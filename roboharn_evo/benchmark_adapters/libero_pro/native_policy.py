"""LIBERO-native OpenPI policy boundary.

Importing this module does not import OpenPI, Torch, JAX, robosuite, or the
LIBERO simulator.  Heavy policy code is loaded only after an explicit local
checkpoint preflight succeeds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from roboharn_evo.benchmark_adapters.base import NeutralObservation


class LiberoNativePolicyError(RuntimeError):
    """Raised when a native policy input, checkpoint, or output is invalid."""


class _Policy(Protocol):
    def infer(
        self,
        observation: Mapping[str, Any],
        *,
        noise: np.ndarray | None = None,
    ) -> Mapping[str, Any]: ...


PolicyFactory = Callable[["LiberoPi05PolicyConfig"], _Policy]
ResizeFunction = Callable[[np.ndarray, int, int], np.ndarray]


@dataclass(frozen=True, slots=True)
class LiberoPi05PolicyConfig:
    """Explicit local configuration for OpenPI's LIBERO π0.5 policy."""

    checkpoint_dir: Path
    source_ref: str
    license_id: str
    config_name: str = "pi05_libero"
    device: str = "cuda"
    resize_size: int = 224
    action_dim: int = 7
    action_low: float = -1.0
    action_high: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_dir",
            Path(self.checkpoint_dir).expanduser(),
        )
        for field_name in ("source_ref", "license_id", "config_name", "device"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty")
            object.__setattr__(self, field_name, value)
        if self.resize_size <= 0:
            raise ValueError("resize_size must be positive")
        if self.action_dim != 7:
            raise ValueError("LIBERO π0.5 action_dim must remain 7")
        if not self.action_low < self.action_high:
            raise ValueError("action_low must be smaller than action_high")


def _required_array(
    observation: Mapping[str, Any],
    key: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if key not in observation:
        raise LiberoNativePolicyError(f"LIBERO π0.5 observation is missing {key!r}")
    value = np.asarray(observation[key])
    if shape is not None and tuple(value.shape) != shape:
        raise LiberoNativePolicyError(
            f"LIBERO π0.5 field {key!r} must have shape {shape}, "
            f"got {tuple(value.shape)}"
        )
    if value.size and np.issubdtype(value.dtype, np.number):
        try:
            finite = np.isfinite(value).all()
        except TypeError as exc:
            raise LiberoNativePolicyError(
                f"LIBERO π0.5 field {key!r} must be numeric"
            ) from exc
        if not finite:
            raise LiberoNativePolicyError(
                f"LIBERO π0.5 field {key!r} contains non-finite values"
            )
    return value


def _camera_image(observation: Mapping[str, Any], key: str) -> np.ndarray:
    image = _required_array(observation, key)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise LiberoNativePolicyError(
            f"LIBERO π0.5 camera {key!r} must be HWC RGB, got {image.shape}"
        )
    if image.dtype != np.uint8:
        raise LiberoNativePolicyError(
            f"LIBERO π0.5 camera {key!r} must be uint8, got {image.dtype}"
        )
    # This is the official policy preprocessing, not an adapter-side claim
    # that the raw benchmark camera has another orientation.
    return np.ascontiguousarray(image[::-1, ::-1])


def _quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert an XYZW quaternion using the official LIBERO evaluation rule."""

    quat = np.asarray(quaternion, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(float(quat[3]))) / denominator


def build_pi05_libero_input(
    observation: NeutralObservation | Mapping[str, Any],
    *,
    prompt: str,
    resize_size: int = 224,
    resize_function: ResizeFunction | None = None,
) -> dict[str, Any]:
    """Build the official OpenPI LIBERO inference input from public fields."""

    prompt = str(prompt).strip()
    if not prompt:
        raise LiberoNativePolicyError("LIBERO π0.5 prompt must be non-empty")
    if resize_size <= 0:
        raise ValueError("resize_size must be positive")
    raw = (
        observation.raw if isinstance(observation, NeutralObservation) else observation
    )
    if not isinstance(raw, Mapping):
        raise LiberoNativePolicyError("LIBERO π0.5 observation must be a mapping")

    if resize_function is None:
        try:
            from openpi_client import image_tools
        except ImportError as exc:
            raise LiberoNativePolicyError(
                "openpi-client is required for official LIBERO image resizing"
            ) from exc
        resize_function = image_tools.resize_with_pad

    base_image = resize_function(
        _camera_image(raw, "agentview_image"),
        resize_size,
        resize_size,
    )
    wrist_image = resize_function(
        _camera_image(raw, "robot0_eye_in_hand_image"),
        resize_size,
        resize_size,
    )
    state = build_pi05_libero_state(raw)
    return {
        "observation/state": state,
        "observation/image": np.asarray(base_image, dtype=np.uint8),
        "observation/wrist_image": np.asarray(wrist_image, dtype=np.uint8),
        "prompt": prompt,
    }


def build_pi05_libero_state(
    observation: NeutralObservation | Mapping[str, Any],
) -> np.ndarray:
    """Project public LIBERO proprioception into the canonical π0.5 8D state."""

    raw = (
        observation.raw if isinstance(observation, NeutralObservation) else observation
    )
    if not isinstance(raw, Mapping):
        raise LiberoNativePolicyError("LIBERO π0.5 observation must be a mapping")
    state = np.concatenate(
        (
            _required_array(raw, "robot0_eef_pos", shape=(3,)),
            _quat_to_axis_angle(_required_array(raw, "robot0_eef_quat", shape=(4,))),
            _required_array(raw, "robot0_gripper_qpos", shape=(2,)),
        )
    ).astype(np.float32, copy=False)
    if tuple(state.shape) != (8,):
        raise LiberoNativePolicyError(
            f"LIBERO π0.5 state must have shape (8,), got {state.shape}"
        )
    return state


class LiberoPi05PolicyBackend:
    """Lazy OpenPI π0.5 executor that returns only native 7D actions."""

    def __init__(
        self,
        config: LiberoPi05PolicyConfig,
        *,
        policy_factory: PolicyFactory | None = None,
        resize_function: ResizeFunction | None = None,
    ) -> None:
        self.config = config
        self._policy_factory = policy_factory or _load_openpi_policy
        self._resize_function = resize_function
        self._policy: _Policy | None = None

    def reset(self) -> None:
        # The official policy has no episode-specific public reset API.
        return None

    def preflight(self) -> dict[str, Any]:
        root = self.config.checkpoint_dir.resolve(strict=True)
        if not root.is_dir():
            raise LiberoNativePolicyError(
                f"LIBERO π0.5 checkpoint is not a directory: {root}"
            )
        weight = root / "model.safetensors"
        norm_stats = root / "physical-intelligence/libero/norm_stats.json"
        for label, path in (("model weights", weight), ("norm stats", norm_stats)):
            if not path.is_file():
                raise LiberoNativePolicyError(f"LIBERO π0.5 {label} is missing: {path}")
        return {
            "policy_family": "openpi_pi05_libero",
            "config_name": self.config.config_name,
            "checkpoint_dir": str(root),
            "source_ref": self.config.source_ref,
            "license_id": self.config.license_id,
            "weight_format": "safetensors",
            "weight_bytes": weight.stat().st_size,
            "norm_stats_path": str(norm_stats),
            "device": self.config.device,
            "input_contract": {
                "base_camera": "agentview_image",
                "wrist_camera": "robot0_eye_in_hand_image",
                "state_shape": [8],
                "policy_image_shape": [
                    self.config.resize_size,
                    self.config.resize_size,
                    3,
                ],
            },
            "output_contract": {
                "action_shape": [self.config.action_dim],
                "environment_nominal_action_low": self.config.action_low,
                "environment_nominal_action_high": self.config.action_high,
                "bounds_handling": (
                    "pass through unchanged; robosuite native OSC controller "
                    "clips normalized arm input and the Panda gripper uses its sign"
                ),
                "conversion": "none; official output transform selects 7 LIBERO dimensions",
            },
        }

    def load(self) -> None:
        self.preflight()
        if self._policy is None:
            self._policy = self._policy_factory(self.config)

    def predict_native_action_chunk(
        self,
        observation: NeutralObservation | Mapping[str, Any],
        *,
        prompt: str,
        sampling_noise: np.ndarray | None = None,
    ) -> np.ndarray:
        self.load()
        assert self._policy is not None
        policy_input = build_pi05_libero_input(
            observation,
            prompt=prompt,
            resize_size=self.config.resize_size,
            resize_function=self._resize_function,
        )
        if sampling_noise is None:
            response = self._policy.infer(policy_input)
        else:
            noise = np.asarray(sampling_noise, dtype=np.float32)
            if noise.ndim != 2 or not noise.size or not np.isfinite(noise).all():
                raise LiberoNativePolicyError(
                    "LIBERO π0.5 sampling noise must be a non-empty finite matrix"
                )
            response = self._policy.infer(policy_input, noise=noise.copy())
        if not isinstance(response, Mapping) or "actions" not in response:
            raise LiberoNativePolicyError(
                "LIBERO π0.5 response must contain an actions array"
            )
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.config.action_dim:
            raise LiberoNativePolicyError(
                f"LIBERO π0.5 actions must have shape (horizon, 7), got {actions.shape}"
            )
        if actions.shape[0] < 1 or not np.isfinite(actions).all():
            raise LiberoNativePolicyError(
                "LIBERO π0.5 actions must be non-empty and finite"
            )
        return actions

    def predict_action_chunk(
        self,
        *,
        observation: Mapping[str, Any],
        task: str,
        subtask: str,
        memory: str,
    ) -> np.ndarray:
        """实现 RoboHarn-Evo 的 ExecutorBackend 协议。"""

        del memory
        prompt = str(subtask).strip() or str(task).strip()
        return self.predict_native_action_chunk(observation, prompt=prompt)


def _load_openpi_policy(config: LiberoPi05PolicyConfig) -> _Policy:
    try:
        from openpi.policies import policy_config
        from openpi.shared import normalize
        from openpi.training import config as training_config
    except ImportError as exc:
        raise LiberoNativePolicyError(
            "the isolated policy environment lacks the OpenPI inference package"
        ) from exc

    root = config.checkpoint_dir.resolve(strict=True)
    norm_stats = normalize.load(root / "physical-intelligence/libero")
    train_config = training_config.get_config(config.config_name)
    return policy_config.create_trained_policy(
        train_config,
        root,
        norm_stats=norm_stats,
        pytorch_device=config.device,
    )


__all__ = [
    "LiberoNativePolicyError",
    "LiberoPi05PolicyBackend",
    "LiberoPi05PolicyConfig",
    "build_pi05_libero_input",
    "build_pi05_libero_state",
]
