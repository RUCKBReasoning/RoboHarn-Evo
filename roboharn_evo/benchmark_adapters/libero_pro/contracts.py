"""Simulator-free contracts for the LIBERO-PRO adapter.

The types in this module describe observed benchmark facts.  They do not
import LIBERO, robosuite, MuJoCo, Gymnasium, or a task registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

CANONICAL_CAMERA_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
)
CANONICAL_PROPRIOCEPTION_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
)

ROBOHARN_AGENT_CAMERA_ROLES = ("head", "left", "right")
ROBOHARN_AGENT_PROPRIOCEPTION_ROLES = (
    "joint_vector",
    "left_endpose",
    "right_endpose",
)
ROBOHARN_AGENT_ACTION_TYPE = "qpos"
ROBOHARN_AGENT_ACTION_SHAPE = (14,)


class LiberoProContractError(ValueError):
    """Raised when a value violates the explicit LIBERO-PRO boundary."""


class LiberoProCapabilityError(RuntimeError):
    """Raised when a caller requests an unsupported Agent integration."""

    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing = tuple(missing)
        detail = ", ".join(self.missing) or "unknown capability"
        super().__init__(
            "LIBERO-PRO cannot be connected to the RMBench-shaped RoboHarn-Evo Agent "
            f"without fabricating unavailable fields; missing: {detail}"
        )


@dataclass(frozen=True, slots=True)
class LiberoProObservationContract:
    """Names of public observation fields copied without transformation.

    ``public_raw_keys`` is an explicit allowlist for additional audited,
    non-oracle fields. Unrecognised keys are not exposed because canonical
    LIBERO observations can contain object-state ground truth.
    """

    camera_keys: tuple[str, ...] = CANONICAL_CAMERA_KEYS
    proprioception_keys: tuple[str, ...] = CANONICAL_PROPRIOCEPTION_KEYS
    public_raw_keys: tuple[str, ...] = ()
    instruction_key: str | None = None
    instruction_attribute: str = "language_instruction"

    def __post_init__(self) -> None:
        _validate_unique_names("camera_keys", self.camera_keys)
        _validate_unique_names("proprioception_keys", self.proprioception_keys)
        _validate_unique_names("public_raw_keys", self.public_raw_keys)
        if self.instruction_key is not None and not self.instruction_key:
            raise LiberoProContractError("instruction_key cannot be empty")
        if not self.instruction_attribute:
            raise LiberoProContractError("instruction_attribute cannot be empty")


@dataclass(frozen=True, slots=True)
class LiberoProActionContract:
    """Audited native action interface; validation never clips or converts."""

    action_type: str = "libero"
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    lower_bounds: tuple[float, ...] | None = None
    upper_bounds: tuple[float, ...] | None = None
    bounds_handling: str = "adapter_validate"
    control_mode: str | None = None
    translation_mode: str | None = None
    rotation_representation: str | None = None
    reference_frame: str | None = None
    gripper_convention: str | None = None

    def __post_init__(self) -> None:
        if not self.action_type:
            raise LiberoProContractError("action_type cannot be empty")
        if self.bounds_handling not in {"adapter_validate", "native_controller"}:
            raise LiberoProContractError(
                "bounds_handling must be 'adapter_validate' or 'native_controller'"
            )
        if self.shape is not None:
            if not self.shape or any(int(size) <= 0 for size in self.shape):
                raise LiberoProContractError(
                    f"action shape must contain positive dimensions, got {self.shape!r}"
                )
            expected_size = 1
            for size in self.shape:
                expected_size *= int(size)
            for name, bounds in (
                ("lower_bounds", self.lower_bounds),
                ("upper_bounds", self.upper_bounds),
            ):
                if bounds is not None and len(bounds) != expected_size:
                    raise LiberoProContractError(
                        f"{name} has {len(bounds)} values for action shape {self.shape}"
                    )
        elif self.lower_bounds is not None or self.upper_bounds is not None:
            raise LiberoProContractError(
                "action bounds require an explicit action shape"
            )
        if (
            self.lower_bounds is not None
            and self.upper_bounds is not None
            and any(
                low > high for low, high in zip(self.lower_bounds, self.upper_bounds)
            )
        ):
            raise LiberoProContractError(
                "action lower bounds cannot exceed upper bounds"
            )


@dataclass(frozen=True, slots=True)
class LiberoProCapabilities:
    """Capabilities established by the selected environment/configuration."""

    camera_keys: tuple[str, ...] = ()
    proprioception_keys: tuple[str, ...] = ()
    camera_roles: Mapping[str, str] = field(default_factory=dict)
    proprioception_roles: Mapping[str, str] = field(default_factory=dict)
    action: LiberoProActionContract = field(default_factory=LiberoProActionContract)

    def __post_init__(self) -> None:
        _validate_unique_names("camera_keys", self.camera_keys)
        _validate_unique_names("proprioception_keys", self.proprioception_keys)
        object.__setattr__(self, "camera_roles", dict(self.camera_roles))
        object.__setattr__(
            self,
            "proprioception_roles",
            dict(self.proprioception_roles),
        )

    def with_observed_fields(
        self,
        *,
        camera_keys: tuple[str, ...],
        proprioception_keys: tuple[str, ...],
    ) -> LiberoProCapabilities:
        """Return capabilities narrowed to fields present in one observation."""
        return LiberoProCapabilities(
            camera_keys=tuple(camera_keys),
            proprioception_keys=tuple(proprioception_keys),
            camera_roles=self.camera_roles,
            proprioception_roles=self.proprioception_roles,
            action=self.action,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "camera_keys": list(self.camera_keys),
            "proprioception_keys": list(self.proprioception_keys),
            "camera_semantics": {
                key: {
                    "orientation": "benchmark_raw",
                    "spatial_transform": "none",
                }
                for key in self.camera_keys
            },
            "camera_roles": dict(self.camera_roles),
            "proprioception_roles": dict(self.proprioception_roles),
            "action": {
                "action_type": self.action.action_type,
                "shape": None if self.action.shape is None else list(self.action.shape),
                "dtype": self.action.dtype,
                "lower_bounds": (
                    None
                    if self.action.lower_bounds is None
                    else list(self.action.lower_bounds)
                ),
                "upper_bounds": (
                    None
                    if self.action.upper_bounds is None
                    else list(self.action.upper_bounds)
                ),
                "bounds_handling": self.action.bounds_handling,
                "control_mode": self.action.control_mode,
                "translation_mode": self.action.translation_mode,
                "rotation_representation": self.action.rotation_representation,
                "reference_frame": self.action.reference_frame,
                "gripper_convention": self.action.gripper_convention,
            },
        }


@dataclass(frozen=True, slots=True)
class RoboHarnCapabilityReport:
    """Result of checking the existing RMBench-shaped Agent requirements."""

    compatible: bool
    missing: tuple[str, ...]

    def require(self) -> None:
        if not self.compatible:
            raise LiberoProCapabilityError(self.missing)


def assess_roboharn_agent_compatibility(
    capabilities: LiberoProCapabilities,
) -> RoboHarnCapabilityReport:
    """Check facts only; never synthesize aliases or missing sensors."""
    missing: list[str] = []
    available_cameras = set(capabilities.camera_keys)
    for role in ROBOHARN_AGENT_CAMERA_ROLES:
        source_key = capabilities.camera_roles.get(role)
        if not source_key or source_key not in available_cameras:
            missing.append(f"camera_role:{role}")

    available_proprioception = set(capabilities.proprioception_keys)
    for role in ROBOHARN_AGENT_PROPRIOCEPTION_ROLES:
        source_key = capabilities.proprioception_roles.get(role)
        if not source_key or source_key not in available_proprioception:
            missing.append(f"proprioception_role:{role}")

    if capabilities.action.action_type != ROBOHARN_AGENT_ACTION_TYPE:
        missing.append(f"action_type:{ROBOHARN_AGENT_ACTION_TYPE}")
    if capabilities.action.shape != ROBOHARN_AGENT_ACTION_SHAPE:
        missing.append(f"action_shape:{ROBOHARN_AGENT_ACTION_SHAPE}")

    return RoboHarnCapabilityReport(
        compatible=not missing,
        missing=tuple(missing),
    )


def _validate_unique_names(field_name: str, values: tuple[str, ...]) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise LiberoProContractError(f"{field_name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise LiberoProContractError(f"{field_name} contains duplicate names")


__all__ = [
    "CANONICAL_CAMERA_KEYS",
    "CANONICAL_PROPRIOCEPTION_KEYS",
    "LiberoProActionContract",
    "LiberoProCapabilities",
    "LiberoProCapabilityError",
    "LiberoProContractError",
    "LiberoProObservationContract",
    "RoboHarnCapabilityReport",
    "assess_roboharn_agent_compatibility",
]
