"""HTTP service for the RMBench PI0.5 executor contract.

The default OpenPI source is the code-only copy vendored below
benchmarks/rmbench/policy/pi05. Trained checkpoints remain external, read-only
operator inputs. OpenPI is imported lazily, so importing this module and
showing CLI help do not load a config, checkpoint, or model. Path-only
preflight imports the selected source and resolves the requested train config,
but never constructs a policy or reads checkpoint contents.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from roboharn_evo.services.http_safety import (
    LEGACY_RMBENCH_DONOR_ROOT,
    RequestBodyError,
    ROBOHARN_EVAL_RESULT_ROOT,
    ROBOHARN_PROJECT_ROOT,
    add_allow_remote_argument,
    read_json_object_body,
    validate_bind_host,
)

PROMPT_MODES = ("subtask_only", "task_subtask", "task_memory_subtask")
DEFAULT_MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024
DEFAULT_OPENPI_REPOSITORY = (
    ROBOHARN_PROJECT_ROOT / "benchmarks" / "rmbench" / "policy" / "pi05"
)
DEFAULT_CONFIG_NAME = "pi05_aloha_full_base"
DEFAULT_CACHE_ROOT = ROBOHARN_PROJECT_ROOT / "eval_result" / "service_caches" / "pi05"
LEGACY_RMBENCH_DONOR = LEGACY_RMBENCH_DONOR_ROOT
_CACHE_ENVIRONMENT_LAYOUT = {
    "JAX_COMPILATION_CACHE_DIR": "jax_compilation_cache",
    "XDG_CACHE_HOME": "xdg",
    "HF_HOME": "huggingface",
    "HF_HUB_CACHE": "huggingface/hub",
    "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
    "TRANSFORMERS_CACHE": "huggingface/transformers",
    "TORCH_HOME": "torch",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda",
}


@dataclass(frozen=True)
class OpenPiBindings:
    """The two lazy OpenPI callables required by the serving runtime."""

    get_config: Callable[[str], Any]
    create_trained_policy: Callable[[Any, Path], Any]


def _existing_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is not an existing directory: {path}")
    return path


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second or first.is_relative_to(second) or second.is_relative_to(first)
    )


def pi05_protected_read_roots(
    *,
    openpi_repo: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Path, ...]:
    """Collect immutable source/asset/checkpoint boundaries for this process."""

    candidates = [LEGACY_RMBENCH_DONOR, DEFAULT_OPENPI_REPOSITORY.resolve()]
    if openpi_repo is not None:
        candidates.append(Path(openpi_repo).expanduser().resolve())
    if checkpoint_dir is not None:
        candidates.append(Path(checkpoint_dir).expanduser().resolve())
    for variable in ("RMBENCH_ROOT", "RMBENCH_ASSETS_ROOT"):
        raw_boundary = os.environ.get(variable, "").strip()
        if raw_boundary:
            candidates.append(Path(raw_boundary).expanduser().resolve())
    return tuple(dict.fromkeys(candidates))


def configure_pi05_cache_root(
    value: str | Path,
    *,
    openpi_repo: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> Path:
    """验证服务缓存位于 RoboHarn-Evo 的 eval_result 目录。"""

    cache_root = Path(value).expanduser().resolve()
    writable_root = ROBOHARN_EVAL_RESULT_ROOT.resolve()
    if not cache_root.is_relative_to(writable_root):
        raise ValueError(
            f"PI0.5 cache root must resolve inside the RoboHarn-Evo eval_result root "
            f"{writable_root}: "
            f"{cache_root}"
        )

    named_boundaries = {
        LEGACY_RMBENCH_DONOR: "legacy RMBench donor",
        DEFAULT_OPENPI_REPOSITORY.resolve(): "copied OpenPI source",
    }
    if openpi_repo is not None:
        named_boundaries[Path(openpi_repo).expanduser().resolve()] = (
            "selected OpenPI source"
        )
    if checkpoint_dir is not None:
        named_boundaries[Path(checkpoint_dir).expanduser().resolve()] = (
            "PI0.5 checkpoint"
        )
    for variable in ("RMBENCH_ROOT", "RMBENCH_ASSETS_ROOT"):
        raw_boundary = os.environ.get(variable, "").strip()
        if raw_boundary:
            named_boundaries[Path(raw_boundary).expanduser().resolve()] = variable
    for boundary, label in named_boundaries.items():
        if _paths_overlap(cache_root, boundary):
            raise ValueError(
                f"PI0.5 cache root {cache_root} overlaps protected {label}: {boundary}"
            )

    cache_root.mkdir(parents=True, exist_ok=True)
    if not cache_root.is_dir():
        raise NotADirectoryError(f"PI0.5 cache root is not a directory: {cache_root}")

    os.environ["OPENPI_DATA_HOME"] = str(cache_root)
    os.environ.update(
        {
            variable: str(cache_root / relative_path)
            for variable, relative_path in _CACHE_ENVIRONMENT_LAYOUT.items()
        }
    )
    return cache_root


def validate_openpi_repository(value: str | Path) -> Path:
    """Validate the explicitly selected OpenPI source tree."""

    root = _existing_directory(value, label="OpenPI repository")
    if root.is_relative_to(LEGACY_RMBENCH_DONOR):
        raise ValueError(
            "The legacy RMBench checkout is read-only provenance/runtime data, "
            f"not an OpenPI execution source: {root}"
        )
    package = root / "src" / "openpi"
    if not package.is_dir():
        raise FileNotFoundError(
            "OpenPI repository must contain src/openpi; "
            f"the supplied checkout does not: {root}"
        )
    return root


def _validate_openpi_module_origin(
    module_name: str,
    module: Any,
    *,
    package_root: Path,
) -> Path:
    raw_path = getattr(module, "__file__", None)
    if isinstance(raw_path, str):
        origin_paths = (Path(raw_path).resolve(),)
    else:
        raw_namespace_paths = getattr(module, "__path__", None)
        if raw_namespace_paths is None:
            raise ImportError(
                f"OpenPI module {module_name!r} has neither an auditable __file__ "
                f"nor namespace __path__; expected it below {package_root}"
            )
        origin_paths = tuple(Path(path).resolve() for path in raw_namespace_paths)
        if not origin_paths:
            raise ImportError(
                f"OpenPI namespace module {module_name!r} has an empty __path__; "
                f"expected it below {package_root}"
            )
    foreign_paths = [
        path for path in origin_paths if not path.is_relative_to(package_root)
    ]
    if foreign_paths:
        raise ImportError(
            f"OpenPI module {module_name!r} came from {foreign_paths}, outside the "
            f"explicitly selected repository package {package_root}"
        )
    return origin_paths[0]


def _validate_loaded_openpi_origins(*, package_root: Path) -> None:
    for module_name, module in tuple(sys.modules.items()):
        if module_name == "openpi" or module_name.startswith("openpi."):
            _validate_openpi_module_origin(
                module_name,
                module,
                package_root=package_root,
            )


def configure_openpi_import_paths(value: str | Path) -> Path:
    """Add only the explicitly selected OpenPI checkout to import lookup."""

    root = validate_openpi_repository(value)
    candidates = (
        root / "src",
        root / "packages" / "openpi-client" / "src",
    )
    additions = [
        str(path) for path in candidates if path.is_dir() and str(path) not in sys.path
    ]
    sys.path[0:0] = additions
    return root


def _load_openpi_training_config(openpi_repo: str | Path) -> tuple[Path, Any]:
    """Import the selected repository's config registry without loading a policy."""

    root = validate_openpi_repository(openpi_repo)
    package_root = root / "src" / "openpi"
    _validate_loaded_openpi_origins(package_root=package_root)
    sys.dont_write_bytecode = True
    configure_openpi_import_paths(root)
    training_config = importlib.import_module("openpi.training.config")
    _validate_openpi_module_origin(
        "openpi.training.config",
        training_config,
        package_root=package_root,
    )
    _validate_loaded_openpi_origins(package_root=package_root)
    return root, training_config


def resolve_openpi_train_config(openpi_repo: str | Path, config_name: str) -> Any:
    """Resolve one named config from the explicitly selected OpenPI source tree."""

    requested_name = str(config_name).strip()
    if not requested_name:
        raise ValueError("config_name must not be empty")
    root, training_config = _load_openpi_training_config(openpi_repo)
    train_config = training_config.get_config(requested_name)
    resolved_name = str(getattr(train_config, "name", "")).strip()
    if resolved_name and resolved_name != requested_name:
        raise ValueError(
            f"OpenPI config registry returned {resolved_name!r} for "
            f"requested config {requested_name!r} from {root}"
        )
    _validate_loaded_openpi_origins(package_root=root / "src" / "openpi")
    return train_config


def load_openpi_bindings(openpi_repo: str | Path) -> OpenPiBindings:
    """Import OpenPI only when the real model runtime is constructed."""

    root, training_config = _load_openpi_training_config(openpi_repo)
    package_root = root / "src" / "openpi"
    policy_config = importlib.import_module("openpi.policies.policy_config")
    _validate_openpi_module_origin(
        "openpi.policies.policy_config",
        policy_config,
        package_root=package_root,
    )
    _validate_loaded_openpi_origins(package_root=package_root)
    return OpenPiBindings(
        get_config=training_config.get_config,
        create_trained_policy=policy_config.create_trained_policy,
    )


def _positive_int(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _required_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _camera_image(observation: Mapping[str, Any], camera: str) -> np.ndarray:
    try:
        raw_image = observation["observation"][camera]["rgb"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"observation is missing observation.{camera}.rgb") from exc
    image = np.asarray(raw_image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(
            f"observation.{camera}.rgb must have HWC shape with 3 or 4 channels, "
            f"got {image.shape}"
        )
    return np.transpose(image[..., :3], (2, 0, 1))


def encode_rmbench_observation(
    observation: Mapping[str, Any], prompt: str
) -> dict[str, Any]:
    """Translate the legacy RMBench observation into OpenPI's ALOHA schema."""

    observation = _required_mapping(observation, field="observation")
    try:
        raw_state = observation["joint_action"]["vector"]
    except (KeyError, TypeError) as exc:
        raise ValueError("observation is missing joint_action.vector") from exc
    state = np.asarray(raw_state, dtype=np.float32)
    if state.ndim != 1:
        raise ValueError(
            f"joint_action.vector must be one-dimensional, got {state.shape}"
        )
    return {
        "state": state,
        "images": {
            "cam_high": _camera_image(observation, "head_camera"),
            "cam_left_wrist": _camera_image(observation, "left_camera"),
            "cam_right_wrist": _camera_image(observation, "right_camera"),
        },
        "prompt": prompt,
    }


class Pi05ExecutorRuntime:
    """Own one OpenPI policy and serialize inference across HTTP threads."""

    def __init__(
        self,
        *,
        openpi_repo: str | Path,
        config_name: str,
        checkpoint_dir: str | Path,
        pi0_step: int,
        prompt_mode: str,
        asset_id: str | None,
        cache_root: str | Path = DEFAULT_CACHE_ROOT,
        bindings: OpenPiBindings | None = None,
        action_horizon: int | None = None,
    ) -> None:
        self.openpi_repo = validate_openpi_repository(openpi_repo)
        self.checkpoint_dir = _existing_directory(
            checkpoint_dir, label="PI0.5 checkpoint"
        )
        self.cache_root = configure_pi05_cache_root(
            cache_root,
            openpi_repo=self.openpi_repo,
            checkpoint_dir=self.checkpoint_dir,
        )
        self.protected_read_roots = pi05_protected_read_roots(
            openpi_repo=self.openpi_repo,
            checkpoint_dir=self.checkpoint_dir,
        )
        self.config_name = str(config_name).strip()
        if not self.config_name:
            raise ValueError("config_name must not be empty")
        self.pi0_step = _positive_int(pi0_step, field="pi0_step")
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
        self.prompt_mode = prompt_mode
        self.asset_id = str(asset_id).strip() if asset_id else None
        self._inference_lock = threading.Lock()
        self._last_prompt: str | None = None
        self._observation_window: dict[str, Any] | None = None

        resolved_bindings = bindings or load_openpi_bindings(self.openpi_repo)
        train_config = resolved_bindings.get_config(self.config_name)
        if action_horizon is not None:
            train_config = dataclasses.replace(train_config, model=dataclasses.replace(
                train_config.model, action_horizon=_positive_int(action_horizon, field="action_horizon")))
        self.action_horizon = getattr(getattr(train_config, "model", None), "action_horizon", None)
        train_config = self._select_assets(train_config)
        self._policy = resolved_bindings.create_trained_policy(
            train_config,
            self.checkpoint_dir,
        )

    def _select_assets(self, train_config: Any) -> Any:
        resolved_asset_id = self.asset_id
        if resolved_asset_id is None:
            assets_dir = self.checkpoint_dir / "assets"
            if assets_dir.is_dir():
                entries = sorted(
                    path.name for path in assets_dir.iterdir() if path.is_dir()
                )
                if entries:
                    resolved_asset_id = entries[0]
        self.asset_id = resolved_asset_id
        if resolved_asset_id is None:
            return train_config
        try:
            updated_assets = dataclasses.replace(
                train_config.data.assets, asset_id=resolved_asset_id
            )
            updated_data = dataclasses.replace(train_config.data, assets=updated_assets)
            return dataclasses.replace(train_config, data=updated_data)
        except (AttributeError, TypeError) as exc:
            raise ValueError(
                "OpenPI train config does not expose dataclass data.assets needed "
                "to apply --asset-id"
            ) from exc

    def reset(self) -> None:
        with self._inference_lock:
            self._last_prompt = None
            self._observation_window = None

    def build_prompt(self, *, task: str, subtask: str, memory: str) -> str:
        subtask_only = subtask.strip()
        if self.prompt_mode == "subtask_only":
            return subtask_only
        if self.prompt_mode == "task_subtask":
            return f"Global task: {task.strip()}\nCurrent subtask: {subtask_only}"
        if self.prompt_mode == "task_memory_subtask":
            return (
                f"Global task: {task.strip()}\n"
                f"Committed memory: {memory.strip()}\n"
                f"Current subtask: {subtask_only}"
            )
        raise AssertionError(f"unreachable prompt mode: {self.prompt_mode}")

    def predict_action_chunk(
        self,
        *,
        observation: Mapping[str, Any],
        task: str,
        subtask: str,
        memory: str,
        max_chunk_steps: int,
    ) -> np.ndarray:
        max_chunk_steps = _positive_int(max_chunk_steps, field="max_chunk_steps")
        prompt = self.build_prompt(task=task, subtask=subtask, memory=memory)
        observation_window = encode_rmbench_observation(observation, prompt)
        with self._inference_lock:
            if prompt != self._last_prompt:
                self._observation_window = None
                self._last_prompt = prompt
            self._observation_window = observation_window
            result = self._policy.infer(self._observation_window)
        if not isinstance(result, Mapping) or "actions" not in result:
            raise ValueError("OpenPI infer result must contain an 'actions' field")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"Expected action chunk with shape (T, D), got {actions.shape}"
            )
        return actions[:max_chunk_steps]

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "pi05",
            "config_name": self.config_name,
            "checkpoint_dir": str(self.checkpoint_dir),
            "openpi_repo": str(self.openpi_repo),
            "prompt_mode": self.prompt_mode,
            "asset_id": self.asset_id,
            "cache_root": str(self.cache_root),
            "project_root": str(ROBOHARN_PROJECT_ROOT.resolve()),
            "protected_read_roots": [str(path) for path in self.protected_read_roots],
            "config_resolved": True,
            "checkpoint_loaded": True,
            "model_loaded": True,
        }


def handle_act_payload(
    runtime: Pi05ExecutorRuntime,
    payload: Mapping[str, Any],
    *,
    default_action_dim: int,
) -> dict[str, Any]:
    """Validate one ``POST /act`` body and build its JSON response."""

    payload = _required_mapping(payload, field="request body")
    if "observation" not in payload:
        raise ValueError("request body is missing observation")
    max_chunk_steps = _positive_int(
        payload.get("max_chunk_steps", runtime.pi0_step),
        field="max_chunk_steps",
    )
    action_dim = _positive_int(
        payload.get("action_dim", default_action_dim), field="action_dim"
    )
    actions = runtime.predict_action_chunk(
        observation=_required_mapping(payload["observation"], field="observation"),
        task=str(payload.get("task", "")),
        subtask=str(payload.get("subtask", "")),
        memory=str(payload.get("memory", "")),
        max_chunk_steps=max_chunk_steps,
    )
    if actions.shape[1] != action_dim:
        raise ValueError(
            f"Executor produced action_dim={actions.shape[1]}, expected {action_dim}"
        )
    return {
        "action_chunk": actions.tolist(),
        "model_info": runtime.model_info(),
    }


class Pi05ExecutorHandler(BaseHTTPRequestHandler):
    runtime: Pi05ExecutorRuntime | None = None
    action_dim: int = 14
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return
        if self.runtime is None:
            self._send_json(
                {"status": "error", "error": "runtime is not initialized"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json(
            {
                "status": "ok",
                **self.runtime.model_info(),
                "pi0_step": self.runtime.pi0_step,
                "action_horizon": getattr(self.runtime, "action_horizon", None),
                "action_dim": self.action_dim,
            }
        )

    def do_POST(self) -> None:
        if self.runtime is None:
            self._send_json(
                {"status": "error", "error": "runtime is not initialized"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        try:
            if self.path == "/reset":
                self.runtime.reset()
                self._send_json({"status": "ok"})
                return
            if self.path != "/act":
                self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
                return
            payload = read_json_object_body(
                headers=self.headers,
                stream=self.rfile,
                max_bytes=self.max_request_bytes,
            )
            response = handle_act_payload(
                self.runtime,
                payload,
                default_action_dim=self.action_dim,
            )
        except RequestBodyError as exc:
            self._send_json(
                {"status": "error", "error": str(exc)},
                status=exc.status,
            )
            return
        except (TypeError, ValueError) as exc:
            self._send_json(
                {"status": "error", "error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {"status": "error", "error": f"executor inference failed: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json(response)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(
        self, payload: Mapping[str, Any], *, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def build_handler(
    runtime: Pi05ExecutorRuntime,
    *,
    action_dim: int,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
) -> type[Pi05ExecutorHandler]:
    """Bind runtime state without mutating the global handler class."""

    validated_action_dim = _positive_int(action_dim, field="action_dim")
    validated_max_request_bytes = _positive_int(
        max_request_bytes,
        field="max_request_bytes",
    )

    class BoundPi05ExecutorHandler(Pi05ExecutorHandler):
        pass

    BoundPi05ExecutorHandler.runtime = runtime
    BoundPi05ExecutorHandler.action_dim = validated_action_dim
    BoundPi05ExecutorHandler.max_request_bytes = validated_max_request_bytes
    return BoundPi05ExecutorHandler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the RMBench RoboHarn-Evo PI0.5 executor over HTTP.",
        epilog=(
            "Invoke this script with the Python executable from the OpenPI environment. "
            "The default source is RoboHarn-Evo's code-only OpenPI copy; checkpoints remain external."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    add_allow_remote_argument(parser)
    parser.add_argument("--port", type=int, default=9201)
    parser.add_argument(
        "--openpi-repo",
        type=Path,
        default=DEFAULT_OPENPI_REPOSITORY,
        help="OpenPI source tree (default: RoboHarn-Evo's benchmarks/rmbench/policy/pi05 copy).",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help=f"OpenPI train config name (default: {DEFAULT_CONFIG_NAME}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="External read-only checkpoint directory; weights are not copied into RoboHarn-Evo.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Writable cache root inside RoboHarn-Evo eval_result (default: eval_result/service_caches/pi05).",
    )
    parser.add_argument("--pi0-step", type=int, default=50)
    parser.add_argument("--action-horizon", type=int, default=None,
                        help="Explicit checkpoint inference horizon; omit to preserve the selected config.")
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=DEFAULT_MAX_REQUEST_BODY_BYTES,
        help="Maximum accepted UTF-8 JSON request-body size.",
    )
    parser.add_argument("--asset-id", default="")
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="subtask_only")
    parser.add_argument(
        "--check-paths-only",
        action="store_true",
        help=(
            "validate paths, import the selected OpenPI config registry, and resolve "
            "--config-name without constructing a policy or loading checkpoint/model data"
        ),
    )
    return parser


def run_server(args: argparse.Namespace) -> None:
    validate_bind_host(args.host, allow_remote=bool(args.allow_remote))
    openpi_repo = validate_openpi_repository(args.openpi_repo)
    checkpoint_dir = _existing_directory(args.checkpoint_dir, label="PI0.5 checkpoint")
    cache_root = configure_pi05_cache_root(
        args.cache_root,
        openpi_repo=openpi_repo,
        checkpoint_dir=checkpoint_dir,
    )
    protected_read_roots = pi05_protected_read_roots(
        openpi_repo=openpi_repo,
        checkpoint_dir=checkpoint_dir,
    )
    _positive_int(args.port, field="port")
    _positive_int(args.pi0_step, field="pi0_step")
    if args.action_horizon is not None:
        _positive_int(args.action_horizon, field="action_horizon")
    _positive_int(args.action_dim, field="action_dim")
    _positive_int(args.max_request_bytes, field="max_request_bytes")
    if args.check_paths_only:
        train_config = resolve_openpi_train_config(openpi_repo, args.config_name)
        resolved_config_name = str(getattr(train_config, "name", args.config_name))
        print(
            json.dumps(
                {
                    "status": "ok",
                    "endpoint": f"http://{args.host}:{args.port}/act",
                    "python_executable": sys.executable,
                    "openpi_repo": str(openpi_repo),
                    "checkpoint_dir": str(checkpoint_dir),
                    "config_name": args.config_name,
                    "resolved_config_name": resolved_config_name,
                    "config_resolved": True,
                    "cache_root": str(cache_root),
                    "project_root": str(ROBOHARN_PROJECT_ROOT.resolve()),
                    "protected_read_roots": [
                        str(path) for path in protected_read_roots
                    ],
                    "openpi_data_home": os.environ["OPENPI_DATA_HOME"],
                    "dont_write_bytecode": sys.dont_write_bytecode,
                    "checkpoint_loaded": False,
                    "model_loaded": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    runtime = Pi05ExecutorRuntime(
        openpi_repo=openpi_repo,
        config_name=args.config_name,
        checkpoint_dir=checkpoint_dir,
        pi0_step=args.pi0_step,
        prompt_mode=args.prompt_mode,
        asset_id=args.asset_id.strip() or None,
        cache_root=cache_root,
        action_horizon=args.action_horizon,
    )
    handler = build_handler(
        runtime,
        action_dim=args.action_dim,
        max_request_bytes=args.max_request_bytes,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        "[pi05-executor] listening on "
        f"http://{args.host}:{args.port} "
        f"config_name={runtime.config_name} checkpoint_dir={runtime.checkpoint_dir} "
        f"openpi_repo={runtime.openpi_repo} prompt_mode={runtime.prompt_mode} "
        f"asset_id={runtime.asset_id} cache_root={runtime.cache_root}",
        flush=True,
    )
    server.serve_forever()


def main(argv: Sequence[str] | None = None) -> None:
    run_server(build_arg_parser().parse_args(argv))


__all__ = [
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_OPENPI_REPOSITORY",
    "OpenPiBindings",
    "Pi05ExecutorHandler",
    "Pi05ExecutorRuntime",
    "build_arg_parser",
    "build_handler",
    "configure_openpi_import_paths",
    "configure_pi05_cache_root",
    "encode_rmbench_observation",
    "handle_act_payload",
    "load_openpi_bindings",
    "main",
    "pi05_protected_read_roots",
    "resolve_openpi_train_config",
    "run_server",
    "validate_openpi_repository",
]
