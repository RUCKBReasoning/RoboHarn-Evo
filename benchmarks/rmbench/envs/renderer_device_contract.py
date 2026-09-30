"""Optional, explicit SAPIEN renderer-device binding for rollout processes.

The RMBench environment historically lets SAPIEN select its default Vulkan
device.  That remains the exact behavior when ``RMBENCH_RENDER_DEVICE`` is
unset.  Parallel launchers opt in to an explicit physical render device with
the canonical ``pci:dddd:bb:dd.f`` alias while exposing exactly that GPU to
CUDA by UUID.

This module is deliberately independent of the Agent and task semantics.  It
only constructs the renderer/scene infrastructure and records which device
SAPIEN actually selected.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


RENDER_DEVICE_ENV = "RMBENCH_RENDER_DEVICE"
RENDER_DEVICE_STRICT_ENV = "RMBENCH_RENDER_DEVICE_STRICT"
EXPECTED_RENDER_CUDA_ID_ENV = "RMBENCH_EXPECTED_RENDER_CUDA_ID"
EXPECTED_RENDER_PCI_BUS_ID_ENV = "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID"
EXPECTED_PHYSICAL_GPU_ENV = "RMBENCH_EXPECTED_PHYSICAL_GPU"
RENDER_DEVICE_PROVENANCE_PATH_ENV = "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH"
RAY_TRACING_DENOISER_ENV = "RMBENCH_RAY_TRACING_DENOISER"
SUPPORTED_RAY_TRACING_DENOISERS = frozenset({"none", "oidn", "optix"})
_RMBENCH_OUTPUT_ROOT_ENV = "RMBENCH_OUTPUT_ROOT"
_ROBOHARN_OUTPUT_ROOT_ENV = "ROBOHARN_EVO_OUTPUT_ROOT"
_READONLY_ROOT_ENVS = ("RMBENCH_ROOT", "RMBENCH_ASSETS_ROOT")
_PROVENANCE_FILENAME = "renderer_device_provenance.json"


class RendererDeviceContractError(RuntimeError):
    """Raised when an explicitly requested renderer-device contract fails.

    ``provenance`` is populated once the selected device can be inspected.  It
    lets the caller persist a structured failure record before re-raising the
    infrastructure error.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str = "configuration",
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.provenance = None if provenance is None else dict(provenance)


def configured_ray_tracing_denoiser(
    environ: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ if environ is None else environ
    value = str(
        environment.get(RAY_TRACING_DENOISER_ENV, "oidn") or "oidn"
    ).strip().lower()
    if value not in SUPPORTED_RAY_TRACING_DENOISERS:
        raise RendererDeviceContractError(
            f"{RAY_TRACING_DENOISER_ENV} must be one of "
            f"{sorted(SUPPORTED_RAY_TRACING_DENOISERS)}, got {value!r}"
        )
    return value


def _roboharn_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _validated_provenance_destination(
    raw_path: str,
    *,
    environment: Mapping[str, str],
) -> Path:
    """要求来源记录位于 RoboHarn-Evo 输出目录。"""

    rmbench_output_raw = str(
        environment.get(_RMBENCH_OUTPUT_ROOT_ENV, "") or ""
    ).strip()
    roboharn_output_raw = str(
        environment.get(_ROBOHARN_OUTPUT_ROOT_ENV, "") or ""
    ).strip()
    if not rmbench_output_raw or not roboharn_output_raw:
        raise RendererDeviceContractError(
            "renderer-device provenance requires both "
            f"{_RMBENCH_OUTPUT_ROOT_ENV} and {_ROBOHARN_OUTPUT_ROOT_ENV}"
        )
    rmbench_output = Path(rmbench_output_raw).expanduser().resolve()
    roboharn_output = Path(roboharn_output_raw).expanduser().resolve()
    if rmbench_output != roboharn_output:
        raise RendererDeviceContractError(
            "renderer-device output roots disagree: "
            f"{rmbench_output} vs {roboharn_output}"
        )
    runtime_base = (_roboharn_project_root() / "eval_result").resolve()
    if (
        rmbench_output != runtime_base
        and runtime_base not in rmbench_output.parents
    ):
        raise RendererDeviceContractError(
            "renderer-device output must be inside the dedicated RoboHarn-Evo "
            f"eval_result tree: {rmbench_output} vs {runtime_base}"
        )
    for env_name in _READONLY_ROOT_ENVS:
        readonly_raw = str(environment.get(env_name, "") or "").strip()
        if not readonly_raw:
            continue
        readonly = Path(readonly_raw).expanduser().resolve()
        if _paths_overlap(rmbench_output, readonly):
            raise RendererDeviceContractError(
                "renderer-device output must not overlap read-only RMBench "
                f"source/assets root: {rmbench_output} vs {readonly}"
            )
    destination = Path(raw_path).expanduser().resolve()
    expected = (rmbench_output / _PROVENANCE_FILENAME).resolve()
    if destination != expected:
        raise RendererDeviceContractError(
            "renderer-device provenance path must be the output-owned fixed "
            f"file {expected}, got {destination}"
        )
    return destination


@dataclass
class RendererBinding:
    """Renderer construction result retained until the scene is available."""

    mode: str
    renderer: Any
    device: Any | None
    requested_device: str | None
    strict: bool
    environment: Mapping[str, str]
    render_system: Any | None = None


def _parse_bool(value: str | None, *, name: str, default: bool = False) -> bool:
    if value is None or not str(value).strip():
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RendererDeviceContractError(
        f"{name} must be one of 0/1/false/true/no/yes/off/on, got {value!r}"
    )


def _optional_nonnegative_int(value: str | None, *, name: str) -> int | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    if not text.isdigit():
        raise RendererDeviceContractError(
            f"{name} must be a non-negative integer, got {value!r}"
        )
    return int(text)


def _visible_cuda_tokens(value: str | None) -> list[str]:
    if value is None:
        return []
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _normalize_pci_bus_id(value: str | None) -> str | None:
    """Normalize NVIDIA/SAPIEN PCI spellings to ``dddd:bb:dd.f``.

    ``nvidia-smi`` commonly emits an eight-digit domain while SAPIEN emits a
    four-digit domain.  Both represent the same PCI address.
    """

    if value is None or not str(value).strip():
        return None
    text = str(value).strip().lower()
    match = re.fullmatch(
        r"(?P<domain>[0-9a-f]{1,8}):(?P<bus>[0-9a-f]{1,2}):"
        r"(?P<device>[0-9a-f]{1,2})\.(?P<function>[0-7])",
        text,
    )
    if match is None:
        raise RendererDeviceContractError(f"invalid PCI bus ID: {value!r}")
    domain = int(match.group("domain"), 16)
    if domain > 0xFFFF:
        raise RendererDeviceContractError(f"PCI domain is out of range: {value!r}")
    return (
        f"{domain:04x}:{int(match.group('bus'), 16):02x}:"
        f"{int(match.group('device'), 16):02x}.{match.group('function')}"
    )


def _device_info(device: Any | None) -> dict[str, Any] | None:
    if device is None:
        return None

    def property_value(name: str, default: Any = None) -> Any:
        try:
            return getattr(device, name)
        except Exception:
            return default

    def method_value(name: str, default: Any = None) -> Any:
        method = property_value(name)
        if not callable(method):
            return default
        try:
            return method()
        except Exception:
            return default

    pci_string = property_value("pci_string")
    try:
        normalized_pci = _normalize_pci_bus_id(pci_string)
    except RendererDeviceContractError:
        normalized_pci = None
    return {
        "repr": str(device),
        "name": property_value("name"),
        "cuda_id": property_value("cuda_id"),
        "pci_bus_id": pci_string,
        "pci_bus_id_normalized": normalized_pci,
        "is_cuda": method_value("is_cuda"),
        "can_render": method_value("can_render"),
    }


def renderer_contract_failure_provenance(
    error: RendererDeviceContractError,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return an auditable failure record without changing failure semantics."""

    if error.provenance is not None:
        result = dict(error.provenance)
    else:
        environment = os.environ if environ is None else environ
        expected_pci_raw = str(
            environment.get(EXPECTED_RENDER_PCI_BUS_ID_ENV, "") or ""
        ).strip()
        try:
            expected_pci = _normalize_pci_bus_id(expected_pci_raw)
        except RendererDeviceContractError:
            expected_pci = None
        result = {
            "schema": "rmbench/renderer_device_binding/v1",
            "mode": "explicit",
            "strict": True,
            "requested_device": str(
                environment.get(RENDER_DEVICE_ENV, "") or ""
            ).strip()
            or None,
            "cuda_visible_devices": str(
                environment.get("CUDA_VISIBLE_DEVICES", "") or ""
            ),
            "cuda_device_order": str(
                environment.get("CUDA_DEVICE_ORDER", "") or ""
            ).strip()
            or None,
            "expected_logical_cuda_id": str(
                environment.get(EXPECTED_RENDER_CUDA_ID_ENV, "") or ""
            ).strip()
            or None,
            "expected_physical_gpu": str(
                environment.get(EXPECTED_PHYSICAL_GPU_ENV, "") or ""
            ).strip()
            or None,
            "expected_pci_bus_id": expected_pci,
            "validation_passed": False,
            "validation_errors": [str(error)],
            "pid": os.getpid(),
        }
    result["validation_passed"] = False
    result.setdefault("validation_errors", [str(error)])
    result["failure_stage"] = error.stage
    result["failure_type"] = type(error).__name__
    return result


def create_renderer_binding(
    sapien_api: Any,
    engine: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> RendererBinding:
    """Create a renderer while preserving legacy behavior unless opted in."""

    environment = dict(os.environ if environ is None else environ)
    raw_requested = str(environment.get(RENDER_DEVICE_ENV, "") or "").strip()
    if not raw_requested:
        renderer = sapien_api.SapienRenderer()
        engine.set_renderer(renderer)
        return RendererBinding(
            mode="legacy_auto",
            renderer=renderer,
            device=None,
            requested_device=None,
            strict=False,
            environment=environment,
        )

    strict = _parse_bool(
        environment.get(RENDER_DEVICE_STRICT_ENV),
        name=RENDER_DEVICE_STRICT_ENV,
        default=False,
    )
    if strict:
        if raw_requested.startswith("cuda"):
            if raw_requested != "cuda:0":
                raise RendererDeviceContractError(
                    "strict single-GPU rollout binding requires "
                    f"{RENDER_DEVICE_ENV}=cuda:0 inside the CUDA-visible "
                    f"namespace, got {raw_requested!r}"
                )
        elif raw_requested.startswith("pci:"):
            expected_pci_raw = str(
                environment.get(EXPECTED_RENDER_PCI_BUS_ID_ENV, "") or ""
            ).strip()
            expected_pci = _normalize_pci_bus_id(expected_pci_raw)
            if expected_pci is None:
                raise RendererDeviceContractError(
                    "strict PCI renderer binding requires "
                    f"{EXPECTED_RENDER_PCI_BUS_ID_ENV}"
                )
            expected_alias = f"pci:{expected_pci}"
            if raw_requested != expected_alias:
                raise RendererDeviceContractError(
                    "strict PCI renderer alias must exactly match the canonical "
                    f"expected PCI address: requested={raw_requested!r}, "
                    f"expected={expected_alias!r}"
                )
        else:
            raise RendererDeviceContractError(
                "strict renderer binding supports only logical cuda:0 or a "
                f"canonical pci:<bus-id> alias, got {raw_requested!r}"
            )

    # Do not use sapien_api.SapienRenderer here.  In SAPIEN 3.0.0b1 that
    # compatibility wrapper accepts **kwargs but discards the device.  The
    # lower-level class in sapien.render honors the Device argument.
    try:
        device = sapien_api.Device(raw_requested)
        renderer = sapien_api.render.SapienRenderer(device)
        engine.set_renderer(renderer)
    except Exception as exc:
        raise RendererDeviceContractError(
            f"explicit SAPIEN renderer construction failed for {raw_requested!r}: {exc}",
            stage="renderer_construction",
        ) from exc
    return RendererBinding(
        mode="explicit",
        renderer=renderer,
        device=device,
        requested_device=raw_requested,
        strict=strict,
        environment=environment,
    )


def create_scene_for_binding(
    sapien_api: Any,
    engine: Any,
    scene_config: Any,
    binding: RendererBinding,
) -> Any:
    """Create a scene whose RenderSystem shares the explicitly selected device."""

    if binding.mode == "legacy_auto":
        return engine.create_scene(scene_config)

    # This reproduces Engine.create_scene's SceneConfig behavior before
    # replacing only the default RenderSystem with a device-bound instance.
    try:
        sapien_api.physx.set_scene_config(scene_config)
        physx_system = sapien_api.physx.PhysxCpuSystem()
        render_system = sapien_api.render.RenderSystem(binding.device)
        scene = sapien_api.Scene([physx_system, render_system])
        binding.render_system = render_system
        return scene
    except Exception as exc:
        raise RendererDeviceContractError(
            "explicit SAPIEN scene construction failed on the requested render device: "
            f"{exc}",
            stage="scene_construction",
        ) from exc


def validate_renderer_binding(binding: RendererBinding, scene: Any) -> dict[str, Any]:
    """Inspect the selected scene device and optionally enforce expectations."""

    environment = binding.environment
    if binding.mode == "legacy_auto":
        return {
            "schema": "rmbench/renderer_device_binding/v1",
            "mode": "legacy_auto",
            "strict": False,
            "validation_passed": True,
        }

    try:
        selected_render_system = scene.render_system
        selected_device = selected_render_system.device
    except Exception as exc:
        raise RendererDeviceContractError(
            "explicit renderer binding could not inspect scene.render_system.device"
        ) from exc

    requested_info = _device_info(binding.device)
    selected_info = _device_info(selected_device)
    visible_devices = str(environment.get("CUDA_VISIBLE_DEVICES", "") or "")
    visible_tokens = _visible_cuda_tokens(visible_devices)
    expected_cuda_id = _optional_nonnegative_int(
        environment.get(EXPECTED_RENDER_CUDA_ID_ENV),
        name=EXPECTED_RENDER_CUDA_ID_ENV,
    )
    requested_device = str(binding.requested_device)
    explicit_gpu_device = requested_device.startswith(("cuda", "pci:"))
    if expected_cuda_id is None and explicit_gpu_device:
        expected_cuda_id = 0
    expected_pci_raw = str(
        environment.get(EXPECTED_RENDER_PCI_BUS_ID_ENV, "") or ""
    ).strip()
    expected_pci = _normalize_pci_bus_id(expected_pci_raw)
    selected_pci = None if selected_info is None else selected_info.get(
        "pci_bus_id_normalized"
    )

    errors: list[str] = []
    if binding.render_system is None or selected_render_system is not binding.render_system:
        errors.append(
            "the SAPIEN scene did not retain the explicitly constructed RenderSystem"
        )
    if explicit_gpu_device:
        if len(visible_tokens) != 1:
            errors.append(
                "explicit GPU renderer binding requires exactly one "
                f"CUDA_VISIBLE_DEVICES entry, got {visible_tokens!r}"
            )
        elif (
            binding.strict
            and requested_device.startswith("pci:")
            and not visible_tokens[0].startswith("GPU-")
        ):
            errors.append(
                "strict PCI renderer binding requires CUDA_VISIBLE_DEVICES "
                f"to contain one immutable GPU UUID, got {visible_tokens[0]!r}"
            )
        if requested_info is None or requested_info.get("is_cuda") is not True:
            errors.append("the requested SAPIEN render device is not CUDA-capable")
        if selected_info is None or selected_info.get("is_cuda") is not True:
            errors.append("the selected SAPIEN render device is not CUDA-capable")
        if expected_cuda_id is not None and (
            requested_info is None
            or requested_info.get("cuda_id") != expected_cuda_id
        ):
            actual = (
                None if requested_info is None else requested_info.get("cuda_id")
            )
            errors.append(
                f"requested logical CUDA ID {actual!r} does not match "
                f"expected {expected_cuda_id}"
            )
        if expected_cuda_id is not None and (
            selected_info is None or selected_info.get("cuda_id") != expected_cuda_id
        ):
            actual = None if selected_info is None else selected_info.get("cuda_id")
            errors.append(
                f"selected logical CUDA ID {actual!r} does not match "
                f"expected {expected_cuda_id}"
            )
        requested_cuda_id = (
            None if requested_info is None else requested_info.get("cuda_id")
        )
        selected_cuda_id = (
            None if selected_info is None else selected_info.get("cuda_id")
        )
        if requested_cuda_id != selected_cuda_id:
            errors.append(
                "requested and selected renderer CUDA IDs differ: "
                f"requested={requested_cuda_id!r}, selected={selected_cuda_id!r}"
            )
    if requested_info is None or requested_info.get("can_render") is not True:
        errors.append("the requested SAPIEN device cannot render")
    if selected_info is None or selected_info.get("can_render") is not True:
        errors.append("the selected SAPIEN device cannot render")
    requested_pci = None if requested_info is None else requested_info.get(
        "pci_bus_id_normalized"
    )
    if expected_pci is not None and requested_pci != expected_pci:
        errors.append(
            f"requested renderer PCI address {requested_pci!r} does not match "
            f"expected {expected_pci!r}"
        )
    if expected_pci is not None and selected_pci != expected_pci:
        errors.append(
            f"selected renderer PCI address {selected_pci!r} does not match "
            f"expected {expected_pci!r}"
        )
    if requested_pci != selected_pci:
        errors.append(
            "requested and selected renderer PCI addresses differ: "
            f"requested={requested_pci!r}, selected={selected_pci!r}"
        )

    provenance = {
        "schema": "rmbench/renderer_device_binding/v1",
        "mode": binding.mode,
        "strict": binding.strict,
        "requested_device": binding.requested_device,
        "cuda_visible_devices": visible_devices,
        "cuda_device_order": str(
            environment.get("CUDA_DEVICE_ORDER", "") or ""
        ).strip()
        or None,
        "cuda_visible_device_tokens": visible_tokens,
        "expected_logical_cuda_id": expected_cuda_id,
        "expected_physical_gpu": str(
            environment.get(EXPECTED_PHYSICAL_GPU_ENV, "") or ""
        ).strip()
        or None,
        "expected_pci_bus_id": expected_pci,
        "requested_device_info": requested_info,
        "selected_device_info": selected_info,
        "validation_passed": not errors,
        "validation_errors": errors,
        "pid": os.getpid(),
    }
    if errors and binding.strict:
        raise RendererDeviceContractError(
            "strict renderer-device validation failed: " + "; ".join(errors),
            stage="device_validation",
            provenance=provenance,
        )
    return provenance


def write_renderer_device_provenance(
    provenance: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Atomically write optional infrastructure provenance outside Agent logs."""

    environment = os.environ if environ is None else environ
    raw_path = str(
        environment.get(RENDER_DEVICE_PROVENANCE_PATH_ENV, "") or ""
    ).strip()
    if not raw_path:
        return None
    destination = _validated_provenance_destination(
        raw_path,
        environment=environment,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(provenance), ensure_ascii=False, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RendererDeviceContractError(
            f"could not write renderer-device provenance to {destination}: {exc}"
        ) from exc
    return destination
