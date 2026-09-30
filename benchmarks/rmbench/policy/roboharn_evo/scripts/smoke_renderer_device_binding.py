#!/usr/bin/env python3
"""Real-GPU smoke test for RMBench's explicit SAPIEN renderer binding.

The parent process resolves a physical NVIDIA GPU to an immutable UUID/PCI
record before it starts a clean child process.  Only the child imports SAPIEN
and ``envs._base_task``.  This ordering is important: Vulkan/CUDA device
enumeration may be initialized at import time, so setting the environment in
the already-imported process is not a meaningful isolation test.

This script intentionally does not start an Agent, GPT backend, or SAM server.
It creates the normal RMBench base scene, renders one 64x64 color frame, and
checks the renderer provenance written by the production device contract.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "rmbench/renderer_device_smoke/v1"
NVIDIA_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,uuid,pci.bus_id",
    "--format=csv,noheader,nounits",
)
RENDER_ENV_NAMES = (
    "RMBENCH_RENDER_DEVICE",
    "RMBENCH_RENDER_DEVICE_STRICT",
    "RMBENCH_EXPECTED_RENDER_CUDA_ID",
    "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID",
    "RMBENCH_EXPECTED_PHYSICAL_GPU",
    "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH",
)


class SmokeError(RuntimeError):
    """Expected infrastructure or validation failure."""


def _normalize_pci_bus_id(value: str) -> str:
    text = str(value).strip().lower()
    match = re.fullmatch(
        r"(?P<domain>[0-9a-f]{1,8}):(?P<bus>[0-9a-f]{1,2}):"
        r"(?P<device>[0-9a-f]{1,2})\.(?P<function>[0-7])",
        text,
    )
    if match is None:
        raise SmokeError(f"invalid NVIDIA PCI bus ID: {value!r}")
    domain = int(match.group("domain"), 16)
    if domain > 0xFFFF:
        raise SmokeError(f"PCI domain is out of range: {value!r}")
    return (
        f"{domain:04x}:{int(match.group('bus'), 16):02x}:"
        f"{int(match.group('device'), 16):02x}.{match.group('function')}"
    )


def parse_gpu_inventory(output: str) -> dict[int, dict[str, Any]]:
    """Parse and validate the exact inventory returned by ``nvidia-smi``."""

    inventory: dict[int, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(str(output).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise SmokeError(
                f"malformed nvidia-smi inventory line {line_number}: {raw_line!r}"
            )
        index_text, uuid, pci_bus_id_raw = fields
        if not index_text.isdigit():
            raise SmokeError(
                f"invalid GPU index on inventory line {line_number}: {index_text!r}"
            )
        index = int(index_text)
        if index in inventory:
            raise SmokeError(f"duplicate GPU index {index} in nvidia-smi inventory")
        if not uuid or uuid.lower() in {"n/a", "[n/a]"}:
            raise SmokeError(f"GPU {index} has no usable UUID")
        pci_bus_id = _normalize_pci_bus_id(pci_bus_id_raw)
        inventory[index] = {
            "index": index,
            "uuid": uuid,
            "pci_bus_id": pci_bus_id,
            "pci_bus_id_raw": pci_bus_id_raw,
        }
    if not inventory:
        raise SmokeError("nvidia-smi returned an empty GPU inventory")
    return inventory


def query_gpu_inventory() -> dict[int, dict[str, Any]]:
    try:
        completed = subprocess.run(
            list(NVIDIA_QUERY),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SmokeError(f"could not execute nvidia-smi: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise SmokeError(
            f"nvidia-smi inventory query failed with exit code "
            f"{completed.returncode}: {detail[-1000:]}"
        )
    return parse_gpu_inventory(completed.stdout)


def build_child_environment(
    parent_environment: Mapping[str, str],
    *,
    gpu: Mapping[str, Any],
    provenance_path: Path | None,
) -> dict[str, str]:
    """Return a single-GPU environment containing the six contract fields."""

    environment = dict(parent_environment)
    index = int(gpu["index"])
    gpu_uuid = str(gpu.get("uuid", "")).strip()
    if not gpu_uuid:
        raise SmokeError(f"physical GPU {index} has no usable UUID")
    pci_bus_id = _normalize_pci_bus_id(str(gpu["pci_bus_id"]))
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            # UUID binding does not depend on CUDA's numeric device ordering.
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "RMBENCH_RENDER_DEVICE": f"pci:{pci_bus_id}",
            "RMBENCH_RENDER_DEVICE_STRICT": "1",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": pci_bus_id,
            "RMBENCH_EXPECTED_PHYSICAL_GPU": str(index),
            "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH": (
                "" if provenance_path is None else str(provenance_path)
            ),
        }
    )
    return environment


def _child_command(
    args: argparse.Namespace,
    *,
    gpu: Mapping[str, Any],
    provenance_path: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        "--gpu-index",
        str(int(gpu["index"])),
        "--expected-gpu-uuid",
        str(gpu["uuid"]),
        "--expected-pci-bus-id",
        str(gpu["pci_bus_id"]),
    ]
    if provenance_path is not None:
        command.extend(["--provenance-path", str(provenance_path)])
    return command


def _extract_child_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(str(stdout).splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("schema") == SCHEMA:
            return candidate
    return None


def run_parent(args: argparse.Namespace) -> int:
    if args.gpu_index < 0:
        raise SmokeError("--gpu-index must be non-negative")
    if not math.isfinite(args.timeout_sec) or args.timeout_sec <= 0:
        raise SmokeError("--timeout-sec must be a positive finite value")
    inventory = query_gpu_inventory()
    if args.gpu_index not in inventory:
        raise SmokeError(
            f"requested physical GPU {args.gpu_index} is absent; "
            f"available indices are {sorted(inventory)}"
        )
    gpu = inventory[args.gpu_index]
    provenance_path = (
        None
        if args.provenance_path is None
        else args.provenance_path.expanduser().resolve()
    )
    child_environment = build_child_environment(
        os.environ,
        gpu=gpu,
        provenance_path=provenance_path,
    )
    try:
        completed = subprocess.run(
            _child_command(args, gpu=gpu, provenance_path=provenance_path),
            check=False,
            capture_output=True,
            text=True,
            env=child_environment,
            timeout=args.timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise SmokeError(
            f"renderer smoke child exceeded {args.timeout_sec:g} seconds"
        ) from exc
    except OSError as exc:
        raise SmokeError(f"could not start renderer smoke child: {exc}") from exc

    result = _extract_child_result(completed.stdout)
    if result is None:
        raise SmokeError(
            "renderer smoke child produced no structured result; "
            f"exit_code={completed.returncode}, "
            f"stdout_tail={completed.stdout[-1000:]!r}, "
            f"stderr_tail={completed.stderr[-1000:]!r}"
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if completed.returncode != 0 or result.get("ok") is not True:
        return completed.returncode if completed.returncode != 0 else 1
    return 0


def _validate_child_environment(args: argparse.Namespace) -> None:
    expected_pci = _normalize_pci_bus_id(args.expected_pci_bus_id)
    expected = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(args.expected_gpu_uuid),
        "RMBENCH_RENDER_DEVICE": f"pci:{expected_pci}",
        "RMBENCH_RENDER_DEVICE_STRICT": "1",
        "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
        "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": expected_pci,
        "RMBENCH_EXPECTED_PHYSICAL_GPU": str(args.gpu_index),
        "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH": (
            ""
            if args.provenance_path is None
            else str(args.provenance_path.expanduser().resolve())
        ),
    }
    errors = [
        f"{name}={os.environ.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if os.environ.get(name) != value
    ]
    if errors:
        raise SmokeError("child environment contract mismatch: " + "; ".join(errors))


def _run_child(args: argparse.Namespace) -> dict[str, Any]:
    """Import RMBench after isolation, render one real frame, and validate it."""

    _validate_child_environment(args)
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    # These imports intentionally occur only after the child environment was
    # validated.  In particular, importing Base_Task imports SAPIEN and torch.
    import numpy as np
    import sapien.core as sapien
    from sapien.render import clear_cache as sapien_clear_cache

    from envs._base_task import Base_Task

    task: Any | None = None
    camera: Any | None = None
    try:
        task = Base_Task.__new__(Base_Task)
        task.random_light = False
        task.render_freq = 0
        # Suppress Base_Task's verbose provenance print; this smoke emits one
        # concise, machine-readable JSON record of its own.
        with contextlib.redirect_stdout(io.StringIO()):
            task.setup_scene()

        camera = task.scene.add_camera(
            name="renderer_device_smoke_camera",
            width=64,
            height=64,
            fovy=math.radians(60.0),
            near=0.01,
            far=10.0,
        )
        camera_pose = np.eye(4, dtype=np.float64)
        camera_pose[:3, 0] = [0.0, 0.0, -1.0]  # camera forward
        camera_pose[:3, 1] = [0.0, 1.0, 0.0]   # camera left
        camera_pose[:3, 2] = [1.0, 0.0, 0.0]   # camera up
        camera_pose[:3, 3] = [0.0, 0.0, 1.0]
        camera.entity.set_pose(sapien.Pose(camera_pose))
        task.scene.step()
        task.scene.update_render()
        camera.take_picture()
        color = np.asarray(camera.get_picture("Color"))
        if color.ndim != 3 or color.shape[:2] != (64, 64) or color.shape[2] < 3:
            raise SmokeError(f"unexpected Color frame shape: {list(color.shape)}")
        if color.size == 0 or not bool(np.isfinite(color).all()):
            raise SmokeError("Color frame is empty or contains non-finite values")

        provenance = dict(task.renderer_device_provenance)
        requested = dict(provenance.get("requested_device_info") or {})
        selected = dict(provenance.get("selected_device_info") or {})
        expected_pci = _normalize_pci_bus_id(args.expected_pci_bus_id)
        expected_alias = f"pci:{expected_pci}"
        requested_pci = requested.get("pci_bus_id_normalized")
        selected_pci = selected.get("pci_bus_id_normalized")
        if provenance.get("validation_passed") is not True:
            raise SmokeError(f"renderer provenance did not validate: {provenance!r}")
        if provenance.get("requested_device") != expected_alias:
            raise SmokeError(
                "renderer requested the wrong device alias: "
                f"{provenance.get('requested_device')!r}, expected {expected_alias!r}"
            )
        if requested.get("cuda_id") != 0:
            raise SmokeError(
                "requested renderer logical CUDA ID "
                f"{requested.get('cuda_id')!r}, expected 0"
            )
        if selected.get("cuda_id") != 0:
            raise SmokeError(
                f"renderer selected logical CUDA ID {selected.get('cuda_id')!r}, expected 0"
            )
        if requested_pci != expected_pci:
            raise SmokeError(
                f"requested renderer PCI {requested_pci!r}, expected {expected_pci!r}"
            )
        if selected_pci != expected_pci:
            raise SmokeError(
                f"renderer selected PCI {selected_pci!r}, expected {expected_pci!r}"
            )

        provenance_path = (
            None
            if args.provenance_path is None
            else args.provenance_path.expanduser().resolve()
        )
        if provenance_path is not None:
            if not provenance_path.is_file():
                raise SmokeError(f"provenance file was not written: {provenance_path}")
            on_disk = json.loads(provenance_path.read_text(encoding="utf-8"))
            if on_disk.get("validation_passed") is not True:
                raise SmokeError("on-disk renderer provenance did not validate")
            disk_selected = dict(on_disk.get("selected_device_info") or {})
            disk_requested = dict(on_disk.get("requested_device_info") or {})
            if on_disk.get("requested_device") != expected_alias:
                raise SmokeError("on-disk provenance has the wrong requested alias")
            if disk_requested.get("pci_bus_id_normalized") != expected_pci:
                raise SmokeError("on-disk requested renderer has the wrong PCI address")
            if disk_selected.get("pci_bus_id_normalized") != expected_pci:
                raise SmokeError("on-disk renderer provenance has the wrong PCI address")

        return {
            "schema": SCHEMA,
            "ok": True,
            "physical_gpu_index": args.gpu_index,
            "gpu_uuid": args.expected_gpu_uuid,
            "expected_pci_bus_id": expected_pci,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "renderer": {
                "mode": provenance.get("mode"),
                "requested_device": provenance.get("requested_device"),
                "logical_cuda_id": selected.get("cuda_id"),
                "pci_bus_id": selected_pci,
                "validation_passed": True,
            },
            "frame": {
                "shape": list(color.shape),
                "dtype": str(color.dtype),
                "finite": True,
                "min": float(color.min()),
                "max": float(color.max()),
            },
            "provenance_path": (
                None if provenance_path is None else str(provenance_path)
            ),
        }
    finally:
        # Release SAPIEN resources in dependency order.  Some SAPIEN builds do
        # not expose close() on Scene/Engine, so dropping references is the
        # portable cleanup path.
        camera = None
        if task is not None:
            for attribute in ("scene", "renderer", "engine"):
                try:
                    setattr(task, attribute, None)
                except Exception:
                    pass
        task = None
        gc.collect()
        try:
            sapien_clear_cache()
        except Exception:
            pass


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one real SAPIEN frame and verify exact GPU binding."
    )
    parser.add_argument("--gpu-index", type=int, default=4)
    parser.add_argument("--provenance-path", type=Path)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-gpu-uuid", help=argparse.SUPPRESS)
    parser.add_argument("--expected-pci-bus-id", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args._child:
            if not args.expected_gpu_uuid or not args.expected_pci_bus_id:
                raise SmokeError("internal child invocation lacks GPU UUID/PCI identity")
            result = _run_child(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            return 0
        return run_parent(args)
    except Exception as exc:
        result = {
            "schema": SCHEMA,
            "ok": False,
            "stage": "child" if args._child else "parent",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "physical_gpu_index": args.gpu_index,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
