from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import types
from typing import Any, Sequence

import numpy as np
from PIL import Image

from roboharn_evo.agent.paths import eval_result_dir
from roboharn_evo.services.http_safety import (
    RequestBodyError,
    ROBOHARN_EVAL_RESULT_ROOT,
    ROBOHARN_PROJECT_ROOT,
    add_allow_remote_argument,
    read_json_object_body,
    readonly_rmbench_boundaries,
    reject_path_overlaps,
    require_path_within,
    validate_bind_host,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9301
DEFAULT_MAX_REQUEST_BODY_BYTES = 1024 * 1024
DEFAULT_MAX_TOP_K = 64
DEFAULT_CACHE_ROOT = ROBOHARN_EVAL_RESULT_ROOT / "service_caches" / "sam3"
_CACHE_ENVIRONMENT_LAYOUT = {
    "HF_HOME": "huggingface",
    "HF_HUB_CACHE": "huggingface/hub",
    "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
    "TRANSFORMERS_CACHE": "huggingface/transformers",
    "TORCH_HOME": "torch",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda",
    "XDG_CACHE_HOME": "xdg",
}


@dataclass(frozen=True)
class SAM3ExternalResources:
    """Validated, operator-supplied SAM3 code and model resources."""

    repo: Path
    checkpoint: Path
    bpe_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve SAM3 text-prompt image segmentation over HTTP."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    add_allow_remote_argument(parser)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--sam3-repo",
        required=True,
        help="Path to an external SAM3 source checkout containing sam3/.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to an external SAM3 checkpoint file.",
    )
    parser.add_argument(
        "--bpe-path",
        required=True,
        help="Path to the external SAM3 BPE vocabulary file.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "none"),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-root",
        default=str(eval_result_dir("roboharn_sam3_segmentation")),
        help="Writable default directory for generated masks.",
    )
    parser.add_argument(
        "--allowed-output-root",
        default="",
        help=(
            "Service write boundary. Requests whose output_dir is outside this "
            "directory are rejected. Defaults to the resolved --output-root."
        ),
    )
    parser.add_argument(
        "--allowed-input-root",
        action="append",
        required=True,
        help=(
            "Existing shared directory from which image_path may be read. Repeat "
            "the option to allow multiple independent producer directories."
        ),
    )
    parser.add_argument("--additional-output-root", action="append", default=[], help="Additional independent mask output directory; repeat for parallel jobs.")
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help=(
            "Writable cache boundary for SAM3 dependencies. Must resolve below "
            "this RoboHarn-Evo project's eval_result/ and remain disjoint from inputs and "
            "read-only roots."
        ),
    )
    parser.add_argument(
        "--instance-id",
        default="",
        help="Optional orchestration identity reported by /health.",
    )
    parser.add_argument(
        "--check-paths-only",
        action="store_true",
        help=(
            "validate external resources and read/write boundaries, then exit "
            "without importing SAM3, loading the checkpoint, or binding a port"
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _resolve_within(
    value: str | Path,
    *,
    allowed_roots: Sequence[str | Path],
    field_name: str,
) -> Path:
    resolved = _resolved_path(value)
    roots = tuple(_resolved_path(root) for root in allowed_roots)
    if not roots:
        raise RuntimeError(f"{field_name} has no configured allowed root")
    for root in roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    rendered_roots = ", ".join(str(root) for root in roots)
    raise ValueError(
        f"{field_name} must resolve within an allowed root: {rendered_roots}"
    )


def validate_io_roots(
    *,
    output_root: str | Path,
    allowed_output_root: str | Path | None,
    allowed_input_roots: Sequence[str | Path],
    project_root: str | Path | None = None,
    protected_roots: Sequence[tuple[str, str | Path]] = (),
) -> tuple[Path, Path, tuple[Path, ...]]:
    resolved_project_root = _resolved_path(project_root or ROBOHARN_PROJECT_ROOT)
    resolved_runtime_root = resolved_project_root / "eval_result"
    resolved_output_root = require_path_within(
        output_root,
        root=resolved_runtime_root,
        label="output_root",
    )
    resolved_allowed_output_root = require_path_within(
        allowed_output_root
        if allowed_output_root is not None and str(allowed_output_root).strip()
        else resolved_output_root,
        root=resolved_runtime_root,
        label="allowed_output_root",
    )
    try:
        resolved_output_root.relative_to(resolved_allowed_output_root)
    except ValueError as exc:
        raise RuntimeError(
            "output_root must resolve within allowed_output_root: "
            f"output_root={resolved_output_root}, "
            f"allowed_output_root={resolved_allowed_output_root}"
        ) from exc

    if not allowed_input_roots:
        raise RuntimeError("at least one allowed_input_root is required")
    resolved_input_roots: list[Path] = []
    for value in allowed_input_roots:
        root = _resolved_path(value)
        if not root.is_dir():
            raise RuntimeError(f"allowed_input_root is not an existing directory: {root}")
        if root not in resolved_input_roots:
            resolved_input_roots.append(root)

    write_boundaries: list[tuple[str, str | Path]] = [
        *readonly_rmbench_boundaries(),
        *protected_roots,
        *(
            (f"allowed_input_root[{index}]", root)
            for index, root in enumerate(resolved_input_roots)
        ),
    ]
    reject_path_overlaps(
        resolved_output_root,
        boundaries=write_boundaries,
        label="output_root",
    )
    reject_path_overlaps(
        resolved_allowed_output_root,
        boundaries=write_boundaries,
        label="allowed_output_root",
    )
    return (
        resolved_output_root,
        resolved_allowed_output_root,
        tuple(resolved_input_roots),
    )


def additional_output_roots(args, *, project_root=None, protected_roots=()) -> tuple[Path, ...]:
    return tuple(validate_io_roots(output_root=root, allowed_output_root=root, allowed_input_roots=args.allowed_input_root, project_root=project_root, protected_roots=protected_roots)[0] for root in args.additional_output_root)


def configure_sam3_cache_root(
    value: str | Path,
    *,
    project_root: str | Path | None = None,
    protected_roots: Sequence[tuple[str, str | Path]] = (),
) -> tuple[Path, dict[str, str]]:
    """将服务缓存设置到 RoboHarn-Evo 的缓存目录。"""

    resolved_project_root = _resolved_path(project_root or ROBOHARN_PROJECT_ROOT)
    resolved_runtime_root = resolved_project_root / "eval_result"
    cache_root = require_path_within(
        value,
        root=resolved_runtime_root,
        label="cache_root",
    )
    reject_path_overlaps(
        cache_root,
        boundaries=[*readonly_rmbench_boundaries(), *protected_roots],
        label="cache_root",
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    if not cache_root.is_dir():
        raise NotADirectoryError(f"SAM3 cache root is not a directory: {cache_root}")

    cache_environment = {
        variable: str(cache_root / relative_path)
        for variable, relative_path in _CACHE_ENVIRONMENT_LAYOUT.items()
    }
    os.environ.update(cache_environment)
    return cache_root, cache_environment


def validate_external_resources(
    *, sam3_repo: str | Path, checkpoint: str | Path, bpe_path: str | Path
) -> SAM3ExternalResources:
    repo = _resolved_path(sam3_repo)
    checkpoint_path = _resolved_path(checkpoint)
    bpe_file = _resolved_path(bpe_path)
    if not repo.is_dir():
        raise RuntimeError(f"SAM3 repo directory does not exist: {repo}")
    if not (repo / "sam3").is_dir():
        raise RuntimeError(f"SAM3 repo must contain a sam3 package directory: {repo}")
    if not checkpoint_path.is_file():
        raise RuntimeError(f"SAM3 checkpoint file does not exist: {checkpoint_path}")
    if not bpe_file.is_file():
        raise RuntimeError(f"SAM3 BPE vocabulary file does not exist: {bpe_file}")
    return SAM3ExternalResources(
        repo=repo,
        checkpoint=checkpoint_path,
        bpe_path=bpe_file,
    )


def add_sam3_repo_to_import_path(repo: Path) -> None:
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)


def ensure_pkg_resources_compat() -> None:
    try:
        import pkg_resources  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("pkg_resources")

    def resource_filename(package: str, resource: str) -> str:
        return str(resources.files(package).joinpath(resource))

    module.resource_filename = resource_filename
    sys.modules["pkg_resources"] = module


def _optional_torch() -> Any | None:
    try:
        import torch
    except (ImportError, OSError):
        return None
    return torch


def _require_torch() -> Any:
    torch = _optional_torch()
    if torch is None:
        raise RuntimeError(
            "PyTorch is required by the SAM3 service; start it in an environment "
            "that provides torch and the external SAM3 dependencies"
        )
    return torch


def resolve_device(requested: str) -> str:
    torch = _require_torch()
    device = str(requested).strip()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device} was requested, but torch.cuda.is_available() is false"
        )
    return device


def load_processor(
    *, checkpoint: Path, bpe_path: Path, device: str, confidence_threshold: float
) -> Any:
    _require_torch()
    ensure_pkg_resources_compat()
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device=device,
    )
    return Sam3Processor(
        model,
        device=device,
        confidence_threshold=float(confidence_threshold),
    )


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def safe_name(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in value
    ).strip("_")
    return normalized or "object"


def mask_summary(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask).astype(bool)
    ys, xs = np.where(binary)
    area = int(binary.sum())
    if area == 0:
        return {
            "area_px": 0,
            "bbox_xyxy": [],
            "centroid_px": [],
            "mask_shape": list(binary.shape),
        }
    return {
        "area_px": area,
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_px": [round(float(xs.mean()), 3), round(float(ys.mean()), 3)],
        "mask_shape": list(binary.shape),
    }


def save_mask(mask: np.ndarray, output_dir: Path, object_id: str, rank: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{safe_name(object_id)}_{rank:02d}_{time.time_ns()}.png"
    )
    Image.fromarray(
        np.asarray(mask).astype(np.uint8) * 255,
        mode="L",
    ).save(output_path)
    return output_path


def tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,))
    torch = _optional_torch()
    if torch is not None and isinstance(value, torch.Tensor):
        if value.dtype in {torch.bfloat16, torch.float16}:
            value = value.to(dtype=torch.float32)
        return value.detach().cpu().numpy()
    return np.asarray(value)


def inference_mode_context() -> Any:
    torch = _optional_torch()
    return torch.inference_mode() if torch is not None else nullcontext()


def autocast_context(device: str, amp_dtype: str) -> Any:
    torch = _optional_torch()
    if torch is None or not device.startswith("cuda") or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


class SAM3SegmentationHandler(BaseHTTPRequestHandler):
    processor: Any = None
    output_root = eval_result_dir("roboharn_sam3_segmentation")
    allowed_output_root = output_root
    additional_output_roots: tuple[Path, ...] = ()
    allowed_input_roots: tuple[Path, ...] = ()
    project_root = ROBOHARN_PROJECT_ROOT
    runtime_root = ROBOHARN_EVAL_RESULT_ROOT
    cache_root = DEFAULT_CACHE_ROOT
    protected_read_roots: tuple[Path, ...] = ()
    sam3_repo = ""
    checkpoint = ""
    bpe_path = ""
    device = ""
    amp_dtype = "bfloat16"
    confidence_threshold = 0.5
    instance_id = ""
    inference_lock = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return
        torch = _optional_torch()
        cuda_available = bool(torch is not None and torch.cuda.is_available())
        cuda_current_device = None
        if self.device.startswith("cuda") and cuda_available:
            cuda_current_device = int(torch.cuda.current_device())
        self._send_json(
            {
                "status": "ok",
                "backend": "sam3_image_text_prompt",
                "sam3_repo": self.sam3_repo,
                "checkpoint": self.checkpoint,
                "bpe_path": self.bpe_path,
                "device": self.device,
                "amp_dtype": self.amp_dtype,
                "confidence_threshold": self.confidence_threshold,
                "cuda_available": cuda_available,
                "cuda_current_device": cuda_current_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", ""),
                "instance_id": self.instance_id,
                "pid": os.getpid(),
                "output_root": str(self.output_root),
                "allowed_output_root": (
                    str(self.allowed_output_root)
                ),
                "allowed_output_roots": [str(self.allowed_output_root), *(str(root) for root in self.additional_output_roots)],
                "allowed_input_roots": [
                    str(root) for root in self.allowed_input_roots
                ],
                "project_root": str(self.project_root),
                "runtime_root": str(self.runtime_root),
                "cache_root": str(self.cache_root),
                "protected_read_roots": [
                    str(root) for root in self.protected_read_roots
                ],
            }
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/segment_image":
            self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return
        self._handle_segment_image()

    def _handle_segment_image(self) -> None:
        if self.processor is None:
            self.send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "processor is not initialized",
            )
            return
        try:
            payload = self._read_json()
            result = self._segment_image(payload)
        except RequestBodyError as exc:
            self._send_json(
                {"success": False, "error": str(exc)},
                status=exc.status,
            )
            return
        except Exception as exc:
            self._send_json(
                {"success": False, "error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        self._send_json(result)

    def _segment_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_path = _resolve_within(
            str(payload["image_path"]),
            allowed_roots=self.allowed_input_roots,
            field_name="image_path",
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"image_path does not exist: {image_path}")
        text_prompt = str(payload.get("text_prompt", "")).strip()
        if not text_prompt:
            raise ValueError("text_prompt is required")
        object_id = str(payload.get("object_id", text_prompt)).strip() or text_prompt
        top_k = int(payload.get("top_k", 3))
        if not 1 <= top_k <= DEFAULT_MAX_TOP_K:
            raise ValueError(f"top_k must be within [1, {DEFAULT_MAX_TOP_K}]")
        allowed_output_root = _resolved_path(
            self.allowed_output_root or self.output_root
        )
        output_root = _resolve_within(
            self.output_root,
            allowed_roots=(allowed_output_root,),
            field_name="output_root",
        )
        output_dir = _resolve_within(
            payload.get("output_dir") or output_root,
            allowed_roots=(allowed_output_root, *self.additional_output_roots),
            field_name="output_dir",
        )

        image = load_rgb(image_path)
        with self.inference_lock:
            request_threshold = float(
                payload.get("confidence_threshold", self.confidence_threshold)
            )
            if not math.isfinite(request_threshold) or not 0.0 <= request_threshold <= 1.0:
                raise ValueError(
                    "confidence_threshold must be finite and within [0, 1]"
                )
            self.processor.confidence_threshold = request_threshold
            try:
                with inference_mode_context(), autocast_context(
                    self.device, self.amp_dtype
                ):
                    state = self.processor.set_image(image)
                    state = self.processor.set_text_prompt(
                        state=state,
                        prompt=text_prompt,
                    )
                masks = tensor_to_numpy(state.get("masks"))
                boxes = tensor_to_numpy(state.get("boxes"))
                scores = tensor_to_numpy(state.get("scores")).reshape(-1)
            finally:
                self.processor.confidence_threshold = self.confidence_threshold

        if masks.size == 0 or scores.size == 0:
            return {
                "success": True,
                "object_id": object_id,
                "text_prompt": text_prompt,
                "image_path": str(image_path),
                "detections": [],
                "num_detections": 0,
            }
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = np.squeeze(masks, axis=1)
        elif masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        order = np.argsort(-scores)[:top_k]
        detections: list[dict[str, Any]] = []
        for rank, index in enumerate(order):
            mask = masks[int(index)].astype(bool)
            mask_path = save_mask(mask, output_dir, object_id, rank)
            box = boxes[int(index)].reshape(-1).tolist() if boxes.size else []
            detections.append(
                {
                    "rank": rank,
                    "score": round(float(scores[int(index)]), 6),
                    "box_xyxy": [round(float(item), 3) for item in box[:4]],
                    "mask_path": str(mask_path),
                    **mask_summary(mask),
                }
            )
        best = detections[0] if detections else {}
        return {
            "success": True,
            "object_id": object_id,
            "text_prompt": text_prompt,
            "image_path": str(image_path),
            "num_detections": int(scores.size),
            "detections": detections,
            "mask_path": best.get("mask_path", ""),
            "bbox_xyxy": best.get("bbox_xyxy", best.get("box_xyxy", [])),
            "centroid_px": best.get("centroid_px", []),
            "area_px": best.get("area_px", 0),
            "score": best.get("score"),
        }

    def _read_json(self) -> dict[str, Any]:
        return read_json_object_body(
            headers=self.headers,
            stream=self.rfile,
            max_bytes=DEFAULT_MAX_REQUEST_BODY_BYTES,
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def configure_handler(
    args: argparse.Namespace,
    *,
    project_root: str | Path | None = None,
) -> type[SAM3SegmentationHandler]:
    sys.dont_write_bytecode = True
    resolved_project_root = _resolved_path(project_root or ROBOHARN_PROJECT_ROOT)
    resources_config = validate_external_resources(
        sam3_repo=args.sam3_repo,
        checkpoint=args.checkpoint,
        bpe_path=args.bpe_path,
    )
    protected_read_roots: tuple[tuple[str, str | Path], ...] = (
        *readonly_rmbench_boundaries(),
        ("selected SAM3 source", resources_config.repo),
        ("selected SAM3 checkpoint", resources_config.checkpoint),
        ("selected SAM3 BPE vocabulary", resources_config.bpe_path),
    )
    output_root, allowed_output_root, allowed_input_roots = validate_io_roots(
        output_root=args.output_root,
        allowed_output_root=args.allowed_output_root,
        allowed_input_roots=args.allowed_input_root,
        project_root=resolved_project_root,
        protected_roots=protected_read_roots,
    )
    extra_outputs = additional_output_roots(args, project_root=resolved_project_root, protected_roots=protected_read_roots)
    cache_root, _cache_environment = configure_sam3_cache_root(
        args.cache_root,
        project_root=resolved_project_root,
        protected_roots=(
            *protected_read_roots,
            ("output_root", output_root),
            ("allowed_output_root", allowed_output_root),
            *(("additional_output_root", root) for root in extra_outputs),
            *(
                (f"allowed_input_root[{index}]", root)
                for index, root in enumerate(allowed_input_roots)
            ),
        ),
    )
    confidence_threshold = float(args.confidence_threshold)
    if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise RuntimeError("confidence-threshold must be finite and within [0, 1]")
    add_sam3_repo_to_import_path(resources_config.repo)
    device = resolve_device(args.device)
    SAM3SegmentationHandler.processor = load_processor(
        checkpoint=resources_config.checkpoint,
        bpe_path=resources_config.bpe_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    SAM3SegmentationHandler.output_root = output_root
    SAM3SegmentationHandler.allowed_output_root = allowed_output_root
    SAM3SegmentationHandler.additional_output_roots = extra_outputs
    SAM3SegmentationHandler.allowed_input_roots = allowed_input_roots
    SAM3SegmentationHandler.project_root = resolved_project_root
    SAM3SegmentationHandler.runtime_root = resolved_project_root / "eval_result"
    SAM3SegmentationHandler.cache_root = cache_root
    SAM3SegmentationHandler.protected_read_roots = tuple(
        _resolved_path(root) for _label, root in protected_read_roots
    )
    SAM3SegmentationHandler.sam3_repo = str(resources_config.repo)
    SAM3SegmentationHandler.checkpoint = str(resources_config.checkpoint)
    SAM3SegmentationHandler.bpe_path = str(resources_config.bpe_path)
    SAM3SegmentationHandler.device = device
    SAM3SegmentationHandler.amp_dtype = str(args.amp_dtype)
    SAM3SegmentationHandler.confidence_threshold = confidence_threshold
    SAM3SegmentationHandler.instance_id = str(args.instance_id)
    return SAM3SegmentationHandler


def main(argv: Sequence[str] | None = None) -> None:
    sys.dont_write_bytecode = True
    args = parse_args(argv)
    validate_bind_host(args.host, allow_remote=bool(args.allow_remote))
    if args.check_paths_only:
        resources_config = validate_external_resources(
            sam3_repo=args.sam3_repo,
            checkpoint=args.checkpoint,
            bpe_path=args.bpe_path,
        )
        protected_read_roots: tuple[tuple[str, str | Path], ...] = (
            *readonly_rmbench_boundaries(),
            ("selected SAM3 source", resources_config.repo),
            ("selected SAM3 checkpoint", resources_config.checkpoint),
            ("selected SAM3 BPE vocabulary", resources_config.bpe_path),
        )
        output_root, allowed_output_root, allowed_input_roots = validate_io_roots(
            output_root=args.output_root,
            allowed_output_root=args.allowed_output_root,
            allowed_input_roots=args.allowed_input_root,
            protected_roots=protected_read_roots,
        )
        extra_outputs = additional_output_roots(args, protected_roots=protected_read_roots)
        cache_root, cache_environment = configure_sam3_cache_root(
            args.cache_root,
            protected_roots=(
                *protected_read_roots,
                ("output_root", output_root),
                ("allowed_output_root", allowed_output_root),
                *(("additional_output_root", root) for root in extra_outputs),
                *(
                    (f"allowed_input_root[{index}]", root)
                    for index, root in enumerate(allowed_input_roots)
                ),
            ),
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "endpoint": f"http://{args.host}:{args.port}/segment_image",
                    "sam3_repo": str(resources_config.repo),
                    "checkpoint": str(resources_config.checkpoint),
                    "bpe_path": str(resources_config.bpe_path),
                    "output_root": str(output_root),
                    "allowed_output_root": str(allowed_output_root),
                    "allowed_output_roots": [str(allowed_output_root), *(str(root) for root in extra_outputs)],
                    "allowed_input_roots": [str(root) for root in allowed_input_roots],
                    "project_root": str(ROBOHARN_PROJECT_ROOT.resolve()),
                    "runtime_root": str(ROBOHARN_EVAL_RESULT_ROOT.resolve()),
                    "cache_root": str(cache_root),
                    "cache_environment": cache_environment,
                    "protected_read_roots": [
                        str(_resolved_path(root))
                        for _label, root in protected_read_roots
                    ],
                    "model_loaded": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    handler = configure_handler(args)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"[sam3-segmentation] listening on http://{args.host}:{args.port} "
        f"device={handler.device}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
