from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image
import torch

from roboharn_evo.agent.paths import eval_result_dir, sam3_bpe_path, sam3_checkpoint_path, sam3_repo_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve SAM3 text-prompt image segmentation over HTTP.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9301)
    parser.add_argument("--sam3-repo", type=str, default=str(sam3_repo_dir()))
    parser.add_argument("--checkpoint", type=str, default=str(sam3_checkpoint_path()))
    parser.add_argument("--bpe-path", type=str, default=str(sam3_bpe_path()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "none"])
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--output-root", type=str, default=str(eval_result_dir("tcm_sam3_segmentation")))
    parser.add_argument(
        "--allowed-output-root",
        type=str,
        default="",
        help=(
            "Optional service write boundary. Requests whose output_dir is outside "
            "this directory are rejected. Empty preserves the legacy single-service behavior."
        ),
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        default="",
        help="Optional orchestration identity reported by /health.",
    )
    return parser.parse_args()


def ensure_sam3_repo(repo_path: str) -> None:
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"SAM3 repo does not exist: {path}")
    sys.path.insert(0, str(path))


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


def load_processor(*, checkpoint: str, bpe_path: str, device: str, confidence_threshold: float):
    ensure_pkg_resources_compat()
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested, but torch.cuda.is_available() is false")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint,
        load_from_HF=False,
        device=device,
    )
    return Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_") or "object"


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
    output_path = output_dir / f"{safe_name(object_id)}_{rank:02d}_{int(time.time() * 1000)}.png"
    Image.fromarray((np.asarray(mask).astype(np.uint8) * 255), mode="L").save(output_path)
    return output_path


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.dtype in {torch.bfloat16, torch.float16}:
            value = value.to(dtype=torch.float32)
        return value.detach().cpu().numpy()
    return np.asarray(value)


def autocast_context(device: str, amp_dtype: str):
    if device != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


class SAM3SegmentationHandler(BaseHTTPRequestHandler):
    processor = None
    output_root = eval_result_dir("tcm_sam3_segmentation")
    allowed_output_root: Path | None = None
    checkpoint = ""
    device = ""
    amp_dtype = "bfloat16"
    confidence_threshold = 0.5
    instance_id = ""
    inference_lock = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            cuda_current_device = None
            if self.device.startswith("cuda") and torch.cuda.is_available():
                cuda_current_device = int(torch.cuda.current_device())
            self._send_json(
                {
                    "status": "ok",
                    "backend": "sam3_image_text_prompt",
                    "checkpoint": self.checkpoint,
                    "device": self.device,
                    "amp_dtype": self.amp_dtype,
                    "confidence_threshold": self.confidence_threshold,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_current_device": cuda_current_device,
                    "cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES", ""
                    ),
                    "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", ""),
                    "instance_id": self.instance_id,
                    "pid": os.getpid(),
                    "output_root": str(self.output_root),
                    "allowed_output_root": (
                        str(self.allowed_output_root)
                        if self.allowed_output_root is not None
                        else ""
                    ),
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/segment_image":
            self._handle_segment_image()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def _handle_segment_image(self) -> None:
        if self.processor is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "processor is not initialized")
            return
        try:
            payload = self._read_json()
            result = self._segment_image(payload)
        except Exception as exc:
            self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(result)

    def _segment_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_path = Path(str(payload["image_path"])).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"image_path does not exist: {image_path}")
        text_prompt = str(payload.get("text_prompt", "")).strip()
        if not text_prompt:
            raise ValueError("text_prompt is required")
        object_id = str(payload.get("object_id", text_prompt)).strip() or text_prompt
        top_k = max(1, int(payload.get("top_k", 3)))
        output_root = Path(self.output_root).expanduser().resolve()
        output_dir = Path(str(payload.get("output_dir") or output_root)).expanduser().resolve()
        allowed_output_root = self.allowed_output_root
        if allowed_output_root is not None:
            allowed_output_root = Path(allowed_output_root).expanduser().resolve()
            try:
                output_dir.relative_to(allowed_output_root)
            except ValueError as exc:
                raise ValueError(
                    "output_dir must be within this SAM3 instance "
                    f"allowed_output_root: {allowed_output_root}"
                ) from exc
        image = load_rgb(image_path)
        # The processor has mutable request state (including the confidence
        # threshold).  A pool normally gives each rollout its own process, but
        # this lock also prevents accidental re-entry within one instance.
        with self.inference_lock:
            request_threshold = float(
                payload.get("confidence_threshold", self.confidence_threshold)
            )
            if not math.isfinite(request_threshold) or not 0.0 <= request_threshold <= 1.0:
                raise ValueError("confidence_threshold must be finite and within [0, 1]")
            self.processor.confidence_threshold = request_threshold
            try:
                with torch.inference_mode(), autocast_context(self.device, self.amp_dtype):
                    state = self.processor.set_image(image)
                    state = self.processor.set_text_prompt(state=state, prompt=text_prompt)
                # Keep GPU materialization inside the same lock as the mutable
                # processor state.  Post-processing and mask writes are CPU-only.
                masks = tensor_to_numpy(state.get("masks"))
                boxes = tensor_to_numpy(state.get("boxes"))
                scores = tensor_to_numpy(state.get("scores")).reshape(-1)
            finally:
                # A request-specific diagnostic threshold must never leak into
                # the next rollout request handled by this service instance.
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
        masks = np.squeeze(masks, axis=1) if masks.ndim == 4 and masks.shape[1] == 1 else masks
        order = np.argsort(-scores)[:top_k]
        detections: list[dict[str, Any]] = []
        for rank, idx in enumerate(order):
            mask = masks[int(idx)].astype(bool)
            mask_path = save_mask(mask, output_dir, object_id, rank)
            box = boxes[int(idx)].reshape(-1).tolist() if boxes.size else []
            detections.append(
                {
                    "rank": rank,
                    "score": round(float(scores[int(idx)]), 6),
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
        body_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(body_length)
        payload = json.loads(raw_body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    args = parse_args()
    ensure_sam3_repo(args.sam3_repo)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    SAM3SegmentationHandler.processor = load_processor(
        checkpoint=args.checkpoint,
        bpe_path=args.bpe_path,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )
    SAM3SegmentationHandler.output_root = Path(args.output_root).expanduser().resolve()
    SAM3SegmentationHandler.allowed_output_root = (
        Path(args.allowed_output_root).expanduser().resolve()
        if str(args.allowed_output_root).strip()
        else None
    )
    SAM3SegmentationHandler.checkpoint = args.checkpoint
    SAM3SegmentationHandler.device = device
    SAM3SegmentationHandler.amp_dtype = args.amp_dtype
    SAM3SegmentationHandler.confidence_threshold = float(args.confidence_threshold)
    SAM3SegmentationHandler.instance_id = str(args.instance_id)
    server = ThreadingHTTPServer((args.host, args.port), SAM3SegmentationHandler)
    print(f"[sam3-segmentation] listening on http://{args.host}:{args.port} device={device}")
    server.serve_forever()


if __name__ == "__main__":
    main()
