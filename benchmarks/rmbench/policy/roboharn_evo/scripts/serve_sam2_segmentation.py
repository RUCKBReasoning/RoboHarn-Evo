from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from policy.roboharn_evo.agent.paths import eval_result_dir, sam2_checkpoint_path, sam2_repo_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve SAM2 promptable image segmentation over HTTP.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9201)
    parser.add_argument("--sam2-repo", type=str, default=str(sam2_repo_dir()))
    parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--checkpoint", type=str, default=str(sam2_checkpoint_path()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=str, default=str(eval_result_dir("tcm_sam2_segmentation")))
    parser.add_argument("--single-mask", action="store_true", help="Ask SAM2 for one mask instead of multimask candidates.")
    return parser.parse_args()


def ensure_sam2_repo(repo_path: str) -> None:
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"SAM2 repo does not exist: {path}")
    sys.path.insert(0, str(path))


def load_predictor(*, config: str, checkpoint: str, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested, but torch.cuda.is_available() is false")
    model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(model)


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def normalize_box(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 4 or not np.all(np.isfinite(array)):
        raise ValueError("bbox_xyxy must contain four finite numbers")
    return array


def normalize_points(coords: Any, labels: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if coords is None:
        return None, None
    point_coords = np.asarray(coords, dtype=np.float32)
    if point_coords.ndim != 2 or point_coords.shape[1] != 2 or not np.all(np.isfinite(point_coords)):
        raise ValueError("point_coords must be an Nx2 finite array")
    if labels is None:
        point_labels = np.ones((point_coords.shape[0],), dtype=np.int32)
    else:
        point_labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if point_labels.shape[0] != point_coords.shape[0]:
        raise ValueError("point_labels length must match point_coords")
    return point_coords, point_labels


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


def save_mask(mask: np.ndarray, output_dir: Path, object_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_object_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in object_id).strip("_") or "object"
    output_path = output_dir / f"{safe_object_id}_{int(time.time() * 1000)}.png"
    image = Image.fromarray((np.asarray(mask).astype(np.uint8) * 255), mode="L")
    image.save(output_path)
    return output_path


class SAM2SegmentationHandler(BaseHTTPRequestHandler):
    predictor = None
    output_root = eval_result_dir("tcm_sam2_segmentation")
    multimask_output = True
    config = ""
    checkpoint = ""
    device = ""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "backend": "sam2_image_predictor",
                    "config": self.config,
                    "checkpoint": self.checkpoint,
                    "device": self.device,
                    "cuda_available": torch.cuda.is_available(),
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
        if self.predictor is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "predictor is not initialized")
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
        object_id = str(payload.get("object_id", "object")).strip() or "object"
        bbox = normalize_box(payload.get("bbox_xyxy"))
        point_coords, point_labels = normalize_points(payload.get("point_coords"), payload.get("point_labels"))
        if bbox is None and point_coords is None:
            raise ValueError("bbox_xyxy or point_coords is required")

        output_dir = Path(str(payload.get("output_dir") or self.output_root)).expanduser().resolve()
        image = load_rgb(image_path)
        self.predictor.set_image(image)
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=bbox,
                multimask_output=bool(payload.get("multimask_output", self.multimask_output)),
                return_logits=False,
            )

        if masks.ndim != 3:
            raise RuntimeError(f"unexpected SAM2 masks shape: {masks.shape}")
        score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
        best_idx = int(np.argmax(score_array)) if score_array.size else 0
        best_mask = masks[best_idx].astype(bool)
        mask_path = save_mask(best_mask, output_dir, object_id)
        summary = mask_summary(best_mask)
        return {
            "success": True,
            "object_id": object_id,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "score": round(float(score_array[best_idx]), 6) if score_array.size else None,
            "candidate_scores": [round(float(item), 6) for item in score_array.tolist()],
            **summary,
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
    ensure_sam2_repo(args.sam2_repo)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    SAM2SegmentationHandler.predictor = load_predictor(config=args.config, checkpoint=args.checkpoint, device=device)
    SAM2SegmentationHandler.output_root = Path(args.output_root).expanduser().resolve()
    SAM2SegmentationHandler.multimask_output = not bool(args.single_mask)
    SAM2SegmentationHandler.config = args.config
    SAM2SegmentationHandler.checkpoint = args.checkpoint
    SAM2SegmentationHandler.device = device
    server = ThreadingHTTPServer((args.host, args.port), SAM2SegmentationHandler)
    print(f"[sam2-segmentation] listening on http://{args.host}:{args.port} device={device}")
    server.serve_forever()


if __name__ == "__main__":
    main()
