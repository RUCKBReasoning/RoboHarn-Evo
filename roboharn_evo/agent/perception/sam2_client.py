from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error, request


class SAM2SegmentationClient:
    def __init__(self, base_url: str = "http://127.0.0.1:9201", timeout_sec: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def segment_image(
        self,
        *,
        image_path: str | Path,
        object_id: str,
        bbox_xyxy: list[float] | None = None,
        point_coords: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
        output_dir: str | Path | None = None,
        multimask_output: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_path": str(image_path),
            "object_id": object_id,
            "multimask_output": bool(multimask_output),
        }
        if bbox_xyxy is not None:
            payload["bbox_xyxy"] = bbox_xyxy
        if point_coords is not None:
            payload["point_coords"] = point_coords
        if point_labels is not None:
            payload["point_labels"] = point_labels
        if output_dir is not None:
            payload["output_dir"] = str(output_dir)
        return self._request("POST", "/segment_image", payload)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SAM2 service HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"SAM2 service unavailable at {self.base_url}: {exc.reason}") from exc

        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"SAM2 service returned non-object JSON: {body[:200]}")
        return parsed
