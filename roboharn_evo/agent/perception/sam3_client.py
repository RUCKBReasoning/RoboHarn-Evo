from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error, request


class SAM3SegmentationClient:
    def __init__(self, base_url: str = "http://127.0.0.1:9301", timeout_sec: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def segment_image(
        self,
        *,
        image_path: str | Path,
        text_prompt: str,
        object_id: str = "",
        output_dir: str | Path | None = None,
        top_k: int = 3,
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_path": str(image_path),
            "text_prompt": text_prompt,
            "object_id": object_id or text_prompt,
            "top_k": int(top_k),
        }
        if output_dir is not None:
            payload["output_dir"] = str(output_dir)
        if confidence_threshold is not None:
            payload["confidence_threshold"] = float(confidence_threshold)
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
            raise RuntimeError(f"SAM3 service HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"SAM3 service unavailable at {self.base_url}: {exc.reason}") from exc

        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"SAM3 service returned non-object JSON: {body[:200]}")
        return parsed
