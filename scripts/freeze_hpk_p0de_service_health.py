from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib import request


class ServiceHealthFreezeError(RuntimeError):
    pass


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _fetch_once(url: str, *, timeout: float, label: str) -> dict[str, Any]:
    outbound = request.Request(url.rstrip("/") + "/health", method="GET")
    try:
        with request.urlopen(outbound, timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
    except Exception as exc:
        raise ServiceHealthFreezeError(
            f"{label} health fetch failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not raw or len(raw) > 1024 * 1024:
        raise ServiceHealthFreezeError(f"{label} health response size is invalid")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceHealthFreezeError(
            f"{label} health response is invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ServiceHealthFreezeError(f"{label} health response must be an object")
    return payload


def _write_once(path: Path, raw: bytes) -> str:
    if not path.is_absolute() or not path.parent.is_dir() or path.parent.is_symlink():
        raise ServiceHealthFreezeError("health artifact path/parent is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ServiceHealthFreezeError(
            f"health artifact already exists: {path}"
        ) from exc
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != len(raw):
            raise ServiceHealthFreezeError("health artifact write was incomplete")
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return hashlib.sha256(raw).hexdigest()


def freeze_service_health(
    *,
    agent_url: str,
    sam_url: str,
    segmentation_artifact_root: Path,
    output_dir: Path,
    timeout: float = 30.0,
) -> dict[str, Any]:
    if not output_dir.is_absolute() or not output_dir.is_dir():
        raise ServiceHealthFreezeError(
            "output_dir must be an existing absolute directory"
        )
    if output_dir.is_symlink():
        raise ServiceHealthFreezeError("output_dir must not be a symlink")
    root = segmentation_artifact_root.absolute()
    agent = _fetch_once(agent_url, timeout=timeout, label="agent_api")
    sam = _fetch_once(sam_url, timeout=timeout, label="sam3")
    if (
        str(agent.get("status", "")).lower() != "ok"
        or agent.get("backend") != "codex-account"
        or agent.get("provider") != "openai"
        or agent.get("model") != "gpt-5.5"
        or agent.get("api_mode") != "responses_compat"
        or agent.get("reasoning_effort") != "xhigh"
        or agent.get("max_concurrent_requests") != 1
        or not isinstance(agent.get("responses_capabilities"), Mapping)
        or agent["responses_capabilities"].get("max_images_per_request") != 16
        or agent.get("max_retries") != 0
        or agent.get("response_storage") != "account_default"
    ):
        raise ServiceHealthFreezeError("Agent API health identity/config mismatch")
    if (
        str(sam.get("status", "")).lower() != "ok"
        or sam.get("backend") != "sam3_image_text_prompt"
        or sam.get("output_root") != str(root / "masks")
        or sam.get("allowed_output_root") != str(root / "masks")
        or sam.get("allowed_input_roots") != [str(root / "inputs")]
        or sam.get("cuda_available") is not True
    ):
        raise ServiceHealthFreezeError("SAM3 health identity/config mismatch")
    agent_raw = _canonical(agent)
    sam_raw = _canonical(sam)
    agent_path = output_dir / "agent_service_identity.json"
    sam_path = output_dir / "sam3_service_identity.json"
    agent_sha = _write_once(agent_path, agent_raw)
    sam_sha = _write_once(sam_path, sam_raw)
    receipt = {
        "schema": "roboharn_evo/hpk/p0de_service_health_freeze/v1",
        "agent_api": {
            "path": str(agent_path),
            "sha256": agent_sha,
            "health_call_count": 1,
        },
        "sam3": {
            "path": str(sam_path),
            "sha256": sam_sha,
            "health_call_count": 1,
        },
        "total_http_calls": 2,
        "automatic_retry_count": 0,
    }
    receipt_raw = _canonical(receipt)
    receipt_path = output_dir / "service_health_freeze_receipt.json"
    receipt_sha = _write_once(receipt_path, receipt_raw)
    return {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-url", required=True)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--segmentation-artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = freeze_service_health(
        agent_url=args.agent_url,
        sam_url=args.sam_url,
        segmentation_artifact_root=args.segmentation_artifact_root,
        output_dir=args.output_dir,
        timeout=args.timeout,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
