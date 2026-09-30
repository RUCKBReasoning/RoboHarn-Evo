from __future__ import annotations

import argparse
from collections.abc import Mapping
from http import HTTPStatus
import ipaddress
import json
import os
from pathlib import Path
from typing import Any, BinaryIO


ROBOHARN_PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOHARN_EVAL_RESULT_ROOT = ROBOHARN_PROJECT_ROOT / "eval_result"
# RMBench 数据目录保持只读，额外目录通过环境变量指定。
LEGACY_RMBENCH_DONOR_ROOT = (ROBOHARN_PROJECT_ROOT.parent / "RMBench").resolve()


class RequestBodyError(ValueError):
    """A client request-body error with an explicit HTTP response status."""

    def __init__(self, message: str, *, status: HTTPStatus) -> None:
        super().__init__(message)
        self.status = status


def add_allow_remote_argument(parser: argparse.ArgumentParser) -> None:
    """Add the opt-in required before binding a service beyond loopback."""

    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help=(
            "explicitly allow binding to a non-loopback host; disabled by "
            "default because these service APIs do not provide authentication"
        ),
    )


def validate_bind_host(host: str, *, allow_remote: bool) -> str:
    """Reject a non-loopback bind unless the operator explicitly opted in."""

    normalized = str(host).strip()
    candidate = (
        normalized[1:-1]
        if normalized.startswith("[") and normalized.endswith("]")
        else normalized
    )
    is_loopback = candidate.lower() == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(candidate).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback and not allow_remote:
        raise ValueError(
            f"refusing non-loopback bind host {host!r}; pass --allow-remote "
            "only when remote exposure is intentional and network access is protected"
        )
    return normalized


def resolved_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a path, including existing symlinks, without creating it."""

    return Path(value).expanduser().resolve(strict=False)


def path_is_within(
    value: str | os.PathLike[str],
    root: str | os.PathLike[str],
) -> bool:
    """Return whether ``value`` resolves to ``root`` or one of its descendants."""

    candidate = resolved_path(value)
    boundary = resolved_path(root)
    return candidate == boundary or boundary in candidate.parents


def paths_overlap(
    first: str | os.PathLike[str],
    second: str | os.PathLike[str],
) -> bool:
    """Return whether either resolved path contains the other."""

    first_path = resolved_path(first)
    second_path = resolved_path(second)
    return path_is_within(first_path, second_path) or path_is_within(
        second_path, first_path
    )


def require_path_within(
    value: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str],
    label: str,
) -> Path:
    """Resolve ``value`` and fail unless it remains within ``root``."""

    candidate = resolved_path(value)
    boundary = resolved_path(root)
    if not path_is_within(candidate, boundary):
        raise ValueError(
            f"{label} must resolve inside the guarded RoboHarn-Evo root {boundary}: {candidate}"
        )
    return candidate


def readonly_rmbench_boundaries(
    environ: Mapping[str, str] | None = None,
) -> tuple[tuple[str, Path], ...]:
    """Return old/caller-supplied RMBench roots that services must never write."""

    source = os.environ if environ is None else environ
    candidates: list[tuple[str, Path]] = [
        ("legacy RMBench donor", LEGACY_RMBENCH_DONOR_ROOT),
    ]
    for variable in ("RMBENCH_ROOT", "RMBENCH_ASSETS_ROOT"):
        raw_value = str(source.get(variable, "")).strip()
        if raw_value:
            candidates.append((variable, resolved_path(raw_value)))

    unique: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, boundary in candidates:
        if boundary not in seen:
            unique.append((label, boundary))
            seen.add(boundary)
    return tuple(unique)


def reject_path_overlaps(
    value: str | os.PathLike[str],
    *,
    boundaries: Mapping[str, str | os.PathLike[str]]
    | tuple[tuple[str, str | os.PathLike[str]], ...]
    | list[tuple[str, str | os.PathLike[str]]],
    label: str,
) -> Path:
    """Resolve a path and reject ancestor/descendant overlap with protected roots."""

    candidate = resolved_path(value)
    items = boundaries.items() if isinstance(boundaries, Mapping) else boundaries
    for boundary_label, raw_boundary in items:
        boundary = resolved_path(raw_boundary)
        if paths_overlap(candidate, boundary):
            raise ValueError(
                f"{label} {candidate} overlaps protected {boundary_label}: {boundary}"
            )
    return candidate


def read_json_object_body(
    *,
    headers: Mapping[str, Any],
    stream: BinaryIO,
    max_bytes: int,
) -> dict[str, Any]:
    """Read one bounded UTF-8 JSON object or raise a status-bearing error."""

    if max_bytes <= 0:
        raise RuntimeError("max request-body size must be a positive integer")
    content_length_text = headers.get("Content-Length")
    if content_length_text is None:
        raise RequestBodyError(
            "Content-Length header is required",
            status=HTTPStatus.BAD_REQUEST,
        )
    try:
        content_length = int(content_length_text)
    except (TypeError, ValueError) as exc:
        raise RequestBodyError(
            "Content-Length must be an integer",
            status=HTTPStatus.BAD_REQUEST,
        ) from exc
    if content_length < 0:
        raise RequestBodyError(
            "Content-Length must not be negative",
            status=HTTPStatus.BAD_REQUEST,
        )
    if content_length > max_bytes:
        raise RequestBodyError(
            f"request body exceeds maximum size of {max_bytes} bytes",
            status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )
    raw_body = stream.read(content_length)
    if len(raw_body) != content_length:
        raise RequestBodyError(
            "request body ended before Content-Length bytes were received",
            status=HTTPStatus.BAD_REQUEST,
        )
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RequestBodyError(
            "request body must be valid UTF-8 JSON",
            status=HTTPStatus.BAD_REQUEST,
        ) from exc
    if not isinstance(payload, dict):
        raise RequestBodyError(
            "request body must be a JSON object",
            status=HTTPStatus.BAD_REQUEST,
        )
    return payload


__all__ = [
    "LEGACY_RMBENCH_DONOR_ROOT",
    "RequestBodyError",
    "ROBOHARN_EVAL_RESULT_ROOT",
    "ROBOHARN_PROJECT_ROOT",
    "add_allow_remote_argument",
    "path_is_within",
    "paths_overlap",
    "read_json_object_body",
    "readonly_rmbench_boundaries",
    "reject_path_overlaps",
    "require_path_within",
    "resolved_path",
    "validate_bind_host",
]
