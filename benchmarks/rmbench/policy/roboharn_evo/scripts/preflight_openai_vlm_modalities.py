#!/usr/bin/env python3
"""Fail-closed modality smoke test for an OpenAI-compatible local VLM.

The report intentionally contains only request metadata and content hashes.  It
does not archive prompts, images, or generated text.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

try:
    from policy.roboharn_evo.scripts.local_models_identity import models_identity_sha256
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from local_models_identity import models_identity_sha256


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
REQUEST_MODES = ("text_only", "head_image", "head_then_third_images")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
RED_CIRCLE_RELATIONS = (
    "left_of_blue_blocks",
    "right_of_blue_blocks",
    "within_blue_block_span",
    "not_visible",
)
TEXT_ARITHMETIC_RESULT = 42


class PreflightError(RuntimeError):
    """A validation or transport failure that must fail the preflight."""


@dataclass(frozen=True)
class ImageArtifact:
    role: str
    path: Path
    media_type: str
    sha256: str
    byte_count: int
    data_url: str


@dataclass(frozen=True)
class HttpJsonResponse:
    status_code: int
    body: bytes
    json_payload: dict[str, Any]
    latency_ms: float


@dataclass(frozen=True)
class VisualFacts:
    blue_block_count: int
    red_circle_horizontal_relation: str


UrlOpen = Callable[..., Any]
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _direct_urlopen(request: urllib.request.Request, *, timeout: float) -> Any:
    """Open only the requested engine URL without consulting proxy variables."""

    return _NO_PROXY_OPENER.open(request, timeout=timeout)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify text, one-image, and ordered two-image support on a direct "
            "OpenAI-compatible VLM engine."
        )
    )
    parser.add_argument(
        "--base-url",
        "--engine-base-url",
        dest="base_url",
        default=DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--model",
        "--model-id",
        dest="model",
        required=True,
        help="Exact served model id.",
    )
    parser.add_argument("--head-image", type=Path, required=True)
    parser.add_argument("--third-image", type=Path, required=True)
    parser.add_argument(
        "--expected-head-blue-block-count",
        type=int,
        required=True,
        help="Human-verified number of blue block-shaped objects in the head image.",
    )
    parser.add_argument(
        "--expected-head-red-circle-relation",
        choices=RED_CIRCLE_RELATIONS,
        required=True,
        help="Human-verified horizontal relation in the head image.",
    )
    parser.add_argument(
        "--expected-third-blue-block-count",
        type=int,
        required=True,
        help="Human-verified number of blue block-shaped objects in the third image.",
    )
    parser.add_argument(
        "--expected-third-red-circle-relation",
        choices=RED_CIRCLE_RELATIONS,
        required=True,
        help="Human-verified horizontal relation in the third image.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--response-format-json-object",
        action="store_true",
        help=(
            "Send response_format={type:json_object}. Disabled by default so "
            "the smoke test also works with engines that rely on the JSON-only prompt."
        ),
    )
    args = parser.parse_args(argv)
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if not args.model.strip():
        parser.error("--model must be nonempty")
    if args.expected_head_blue_block_count < 0:
        parser.error("--expected-head-blue-block-count must be non-negative")
    if args.expected_third_blue_block_count < 0:
        parser.error("--expected-third-blue-block-count must be non-negative")
    return args


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _media_type(path: Path, data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise PreflightError(
        f"unsupported or invalid image file for {path.name}; expected PNG, JPEG, GIF, or WEBP"
    )


def load_image(role: str, path: Path) -> ImageArtifact:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PreflightError(f"{role} image is not a regular file: {resolved}")
    data = resolved.read_bytes()
    if not data:
        raise PreflightError(f"{role} image is empty: {resolved}")
    media_type = _media_type(resolved, data)
    return ImageArtifact(
        role=role,
        path=resolved,
        media_type=media_type,
        sha256=_sha256(data),
        byte_count=len(data),
        data_url=f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}",
    )


def _http_json(
    method: str,
    url: str,
    *,
    timeout_sec: float,
    payload: dict[str, Any] | None = None,
    opener: UrlOpen = _direct_urlopen,
) -> HttpJsonResponse:
    request_body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=request_body,
        headers=headers,
        method=method,
    )
    started = time.monotonic()
    try:
        with opener(request, timeout=timeout_sec) as response:
            status = int(getattr(response, "status", response.getcode()))
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise PreflightError(f"{method} {url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = type(exc.reason).__name__ if exc.reason is not None else "unknown"
        raise PreflightError(f"{method} {url} transport failure ({reason})") from exc
    except TimeoutError as exc:
        raise PreflightError(f"{method} {url} timed out") from exc
    latency_ms = (time.monotonic() - started) * 1000.0
    if status < 200 or status >= 300:
        raise PreflightError(f"{method} {url} returned HTTP {status}")
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{method} {url} returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise PreflightError(f"{method} {url} response must be a JSON object")
    return HttpJsonResponse(
        status_code=status,
        body=body,
        json_payload=decoded,
        latency_ms=latency_ms,
    )


def _balanced_json_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    start: int | None = None
    depth = 0
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start = index
                depth = 1
                quoted = False
                escaped = False
            continue
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(value, dict):
                        objects.append(value)
                start = None
    return objects


def parse_generated_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if not stripped:
        raise PreflightError("chat completion message.content is empty")
    try:
        direct = json.loads(stripped)
    except json.JSONDecodeError:
        direct = None
    if isinstance(direct, dict):
        return direct
    objects = _balanced_json_objects(stripped)
    if len(objects) != 1:
        raise PreflightError(
            f"chat completion must contain exactly one JSON object; found {len(objects)}"
        )
    return objects[0]


def _validate_models(response: HttpJsonResponse, exact_model: str) -> dict[str, Any]:
    data = response.json_payload.get("data")
    if not isinstance(data, list):
        raise PreflightError("GET /models response is missing a data list")
    model_ids = [item.get("id") for item in data if isinstance(item, dict)]
    if exact_model not in model_ids:
        raise PreflightError(
            f"GET /models did not contain the exact required model id: {exact_model}"
        )
    return {
        "status_code": response.status_code,
        "latency_ms": round(response.latency_ms, 3),
        "response_sha256": _sha256(response.body),
        "identity_sha256": models_identity_sha256(response.json_payload),
        "response_byte_count": len(response.body),
        "listed_model_count": len(model_ids),
        "exact_model_found": True,
    }


def _user_content(mode: str, images: list[ImageArtifact]) -> str | list[dict[str, Any]]:
    if mode == "text_only":
        instruction = (
            "Compute seventeen plus twenty-five. Return only the requested JSON object; "
            "do not describe your reasoning."
        )
    elif mode == "head_image":
        instruction = (
            "Inspect image 1 (the head-camera image). Count distinct blue block-shaped "
            "objects. Compare the center of the red circular object with the horizontal "
            "span of the blue-block centers: use left_of_blue_blocks, "
            "right_of_blue_blocks, within_blue_block_span, or not_visible. Return one "
            "observation for image_index 1."
        )
    elif mode == "head_then_third_images":
        instruction = (
            "Inspect both supplied images independently. Image 1 is the head-camera "
            "image and image 2 is the third-camera image. For each image, count distinct "
            "blue block-shaped objects and compare the center of the red circular object "
            "with the horizontal span of the blue-block centers using "
            "left_of_blue_blocks, right_of_blue_blocks, within_blue_block_span, or "
            "not_visible. Return observations in image order: image_index 1, then 2."
        )
    else:
        raise PreflightError(f"unknown request mode: {mode}")
    if not images:
        return instruction
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    for artifact in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": artifact.data_url},
            }
        )
    return content


def build_chat_payload(
    *,
    model: str,
    mode: str,
    images: list[ImageArtifact],
    max_tokens: int,
    response_format_json_object: bool,
) -> dict[str, Any]:
    if mode == "text_only":
        schema_instruction = (
            "Return exactly one valid JSON object and no other text. It must contain "
            f"modality (string, set to {mode}) and arithmetic_result (integer)."
        )
    else:
        schema_instruction = (
            "Return exactly one valid JSON object and no other text. It must contain "
            f"modality (string, set to {mode}) and observations (array). Every "
            "observation must contain image_index (integer), blue_block_count "
            "(non-negative integer), and red_circle_horizontal_relation (string)."
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": schema_instruction,
            },
            {"role": "user", "content": _user_content(mode, images)},
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if response_format_json_object:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _usage_counters(response: dict[str, Any]) -> dict[str, int | None]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    def counter(*names: str) -> int | None:
        for name in names:
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    return {
        "input_tokens": counter("prompt_tokens", "input_tokens"),
        "output_tokens": counter("completion_tokens", "output_tokens"),
        "total_tokens": counter("total_tokens"),
    }


def _validate_chat_response(
    response: HttpJsonResponse,
    *,
    exact_model: str,
    mode: str,
    expected_visual_facts: list[VisualFacts],
) -> dict[str, Any]:
    body = response.json_payload
    if not isinstance(body.get("id"), str) or not body["id"]:
        raise PreflightError(f"{mode}: response is missing a nonempty id")
    if body.get("object") not in {"chat.completion", "chat.completion.chunk"}:
        raise PreflightError(f"{mode}: response has an invalid object type")
    if body.get("model") != exact_model:
        raise PreflightError(f"{mode}: response model does not exactly match the requested model")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise PreflightError(f"{mode}: response is missing choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise PreflightError(f"{mode}: choices[0].message is not an assistant message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise PreflightError(f"{mode}: response has empty message.content")
    generated = parse_generated_json(content)
    if generated.get("modality") != mode:
        raise PreflightError(f"{mode}: generated JSON has the wrong modality field")
    if mode == "text_only":
        result = generated.get("arithmetic_result")
        if not isinstance(result, int) or isinstance(result, bool):
            raise PreflightError(f"{mode}: arithmetic_result must be an integer")
        if result != TEXT_ARITHMETIC_RESULT:
            raise PreflightError(f"{mode}: arithmetic result is incorrect")
    else:
        observations = generated.get("observations")
        if not isinstance(observations, list):
            raise PreflightError(f"{mode}: observations must be an array")
        if len(observations) != len(expected_visual_facts):
            raise PreflightError(
                f"{mode}: expected {len(expected_visual_facts)} ordered observations; "
                f"received {len(observations)}"
            )
        for index, (observation, expected) in enumerate(
            zip(observations, expected_visual_facts, strict=True),
            start=1,
        ):
            if not isinstance(observation, dict):
                raise PreflightError(f"{mode}: observations[{index - 1}] must be an object")
            if observation.get("image_index") != index:
                raise PreflightError(
                    f"{mode}: observations must preserve supplied image order"
                )
            count = observation.get("blue_block_count")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise PreflightError(
                    f"{mode}: observations[{index - 1}].blue_block_count must be a "
                    "non-negative integer"
                )
            relation = observation.get("red_circle_horizontal_relation")
            if relation not in RED_CIRCLE_RELATIONS:
                raise PreflightError(
                    f"{mode}: observations[{index - 1}] has an invalid red-circle relation"
                )
            if count != expected.blue_block_count:
                raise PreflightError(
                    f"{mode}: image {index} blue-block count does not match the "
                    "independently supplied expected fact"
                )
            if relation != expected.red_circle_horizontal_relation:
                raise PreflightError(
                    f"{mode}: image {index} red-circle relation does not match the "
                    "independently supplied expected fact"
                )
    return {
        "status_code": response.status_code,
        "latency_ms": round(response.latency_ms, 3),
        "response_sha256": _sha256(response.body),
        "response_byte_count": len(response.body),
        "usage": _usage_counters(body),
        "schema_valid": True,
    }


def run_preflight(
    args: argparse.Namespace,
    *,
    opener: UrlOpen = _direct_urlopen,
) -> dict[str, Any]:
    base_url = str(args.base_url).strip().rstrip("/")
    parsed_base_url = urllib.parse.urlsplit(base_url)
    if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.hostname:
        raise PreflightError("--base-url must be an absolute http:// or https:// URL")
    if (
        parsed_base_url.username is not None
        or parsed_base_url.password is not None
        or "?" in base_url
        or "#" in base_url
    ):
        raise PreflightError("--base-url must not contain credentials, a query, or a fragment")
    if parsed_base_url.hostname.lower() not in LOCAL_HOSTS:
        raise PreflightError(
            "--base-url must use localhost, 127.0.0.1, or ::1; remote engines are forbidden"
        )
    head = load_image("head", args.head_image)
    third = load_image("third", args.third_image)
    expected_by_role = {
        "head": VisualFacts(
            blue_block_count=args.expected_head_blue_block_count,
            red_circle_horizontal_relation=args.expected_head_red_circle_relation,
        ),
        "third": VisualFacts(
            blue_block_count=args.expected_third_blue_block_count,
            red_circle_horizontal_relation=args.expected_third_red_circle_relation,
        ),
    }
    model_url = f"{base_url}/models"
    completion_url = f"{base_url}/chat/completions"
    model_response = _http_json(
        "GET",
        model_url,
        timeout_sec=args.timeout_sec,
        opener=opener,
    )
    model_check = _validate_models(model_response, args.model)
    model_check["url"] = model_url

    mode_images = {
        "text_only": [],
        "head_image": [head],
        "head_then_third_images": [head, third],
    }
    mode_expected_facts = {
        "text_only": [],
        "head_image": [expected_by_role["head"]],
        "head_then_third_images": [
            expected_by_role["head"],
            expected_by_role["third"],
        ],
    }
    requests: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for mode in REQUEST_MODES:
        artifacts = mode_images[mode]
        metadata: dict[str, Any] = {
            "url": completion_url,
            "model": args.model,
            "request_mode": mode,
            "image_count": len(artifacts),
            "image_order": [artifact.role for artifact in artifacts],
            "image_sha256": [artifact.sha256 for artifact in artifacts],
            "fallback_count": 0,
        }
        payload = build_chat_payload(
            model=args.model,
            mode=mode,
            images=artifacts,
            max_tokens=args.max_tokens,
            response_format_json_object=args.response_format_json_object,
        )
        try:
            response = _http_json(
                "POST",
                completion_url,
                timeout_sec=args.timeout_sec,
                payload=payload,
                opener=opener,
            )
            metadata.update(
                _validate_chat_response(
                    response,
                    exact_model=args.model,
                    mode=mode,
                    expected_visual_facts=mode_expected_facts[mode],
                )
            )
            metadata["status"] = "passed"
        except PreflightError as exc:
            metadata["status"] = "failed"
            metadata["error"] = str(exc)
            errors.append({"request_mode": mode, "error": str(exc)})
        requests.append(metadata)

    report = {
        "schema_version": 1,
        "recorded_at_utc": _utc_now(),
        "status": "failed" if errors else "passed",
        "engine_base_url": base_url,
        "model": args.model,
        "expected_visual_facts": {
            role: {
                "blue_block_count": facts.blue_block_count,
                "red_circle_horizontal_relation": facts.red_circle_horizontal_relation,
            }
            for role, facts in expected_by_role.items()
        },
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": args.max_tokens,
            "thinking_enabled": True,
            "response_format_json_object": bool(args.response_format_json_object),
        },
        "model_check": model_check,
        "requests": requests,
        "fallback_count": 0,
        "raw_images_archived": False,
        "raw_prompts_archived": False,
        "raw_responses_archived": False,
        "errors": errors,
    }
    return report


def write_report_atomic(path: Path, report: dict[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def execute(args: argparse.Namespace, *, opener: UrlOpen = _direct_urlopen) -> int:
    try:
        report = run_preflight(args, opener=opener)
    except PreflightError as exc:
        report = {
            "schema_version": 1,
            "recorded_at_utc": _utc_now(),
            "status": "failed",
            "engine_base_url": str(args.base_url).rstrip("/"),
            "model": args.model,
            "fallback_count": 0,
            "raw_images_archived": False,
            "raw_prompts_archived": False,
            "raw_responses_archived": False,
            "errors": [{"stage": "setup_or_model_check", "error": str(exc)}],
        }
    write_report_atomic(args.output, report)
    print(json.dumps({"status": report["status"], "report": str(args.output)}, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
