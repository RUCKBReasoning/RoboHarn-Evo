from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import re
import time
import uuid
from typing import Any


_DATA_IMAGE_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)
_SUPPORTED_FIELDS = {
    "background",
    "include",
    "input",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "model",
    "parallel_tool_calls",
    "previous_response_id",
    "prompt",
    "reasoning",
    "service_tier",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "truncation",
    "user",
}


class ResponsesCompatError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str = "invalid_request",
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.param = param
        self.code = code
        self.status_code = status_code

    def as_openai_error(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": "invalid_request_error",
                "param": self.param,
                "code": self.code,
            }
        }


@dataclass(frozen=True)
class PreparedResponsesRequest:
    model: str
    instructions: Any
    developer_instructions: str
    user_input: list[dict[str, Any]]
    effort: str | None
    summary: str | None
    output_schema: dict[str, Any] | None
    text_config: dict[str, Any]
    stream: bool
    store: bool
    metadata: dict[str, str]


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _unsupported(payload: dict[str, Any], name: str, *, allowed: tuple[Any, ...] = (None,)) -> None:
    if name in payload and payload[name] not in allowed:
        raise ResponsesCompatError(
            f"Parameter '{name}' is not supported by the local Codex account adapter.",
            param=name,
            code="unsupported_parameter",
        )


def _validate_data_image(url: str, *, param: str, max_image_bytes: int) -> None:
    match = _DATA_IMAGE_RE.match(url)
    if match is None:
        raise ResponsesCompatError(
            "Only base64 data URLs are supported for input images by this local adapter.",
            param=param,
            code="unsupported_image_url",
        )
    try:
        decoded = base64.b64decode(match.group(2), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ResponsesCompatError(
            "The input image contains invalid base64 data.",
            param=param,
            code="invalid_image",
        ) from exc
    if len(decoded) > max_image_bytes:
        raise ResponsesCompatError(
            f"The decoded input image exceeds the {max_image_bytes}-byte limit.",
            param=param,
            code="image_too_large",
            status_code=413,
        )


def _content_parts(
    content: Any,
    *,
    param: str,
    allow_images: bool,
    max_image_bytes: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    if isinstance(content, str):
        return [content], []
    if not isinstance(content, list):
        raise ResponsesCompatError(
            "Message content must be a string or an array of content parts.",
            param=param,
        )

    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for index, item in enumerate(content):
        item_param = f"{param}[{index}]"
        if isinstance(item, str):
            texts.append(item)
            continue
        if not isinstance(item, dict):
            raise ResponsesCompatError("Content parts must be objects.", param=item_param)
        item_type = str(item.get("type", ""))
        if item_type in {"input_text", "output_text", "text"}:
            texts.append(str(item.get("text", "")))
            continue
        if item_type in {"input_image", "image_url"}:
            if not allow_images:
                raise ResponsesCompatError(
                    "Images are not supported inside system or developer instructions.",
                    param=item_param,
                    code="unsupported_instruction_content",
                )
            raw_url = item.get("image_url", item.get("url", ""))
            if isinstance(raw_url, dict):
                raw_url = raw_url.get("url", "")
            url = str(raw_url)
            if not url:
                raise ResponsesCompatError("An input image URL is required.", param=item_param)
            _validate_data_image(url, param=item_param, max_image_bytes=max_image_bytes)
            detail = item.get("detail")
            if detail is None and isinstance(item.get("image_url"), dict):
                detail = item["image_url"].get("detail")
            if detail is not None and (
                not isinstance(detail, str)
                or detail not in {"auto", "low", "high", "original"}
            ):
                raise ResponsesCompatError(
                    "Input image detail must be one of auto, low, high, or original.",
                    param=f"{item_param}.detail",
                )
            image: dict[str, Any] = {"type": "image", "url": url}
            if detail in {"auto", "low", "high", "original"}:
                image["detail"] = detail
            images.append(image)
            continue
        raise ResponsesCompatError(
            f"Input content type '{item_type or '<missing>'}' is not supported.",
            param=item_param,
            code="unsupported_content_type",
        )
    return texts, images


def _format_trusted_message(role: str, texts: list[str]) -> str:
    content = "\n".join(text for text in texts if text)
    return f"[{role.upper()}]\n{content}" if content else ""


def _append_conversation_text(user_input: list[dict[str, Any]], role: str, texts: list[str]) -> None:
    text = "\n".join(part for part in texts if part)
    if not text:
        return
    if role == "user":
        user_input.append({"type": "text", "text": text})
    else:
        user_input.append(
            {
                "type": "text",
                "text": f"<prior_message role={role!r}>\n{text}\n</prior_message>",
            }
        )


def _parse_message(
    item: dict[str, Any],
    *,
    param: str,
    trusted: list[str],
    user_input: list[dict[str, Any]],
    max_image_bytes: int,
) -> int:
    role = str(item.get("role", "user")).lower()
    if role not in {"system", "developer", "user", "assistant"}:
        raise ResponsesCompatError(f"Unsupported message role '{role}'.", param=f"{param}.role")
    content = item.get("content", "")
    texts, images = _content_parts(
        content,
        param=f"{param}.content",
        allow_images=role in {"user", "assistant"},
        max_image_bytes=max_image_bytes,
    )
    if role in {"system", "developer"}:
        rendered = _format_trusted_message(role, texts)
        if rendered:
            trusted.append(rendered)
        return 0
    _append_conversation_text(user_input, role, texts)
    user_input.extend(images)
    return len(images)


def _parse_instruction_value(
    value: Any,
    *,
    trusted: list[str],
    max_image_bytes: int,
) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if value:
            trusted.append(value)
        return
    if not isinstance(value, list):
        raise ResponsesCompatError(
            "'instructions' must be a string or an array of instruction messages.",
            param="instructions",
        )
    ignored_user_input: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ResponsesCompatError("Instruction items must be objects.", param=f"instructions[{index}]")
        role = str(item.get("role", "developer")).lower()
        if role not in {"system", "developer"}:
            raise ResponsesCompatError(
                "Only system and developer messages are supported in the instructions array.",
                param=f"instructions[{index}].role",
                code="unsupported_instruction_role",
            )
        _parse_message(
            item,
            param=f"instructions[{index}]",
            trusted=trusted,
            user_input=ignored_user_input,
            max_image_bytes=max_image_bytes,
        )


def _parse_text_config(value: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if value is None:
        return {"format": {"type": "text"}}, None
    if not isinstance(value, dict):
        raise ResponsesCompatError("'text' must be an object.", param="text")
    if value.get("verbosity") is not None:
        raise ResponsesCompatError(
            "Parameter 'text.verbosity' is not supported by the local Codex account adapter.",
            param="text.verbosity",
            code="unsupported_parameter",
        )
    format_value = value.get("format", {"type": "text"})
    if not isinstance(format_value, dict):
        raise ResponsesCompatError("'text.format' must be an object.", param="text.format")
    format_type = str(format_value.get("type", "text"))
    if format_type == "text":
        return {"format": {"type": "text"}}, None
    if format_type == "json_object":
        # app-server outputSchema is strict Structured Outputs, not JSON mode.
        # A schema with only {"type": "object"} is rejected upstream, and
        # closing it with additionalProperties=false would allow only {}.
        return {"format": dict(format_value)}, None
    if format_type == "json_schema":
        name = format_value.get("name")
        if not isinstance(name, str) or not name:
            raise ResponsesCompatError(
                "'text.format.name' is required for json_schema output.",
                param="text.format.name",
                code="missing_required_parameter",
            )
        if "strict" in format_value and not isinstance(format_value.get("strict"), bool):
            raise ResponsesCompatError(
                "'text.format.strict' must be a boolean.",
                param="text.format.strict",
            )
        schema = format_value.get("schema")
        if not isinstance(schema, dict):
            raise ResponsesCompatError(
                "'text.format.schema' must be a JSON Schema object.",
                param="text.format.schema",
            )
        return {"format": dict(format_value)}, schema
    raise ResponsesCompatError(
        f"Text format '{format_type}' is not supported.",
        param="text.format.type",
        code="unsupported_text_format",
    )


def prepare_responses_request(
    payload: Any,
    *,
    expected_model: str,
    default_effort: str | None,
    max_images: int,
    max_image_bytes: int = 32 * 1024 * 1024,
) -> PreparedResponsesRequest:
    if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images <= 0:
        raise ValueError("max_images must be a positive integer")
    if not isinstance(payload, dict):
        raise ResponsesCompatError("The request body must be a JSON object.")

    unknown_fields = sorted(set(payload) - _SUPPORTED_FIELDS)
    if unknown_fields:
        name = unknown_fields[0]
        raise ResponsesCompatError(
            f"Parameter '{name}' is not supported by the local Codex account adapter.",
            param=name,
            code="unsupported_parameter",
        )

    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ResponsesCompatError("The 'model' parameter is required.", param="model", code="missing_required_parameter")
    if model != expected_model:
        raise ResponsesCompatError(
            f"Model '{model}' is not available on this endpoint; use '{expected_model}'.",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    _unsupported(payload, "background", allowed=(None, False))
    for name in (
        "conversation",
        "max_output_tokens",
        "max_tool_calls",
        "previous_response_id",
        "prompt",
        "temperature",
        "top_logprobs",
        "top_p",
    ):
        _unsupported(payload, name)
    if payload.get("tools") not in (None, []):
        _unsupported(payload, "tools")
    if payload.get("tool_choice") not in (None, "none", "auto"):
        _unsupported(payload, "tool_choice")
    if payload.get("include") not in (None, []):
        _unsupported(payload, "include")
    if payload.get("truncation") not in (None, "disabled"):
        _unsupported(payload, "truncation")
    if payload.get("service_tier") is not None:
        _unsupported(payload, "service_tier")
    if payload.get("stream_options") not in (None, {}):
        _unsupported(payload, "stream_options")
    if payload.get("parallel_tool_calls") not in (None, False, True):
        raise ResponsesCompatError("'parallel_tool_calls' must be a boolean.", param="parallel_tool_calls")

    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise ResponsesCompatError("'stream' must be a boolean.", param="stream")
    store = payload.get("store", False)
    if not isinstance(store, bool):
        raise ResponsesCompatError("'store' must be a boolean.", param="store")
    if store:
        raise ResponsesCompatError(
            "The ChatGPT/Codex account adapter supports only ephemeral local responses; use store=false.",
            param="store",
            code="unsupported_parameter",
        )

    metadata_value = payload.get("metadata", {})
    if metadata_value is None:
        metadata_value = {}
    if not isinstance(metadata_value, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata_value.items()
    ):
        raise ResponsesCompatError("'metadata' must be an object of string values.", param="metadata")

    reasoning = payload.get("reasoning", {})
    if reasoning is None:
        reasoning = {}
    if not isinstance(reasoning, dict):
        raise ResponsesCompatError("'reasoning' must be an object.", param="reasoning")
    effort_value = reasoning.get("effort", default_effort)
    if effort_value is not None and not isinstance(effort_value, str):
        raise ResponsesCompatError("'reasoning.effort' must be a string.", param="reasoning.effort")
    summary_value = reasoning.get("summary", reasoning.get("generate_summary"))
    if summary_value is not None and (
        not isinstance(summary_value, str)
        or summary_value not in {"auto", "concise", "detailed", "none"}
    ):
        raise ResponsesCompatError("Unsupported reasoning summary value.", param="reasoning.summary")

    text_config, output_schema = _parse_text_config(payload.get("text"))
    trusted: list[str] = []
    _parse_instruction_value(payload.get("instructions"), trusted=trusted, max_image_bytes=max_image_bytes)

    if "input" not in payload:
        raise ResponsesCompatError("The 'input' parameter is required.", param="input", code="missing_required_parameter")
    raw_input = payload["input"]
    user_input: list[dict[str, Any]] = []
    image_count = 0
    if isinstance(raw_input, str):
        user_input.append({"type": "text", "text": raw_input})
    elif isinstance(raw_input, list):
        for index, item in enumerate(raw_input):
            param = f"input[{index}]"
            if isinstance(item, str):
                user_input.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                raise ResponsesCompatError("Input items must be strings or objects.", param=param)
            item_type = str(item.get("type", ""))
            if item_type in {"input_text", "input_image"}:
                texts, images = _content_parts(
                    [item],
                    param=param,
                    allow_images=True,
                    max_image_bytes=max_image_bytes,
                )
                _append_conversation_text(user_input, "user", texts)
                user_input.extend(images)
                image_count += len(images)
                continue
            if item_type in {"message", ""} or "role" in item:
                image_count += _parse_message(
                    item,
                    param=param,
                    trusted=trusted,
                    user_input=user_input,
                    max_image_bytes=max_image_bytes,
                )
                continue
            raise ResponsesCompatError(
                f"Input item type '{item_type}' is not supported.",
                param=f"{param}.type",
                code="unsupported_input_item",
            )
    else:
        raise ResponsesCompatError("'input' must be a string or an array.", param="input")

    if image_count > max_images:
        raise ResponsesCompatError(
            f"The request exceeds the {max_images}-image limit.",
            param="input",
            code="too_many_images",
        )
    if not user_input:
        user_input.append({"type": "text", "text": ""})

    return PreparedResponsesRequest(
        model=model,
        instructions=payload.get("instructions"),
        developer_instructions="\n\n".join(part for part in trusted if part),
        user_input=user_input,
        effort=effort_value,
        summary=summary_value,
        output_schema=output_schema,
        text_config=text_config,
        stream=stream,
        store=store,
        metadata=dict(metadata_value),
    )


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    source = usage.get("last") if isinstance(usage.get("last"), dict) else usage
    input_tokens = int(source.get("inputTokens", source.get("input_tokens", 0)) or 0)
    cached_tokens = int(source.get("cachedInputTokens", source.get("cached_input_tokens", 0)) or 0)
    output_tokens = int(source.get("outputTokens", source.get("output_tokens", 0)) or 0)
    reasoning_tokens = int(
        source.get("reasoningOutputTokens", source.get("reasoning_output_tokens", 0)) or 0
    )
    total_tokens = int(source.get("totalTokens", source.get("total_tokens", input_tokens + output_tokens)) or 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": total_tokens,
    }


def build_response_object(
    request: PreparedResponsesRequest,
    *,
    text: str,
    usage: dict[str, Any] | None,
    response_id: str | None = None,
    message_id: str | None = None,
    created_at: int | None = None,
    status: str = "completed",
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response_id = response_id or new_response_id()
    message_id = message_id or new_message_id()
    created_at = int(time.time()) if created_at is None else int(created_at)
    output: list[dict[str, Any]] = []
    if status == "completed":
        output = [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ]
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": int(time.time()) if status == "completed" else None,
        "status": status,
        "error": error,
        "incomplete_details": None,
        "instructions": request.instructions,
        "max_output_tokens": None,
        "model": request.model,
        "output": output,
        "output_text": text if status == "completed" else "",
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": {"effort": request.effort, "summary": request.summary},
        "store": request.store,
        "text": request.text_config,
        "tool_choice": "none",
        "tools": [],
        "metadata": request.metadata,
        "usage": normalize_usage(usage),
    }
