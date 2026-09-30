"""Fail-closed multimodal transport for offline Phase A2 reflection.

This module owns only model I/O and its audit boundary.  It does not discover
expert data, select keyframes, validate reflection semantics, publish an
experience, or mutate runtime retrieval.  Callers must provide already bounded
image bytes and an explicit operator authorization for image egress.

The capability preflight is active rather than declarative: it sends a
synthetic, randomly permuted four-colour image and accepts the backend only if
strict structured output identifies the permutation exactly.  A text-only
backend therefore cannot silently pass by echoing a capability flag.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
from io import BytesIO
import json
import re
import secrets
from typing import Any, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw


_DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_TOTAL_IMAGE_BYTES = 32 * 1024 * 1024
_DEFAULT_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_SCHEMA_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_SAFE_ERROR_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.\[\]-]{1,160}$")
_SAFE_PROVIDER_ERROR_TYPES = frozenset(
    {
        "api_error",
        "authentication_error",
        "invalid_request_error",
        "not_found_error",
        "permission_error",
        "rate_limit_error",
        "server_error",
    }
)
_SAFE_PROVIDER_ERROR_CODES = frozenset(
    {
        "codex_app_server_error",
        "context_length_exceeded",
        "image_too_large",
        "invalid_image",
        "invalid_json_schema",
        "invalid_request",
        "method_not_allowed",
        "missing_required_parameter",
        "model_not_found",
        "not_found",
        "rate_limit_exceeded",
        "server_overloaded",
        "too_many_images",
        "unsupported_backend",
        "unsupported_content_type",
        "unsupported_image_url",
        "unsupported_input_item",
        "unsupported_instruction_content",
        "unsupported_instruction_role",
        "unsupported_parameter",
        "unsupported_text_format",
        "upstream_timeout",
    }
)
_SAFE_PROVIDER_ERROR_PARAM_ROOTS = frozenset(
    {
        "background",
        "include",
        "input",
        "instructions",
        "metadata",
        "model",
        "reasoning",
        "service_tier",
        "store",
        "stream",
        "text",
        "tool_choice",
        "tools",
        "truncation",
    }
)
_SECRET_VALUE_RE = re.compile(
    r"(?:"
    r"authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/-]{4,}"
    r"|\bbearer\s+[A-Za-z0-9._~+/-]{8,}"
    r"|\bsk-[A-Za-z0-9_-]{8,}"
    r"|[\"'](?:api[_-]?key|access[_-]?token|secret)[\"']\s*:\s*"
    r"[\"'][^\"']{4,}[\"']"
    r")",
    re.IGNORECASE,
)
_DATA_URL_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,", re.IGNORECASE)
_RAW_SHA256_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_LOCAL_PATH_RE = re.compile(
    r"(?:"
    r"file://"
    r"|(?<![A-Za-z0-9])/(?!/)[^\s\"']+"
    r"|(?<![A-Za-z0-9])\.\.[/\\]"
    r"|(?<![A-Za-z0-9])\.[/\\]"
    r"|(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+[/\\])+[A-Za-z0-9_.-]+\.(?:jsonl?|ya?ml|toml|png|jpe?g|np[yz]|pkl|hdf5|csv|txt|log|py)\b"
    r"|(?<![A-Za-z0-9])[A-Za-z]:\\[^\s\"']+"
    r")",
    re.IGNORECASE,
)
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|secret|token)"
    r"\s*[:=]\s*[^\s\"',}]{4,}",
    re.IGNORECASE,
)

_COLOUR_RGB: dict[str, tuple[int, int, int]] = {
    "blue": (25, 80, 220),
    "green": (20, 175, 75),
    "red": (220, 35, 35),
    "yellow": (245, 205, 25),
}


class MultimodalTransportError(RuntimeError):
    """Base error whose messages never contain request/response bodies."""


class MultimodalHTTPError(MultimodalTransportError):
    """Body-free HTTP failure with an allowlisted diagnostic fingerprint."""

    def __init__(
        self,
        status_code: int,
        *,
        error_type: str | None = None,
        error_code: str | None = None,
        error_param: str | None = None,
        error_body_sha256: str | None = None,
    ) -> None:
        super().__init__(f"multimodal endpoint returned HTTP status {status_code}")
        self.status_code = int(status_code)
        self.error_type = error_type
        self.error_code = error_code
        self.error_param = error_param
        self.error_body_sha256 = error_body_sha256

    def audit_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "transport_failure_stage": "http_response",
            "http_status": self.status_code,
        }
        for key, value in (
            ("provider_error_type", self.error_type),
            ("provider_error_code", self.error_code),
            ("provider_error_param", self.error_param),
            ("error_body_sha256", self.error_body_sha256),
        ):
            if value is not None:
                result[key] = value
        return result


class ImageEgressDenied(MultimodalTransportError):
    """Raised before network I/O when image egress was not authorized."""


class CapabilityPreflightError(MultimodalTransportError):
    """Raised when image or structured-output capability is not proven."""


class MultimodalResponseError(MultimodalTransportError):
    """Raised when an untrusted backend response is invalid or too large."""


class ModelInputProjectionError(MultimodalTransportError):
    """Raised before I/O when projected model text contains private material."""


class ProviderSchemaProjectionError(MultimodalTransportError):
    """Raised before I/O when a local schema cannot be safely projected."""


@dataclass(frozen=True, slots=True)
class ImageEgressAuthorization:
    """Explicit operator decision for one bounded Phase A2 image-egress scope.

    ``granted`` defaults to false.  The assertion is audit metadata, not an API
    credential, and only its digest is emitted by this module.
    """

    granted: bool = False
    scope: str = "phase_a2_offline_keyframes"
    operator_assertion: str = "not_authorized"

    def __post_init__(self) -> None:
        if type(self.granted) is not bool:
            raise ValueError("authorization granted must be a boolean")
        if (
            not isinstance(self.scope, str)
            or not self.scope.strip()
            or len(self.scope) > 160
        ):
            raise ValueError("authorization scope must be 1..160 characters")
        if (
            not isinstance(self.operator_assertion, str)
            or not self.operator_assertion.strip()
            or len(self.operator_assertion) > 256
        ):
            raise ValueError("authorization assertion must be 1..256 characters")
        if any(character in "\r\n\x00" for character in self.operator_assertion):
            raise ValueError("authorization assertion contains a control character")

    @classmethod
    def operator_granted(
        cls,
        *,
        assertion: str = "explicit_cli_authorization",
        scope: str = "phase_a2_offline_keyframes",
    ) -> ImageEgressAuthorization:
        if (
            not isinstance(assertion, str)
            or not isinstance(scope, str)
            or not assertion.strip()
            or not scope.strip()
        ):
            raise ValueError("authorization assertion and scope must be non-empty")
        return cls(granted=True, scope=scope, operator_assertion=assertion)

    def audit_sha256(self) -> str:
        payload = {
            "granted": self.granted,
            "operator_assertion": self.operator_assertion,
            "scope": self.scope,
        }
        return _canonical_json_sha256(payload)

    def require(self) -> None:
        if not self.granted:
            raise ImageEgressDenied(
                "image egress is not authorized; no capability probe or expert "
                "keyframe was sent"
            )


@dataclass(frozen=True, slots=True)
class MultimodalImage:
    """One bounded image tied to a validated evidence item."""

    evidence_id: str
    mime_type: str
    content: bytes = field(repr=False)
    detail: str = "high"

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("image evidence_id must be non-empty")
        if self.mime_type not in _SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"unsupported image mime type: {self.mime_type!r}")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("image content must be non-empty bytes")
        if self.detail not in {"auto", "low", "high", "original"}:
            raise ValueError("image detail must be auto, low, high, or original")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """Audit-safe proof that one transport instance passed active preflight."""

    backend: str
    service_url: str
    model: str
    supports_images: bool
    image_roundtrip_verified: bool
    structured_output_verified: bool
    text_only_fallback_detected: bool
    request_sha256: str
    response_sha256: str
    challenge_image_sha256: str
    authorization_sha256: str
    configuration_sha256: str
    service_max_images_per_request: int | None
    service_schema_profile: str | None
    service_health_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "service_url": self.service_url,
            "model": self.model,
            "supports_images": self.supports_images,
            "image_roundtrip_verified": self.image_roundtrip_verified,
            "structured_output_verified": self.structured_output_verified,
            "text_only_fallback_detected": self.text_only_fallback_detected,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "challenge_image_sha256": self.challenge_image_sha256,
            "authorization_sha256": self.authorization_sha256,
            "configuration_sha256": self.configuration_sha256,
            "service_max_images_per_request": self.service_max_images_per_request,
            "service_schema_profile": self.service_schema_profile,
            "service_health_sha256": self.service_health_sha256,
        }


@dataclass(frozen=True, slots=True)
class MultimodalCompletion:
    """Parsed untrusted JSON plus a body-free transport audit."""

    output: Mapping[str, Any]
    audit: Mapping[str, Any]


@runtime_checkable
class JsonHttpSender(Protocol):
    """Injectable HTTP boundary used by deterministic tests."""

    def __call__(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_sec: float,
        max_response_bytes: int,
    ) -> bytes | Mapping[str, Any]: ...


@runtime_checkable
class ServiceCapabilityProbe(Protocol):
    """Read the local gateway's body-bounded, secret-free health contract."""

    def __call__(
        self,
        *,
        url: str,
        timeout_sec: float,
        max_response_bytes: int,
    ) -> bytes | Mapping[str, Any]: ...


@runtime_checkable
class EvidenceImageResolver(Protocol):
    """Resolve only backend-selected evidence IDs to bounded image bytes."""

    def __call__(
        self,
        request: Any,
        evidence_ids: Sequence[str],
    ) -> Sequence[MultimodalImage]: ...


def _strict_json_object(raw: bytes | str, *, label: str) -> dict[str, Any]:
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MultimodalResponseError(f"{label} is not UTF-8 JSON") from exc
    else:
        text = raw

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise MultimodalResponseError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise MultimodalResponseError(f"{label} must be a JSON object")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not strict JSON") from exc


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


_PROVIDER_SCHEMA_PROJECTION_VERSION = "openai_strict_json_schema_subset_v1"
_PROVIDER_SCHEMA_DROPPED_KEYWORDS = frozenset(
    {
        "$schema",
        # These assertions are enforced again by the local typed quality gates,
        # but are not accepted by OpenAI Structured Outputs.
        "uniqueItems",
        "minLength",
        "maxLength",
    }
)
_PROVIDER_SCHEMA_FORBIDDEN_COMPOSITION = frozenset(
    {
        "allOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
    }
)
_PROVIDER_SCHEMA_ALLOWED_KEYWORDS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "const",
        "definitions",
        "description",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "items",
        "maxItems",
        "maximum",
        "minItems",
        "minimum",
        "multipleOf",
        "pattern",
        "properties",
        "required",
        "title",
        "type",
    }
)


def _json_type_for_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise ProviderSchemaProjectionError(
        "provider schema contains a non-JSON literal"
    )


def _inferred_schema_type(node: Mapping[str, Any]) -> str | list[str] | None:
    if "const" in node:
        return _json_type_for_literal(node["const"])
    enum = node.get("enum")
    if not isinstance(enum, list) or not enum:
        return None
    types = {_json_type_for_literal(value) for value in enum}
    if types == {"integer", "number"}:
        return "number"
    ordered = [
        value
        for value in (
            "string",
            "number",
            "integer",
            "boolean",
            "object",
            "array",
            "null",
        )
        if value in types
    ]
    return ordered[0] if len(ordered) == 1 else ordered


def project_openai_strict_output_schema(
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a local strict schema into OpenAI's supported JSON subset.

    Local quality gates remain authoritative for constraints removed here.  In
    particular, array uniqueness and non-empty strings are still checked after
    parsing; this projection only prevents the provider from rejecting the
    request before generation begins.
    """

    if not isinstance(schema, Mapping) or not schema:
        raise ProviderSchemaProjectionError(
            "provider output schema must be a non-empty object"
        )

    def clone(value: Any) -> Any:
        return json.loads(_canonical_json_bytes(value))

    def visit(value: Any, *, path: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ProviderSchemaProjectionError(
                f"provider schema node {path} must be an object"
            )
        forbidden = set(value).intersection(_PROVIDER_SCHEMA_FORBIDDEN_COMPOSITION)
        if forbidden:
            raise ProviderSchemaProjectionError(
                f"provider schema node {path} uses an unsupported composition keyword"
            )
        unknown = set(value).difference(
            _PROVIDER_SCHEMA_ALLOWED_KEYWORDS,
            _PROVIDER_SCHEMA_DROPPED_KEYWORDS,
            _PROVIDER_SCHEMA_FORBIDDEN_COMPOSITION,
        )
        if unknown:
            raise ProviderSchemaProjectionError(
                f"provider schema node {path} uses an unknown keyword"
            )
        output: dict[str, Any] = {}
        for key, child in value.items():
            if key in _PROVIDER_SCHEMA_DROPPED_KEYWORDS:
                continue
            if key == "properties":
                if not isinstance(child, Mapping):
                    raise ProviderSchemaProjectionError(
                        f"provider schema node {path}.properties must be an object"
                    )
                output[key] = {
                    str(name): visit(item, path=f"{path}.properties.{name}")
                    for name, item in child.items()
                }
            elif key == "items":
                output[key] = visit(child, path=f"{path}.items")
            elif key == "anyOf":
                if not isinstance(child, list) or not child:
                    raise ProviderSchemaProjectionError(
                        f"provider schema node {path}.anyOf must be a non-empty array"
                    )
                output[key] = [
                    visit(item, path=f"{path}.anyOf[{index}]")
                    for index, item in enumerate(child)
                ]
            elif key in {"$defs", "definitions"}:
                if not isinstance(child, Mapping):
                    raise ProviderSchemaProjectionError(
                        f"provider schema node {path}.{key} must be an object"
                    )
                output[key] = {
                    str(name): visit(item, path=f"{path}.{key}.{name}")
                    for name, item in child.items()
                }
            else:
                output[key] = clone(child)

        if "type" not in output:
            inferred = _inferred_schema_type(output)
            if inferred is not None:
                output["type"] = inferred
            elif "anyOf" not in output and "$ref" not in output:
                raise ProviderSchemaProjectionError(
                    f"provider schema node {path} has no explicit type"
                )

        node_type = output.get("type")
        if node_type == "object":
            properties = output.get("properties")
            if not isinstance(properties, dict):
                raise ProviderSchemaProjectionError(
                    f"provider object schema {path} has no properties"
                )
            if output.get("additionalProperties") is not False:
                raise ProviderSchemaProjectionError(
                    f"provider object schema {path} must forbid additional properties"
                )
            required = output.get("required")
            if not isinstance(required, list) or set(required) != set(properties):
                raise ProviderSchemaProjectionError(
                    f"provider object schema {path} must require every property"
                )
        if node_type == "array" and not isinstance(output.get("items"), dict):
            raise ProviderSchemaProjectionError(
                f"provider array schema {path} has no item schema"
            )
        return output

    projected = visit(schema, path="$")
    if projected.get("type") != "object":
        raise ProviderSchemaProjectionError(
            "provider output schema root must be an object"
        )
    return projected


def _provider_schema_removed_keyword_counts(
    schema: Mapping[str, Any],
) -> dict[str, int]:
    counts = {key: 0 for key in sorted(_PROVIDER_SCHEMA_DROPPED_KEYWORDS)}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in counts:
                    counts[key] += 1
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return {key: value for key, value in counts.items() if value}


def _safe_service_url(value: str) -> tuple[str, str]:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("service_url must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("service_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("service_url must not contain a query or fragment")
    normalized_path = parsed.path.rstrip("/")
    if normalized_path.endswith("/v1/responses"):
        endpoint_path = normalized_path
        base_path = normalized_path[: -len("/v1/responses")]
    else:
        base_path = normalized_path
        endpoint_path = f"{base_path}/v1/responses"
    safe_base = urlunsplit((parsed.scheme, parsed.netloc, base_path, "", ""))
    endpoint = urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))
    return safe_base, endpoint


def _default_service_capability_probe(
    *,
    url: str,
    timeout_sec: float,
    max_response_bytes: int,
) -> bytes:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout_sec) as response:  # noqa: S310
            raw = response.read(max_response_bytes + 1)
    except HTTPError as exc:
        raise CapabilityPreflightError(
            f"service capability endpoint returned HTTP status {exc.code}"
        ) from exc
    except URLError as exc:
        raise CapabilityPreflightError(
            "service capability endpoint connection failed"
        ) from exc
    if len(raw) > max_response_bytes:
        raise CapabilityPreflightError(
            "service capability response exceeds byte budget"
        )
    return raw


def _default_json_sender(
    *,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_sec: float,
    max_response_bytes: int,
) -> bytes:
    encoded = _canonical_json_bytes(payload)
    request = Request(
        url,
        data=encoded,
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:  # noqa: S310
            raw = response.read(max_response_bytes + 1)
    except HTTPError as exc:
        # Read only a bounded body and retain allowlisted OpenAI error tokens.
        # The message and body are never included in the exception or audit.
        error_type: str | None = None
        error_code: str | None = None
        error_param: str | None = None
        error_body_sha256: str | None = None
        try:
            error_raw = exc.read(min(max_response_bytes, 64 * 1024) + 1)
        except Exception:
            error_raw = b""
        if error_raw and len(error_raw) <= min(max_response_bytes, 64 * 1024):
            error_body_sha256 = hashlib.sha256(error_raw).hexdigest()
            try:
                parsed_error = _strict_json_object(
                    error_raw,
                    label="multimodal error response",
                )
            except MultimodalResponseError:
                parsed_error = {}
            error = parsed_error.get("error")
            if isinstance(error, Mapping):
                tokens: list[str | None] = []
                for field_name in ("type", "code", "param"):
                    value = error.get(field_name)
                    safe_value: str | None = None
                    if (
                        isinstance(value, str)
                        and _SAFE_ERROR_TOKEN_RE.fullmatch(value)
                        and _SECRET_VALUE_RE.search(value) is None
                        and _RAW_SHA256_RE.search(value) is None
                    ):
                        if (
                            field_name == "type"
                            and value in _SAFE_PROVIDER_ERROR_TYPES
                        ):
                            safe_value = value
                        elif (
                            field_name == "code"
                            and value in _SAFE_PROVIDER_ERROR_CODES
                        ):
                            safe_value = value
                        elif field_name == "param":
                            root = re.split(r"[.\[]", value, maxsplit=1)[0]
                            if root in _SAFE_PROVIDER_ERROR_PARAM_ROOTS:
                                safe_value = value
                    tokens.append(safe_value)
                error_type, error_code, error_param = tokens
        raise MultimodalHTTPError(
            int(exc.code),
            error_type=error_type,
            error_code=error_code,
            error_param=error_param,
            error_body_sha256=error_body_sha256,
        ) from exc
    except URLError as exc:
        raise MultimodalTransportError(
            "multimodal endpoint connection failed"
        ) from exc
    if len(raw) > max_response_bytes:
        raise MultimodalResponseError("multimodal response exceeds byte budget")
    return raw


def _response_object(value: bytes | Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    if isinstance(value, Mapping):
        raw = _canonical_json_bytes(dict(value))
        return dict(value), raw
    if not isinstance(value, bytes):
        raise MultimodalResponseError("HTTP sender returned an unsupported value")
    return _strict_json_object(value, label="multimodal response"), value


def _response_output_text(response: Mapping[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    output = response.get("output")
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                text = part.get("text")
                if part.get("type") == "output_text" and isinstance(text, str):
                    texts.append(text)
        if texts:
            return "\n".join(texts)
    raise MultimodalResponseError("multimodal response has no output text")


def _challenge_png() -> tuple[bytes, tuple[str, str, str, str]]:
    order = list(_COLOUR_RGB)
    secrets.SystemRandom().shuffle(order)
    image = Image.new("RGB", (192, 192), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    boxes = ((0, 0, 95, 95), (96, 0, 191, 95), (0, 96, 95, 191), (96, 96, 191, 191))
    for colour, box in zip(order, boxes, strict=True):
        draw.rectangle(box, fill=_COLOUR_RGB[colour])
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue(), tuple(order)  # type: ignore[return-value]


def _data_url(image: MultimodalImage) -> str:
    encoded = base64.b64encode(image.content).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


@dataclass(frozen=True, slots=True)
class _RequestLocalAliases:
    segment_to_alias: Mapping[str, str]
    alias_to_segment: Mapping[str, str]
    evidence_to_alias: Mapping[str, str]
    alias_to_evidence: Mapping[str, str]
    annotation_to_alias: Mapping[str, str]
    alias_to_annotation: Mapping[str, str]
    transition_to_alias: Mapping[str, str]
    alias_to_transition: Mapping[str, str]

    def audit_dict(self) -> dict[str, Any]:
        return {
            "segments": dict(self.alias_to_segment),
            "visual_evidence": dict(self.alias_to_evidence),
            "annotations": dict(self.alias_to_annotation),
            "gripper_transitions": dict(self.alias_to_transition),
        }


def _request_local_aliases(
    request: Any,
    selected_ids: Sequence[str],
) -> _RequestLocalAliases:
    segment_ids = [str(value["segment_id"]) for value in request.segments]
    segment_to_alias = {
        value: f"seg_{index:03d}" for index, value in enumerate(segment_ids)
    }
    evidence_to_alias = {
        value: f"ev_{index:03d}" for index, value in enumerate(selected_ids)
    }
    annotation_to_alias: dict[str, str] = {}
    for index, value in enumerate(request.annotation_evidence_refs):
        if isinstance(value, Mapping) and isinstance(value.get("evidence_ref"), str):
            annotation_to_alias[value["evidence_ref"]] = f"ann_{index:03d}"
    transition_to_alias: dict[str, str] = {}
    transitions = request.visual_evidence.get("gripper_transitions", [])
    if isinstance(transitions, Sequence) and not isinstance(transitions, (str, bytes)):
        for index, value in enumerate(transitions):
            if isinstance(value, Mapping) and isinstance(
                value.get("transition_id"), str
            ):
                transition_to_alias[value["transition_id"]] = f"gt_{index:03d}"
    return _RequestLocalAliases(
        segment_to_alias=segment_to_alias,
        alias_to_segment={value: key for key, value in segment_to_alias.items()},
        evidence_to_alias=evidence_to_alias,
        alias_to_evidence={value: key for key, value in evidence_to_alias.items()},
        annotation_to_alias=annotation_to_alias,
        alias_to_annotation={value: key for key, value in annotation_to_alias.items()},
        transition_to_alias=transition_to_alias,
        alias_to_transition={value: key for key, value in transition_to_alias.items()},
    )


def _evidence_frame_index(item: Mapping[str, Any], fallback: int) -> int:
    for key in ("frame_index", "step_index"):
        value = item.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    for key in (
        "frame",
        "frame_reference",
        "frame_ref",
        "step_reference",
        "locator",
    ):
        nested = item.get(key)
        if not isinstance(nested, Mapping):
            continue
        for index_key in ("frame_index", "step_index", "index"):
            value = nested.get(index_key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return fallback


def _visual_evidence_items(visual_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("items", "evidence_items", "visual_evidence"):
        values = visual_evidence.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            result = [dict(value) for value in values if isinstance(value, Mapping)]
            if result:
                return result
    return []


def _select_spanning_evidence_ids(
    visual_evidence: Mapping[str, Any],
    *,
    limit: int,
) -> tuple[str, ...]:
    """Select role-paired, segment-spanning images without task knowledge."""

    items = _visual_evidence_items(visual_evidence)
    segment_order = visual_evidence.get("segment_order", [])
    if not isinstance(segment_order, Sequence) or isinstance(
        segment_order, (str, bytes)
    ):
        segment_order = []
    segment_ids = [value for value in segment_order if isinstance(value, str)]
    capabilities = visual_evidence.get("data_capabilities", {})
    rgb = capabilities.get("rgb", {}) if isinstance(capabilities, Mapping) else {}
    cameras_value = rgb.get("cameras", []) if isinstance(rgb, Mapping) else []
    cameras = (
        [value for value in cameras_value if isinstance(value, str)]
        if isinstance(cameras_value, Sequence)
        and not isinstance(cameras_value, (str, bytes))
        else []
    )
    segment_rank = {value: index for index, value in enumerate(segment_ids)}
    camera_rank = {value: index for index, value in enumerate(cameras)}
    role_rank = {
        "before": 0,
        "during": 1,
        "after": 2,
        "settled": 3,
        "task_feedback": 4,
        "context": 5,
    }

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for ordinal, item in enumerate(items):
        evidence_id = item.get("evidence_id")
        if (
            not isinstance(evidence_id, str)
            or not evidence_id.strip()
            or evidence_id in seen_ids
        ):
            continue
        seen_ids.add(evidence_id)
        image = item.get("image")
        image_sha256 = (
            image.get("sha256")
            if isinstance(image, Mapping) and isinstance(image.get("sha256"), str)
            else None
        )
        provenance = item.get("extraction_provenance")
        selection_reasons = (
            provenance.get("selection_reasons", [])
            if isinstance(provenance, Mapping)
            else []
        )
        reasons = tuple(
            value
            for value in selection_reasons
            if isinstance(value, str) and value
        )
        normalized.append(
            {
                "evidence_id": evidence_id,
                "segment_id": item.get("segment_id"),
                "camera": item.get("camera"),
                "temporal_role": item.get("temporal_role"),
                "frame_index": _evidence_frame_index(item, ordinal),
                "ordinal": ordinal,
                "selection_reasons": reasons,
                "image_key": (
                    f"sha256:{image_sha256}"
                    if image_sha256
                    else f"evidence:{evidence_id}"
                ),
            }
        )
    normalized.sort(
        key=lambda item: (
            item["frame_index"],
            segment_rank.get(item["segment_id"], len(segment_rank)),
            camera_rank.get(item["camera"], len(camera_rank)),
            role_rank.get(item["temporal_role"], 9),
            item["ordinal"],
        )
    )
    if len(normalized) <= limit:
        return tuple(item["evidence_id"] for item in normalized)

    if not segment_ids:
        selected_indices: list[int] = []
        for slot in range(limit):
            index = (
                0
                if limit == 1
                else round(slot * (len(normalized) - 1) / (limit - 1))
            )
            if index not in selected_indices:
                selected_indices.append(index)
        selected_indices.extend(
            index
            for index in range(len(normalized))
            if index not in selected_indices
        )
        return tuple(
            normalized[index]["evidence_id"]
            for index in sorted(selected_indices[:limit])
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_image_keys: set[str] = set()
    max_logical_evidence = limit * 2

    def add(item: Mapping[str, Any] | None) -> None:
        if item is None or len(selected) >= max_logical_evidence:
            return
        evidence_id = str(item["evidence_id"])
        if evidence_id in selected_ids:
            return
        image_key = str(item["image_key"])
        if image_key not in selected_image_keys and len(selected_image_keys) >= limit:
            return
        selected.append(dict(item))
        selected_ids.add(evidence_id)
        selected_image_keys.add(image_key)

    def can_add_group(items: Sequence[Mapping[str, Any] | None]) -> bool:
        candidates = [item for item in items if item is not None]
        if len(selected) + len(candidates) > max_logical_evidence:
            return False
        new_keys = {
            str(item["image_key"])
            for item in candidates
            if str(item["evidence_id"]) not in selected_ids
            and str(item["image_key"]) not in selected_image_keys
        }
        return len(selected_image_keys) + len(new_keys) <= limit

    # Complete before/after pairs consume two unique-image budget units when
    # their content is distinct. When there are more segments than the
    # configured request budget covers, choose them evenly over the whole
    # trajectory. Camera order comes only from the adapter bundle.
    pair_segment_ids = segment_ids
    if len(pair_segment_ids) > max(1, limit // 2):
        pair_count = max(1, limit // 2)
        if pair_count == 1:
            pair_segment_ids = [pair_segment_ids[len(pair_segment_ids) // 2]]
        else:
            pair_segment_ids = [
                pair_segment_ids[
                    round(slot * (len(pair_segment_ids) - 1) / (pair_count - 1))
                ]
                for slot in range(pair_count)
            ]
    primary_camera = cameras[0] if cameras else None
    for segment_id in pair_segment_ids:
        for role in ("before", "after"):
            boundary_reason = "segment_start" if role == "before" else "segment_end"
            matches = [
                item
                for item in normalized
                if item["segment_id"] == segment_id
                and item["temporal_role"] == role
                and (primary_camera is None or item["camera"] == primary_camera)
            ]
            matches.sort(
                key=lambda item: (
                    boundary_reason not in item["selection_reasons"],
                    item["frame_index"] if role == "before" else -item["frame_index"],
                    item["ordinal"],
                )
            )
            if not matches:
                matches = [
                    item
                    for item in normalized
                    if item["segment_id"] == segment_id
                    and item["temporal_role"] == role
                ]
                matches.sort(
                    key=lambda item: (
                        boundary_reason not in item["selection_reasons"],
                        item["frame_index"] if role == "before" else -item["frame_index"],
                        item["ordinal"],
                    )
                )
            add(matches[0] if matches else None)

    # A bounded low-dimensional gripper scan may identify action-stage state
    # transitions that do not coincide with segment endpoints.  Before using
    # spare slots for extra cameras or generic context, cover one transition's
    # primary-camera before/after pair.  Selection depends only on structured
    # refs and adapter camera order, never instruction or annotation text.
    normalized_by_id = {item["evidence_id"]: item for item in normalized}
    transitions = visual_evidence.get("gripper_transitions", [])
    if isinstance(transitions, Sequence) and not isinstance(transitions, (str, bytes)):
        selected_ids_now = {item["evidence_id"] for item in selected}
        for transition in transitions:
            if not isinstance(transition, Mapping):
                continue
            refs = transition.get("supporting_evidence_refs")
            if not isinstance(refs, Mapping):
                continue

            def primary_ref(role: str) -> dict[str, Any] | None:
                values = refs.get(role, [])
                if not isinstance(values, Sequence) or isinstance(
                    values, (str, bytes)
                ):
                    return None
                candidates = [
                    normalized_by_id[value]
                    for value in values
                    if isinstance(value, str)
                    and value in normalized_by_id
                    and (
                        primary_camera is None
                        or normalized_by_id[value]["camera"] == primary_camera
                    )
                ]
                return candidates[0] if candidates else None

            pair = (primary_ref("before"), primary_ref("after"))
            if any(value is None for value in pair):
                continue
            uncovered = [
                value
                for value in pair
                if value is not None and value["evidence_id"] not in selected_ids_now
            ]
            if not uncovered:
                continue
            if not can_add_group(uncovered):
                continue
            for value in pair:
                add(value)
            break

    # Release-settled and task-feedback observations have a distinct evidence
    # role and take precedence over redundant camera coverage.  Derived ROI
    # presentations are planned by the hierarchical presentation layer, not by
    # this legacy original-frame selector.
    if len(selected_image_keys) < limit:
        for role in ("settled", "task_feedback"):
            for item in normalized:
                if item["temporal_role"] == role:
                    add(item)

    # Use remaining slots for cross-camera trajectory endpoints, then optional
    # during/context evidence and finally chronological fallback. No text or
    # task identifier affects this selection.
    if segment_ids:
        for camera in cameras[1:]:
            for segment_id, role in (
                (segment_ids[0], "before"),
                (segment_ids[-1], "after"),
            ):
                matches = [
                    item
                    for item in normalized
                    if item["segment_id"] == segment_id
                    and item["temporal_role"] == role
                    and item["camera"] == camera
                ]
                boundary_reason = (
                    "segment_start" if role == "before" else "segment_end"
                )
                matches.sort(
                    key=lambda item: (
                        boundary_reason not in item["selection_reasons"],
                        item["frame_index"] if role == "before" else -item["frame_index"],
                        item["ordinal"],
                    )
                )
                add(matches[0] if matches else None)
    if len(selected_image_keys) < limit:
        for role in (
            "during",
            "context",
            "before",
            "after",
        ):
            for item in normalized:
                if item["temporal_role"] == role:
                    add(item)
    if len(selected_image_keys) < limit:
        for item in normalized:
            add(item)
    selected.sort(key=lambda item: (item["frame_index"], item["ordinal"]))
    return tuple(item["evidence_id"] for item in selected)


def _model_request_projection(
    request: Any,
    *,
    selected_ids: Sequence[str],
    aliases: _RequestLocalAliases,
) -> dict[str, Any]:
    """Build an explicit allowlist projection for the external model.

    Local paths, source IDs, hashes, raw robot values, dataset paths, episode
    seeds, and provenance never enter this projection.  The complete request
    remains available locally to the quality gate and audit sidecars.
    """

    selected = set(selected_ids)
    visual = request.visual_evidence
    items: list[dict[str, Any]] = []
    for raw in _visual_evidence_items(visual):
        evidence_id = raw.get("evidence_id")
        if evidence_id not in selected:
            continue
        frame = raw.get("frame")
        frame_index = frame.get("index") if isinstance(frame, Mapping) else None
        action = raw.get("action_range")
        action_projection: dict[str, Any] = {}
        if isinstance(action, Mapping):
            for key in ("source_type", "start_inclusive", "end_inclusive", "semantics"):
                if key in action:
                    action_projection[key] = action[key]
        grippers: list[dict[str, Any]] = []
        values = raw.get("gripper_states", [])
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for value in values:
                if not isinstance(value, Mapping):
                    continue
                grippers.append(
                    {
                        key: value[key]
                        for key in (
                            "channel_id",
                            "source_type",
                            "state",
                            "confidence",
                        )
                        if key in value
                    }
                )
        items.append(
            {
                "evidence_id": aliases.evidence_to_alias[evidence_id],
                "segment_id": aliases.segment_to_alias.get(
                    str(raw.get("segment_id")), "invalid_segment_alias"
                ),
                "segment_index": raw.get("segment_index"),
                "frame_index": frame_index,
                "camera": raw.get("camera"),
                "temporal_role": raw.get("temporal_role"),
                "action_range": action_projection,
                "gripper_states": grippers,
                "evidence_source": raw.get("evidence_source"),
                "confidence": raw.get("confidence"),
            }
        )

    comparisons: list[dict[str, Any]] = []
    values = visual.get("comparisons", [])
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for raw in values:
            if not isinstance(raw, Mapping):
                continue
            before_id = raw.get("before_evidence_id")
            after_id = raw.get("after_evidence_id")
            if before_id not in selected or after_id not in selected:
                continue
            visual_delta = raw.get("visual_delta")
            projected_delta: dict[str, Any] = {}
            if isinstance(visual_delta, Mapping):
                for key in (
                    "source_type",
                    "metric",
                    "shape_equal",
                    "hash_changed",
                    "normalized_mean_absolute_difference",
                    "semantic_interpretation",
                    "causal_attribution",
                ):
                    if key in visual_delta:
                        projected_delta[key] = visual_delta[key]
            gripper_deltas: list[dict[str, Any]] = []
            raw_deltas = raw.get("gripper_deltas", [])
            if isinstance(raw_deltas, Sequence) and not isinstance(
                raw_deltas, (str, bytes)
            ):
                for delta in raw_deltas:
                    if not isinstance(delta, Mapping):
                        continue
                    gripper_deltas.append(
                        {
                            **{
                                key: delta[key]
                                for key in (
                                    "channel_id",
                                    "source_type",
                                    "before_state",
                                    "after_state",
                                    "state_changed",
                                )
                                if key in delta
                            },
                            "supporting_evidence_refs": [
                                aliases.evidence_to_alias.get(
                                    str(ref), "invalid_evidence_alias"
                                )
                                for ref in delta.get("supporting_evidence_refs", [])
                            ],
                        }
                    )
            comparisons.append(
                {
                    "segment_id": aliases.segment_to_alias.get(
                        str(raw.get("segment_id")), "invalid_segment_alias"
                    ),
                    "segment_index": raw.get("segment_index"),
                    "camera": raw.get("camera"),
                    "before_evidence_id": aliases.evidence_to_alias[before_id],
                    "after_evidence_id": aliases.evidence_to_alias[after_id],
                    "visual_delta": projected_delta,
                    "gripper_deltas": gripper_deltas,
                    "confidence": raw.get("confidence"),
                    "limitations": raw.get("limitations", []),
                }
            )

    transition_summaries: list[dict[str, Any]] = []
    item_by_evidence_id = {
        value.get("evidence_id"): value
        for value in _visual_evidence_items(visual)
        if isinstance(value.get("evidence_id"), str)
    }
    rgb_capability = visual.get("data_capabilities", {})
    rgb_capability = (
        rgb_capability.get("rgb", {})
        if isinstance(rgb_capability, Mapping)
        else {}
    )
    camera_order = (
        rgb_capability.get("cameras", [])
        if isinstance(rgb_capability, Mapping)
        else []
    )
    primary_camera = (
        camera_order[0]
        if isinstance(camera_order, list)
        and camera_order
        and isinstance(camera_order[0], str)
        else None
    )
    raw_transitions = visual.get("gripper_transitions", [])
    if isinstance(raw_transitions, Sequence) and not isinstance(
        raw_transitions, (str, bytes)
    ):
        for raw in raw_transitions:
            if not isinstance(raw, Mapping):
                continue
            refs = raw.get("supporting_evidence_refs")
            if not isinstance(refs, Mapping):
                continue

            def projected_ref(role: str) -> str | None:
                values = refs.get(role, [])
                if not isinstance(values, Sequence) or isinstance(
                    values, (str, bytes)
                ):
                    return None
                for value in values:
                    if value not in selected or value not in item_by_evidence_id:
                        continue
                    item = item_by_evidence_id[value]
                    if primary_camera is None or item.get("camera") == primary_camera:
                        return aliases.evidence_to_alias.get(str(value))
                return None

            before_alias = projected_ref("before")
            after_alias = projected_ref("after")
            if before_alias is None or after_alias is None:
                continue
            before_state_ref = raw.get("before_state_ref", {})
            after_state_ref = raw.get("after_state_ref", {})
            transition_summaries.append(
                {
                    "transition_alias": aliases.transition_to_alias.get(
                        str(raw.get("transition_id")), "invalid_transition_alias"
                    ),
                    "channel_id": raw.get("channel_id"),
                    "source_type": raw.get("source_type"),
                    "from_state": raw.get("from_state"),
                    "to_state": raw.get("to_state"),
                    "before_segment_id": aliases.segment_to_alias.get(
                        str(raw.get("before_segment_id")), "invalid_segment_alias"
                    ),
                    "after_segment_id": aliases.segment_to_alias.get(
                        str(raw.get("after_segment_id")), "invalid_segment_alias"
                    ),
                    "before_evidence_ref": before_alias,
                    "after_evidence_ref": after_alias,
                    "before_frame_index": (
                        before_state_ref.get("frame_index")
                        if isinstance(before_state_ref, Mapping)
                        else None
                    ),
                    "after_frame_index": (
                        after_state_ref.get("frame_index")
                        if isinstance(after_state_ref, Mapping)
                        else None
                    ),
                    "confidence": raw.get("confidence"),
                    "limitations": [
                        "robot_state_transition_does_not_prove_visual_effect",
                        "transition_does_not_establish_task_causality",
                    ],
                }
            )

    annotation_refs = {
        value.get("segment_id"): aliases.annotation_to_alias.get(
            str(value.get("evidence_ref")), "invalid_annotation_alias"
        )
        for value in request.annotation_evidence_refs
        if isinstance(value, Mapping)
    }
    segments: list[dict[str, Any]] = []
    task_families: list[str] = []
    for raw in request.segments:
        context = raw.get("context", {})
        derivation = raw.get("derivation", {})
        frame_range = (
            derivation.get("frame_range", {})
            if isinstance(derivation, Mapping)
            else {}
        )
        segment_id = raw.get("segment_id")
        task_family = (
            context.get("task_family") if isinstance(context, Mapping) else None
        )
        if isinstance(task_family, str) and task_family:
            task_families.append(task_family)
        segments.append(
            {
                "segment_id": aliases.segment_to_alias.get(
                    str(segment_id), "invalid_segment_alias"
                ),
                "segment_index": raw.get("segment_index"),
                "annotation": (
                    context.get("subtask_instruction")
                    if isinstance(context, Mapping)
                    else None
                ),
                "annotation_evidence_ref": annotation_refs.get(segment_id),
                "task_family": task_family,
                "frame_range": {
                    key: frame_range.get(key)
                    for key in ("start_inclusive", "end_inclusive")
                    if isinstance(frame_range, Mapping) and key in frame_range
                },
            }
        )

    capabilities: dict[str, Any] = {}
    raw_capabilities = visual.get("data_capabilities", {})
    if isinstance(raw_capabilities, Mapping):
        for name, raw in raw_capabilities.items():
            if not isinstance(raw, Mapping):
                continue
            capability: dict[str, Any] = {}
            for key in ("available", "reason"):
                if key in raw:
                    capability[key] = raw[key]
            if name == "rgb" and isinstance(raw.get("cameras"), list):
                capability["cameras"] = list(raw["cameras"])
            capabilities[str(name)] = capability

    outcome = request.trajectory_outcome
    if outcome.get("explicit_in_artifact") is True:
        outcome_projection = {
            key: outcome[key]
            for key in (
                "status",
                "evidence_source",
                "explicit_in_artifact",
                "evidence_refs",
            )
            if key in outcome
        }
        outcome_projection["available"] = True
    else:
        # A collection-contract label without an artifact-local evidence ref
        # is not evidence the model may cite.  Do not expose a bare success
        # status while simultaneously telling the model it is unverifiable.
        outcome_projection = {
            "available": False,
            "verification_status": "unverified",
            "explicit_in_artifact": False,
        }
    unique_task_families = list(dict.fromkeys(task_families))
    return {
        "instruction": request.instruction,
        "task_family": (
            unique_task_families[0] if len(unique_task_families) == 1 else None
        ),
        "trajectory_outcome": outcome_projection,
        "ordered_segments": segments,
        "annotation_evidence_refs": [
            {
                "segment_id": aliases.segment_to_alias.get(
                    str(value.get("segment_id")), "invalid_segment_alias"
                ),
                "evidence_ref": aliases.annotation_to_alias.get(
                    str(value.get("evidence_ref")), "invalid_annotation_alias"
                ),
            }
            for value in request.annotation_evidence_refs
            if isinstance(value, Mapping)
        ],
        "selected_visual_evidence": items,
        "selected_before_after_comparisons": comparisons,
        "selected_gripper_transitions": transition_summaries,
        "data_capabilities": capabilities,
        "uncertainties": list(request.uncertainties),
        "selected_image_evidence_ids": [
            aliases.evidence_to_alias[value] for value in selected_ids
        ],
        "image_binding": "Images are supplied in selected_image_evidence_ids order.",
    }


def _invalid_external_ref(kind: str, value: Any) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    return f"invalid_external_{kind}_ref_{digest}"


def _restore_external_output_refs(
    output: Mapping[str, Any],
    *,
    aliases: _RequestLocalAliases,
) -> dict[str, Any]:
    """Map only typed ref fields back to local IDs; unknown aliases stay invalid."""

    restored = json.loads(_canonical_json_bytes(dict(output)).decode("utf-8"))
    assert isinstance(restored, dict)
    evidence_aliases = {
        **aliases.alias_to_evidence,
        **aliases.alias_to_annotation,
    }

    def evidence_refs(value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [
            evidence_aliases.get(item, _invalid_external_ref("evidence", item))
            if isinstance(item, str)
            else item
            for item in value
        ]

    def segment_refs(value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [
            aliases.alias_to_segment.get(
                item,
                _invalid_external_ref("segment", item),
            )
            if isinstance(item, str)
            else item
            for item in value
        ]

    for container_name in ("episode_specific_facts", "attributions"):
        containers = restored.get(container_name)
        if not isinstance(containers, list):
            continue
        for container in containers:
            if not isinstance(container, dict):
                continue
            if "supporting_evidence_refs" in container:
                container["supporting_evidence_refs"] = evidence_refs(
                    container["supporting_evidence_refs"]
                )
            if "supporting_segment_refs" in container:
                container["supporting_segment_refs"] = segment_refs(
                    container["supporting_segment_refs"]
                )
    if "supporting_evidence_refs" in restored:
        restored["supporting_evidence_refs"] = evidence_refs(
            restored["supporting_evidence_refs"]
        )
    if "supporting_segment_refs" in restored:
        restored["supporting_segment_refs"] = segment_refs(
            restored["supporting_segment_refs"]
        )
    return restored


def contains_secret_or_image_payload(value: str) -> bool:
    """Conservative log-line guard used by the runtime and its tests."""

    return bool(
        _SECRET_VALUE_RE.search(value)
        or _DATA_URL_RE.search(value)
        or ";base64," in value.casefold()
    )


def validate_model_projection_text(value: str) -> None:
    """Reject sensitive/private text before any external model call."""

    if not isinstance(value, str) or not value.strip():
        raise ModelInputProjectionError("model projection text must be non-empty")
    if value.lstrip().startswith(("{", "[")):
        parsed = json.loads(value)

        def strings(item):
            if isinstance(item, dict):
                for key, child in item.items():
                    yield key
                    yield from strings(child)
            elif isinstance(item, list):
                for child in item:
                    yield from strings(child)
            elif isinstance(item, str):
                yield item

        # 检查 JSON 中的文本内容，保留正常换行和自然语言表达。
        value = "\n".join(strings(parsed))
    if contains_secret_or_image_payload(value):
        raise ModelInputProjectionError(
            "model projection contains a secret or encoded image marker"
        )
    if _RAW_SHA256_RE.search(value):
        raise ModelInputProjectionError("model projection contains a raw SHA-256 value")
    if _LOCAL_PATH_RE.search(value):
        raise ModelInputProjectionError("model projection contains a local path")
    if _CREDENTIAL_ASSIGNMENT_RE.search(value):
        raise ModelInputProjectionError(
            "model projection contains a credential-like assignment"
        )


def generic_reflection_system_prompt() -> str:
    """Return the task-neutral production prompt for an evidence-only model.

    It intentionally contains no benchmark/task name, expected policy, fixed
    configuration, known frame number, or task-specific few-shot example.
    """

    return (
        "You are an offline trajectory evidence analyst. Analyze the complete "
        "ordered trajectory using only the supplied instruction, segments, "
        "robot-state records, and referenced images. Keep directly visible "
        "facts, robot-state facts, annotations, and cross-segment inferences "
        "distinct. Every claim must cite existing evidence identifiers and "
        "state uncertainty. Annotation claims must cite only the supplied "
        "annotation_evidence_refs, never a visual frame. Keep all concrete "
        "episode facts private, and provide evidence attribution for every "
        "transferable claim. Attribution target_path values are relative to "
        "the transferable guidance object and must follow the supplied schema; "
        "never add that container name as a prefix. Attribute the condition, every ordered "
        "step, each emitted feedback-policy component, the avoid list, and every "
        "predicted effect. When a supplied image pair visibly supports a change, "
        "include an observed_visual fact with temporal_scope=before_after and "
        "cite both evidence items; if no visual change can be supported, abstain. "
        "Derive only transferable procedure guidance; keep "
        "episode-specific values in the private episode-facts field. Never "
        "turn one episode's concrete successful value into a general answer. "
        "Do not invent precision, contact, pose, causality, or success that the "
        "evidence does not support. Return only JSON matching the supplied "
        "schema. If evidence is insufficient, return an abstention rather than "
        "copying annotations or guessing."
    )


def generic_leakage_critic_prompt() -> str:
    """Return the independent, task-neutral episode-leakage critic prompt."""

    return (
        "You are an independent auditor of a proposed transferable procedure. "
        "Compare the private episode facts with the proposed transferable "
        "fields. Reject if a concrete value, ordering outcome, identifier, "
        "coordinate, pose, or single-episode successful assignment is stated "
        "or implied to be a generally correct answer. Reject guidance that "
        "bypasses feedback by replaying one episode's answer. Do not revise the "
        "candidate and do not infer task answers. Return only the requested "
        "structured verdict. When equivalence is uncertain, abstain."
    )


class OpenAICompatibleMultimodalTransport:
    """OpenAI Responses-compatible multimodal structured-output client."""

    backend_name = "openai_compatible_multimodal_responses"

    def __init__(
        self,
        *,
        service_url: str = "http://127.0.0.1:9104",
        model: str = "gpt-5.5",
        reasoning_effort: str = "xhigh",
        timeout_sec: float = 600.0,
        api_key: str | None = None,
        sender: JsonHttpSender | None = None,
        service_capability_probe: ServiceCapabilityProbe | None = None,
        max_images: int = 16,
        max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
        max_total_image_bytes: int = _DEFAULT_MAX_TOTAL_IMAGE_BYTES,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
        challenge_factory: Callable[[], tuple[bytes, tuple[str, str, str, str]]]
        | None = None,
    ) -> None:
        safe_url, endpoint = _safe_service_url(service_url)
        if not model.strip() or not reasoning_effort.strip():
            raise ValueError("model and reasoning_effort must be non-empty")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        for label, value in {
            "max_images": max_images,
            "max_image_bytes": max_image_bytes,
            "max_total_image_bytes": max_total_image_bytes,
            "max_response_bytes": max_response_bytes,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        self._service_url = safe_url
        self._endpoint = endpoint
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._timeout_sec = float(timeout_sec)
        self._api_key = api_key
        self._sender = sender or _default_json_sender
        self._service_capability_probe = (
            service_capability_probe
            if service_capability_probe is not None
            else _default_service_capability_probe if sender is None else None
        )
        self._health_endpoint = f"{safe_url.rstrip('/')}/health"
        self._max_images = max_images
        self._max_image_bytes = max_image_bytes
        self._max_total_image_bytes = max_total_image_bytes
        self._max_response_bytes = max_response_bytes
        self._challenge_factory = challenge_factory or _challenge_png
        self._verified_authorization_sha256: str | None = None
        self._capability_report: CapabilityReport | None = None
        self._last_preflight_audit: dict[str, Any] | None = None
        self._last_multimodal_completion_audit: dict[str, Any] | None = None
        self._last_text_completion_audit: dict[str, Any] | None = None

    def configuration_identity(self) -> dict[str, Any]:
        """Return a secret-free, serializable transport identity."""

        return {
            "backend": self.backend_name,
            "service_url": self._service_url,
            "endpoint_path": "/v1/responses",
            "model": self._model,
            "reasoning_effort": self._reasoning_effort,
            "timeout_sec": self._timeout_sec,
            "max_images": self._max_images,
            "max_image_bytes": self._max_image_bytes,
            "max_total_image_bytes": self._max_total_image_bytes,
            "max_response_bytes": self._max_response_bytes,
            "service_capability_check_required": (
                self._service_capability_probe is not None
            ),
            "credential_configured": bool(self._api_key),
            "response_storage": False,
        }

    @property
    def configuration_sha256(self) -> str:
        return _canonical_json_sha256(self.configuration_identity())

    @property
    def capability_report(self) -> CapabilityReport | None:
        return self._capability_report

    @property
    def last_preflight_audit(self) -> Mapping[str, Any] | None:
        return (
            None
            if self._last_preflight_audit is None
            else dict(self._last_preflight_audit)
        )

    @property
    def last_multimodal_completion_audit(self) -> Mapping[str, Any] | None:
        return (
            None
            if self._last_multimodal_completion_audit is None
            else dict(self._last_multimodal_completion_audit)
        )

    @property
    def last_text_completion_audit(self) -> Mapping[str, Any] | None:
        return (
            None
            if self._last_text_completion_audit is None
            else dict(self._last_text_completion_audit)
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _send(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
        try:
            value = self._sender(
                url=self._endpoint,
                payload=payload,
                headers=self._headers(),
                timeout_sec=self._timeout_sec,
                max_response_bytes=self._max_response_bytes,
            )
        except MultimodalTransportError:
            raise
        except Exception as exc:
            # The injected sender is untrusted too; do not include its message,
            # which may contain a serialized request or credential.
            raise MultimodalTransportError(
                f"multimodal sender failed ({type(exc).__name__})"
            ) from exc
        response, raw = _response_object(value)
        if len(raw) > self._max_response_bytes:
            raise MultimodalResponseError("multimodal response exceeds byte budget")
        return response, raw

    def capability_preflight(
        self,
        *,
        authorization: ImageEgressAuthorization,
    ) -> CapabilityReport:
        """Actively prove image ingestion and strict structured output.

        The explicit authorization gate is evaluated before the synthetic image
        is created or any network sender is invoked.
        """

        self._last_preflight_audit = None
        authorization.require()
        service_max_images: int | None = None
        schema_profile: str | None = None
        service_health_sha256: str | None = None
        service_capability_audit: dict[str, Any]
        if self._service_capability_probe is None:
            service_capability_audit = {
                "service_capability_check": "skipped_injected_sender",
            }
        else:
            try:
                health_value = self._service_capability_probe(
                    url=self._health_endpoint,
                    timeout_sec=min(self._timeout_sec, 30.0),
                    max_response_bytes=min(self._max_response_bytes, 64 * 1024),
                )
                health, health_raw = _response_object(health_value)
                capabilities = health.get("responses_capabilities")
                service_max_images = (
                    capabilities.get("max_images_per_request")
                    if isinstance(capabilities, Mapping)
                    else None
                )
                schema_profile = (
                    capabilities.get("structured_output_schema_profile")
                    if isinstance(capabilities, Mapping)
                    else None
                )
                if (
                    health.get("status") != "ok"
                    or health.get("model") != self._model
                    or health.get("api_mode") != "responses_compat"
                    or health.get("reasoning_effort") != self._reasoning_effort
                    or isinstance(service_max_images, bool)
                    or not isinstance(service_max_images, int)
                    or service_max_images < self._max_images
                    or schema_profile != "openai_strict_json_schema_subset"
                ):
                    raise CapabilityPreflightError(
                        "service health does not satisfy the requested model, "
                        "reasoning, image-count, and schema-profile contract"
                    )
            except Exception as exc:
                self._last_preflight_audit = {
                    "backend": self.backend_name,
                    "service_url": self._service_url,
                    "model": self._model,
                    "status": "local_failed",
                    "delivery_status": "not_sent",
                    "transport_failure_stage": "service_capability_check",
                    "configuration_sha256": self.configuration_sha256,
                    "authorization_sha256": authorization.audit_sha256(),
                    "image_count": 0,
                    "images_sent": 0,
                    "synthetic_only": True,
                    "image_payload_logged": False,
                    "secret_logged": False,
                    "store": False,
                }
                if isinstance(exc, CapabilityPreflightError):
                    raise
                raise CapabilityPreflightError(
                    "service capability check failed before model I/O"
                ) from exc
            service_capability_audit = {
                "service_capability_check": "passed",
                "service_health_sha256": hashlib.sha256(health_raw).hexdigest(),
                "service_max_images_per_request": service_max_images,
                "service_schema_profile": schema_profile,
            }
            service_health_sha256 = service_capability_audit[
                "service_health_sha256"
            ]
        image_bytes, expected_order = self._challenge_factory()
        probe = MultimodalImage(
            evidence_id="synthetic_capability_probe",
            mime_type="image/png",
            content=image_bytes,
            detail="high",
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["image_received", "quadrants", "structured_output"],
            "properties": {
                "image_received": {"type": "boolean", "const": True},
                "quadrants": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "string", "enum": sorted(_COLOUR_RGB)},
                },
                "structured_output": {"type": "boolean", "const": True},
            },
        }
        provider_schema = project_openai_strict_output_schema(schema)
        instructions = (
            "Inspect the supplied synthetic image. Return the four quadrant "
            "colours in reading order: top-left, top-right, bottom-left, "
            "bottom-right. Return only JSON matching the supplied schema."
        )
        payload = {
            "model": self._model,
            "instructions": instructions,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Report the visible quadrant colours.",
                        },
                        {
                            "type": "input_image",
                            "image_url": _data_url(probe),
                            "detail": probe.detail,
                        },
                    ],
                }
            ],
            "reasoning": {"effort": self._reasoning_effort},
            "store": False,
            "stream": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "phase_a2_multimodal_capability",
                    "strict": True,
                    "schema": provider_schema,
                }
            },
        }
        self._last_preflight_audit = {
            "backend": self.backend_name,
            "service_url": self._service_url,
            "model": self._model,
            "status": "attempted",
            "configuration_sha256": self.configuration_sha256,
            "authorization_sha256": authorization.audit_sha256(),
            "prompt_template_sha256": hashlib.sha256(
                instructions.encode("utf-8")
            ).hexdigest(),
            "request_sha256": _canonical_json_sha256(payload),
            "challenge_image_sha256": probe.sha256,
            "image_count": 1,
            "synthetic_only": True,
            "image_payload_logged": False,
            "secret_logged": False,
            "store": False,
            "source_output_schema_sha256": _canonical_json_sha256(schema),
            "provider_output_schema_sha256": _canonical_json_sha256(
                provider_schema
            ),
            "provider_schema_projection_version": (
                _PROVIDER_SCHEMA_PROJECTION_VERSION
            ),
            "provider_schema_removed_keyword_counts": (
                _provider_schema_removed_keyword_counts(schema)
            ),
            **service_capability_audit,
        }
        try:
            response, raw_response = self._send(payload)
        except MultimodalTransportError as exc:
            self._last_preflight_audit = {
                **self._last_preflight_audit,
                "status": "failed",
                "delivery_status": "sent_or_attempted",
                "transport_failure_stage": "request",
                **(
                    exc.audit_metadata()
                    if isinstance(exc, MultimodalHTTPError)
                    else {}
                ),
            }
            raise
        parsed = _strict_json_object(
            _response_output_text(response),
            label="capability preflight output",
        )
        exact = (
            parsed.get("image_received") is True
            and parsed.get("structured_output") is True
            and parsed.get("quadrants") == list(expected_order)
            and set(parsed) == {"image_received", "quadrants", "structured_output"}
        )
        if not exact:
            raise CapabilityPreflightError(
                "active image/structured-output challenge was not satisfied; "
                "text-only fallback or image loss is possible"
            )
        authorization_sha256 = authorization.audit_sha256()
        report = CapabilityReport(
            backend=self.backend_name,
            service_url=self._service_url,
            model=self._model,
            supports_images=True,
            image_roundtrip_verified=True,
            structured_output_verified=True,
            text_only_fallback_detected=False,
            request_sha256=_canonical_json_sha256(payload),
            response_sha256=hashlib.sha256(raw_response).hexdigest(),
            challenge_image_sha256=probe.sha256,
            authorization_sha256=authorization_sha256,
            configuration_sha256=self.configuration_sha256,
            service_max_images_per_request=service_max_images,
            service_schema_profile=schema_profile,
            service_health_sha256=service_health_sha256,
        )
        self._verified_authorization_sha256 = authorization_sha256
        self._capability_report = report
        self._last_preflight_audit = {
            **self._last_preflight_audit,
            "status": "passed",
            "delivery_status": "sent",
            "images_sent": 1,
            "response_sha256": report.response_sha256,
        }
        return report

    def complete(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[MultimodalImage],
        output_schema: Mapping[str, Any],
        schema_name: str,
        authorization: ImageEgressAuthorization,
    ) -> MultimodalCompletion:
        """Send bounded expert evidence after a successful active preflight."""

        self._last_multimodal_completion_audit = None
        authorization.require()
        if self._capability_report is None or not (
            self._capability_report.image_roundtrip_verified
            and self._capability_report.structured_output_verified
        ):
            raise CapabilityPreflightError(
                "active multimodal capability preflight has not passed"
            )
        if authorization.audit_sha256() != self._verified_authorization_sha256:
            raise ImageEgressDenied(
                "image egress authorization does not match the preflight scope"
            )
        if not instructions.strip() or not input_text.strip():
            raise ValueError("instructions and input_text must be non-empty")
        if self._api_key and (
            self._api_key in instructions or self._api_key in input_text
        ):
            raise ModelInputProjectionError(
                "model projection contains the configured API credential"
            )
        validate_model_projection_text(instructions)
        validate_model_projection_text(input_text)
        if not _SCHEMA_NAME_RE.fullmatch(schema_name):
            raise ValueError("schema_name is not a safe JSON-schema identifier")
        if not isinstance(output_schema, Mapping) or not output_schema:
            raise ValueError("output_schema must be a non-empty mapping")
        bounded_images = tuple(images)
        if not bounded_images:
            raise ValueError("multimodal reflection requires at least one image")
        if len(bounded_images) > self._max_images:
            raise ValueError("multimodal request exceeds the image-count budget")
        total_bytes = 0
        evidence_ids: list[str] = []
        image_hashes: list[str] = []
        for image in bounded_images:
            if not isinstance(image, MultimodalImage):
                raise TypeError("images must contain MultimodalImage values")
            if len(image.content) > self._max_image_bytes:
                raise ValueError("one image exceeds the per-image byte budget")
            total_bytes += len(image.content)
            evidence_ids.append(image.evidence_id)
            image_hashes.append(image.sha256)
        if total_bytes > self._max_total_image_bytes:
            raise ValueError("multimodal request exceeds the total image byte budget")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("multimodal request contains duplicate evidence IDs")

        source_schema_sha256 = _canonical_json_sha256(output_schema)
        try:
            provider_schema = project_openai_strict_output_schema(output_schema)
        except ProviderSchemaProjectionError:
            self._last_multimodal_completion_audit = {
                "backend": self.backend_name,
                "status": "local_failed",
                "delivery_status": "not_sent",
                "transport_failure_stage": "provider_schema_projection",
                "service_url": self._service_url,
                "model": self._model,
                "configuration_sha256": self.configuration_sha256,
                "authorization_sha256": authorization.audit_sha256(),
                "source_output_schema_sha256": source_schema_sha256,
                "image_count": len(bounded_images),
                "image_total_bytes": total_bytes,
                "images_sent": 0,
                "image_bytes_sent": 0,
                "image_payload_logged": False,
                "secret_logged": False,
                "store": False,
            }
            raise
        provider_schema_sha256 = _canonical_json_sha256(provider_schema)

        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": input_text}
        ]
        content.extend(
            {
                "type": "input_image",
                "image_url": _data_url(image),
                "detail": image.detail,
            }
            for image in bounded_images
        )
        payload = {
            "model": self._model,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": self._reasoning_effort},
            "store": False,
            "stream": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": provider_schema,
                }
            },
        }
        base_audit = {
            "backend": self.backend_name,
            "status": "attempted",
            "service_url": self._service_url,
            "model": self._model,
            "configuration_sha256": self.configuration_sha256,
            "capability_report_sha256": _canonical_json_sha256(
                self._capability_report.to_dict()
            ),
            "authorization_sha256": authorization.audit_sha256(),
            "prompt_template_sha256": hashlib.sha256(
                instructions.encode("utf-8")
            ).hexdigest(),
            "input_text_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
            "request_sha256": _canonical_json_sha256(payload),
            "image_count": len(bounded_images),
            "image_total_bytes": total_bytes,
            "image_evidence_ids": evidence_ids,
            "image_sha256": image_hashes,
            "image_payload_logged": False,
            "secret_logged": False,
            "store": False,
            "source_output_schema_sha256": source_schema_sha256,
            "provider_output_schema_sha256": provider_schema_sha256,
            "provider_schema_projection_version": (
                _PROVIDER_SCHEMA_PROJECTION_VERSION
            ),
            "provider_schema_removed_keyword_counts": (
                _provider_schema_removed_keyword_counts(output_schema)
            ),
        }
        self._last_multimodal_completion_audit = base_audit
        try:
            response, raw_response = self._send(payload)
        except MultimodalTransportError as exc:
            self._last_multimodal_completion_audit = {
                **base_audit,
                "status": "failed",
                "delivery_status": "sent_or_attempted",
                "transport_failure_stage": "request",
                **(
                    exc.audit_metadata()
                    if isinstance(exc, MultimodalHTTPError)
                    else {}
                ),
            }
            raise
        self._last_multimodal_completion_audit = {
            **base_audit,
            "status": "response_received",
            "response_sha256": hashlib.sha256(raw_response).hexdigest(),
        }
        output_text = _response_output_text(response)
        parsed = _strict_json_object(output_text, label="reflection output")
        audit = {
            **base_audit,
            "status": "completed",
            "response_sha256": hashlib.sha256(raw_response).hexdigest(),
            "model_output_sha256": hashlib.sha256(
                output_text.encode("utf-8")
            ).hexdigest(),
        }
        self._last_multimodal_completion_audit = audit
        return MultimodalCompletion(output=parsed, audit=audit)

    def complete_text(
        self,
        *,
        instructions: str,
        input_text: str,
        output_schema: Mapping[str, Any],
        schema_name: str,
        authorization: ImageEgressAuthorization,
        purpose: str,
    ) -> MultimodalCompletion:
        """Run an independent structured text call after multimodal preflight.

        Phase A2 uses this for the leakage critic.  Requiring the same explicit
        authorization and verified transport scope prevents a second, less
        visible model call from escaping the audited run boundary.  No image
        data URL is constructed for this request.
        """

        self._last_text_completion_audit = None
        authorization.require()
        if self._capability_report is None or not (
            self._capability_report.image_roundtrip_verified
            and self._capability_report.structured_output_verified
        ):
            raise CapabilityPreflightError(
                "active multimodal capability preflight has not passed"
            )
        if authorization.audit_sha256() != self._verified_authorization_sha256:
            raise ImageEgressDenied(
                "model-call authorization does not match the preflight scope"
            )
        if not instructions.strip() or not input_text.strip() or not purpose.strip():
            raise ValueError("instructions, input_text, and purpose must be non-empty")
        if self._api_key and (
            self._api_key in instructions or self._api_key in input_text
        ):
            raise ModelInputProjectionError(
                "model projection contains the configured API credential"
            )
        validate_model_projection_text(instructions)
        validate_model_projection_text(input_text)
        if not _SCHEMA_NAME_RE.fullmatch(schema_name):
            raise ValueError("schema_name is not a safe JSON-schema identifier")
        if not isinstance(output_schema, Mapping) or not output_schema:
            raise ValueError("output_schema must be a non-empty mapping")

        source_schema_sha256 = _canonical_json_sha256(output_schema)
        try:
            provider_schema = project_openai_strict_output_schema(output_schema)
        except ProviderSchemaProjectionError:
            self._last_text_completion_audit = {
                "backend": self.backend_name,
                "purpose": purpose,
                "status": "local_failed",
                "delivery_status": "not_sent",
                "transport_failure_stage": "provider_schema_projection",
                "service_url": self._service_url,
                "model": self._model,
                "configuration_sha256": self.configuration_sha256,
                "authorization_sha256": authorization.audit_sha256(),
                "source_output_schema_sha256": source_schema_sha256,
                "image_count": 0,
                "images_sent": 0,
                "image_bytes_sent": 0,
                "image_payload_logged": False,
                "secret_logged": False,
                "store": False,
            }
            raise
        provider_schema_sha256 = _canonical_json_sha256(provider_schema)

        payload = {
            "model": self._model,
            "instructions": instructions,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_text}],
                }
            ],
            "reasoning": {"effort": self._reasoning_effort},
            "store": False,
            "stream": False,
            "metadata": {"phase": "self_evolution_phase_a2", "purpose": purpose},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": provider_schema,
                }
            },
        }
        base_audit = {
            "backend": self.backend_name,
            "purpose": purpose,
            "status": "attempted",
            "service_url": self._service_url,
            "model": self._model,
            "configuration_sha256": self.configuration_sha256,
            "capability_report_sha256": _canonical_json_sha256(
                self._capability_report.to_dict()
            ),
            "authorization_sha256": authorization.audit_sha256(),
            "prompt_template_sha256": hashlib.sha256(
                instructions.encode("utf-8")
            ).hexdigest(),
            "input_text_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
            "request_sha256": _canonical_json_sha256(payload),
            "image_count": 0,
            "image_payload_logged": False,
            "secret_logged": False,
            "store": False,
            "source_output_schema_sha256": source_schema_sha256,
            "provider_output_schema_sha256": provider_schema_sha256,
            "provider_schema_projection_version": (
                _PROVIDER_SCHEMA_PROJECTION_VERSION
            ),
            "provider_schema_removed_keyword_counts": (
                _provider_schema_removed_keyword_counts(output_schema)
            ),
        }
        self._last_text_completion_audit = base_audit
        try:
            response, raw_response = self._send(payload)
        except MultimodalTransportError as exc:
            self._last_text_completion_audit = {
                **base_audit,
                "status": "failed",
                "delivery_status": "sent_or_attempted",
                "transport_failure_stage": "request",
                **(
                    exc.audit_metadata()
                    if isinstance(exc, MultimodalHTTPError)
                    else {}
                ),
            }
            raise
        self._last_text_completion_audit = {
            **base_audit,
            "status": "response_received",
            "response_sha256": hashlib.sha256(raw_response).hexdigest(),
        }
        output_text = _response_output_text(response)
        parsed = _strict_json_object(output_text, label=f"{purpose} output")
        audit = {
            **base_audit,
            "status": "completed",
            "response_sha256": hashlib.sha256(raw_response).hexdigest(),
            "model_output_sha256": hashlib.sha256(
                output_text.encode("utf-8")
            ).hexdigest(),
        }
        self._last_text_completion_audit = audit
        return MultimodalCompletion(output=parsed, audit=audit)


class OpenAICompatibleLeakageCritic:
    """Independent structured critic for episode-answer leakage.

    This critic is deliberately separate from deterministic schema/field
    checks.  It can add evidence for semantic equivalence attacks, but it does
    not claim to prove the absence of every possible natural-language leak.
    """

    backend_name = "openai_compatible_episode_leakage_critic"

    def __init__(
        self,
        transport: OpenAICompatibleMultimodalTransport,
        *,
        authorization: ImageEgressAuthorization,
    ) -> None:
        self._transport = transport
        self._authorization = authorization
        self._last_call_audit: dict[str, Any] | None = None

    @property
    def last_call_audit(self) -> Mapping[str, Any] | None:
        return None if self._last_call_audit is None else dict(self._last_call_audit)

    @staticmethod
    def output_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "verdict",
                "episode_specific_leakage",
                "reasons",
                "confidence",
            ],
            "properties": {
                "verdict": {"type": "string", "enum": ["safe", "reject", "abstain"]},
                "episode_specific_leakage": {"type": "boolean"},
                "reasons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["code", "offending_field", "explanation"],
                        "properties": {
                            "code": {"type": "string", "maxLength": 120},
                            "offending_field": {"type": "string", "maxLength": 240},
                            "explanation": {"type": "string", "maxLength": 800},
                        },
                    },
                },
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
        }

    def audit(
        self,
        transferable_guidance: Mapping[str, Any],
        episode_specific_facts: Sequence[Mapping[str, Any]],
    ) -> Any:
        from .quality import LeakageCriticResult, QualityIssue

        self._last_call_audit = None

        private_fact_projection = []
        for fact in episode_specific_facts:
            private_fact_projection.append(
                {
                    key: fact[key]
                    for key in (
                        "claim",
                        "source_type",
                        "confidence",
                        "uncertainty",
                        "temporal_scope",
                        "transferable_allowed",
                        "structured_value",
                    )
                    if key in fact
                }
            )
        input_payload = {
            "transferable_candidate": dict(transferable_guidance),
            "private_episode_facts": private_fact_projection,
        }
        try:
            completion = self._transport.complete_text(
                instructions=generic_leakage_critic_prompt(),
                input_text=_canonical_json_bytes(input_payload).decode("utf-8"),
                output_schema=self.output_schema(),
                schema_name="phase_a2_episode_leakage_critic",
                authorization=self._authorization,
                purpose="episode_leakage_critic",
            )
        except Exception:
            attempt = self._transport.last_text_completion_audit
            self._last_call_audit = None if attempt is None else dict(attempt)
            raise
        self._last_call_audit = dict(completion.audit)
        output = completion.output
        verdict = output.get("verdict")
        leakage = output.get("episode_specific_leakage")
        reasons = output.get("reasons")
        confidence = output.get("confidence")
        if verdict not in {"safe", "reject", "abstain"}:
            raise MultimodalResponseError("leakage critic returned an invalid verdict")
        if not isinstance(leakage, bool):
            raise MultimodalResponseError("leakage critic returned an invalid leak flag")
        if confidence not in {"low", "medium", "high"}:
            raise MultimodalResponseError("leakage critic returned invalid confidence")
        if (
            isinstance(reasons, (str, bytes))
            or not isinstance(reasons, Sequence)
            or any(not isinstance(value, Mapping) for value in reasons)
        ):
            raise MultimodalResponseError("leakage critic returned invalid reasons")
        normalized_reasons: list[dict[str, Any]] = []
        for value in reasons:
            reason = dict(value)
            if set(reason) != {"code", "offending_field", "explanation"} or any(
                not isinstance(reason[key], str) or not reason[key].strip()
                for key in reason
            ):
                raise MultimodalResponseError(
                    "leakage critic returned a malformed reason"
                )
            normalized_reasons.append(reason)
        if verdict == "safe" and (leakage or normalized_reasons):
            raise MultimodalResponseError(
                "leakage critic returned an internally inconsistent safe verdict"
            )
        if verdict == "reject" and not leakage:
            raise MultimodalResponseError(
                "leakage critic rejected without identifying episode leakage"
            )
        issues: list[QualityIssue] = []
        if verdict == "reject":
            issues.extend(
                QualityIssue(
                    code="episode_specific_value_leakage",
                    offending_field=reason["offending_field"],
                    detail=(
                        f"independent critic {reason['code']}: "
                        f"{reason['explanation']}"
                    ),
                )
                for reason in normalized_reasons
            )
            if not issues:
                issues.append(
                    QualityIssue(
                        code="episode_specific_value_leakage",
                        offending_field="transferable_guidance",
                        detail="independent critic detected episode-specific leakage",
                    )
                )
        elif verdict == "abstain":
            issues.append(
                QualityIssue(
                    code="leakage_critic_abstained",
                    offending_field="transferable_guidance",
                    detail="independent critic could not reach a reliable verdict",
                )
            )
        return LeakageCriticResult(
            completed=verdict != "abstain",
            issues=tuple(issues),
            auditor=self.backend_name,
            prompt_template_hash=str(
                completion.audit["prompt_template_sha256"]
            ),
            output_sha256=str(completion.audit["model_output_sha256"]),
        )


class OpenAICompatibleMultimodalBackend:
    """Concrete ``MultimodalReflectionBackend`` over the Responses transport.

    The backend selects a trajectory-spanning subset under the caller-provided
    image budget, resolves only those IDs, constructs a task-neutral request, and
    wraps the parsed response as ``UntrustedReflectionOutput``.  Semantic
    admission, including the independently injected leakage critic, remains
    the responsibility of ``WholeTrajectoryReflector`` and its quality gate.
    """

    backend_name = "openai_compatible_whole_trajectory_multimodal"

    def __init__(
        self,
        transport: OpenAICompatibleMultimodalTransport,
        *,
        authorization: ImageEgressAuthorization,
        image_resolver: EvidenceImageResolver,
        output_schema: Mapping[str, Any] | None = None,
        max_images: int = 8,
    ) -> None:
        if (
            isinstance(max_images, bool)
            or not isinstance(max_images, int)
            or max_images <= 0
        ):
            raise ValueError("max_images must be a positive integer")
        if output_schema is None:
            from .whole_trajectory import reflection_output_json_schema

            output_schema = reflection_output_json_schema()
        if not isinstance(output_schema, Mapping) or not output_schema:
            raise ValueError("output_schema must be a non-empty mapping")
        self._transport = transport
        self._authorization = authorization
        self._image_resolver = image_resolver
        self._output_schema = dict(output_schema)
        self._max_images = max_images
        self._last_call_audit: dict[str, Any] | None = None
        self._last_selected_evidence_ids: tuple[str, ...] = ()

    @property
    def last_call_audit(self) -> Mapping[str, Any] | None:
        return None if self._last_call_audit is None else dict(self._last_call_audit)

    @property
    def last_selected_evidence_ids(self) -> tuple[str, ...]:
        return self._last_selected_evidence_ids

    def reflect(self, request: Any) -> Any:
        # Import lazily to preserve the transport module's standalone test seam
        # and avoid a module cycle with the provider-neutral protocol.
        from .whole_trajectory import (
            UntrustedReflectionOutput,
            WholeTrajectoryReflectionRequest,
        )

        self._last_call_audit = None
        self._last_selected_evidence_ids = ()
        if not isinstance(request, WholeTrajectoryReflectionRequest):
            raise TypeError("request must be WholeTrajectoryReflectionRequest")
        selected_ids = _select_spanning_evidence_ids(
            request.visual_evidence,
            limit=self._max_images,
        )
        if not selected_ids:
            raise ValueError("reflection request has no resolvable visual evidence IDs")
        resolved = tuple(self._image_resolver(request, selected_ids))
        if tuple(image.evidence_id for image in resolved) != selected_ids:
            raise ValueError(
                "image resolver must return exactly the selected evidence IDs in order"
            )

        # Keep every logical evidence ref for attribution while sending each
        # byte-identical camera frame only once.  This prevents adjacent
        # segment boundaries from silently consuming the user-approved image
        # budget and records the many-to-one presentation binding explicitly.
        payload_images: list[MultimodalImage] = []
        payload_source_ids: list[list[str]] = []
        payload_index_by_content: dict[tuple[str, str], int] = {}
        for image in resolved:
            content_key = (image.mime_type, image.sha256)
            payload_index = payload_index_by_content.get(content_key)
            if payload_index is None:
                payload_index = len(payload_images)
                payload_index_by_content[content_key] = payload_index
                payload_images.append(
                    MultimodalImage(
                        evidence_id=f"model_image_{payload_index:03d}",
                        mime_type=image.mime_type,
                        content=image.content,
                        detail=image.detail,
                    )
                )
                payload_source_ids.append([])
            payload_source_ids[payload_index].append(image.evidence_id)
        if len(payload_images) > self._max_images:
            raise ValueError("deduplicated model image payload exceeds max_images")

        aliases = _request_local_aliases(request, selected_ids)
        request_payload = _model_request_projection(
            request,
            selected_ids=selected_ids,
            aliases=aliases,
        )
        model_image_bindings = [
            {
                "model_image_id": image.evidence_id,
                "source_evidence_ids": [
                    aliases.evidence_to_alias[value]
                    for value in payload_source_ids[index]
                ],
                "presentation": "original_frame",
                "byte_identical_sources_shared": (
                    len(payload_source_ids[index]) > 1
                ),
            }
            for index, image in enumerate(payload_images)
        ]
        request_payload["model_image_bindings"] = model_image_bindings
        request_payload["image_binding"] = (
            "Payload images follow model_image_bindings order. Cite the bound "
            "source_evidence_ids; model_image_id is presentation-only evidence."
        )
        instructions = generic_reflection_system_prompt()
        prompt_hash = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
        try:
            completion = self._transport.complete(
                instructions=instructions,
                input_text=_canonical_json_bytes(request_payload).decode("utf-8"),
                images=tuple(payload_images),
                output_schema=self._output_schema,
                schema_name="phase_a2_whole_trajectory_reflection",
                authorization=self._authorization,
            )
        except Exception:
            attempt = self._transport.last_multimodal_completion_audit
            self._last_call_audit = {
                **({} if attempt is None else dict(attempt)),
                "request_local_aliases": aliases.audit_dict(),
                "logical_evidence_ids": list(selected_ids),
                "model_image_bindings": model_image_bindings,
            }
            raise
        self._last_call_audit = {
            **dict(completion.audit),
            "request_local_aliases": aliases.audit_dict(),
            "logical_evidence_ids": list(selected_ids),
            "logical_evidence_ref_count": len(selected_ids),
            "unique_source_content_count": len(payload_images),
            "model_image_bindings": model_image_bindings,
        }
        self._last_selected_evidence_ids = selected_ids

        report = self._transport.capability_report
        if report is None:  # complete() already proves this; defense in depth.
            raise CapabilityPreflightError("capability report disappeared")
        capabilities = {
            "image_input_supported": report.supports_images,
            "image_input_acknowledged": report.image_roundtrip_verified,
            "structured_output_supported": report.structured_output_verified,
            "text_only_fallback": report.text_only_fallback_detected,
            "scripted_backend": False,
        }
        envelope = UntrustedReflectionOutput.from_raw(
            completion.output,
            backend=self.backend_name,
            prompt_template_hash=prompt_hash,
            capabilities=capabilities,
        )
        return replace(
            envelope,
            parsed=_restore_external_output_refs(
                completion.output,
                aliases=aliases,
            ),
        )


__all__ = [
    "CapabilityPreflightError",
    "CapabilityReport",
    "EvidenceImageResolver",
    "ImageEgressAuthorization",
    "ImageEgressDenied",
    "JsonHttpSender",
    "MultimodalCompletion",
    "MultimodalHTTPError",
    "MultimodalImage",
    "ModelInputProjectionError",
    "MultimodalResponseError",
    "MultimodalTransportError",
    "OpenAICompatibleLeakageCritic",
    "OpenAICompatibleMultimodalBackend",
    "OpenAICompatibleMultimodalTransport",
    "ProviderSchemaProjectionError",
    "ServiceCapabilityProbe",
    "contains_secret_or_image_payload",
    "generic_leakage_critic_prompt",
    "generic_reflection_system_prompt",
    "project_openai_strict_output_schema",
    "validate_model_projection_text",
]
