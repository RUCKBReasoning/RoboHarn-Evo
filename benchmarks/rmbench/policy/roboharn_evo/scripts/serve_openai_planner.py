from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from openai import DefaultHttpxClient, OpenAI
except ImportError:  # pragma: no cover - exercised only in minimal test envs
    DefaultHttpxClient = None  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment]

from policy.roboharn_evo.scripts.codex_app_server_client import (
    CodexAppServerClient,
    CodexAppServerTimeoutError,
)
from policy.roboharn_evo.scripts.local_models_identity import models_identity_sha256
from policy.roboharn_evo.scripts.openai_responses_compat import (
    PreparedResponsesRequest,
    ResponsesCompatError,
    build_response_object,
    new_message_id,
    new_response_id,
    prepare_responses_request,
)
from policy.roboharn_evo.scripts.serve_qwen_planner import (
    QwenPlannerHandler,
    extract_json_object,
)

DEFAULT_CODEX_AUTH_PATH = str(Path.home() / ".codex" / "auth.json")
DEFAULT_CODEX_CONFIG_PATH = str(Path.home() / ".codex" / "config.toml")

_BUSINESS_ENDPOINTS = frozenset(
    {
        "/plan",
        "/ood",
        "/recover",
        "/perception_queries",
        "/normalize_perception_queries",
    }
)


class ModelOutputError(RuntimeError):
    """Raised when the sole configured upstream returns an invalid business response."""


class GatewayConfigurationError(RuntimeError):
    """Raised when a request reaches a gateway that has no configured upstream client."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve an OpenAI planner VLM over the same HTTP API as the Qwen planner service."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9102)
    parser.add_argument(
        "--backend",
        choices=("openai", "codex-account"),
        default="openai",
        help=(
            "Upstream transport. 'openai' uses an API key with the OpenAI SDK; "
            "'codex-account' uses the locally authenticated Codex CLI without exposing OAuth tokens."
        ),
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=DEFAULT_CODEX_CONFIG_PATH,
        help="Optional Codex config.toml to read model/provider defaults from.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable that contains the OpenAI API key. Takes precedence over --api-key-file.",
    )
    parser.add_argument(
        "--api-key-file",
        type=str,
        default=DEFAULT_CODEX_AUTH_PATH,
        help="Optional JSON file containing an OpenAI API key. Defaults to Codex auth.json.",
    )
    parser.add_argument(
        "--api-key-json-key",
        type=str,
        default="OPENAI_API_KEY",
        help="JSON key to read from --api-key-file.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL. Overrides Codex config provider base_url.",
    )
    parser.add_argument(
        "--provider-name",
        type=str,
        default=None,
        help="Optional provider identity override reported by /health.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help=(
            "Fail-closed local serving mode: require a loopback OpenAI-compatible "
            "upstream, ignore all API-key/proxy environment variables, and verify "
            "the exact upstream model before accepting requests."
        ),
    )
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=2,
        help="Maximum simultaneous upstream model requests; excess local requests wait in the HTTP server.",
    )
    parser.add_argument(
        "--request-queue-timeout-sec",
        type=float,
        default=5.0,
        help="Seconds an excess request may wait for a model slot before receiving HTTP 503.",
    )
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Maximum accepted JSON request-body size.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="OpenAI SDK retries for transient upstream failures.",
    )
    parser.add_argument(
        "--api-mode",
        choices=("chat", "responses"),
        default=None,
        help="API surface to use. Defaults from Codex provider wire_api when available.",
    )
    parser.add_argument(
        "--response-format-json",
        action="store_true",
        help="Request JSON object output in chat mode when the selected model supports it.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Optional reasoning effort for supported models/providers. Overrides Codex model_reasoning_effort.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=0,
        help="Optional max output token cap. Uses max_completion_tokens in chat mode.",
    )
    parser.add_argument(
        "--planner-max-output-tokens",
        type=int,
        default=0,
        help=(
            "Optional /plan-only output cap. Zero inherits --max-output-tokens. "
            "Currently restricted to --local-only serving."
        ),
    )
    parser.add_argument(
        "--planner-prompt-mode",
        choices=("legacy_duplicate", "rendered_system_once"),
        default="legacy_duplicate",
        help=(
            "Planner context mapping. rendered_system_once keeps task/memory/state "
            "only in the adapter-rendered system prompt and is restricted to "
            "--local-only serving."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Omitted from upstream requests when unset.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus-sampling probability. Omitted when unset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional provider-specific top-k value sent through chat extra_body.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Optional provider-specific min-p value sent through chat extra_body.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("provider_default", "enabled", "disabled"),
        default="provider_default",
        help=(
            "Thinking-mode identity. In chat mode, enabled/disabled is enforced via "
            "chat_template_kwargs.enable_thinking; provider_default sends no override."
        ),
    )
    parser.add_argument("--model-source", type=str, default="")
    parser.add_argument("--model-revision", type=str, default="")
    parser.add_argument("--quantization", type=str, default="")
    parser.add_argument("--serving-framework", type=str, default="")
    parser.add_argument(
        "--context-limit",
        type=int,
        default=0,
        help="Configured upstream context limit for identity/provenance reporting.",
    )
    parser.add_argument("--model-artifact-path", type=str, default="")
    parser.add_argument("--model-artifact-manifest-path", type=str, default="")
    parser.add_argument("--model-artifact-manifest-sha256", type=str, default="")
    parser.add_argument(
        "--upstream-models-response-sha256",
        type=str,
        default="",
        help="SHA256 of the upstream /v1/models response verified before gateway startup.",
    )
    parser.add_argument(
        "--upstream-models-identity-sha256",
        type=str,
        default="",
        help=(
            "SHA256 of stable /v1/models identity fields. Unlike the raw response "
            "hash, this excludes per-request timestamps and permission IDs."
        ),
    )
    parser.add_argument(
        "--upstream-model-verified",
        action="store_true",
        help="Report that startup verified the configured model in the upstream model list.",
    )
    parser.add_argument("--agent-contract-path", type=str, default="")
    parser.add_argument("--agent-contract-manifest-sha256", type=str, default="")
    parser.add_argument(
        "--serving-runtime-identity-path",
        type=str,
        default="",
        help=(
            "Optional absolute path to a content-addressed JSON description of the "
            "serving runtime. Must be paired with --serving-runtime-identity-sha256."
        ),
    )
    parser.add_argument(
        "--serving-runtime-identity-sha256",
        type=str,
        default="",
        help="SHA256 of the exact serving-runtime-identity JSON bytes.",
    )
    parser.add_argument(
        "--codex-bin",
        type=str,
        default="codex",
        help="Codex CLI executable used by the codex-account backend.",
    )
    parser.add_argument(
        "--codex-auth-file",
        type=str,
        default=DEFAULT_CODEX_AUTH_PATH,
        help="Codex auth file used only to verify that ChatGPT account login is present.",
    )
    parser.add_argument(
        "--codex-workdir",
        type=str,
        default="/tmp",
        help="Read-only working directory for ephemeral codex-account inference sessions.",
    )
    storage_group = parser.add_mutually_exclusive_group()
    storage_group.add_argument(
        "--disable-response-storage",
        dest="disable_response_storage",
        action="store_true",
        default=None,
        help="Send store=false for providers that support response storage control.",
    )
    storage_group.add_argument(
        "--enable-response-storage",
        dest="disable_response_storage",
        action="store_false",
        help="Do not force store=false, even if Codex config disables storage.",
    )
    return parser.parse_args(argv)


def _strip_toml_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None:
            return line[:index]
    return line


def _parse_toml_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def _load_toml(path: str) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not path or not config_path.exists():
        return {}
    text = config_path.read_text(encoding="utf-8")
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]
    if tomllib is not None:
        return tomllib.loads(text)

    try:
        import tomli  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        tomli = None  # type: ignore[assignment]
    if tomli is not None:
        return tomli.loads(text)

    data: dict[str, Any] = {}
    current = data
    for raw_line in text.splitlines():
        line = _strip_toml_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            table = line[1:-1].strip()
            current = data
            for part in table.split("."):
                current = current.setdefault(part, {})
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        current[key.strip()] = _parse_toml_scalar(value)
    return data


def _provider_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    provider_name = str(config.get("model_provider", "") or "")
    providers = config.get("model_providers", {})
    if not provider_name or not isinstance(providers, dict):
        return provider_name, {}
    provider = providers.get(provider_name, {})
    if not isinstance(provider, dict):
        return provider_name, {}
    return provider_name, provider


def _api_mode_from_wire_api(wire_api: Any) -> str:
    value = str(wire_api or "").strip().lower().replace("-", "_")
    if value in {"responses", "response"}:
        return "responses"
    if value in {"chat", "chat_completions", "chat_completions_api"}:
        return "chat"
    return ""


def _resolve_runtime_options(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_toml(str(args.config_file or ""))
    provider_name, provider = _provider_config(config)
    if args.provider_name is not None:
        provider_name = str(args.provider_name)
    configured_model = str(config.get("model") or "")
    model = str(args.model or configured_model or "gpt-5.5")
    base_url = args.base_url
    if base_url is None:
        base_url = str(provider.get("base_url") or os.getenv("OPENAI_BASE_URL", ""))
    api_mode = args.api_mode or _api_mode_from_wire_api(provider.get("wire_api")) or "chat"
    reasoning_effort = args.reasoning_effort
    if reasoning_effort is None:
        # A Codex config default belongs to its configured model.  Reusing it for an
        # explicitly different OpenAI-compatible model can silently send values such
        # as ``xhigh`` to a local provider that does not implement that contract.
        inherit_config_reasoning = not (
            args.model is not None
            and configured_model
            and str(args.model) != configured_model
        )
        reasoning_effort = (
            str(config.get("model_reasoning_effort") or "")
            if inherit_config_reasoning
            else ""
        )
    disable_response_storage = args.disable_response_storage
    if disable_response_storage is None:
        disable_response_storage = bool(config.get("disable_response_storage", False))
    return {
        "provider_name": provider_name,
        "model": model,
        "base_url": str(base_url or ""),
        "api_mode": api_mode,
        "reasoning_effort": str(reasoning_effort or ""),
        "disable_response_storage": bool(disable_response_storage),
    }


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _validated_loopback_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(str(value).strip())
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("--local-only requires an explicit loopback http(s) --base-url")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "--local-only --base-url must not contain credentials, a query, or a fragment"
        )
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
    )


def _safe_url_for_reporting(value: str) -> str:
    raw = str(value or "")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return _redact_sensitive_text(raw, limit=500)
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _validate_identity_file(path_value: str, expected_sha256: str, *, label: str) -> Path:
    if not _SHA256_RE.fullmatch(str(expected_sha256)):
        raise ValueError(f"{label} SHA256 must be 64 lowercase hexadecimal characters")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}")
    return path


def _verify_local_upstream_model_details(
    *,
    base_url: str,
    model: str,
    timeout_sec: float,
    opener: Any | None = None,
) -> dict[str, str]:
    normalized = _validated_loopback_base_url(base_url)
    models_url = f"{normalized}/models"
    direct_opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    http_request = urllib.request.Request(
        models_url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with direct_opener.open(http_request, timeout=timeout_sec) as response:
            status = int(getattr(response, "status", response.getcode()))
            body = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"local upstream model verification failed ({type(exc).__name__})"
        ) from exc
    if not 200 <= status < 300:
        raise RuntimeError(f"local upstream /models returned HTTP {status}")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("local upstream /models returned invalid JSON") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    model_ids = [
        item.get("id") for item in data or [] if isinstance(item, dict)
    ] if isinstance(data, list) else []
    if model not in model_ids:
        raise RuntimeError(f"local upstream /models does not list the exact model {model!r}")
    return {
        "response_sha256": hashlib.sha256(body).hexdigest(),
        "identity_sha256": models_identity_sha256(payload),
    }


def _verify_local_upstream_model(
    *,
    base_url: str,
    model: str,
    timeout_sec: float,
    opener: Any | None = None,
) -> str:
    """Backward-compatible raw response digest used by existing callers/tests."""

    return _verify_local_upstream_model_details(
        base_url=base_url,
        model=model,
        timeout_sec=timeout_sec,
        opener=opener,
    )["response_sha256"]


def _validate_runtime_configuration(
    args: argparse.Namespace,
    options: dict[str, Any],
) -> None:
    if args.max_output_tokens < 0:
        raise ValueError("--max-output-tokens must be non-negative")
    if args.planner_max_output_tokens < 0:
        raise ValueError("--planner-max-output-tokens must be non-negative")
    if (
        args.max_output_tokens > 0
        and args.planner_max_output_tokens > args.max_output_tokens
    ):
        raise ValueError(
            "--planner-max-output-tokens cannot exceed --max-output-tokens"
        )
    if args.context_limit < 0:
        raise ValueError("--context-limit must be non-negative")
    if (
        args.context_limit > 0
        and args.planner_max_output_tokens > args.context_limit
    ):
        raise ValueError(
            "--planner-max-output-tokens cannot exceed --context-limit"
        )
    if (
        args.planner_prompt_mode != "legacy_duplicate"
        or args.planner_max_output_tokens > 0
    ) and not args.local_only:
        raise ValueError(
            "non-default --planner-prompt-mode and --planner-max-output-tokens "
            "are supported only with --local-only"
        )
    if args.temperature is not None and args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.top_p is not None and not 0.0 <= args.top_p <= 1.0:
        raise ValueError("--top-p must be between 0 and 1")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.min_p is not None and not 0.0 <= args.min_p <= 1.0:
        raise ValueError("--min-p must be between 0 and 1")
    if args.upstream_model_verified and not _SHA256_RE.fullmatch(
        str(args.upstream_models_response_sha256)
    ):
        raise ValueError(
            "--upstream-model-verified requires a 64-character lowercase models-response SHA256"
        )
    if args.upstream_models_identity_sha256 and not _SHA256_RE.fullmatch(
        str(args.upstream_models_identity_sha256)
    ):
        raise ValueError(
            "--upstream-models-identity-sha256 must be a 64-character lowercase SHA256"
        )

    runtime_identity_path = str(args.serving_runtime_identity_path or "")
    runtime_identity_sha256 = str(args.serving_runtime_identity_sha256 or "")
    if bool(runtime_identity_path) != bool(runtime_identity_sha256):
        raise ValueError(
            "--serving-runtime-identity-path and "
            "--serving-runtime-identity-sha256 must be supplied together"
        )
    if runtime_identity_path:
        if not Path(runtime_identity_path).is_absolute():
            raise ValueError("--serving-runtime-identity-path must be absolute")
        runtime_identity_file = _validate_identity_file(
            runtime_identity_path,
            runtime_identity_sha256,
            label="serving runtime identity",
        )
        try:
            runtime_identity = json.loads(runtime_identity_file.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("serving runtime identity must contain valid UTF-8 JSON") from exc
        if not isinstance(runtime_identity, dict):
            raise ValueError("serving runtime identity must be a JSON object")

    provider_specific_controls: list[str] = []
    if args.top_k is not None:
        provider_specific_controls.append("--top-k")
    if args.min_p is not None:
        provider_specific_controls.append("--min-p")
    if args.thinking_mode != "provider_default":
        provider_specific_controls.append("--thinking-mode")

    effective_api_mode = (
        "responses_compat" if args.backend == "codex-account" else str(options["api_mode"])
    )
    if effective_api_mode != "chat" and provider_specific_controls:
        controls = ", ".join(provider_specific_controls)
        raise ValueError(
            f"{controls} require --api-mode chat; effective API mode is {effective_api_mode}"
        )

    # The local Codex account adapter intentionally does not implement sampling or
    # output caps.  Refuse explicit values instead of reporting settings in /health
    # that would never reach the model.  Its existing defaults remain unchanged.
    if args.backend == "codex-account":
        ignored_controls: list[str] = []
        if args.temperature is not None:
            ignored_controls.append("--temperature")
        if args.top_p is not None:
            ignored_controls.append("--top-p")
        if args.max_output_tokens > 0:
            ignored_controls.append("--max-output-tokens")
        if ignored_controls:
            controls = ", ".join(ignored_controls)
            raise ValueError(f"{controls} are not supported by --backend codex-account")

    if args.local_only:
        if args.backend != "openai":
            raise ValueError("--local-only is supported only with --backend openai")
        if str(args.config_file or ""):
            raise ValueError("--local-only requires --config-file '' to prevent inherited routes")
        if args.base_url is None or args.model is None or args.provider_name is None:
            raise ValueError(
                "--local-only requires explicit --base-url, --model, and --provider-name"
            )
        normalized = _validated_loopback_base_url(str(options["base_url"]))
        options["base_url"] = normalized
        if options["api_mode"] != "chat":
            raise ValueError("--local-only currently requires --api-mode chat")
        if int(args.max_retries) != 0:
            raise ValueError("--local-only requires --max-retries 0")
        if args.disable_response_storage is not True:
            raise ValueError("--local-only requires --disable-response-storage")
        model_manifest = _validate_identity_file(
            args.model_artifact_manifest_path,
            args.model_artifact_manifest_sha256,
            label="model artifact manifest",
        )
        contract_manifest = _validate_identity_file(
            args.agent_contract_path,
            args.agent_contract_manifest_sha256,
            label="Agent contract manifest",
        )
        if not Path(args.model_artifact_path).expanduser().resolve().is_dir():
            raise ValueError("--local-only model artifact path must be an existing directory")
        try:
            model_identity = json.loads(model_manifest.read_text(encoding="utf-8"))
            agent_contract = json.loads(contract_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("--local-only identity manifests must contain valid JSON") from exc
        if model_identity.get("model_directory") != str(
            Path(args.model_artifact_path).expanduser().resolve()
        ):
            raise ValueError("model artifact manifest does not identify --model-artifact-path")
        if model_identity.get("source", {}).get("repo_id") != options["model"]:
            raise ValueError("model artifact manifest does not identify the configured model")
        required_endpoints = agent_contract.get("shared_model_role_contract", {}).get(
            "required_endpoints"
        )
        if set(required_endpoints or []) != set(_BUSINESS_ENDPOINTS):
            raise ValueError("Agent contract manifest does not contain the five required endpoints")


def _load_api_key_from_json(path: str, key_name: str) -> str:
    if not path:
        return ""
    key_path = Path(path).expanduser()
    if not key_path.exists():
        return ""
    try:
        payload = json.loads(key_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to read OpenAI API key file {key_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"OpenAI API key file {key_path} must contain a JSON object")
    value = payload.get(key_name, "")
    if not value:
        return ""
    if not isinstance(value, str):
        raise RuntimeError(f"OpenAI API key field {key_name!r} in {key_path} must be a string")
    return value.strip()


def build_client(
    *,
    api_key_env: str,
    api_key_file: str,
    api_key_json_key: str,
    base_url: str,
    max_retries: int = 3,
    local_only: bool = False,
) -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed; install it before starting the OpenAI planner service")
    if local_only:
        api_key = "EMPTY"
    else:
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            api_key = _load_api_key_from_json(api_key_file, api_key_json_key)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set and {api_key_json_key!r} was not found in {api_key_file}")
    kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": max(0, int(max_retries))}
    if base_url:
        kwargs["base_url"] = base_url
    if local_only:
        if DefaultHttpxClient is None:  # pragma: no cover - OpenAI imports it in supported SDKs
            raise RuntimeError("the installed OpenAI SDK cannot create a proxy-free local client")
        kwargs["http_client"] = DefaultHttpxClient(trust_env=False)
    return OpenAI(**kwargs)


def _validate_codex_account_backend(*, codex_bin: str, auth_file: str, workdir: str) -> str:
    resolved_bin = shutil.which(codex_bin)
    if resolved_bin is None:
        candidate = Path(codex_bin).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved_bin = str(candidate.resolve())
    if resolved_bin is None:
        raise RuntimeError(f"Codex CLI executable was not found: {codex_bin}")

    workdir_path = Path(workdir).expanduser()
    if not workdir_path.is_dir():
        raise RuntimeError(f"Codex inference workdir is not a directory: {workdir_path}")

    auth_path = Path(auth_file).expanduser()
    try:
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to read Codex account auth metadata from {auth_path}: {exc}") from exc
    if not isinstance(auth, dict) or auth.get("auth_mode") != "chatgpt":
        raise RuntimeError(f"Codex auth at {auth_path} is not a ChatGPT account login")
    tokens = auth.get("tokens")
    if not isinstance(tokens, dict) or not any(
        isinstance(tokens.get(name), str) and bool(tokens.get(name))
        for name in ("access_token", "refresh_token")
    ):
        raise RuntimeError(f"Codex auth at {auth_path} does not contain usable account tokens")
    return resolved_bin


_DATA_IMAGE_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)
_SECRET_REPLACEMENTS = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"), "Bearer [REDACTED]"),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
        "[REDACTED_JWT]",
    ),
)
_CODEX_ACCOUNT_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "multi_agent",
    "browser_use",
    "computer_use",
    "in_app_browser",
    "image_generation",
    "plugins",
    "remote_plugin",
    "tool_suggest",
    "goals",
)


def _image_suffix(mime_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(mime_type.lower(), ".img")


def _materialize_codex_messages(
    messages: list[dict[str, Any]],
    *,
    temp_dir: Path,
    max_images: int = 8,
    max_image_bytes: int = 32 * 1024 * 1024,
) -> tuple[str, list[Path]]:
    sections: list[str] = [
        "Act only as a stateless robot-planner inference backend.",
        "Do not use tools, inspect files, run commands, browse, or modify state.",
        "Follow the trusted SYSTEM specification below and return exactly one JSON object with no markdown.",
    ]
    image_paths: list[Path] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = message.get("content", "")
        rendered: list[str] = []
        if isinstance(content, str):
            rendered.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    rendered.append(str(item))
                    continue
                if item.get("type") == "text":
                    rendered.append(str(item.get("text", "")))
                    continue
                if item.get("type") != "image_url":
                    rendered.append(str(item))
                    continue
                image_url = item.get("image_url", {})
                url = str(image_url.get("url", "")) if isinstance(image_url, dict) else str(image_url)
                match = _DATA_IMAGE_RE.match(url)
                if match is None:
                    rendered.append("[remote image omitted: only embedded image data is accepted]")
                    continue
                if len(image_paths) >= max_images:
                    raise ValueError(f"codex-account request exceeds the {max_images}-image limit")
                try:
                    image_bytes = base64.b64decode(match.group(2), validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise ValueError("codex-account request contains invalid base64 image data") from exc
                if len(image_bytes) > max_image_bytes:
                    raise ValueError(
                        f"codex-account image exceeds the {max_image_bytes}-byte decoded size limit"
                    )
                image_index = len(image_paths) + 1
                image_path = temp_dir / f"input_{image_index:02d}{_image_suffix(match.group(1))}"
                image_path.write_bytes(image_bytes)
                image_paths.append(image_path)
                rendered.append(f"[attached image {image_index}]")
        else:
            rendered.append(str(content))
        sections.append(f"\n{role}:\n" + "\n".join(rendered))
    return "\n".join(sections), image_paths


def _codex_error_summary(stderr: str) -> str:
    clean = _redact_sensitive_text(stderr or "", limit=10_000)
    candidates = [
        line.strip()
        for line in clean.splitlines()
        if line.strip().lower().startswith(("error", "stream error", "failed"))
    ]
    if not candidates:
        return "no diagnostic was emitted"
    return candidates[-1][:400]


def _redact_sensitive_text(text: str, *, limit: int = 1000) -> str:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text or "")
    for pattern, replacement in _SECRET_REPLACEMENTS:
        clean = pattern.sub(replacement, clean)
    return clean[: max(0, int(limit))]


def _value_from_mapping_or_object(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _completion_usage_payload(usage: Any) -> dict[str, int] | None:
    """Return token counters only; never serialize arbitrary provider metadata."""

    if usage is None:
        return None
    input_tokens = _value_from_mapping_or_object(
        usage, "input_tokens", "prompt_tokens", "inputTokens"
    )
    output_tokens = _value_from_mapping_or_object(
        usage, "output_tokens", "completion_tokens", "outputTokens"
    )
    total_tokens = _value_from_mapping_or_object(usage, "total_tokens", "totalTokens")
    cached_input_tokens = _value_from_mapping_or_object(
        usage, "cached_input_tokens", "cachedInputTokens"
    )
    reasoning_output_tokens = _value_from_mapping_or_object(
        usage, "reasoning_output_tokens", "reasoningOutputTokens"
    )

    prompt_details = _value_from_mapping_or_object(usage, "prompt_tokens_details")
    completion_details = _value_from_mapping_or_object(usage, "completion_tokens_details")
    if cached_input_tokens is None:
        cached_input_tokens = _value_from_mapping_or_object(prompt_details, "cached_tokens")
    if reasoning_output_tokens is None:
        reasoning_output_tokens = _value_from_mapping_or_object(
            completion_details, "reasoning_tokens"
        )

    counters = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
    }
    result: dict[str, int] = {}
    for name, raw_value in counters.items():
        if isinstance(raw_value, bool):
            continue
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed >= 0:
            result[name] = parsed
    return result or None


def _normalize_completion_json(
    text: str,
    normalizer: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    try:
        normalized = normalizer(extract_json_object(text))
    except Exception as exc:
        raise ModelOutputError(
            f"upstream model returned invalid structured output ({type(exc).__name__})"
        ) from exc
    if not isinstance(normalized, dict):
        raise ModelOutputError("upstream model normalizer did not return a JSON object")
    return normalized


def _exception_status_code(exc: Exception) -> int | None:
    candidates = (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    )
    for value in candidates:
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 100 <= parsed <= 599:
            return parsed
    return None


def _classify_business_exception(exc: Exception) -> tuple[HTTPStatus, str, str]:
    if isinstance(exc, ResponsesCompatError):
        status = HTTPStatus(exc.status_code)
        category = "malformed_json" if exc.code == "invalid_json" else "invalid_request"
        return status, category, _redact_sensitive_text(str(exc), limit=500)
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        return HTTPStatus.BAD_REQUEST, "malformed_json", "The request body is not valid JSON."
    if isinstance(exc, (KeyError, TypeError, ValueError)) and not isinstance(
        exc, ModelOutputError
    ):
        return (
            HTTPStatus.BAD_REQUEST,
            "invalid_request",
            _redact_sensitive_text(str(exc), limit=500) or "The request payload is invalid.",
        )
    if isinstance(exc, GatewayConfigurationError):
        return (
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "gateway_configuration_error",
            "The model gateway is not configured.",
        )

    class_name = type(exc).__name__.lower()
    if isinstance(exc, (TimeoutError, CodexAppServerTimeoutError)) or "timeout" in class_name:
        return (
            HTTPStatus.GATEWAY_TIMEOUT,
            "upstream_timeout",
            "The upstream model request timed out.",
        )

    status_code = _exception_status_code(exc)
    if status_code in {400, 413, 422}:
        return (
            HTTPStatus.BAD_GATEWAY,
            "upstream_rejected_request",
            f"The upstream model rejected the request with HTTP {status_code}.",
        )
    message = str(exc).lower()
    oom = (
        isinstance(exc, MemoryError)
        or "out of memory" in message
        or re.search(r"\boom\b", message) is not None
    )
    unavailable = status_code == 503 or "service unavailable" in message
    if oom or unavailable:
        return (
            HTTPStatus.SERVICE_UNAVAILABLE,
            "upstream_oom" if oom else "upstream_unavailable",
            "The upstream model service is unavailable.",
        )
    if isinstance(exc, ModelOutputError):
        return (
            HTTPStatus.BAD_GATEWAY,
            "invalid_upstream_output",
            "The upstream model returned invalid structured output.",
        )
    return (
        HTTPStatus.BAD_GATEWAY,
        "upstream_error",
        "The upstream model request failed.",
    )


def _convert_chat_content_to_responses(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    converted: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            converted.append({"type": "input_text", "text": str(item)})
            continue
        item_type = item.get("type")
        if item_type == "text":
            converted.append({"type": "input_text", "text": str(item.get("text", ""))})
            continue
        if item_type == "image_url":
            image_url = item.get("image_url", {})
            if isinstance(image_url, dict):
                url = str(image_url.get("url", ""))
            else:
                url = str(image_url)
            if url:
                converted.append({"type": "input_image", "image_url": url})
            continue
        converted.append({"type": "input_text", "text": str(item)})
    return converted


def _convert_messages_to_responses_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    response_input: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if role == "system":
            if isinstance(content, str):
                instructions.append(content)
            else:
                instructions.append(str(content))
            continue
        response_role = role if role in {"user", "assistant", "developer"} else "user"
        response_input.append(
            {
                "role": response_role,
                "content": _convert_chat_content_to_responses(content),
            }
        )
    return "\n\n".join(item for item in instructions if item), response_input


class OpenAIPlannerHandler(QwenPlannerHandler):
    backend: str = "openai"
    api_mode: str = "chat"
    response_format_json: bool = False
    reasoning_effort: str = ""
    max_output_tokens: int = 0
    planner_max_output_tokens: int = 0
    planner_prompt_mode: str = "legacy_duplicate"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    thinking_mode: str = "provider_default"
    local_only: bool = False
    base_url: str = ""
    provider_name: str = ""
    model_source: str = ""
    model_revision: str = ""
    quantization: str = ""
    serving_framework: str = ""
    context_limit: int = 0
    model_artifact_path: str = ""
    model_artifact_manifest_path: str = ""
    model_artifact_manifest_sha256: str = ""
    upstream_models_response_sha256: str = ""
    upstream_models_identity_sha256: str = ""
    upstream_model_verified: bool = False
    service_url: str = ""
    agent_contract_path: str = ""
    agent_contract_manifest_sha256: str = ""
    serving_runtime_identity_path: str = ""
    serving_runtime_identity_sha256: str = ""
    disable_response_storage: bool = False
    max_concurrent_requests: int = 2
    max_retries: int = 3
    request_queue_timeout_sec: float = 5.0
    max_request_bytes: int = 64 * 1024 * 1024
    request_semaphore: threading.BoundedSemaphore | None = None
    request_body_semaphore: threading.BoundedSemaphore | None = None
    sse_heartbeat_interval_sec: float = 15.0
    codex_bin: str = "codex"
    codex_home: str = str(Path.home() / ".codex")
    codex_workdir: str = "/tmp"
    codex_app_server: CodexAppServerClient | None = None
    _last_completion_usage: dict[str, int] | None = None

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            app_server = type(self).codex_app_server
            app_server_running = app_server.is_running if app_server is not None else None
            self._send_json(
                {
                    "status": (
                        "degraded"
                        if self.backend == "codex-account" and not app_server_running
                        else "ok"
                    ),
                    "backend": self.backend,
                    "local_only": self.local_only,
                    "service_url": self.service_url,
                    "model": self.model,
                    "api_mode": self.api_mode,
                    "base_url": self.base_url or "default",
                    "provider": self.provider_name or "default",
                    "reasoning_effort": self.reasoning_effort,
                    "thinking_mode": self.thinking_mode,
                    "model_source": self.model_source,
                    "model_revision": self.model_revision,
                    "quantization": self.quantization,
                    "serving_framework": self.serving_framework,
                    "context_limit": self.context_limit,
                    "max_output_tokens": self.max_output_tokens,
                    "planner_max_output_tokens": self.planner_max_output_tokens,
                    "planner_prompt_mode": self.planner_prompt_mode,
                    "sampling": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "min_p": self.min_p,
                    },
                    "model_artifact": {
                        "path": self.model_artifact_path,
                        "manifest_path": self.model_artifact_manifest_path,
                        "manifest_sha256": self.model_artifact_manifest_sha256,
                    },
                    "upstream_model_identity": {
                        "verified": self.upstream_model_verified,
                        "expected_model": self.model,
                        "models_response_sha256": self.upstream_models_response_sha256,
                        "models_identity_sha256": self.upstream_models_identity_sha256,
                    },
                    "agent_contract": {
                        "path": self.agent_contract_path,
                        "manifest_sha256": self.agent_contract_manifest_sha256,
                    },
                    **(
                        {
                            "serving_runtime_identity": {
                                "path": self.serving_runtime_identity_path,
                                "sha256": self.serving_runtime_identity_sha256,
                            }
                        }
                        if self.serving_runtime_identity_path
                        else {}
                    ),
                    "response_storage": (
                        "account_default"
                        if self.backend == "codex-account"
                        else "disabled" if self.disable_response_storage else "provider_default"
                    ),
                    "timeout_sec": self.timeout_sec,
                    "max_concurrent_requests": self.max_concurrent_requests,
                    "max_retries": self.max_retries,
                    "auth_mode": "chatgpt" if self.backend == "codex-account" else "api_key",
                    "responses_endpoint": (
                        "/v1/responses" if self.backend == "codex-account" else None
                    ),
                    "app_server_running": app_server_running,
                    "fallback_enabled": False,
                }
            )
            return
        if path in {"/v1/responses", "/responses"}:
            self._send_openai_error(
                HTTPStatus.METHOD_NOT_ALLOWED,
                "Only POST is supported for this endpoint.",
                error_type="invalid_request_error",
                code="method_not_allowed",
            )
            return
        if path.startswith("/v1/"):
            self._send_openai_error(
                HTTPStatus.NOT_FOUND,
                "Unknown API endpoint.",
                error_type="invalid_request_error",
                code="not_found",
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in {"/v1/responses", "/responses"}:
            try:
                self._handle_responses_api()
            except Exception as exc:
                safe_error = _redact_sensitive_text(str(exc), limit=500)
                print(
                    f"[openai-planner] request failed: {type(exc).__name__}: {safe_error}",
                    file=sys.stderr,
                )
                self._send_codex_openai_error(exc)
            return

        request_started = time.monotonic()
        is_business_endpoint = path in _BUSINESS_ENDPOINTS
        semaphore = type(self).request_semaphore
        acquired = True
        if semaphore is not None:
            acquired = semaphore.acquire(timeout=max(0.0, self.request_queue_timeout_sec))
        if not acquired:
            if is_business_endpoint:
                self._emit_completion_audit(
                    endpoint=path,
                    started_at=request_started,
                    success=False,
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                    error_category="gateway_overloaded",
                )
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": "openai_planner_overloaded",
                    "error_category": "gateway_overloaded",
                    "message": "The local model concurrency limit is full; retry later.",
                    "fallback_used": False,
                },
            )
            return
        try:
            try:
                if is_business_endpoint:
                    self._last_completion_usage = None
                    self._current_request_mode = ""
                    if self.client is None:
                        raise GatewayConfigurationError("client is not initialized")
                    self._prepare_business_request_replay()
                super().do_POST()
            except Exception as exc:
                if is_business_endpoint:
                    status, category, public_message = _classify_business_exception(exc)
                    self._emit_completion_audit(
                        endpoint=path,
                        started_at=request_started,
                        success=False,
                        status=status,
                        error_category=category,
                    )
                    self._send_error_json(
                        status,
                        {
                            "error": "openai_planner_request_failed",
                            "error_category": category,
                            "error_type": type(exc).__name__,
                            "message": public_message,
                            "fallback_used": False,
                        },
                    )
                else:
                    safe_error = _redact_sensitive_text(str(exc), limit=500)
                    print(
                        f"[openai-planner] request failed: {type(exc).__name__}: {safe_error}",
                        file=sys.stderr,
                    )
                    self._send_error_json(
                        HTTPStatus.BAD_GATEWAY,
                        {
                            "error": "openai_planner_request_failed",
                            "error_type": type(exc).__name__,
                            "message": safe_error,
                            "fallback_used": False,
                        },
                    )
            else:
                if is_business_endpoint:
                    self._emit_completion_audit(
                        endpoint=path,
                        started_at=request_started,
                        success=True,
                        status=HTTPStatus.OK,
                        error_category=None,
                    )
        finally:
            if semaphore is not None:
                semaphore.release()

    def _prepare_business_request_replay(self) -> None:
        payload = self._read_json_request()
        if not isinstance(payload, dict):
            raise ResponsesCompatError("The request body must be a JSON object.")
        path = self.path.split("?", 1)[0]
        request_mode = {
            "/plan": "planner",
            "/ood": "ood_detection",
            "/perception_queries": "perception_query_generation",
            "/normalize_perception_queries": "perception_query_normalization",
        }.get(path, "")
        if path == "/recover":
            recovery_payload = payload.get("recovery_payload", {})
            request_mode = (
                "action_effect_verification"
                if isinstance(recovery_payload, dict)
                and recovery_payload.get("mode") == "action_effect_verification"
                else "recovery_planning"
            )
        self._current_request_mode = request_mode
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.rfile = io.BytesIO(encoded)
        if "Content-Length" in self.headers and hasattr(self.headers, "replace_header"):
            self.headers.replace_header("Content-Length", str(len(encoded)))
        else:
            self.headers["Content-Length"] = str(len(encoded))

    def _emit_completion_audit(
        self,
        *,
        endpoint: str,
        started_at: float,
        success: bool,
        status: HTTPStatus | int,
        error_category: str | None,
    ) -> None:
        elapsed_ms = max(0, round((time.monotonic() - started_at) * 1000.0))
        payload: dict[str, Any] = {
            "event": "agent_model_request_complete",
            "endpoint": endpoint,
            "mode": getattr(self, "_current_request_mode", ""),
            "provider": self.provider_name or "default",
            "model": self.model,
            "backend": self.backend,
            "api_mode": self.api_mode,
            "success": bool(success),
            "status_code": int(status),
            "error_category": error_category,
            "elapsed_ms": elapsed_ms,
            "fallback_used": False,
            "configured_max_retries": int(self.max_retries),
            "retry_count": 0 if int(self.max_retries) == 0 else None,
        }
        usage = getattr(self, "_last_completion_usage", None)
        if usage:
            payload["usage"] = usage
        print(
            "[openai-planner-audit] "
            + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    def _read_json_request(self) -> Any:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ResponsesCompatError(
                "Content-Type must be application/json.",
                code="unsupported_media_type",
                status_code=415,
            )
        content_length_text = self.headers.get("Content-Length")
        if content_length_text is None:
            raise ResponsesCompatError(
                "A Content-Length header is required.",
                code="missing_content_length",
                status_code=411,
            )
        try:
            content_length = int(content_length_text)
        except ValueError as exc:
            raise ResponsesCompatError("Content-Length must be an integer.") from exc
        if content_length < 0:
            raise ResponsesCompatError("Content-Length must not be negative.")
        if content_length > self.max_request_bytes:
            raise ResponsesCompatError(
                f"The request body exceeds the {self.max_request_bytes}-byte limit.",
                code="request_too_large",
                status_code=413,
            )
        old_timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(30.0)
            raw_body = self.rfile.read(content_length)
            if len(raw_body) != content_length:
                raise ResponsesCompatError(
                    "The request body ended before Content-Length bytes were received.",
                    code="incomplete_request_body",
                )
            return json.loads(raw_body.decode("utf-8"))
        except (TimeoutError, OSError) as exc:
            self.close_connection = True
            raise ResponsesCompatError(
                "Timed out while reading the request body.",
                code="request_timeout",
                status_code=408,
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResponsesCompatError("The request body is not valid UTF-8 JSON.", code="invalid_json") from exc
        finally:
            try:
                self.connection.settimeout(old_timeout)
            except OSError:
                pass

    @staticmethod
    def _codex_usage_dict(result: Any) -> dict[str, Any] | None:
        usage = getattr(result, "usage", None)
        if usage is None:
            return None
        return {
            "inputTokens": usage.input_tokens,
            "cachedInputTokens": usage.cached_input_tokens,
            "outputTokens": usage.output_tokens,
            "reasoningOutputTokens": usage.reasoning_output_tokens,
            "totalTokens": usage.total_tokens,
        }

    @staticmethod
    def _codex_error_status(exc: Exception) -> tuple[HTTPStatus, str]:
        if isinstance(exc, CodexAppServerTimeoutError):
            return HTTPStatus.GATEWAY_TIMEOUT, "server_error"
        code = getattr(exc, "error_code", None)
        if code is None:
            code = getattr(exc, "code", None)
        if code in {"usageLimitExceeded", "serverOverloaded", -32001}:
            return HTTPStatus.TOO_MANY_REQUESTS, "rate_limit_error"
        if code == "unauthorized":
            return HTTPStatus.UNAUTHORIZED, "authentication_error"
        if code in {"badRequest", "contextWindowExceeded"}:
            return HTTPStatus.BAD_REQUEST, "invalid_request_error"
        return HTTPStatus.BAD_GATEWAY, "server_error"

    def _send_openai_error(
        self,
        status: HTTPStatus | int,
        message: str,
        *,
        error_type: str,
        param: str | None = None,
        code: str | int | None = None,
    ) -> None:
        self._send_error_json(
            HTTPStatus(int(status)),
            {
                "error": {
                    "message": message[:1000],
                    "type": error_type,
                    "param": param,
                    "code": code,
                }
            },
        )

    def _send_codex_openai_error(self, exc: Exception) -> None:
        if isinstance(exc, ResponsesCompatError):
            self._send_error_json(HTTPStatus(exc.status_code), exc.as_openai_error())
            return
        status, error_type = self._codex_error_status(exc)
        error_code: str | int | None
        if isinstance(exc, CodexAppServerTimeoutError):
            error_code = "upstream_timeout"
        else:
            error_code = getattr(
                exc,
                "error_code",
                getattr(exc, "code", "codex_app_server_error"),
            )
        self._send_openai_error(
            status,
            _redact_sensitive_text(str(exc), limit=1000)
            or "The local Codex account adapter failed.",
            error_type=error_type,
            code=error_code,
        )

    def _run_prepared_codex_request(
        self,
        request: PreparedResponsesRequest,
        *,
        on_delta: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        app_server = type(self).codex_app_server
        if self.backend != "codex-account" or app_server is None:
            raise ResponsesCompatError(
                "The local /v1/responses endpoint is available only with --backend codex-account.",
                code="unsupported_backend",
                status_code=404,
            )
        return app_server.run_turn(
            request.user_input,
            model=request.model,
            developer_instructions=request.developer_instructions,
            cwd=self.codex_workdir,
            effort=request.effort,
            summary=request.summary,
            output_schema=request.output_schema,
            timeout=max(1, int(self.timeout_sec)),
            on_delta=on_delta,
            cancel_event=cancel_event,
        )

    def _handle_responses_api(self) -> None:
        body_semaphore = type(self).request_body_semaphore
        body_acquired = True
        if body_semaphore is not None:
            body_acquired = body_semaphore.acquire(
                timeout=max(0.0, self.request_queue_timeout_sec)
            )
        if not body_acquired:
            self._send_openai_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "The local request-body admission limit is full; retry later.",
                error_type="server_error",
                code="server_overloaded",
            )
            return
        semaphore = type(self).request_semaphore
        acquired = True
        try:
            try:
                request = prepare_responses_request(
                    self._read_json_request(),
                    expected_model=self.model,
                    default_effort=self.reasoning_effort or None,
                )
            except ResponsesCompatError as exc:
                self._send_error_json(HTTPStatus(exc.status_code), exc.as_openai_error())
                return
            if semaphore is not None:
                acquired = semaphore.acquire(
                    timeout=max(0.0, self.request_queue_timeout_sec)
                )
        finally:
            if body_semaphore is not None:
                body_semaphore.release()

        if not acquired:
            self._send_openai_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "The local Codex account adapter is at its concurrency limit; retry later.",
                error_type="server_error",
                code="server_overloaded",
            )
            return
        try:
            if request.stream:
                self._handle_streaming_response(request)
                return
            try:
                result = self._run_prepared_codex_request(request)
            except Exception as exc:
                self._send_codex_openai_error(exc)
                return
            response = build_response_object(
                request,
                text=result.text,
                usage=self._codex_usage_dict(result),
            )
            self._send_json(response)
        finally:
            if semaphore is not None:
                semaphore.release()

    def _write_sse(self, event_type: str, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.wfile.write(f"event: {event_type}\n".encode("ascii"))
        self.wfile.write(b"data: " + encoded + b"\n\n")
        self.wfile.flush()

    def _handle_streaming_response(self, request: PreparedResponsesRequest) -> None:
        response_id = new_response_id()
        message_id = new_message_id()
        created_at = int(time.time())
        sequence_number = 0
        item_started = False
        stream_lock = threading.Lock()
        stream_cancelled = threading.Event()
        heartbeat_stop = threading.Event()
        heartbeat_thread: threading.Thread | None = None

        self.connection.settimeout(5.0)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(event_type: str, payload: dict[str, Any]) -> None:
            nonlocal sequence_number
            with stream_lock:
                payload.setdefault("type", event_type)
                payload.setdefault("sequence_number", sequence_number)
                sequence_number += 1
                self._write_sse(event_type, payload)

        in_progress = build_response_object(
            request,
            text="",
            usage=None,
            response_id=response_id,
            message_id=message_id,
            created_at=created_at,
            status="in_progress",
        )
        try:
            emit("response.created", {"response": in_progress})
            emit("response.in_progress", {"response": in_progress})

            def send_heartbeats() -> None:
                while not heartbeat_stop.wait(
                    timeout=max(1.0, float(self.sse_heartbeat_interval_sec))
                ):
                    try:
                        with stream_lock:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                    except Exception:
                        stream_cancelled.set()
                        return

            heartbeat_thread = threading.Thread(
                target=send_heartbeats,
                name=f"responses-heartbeat-{response_id[-8:]}",
                daemon=True,
            )
            heartbeat_thread.start()

            def ensure_item_started() -> None:
                nonlocal item_started
                if item_started:
                    return
                emit(
                    "response.output_item.added",
                    {
                        "output_index": 0,
                        "item": {
                            "id": message_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                )
                emit(
                    "response.content_part.added",
                    {
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                )
                item_started = True

            def on_delta(delta: str) -> None:
                ensure_item_started()
                emit(
                    "response.output_text.delta",
                    {
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta,
                        "logprobs": [],
                    },
                )

            try:
                result = self._run_prepared_codex_request(
                    request,
                    on_delta=on_delta,
                    cancel_event=stream_cancelled,
                )
            finally:
                heartbeat_stop.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=1.0)
            ensure_item_started()
            final_text = result.text
            emit(
                "response.output_text.done",
                {
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": final_text,
                    "logprobs": [],
                },
            )
            part = {"type": "output_text", "text": final_text, "annotations": []}
            emit(
                "response.content_part.done",
                {
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": part,
                },
            )
            item = {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [part],
            }
            emit("response.output_item.done", {"output_index": 0, "item": item})
            completed = build_response_object(
                request,
                text=final_text,
                usage=self._codex_usage_dict(result),
                response_id=response_id,
                message_id=message_id,
                created_at=created_at,
            )
            emit("response.completed", {"response": completed})
        except OSError:
            return
        except Exception as exc:
            status, error_type = self._codex_error_status(exc)
            try:
                emit(
                    "error",
                    {
                        "code": (
                            "upstream_timeout"
                            if isinstance(exc, CodexAppServerTimeoutError)
                            else getattr(
                                exc,
                                "error_code",
                                getattr(exc, "code", "codex_app_server_error"),
                            )
                        ),
                        "message": _redact_sensitive_text(str(exc), limit=1000),
                        "param": None,
                        "status": int(status),
                        "error_type": error_type,
                    },
                )
            except OSError:
                return

    def _run_completion(self, *, messages: list[dict[str, Any]], normalizer: Callable[[dict[str, Any]], dict[str, Any]]):
        if self.client is None:
            raise RuntimeError("client is not initialized")
        if self.backend == "codex-account":
            return self._run_codex_account_completion(messages=messages, normalizer=normalizer)
        if self.api_mode == "responses":
            return self._run_responses_completion(messages=messages, normalizer=normalizer)
        return self._run_chat_completion(messages=messages, normalizer=normalizer)

    def _run_codex_account_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        normalizer: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        instructions, response_input = _convert_messages_to_responses_input(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": response_input,
            "stream": False,
            "store": False,
            "text": {"format": {"type": "json_object"}},
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        request = prepare_responses_request(
            payload,
            expected_model=self.model,
            default_effort=self.reasoning_effort or None,
        )
        result = self._run_prepared_codex_request(request)
        self._last_completion_usage = _completion_usage_payload(
            self._codex_usage_dict(result)
        )
        return _normalize_completion_json(result.text, normalizer)

    def _run_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        normalizer: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "timeout": self.timeout_sec,
        }
        if self.response_format_json:
            request["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        effective_max_output_tokens = self._effective_max_output_tokens()
        if effective_max_output_tokens > 0:
            request["max_completion_tokens"] = effective_max_output_tokens
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.top_p is not None:
            request["top_p"] = self.top_p
        extra_body: dict[str, Any] = {}
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if self.thinking_mode in {"enabled", "disabled"}:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": self.thinking_mode == "enabled"
            }
        if extra_body:
            request["extra_body"] = extra_body
        if self.disable_response_storage:
            request["store"] = False
        response = self.client.chat.completions.create(**request)
        self._last_completion_usage = _completion_usage_payload(
            getattr(response, "usage", None)
        )
        text = response.choices[0].message.content or ""
        return _normalize_completion_json(text, normalizer)

    def _effective_max_output_tokens(self) -> int:
        if (
            getattr(self, "_current_request_mode", "") == "planner"
            and self.planner_max_output_tokens > 0
        ):
            return int(self.planner_max_output_tokens)
        return int(self.max_output_tokens)

    def _run_responses_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        normalizer: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        instructions, response_input = _convert_messages_to_responses_input(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "input": response_input,
            "instructions": instructions,
            "stream": False,
            "timeout": self.timeout_sec,
        }
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}
        effective_max_output_tokens = self._effective_max_output_tokens()
        if effective_max_output_tokens > 0:
            request["max_output_tokens"] = effective_max_output_tokens
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.top_p is not None:
            request["top_p"] = self.top_p
        if self.disable_response_storage:
            request["store"] = False
        response = self.client.responses.create(**request)
        self._last_completion_usage = _completion_usage_payload(
            getattr(response, "usage", None)
        )
        text = getattr(response, "output_text", "") or ""
        return _normalize_completion_json(text, normalizer)

    def _send_error_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    args = parse_args()
    options = _resolve_runtime_options(args)
    _validate_runtime_configuration(args, options)
    if args.local_only:
        verified_identity = _verify_local_upstream_model_details(
            base_url=options["base_url"],
            model=options["model"],
            timeout_sec=max(1.0, float(args.timeout_sec)),
        )
        verified_sha256 = verified_identity["response_sha256"]
        verified_identity_sha256 = verified_identity["identity_sha256"]
        if (
            args.upstream_models_response_sha256
            and args.upstream_models_response_sha256 != verified_sha256
        ):
            raise RuntimeError(
                "independent local /models response hash does not match the supplied hash"
            )
        args.upstream_models_response_sha256 = verified_sha256
        if (
            args.upstream_models_identity_sha256
            and args.upstream_models_identity_sha256 != verified_identity_sha256
        ):
            raise RuntimeError(
                "independent local /models stable identity hash does not match the supplied hash"
            )
        args.upstream_models_identity_sha256 = verified_identity_sha256
        args.upstream_model_verified = True
    OpenAIPlannerHandler.backend = args.backend
    if args.backend == "codex-account":
        OpenAIPlannerHandler.codex_bin = _validate_codex_account_backend(
            codex_bin=args.codex_bin,
            auth_file=args.codex_auth_file,
            workdir=args.codex_workdir,
        )
        OpenAIPlannerHandler.codex_home = str(
            Path(args.codex_auth_file).expanduser().resolve().parent
        )
        OpenAIPlannerHandler.codex_workdir = str(Path(args.codex_workdir).expanduser().resolve())
        OpenAIPlannerHandler.codex_app_server = CodexAppServerClient(
            codex_bin=OpenAIPlannerHandler.codex_bin,
            codex_home=OpenAIPlannerHandler.codex_home,
            process_cwd=OpenAIPlannerHandler.codex_workdir,
            startup_timeout=min(60.0, max(5.0, float(args.timeout_sec))),
            rpc_timeout=min(60.0, max(5.0, float(args.timeout_sec))),
        ).start()
        OpenAIPlannerHandler.client = object()  # Local authenticated Codex CLI transport sentinel.
        options["provider_name"] = "openai"
        options["base_url"] = "codex-app-server"
        options["api_mode"] = "responses_compat"
    else:
        OpenAIPlannerHandler.client = build_client(
            api_key_env=args.api_key_env,
            api_key_file=args.api_key_file,
            api_key_json_key=args.api_key_json_key,
            base_url=options["base_url"],
            max_retries=args.max_retries,
            local_only=bool(args.local_only),
        )
    OpenAIPlannerHandler.model = options["model"]
    OpenAIPlannerHandler.timeout_sec = args.timeout_sec
    OpenAIPlannerHandler.api_mode = options["api_mode"]
    OpenAIPlannerHandler.response_format_json = bool(args.response_format_json)
    OpenAIPlannerHandler.reasoning_effort = options["reasoning_effort"]
    OpenAIPlannerHandler.max_output_tokens = int(args.max_output_tokens)
    OpenAIPlannerHandler.planner_max_output_tokens = int(
        args.planner_max_output_tokens
    )
    OpenAIPlannerHandler.planner_prompt_mode = str(args.planner_prompt_mode)
    OpenAIPlannerHandler.temperature = args.temperature
    OpenAIPlannerHandler.top_p = args.top_p
    OpenAIPlannerHandler.top_k = args.top_k
    OpenAIPlannerHandler.min_p = args.min_p
    OpenAIPlannerHandler.thinking_mode = str(args.thinking_mode)
    OpenAIPlannerHandler.local_only = bool(args.local_only)
    OpenAIPlannerHandler.base_url = _safe_url_for_reporting(options["base_url"])
    OpenAIPlannerHandler.provider_name = options["provider_name"]
    OpenAIPlannerHandler.model_source = str(args.model_source)
    OpenAIPlannerHandler.model_revision = str(args.model_revision)
    OpenAIPlannerHandler.quantization = str(args.quantization)
    OpenAIPlannerHandler.serving_framework = str(args.serving_framework)
    OpenAIPlannerHandler.context_limit = int(args.context_limit)
    OpenAIPlannerHandler.model_artifact_path = str(args.model_artifact_path)
    OpenAIPlannerHandler.model_artifact_manifest_path = str(
        args.model_artifact_manifest_path
    )
    OpenAIPlannerHandler.model_artifact_manifest_sha256 = str(
        args.model_artifact_manifest_sha256
    )
    OpenAIPlannerHandler.upstream_models_response_sha256 = str(
        args.upstream_models_response_sha256
    )
    OpenAIPlannerHandler.upstream_models_identity_sha256 = str(
        args.upstream_models_identity_sha256
    )
    OpenAIPlannerHandler.upstream_model_verified = bool(args.upstream_model_verified)
    OpenAIPlannerHandler.service_url = f"http://{args.host}:{args.port}"
    OpenAIPlannerHandler.agent_contract_path = str(args.agent_contract_path)
    OpenAIPlannerHandler.agent_contract_manifest_sha256 = str(
        args.agent_contract_manifest_sha256
    )
    OpenAIPlannerHandler.serving_runtime_identity_path = (
        str(Path(args.serving_runtime_identity_path).resolve())
        if args.serving_runtime_identity_path
        else ""
    )
    OpenAIPlannerHandler.serving_runtime_identity_sha256 = str(
        args.serving_runtime_identity_sha256
    )
    OpenAIPlannerHandler.disable_response_storage = options["disable_response_storage"]
    OpenAIPlannerHandler.max_concurrent_requests = max(1, int(args.max_concurrent_requests))
    OpenAIPlannerHandler.max_retries = max(0, int(args.max_retries))
    OpenAIPlannerHandler.request_queue_timeout_sec = max(0.0, float(args.request_queue_timeout_sec))
    OpenAIPlannerHandler.max_request_bytes = max(1, int(args.max_request_bytes))
    OpenAIPlannerHandler.request_semaphore = threading.BoundedSemaphore(
        OpenAIPlannerHandler.max_concurrent_requests
    )
    OpenAIPlannerHandler.request_body_semaphore = threading.BoundedSemaphore(
        OpenAIPlannerHandler.max_concurrent_requests
    )
    try:
        server = ThreadingHTTPServer((args.host, args.port), OpenAIPlannerHandler)
    except BaseException:
        app_server = OpenAIPlannerHandler.codex_app_server
        if app_server is not None:
            app_server.close()
        raise
    print(
        f"[openai-planner] listening on http://{args.host}:{args.port} "
        f"backend={OpenAIPlannerHandler.backend} "
        f"model={OpenAIPlannerHandler.model} api_mode={OpenAIPlannerHandler.api_mode} "
        f"provider={OpenAIPlannerHandler.provider_name or 'default'} "
        f"base_url={OpenAIPlannerHandler.base_url or 'default'} "
        f"max_concurrent_requests={OpenAIPlannerHandler.max_concurrent_requests} "
        f"max_retries={OpenAIPlannerHandler.max_retries}"
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        app_server = OpenAIPlannerHandler.codex_app_server
        if app_server is not None:
            app_server.close()


if __name__ == "__main__":
    main()
