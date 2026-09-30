#!/usr/bin/env python3
"""Bind fresh local-model preflights to one content-addressed gateway identity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_MODALITY_MODES = {
    "text_only",
    "head_image",
    "head_then_third_images",
}
REQUIRED_FULLSTACK_MODES = {
    "planner_context_contract",
    "planner",
    "ood_detection",
    "recovery_planning",
    "action_effect_verification",
    "perception_query_generation",
    "perception_query_normalization",
    "malformed_json_negative",
}


class EvidenceBindingError(ValueError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and content-address two real preflight reports against an "
            "unchanged local gateway health identity."
        )
    )
    parser.add_argument("--gateway-health-before", type=Path, required=True)
    parser.add_argument("--gateway-health-after", type=Path, required=True)
    parser.add_argument("--modality-report", type=Path, required=True)
    parser.add_argument("--fullstack-report", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-engine-base-url", required=True)
    parser.add_argument("--expected-gateway-base-url", required=True)
    parser.add_argument("--expected-head-image-sha256", required=True)
    parser.add_argument("--expected-third-image-sha256", required=True)
    parser.add_argument(
        "--expected-planner-task-min-bytes",
        type=int,
        required=True,
        help="Minimum UTF-8 byte count required from the bound long /plan task probe.",
    )
    parser.add_argument(
        "--expected-planner-max-output-tokens",
        type=int,
        required=True,
        help="Expected route-specific /plan output-token cap from gateway health.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _load_json_object(path: Path, *, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    if not path.is_absolute():
        raise EvidenceBindingError(f"{label} path must be absolute: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise EvidenceBindingError(f"{label} is not a regular file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceBindingError(f"{label} is not valid UTF-8 JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise EvidenceBindingError(f"{label} must be a JSON object")
    return resolved, raw, payload


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise EvidenceBindingError(f"{label}={actual!r}, expected {expected!r}")


def _require_int_at_least(actual: Any, minimum: int, *, label: str) -> int:
    if not isinstance(actual, int) or isinstance(actual, bool):
        raise EvidenceBindingError(f"{label} must be an integer")
    if actual < minimum:
        raise EvidenceBindingError(f"{label}={actual}, expected at least {minimum}")
    return actual


def build_binding(
    *,
    gateway_health_before_path: Path,
    gateway_health_after_path: Path,
    modality_report_path: Path,
    fullstack_report_path: Path,
    expected_model: str,
    expected_engine_base_url: str,
    expected_gateway_base_url: str,
    expected_head_image_sha256: str,
    expected_third_image_sha256: str,
    expected_planner_task_min_bytes: int,
    expected_planner_max_output_tokens: int,
) -> tuple[dict[str, Any], Path]:
    if expected_planner_task_min_bytes <= 0:
        raise EvidenceBindingError("expected planner task minimum bytes must be positive")
    if expected_planner_max_output_tokens <= 0:
        raise EvidenceBindingError("expected planner max output tokens must be positive")
    for label, digest in (
        ("expected head image", expected_head_image_sha256),
        ("expected third image", expected_third_image_sha256),
    ):
        if not SHA256_RE.fullmatch(digest):
            raise EvidenceBindingError(f"{label} SHA256 is invalid")
    before_path, before_raw, before = _load_json_object(
        gateway_health_before_path, label="gateway health before"
    )
    after_path, after_raw, after = _load_json_object(
        gateway_health_after_path, label="gateway health after"
    )
    if before_raw != after_raw:
        raise EvidenceBindingError(
            "gateway health identity changed while the preflights were running"
        )
    gateway_identity_sha256 = _sha256(after_raw)
    expected_health = {
        "status": "ok",
        "local_only": True,
        "model": expected_model,
        "service_url": expected_gateway_base_url.rstrip("/"),
        "base_url": expected_engine_base_url.rstrip("/"),
        "planner_prompt_mode": "rendered_system_once",
        "planner_max_output_tokens": expected_planner_max_output_tokens,
        "fallback_enabled": False,
        "response_storage": "disabled",
    }
    for key, expected_value in expected_health.items():
        _require_equal(after.get(key), expected_value, label=f"gateway health {key}")

    upstream_identity = after.get("upstream_model_identity")
    if not isinstance(upstream_identity, dict):
        raise EvidenceBindingError("gateway health lacks upstream_model_identity")
    _require_equal(
        upstream_identity.get("verified"), True, label="upstream_model_identity.verified"
    )
    _require_equal(
        upstream_identity.get("expected_model"),
        expected_model,
        label="upstream_model_identity.expected_model",
    )
    upstream_models_sha256 = upstream_identity.get("models_response_sha256")
    if not isinstance(upstream_models_sha256, str) or not SHA256_RE.fullmatch(
        upstream_models_sha256
    ):
        raise EvidenceBindingError(
            "upstream_model_identity.models_response_sha256 is not a SHA256"
        )
    upstream_models_identity_sha256 = upstream_identity.get("models_identity_sha256")
    if not isinstance(
        upstream_models_identity_sha256, str
    ) or not SHA256_RE.fullmatch(upstream_models_identity_sha256):
        raise EvidenceBindingError(
            "upstream_model_identity.models_identity_sha256 is not a SHA256"
        )

    runtime_reference = after.get("serving_runtime_identity")
    if not isinstance(runtime_reference, dict):
        raise EvidenceBindingError("gateway health lacks serving_runtime_identity")
    runtime_path_text = runtime_reference.get("path")
    runtime_sha256 = runtime_reference.get("sha256")
    if not isinstance(runtime_path_text, str) or not Path(runtime_path_text).is_absolute():
        raise EvidenceBindingError("serving_runtime_identity.path must be absolute")
    if not isinstance(runtime_sha256, str) or not SHA256_RE.fullmatch(runtime_sha256):
        raise EvidenceBindingError("serving_runtime_identity.sha256 is not a SHA256")
    runtime_path, runtime_raw, runtime = _load_json_object(
        Path(runtime_path_text), label="serving runtime identity"
    )
    _require_equal(
        _sha256(runtime_raw), runtime_sha256, label="serving runtime identity SHA256"
    )
    serving_framework = after.get("serving_framework")
    if not isinstance(serving_framework, str) or not serving_framework.strip():
        raise EvidenceBindingError(
            "gateway health serving_framework must be a non-empty string"
        )
    runtime_vllm = runtime.get("vllm")
    if not isinstance(runtime_vllm, dict):
        raise EvidenceBindingError(
            "serving runtime identity vllm must be a JSON object"
        )
    runtime_vllm_version = runtime_vllm.get("version_output")
    if not isinstance(runtime_vllm_version, str) or not runtime_vllm_version.strip():
        raise EvidenceBindingError(
            "serving runtime identity vllm.version_output must be a non-empty string"
        )
    if serving_framework != runtime_vllm_version:
        raise EvidenceBindingError(
            "gateway health serving_framework does not match serving runtime "
            "identity vllm.version_output"
        )

    modality_path, modality_raw, modality = _load_json_object(
        modality_report_path, label="modality preflight report"
    )
    _require_equal(modality.get("status"), "passed", label="modality report status")
    _require_equal(modality.get("model"), expected_model, label="modality report model")
    _require_equal(
        modality.get("engine_base_url"),
        expected_engine_base_url.rstrip("/"),
        label="modality report engine_base_url",
    )
    _require_equal(modality.get("fallback_count"), 0, label="modality fallback_count")
    _require_equal(modality.get("errors"), [], label="modality errors")
    modality_model_check = modality.get("model_check")
    if not isinstance(modality_model_check, dict):
        raise EvidenceBindingError("modality report lacks model_check")
    _require_equal(
        modality_model_check.get("exact_model_found"),
        True,
        label="modality model_check.exact_model_found",
    )
    modality_models_response_sha256 = modality_model_check.get("response_sha256")
    if not isinstance(
        modality_models_response_sha256, str
    ) or not SHA256_RE.fullmatch(modality_models_response_sha256):
        raise EvidenceBindingError("modality model_check.response_sha256 is not a SHA256")
    _require_equal(
        modality_model_check.get("identity_sha256"),
        upstream_models_identity_sha256,
        label="modality /models stable identity SHA256",
    )
    modality_requests = modality.get("requests")
    if not isinstance(modality_requests, list):
        raise EvidenceBindingError("modality report requests must be a list")
    _require_equal(
        len(modality_requests), len(REQUIRED_MODALITY_MODES), label="modality request count"
    )
    modality_modes = {
        item.get("request_mode")
        for item in modality_requests
        if isinstance(item, dict)
        and isinstance(item.get("request_mode"), str)
        and item.get("status") == "passed"
        and item.get("fallback_count") == 0
    }
    if modality_modes != REQUIRED_MODALITY_MODES:
        raise EvidenceBindingError(
            f"modality preflight passed modes={sorted(modality_modes)!r}, "
            f"expected {sorted(REQUIRED_MODALITY_MODES)!r}"
        )
    modality_by_mode = {item["request_mode"]: item for item in modality_requests}
    expected_modality_image_hashes = {
        "text_only": [],
        "head_image": [expected_head_image_sha256],
        "head_then_third_images": [
            expected_head_image_sha256,
            expected_third_image_sha256,
        ],
    }
    for mode, expected_hashes in expected_modality_image_hashes.items():
        _require_equal(
            modality_by_mode[mode].get("image_sha256"),
            expected_hashes,
            label=f"modality {mode} image_sha256",
        )

    fullstack_path, fullstack_raw, fullstack = _load_json_object(
        fullstack_report_path, label="full-stack preflight report"
    )
    _require_equal(fullstack.get("passed"), True, label="full-stack report passed")
    _require_equal(
        fullstack.get("gateway_base_url"),
        expected_gateway_base_url.rstrip("/"),
        label="full-stack report gateway_base_url",
    )
    _require_equal(fullstack.get("fallback_count"), 0, label="full-stack fallback_count")
    _require_equal(fullstack.get("failed_probe_count"), 0, label="failed_probe_count")
    _require_equal(fullstack.get("failures"), [], label="full-stack failures")
    source_images = fullstack.get("source_images")
    if not isinstance(source_images, list) or len(source_images) != 2:
        raise EvidenceBindingError("full-stack source_images must contain head and third")
    for index, (camera, digest) in enumerate(
        (
            ("head", expected_head_image_sha256),
            ("third", expected_third_image_sha256),
        )
    ):
        image = source_images[index]
        if not isinstance(image, dict):
            raise EvidenceBindingError(f"full-stack source_images[{index}] must be an object")
        _require_equal(image.get("order"), index, label=f"source_images[{index}].order")
        _require_equal(image.get("camera"), camera, label=f"source_images[{index}].camera")
        _require_equal(image.get("sha256"), digest, label=f"source_images[{index}].sha256")
    probes = fullstack.get("probes")
    if not isinstance(probes, list):
        raise EvidenceBindingError("full-stack report probes must be a list")
    _require_equal(len(probes), len(REQUIRED_FULLSTACK_MODES), label="full-stack probe count")
    passed_modes = {
        item.get("mode")
        for item in probes
        if isinstance(item, dict)
        and isinstance(item.get("mode"), str)
        and item.get("status") == "passed"
        and item.get("fallback_count") == 0
    }
    if passed_modes != REQUIRED_FULLSTACK_MODES:
        raise EvidenceBindingError(
            f"full-stack preflight passed modes={sorted(passed_modes)!r}, "
            f"expected {sorted(REQUIRED_FULLSTACK_MODES)!r}"
        )
    _require_equal(
        fullstack.get("passed_probe_count"), len(probes), label="passed_probe_count"
    )
    _require_equal(fullstack.get("probe_count"), len(probes), label="probe_count")

    probes_by_mode = {
        item["mode"]: item
        for item in probes
        if isinstance(item, dict) and isinstance(item.get("mode"), str)
    }
    context_probe = probes_by_mode["planner_context_contract"]
    _require_equal(
        context_probe.get("request_image_count"),
        0,
        label="planner context contract request_image_count",
    )
    _require_equal(
        context_probe.get("planner_prompt_mode"),
        after["planner_prompt_mode"],
        label="planner context contract planner_prompt_mode",
    )
    _require_equal(
        context_probe.get("expected_planner_prompt_mode"),
        after["planner_prompt_mode"],
        label="planner context contract expected_planner_prompt_mode",
    )
    _require_equal(
        context_probe.get("planner_max_output_tokens"),
        after["planner_max_output_tokens"],
        label="planner context contract planner_max_output_tokens",
    )
    _require_equal(
        context_probe.get("expected_planner_max_output_tokens"),
        after["planner_max_output_tokens"],
        label="planner context contract expected_planner_max_output_tokens",
    )

    planner_payload = fullstack.get("planner_payload")
    if not isinstance(planner_payload, dict):
        raise EvidenceBindingError("full-stack report lacks planner_payload")
    _require_equal(
        planner_payload.get("requested_min_task_utf8_byte_count"),
        expected_planner_task_min_bytes,
        label="planner payload requested_min_task_utf8_byte_count",
    )
    planner_task_bytes = _require_int_at_least(
        planner_payload.get("actual_task_utf8_byte_count"),
        expected_planner_task_min_bytes,
        label="planner payload actual_task_utf8_byte_count",
    )
    planner_non_image_bytes = _require_int_at_least(
        planner_payload.get("actual_non_image_byte_count"),
        planner_task_bytes,
        label="planner payload actual_non_image_byte_count",
    )
    planner_request_bytes = _require_int_at_least(
        planner_payload.get("actual_request_body_byte_count"),
        planner_non_image_bytes,
        label="planner payload actual_request_body_byte_count",
    )
    _require_equal(
        planner_payload.get("request_image_count"),
        2,
        label="planner payload request_image_count",
    )
    _require_equal(
        planner_payload.get("rendered_prompt_present"),
        True,
        label="planner payload rendered_prompt_present",
    )

    planner_contract = fullstack.get("planner_context_contract")
    if not isinstance(planner_contract, dict):
        raise EvidenceBindingError("full-stack report lacks planner_context_contract")
    _require_equal(
        planner_contract.get("expected_prompt_mode"),
        after["planner_prompt_mode"],
        label="planner_context_contract expected_prompt_mode",
    )
    _require_equal(
        planner_contract.get("expected_max_output_tokens"),
        after["planner_max_output_tokens"],
        label="planner_context_contract expected_max_output_tokens",
    )

    planner_probe = probes_by_mode["planner"]
    _require_equal(
        planner_probe.get("request_image_count"),
        2,
        label="planner probe request_image_count",
    )
    planner_request_images = planner_probe.get("request_images")
    if not isinstance(planner_request_images, list) or len(planner_request_images) != 2:
        raise EvidenceBindingError("planner probe request_images must contain head and third")
    for index, (camera, digest) in enumerate(
        (("head", expected_head_image_sha256), ("third", expected_third_image_sha256))
    ):
        image = planner_request_images[index]
        if not isinstance(image, dict):
            raise EvidenceBindingError(f"planner probe request_images[{index}] must be an object")
        _require_equal(image.get("order"), index, label=f"planner request_images[{index}].order")
        _require_equal(image.get("camera"), camera, label=f"planner request_images[{index}].camera")
        _require_equal(image.get("sha256"), digest, label=f"planner request_images[{index}].sha256")
    _require_equal(
        planner_probe.get("planner_task_requested_min_utf8_byte_count"),
        expected_planner_task_min_bytes,
        label="planner probe requested task byte count",
    )
    _require_equal(
        planner_probe.get("planner_task_actual_utf8_byte_count"),
        planner_task_bytes,
        label="planner probe actual task byte count",
    )
    _require_equal(
        planner_probe.get("planner_payload_actual_non_image_byte_count"),
        planner_non_image_bytes,
        label="planner probe non-image byte count",
    )
    _require_equal(
        planner_probe.get("planner_payload_actual_request_body_byte_count"),
        planner_request_bytes,
        label="planner probe request body byte count",
    )
    planner_prompt_bytes = _require_int_at_least(
        planner_probe.get("planner_rendered_prompt_byte_count"),
        planner_task_bytes,
        label="planner probe rendered prompt byte count",
    )
    _require_int_at_least(
        planner_probe.get("planner_diagnostic_context_record_count"),
        1,
        label="planner probe diagnostic context record count",
    )
    if planner_non_image_bytes <= planner_task_bytes + planner_prompt_bytes:
        raise EvidenceBindingError(
            "planner non-image payload must contain both task and rendered prompt"
        )
    if planner_request_bytes <= planner_non_image_bytes:
        raise EvidenceBindingError(
            "planner request body must be larger than its image-scrubbed payload"
        )
    _require_equal(
        planner_probe.get("planner_rendered_prompt_present"),
        True,
        label="planner probe rendered prompt present",
    )

    binding = {
        "schema_version": 3,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed",
        "model": expected_model,
        "engine_base_url": expected_engine_base_url.rstrip("/"),
        "gateway_base_url": expected_gateway_base_url.rstrip("/"),
        "source_images": {
            "head_sha256": expected_head_image_sha256,
            "third_sha256": expected_third_image_sha256,
        },
        "gateway_service_identity": {
            "sha256": gateway_identity_sha256,
            "health_before_path": str(before_path),
            "health_after_path": str(after_path),
            "unchanged_during_preflights": True,
            "serving_framework": serving_framework,
            "planner_prompt_mode": after["planner_prompt_mode"],
            "planner_max_output_tokens": after["planner_max_output_tokens"],
            "upstream_models_response_sha256": upstream_models_sha256,
            "modality_models_response_sha256": modality_models_response_sha256,
            "upstream_models_identity_sha256": upstream_models_identity_sha256,
        },
        "serving_runtime_identity": {
            "path": str(runtime_path),
            "sha256": runtime_sha256,
        },
        "modality_preflight": {
            "path": str(modality_path),
            "sha256": _sha256(modality_raw),
            "passed_modes": sorted(modality_modes),
        },
        "agent_fullstack_preflight": {
            "path": str(fullstack_path),
            "sha256": _sha256(fullstack_raw),
            "passed_modes": sorted(passed_modes),
            "planner_task_requested_min_utf8_byte_count": expected_planner_task_min_bytes,
            "planner_task_actual_utf8_byte_count": planner_task_bytes,
            "planner_rendered_prompt_present": True,
            "planner_request_image_count": 2,
        },
        "fallback_count": 0,
    }
    return binding, runtime_path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise EvidenceBindingError(f"output path must be absolute: {path}")
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binding, runtime_path = build_binding(
        gateway_health_before_path=args.gateway_health_before,
        gateway_health_after_path=args.gateway_health_after,
        modality_report_path=args.modality_report,
        fullstack_report_path=args.fullstack_report,
        expected_model=args.expected_model,
        expected_engine_base_url=args.expected_engine_base_url,
        expected_gateway_base_url=args.expected_gateway_base_url,
        expected_head_image_sha256=args.expected_head_image_sha256,
        expected_third_image_sha256=args.expected_third_image_sha256,
        expected_planner_task_min_bytes=args.expected_planner_task_min_bytes,
        expected_planner_max_output_tokens=args.expected_planner_max_output_tokens,
    )
    _write_json_atomic(args.output, binding)
    print(
        json.dumps(
            {
                "binding": str(args.output.resolve()),
                "gateway_service_identity_sha256": binding["gateway_service_identity"][
                    "sha256"
                ],
                "serving_runtime_identity": str(runtime_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
