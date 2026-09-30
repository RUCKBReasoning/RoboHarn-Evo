#!/usr/bin/env python3
"""Record RMBench runtime provenance from source content, without Git identity."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any
from urllib.parse import urlsplit


RUNTIME_PROVENANCE_SCHEMA = "roboharn_evo/rmbench_runtime_provenance/content/v2"
RUNTIME_PROVENANCE_SCHEMA_VERSION = 2
SOURCE_IDENTITY_MODE = "content"

# 记录 Runtime、benchmark 入口、启动程序与实验输入；路径相对于 --repo-root。
DEFAULT_RUNTIME_PATHS = (
    "roboharn_evo/__init__.py",
    "roboharn_evo/agent",
    "roboharn_evo/models",
    "roboharn_evo/resources",
    "roboharn_evo/utils",
    "benchmarks/rmbench/__init__.py",
    "benchmarks/rmbench/integration.py",
    "benchmarks/rmbench/paths.py",
    "benchmarks/rmbench/runtime_assets.py",
    "benchmarks/rmbench/envs",
    "benchmarks/rmbench/task_config",
    "benchmarks/rmbench/description/task_instruction",
    "benchmarks/rmbench/description/task_instruction_sets",
    "benchmarks/rmbench/description/utils",
    "benchmarks/rmbench/policy/__init__.py",
    "benchmarks/rmbench/policy/roboharn_evo/__init__.py",
    "benchmarks/rmbench/policy/roboharn_evo/deploy_policy.py",
    "benchmarks/rmbench/policy/roboharn_evo/deploy_policy.yml",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/launch_gpt55_formal4x20_from_seed_manifest.sh",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/run_gpt55_parallel4_isolated.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/run_gpt55_six_tasks_sequential.sh",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/run_gpt55_pure_tool_control_8way.sh",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/check_pure_tool_control_early_stop.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/serve_openai_planner.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/serve_qwen_planner.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/openai_responses_compat.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/codex_app_server_client.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/local_models_identity.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/serve_sam3_segmentation.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/run_gpt55_object_info_ablation_8way.sh",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/summarize_object_info_ablation.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/record_runtime_provenance.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/record_local_model_identity.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/record_local_serving_runtime_identity.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/record_agent_contract_manifest.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/preflight_agent_fullstack.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/preflight_openai_vlm_modalities.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/bind_local_fullstack_preflight_evidence.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/record_gpu_memory_gate.py",
    "benchmarks/rmbench/policy/roboharn_evo/scripts/run_local_qwen35_fullstack_services.sh",
    "benchmarks/rmbench/script/eval_policy.py",
    "benchmarks/rmbench/script/test_render.py",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_FORBIDDEN_GIT_IDENTITY_FIELDS = frozenset(
    {
        "git_head",
        "git_dirty_for_runtime_paths",
        "git_status_short",
        "tracked_runtime_diff_sha256",
    }
)

# Only these environment values are allowed to influence the recorded
# effective configuration.  In particular, no generic environment dump and
# no credential-bearing variable is ever serialized.
EFFECTIVE_CONFIG_ENV_KEYS = (
    "TASK_NAME",
    "TASK_CONFIG",
    "INSTRUCTION_SET",
    "POLICY_NAME",
    "PERCEPTION_CONDITION",
    "EVAL_START_SEEDS",
    "NUM_WORKERS",
    "GPU_IDS",
    "AGENT_API_BASE_URL",
    "SAM3_SERVICE_URL",
    "EXPECTED_AGENT_MODEL",
    "EXPECTED_AGENT_API_MODE",
    "EXPECTED_REASONING_EFFORT",
    "EXPECTED_RESPONSE_STORAGE",
    "EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS",
    "MAX_ROUNDS",
    "MAX_CONTROL_TURNS",
    "MAX_NO_PROGRESS_CONTROL_TURNS",
    "EVAL_STEP_LIMIT",
    "MAX_OBJECTS",
    "N_PER_WORKER",
    "SEED_OFFSET",
    "NON_FORMAL_DIAGNOSTIC",
    "ROBOHARN_EVO_FORMAL_PROTOCOL",
    "ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION",
    "REQUIRE_EXPLICIT_EVAL_START_SEEDS",
    "REQUIRE_SAM3_PREFLIGHT",
    "SKIP_PREFLIGHT",
    "SKIP_AGENT_IDENTITY_CHECK",
    "REQUIRE_AGENT_INFERENCE_PREFLIGHT",
    "RECORD_RUNTIME_PROVENANCE",
    "PREFLIGHT_ONLY",
    "PLANNER_TIMEOUT_SEC",
    "RECOVERY_TIMEOUT_SEC",
    "OOD_TIMEOUT_SEC",
    "QUERY_TIMEOUT_SEC",
    "PREFLIGHT_TIMEOUT_SEC",
    "AGENT_INFERENCE_PREFLIGHT_TIMEOUT_SEC",
    "MIN_AGENT_TIMEOUT_SEC",
    "WORKER_START_DELAY_SEC",
    "MAX_WAIT_STEPS",
    "RETRY_BUDGET",
    "BACKEND_ERROR_BUDGET",
    "EMPTY_PLAN_REPLAN_THRESHOLD",
    "SHUTDOWN_GRACE_SEC",
    "HEARTBEAT_SEC",
    "INTERRUPT_ESCALATION_POLICY",
)

_EFFECTIVE_CONFIG_FIELDS = frozenset(
    {
        "task_name",
        "task_config",
        "instruction_set",
        "policy_name",
        "perception_condition",
        "eval_start_seed",
        "eval_start_seeds",
        "num_workers",
        "gpu_ids",
        "agent_api_base_url",
        "sam3_service_url",
        "expected_agent_model",
        "expected_agent_api_mode",
        "expected_reasoning_effort",
        "expected_response_storage",
        "expected_agent_max_concurrent_requests",
        "max_rounds",
        "max_control_turns",
        "max_no_progress_control_turns",
        "eval_step_limit",
        "max_objects",
        "n_per_worker",
        "seed_offset",
        "pure_tool_control",
        "oracle_objects_enabled",
        "non_formal_diagnostic",
        "formal_protocol",
        "formal_protocol_version",
        "require_explicit_eval_start_seeds",
        "require_sam3_preflight",
        "skip_preflight",
        "skip_agent_identity_check",
        "require_agent_inference_preflight",
        "record_runtime_provenance",
        "preflight_only",
        "planner_timeout_sec",
        "recovery_timeout_sec",
        "ood_timeout_sec",
        "query_timeout_sec",
        "preflight_timeout_sec",
        "agent_inference_preflight_timeout_sec",
        "min_agent_timeout_sec",
        "worker_start_delay_sec",
        "max_wait_steps",
        "retry_budget",
        "backend_error_budget",
        "empty_plan_replan_threshold",
        "shutdown_grace_sec",
        "heartbeat_sec",
        "interrupt_escalation_policy",
    }
)

_REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "recorded_at_utc",
        "repository_root",
        "source_identity_mode",
        "runtime_content_manifest",
        "runtime_tree_sha256",
        "runtime_file_count",
        "runtime_paths",
        "runtime_file_hashes_sha256",
        "effective_config",
        "effective_config_sha256",
        "secrets_recorded",
        "note",
    }
)
_OPTIONAL_PROVENANCE_FIELDS = frozenset(
    {
        "agent_service_identity",
        "agent_service_identity_sha256",
        "agent_service_identity_hash_mode",
        "agent_service_identity_bytes_sha256",
        "agent_service_identity_canonical_sha256",
        "external_evidence",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a content-addressed snapshot of RoboHarn-Evo runtime sources without "
            "copying source text or depending on Git metadata."
        )
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--service-identity",
        type=Path,
        default=None,
        help=(
            "Optional JSON health/identity snapshot for the Agent model service. "
            "The parsed object and the SHA256 of its exact bytes are embedded in "
            "the runtime provenance manifest."
        ),
    )
    parser.add_argument(
        "--external-evidence",
        action="append",
        default=[],
        metavar="LABEL=ABS_PATH",
        help=(
            "Optional content-addressed JSON evidence. May be repeated. Paths must "
            "be absolute; labels must be unique. Parsed JSON and the SHA256 of the "
            "file's exact bytes are embedded in the runtime provenance manifest."
        ),
    )
    parser.add_argument(
        "--path",
        dest="paths",
        action="append",
        default=[],
        help="RoboHarn-Evo-project-relative runtime file/directory. Defaults to the formal runtime set.",
    )
    return parser.parse_args()


def _is_ignored(path: Path) -> bool:
    return any(part in {"__pycache__", ".pytest_cache"} for part in path.parts) or path.suffix in {
        ".pyc",
        ".pyo",
    }


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _runtime_input(repo_root: Path, relative: str | os.PathLike[str]) -> Path:
    root = repo_root.resolve()
    raw = os.fspath(relative)
    if not raw:
        raise ValueError("runtime provenance path must not be empty")
    declared = Path(raw)
    if declared.is_absolute():
        raise ValueError(f"runtime provenance path must be repository-relative: {raw}")
    candidate = (root / declared).resolve()
    if not _inside(candidate, root):
        raise ValueError(f"runtime provenance path escapes repository: {raw}")
    if not candidate.exists():
        raise FileNotFoundError(f"runtime provenance path does not exist: {raw}")
    return candidate


def runtime_files(
    repo_root: Path,
    relative_paths: Iterable[str | os.PathLike[str]],
) -> list[Path]:
    root = repo_root.resolve()
    files: dict[str, Path] = {}
    for relative in relative_paths:
        candidate = _runtime_input(root, relative)
        discovered = [candidate] if candidate.is_file() else candidate.rglob("*")
        for path in discovered:
            if not path.is_file() or _is_ignored(path):
                continue
            resolved = path.resolve()
            if not _inside(resolved, root):
                raise ValueError(
                    f"runtime provenance file escapes repository: {path}"
                )
            relative_text = resolved.relative_to(root).as_posix()
            files[relative_text] = resolved
    return [files[key] for key in sorted(files)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance value must be finite JSON data") from exc
    return encoded.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def decode_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    """Decode one finite JSON object while rejecting duplicate keys."""

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        decoded: dict[str, object] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            decoded[key] = value
        return decoded

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-standard JSON constant: {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonstandard_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    canonical_json_bytes(payload)
    return payload


def _effective_text(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str | None = None,
) -> str:
    raw = environ.get(name, default)
    if raw is None:
        raise ValueError(f"formal runtime environment is missing {name}")
    value = str(raw).strip()
    if not value or len(value) > 4096 or any(ord(character) < 32 for character in value):
        raise ValueError(f"formal runtime environment has invalid {name}")
    return value


def _effective_optional_text(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str = "",
) -> str:
    value = str(environ.get(name, default))
    if len(value) > 4096 or any(ord(character) < 32 for character in value):
        raise ValueError(f"formal runtime environment has invalid {name}")
    return value.strip()


def _effective_int(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str | None = None,
    minimum: int = 0,
) -> int:
    raw = _effective_text(environ, name, default=default)
    if not re.fullmatch(r"-?[0-9]+", raw):
        raise ValueError(f"formal runtime environment {name} must be an integer")
    value = int(raw, 10)
    if value < minimum:
        raise ValueError(f"formal runtime environment {name} must be >= {minimum}")
    return value


def _effective_float(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str,
    minimum: float = 0.0,
) -> float:
    raw = _effective_text(environ, name, default=default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"formal runtime environment {name} must be a finite number"
        ) from exc
    if not (value >= minimum and value < float("inf")):
        raise ValueError(
            f"formal runtime environment {name} must be finite and >= {minimum}"
        )
    return value


def _effective_bool(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str,
) -> bool:
    raw = _effective_text(environ, name, default=default)
    if raw not in {"0", "1"}:
        raise ValueError(f"formal runtime environment {name} must be 0 or 1")
    return raw == "1"


def _effective_int_csv(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str | None = None,
    allow_empty: bool = False,
) -> list[int]:
    raw = _effective_optional_text(
        environ,
        name,
        default="" if default is None else default,
    )
    if not raw and allow_empty:
        return []
    if not raw:
        raise ValueError(f"formal runtime environment is missing {name}")
    pieces = raw.split(",")
    if any(not piece or piece.strip() != piece for piece in pieces):
        raise ValueError(
            f"formal runtime environment {name} must be a canonical integer CSV"
        )
    values: list[int] = []
    for piece in pieces:
        if not re.fullmatch(r"[0-9]+", piece):
            raise ValueError(
                f"formal runtime environment {name} must be a canonical integer CSV"
            )
        values.append(int(piece, 10))
    if len(set(values)) != len(values):
        raise ValueError(f"formal runtime environment {name} contains duplicates")
    return values


def _effective_service_url(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str,
) -> str:
    value = _effective_text(environ, name, default=default).rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"formal runtime environment {name} must be an HTTP(S) URL without credentials"
        )
    return value


def _validate_effective_config(config: object) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("effective_config must be a JSON object")
    if set(config) != _EFFECTIVE_CONFIG_FIELDS:
        missing = sorted(_EFFECTIVE_CONFIG_FIELDS - set(config))
        extra = sorted(set(config) - _EFFECTIVE_CONFIG_FIELDS)
        raise ValueError(
            "effective_config fields do not match the strict whitelist; "
            f"missing={missing}, extra={extra}"
        )
    for key in (
        "task_name",
        "task_config",
        "instruction_set",
        "policy_name",
        "agent_api_base_url",
        "sam3_service_url",
        "expected_agent_model",
    ):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"effective_config {key} must be a non-empty string")
    for key in (
        "perception_condition",
        "expected_agent_api_mode",
        "expected_reasoning_effort",
        "expected_response_storage",
    ):
        value = config[key]
        if (
            not isinstance(value, str)
            or len(value) > 4096
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError(f"effective_config {key} must be a safe string")
    if config["perception_condition"] not in {"", "no_oracle", "oracle"}:
        raise ValueError("effective_config perception_condition is invalid")
    for key in ("agent_api_base_url", "sam3_service_url"):
        parsed_url = urlsplit(config[key])
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError(
                f"effective_config {key} must be an HTTP(S) URL without credentials"
            )

    integer_minima = {
        "num_workers": 1,
        "expected_agent_max_concurrent_requests": 1,
        "max_rounds": 0,
        "max_control_turns": 0,
        "max_no_progress_control_turns": 0,
        "eval_step_limit": 1,
        "max_objects": 1,
        "n_per_worker": 1,
        "seed_offset": 0,
        "formal_protocol_version": 0,
        "max_wait_steps": 0,
        "retry_budget": 0,
        "backend_error_budget": 0,
        "empty_plan_replan_threshold": 0,
    }
    for key, minimum in integer_minima.items():
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"effective_config {key} must be an integer >= {minimum}")
    eval_start_seeds = config["eval_start_seeds"]
    if (
        not isinstance(eval_start_seeds, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in eval_start_seeds
        )
        or len(set(eval_start_seeds)) != len(eval_start_seeds)
    ):
        raise ValueError(
            "effective_config eval_start_seeds must be unique non-negative integers"
        )
    expected_single_seed = eval_start_seeds[0] if len(eval_start_seeds) == 1 else None
    if config["eval_start_seed"] != expected_single_seed:
        raise ValueError(
            "effective_config eval_start_seed must be the sole start seed or null"
        )
    gpu_ids = config["gpu_ids"]
    if (
        not isinstance(gpu_ids, list)
        or not gpu_ids
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("effective_config gpu_ids must be unique non-negative integers")
    if len(gpu_ids) != config["num_workers"]:
        raise ValueError("effective_config gpu_ids must match num_workers")
    if eval_start_seeds and len(eval_start_seeds) != config["num_workers"]:
        raise ValueError("effective_config eval_start_seeds must match num_workers")

    positive_timeout_fields = (
        "planner_timeout_sec",
        "recovery_timeout_sec",
        "ood_timeout_sec",
        "query_timeout_sec",
        "preflight_timeout_sec",
        "agent_inference_preflight_timeout_sec",
        "min_agent_timeout_sec",
    )
    non_negative_duration_fields = (
        "worker_start_delay_sec",
        "shutdown_grace_sec",
        "heartbeat_sec",
    )
    for key in positive_timeout_fields + non_negative_duration_fields:
        value = config[key]
        minimum = 0.0 if key in non_negative_duration_fields else 0.0
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not (float(value) > minimum if key in positive_timeout_fields else float(value) >= minimum)
            or not float(value) < float("inf")
        ):
            comparison = "> 0" if key in positive_timeout_fields else ">= 0"
            raise ValueError(
                f"effective_config {key} must be a finite number {comparison}"
            )
    if config["interrupt_escalation_policy"] not in {
        "term_then_kill",
        "ctrl_c_only",
    }:
        raise ValueError(
            "effective_config interrupt_escalation_policy is invalid"
        )

    bool_fields = (
        "pure_tool_control",
        "oracle_objects_enabled",
        "non_formal_diagnostic",
        "formal_protocol",
        "require_explicit_eval_start_seeds",
        "require_sam3_preflight",
        "skip_preflight",
        "skip_agent_identity_check",
        "require_agent_inference_preflight",
        "record_runtime_provenance",
        "preflight_only",
    )
    if any(not isinstance(config[key], bool) for key in bool_fields):
        raise ValueError("effective_config formal flags must be booleans")
    if config["pure_tool_control"] is not True:
        raise ValueError("effective_config pure_tool_control must be true")
    if config["formal_protocol"] is config["non_formal_diagnostic"]:
        raise ValueError(
            "effective_config formal_protocol and non_formal_diagnostic disagree"
        )
    if config["oracle_objects_enabled"] is not (
        config["perception_condition"] == "oracle"
    ):
        raise ValueError("effective_config oracle_objects_enabled is inconsistent")
    if config["formal_protocol"]:
        formal_requirements = {
            "policy_name": "policy.roboharn_evo.deploy_policy",
            "formal_protocol_version": 3,
            "max_rounds": 10,
            "max_control_turns": 64,
            "max_no_progress_control_turns": 10,
            "require_explicit_eval_start_seeds": True,
            "skip_preflight": False,
            "skip_agent_identity_check": False,
            "require_agent_inference_preflight": True,
            "record_runtime_provenance": True,
        }
        mismatched = [
            key for key, expected in formal_requirements.items()
            if config[key] != expected
        ]
        if mismatched:
            raise ValueError(
                "effective_config violates the donor formal protocol: "
                + ", ".join(mismatched)
            )
        if not eval_start_seeds:
            raise ValueError("effective_config formal protocol requires start seeds")
        if config["perception_condition"] not in {"no_oracle", "oracle"}:
            raise ValueError(
                "effective_config formal perception_condition is invalid"
            )
        expected_sam_preflight = config["perception_condition"] == "no_oracle"
        if config["require_sam3_preflight"] is not expected_sam_preflight:
            raise ValueError(
                "effective_config require_sam3_preflight is inconsistent"
            )
    elif config["formal_protocol_version"] != 0:
        raise ValueError(
            "effective_config non-formal protocol_version must be zero"
        )
    if config["require_explicit_eval_start_seeds"] and not eval_start_seeds:
        raise ValueError(
            "effective_config requires explicit start seeds but none were recorded"
        )
    _assert_identity_has_no_secret_fields(config, "effective_config")
    return json.loads(canonical_json_bytes(config).decode("utf-8"))


def effective_config_from_environment(
    environ: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    """Build a typed pure-tool runtime config from a secret-free env allowlist."""

    task_name = _effective_text(environ, "TASK_NAME")
    perception = _effective_optional_text(environ, "PERCEPTION_CONDITION")
    seeds = _effective_int_csv(environ, "EVAL_START_SEEDS", allow_empty=True)
    num_workers = _effective_int(environ, "NUM_WORKERS", default="1", minimum=1)
    gpu_ids = _effective_int_csv(environ, "GPU_IDS")
    non_formal = _effective_bool(environ, "NON_FORMAL_DIAGNOSTIC", default="0")
    formal_protocol = _effective_bool(
        environ,
        "ROBOHARN_EVO_FORMAL_PROTOCOL",
        default="0" if non_formal else "1",
    )
    formal_version = _effective_int(
        environ,
        "ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION",
        default="3" if formal_protocol else "0",
        minimum=0,
    )
    config = {
        "task_name": task_name,
        "task_config": _effective_text(environ, "TASK_CONFIG", default="demo_clean"),
        "instruction_set": _effective_text(
            environ, "INSTRUCTION_SET", default="rmbench_original"
        ),
        "policy_name": _effective_text(
            environ, "POLICY_NAME", default="policy.roboharn_evo.deploy_policy"
        ),
        "perception_condition": perception,
        "eval_start_seed": seeds[0] if len(seeds) == 1 else None,
        "eval_start_seeds": seeds,
        "num_workers": num_workers,
        "gpu_ids": gpu_ids,
        "agent_api_base_url": _effective_service_url(
            environ, "AGENT_API_BASE_URL", default="http://127.0.0.1:9104"
        ),
        "sam3_service_url": _effective_service_url(
            environ, "SAM3_SERVICE_URL", default="http://127.0.0.1:9301"
        ),
        "expected_agent_model": _effective_text(
            environ, "EXPECTED_AGENT_MODEL", default="gpt-5.5"
        ),
        "expected_agent_api_mode": _effective_optional_text(
            environ, "EXPECTED_AGENT_API_MODE", default="responses_compat"
        ),
        "expected_reasoning_effort": _effective_optional_text(
            environ, "EXPECTED_REASONING_EFFORT", default="xhigh"
        ),
        "expected_response_storage": _effective_optional_text(
            environ, "EXPECTED_RESPONSE_STORAGE", default="account_default"
        ),
        "expected_agent_max_concurrent_requests": _effective_int(
            environ,
            "EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS",
            default="4",
            minimum=1,
        ),
        "max_rounds": _effective_int(environ, "MAX_ROUNDS", default="10"),
        "max_control_turns": _effective_int(
            environ, "MAX_CONTROL_TURNS", default="64"
        ),
        "max_no_progress_control_turns": _effective_int(
            environ, "MAX_NO_PROGRESS_CONTROL_TURNS", default="10"
        ),
        "eval_step_limit": _effective_int(
            environ, "EVAL_STEP_LIMIT", default="150", minimum=1
        ),
        "max_objects": _effective_int(environ, "MAX_OBJECTS", default="8", minimum=1),
        "n_per_worker": _effective_int(environ, "N_PER_WORKER", default="1", minimum=1),
        "seed_offset": _effective_int(environ, "SEED_OFFSET", default="0"),
        "pure_tool_control": True,
        "oracle_objects_enabled": perception == "oracle",
        "non_formal_diagnostic": non_formal,
        "formal_protocol": formal_protocol,
        "formal_protocol_version": formal_version,
        "require_explicit_eval_start_seeds": _effective_bool(
            environ, "REQUIRE_EXPLICIT_EVAL_START_SEEDS", default="1"
        ),
        "require_sam3_preflight": _effective_bool(
            environ, "REQUIRE_SAM3_PREFLIGHT", default="1"
        ),
        "skip_preflight": _effective_bool(environ, "SKIP_PREFLIGHT", default="0"),
        "skip_agent_identity_check": _effective_bool(
            environ, "SKIP_AGENT_IDENTITY_CHECK", default="0"
        ),
        "require_agent_inference_preflight": _effective_bool(
            environ, "REQUIRE_AGENT_INFERENCE_PREFLIGHT", default="1"
        ),
        "record_runtime_provenance": _effective_bool(
            environ, "RECORD_RUNTIME_PROVENANCE", default="1"
        ),
        "preflight_only": _effective_bool(
            environ, "PREFLIGHT_ONLY", default="0"
        ),
        "planner_timeout_sec": _effective_float(
            environ, "PLANNER_TIMEOUT_SEC", default="1800", minimum=0.0
        ),
        "recovery_timeout_sec": _effective_float(
            environ, "RECOVERY_TIMEOUT_SEC", default="1800", minimum=0.0
        ),
        "ood_timeout_sec": _effective_float(
            environ, "OOD_TIMEOUT_SEC", default="300", minimum=0.0
        ),
        "query_timeout_sec": _effective_float(
            environ, "QUERY_TIMEOUT_SEC", default="900", minimum=0.0
        ),
        "preflight_timeout_sec": _effective_float(
            environ, "PREFLIGHT_TIMEOUT_SEC", default="10", minimum=0.0
        ),
        "agent_inference_preflight_timeout_sec": _effective_float(
            environ,
            "AGENT_INFERENCE_PREFLIGHT_TIMEOUT_SEC",
            default="650",
            minimum=0.0,
        ),
        "min_agent_timeout_sec": _effective_float(
            environ, "MIN_AGENT_TIMEOUT_SEC", default="600", minimum=0.0
        ),
        "worker_start_delay_sec": _effective_float(
            environ, "WORKER_START_DELAY_SEC", default="5", minimum=0.0
        ),
        "max_wait_steps": _effective_int(
            environ, "MAX_WAIT_STEPS", default="2"
        ),
        "retry_budget": _effective_int(
            environ, "RETRY_BUDGET", default="1"
        ),
        "backend_error_budget": _effective_int(
            environ, "BACKEND_ERROR_BUDGET", default="5"
        ),
        "empty_plan_replan_threshold": _effective_int(
            environ, "EMPTY_PLAN_REPLAN_THRESHOLD", default="2"
        ),
        "shutdown_grace_sec": _effective_float(
            environ, "SHUTDOWN_GRACE_SEC", default="90", minimum=0.0
        ),
        "heartbeat_sec": _effective_float(
            environ, "HEARTBEAT_SEC", default="60", minimum=0.0
        ),
        "interrupt_escalation_policy": _effective_text(
            environ,
            "INTERRUPT_ESCALATION_POLICY",
            default="term_then_kill",
        ),
    }
    return _validate_effective_config(config)


_SECRET_KEY_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "credential",
    "credentials",
    "client_secret",
    "private_key",
    "session_token",
    "token",
    "access_token",
    "refresh_token",
    "bearer_token",
}

_EVIDENCE_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


def _assert_identity_has_no_secret_fields(value: object, path: str = "identity") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SECRET_KEY_NAMES or normalized.endswith(
                (
                    "_api_key",
                    "_password",
                    "_secret",
                    "_credential",
                    "_credentials",
                    "_client_secret",
                    "_private_key",
                    "_token",
                )
            ):
                raise ValueError(
                    f"service identity contains a forbidden credential-like field: {path}.{key}"
                )
            _assert_identity_has_no_secret_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_identity_has_no_secret_fields(child, f"{path}[{index}]")


def load_service_identity(path: Path) -> tuple[dict, str]:
    identity_path = path.expanduser().resolve()
    raw = identity_path.read_bytes()
    payload = decode_json_object(raw, label=f"service identity {identity_path}")
    _assert_identity_has_no_secret_fields(payload)
    return payload, hashlib.sha256(raw).hexdigest()


def parse_external_evidence_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    """Parse repeatable LABEL=ABS_PATH values without shell-style evaluation."""

    parsed: list[tuple[str, Path]] = []
    seen_labels: set[str] = set()
    for spec in specs:
        label, separator, raw_path = str(spec).partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("external evidence must use the form LABEL=ABS_PATH")
        if not _EVIDENCE_LABEL_RE.fullmatch(label):
            raise ValueError(
                "external evidence label must start with a letter and contain only "
                "letters, digits, '.', '_', or '-'"
            )
        if label in seen_labels:
            raise ValueError(f"duplicate external evidence label: {label}")
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError(
                f"external evidence path must be absolute for {label}: {raw_path}"
            )
        if path.suffix.lower() != ".json":
            raise ValueError(
                f"external evidence must be a .json file for {label}: {raw_path}"
            )
        parsed.append((label, path))
        seen_labels.add(label)
    return parsed


def load_external_evidence(path: Path, *, label: str) -> dict:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"external evidence is not a regular file for {label}: {resolved}")
    raw = resolved.read_bytes()
    payload = decode_json_object(
        raw, label=f"external evidence for {label} at {resolved}"
    )
    _assert_identity_has_no_secret_fields(payload, f"external_evidence.{label}")
    return {
        "path": str(resolved),
        "bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_sha256": canonical_json_sha256(payload),
        "json": payload,
    }


def _content_entries(repo_root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    root = repo_root.resolve()
    entries: list[dict[str, Any]] = []
    for path in files:
        content = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    return entries


def _manifest_relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a canonical repository-relative POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} must be a canonical repository-relative POSIX path")
    return relative


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def validate_runtime_content_manifest(
    manifest: object,
    repo_root: Path,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate and normalize a reusable content manifest.

    When ``verify_files`` is true, every entry is checked against the current
    bytes beneath ``repo_root``.  No Git repository or command is consulted.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("runtime_content_manifest must be a JSON object")
    expected_manifest_fields = {
        "hash_algorithm",
        "root",
        "entries",
        "file_count",
        "aggregate_sha256",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError(
            "runtime_content_manifest fields do not match the strict schema"
        )
    if manifest.get("hash_algorithm") != "sha256":
        raise ValueError("runtime content hash_algorithm must be sha256")
    if manifest.get("root") != ".":
        raise ValueError("runtime content root must be '.'")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("runtime content manifest contains no files")

    root = repo_root.resolve()
    normalized: list[dict[str, Any]] = []
    observed_paths: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"runtime content entry {index} must be a JSON object")
        if set(entry) != {"path", "sha256", "size_bytes"}:
            raise ValueError(
                f"runtime content entry {index} must contain path, sha256, and size_bytes"
            )
        relative = _manifest_relative_path(
            entry.get("path"), label=f"runtime content entry {index} path"
        )
        relative_text = relative.as_posix()
        digest = entry.get("sha256")
        if not _is_sha256(digest):
            raise ValueError(f"runtime content entry {relative_text} has invalid sha256")
        raw_size = entry.get("size_bytes")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
            raise ValueError(f"runtime content entry {relative_text} has invalid size_bytes")

        normalized_entry = {
            "path": relative_text,
            "sha256": str(digest).lower(),
            "size_bytes": raw_size,
        }
        if verify_files:
            runtime_file = (root / relative_text).resolve()
            if not _inside(runtime_file, root) or not runtime_file.is_file():
                raise ValueError(
                    f"runtime content file is absent or outside repository: {relative_text}"
                )
            if runtime_file.relative_to(root).as_posix() != relative_text:
                raise ValueError(
                    f"runtime content path resolves through an alias: {relative_text}"
                )
            content = runtime_file.read_bytes()
            if len(content) != raw_size:
                raise ValueError(f"runtime content size changed for {relative_text}")
            if hashlib.sha256(content).hexdigest() != normalized_entry["sha256"]:
                raise ValueError(f"runtime content hash changed for {relative_text}")
        observed_paths.append(relative_text)
        normalized.append(normalized_entry)

    if observed_paths != sorted(set(observed_paths)):
        raise ValueError("runtime content paths must be unique and sorted")
    if manifest.get("file_count") != len(normalized):
        raise ValueError("runtime content file_count does not match entries")
    aggregate = canonical_json_sha256(normalized)
    if manifest.get("aggregate_sha256") != aggregate:
        raise ValueError("runtime content aggregate_sha256 is invalid")
    return {
        "hash_algorithm": "sha256",
        "root": ".",
        "entries": normalized,
        "file_count": len(normalized),
        "aggregate_sha256": aggregate,
    }


def validate_runtime_provenance(
    payload: object,
    repo_root: Path,
    *,
    verify_files: bool = True,
    verify_external_evidence: bool = True,
    expected_effective_config: Mapping[str, Any] | None = None,
    required_runtime_paths: Iterable[str | os.PathLike[str]] = (),
) -> dict[str, Any]:
    """Validate a content-provenance payload for reuse by launch/eval code.

    Evaluators should pass ``verify_external_evidence=False`` because scheduler
    evidence may legitimately be appended after this manifest is recorded.
    They can bind the live rollout environment by passing
    ``expected_effective_config=effective_config_from_environment()``.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("runtime provenance must be a JSON object")
    # Reject NaN/infinity and non-JSON values before inspecting individual
    # fields.  This also ensures the exact object being archived can be
    # represented canonically.
    canonical_json_bytes(payload)
    forbidden = sorted(
        key
        for key in payload
        if key in _FORBIDDEN_GIT_IDENTITY_FIELDS or str(key).startswith("git_")
    )
    if forbidden:
        raise ValueError(
            "content runtime provenance must not contain Git identity fields: "
            + ", ".join(forbidden)
        )
    observed_fields = set(payload)
    missing_fields = sorted(_REQUIRED_PROVENANCE_FIELDS - observed_fields)
    extra_fields = sorted(
        observed_fields
        - _REQUIRED_PROVENANCE_FIELDS
        - _OPTIONAL_PROVENANCE_FIELDS
    )
    if missing_fields or extra_fields:
        raise ValueError(
            "runtime provenance fields do not match the strict schema; "
            f"missing={missing_fields}, extra={extra_fields}"
        )
    if payload.get("schema") not in {RUNTIME_PROVENANCE_SCHEMA, "tcm/rmbench_runtime_provenance/content/v2"}:
        raise ValueError(f"unsupported runtime provenance schema: {payload.get('schema')!r}")
    if payload.get("schema_version") != RUNTIME_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported runtime provenance schema_version")
    if payload.get("source_identity_mode") != SOURCE_IDENTITY_MODE:
        raise ValueError("runtime provenance source_identity_mode must be content")
    if payload.get("secrets_recorded") is not False:
        raise ValueError("runtime provenance must affirm secrets_recorded=false")
    if not isinstance(payload.get("recorded_at_utc"), str) or not payload[
        "recorded_at_utc"
    ]:
        raise ValueError("runtime provenance recorded_at_utc must be non-empty")
    if not isinstance(payload.get("note"), str) or not payload["note"]:
        raise ValueError("runtime provenance note must be non-empty")
    _assert_identity_has_no_secret_fields(payload, "runtime_provenance")

    root = repo_root.resolve()
    if payload.get("repository_root") != str(root):
        raise ValueError("runtime provenance repository_root does not match repo_root")
    manifest = validate_runtime_content_manifest(
        payload.get("runtime_content_manifest"),
        root,
        verify_files=verify_files,
    )
    file_count = manifest["file_count"]
    aggregate = manifest["aggregate_sha256"]
    if payload.get("runtime_file_count") != file_count:
        raise ValueError("runtime_file_count does not match runtime content entries")
    if payload.get("runtime_tree_sha256") != aggregate:
        raise ValueError("runtime_tree_sha256 does not match content aggregate")

    effective_config = _validate_effective_config(payload.get("effective_config"))
    effective_config_sha256 = canonical_json_sha256(effective_config)
    if payload.get("effective_config_sha256") != effective_config_sha256:
        raise ValueError("effective_config_sha256 does not match effective_config")
    if expected_effective_config is not None:
        expected = _validate_effective_config(expected_effective_config)
        if effective_config != expected:
            raise ValueError("effective_config does not match the current formal environment")

    hashes = payload.get("runtime_file_hashes_sha256")
    expected_hashes = {
        entry["path"]: entry["sha256"] for entry in manifest["entries"]
    }
    if hashes != expected_hashes:
        raise ValueError("runtime_file_hashes_sha256 does not match content entries")

    selected_paths = payload.get("runtime_paths")
    if not isinstance(selected_paths, list) or not selected_paths:
        raise ValueError("runtime_paths must be a non-empty list")
    declared_files = [
        path.relative_to(root).as_posix()
        for path in runtime_files(root, selected_paths)
    ]
    if declared_files != [entry["path"] for entry in manifest["entries"]]:
        raise ValueError("runtime_paths do not select the recorded content entries")
    required_paths = tuple(required_runtime_paths)
    if required_paths:
        required_files = {
            path.relative_to(root).as_posix()
            for path in runtime_files(root, required_paths)
        }
        recorded_files = {entry["path"] for entry in manifest["entries"]}
        missing_required_files = sorted(required_files - recorded_files)
        if missing_required_files:
            raise ValueError(
                "runtime content manifest omits required formal runtime files: "
                + ", ".join(missing_required_files)
            )

    identity_fields = {
        "agent_service_identity",
        "agent_service_identity_sha256",
        "agent_service_identity_hash_mode",
        "agent_service_identity_bytes_sha256",
        "agent_service_identity_canonical_sha256",
    }
    present_identity_fields = observed_fields & identity_fields
    if present_identity_fields and present_identity_fields != identity_fields:
        raise ValueError(
            "agent service identity fields must be present as one complete group"
        )
    identity = payload.get("agent_service_identity")
    identity_hash = payload.get("agent_service_identity_sha256")
    if identity is not None or identity_hash is not None:
        if not isinstance(identity, dict):
            raise ValueError("agent_service_identity must be a JSON object")
        _assert_identity_has_no_secret_fields(identity)
        if not _is_sha256(identity_hash):
            raise ValueError("agent_service_identity_sha256 is invalid")
        if payload.get("agent_service_identity_hash_mode") != "exact_source_json_bytes":
            raise ValueError(
                "agent_service_identity_hash_mode must be exact_source_json_bytes"
            )
        if payload.get("agent_service_identity_bytes_sha256") != identity_hash:
            raise ValueError(
                "agent_service_identity_bytes_sha256 must match the donor-compatible hash"
            )
        canonical_identity_hash = canonical_json_sha256(identity)
        if payload.get("agent_service_identity_canonical_sha256") != canonical_identity_hash:
            raise ValueError(
                "agent_service_identity_canonical_sha256 does not match the embedded identity"
            )

    evidence = payload.get("external_evidence")
    if evidence is not None:
        if not isinstance(evidence, dict) or not evidence:
            raise ValueError("external_evidence must be a non-empty JSON object")
        for label, record in evidence.items():
            if not _EVIDENCE_LABEL_RE.fullmatch(str(label)):
                raise ValueError(f"invalid external evidence label: {label}")
            if not isinstance(record, dict):
                raise ValueError(f"external evidence record must be an object: {label}")
            if set(record) != {
                "path",
                "bytes_sha256",
                "canonical_sha256",
                "json",
            }:
                raise ValueError(
                    f"external evidence record has unexpected fields: {label}"
                )
            path = record.get("path")
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ValueError(f"external evidence path must be absolute: {label}")
            if not _is_sha256(record.get("bytes_sha256")):
                raise ValueError(f"external evidence bytes_sha256 is invalid: {label}")
            evidence_json = record.get("json")
            if not isinstance(evidence_json, dict):
                raise ValueError(f"external evidence JSON must be an object: {label}")
            if record.get("canonical_sha256") != canonical_json_sha256(evidence_json):
                raise ValueError(
                    f"external evidence canonical_sha256 does not match JSON: {label}"
                )
            _assert_identity_has_no_secret_fields(
                evidence_json, f"external_evidence.{label}"
            )
            if verify_external_evidence:
                observed = load_external_evidence(Path(path), label=str(label))
                if observed != record:
                    raise ValueError(f"external evidence content changed: {label}")

    return dict(payload)


def roboharn_project_root(repo_root: Path) -> Path:
    """从项目目录或 RMBench 目录解析 RoboHarn-Evo 根目录。"""

    root = repo_root.expanduser().resolve()
    if root.name == "rmbench" and root.parent.name == "benchmarks":
        project = root.parents[1]
    else:
        project = root
    benchmark = project / "benchmarks" / "rmbench"
    if not benchmark.is_dir():
        raise ValueError(
            "repo_root must be the RoboHarn-Evo project root or copied benchmark root: "
            f"{root}"
        )
    return project


def roboharn_eval_result_root(repo_root: Path) -> Path:
    """解析 RoboHarn-Evo 的 RMBench 结果目录。"""

    return (roboharn_project_root(repo_root) / "eval_result" / "rmbench").resolve()


def roboharn_self_evolution_result_root(repo_root: Path) -> Path:
    """解析 RoboHarn-Evo 的自进化实验结果目录。"""

    return (roboharn_project_root(repo_root) / "eval_result" / "self_evolution").resolve()


def validate_output_path(
    repo_root: Path,
    output: Path,
    *,
    formal_protocol: bool = True,
) -> Path:
    """Return a safe provenance path inside the protocol-owned result roots.

    Formal RMBench provenance remains confined to ``eval_result/rmbench``.
    Non-formal development runs may additionally archive paired self-evolution
    evidence below ``eval_result/self_evolution``.
    """

    output_roots = [roboharn_eval_result_root(repo_root)]
    if not formal_protocol:
        output_roots.append(roboharn_self_evolution_result_root(repo_root))
    resolved = output.expanduser().resolve()
    if not any(
        resolved != output_root and _inside(resolved, output_root)
        for output_root in output_roots
    ):
        allowed = ", ".join(str(output_root) for output_root in output_roots)
        raise ValueError(
            "runtime provenance output must be a file inside an allowed result "
            f"root ({allowed}): {resolved}"
        )
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"runtime provenance output is not a regular file: {resolved}")
    return resolved


def build_runtime_provenance(
    repo_root: Path,
    relative_paths: Iterable[str | os.PathLike[str]],
    service_identity_path: Path | None = None,
    external_evidence_specs: Iterable[str] = (),
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve()
    selected_paths = tuple(os.fspath(path) for path in relative_paths)
    files = runtime_files(root, selected_paths)
    if not files:
        raise ValueError("runtime provenance contains no runtime files")
    entries = _content_entries(root, files)
    aggregate = canonical_json_sha256(entries)
    effective_config = effective_config_from_environment(
        os.environ if environ is None else environ
    )
    payload: dict[str, Any] = {
        "schema": RUNTIME_PROVENANCE_SCHEMA,
        "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repository_root": str(root),
        "source_identity_mode": SOURCE_IDENTITY_MODE,
        "runtime_content_manifest": {
            "hash_algorithm": "sha256",
            "root": ".",
            "entries": entries,
            "file_count": len(entries),
            "aggregate_sha256": aggregate,
        },
        "runtime_tree_sha256": aggregate,
        "runtime_file_count": len(entries),
        "runtime_paths": list(selected_paths),
        "runtime_file_hashes_sha256": {
            entry["path"]: entry["sha256"] for entry in entries
        },
        "effective_config": effective_config,
        "effective_config_sha256": canonical_json_sha256(effective_config),
        "secrets_recorded": False,
        "note": (
            "Source identity is derived only from the declared runtime file paths, "
            "byte sizes, and SHA-256 digests. Source contents and Git metadata are "
            "not copied into the manifest."
        ),
    }
    if service_identity_path is not None:
        identity, identity_sha256 = load_service_identity(service_identity_path)
        payload["agent_service_identity"] = identity
        # Preserve the donor field as the hash of the exact evidence bytes,
        # label that mode explicitly, and add a separately verifiable hash of
        # the embedded parsed object.
        payload["agent_service_identity_sha256"] = identity_sha256
        payload["agent_service_identity_hash_mode"] = "exact_source_json_bytes"
        payload["agent_service_identity_bytes_sha256"] = identity_sha256
        payload["agent_service_identity_canonical_sha256"] = canonical_json_sha256(
            identity
        )
    parsed_external_evidence = parse_external_evidence_specs(external_evidence_specs)
    if parsed_external_evidence:
        payload["external_evidence"] = {
            label: load_external_evidence(path, label=label)
            for label, path in parsed_external_evidence
        }
    validate_runtime_provenance(
        payload,
        root,
        expected_effective_config=effective_config,
    )
    return payload


def write_runtime_provenance(
    repo_root: Path,
    output: Path,
    payload: Mapping[str, Any],
) -> Path:
    """Validate and atomically write provenance inside its protocol-owned root."""

    validated = validate_runtime_provenance(payload, repo_root)
    formal_protocol = validated["effective_config"]["formal_protocol"]
    destination = validate_output_path(
        repo_root,
        output,
        formal_protocol=formal_protocol,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = validate_output_path(
        repo_root,
        destination,
        formal_protocol=formal_protocol,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def main() -> None:
    args = parse_args()
    paths = tuple(args.paths) or DEFAULT_RUNTIME_PATHS
    payload = build_runtime_provenance(
        args.repo_root,
        paths,
        service_identity_path=args.service_identity,
        external_evidence_specs=args.external_evidence,
    )
    output = write_runtime_provenance(args.repo_root, args.output, payload)
    print(
        json.dumps(
            {
                "runtime_provenance": str(output),
                "source_identity_mode": payload["source_identity_mode"],
                "runtime_tree_sha256": payload["runtime_tree_sha256"],
                "runtime_file_count": payload["runtime_file_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
