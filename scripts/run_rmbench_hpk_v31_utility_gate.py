#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.summarize_rmbench_hpk_v31_usage import summarize  # noqa: E402


RUN_PLAN_SCHEMA = "roboharn_evo/rmbench/v31/utility_gate_run_plan"
RUN_RESULT_SCHEMA = "roboharn_evo/rmbench/v31/utility_gate_run_result"
CONDITIONS = ("c0", "c1", "c2")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class UtilityGateRunError(RuntimeError):
    """The preregistered cell cannot be executed exactly once."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UtilityGateRunError(f"cannot read JSON object {path}") from exc
    if not isinstance(value, dict):
        raise UtilityGateRunError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise UtilityGateRunError(f"{label} must be an explicit path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise UtilityGateRunError(f"{label} does not exist: {candidate}") from exc


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) != 1:
        raise UtilityGateRunError(
            f"{root} must contain exactly one {pattern}; found {len(matches)}"
        )
    return matches[0]


def _safe_name(value: str) -> str:
    result = _SAFE_NAME.sub("_", str(value).strip()).strip("_.-")
    if not result:
        raise UtilityGateRunError("boundary ID cannot form a launcher name")
    return result[:120]


def _strict_cell(value: Any, *, base: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise UtilityGateRunError("each run-plan cell must be an object")
    expected = {
        "boundary_id",
        "condition",
        "seed",
        "cell_root",
        "overrides",
    }
    if set(value) != expected:
        raise UtilityGateRunError("run-plan cell fields mismatch")
    boundary_id = str(value["boundary_id"] or "").strip()
    condition = str(value["condition"] or "").strip().lower()
    seed = value["seed"]
    overrides = value["overrides"]
    if not boundary_id or len(boundary_id) > 200:
        raise UtilityGateRunError("boundary_id must be bounded and non-empty")
    if condition not in CONDITIONS:
        raise UtilityGateRunError(f"unsupported condition: {condition}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UtilityGateRunError("cell seed must be a non-negative integer")
    if (
        not isinstance(overrides, list)
        or not overrides
        or any(not isinstance(item, str) or not item for item in overrides)
    ):
        raise UtilityGateRunError("cell overrides must be non-empty strings")
    return {
        "boundary_id": boundary_id,
        "condition": condition,
        "seed": seed,
        "cell_root": _path(value["cell_root"], base=base, label="cell_root"),
        "overrides": list(overrides),
    }


def load_run_plan(path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    source = path.resolve(strict=True)
    payload = _read_json(source)
    if payload.get("schema") not in {RUN_PLAN_SCHEMA, "tcm/rmbench/v31/utility_gate_run_plan"}:
        raise UtilityGateRunError("run-plan schema mismatch")
    if payload.get("automatic_retry") is not False:
        raise UtilityGateRunError("run plan must explicitly disable retries")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise UtilityGateRunError("run plan must contain cells")
    typed = tuple(_strict_cell(value, base=source.parent) for value in cells)
    if payload.get("cell_count") != len(typed):
        raise UtilityGateRunError("run-plan cell_count mismatch")
    if payload.get("conditions") != list(CONDITIONS):
        raise UtilityGateRunError("run-plan condition order mismatch")
    contract = payload.get("experiment_contract")
    expected_contract = {
        "task",
        "shared_core_baseline",
        "runtime_commit",
        "store_root",
        "agent_model",
        "reasoning_effort",
        "decoding",
        "task_config",
        "instruction_set",
        "grasp_transport_policy",
        "environment_action_budget",
        "control_turn_budget",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected_contract:
        raise UtilityGateRunError("run-plan experiment contract fields mismatch")
    frozen = dict(contract)
    for field in (
        "task",
        "shared_core_baseline",
        "runtime_commit",
        "store_root",
        "agent_model",
        "reasoning_effort",
        "task_config",
        "instruction_set",
        "grasp_transport_policy",
    ):
        if not isinstance(frozen[field], str) or not frozen[field].strip():
            raise UtilityGateRunError(f"experiment_contract.{field} is empty")
    if frozen["shared_core_baseline"] != "8276000":
        raise UtilityGateRunError("shared-core baseline must be 8276000")
    if frozen["grasp_transport_policy"] != "strict":
        raise UtilityGateRunError("utility gate requires strict grasp verification")
    if not isinstance(frozen["decoding"], Mapping) or not frozen["decoding"]:
        raise UtilityGateRunError("experiment decoding must be non-empty")
    for field in ("environment_action_budget", "control_turn_budget"):
        value = frozen[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise UtilityGateRunError(f"experiment_contract.{field} must be positive")
    identities = [(cell["boundary_id"], cell["condition"]) for cell in typed]
    if len(set(identities)) != len(identities):
        raise UtilityGateRunError("run plan contains duplicate cells")
    return frozen, typed


def _audit_window(path: Path, start: int) -> bytes:
    size = path.stat().st_size
    if size < start:
        raise UtilityGateRunError("Agent service audit log was truncated during cell")
    with path.open("rb") as handle:
        handle.seek(start)
        return handle.read()


def _assert_cell_unused(root: Path) -> None:
    forbidden = (
        "runner_result.json",
        "usage.json",
        "episode_*_agent_trace.jsonl",
        "failure_boundary_result.json",
    )
    for pattern in forbidden:
        if any(path.is_file() for path in root.rglob(pattern)):
            raise UtilityGateRunError(
                f"cell already has execution output ({pattern}); no automatic rerun"
            )


def _store_bytes(root_value: Any) -> dict[str, bytes]:
    root = Path(str(root_value)).expanduser().resolve(strict=True)
    result: dict[str, bytes] = {}
    for name in ("task_knowledge.jsonl", "action_knowledge.jsonl"):
        path = root / name
        try:
            result[name] = path.read_bytes()
        except OSError as exc:
            raise UtilityGateRunError(
                f"cannot read frozen Store member {path}"
            ) from exc
    return result


def _launcher_environment(
    *,
    cell: Mapping[str, Any],
    contract: Mapping[str, Any],
    assets_root: Path,
    agent_api_base_url: str,
    sam3_service_url: str,
    gpu: str,
) -> dict[str, str]:
    root = Path(cell["cell_root"])
    boundary_name = _safe_name(str(cell["boundary_id"]))
    condition = str(cell["condition"])
    short_cache_root = Path("/tmp") / (
        f"av31-{os.getpid()}-{boundary_name[:24]}-{condition}"
    )
    short_temp_root = short_cache_root / "tmp"
    result = dict(os.environ)
    result.update(
        {
            "NUM_WORKERS": "1",
            "GPU_IDS": "0",
            "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN": str(gpu),
            "TASK_NAME": str(contract["task"]),
            "TASK_CONFIG": str(contract["task_config"]),
            "INSTRUCTION_SET": str(contract["instruction_set"]),
            "N_PER_WORKER": "1",
            "EVAL_START_SEEDS": str(cell["seed"]),
            "REQUIRE_EXPLICIT_EVAL_START_SEEDS": "1",
            "BASE_CKPT": f"hpk_v31_gate_{boundary_name}_{condition}",
            "RUN_STAMP": "single_continuation",
            "RMBENCH_OUTPUT_ROOT": str(root / "run_output"),
            "LOG_ROOT": str(root / "launcher"),
            "SEGMENTATION_ARTIFACT_ROOT": str(root / "segmentation_artifacts"),
            "RMBENCH_ASSETS_ROOT": str(assets_root),
            "AGENT_API_BASE_URL": agent_api_base_url.rstrip("/"),
            "SAM3_SERVICE_URL": sam3_service_url.rstrip("/"),
            # NVRTC rejects long temporary paths.  Keep compilation/cache
            # scratch in a short process-scoped /tmp path while all durable
            # experiment outputs remain under the cell root.
            "RUNTIME_CACHE_ROOT": str(short_cache_root),
            "TMPDIR": str(short_temp_root),
            "TMP": str(short_temp_root),
            "TEMP": str(short_temp_root),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "NON_FORMAL_DIAGNOSTIC": "1",
            "PERCEPTION_CONDITION": "no_oracle",
            "RETRY_BUDGET": "0",
            "BACKEND_ERROR_BUDGET": "0",
            "MAX_CONTROL_TURNS": str(contract["control_turn_budget"]),
            "MAX_NO_PROGRESS_CONTROL_TURNS": "10",
            "MAX_ROUNDS": "10",
            "EXPECTED_AGENT_MODEL": str(contract["agent_model"]),
            "EXPECTED_REASONING_EFFORT": str(contract["reasoning_effort"]),
            "EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS": "1",
            # Health/identity and SAM3 checks still run.  The inference
            # preflight is disabled because it is an extra model call outside
            # the one-cell continuation and would contaminate usage accounting.
            "REQUIRE_AGENT_INFERENCE_PREFLIGHT": "0",
            "REQUIRE_SAM3_PREFLIGHT": "1",
            "RECORD_RUNTIME_PROVENANCE": "1",
            "DETACH": "0",
        }
    )
    return result


def run_cell(
    cell: Mapping[str, Any],
    *,
    launcher: Path,
    service_audit_log: Path,
    contract: Mapping[str, Any],
    assets_root: Path,
    agent_api_base_url: str,
    sam3_service_url: str,
    gpu: str,
) -> dict[str, Any]:
    root = Path(cell["cell_root"])
    _assert_cell_unused(root)
    store_before = _store_bytes(contract["store_root"])
    before = service_audit_log.stat().st_size
    environment = _launcher_environment(
        cell=cell,
        contract=contract,
        assets_root=assets_root,
        agent_api_base_url=agent_api_base_url,
        sam3_service_url=sam3_service_url,
        gpu=gpu,
    )
    command = [
        str(launcher),
        "--eval.exact_seed_fail_closed",
        "True",
        "--eval.step_limit",
        str(contract["environment_action_budget"]),
        *cell["overrides"],
    ]
    completed = subprocess.run(
        command,
        cwd=launcher.parent.parent.parent,
        env=environment,
        check=False,
    )
    audit_path = root / "agent_api_audit_window.log"
    audit_path.write_bytes(_audit_window(service_audit_log, before))
    result: dict[str, Any] = {
        "schema": RUN_RESULT_SCHEMA,
        "boundary_id": cell["boundary_id"],
        "condition": cell["condition"],
        "launcher_exit_code": completed.returncode,
        "command": command,
        "automatic_retry": False,
        "service_audit_window": str(audit_path),
    }
    store_after = _store_bytes(contract["store_root"])
    store_unchanged = store_before == store_after
    result["store_bytes_unchanged"] = store_unchanged
    result["store_member_sizes"] = {
        name: len(value) for name, value in sorted(store_after.items())
    }
    if not store_unchanged:
        result["status"] = "failed"
        _write_json(root / "runner_result.json", result)
        raise UtilityGateRunError("frozen Task/Action Store changed during cell")
    if completed.returncode != 0:
        result["status"] = "failed"
        _write_json(root / "runner_result.json", result)
        raise UtilityGateRunError(
            f"cell {cell['boundary_id']}/{cell['condition']} failed with "
            f"exit {completed.returncode}; no retry was attempted"
        )
    trace = _one(root / "run_output", "episode_*_agent_trace.jsonl")
    _one(root / "run_output", "failure_boundary_result.json")
    usage = summarize(
        audit_path,
        trace,
        expected_model=str(contract["agent_model"]),
        expected_reasoning_effort=str(contract["reasoning_effort"]),
    )
    _write_json(root / "usage.json", usage)
    result.update(
        {
            "status": "completed",
            "trace_path": str(trace),
            "usage_path": str(root / "usage.json"),
        }
    )
    _write_json(root / "runner_result.json", result)
    return result


def run_plan(
    plan_path: Path,
    *,
    launcher: Path,
    service_audit_log: Path,
    assets_root: Path,
    agent_api_base_url: str,
    sam3_service_url: str,
    gpu: str,
    boundary_id: str | None = None,
    condition: str | None = None,
) -> list[dict[str, Any]]:
    contract, cells = load_run_plan(plan_path)
    selected = [
        cell
        for cell in cells
        if (boundary_id is None or cell["boundary_id"] == boundary_id)
        and (condition is None or cell["condition"] == condition)
    ]
    if not selected:
        raise UtilityGateRunError("cell selection is empty")
    return [
        run_cell(
            cell,
            launcher=launcher,
            service_audit_log=service_audit_log,
            contract=contract,
            assets_root=assets_root,
            agent_api_base_url=agent_api_base_url,
            sam3_service_url=sam3_service_url,
            gpu=gpu,
        )
        for cell in selected
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--service-audit-log", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--agent-api-base-url", required=True)
    parser.add_argument("--sam3-service-url", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--boundary-id")
    parser.add_argument("--condition", choices=CONDITIONS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    launcher = args.launcher.resolve(strict=True)
    if not os.access(launcher, os.X_OK):
        raise UtilityGateRunError("launcher must be executable")
    results = run_plan(
        args.run_plan,
        launcher=launcher,
        service_audit_log=args.service_audit_log.resolve(strict=True),
        assets_root=args.assets_root.resolve(strict=True),
        agent_api_base_url=str(args.agent_api_base_url).strip(),
        sam3_service_url=str(args.sam3_service_url).strip(),
        gpu=str(args.gpu).strip(),
        boundary_id=args.boundary_id,
        condition=args.condition,
    )
    print(json.dumps(results, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UtilityGateRunError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
