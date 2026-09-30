"""Result-blind Harness baseline matrix and native-evaluator result collection."""
from __future__ import annotations

import csv
import json
import math
import signal
import shutil
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]


def stop_worker(process):
    """Stop only this owned worker and its descendants, including video/SDK sessions."""
    if process.poll() is not None:
        return
    import psutil
    try:
        root = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        process.wait()
        return
    owned = root.children(recursive=True) + [root]
    root.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    for child in owned:
        try:
            if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                child.send_signal(signal.SIGINT if child.name() == "ffmpeg" else signal.SIGTERM)
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(owned, timeout=5)
    for child in remaining:
        try:
            if child.status() != psutil.STATUS_ZOMBIE:
                child.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=5)


def wait_worker(process, wall_timeout_sec):
    """Enforce the wall cap outside native simulator/driver calls."""
    try:
        return process.wait(timeout=wall_timeout_sec)
    except subprocess.TimeoutExpired:
        stop_worker(process)
        return 124


def plan(profile_path, *, evaluation_seeds_per_task=None, tasks=None):
    profile = yaml.safe_load(Path(profile_path).read_text())
    protocol = yaml.safe_load((ROOT / profile["evaluation_protocol"]).read_text())
    if evaluation_seeds_per_task is not None and evaluation_seeds_per_task < 1:
        raise ValueError("evaluation_seeds_per_task must be positive")
    if tasks is not None and set(tasks) - {task["name"] for task in protocol["tasks"]}:
        raise ValueError("selected task is outside the frozen task matrix")
    reference = profile["bootstrap"]["reference_seed"]
    cells = []
    for task in protocol["tasks"]:
        if tasks is not None and task["name"] not in tasks:
            continue
        if reference in task["evaluation_seeds"]:
            raise ValueError("reference/evaluation seed overlap")
        cells.append({"task": task["name"], "seed": reference, "phase": "bootstrap"})
        cells.extend({"task": task["name"], "seed": seed, "phase": "evaluation"}
                     for seed in task["evaluation_seeds"][:evaluation_seeds_per_task])
    return {"schema": "harness_vla/rmbench_campaign/v1", "profile": profile,
            "task_config": protocol["task_config"], "instruction_set": protocol["instruction_set"],
            "instruction_type": protocol["instruction_type"], "cells": cells}


def collect_cell(directory):
    directory = Path(directory)
    starts, ends = [], []
    for path in directory.rglob("*_agent_trace.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "episode_end":
                ends.append(event)
            elif event.get("event") == "episode_start":
                starts.append(event)
    if len(ends) != 1:
        return {"validity": "missing_or_ambiguous_episode_end", "denominator_eligible": False,
                "official_success": None, "directory": str(directory)}
    end = ends[0]
    metadata = list(directory.rglob("harness_result.json"))
    result = json.loads(metadata[0].read_text()) if len(metadata) == 1 else {}
    planner = list(directory.rglob("planner_result.json"))
    stats = json.loads(planner[0].read_text()).get("stats", {}) if len(planner) == 1 else {}
    streams = list(directory.rglob("codex.txt.stream.jsonl"))
    reconnects = None
    if len(streams) == 1:
        reconnects = 0
        with streams[0].open() as stream:
            for line in stream:
                if '"response_stream_disconnected"' not in line:
                    continue
                event = json.loads(line)
                payload = event.get("payload") or {}
                info = (payload.get("error") or {}).get("codex_error_info") or {}
                if (event.get("method") == "error" and payload.get("will_retry") is True
                        and "response_stream_disconnected" in info):
                    reconnects += 1
    validity = end.get("episode_validity", {})
    elapsed = (end["timestamp"] - starts[0]["timestamp"]
               if len(starts) == 1 and "timestamp" in starts[0] and "timestamp" in end else None)
    return {"task": end["task_name"], "seed": end["seed"], "phase": result.get("phase"),
            "validity": validity.get("label"),
            "denominator_eligible": bool(validity.get("benchmark_denominator_eligible")),
            "official_success": bool(end["success"]), "environment_steps": end["total_steps"],
            "environment_step_limit": end.get("step_limit"),
            "step_limit_provenance": end.get("step_limit_provenance"),
            "environment_actions_including_resets": result.get("total_environment_actions_including_resets", end["total_steps"]),
            "reference_resets": result.get("reset_count", 0), "episode_wall_seconds": elapsed,
            "vla_calls": result.get("vla_calls"), "sam_calls": result.get("sam_calls"),
            "model_responses": stats.get("model_responses"),
            "stream_reconnect_events": reconnects,
            "input_tokens": stats.get("total_input_tokens"), "output_tokens": stats.get("total_output_tokens"),
            "cached_input_tokens": stats.get("total_cached_input_tokens"),
            "reasoning_output_tokens": stats.get("total_reasoning_output_tokens"),
            "planner_wall_seconds": stats.get("elapsed_s"), "cost_cny": None,
            "directory": str(directory)}


def export_reference_recipe(directory, task, reset_count):
    """Export real commands, as RPent's write_recipe_from_states does.

    Keep successful segmentation queries and non-error motion commands in order.
    Same-seed reset starts a new attempt, so earlier failed attempts remain in
    the audit trail but cannot become part of the final attempt's recipe.
    Coordinates are reference-scene bindings to re-ground, not universal poses.
    """
    directory = Path(directory)
    actions = {"move_to", "move_pose", "rotate_wrist", "rotate_pitch",
               "set_gripper", "release", "vla_act", "segment"}
    events = []
    for path in directory.rglob("*_agent_trace.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if (event.get("event") == "harness_primitive"
                    and event.get("before", {}).get("reset_count", 0) == reset_count
                    and event.get("tool_name") in actions
                    and not event.get("result", {}).get("error")):
                events.append(event)
    events.sort(key=lambda event: event["tool_index"])
    commands = [{"action": event["tool_name"], **event["args"]} for event in events]
    recipe = directory / f"recipe_{task}.jsonl"
    recipe.write_text("".join(json.dumps(command, ensure_ascii=False) + "\n" for command in commands))
    return recipe


def write_source_receipt(memory_dir, directory, task, seed):
    """Accept source outcomes only after the native evaluator finalized the run."""
    directory = Path(directory).resolve()
    outcome = collect_cell(directory)
    for key, expected in {"task": task, "seed": seed, "phase": "bootstrap"}.items():
        if outcome.get(key) is not None and outcome[key] != expected:
            raise ValueError("native reference outcome does not match the requested task/seed")
    receipt = {"task": task, "reference_seed": seed,
               "official_success": outcome["official_success"],
               "denominator_eligible": outcome["denominator_eligible"],
               "validity": outcome["validity"], "run_directory": str(directory),
               "source": "RPent autonomous reference-seed exploration; native episode_end"}
    memory_dir = Path(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    if outcome["denominator_eligible"]:
        recipe = export_reference_recipe(directory, task, outcome["reference_resets"])
        if outcome["official_success"] and not recipe.read_text().strip():
            raise ValueError("successful reference has no recorded primitive recipe")
        destination = memory_dir / f"{task}.jsonl"
        # Preserve any earlier recipe/model-written draft for audit, but use
        # actual execution, not model recollection, as the current recipe.
        draft = directory / f"prior_recipe_{task}.jsonl"
        if destination.is_file() and not draft.exists() and destination.read_bytes() != recipe.read_bytes():
            shutil.copy2(destination, draft)
        shutil.copy2(recipe, destination)
        receipt["recipe_source"] = "executed primitive trace, final reference attempt"
    (memory_dir / f"{task}.source.json").write_text(json.dumps(receipt, indent=2))
    snapshot = Path(directory) / "reference_memory_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in [f"{task}.json", f"{task}.jsonl", f"{task}.source.json", "MEMORY.md"]:
        if (memory_dir / name).is_file() and not (snapshot / name).exists():
            shutil.copy2(memory_dir / name, snapshot / name)
    return receipt


def remaining_reference_budget(budget, task, seed, records):
    """Continue the same reference seed, charging all earlier completed sessions."""
    wall_used, responses_used = 0., 0
    for record in records:
        if (record.get("phase"), record.get("task"), record.get("seed")) != ("bootstrap", task, seed):
            raise ValueError("reference history belongs to another task, seed or phase")
        wall = record.get("episode_wall_seconds")
        responses = record.get("model_responses")
        if (not isinstance(wall, (int, float))
                or not math.isfinite(wall) or wall < 0 or not isinstance(responses, int) or responses < 0):
            raise ValueError("reference history has incomplete outcome or budget accounting")
        wall_used += wall
        responses_used += responses
    remaining = {**budget, "wall_timeout_sec": math.floor(budget["wall_timeout_sec"] - wall_used),
                 "max_turns": budget["max_turns"] - responses_used}
    if remaining["wall_timeout_sec"] <= 0 or remaining["max_turns"] <= 0:
        raise ValueError("reference exploration budget exhausted")
    return remaining


def reference_history(directory):
    """Read native usage, or an explicitly audited interruption budget ledger.

    Budget accounting never makes an infrastructure-invalid outcome eligible.
    """
    record = collect_cell(directory)
    if record.get("denominator_eligible"):
        return record
    ledger = Path(directory) / "reference_accounting.json"
    if ledger.is_file():
        accounting = json.loads(ledger.read_text())
        return {**accounting, "directory": str(Path(directory).resolve()),
                "episode_wall_seconds": accounting["budget_wall_charge_seconds"],
                "model_responses": accounting["budget_model_response_charge"]}
    return record


def require_reference_memory(memory_dir, task):
    """Validate a positive recipe, not admission to evaluation."""
    memory_dir = Path(memory_dir)
    for name in [f"{task}.json", f"{task}.jsonl", f"{task}.source.json", "MEMORY.md"]:
        if not (memory_dir / name).is_file():
            raise ValueError(f"reference memory is incomplete for {task}: {name}")
    source = json.loads((memory_dir / f"{task}.source.json").read_text())
    if source.get("task") != task or not source.get("official_success") or not source.get("denominator_eligible"):
        raise ValueError(f"reference recipe lacks a valid native successful outcome: {task}")


def freeze_reference_memory(memory_dir, destination, tasks):
    """Freeze optional lessons; only native-success sequences are positive recipes."""
    source, destination = Path(memory_dir), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    coverage = {"source_directory": str(source.resolve()), "tasks": {}}
    for task in sorted(set(tasks)):
        receipt_path = source / f"{task}.source.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        valid = receipt.get("task") == task and receipt.get("denominator_eligible") is True
        recipe = source / f"{task}.jsonl"
        available = (valid and receipt.get("official_success") is True
                     and recipe.is_file() and bool(recipe.read_text().strip()))
        for name in [f"{task}.json", f"{task}.source.json"]:
            if (source / name).is_file():
                shutil.copy2(source / name, destination / name)
        if available:
            shutil.copy2(recipe, destination / recipe.name)
        coverage["tasks"][task] = {
            "source_outcome": ("success" if receipt.get("official_success") else "failure") if valid else "unverified_or_absent",
            "successful_recipe_available": available,
            "audit_available": (destination / f"{task}.json").is_file(),
        }
    coverage["successful_recipe_count"] = sum(row["successful_recipe_available"] for row in coverage["tasks"].values())
    coverage["task_count"] = len(coverage["tasks"])
    notes = source / "MEMORY.md"
    (destination / "MEMORY.md").write_text(
        "# Frozen reference notes\n\nTask recipes are optional. Read reference_coverage.json "
        "for native source outcomes. Audits and global notes may describe failed attempts "
        "or unverified hypotheses, not successful solutions. Missing successful recipes "
        "do not prevent evaluation: use current observations and the available tools. "
        "Re-ground all reference coordinates in the current scene.\n\n"
        + (notes.read_text() if notes.is_file() else "No reference lessons are available.\n"))
    (destination / "reference_coverage.json").write_text(json.dumps(coverage, indent=2))
    return coverage


def write_summary(root, records):
    root = Path(root)
    previous = root / "summary.json"
    by_cell = {}
    if previous.exists():
        for record in json.loads(previous.read_text()).get("records", []):
            by_cell[(record.get("phase"), record.get("task"), record.get("seed"))] = record
    for record in records:
        by_cell[(record.get("phase"), record.get("task"), record.get("seed"))] = record
    records = list(by_cell.values())
    evaluation = [r for r in records if r.get("phase") == "evaluation"]
    valid = [r for r in evaluation if r.get("denominator_eligible")]
    successes = sum(r.get("official_success") is True for r in valid)
    expected = set()
    manifest = root / "campaign_plan.json"
    if manifest.exists():
        manifest_data = json.loads(manifest.read_text())
        expected = {(c["task"], c["seed"]) for c in manifest_data["cells"] if c["phase"] == "evaluation"}
    else:
        manifest_data = {}
    observed = {(r.get("task"), r.get("seed")) for r in valid}
    by_task = {}
    for task in sorted({task for task, _ in expected} | {r["task"] for r in evaluation}):
        attempted = [r for r in evaluation if r["task"] == task]
        eligible = [r for r in attempted if r.get("denominator_eligible")]
        won = sum(r.get("official_success") is True for r in eligible)
        by_task[task] = {"planned": sum(name == task for name, _ in expected),
                         "attempted": len(attempted), "valid": len(eligible), "successes": won,
                         "task_failures": len(eligible) - won,
                         "invalid_or_interrupted": len(attempted) - len(eligible),
                         "success_rate": won / len(eligible) if eligible else None}
    summary = {"evaluation_attempts": len(evaluation), "valid_evaluation_outcomes": len(valid),
               "successful_outcomes": successes, "success_rate": successes / len(valid) if valid else None,
               "planned_evaluation_count": len(expected),
               "complete_selected_denominator": bool(expected) and observed == expected,
               "complete_formal_denominator": len(expected) == 60 and observed == expected,
               "account_billing": "ChatGPT subscription; per-run cash cost unavailable",
               "by_task": by_task, "records": records}
    summary["evaluation_budget"] = manifest_data.get("profile", {}).get("evaluation", {})
    coverage_path = root / "frozen_memory" / "reference_coverage.json"
    if coverage_path.is_file():
        summary["reference_coverage"] = json.loads(coverage_path.read_text())
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Harness VLA — RMBench evaluation", "",
             f"Valid held-out outcomes: {len(valid)} / {len(expected)}. "
             + ("Formal denominator complete." if summary["complete_formal_denominator"] else
                "Selected subset complete; not the full 60-trial benchmark." if summary["complete_selected_denominator"] else
                "INCOMPLETE; not a final benchmark result."),
             "", "Reference exploration is excluded from this table. SR uses only valid held-out outcomes; "
             "invalid/interrupted trials are shown separately, not silently replaced or counted as task failures.",
             "", "| Task | Valid / planned | Successes | Task failures | Invalid / interrupted | SR (valid) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for task, row in by_task.items():
        rate = "—" if row["success_rate"] is None else f'{row["success_rate"]:.1%}'
        lines.append(f'| {task} | {row["valid"]} / {row["planned"]} | {row["successes"]} | '
                     f'{row["task_failures"]} | {row["invalid_or_interrupted"]} | {rate} |')
    if summary["evaluation_budget"]:
        budget = summary["evaluation_budget"]
        lines.extend(["", f'Episode action limit: {budget["action_limit"]}; '
                      f'native-horizon mode: {budget.get("step_limit_mode", "cap")}. '
                      "Compare only with trials using the same budget."])
    if "reference_coverage" in summary:
        coverage = summary["reference_coverage"]
        lines.extend(["", f'Successful reference recipes: {coverage["successful_recipe_count"]} / {coverage["task_count"]}. '
                      "Missing recipes do not block evaluation. Failed command sequences are excluded; "
                      "source audits and global failure lessons remain available read-only."])
    references = [r for r in records if r.get("phase") == "bootstrap"]
    if references:
        lines.extend(["", "## Reference exploration — current campaign", "",
                      "These are not held-out scores. A valid failure is not a successful recipe. "
                      "Earlier charged reference sessions remain in their original campaign folders.", "",
                      "| Task | Seed | Native result | Resets | Actions including resets | Model responses |",
                      "| --- | ---: | --- | ---: | ---: | ---: |"])
        for row in sorted(references, key=lambda r: (r["task"], r["seed"])):
            outcome = ("success" if row.get("official_success") else "failure") if row.get("denominator_eligible") else "invalid/unverified"
            lines.append(f'| {row["task"]} | {row["seed"]} | {outcome} | '
                         f'{row.get("reference_resets", "—")} | '
                         f'{row.get("environment_actions_including_resets", "—")} | '
                         f'{row.get("model_responses", "—")} |')
    lines.extend(["", "Billing: ChatGPT subscription; measured token counts are in results.csv. "
                  "A per-run cash invoice is unavailable, not zero.", ""])
    (root / "results.md").write_text("\n".join(lines))
    if records:
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (root / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    return summary
