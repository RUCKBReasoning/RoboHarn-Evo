"""Run the frozen 3-init by 5-condition LIBERO read-only feasibility pilot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from roboharn_evo.benchmark_adapters.libero_pro.p0_feasibility import CONDITIONS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "scripts/run_libero_pro.py"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_init_states(value: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("init states must be integers") from exc
    if len(values) != 3 or len(set(values)) != 3 or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "init states must contain three distinct non-negative integers"
        )
    return values


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--capability-mapping", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--policy-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--policy-source-ref", required=True)
    parser.add_argument("--policy-license-id", required=True)
    parser.add_argument("--planner-server-url", required=True)
    parser.add_argument("--sam3-server-url", required=True)
    parser.add_argument("--perception-root", type=Path, required=True)
    parser.add_argument("--source-domain", default="RMBench")
    parser.add_argument("--suite", default="libero_spatial_swap")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-states", type=_parse_init_states, default=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=220)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--planner-calls", type=int, default=21)
    parser.add_argument("--family-exhaustive-threshold", type=int, default=1)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--episode-timeout-sec", type=int, default=3600)
    parser.add_argument("--mujoco-gl", choices=("egl", "osmesa", "glfw"), default="egl")
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _store_bytes(root: Path) -> dict[str, bytes]:
    files = sorted(path for path in root.iterdir() if path.is_file())
    if not files:
        raise ValueError("Store contains no files")
    return {path.name: path.read_bytes() for path in files}


def _condition_args(condition: str) -> list[str]:
    modes = {
        "native_pi05": "--native-policy-off",
        "agent_off": "--agent-loop-off",
        "task_hpk": "--agent-loop-task-hpk",
        "action_hpk": "--agent-loop-action-hpk",
        "task_action_hpk": "--agent-loop-task-action-hpk",
    }
    return [modes[condition]]


def main() -> int:
    args = _args()
    root = args.output_root.expanduser().resolve()
    store = args.store_root.expanduser().resolve(strict=True)
    assets = args.assets_root.expanduser().resolve(strict=True)
    checkpoint = args.policy_checkpoint_dir.expanduser().resolve(strict=True)
    capability_mapping_path = args.capability_mapping.expanduser().resolve(strict=True)
    perception_root = args.perception_root.expanduser().resolve(strict=True)
    for child in (perception_root / "inputs", perception_root / "masks"):
        if not child.is_dir() or any(child.iterdir()):
            raise ValueError(
                "perception root must start with empty inputs and masks directories"
            )
    capability_mapping = json.loads(capability_mapping_path.read_text(encoding="utf-8"))
    if not isinstance(capability_mapping, dict) or not capability_mapping.get(
        "mappings"
    ):
        raise ValueError("capability mapping must contain predeclared mappings")
    if root.exists():
        raise FileExistsError(f"output root already exists: {root}")
    for value, label in (
        (args.horizon, "horizon"),
        (args.replan_steps, "replan steps"),
        (args.planner_calls, "planner calls"),
        (args.family_exhaustive_threshold, "Family threshold"),
        (args.episode_timeout_sec, "episode timeout"),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    if args.settle_steps < 0:
        raise ValueError("settle steps must be non-negative")

    root.mkdir(parents=True)
    _write_json(root / "capability_mapping.json", capability_mapping)
    before_store = _store_bytes(store)
    worktree_status = _git("status", "--short")
    untracked_sources = tuple(
        value
        for value in _git("ls-files", "--others", "--exclude-standard").splitlines()
        if value
    )
    if worktree_status:
        (root / "working_tree_status.txt").write_text(
            worktree_status.rstrip() + "\n",
            encoding="utf-8",
        )
        (root / "working_tree.patch").write_text(
            _git("diff", "--no-ext-diff") + "\n",
            encoding="utf-8",
        )
        for relative in untracked_sources:
            source = REPOSITORY_ROOT / relative
            if source.is_file():
                destination = root / "untracked_source_snapshot" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    commands: list[dict[str, Any]] = []
    common = [
        str(RUNNER),
        "--suite",
        args.suite,
        "--task-id",
        str(args.task_id),
        "--seed",
        str(args.seed),
        "--horizon",
        str(args.horizon),
        "--policy-replan-steps",
        str(args.replan_steps),
        "--policy-settle-steps",
        str(args.settle_steps),
        "--agent-max-planner-calls",
        str(args.planner_calls),
        "--assets-root",
        str(assets),
        "--policy-checkpoint-dir",
        str(checkpoint),
        "--policy-source-ref",
        args.policy_source_ref,
        "--policy-license-id",
        args.policy_license_id,
        "--policy-device",
        args.policy_device,
        "--mujoco-gl",
        args.mujoco_gl,
    ]
    for init_state in args.init_states:
        for condition in CONDITIONS:
            output = root / f"init_{init_state}" / condition
            command = [
                sys.executable,
                *common,
                *_condition_args(condition),
                "--init-state-id",
                str(init_state),
                "--output-dir",
                str(output),
            ]
            if condition != "native_pi05":
                command.extend(
                    [
                        "--planner-server-url",
                        args.planner_server_url,
                        "--agent-loop-profile",
                        "perception_memory",
                        "--sam3-server-url",
                        args.sam3_server_url,
                        "--perception-artifact-root",
                        str(perception_root),
                    ]
                )
            if condition in {"task_hpk", "action_hpk", "task_action_hpk"}:
                command.extend(
                    [
                        "--hpk-task-store-root",
                        str(store),
                        "--hpk-source-domain",
                        args.source_domain,
                        "--hpk-family-exhaustive-threshold",
                        str(args.family_exhaustive_threshold),
                    ]
                )
            commands.append(
                {
                    "init_state_id": init_state,
                    "condition": condition,
                    "output": str(output),
                    "command": command,
                }
            )

    _write_json(
        root / "run_config.json",
        {
            "schema": "roboharn_evo/libero_p0_read_only_feasibility_config/v1",
            "status": "DEVELOPMENT",
            "producing_commit": _git("rev-parse", "HEAD"),
            "working_tree_clean": not bool(worktree_status),
            "untracked_source_files": list(untracked_sources),
            "suite": args.suite,
            "task_id": args.task_id,
            "init_states": list(args.init_states),
            "seed": args.seed,
            "conditions": list(CONDITIONS),
            "horizon": args.horizon,
            "replan_steps": args.replan_steps,
            "settle_steps": args.settle_steps,
            "planner_calls": args.planner_calls,
            "source_store": str(store),
            "source_domain": args.source_domain,
            "perception_root": str(perception_root),
            "family_exhaustive_threshold": args.family_exhaustive_threshold,
            "persistent_store_write": False,
            "online_self_evolution": False,
            "commands": commands,
        },
    )
    _write_json(
        root / "run_status.json",
        {"completed": [], "active": None, "failed": None},
    )
    completed: list[dict[str, Any]] = []
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for item in commands:
        status = {"completed": completed, "active": item, "failed": None}
        _write_json(root / "run_status.json", status)
        print(
            f"running init={item['init_state_id']} condition={item['condition']}",
            flush=True,
        )
        log_path = Path(item["output"]).parent / f"{item['condition']}_console.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("x", encoding="utf-8") as stream:
                subprocess.run(
                    item["command"],
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=args.episode_timeout_sec,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            _write_json(
                root / "run_status.json",
                {
                    "completed": completed,
                    "active": None,
                    "failed": {**item, "error": type(exc).__name__},
                },
            )
            raise
        completed.append(
            {
                "init_state_id": item["init_state_id"],
                "condition": item["condition"],
            }
        )
        if item["condition"] != "native_pi05":
            archive = (
                root
                / "perception_artifacts"
                / f"init_{item['init_state_id']}"
                / item["condition"]
            )
            archive.mkdir(parents=True)
            for name in ("inputs", "masks"):
                source = perception_root / name
                source.rename(archive / name)
                source.mkdir()
    after_store = _store_bytes(store)
    _write_json(
        root / "store_read_only_check.json",
        {
            "source_store": str(store),
            "file_names_before": sorted(before_store),
            "file_names_after": sorted(after_store),
            "unchanged": before_store == after_store,
        },
    )
    _write_json(
        root / "run_status.json",
        {"completed": completed, "active": None, "failed": None},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
