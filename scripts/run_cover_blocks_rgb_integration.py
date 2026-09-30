from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_rmbench_hpk_v31_gate_with_services import _health, _wait_health
from scripts.run_rmbench_parallel import describe, load_plan, run_parallel
from roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store
from roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def integration_plan(config: dict, *, source_root: Path, gpus: list[str], agent_port: int, sam3_port: int) -> dict:
    integration = config["integration"]
    common = {
        "eval": {"step_limit": integration["action_limit"], "step_limit_mode": "cap"},
        "planner": {"agent_api": {"timeout_sec": 1200}},
        "recovery": {"agent_api": {"timeout_sec": 1200}},
        "agent": {
            "hpk": {"mode": "off"}, "procedure_experience": {"mode": "off"},
            "pure_tool_control": {"enabled": True, "trigger_step": 0, "signal": "task_level_recovery_control", "wait_for_scene_memory": True, "max_wait_steps": 6, "retry_budget": 0, "max_rounds": integration["control_turn_limit"], "max_control_turns": integration["control_turn_limit"], "max_no_progress_control_turns": 10, "backend_error_budget": 1, "bootstrap_with_planner": True, "skip_vla_rollout": True},
            "recovery": {"enable_retry": False, "enable_reobserve": False},
            "rollout_dump": {"enabled": True, "video_fps": 30},
            "observation_preprocess": {"query_timeout_sec": 1200, "max_objects": 12, "oracle_objects": {"enabled": False}},
        },
    }
    jobs = []
    for seed in integration["seeds"]:
        for method in integration["methods"]:
            hpk = {"mode": method, "knowledge_updates_enabled": False}
            if method == "full":
                hpk.update(store_root=str(source_root), retrieval_rgb_enabled=True, maintenance_rgb_enabled=True, historical_rgb_cameras=["head_camera"], max_rgb_images=32, retrieval_model=config["model"]["model"], retrieval_reasoning_effort=config["model"]["reasoning_effort"], hpk_goal_consistency_enabled=True, hierarchical_reflection=False, timeout_sec=1200)
            jobs.append({"task": config["task"], "seed": seed, "overrides": {"agent": {"hpk_v3": hpk}}})
    return {"output_root": str(source_root / "integration_runs"), "segmentation_root": str(source_root / "integration_segmentation"), "assets_root": os.environ["RMBENCH_ASSETS_ROOT"], "python": os.environ["RMBENCH_PYTHON"], "base_config": "configs/rmbench_deploy_policy.yaml", "task_config": config["task_config"], "instruction_type": config["instruction_type"], "instruction_set": "rmbench_original", "episode_timeout_sec": integration["episode_timeout_seconds"], "workers": [{"gpu": gpu, "agent_api_url": f"http://127.0.0.1:{agent_port}", "sam3_url": f"http://127.0.0.1:{sam3_port}"} for gpu in gpus], "overrides": common, "jobs": jobs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "benchmarks/rmbench/experiments/cover_blocks_rgb_source_v1.yaml")
    parser.add_argument("--gpus", nargs="+", default=["6", "7", "6", "7"])
    parser.add_argument("--agent-port", type=int, default=19210)
    parser.add_argument("--sam3-port", type=int, default=19211)
    parser.add_argument("--methods", nargs="+", choices=["off", "full"])
    parser.add_argument("--run-name")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    source_root = (REPO_ROOT / config["output_root"]).resolve(strict=True)
    tasks, actions = load_hierarchical_store(source_root)
    RGBEvidenceIndex.load(source_root).validate_knowledge(tasks, actions)
    raw_plan = integration_plan(config, source_root=source_root, gpus=args.gpus, agent_port=args.agent_port, sam3_port=args.sam3_port)
    if args.run_name is not None:
        if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
            raise ValueError("run-name must be one directory name")
        raw_plan["output_root"] = str(source_root / "integration_runs" / args.run_name)
        raw_plan["segmentation_root"] = str(source_root / "integration_segmentation" / args.run_name)
    if args.methods is not None:
        raw_plan["jobs"] = [job for job in raw_plan["jobs"] if job["overrides"]["agent"]["hpk_v3"]["mode"] in args.methods]
    plan_path = source_root / ("integration_plan.json" if args.run_name is None else f"integration_plan_{args.run_name}.json")
    write_json(plan_path, raw_plan)
    plan = load_plan(plan_path)
    if not args.run:
        print(json.dumps(describe(plan), indent=2))
        return
    if any(_health(f"http://127.0.0.1:{port}") is not None for port in (args.agent_port, args.sam3_port)):
        raise RuntimeError("integration service ports are already in use")
    cases_path = source_root / "integration_runs" / "cases.json"
    initial = load_plan(source_root / "integration_plan.json")
    cases = {(job["seed"], job["config"]["agent"]["hpk_v3"]["mode"]): {"seed": job["seed"], "method": job["config"]["agent"]["hpk_v3"]["mode"], "directory": str(job["directory"].relative_to(source_root / "integration_runs"))} for job in initial.jobs}
    if cases_path.is_file():
        cases.update({(case["seed"], case["method"]): case for case in json.loads(cases_path.read_text())["cases"]})
    for job in plan.jobs:
        method = job["config"]["agent"]["hpk_v3"]["mode"]
        cases[(job["seed"], method)] = {"seed": job["seed"], "method": method, "directory": str(job["directory"].relative_to(source_root / "integration_runs"))}
    selected_cases = {"cases": list(cases.values()), "selection": "explicit configuration before execution"}
    services = source_root / "integration_services" / (args.run_name or "initial")
    services.mkdir(parents=True, exist_ok=True)
    work = REPO_ROOT / "eval_result/hpk/q4/cover_blocks_service_work" / str(args.agent_port)
    work.mkdir(parents=True, exist_ok=True)
    cache = REPO_ROOT / "eval_result/service_caches/cover_blocks_sam3"
    cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(work), TMP=str(work), TEMP=str(work), ROBOHARN_EVO_PROCEDURE_EXPERIENCE_MODE="off")
    commands = {
        "agent": [
            plan.python, "-u", str(REPO_ROOT / "scripts/serve_rmbench_agent_api.py"),
            "--host", "127.0.0.1", "--port", str(args.agent_port),
            "--provider", "openai", "--backend", "codex-account",
            "--model", config["model"]["model"],
            "--reasoning-effort", config["model"]["reasoning_effort"],
            "--responses-max-images", "64", "--timeout-sec", "1200",
            "--max-retries", "0", "--max-concurrent-requests", str(len(args.gpus)),
            "--codex-bin", os.environ.get("CODEX_BIN", "codex"),
            "--codex-auth-file", os.environ["ROBOHARN_EVO_PROVIDER_AUTH_FILE"],
            "--config-file", os.environ["ROBOHARN_EVO_PROVIDER_CONFIG_FILE"],
            "--codex-workdir", str(work / "agent"),
            "--codex-runtime-root", str(work / "runtime"),
        ],
        "sam3": [os.environ["SAM3_PYTHON"], "-u", str(REPO_ROOT / "scripts/serve_rmbench_sam3.py"), "--host", "127.0.0.1", "--port", str(args.sam3_port), "--sam3-repo", os.environ["SAM3_REPO"], "--checkpoint", os.environ["SAM3_CHECKPOINT"], "--bpe-path", os.environ["SAM3_BPE_PATH"], "--device", "cuda", "--amp-dtype", "bfloat16", "--confidence-threshold", "0.1", "--cache-root", str(cache), "--output-root", str(services / "masks"), "--allowed-output-root", str(services / "masks"), "--instance-id", "cover-blocks-rgb-integration"],
    }
    for job in plan.jobs:
        inputs, masks = job["segmentation"] / "inputs", job["segmentation"] / "masks"
        inputs.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        commands["sam3"].extend(["--allowed-input-root", str(inputs), "--additional-output-root", str(masks)])
    (work / "agent").mkdir(exist_ok=True)
    stop = threading.Event()
    handlers = {sig: signal.signal(sig, lambda _sig, _frame: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    processes = []
    with ExitStack() as stack:
        try:
            for name, command in commands.items():
                log = stack.enter_context((services / f"{name}.log").open("ab"))
                env = dict(environment)
                if name == "sam3":
                    env["CUDA_VISIBLE_DEVICES"] = args.gpus[-1]
                process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                processes.append(process)
                port = args.agent_port if name == "agent" else args.sam3_port
                health = _wait_health(process, f"http://127.0.0.1:{port}", timeout_sec=180)
                if name == "agent" and (health["model"] != config["model"]["model"] or health["reasoning_effort"] != config["model"]["reasoning_effort"]):
                    raise ValueError("integration service model differs from the fixed protocol")
                write_json(services / f"{name}_identity.json", {"pid": process.pid, "command": command, "health": health})
            if args.run_name is not None:
                write_json(cases_path, selected_cases)
            results = run_parallel(plan, stop=stop)
            if args.run_name is None:
                write_json(cases_path, selected_cases)
            print(json.dumps(results, indent=2), flush=True)
            if any(result["status"] != "completed" for result in results):
                raise RuntimeError("integration executions did not all produce final benchmark results")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
