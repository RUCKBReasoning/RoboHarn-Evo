from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402
from benchmarks.rmbench.integration import build_run_config, discover_tasks  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def merge_mapping(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_mapping(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass
class Plan:
    output: Path
    assets: Path
    python: str
    workers: list[dict]
    jobs: list[dict]
    timeout: float
    shutdown_grace: float


def load_plan(path: Path) -> Plan:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("parallel plan must be a YAML/JSON object")

    def resolve(value: str) -> Path:
        p = Path(value).expanduser()
        return (REPO_ROOT / p).resolve() if not p.is_absolute() else p.resolve()

    output = resolve(raw["output_root"])
    runtime_root = (REPO_ROOT / "eval_result").resolve()
    if output == runtime_root or not output.is_relative_to(runtime_root):
        raise ValueError("output_root must be a new subdirectory of RoboHarn-Evo/eval_result")
    assets = resolve(raw["assets_root"])
    if output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError("parallel output must not overlap simulator assets")
    workers = deepcopy(raw["workers"])
    if not isinstance(workers, list) or not workers:
        raise ValueError("workers must be a non-empty list")
    for worker in workers:
        worker["gpu"] = str(worker["gpu"])
        for name in ("agent_api_url", "sam3_url"):
            url = str(worker[name]).rstrip("/")
            if urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).hostname:
                raise ValueError(f"worker {name} must be an HTTP(S) service URL")
            worker[name] = url
    jobs = []
    available_tasks = set(discover_tasks())
    for index, job in enumerate(raw["jobs"]):
        task, seed = job["task"], job["seed"]
        if task not in available_tasks:
            raise ValueError(f"unknown RMBench task: {task}")
        if type(seed) is not int or seed < 0:
            raise ValueError("each job seed must be a non-negative integer")
        job_id = f"job_{index:04d}_{task}_seed_{seed}"
        directory = output / job_id
        segment = (resolve(raw["segmentation_root"]) / job_id
                   if raw.get("segmentation_root") else directory / "segmentation")
        if not segment.is_relative_to(runtime_root) or segment.is_relative_to(assets):
            raise ValueError("segmentation_root must be under RoboHarn-Evo/eval_result, outside assets")
        task_config = str(job.get("task_config", raw.get("task_config", "demo_clean")))
        if Path(task_config).suffix in {".yaml", ".yml"} or Path(task_config).parent != Path("."):
            task_config = str(resolve(task_config))
        config = build_run_config(
            config_path=resolve(job.get("config", raw["base_config"])),
            task_name=task, seed=seed, output=directory / "rollout",
            task_config=task_config,
            instruction_type=raw.get("instruction_type", "unseen"),
            instruction_set=raw.get("instruction_set", "rmbench_original"),
            checkpoint_setting=job_id,
            overrides=merge_mapping(raw.get("overrides", {}), job.get("overrides", {})),
        )
        # The scheduler always launches one exact episode, regardless of a
        # reused config's old test_num, seed or artifact location.
        config.update(seed=seed, output_root=str(directory / "rollout"), ckpt_setting=job_id)
        config.setdefault("eval", {}).update(test_num=1, start_seed=seed, exact_seed_fail_closed=True)
        agent = config.setdefault("agent", {})
        agent.setdefault("rollout_dump", {})["root"] = str(directory / "rollout")
        v3 = agent.get("hpk_v3", {})
        store_source = None
        store_read_only = False
        if v3.get("mode") == "full":
            store_source = resolve(v3["store_root"])
            if not store_source.is_dir():
                raise ValueError(f"HPK v3 input Store does not exist: {store_source}")
            store_read_only = v3.get("knowledge_updates_enabled", True) is False
            if not store_read_only and directory.is_relative_to(store_source):
                raise ValueError("parallel output must not be inside the input HPK Store")
            v3["store_root"] = str(store_source if store_read_only else directory / "hpk_v3_store")
            if v3.get("retrieval_rgb_enabled", False):
                v3["rgb_audit_root"] = str(directory / "hpk_rgb_retrieval")
        hpk = agent.get("hpk", {})
        if hpk.get("mode") == "evolving":
            hpk["snapshot_output_root"] = str(directory / "hpk_snapshots")
        jobs.append({"job_id": job_id, "task": task, "seed": seed, "config": config,
                     "directory": directory, "segmentation": segment, "store_source": store_source,
                     "store_read_only": store_read_only})
    if not jobs:
        raise ValueError("jobs must not be empty")
    timeout = float(raw.get("episode_timeout_sec", 7200))
    grace = float(raw.get("shutdown_grace_sec", 45))
    if not 0 < timeout < float("inf") or not 0 < grace <= 60:
        raise ValueError("episode timeout must be positive; shutdown grace must be in (0, 60]")
    return Plan(output, assets, str(raw.get("python", sys.executable)),
                workers, jobs, timeout, grace)


def health(url: str) -> dict:
    with build_opener(ProxyHandler({})).open(url + "/health", timeout=5) as response:
        value = json.load(response)
    if value.get("status") != "ok":
        raise ValueError(f"service is not healthy: {url}")
    return value


def gpu_inventory() -> list[dict]:
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits",
    ], text=True)
    records = []
    for line in text.splitlines():
        index, uuid, pci = (part.strip() for part in line.split(","))
        domain, bus, device = pci.lower().split(":")
        records.append({"index": index, "uuid": uuid, "pci": f"{domain[-4:]}:{bus}:{device}"})
    return records


def preflight(plan: Plan) -> tuple[list[dict], dict]:
    """Read-only resource checks; no model, SAM inference, or simulator call."""
    for name in ("embodiments", "objects"):
        if not (plan.assets / name).is_dir():
            raise ValueError(f"RMBench asset directory is missing {name}: {plan.assets}")
    services = {}
    counts = Counter(w["agent_api_url"] for w in plan.workers)
    for url, slots in counts.items():
        info = services[url] = health(url)
        required = min(slots, len(plan.jobs))
        capacity = info.get("max_concurrent_requests")
        if not isinstance(capacity, int) or capacity < required:
            raise ValueError(
                f"{url}: max_concurrent_requests={capacity}, need >= {required}; "
                "restart that gateway with --max-concurrent-requests or use separate gateways"
            )
    for url in {w["sam3_url"] for w in plan.workers}:
        info = services[url] = health(url)
        # Shared SAM3 is allowed (its inference lock serializes GPU work), but
        # every job's separate input/mask paths must be allowed by that server.
        inputs = [Path(p).resolve() for p in info.get("allowed_input_roots", [])]
        mask_roots = [Path(path).resolve() for path in info.get("allowed_output_roots", [info.get("allowed_output_root", info.get("output_root", ""))])]
        for job in plan.jobs:
            segment = job["segmentation"]
            if not any((segment / "inputs").resolve().is_relative_to(p) for p in inputs):
                raise ValueError(f"SAM3 {url} must allow input files below {plan.output}")
            if not any((segment / "masks").resolve().is_relative_to(root) for root in mask_roots):
                raise ValueError(f"SAM3 {url} must allow output files below {plan.output}")
    inventory = gpu_inventory()
    bindings = []
    for worker in plan.workers:
        matches = [g for g in inventory if worker["gpu"] in (g["index"], g["uuid"])]
        if len(matches) != 1:
            raise ValueError(f"GPU not found by physical index or UUID: {worker['gpu']}")
        bindings.append(matches[0])
    return bindings, services


def job_config(job: dict, worker: dict) -> dict:
    config = deepcopy(job["config"])
    for section, endpoint in (("planner", "plan"), ("ood", "ood"), ("recovery", "recover")):
        service = config.setdefault(section, {})
        service["backend"] = "agent_api"
        service.setdefault("agent_api", {})["server_url"] = worker["agent_api_url"] + "/" + endpoint
    preprocess = config["agent"].setdefault("observation_preprocess", {})
    preprocess["query_url"] = worker["agent_api_url"] + "/perception_queries"
    preprocess["normalization_url"] = worker["agent_api_url"] + "/normalize_perception_queries"
    preprocess.setdefault("segmentation", {})["service_url"] = worker["sam3_url"]
    return config


def job_environment(job: dict, worker: dict, gpu: dict) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": os.pathsep.join((str(REPO_ROOT), str(REPO_ROOT / "benchmarks/rmbench"))),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": gpu["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "RMBENCH_RENDER_DEVICE": "pci:" + gpu["pci"], "RMBENCH_RENDER_DEVICE_STRICT": "1",
        "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0", "RMBENCH_EXPECTED_PHYSICAL_GPU": gpu["index"],
        "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": gpu["pci"],
        "ROBOHARN_EVO_SEGMENTATION_ARTIFACT_DIR": str(job["segmentation"]),
        "RMBENCH_RUNTIME_CONFIG_ROOT": str(job["directory"] / "rollout/runtime_configs"),
        "RMBENCH_OUTPUT_ROOT": str(job["directory"] / "rollout"),
        "ROBOHARN_EVO_OUTPUT_ROOT": str(job["directory"] / "rollout"),
        "ROBOHARN_EVO_WORKSPACE_ROOT": str(job["directory"] / "rollout/runtime_workspace"),
        "ROBOHARN_EVO_SAM3_SERVICE_URL": worker["sam3_url"],
        "ROBOHARN_EVO_FORMAL_PROTOCOL": "0", "ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION": "0",
    })
    # Inherit operator-selected ICD settings; supply NVIDIA defaults only when
    # available. Explicit PCI binding prevents Vulkan silently choosing GPU 0.
    for key, path in {
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    }.items():
        if Path(path).is_file():
            environment.setdefault(key, path)
    return environment


def job_command(plan: Plan, job: dict) -> list[str]:
    return [plan.python, "-u", str(REPO_ROOT / "scripts/run_rmbench.py"), "--run",
            "--task", job["task"], "--seed", str(job["seed"]),
            "--task-config", job["config"]["task_config_path"],
            "--config", str(job["directory"] / "run_config.json"),
            "--output", str(job["directory"] / "rollout"), "--assets-root", str(plan.assets),
            "--checkpoint-setting", job["job_id"],
            "--instruction-type", job["config"]["instruction_type"],
            "--instruction-set", job["config"]["instruction_set"]]


def episode_result(job: dict, returncode: int, reason: str = "") -> dict:
    endings = []
    log = job["directory"] / "rollout.log"
    if log.exists():
        with log.open(errors="replace") as stream:
            for line in stream:
                if not line.startswith("[eval] "):
                    continue
                try:
                    event = json.loads(line[7:])
                except ValueError:
                    continue
                if event.get("event") == "episode_end":
                    endings.append(event)
    ending = endings[0] if len(endings) == 1 else {}
    valid = (returncode == 0 and ending.get("task_name") == job["task"]
             and ending.get("seed") == job["seed"] and type(ending.get("success")) is bool)
    videos = sorted((job["directory"] / "rollout").rglob("episode_0000_rollout/video/*.mp4"))
    if not videos:
        videos = sorted((job["directory"] / "rollout").rglob("video/*.mp4"))
    return {"job_id": job["job_id"], "task": job["task"], "seed": job["seed"],
            "status": reason or ("completed" if valid else "infrastructure_error"),
            "exit_code": returncode, "success": ending.get("success") if valid and not reason else None,
            "total_steps": ending.get("total_steps"), "episode_end": ending,
            "video_paths": [str(p) for p in videos],
            "automatic_retry": False}


def run_parallel(plan: Plan, *, stop: threading.Event | None = None) -> list[dict]:
    stop = stop if stop is not None else threading.Event()
    if plan.output.exists():
        raise ValueError(f"output already exists; choose a new run directory: {plan.output}")
    bindings, services = preflight(plan)
    plan.output.mkdir(parents=True, exist_ok=False)
    write_json(plan.output / "services.json", services)
    # Snapshot every Store before any worker starts so all jobs use the same
    # initial knowledge, never a sibling's partially updated JSONL files.
    frozen_stores = {}
    for job in plan.jobs:
        job["directory"].mkdir()
        for part in ("inputs", "masks"):
            (job["segmentation"] / part).mkdir(parents=True, exist_ok=True)
        if job["store_source"] is not None and job.get("store_read_only", False):
            source = job["store_source"]
            job["store_before"] = {name: (source / name).read_bytes() for name in ("task_knowledge.jsonl", "action_knowledge.jsonl", "knowledge_families.json", "evidence_index.jsonl") if (source / name).is_file()}
        elif job["store_source"] is not None:
            source = frozen_stores.get(job["store_source"], job["store_source"])
            destination = job["directory"] / "hpk_v3_store"
            shutil.copytree(source, destination)
            frozen_stores.setdefault(job["store_source"], destination)
    pending: queue.Queue = queue.Queue()
    for job in plan.jobs:
        pending.put(job)
    records: dict[str, dict] = {}
    lock = threading.Lock()

    def record(job: dict, value: dict) -> None:
        with lock:
            records[job["job_id"]] = value
            write_json(job["directory"] / "status.json", value)
            with (plan.output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    def worker_loop(slot: int, worker: dict) -> None:
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            if stop.is_set():
                return
            process = None
            started = time.monotonic()
            try:
                write_json(job["directory"] / "run_config.json", job_config(job, worker))
                with (job["directory"] / "rollout.log").open("xb") as log:
                    process = subprocess.Popen(job_command(plan, job), cwd=REPO_ROOT,
                                               env=job_environment(job, worker, bindings[slot]),
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    record(job, {"job_id": job["job_id"], "status": "running", "pid": process.pid,
                                 "worker": slot, "gpu": worker["gpu"], "started_utc": now()})
                    reason, signalled_at = "", None
                    while process.poll() is None:
                        if signalled_at is None and (stop.is_set() or time.monotonic() - started >= plan.timeout):
                            reason = "cancelled" if stop.is_set() else "timeout"
                            # Signal only the evaluator first: ffmpeg children must
                            # stay alive to finish video encoding in its finally block.
                            process.terminate()
                            signalled_at = time.monotonic()
                        if signalled_at is not None and time.monotonic() - signalled_at >= plan.shutdown_grace:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            break
                        time.sleep(0.1)
                result = episode_result(job, process.returncode, reason)
                if job.get("store_read_only", False):
                    unchanged = all((job["store_source"] / name).read_bytes() == content for name, content in job["store_before"].items())
                    write_json(job["directory"] / "read_only_store_check.json", {"unchanged": unchanged, "files": sorted(job["store_before"])})
                    if not unchanged:
                        raise RuntimeError("read-only HPK knowledge changed during evaluation")
                    result["knowledge_store_unchanged"] = True
            except Exception as error:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=plan.shutdown_grace)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                result = {"job_id": job["job_id"], "task": job["task"], "seed": job["seed"],
                          "status": "infrastructure_error", "success": None, "automatic_retry": False,
                          "error": f"{type(error).__name__}: {error}"}
            result.update(worker=slot, elapsed_sec=round(time.monotonic() - started, 3), finished_utc=now())
            record(job, result)

    write_json(plan.output / "run.json", {**describe(plan), "status": "running", "pid": os.getpid()})
    with ThreadPoolExecutor(max_workers=len(plan.workers)) as pool:
        futures = [pool.submit(worker_loop, index, worker) for index, worker in enumerate(plan.workers)]
        try:
            for future in futures:
                future.result()
        except BaseException:
            stop.set()
            raise
    for job in plan.jobs:
        if job["job_id"] not in records:
            record(job, {"job_id": job["job_id"], "task": job["task"], "seed": job["seed"],
                         "status": "not_started", "success": None, "automatic_retry": False})
    results = [records[job["job_id"]] for job in plan.jobs]
    write_json(plan.output / "results.json", results)
    write_json(plan.output / "run.json", {**describe(plan), "status": "cancelled" if stop.is_set() else "finished",
                                         "counts": dict(Counter(r["status"] for r in results))})
    return results


def describe(plan: Plan) -> dict:
    return {"output_root": str(plan.output), "workers": plan.workers,
            "jobs": [{"job_id": j["job_id"], "task": j["task"], "seed": j["seed"],
                      "segmentation_root": str(j["segmentation"]),
                      "directory": str(j["directory"]), "hpk_store_read_only": j.get("store_read_only", False),
                      "hpk_store_isolated": j["store_source"] is not None and not j.get("store_read_only", False)}
                     for j in plan.jobs], "automatic_retry": False,
            "episode_timeout_sec": plan.timeout, "services_managed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="YAML/JSON parallel run plan")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true", help="start actual independent simulator episodes")
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    if not args.run:
        print(json.dumps({**describe(plan), "mode": "dry-run", "rollout_started": False}, indent=2))
        return 0
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda _sig, _frame: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        results = run_parallel(plan, stop=stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 130 if stop.is_set() else (0 if all(r["status"] == "completed" for r in results) else 1)


if __name__ == "__main__":
    raise SystemExit(main())
