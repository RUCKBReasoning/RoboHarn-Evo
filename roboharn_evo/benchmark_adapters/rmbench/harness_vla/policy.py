"""Native RMBench policy entry: the original evaluator owns reset and scoring.

RPent runs in its own Python environment. RPC requests are executed on this
simulator's main thread; neither planner threads nor another worker touch it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from roboharn_evo.models.pi05_api_executor_adapter import Pi05ApiExecutorAdapter, Pi05ApiExecutorConfig
from .primitives import RMBenchPrimitives


ROOT = Path(__file__).resolve().parents[4]


class HarnessPolicy:
    def __init__(self, config):
        self.config = dict(config["harness_vla"])
        self.run_config = config
        # 仅提供评测状态接口。
        self.session = SimpleNamespace(agent=self)
        self.status = {"task_finished": False, "terminal_failure": False}
        self.trace_file = None
        self.metadata = {}
        self.rollout_dir = Path(self.config["output_dir"])

    def current_status(self):
        return dict(self.status)

    def set_trace_file(self, path, *, episode_id, seed):
        self.trace_file = Path(path)
        self.episode_id, self.seed = episode_id, seed

    def set_rollout_dump_dir(self, path):
        self.rollout_dir = Path(path)

    def write_rollout_meta(self, metadata):
        self.metadata = {**metadata, "baseline": "Harness VLA / RPent adapted to RMBench",
                         "harness_phase": self.config["phase"], "model": "gpt-5.5", "reasoning_effort": "high"}
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        (self.rollout_dir / "metadata.json").write_text(json.dumps(self.metadata, indent=2))

    def run(self, env):
        sys.path.insert(0, self.config["rpent_root"])
        from rpent.utils.http_rpc import HttpRpcServer

        vla = Pi05ApiExecutorAdapter(Pi05ApiExecutorConfig(
            server_url=self.config["vla_url"], timeout_sec=180, auth_token="", auth_header="Authorization",
            extra_headers={}, extra_body={}, max_chunk_steps=self.config.get("action_chunk", 32), action_dim=14,
        ))
        primitives = RMBenchPrimitives(env, output_dir=self.rollout_dir, vla=vla,
                                       sam_url=self.config["sam_url"],
                                       deadline=time.monotonic() + self.config["wall_timeout_sec"],
                                       reset=(lambda: self.reset_reference(env)) if self.config["phase"] == "bootstrap" else None)
        primitives.save_observation()
        pending = queue.Queue()

        def dispatch(method, args, kwargs):
            if method == "healthz":
                return {"status": "ok"}
            if method != "execute":
                raise ValueError(f"unknown method {method}")
            event, box = threading.Event(), {}
            pending.put((kwargs, event, box))
            if not event.wait(timeout=3550):
                raise TimeoutError("primitive execution timed out; physical action is NOT retried")
            if "error" in box:
                raise RuntimeError(box["error"])
            return box["result"]

        server = HttpRpcServer(("127.0.0.1", 0), dispatch)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config = {**self.config, "env_url": f"http://127.0.0.1:{server.server_address[1]}",
                  "output_dir": str(self.rollout_dir), "task": self.run_config["task_name"],
                  "seed": self.seed, "instruction": env.get_instruction()}
        launch = self.rollout_dir / "planner_config.json"
        launch.write_text(json.dumps(config, indent=2))
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = os.pathsep.join([str(ROOT), self.config["rpent_root"]])
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        process = None
        try:
            with (self.rollout_dir / "planner.log").open("w") as log:
                process = subprocess.Popen([
                    self.config["planner_python"], "-m", "roboharn_evo.benchmark_adapters.rmbench.harness_vla.run_planner",
                    "--config", str(launch),
                ], cwd=self.rollout_dir, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + self.config["wall_timeout_sec"] + 30
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Harness VLA episode wall-clock budget exhausted")
                    try:
                        kwargs, event, box = pending.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        box["result"] = primitives.execute(**kwargs)
                    except Exception as exc:
                        box["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        event.set()
                if process.returncode:
                    raise RuntimeError(f"RPent planner exited {process.returncode}; see planner.log")
            result = json.loads((self.rollout_dir / "planner_result.json").read_text())
            if result.get("error"):
                raise RuntimeError(f"RPent planner failure: {result['error']}")
            self.status.update(task_finished=True, monitor_status="harness_planner_finished")
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            server.shutdown()
            server.server_close()
            summary = {**primitives.state(), "phase": self.config["phase"], "seed": self.seed,
                       "task": self.run_config["task_name"], "scorer": "RMBench eval_success",
                       "planner_model": "gpt-5.5", "reasoning_effort": "high",
                       "auth": "ChatGPT account", "memory_dir": self.config["memory_dir"]}
            (self.rollout_dir / "harness_result.json").write_text(json.dumps(summary, indent=2))

    def reset_reference(self, env):
        if self.config["phase"] != "bootstrap":
            raise ValueError("formal evaluation forbids reset")
        from scripts.smoke_rmbench_failure_boundary_replay import _load_task_args
        args = _load_task_args(self.run_config["task_name"], self.run_config["task_config"])
        args["data_type"].update(self.run_config.get("data_type", {}))
        args.update(save_path=str(self.rollout_dir / "reference_reset_data"),
                    policy_name=self.run_config["policy_name"], ckpt_setting=self.run_config["ckpt_setting"])
        callback = getattr(env, "_eval_video_frame_callback", None)
        instruction = env.get_instruction()
        env.close_env()
        env.setup_demo(now_ep_num=self.episode_id, seed=self.seed, is_test=True, **args)
        env.step_lim = min(env.step_lim, self.config["action_limit"])
        env.set_instruction(instruction)
        env._set_eval_video_frame_callback(callback)


def get_model(usr_args):
    return HarnessPolicy(usr_args)


def reset_model(model):
    model.status = {"task_finished": False, "terminal_failure": False}


def eval(TASK_ENV, model, observation):
    model.run(TASK_ENV)
