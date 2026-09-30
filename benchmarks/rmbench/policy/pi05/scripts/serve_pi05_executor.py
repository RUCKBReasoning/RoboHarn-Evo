from __future__ import annotations

import argparse
import dataclasses
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.pi05.deploy_policy import encode_obs
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class Pi05ExecutorRuntime:
    def __init__(
        self,
        *,
        config_name: str,
        checkpoint_dir: str,
        pi0_step: int,
        prompt_mode: str,
        asset_id: str | None,
    ) -> None:
        self.config_name = config_name
        self.checkpoint_dir = str(Path(checkpoint_dir).resolve())
        self.pi0_step = pi0_step
        self.prompt_mode = prompt_mode
        self.asset_id = asset_id
        self._train_config = _config.get_config(config_name)
        self._policy = self._create_policy()
        self._last_prompt: str | None = None
        self._observation_window: dict[str, Any] | None = None

    def _create_policy(self):
        checkpoint_dir = Path(self.checkpoint_dir)
        train_config = self._train_config
        resolved_asset_id = self.asset_id
        if resolved_asset_id is None:
            assets_dir = checkpoint_dir / "assets"
            if assets_dir.exists() and assets_dir.is_dir():
                entries = sorted(path.name for path in assets_dir.iterdir() if path.is_dir())
                if entries:
                    resolved_asset_id = entries[0]
        self.asset_id = resolved_asset_id

        if resolved_asset_id is not None:
            updated_assets = dataclasses.replace(train_config.data.assets, asset_id=resolved_asset_id)
            updated_data = dataclasses.replace(train_config.data, assets=updated_assets)
            train_config = dataclasses.replace(train_config, data=updated_data)

        return _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
        )

    def reset(self) -> None:
        self._last_prompt = None
        self._observation_window = None

    def build_prompt(self, *, task: str, subtask: str, memory: str) -> str:
        subtask_only = subtask.strip()
        if self.prompt_mode == "subtask_only":
            return subtask_only
        if self.prompt_mode == "task_subtask":
            return (
                f"Global task: {task.strip()}\n"
                f"Current subtask: {subtask_only}"
            )
        if self.prompt_mode == "task_memory_subtask":
            return (
                f"Global task: {task.strip()}\n"
                f"Committed memory: {memory.strip()}\n"
                f"Current subtask: {subtask_only}"
            )
        raise ValueError(f"Unsupported prompt_mode: {self.prompt_mode}")

    def _build_observation_window(self, observation: dict[str, Any], prompt: str) -> dict[str, Any]:
        input_rgb_arr, input_state = encode_obs(observation)
        img_front = np.transpose(np.asarray(input_rgb_arr[0], dtype=np.uint8), (2, 0, 1))
        img_right = np.transpose(np.asarray(input_rgb_arr[1], dtype=np.uint8), (2, 0, 1))
        img_left = np.transpose(np.asarray(input_rgb_arr[2], dtype=np.uint8), (2, 0, 1))
        state = np.asarray(input_state, dtype=np.float32)
        return {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": prompt,
        }

    def predict_action_chunk(
        self,
        *,
        observation: dict[str, Any],
        task: str,
        subtask: str,
        memory: str,
        max_chunk_steps: int,
    ) -> np.ndarray:
        prompt = self.build_prompt(task=task, subtask=subtask, memory=memory)
        if prompt != self._last_prompt:
            self._observation_window = None
            self._last_prompt = prompt

        self._observation_window = self._build_observation_window(observation, prompt)
        actions = np.asarray(self._policy.infer(self._observation_window)["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected action chunk with shape (T, D), got {actions.shape}")
        return actions[:max_chunk_steps]


class Pi05ExecutorHandler(BaseHTTPRequestHandler):
    runtime: Pi05ExecutorRuntime | None = None
    action_dim: int = 14

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            if self.runtime is None:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "runtime is not initialized")
                return
            self._send_json(
                {
                    "status": "ok",
                    "backend": "pi05",
                    "config_name": self.runtime.config_name,
                    "checkpoint_dir": self.runtime.checkpoint_dir,
                    "pi0_step": self.runtime.pi0_step,
                    "prompt_mode": self.runtime.prompt_mode,
                    "asset_id": self.runtime.asset_id,
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802
        if self.runtime is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "runtime is not initialized")
            return
        if self.path == "/reset":
            self.runtime.reset()
            self._send_json({"status": "ok"})
            return
        if self.path != "/act":
            self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return

        body_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(body_length)
        payload = json.loads(raw_body.decode("utf-8"))

        task = str(payload.get("task", ""))
        subtask = str(payload.get("subtask", ""))
        memory = str(payload.get("memory", ""))
        observation = payload["observation"]
        max_chunk_steps = int(payload.get("max_chunk_steps", self.runtime.pi0_step))
        action_dim = int(payload.get("action_dim", self.action_dim))

        actions = self.runtime.predict_action_chunk(
            observation=observation,
            task=task,
            subtask=subtask,
            memory=memory,
            max_chunk_steps=max_chunk_steps,
        )
        if actions.shape[1] != action_dim:
            raise ValueError(f"Executor produced action_dim={actions.shape[1]}, expected {action_dim}")
        self._send_json(
            {
                "action_chunk": actions.tolist(),
                "model_info": {
                    "backend": "pi05",
                    "config_name": self.runtime.config_name,
                    "checkpoint_dir": self.runtime.checkpoint_dir,
                    "prompt_mode": self.runtime.prompt_mode,
                    "asset_id": self.runtime.asset_id,
                },
            }
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve PI0.5 executor over HTTP for RoboHarn-Evo deployment.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9201)
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--pi0-step", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--asset-id", type=str, default="")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="subtask_only",
        choices=["subtask_only", "task_subtask", "task_memory_subtask"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Pi05ExecutorHandler.runtime = Pi05ExecutorRuntime(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        pi0_step=args.pi0_step,
        prompt_mode=args.prompt_mode,
        asset_id=args.asset_id.strip() or None,
    )
    Pi05ExecutorHandler.action_dim = args.action_dim
    server = ThreadingHTTPServer((args.host, args.port), Pi05ExecutorHandler)
    print(
        "[pi05-executor] listening on "
        f"http://{args.host}:{args.port} "
        f"config_name={args.config_name} checkpoint_dir={Path(args.checkpoint_dir).resolve()} "
        f"prompt_mode={args.prompt_mode} asset_id={args.asset_id.strip() or None}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
