from __future__ import annotations

import numpy as np

from policy.roboharn_evo.agent.vla import VLAInstructionBuilder
from policy.roboharn_evo.models.pi05_api_executor_adapter import Pi05ApiExecutorAdapter, Pi05ApiExecutorConfig
from policy.roboharn_evo.models.pi05_adapter import Pi05ExecutorAdapter, Pi05ExecutorConfig
from policy.roboharn_evo.models.world_action_model_adapter import WorldActionModelExecutorAdapter, WorldActionModelExecutorConfig


def test_vla_instruction_text_is_subtask_only() -> None:
    builder = VLAInstructionBuilder()
    payload = builder.build(
        global_task="press the button three times",
        current_subtask="press the button",
        committed_memory="button has already been pressed twice",
        retry_count=2,
        recovered=True,
        recovery_reason="motion blocked",
    )

    assert builder.render_text(payload) == "press the button"


def test_local_pi05_prompt_ignores_task_and_memory() -> None:
    adapter = Pi05ExecutorAdapter(
        Pi05ExecutorConfig(
            train_config_name="unused",
            model_name="unused",
            checkpoint_id=0,
            pi0_step=50,
            prompt_template="Global task: {task}\nCommitted memory: {memory}\nCurrent subtask: {subtask}\n",
        )
    )

    assert adapter.build_prompt(
        task="press the button three times",
        subtask="press the button",
        memory="button has already been pressed twice",
    ) == "press the button"


def test_pi05_api_payload_clears_task_and_memory(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post_json(*, url: str, payload: dict, timeout_sec: int, headers: dict | None = None) -> dict:
        captured.update(payload)
        return {"action_chunk": np.zeros((1, 14), dtype=np.float32).tolist()}

    monkeypatch.setattr(
        "policy.roboharn_evo.models.pi05_api_executor_adapter._post_json",
        fake_post_json,
    )
    adapter = Pi05ApiExecutorAdapter(
        Pi05ApiExecutorConfig(
            server_url="http://127.0.0.1:9201/act",
            timeout_sec=1,
            auth_token="",
            auth_header="Authorization",
            extra_headers={},
            extra_body={"task": "bad task", "memory": "bad memory"},
            max_chunk_steps=1,
            action_dim=14,
        )
    )

    adapter.predict_action_chunk(
        observation={"observation": {}},
        task="press the button three times",
        subtask="press the button",
        memory="button has already been pressed twice",
    )

    assert captured["task"] == ""
    assert captured["subtask"] == "press the button"
    assert captured["memory"] == ""


def test_world_action_model_payload_clears_task_and_memory(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post_json(*, url: str, payload: dict, timeout_sec: int, headers: dict | None = None) -> dict:
        captured.update(payload)
        return {"actions": np.zeros((1, 14), dtype=np.float32).tolist()}

    monkeypatch.setattr(
        "policy.roboharn_evo.models.world_action_model_adapter._post_json",
        fake_post_json,
    )
    adapter = WorldActionModelExecutorAdapter(
        WorldActionModelExecutorConfig(
            server_url="http://127.0.0.1:9201/act",
            timeout_sec=1,
            auth_token="",
            auth_header="Authorization",
            extra_headers={},
            extra_body={"task": "bad task", "memory": "bad memory"},
            max_chunk_steps=1,
            action_dim=14,
        )
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    adapter.predict_action_chunk(
        observation={
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            },
            "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
        },
        task="press the button three times",
        subtask="press the button",
        memory="button has already been pressed twice",
    )

    assert captured["task"] == ""
    assert captured["subtask"] == "press the button"
    assert captured["memory"] == ""
