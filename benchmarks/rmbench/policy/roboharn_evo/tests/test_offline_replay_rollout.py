from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from policy.roboharn_evo.models.backend_factory import build_executor_backend, build_planner_backend
from policy.roboharn_evo.scripts.run_offline_traj_rollout import configure_offline_agent, infer_offline_chunk_steps, load_hdf5_trajectory, override_contains


def test_offline_replay_executor_returns_dummy_chunk() -> None:
    executor = build_executor_backend(
        {
            "backend": "offline_replay",
            "offline_replay": {
                "action_dim": 14,
                "chunk_steps": 2,
            },
        }
    )

    actions = executor.predict_action_chunk(
        observation={},
        task="global task should be ignored",
        subtask="press the button",
        memory="memory should be ignored",
    )

    assert actions.shape == (2, 14)
    assert np.allclose(actions, 0.0)


def test_offline_chunk_steps_infers_from_pi05_api_config() -> None:
    config = {
        "executor": {
            "backend": "pi05_api",
            "pi05_api": {
                "max_chunk_steps": 32,
            },
        },
    }

    assert infer_offline_chunk_steps(config) == 32
    assert infer_offline_chunk_steps(config, explicit_chunk_steps=7) == 7


def test_configure_offline_agent_uses_chunked_replay_and_local_ood() -> None:
    config = {
        "agent": {
            "enabled": True,
            "observation_preprocess": {
                "enabled": True,
                "every_n_steps": 5,
            },
        },
        "planner": {
            "backend": "agent_api",
        },
        "executor": {
            "backend": "pi05_api",
            "pi05_api": {
                "max_chunk_steps": 24,
            },
        },
        "ood": {
            "backend": "agent_api",
        },
    }

    configured = configure_offline_agent(
        config,
        offline_planner=False,
        chunk_steps=0,
        use_vlm_ood=False,
        preprocess_every_chunk=True,
        preprocess_interval_overridden=False,
    )

    assert configured["planner"]["backend"] == "agent_api"
    assert configured["executor"]["backend"] == "offline_replay"
    assert configured["executor"]["offline_replay"]["chunk_steps"] == 24
    assert configured["agent"]["decision_interval"] == 24
    assert configured["agent"]["observation_preprocess"]["every_n_steps"] == 24
    assert configured["ood"]["backend"] == "offline_replay"


def test_configure_offline_agent_respects_preprocess_interval_override() -> None:
    config = {
        "agent": {
            "observation_preprocess": {
                "every_n_steps": 3,
            },
        },
        "executor": {
            "backend": "pi05_api",
            "pi05_api": {
                "max_chunk_steps": 12,
            },
        },
    }

    configured = configure_offline_agent(
        config,
        offline_planner=True,
        offline_subtask="cover the block",
        chunk_steps=6,
        use_vlm_ood=True,
        preprocess_every_chunk=True,
        preprocess_interval_overridden=True,
        decision_interval_overridden=True,
    )

    assert configured["planner"]["backend"] == "offline_replay"
    assert configured["planner"]["offline_replay"]["subtask_text"] == "cover the block"
    assert configured["executor"]["offline_replay"]["chunk_steps"] == 6
    assert "decision_interval" not in configured["agent"]
    assert configured["agent"]["observation_preprocess"]["every_n_steps"] == 3
    assert "ood" not in configured


def test_override_contains_matches_dotted_key_with_optional_dash_prefix() -> None:
    overrides = ["--agent.observation_preprocess.every_n_steps", "2"]

    assert override_contains(overrides, "agent.observation_preprocess.every_n_steps")
    assert not override_contains(overrides, "agent.decision_interval")


def test_offline_replay_planner_returns_subtask() -> None:
    runtime = build_planner_backend(
        {
            "backend": "offline_replay",
            "offline_replay": {
                "subtask_text": "press the button",
            },
        }
    )

    prediction = runtime.predict_planner_step(
        task="press the button three times",
        previous_memory_text="started",
        planner_start_image=np.zeros((8, 8, 3), dtype=np.uint8),
        planner_end_image=np.zeros((8, 8, 3), dtype=np.uint8),
        planner_state=np.zeros(42, dtype=np.float32),
    )

    assert prediction["subtask_text"] == "press the button"
    assert prediction["planner_backend"] == "offline_replay"


def test_offline_replay_planner_uses_global_task_from_control_payload() -> None:
    runtime = build_planner_backend({"backend": "offline_replay"})

    prediction = runtime.predict_planner_step(
        task='{"global_task": "cover the block", "agent_state": {}}',
        previous_memory_text="started",
        planner_start_image=np.zeros((8, 8, 3), dtype=np.uint8),
        planner_end_image=np.zeros((8, 8, 3), dtype=np.uint8),
        planner_state=np.zeros(42, dtype=np.float32),
    )

    assert prediction["subtask_text"] == "cover the block"


def test_load_hdf5_trajectory_builds_rmbench_observations(tmp_path: Path) -> None:
    h5_path = tmp_path / "episode0.hdf5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("joint_action/vector", data=np.zeros((3, 14), dtype=np.float32))
        for camera in ("head", "left", "right"):
            group = handle.create_group(f"observation/{camera}_camera")
            group.create_dataset("rgb", data=np.zeros((3, 8, 8, 3), dtype=np.uint8))
            group.create_dataset("depth", data=np.full((3, 8, 8), 1000.0, dtype=np.float32))
            group.create_dataset("intrinsic_cv", data=np.eye(3, dtype=np.float32))
            group.create_dataset("cam2world_gl", data=np.eye(4, dtype=np.float32))
        endpose = handle.create_group("endpose")
        endpose.create_dataset("left_endpose", data=np.tile(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), (3, 1)))
        endpose.create_dataset("right_endpose", data=np.tile(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), (3, 1)))

    trajectory = load_hdf5_trajectory(
        h5_path,
        instruction="cover the block",
        task_name="cover_blocks",
        episode_id=0,
        max_steps=2,
    )

    assert len(trajectory.observations) == 3
    observation = trajectory.observations[0]
    assert observation["instruction"] == "cover the block"
    assert observation["observation"]["head_camera"]["rgb"].shape == (8, 8, 3)
    assert observation["observation"]["head_camera"]["depth"].shape == (8, 8)
    assert observation["observation"]["head_camera"]["intrinsic_cv"].shape == (3, 3)
    assert observation["endpose"]["left_endpose"].shape == (7,)


def test_load_hdf5_trajectory_uses_full_episode_when_max_steps_is_zero(tmp_path: Path) -> None:
    h5_path = tmp_path / "episode0.hdf5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("joint_action/vector", data=np.zeros((5, 14), dtype=np.float32))
        for camera in ("head", "left", "right"):
            group = handle.create_group(f"observation/{camera}_camera")
            group.create_dataset("rgb", data=np.zeros((5, 8, 8, 3), dtype=np.uint8))

    trajectory = load_hdf5_trajectory(
        h5_path,
        instruction="cover the block",
        task_name="cover_blocks",
        episode_id=0,
        max_steps=0,
    )

    assert len(trajectory.observations) == 5
