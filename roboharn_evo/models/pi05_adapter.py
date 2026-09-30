from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

import numpy as np

from roboharn_evo.models.paths import pi05_checkpoint_root, pi05_repository_dir


def _ensure_pi05_import_paths() -> None:
    pi05_dir = pi05_repository_dir()
    pi05_src_dir = pi05_dir / "src"
    for path in (pi05_dir, pi05_src_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


@dataclass(frozen=True)
class Pi05ExecutorConfig:
    train_config_name: str
    model_name: str
    checkpoint_id: int
    pi0_step: int
    prompt_template: str


class Pi05ExecutorAdapter:
    def __init__(self, config: Pi05ExecutorConfig) -> None:
        self.config = config
        self._policy_model: Any | None = None
        self._last_prompt: str | None = None

    def _lazy_load(self) -> Any:
        if self._policy_model is not None:
            return self._policy_model

        from roboharn_evo.models.pi05_model import Pi05PolicyModel

        checkpoint_dir = (
            pi05_checkpoint_root()
            / "pi05_aloha_robotwin_full"
            / self.config.model_name
            / str(self.config.checkpoint_id)
        )
        assert checkpoint_dir.exists(), f"Missing pi05 checkpoint: {checkpoint_dir}"
        self._policy_model = Pi05PolicyModel(
            train_config_name=self.config.train_config_name,
            model_name=self.config.model_name,
            checkpoint_id=self.config.checkpoint_id,
            pi0_step=self.config.pi0_step,
        )
        return self._policy_model

    def reset(self) -> None:
        if self._policy_model is not None:
            self._policy_model.reset_obsrvationwindows()
        self._last_prompt = None

    def build_prompt(self, *, task: str, subtask: str, memory: str) -> str:
        return subtask.strip()

    def predict_action_chunk(
        self,
        *,
        observation: dict[str, Any],
        task: str,
        subtask: str,
        memory: str,
    ) -> np.ndarray:
        model = self._lazy_load()
        prompt = self.build_prompt(task=task, subtask=subtask, memory=memory)
        if prompt != self._last_prompt:
            model.reset_obsrvationwindows()
            model.set_language(prompt)
            self._last_prompt = prompt

        image_triplet, state_vector = self._extract_images_and_state(observation)
        model.update_observation_window(image_triplet, state_vector)
        return np.asarray(model.get_action()[: model.pi0_step], dtype=np.float32)

    def _extract_images_and_state(self, observation: dict[str, Any]) -> tuple[list[np.ndarray], np.ndarray]:
        if "observation" in observation and "joint_action" in observation:
            return [
                np.asarray(observation["observation"]["head_camera"]["rgb"], dtype=np.uint8),
                np.asarray(observation["observation"]["right_camera"]["rgb"], dtype=np.uint8),
                np.asarray(observation["observation"]["left_camera"]["rgb"], dtype=np.uint8),
            ], np.asarray(observation["joint_action"]["vector"], dtype=np.float32)

        if "executor_images" not in observation or "executor_state" not in observation:
            raise KeyError(
                "Pi05 executor adapter expects either raw RMBench observation keys "
                "or executor_images/executor_state."
            )

        images = np.asarray(observation["executor_images"], dtype=np.uint8)
        state = np.asarray(observation["executor_state"], dtype=np.float32)
        if images.ndim == 5:
            current_images = images[-1]
        elif images.ndim == 4:
            current_images = images
        else:
            raise ValueError(f"Unsupported executor_images shape: {images.shape}")

        if state.ndim == 2:
            current_state = state[-1]
        elif state.ndim == 1:
            current_state = state
        else:
            raise ValueError(f"Unsupported executor_state shape: {state.shape}")

        if current_images.shape[0] < 3:
            raise ValueError(f"Expected at least 3 camera views, got {current_images.shape}")

        # 数据集视角顺序为 [head, left, right]，PI0.update_observation_window 使用 [head, right, left]。
        head_image = np.asarray(current_images[0], dtype=np.uint8)
        left_image = np.asarray(current_images[1], dtype=np.uint8)
        right_image = np.asarray(current_images[2], dtype=np.uint8)
        return [head_image, right_image, left_image], np.asarray(current_state, dtype=np.float32)
