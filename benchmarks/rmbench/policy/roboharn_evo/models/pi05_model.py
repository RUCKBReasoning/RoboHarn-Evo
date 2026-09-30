from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..utils.openpi_utils import build_robotwin_runtime_config, create_trained_policy


class Pi05PolicyModel:
    def __init__(self, train_config_name: str, model_name: str, checkpoint_id: int, pi0_step: int) -> None:
        self.train_config_name = str(train_config_name)
        self.model_name = str(model_name)
        self.checkpoint_id = int(checkpoint_id)
        self.pi0_step = int(pi0_step)

        repo_root = Path(__file__).resolve().parents[3]
        checkpoint_dir = (
            repo_root
            / "policy"
            / "roboharn_evo"
            / "checkpoints"
            / "pi05"
            / self.train_config_name
            / self.model_name
            / str(self.checkpoint_id)
        )
        assets_dir = checkpoint_dir / "assets"
        assert checkpoint_dir.exists(), f"Missing RoboHarn-Evo pi05 checkpoint: {checkpoint_dir}"
        assert assets_dir.exists(), f"Missing assets directory: {assets_dir}"

        asset_entries = sorted(path.name for path in assets_dir.iterdir() if path.is_dir())
        assert asset_entries, f"Expected norm stats under {assets_dir}"
        self.robotwin_repo_id = asset_entries[0]

        config = build_robotwin_runtime_config(
            repo_id=self.robotwin_repo_id,
            checkpoint_base_dir=str(repo_root / "policy" / "roboharn_evo" / "checkpoints" / "pi05"),
            assets_base_dir=str(repo_root / "policy" / "roboharn_evo" / "assets" / "pi05"),
        )
        self.policy = create_trained_policy(
            config,
            checkpoint_dir,
            robotwin_repo_id=self.robotwin_repo_id,
        )
        self.observation_window: dict[str, Any] | None = None
        self.instruction: str | None = None

    def set_language(self, instruction: str) -> None:
        self.instruction = str(instruction)

    def update_observation_window(self, img_arr: list[np.ndarray], state: np.ndarray) -> None:
        img_front, img_right, img_left = img_arr
        img_front = np.transpose(np.asarray(img_front, dtype=np.uint8), (2, 0, 1))
        img_right = np.transpose(np.asarray(img_right, dtype=np.uint8), (2, 0, 1))
        img_left = np.transpose(np.asarray(img_left, dtype=np.uint8), (2, 0, 1))
        self.observation_window = {
            "state": np.asarray(state, dtype=np.float32),
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self) -> np.ndarray:
        assert self.observation_window is not None, "Call update_observation_window before get_action."
        return np.asarray(self.policy.infer(self.observation_window)["actions"], dtype=np.float32)

    def reset_obsrvationwindows(self) -> None:
        self.instruction = None
        self.observation_window = None
