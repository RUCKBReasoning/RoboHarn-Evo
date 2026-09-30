from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Literal

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tqdm
import json

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.utils.io import decode_rgb_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RMBench RobotWin trajectories to LeRobot for pi05.")
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--task-config", type=str, default="demo_clean")
    parser.add_argument("--instruction-type", type=str, default="seen", choices=["seen", "unseen"])
    return parser.parse_args()


def create_empty_dataset(repo_id: str, *, mode: Literal["video", "image"] = "image") -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": [motors]},
        "action": {"dtype": "float32", "shape": (14,), "names": [motors]},
        "observation.images.cam_high": {"dtype": mode, "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
        "observation.images.cam_left_wrist": {"dtype": mode, "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
        "observation.images.cam_right_wrist": {"dtype": mode, "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
    }
    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type="aloha",
        features=features,
        use_videos=False,
        tolerance_s=0.0001,
        image_writer_processes=10,
        image_writer_threads=5,
        video_backend=None,
    )


def iter_episode_paths(raw_dir: Path, task_config: str) -> list[tuple[str, Path, Path]]:
    results: list[tuple[str, Path, Path]] = []
    data_root = raw_dir / "data" / "data"
    for task_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        config_dir = task_dir / task_config
        hdf5_dir = config_dir / "data"
        instruction_dir = config_dir / "instructions"
        if not hdf5_dir.exists() or not instruction_dir.exists():
            continue
        for hdf5_path in sorted(hdf5_dir.glob("episode*.hdf5")):
            results.append((task_dir.name, hdf5_path, instruction_dir))
    return results


def load_instruction(instruction_dir: Path, episode_index: int, instruction_type: str) -> str:
    instruction_path = instruction_dir / f"episode{episode_index}.json"
    with instruction_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return str(payload[instruction_type][0])


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    dataset = create_empty_dataset(args.repo_id)
    for task_name, hdf5_path, instruction_dir in tqdm.tqdm(iter_episode_paths(raw_dir, args.task_config)):
        del task_name
        episode_index = int(hdf5_path.stem.replace("episode", ""))
        instruction = load_instruction(instruction_dir, episode_index, args.instruction_type)
        with h5py.File(hdf5_path, "r") as handle:
            states = handle["joint_action/vector"][()]
            head = handle["observation/head_camera/rgb"]
            left = handle["observation/left_camera/rgb"]
            right = handle["observation/right_camera/rgb"]
            for frame_index in range(states.shape[0]):
                next_index = min(frame_index + 1, states.shape[0] - 1)
                frame = {
                    "observation.state": states[frame_index].astype(np.float32),
                    "action": states[next_index].astype(np.float32),
                    "task": instruction,
                    "observation.images.cam_high": decode_rgb_frame(head[frame_index]),
                    "observation.images.cam_left_wrist": decode_rgb_frame(left[frame_index]),
                    "observation.images.cam_right_wrist": decode_rgb_frame(right[frame_index]),
                }
                dataset.add_frame(frame)
            dataset.save_episode()


if __name__ == "__main__":
    main()
