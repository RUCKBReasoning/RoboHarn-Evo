from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_hpk_v3_gt_hierarchical_outputs import (
    _arm_sensor_facts,
    _decode_hdf5_rgb,
    _load_chunks,
    _write_json,
)
from scripts.build_rmbench_hpk_formal_q1_source import _instruction
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import build_hierarchical_reflection_input
from roboharn_evo.agent.reflector.multimodal_transport import MultimodalImage


DEFAULT_CONFIG = REPO_ROOT / "benchmarks/rmbench/experiments/cover_blocks_rgb_source_v1.yaml"


def observation_frames(handle: h5py.File, start: int, end: int) -> list[int]:
    # 仅根据连续夹爪信号和测得的高度选帧；不指定动作语义或成功状态。
    frames = {start, end}
    changes = []
    for arm in ("left", "right"):
        values = np.asarray(handle[f"endpose/{arm}_gripper"][start : end + 1])
        changed = np.flatnonzero(np.abs(np.diff(values)) > 1e-5) + start + 1
        if not len(changed):
            continue
        for run in np.split(changed, np.flatnonzero(np.diff(changed) > 1) + 1):
            before, after = int(run[0]) - 1, int(run[-1])
            frames.update((before, after))
            changes.append((arm, before, after))
    for arm, before, after in changes:
        signal = handle[f"endpose/{arm}_gripper"]
        if signal[after] >= signal[before]:
            continue
        later = [a for name, a, b in changes if name == arm and a > after]
        finish = min(later, default=end)
        heights = handle[f"endpose/{arm}_endpose"][after : finish + 1, 2]
        frames.add(int(after + np.argmax(heights)))
    return sorted(frames)


def inspect_sources(config: dict, data_root: Path) -> tuple[str, list[dict]]:
    episodes = config["source_episode_indices"]
    protocol = yaml.safe_load((REPO_ROOT / config["source_protocol"]).read_text())
    task = next(item for item in protocol["tasks"] if item["name"] == config["task"])
    if episodes != protocol["source"]["episode_indices"]:
        raise ValueError("source episode list differs from the fixed source split")
    seeds = [int(item) for item in (data_root / "seed.txt").read_text().split()]
    selected_seeds = [seeds[index] for index in episodes]
    if selected_seeds != config["source_seeds"] or selected_seeds != task["source_seeds"]:
        raise ValueError("source seeds differ from the fixed source split")
    if set(config["integration"]["seeds"]) & set(selected_seeds + task["evaluation_seeds"]):
        raise ValueError("integration configurations overlap source or formal target")
    instruction = _instruction(data_root, episodes, config["instruction_type"])
    current = json.loads(
        (REPO_ROOT / f"benchmarks/rmbench/description/task_instruction/{config['task']}.json").read_text()
    )[config["instruction_type"]]
    if current != [instruction]:
        raise ValueError("source instruction differs from the current benchmark")
    sources = []
    for episode, seed in zip(episodes, selected_seeds, strict=True):
        files = {
            "hdf5": data_root / "data" / f"episode{episode}.hdf5",
            "planned_joint_trajectory": data_root / "_traj_data" / f"episode{episode}.pkl",
            "instruction": data_root / "instructions" / f"episode{episode}.json",
            "video": data_root / "video" / f"episode{episode}.mp4",
        }
        for path in files.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        chunks = _load_chunks(data_root / "language_annotation.json", episode)
        with h5py.File(files["hdf5"], "r") as handle:
            frames = len(handle["observation/head_camera/rgb"])
            if not chunks or chunks[-1][2] != frames - 1:
                raise ValueError(f"episode {episode}: annotation does not cover the full trajectory")
            datasets = {}
            def visit(name, item):
                if isinstance(item, h5py.Dataset):
                    if len(item) != frames:
                        raise ValueError(f"episode {episode}: unaligned dataset {name}")
                    datasets[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}
            handle.visititems(visit)
        sources.append({
            "episode": episode,
            "seed": seed,
            "raw_files": {key: str(path) for key, path in files.items()},
            "frame_count": frames,
            "annotation_count": len(chunks),
            "datasets": datasets,
            "official_success": None,
            "official_success_note": "No terminal scorer value is stored in the source HDF5 or scene_info; do not infer success from the expert label.",
            "timestamp_note": "Native frame indices are available; capture timestamps are absent. The collector also saves motion boundaries, so frame index times save_freq is not an exact timestamp.",
        })
    return instruction, sources


def export_planned_actions(source: dict, episode_root: Path) -> dict:
    # 这是用户指定的本地 benchmark 原始文件；便携包改存无 pickle 的数组。
    with Path(source["raw_files"]["planned_joint_trajectory"]).open("rb") as stream:
        plans = pickle.load(stream)
    arrays, segments = {}, []
    for arm, paths in plans.items():
        for index, path in enumerate(paths):
            keys = {}
            for field in ("position", "velocity"):
                key = f"{arm}/{index}/{field}"
                value = np.asarray(path[field])
                if value.ndim != 2 or not np.isfinite(value).all():
                    raise ValueError(f"invalid recorded joint plan {key}")
                arrays[key] = value
                keys[field] = key
            segments.append({"arm_path": arm, "segment": index, "planning_status": path["status"], "arrays": keys})
    np.savez_compressed(episode_root / "planned_joint_actions.npz", **arrays)
    summary = {
        "segments": segments,
        "alignment_note": "These are per-arm planned joint commands. The original artifact does not provide an exact command-to-RGB timestamp mapping; do not invent one.",
        "hdf5_joint_action_note": "The collector fills HDF5 joint_action with measured get_left/right_arm_jointState values, not these planned commands.",
        "planning_status_note": "Motion planner Success is not a physical-effect verdict or a terminal task score.",
    }
    _write_json(episode_root / "planned_joint_actions.json", summary)
    return summary


def prepare_episode(source: dict, instruction: str, data_root: Path, output_root: Path) -> dict:
    episode = source["episode"]
    episode_root = output_root / f"episode_{episode}"
    episode_root.mkdir(parents=True, exist_ok=False)
    export_planned_actions(source, episode_root)
    chunks = _load_chunks(data_root / "language_annotation.json", episode)
    image_root = output_root / "evidence" / f"episode_{episode}"
    chunk_inputs, descriptions, image_records, spans, operations = [], [], [], [], []
    with h5py.File(source["raw_files"]["hdf5"], "r") as handle:
        arrays = {}
        for group in ("endpose", "joint_action"):
            for name, dataset in handle[group].items():
                value = np.asarray(dataset)
                if not np.isfinite(value).all():
                    raise ValueError(f"episode {episode}: non-finite {group}/{name}")
                arrays[f"{group}/{name}"] = value
        np.savez_compressed(episode_root / "robot_state_and_actions.npz", **arrays)
        cameras = {name: f"observation/{name}/rgb" for name in handle["observation"] if "rgb" in handle[f"observation/{name}"]}
        if "third_view_rgb" in handle:
            cameras["third_view"] = "third_view_rgb"
        for annotation_index, (annotation, start, end) in enumerate(chunks):
            frames = observation_frames(handle, start, end)
            operation = {"annotation_index": annotation_index, "annotation": annotation, "start_frame": start, "end_frame": end, "selected_frames": frames, "ordered_chunk_indices": []}
            for frame in frames:
                images = {}
                for camera, dataset_name in cameras.items():
                    image_path = image_root / camera / f"frame_{frame:06d}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    _decode_hdf5_rgb(handle[dataset_name][frame]).save(image_path)
                    images[camera] = image_path.relative_to(output_root).as_posix()
                # 路径、原始帧号仅在 sidecar；语义知识中不嵌入这些字段。
                image_records.append({"frame": frame, "timestamp_seconds": None, "annotation_index": annotation_index, "images": images})
                descriptions.append(f"Head-camera observation {len(descriptions) + 1}, in chronological order within annotation {annotation_index + 1}: {annotation.rstrip('.')}")
            for before, after in zip(frames, frames[1:]):
                chunk_index = len(chunk_inputs)
                operation["ordered_chunk_indices"].append(chunk_index)
                chunk_inputs.append({
                    "annotation": annotation.rstrip("."),
                    "observed_facts": _arm_sensor_facts(handle, before, after),
                    "visual_observations": [f"Compare head-camera observations {len(descriptions) - len(frames) + frames.index(before) + 1} and {len(descriptions) - len(frames) + frames.index(after) + 1}; later observations may resolve this local effect."],
                })
                spans.append({"chunk_index": chunk_index, "annotation_index": annotation_index, "start_frame": before, "end_frame": after, "before_observation": len(image_records) - len(frames) + frames.index(before), "after_observation": len(image_records) - len(frames) + frames.index(after)})
            operations.append(operation)
    available_state = [
        "Robot end-effector motion and gripper signals are measured; object poses and attachment flags are not present.",
        "The source provides no terminal success flag. Distinguish visually observed effects from the official task score.",
        "A closed gripper alone does not establish attachment; later visible object motion may supply delayed evidence.",
    ]
    payload = {"input_format": "raw ordered action chunk observations", "instruction": instruction, "ordered_action_chunks": chunk_inputs, "key_observations": descriptions, "available_state": available_state}
    # 直接使用共享 reflector 的校验，不另建一套语义规则。
    build_hierarchical_reflection_input(instruction=instruction, ordered_action_chunks=chunk_inputs, key_observations=descriptions, available_state=available_state, image_count=len(descriptions))
    _write_json(episode_root / "reflector_input.json", payload)
    _write_json(episode_root / "source_observations.json", {"source": source, "observations": image_records, "chunks": spans, "operations": operations, "reflection_camera": "head_camera", "rgb_decoding": "shared _decode_hdf5_rgb; undo historical OpenCV RGB-as-BGR JPEG encoding"})
    summary = {"episode": episode, "frame_count": source["frame_count"], "annotation_count": len(chunks), "ordered_chunks": len(chunk_inputs), "reflection_images": len(descriptions), "saved_images": len(image_records) * len(cameras), "cameras": list(cameras)}
    print(json.dumps(summary), flush=True)
    return summary


def prepare(config_path: Path, data_root: Path | None, output_root: Path | None) -> dict:
    config = yaml.safe_load(config_path.read_text())
    data_root = (data_root or Path(config["source_data_root"])).resolve()
    output_root = (output_root or REPO_ROOT / config["output_root"]).resolve()
    instruction, sources = inspect_sources(config, data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "source_trajectories.json"
    if manifest.exists():
        raise FileExistsError(f"already prepared source manifest: {manifest}")
    # 清单先于提取与验证固定，缺失的源数据不替换成其他 episode。
    _write_json(manifest, {"schema": "roboharn_evo/rmbench/cover_blocks_source/v1", "config": config, "instruction": instruction, "source_data_root": str(data_root), "source_trajectories": sources})
    summaries = [prepare_episode(source, instruction, data_root, output_root) for source in sources]
    summary = {"source_trajectory_count": len(summaries), "total_source_frames": sum(x["frame_count"] for x in summaries), "total_saved_images": sum(x["saved_images"] for x in summaries), "episodes": summaries, "external_model_calls": 0, "sam3_calls": 0, "simulator_runs": 0}
    _write_json(output_root / "preparation_summary.json", summary)
    return summary


def load_prepared_source(output_root: Path, episode: int) -> tuple[dict, list[MultimodalImage]]:
    episode_root = output_root / f"episode_{episode}"
    payload = json.loads((episode_root / "reflector_input.json").read_text())
    sidecar = json.loads((episode_root / "source_observations.json").read_text())
    images = [
        MultimodalImage(
            evidence_id=f"episode {episode} observation {ordinal}",
            mime_type="image/png",
            content=(output_root / observation["images"][sidecar["reflection_camera"]]).read_bytes(),
            detail="high",
        )
        for ordinal, observation in enumerate(sidecar["observations"])
    ]
    build_hierarchical_reflection_input(
        instruction=payload["instruction"],
        ordered_action_chunks=payload["ordered_action_chunks"],
        key_observations=payload["key_observations"],
        available_state=payload["available_state"],
        image_count=len(images),
    )
    return payload, images


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed Cover Blocks source trajectories and real RGB evidence")
    parser.add_argument("stage", choices=("prepare",))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    result = prepare(args.config, args.data_root, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
