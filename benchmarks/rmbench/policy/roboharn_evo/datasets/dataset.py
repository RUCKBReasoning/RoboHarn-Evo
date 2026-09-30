from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import h5py
import torch
from torch.utils.data import Dataset

from ..utils.io import decode_rgb_frame, image_to_tensor, parse_episode_id
from ..utils.text import VocabularySet


STATE_SUMMARY_KEYS = [
    "joint_action_vector_start",
    "joint_action_vector_end",
    "left_endpose_end",
    "right_endpose_end",
]


@dataclass(frozen=True)
class SampleSpec:
    row: dict[str, Any]
    window_start: int
    sample_type: str
    prev_memory_text: str
    commit_label: str
    planner_only: bool
    apply_write_loss: bool
    apply_subtask_loss: bool


def split_rows_by_episode(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["episode_id"]), []).append(row)

    episode_ids = sorted(grouped.keys())
    random.Random(seed).shuffle(episode_ids)
    split_index = max(1, int(len(episode_ids) * float(train_ratio)))
    train_episode_ids = set(episode_ids[:split_index])

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for episode_id, episode_rows in grouped.items():
        if episode_id in train_episode_ids:
            train_rows.extend(episode_rows)
        else:
            val_rows.extend(episode_rows)
    return train_rows, val_rows


class RMBenchRoboHarnDataset(Dataset):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        rmbench_root: str,
        task_config: str,
        vocab: VocabularySet,
        image_size: int,
        planner_camera_key: str,
        executor_camera_keys: list[str],
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        action_stride: int,
        evidence_state_dim: int,
        max_memory_candidates: int,
        include_hard_negatives: bool,
    ) -> None:
        super().__init__()
        self.rows = sorted(rows, key=lambda item: (str(item["episode_id"]), int(item["segment_id"])))
        self.rmbench_root = Path(rmbench_root)
        self.task_config = str(task_config)
        self.vocab = vocab
        self.image_size = int(image_size)
        self.planner_camera_key = str(planner_camera_key)
        self.executor_camera_keys = list(executor_camera_keys)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.action_stride = int(action_stride)
        self.evidence_state_dim = int(evidence_state_dim)
        self.max_memory_candidates = int(max_memory_candidates)
        self.include_hard_negatives = bool(include_hard_negatives)

        self.rows_by_episode: dict[str, list[dict[str, Any]]] = {}
        self.rows_by_subtask: dict[str, list[dict[str, Any]]] = {}
        for row in self.rows:
            episode_id = str(row["episode_id"])
            self.rows_by_episode.setdefault(episode_id, []).append(row)
            subtask = str(row["subtask"])
            self.rows_by_subtask.setdefault(subtask, []).append(row)

        for episode_rows in self.rows_by_episode.values():
            episode_rows.sort(key=lambda item: int(item["segment_id"]))

        self.samples = self._build_samples()

    def _build_samples(self) -> list[SampleSpec]:
        specs: list[SampleSpec] = []
        for row in self.rows:
            t_start = int(row["t_start"])
            t_end = int(row["t_end"])
            valid_starts = list(range(t_start, t_end, self.action_stride))
            if not valid_starts:
                valid_starts = [t_start]

            if bool(row.get("update_required", True)):
                commit_label = str(row.get("update_trigger", "state_change"))
            else:
                commit_label = "no_update"

            for window_start in valid_starts:
                specs.append(
                    SampleSpec(
                        row=row,
                        window_start=window_start,
                        sample_type="positive",
                        prev_memory_text=str(row["previous_memory_text"]),
                        commit_label=commit_label,
                        planner_only=False,
                        apply_write_loss=commit_label != "no_update",
                        apply_subtask_loss=True,
                    )
                )

            if self.include_hard_negatives and commit_label != "no_update":
                specs.append(
                    SampleSpec(
                        row=row,
                        window_start=t_start,
                        sample_type="missed_update",
                        prev_memory_text=str(row["previous_memory_text"]),
                        commit_label="no_update",
                        planner_only=True,
                        apply_write_loss=False,
                        apply_subtask_loss=False,
                    )
                )
                inconsistent_prev = self._pick_inconsistent_previous_memory(row=row)
                if inconsistent_prev is not None:
                    specs.append(
                        SampleSpec(
                            row=row,
                            window_start=t_start,
                            sample_type="history_inconsistent",
                            prev_memory_text=inconsistent_prev,
                            commit_label="no_update",
                            planner_only=True,
                            apply_write_loss=False,
                            apply_subtask_loss=False,
                        )
                    )
        return specs

    def _pick_inconsistent_previous_memory(self, *, row: dict[str, Any]) -> str | None:
        episode_rows = self.rows_by_episode[str(row["episode_id"])]
        for candidate in reversed(episode_rows):
            memory_text = str(candidate["memory_text"])
            if memory_text != str(row["previous_memory_text"]):
                return memory_text
        return None

    def _episode_hdf5_path(self, episode_id: str) -> Path:
        task_name, episode_index = parse_episode_id(episode_id)
        hdf5_path = self.rmbench_root / "data" / "data" / task_name / self.task_config / "data" / f"episode{episode_index}.hdf5"
        assert hdf5_path.exists(), f"Missing HDF5 for {episode_id}: {hdf5_path}"
        return hdf5_path

    def _flatten_state_summary(self, row: dict[str, Any], hdf5_file: h5py.File) -> torch.Tensor:
        summary = row.get("chunk_evidence", {}).get("state_summary")
        if isinstance(summary, dict):
            flat_values: list[float] = []
            for key in STATE_SUMMARY_KEYS:
                values = summary.get(key, [])
                flat_values.extend(float(value) for value in values)
        else:
            t_start = int(row["t_start"])
            t_end = int(row["t_end"])
            flat_values = []
            flat_values.extend(float(value) for value in hdf5_file["joint_action/vector"][t_start].tolist())
            flat_values.extend(float(value) for value in hdf5_file["joint_action/vector"][t_end].tolist())
            flat_values.extend(float(value) for value in hdf5_file["endpose/left_endpose"][t_end].tolist())
            flat_values.extend(float(value) for value in hdf5_file["endpose/right_endpose"][t_end].tolist())

        if len(flat_values) < self.evidence_state_dim:
            flat_values.extend([0.0] * (self.evidence_state_dim - len(flat_values)))
        flat_values = flat_values[: self.evidence_state_dim]
        return torch.tensor(flat_values, dtype=torch.float32)

    def _memory_candidates(self, spec: SampleSpec) -> tuple[torch.Tensor, torch.Tensor]:
        row = spec.row
        if not spec.apply_write_loss:
            candidates = [spec.prev_memory_text]
        else:
            positive = str(row["memory_text"])
            candidates = [positive]
            episode_rows = self.rows_by_episode[str(row["episode_id"])]
            later_rows = [item for item in episode_rows if int(item["segment_id"]) > int(row["segment_id"])]
            for later in later_rows:
                later_memory = str(later["memory_text"])
                if later_memory not in candidates:
                    candidates.append(later_memory)
                if len(candidates) >= self.max_memory_candidates:
                    break

            same_subtask_rows = self.rows_by_subtask[str(row["subtask"])]
            for same_subtask_row in same_subtask_rows:
                negative_memory = str(same_subtask_row["memory_text"])
                if negative_memory not in candidates:
                    candidates.append(negative_memory)
                if len(candidates) >= self.max_memory_candidates:
                    break

            previous_memory = str(row["previous_memory_text"])
            if previous_memory not in candidates and len(candidates) < self.max_memory_candidates:
                candidates.append(previous_memory)

        candidate_mask = torch.zeros(self.max_memory_candidates, dtype=torch.bool)
        candidate_tokens = torch.full(
            (self.max_memory_candidates, self.vocab.max_memory_tokens),
            fill_value=self.vocab.text_tokenizer.pad_id,
            dtype=torch.long,
        )

        for index, candidate_text in enumerate(candidates[: self.max_memory_candidates]):
            encoded = self.vocab.encode_memory(candidate_text)
            candidate_tokens[index] = torch.tensor(encoded, dtype=torch.long)
            candidate_mask[index] = True
        return candidate_tokens, candidate_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        spec = self.samples[index]
        row = spec.row
        hdf5_path = self._episode_hdf5_path(str(row["episode_id"]))

        task_tokens = torch.tensor(self.vocab.encode_task(str(row["task"])), dtype=torch.long)
        prev_memory_tokens = torch.tensor(self.vocab.encode_memory(spec.prev_memory_text), dtype=torch.long)
        commit_label = torch.tensor(self.vocab.commit_to_id[spec.commit_label], dtype=torch.long)
        subtask_tokens = torch.tensor(self.vocab.encode_subtask(str(row["subtask"])), dtype=torch.long)
        subtask_label = torch.tensor(self.vocab.subtask_to_id[str(row["subtask"])], dtype=torch.long)

        if spec.commit_label == "no_update":
            committed_memory_text = spec.prev_memory_text
        else:
            committed_memory_text = str(row["memory_text"])
        committed_memory_tokens = torch.tensor(self.vocab.encode_memory(committed_memory_text), dtype=torch.long)

        memory_candidate_tokens, memory_candidate_mask = self._memory_candidates(spec)
        memory_target_index = torch.tensor(0, dtype=torch.long)

        num_views = len(self.executor_camera_keys)
        obs_images = torch.zeros(
            (self.action_horizon, num_views, 3, self.image_size, self.image_size),
            dtype=torch.float32,
        )
        # s: (T, D_s)
        state_seq = torch.zeros((self.action_horizon, self.state_dim), dtype=torch.float32)
        # a: (T, D_a)
        action_seq = torch.zeros((self.action_horizon, self.action_dim), dtype=torch.float32)
        sequence_mask = torch.zeros((self.action_horizon,), dtype=torch.bool)
        # t: (T, 1)
        time_seq = torch.zeros((self.action_horizon, 1), dtype=torch.float32)

        with h5py.File(hdf5_path, "r") as hdf5_file:
            evidence_start_image = image_to_tensor(
                decode_rgb_frame(hdf5_file[self.planner_camera_key][int(row["t_start"])]),
                image_size=self.image_size,
            )
            evidence_end_image = image_to_tensor(
                decode_rgb_frame(hdf5_file[self.planner_camera_key][int(row["t_end"])]),
                image_size=self.image_size,
            )
            evidence_state = self._flatten_state_summary(row=row, hdf5_file=hdf5_file)

            if not spec.planner_only:
                max_valid_steps = max(0, int(row["t_end"]) - int(spec.window_start))
                valid_steps = min(self.action_horizon, max_valid_steps)
                for step_index in range(valid_steps):
                    frame_index = int(spec.window_start) + step_index
                    next_frame_index = frame_index + 1
                    current_state = torch.tensor(
                        hdf5_file["joint_action/vector"][frame_index][: self.state_dim],
                        dtype=torch.float32,
                    )
                    next_action = torch.tensor(
                        hdf5_file["joint_action/vector"][next_frame_index][: self.action_dim],
                        dtype=torch.float32,
                    )
                    state_seq[step_index] = current_state
                    action_seq[step_index] = next_action
                    sequence_mask[step_index] = True
                    if self.action_horizon > 1:
                        time_value = float(step_index) / float(self.action_horizon - 1)
                    else:
                        time_value = 0.0
                    time_seq[step_index, 0] = time_value

                    for view_index, camera_name in enumerate(self.executor_camera_keys):
                        camera_key = f"observation/{camera_name}/rgb"
                        assert camera_key in hdf5_file, f"Missing camera key {camera_key} in {hdf5_path}"
                        obs_images[step_index, view_index] = image_to_tensor(
                            decode_rgb_frame(hdf5_file[camera_key][frame_index]),
                            image_size=self.image_size,
                        )
            else:
                evidence_state = evidence_state

        return {
            "task_tokens": task_tokens,
            "prev_memory_tokens": prev_memory_tokens,
            "evidence_start_image": evidence_start_image,
            "evidence_end_image": evidence_end_image,
            "evidence_state": evidence_state,
            "memory_candidate_tokens": memory_candidate_tokens,
            "memory_candidate_mask": memory_candidate_mask,
            "memory_target_index": memory_target_index,
            "commit_label": commit_label,
            "subtask_tokens": subtask_tokens,
            "subtask_label": subtask_label,
            "committed_memory_tokens": committed_memory_tokens,
            "obs_images": obs_images,
            "state_seq": state_seq,
            "action_seq": action_seq,
            "sequence_mask": sequence_mask,
            "time_seq": time_seq,
            "apply_write_loss": torch.tensor(spec.apply_write_loss, dtype=torch.bool),
            "apply_subtask_loss": torch.tensor(spec.apply_subtask_loss, dtype=torch.bool),
        }
