from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.hierarchical_knowledge import ActionKnowledgeV3, SubtaskKnowledgeV3
from roboharn_evo.agent.reflector.multimodal_transport import MultimodalImage


EVIDENCE_INDEX_FILENAME = "evidence_index.jsonl"


def evidence_counts(records: Sequence[Mapping[str, Any]], *, kind: str) -> dict[str, int]:
    events: dict[str, str] = {}
    for record in records:
        if not record.get("valid_for_current_knowledge", True):
            continue
        event = record["execution_event"]
        verdict = record["verdict"]
        if event in events and events[event] != verdict:
            raise ValueError(f"conflicting verdicts for execution event {event}")
        events[event] = verdict
    counts = {verdict: sum(value == verdict for value in events.values()) for verdict in ("support", "oppose", "unverified")}
    if kind == "action":
        counts["independent_verified_trials"] = counts["support"] + counts["oppose"]
    return counts


def knowledge_with_evidence(value: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *, kind: str):
    counts = evidence_counts(records, kind=kind)
    payload = dict(value)
    payload["evidence_summary"] = counts
    payload["status"] = "contested" if counts["oppose"] else "supported" if counts["support"] else "candidate"
    record_type = SubtaskKnowledgeV3 if kind == "task" else ActionKnowledgeV3
    return record_type(payload)


class RGBEvidenceIndex:
    def __init__(self, root: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
        self.root = Path(root).resolve()
        self.records = tuple(copy.deepcopy(dict(record)) for record in records)
        for record in self.records:
            if record["knowledge_type"] not in {"task", "action"}:
                raise ValueError("invalid evidence knowledge type")
            if type(record["knowledge_index"]) is not int or record["knowledge_index"] < 0:
                raise ValueError("invalid evidence knowledge index")
            if record["verdict"] not in {"support", "oppose", "unverified"}:
                raise ValueError("invalid evidence verdict")
            if not record["execution_event"] or not record["before"] or not record["after"]:
                raise ValueError("execution evidence requires an event and before/after images")
            for role in ("before", "after", "delayed"):
                for observation in record[role]:
                    self.image_path(observation["path"])

    def image_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("evidence paths must be relative to the knowledge package")
        resolved = (self.root / path).resolve(strict=True)
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ValueError("evidence image is outside the knowledge package")
        return resolved

    @classmethod
    def load(cls, root: str | Path, *, filename: str = EVIDENCE_INDEX_FILENAME) -> RGBEvidenceIndex:
        directory = Path(root)
        with (directory / filename).open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        return cls(directory, records)

    def for_knowledge(self, kind: str, index: int) -> tuple[dict[str, Any], ...]:
        return tuple(record for record in self.records if record["knowledge_type"] == kind and record["knowledge_index"] == index)

    def eligible(self, kind: str, index: int) -> bool:
        records = self.for_knowledge(kind, index)
        if not records:
            raise ValueError(f"missing RGB evidence for {kind} knowledge {index}")
        return all(record.get("retrieval_enabled", True) for record in records)

    def validate_knowledge(self, task_knowledge: Sequence[Any], action_knowledge: Sequence[Any]) -> None:
        for kind, values in (("task", task_knowledge), ("action", action_knowledge)):
            expected = set(range(len(values)))
            actual = {record["knowledge_index"] for record in self.records if record["knowledge_type"] == kind}
            if actual != expected:
                raise ValueError(f"{kind} evidence index does not cover the current knowledge file")
            for index, value in enumerate(values):
                if dict(value["evidence_summary"]) != evidence_counts(self.for_knowledge(kind, index), kind=kind):
                    raise ValueError(f"{kind} knowledge {index} evidence counts disagree with executions")

    def save(self, root: str | Path | None = None, *, filename: str = EVIDENCE_INDEX_FILENAME) -> Path:
        destination = self.root if root is None else Path(root).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        if destination != self.root:
            paths = {observation["path"] for record in self.records for role in ("before", "after", "delayed") for observation in record[role]}
            for relative in sorted(paths):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.image_path(relative), target)
            for source_name in {record["source_trajectory"] for record in self.records}:
                source = (self.root / source_name).resolve(strict=True)
                if not source.is_relative_to(self.root) or source == self.root:
                    raise ValueError("trajectory provenance must be inside the source package")
                target = destination / source_name
                if source.is_dir():
                    shutil.copytree(source, target, dirs_exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in self.records), encoding="utf-8")
        return target

    def regroup(self, *, task_groups: Sequence[Sequence[int]], action_groups: Sequence[Sequence[int]]) -> RGBEvidenceIndex:
        records = []
        for kind, groups in (("task", task_groups), ("action", action_groups)):
            for target_index, sources in enumerate(groups):
                seen = {}
                for source_index in sources:
                    for original in self.for_knowledge(kind, source_index):
                        if original["execution_event"] in seen:
                            existing = seen[original["execution_event"]]
                            if existing["verdict"] != original["verdict"]:
                                raise ValueError("one execution has conflicting verdicts during consolidation")
                            for role in ("before", "after", "delayed"):
                                known = {(image["camera"], image["frame"], image["path"]) for image in existing[role]}
                                existing[role].extend(copy.deepcopy(image) for image in original[role] if (image["camera"], image["frame"], image["path"]) not in known)
                            continue
                        record = copy.deepcopy(original)
                        record["knowledge_index"] = target_index
                        seen[original["execution_event"]] = record
                        records.append(record)
        return RGBEvidenceIndex(self.root, records)

    def append(self, other: RGBEvidenceIndex, *, task_offset: int, action_offset: int) -> RGBEvidenceIndex:
        incoming = copy.deepcopy(list(other.records))
        if other.root != self.root:
            source_count = len({record["source_trajectory"] for record in self.records})
            prefix = Path("evidence") / f"source_{source_count}"
            copied = {}
            copied_sources = set()
            for record in incoming:
                source_name = record["source_trajectory"]
                if source_name not in copied_sources:
                    source = (other.root / source_name).resolve(strict=True)
                    if not source.is_relative_to(other.root) or source == other.root:
                        raise ValueError("incoming trajectory provenance must be inside its package")
                    target = self.root / prefix / source_name
                    if source.is_dir():
                        shutil.copytree(source, target, dirs_exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                    copied_sources.add(source_name)
                record["source_trajectory"] = (prefix / source_name).as_posix()
                for role in ("before", "after", "delayed"):
                    for observation in record[role]:
                        source = observation["path"]
                        relative = (prefix / source).as_posix()
                        if source not in copied:
                            target = self.root / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if target.exists() and target.read_bytes() != other.image_path(source).read_bytes():
                                raise FileExistsError(target)
                            shutil.copy2(other.image_path(source), target)
                            copied[source] = relative
                        observation["path"] = copied[source]
        for record in incoming:
            record["knowledge_index"] += task_offset if record["knowledge_type"] == "task" else action_offset
        return RGBEvidenceIndex(self.root, [*self.records, *incoming])

    def image_batches(self, records: Sequence[Mapping[str, Any]], *, max_images: int, current_images: Sequence[MultimodalImage] = (), camera_names: Sequence[str] | None = None):
        if max_images <= len(current_images):
            raise ValueError("image capacity must include current and historical observations")
        batch_images = list(current_images)
        batch_bindings = [{"image_index": index, "role": "current observation"} for index in range(len(current_images))]
        for event_index, record in enumerate(records):
            # 每个相机的前后图片一起进入请求，延迟图片随后附带。
            cameras = sorted({item["camera"] for item in record["before"]} & {item["camera"] for item in record["after"]})
            if camera_names is not None:
                cameras = [camera for camera in cameras if camera in camera_names]
            if not cameras:
                raise ValueError("before and after evidence have no common camera")
            for camera in cameras:
                observations = [(role, item) for role in ("before", "after", "delayed") for item in record[role] if item["camera"] == camera]
                if len(observations) + len(current_images) > max_images:
                    raise ValueError("one paired execution exceeds the service image capacity")
                if len(batch_images) + len(observations) > max_images:
                    yield tuple(batch_images), batch_bindings
                    batch_images = list(current_images)
                    batch_bindings = [{"image_index": index, "role": "current observation"} for index in range(len(current_images))]
                for role, observation in observations:
                    batch_bindings.append({"image_index": len(batch_images), "knowledge_index": record["knowledge_index"], "event_index": event_index, "camera": camera.replace("_", " "), "role": role})
                    batch_images.append(MultimodalImage(evidence_id=f"historical observation {len(batch_images)}", mime_type="image/png", content=self.image_path(observation["path"]).read_bytes(), detail="high"))
        if len(batch_images) > len(current_images):
            yield tuple(batch_images), batch_bindings


def bind_reflection_evidence(*, root: Path, source_name: str, source_observations: Mapping[str, Any], package: Any, atomic_knowledge: Mapping[str, Any], bindings: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[dict]], RGBEvidenceIndex]:
    payload = package.to_dict()
    subtasks = payload["task_strategy"]["subtasks"]
    offsets = []
    action_count = 0
    expected = set()
    for subtask_index, subtask in enumerate(subtasks):
        offsets.append(action_count)
        expected.add((subtask_index, None, None))
        for action_index, action in enumerate(subtask["action_knowledge"]):
            for evidence_index in range(len(action["evidence"])):
                expected.add((subtask_index, action_index, evidence_index))
        action_count += len(subtask["action_knowledge"])
    seen, records = set(), []
    covered_task_chunks = set()
    observations = source_observations["observations"]
    chunks = source_observations["chunks"]
    source = source_observations["source"]
    execution_source = source.get("execution_source") or source["raw_files"].get("hdf5")
    if not execution_source:
        raise ValueError("RGB bindings require their actual execution source")
    for binding in bindings:
        key = (binding["subtask_index"], binding["action_index"], binding["evidence_index"])
        if key not in expected or key in seen:
            raise ValueError("reflection evidence bindings contain an unexpected or repeated entry")
        seen.add(key)
        task_index, action_index, evidence_index = key
        is_task = action_index is None
        kind = "task" if is_task else "action"
        knowledge_index = task_index if is_task else offsets[task_index] + action_index
        first, last = binding["chunk_start"], binding["chunk_end"]
        before, after = binding["before_observation"], binding["after_observation"]
        if not 0 <= first <= last < len(chunks) or not 0 <= before < after < len(observations):
            raise ValueError("reflection evidence has an invalid temporal range")
        if observations[before]["frame"] > chunks[first]["start_frame"] or observations[after]["frame"] < chunks[last]["end_frame"]:
            raise ValueError(f"binding {key}: image frames {observations[before]['frame']} to {observations[after]['frame']} do not bracket execution frames {chunks[first]['start_frame']} to {chunks[last]['end_frame']}")
        delayed = binding["delayed_observations"]
        if any(type(index) is not int or not after < index < len(observations) for index in delayed):
            raise ValueError("delayed evidence must follow the after observation")
        if is_task:
            covered_task_chunks.update(range(first, last + 1))
            verdict = binding["task_verdict"]
            observed_result = binding["observed_result"]
        else:
            action_evidence = subtasks[task_index]["action_knowledge"][action_index]["evidence"][evidence_index]
            verdict = action_evidence["verdict"]
            observed_result = action_evidence["observed_result"]
        def pictures(indices):
            return [{"path": path, "camera": camera, "frame": observations[index]["frame"], "timestamp_seconds": observations[index].get("timestamp_seconds")} for index in indices for camera, path in observations[index]["images"].items()]
        records.append({
            "knowledge_type": kind, "knowledge_index": knowledge_index,
            "source_trajectory": source_name,
            "execution_event": f"{execution_source}#{kind}:{first}-{last}",
            "action_range": {"start_frame": chunks[first]["start_frame"], "end_frame": chunks[last]["end_frame"]},
            "before": pictures([before]), "after": pictures([after]), "delayed": pictures(delayed),
            "verdict": verdict, "observed_result": observed_result,
            "missing_evidence": binding["missing_evidence"],
            "tested_knowledge": {name: value for name, value in atomic_knowledge[f"{kind}_knowledge"][knowledge_index].items() if name not in {"evidence_summary", "status"}},
            "execution": {"before_state": subtasks[task_index]["before_state"], "completion_condition": subtasks[task_index]["completion_condition"], "observed_result": observed_result} if is_task else copy.deepcopy(action_evidence),
            "valid_for_current_knowledge": True, "retrieval_enabled": True,
        })
    if seen != expected:
        raise ValueError("reflection must bind every subtask and every action execution")
    if covered_task_chunks != set(range(len(chunks))):
        raise ValueError(f"subtask bindings omit source chunks {sorted(set(range(len(chunks))) - covered_task_chunks)}")
    index = RGBEvidenceIndex(root, records)
    result = {}
    for kind in ("task", "action"):
        result[f"{kind}_knowledge"] = [knowledge_with_evidence(value, index.for_knowledge(kind, ordinal), kind=kind).to_dict() for ordinal, value in enumerate(atomic_knowledge[f"{kind}_knowledge"])]
    return result, index
