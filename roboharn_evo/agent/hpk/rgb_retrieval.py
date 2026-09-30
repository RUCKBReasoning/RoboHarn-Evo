from __future__ import annotations

import copy
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image

from roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import ExistingGPTMultimodalReflectorBackend, HierarchicalBackendCompletion, _strict_response
from roboharn_evo.agent.reflector.multimodal_transport import ImageEgressAuthorization, MultimodalImage, OpenAICompatibleMultimodalTransport


class RuntimeMultimodalBackend:
    def __init__(self, planner_url: str, *, model: str, reasoning_effort: str, max_images: int, timeout_sec: int) -> None:
        parts = urlsplit(planner_url)
        service_url = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        self.authorization = ImageEgressAuthorization.operator_granted(assertion="authorized HPK RGB retrieval and evidence maintenance", scope="task source and execution observations")
        self.transport = OpenAICompatibleMultimodalTransport(service_url=service_url, model=model, reasoning_effort=reasoning_effort, max_images=max_images, timeout_sec=timeout_sec)
        self.backend = ExistingGPTMultimodalReflectorBackend(self.transport, authorization=self.authorization)

    def complete(self, **kwargs):
        if self.transport.capability_report is None:
            self.transport.capability_preflight(authorization=self.authorization)
        return self.backend.complete(**kwargs)


def current_rgb_image(array: np.ndarray, *, camera: str) -> MultimodalImage:
    value = np.asarray(array)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("current RGB must be a uint8 array with three channels")
    buffer = BytesIO()
    Image.fromarray(value).save(buffer, format="PNG")
    return MultimodalImage(evidence_id=f"current {camera}", mime_type="image/png", content=buffer.getvalue(), detail="high")


class RGBRetrievalContext:
    def __init__(self, evidence: RGBEvidenceIndex, *, task_knowledge, action_knowledge, max_images: int = 32, camera_names=None) -> None:
        evidence.validate_knowledge(task_knowledge, action_knowledge)
        self.evidence = evidence
        self.knowledge = {"task": tuple(task_knowledge), "action": tuple(action_knowledge)}
        self.max_images = max_images
        self.camera_names = camera_names
        self.current_images = ()
        self.audit_root: Path | None = None
        self.request_count = 0

    def source_index(self, kind: str, candidate: Any) -> int:
        matches = [index for index, value in enumerate(self.knowledge[kind]) if value.to_dict() == candidate.to_dict()]
        if len(matches) != 1:
            raise ValueError("RGB retrieval candidate must resolve to one stored knowledge entry")
        return matches[0]

    def eligible(self, kind: str, candidate: Any) -> bool:
        return self.evidence.eligible(kind, self.source_index(kind, candidate))

    def set_current(self, images, *, audit_root: str | Path | None = None) -> None:
        if not images:
            raise ValueError("RGB retrieval requires current observations")
        self.current_images = tuple(images)
        if audit_root is not None:
            self.audit_root = Path(audit_root)

    def _records(self, kind: str, candidates, *, one_event: bool = False):
        records = []
        for local_index, candidate in enumerate(candidates):
            available = self.evidence.for_knowledge(kind, self.source_index(kind, candidate))
            # 每种实际 verdict 保留一个最近执行，批次记录对应的执行事件。
            chosen = []
            for verdict in ("support", "oppose", "unverified"):
                matches = [record for record in available if record["verdict"] == verdict and record.get("valid_for_current_knowledge", True)]
                if matches:
                    chosen.append(matches[-1])
            if one_event:
                chosen = chosen[:1]
            for record in chosen:
                value = copy.deepcopy(record)
                value["knowledge_index"] = local_index
                records.append(value)
        return records

    def complete(self, backend, *, kind: str, candidates, payload: dict, instructions: str, output_schema: dict, schema_name: str):
        if not self.current_images:
            raise ValueError("RGB retrieval has no current scene observation")
        if self.audit_root is None:
            raise ValueError("RGB retrieval requires an explicit request audit directory")
        candidate_key = "supported_task_knowledge" if kind == "task" else "supported_action_knowledge"
        histories, calls = [], []
        current_candidates = tuple(candidates)
        final_completion = None
        while current_candidates:
            comparison_round = bool(histories)
            records = self._records(kind, current_candidates, one_event=bool(histories))
            batches = tuple(self.evidence.image_batches(records, max_images=self.max_images, current_images=self.current_images, camera_names=self.camera_names))
            if not batches:
                raise ValueError("RGB retrieval candidates have no applicable image evidence")
            winners = {}
            for images, bindings in batches:
                visible = {item["knowledge_index"] for item in bindings if "knowledge_index" in item}
                request_payload = copy.deepcopy(payload)
                original_cards = payload[candidate_key]
                cards_by_source = {index: original_cards[candidates.index(candidate)]["knowledge"] for index, candidate in enumerate(current_candidates)}
                request_payload[candidate_key] = [{"source_index": index, "knowledge": cards_by_source[index]} for index in sorted(visible)]
                request_payload["image_bindings"] = bindings
                request_payload["earlier_visual_assessments"] = histories
                input_text = json.dumps(request_payload, ensure_ascii=False, separators=(",", ":"))
                directory = self.audit_root / f"request_{self.request_count:05d}"
                directory.mkdir(parents=True, exist_ok=False)
                self.request_count += 1
                image_paths = []
                for image_index, image in enumerate(images):
                    name = f"image_{image_index:03d}.png"
                    (directory / name).write_bytes(image.content)
                    image_paths.append(name)
                request_record = {"schema_name": schema_name, "instructions": instructions, "input": request_payload, "images": image_paths, "execution_events": [record["execution_event"] for record in records]}
                (directory / "request.json").write_text(json.dumps(request_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                completion = backend.complete(instructions=instructions + "\nUse the current scene and the paired historical images. Select only a source_index present in this batch. Image positions and candidate indices are temporary references.", input_text=input_text, images=images, output_schema=output_schema, schema_name=schema_name)
                result = _strict_response(completion.output)
                selected = result["selected_knowledge_index"]
                if selected is not None and (type(selected) is not int or selected not in visible):
                    raise ValueError("RGB retrieval selected a candidate absent from the image batch")
                (directory / "response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                calls.append({"directory": str(directory), "image_count": len(images), "input_characters": len(input_text), "provider_audit": completion.audit})
                final_completion = completion
                histories.append({"selected_knowledge": None if selected is None else cards_by_source[selected], "reason": result["reason"]})
                if selected is not None:
                    winners[selected] = (current_candidates[selected], result)
            if not winners:
                break
            if len(winners) == 1:
                candidate, result = next(iter(winners.values()))
                result["selected_knowledge_index"] = candidates.index(candidate)
                final_completion = HierarchicalBackendCompletion(output=result)
                break
            next_candidates = tuple(value[0] for value in winners.values())
            if comparison_round and len(next_candidates) >= len(current_candidates):
                raise ValueError("service image capacity cannot compare the selected knowledge entries")
            current_candidates = next_candidates
        if final_completion is None:
            raise ValueError("RGB retrieval completed no model request")
        return HierarchicalBackendCompletion(output=final_completion.output, audit={"retrieval_calls": len(calls), "retrieval_input_characters": sum(call["input_characters"] for call in calls), "rgb_requests": calls})
