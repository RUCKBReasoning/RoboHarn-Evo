from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex, knowledge_with_evidence
from roboharn_evo.agent.hpk.semantic_consolidator import semantic_consolidation_json_schema
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import _strict_response


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


class RGBKnowledgeReviewer:
    def __init__(self, backend, *, max_images: int = 32, camera_names=None) -> None:
        self.backend = backend
        self.max_images = max_images
        self.camera_names = camera_names

    def review(self, *, kind: str, knowledge_index: int, knowledge, evidence: RGBEvidenceIndex, maintenance_rgb_enabled: bool, output_root: Path):
        if not isinstance(maintenance_rgb_enabled, bool):
            raise TypeError("maintenance_rgb_enabled must be boolean")
        records = copy.deepcopy(list(evidence.for_knowledge(kind, knowledge_index)))
        if not records:
            raise ValueError("knowledge review requires recorded executions")
        output_root.mkdir(parents=True, exist_ok=False)
        events = [{"event_index": index, "tested_knowledge": record["tested_knowledge"], "execution": record["execution"], "observed_result": record["observed_result"], "verdict": record["verdict"], "missing_evidence": record["missing_evidence"]} for index, record in enumerate(records)]
        canonical = {key: value for key, value in knowledge.to_dict().items() if key not in {"evidence_summary", "status"}}
        canonical_schema = semantic_consolidation_json_schema()["properties"][f"{kind}_groups"]["items"]["properties"]["canonical"]
        decision_schema = object_schema({
            "decision": {"type": "string", "enum": ["retain", "revise", "deactivate", "reinstate", "insufficient evidence"]},
            "canonical": canonical_schema,
            "reason": {"type": "string"},
        })
        batches = tuple(evidence.image_batches(records, max_images=self.max_images, camera_names=self.camera_names))
        if not batches:
            raise ValueError("knowledge review has no readable execution images")
        calls = []

        def complete(name, payload, images, schema, instructions):
            directory = output_root / f"call_{len(calls):03d}"
            directory.mkdir()
            paths = []
            for index, image in enumerate(images if maintenance_rgb_enabled else ()):
                path = directory / f"image_{index:03d}.png"
                path.write_bytes(image.content)
                paths.append(path.name)
            request_record = {"instructions": instructions, "input": payload, "schema_name": name, "images": paths}
            (directory / "request.json").write_text(json.dumps(request_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            completion = self.backend.complete(instructions=instructions, input_text=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), images=images if maintenance_rgb_enabled else (), output_schema=schema, schema_name=name)
            result = _strict_response(completion.output)
            (directory / "response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            calls.append({"request": str(directory / "request.json"), "response": str(directory / "response.json"), "images_sent": len(paths), "provider_audit": completion.audit})
            return result

        event_schema = object_schema({"event_index": {"type": "integer", "minimum": 0}, "applicable": {"type": "boolean"}, "verdict": {"type": "string", "enum": ["support", "oppose", "unverified"]}, "reason": {"type": "string"}})
        prior_assessments = []
        if len(batches) > 1:
            for images, bindings in batches:
                indices = sorted({item["event_index"] for item in bindings})
                observed = complete(
                    "hpk_rgb_execution_assessment",
                    {"knowledge_type": kind, "current_knowledge": canonical, "execution_evidence": [events[index] for index in indices], "image_bindings": bindings},
                    images, object_schema({"assessments": {"type": "array", "items": event_schema}}),
                    "Inspect every supplied execution before knowledge maintenance. Assess whether its actual condition and executed strategy match the current knowledge and whether its observed physical result supports, opposes, or leaves that knowledge unverified. Describe any condition or strategy distinction relevant to a revision. Judge task completion independently of individual action effects. Use the supplied text and any supplied images. Return exactly one assessment per supplied event index as strict JSON.",
                )
                if sorted(item["event_index"] for item in observed["assessments"]) != indices:
                    raise ValueError("maintenance did not inspect every supplied execution")
                prior_assessments.extend(observed["assessments"])
        first_images, first_bindings = batches[0]
        proposal = complete(
            "hpk_rgb_knowledge_review",
            {"knowledge_type": kind, "current_knowledge": canonical, "current_retrieval_enabled": evidence.eligible(kind, knowledge_index), "execution_evidence": events, "image_bindings": first_bindings, "all_batch_assessments": prior_assessments},
            first_images, decision_schema,
            "Review one historical manipulation knowledge entry using its actual execution evidence. Return retain, revise, deactivate, reinstate, or insufficient evidence. A revision may change applicability or strategy only when actual recorded behavior supports it. Keep the action primitive unchanged. Task knowledge is judged by its own completion condition; action knowledge by its physical effect. Treat explanations as inferred unless observed. For retain, deactivate, or insufficient evidence, copy the current canonical content exactly. A proposed revision or reinstatement will be checked against every execution separately. Use only supplied text and any supplied images; state uncertainty when evidence is incomplete. Indices refer only to this request. Return strict JSON.",
        )
        decision = proposal["decision"]
        revised = proposal["canonical"]
        for field in ("condition", "geometric_strategy"):
            if field in revised:
                revised[field] = {key: value for key, value in revised[field].items() if value is not None}
        if decision in {"retain", "deactivate", "insufficient evidence"} and revised != canonical:
            raise ValueError("maintenance decision changed content without a revision")
        if kind == "action" and revised["condition"]["action"] != canonical["condition"]["action"]:
            raise ValueError("knowledge review changed the action primitive")
        if decision != "insufficient evidence":
            assessments: dict[int, list[dict[str, Any]]] = {}
            for images, bindings in batches:
                indices = sorted({item["event_index"] for item in bindings})
                assessment = complete(
                    "hpk_rgb_revised_evidence_check",
                    {"knowledge_type": kind, "proposed_knowledge": revised, "execution_evidence": [events[index] for index in indices], "image_bindings": bindings},
                    images, object_schema({"assessments": {"type": "array", "items": event_schema}}),
                    "Check each supplied real execution against the proposed knowledge. Applicability requires that both its condition and its actually executed strategy match this knowledge. Support requires observed completion of that subtask or the expected local physical effect. Oppose requires applicable, completed execution and clear contradictory physical evidence. Otherwise return unverified. A pose reached or a closed gripper alone does not establish attachment. Later images may confirm an earlier execution; count it once. Return one assessment for every supplied event index, with a short factual reason. Use only the supplied text and any supplied images. Return strict JSON.",
                )
                results = assessment["assessments"]
                if sorted(item["event_index"] for item in results) != indices:
                    raise ValueError("maintenance did not assess each requested execution exactly once")
                for item in results:
                    assessments.setdefault(item["event_index"], []).append(item)
            for event_index, record in enumerate(records):
                values = assessments[event_index]
                applicable = all(item["applicable"] for item in values)
                verified = {item["verdict"] for item in values if item["verdict"] != "unverified"}
                record["valid_for_current_knowledge"] = applicable
                record["verdict"] = next(iter(verified)) if applicable and len(verified) == 1 else "unverified"
                record["review_reason"] = "; ".join(item["reason"] for item in values)
                if decision in {"revise", "reinstate"}:
                    record["retrieval_enabled"] = True
        if decision == "deactivate":
            for record in records:
                record["retrieval_enabled"] = False
        current = knowledge_with_evidence(revised, records, kind=kind)
        if decision == "reinstate" and current["status"] != "supported":
            raise ValueError("reinstatement requires supporting evidence for the current content")
        result_index = RGBEvidenceIndex(evidence.root, [*([record for record in evidence.records if (record["knowledge_type"], record["knowledge_index"]) != (kind, knowledge_index)]), *records])
        report = {"decision": decision, "reason": proposal["reason"], "maintenance_rgb_enabled": maintenance_rgb_enabled, "knowledge": current.to_dict(), "calls": calls}
        (output_root / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return current, result_index, report
