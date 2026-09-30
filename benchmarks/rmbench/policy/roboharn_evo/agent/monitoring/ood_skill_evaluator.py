from __future__ import annotations

import json
from typing import Any

from ..experience import normalize_semantic_tags
from ...models.backend_factory import build_ood_backend

from .signals import GRASP_LOST, MOTION_BLOCKED, OBJECT_NOT_VISIBLE, REQUIRES_REPLAN, SCENE_DRIFT_DETECTED


OOD_SCENARIO_TO_SIGNAL = {
    "object_not_visible": OBJECT_NOT_VISIBLE,
    "motion_blocked": MOTION_BLOCKED,
    "grasp_lost": GRASP_LOST,
    "scene_drift_detected": SCENE_DRIFT_DETECTED,
    "requires_replan": REQUIRES_REPLAN,
}


class OODSkillEvaluator:
    def __init__(self, *, skill_registry, backend_config: dict[str, Any] | None = None) -> None:
        self._skill_registry = skill_registry
        self._ood_backend = build_ood_backend(backend_config or {
            "backend": "agent_api",
            "agent_api": {
                "server_url": "http://127.0.0.1:9101/ood",
                "timeout_sec": 120,
                "prompt_template": (
                    "You are the OOD detector VLM in a robot manipulation system.\n"
                    "Given the structured skill payload below, return JSON only with this schema:\n"
                    "{\n"
                    '  \"OOD_scenario\": \"none | object_not_visible | motion_blocked | grasp_lost | scene_drift_detected | requires_replan\",\n'
                    '  \"reason\": \"short evidence-based explanation\",\n'
                    '  \"confidence\": 0.0,\n'
                    '  \"semantic_tags\": {\n'
                    '    \"task_family\": \"pick_and_place | open_drawer | close_drawer | articulated_object | tool_use | other\",\n'
                    '    \"subtask_type\": \"grasp | place | open | close | move | align | reobserve | recover | other\",\n'
                    '    \"state_tags\": {\"object_state\": \"\", \"visibility_state\": \"\", \"gripper_state\": \"\", \"motion_state\": \"\"}\n'
                    "  }\n"
                    "}\n\n"
                    "Do not output summaries for the runtime to classify later.\n"
                    "semantic_tags is optional; leave uncertain state tag values as empty strings.\n"
                    "Skill payload: {skill_payload}"
                ),
                "auth_token": "",
                "auth_header": "Authorization",
                "extra_headers": {},
                "extra_body": {},
            },
        })

    def _normalize_ood_scenario(self, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        return normalized

    def _coerce_score(self, value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, score))

    def _build_signal(self, *, skill_name: str, scenario: str, reason: str, confidence: float, details: dict[str, Any] | None = None) -> dict[str, Any]:
        signal_name = OOD_SCENARIO_TO_SIGNAL[scenario]
        merged_details = {
            "source": skill_name,
            "mode": "structured-ood-scenario",
            "ood_scenario": scenario,
        }
        if details:
            merged_details.update(details)
        return {
            "name": signal_name,
            "level": "warning",
            "reason": reason,
            "score": confidence,
            "details": merged_details,
        }

    def evaluate(self, *, skill_payload: str) -> list[dict[str, Any]]:
        skill = self._skill_registry.get_skill("ood-detection", refresh=True)
        if skill is None:
            return []
        try:
            raw_result = self._ood_backend.evaluate_ood(skill_payload=skill_payload)
        except Exception:
            return []

        scenario = self._normalize_ood_scenario(raw_result.get("OOD_scenario", ""))
        if not scenario or scenario == "none":
            return []
        if scenario not in OOD_SCENARIO_TO_SIGNAL:
            return []

        reason = str(raw_result.get("reason", "")).strip() or f"VLM classified rollout as {scenario}"
        confidence = self._coerce_score(raw_result.get("confidence", raw_result.get("score", 0.0)))
        details: dict[str, Any] = {}
        if "analysis_note" in raw_result:
            details["analysis_note"] = raw_result.get("analysis_note")
        if isinstance(raw_result.get("semantic_tags"), dict):
            details["semantic_tags"] = normalize_semantic_tags(raw_result.get("semantic_tags"), default_source="ood_vlm")
        try:
            parsed_payload = json.loads(skill_payload)
        except json.JSONDecodeError:
            parsed_payload = {}
        if "current_subtask" in parsed_payload:
            details["current_subtask"] = parsed_payload.get("current_subtask")
        return [
            self._build_signal(
                skill_name=skill.name,
                scenario=scenario,
                reason=reason,
                confidence=confidence,
                details=details,
            )
        ]
