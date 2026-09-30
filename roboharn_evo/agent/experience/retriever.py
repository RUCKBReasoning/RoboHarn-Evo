from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from roboharn_evo.resources import skills_path

from .schemas import ACCEPTED_LESSON_STATUS, FAILURE_OUTCOMES, SUCCESS_OUTCOMES, normalize_ood_scenario, validate_retrieval_payload


@dataclass(slots=True)
class ExperienceQuery:
    OOD_scenario: str = ""
    task_family: str = ""
    subtask_type: str = ""
    object_state: str = ""
    visibility_state: str = ""
    gripper_state: str = ""
    motion_state: str = ""
    current_subtask: str = ""
    available_tools: list[str] = field(default_factory=list)
    tag_source: str = "unknown"

    @property
    def tags(self) -> set[str]:
        values = {
            self.OOD_scenario,
            self.task_family,
            self.subtask_type,
            self.object_state,
            self.visibility_state,
            self.gripper_state,
            self.motion_state,
        }
        return {str(value).strip() for value in values if str(value).strip()}


class ExperienceRetriever:
    """Deterministic v1 retriever for compact recovery experience.

    It intentionally avoids vector search. Accepted lessons and compact case
    summaries can be injected into recovery planning; raw traces stay outside
    runtime prompts.
    """

    def __init__(self, experience_root: str | Path | None = None) -> None:
        self.experience_root = (
            Path(experience_root).expanduser().resolve()
            if experience_root is not None
            else skills_path() / "experience"
        )
        self.index_path = self.experience_root / "retrieval-index" / "index.jsonl"

    def retrieve(
        self,
        query: ExperienceQuery,
        *,
        top_k_lessons: int = 3,
        top_k_cases: int = 2,
    ) -> dict[str, Any]:
        records = self._load_index_records()
        scored = [
            (self._score_record(record, query), record)
            for record in records
        ]
        scored = [(score, record) for score, record in scored if score > 0]
        scored.sort(key=lambda item: (-item[0], str(item[1].get("case_id", ""))))

        lessons: list[dict[str, Any]] = []
        cases: list[dict[str, Any]] = []
        avoid_patterns: list[dict[str, Any]] = []
        seen_lessons: set[str] = set()
        seen_cases: set[str] = set()

        for score, record in scored:
            if len(cases) < top_k_cases:
                case = self._load_case_summary(record, score=score)
                case_id = str(case.get("case_id", record.get("case_id", "")))
                if case and case_id not in seen_cases:
                    cases.append(case)
                    seen_cases.add(case_id)
            if self._is_avoid_pattern(record):
                avoid_patterns.append(self._compact_avoid_pattern(record, score=score))
            for lesson_path in record.get("lesson_paths", []) or []:
                if len(lessons) >= top_k_lessons:
                    break
                lesson = self._load_lesson(lesson_path, score=score)
                lesson_id = str(lesson.get("lesson_id", lesson.get("path", "")))
                if lesson and lesson_id not in seen_lessons:
                    lessons.append(lesson)
                    seen_lessons.add(lesson_id)
            if len(lessons) >= top_k_lessons and len(cases) >= top_k_cases:
                break

        payload = {
            "query": {
                "OOD_scenario": normalize_ood_scenario(query.OOD_scenario),
                "task_family": query.task_family,
                "subtask_type": query.subtask_type,
                "object_state": query.object_state,
                "visibility_state": query.visibility_state,
                "gripper_state": query.gripper_state,
                "motion_state": query.motion_state,
                "tag_source": query.tag_source,
            },
            "lessons": lessons[:top_k_lessons],
            "similar_cases": cases[:top_k_cases],
            "avoid_patterns": avoid_patterns[:top_k_cases],
        }
        validate_retrieval_payload(payload)
        return payload

    def _load_index_records(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    def _score_record(self, record: dict[str, Any], query: ExperienceQuery) -> int:
        score = 0
        scenario = normalize_ood_scenario(record.get("OOD_scenario", ""))
        query_scenario = normalize_ood_scenario(query.OOD_scenario)
        if query_scenario != "none" and scenario == query_scenario:
            score += 4
        if query.task_family and record.get("task_family") == query.task_family:
            score += 2
        if query.subtask_type and record.get("subtask_type") == query.subtask_type:
            score += 2
        record_tags = {str(tag).strip() for tag in record.get("retrieval_tags", []) if str(tag).strip()}
        score += len(record_tags & query.tags)
        outcome = str(record.get("outcome", "")).strip()
        support_count = self._safe_int(record.get("support_count", 0))
        opposing_count = self._safe_int(record.get("opposing_count", 0))
        failure_penalty = self._safe_int(record.get("failure_penalty", 0))
        confidence = str(record.get("confidence", "")).strip()
        if outcome in SUCCESS_OUTCOMES:
            score += 1
        if self._is_avoid_pattern(record):
            score += 1
        score += min(support_count, 4)
        if confidence == "high":
            score += 2
        elif confidence == "medium":
            score += 1
        score -= min(opposing_count + failure_penalty, 4)
        if self._is_avoid_pattern(record):
            score = max(score, 1)
        return score

    def _safe_int(self, value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _is_avoid_pattern(self, record: dict[str, Any]) -> bool:
        outcome = str(record.get("outcome", "")).strip()
        return bool(record.get("avoid_pattern", False)) or outcome in FAILURE_OUTCOMES

    def _resolve_experience_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (self.experience_root / path).resolve()
        return path

    def _load_case_summary(self, record: dict[str, Any], *, score: int) -> dict[str, Any]:
        path_value = str(record.get("summary_path", "")).strip()
        text = ""
        path = ""
        if path_value:
            summary_path = self._resolve_experience_path(path_value)
            path = str(summary_path)
            if summary_path.exists():
                text = self._truncate(summary_path.read_text(encoding="utf-8"), max_chars=900)
        return {
            "case_id": str(record.get("case_id", "")),
            "score": score,
            "OOD_scenario": normalize_ood_scenario(record.get("OOD_scenario", "")),
            "task_family": str(record.get("task_family", "")),
            "subtask_type": str(record.get("subtask_type", "")),
            "workflow": str(record.get("recovery_workflow", "")),
            "outcome": str(record.get("outcome", "")),
            "grounding_summary": dict(record.get("grounding_summary", {}) or {}),
            "summary": text,
            "path": path,
        }

    def _load_lesson(self, raw_path: str, *, score: int) -> dict[str, Any]:
        path = self._resolve_experience_path(raw_path)
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8")
        status = self._extract_lesson_status(text)
        if status != ACCEPTED_LESSON_STATUS:
            return {}
        return {
            "lesson_id": path.stem,
            "score": score,
            "status": status,
            "text": self._truncate(text, max_chars=900),
            "path": str(path),
        }

    def _extract_lesson_status(self, text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("- status:"):
                return stripped.split(":", 1)[1].strip().strip("`").lower()
            if stripped.lower().startswith("status:"):
                return stripped.split(":", 1)[1].strip().strip("`").lower()
        return ""

    def _compact_avoid_pattern(self, record: dict[str, Any], *, score: int) -> dict[str, Any]:
        return {
            "case_id": str(record.get("case_id", "")),
            "score": score,
            "OOD_scenario": normalize_ood_scenario(record.get("OOD_scenario", "")),
            "workflow": str(record.get("recovery_workflow", "")),
            "tool_sequence": list(record.get("recovery_primitives", []) or record.get("tool_sequence", []) or []),
            "outcome": str(record.get("outcome", "")),
            "support_count": self._safe_int(record.get("support_count", 0)),
            "opposing_count": self._safe_int(record.get("opposing_count", 0)),
            "failure_penalty": self._safe_int(record.get("failure_penalty", 0)),
            "grounding_summary": dict(record.get("grounding_summary", {}) or {}),
            "notes": self._truncate(str(record.get("notes", "")), max_chars=240),
        }

    def _truncate(self, text: str, *, max_chars: int) -> str:
        compact = "\n".join(line.rstrip() for line in text.strip().splitlines() if line.strip())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."
