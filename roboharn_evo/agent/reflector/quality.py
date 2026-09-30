"""Deterministic admission and leakage gates for Phase A2 reflection.

The model response is never trusted.  This module validates its complete
shape, resolves every cited item against the supplied request, keeps private
episode facts outside the planner-facing candidate, and emits an auditable
abstention instead of silently falling back to scripted guidance.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import re
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .schemas import (
    ProcedureExperienceV1,
    SchemaValidationError,
    _find_forbidden_transfer_fields,
    stable_experience_id,
    validate_reflector_candidate,
)
from .whole_trajectory import (
    UntrustedReflectionOutput,
    WholeTrajectoryReflectionRequest,
    WholeTrajectoryReflectionResult,
    annotation_evidence_ref,
)


_OUTPUT_SCHEMA = "roboharn_evo/whole_trajectory_reflection_output/v1"
_ABSTENTION_SCHEMA = "roboharn_evo/reflection_abstention/v1"
_CONFIDENCE = frozenset({"low", "medium", "high"})
_SOURCE_TYPES = frozenset(
    {
        "observed_visual",
        "observed_robot_state",
        "annotation",
        "cross_segment_inference",
    }
)
_TEMPORAL_SCOPES = frozenset({"single_frame", "before_after", "cross_segment"})
_PREDICTION_STATUSES = frozenset({"evidence_supported", "pending_validation"})
_MAX_RAW_BYTES = 256 * 1024
_MAX_STRING_CHARS = 4096
_MAX_COLLECTION_ITEMS = 256

# Conservative capability boundary, not a task-answer or paraphrase matcher.
# The current evidence schema explicitly declares that depth/object pose/contact
# may be unavailable, so exact physical quantities and proven-contact claims are
# never transferable from RGB-only evidence. Semantic-equivalence leakage is
# handled by the independent critic instead of expanding this list per fixture.
_PRECISE_PHYSICS_RE = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|deg(?:ree)?s?|rad(?:ian)?s?|n(?:ewtons?)?)\b"
    r"|exact\s+(?:insertion\s+depth|pose|contact|collision)"
    r"|precise\s+(?:insertion\s+depth|pose|contact|collision)"
    r"|proven\s+(?:physical\s+)?stability"
    r"|精确(?:插入深度|物体位姿|位姿|碰撞状态)"
    r"|完整接触关系|已证明物理稳定"
    r")",
    re.IGNORECASE,
)
_RAW_SHA256_VALUE_RE = re.compile(
    r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])"
)
_RUNTIME_PRIVATE_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:track|candidate|geometry|instance|oracle)"
    r"(?:[_-]?(?:id|uid|uuid))?[_-][A-Za-z0-9_.-]{4,}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_REQUEST_ALIAS_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:seg|ann|ev)_\d{3}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_COORDINATE_VECTOR_RE = re.compile(
    r"[\[(]\s*[+-]?\d+(?:\.\d+)?\s*,\s*"
    r"[+-]?\d+(?:\.\d+)?\s*,\s*"
    r"[+-]?\d+(?:\.\d+)?(?:\s*,\s*[+-]?\d+(?:\.\d+)?){0,4}\s*[\])]"
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_text(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _all_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for child in value.values():
            result.extend(_all_text(child))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for child in value:
            result.extend(_all_text(child))
        return result
    return []


def _deduplicate(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One deterministic reason a backend response cannot be admitted."""

    code: str
    offending_field: str
    detail: str
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "offending_field": self.offending_field,
            "detail": self.detail,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class ReflectionAbstentionAudit:
    """Persistable, raw-output-free audit record for one rejection."""

    trajectory_id: str
    issues: tuple[QualityIssue, ...]
    backend: str
    prompt_template_hash: str
    output_sha256: str
    created_at: str

    @property
    def reason(self) -> str:
        return self.issues[0].code if self.issues else "unknown_rejection"

    def to_dict(self) -> dict[str, Any]:
        first = self.issues[0] if self.issues else None
        refs = _deduplicate(
            tuple(ref for issue in self.issues for ref in issue.evidence_refs)
        )
        return {
            "schema": _ABSTENTION_SCHEMA,
            "schema_version": 1,
            "trajectory_id": self.trajectory_id,
            "status": "abstained",
            "reason": self.reason,
            "offending_field": "" if first is None else first.offending_field,
            "evidence_refs": list(refs),
            "issues": [issue.to_dict() for issue in self.issues],
            "backend": self.backend,
            "prompt_template_hash": self.prompt_template_hash,
            "output_sha256": self.output_sha256,
            "created_at": self.created_at,
        }


@runtime_checkable
class EpisodeSpecificLeakageCritic(Protocol):
    """Independent second-layer audit for natural-language answer leakage."""

    def audit(
        self,
        transferable_guidance: Mapping[str, Any],
        episode_specific_facts: Sequence[Mapping[str, Any]],
    ) -> "LeakageCriticResult": ...


@dataclass(frozen=True, slots=True)
class LeakageCriticResult:
    """Hash-bound result of the independent leakage-auditor stage."""

    completed: bool
    issues: tuple[QualityIssue, ...]
    auditor: str
    prompt_template_hash: str
    output_sha256: str

    @property
    def passed(self) -> bool:
        return self.completed and not self.issues


class DeterministicLeakageCritic:
    """Conservative reproducible critic for defined leakage attack classes.

    It complements structural isolation; it does not claim to solve arbitrary
    semantic equivalence.  It only detects exact private claims and structured
    values after normalization.  Paraphrase detection belongs to the separate
    semantic critic, not to a production synonym list.
    """

    def audit(
        self,
        transferable_guidance: Mapping[str, Any],
        episode_specific_facts: Sequence[Mapping[str, Any]],
    ) -> LeakageCriticResult:
        private_facts = [
            fact
            for fact in episode_specific_facts
            if fact.get("transferable_allowed") is False
        ]
        if not private_facts:
            return self._result(())

        transfer_text = "\n".join(_all_text(transferable_guidance))
        normalized_transfer = _normalize_text(transfer_text)
        protected_phrases: list[str] = []
        for fact in private_facts:
            claim = fact.get("claim")
            if isinstance(claim, str):
                protected_phrases.append(claim)
            structured_value = fact.get("structured_value")
            if isinstance(structured_value, str) and structured_value.strip():
                protected_phrases.append(structured_value)
        for phrase in protected_phrases:
            normalized = _normalize_text(phrase)
            if len(normalized) >= 8 and normalized in normalized_transfer:
                return self._result(
                    (
                        QualityIssue(
                        code="episode_specific_value_leakage",
                        offending_field="transferable_guidance",
                        detail="transferable text contains a protected episode-specific phrase",
                    ),
                    )
                )
        return self._result(())

    @staticmethod
    def _result(issues: tuple[QualityIssue, ...]) -> LeakageCriticResult:
        prompt_hash = hashlib.sha256(
            b"tcm-deterministic-episode-specific-leakage-critic-v1"
        ).hexdigest()
        output_payload = [issue.to_dict() for issue in issues]
        output_hash = hashlib.sha256(_canonical(output_payload).encode("utf-8")).hexdigest()
        return LeakageCriticResult(
            completed=True,
            issues=issues,
            auditor="deterministic_episode_specific_leakage_critic_v1",
            prompt_template_hash=prompt_hash,
            output_sha256=output_hash,
        )


class ReflectionQualityGate:
    """Apply the complete deterministic Phase A2 candidate admission policy."""

    def __init__(
        self,
        *,
        leakage_critic: EpisodeSpecificLeakageCritic | None = None,
        max_raw_bytes: int = _MAX_RAW_BYTES,
        max_string_chars: int = _MAX_STRING_CHARS,
        max_collection_items: int = _MAX_COLLECTION_ITEMS,
    ) -> None:
        if max_raw_bytes <= 0 or max_string_chars <= 0 or max_collection_items <= 0:
            raise ValueError("quality gate budgets must be positive")
        self._critic = leakage_critic or DeterministicLeakageCritic()
        self._max_raw_bytes = max_raw_bytes
        self._max_string_chars = max_string_chars
        self._max_collection_items = max_collection_items

    def backend_abstention(
        self,
        request: WholeTrajectoryReflectionRequest,
        *,
        backend: str,
        reason: str,
        detail: str = "",
    ) -> WholeTrajectoryReflectionResult:
        prompt_hash = hashlib.sha256(b"not-applicable").hexdigest()
        output_hash = hashlib.sha256(b"").hexdigest()
        audit = self._audit(
            request,
            (
                QualityIssue(
                    code=reason,
                    offending_field="backend",
                    detail=detail or reason,
                ),
            ),
            backend=str(backend),
            prompt_template_hash=prompt_hash,
            output_sha256=output_hash,
        )
        return WholeTrajectoryReflectionResult(
            candidate_experience=None,
            episode_specific_facts=(),
            attributions=(),
            abstention=audit.to_dict(),
            output_sha256=output_hash,
        )

    def evaluate(
        self,
        request: WholeTrajectoryReflectionRequest,
        output: UntrustedReflectionOutput,
        *,
        producer_name: str = "whole_trajectory_reflector",
        producer_version: str = "1",
    ) -> WholeTrajectoryReflectionResult:
        issues: list[QualityIssue] = []
        raw_size = len(output.raw_text.encode("utf-8"))
        if raw_size > self._max_raw_bytes:
            issues.append(
                QualityIssue(
                    code="malformed_output",
                    offending_field="backend_output",
                    detail=f"raw output exceeds {self._max_raw_bytes} bytes",
                )
            )

        capabilities = output.capabilities
        if not (
            capabilities.get("image_input_supported") is True
            and capabilities.get("image_input_acknowledged") is True
            and capabilities.get("structured_output_supported") is True
            and capabilities.get("text_only_fallback") is False
            and capabilities.get("scripted_backend") is False
        ):
            issues.append(
                QualityIssue(
                    code="backend_not_semantic_multimodal",
                    offending_field="backend.capabilities",
                    detail="semantic admission requires acknowledged images and structured output",
                )
            )

        if output.parsed is None:
            issues.append(
                QualityIssue(
                    code="malformed_output",
                    offending_field="backend_output",
                    detail=output.parse_error or "backend output is not a JSON object",
                )
            )
            return self._rejected(request, output, issues)

        payload = output.parsed_dict()
        assert payload is not None
        issues.extend(self._budget_issues(payload))
        shape_issue = self._validate_output_shape(payload)
        if shape_issue is not None:
            issues.append(shape_issue)
            return self._rejected(request, output, issues)

        facts = tuple(deepcopy(payload["episode_specific_facts"]))
        attributions = tuple(deepcopy(payload["attributions"]))
        if payload["status"] == "abstained":
            issues.append(
                QualityIssue(
                    code="backend_abstained",
                    offending_field="backend_output.abstention_reason",
                    detail=str(payload["abstention_reason"]),
                    evidence_refs=tuple(
                        ref
                        for ref in payload["supporting_evidence_refs"]
                        if isinstance(ref, str)
                    ),
                )
            )
            return self._rejected(
                request,
                output,
                issues,
                episode_specific_facts=facts,
                attributions=attributions,
            )

        transferable = deepcopy(payload["transferable_guidance"])
        assert isinstance(transferable, Mapping)
        try:
            issues.extend(self._semantic_issues(request, payload))
        except Exception as exc:
            # Every backend value is untrusted even after strict JSON parsing.
            # A malformed nested list/object must become an auditable abstention,
            # never an exception that aborts the complete offline batch.
            issues.append(
                QualityIssue(
                    code="malformed_output",
                    offending_field="backend_output",
                    detail=(
                        "nested semantic validation failed safely "
                        f"({type(exc).__name__})"
                    ),
                )
            )
        try:
            critic_result = self._critic.audit(transferable, facts)
        except Exception as exc:
            critic_result = LeakageCriticResult(
                completed=False,
                issues=(
                    QualityIssue(
                        code="leakage_critic_unavailable",
                        offending_field="transferable_guidance",
                        detail=f"critic raised {type(exc).__name__}",
                    ),
                ),
                auditor=self._critic.__class__.__name__,
                prompt_template_hash=hashlib.sha256(b"unavailable").hexdigest(),
                output_sha256=hashlib.sha256(b"").hexdigest(),
            )
        if not isinstance(critic_result, LeakageCriticResult):
            issues.append(
                QualityIssue(
                    code="leakage_critic_unavailable",
                    offending_field="transferable_guidance",
                    detail="critic returned an invalid result envelope",
                )
            )
        else:
            issues.extend(critic_result.issues)
            if not critic_result.completed:
                issues.append(
                    QualityIssue(
                        code="leakage_critic_unavailable",
                        offending_field="transferable_guidance",
                        detail="critic did not complete",
                    )
                )

        if issues:
            return self._rejected(
                request,
                output,
                issues,
                episode_specific_facts=facts,
                attributions=attributions,
            )

        candidate = self._build_candidate(
            request,
            output,
            payload,
            producer_name=producer_name,
            producer_version=producer_version,
        )
        try:
            record = ProcedureExperienceV1.from_dict(
                candidate,
                reflector_output=True,
            )
            validate_reflector_candidate(record)
            candidate = record.to_dict()
        except (SchemaValidationError, TypeError, ValueError) as exc:
            return self._rejected(
                request,
                output,
                [
                    QualityIssue(
                        code="malformed_output",
                        offending_field="transferable_guidance",
                        detail=f"candidate schema rejected output: {exc}",
                    )
                ],
                episode_specific_facts=facts,
                attributions=attributions,
            )
        return WholeTrajectoryReflectionResult(
            candidate_experience=candidate,
            episode_specific_facts=facts,
            attributions=attributions,
            abstention=None,
            output_sha256=output.output_sha256,
        )

    def _rejected(
        self,
        request: WholeTrajectoryReflectionRequest,
        output: UntrustedReflectionOutput,
        issues: Sequence[QualityIssue],
        *,
        episode_specific_facts: Sequence[Mapping[str, Any]] = (),
        attributions: Sequence[Mapping[str, Any]] = (),
    ) -> WholeTrajectoryReflectionResult:
        unique: list[QualityIssue] = []
        seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
        for issue in issues:
            identity = (
                issue.code,
                issue.offending_field,
                issue.detail,
                issue.evidence_refs,
            )
            if identity not in seen:
                seen.add(identity)
                unique.append(issue)
        audit = self._audit(
            request,
            tuple(unique),
            backend=output.backend,
            prompt_template_hash=output.prompt_template_hash,
            output_sha256=output.output_sha256,
        )
        return WholeTrajectoryReflectionResult(
            candidate_experience=None,
            episode_specific_facts=tuple(deepcopy(list(episode_specific_facts))),
            attributions=tuple(deepcopy(list(attributions))),
            abstention=audit.to_dict(),
            output_sha256=output.output_sha256,
        )

    @staticmethod
    def _audit(
        request: WholeTrajectoryReflectionRequest,
        issues: tuple[QualityIssue, ...],
        *,
        backend: str,
        prompt_template_hash: str,
        output_sha256: str,
    ) -> ReflectionAbstentionAudit:
        created_at = str(request.provenance.get("created_at", ""))
        return ReflectionAbstentionAudit(
            trajectory_id=request.trajectory_id,
            issues=issues,
            backend=backend,
            prompt_template_hash=prompt_template_hash,
            output_sha256=output_sha256,
            created_at=created_at,
        )

    def _budget_issues(self, payload: Any) -> list[QualityIssue]:
        issues: list[QualityIssue] = []

        def visit(value: Any, path: str) -> None:
            if isinstance(value, str):
                if len(value) > self._max_string_chars:
                    issues.append(
                        QualityIssue(
                            code="malformed_output",
                            offending_field=path,
                            detail=(
                                f"string exceeds {self._max_string_chars} characters"
                            ),
                        )
                    )
                return
            if isinstance(value, Mapping):
                if len(value) > self._max_collection_items:
                    issues.append(
                        QualityIssue(
                            code="malformed_output",
                            offending_field=path,
                            detail="object exceeds item budget",
                        )
                    )
                for key, child in value.items():
                    visit(child, f"{path}.{key}")
                return
            if isinstance(value, list):
                if len(value) > self._max_collection_items:
                    issues.append(
                        QualityIssue(
                            code="malformed_output",
                            offending_field=path,
                            detail="array exceeds item budget",
                        )
                    )
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")

        visit(payload, "backend_output")
        return issues

    @staticmethod
    def _validate_output_shape(payload: Mapping[str, Any]) -> QualityIssue | None:
        allowed = {
            "schema",
            "schema_version",
            "status",
            "abstention_reason",
            "summary",
            "confidence",
            "episode_specific_facts",
            "transferable_guidance",
            "attributions",
            "supporting_evidence_refs",
            "supporting_segment_refs",
        }
        if set(payload) != allowed:
            return QualityIssue(
                code="malformed_output",
                offending_field="backend_output",
                detail="top-level fields must match the v1 output schema exactly",
            )
        if payload.get("schema") not in {
            _OUTPUT_SCHEMA, "tcm/whole_trajectory_reflection_output/v1"
        } or payload.get("schema_version") != 1:
            return QualityIssue(
                code="malformed_output",
                offending_field="backend_output.schema",
                detail="unknown reflection output schema",
            )
        status = payload.get("status")
        if not isinstance(status, str) or status not in {"candidate", "abstained"}:
            return QualityIssue(
                code="forbidden_candidate_status",
                offending_field="backend_output.status",
                detail="Reflector output must be candidate or explicitly abstained",
            )
        if not isinstance(payload.get("summary"), str) or not payload["summary"].strip():
            return QualityIssue(
                code="malformed_output",
                offending_field="backend_output.summary",
                detail="summary must be a non-empty string",
            )
        confidence = payload.get("confidence")
        if not isinstance(confidence, str) or confidence not in _CONFIDENCE:
            return QualityIssue(
                code="malformed_output",
                offending_field="backend_output.confidence",
                detail="unknown confidence",
            )
        for key in (
            "episode_specific_facts",
            "attributions",
            "supporting_evidence_refs",
            "supporting_segment_refs",
        ):
            if not isinstance(payload.get(key), list):
                return QualityIssue(
                    code="malformed_output",
                    offending_field=f"backend_output.{key}",
                    detail="must be an array",
                )
        reason = payload.get("abstention_reason")
        if not isinstance(reason, str) or (status == "abstained" and not reason.strip()):
            return QualityIssue(
                code="malformed_output",
                offending_field="backend_output.abstention_reason",
                detail="abstained output requires a non-empty reason",
            )
        if status == "candidate" and reason:
            return QualityIssue(
                code="malformed_output",
                offending_field="backend_output.abstention_reason",
                detail="candidate output must use an empty abstention_reason",
            )
        if status == "candidate" and not isinstance(
            payload.get("transferable_guidance"), Mapping
        ):
            return QualityIssue(
                code="malformed_output",
                offending_field="backend_output.transferable_guidance",
                detail="must be an object",
            )
        if status == "abstained" and payload.get("transferable_guidance") is not None:
            return QualityIssue(
                code="malformed_output",
                offending_field="backend_output.transferable_guidance",
                detail="abstained output must not include transferable guidance",
            )
        return None

    def _semantic_issues(
        self,
        request: WholeTrajectoryReflectionRequest,
        payload: Mapping[str, Any],
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        facts, fact_issues = self._validate_facts(payload["episode_specific_facts"])
        issues.extend(fact_issues)
        attributions, attribution_issues = self._validate_attributions(
            payload["attributions"]
        )
        issues.extend(attribution_issues)
        if not facts:
            issues.append(
                QualityIssue(
                    code="missing_episode_specific_facts",
                    offending_field="episode_specific_facts",
                    detail="candidate needs private evidence-backed episode facts",
                )
            )
        if not attributions:
            issues.append(
                QualityIssue(
                    code="missing_evidence_attribution",
                    offending_field="attributions",
                    detail="candidate needs evidence attribution records",
                )
            )

        transfer = payload["transferable_guidance"]
        if set(transfer) != {"condition", "guidance", "predicted_effects"}:
            issues.append(
                QualityIssue(
                    code="malformed_output",
                    offending_field="transferable_guidance",
                    detail="expected condition, guidance, and predicted_effects only",
                )
            )
            return issues
        condition = transfer.get("condition")
        guidance = transfer.get("guidance")
        effects = transfer.get("predicted_effects")
        if not isinstance(condition, Mapping) or not condition:
            issues.append(
                QualityIssue(
                    code="missing_applicability_condition",
                    offending_field="transferable_guidance.condition",
                    detail="applicability condition must be a non-empty object",
                )
            )
        elif set(condition) != {"task_family", "subtask_type", "preconditions"} or (
            not isinstance(condition.get("task_family"), str)
            or not condition["task_family"].strip()
            or not isinstance(condition.get("subtask_type"), str)
            or not condition["subtask_type"].strip()
            or not self._valid_string_list(condition.get("preconditions"), nonempty=False)
        ):
            issues.append(
                QualityIssue(
                    code="malformed_output",
                    offending_field="transferable_guidance.condition",
                    detail="condition fields do not match the strict generic schema",
                )
            )
        if not isinstance(guidance, Mapping):
            issues.append(
                QualityIssue(
                    code="malformed_output",
                    offending_field="transferable_guidance.guidance",
                    detail="guidance must be an object",
                )
            )
            return issues
        if set(guidance) != {"ordered_steps", "feedback_policy", "avoid"}:
            issues.append(
                QualityIssue(
                    code="malformed_output",
                    offending_field="transferable_guidance.guidance",
                    detail="guidance fields do not match the strict generic schema",
                )
            )
        if not isinstance(effects, list) or not effects:
            issues.append(
                QualityIssue(
                    code="empty_predicted_effects",
                    offending_field="transferable_guidance.predicted_effects",
                    detail="candidate predicted_effects must be non-empty",
                )
            )

        steps = guidance.get("ordered_steps")
        if not isinstance(steps, list) or not steps:
            issues.append(
                QualityIssue(
                    code="malformed_output",
                    offending_field="transferable_guidance.guidance.ordered_steps",
                    detail="ordered_steps must be a non-empty array",
                )
            )
        elif any(
            not isinstance(step, Mapping)
            or set(step) != {"action_pattern", "instruction"}
            or not isinstance(step.get("action_pattern"), str)
            or not step["action_pattern"].strip()
            or not isinstance(step.get("instruction"), str)
            or not step["instruction"].strip()
            for step in steps
        ):
            issues.append(
                QualityIssue(
                    code="malformed_output",
                    offending_field="transferable_guidance.guidance.ordered_steps",
                    detail="ordered step fields do not match the strict generic schema",
                )
            )
        avoid = guidance.get("avoid", [])
        if not isinstance(avoid, list):
            issues.append(
                QualityIssue(
                    code="malformed_output",
                    offending_field="transferable_guidance.guidance.avoid",
                    detail="avoid must be an array",
                )
            )
        elif not self._valid_string_list(avoid, nonempty=True):
            issues.append(
                QualityIssue(
                    code="missing_repeat_prevention",
                    offending_field="transferable_guidance.guidance.avoid",
                    detail="feedback guidance must include at least one explicit avoidance",
                )
            )

        policy = guidance.get("feedback_policy")
        if not isinstance(policy, Mapping):
            issues.extend(
                (
                    QualityIssue(
                        code="missing_observation_rule",
                        offending_field="transferable_guidance.guidance.feedback_policy",
                        detail="whole-trajectory candidate requires feedback observations",
                    ),
                    QualityIssue(
                        code="missing_failure_branch",
                        offending_field="transferable_guidance.guidance.feedback_policy",
                        detail="whole-trajectory candidate requires a failure branch",
                    ),
                    QualityIssue(
                        code="missing_termination_condition",
                        offending_field="transferable_guidance.guidance.feedback_policy",
                        detail="whole-trajectory candidate requires a termination condition",
                    ),
                )
            )
        else:
            expected_policy_keys = {
                "state_variables",
                "attempt_tracking",
                "observation_rules",
                "failure_branch",
                "success_branch",
                "termination_conditions",
            }
            if set(policy) != expected_policy_keys:
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=(
                            "transferable_guidance.guidance.feedback_policy"
                        ),
                        detail="feedback policy fields do not match the strict schema",
                    )
                )
            state_variables = policy.get("state_variables")
            if not self._valid_string_list(state_variables, nonempty=True):
                issues.append(
                    QualityIssue(
                        code="missing_state_tracking",
                        offending_field=(
                            "transferable_guidance.guidance.feedback_policy."
                            "state_variables"
                        ),
                        detail="feedback policy must declare maintained episode state",
                    )
                )
            tracking = policy.get("attempt_tracking")
            if (
                not isinstance(tracking, Mapping)
                or set(tracking)
                != {
                    "state_variable",
                    "record_after_attempt",
                    "exclude_previously_recorded",
                }
                or tracking.get("state_variable") not in (state_variables or [])
                or tracking.get("record_after_attempt") is not True
                or tracking.get("exclude_previously_recorded") is not True
            ):
                issues.append(
                    QualityIssue(
                        code="missing_repeat_prevention",
                        offending_field=(
                            "transferable_guidance.guidance.feedback_policy."
                            "attempt_tracking"
                        ),
                        detail=(
                            "typed attempt tracking must record each attempt and "
                            "exclude previously recorded attempts"
                        ),
                    )
                )
            if not isinstance(policy.get("observation_rules"), list) or not policy.get(
                "observation_rules"
            ):
                issues.append(
                    QualityIssue(
                        code="missing_observation_rule",
                        offending_field=(
                            "transferable_guidance.guidance.feedback_policy."
                            "observation_rules"
                        ),
                        detail="feedback policy has no post-action observation rule",
                    )
                )
            elif any(
                not isinstance(rule, Mapping)
                or set(rule) != {"after_action_pattern", "observe_signal"}
                or not isinstance(rule.get("after_action_pattern"), str)
                or not rule["after_action_pattern"].strip()
                or not isinstance(rule.get("observe_signal"), str)
                or not rule["observe_signal"].strip()
                for rule in policy["observation_rules"]
            ):
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=(
                            "transferable_guidance.guidance.feedback_policy."
                            "observation_rules"
                        ),
                        detail="observation rule fields do not match the strict schema",
                    )
                )
            if not isinstance(policy.get("failure_branch"), list) or not policy.get(
                "failure_branch"
            ):
                issues.append(
                    QualityIssue(
                        code="missing_failure_branch",
                        offending_field=(
                            "transferable_guidance.guidance.feedback_policy.failure_branch"
                        ),
                        detail="feedback policy has no unsuccessful-outcome branch",
                    )
                )
            elif not self._valid_string_list(policy["failure_branch"], nonempty=True):
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=(
                            "transferable_guidance.guidance.feedback_policy.failure_branch"
                        ),
                        detail="failure_branch must contain unique non-empty actions",
                    )
                )
            if not isinstance(policy.get("success_branch"), list) or not policy.get(
                "success_branch"
            ):
                issues.append(
                    QualityIssue(
                        code="missing_success_branch",
                        offending_field=(
                            "transferable_guidance.guidance.feedback_policy.success_branch"
                        ),
                        detail="feedback policy has no successful-outcome branch",
                    )
                )
            elif not self._valid_string_list(policy["success_branch"], nonempty=True):
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=(
                            "transferable_guidance.guidance.feedback_policy.success_branch"
                        ),
                        detail="success_branch must contain unique non-empty actions",
                    )
                )
            if not isinstance(policy.get("termination_conditions"), list) or not policy.get(
                "termination_conditions"
            ):
                issues.append(
                    QualityIssue(
                        code="missing_termination_condition",
                        offending_field=(
                            "transferable_guidance.guidance.feedback_policy."
                            "termination_conditions"
                        ),
                        detail="feedback policy has no termination condition",
                    )
                )
            elif not self._valid_string_list(
                policy["termination_conditions"], nonempty=True
            ):
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=(
                            "transferable_guidance.guidance.feedback_policy."
                            "termination_conditions"
                        ),
                        detail="termination conditions must be unique non-empty strings",
                    )
                )

        if isinstance(effects, list) and effects and any(
            not isinstance(effect, Mapping)
            or set(effect) != {"effect", "validation_status"}
            or not isinstance(effect.get("effect"), str)
            or not effect["effect"].strip()
            or effect.get("validation_status")
            not in {"evidence_supported", "pending_validation"}
            for effect in effects
        ):
            issues.append(
                QualityIssue(
                    code="malformed_output",
                    offending_field="transferable_guidance.predicted_effects",
                    detail="predicted effect fields do not match the strict schema",
                )
            )

        private_fields = _find_forbidden_transfer_fields(
            transfer,
            path="transferable_guidance",
        )
        if private_fields:
            issues.append(
                QualityIssue(
                    code="runtime_private_field",
                    offending_field=private_fields[0],
                    detail="transferable guidance contains runtime-private data",
                )
            )
        request_private_values: list[str] = [request.trajectory_id]
        request_private_values.extend(
            str(segment["segment_id"]) for segment in request.segments
        )
        request_private_values.extend(
            str(value["evidence_ref"])
            for value in request.annotation_evidence_refs
        )
        visual_items = request.visual_evidence.get("items", [])
        if isinstance(visual_items, list):
            for item in visual_items:
                if not isinstance(item, Mapping):
                    continue
                for key in ("evidence_id", "comparison_id"):
                    value = item.get(key)
                    if isinstance(value, str):
                        request_private_values.append(value)
                frame = item.get("frame")
                if isinstance(frame, Mapping):
                    value = frame.get("evidence_ref")
                    if isinstance(value, str):
                        request_private_values.append(value)
        comparisons = request.visual_evidence.get("comparisons", [])
        if isinstance(comparisons, list):
            for comparison in comparisons:
                if isinstance(comparison, Mapping):
                    value = comparison.get("comparison_id")
                    if isinstance(value, str):
                        request_private_values.append(value)
        normalized_private_values = {
            normalized
            for value in request_private_values
            if len(normalized := _normalize_text(value)) >= 8
        }
        for path, text in self._text_paths(transfer, "transferable_guidance"):
            if _PRECISE_PHYSICS_RE.search(text):
                issues.append(
                    QualityIssue(
                        code="unsupported_precise_physical_claim",
                        offending_field=path,
                        detail="claim exceeds the available visual/robot-state evidence",
                    )
                )
                break
            normalized_text = _normalize_text(text)
            if (
                _RAW_SHA256_VALUE_RE.search(text)
                or _RUNTIME_PRIVATE_VALUE_RE.search(text)
                or _REQUEST_ALIAS_VALUE_RE.search(text)
                or any(
                    private_value in normalized_text
                    for private_value in normalized_private_values
                )
            ):
                issues.append(
                    QualityIssue(
                        code="runtime_private_value",
                        offending_field=path,
                        detail=(
                            "transferable guidance contains a request-local or "
                            "runtime-private value"
                        ),
                    )
                )
                break
            if _COORDINATE_VECTOR_RE.search(text):
                issues.append(
                    QualityIssue(
                        code="unsupported_precise_physical_claim",
                        offending_field=path,
                        detail=(
                            "transferable guidance contains an unsupported "
                            "coordinate-like numeric vector"
                        ),
                    )
                )
                break
        # ``uncertainty`` is where the backend must state limitations such as
        # “precise pose is unavailable”.  Scan asserted claims only; treating a
        # negative capability caveat as a physical claim would invert the
        # epistemic contract and reject appropriately cautious output.
        asserted_claims = [
            *(
                (f"episode_specific_facts[{index}].claim", str(fact["claim"]))
                for index, fact in enumerate(facts)
            ),
            *(
                (f"attributions[{index}].claim", str(attribution["claim"]))
                for index, attribution in enumerate(attributions)
            ),
        ]
        for path, text in asserted_claims:
            if _PRECISE_PHYSICS_RE.search(text):
                issues.append(
                    QualityIssue(
                        code="unsupported_precise_physical_claim",
                        offending_field=path,
                        detail="claim exceeds the available visual/robot-state evidence",
                    )
                )
                break

        (
            valid_evidence,
            evidence_roles,
            evidence_source_types,
            robot_state_values,
            evidence_segments,
        ) = self._evidence_registry(request)
        visual_evidence = {
            ref
            for ref, source_types in evidence_source_types.items()
            if "observed_visual" in source_types
        }
        robot_state_evidence = {
            ref
            for ref, source_types in evidence_source_types.items()
            if "observed_robot_state" in source_types
        }
        valid_segments = {str(segment["segment_id"]) for segment in request.segments}
        cited_refs: list[str] = []
        cited_segments: list[str] = []
        for field_name in ("supporting_evidence_refs", "supporting_segment_refs"):
            values = payload[field_name]
            if any(not isinstance(value, str) or not value.strip() for value in values):
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=f"backend_output.{field_name}",
                        detail="references must be non-empty strings",
                    )
                )
            if len(set(values)) != len(values):
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=f"backend_output.{field_name}",
                        detail="duplicate reference",
                    )
                )
        cited_refs.extend(
            value for value in payload["supporting_evidence_refs"] if isinstance(value, str)
        )
        cited_segments.extend(
            value for value in payload["supporting_segment_refs"] if isinstance(value, str)
        )
        for fact in facts:
            refs = fact["supporting_evidence_refs"]
            segments = fact["supporting_segment_refs"]
            cited_refs.extend(fact["supporting_evidence_refs"])
            cited_segments.extend(fact["supporting_segment_refs"])
            source_type = fact["source_type"]
            expected_source = source_type if source_type != "cross_segment_inference" else None
            if expected_source is not None and not all(
                expected_source in evidence_source_types.get(ref, set()) for ref in refs
            ):
                issues.append(
                    QualityIssue(
                        code="source_type_confusion",
                        offending_field=f"episode_specific_facts[{fact['fact_id']}]",
                        detail=f"{source_type} fact cites a different evidence source",
                        evidence_refs=tuple(refs),
                    )
                )
            fact_temporal_scope = fact["temporal_scope"]
            if source_type == "cross_segment_inference" and (
                len(set(segments)) < 2 or fact_temporal_scope != "cross_segment"
            ):
                issues.append(
                    QualityIssue(
                        code="source_type_confusion",
                        offending_field=f"episode_specific_facts[{fact['fact_id']}]",
                        detail="cross-segment fact must cite at least two segments",
                        evidence_refs=tuple(refs),
                    )
                )
            if source_type == "cross_segment_inference":
                cited_evidence_segments = {
                    evidence_segments[ref]
                    for ref in refs
                    if ref in evidence_segments
                }
                if (
                    len(cited_evidence_segments) < 2
                    or cited_evidence_segments != set(segments)
                ):
                    issues.append(
                        QualityIssue(
                            code="cross_segment_evidence_mismatch",
                            offending_field=f"episode_specific_facts[{fact['fact_id']}]",
                            detail=(
                                "cross-segment fact refs must resolve across exactly "
                                "the declared supporting segments"
                            ),
                            evidence_refs=tuple(refs),
                        )
                    )
            if source_type == "observed_robot_state":
                supported_values = {
                    value
                    for ref in refs
                    for value in robot_state_values.get(ref, set())
                }
                if fact["structured_value"] not in supported_values:
                    issues.append(
                        QualityIssue(
                            code="source_type_confusion",
                            offending_field=(
                                f"episode_specific_facts[{fact['fact_id']}]."
                                "structured_value"
                            ),
                            detail=(
                                "structured robot-state fact does not match the "
                                "cited item"
                            ),
                            evidence_refs=tuple(refs),
                        )
                    )
            if (
                source_type == "observed_visual"
                and fact_temporal_scope == "before_after"
            ):
                roles = {evidence_roles.get(ref) for ref in refs}
                if not {"before", "after"}.issubset(roles):
                    issues.append(
                        QualityIssue(
                            code="state_change_without_before_after_evidence",
                            offending_field=f"episode_specific_facts[{fact['fact_id']}]",
                            detail="visual state change needs both before and after evidence",
                            evidence_refs=tuple(refs),
                        )
                    )
        for attribution in attributions:
            refs = attribution["supporting_evidence_refs"]
            segments = attribution["supporting_segment_refs"]
            cited_refs.extend(refs)
            cited_segments.extend(segments)
            source_type = attribution["source_type"]
            temporal_scope = attribution["temporal_scope"]
            expected_source = source_type if source_type != "cross_segment_inference" else None
            if expected_source is not None and not all(
                expected_source in evidence_source_types.get(ref, set()) for ref in refs
            ):
                issues.append(
                    QualityIssue(
                        code="source_type_confusion",
                        offending_field=f"attributions[{attribution['attribution_id']}]",
                        detail=f"{source_type} attribution cites a different evidence source",
                        evidence_refs=tuple(refs),
                    )
                )
            if source_type == "observed_visual" and temporal_scope == "before_after":
                roles = {evidence_roles.get(ref) for ref in refs}
                if not {"before", "after"}.issubset(roles):
                    issues.append(
                        QualityIssue(
                            code="state_change_without_before_after_evidence",
                            offending_field=(
                                f"attributions[{attribution['attribution_id']}]"
                            ),
                            detail="visual state change needs both before and after evidence",
                            evidence_refs=tuple(refs),
                        )
                    )
            if source_type == "cross_segment_inference" and (
                temporal_scope != "cross_segment" or len(set(segments)) < 2
            ):
                issues.append(
                    QualityIssue(
                        code="source_type_confusion",
                        offending_field=f"attributions[{attribution['attribution_id']}]",
                        detail="cross-segment inference must cite at least two segments",
                        evidence_refs=tuple(refs),
                    )
                )
            if source_type == "cross_segment_inference":
                cited_evidence_segments = {
                    evidence_segments[ref]
                    for ref in refs
                    if ref in evidence_segments
                }
                if (
                    len(cited_evidence_segments) < 2
                    or cited_evidence_segments != set(segments)
                ):
                    issues.append(
                        QualityIssue(
                            code="cross_segment_evidence_mismatch",
                            offending_field=(
                                f"attributions[{attribution['attribution_id']}]"
                            ),
                            detail=(
                                "cross-segment attribution refs must resolve across "
                                "exactly the declared supporting segments"
                            ),
                            evidence_refs=tuple(refs),
                        )
                    )

        fact_source_types = {fact["source_type"] for fact in facts}
        if visual_evidence and "observed_visual" not in fact_source_types:
            issues.append(
                QualityIssue(
                    code="missing_visual_fact",
                    offending_field="episode_specific_facts",
                    detail="visual input exists but no observed_visual fact was produced",
                )
            )
        if visual_evidence and not any(
            fact["source_type"] == "observed_visual"
            and fact["temporal_scope"] == "before_after"
            for fact in facts
        ):
            issues.append(
                QualityIssue(
                    code="missing_visual_change_fact",
                    offending_field="episode_specific_facts",
                    detail=(
                        "visual trajectory evidence requires at least one "
                        "before_after observed_visual fact"
                    ),
                )
            )
        if robot_state_evidence and "observed_robot_state" not in fact_source_types:
            issues.append(
                QualityIssue(
                    code="missing_robot_state_fact",
                    offending_field="episode_specific_facts",
                    detail="gripper state exists but no observed_robot_state fact was produced",
                )
            )
        if not any(
            attribution["source_type"] == "cross_segment_inference"
            and len(set(attribution["supporting_segment_refs"])) >= 2
            for attribution in attributions
        ):
            issues.append(
                QualityIssue(
                    code="single_segment_strategy",
                    offending_field="attributions",
                    detail="candidate lacks a grounded cross-segment inference",
                )
            )

        invalid_refs = sorted(set(cited_refs) - valid_evidence)
        invalid_segments = sorted(set(cited_segments) - valid_segments)
        if invalid_refs or invalid_segments:
            issues.append(
                QualityIssue(
                    code="invalid_evidence_ref",
                    offending_field="backend_output",
                    detail="one or more evidence/segment refs do not resolve",
                    evidence_refs=tuple(invalid_refs),
                )
            )
        if len(set(cited_segments) & valid_segments) < 2:
            issues.append(
                QualityIssue(
                    code="single_segment_strategy",
                    offending_field="supporting_segment_refs",
                    detail="whole-trajectory strategy must use multiple segments",
                )
            )

        issues.extend(self._attribution_coverage_issues(transfer, attributions))
        issues.extend(self._effect_basis_issues(effects, attributions))
        issues.extend(self._annotation_copy_issues(request, payload, transfer, attributions))
        issues.extend(self._structural_leakage_issues(transfer, facts))

        return issues

    @staticmethod
    def _validate_facts(
        raw_facts: Sequence[Any],
    ) -> tuple[list[dict[str, Any]], list[QualityIssue]]:
        allowed = {
            "fact_id",
            "claim",
            "source_type",
            "supporting_evidence_refs",
            "supporting_segment_refs",
            "confidence",
            "uncertainty",
            "temporal_scope",
            "transferable_allowed",
            "structured_value",
        }
        required = allowed
        facts: list[dict[str, Any]] = []
        issues: list[QualityIssue] = []
        seen_ids: set[str] = set()
        for number, value in enumerate(raw_facts):
            path = f"episode_specific_facts[{number}]"
            if not isinstance(value, Mapping) or set(value) - allowed or required - set(value):
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=path,
                        detail="fact fields do not match schema",
                    )
                )
                continue
            fact = deepcopy(dict(value))
            fact_id = fact.get("fact_id")
            if not isinstance(fact_id, str) or not fact_id.strip() or fact_id in seen_ids:
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=f"{path}.fact_id",
                        detail="fact_id must be non-empty and unique",
                    )
                )
                continue
            seen_ids.add(fact_id)
            if (
                not isinstance(fact.get("claim"), str)
                or not fact["claim"].strip()
                or fact.get("source_type") not in _SOURCE_TYPES
                or fact.get("confidence") not in _CONFIDENCE
                or not isinstance(fact.get("uncertainty"), str)
                or fact.get("temporal_scope") not in _TEMPORAL_SCOPES
                or fact.get("transferable_allowed") is not False
                or not isinstance(fact.get("structured_value"), str)
                or not ReflectionQualityGate._valid_string_list(
                    fact.get("supporting_evidence_refs"), nonempty=True
                )
                or not ReflectionQualityGate._valid_string_list(
                    fact.get("supporting_segment_refs"), nonempty=True
                )
            ):
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=path,
                        detail="fact contains invalid typed fields",
                    )
                )
                continue
            facts.append(fact)
        return facts, issues

    @staticmethod
    def _validate_attributions(
        raw_attributions: Sequence[Any],
    ) -> tuple[list[dict[str, Any]], list[QualityIssue]]:
        allowed = {
            "attribution_id",
            "target_path",
            "claim",
            "source_type",
            "supporting_evidence_refs",
            "supporting_segment_refs",
            "confidence",
            "uncertainty",
            "temporal_scope",
            "prediction_status",
        }
        attributions: list[dict[str, Any]] = []
        issues: list[QualityIssue] = []
        seen_ids: set[str] = set()
        for number, value in enumerate(raw_attributions):
            path = f"attributions[{number}]"
            if not isinstance(value, Mapping) or set(value) != allowed:
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=path,
                        detail="attribution fields do not match schema",
                    )
                )
                continue
            attribution = deepcopy(dict(value))
            attribution_id = attribution.get("attribution_id")
            if (
                not isinstance(attribution_id, str)
                or not attribution_id.strip()
                or attribution_id in seen_ids
            ):
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=f"{path}.attribution_id",
                        detail="attribution_id must be non-empty and unique",
                    )
                )
                continue
            seen_ids.add(attribution_id)
            if (
                not isinstance(attribution.get("target_path"), str)
                or not attribution["target_path"].startswith("/")
                or not isinstance(attribution.get("claim"), str)
                or not attribution["claim"].strip()
                or attribution.get("source_type") not in _SOURCE_TYPES
                or attribution.get("confidence") not in _CONFIDENCE
                or not isinstance(attribution.get("uncertainty"), str)
                or attribution.get("temporal_scope") not in _TEMPORAL_SCOPES
                or attribution.get("prediction_status") not in _PREDICTION_STATUSES
                or not ReflectionQualityGate._valid_string_list(
                    attribution.get("supporting_segment_refs"), nonempty=True
                )
            ):
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=path,
                        detail="attribution contains invalid typed fields",
                    )
                )
                continue
            refs = attribution.get("supporting_evidence_refs")
            refs_may_be_empty = attribution["prediction_status"] == "pending_validation"
            if not ReflectionQualityGate._valid_string_list(
                refs,
                nonempty=not refs_may_be_empty,
            ):
                issues.append(
                    QualityIssue(
                        code="malformed_output",
                        offending_field=f"{path}.supporting_evidence_refs",
                        detail="evidence refs are invalid for prediction status",
                    )
                )
                continue
            attributions.append(attribution)
        return attributions, issues

    @staticmethod
    def _valid_string_list(value: Any, *, nonempty: bool) -> bool:
        return (
            isinstance(value, list)
            and (bool(value) or not nonempty)
            and all(isinstance(item, str) and bool(item.strip()) for item in value)
            and len(set(value)) == len(value)
        )

    @staticmethod
    def _evidence_registry(
        request: WholeTrajectoryReflectionRequest,
    ) -> tuple[
        set[str],
        dict[str, str],
        dict[str, set[str]],
        dict[str, set[str]],
        dict[str, str],
    ]:
        valid: set[str] = set()
        roles: dict[str, str] = {}
        source_types: dict[str, set[str]] = {}
        robot_values: dict[str, set[str]] = {}
        evidence_segments: dict[str, str] = {}
        visual = request.visual_evidence
        raw_items = visual.get("items", visual.get("evidence_items", []))
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, Mapping):
                    continue
                identifiers: list[str] = []
                evidence_id = item.get("evidence_id")
                if isinstance(evidence_id, str) and evidence_id:
                    identifiers.append(evidence_id)
                frame = item.get("frame")
                if isinstance(frame, Mapping):
                    evidence_ref = frame.get("evidence_ref")
                    if isinstance(evidence_ref, str) and evidence_ref:
                        identifiers.append(evidence_ref)
                role = item.get("temporal_role")
                segment_value = item.get("segment_id")
                for identifier in identifiers:
                    valid.add(identifier)
                    source_types.setdefault(identifier, set())
                    if isinstance(segment_value, str) and segment_value:
                        evidence_segments[identifier] = segment_value
                    image = item.get("image")
                    if (
                        isinstance(image, Mapping)
                        and image.get("source_type") == "observed_visual"
                        and isinstance(item.get("evidence_source"), str)
                        and item["evidence_source"]
                    ):
                        source_types[identifier].add("observed_visual")
                    if role in {"before", "after", "context"}:
                        roles[identifier] = str(role)
                    gripper = item.get("gripper_states")
                    action_range = item.get("action_range")
                    has_robot_state = (
                        isinstance(action_range, Mapping)
                        and action_range.get("source_type") == "observed_robot_state"
                    ) or (
                        isinstance(gripper, list)
                        and bool(gripper)
                        and all(
                            isinstance(state, Mapping)
                            and state.get("source_type") == "observed_robot_state"
                            for state in gripper
                        )
                    )
                    if has_robot_state:
                        source_types[identifier].add("observed_robot_state")
                    if isinstance(gripper, list):
                        for state in gripper:
                            if not isinstance(state, Mapping):
                                continue
                            channel_id = state.get("channel_id")
                            state_value = state.get("state")
                            if (
                                state.get("source_type") == "observed_robot_state"
                                and isinstance(channel_id, str)
                                and channel_id
                                and isinstance(state_value, str)
                                and state_value
                            ):
                                robot_values.setdefault(identifier, set()).add(
                                    f"{channel_id}={state_value}"
                                )
        for segment in request.segments:
            transition = segment.get("transition", {})
            derivation = segment.get("derivation", {})
            for key in ("before_event_refs", "action_event_refs", "after_event_refs"):
                refs = transition.get(key, []) if isinstance(transition, Mapping) else []
                if isinstance(refs, list):
                    for value in refs:
                        if isinstance(value, str):
                            valid.add(value)
                            evidence_segments[value] = str(segment["segment_id"])
            refs = (
                derivation.get("evidence_event_refs", [])
                if isinstance(derivation, Mapping)
                else []
            )
            if isinstance(refs, list):
                for value in refs:
                    if isinstance(value, str):
                        valid.add(value)
                        evidence_segments[value] = str(segment["segment_id"])
            segment_id = str(segment["segment_id"])
            annotation_ref = annotation_evidence_ref(segment_id)
            valid.add(annotation_ref)
            source_types.setdefault(annotation_ref, set()).add("annotation")
            evidence_segments[annotation_ref] = segment_id
        return valid, roles, source_types, robot_values, evidence_segments

    @staticmethod
    def _text_paths(value: Any, path: str) -> list[tuple[str, str]]:
        if isinstance(value, str):
            return [(path, value)]
        if isinstance(value, Mapping):
            result: list[tuple[str, str]] = []
            for key, child in value.items():
                result.extend(ReflectionQualityGate._text_paths(child, f"{path}.{key}"))
            return result
        if isinstance(value, list):
            result = []
            for index, child in enumerate(value):
                result.extend(
                    ReflectionQualityGate._text_paths(child, f"{path}[{index}]")
                )
            return result
        return []

    @staticmethod
    def _attribution_coverage_issues(
        transfer: Mapping[str, Any],
        attributions: Sequence[Mapping[str, Any]],
    ) -> list[QualityIssue]:
        guidance = transfer.get("guidance", {})
        effects = transfer.get("predicted_effects", [])
        required_paths: list[str] = ["/condition"]
        steps = guidance.get("ordered_steps", []) if isinstance(guidance, Mapping) else []
        if isinstance(steps, list):
            required_paths.extend(f"/guidance/ordered_steps/{index}" for index in range(len(steps)))
        policy = guidance.get("feedback_policy") if isinstance(guidance, Mapping) else None
        if isinstance(policy, Mapping):
            required_paths.extend(
                (
                    "/guidance/feedback_policy/state_variables",
                    "/guidance/feedback_policy/attempt_tracking",
                    "/guidance/feedback_policy/observation_rules",
                    "/guidance/feedback_policy/failure_branch",
                    "/guidance/feedback_policy/success_branch",
                    "/guidance/feedback_policy/termination_conditions",
                )
            )
        required_paths.append("/guidance/avoid")
        if isinstance(effects, list):
            required_paths.extend(
                f"/predicted_effects/{index}" for index in range(len(effects))
            )
        declared = {str(item["target_path"]) for item in attributions}
        return [
            QualityIssue(
                code="missing_evidence_attribution",
                offending_field=path,
                detail="key transferable conclusion has no attribution",
            )
            for path in required_paths
            if path not in declared
        ]

    @staticmethod
    def _effect_basis_issues(
        effects: Any,
        attributions: Sequence[Mapping[str, Any]],
    ) -> list[QualityIssue]:
        if not isinstance(effects, list):
            return []
        by_path = {str(item["target_path"]): item for item in attributions}
        issues: list[QualityIssue] = []
        for index, effect in enumerate(effects):
            path = f"/predicted_effects/{index}"
            attribution = by_path.get(path)
            if attribution is None:
                continue
            if attribution["prediction_status"] == "pending_validation":
                if not isinstance(effect, Mapping) or effect.get("validation_status") != (
                    "pending_validation"
                ):
                    issues.append(
                        QualityIssue(
                            code="unsupported_predicted_effect",
                            offending_field=path,
                            detail="unverified effect must be marked pending_validation",
                        )
                    )
            elif not attribution["supporting_evidence_refs"]:
                issues.append(
                    QualityIssue(
                        code="unsupported_predicted_effect",
                        offending_field=path,
                        detail="evidence-supported effect has no evidence refs",
                    )
                )
        return issues

    @staticmethod
    def _annotation_copy_issues(
        request: WholeTrajectoryReflectionRequest,
        payload: Mapping[str, Any],
        transfer: Mapping[str, Any],
        attributions: Sequence[Mapping[str, Any]],
    ) -> list[QualityIssue]:
        annotations = [request.instruction]
        for segment in request.segments:
            context = segment.get("context", {})
            if isinstance(context, Mapping):
                instruction = context.get("subtask_instruction")
                if isinstance(instruction, str) and instruction.strip():
                    annotations.append(instruction)
        annotation_norms = [_normalize_text(value) for value in annotations]
        candidate_texts = [str(payload["summary"]), *_all_text(transfer)]
        candidate_norms = [
            (path, normalized)
            for path, text in enumerate(candidate_texts)
            if (normalized := _normalize_text(text))
        ]
        for index, candidate in candidate_norms:
            if any(candidate == annotation for annotation in annotation_norms if annotation):
                return [
                    QualityIssue(
                        code="annotation_copy",
                        offending_field=f"transferable_text[{index}]",
                        detail="transferable output directly copies an input annotation",
                    )
                ]
        has_cross_segment = any(
            item.get("source_type") == "cross_segment_inference"
            and len(set(item.get("supporting_segment_refs", []))) >= 2
            for item in attributions
        )
        feedback = transfer.get("guidance", {}).get("feedback_policy")
        if not has_cross_segment or not isinstance(feedback, Mapping):
            for index, candidate in candidate_norms:
                for annotation in annotation_norms:
                    if (
                        len(candidate) >= 12
                        and len(annotation) >= 12
                        and SequenceMatcher(None, candidate, annotation).ratio() >= 0.78
                    ):
                        return [
                            QualityIssue(
                                code="annotation_paraphrase_without_feedback",
                                offending_field=f"transferable_text[{index}]",
                                detail=(
                                    "annotation paraphrase adds no grounded cross-segment "
                                    "feedback structure"
                                ),
                            )
                        ]
        return []

    @staticmethod
    def _structural_leakage_issues(
        transferable: Mapping[str, Any],
        facts: Sequence[Mapping[str, Any]],
    ) -> list[QualityIssue]:
        transfer_canonical = _canonical(transferable)
        transfer_text = _normalize_text("\n".join(_all_text(transferable)))
        for fact in facts:
            if fact.get("transferable_allowed") is not False:
                continue
            if "structured_value" in fact:
                value = fact["structured_value"]
                serialized = _canonical(value)
                normalized_value = (
                    _normalize_text(value) if isinstance(value, str) else ""
                )
                exact_match = len(serialized) >= 4 and serialized in transfer_canonical
                embedded_string_match = (
                    len(normalized_value) >= 4
                    and normalized_value in transfer_text
                )
                if exact_match or embedded_string_match:
                    return [
                        QualityIssue(
                            code="episode_specific_value_leakage",
                            offending_field="transferable_guidance",
                            detail="private structured episode value appears in guidance",
                        )
                    ]
            claim = fact.get("claim")
            if isinstance(claim, str):
                normalized = _normalize_text(claim)
                if len(normalized) >= 12 and normalized in transfer_text:
                    return [
                        QualityIssue(
                            code="episode_specific_value_leakage",
                            offending_field="transferable_guidance",
                            detail="private episode fact appears in guidance",
                        )
                    ]
        return []

    @staticmethod
    def _build_candidate(
        request: WholeTrajectoryReflectionRequest,
        output: UntrustedReflectionOutput,
        payload: Mapping[str, Any],
        *,
        producer_name: str,
        producer_version: str,
    ) -> dict[str, Any]:
        transferable = deepcopy(payload["transferable_guidance"])
        condition = deepcopy(transferable["condition"])
        guidance = deepcopy(transferable["guidance"])
        effects = deepcopy(transferable["predicted_effects"])
        segment_refs = list(dict.fromkeys(payload["supporting_segment_refs"]))
        provenance = deepcopy(dict(request.provenance))
        provenance["producer"] = {
            "kind": "reflector",
            "name": producer_name,
            "version": producer_version,
            "prompt_hash": output.prompt_template_hash,
        }
        return {
            "schema": ProcedureExperienceV1.SCHEMA,
            "schema_version": 1,
            "experience_id": stable_experience_id(
                "procedure",
                condition,
                guidance,
                effects,
                prefix="pexp",
            ),
            "kind": "procedure",
            "status": "candidate",
            "confidence": payload["confidence"],
            "condition": condition,
            "guidance": guidance,
            "predicted_effects": effects,
            "evidence": {
                "supporting_segment_refs": segment_refs,
                "opposing_segment_refs": [],
                "support_count": len(segment_refs),
                "opposing_count": 0,
                "evaluation_status": "not_evaluated",
                "evaluation_ref": None,
                "reflection_attributions": deepcopy(payload["attributions"]),
                "episode_specific_facts": deepcopy(
                    payload["episode_specific_facts"]
                ),
                "supporting_evidence_refs": deepcopy(
                    payload["supporting_evidence_refs"]
                ),
            },
            "last_validated_at": None,
            "provenance": provenance,
            "development_only": True,
            "runtime_retrieval": False,
            "human_review_required": True,
        }


__all__ = [
    "DeterministicLeakageCritic",
    "EpisodeSpecificLeakageCritic",
    "LeakageCriticResult",
    "QualityIssue",
    "ReflectionAbstentionAudit",
    "ReflectionQualityGate",
]
