from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.action_transition import ActionEffectTransitionV1
from roboharn_evo.agent.hpk.episode_finalizer import TransitionStrategyContext
from roboharn_evo.agent.hpk.schemas import HPKUnresolved, EntryV1
from roboharn_evo.agent.hpk.semantic_knowledge import (
    semantic_evidence_from_transition,
    semantic_knowledge_from_strategy,
)
from roboharn_evo.agent.hpk.semantic_store import (
    load_semantic_knowledge,
    update_semantic_knowledge,
)
from roboharn_evo.agent.hpk.strategy_extractor import extract_baseline_strategy
from roboharn_evo.agent.hpk.updater import same_semantic_strategy


class SemanticFinalizationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SemanticFinalizationResult:
    status: str
    knowledge_path: Path
    knowledge: tuple[dict[str, Any], ...]
    newly_accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "knowledge_path": str(self.knowledge_path),
            "knowledge": [dict(value) for value in self.knowledge],
            "newly_accepted": self.newly_accepted,
        }


def _strategy_components(
    transition: ActionEffectTransitionV1,
    context: TransitionStrategyContext,
    selected_entry: EntryV1 | Mapping[str, Any] | None,
) -> tuple[Any, Any]:
    if context.proposed_geometric_strategy is not None:
        assert context.proposed_expected_effect is not None
        return context.proposed_geometric_strategy, context.proposed_expected_effect
    if selected_entry is not None:
        entry = (
            selected_entry
            if isinstance(selected_entry, EntryV1)
            else EntryV1.from_dict(selected_entry)
        )
        return entry["geometric_strategy"], entry["expected_effect"]
    features = transition["candidate_geometry_features"]
    expected = transition["expected_effect"]
    if not isinstance(features, Mapping) or not isinstance(expected, Mapping):
        raise SemanticFinalizationError(
            "the action has no transferable geometric strategy"
        )
    extracted = extract_baseline_strategy(
        condition=context.condition,
        task_strategy=context.task_strategy,
        candidate_features=features,
        expected_effect=expected,
        selected_hpk_entry_id=None,
        allow_oracle_geometry=False,
    )
    if isinstance(extracted, HPKUnresolved):
        raise SemanticFinalizationError(extracted.reason)
    return extracted.geometric_strategy, extracted.expected_effect


def knowledge_attempt_from_transition(
    transition: ActionEffectTransitionV1 | Mapping[str, Any],
    context: TransitionStrategyContext,
    *,
    selected_entry: EntryV1 | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project one private runtime transition into semantic knowledge/evidence."""

    typed_transition = (
        transition
        if isinstance(transition, ActionEffectTransitionV1)
        else ActionEffectTransitionV1(transition)
    )
    geometry, expected = _strategy_components(typed_transition, context, selected_entry)
    evidence = semantic_evidence_from_transition(
        typed_transition,
        object_semantics=context.semantic_object,
    )
    candidate = semantic_knowledge_from_strategy(
        condition=context.condition,
        task_strategy=context.task_strategy,
        geometric_strategy=geometry,
        expected_effect=expected,
        object_semantics=context.semantic_object,
        reasoning=context.reasoning,
        status="candidate",
    )
    return candidate, evidence


class SemanticEpisodeFinalizer:
    """Small finalizer: semantic match, inline evidence, one JSONL file."""

    def __init__(
        self,
        *,
        knowledge_path: str | Path,
        promotion_policy: Mapping[str, Any] | Any,
    ) -> None:
        self.knowledge_path = Path(knowledge_path)
        self.promotion_policy = promotion_policy

    def finalize(
        self,
        attempts: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ) -> SemanticFinalizationResult:
        if not attempts:
            return SemanticFinalizationResult(
                status="no action evidence; knowledge unchanged",
                knowledge_path=self.knowledge_path,
                knowledge=(),
                newly_accepted=False,
            )
        records: tuple[dict[str, Any], ...] = ()
        newly_accepted = False
        for candidate, evidence in attempts:
            before = (
                load_semantic_knowledge(self.knowledge_path)
                if self.knowledge_path.exists()
                else ()
            )
            previously_accepted = any(
                value["status"] == "accepted"
                and same_semantic_strategy(value, candidate)
                for value in before
            )
            updated, records = update_semantic_knowledge(
                self.knowledge_path,
                candidate,
                evidence,
                promotion_policy=self.promotion_policy,
            )
            newly_accepted = newly_accepted or (
                updated["status"] == "accepted" and not previously_accepted
            )
        return SemanticFinalizationResult(
            status=(
                "knowledge updated"
                if newly_accepted
                else "evidence saved; no accepted knowledge"
            ),
            knowledge_path=self.knowledge_path,
            knowledge=records,
            newly_accepted=newly_accepted,
        )


__all__ = [
    "SemanticEpisodeFinalizer",
    "SemanticFinalizationError",
    "SemanticFinalizationResult",
    "knowledge_attempt_from_transition",
]
