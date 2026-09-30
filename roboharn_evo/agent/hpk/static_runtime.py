from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roboharn_evo.agent.hpk.audit import (
    PrivateRankingAuditSink,
    build_public_usage_audit,
)
from roboharn_evo.agent.hpk.condition_builder import HPKConditionBuilder
from roboharn_evo.agent.hpk.retriever import (
    HPKDomainCapabilities,
    HPKPrePlannerQuery,
    HPKRetrievalResult,
    HPKRetriever,
    build_hierarchical_physical_knowledge_context,
    validate_hpk_task_usage_transport_audit,
)
from roboharn_evo.agent.hpk.runtime_policy import HPKRuntimePolicy
from roboharn_evo.agent.hpk.schemas import (
    HPK_GEOMETRY_SCORE_PROFILE,
    HPKUnresolved,
    AuditV1,
    ConditionV1,
    TaskStrategyV1,
    contains_private_transfer_text,
)
from roboharn_evo.agent.hpk.semantic_knowledge import semantic_object_from_mapping
from roboharn_evo.agent.hpk.store import LoadedHPKSnapshot
from roboharn_evo.agent.hpk.task_strategy import TaskStrategyNormalizer

if TYPE_CHECKING:
    from roboharn_evo.agent.hpk.policy_config import LoadedHPKPolicy


_SUPPORTED_OPERATIONS = ("contact", "grasp", "place")
_SUPPORTED_STRATEGY_FAMILIES = (
    "contact_relation",
    "observed_grasp_geometry",
    "placement_relation",
)
_TARGET_KIND_ROLES = {
    "free_support": "current_free_support_region",
    "object_top": "current_support_object",
    "reference_region": "current_reference_region",
    "relational_reference_region": "current_reference_region",
    "vacated_pose": "current_vacated_support_region",
}

# The wire TaskStrategyV1 schema keeps the existing normalizer version.  This
# profile names the narrower post-binding fallback used when an accepted active
# skill's instruction contains runtime-private identity/geometry.  The fallback
# is rendered only from the already-bound operation; it never inspects task
# names, seeds, object IDs, coordinates, or snapshot contents.
BOUND_OPERATION_TASK_STRATEGY_PROFILE = "hpk_bound_operation_task_strategy/v1"


class _EmptyHPKRetriever:
    """Validated no-match projection for a legitimate empty evolving head."""

    entries: tuple[()] = ()

    def retrieve_pre_planner(
        self,
        query: HPKPrePlannerQuery | Mapping[str, Any],
        *,
        current_capabilities: HPKDomainCapabilities | Mapping[str, Any],
        max_prompt_chars: int,
    ) -> HPKRetrievalResult:
        typed = HPKPrePlannerQuery.from_mapping(
            query.to_dict() if isinstance(query, HPKPrePlannerQuery) else query
        )
        HPKDomainCapabilities.from_mapping(
            current_capabilities.to_dict()
            if isinstance(current_capabilities, HPKDomainCapabilities)
            else current_capabilities
        )
        if isinstance(max_prompt_chars, bool) or not isinstance(max_prompt_chars, int):
            raise TypeError("max_prompt_chars must be an integer")
        if max_prompt_chars < 1:
            raise ValueError("max_prompt_chars must be >= 1")
        return HPKRetrievalResult(
            stage="pre_planner_task",
            selected_entry=None,
            context=None,
            geometric_strategy=None,
            match_reason=None,
            rejection_reasons=(
                (
                    "no_exact_match"
                    if typed.has_required_information
                    else "insufficient_information"
                ),
            ),
        )

    def retrieve_geometry(
        self,
        condition: Any,
        task_strategy: Any,
        *,
        current_capabilities: HPKDomainCapabilities | Mapping[str, Any],
        target_bound: bool,
        operation_hint: str | None = None,
    ) -> HPKRetrievalResult:
        HPKDomainCapabilities.from_mapping(
            current_capabilities.to_dict()
            if isinstance(current_capabilities, HPKDomainCapabilities)
            else current_capabilities
        )
        reasons: list[str] = []
        if not isinstance(condition, ConditionV1):
            reasons.append("condition_unresolved")
        if not isinstance(task_strategy, TaskStrategyV1):
            reasons.append("task_strategy_unresolved")
        if operation_hint == "place" and not target_bound:
            reasons.append("target_unbound")
        if not reasons:
            reasons.append("no_exact_match")
        return HPKRetrievalResult(
            stage="post_binding_geometry",
            selected_entry=None,
            context=None,
            geometric_strategy=None,
            match_reason=None,
            rejection_reasons=tuple(dict.fromkeys(reasons)),
        )


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _token(value: Any) -> str:
    return _text(value).lower().replace("-", "_").replace(" ", "_")


def _safe_role(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text and not contains_private_transfer_text(text):
            return text
    return ""


def _unique_safe_role(*values: Any) -> tuple[str, bool]:
    roles = list(dict.fromkeys(text for value in values if (text := _safe_role(value))))
    return (roles[0], False) if len(roles) == 1 else ("", len(roles) > 1)


def _source_disclosure(result: HPKRetrievalResult) -> dict[str, Any] | None:
    if result.selected_entry is None:
        return None
    context = build_hierarchical_physical_knowledge_context(result.selected_entry)
    return dict(context["source_disclosure"])


def _bound_operation_purpose(operation: str) -> str:
    if operation not in _SUPPORTED_OPERATIONS:
        raise ValueError("bound operation purpose requires a supported operation")
    return f"execute current bound {operation} subgoal"


@dataclass(frozen=True, slots=True)
class HPKTaskDecision:
    retrieval: HPKRetrievalResult
    audit: AuditV1


@dataclass(frozen=True, slots=True)
class HPKBoundGeometryQuery:
    condition: ConditionV1 | HPKUnresolved
    task_strategy: TaskStrategyV1 | HPKUnresolved
    target_bound: bool
    operation: str
    semantic_object: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class HPKGeometryDecision:
    retrieval: HPKRetrievalResult
    condition: ConditionV1 | HPKUnresolved
    task_strategy: TaskStrategyV1 | HPKUnresolved
    audit: AuditV1 | None


class HPKStaticRuntime:
    """One immutable snapshot plus deterministic, transient retrieval state."""

    def __init__(
        self,
        *,
        policy: HPKRuntimePolicy,
        snapshot: LoadedHPKSnapshot | Any,
        task_family: str,
        domain_id: str,
        current_capabilities: HPKDomainCapabilities | Mapping[str, Any] | None = None,
        snapshot_capabilities: HPKDomainCapabilities | Mapping[str, Any] | None = None,
        geometry_policy: LoadedHPKPolicy | None = None,
        allow_empty_snapshot: bool = False,
    ) -> None:
        if not policy.retrieves:
            raise ValueError("HPKStaticRuntime requires a retrieval-enabled mode")
        family = _text(task_family)
        domain = _text(domain_id)
        if not family or not domain:
            raise ValueError("static HPK runtime requires task_family and domain_id")
        self.policy = policy
        self.geometry_policy = geometry_policy or policy.load_geometry_policy()
        self.snapshot = snapshot
        self.task_family = family
        self.domain_id = domain
        self.current_capabilities = (
            current_capabilities
            if isinstance(current_capabilities, HPKDomainCapabilities)
            else HPKDomainCapabilities.from_mapping(
                current_capabilities
                if isinstance(current_capabilities, Mapping)
                else self._default_capabilities()
            )
        )
        raw_snapshot_capabilities = snapshot_capabilities
        if raw_snapshot_capabilities is None:
            raw_manifest = getattr(snapshot, "manifest", {})
            if isinstance(raw_manifest, Mapping):
                raw_snapshot_capabilities = raw_manifest.get("capabilities")
        typed_snapshot_capabilities = (
            raw_snapshot_capabilities
            if isinstance(raw_snapshot_capabilities, HPKDomainCapabilities)
            else HPKDomainCapabilities.from_mapping(
                raw_snapshot_capabilities
                if isinstance(raw_snapshot_capabilities, Mapping)
                else self.current_capabilities.to_dict()
            )
        )
        self._install_snapshot_projection(
            snapshot,
            snapshot_capabilities=typed_snapshot_capabilities,
            allow_empty_snapshot=allow_empty_snapshot,
        )
        self._pending_preplanner_query: HPKPrePlannerQuery | None = None
        self._last_task_decision: HPKTaskDecision | None = None
        self._last_geometry_decision: HPKGeometryDecision | None = None
        self._load_count = 1

    def _install_snapshot_projection(
        self,
        snapshot: Any,
        *,
        snapshot_capabilities: HPKDomainCapabilities,
        allow_empty_snapshot: bool,
    ) -> None:
        entries = tuple(snapshot.entries)
        if entries:
            self.retriever: HPKRetriever | _EmptyHPKRetriever = HPKRetriever(
                entries,
                snapshot_capabilities=snapshot_capabilities,
            )
        elif allow_empty_snapshot:
            self.retriever = _EmptyHPKRetriever()
        else:
            raise ValueError("static HPK snapshot must contain an accepted entry")
        self.snapshot = snapshot

    def _disable_legacy_entry_retrieval(self) -> None:
        """Keep a legacy snapshot readable while preventing new runtime use."""

        self.retriever = _EmptyHPKRetriever()

    @classmethod
    def from_policy(
        cls,
        policy: HPKRuntimePolicy,
        *,
        task_family: str,
        domain_id: str,
        current_capabilities: HPKDomainCapabilities | Mapping[str, Any] | None = None,
    ) -> "HPKStaticRuntime":
        snapshot = policy.load_static_snapshot()
        return cls(
            policy=policy,
            snapshot=snapshot,
            task_family=task_family,
            domain_id=domain_id,
            current_capabilities=current_capabilities,
        )

    @property
    def load_count(self) -> int:
        return self._load_count

    @property
    def last_task_audit(self) -> AuditV1 | None:
        return (
            None if self._last_task_decision is None else self._last_task_decision.audit
        )

    @property
    def last_geometry_audit(self) -> AuditV1 | None:
        decision = self._last_geometry_decision
        return None if decision is None else decision.audit

    def reset_episode(self) -> None:
        self._pending_preplanner_query = None
        self._last_task_decision = None
        self._last_geometry_decision = None

    def _default_capabilities(self) -> dict[str, Any]:
        geometry_sources = ["rgbd_observed", "runtime_relational", "unknown"]
        if self.policy.allow_oracle_evidence:
            geometry_sources.append("oracle")
        return {
            "task_families": [self.task_family],
            "domain_ids": [self.domain_id],
            "operations": sorted(_SUPPORTED_OPERATIONS),
            "strategy_families": sorted(_SUPPORTED_STRATEGY_FAMILIES),
            "target_relations": ["center_of"],
            "hard_constraints": ["support_valid", "target_region_free"],
            "geometry_source_classes": sorted(geometry_sources),
            "semantic_part_observed": False,
        }

    def set_preplanner_query(
        self,
        query: HPKPrePlannerQuery | Mapping[str, Any] | None,
    ) -> None:
        if query is None:
            self._pending_preplanner_query = None
            return
        self._pending_preplanner_query = (
            query
            if isinstance(query, HPKPrePlannerQuery)
            else HPKPrePlannerQuery.from_mapping(query)
        )

    def build_preplanner_query(
        self,
        *,
        scene_memory: Mapping[str, Any] | None,
        active_skill: Mapping[str, Any] | Any | None,
    ) -> HPKPrePlannerQuery:
        """Project only explicit pre-planner facts; absent facts remain null."""

        scene = _mapping(scene_memory)
        active = _mapping(active_skill)
        tags = _mapping(active.get("semantic_tags"))
        focus = _mapping(scene.get("task_focus"))
        operation = _token(tags.get("subtask_type"))
        if operation not in _SUPPORTED_OPERATIONS:
            operation = ""
        phases = {
            _safe_role(_mapping(state).get("phase"))
            for state in _mapping(scene.get("manipulation_state")).values()
            if _mapping(state).get("phase")
        }
        phase = next(iter(phases)) if len(phases) == 1 else ""
        manipulated_role = _safe_role(focus.get("manipulated_role"))
        target_role = _safe_role(focus.get("target_role"))
        target_relation = _token(focus.get("target_relation"))
        if target_relation != "center_of":
            target_relation = ""
        held_state = _token(focus.get("held_state"))
        if held_state not in {"held", "not_held"}:
            held_state = ""
        return HPKPrePlannerQuery.from_mapping(
            {
                "task_family": self.task_family,
                "manipulation_phase": phase or None,
                "operation": operation or None,
                "manipulated_role": manipulated_role or None,
                "held_state": held_state or None,
                "target_role": target_role or None,
                "target_relation": target_relation or None,
                "positive_scene_predicates": [
                    value
                    for value in ("support_valid", "target_region_free")
                    if scene.get(value) is True
                ],
                "satisfied_preconditions": [],
            }
        )

    def retrieve_preplanner(self) -> HPKTaskDecision:
        query = self._pending_preplanner_query
        self._pending_preplanner_query = None
        if query is None:
            query = HPKPrePlannerQuery.from_mapping(
                {
                    "task_family": self.task_family,
                    "manipulation_phase": None,
                    "operation": None,
                    "manipulated_role": None,
                    "held_state": None,
                    "target_role": None,
                    "target_relation": None,
                    "positive_scene_predicates": [],
                    "satisfied_preconditions": [],
                }
            )
        retrieval = self.retriever.retrieve_pre_planner(
            query,
            current_capabilities=self.current_capabilities,
            max_prompt_chars=self.policy.max_prompt_chars,
        )
        injected = retrieval.context is not None
        audit = build_public_usage_audit(
            retrieval_stage="pre_planner_task",
            snapshot_id=self.snapshot.snapshot_id,
            snapshot_manifest_sha256=self.snapshot.manifest_sha256,
            retrieved_entry_ids=retrieval.retrieved_entry_ids,
            selected_entry_id=retrieval.selected_entry_id,
            match_reason=retrieval.match_reason,
            rejection_reasons=retrieval.rejection_reasons,
            behavior_changed="unverified" if injected else False,
            behavior_change_channel="unverified" if injected else "none",
            source_disclosure=_source_disclosure(retrieval),
        )
        decision = HPKTaskDecision(retrieval=retrieval, audit=audit)
        self._last_task_decision = decision
        return decision

    def accept_task_transport_receipt(
        self,
        decision: HPKTaskDecision,
        receipt: Mapping[str, Any],
    ) -> HPKTaskDecision:
        context = decision.retrieval.context
        if context is None:
            raise ValueError("HPK task receipt is unexpected without a context")
        validated = validate_hpk_task_usage_transport_audit(
            receipt,
            expected_context=context,
        )
        audit = build_public_usage_audit(
            retrieval_stage="pre_planner_task",
            snapshot_id=self.snapshot.snapshot_id,
            snapshot_manifest_sha256=self.snapshot.manifest_sha256,
            retrieved_entry_ids=decision.retrieval.retrieved_entry_ids,
            selected_entry_id=decision.retrieval.selected_entry_id,
            match_reason=decision.retrieval.match_reason,
            rejection_reasons=decision.retrieval.rejection_reasons,
            planner_context_sha256=validated["context_sha256"],
            planner_context_rendered=True,
            planner_usage="injected_use_unverified",
            behavior_changed="unverified",
            behavior_change_channel="unverified",
            source_disclosure=_source_disclosure(decision.retrieval),
        )
        finalized = HPKTaskDecision(retrieval=decision.retrieval, audit=audit)
        self._last_task_decision = finalized
        return finalized

    def build_bound_geometry_query(
        self,
        *,
        scene_memory: Mapping[str, Any],
        instance: Mapping[str, Any],
        action_mode: str,
        arm: str,
        requested_target_id: Any = None,
        active_skill: Mapping[str, Any] | Any | None = None,
    ) -> HPKBoundGeometryQuery:
        scene = _mapping(scene_memory)
        current_instance = _mapping(instance)
        operation = _token(action_mode)
        normalized_arm = _token(arm)
        active = _mapping(active_skill)
        if operation not in _SUPPORTED_OPERATIONS:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="operation_unresolved",
                missing_fields=("operation",),
            )
            return HPKBoundGeometryQuery(unresolved, unresolved, False, operation)

        target_id = _text(requested_target_id)
        raw_targets = scene.get("operation_targets")
        targets = [
            dict(item)
            for item in (raw_targets if isinstance(raw_targets, list) else [])
            if isinstance(item, Mapping) and _text(item.get("target_id")) == target_id
        ]
        target_bound = operation != "place" or (bool(target_id) and len(targets) == 1)
        if operation == "place" and not target_bound:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="place_target_unbound",
                missing_fields=("target_id",),
            )
            return HPKBoundGeometryQuery(unresolved, unresolved, False, operation)
        target = targets[0] if targets else {}

        manipulated_role, manipulated_role_conflict = _unique_safe_role(
            current_instance.get("semantic_role"),
            current_instance.get("task_role"),
            current_instance.get("role"),
            current_instance.get("query_role"),
        )
        if manipulated_role_conflict:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="manipulated_role_conflict",
                missing_fields=("manipulated_role",),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )
        if not manipulated_role:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="manipulated_role_unresolved",
                missing_fields=("manipulated_role",),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )

        raw_manipulation_state = scene.get("manipulation_state")
        manipulation_state_is_trustworthy = isinstance(raw_manipulation_state, Mapping)
        manipulation_state = _mapping(raw_manipulation_state)
        arm_state = _mapping(manipulation_state.get(normalized_arm))
        held_refs = {
            _text(_mapping(value).get("held_instance_id"))
            for value in manipulation_state.values()
            if _text(_mapping(value).get("held_instance_id"))
        }
        no_object_held = manipulation_state_is_trustworthy and not held_refs
        phase = _safe_role(arm_state.get("phase"))
        if not phase and operation in {"grasp", "contact"} and no_object_held:
            # This is the actual lifecycle boundary at which the runtime is
            # selecting an operation candidate.  It is not inferred from task
            # language or supplied by the snapshot.
            phase = f"{operation}_candidate"
        if not phase:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="manipulation_phase_unresolved",
                missing_fields=("manipulation_state.arm.phase",),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )
        instance_ref = _text(
            current_instance.get("instance_id") or current_instance.get("track_id")
        )
        if not instance_ref:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="held_state_unresolved",
                missing_fields=("manipulated_instance_id",),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )
        if "held_instance_id" in arm_state:
            held_ref = _text(arm_state.get("held_instance_id"))
            if held_ref and held_ref == instance_ref:
                held_state = "held"
            else:
                held_state = "not_held"
        elif operation in {"grasp", "contact"} and no_object_held:
            held_state = "not_held"
        else:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="held_state_unresolved",
                missing_fields=("manipulation_state.arm.held_instance_id",),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )
        if operation == "place" and held_state != "held":
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="place_held_state_unresolved",
                missing_fields=("manipulated_object.held_state",),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )

        relation_value = target.get("placement_relation", target.get("target_relation"))
        if isinstance(relation_value, Mapping):
            relation_value = relation_value.get("relation")
        target_relation = _token(relation_value)
        if target_relation != "center_of":
            target_relation = ""
        target_role, target_role_conflict = _unique_safe_role(
            target.get("target_role"),
            target.get("reference_role"),
            target.get("role"),
        )
        if target_role_conflict:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="target_role_conflict",
                missing_fields=("target_role",),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )
        if not target_role:
            target_role = _TARGET_KIND_ROLES.get(_token(target.get("target_kind")), "")
        if operation == "place" and (not target_role or not target_relation):
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="place_target_semantics_unresolved",
                missing_fields=tuple(
                    name
                    for name, value in (
                        ("target_role", target_role),
                        ("target_relation", target_relation),
                    )
                    if not value
                ),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )

        semantic_class, semantic_class_conflict = _unique_safe_role(
            current_instance.get("semantic_class"),
            current_instance.get("object_class"),
            current_instance.get("class_name"),
            current_instance.get("class"),
            current_instance.get("category"),
        )
        geometry_class, geometry_class_conflict = _unique_safe_role(
            current_instance.get("geometry_class"),
            current_instance.get("shape_class"),
        )
        if semantic_class_conflict or geometry_class_conflict:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="manipulated_descriptor_conflict",
                missing_fields=tuple(
                    name
                    for name, conflict in (
                        ("manipulated_object.semantic_class", semantic_class_conflict),
                        ("manipulated_object.geometry_class", geometry_class_conflict),
                    )
                    if conflict
                ),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )
        if not semantic_class:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="semantic_class_unresolved",
                missing_fields=("manipulated_object.semantic_class",),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )
        geometry_class = geometry_class or "unknown"
        target_semantic_class, target_semantic_conflict = _unique_safe_role(
            target.get("semantic_class"),
            target.get("object_class"),
        )
        if not target_semantic_class:
            target_semantic_class = _safe_role(target.get("target_kind"))
        target_geometry_class, target_geometry_conflict = _unique_safe_role(
            target.get("geometry_class"),
            target.get("shape_class"),
        )
        if target_semantic_conflict or target_geometry_conflict:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="target_descriptor_conflict",
                missing_fields=tuple(
                    name
                    for name, conflict in (
                        ("target.semantic_class", target_semantic_conflict),
                        ("target.geometry_class", target_geometry_conflict),
                    )
                    if conflict
                ),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )
        target_descriptor = {
            "semantic_class": target_semantic_class or None,
            "geometry_class": target_geometry_class or None,
            "role": target_role or None,
            "relation": target_relation or None,
        }
        active_tags = _mapping(active.get("semantic_tags"))
        raw_current_instruction = next(
            (
                text
                for value in (
                    active.get("subgoal_purpose"),
                    active_tags.get("subgoal_purpose"),
                    active.get("instruction"),
                    active.get("subtask_text"),
                )
                if (text := _text(value))
            ),
            "",
        )
        skill_name = _safe_role(
            active.get("skill_name"),
            active.get("selected_skill"),
            active.get("name"),
        )
        if not raw_current_instruction or not skill_name:
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="current_task_strategy_source_unresolved",
                missing_fields=tuple(
                    name
                    for name, value in (
                        ("active_skill.instruction", raw_current_instruction),
                        ("active_skill.skill_name", skill_name),
                    )
                    if not value
                ),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )
        current_instruction = _safe_role(raw_current_instruction)
        if not current_instruction:
            # TaskStrategyV1.source cannot transfer private instance IDs,
            # coordinates, or paths.  Preserve the accepted active skill name
            # while projecting a deterministic operation-level purpose.  The
            # snapshot never supplies or repairs this current runtime value.
            current_instruction = _bound_operation_purpose(operation)
        active_preferred_arm = _token(active.get("preferred_arm"))
        if active_preferred_arm not in {"left", "right", "either"}:
            active_preferred_arm = normalized_arm
        if (
            active_preferred_arm in {"left", "right"}
            and active_preferred_arm != normalized_arm
        ):
            unresolved = HPKUnresolved(
                component="p0c_runtime_binding",
                reason="preferred_arm_binding_conflict",
                missing_fields=("preferred_arm",),
            )
            return HPKBoundGeometryQuery(
                unresolved, unresolved, target_bound, operation
            )
        bound = {
            "operation": operation,
            "action_mode": operation,
            "manipulated_role": manipulated_role,
            "target_role": target_role,
            "target_relation": target_relation,
            "manipulation_phase": phase,
            "preferred_arm": active_preferred_arm,
            "arm": normalized_arm,
            "manipulated_instance_id": instance_ref,
            "manipulated_object": {
                "semantic_class": semantic_class,
                "geometry_class": geometry_class,
                "role": manipulated_role,
                "held_state": held_state,
            },
            "target": target_descriptor,
            "support_valid": target.get("support_valid"),
            "free": target.get("free"),
        }
        planner_projection = {
            "subtask_text": current_instruction,
            "selected_skill": skill_name,
            "operation": operation,
            "manipulated_role": manipulated_role,
            "target_role": target_role,
            "target_relation": target_relation,
            "manipulation_phase": phase,
            "subgoal_purpose": current_instruction,
            "preferred_arm": active_preferred_arm,
            "semantic_tags": active_tags,
        }
        task_strategy = TaskStrategyNormalizer().normalize(
            planner_projection,
            scene,
            bound,
            active,
        )
        runtime_state = {
            "task_family": self.task_family,
            "operation": operation,
            "manipulation_phase": phase,
            "manipulated_role": manipulated_role,
            "target_role": target_role,
            "target_relation": target_relation,
            "manipulation_state": manipulation_state,
            "manipulated_instance_id": instance_ref,
            "manipulated_object": bound["manipulated_object"],
            "target": target_descriptor,
            "support_valid": target.get("support_valid"),
            "free": target.get("free"),
            "preconditions": [],
        }
        condition = HPKConditionBuilder().build(
            scene,
            task_strategy if isinstance(task_strategy, TaskStrategyV1) else None,
            runtime_state,
            bound,
            task_family=self.task_family,
        )
        semantic_source = dict(current_instance)
        semantic_source["role"] = manipulated_role
        semantic_source["held_state"] = held_state
        return HPKBoundGeometryQuery(
            condition,
            task_strategy,
            target_bound,
            operation,
            semantic_object_from_mapping(semantic_source),
        )

    def retrieve_geometry(
        self,
        query: HPKBoundGeometryQuery,
    ) -> HPKGeometryDecision:
        retrieval = self.retriever.retrieve_geometry(
            query.condition,
            query.task_strategy,
            current_capabilities=self.current_capabilities,
            target_bound=query.target_bound,
            operation_hint=query.operation,
        )
        condition_id = (
            query.condition.stable_id
            if isinstance(query.condition, ConditionV1)
            else None
        )
        task_strategy_id = (
            query.task_strategy.stable_id
            if isinstance(query.task_strategy, TaskStrategyV1)
            else None
        )
        audit: AuditV1 | None = None
        if retrieval.selected_entry is None:
            audit = build_public_usage_audit(
                retrieval_stage="post_binding_geometry",
                snapshot_id=self.snapshot.snapshot_id,
                snapshot_manifest_sha256=self.snapshot.manifest_sha256,
                condition_id=condition_id,
                task_strategy_id=task_strategy_id,
                rejection_reasons=retrieval.rejection_reasons,
                behavior_changed=False,
                behavior_change_channel="none",
            )
        decision = HPKGeometryDecision(
            retrieval=retrieval,
            condition=query.condition,
            task_strategy=query.task_strategy,
            audit=audit,
        )
        self._last_geometry_decision = decision
        return decision

    def new_private_ranking_sink(
        self,
        decision: HPKGeometryDecision,
    ) -> PrivateRankingAuditSink:
        entry = decision.retrieval.selected_entry
        strategy = decision.retrieval.geometric_strategy
        if entry is None or strategy is None:
            raise ValueError("private ranking sink requires a selected geometry entry")
        return PrivateRankingAuditSink(
            snapshot_id=self.snapshot.snapshot_id,
            snapshot_manifest_sha256=self.snapshot.manifest_sha256,
            selected_entry_id=entry["entry_id"],
            selected_geometric_strategy_id=strategy.stable_id,
        )

    def finalize_geometry_ranking(
        self,
        decision: HPKGeometryDecision,
        sink: PrivateRankingAuditSink,
    ) -> HPKGeometryDecision:
        entry = decision.retrieval.selected_entry
        strategy = decision.retrieval.geometric_strategy
        if entry is None or strategy is None:
            raise ValueError("geometry finalization requires a selected entry")
        if not sink.accepted_low_level_ranking:
            raise RuntimeError("private ranking audit was not captured")
        reasons = list(decision.retrieval.rejection_reasons)
        if sink.zero_compliant_candidates:
            reasons.append("zero_compliant_candidates")
        changed = sink.behavior_changed
        audit = build_public_usage_audit(
            retrieval_stage="post_binding_geometry",
            snapshot_id=self.snapshot.snapshot_id,
            snapshot_manifest_sha256=self.snapshot.manifest_sha256,
            retrieved_entry_ids=decision.retrieval.retrieved_entry_ids,
            selected_entry_id=decision.retrieval.selected_entry_id,
            condition_id=(
                decision.condition.stable_id
                if isinstance(decision.condition, ConditionV1)
                else None
            ),
            task_strategy_id=(
                decision.task_strategy.stable_id
                if isinstance(decision.task_strategy, TaskStrategyV1)
                else None
            ),
            selected_geometric_strategy_id=strategy.stable_id,
            match_reason=decision.retrieval.match_reason,
            rejection_reasons=reasons,
            behavior_changed=changed,
            behavior_change_channel="geometry" if changed else "none",
            geometric_compliance=sink.geometric_compliance,
            scoring_profile=HPK_GEOMETRY_SCORE_PROFILE,
            source_disclosure=_source_disclosure(decision.retrieval),
        )
        sink.bind_public_usage_audit_record(audit)
        finalized = HPKGeometryDecision(
            retrieval=decision.retrieval,
            condition=decision.condition,
            task_strategy=decision.task_strategy,
            audit=audit,
        )
        self._last_geometry_decision = finalized
        return finalized

    def geometry_trace_binding(
        self,
        decision: HPKGeometryDecision,
        sink: PrivateRankingAuditSink,
    ) -> dict[str, Any]:
        if decision.audit is None or not sink.finalized:
            raise RuntimeError("geometry usage must be fully audited before motion")
        entry = decision.retrieval.selected_entry
        strategy = decision.retrieval.geometric_strategy
        if entry is None or strategy is None:
            raise RuntimeError("geometry trace binding requires a selected entry")
        return {
            "snapshot_id": self.snapshot.snapshot_id,
            "snapshot_manifest_sha256": self.snapshot.manifest_sha256,
            "selected_entry_id": entry["entry_id"],
            "condition_id": (
                decision.condition.stable_id
                if isinstance(decision.condition, ConditionV1)
                else None
            ),
            "task_strategy_id": (
                decision.task_strategy.stable_id
                if isinstance(decision.task_strategy, TaskStrategyV1)
                else None
            ),
            "selected_geometric_strategy_id": strategy.stable_id,
            "public_usage_audit_id": decision.audit.stable_id,
            "private_ranking_audit_id": sink.private_ranking_audit_id,
            "geometry_policy_id": self.geometry_policy.policy_id,
            "geometry_policy_sha256": self.geometry_policy.config_sha256,
            "strategy_realization_status": (
                "satisfied"
                if sink.geometric_compliance is True
                and sink.selected_candidate_private_ref is not None
                else "violated"
                if sink.zero_compliant_candidates
                else "unknown"
            ),
            "motion_status": "pending",
            "source_disclosure": _source_disclosure(decision.retrieval),
        }


def build_static_hpk_runtime(
    policy: HPKRuntimePolicy,
    *,
    task_family: str,
    domain_id: str,
    current_capabilities: HPKDomainCapabilities | Mapping[str, Any] | None = None,
) -> HPKStaticRuntime | None:
    """Build static once; off/observe return ``None`` without snapshot I/O."""

    if not policy.static_enabled:
        return None
    return HPKStaticRuntime.from_policy(
        policy,
        task_family=task_family,
        domain_id=domain_id,
        current_capabilities=current_capabilities,
    )


__all__ = [
    "HPKBoundGeometryQuery",
    "HPKGeometryDecision",
    "HPKStaticRuntime",
    "HPKTaskDecision",
    "build_static_hpk_runtime",
]
