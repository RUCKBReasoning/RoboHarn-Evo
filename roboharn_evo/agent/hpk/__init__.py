from __future__ import annotations

from importlib import import_module
from typing import Any

from roboharn_evo.agent.hpk.audit import (
    PrivateRankingAuditSink,
    PrivateRankingAuditV1,
    build_observe_audit,
    build_public_usage_audit,
)
from roboharn_evo.agent.hpk.candidate_features import (
    candidate_semantic_features,
    extract_candidate_features,
)
from roboharn_evo.agent.hpk.condition_builder import (
    HPKConditionBuilder,
    ConditionBuilder,
    build_condition,
)
from roboharn_evo.agent.hpk.effect_extractor import (
    AbstractEffectExtractor,
    extract_abstract_effect,
)
from roboharn_evo.agent.hpk.evidence import (
    build_action_effect_evidence,
    build_evidence,
    determine_evidence_verdict,
    evidence_verdict,
)
from roboharn_evo.agent.hpk.geometry_policy import (
    HPKGeometryPolicy,
    GeometricRankingRequest,
    GeometryRankingResult,
    rank_operation_pose_candidates,
)
from roboharn_evo.agent.hpk.runtime_policy import (
    HPK_OBSERVE_MODE,
    HPK_OFF_MODE,
    HPK_STATIC_MODE,
    HPKRuntimePolicy,
    parse_hpk_runtime_policy,
)
from roboharn_evo.agent.hpk.schemas import (
    ABSTRACT_EFFECT_SCHEMA,
    AbstractEffectV1,
    HPKAbstractEffectV1,
    HPKConditionV1,
    HPKEvidenceV1,
    HPKGeometricStrategyV1,
    HPKTaskStrategyV1,
    HPKUnresolved,
    HPKValidationError,
    AuditV1,
    CandidateGeometryFeaturesV1,
    ConditionV1,
    ContextV1,
    EntryV1,
    EvidenceV1,
    GeometricStrategyV1,
    ObserveAuditRecord,
    SnapshotV1,
    TaskStrategyV1,
    canonical_json,
    reject_private_transferable,
    stable_content_id,
)
from roboharn_evo.agent.hpk.task_strategy import (
    TaskStrategyNormalizer,
    normalize_task_strategy,
)

_P0C_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "migrate_action_effect_transition": (
        "roboharn_evo.agent.hpk.action_transition",
        "migrate_action_effect_transition",
    ),
    "ActionKnowledgeFamilyV1": (
        "roboharn_evo.agent.hpk.knowledge_family",
        "ActionKnowledgeFamilyV1",
    ),
    "HPKV3ValidationError": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "HPKV3ValidationError",
    ),
    "ActionEvidenceV3": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "ActionEvidenceV3",
    ),
    "ActionEvidenceV31": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "ActionEvidenceV31",
    ),
    "ActionKnowledgeCandidateV3": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "ActionKnowledgeCandidateV3",
    ),
    "ActionKnowledgeV3": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "ActionKnowledgeV3",
    ),
    "FamilyRoutingConfig": (
        "roboharn_evo.agent.hpk.family_router",
        "FamilyRoutingConfig",
    ),
    "ActionKnowledgeQuery": (
        "roboharn_evo.agent.hpk.hierarchical_retriever",
        "ActionKnowledgeQuery",
    ),
    "HPKDomainCapabilities": (
        "roboharn_evo.agent.hpk.retriever",
        "HPKDomainCapabilities",
    ),
    "HPKPrePlannerQuery": (
        "roboharn_evo.agent.hpk.retriever",
        "HPKPrePlannerQuery",
    ),
    "HPKRetrievalResult": (
        "roboharn_evo.agent.hpk.retriever",
        "HPKRetrievalResult",
    ),
    "HPKRetriever": ("roboharn_evo.agent.hpk.retriever", "HPKRetriever"),
    "HPKStaticRuntime": (
        "roboharn_evo.agent.hpk.static_runtime",
        "HPKStaticRuntime",
    ),
    "LoadedHPKSnapshot": (
        "roboharn_evo.agent.hpk.store",
        "LoadedHPKSnapshot",
    ),
    "HierarchicalHPKRetrievalRuntime": (
        "roboharn_evo.agent.hpk.hierarchical_retriever",
        "HierarchicalHPKRetrievalRuntime",
    ),
    "IncrementalKnowledgeMaintainer": (
        "roboharn_evo.agent.hpk.incremental_maintainer",
        "IncrementalKnowledgeMaintainer",
    ),
    "KnowledgeFamilyCatalogV1": (
        "roboharn_evo.agent.hpk.knowledge_family",
        "KnowledgeFamilyCatalogV1",
    ),
    "KnowledgeFamilyRouter": (
        "roboharn_evo.agent.hpk.family_router",
        "KnowledgeFamilyRouter",
    ),
    "RealizationHypothesisV31": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "RealizationHypothesisV31",
    ),
    "RuntimeFeasibilityReportV31": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "RuntimeFeasibilityReportV31",
    ),
    "RuntimeGoalBindingV31": (
        "roboharn_evo.agent.hpk.goal_consistency",
        "RuntimeGoalBindingV31",
    ),
    "SubtaskGoalContractV31": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "SubtaskGoalContractV31",
    ),
    "SubtaskKnowledgeQuery": (
        "roboharn_evo.agent.hpk.hierarchical_retriever",
        "SubtaskKnowledgeQuery",
    ),
    "VLMActionKnowledgeRetriever": (
        "roboharn_evo.agent.hpk.hierarchical_retriever",
        "VLMActionKnowledgeRetriever",
    ),
    "VLMKnowledgeConsolidator": (
        "roboharn_evo.agent.hpk.semantic_consolidator",
        "VLMKnowledgeConsolidator",
    ),
    "VLMSubtaskKnowledgeRetriever": (
        "roboharn_evo.agent.hpk.hierarchical_retriever",
        "VLMSubtaskKnowledgeRetriever",
    ),
    "SubtaskKnowledgeV3": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "SubtaskKnowledgeV3",
    ),
    "TaskKnowledgeFamilyV1": (
        "roboharn_evo.agent.hpk.knowledge_family",
        "TaskKnowledgeFamilyV1",
    ),
    "SubtaskPackageV3": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "SubtaskPackageV3",
    ),
    "TrajectoryKnowledgePackageV3": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "TrajectoryKnowledgePackageV3",
    ),
    "VLMHierarchicalReflector": (
        "roboharn_evo.agent.hpk.vlm_hierarchical_reflector",
        "VLMHierarchicalReflector",
    ),
    "atomize_package": (
        "roboharn_evo.agent.hpk.hierarchical_knowledge",
        "atomize_package",
    ),
    "build_hierarchical_physical_knowledge_context": (
        "roboharn_evo.agent.hpk.retriever",
        "build_hierarchical_physical_knowledge_context",
    ),
    "build_realization_hypothesis_v31": (
        "roboharn_evo.agent.hpk.goal_consistency",
        "build_realization_hypothesis_v31",
    ),
    "build_subtask_goal_contract_v31": (
        "roboharn_evo.agent.hpk.goal_consistency",
        "build_subtask_goal_contract_v31",
    ),
    "filter_goal_consistent_candidates_v31": (
        "roboharn_evo.agent.hpk.goal_consistency",
        "filter_goal_consistent_candidates_v31",
    ),
    "build_static_hpk_runtime": (
        "roboharn_evo.agent.hpk.static_runtime",
        "build_static_hpk_runtime",
    ),
    "load_hpk_snapshot": ("roboharn_evo.agent.hpk.store", "load_hpk_snapshot"),
    "load_hierarchical_store": (
        "roboharn_evo.agent.hpk.hierarchical_store",
        "load_hierarchical_store",
    ),
    "load_knowledge_family_catalog": (
        "roboharn_evo.agent.hpk.family_store",
        "load_knowledge_family_catalog",
    ),
    "merge_action_units": (
        "roboharn_evo.agent.hpk.hierarchical_store",
        "merge_action_units",
    ),
    "merge_subtask_units": (
        "roboharn_evo.agent.hpk.hierarchical_store",
        "merge_subtask_units",
    ),
    "render_hierarchical_physical_knowledge_context": (
        "roboharn_evo.agent.hpk.retriever",
        "render_hierarchical_physical_knowledge_context",
    ),
    "save_hierarchical_store": (
        "roboharn_evo.agent.hpk.hierarchical_store",
        "save_hierarchical_store",
    ),
    "save_hierarchical_store_with_catalog": (
        "roboharn_evo.agent.hpk.family_store",
        "save_hierarchical_store_with_catalog",
    ),
}


__all__ = [  # noqa: PLE0604 - lazy-export names are the dictionary's string keys
    "ABSTRACT_EFFECT_SCHEMA",
    "HPKAbstractEffectV1",
    "HPKConditionBuilder",
    "HPKConditionV1",
    "HPKEvidenceV1",
    "HPKGeometricStrategyV1",
    "HPKGeometryPolicy",
    "HPKTaskStrategyV1",
    "HPKUnresolved",
    "HPKValidationError",
    "HPK_OBSERVE_MODE",
    "HPK_OFF_MODE",
    "HPK_STATIC_MODE",
    "HPKRuntimePolicy",
    "AbstractEffectExtractor",
    "AbstractEffectV1",
    "AuditV1",
    "CandidateGeometryFeaturesV1",
    "ConditionBuilder",
    "ConditionV1",
    "ContextV1",
    "EntryV1",
    "EvidenceV1",
    "GeometricStrategyV1",
    "GeometricRankingRequest",
    "GeometryRankingResult",
    "ObserveAuditRecord",
    "PrivateRankingAuditSink",
    "PrivateRankingAuditV1",
    "SnapshotV1",
    "TaskStrategyNormalizer",
    "TaskStrategyV1",
    "build_action_effect_evidence",
    "build_condition",
    "build_evidence",
    "build_observe_audit",
    "build_public_usage_audit",
    "candidate_semantic_features",
    "canonical_json",
    "determine_evidence_verdict",
    "evidence_verdict",
    "extract_abstract_effect",
    "extract_candidate_features",
    "normalize_task_strategy",
    "parse_hpk_runtime_policy",
    "rank_operation_pose_candidates",
    "reject_private_transferable",
    "stable_content_id",
    *_P0C_LAZY_EXPORTS,
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _P0C_LAZY_EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module protocol
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
