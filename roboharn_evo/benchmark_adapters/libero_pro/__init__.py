"""Public, simulator-free LIBERO-PRO adapter boundary."""

from __future__ import annotations

from typing import Any

from roboharn_evo.benchmark_adapters.base import BenchmarkAdapter
from roboharn_evo.benchmark_adapters.libero_pro.adapter import LiberoProAdapter
from roboharn_evo.benchmark_adapters.libero_pro.agent_bridge import (
    LIBERO_AGENT_PLANNER_PROMPT,
    LiberoProAgentBridge,
)
from roboharn_evo.benchmark_adapters.libero_pro.contracts import (
    CANONICAL_CAMERA_KEYS,
    CANONICAL_PROPRIOCEPTION_KEYS,
    LiberoProActionContract,
    LiberoProCapabilities,
    LiberoProCapabilityError,
    LiberoProContractError,
    LiberoProObservationContract,
    RoboHarnCapabilityReport,
    assess_roboharn_agent_compatibility,
)
from roboharn_evo.benchmark_adapters.libero_pro.effect_verifier import (
    LiberoDeterministicEffectVerifier,
    LiberoEffectVerdict,
    LiberoEffectVerifierConfig,
)
from roboharn_evo.benchmark_adapters.libero_pro.expert_task_knowledge import (
    LiberoExpertKnowledgeError,
    LiberoExpertReflectionSource,
    LiberoTaskKnowledgeBuildResult,
    build_task_only_runtime,
    build_task_only_store_from_expert,
    load_expert_reflection_source,
    prepare_expert_reflection,
    uniform_temporal_boundaries,
)
from roboharn_evo.benchmark_adapters.libero_pro.loop_ablation import (
    LiberoLoopAblationError,
    audit_loop_profile_ablation,
)
from roboharn_evo.benchmark_adapters.libero_pro.native_policy import (
    LiberoNativePolicyError,
    LiberoPi05PolicyBackend,
    LiberoPi05PolicyConfig,
    build_pi05_libero_input,
    build_pi05_libero_state,
)
from roboharn_evo.benchmark_adapters.libero_pro.native_rollout import (
    LiberoNativeChunkPolicy,
    LiberoNativeOffResult,
    LiberoNativeRolloutError,
    NativeActionRecord,
    run_libero_native_off_episode,
    validate_libero_native_action_chunk,
)
from roboharn_evo.benchmark_adapters.libero_pro.p0_feasibility import (
    LiberoP0FeasibilityError,
    audit_p0_feasibility_matrix,
)
from roboharn_evo.benchmark_adapters.libero_pro.perception import (
    LiberoPerceptionError,
    LiberoSAM3PerceptionClient,
    PerceptionDetection,
    PerceptionFrame,
    PerceptionQuery,
)
from roboharn_evo.benchmark_adapters.libero_pro.scene_memory import (
    LiberoPerceptionMemoryRuntime,
    LiberoSceneMemory,
)
from roboharn_evo.benchmark_adapters.libero_pro.transfer_evaluation import (
    LiberoTransferEvaluationError,
    audit_transfer_experiment,
)
from roboharn_evo.benchmark_adapters.libero_pro.transfer_knowledge import (
    LiberoActionKnowledgePromptRuntime,
    LiberoTaskActionRuntimes,
    LiberoTransferKnowledgeError,
    LiberoTransferStoreBundle,
    action_prompt_grounding_json_schema,
    build_action_only_runtime,
    build_augmented_action_store,
    build_task_action_runtimes,
    build_transfer_store_bundle,
    build_transfer_task_only_runtime,
)


class LiberoProAdapterBoundary(BenchmarkAdapter[Any]):
    """Deprecated abstract marker retained for import compatibility.

    New integrations should instantiate :class:`LiberoProAdapter`.
    """


__all__ = [
    "CANONICAL_CAMERA_KEYS",
    "CANONICAL_PROPRIOCEPTION_KEYS",
    "LIBERO_AGENT_PLANNER_PROMPT",
    "LiberoActionKnowledgePromptRuntime",
    "LiberoDeterministicEffectVerifier",
    "LiberoEffectVerdict",
    "LiberoEffectVerifierConfig",
    "LiberoExpertKnowledgeError",
    "LiberoExpertReflectionSource",
    "LiberoLoopAblationError",
    "LiberoNativeChunkPolicy",
    "LiberoNativeOffResult",
    "LiberoNativePolicyError",
    "LiberoNativeRolloutError",
    "LiberoP0FeasibilityError",
    "LiberoPerceptionError",
    "LiberoPerceptionMemoryRuntime",
    "LiberoPi05PolicyBackend",
    "LiberoPi05PolicyConfig",
    "LiberoProActionContract",
    "LiberoProAdapter",
    "LiberoProAdapterBoundary",
    "LiberoProAgentBridge",
    "LiberoProCapabilities",
    "LiberoProCapabilityError",
    "LiberoProContractError",
    "LiberoProObservationContract",
    "LiberoSAM3PerceptionClient",
    "LiberoSceneMemory",
    "LiberoTaskActionRuntimes",
    "LiberoTaskKnowledgeBuildResult",
    "LiberoTransferEvaluationError",
    "LiberoTransferKnowledgeError",
    "LiberoTransferStoreBundle",
    "NativeActionRecord",
    "PerceptionDetection",
    "PerceptionFrame",
    "PerceptionQuery",
    "RoboHarnCapabilityReport",
    "action_prompt_grounding_json_schema",
    "assess_roboharn_agent_compatibility",
    "audit_loop_profile_ablation",
    "audit_p0_feasibility_matrix",
    "audit_transfer_experiment",
    "build_action_only_runtime",
    "build_augmented_action_store",
    "build_pi05_libero_input",
    "build_pi05_libero_state",
    "build_task_action_runtimes",
    "build_task_only_runtime",
    "build_task_only_store_from_expert",
    "build_transfer_store_bundle",
    "build_transfer_task_only_runtime",
    "load_expert_reflection_source",
    "prepare_expert_reflection",
    "run_libero_native_off_episode",
    "uniform_temporal_boundaries",
    "validate_libero_native_action_chunk",
]
