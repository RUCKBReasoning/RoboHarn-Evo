from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboharn_evo.agent.action_geometry_repair_policy import (
    STRICT_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
    normalize_action_geometry_repair_pending_policy,
)
from roboharn_evo.agent.core import AgentSession, BaseAgentCard
from roboharn_evo.agent.hpk.compatibility import normalize_agent_knowledge_config
from roboharn_evo.agent.planner_scene_view import (
    LEGACY_PLANNER_CONTEXT_MODE,
    normalize_planner_context_mode,
)
from roboharn_evo.agent.skills.registry import SkillRegistry, build_default_skill_registry
from roboharn_evo.agent.trace_compaction import (
    COMPACT_TRACE_PAYLOAD_MODE,
    normalize_trace_payload_mode,
)
from roboharn_evo.models.backend_factory import build_executor_backend, build_planner_backend
from roboharn_evo.models.backend_interfaces import ExecutorBackend, PlannerBackend

VALID_OBSERVATION_CAMERAS = {
    "head",
    "left",
    "right",
    "third",
}

VALID_GRASP_TRANSPORT_POLICIES = {
    "strict",
    "evidence_only",
}


def _parse_grasp_transport_policy(value: Any) -> str:
    policy = str(value or "").strip().lower().replace("-", "_")
    if policy not in VALID_GRASP_TRANSPORT_POLICIES:
        choices = ", ".join(sorted(VALID_GRASP_TRANSPORT_POLICIES))
        raise ValueError(
            "agent.recovery.grasp_transport_policy must be one of "
            f"{{{choices}}}; got {value!r}"
        )
    return policy


def _parse_action_geometry_repair_pending_policy(value: Any) -> str:
    return normalize_action_geometry_repair_pending_policy(
        value,
        field_name=(
            "agent.recovery.action_geometry_repair_pending_policy"
        ),
    )


def _parse_planner_context_mode(value: Any) -> str:
    return normalize_planner_context_mode(
        value,
        field_name="agent.recovery.planner_context_mode",
    )


def _parse_observation_trace_payload_mode(value: Any) -> str:
    return normalize_trace_payload_mode(
        value,
        field_name="agent.rollout_dump.trace_payload_mode",
    )


def _normalize_observation_camera(value: Any, *, default: str = "head") -> str:
    text = str(value or "").strip().lower().replace("_camera", "")
    if text in VALID_OBSERVATION_CAMERAS:
        return text
    return default


def _parse_observation_cameras(value: Any, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or value == "":
        raw_items: list[Any] = []
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        raw_items = [item.strip().strip("'\"") for item in text.split(",") if item.strip()]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    cameras: list[str] = []
    for item in raw_items:
        camera = _normalize_observation_camera(item, default="")
        if camera and camera not in cameras:
            cameras.append(camera)
    if cameras:
        return tuple(cameras)
    return tuple(_normalize_observation_camera(item) for item in default)


@dataclass(frozen=True)
class AgentRuntimeConfig:
    enabled: bool
    mode: str
    decision_interval: int
    observation_preprocess_enabled: bool
    observation_preprocess_every_n_steps: int
    observation_preprocess_auto_objects: bool
    observation_preprocess_query_url: str
    observation_preprocess_normalization_url: str
    observation_preprocess_query_timeout_sec: int
    observation_preprocess_max_objects: int
    observation_preprocess_objects: tuple[dict[str, str], ...]
    observation_preprocess_backend: str
    observation_preprocess_camera: str
    observation_preprocess_cameras: tuple[str, ...]
    observation_verification_cameras: tuple[str, ...]
    observation_preprocess_service_url: str
    oracle_objects_enabled: bool
    oracle_objects_include_all: bool
    oracle_objects_max_objects: int
    observation_grounding_enabled: bool
    observation_grounding_camera: str
    observation_grounding_cameras: tuple[str, ...]
    observation_grounding_min_valid_ratio: float
    observation_grounding_max_points: int
    observation_grounding_approach_height_m: float
    scene_memory_enabled: bool
    scene_memory_max_missing_steps: int
    scene_memory_stable_distance_m: float
    scene_memory_temporal_action_geometry_distance_m: float
    scene_memory_temporal_window_size: int
    scene_memory_min_candidate_score: float
    scene_memory_max_object_z_extent_m: float
    scene_memory_max_object_xy_extent_m: float
    scene_memory_max_object_world_z_m: float
    scene_memory_robot_self_filter_radius_m: float
    scene_memory_robot_self_filter_z_margin_m: float
    stall_patience: int
    max_steps_per_skill: int
    max_retries_per_skill: int
    interrupt_on_skill_change: bool
    initial_memory_text: str
    recovery_reobserve_scene_enabled: bool
    execution_evidence_enabled: bool
    grasp_transport_policy: str
    release_guard_enabled: bool
    planner_context_mode: str
    observation_trace_payload_mode: str
    raw_trace_sidecar_enabled: bool
    gpt_scene_memory_sidecar_enabled: bool
    action_geometry_repair_pending_policy: str
    enable_executable_candidate_temporal_consistency: bool
    allow_memory_valid_final_grounded_action: bool
    enable_automatic_self_occlusion_visual_clearance: bool
    debug_recovery_enabled: bool
    debug_recovery_trigger_step: int
    debug_recovery_signal: str
    debug_recovery_reason: str
    debug_recovery_subtask: str
    debug_recovery_skill: str
    debug_recovery_wait_for_scene_memory: bool
    debug_recovery_max_wait_steps: int
    debug_recovery_retry_budget: int
    debug_recovery_pure_control: bool
    debug_recovery_max_rounds: int
    debug_recovery_repeat_signal: bool
    debug_recovery_bootstrap_with_planner: bool
    debug_recovery_skip_vla_rollout: bool
    pure_tool_control_enabled: bool
    pure_tool_control_trigger_step: int
    pure_tool_control_signal: str
    pure_tool_control_reason: str
    pure_tool_control_wait_for_scene_memory: bool
    pure_tool_control_max_wait_steps: int
    pure_tool_control_retry_budget: int
    pure_tool_control_max_rounds: int
    pure_tool_control_max_control_turns: int
    pure_tool_control_max_no_progress_control_turns: int
    pure_tool_control_backend_error_budget: int
    pure_tool_control_empty_plan_replan_threshold: int
    pure_tool_control_repeat_signal: bool
    pure_tool_control_bootstrap_with_planner: bool
    pure_tool_control_skip_vla_rollout: bool
    hpk_mode: str


@dataclass(frozen=True)
class HPKDurableFinalization:
    coordinated_result: Any
    receipt: Any
    semantic_result: Any = None


class RoboHarnAgentRuntime:
    """Compatibility adapter over the new top-level agent runtime skeleton."""

    def __init__(
        self,
        *,
        control_backend: PlannerBackend,
        executor_backend: ExecutorBackend,
        config: AgentRuntimeConfig,
        skill_registry: SkillRegistry | None = None,
        control_model_name: str = "",
        executor_name: str = "",
        ood_backend_config: dict[str, Any] | None = None,
        recovery_backend_config: dict[str, Any] | None = None,
        hpk_runtime: Any = None,
    ) -> None:
        self.control_runtime = control_backend
        self.executor_runtime = executor_backend
        self.config = config
        self.skill_registry = skill_registry or build_default_skill_registry()
        self.control_model_name = control_model_name
        self.executor_name = executor_name
        self.ood_backend_config = ood_backend_config
        self.recovery_backend_config = recovery_backend_config
        self.hpk_runtime = hpk_runtime
        self.agent_card = BaseAgentCard(
            config=config,
            control_runtime=control_backend,
            executor_runtime=executor_backend,
            ood_backend_config=ood_backend_config,
            recovery_backend_config=recovery_backend_config,
            memory_store=None,
            skill_registry=None,
            control_model_name=control_model_name,
            executor_name=executor_name,
            hpk_runtime=hpk_runtime,
        )
        self.session = AgentSession.build(agent_card=self.agent_card)
        self._configure_recovery_tool_ablations()

    def _configure_recovery_tool_ablations(self) -> None:
        """Apply runtime tool ablations after the Agent session is built.

        The dispatcher remains owned by the Agent, while experiment config is
        assembled here.  Keeping the capability mutation on the dispatcher
        makes both planner exposure and direct execution share one authority.
        """

        agent = getattr(self.session, "agent", None)
        dispatcher = getattr(agent, "_recovery_dispatcher", None)
        configure = getattr(
            dispatcher,
            "set_reobserve_scene_enabled",
            None,
        )
        if callable(configure):
            configure(
                self.config.recovery_reobserve_scene_enabled
            )

    @property
    def memory_store(self):
        return self.session.agent.memory_store

    def reset(self) -> None:
        self.session.reset()

    def run_eval_step(self, task_env: Any, observation: dict[str, Any]) -> None:
        self.session.run_eval_once(task_env, observation)

    def consume_recovery_observation(self, task_env: Any) -> dict[str, Any] | None:
        return self.session.agent.consume_recovery_observation(task_env)

    def set_trace_file(self, trace_file: str | None, *, episode_id: int = -1, seed: int = -1) -> None:
        self.session.agent.set_trace_file(trace_file, episode_id=episode_id, seed=seed)

    def set_rollout_dump_dir(self, rollout_dir: str | None) -> None:
        self.session.agent.set_rollout_dump_dir(rollout_dir)

    def finalize_rollout_videos(self, fps: int = 10) -> None:
        self.session.agent.finalize_rollout_videos(fps=fps)

    def write_rollout_meta(self, meta: dict[str, Any]) -> None:
        self.session.agent.write_rollout_meta(meta)

    def finalize_hpk_v3_episode(
        self,
        *,
        rollout_dir: str,
        result: str,
    ) -> dict[str, Any]:
        runtime = self.hpk_runtime
        finalize = getattr(runtime, "finalize_online_episode", None)
        if (
            runtime is None
            or getattr(runtime, "hierarchical_full_enabled", False) is not True
            or not callable(finalize)
        ):
            raise RuntimeError("HPK v3 episode finalization requires full mode")
        summary = finalize(rollout_dir=rollout_dir, result=result)
        if not isinstance(summary, dict):
            raise TypeError("HPK v3 finalization must return a summary object")
        return summary

    def write_hpk_public_episode_event(
        self,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        self.session.agent.write_hpk_public_episode_event(event, payload)

    def bind_hpk_sequential_run(self, run_binding: dict[str, Any]) -> dict[str, Any]:
        runtime = self.hpk_runtime
        binder = getattr(runtime, "bind_sequential_run", None)
        if not callable(binder):
            raise RuntimeError("HPK sequential run binding requires evolving mode")
        return binder(run_binding)

    def finalize_hpk_episode(
        self,
        *,
        manifest_path: str,
        trace_path: str,
        expected_manifest_sha256: str,
        expected_trace_sha256: str,
        created_at: str,
        receipt_path: str,
    ) -> HPKDurableFinalization:
        runtime = self.hpk_runtime
        if (
            runtime is None
            or getattr(runtime, "evolving_enabled", False) is not True
        ):
            raise RuntimeError("HPK episode finalization requires evolving mode")
        manifest = Path(manifest_path).resolve(strict=True)
        prior_result = runtime.finalization_result
        if prior_result is None:
            result = runtime.finalize_episode(
                manifest_path=manifest,
                trace_path=trace_path,
                expected_manifest_sha256=expected_manifest_sha256,
                expected_trace_sha256=expected_trace_sha256,
                created_at=created_at,
            )
        else:
            if runtime.finalized_manifest_path != manifest:
                raise RuntimeError(
                    "HPK receipt retry must use the finalized rollout manifest"
                )
            if (
                hashlib.sha256(manifest.read_bytes()).hexdigest()
                != expected_manifest_sha256
                or hashlib.sha256(Path(trace_path).resolve(strict=True).read_bytes()).hexdigest()
                != expected_trace_sha256
            ):
                raise RuntimeError("HPK receipt retry source hash differs")
            result = prior_result
        from roboharn_evo.agent.hpk.evolving_runtime import (
            publish_finalization_receipt_v1,
        )

        receipt = publish_finalization_receipt_v1(
            rollout_manifest_path=manifest,
            coordinated_result=result,
            receipt_path=receipt_path,
        )
        return HPKDurableFinalization(
            coordinated_result=result,
            receipt=receipt,
            semantic_result=runtime.semantic_finalization_result,
        )

    @property
    def hpk_trace_paths(self) -> dict[str, str | None]:
        agent = self.session.agent
        return {
            "public_trace": (
                None
                if agent._hpk_public_audit_file is None
                else str(agent._hpk_public_audit_file)
            ),
            "private_transition_trace": (
                None
                if agent._hpk_private_trace_file is None
                else str(agent._hpk_private_trace_file)
            ),
        }

    def publish_hpk_rollout_import_manifest(self, manifest_path: str) -> Any:
        runtime = self.hpk_runtime
        if (
            runtime is None
            or getattr(runtime, "evolving_enabled", False) is not True
        ):
            raise RuntimeError("HPK rollout manifest requires evolving mode")
        paths = self.hpk_trace_paths
        public_path = paths["public_trace"]
        private_path = paths["private_transition_trace"]
        if public_path is None or private_path is None:
            raise RuntimeError("HPK public/private trace paths are not configured")
        lease = runtime.episode_lease
        if lease is None:
            raise RuntimeError("HPK rollout manifest requires an active episode")
        from roboharn_evo.agent.hpk.evolving_runtime import (
            publish_rollout_import_manifest_v2,
        )

        return publish_rollout_import_manifest_v2(
            expected_episode_id=lease.expected_episode_id,
            public_trace_path=public_path,
            private_trace_path=private_path,
            manifest_path=manifest_path,
        )


def build_agent_runtime(usr_args: dict[str, Any]) -> RoboHarnAgentRuntime:
    planner_cfg = usr_args.get("planner", {})
    executor_cfg = usr_args.get("executor", {})
    agent_cfg = normalize_agent_knowledge_config(usr_args.get("agent", {}))
    ood_cfg = usr_args.get("ood", {})
    recovery_cfg = usr_args.get("recovery", {})
    preprocess_cfg = dict(agent_cfg.get("observation_preprocess", {}) or {})
    segment_cfg = dict(preprocess_cfg.get("segmentation", {}) or {})
    grounding_cfg = dict(preprocess_cfg.get("grounding", {}) or {})
    oracle_objects_cfg = dict(preprocess_cfg.get("oracle_objects", {}) or {})
    scene_memory_cfg = dict(preprocess_cfg.get("scene_memory", {}) or {})
    recovery_tool_cfg = dict(agent_cfg.get("recovery", {}) or {})
    execution_evidence_enabled = recovery_tool_cfg.get("enable_execution_evidence", False)
    if not isinstance(execution_evidence_enabled, bool):
        raise ValueError("agent.recovery.enable_execution_evidence must be boolean")
    rollout_dump_cfg = dict(agent_cfg.get("rollout_dump", {}) or {})
    from roboharn_evo.agent.experience.procedure_retriever import (
        resolve_procedure_experience_config,
    )

    procedure_experience_config = resolve_procedure_experience_config(
        agent_cfg.get("procedure_experience", {}),
        task_family=str(usr_args.get("task_name", "")),
    )
    raw_hpk_config = agent_cfg.get("hpk", {})
    if raw_hpk_config is None:
        raw_hpk_config = {}
    raw_hpk_mode = (
        str(raw_hpk_config.get("mode", "off") or "off").strip().lower()
        if isinstance(raw_hpk_config, dict)
        else "invalid"
    )
    raw_hpk_v3_config = agent_cfg.get("hpk_v3", {})
    if raw_hpk_v3_config is None:
        raw_hpk_v3_config = {}
    if not execution_evidence_enabled:
        raw_hpk_v3_config = {
            **raw_hpk_v3_config,
            "execution_evidence_enabled": False,
            "hpk_goal_consistency_enabled": False,
            "knowledge_updates_enabled": False,
            "hierarchical_reflection": False,
            "maintenance_rgb_enabled": False,
        }
        if raw_hpk_mode == "evolving":
            raise ValueError("Evolving HPK requires execution evidence; use read-only HPK v3 full")
    raw_hpk_v3_mode = (
        str(raw_hpk_v3_config.get("mode", "off") or "off").strip().lower()
        if isinstance(raw_hpk_v3_config, dict)
        else "invalid"
    )
    if raw_hpk_v3_mode not in {"off", "rule", "full"}:
        raise ValueError("agent.hpk_v3.mode must be off, rule, or full")
    if raw_hpk_mode != "off" and raw_hpk_v3_mode != "off":
        raise ValueError("legacy agent.hpk and agent.hpk_v3 cannot both be active")
    hpk_runtime = None
    hpk_mode = "off"
    if raw_hpk_v3_mode != "off":
        from roboharn_evo.agent.hpk.hierarchical_retriever import (
            build_hierarchical_hpk_runtime,
        )

        planner_agent_api = planner_cfg.get("agent_api", planner_cfg)
        if not isinstance(planner_agent_api, dict):
            raise ValueError("HPK v3 requires planner.agent_api configuration")
        hpk_runtime = build_hierarchical_hpk_runtime(
            raw_hpk_v3_config,
            planner_agent_api=planner_agent_api,
        )
        hpk_mode = f"v3_{raw_hpk_v3_mode}"
    elif raw_hpk_mode != "off":
        from roboharn_evo.agent.hpk.runtime_policy import parse_hpk_runtime_policy

        hpk_policy = parse_hpk_runtime_policy(raw_hpk_config)
        hpk_mode = hpk_policy.mode
        if hpk_policy.static_enabled:
            from roboharn_evo.agent.hpk.static_runtime import build_static_hpk_runtime

            hpk_runtime = build_static_hpk_runtime(
                hpk_policy,
                task_family=str(usr_args.get("task_name", "") or ""),
                domain_id=str(
                    usr_args.get("domain_id")
                    or usr_args.get("benchmark_id")
                    or "rmbench"
                ),
            )
        elif hpk_policy.evolving_enabled:
            from roboharn_evo.agent.hpk.evolving_runtime import build_evolving_hpk_runtime

            proposer = None
            if hpk_policy.proposer_enabled:
                from roboharn_evo.agent.hpk.vlm_strategy_proposer import (
                    AgentApiStrategyProposalBackend,
                    VLMStrategyProposer,
                )

                planner_agent_api = planner_cfg.get("agent_api", planner_cfg)
                if not isinstance(planner_agent_api, dict):
                    raise ValueError(
                        "HPK v2 proposer requires planner.agent_api configuration"
                    )
                proposer_headers = dict(
                    planner_agent_api.get("extra_headers", {}) or {}
                )
                auth_token = str(planner_agent_api.get("auth_token", "") or "")
                if auth_token:
                    proposer_headers[
                        str(
                            planner_agent_api.get(
                                "auth_header", "Authorization"
                            )
                        )
                    ] = auth_token
                proposer = VLMStrategyProposer(
                    AgentApiStrategyProposalBackend(
                        str(planner_agent_api["server_url"]),
                        timeout_sec=int(
                            planner_agent_api.get("timeout_sec", 600)
                        ),
                        headers=proposer_headers,
                    ),
                    max_proposals=hpk_policy.proposer_max_proposals,
                )

            hpk_runtime = build_evolving_hpk_runtime(
                hpk_policy,
                task_family=str(usr_args.get("task_name", "") or ""),
                domain_id=str(
                    usr_args.get("domain_id")
                    or usr_args.get("benchmark_id")
                    or "rmbench"
                ),
                runtime_provenance=usr_args.get("runtime_provenance"),
                proposer=proposer,
            )
    grasp_transport_policy = _parse_grasp_transport_policy(
        recovery_tool_cfg.get("grasp_transport_policy", "strict")
    )
    action_geometry_repair_pending_policy = (
        _parse_action_geometry_repair_pending_policy(
            recovery_tool_cfg.get(
                "action_geometry_repair_pending_policy",
                STRICT_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
            )
        )
    )
    planner_context_mode = _parse_planner_context_mode(
        recovery_tool_cfg.get(
            "planner_context_mode",
            LEGACY_PLANNER_CONTEXT_MODE,
        )
    )
    observation_trace_payload_mode = _parse_observation_trace_payload_mode(
        rollout_dump_cfg.get(
            "trace_payload_mode",
            COMPACT_TRACE_PAYLOAD_MODE,
        )
    )
    raw_trace_sidecar_enabled = bool(
        rollout_dump_cfg.get("raw_trace_sidecar_enabled", False)
    )
    gpt_scene_memory_sidecar_enabled = bool(
        rollout_dump_cfg.get("gpt_scene_memory_sidecar_enabled", True)
    )
    if (
        raw_trace_sidecar_enabled
        and observation_trace_payload_mode != "compact_v1"
    ):
        raise ValueError(
            "agent.rollout_dump.raw_trace_sidecar_enabled requires "
            "agent.rollout_dump.trace_payload_mode=compact_v1 so standard "
            "JSONL files remain compact"
        )
    debug_recovery_cfg = dict(agent_cfg.get("debug_recovery", {}) or {})
    pure_tool_control_cfg = dict(agent_cfg.get("pure_tool_control", {}) or {})
    query_url = str(preprocess_cfg.get("query_url", "http://127.0.0.1:9101/perception_queries"))
    normalization_url = str(preprocess_cfg.get("normalization_url", "") or "")
    if not normalization_url and query_url.endswith("/perception_queries"):
        normalization_url = query_url[: -len("/perception_queries")] + "/normalize_perception_queries"
    segment_objects: list[dict[str, str]] = []
    for item in segment_cfg.get("objects", []):
        if isinstance(item, str):
            name = item.strip()
            if name:
                segment_objects.append({"object_id": name, "text_prompt": name})
        elif isinstance(item, dict):
            object_id = str(item.get("object_id", item.get("name", ""))).strip()
            text_prompt = str(item.get("text_prompt", object_id)).strip()
            if object_id:
                role = str(item.get("role", item.get("query_role", "context"))).strip().lower().replace("-", "_").replace(" ", "_")
                if role not in {"target", "tool", "context"}:
                    role = "context"
                segment_objects.append(
                    {
                        "object_id": object_id,
                        "text_prompt": text_prompt or object_id,
                        "role": role,
                        "instance_hint": str(item.get("instance_hint", item.get("query_instance_hint", ""))).strip(),
                        "reason": str(item.get("reason", item.get("query_reason", ""))).strip(),
                    }
                )
    control_backend = build_planner_backend(
        planner_cfg,
        procedure_experience_config=procedure_experience_config,
        hpk_runtime=hpk_runtime,
        execution_evidence_enabled=execution_evidence_enabled,
    )
    executor_backend = build_executor_backend(executor_cfg)
    control_model_name = str(planner_cfg.get("backend", "reasoner"))
    executor_backend_name = str(executor_cfg.get("backend", "default_manipulation_policy"))
    segment_camera = _normalize_observation_camera(segment_cfg.get("camera", "head"))
    grounding_camera = _normalize_observation_camera(grounding_cfg.get("camera", segment_camera), default=segment_camera)
    segment_cameras = _parse_observation_cameras(segment_cfg.get("cameras", ()), default=(segment_camera,))
    verification_cameras = _parse_observation_cameras(
        segment_cfg.get("verification_cameras", ()),
        default=(),
    )
    grounding_cameras = _parse_observation_cameras(grounding_cfg.get("cameras", ()), default=())
    return RoboHarnAgentRuntime(
        control_backend=control_backend,
        executor_backend=executor_backend,
        config=AgentRuntimeConfig(
            enabled=bool(agent_cfg.get("enabled", False)),
            mode=str(agent_cfg.get("mode", "deployment")),
            decision_interval=max(1, int(agent_cfg.get("decision_interval", 1))),
            observation_preprocess_enabled=bool(preprocess_cfg.get("enabled", False)),
            observation_preprocess_every_n_steps=max(1, int(preprocess_cfg.get("every_n_steps", 1))),
            observation_preprocess_auto_objects=bool(preprocess_cfg.get("auto_objects", True)),
            observation_preprocess_query_url=query_url,
            observation_preprocess_normalization_url=normalization_url,
            observation_preprocess_query_timeout_sec=max(1, int(preprocess_cfg.get("query_timeout_sec", 120))),
            observation_preprocess_max_objects=max(1, int(preprocess_cfg.get("max_objects", 3))),
            observation_preprocess_objects=tuple(segment_objects),
            observation_preprocess_backend=str(segment_cfg.get("backend", "sam3")),
            observation_preprocess_camera=segment_camera,
            observation_preprocess_cameras=segment_cameras,
            observation_verification_cameras=verification_cameras,
            observation_preprocess_service_url=str(segment_cfg.get("service_url", "")),
            oracle_objects_enabled=bool(oracle_objects_cfg.get("enabled", False)),
            oracle_objects_include_all=bool(oracle_objects_cfg.get("include_all", True)),
            oracle_objects_max_objects=max(1, int(oracle_objects_cfg.get("max_objects", preprocess_cfg.get("max_objects", 12)))),
            observation_grounding_enabled=bool(grounding_cfg.get("enabled", False)),
            observation_grounding_camera=grounding_camera,
            observation_grounding_cameras=grounding_cameras,
            observation_grounding_min_valid_ratio=max(0.0, float(grounding_cfg.get("min_valid_ratio", 0.05))),
            observation_grounding_max_points=max(1, int(grounding_cfg.get("max_points", 5000))),
            observation_grounding_approach_height_m=float(grounding_cfg.get("approach_height_m", 0.08)),
            scene_memory_enabled=bool(scene_memory_cfg.get("enabled", True)),
            scene_memory_max_missing_steps=max(1, int(scene_memory_cfg.get("max_missing_steps", 20))),
            scene_memory_stable_distance_m=max(0.0, float(scene_memory_cfg.get("stable_distance_m", 0.08))),
            scene_memory_temporal_action_geometry_distance_m=max(
                0.0,
                float(scene_memory_cfg.get("temporal_action_geometry_distance_m", 0.04)),
            ),
            scene_memory_temporal_window_size=max(1, int(scene_memory_cfg.get("temporal_window_size", 8))),
            scene_memory_min_candidate_score=max(0.0, float(scene_memory_cfg.get("min_candidate_score", 0.12))),
            scene_memory_max_object_z_extent_m=max(0.0, float(scene_memory_cfg.get("max_object_z_extent_m", 0.18))),
            scene_memory_max_object_xy_extent_m=max(0.0, float(scene_memory_cfg.get("max_object_xy_extent_m", 0.30))),
            scene_memory_max_object_world_z_m=max(0.0, float(scene_memory_cfg.get("max_object_world_z_m", 0.95))),
            scene_memory_robot_self_filter_radius_m=max(0.0, float(scene_memory_cfg.get("robot_self_filter_radius_m", 0.08))),
            scene_memory_robot_self_filter_z_margin_m=max(0.0, float(scene_memory_cfg.get("robot_self_filter_z_margin_m", 0.08))),
            stall_patience=max(1, int(agent_cfg.get("monitor", {}).get("stall_patience", 12))),
            max_steps_per_skill=max(1, int(agent_cfg.get("monitor", {}).get("max_steps_per_skill", 80))),
            max_retries_per_skill=max(0, int(agent_cfg.get("monitor", {}).get("max_retries_per_skill", 2))),
            interrupt_on_skill_change=bool(agent_cfg.get("interrupt_on_skill_change", True)),
            initial_memory_text=str(usr_args.get("initial_memory_text", "The task has started.")),
            recovery_reobserve_scene_enabled=bool(
                execution_evidence_enabled and recovery_tool_cfg.get("enable_reobserve", True)
            ),
            execution_evidence_enabled=execution_evidence_enabled,
            grasp_transport_policy=grasp_transport_policy,
            release_guard_enabled=bool(
                recovery_tool_cfg.get("enable_release_guard", False)
            ),
            planner_context_mode=planner_context_mode,
            observation_trace_payload_mode=(
                observation_trace_payload_mode
            ),
            raw_trace_sidecar_enabled=raw_trace_sidecar_enabled,
            gpt_scene_memory_sidecar_enabled=(
                gpt_scene_memory_sidecar_enabled
            ),
            action_geometry_repair_pending_policy=(
                action_geometry_repair_pending_policy
            ),
            enable_executable_candidate_temporal_consistency=bool(
                recovery_tool_cfg.get(
                    "enable_executable_candidate_temporal_consistency",
                    True,
                )
            ),
            allow_memory_valid_final_grounded_action=bool(
                recovery_tool_cfg.get(
                    "allow_memory_valid_final_grounded_action",
                    False,
                )
            ),
            enable_automatic_self_occlusion_visual_clearance=bool(
                recovery_tool_cfg.get(
                    "enable_automatic_self_occlusion_visual_clearance",
                    True,
                )
            ),
            debug_recovery_enabled=bool(debug_recovery_cfg.get("enabled", False)),
            debug_recovery_trigger_step=max(0, int(debug_recovery_cfg.get("trigger_step", 0))),
            debug_recovery_signal=str(debug_recovery_cfg.get("signal", "motion_blocked")),
            debug_recovery_reason=str(debug_recovery_cfg.get("reason", "debug forced recovery loop")),
            debug_recovery_subtask=str(debug_recovery_cfg.get("subtask", "")),
            debug_recovery_skill=str(debug_recovery_cfg.get("skill", "monitored-subtask-execution")),
            debug_recovery_wait_for_scene_memory=bool(debug_recovery_cfg.get("wait_for_scene_memory", True)),
            debug_recovery_max_wait_steps=max(0, int(debug_recovery_cfg.get("max_wait_steps", 10))),
            debug_recovery_retry_budget=max(0, int(debug_recovery_cfg.get("retry_budget", 1))),
            debug_recovery_pure_control=bool(debug_recovery_cfg.get("pure_control", False)),
            debug_recovery_max_rounds=max(1, int(debug_recovery_cfg.get("max_rounds", 100))),
            debug_recovery_repeat_signal=bool(debug_recovery_cfg.get("repeat_signal", False)),
            debug_recovery_bootstrap_with_planner=bool(debug_recovery_cfg.get("bootstrap_with_planner", False)),
            debug_recovery_skip_vla_rollout=bool(debug_recovery_cfg.get("skip_vla_rollout", False)),
            pure_tool_control_enabled=bool(pure_tool_control_cfg.get("enabled", False)),
            pure_tool_control_trigger_step=max(0, int(pure_tool_control_cfg.get("trigger_step", debug_recovery_cfg.get("trigger_step", 0)))),
            pure_tool_control_signal=str(pure_tool_control_cfg.get("signal", "task_level_recovery_control")),
            pure_tool_control_reason=str(pure_tool_control_cfg.get("reason", "pure tool-control ablation")),
            pure_tool_control_wait_for_scene_memory=bool(pure_tool_control_cfg.get("wait_for_scene_memory", debug_recovery_cfg.get("wait_for_scene_memory", True))),
            pure_tool_control_max_wait_steps=max(0, int(pure_tool_control_cfg.get("max_wait_steps", debug_recovery_cfg.get("max_wait_steps", 10)))),
            pure_tool_control_retry_budget=max(0, int(pure_tool_control_cfg.get("retry_budget", debug_recovery_cfg.get("retry_budget", 1)))),
            pure_tool_control_max_rounds=max(1, int(pure_tool_control_cfg.get("max_rounds", 10))),
            pure_tool_control_max_control_turns=max(0, int(pure_tool_control_cfg.get("max_control_turns", 64))),
            pure_tool_control_max_no_progress_control_turns=max(
                0,
                int(pure_tool_control_cfg.get("max_no_progress_control_turns", 10)),
            ),
            pure_tool_control_backend_error_budget=max(1, int(pure_tool_control_cfg.get("backend_error_budget", 5))),
            pure_tool_control_empty_plan_replan_threshold=max(1, int(pure_tool_control_cfg.get("empty_plan_replan_threshold", 2))),
            pure_tool_control_repeat_signal=bool(pure_tool_control_cfg.get("repeat_signal", True)),
            pure_tool_control_bootstrap_with_planner=bool(pure_tool_control_cfg.get("bootstrap_with_planner", True)),
            pure_tool_control_skip_vla_rollout=bool(pure_tool_control_cfg.get("skip_vla_rollout", True)),
            hpk_mode=hpk_mode,
        ),
        control_model_name=control_model_name,
        executor_name=executor_backend_name,
        ood_backend_config=ood_cfg,
        recovery_backend_config=recovery_cfg,
        hpk_runtime=hpk_runtime,
    )
