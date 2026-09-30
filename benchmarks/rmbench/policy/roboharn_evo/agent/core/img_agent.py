from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import math
import os
import time
from pathlib import Path
from typing import Any
from urllib import request

import imageio.v3 as iio
from PIL import Image

from ..action_geometry_repair_policy import (
    DISABLED_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
    SAFE_MOTION_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
    STRICT_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
    normalize_action_geometry_repair_pending_policy,
    plan_repair_pending_safe_motion,
    repair_pending_retreat_is_safe,
)
from ..arm_contract import normalize_physical_arm, normalize_preferred_arm
from ..components.agent_tools.local_skill_registry import LocalSkillRegistry
from .base_agent import BaseAgent
from ..decisions import ControlSignal
from ..environment import EnvSnapshot, RMBenchEnvAdapter
from roboharn_evo.agent.gripper_state import gripper_command_state
from ..grounded_target_pose_contract import (
    authorize_memory_valid_final_grounded_action,
)
from ..evidence_acquisition_policy import (
    EvidenceAcquisitionContext,
    EvidenceAcquisitionPolicy,
    EvidenceSituation,
)
from ..grasp_attachment_contract import (
    apply_grasp_motion_validation,
    capture_grasp_motion_snapshot,
    matched_grasp_motion_validation,
    pending_grasp_observation_attempted,
    validate_grasp_motion_effect,
    validate_pending_grasp_observation_effect,
)
from ..gpt_scene_memory_sidecar import (
    append_gpt_scene_memory_request,
    resolve_gpt_scene_memory_sidecar_path,
)
from ..experience import (
    fallback_semantic_tags,
    has_semantic_tags,
    merge_semantic_tags,
    normalize_semantic_tags,
)
from ..monitoring import (
    HandoffPolicy,
    INVALID_ACTION_PATTERN,
    MonitorSignal,
    MOTION_BLOCKED,
    NEEDS_RECOVERY_TOOLS,
    OBJECT_NOT_VISIBLE,
    OODDetector,
    OODSkillEvaluator,
    ProgressMonitor,
    RUNNING,
    SCENE_DRIFT_DETECTED,
    STALL_DETECTED,
    STEP_BUDGET_EXHAUSTED,
    TASK_SUCCESS,
    TASK_LEVEL_RECOVERY_CONTROL,
    make_signal,
)
from ..operation_candidates import (
    OPERATION_TARGETS_KEY,
    SPATIAL_STATE_KEY,
    SUPPORTED_OPERATION_ACTION_MODES,
    capture_held_object_to_tcp_attachment,
    manipulation_state_allows_transport,
    materialize_operation_candidate,
    operation_action_mode,
    operation_pose_candidates,
    place_target_tolerance_m,
    propagate_held_object_world_m,
    select_operation_pose_candidate,
    validate_place_candidate,
    uses_operation_pose_candidate,
    with_dynamic_place_candidates,
)
from ..operation_candidate_lifecycle import (
    OperationCandidateLifecycle,
    OperationCandidateLifecycleLimits,
    candidate_geometry,
    candidate_geometry_revision,
)
from ..operation_geometry_refresh_contract import (
    OperationGeometryRefreshContract,
)
from ..planner_state_projection import (
    public_manipulation_state,
)
from ..grasp_lifecycle import (
    apply_close_time_grasp_transport_policy,
    apply_grasp_close_boundary_effect,
    bind_pending_grasp_attachment,
    grasp_attempt_metadata,
    is_runtime_grasp_diagnostic_lift_call,
    merge_pending_grasp_validation,
    promote_verified_grasp_attachment,
    stage_grasp_verification_boundary,
)
from ..placement_release_contract import (
    SupportContactLimits,
    authorize_support_contact_release,
    clear_support_contact_release_state,
    create_support_contact_release_evidence,
    drop_redundant_support_contact_settles,
    install_support_contact_release_state,
    release_identity_recovery_anchors,
)
from ..relational_place_targets import (
    REFERENCE_REGIONS_KEY,
    compact_reference_regions,
)
from ..paths import rmbench_root, roboharn_skills_dir
from ..planner_scene_view import (
    COMPACT_PLANNER_CONTEXT_MODE,
    build_planner_scene_view,
    compact_finalized_observation_event,
    normalize_planner_context_mode,
)
from ..trace_compaction import (
    COMPACT_TRACE_PAYLOAD_MODE,
    compact_trace_event,
    normalize_trace_payload_mode,
)
from ..perception import SceneMemoryTracker, ground_segmentation_result
from ..perception.scene_memory import (
    POSITION_CURRENT_VERIFIED,
    POSITION_MEMORY_VALID,
    POSITION_MOTION_UNCERTAIN,
    POSITION_ONLY_ACTION_GEOMETRY_QUARANTINE_LEASE_KEY,
    instance_has_finite_grounding,
    scene_candidate_quality,
)
from ..perception.query_normalization import (
    matching_perception_query_index,
    merge_instance_hints,
    merge_perception_query,
    REFERENCE_QUERY_METADATA_KEYS,
    normalize_entity_scope,
    normalized_relation_metadata,
    normalize_perception_object_id,
    normalize_query_role,
)
from ..reasoners.prompts import build_reasoner_turn_payload
from ..recovery import (
    ActionEffectVerifier,
    ActionEffectVerifierBackendError,
    RecoveryPolicyResolver,
    RecoveryToolDispatcher,
    SkillRecoveryWorkflowLoader,
)
from ..recovery.ambiguous_grasp_return import (
    is_runtime_ambiguous_grasp_return_call,
    plan_ambiguous_grasp_return,
    reduce_ambiguous_grasp_return_results,
)
from ..recovery.tool_specs import (
    REOBSERVE_SCENE_TOOL,
    RecoveryToolCall,
)
from ..recovery.failed_grasp_clearance import (
    apply_failed_grasp_clearance_results,
    is_runtime_failed_grasp_clearance_call,
    stage_failed_grasp_clearance,
)
from ..rollout_video import encode_rollout_video
from ..vla import VLAInstructionBuilder, build_vla_execution_request


_MAX_GROUNDED_CONTACT_OVERTRAVEL_M = 0.01
_MIN_GROUNDED_CONTACT_ALIGNMENT_COSINE = 0.7
_MIN_DUAL_ARM_TARGET_CLEARANCE_M = 0.10
_GROUNDED_SETUP_NO_PROGRESS_EPSILON_M = 0.002
_GROUNDED_SETUP_FAILURE_THRESHOLD = 2
_GROUNDED_GEOMETRY_LEASE_MAX_POSE_DRIFT_M = 0.015
_GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M = 0.01
_GROUNDED_VISUAL_CLEARANCE_MAX_APPROACH_DRIFT_M = 0.03
_PARTIAL_APPROACH_CONTINUATION_MAX_RESIDUAL_M = 0.04
_DIAGNOSTIC_GRASP_MIN_LIFT_M = 0.01
_DIAGNOSTIC_GRASP_MAX_MOTION_M = 0.04
_DIAGNOSTIC_GRASP_MIN_DIRECTION_COSINE = 0.8
_RECOVERY_BACKEND_ERROR = "__pure_tool_control_recovery_backend_error__"
_RECOVERY_EMPTY_PLAN = "__pure_tool_control_empty_recovery_plan__"
_RECOVERY_INTERNAL_PROGRESS = "__pure_tool_control_internal_recovery_progress__"
_MANIPULATION_VERIFICATION_CAMERA_PHASES = {
    "grasp_candidate",
    "ambiguous_grasp_return_pending",
    "holding_provisional",
    "release_pending_verification",
    "release_recovery_required",
}
_ROBOT_EMBODIMENT_QUERY_IDS = {
    "gripper",
    "left_gripper",
    "right_gripper",
    "robot_gripper",
    "left_robot_gripper",
    "right_robot_gripper",
    "end_effector",
    "left_end_effector",
    "right_end_effector",
    "robot_end_effector",
    "tcp",
    "left_tcp",
    "right_tcp",
    "robot_tcp",
    "wrist",
    "left_wrist",
    "right_wrist",
    "robot_wrist",
    "robot_arm",
    "left_robot_arm",
    "right_robot_arm",
}


def _run_agent_tool_sync(coro) -> str:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return str(asyncio.run(coro))
    raise RuntimeError("Cannot run observation preprocess tools while an event loop is already running")


class ImgAgent(BaseAgent):
    def __init__(self, agent_card) -> None:
        super().__init__(agent_card)
        self.current_instruction = ""
        self.segment_start_snapshot: EnvSnapshot | None = None
        self.previous_snapshot: EnvSnapshot | None = None
        self.latest_snapshot: EnvSnapshot | None = None
        self._vla_instruction_builder = VLAInstructionBuilder()
        self._handoff_policy = HandoffPolicy()
        self._progress_monitor = ProgressMonitor()
        self._evidence_acquisition_policy = EvidenceAcquisitionPolicy()
        self._last_evidence_acquisition_decision: dict[str, Any] = {}
        self._pending_internal_recovery_completion = ""
        self._scene_memory_tracker = SceneMemoryTracker(
            max_missing_steps=int(getattr(self.config, "scene_memory_max_missing_steps", 20)),
            stable_distance_m=float(getattr(self.config, "scene_memory_stable_distance_m", 0.08)),
            temporal_action_geometry_distance_m=float(
                getattr(self.config, "scene_memory_temporal_action_geometry_distance_m", 0.04)
            ),
            temporal_window_size=int(getattr(self.config, "scene_memory_temporal_window_size", 8)),
            min_candidate_score=float(getattr(self.config, "scene_memory_min_candidate_score", 0.12)),
            max_object_z_extent_m=float(getattr(self.config, "scene_memory_max_object_z_extent_m", 0.18)),
            max_object_xy_extent_m=float(getattr(self.config, "scene_memory_max_object_xy_extent_m", 0.30)),
            max_object_world_z_m=float(getattr(self.config, "scene_memory_max_object_world_z_m", 0.95)),
            robot_self_filter_radius_m=float(getattr(self.config, "scene_memory_robot_self_filter_radius_m", 0.08)),
            robot_self_filter_z_margin_m=float(getattr(self.config, "scene_memory_robot_self_filter_z_margin_m", 0.08)),
            enable_executable_candidate_temporal_consistency=bool(
                getattr(
                    self.config,
                    "enable_executable_candidate_temporal_consistency",
                    True,
                )
            ),
        )
        self._ood_detector = OODDetector(
            evaluate_ood=OODSkillEvaluator(
                skill_registry=LocalSkillRegistry(
                    configured_paths=[str(roboharn_skills_dir())],
                    workspace_root=str(rmbench_root()),
                ),
                backend_config=agent_card.ood_backend_config,
            ).evaluate
        )
        self._recovery_skill_registry = LocalSkillRegistry(
            configured_paths=[str(roboharn_skills_dir())],
            workspace_root=str(rmbench_root()),
        )
        self._recovery_policy_resolver = RecoveryPolicyResolver(
            SkillRecoveryWorkflowLoader(
                self._recovery_skill_registry,
                backend_config=agent_card.recovery_backend_config,
                request_observer=(
                    self._record_recovery_planner_scene_memory_request
                ),
            )
        )
        self._action_effect_verifier = ActionEffectVerifier(
            skill_registry=self._recovery_skill_registry,
            backend_config=agent_card.recovery_backend_config,
            request_observer=(
                self._record_action_effect_scene_memory_request
            ),
        )
        self._recovery_dispatcher = RecoveryToolDispatcher(
            oracle_objects_enabled=bool(
                getattr(self.config, "oracle_objects_enabled", False)
            )
        )
        self._trace_file: Path | None = None
        self._current_episode_id: int = -1
        self._current_seed: int = -1
        self._rollout_dir: Path | None = None
        self._rollout_events_file: Path | None = None
        self._raw_perception_sidecar_file: Path | None = None
        self._raw_perception_sidecar_capture_keys: set[
            tuple[int, int, int]
        ] = set()
        self._gpt_scene_memory_sidecar_file: Path | None = None
        self._gpt_scene_memory_request_index = 0
        self._last_preprocessed_snapshot_key: tuple[str, ...] | None = None
        self._observation_preprocess_generation = 0
        self._observation_capture_generation = 0
        self._debug_recovery_triggered = False
        self._debug_recovery_rounds = 0
        self._debug_recovery_scene_wait_turns = 0
        self._identity_binding_retry_attempts = 0
        self._identity_binding_retry_skill_id = ""
        self._debug_recovery_planner_bootstrapped = False
        self._pure_tool_control_control_turns = 0
        self._pure_tool_control_no_progress_control_turns = 0
        self._pure_tool_control_control_backend_errors = 0
        self._pure_tool_control_recovery_backend_errors = 0
        self._last_recovery_backend_error = ""
        self._last_recovery_backend_error_stage = ""
        self._pending_action_effect_verification: dict[str, Any] | None = None
        self._pure_tool_control_empty_plan_turns = 0
        self._pure_tool_control_terminal_failure = ""
        self._authoritative_environment_success = False
        self._grounded_setup_failures: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._blocked_grounded_setups: set[tuple[str, str, str]] = set()
        self._operation_candidate_lifecycle = (
            OperationCandidateLifecycle(
                OperationCandidateLifecycleLimits(
                    failure_threshold=(
                        _GROUNDED_SETUP_FAILURE_THRESHOLD
                    ),
                    progress_epsilon_m=(
                        _GROUNDED_SETUP_NO_PROGRESS_EPSILON_M
                    ),
                )
            )
        )
        self._operation_geometry_refresh_contract = (
            OperationGeometryRefreshContract()
        )
        self._grounded_geometry_leases: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._partial_grounded_approach_leases: dict[
            tuple[str, str, str], dict[str, Any]
        ] = {}
        self._pending_recovery_observation: dict[str, Any] | None = None
        self._recent_release_resolutions: dict[str, dict[str, Any]] = {}

    def set_trace_file(self, trace_file: str | None, *, episode_id: int = -1, seed: int = -1) -> None:
        self._trace_file = Path(trace_file) if trace_file else None
        self._current_episode_id = int(episode_id)
        self._current_seed = int(seed)
        self._raw_perception_sidecar_capture_keys.clear()
        self._gpt_scene_memory_request_index = 0
        self._configure_raw_perception_sidecar_file()
        self._configure_gpt_scene_memory_sidecar_file()

    def _trace(self, event: str, **payload: Any) -> None:
        env_step = getattr(self.latest_snapshot, "step_count", -1) if self.latest_snapshot is not None else -1
        record = {
            "event": event,
            "timestamp": time.time(),
            "episode_id": self._current_episode_id,
            "seed": self._current_seed,
            "env_step": env_step,
            **payload,
        }
        try:
            print("[agent] " + json.dumps(record, ensure_ascii=False), flush=True)
        except Exception:
            print(f"[agent] {event} {payload}", flush=True)
        if self._trace_file is not None:
            try:
                self._trace_file.parent.mkdir(parents=True, exist_ok=True)
                with self._trace_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def set_rollout_dump_dir(self, rollout_dir: str | None) -> None:
        self._rollout_dir = Path(rollout_dir) if rollout_dir else None
        self._rollout_events_file = None if self._rollout_dir is None else self._rollout_dir / "events.jsonl"
        self._configure_raw_perception_sidecar_file()
        self._configure_gpt_scene_memory_sidecar_file()
        if self._rollout_dir is not None:
            for name in (
                "head",
                "left",
                "right",
                "third",
                "video",
            ):
                (self._rollout_dir / name).mkdir(parents=True, exist_ok=True)

    def _configure_raw_perception_sidecar_file(self) -> None:
        if self._rollout_dir is not None:
            path = self._rollout_dir / "raw_perception.jsonl"
        elif self._trace_file is not None:
            stem = self._trace_file.stem
            if stem.endswith("_agent_trace"):
                stem = stem[: -len("_agent_trace")] + "_raw_perception"
            else:
                stem = stem + "_raw_perception"
            path = self._trace_file.with_name(stem + ".jsonl")
        else:
            path = None
        self._raw_perception_sidecar_file = path
        if (
            path is None
            or not bool(
                getattr(
                    self.config,
                    "raw_trace_sidecar_enabled",
                    False,
                )
            )
            or not path.is_file()
        ):
            return
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                        key = (
                            int(record.get("episode_id", -1)),
                            int(record.get("seed", -1)),
                            int(record["observation_capture_id"]),
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue
                    self._raw_perception_sidecar_capture_keys.add(key)
        except Exception:
            pass

    def _configure_gpt_scene_memory_sidecar_file(self) -> None:
        """Select the per-episode audit file for model-visible Scene Memory."""

        self._gpt_scene_memory_sidecar_file = (
            resolve_gpt_scene_memory_sidecar_path(
                trace_file=self._trace_file,
                rollout_dir=self._rollout_dir,
            )
        )

    def _record_gpt_scene_memory_request(
        self,
        *,
        consumer: str,
        scene_memory: dict[str, Any],
        request_metadata: dict[str, Any] | None = None,
        scene_memory_views: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """Write the exact Scene Memory section attached to one GPT request.

        The standard trace records state transitions.  This sidecar instead
        records request attempts, including repeated requests with identical
        state, so loops remain visible during review.  Serialization and I/O
        are deliberately fail-open and can never block model inference.
        """

        try:
            if not bool(
                getattr(
                    self.config,
                    "gpt_scene_memory_sidecar_enabled",
                    True,
                )
            ):
                return ""
            path = self._gpt_scene_memory_sidecar_file
            if path is None:
                return ""

            self._gpt_scene_memory_request_index += 1
            request_index = self._gpt_scene_memory_request_index
            perception_status = scene_memory.get("perception_status")
            if not isinstance(perception_status, dict):
                perception_status = {}
            observation_preprocess = (
                self.memory_store.state.working.observation_preprocess
            )
            if not isinstance(observation_preprocess, dict):
                observation_preprocess = {}
            env_step = scene_memory.get("env_step")
            if env_step is None:
                env_step = (
                    getattr(self.latest_snapshot, "step_count", -1)
                    if self.latest_snapshot is not None
                    else -1
                )
            return append_gpt_scene_memory_request(
                path=path,
                request_index=request_index,
                consumer=consumer,
                scene_memory=scene_memory,
                episode_id=self._current_episode_id,
                seed=self._current_seed,
                env_step=env_step,
                planner_context_mode=self._planner_context_mode(),
                observation_generation=perception_status.get(
                    "observation_generation",
                    observation_preprocess.get("observation_generation"),
                ),
                observation_capture_id=perception_status.get(
                    "observation_capture_id",
                    observation_preprocess.get("observation_capture_id"),
                ),
                request_metadata=request_metadata,
                scene_memory_views=scene_memory_views,
            )
        except Exception:
            return ""

    def _record_recovery_planner_scene_memory_request(
        self,
        request_payload: dict[str, Any],
    ) -> None:
        scene_memory = request_payload.get("scene_memory")
        if not isinstance(scene_memory, dict):
            return
        self._record_gpt_scene_memory_request(
            consumer="recovery_planner",
            scene_memory=scene_memory,
            request_metadata={
                "signal_name": request_payload.get("signal_name", ""),
                "ood_scenario": request_payload.get("OOD_scenario", ""),
                "current_subtask": request_payload.get(
                    "current_subtask", ""
                ),
                "preferred_arm": request_payload.get("preferred_arm", ""),
                "skill_payload_mode": request_payload.get(
                    "skill_payload_mode", ""
                ),
            },
        )

    def _record_action_effect_scene_memory_request(
        self,
        request_payload: dict[str, Any],
    ) -> None:
        evidence = request_payload.get("evidence_payload")
        if not isinstance(evidence, dict):
            return
        pre = evidence.get("pre")
        post = evidence.get("post")
        pre_scene = (
            pre.get("scene_memory", {}) if isinstance(pre, dict) else {}
        )
        post_scene = (
            post.get("scene_memory", {}) if isinstance(post, dict) else {}
        )
        if not isinstance(pre_scene, dict):
            pre_scene = {}
        if not isinstance(post_scene, dict):
            post_scene = {}
        self._record_gpt_scene_memory_request(
            consumer="action_effect_verifier",
            scene_memory=post_scene,
            scene_memory_views={
                "pre": pre_scene,
                "post": post_scene,
            },
            request_metadata={
                "workflow": evidence.get("workflow", ""),
                "current_subtask": evidence.get("current_subtask", ""),
                "skill_id": evidence.get("skill_id", ""),
                "post_recovery_intent": evidence.get(
                    "post_recovery_intent", ""
                ),
                "expected_outcome": evidence.get("expected_outcome", ""),
            },
        )

    def _record_trace_and_rollout_event(
        self,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        try:
            serialized = compact_trace_event(
                event,
                payload,
                mode=self._observation_trace_payload_mode(),
            )
        except Exception as exc:
            # Trace serialization must never change physical rollout behavior.
            serialized = {
                "schema": "trace/compaction_error/v1",
                "schema_version": 1,
                "compaction_error": type(exc).__name__,
                "payload_omitted": True,
            }
        if not isinstance(serialized, dict):
            serialized = {"payload": serialized}
        self._trace(event, **serialized)
        self._dump_rollout_event(event, **serialized)

    def _write_raw_perception_sidecar_once(
        self,
        preprocess_payload: dict[str, Any],
    ) -> None:
        if not bool(
            getattr(self.config, "raw_trace_sidecar_enabled", False)
        ):
            return
        if self._observation_trace_payload_mode() != COMPACT_TRACE_PAYLOAD_MODE:
            return
        path = self._raw_perception_sidecar_file
        if path is None:
            return
        try:
            capture_id = int(preprocess_payload["observation_capture_id"])
        except (KeyError, TypeError, ValueError):
            return
        key = (
            int(self._current_episode_id),
            int(self._current_seed),
            capture_id,
        )
        if key in self._raw_perception_sidecar_capture_keys:
            return
        record = {
            "schema": "roboharn_evo/raw_perception_v1",
            "schema_version": 1,
            "event": "observation_preprocess_raw",
            "timestamp": time.time(),
            "episode_id": self._current_episode_id,
            "seed": self._current_seed,
            "env_step": preprocess_payload.get("env_step", -1),
            "observation_capture_id": capture_id,
            "observation_generation": preprocess_payload.get(
                "observation_generation"
            ),
            "payload": preprocess_payload,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            needs_separator = False
            if path.is_file() and path.stat().st_size > 0:
                with path.open("rb") as existing:
                    existing.seek(-1, os.SEEK_END)
                    needs_separator = existing.read(1) != b"\n"
            with path.open("a", encoding="utf-8") as stream:
                if needs_separator:
                    stream.write("\n")
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            return
        self._raw_perception_sidecar_capture_keys.add(key)

    def _dump_rollout_event(self, event: str, **payload: Any) -> None:
        if self._rollout_events_file is None:
            return
        env_step = getattr(self.latest_snapshot, "step_count", -1) if self.latest_snapshot is not None else -1
        record = {
            "event": event,
            "timestamp": time.time(),
            "episode_id": self._current_episode_id,
            "seed": self._current_seed,
            "env_step": env_step,
            **payload,
        }
        try:
            with self._rollout_events_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _dump_snapshot_images(self, snapshot: EnvSnapshot) -> None:
        if self._rollout_dir is None:
            return
        step = int(getattr(snapshot, "step_count", 0))
        try:
            iio.imwrite(self._rollout_dir / "head" / f"step_{step:06d}.png", snapshot.head_rgb)
            iio.imwrite(self._rollout_dir / "left" / f"step_{step:06d}.png", snapshot.left_rgb)
            iio.imwrite(self._rollout_dir / "right" / f"step_{step:06d}.png", snapshot.right_rgb)
            if snapshot.third_rgb is not None:
                iio.imwrite(
                    self._rollout_dir
                    / "third"
                    / f"step_{step:06d}.png",
                    snapshot.third_rgb,
                )
        except Exception:
            pass

    def finalize_rollout_videos(self, fps: int = 10) -> None:
        if self._rollout_dir is None:
            return
        video_dir = self._rollout_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        for name in ("head", "left", "right", "third"):
            encode_rollout_video(
                self._rollout_dir / name,
                video_dir / f"{name}.mp4",
                fps=fps,
            )

    def write_rollout_meta(self, meta: dict[str, Any]) -> None:
        if self._rollout_dir is None:
            return
        try:
            with (self._rollout_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _get_env_summary(self) -> str:
        return self.memory_store.state.working.recent_observation_summary

    def _get_robot_state(self) -> dict[str, Any]:
        if self.latest_snapshot is None:
            return {}
        return RMBenchEnvAdapter.robot_state(self.latest_snapshot)

    @staticmethod
    def _support_contact_limits() -> SupportContactLimits:
        return SupportContactLimits(
            no_progress_m=_GROUNDED_SETUP_NO_PROGRESS_EPSILON_M,
            max_contact_command_m=_MAX_GROUNDED_CONTACT_OVERTRAVEL_M,
            max_support_geometry_drift_m=(
                _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
            ),
            max_pose_drift_m=_GROUNDED_GEOMETRY_LEASE_MAX_POSE_DRIFT_M,
        )

    def _active_skill_id(self) -> str:
        active = self.memory_store.state.active_skill
        return "" if active is None else str(active.skill_id)

    def consume_recovery_observation(self, task_env: Any) -> dict[str, Any] | None:
        """Return a fresh recovery observation once when it still matches the environment step."""

        pending = self._pending_recovery_observation
        self._pending_recovery_observation = None
        if not isinstance(pending, dict):
            return None
        try:
            cached_step = int(pending.get("step_count", -1))
            current_step = int(getattr(task_env, "take_action_cnt", -2))
        except (TypeError, ValueError):
            return None
        raw = pending.get("raw")
        if cached_step != current_step or not isinstance(raw, dict):
            self._trace(
                "recovery_observation_handoff_discarded",
                cached_step=cached_step,
                current_step=current_step,
            )
            return None
        self._trace("recovery_observation_handoff_consumed", cached_step=cached_step)
        return dict(raw)

    def set_instruction(self, instruction: str) -> None:
        self.current_instruction = str(instruction)
        self._pure_tool_control_control_turns = 0
        self._pure_tool_control_no_progress_control_turns = 0
        self._pure_tool_control_control_backend_errors = 0
        self._pure_tool_control_recovery_backend_errors = 0
        self._last_recovery_backend_error = ""
        self._last_recovery_backend_error_stage = ""
        self._pending_action_effect_verification = None
        self._pure_tool_control_empty_plan_turns = 0
        self._pure_tool_control_terminal_failure = ""
        self._identity_binding_retry_attempts = 0
        self._identity_binding_retry_skill_id = ""
        self._authoritative_environment_success = False
        self._grounded_setup_failures.clear()
        self._blocked_grounded_setups.clear()
        self._operation_candidate_lifecycle.reset()
        self._operation_geometry_refresh_contract.reset()
        self._grounded_geometry_leases.clear()
        self._partial_grounded_approach_leases.clear()
        self._pending_recovery_observation = None
        self._recent_release_resolutions.clear()
        self._evidence_acquisition_policy.reset()
        self._last_evidence_acquisition_decision = {}
        self._pending_internal_recovery_completion = ""
        available_tools = [tool["function"]["name"] for tool in self.service_manager.activate_tools_list]
        self.memory_store.reset(
            task=self.current_instruction,
            control_model_name=self._control_model_name,
            available_policies=[self._executor_name or "default_manipulation_policy"],
            available_tools=available_tools,
            mode=self.config.mode,
            retry_budget=self.config.max_retries_per_skill,
            reset_budget=1,
            replan_budget=2,
        )
        self._trace("instruction_set", instruction=self.current_instruction)
        self._dump_rollout_event("instruction_set", instruction=self.current_instruction)

    def reset(self) -> None:
        self.control_runtime.reset()
        self.executor_runtime.reset()
        self.current_instruction = ""
        self.segment_start_snapshot = None
        self.previous_snapshot = None
        self.latest_snapshot = None
        self._scene_memory_tracker.reset()
        self._trace_file = None
        self._rollout_dir = None
        self._rollout_events_file = None
        self._raw_perception_sidecar_file = None
        self._raw_perception_sidecar_capture_keys.clear()
        self._gpt_scene_memory_sidecar_file = None
        self._gpt_scene_memory_request_index = 0
        self._current_episode_id = -1
        self._current_seed = -1
        self._last_preprocessed_snapshot_key = None
        self._observation_preprocess_generation = 0
        self._observation_capture_generation = 0
        self._debug_recovery_triggered = False
        self._debug_recovery_rounds = 0
        self._debug_recovery_scene_wait_turns = 0
        self._identity_binding_retry_attempts = 0
        self._identity_binding_retry_skill_id = ""
        self._debug_recovery_planner_bootstrapped = False
        self._pure_tool_control_control_turns = 0
        self._pure_tool_control_no_progress_control_turns = 0
        self._pure_tool_control_control_backend_errors = 0
        self._pure_tool_control_recovery_backend_errors = 0
        self._last_recovery_backend_error = ""
        self._last_recovery_backend_error_stage = ""
        self._pending_action_effect_verification = None
        self._pure_tool_control_empty_plan_turns = 0
        self._pure_tool_control_terminal_failure = ""
        self._authoritative_environment_success = False
        self._grounded_setup_failures.clear()
        self._blocked_grounded_setups.clear()
        self._operation_candidate_lifecycle.reset()
        self._operation_geometry_refresh_contract.reset()
        self._grounded_geometry_leases.clear()
        self._partial_grounded_approach_leases.clear()
        self._pending_recovery_observation = None
        self._recent_release_resolutions.clear()
        self._evidence_acquisition_policy.reset()
        self._last_evidence_acquisition_decision = {}
        self._pending_internal_recovery_completion = ""

    def update_snapshot(
        self,
        snapshot: EnvSnapshot,
        *,
        force_preprocess: bool = False,
        perception_binding_requirement: dict[str, Any] | None = None,
    ) -> None:
        if snapshot is not self.latest_snapshot:
            self._observation_capture_generation += 1
        self.previous_snapshot = self.latest_snapshot
        self.latest_snapshot = snapshot
        self._authoritative_environment_success = bool(
            self._authoritative_environment_success or snapshot.eval_success
        )
        if self._pure_tool_control_enabled() and self._authoritative_environment_success:
            summary = RMBenchEnvAdapter.summarize(snapshot)
            self.memory_store.record_observation_summary(summary)
            if self.segment_start_snapshot is None:
                self.segment_start_snapshot = snapshot
            self._dump_snapshot_images(snapshot)
            payload = {
                "authority": "environment_eval_success",
                "environment_success": True,
                "reason": "terminal environment snapshot requires no further perception API calls",
            }
            self._trace("terminal_observation_preprocess_skipped", **payload)
            self._dump_rollout_event("terminal_observation_preprocess_skipped", **payload)
            return
        snapshot_key = self._snapshot_preprocess_key(snapshot)
        if (
            not force_preprocess
            and snapshot_key is not None
            and snapshot_key == self._last_preprocessed_snapshot_key
        ):
            return
        if snapshot_key is not None:
            self._last_preprocessed_snapshot_key = snapshot_key
        summary = RMBenchEnvAdapter.summarize(snapshot)
        preprocess_payload = self.preprocess_observation(
            snapshot,
            force_preprocess=force_preprocess,
            perception_binding_requirement=perception_binding_requirement,
        )
        if preprocess_payload:
            self._observation_preprocess_generation += 1
            preprocess_payload["observation_generation"] = (
                self._observation_preprocess_generation
            )
            preprocess_payload["observation_capture_id"] = (
                self._observation_capture_generation
            )
            self._record_trace_and_rollout_event(
                "observation_preprocess",
                preprocess_payload,
            )
            self._write_raw_perception_sidecar_once(preprocess_payload)
            scene_memory = self.update_scene_memory(preprocess_payload, snapshot)
            if scene_memory:
                scene_memory = self._with_runtime_manipulation_state(scene_memory)
                preprocess_payload["scene_memory"] = scene_memory
                self.memory_store.record_scene_memory(scene_memory)
                self._resolve_pending_release_from_runtime(
                    source="scene_memory_update",
                )
                scene_memory = self.memory_store.state.working.scene_memory
                if isinstance(scene_memory, dict):
                    preprocess_payload["scene_memory"] = dict(scene_memory)
                trace_scene_memory = self._scene_memory_trace_payload(
                    scene_memory if isinstance(scene_memory, dict) else {}
                )
                self._record_trace_and_rollout_event(
                    "scene_memory_update",
                    {
                        "observation_generation": preprocess_payload.get(
                            "observation_generation"
                        ),
                        "observation_capture_id": preprocess_payload.get(
                            "observation_capture_id"
                        ),
                        "scene_memory": trace_scene_memory,
                    },
                )
            self.memory_store.record_observation_preprocess(preprocess_payload)
            finalized_trace_payload = self._safe_observation_preprocess_finalized_trace_payload(
                preprocess_payload
            )
            self._record_trace_and_rollout_event(
                "observation_preprocess_finalized",
                finalized_trace_payload,
            )
            summary = self._append_preprocess_summary(summary, preprocess_payload)
        self.memory_store.record_observation_summary(summary)
        if self.segment_start_snapshot is None:
            self.segment_start_snapshot = snapshot
        self._dump_snapshot_images(snapshot)

    def _with_runtime_manipulation_state(
        self,
        scene_memory: dict[str, Any],
    ) -> dict[str, Any]:
        payload = dict(scene_memory or {})
        manipulation_state = dict(
            self.memory_store.state.working.manipulation_state
        )
        if manipulation_state:
            payload["manipulation_state"] = manipulation_state
        else:
            payload.pop("manipulation_state", None)
        return with_dynamic_place_candidates(
            payload,
            manipulation_state=manipulation_state,
            robot_state=self._get_robot_state(),
            tcp_calibration_by_arm=(
                getattr(self.latest_snapshot, "tcp_calibration_by_arm", None)
                if self.latest_snapshot is not None
                else None
            ),
            active_grasp_transport_policy=(
                self._grasp_transport_policy()
            ),
        )

    def _scene_memory_trace_payload(
        self,
        scene_memory: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge current manipulation state without recomputing geometry."""

        payload = dict(scene_memory or {})
        try:
            manipulation_state = dict(
                self.memory_store.state.working.manipulation_state
            )
        except Exception:
            return payload
        if manipulation_state:
            payload["manipulation_state"] = manipulation_state
        else:
            payload.pop("manipulation_state", None)
        return payload

    def _sync_runtime_manipulation_state_to_scene_memory(self) -> None:
        scene_memory = self.memory_store.state.working.scene_memory
        if not isinstance(scene_memory, dict) or not scene_memory:
            return
        self.memory_store.record_scene_memory(
            self._with_runtime_manipulation_state(scene_memory)
        )

    def _runtime_scene_position_events(
        self,
        manipulation_state: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        state = (
            manipulation_state
            if isinstance(manipulation_state, dict)
            else self.memory_store.state.working.manipulation_state
        )
        if not isinstance(state, dict):
            return []
        robot_state = self._get_robot_state()
        calibrations = (
            getattr(
                self.latest_snapshot,
                "tcp_calibration_by_arm",
                {},
            )
            if self.latest_snapshot is not None
            else {}
        ) or {}
        step = (
            int(self.latest_snapshot.step_count)
            if self.latest_snapshot is not None
            else -1
        )
        events: list[dict[str, Any]] = []
        for arm in ("left", "right"):
            arm_state = state.get(arm)
            if not isinstance(arm_state, dict):
                continue
            instance_ref = str(
                arm_state.get("held_instance_id", "") or ""
            ).strip()
            if not instance_ref:
                continue
            phase = str(
                arm_state.get("phase", "") or ""
            ).strip().lower()
            try:
                manipulation_boundary_step = int(
                    arm_state.get("updated_step", step)
                )
            except (TypeError, ValueError):
                manipulation_boundary_step = step
            if (
                phase in {"holding_provisional", "place_aligned"}
                and arm_state.get("holding_confirmed") is not True
                and str(
                    arm_state.get("grasp_transport_policy", "") or ""
                ).strip().lower()
                == "evidence_only"
                and not self._grasp_transport_evidence_only()
            ):
                events.append(
                    {
                        "instance_ref": instance_ref,
                        "position_state": POSITION_MOTION_UNCERTAIN,
                        "source": "inactive_provisional_transport_policy",
                        "reason": (
                            "the active strict grasp policy does not accept "
                            "a persisted evidence-only attachment"
                        ),
                        "env_step": manipulation_boundary_step,
                    }
                )
                continue
            if phase in {
                "grasp_candidate",
                "ambiguous_grasp_return_pending",
                "release_pending_verification",
                "release_recovery_required",
            }:
                events.append(
                    {
                        "instance_ref": instance_ref,
                        "position_state": POSITION_MOTION_UNCERTAIN,
                        "source": (
                            "unverified_grasp_contact"
                            if phase
                            in {
                                "grasp_candidate",
                                "ambiguous_grasp_return_pending",
                            }
                            else "unverified_release"
                        ),
                        "reason": (
                            f"manipulation phase {phase} may have moved "
                            "the object"
                        ),
                        "env_step": manipulation_boundary_step,
                    }
                )
                continue
            if phase not in {
                "holding",
                "holding_provisional",
                "place_aligned",
            }:
                continue
            attachment = arm_state.get(
                "held_object_to_tcp_attachment"
            )
            current_world = propagate_held_object_world_m(
                robot_arm_state=(
                    robot_state.get(arm)
                    if isinstance(robot_state, dict)
                    else None
                ),
                calibration=(
                    calibrations.get(arm)
                    if isinstance(calibrations, dict)
                    else None
                ),
                attachment=attachment,
            )
            if current_world is None:
                events.append(
                    {
                        "instance_ref": instance_ref,
                        "position_state": POSITION_MOTION_UNCERTAIN,
                        "source": "missing_tcp_attachment_geometry",
                        "reason": (
                            "held-object position cannot be propagated "
                            "from the current calibrated TCP"
                        ),
                        "env_step": step,
                    }
                )
                continue
            confirmed = arm_state.get(
                "holding_confirmed"
            ) is True
            events.append(
                {
                    "instance_ref": instance_ref,
                    "position_state": (
                        POSITION_CURRENT_VERIFIED
                        if confirmed
                        else POSITION_MEMORY_VALID
                    ),
                    "world_m": current_world,
                    "confidence": 1.0 if confirmed else 0.5,
                    "source": (
                        "verified_tcp_attachment"
                        if confirmed
                        else "provisional_tcp_attachment"
                    ),
                    "reason": (
                        "confirmed rigid object-to-TCP attachment"
                        if confirmed
                        else (
                            "bounded provisional attachment propagated "
                            "from the current TCP"
                        )
                    ),
                    "env_step": step,
                    "arm": arm,
                }
            )
        return events

    def _apply_scene_runtime_events(
        self,
        events: list[dict[str, Any]],
    ) -> None:
        if not events:
            return
        scene_memory = (
            self.memory_store.state.working.scene_memory
        )
        if not isinstance(scene_memory, dict) or not scene_memory:
            return
        step = (
            int(self.latest_snapshot.step_count)
            if self.latest_snapshot is not None
            else int(scene_memory.get("env_step", -1) or -1)
        )
        updated = self._scene_memory_tracker.apply_runtime_events(
            scene_memory,
            events=events,
            env_step=step,
        )
        self.memory_store.record_scene_memory(
            self._with_runtime_manipulation_state(updated)
        )
        self._trace(
            "scene_memory_runtime_events",
            env_step=step,
            events=events,
        )
        self._dump_rollout_event(
            "scene_memory_runtime_events",
            env_step=step,
            events=events,
        )

    def _pending_release_verification_states(self) -> dict[str, dict[str, Any]]:
        return {
            arm: state
            for arm, state in self._active_manipulation_identity_states().items()
            if state.get("phase") == "release_pending_verification"
        }

    def _active_manipulation_identity_states(
        self,
    ) -> dict[str, dict[str, Any]]:
        """Return manipulation states whose object identity is runtime-owned.

        Once a grasp has produced a concrete held-instance reference, later
        perception-query turns may describe that object but must not replace
        its identity.  The lease remains active through release verification
        and bounded release recovery.
        """
        manipulation = self.memory_store.state.working.manipulation_state
        if not isinstance(manipulation, dict):
            return {}
        identity_phases = {
            "grasp_candidate",
            "ambiguous_grasp_return_pending",
            "holding",
            "holding_provisional",
            "place_aligned",
            "release_pending_verification",
            "release_recovery_required",
        }
        return {
            str(arm): dict(state)
            for arm, state in manipulation.items()
            if (
                str(arm) in {"left", "right"}
                and isinstance(state, dict)
                and state.get("phase") in identity_phases
                and str(state.get("held_instance_id", "") or "").strip()
            )
        }

    def _pending_release_identity_relocation_leases(
        self,
    ) -> list[dict[str, Any]]:
        leases: list[dict[str, Any]] = []
        for arm, state in sorted(
            self._pending_release_verification_states().items()
        ):
            instance_ref = str(
                state.get("held_instance_id", "") or ""
            ).strip()
            target_world = self._xyz_prefix(
                state.get("held_object_target_world_m")
            )
            if not instance_ref or target_world is None:
                continue
            extent = self._xyz_prefix(state.get("held_extent_m"))
            target_tolerance = place_target_tolerance_m(extent)
            recovery_anchors = release_identity_recovery_anchors(
                state
            )
            leases.append(
                {
                    "instance_ref": instance_ref,
                    "target_world_m": target_world,
                    # Association gets a little more sensor allowance than
                    # final placement validation, while remaining local to
                    # the runtime-selected operation target.
                    "tolerance_m": min(
                        0.06,
                        max(0.03, target_tolerance * 1.5),
                    ),
                    "validation_tolerance_m": target_tolerance,
                    "recovery_anchors": recovery_anchors,
                    "arm": arm,
                    "release_step": state.get("updated_step"),
                    "source": (
                        "release_pending_verification_operation_target"
                    ),
                    # Release verification consumes only the object's current
                    # world position.  Scene Memory may therefore retain a
                    # uniquely bound in-target centroid while quarantining
                    # occlusion-polluted approach/grasp/contact geometry.
                    POSITION_ONLY_ACTION_GEOMETRY_QUARANTINE_LEASE_KEY: True,
                }
            )
        # The tracker owns post-release action-geometry repair leases.  Do not
        # copy its prior-frame internal lease back as an external lease: a
        # same-update motion_uncertain event must be able to revoke it before
        # association.  Only active manipulation release state is authoritative
        # input from ImgAgent.
        return leases

    def _authorize_operation_geometry_refresh(
        self,
        *,
        calls: list[RecoveryToolCall],
        results: list[Any],
    ) -> None:
        """Bridge measured clearance to one bounded geometry refresh capture."""

        scene_memory = self.memory_store.state.working.scene_memory
        if not isinstance(scene_memory, dict):
            return
        instances = scene_memory.get("instances", [])
        if not isinstance(instances, list):
            return
        blocked_scopes = (
            self._operation_candidate_lifecycle.planner_payload(
                instances
            )
        )
        if not blocked_scopes:
            return
        latest = self._recovery_dispatcher.latest_snapshot
        env_step = (
            int(getattr(latest, "step_count", 0))
            if latest is not None
            else int(
                getattr(self.latest_snapshot, "step_count", 0)
                if self.latest_snapshot is not None
                else 0
            )
        )
        created = (
            self._operation_geometry_refresh_contract.authorize_from_results(
                calls=calls,
                results=results,
                blocked_scopes=blocked_scopes,
                scene_memory=scene_memory,
                env_step=env_step,
                capture_id=self._observation_capture_generation,
            )
        )
        for lease in created:
            self._trace(
                "operation_geometry_refresh_authorized",
                **lease,
            )
            self._dump_rollout_event(
                "operation_geometry_refresh_authorized",
                **lease,
            )

    def _with_runtime_manipulation_perception_queries(
        self,
        queries: list[dict[str, Any]],
        *,
        max_objects: int,
    ) -> list[dict[str, Any]]:
        runtime_queries: list[dict[str, Any]] = []
        runtime_refs: set[str] = set()
        runtime_descriptor_keys: list[tuple[str, str]] = []
        scene_memory = self.memory_store.state.working.scene_memory
        for arm, state in sorted(
            self._active_manipulation_identity_states().items()
        ):
            instance_ref = str(
                state.get("held_instance_id", "") or ""
            ).strip()
            if not instance_ref:
                continue
            descriptor = state.get(
                "held_object_perception_descriptor"
            )
            if not isinstance(descriptor, dict):
                descriptor = (
                    self._perception_descriptor_for_scene_instance(
                        self._scene_instance_by_ref(
                            scene_memory,
                            instance_ref,
                        )
                    )
                )
            raw_object_id = str(
                descriptor.get("object_id", "") or ""
            ).strip()
            object_id, inferred_hint = (
                normalize_perception_object_id(raw_object_id)
            )
            if not object_id:
                continue
            text_prompt = str(
                descriptor.get("text_prompt", "") or object_id
            ).strip()
            runtime_queries.append(
                {
                    "object_id": object_id,
                    "text_prompt": text_prompt or object_id,
                    "role": "context",
                    "instance_hint": merge_instance_hints(
                        (
                            "runtime-owned manipulated instance for "
                            f"the {arm} arm"
                        ),
                        descriptor.get("instance_hint", ""),
                        inferred_hint,
                    ),
                    "reason": (
                        "Runtime manipulation identity continuity must "
                        "preserve the same instance through grasp, "
                        "transport, placement, and release verification."
                    ),
                    "instance_ref": instance_ref,
                    "identity_binding_required": True,
                }
            )
            runtime_refs.add(instance_ref)
            runtime_descriptor_keys.append(
                (
                    object_id,
                    " ".join(text_prompt.lower().split()),
                )
            )

        if not runtime_queries:
            return [
                dict(item)
                for item in queries
                if isinstance(item, dict)
            ][:max_objects]

        normalized_queries = [
            dict(item)
            for item in queries
            if isinstance(item, dict)
        ]
        object_id_counts: dict[str, int] = {}
        for item in normalized_queries:
            item_object_id, _ = normalize_perception_object_id(
                item.get("object_id", "")
            )
            if item_object_id:
                object_id_counts[item_object_id] = (
                    object_id_counts.get(item_object_id, 0) + 1
                )

        def conflicts_with_runtime_identity(
            item: dict[str, Any],
        ) -> bool:
            instance_ref = str(
                item.get("instance_ref", "") or ""
            ).strip()
            if instance_ref in runtime_refs:
                return True
            item_object_id, _ = normalize_perception_object_id(
                item.get("object_id", "")
            )
            item_prompt = " ".join(
                str(item.get("text_prompt", "") or "")
                .strip()
                .lower()
                .split()
            )
            for descriptor_object_id, descriptor_prompt in (
                runtime_descriptor_keys
            ):
                if item_prompt and item_prompt == descriptor_prompt:
                    return True
                # An object id that occurs only once in this query batch is
                # an unambiguous semantic reference to the runtime-owned
                # object.  Replacing it is safe; repeated generic categories
                # such as several distinct "cube" queries remain separate.
                if (
                    item_object_id
                    and item_object_id == descriptor_object_id
                    and object_id_counts.get(item_object_id, 0) == 1
                ):
                    return True
            return False

        retained = [
            item
            for item in normalized_queries
            if not conflicts_with_runtime_identity(item)
        ]
        combined = [*runtime_queries, *retained][:max_objects]
        replaced_count = sum(
            1
            for item in normalized_queries
            if conflicts_with_runtime_identity(item)
        )
        payload = {
            "instance_refs": sorted(runtime_refs),
            "queries": [dict(item) for item in runtime_queries],
            "replaced_bound_query_count": replaced_count,
        }
        self._trace(
            "runtime_manipulation_perception_query_injected",
            **payload,
        )
        self._dump_rollout_event(
            "runtime_manipulation_perception_query_injected",
            **payload,
        )
        return combined

    def _snapshot_preprocess_key(self, snapshot: EnvSnapshot) -> tuple[str, ...]:
        active = self.memory_store.state.active_skill
        active_skill_id = "" if active is None else active.skill_id
        raw = snapshot.raw if isinstance(snapshot.raw, dict) else {}
        replay_meta = raw.get("_roboharn_evo_offline_replay", raw.get("_tcm_offline_replay"))
        if isinstance(replay_meta, dict):
            try:
                frame_index = int(replay_meta.get("frame_index", snapshot.step_count))
            except Exception:
                frame_index = int(snapshot.step_count)
            return (
                "offline_replay",
                str(replay_meta.get("mode", "offline_replay")),
                str(frame_index),
                active_skill_id,
            )

        digest = hashlib.blake2b(digest_size=16)
        for name in (
            "head_rgb",
            "left_rgb",
            "right_rgb",
            "third_rgb",
            "head_depth",
            "left_depth",
            "right_depth",
            "third_depth",
            "joint_vector",
            "left_endpose",
            "right_endpose",
        ):
            value = getattr(snapshot, name, None)
            if value is None:
                continue
            digest.update(name.encode("ascii"))
            digest.update(str(getattr(value, "shape", "")).encode("ascii"))
            digest.update(str(getattr(value, "dtype", "")).encode("ascii"))
            try:
                digest.update(value.tobytes())
            except (AttributeError, TypeError, ValueError):
                digest.update(repr(value).encode("utf-8", errors="replace"))
        robot_state = RMBenchEnvAdapter.robot_state(snapshot)
        digest.update(
            json.dumps(
                robot_state,
                ensure_ascii=True,
                sort_keys=True,
                default=lambda value: value.tolist() if hasattr(value, "tolist") else repr(value),
            ).encode("utf-8")
        )
        digest.update(str(bool(snapshot.eval_success)).encode("ascii"))
        digest.update(str(bool(snapshot.check_success)).encode("ascii"))
        return "live", digest.hexdigest(), active_skill_id

    def update_scene_memory(self, preprocess_payload: dict[str, Any], snapshot: EnvSnapshot) -> dict[str, Any]:
        if not bool(getattr(self.config, "scene_memory_enabled", True)):
            return {}
        segments = preprocess_payload.get("segmentation", [])
        agent_binding_required = bool(preprocess_payload.get("agent_identity_binding_required"))
        allow_context_binding = bool(
            self._active_manipulation_identity_states()
        )
        if not isinstance(segments, list):
            return {}
        current_subtask = "" if self.memory_store.state.active_skill is None else self.memory_store.state.active_skill.instruction
        previous_scene_memory = self.memory_store.state.working.scene_memory
        if not isinstance(previous_scene_memory, dict):
            previous_scene_memory = {}
        if segments:
            refresh_leases = (
                self._operation_geometry_refresh_contract.pending_relocation_leases(
                    scene_memory=previous_scene_memory,
                    env_step=int(
                        getattr(snapshot, "step_count", 0)
                    ),
                    capture_id=int(
                        preprocess_payload.get(
                            "observation_capture_id",
                            self._observation_capture_generation,
                        )
                        or 0
                    ),
                )
            )
            scene_memory = self._scene_memory_tracker.update(
                segmentation=segments,
                env_step=int(getattr(snapshot, "step_count", 0)),
                global_task=self.current_instruction or str(getattr(snapshot, "instruction", "")),
                current_subtask=current_subtask,
                identity_relocation_leases=(
                    [
                        *refresh_leases,
                        *self._pending_release_identity_relocation_leases(),
                    ]
                ),
                position_events=(
                    self._runtime_scene_position_events()
                ),
                observation_capture_id=(
                    preprocess_payload.get(
                        "observation_capture_id"
                    )
                ),
            )
        elif agent_binding_required:
            scene_memory = dict(previous_scene_memory)
            scene_memory.setdefault("instances", [])
            scene_memory.setdefault("uncertainty", [])
            scene_memory.setdefault("temporal_memory", {})
            scene_memory["env_step"] = int(getattr(snapshot, "step_count", 0))
        else:
            return {}
        perception_queries = [
            dict(item)
            for item in preprocess_payload.get("perception_queries", []) or []
            if isinstance(item, dict)
        ]
        perception_binding_requirement = preprocess_payload.get(
            "perception_binding_requirement"
        )
        if not isinstance(perception_binding_requirement, dict):
            perception_binding_requirement = None
        if agent_binding_required or any(bool(item.get("identity_binding_required")) for item in perception_queries):
            scene_memory, perception_queries = self._bind_scene_memory_focus_with_agent(
                scene_memory=scene_memory,
                perception_queries=perception_queries,
                observation_summary=RMBenchEnvAdapter.summarize(snapshot),
                current_subtask=current_subtask,
                identity_binding_required=agent_binding_required,
                allow_context_binding=allow_context_binding,
                previous_scene_memory=previous_scene_memory,
                perception_binding_requirement=perception_binding_requirement,
            )
            preprocess_payload["perception_queries"] = perception_queries
        return scene_memory

    def preprocess_observation(
        self,
        snapshot: EnvSnapshot,
        *,
        force_preprocess: bool = False,
        perception_binding_requirement: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self.config
        if not bool(getattr(config, "observation_preprocess_enabled", False)):
            return {}
        step = int(getattr(snapshot, "step_count", 0))
        every_n = max(1, int(getattr(config, "observation_preprocess_every_n_steps", 1)))
        if not force_preprocess and step % every_n != 0:
            return {}
        started = time.time()
        summary = RMBenchEnvAdapter.summarize(snapshot)
        objects = tuple(getattr(config, "observation_preprocess_objects", ()) or ())
        auto_objects = bool(
            getattr(config, "observation_preprocess_auto_objects", True)
        )
        query_url = str(
            getattr(config, "observation_preprocess_query_url", "") or ""
        ).strip()
        binding_requirement: dict[str, Any] = {}
        if not objects and auto_objects:
            allow_context_binding = bool(
                self._active_manipulation_identity_states()
            )
            binding_requirement = self._perception_binding_requirement(
                allow_context_binding=allow_context_binding,
                retry_context=perception_binding_requirement,
            )
            objects = tuple(
                self._build_perception_queries(
                    summary,
                    binding_requirement=binding_requirement,
                    allow_context_binding=allow_context_binding,
                )
            )
        elif objects:
            objects = tuple(self._normalize_perception_queries_with_api(list(objects), max_objects=len(objects), observation_summary=summary))
        objects = tuple(
            self._with_runtime_manipulation_perception_queries(
                list(objects),
                max_objects=max(
                    1,
                    int(
                        getattr(
                            config,
                            "observation_preprocess_max_objects",
                            3,
                        )
                    ),
                ),
            )
        )
        if not objects:
            agent_binding_required = bool(auto_objects and query_url)
            if agent_binding_required:
                payload = {
                    "stage": "observation_preprocess",
                    "env_step": step,
                    "latency_sec": round(time.time() - started, 4),
                    "agent_identity_binding_required": True,
                    "perception_queries": [],
                    "segmentation": [],
                    "perception_error": "agent returned no perception queries",
                }
                if binding_requirement:
                    payload["perception_binding_requirement"] = dict(
                        binding_requirement
                    )
                return payload
            return {}
        if bool(getattr(config, "oracle_objects_enabled", False)):
            payload = self._oracle_preprocess_observation(snapshot=snapshot, objects=objects, started=started)
            if payload and binding_requirement:
                payload["perception_binding_requirement"] = dict(
                    binding_requirement
                )
            return payload
        cameras = self._observation_preprocess_cameras()
        results: list[dict[str, Any]] = []
        for item in objects:
            object_id = str(item.get("object_id", "")).strip()
            if not object_id:
                continue
            text_prompt = str(item.get("text_prompt", object_id)).strip() or object_id
            for camera in cameras:
                result = self._preprocess_segment_object(
                    object_id=object_id,
                    text_prompt=text_prompt,
                    camera=camera,
                    backend=str(getattr(config, "observation_preprocess_backend", "sam3")),
                    service_url=str(getattr(config, "observation_preprocess_service_url", "")),
                )
                result.setdefault("camera", camera)
                self._attach_perception_query_metadata(result, item)
                self._attach_candidate_robot_state(result)
                if bool(getattr(config, "observation_grounding_enabled", False)):
                    grounding_camera = self._observation_grounding_camera_for_result(result)
                    result["grounding_3d"] = self._ground_segmentation_result(
                        snapshot=snapshot,
                        result=result,
                        camera=grounding_camera,
                    )
                    self._ground_detection_candidates(
                        snapshot=snapshot,
                        result=result,
                        camera=grounding_camera,
                    )
                    self._filter_grounded_preprocess_candidates(result)
                results.append(result)
        payload = {
            "stage": "observation_preprocess",
            "env_step": step,
            "latency_sec": round(time.time() - started, 4),
            "agent_identity_binding_required": bool(
                getattr(config, "observation_preprocess_auto_objects", True)
                and str(getattr(config, "observation_preprocess_query_url", "") or "").strip()
            ),
            "perception_queries": [dict(item) for item in objects],
            "segmentation": results,
        }
        if binding_requirement:
            payload["perception_binding_requirement"] = dict(
                binding_requirement
            )
        return payload

    def _oracle_preprocess_observation(
        self,
        *,
        snapshot: EnvSnapshot,
        objects: tuple[dict[str, str], ...],
        started: float,
    ) -> dict[str, Any]:
        catalog = self._oracle_catalog_from_snapshot(snapshot)
        if not isinstance(catalog, list) or not catalog:
            return {
                "stage": "observation_preprocess",
                "env_step": int(getattr(snapshot, "step_count", 0)),
                "latency_sec": round(time.time() - started, 4),
                "source": "oracle_simulator",
                "segmentation": [],
                "oracle_error": "snapshot has no oracle object catalog",
            }

        max_objects = max(1, int(getattr(self.config, "oracle_objects_max_objects", getattr(self.config, "observation_preprocess_max_objects", 12))))
        include_all = bool(getattr(self.config, "oracle_objects_include_all", True))
        segments: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for query in objects:
            matches = self._match_oracle_catalog(query=query, catalog=catalog)
            for match in matches:
                key = str(match.get("oracle_id") or match.get("source_path") or id(match))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                segments.append(self._oracle_catalog_entry_to_segment(match, query=query))
                if len(segments) >= max_objects:
                    break
            if len(segments) >= max_objects:
                break

        if include_all and len(segments) < max_objects:
            for entry in catalog:
                if not isinstance(entry, dict):
                    continue
                key = str(entry.get("oracle_id") or entry.get("source_path") or id(entry))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                segments.append(self._oracle_catalog_entry_to_segment(entry, query=None))
                if len(segments) >= max_objects:
                    break

        return {
            "stage": "observation_preprocess",
            "env_step": int(getattr(snapshot, "step_count", 0)),
            "latency_sec": round(time.time() - started, 4),
            "source": "oracle_simulator",
            "oracle_object_count": len([item for item in catalog if isinstance(item, dict)]),
            "agent_identity_binding_required": bool(
                getattr(self.config, "observation_preprocess_auto_objects", True)
                and str(getattr(self.config, "observation_preprocess_query_url", "") or "").strip()
            ),
            "perception_queries": [dict(item) for item in objects],
            "segmentation": segments,
        }

    def _match_oracle_catalog(self, *, query: dict[str, Any], catalog: list[Any]) -> list[dict[str, Any]]:
        explicit_oracle_id = str(query.get("oracle_id", query.get("query_oracle_id", "")) or "").strip()
        explicit_instance_ref = str(query.get("instance_ref", query.get("query_instance_ref", "")) or "").strip()
        resolved_instance_oracle_id = self._oracle_id_for_scene_instance_ref(explicit_instance_ref)
        if resolved_instance_oracle_id and explicit_oracle_id and resolved_instance_oracle_id != explicit_oracle_id:
            return []
        if resolved_instance_oracle_id and not explicit_oracle_id:
            explicit_oracle_id = resolved_instance_oracle_id
        if explicit_oracle_id or explicit_instance_ref:
            exact_matches: list[dict[str, Any]] = []
            for entry in catalog:
                if not isinstance(entry, dict):
                    continue
                entry_oracle_id = str(entry.get("oracle_id", "") or "").strip()
                entry_refs = {
                    entry_oracle_id,
                    str(entry.get("source_path", "") or "").strip(),
                }
                if explicit_oracle_id and entry_oracle_id != explicit_oracle_id:
                    continue
                if explicit_instance_ref and explicit_instance_ref not in entry_refs and not resolved_instance_oracle_id:
                    continue
                exact_matches.append(entry)
            if bool(query.get("identity_binding_required", query.get("_identity_binding_required", False))):
                return exact_matches if len(exact_matches) == 1 else []
            return exact_matches
        if bool(query.get("identity_binding_required", query.get("_identity_binding_required", False))):
            return []
        token_groups = (
            self._oracle_text_tokens(query.get("object_id")),
            self._oracle_text_tokens(query.get("text_prompt")),
            self._oracle_text_tokens(query.get("instance_hint", query.get("query_instance_hint"))),
        )
        for query_tokens in token_groups:
            if not query_tokens:
                continue
            scored: list[tuple[int, str, dict[str, Any]]] = []
            for entry in catalog:
                if not isinstance(entry, dict):
                    continue
                overlap = query_tokens & self._oracle_entry_tokens(entry)
                if not overlap:
                    continue
                scored.append((len(overlap), str(entry.get("source_path", "")), entry))
            if scored:
                scored.sort(key=lambda item: (-item[0], item[1]))
                return [item[2] for item in scored]
        return []

    def _oracle_id_for_scene_instance_ref(self, instance_ref: str) -> str:
        reference = str(instance_ref or "").strip()
        if not reference:
            return ""
        for item in self._scene_instance_catalog_for_query_payload():
            if reference not in {
                str(item.get("instance_id", "") or "").strip(),
                str(item.get("track_id", "") or "").strip(),
            }:
                continue
            return str(item.get("oracle_id", "") or "").strip()
        return ""

    def _oracle_query_tokens(self, query: dict[str, Any]) -> set[str]:
        tokens: set[str] = set()
        for key in ("object_id", "text_prompt", "instance_hint", "query_instance_hint"):
            tokens.update(self._oracle_text_tokens(query.get(key)))
        return tokens

    def _oracle_entry_tokens(self, entry: dict[str, Any]) -> set[str]:
        tokens: set[str] = set()
        for key in ("oracle_id", "class_name", "source_path", "actor_name"):
            tokens.update(self._oracle_text_tokens(entry.get(key)))
        aliases = entry.get("aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                tokens.update(self._oracle_text_tokens(alias))
        return tokens

    def _oracle_text_tokens(self, value: Any) -> set[str]:
        text = str(value or "").strip().lower()
        if not text:
            return set()
        normalized = "".join(char if char.isalnum() else "_" for char in text).strip("_")
        parts = [part for part in normalized.split("_") if part]
        tokens = set(parts)
        if normalized:
            tokens.add(normalized)
        stems: set[str] = set()
        for token in tokens:
            if len(token) > 3 and token.endswith("ies"):
                stems.add(token[:-3] + "y")
            elif len(token) > 2 and token.endswith("ses"):
                stems.add(token[:-2])
            elif len(token) > 1 and token.endswith("s"):
                stems.add(token[:-1])
        tokens.update(stems)
        return {token for token in tokens if token}

    def _oracle_catalog_entry_to_segment(self, entry: dict[str, Any], *, query: dict[str, Any] | None) -> dict[str, Any]:
        class_name = str(entry.get("class_name") or entry.get("actor_name") or entry.get("oracle_id") or "object")
        object_id = class_name
        text_prompt = class_name.replace("_", " ")
        role = "context"
        instance_hint = str(entry.get("source_path", "") or entry.get("oracle_id", ""))
        reason = "oracle simulator object pose"
        if isinstance(query, dict):
            object_id = str(query.get("object_id", object_id) or object_id)
            text_prompt = str(query.get("text_prompt", text_prompt) or text_prompt)
            role = str(query.get("role", query.get("query_role", role)) or role).strip().lower().replace("-", "_").replace(" ", "_")
            if role not in {"target", "tool", "context"}:
                role = "context"
            query_hint = str(query.get("instance_hint", query.get("query_instance_hint", "")) or "").strip()
            instance_hint = query_hint or instance_hint
            reason = str(query.get("reason", query.get("query_reason", "")) or "").strip() or reason
        grounding = entry.get("grounding_3d") if isinstance(entry.get("grounding_3d"), dict) else {}
        result = {
            "success": True,
            "object_id": object_id,
            "text_prompt": text_prompt,
            "num_detections": 1,
            "score": 1.0,
            "backend": "oracle_simulator",
            "camera": "oracle",
            "env_step": getattr(self.latest_snapshot, "step_count", 0) if self.latest_snapshot is not None else 0,
            "query_role": role,
            "query_instance_hint": instance_hint,
            "query_reason": reason,
            "instance_ref": "" if query is None else str(query.get("instance_ref", query.get("query_instance_ref", "")) or "").strip(),
            "query_oracle_id": "" if query is None else str(query.get("oracle_id", query.get("query_oracle_id", "")) or "").strip(),
            "identity_binding_required": False if query is None else bool(query.get("identity_binding_required", query.get("_identity_binding_required", False))),
            "identity_binding_error": "" if query is None else str(query.get("identity_binding_error", "") or "").strip(),
            **normalized_relation_metadata(query),
            "oracle_id": entry.get("oracle_id"),
            "oracle_class_name": class_name,
            "oracle_source_path": entry.get("source_path"),
            "oracle_actor_name": entry.get("actor_name"),
            "oracle_aliases": entry.get("aliases", []),
            "grounding_3d": grounding,
        }
        result["detections"] = [
            {
                "rank": int(entry.get("rank", 0) or 0),
                "score": 1.0,
                "camera": "oracle",
                "object_id": object_id,
                "query_role": role,
                "query_instance_hint": instance_hint,
                "query_reason": reason,
                "instance_ref": result["instance_ref"],
                "query_oracle_id": result["query_oracle_id"],
                "identity_binding_required": result["identity_binding_required"],
                "identity_binding_error": result["identity_binding_error"],
                **normalized_relation_metadata(query),
                "oracle_id": entry.get("oracle_id"),
                "oracle_class_name": class_name,
                "oracle_source_path": entry.get("source_path"),
                "oracle_aliases": entry.get("aliases", []),
                "grounding_3d": grounding,
            }
        ]
        return result

    def _observation_preprocess_cameras(self) -> tuple[str, ...]:
        configured = getattr(self.config, "observation_preprocess_cameras", ()) or ()
        cameras = self._normalize_observation_camera_list(configured)
        if not cameras:
            cameras = (
                self._normalize_observation_camera_list(
                    (
                        getattr(
                            self.config,
                            "observation_preprocess_camera",
                            "head",
                        ),
                    )
                )
                or ("head",)
            )
        verification_cameras = self._normalize_observation_camera_list(
            getattr(
                self.config,
                "observation_verification_cameras",
                (),
            )
            or ()
        )
        augmented = list(cameras)
        for camera in verification_cameras:
            if (
                camera not in augmented
                and self._snapshot_has_calibrated_camera(camera)
            ):
                augmented.append(camera)
        return tuple(augmented)

    def _manipulation_verification_view_required(self) -> bool:
        manipulation = (
            self.memory_store.state.working.manipulation_state
        )
        if isinstance(manipulation, dict):
            for state in manipulation.values():
                if (
                    isinstance(state, dict)
                    and str(
                        state.get("phase", "") or ""
                    ).strip().lower()
                    in _MANIPULATION_VERIFICATION_CAMERA_PHASES
                ):
                    return True
        scene_memory = self.memory_store.state.working.scene_memory
        if not isinstance(scene_memory, dict):
            return False
        return any(
            isinstance(instance, dict)
            and str(
                instance.get("action_geometry_state", "") or ""
            ).strip().lower()
            in {
                "relocation_pending",
                "identity_repair_expired",
                "unavailable",
            }
            for instance in scene_memory.get("instances", []) or []
        )

    def _snapshot_has_calibrated_camera(self, camera: str) -> bool:
        if self.latest_snapshot is None:
            return False
        camera_data = RMBenchEnvAdapter.camera_data(
            self.latest_snapshot,
            camera,
        )
        return all(
            camera_data.get(key) is not None
            for key in (
                "rgb",
                "depth",
                "intrinsic_cv",
                "cam2world_gl",
            )
        )

    def _observation_grounding_camera_for_result(self, result: dict[str, Any]) -> str:
        configured = self._normalize_observation_camera_list(getattr(self.config, "observation_grounding_cameras", ()) or ())
        result_camera = self._normalize_observation_camera(result.get("camera"), default="")
        # A segmentation mask is expressed in the pixel frame of the camera
        # that produced it.  Grounding it with another camera's depth and
        # calibration silently creates a geometrically invalid world pose, even
        # when both images happen to have the same resolution.  Explicit
        # grounding cameras remain useful as a fallback for legacy results that
        # do not carry camera provenance.
        if result_camera:
            return result_camera
        if configured:
            return configured[0]
        preprocess_cameras = self._observation_preprocess_cameras()
        if len(preprocess_cameras) <= 1:
            legacy_camera = self._normalize_observation_camera(getattr(self.config, "observation_grounding_camera", ""), default="")
            if legacy_camera:
                return legacy_camera
        return self._normalize_observation_camera(getattr(self.config, "observation_grounding_camera", "head"))

    def _normalize_observation_camera_list(self, value: Any) -> tuple[str, ...]:
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
            camera = self._normalize_observation_camera(item, default="")
            if camera and camera not in cameras:
                cameras.append(camera)
        return tuple(cameras)

    def _normalize_observation_camera(self, value: Any, *, default: str = "head") -> str:
        camera = str(value or "").strip().lower().replace("_camera", "")
        if camera in {"head", "left", "right", "third"}:
            return camera
        return default

    def _preprocess_segment_object(
        self,
        *,
        object_id: str,
        text_prompt: str,
        camera: str,
        backend: str,
        service_url: str,
    ) -> dict[str, Any]:
        previous_threshold = os.environ.get("ROBOHARN_EVO_SAM3_CONFIDENCE_THRESHOLD")
        threshold = str(os.getenv("ROBOHARN_EVO_OBS_PREPROCESS_SAM3_CONFIDENCE_THRESHOLD", "")).strip()
        if threshold:
            os.environ["ROBOHARN_EVO_SAM3_CONFIDENCE_THRESHOLD"] = threshold

        def segment(prompt: str) -> dict[str, Any]:
            result_text = _run_agent_tool_sync(
                self.agent_tools.segment_object(
                    object_id=object_id,
                    text_prompt=prompt,
                    camera=camera,
                    backend=backend,
                    service_url=service_url,
                )
            )
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                return {"success": False, "object_id": object_id, "text_prompt": prompt, "error": result_text[:300]}
            if not isinstance(result, dict):
                return {"success": False, "object_id": object_id, "text_prompt": prompt, "error": "non-object tool result"}
            return self._compact_segmentation_result(result)

        try:
            primary = segment(text_prompt)
            fallback_prompt = " ".join(str(object_id).replace("_", " ").split())
            should_retry = (
                bool(fallback_prompt)
                and fallback_prompt.casefold() != " ".join(str(text_prompt).split()).casefold()
                and not self._segmentation_result_has_detection(primary)
            )
            if should_retry:
                fallback = segment(fallback_prompt)
                primary["text_prompt_fallback_attempted"] = True
                primary["fallback_text_prompt"] = fallback_prompt
                if self._segmentation_result_has_detection(fallback):
                    fallback["requested_text_prompt"] = text_prompt
                    fallback["fallback_text_prompt"] = fallback_prompt
                    fallback["text_prompt_fallback_attempted"] = True
                    fallback["text_prompt_fallback_used"] = True
                    return fallback
                primary["text_prompt_fallback_used"] = False
                if fallback.get("error"):
                    primary["fallback_error"] = self._compact_summary_text(fallback.get("error"), 180)
            return primary
        finally:
            if threshold:
                if previous_threshold is None:
                    os.environ.pop("ROBOHARN_EVO_SAM3_CONFIDENCE_THRESHOLD", None)
                else:
                    os.environ["ROBOHARN_EVO_SAM3_CONFIDENCE_THRESHOLD"] = previous_threshold

    @staticmethod
    def _segmentation_result_has_detection(result: dict[str, Any]) -> bool:
        detections = result.get("detections")
        if isinstance(detections, list) and detections:
            return True
        try:
            if int(result.get("num_detections", 0) or 0) > 0:
                return True
        except Exception:
            pass
        return bool(result.get("mask_path") or result.get("bbox_xyxy"))

    @staticmethod
    def _required_perception_query_roles(
        binding_requirement: dict[str, Any] | None,
    ) -> tuple[str, ...]:
        if not isinstance(binding_requirement, dict):
            return ()
        roles: list[str] = []
        for value in binding_requirement.get("required_any_roles", []) or []:
            role = (
                str(value or "")
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            if role in {"target", "tool", "context"} and role not in roles:
                roles.append(role)
        return tuple(roles)

    def _perception_queries_satisfy_binding_requirement(
        self,
        queries: list[dict[str, Any]],
        binding_requirement: dict[str, Any] | None,
    ) -> bool:
        required_roles = set(
            self._required_perception_query_roles(binding_requirement)
        )
        if not required_roles:
            return True
        return any(
            normalize_query_role(
                item.get("role", item.get("query_role", "context"))
            )
            in required_roles
            for item in queries
            if isinstance(item, dict)
        )

    def _perception_binding_requirement(
        self,
        *,
        allow_context_binding: bool,
        retry_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        required_roles = ["target", "tool"]
        if allow_context_binding:
            required_roles.append("context")
        requirement: dict[str, Any] = {
            "required_any_roles": required_roles,
            "contract": "at_least_one_bound_task_focus_query",
        }
        if not isinstance(retry_context, dict):
            return requirement
        failure_reason = self._compact_summary_text(
            retry_context.get("failure_reason", ""),
            320,
        )
        if failure_reason:
            requirement["failure_reason"] = failure_reason
        try:
            retry_attempt = max(
                0,
                int(retry_context.get("retry_attempt", 0) or 0),
            )
        except (TypeError, ValueError):
            retry_attempt = 0
        if retry_attempt:
            requirement["retry_attempt"] = retry_attempt
        if bool(retry_context.get("force_refresh")):
            requirement["force_refresh"] = True
        previous_queries = retry_context.get("previous_queries", [])
        if isinstance(previous_queries, list):
            max_objects = max(
                1,
                int(
                    getattr(
                        self.config,
                        "observation_preprocess_max_objects",
                        3,
                    )
                ),
            )
            compact_previous = self._normalize_perception_queries(
                previous_queries,
                max_objects=max_objects,
            )
            if compact_previous:
                requirement["previous_queries"] = compact_previous
        return requirement

    def _trace_perception_binding_contract_violation(
        self,
        *,
        stage: str,
        queries: list[dict[str, Any]],
        binding_requirement: dict[str, Any] | None,
    ) -> None:
        required_roles = list(
            self._required_perception_query_roles(binding_requirement)
        )
        if not required_roles:
            return
        returned_roles = [
            normalize_query_role(
                item.get("role", item.get("query_role", "context"))
            )
            for item in queries
            if isinstance(item, dict)
        ]
        payload = {
            "stage": stage,
            "required_any_roles": required_roles,
            "returned_roles": returned_roles,
            "failure_reason": str(
                (binding_requirement or {}).get("failure_reason", "") or ""
            ),
            "retry_attempt": int(
                (binding_requirement or {}).get("retry_attempt", 0) or 0
            ),
        }
        self._trace(
            "perception_binding_contract_violation",
            **payload,
        )
        self._dump_rollout_event(
            "perception_binding_contract_violation",
            **payload,
        )

    def _build_perception_queries(
        self,
        observation_summary: str,
        *,
        binding_requirement: dict[str, Any] | None = None,
        allow_context_binding: bool | None = None,
    ) -> list[dict[str, str]]:
        query_url = str(getattr(self.config, "observation_preprocess_query_url", "") or "").strip()
        max_objects = max(1, int(getattr(self.config, "observation_preprocess_max_objects", 3)))
        if allow_context_binding is None:
            allow_context_binding = bool(
                self._active_manipulation_identity_states()
            )
        if binding_requirement is None:
            binding_requirement = self._perception_binding_requirement(
                allow_context_binding=allow_context_binding,
            )
        else:
            binding_requirement = dict(binding_requirement)
        queries: list[dict[str, str]] = []
        if query_url:
            try:
                oracle_objects = self._oracle_catalog_for_query_payload()
                scene_instances = self._scene_instance_catalog_for_query_payload()
                # No-oracle perception is deliberately two phase.  Existing
                # tracks are only a partial catalog, so requiring every new
                # task query to bind one of them makes it impossible to
                # discover an object category introduced by a later subtask.
                # Oracle catalogs are complete and may be selected directly;
                # no-oracle queries are bound strictly after SAM has generated
                # current candidates in _bind_scene_memory_focus_with_agent.
                require_catalog_binding = bool(oracle_objects)
                payload = {
                    "global_task": self.current_instruction,
                    "current_subtask": "" if self.memory_store.state.active_skill is None else self.memory_store.state.active_skill.instruction,
                    "committed_memory": self.memory_store.export_reasoner_memory(),
                    "observation_summary": observation_summary,
                    "robot_state": self._get_robot_state(),
                    "active_skill": "" if self.memory_store.state.active_skill is None else self.memory_store.state.active_skill.skill_name,
                    "max_queries": max_objects,
                    "camera": self._observation_preprocess_cameras()[0],
                    "cameras": list(self._observation_preprocess_cameras()),
                    "oracle_objects": oracle_objects,
                    "scene_instances": scene_instances,
                    "require_instance_binding": require_catalog_binding,
                    "instance_binding_phase": (
                        "catalog_selection"
                        if require_catalog_binding
                        else "candidate_discovery"
                    ),
                    "binding_requirement": binding_requirement,
                    "image_b64": self._encode_snapshot_image_b64(self.latest_snapshot, camera=self._observation_preprocess_cameras()[0]),
                    "image_b64_by_camera": {
                        camera: self._encode_snapshot_image_b64(self.latest_snapshot, camera=camera)
                        for camera in self._observation_preprocess_cameras()
                    },
                }
                raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                http_request = request.Request(
                    query_url,
                    data=raw_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                timeout_sec = max(1, int(getattr(self.config, "observation_preprocess_query_timeout_sec", 120)))
                with request.urlopen(http_request, timeout=timeout_sec) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                raw_queries = response_payload.get("queries", []) if isinstance(response_payload, dict) else []
                queries = self._normalize_perception_queries_with_api(
                    raw_queries,
                    max_objects=max_objects,
                    observation_summary=observation_summary,
                    scene_instances=scene_instances,
                    require_instance_binding=require_catalog_binding,
                    binding_requirement=binding_requirement,
                )
            except Exception as exc:
                self._trace("observation_preprocess_query_error", query_url=query_url, error=repr(exc))
        if not queries:
            queries = self._fallback_perception_queries(max_objects=max_objects)
        if query_url:
            queries = self._mark_agent_identity_binding_required(
                queries,
                allow_context_binding=allow_context_binding,
            )
        return queries

    def _normalize_perception_queries_with_api(
        self,
        raw_queries: Any,
        *,
        max_objects: int,
        observation_summary: str,
        scene_instances: list[dict[str, Any]] | None = None,
        require_instance_binding: bool = False,
        binding_requirement: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        schema_queries = self._normalize_perception_queries(raw_queries, max_objects=max_objects)
        required_roles = self._required_perception_query_roles(
            binding_requirement
        )
        if not schema_queries and not required_roles:
            return []
        normalization_url = str(getattr(self.config, "observation_preprocess_normalization_url", "") or "").strip()
        if not normalization_url:
            if self._perception_queries_satisfy_binding_requirement(
                schema_queries,
                binding_requirement,
            ):
                return schema_queries
            self._trace_perception_binding_contract_violation(
                stage="schema_without_normalization_api",
                queries=schema_queries,
                binding_requirement=binding_requirement,
            )
            return []
        try:
            candidate_instances = (
                list(scene_instances)
                if isinstance(scene_instances, list)
                else self._scene_instance_catalog_for_query_payload()
            )
            cameras = self._observation_preprocess_cameras()
            image_b64 = ""
            image_b64_by_camera: dict[str, str] = {}
            if require_instance_binding:
                image_b64 = self._encode_snapshot_image_b64(self.latest_snapshot, camera=cameras[0])
                image_b64_by_camera = {
                    camera: self._encode_snapshot_image_b64(self.latest_snapshot, camera=camera)
                    for camera in cameras
                }
            payload = {
                "global_task": self.current_instruction,
                "current_subtask": "" if self.memory_store.state.active_skill is None else self.memory_store.state.active_skill.instruction,
                "committed_memory": self.memory_store.export_reasoner_memory(),
                "observation_summary": observation_summary,
                "raw_queries": schema_queries,
                "max_queries": max_objects,
                "oracle_objects": self._oracle_catalog_for_query_payload(),
                "scene_instances": candidate_instances,
                "require_instance_binding": bool(require_instance_binding),
                "instance_binding_phase": (
                    "post_detection_selection"
                    if require_instance_binding
                    else "candidate_discovery"
                ),
                "binding_requirement": dict(binding_requirement or {}),
                "image_b64": image_b64,
                "image_b64_by_camera": image_b64_by_camera,
            }
            raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            http_request = request.Request(
                normalization_url,
                data=raw_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            timeout_sec = max(1, int(getattr(self.config, "observation_preprocess_query_timeout_sec", 120)))
            with request.urlopen(http_request, timeout=timeout_sec) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            normalized = response_payload.get("queries", []) if isinstance(response_payload, dict) else []
            validated = self._normalize_perception_queries(normalized, max_objects=max_objects)
            if validated and self._perception_queries_satisfy_binding_requirement(
                validated,
                binding_requirement,
            ):
                return validated
            if validated:
                self._trace_perception_binding_contract_violation(
                    stage="normalization_response",
                    queries=validated,
                    binding_requirement=binding_requirement,
                )
            if self._perception_queries_satisfy_binding_requirement(
                schema_queries,
                binding_requirement,
            ):
                return schema_queries
            if not validated:
                self._trace_perception_binding_contract_violation(
                    stage="empty_normalization_response",
                    queries=[],
                    binding_requirement=binding_requirement,
                )
            return []
        except Exception as exc:
            self._trace("observation_preprocess_normalization_error", normalization_url=normalization_url, error=repr(exc))
            if self._perception_queries_satisfy_binding_requirement(
                schema_queries,
                binding_requirement,
            ):
                return schema_queries
            self._trace_perception_binding_contract_violation(
                stage="normalization_error_fallback",
                queries=schema_queries,
                binding_requirement=binding_requirement,
            )
            return []

    def _normalize_perception_queries(self, raw_queries: Any, *, max_objects: int) -> list[dict[str, Any]]:
        if not isinstance(raw_queries, list):
            return []
        queries: list[dict[str, Any]] = []
        for item in raw_queries:
            if isinstance(item, str):
                raw_object_id = item.strip()
                object_id, inferred_hint = normalize_perception_object_id(raw_object_id)
                text_prompt = raw_object_id or object_id
                role = "context"
                instance_hint = inferred_hint
                reason = ""
                instance_ref = ""
                oracle_id = ""
                relation_metadata: dict[str, Any] = {}
            elif isinstance(item, dict):
                raw_object_id = str(item.get("object_id", item.get("name", ""))).strip()
                object_id, inferred_hint = normalize_perception_object_id(raw_object_id)
                text_prompt = str(item.get("text_prompt", raw_object_id or object_id)).strip() or object_id
                role = normalize_query_role(item.get("role", item.get("query_role", "context")))
                instance_hint = merge_instance_hints(item.get("instance_hint", item.get("query_instance_hint", "")), inferred_hint)
                reason = str(item.get("reason", item.get("query_reason", ""))).strip()
                instance_ref = str(item.get("instance_ref", item.get("track_id", item.get("instance_id", ""))) or "").strip()
                oracle_id = str(item.get("oracle_id", item.get("query_oracle_id", "")) or "").strip()
                relation_metadata = normalized_relation_metadata(item)
            else:
                continue
            if not object_id:
                continue
            query = {
                "object_id": object_id,
                "text_prompt": text_prompt,
                "role": role,
                "instance_hint": instance_hint,
                "reason": reason,
            }
            query.update(relation_metadata)
            if query.get("entity_scope") == "reference_set":
                instance_ref = ""
                oracle_id = ""
            if instance_ref:
                query["instance_ref"] = instance_ref[:160]
            if oracle_id:
                query["oracle_id"] = oracle_id[:160]
            merge_index = matching_perception_query_index(queries, query)
            if merge_index is not None:
                queries[merge_index] = merge_perception_query(
                    queries[merge_index],
                    query,
                )
                continue
            queries.append(query)
            if len(queries) >= max_objects:
                break
        return queries

    def _fallback_perception_queries(self, *, max_objects: int) -> list[dict[str, str]]:
        return []

    def _mark_agent_identity_binding_required(
        self,
        queries: list[dict[str, Any]],
        *,
        allow_context_binding: bool = False,
    ) -> list[dict[str, Any]]:
        marked: list[dict[str, Any]] = []
        binding_roles = {"target", "tool"}
        if allow_context_binding:
            binding_roles.add("context")
        for item in queries:
            if not isinstance(item, dict):
                continue
            query = dict(item)
            role = normalize_query_role(query.get("role", query.get("query_role", "context")))
            if normalize_entity_scope(
                query.get("entity_scope")
            ) == "reference_set":
                query["identity_binding_required"] = False
                query.pop("identity_binding_error", None)
                query.pop("instance_ref", None)
                query.pop("oracle_id", None)
            elif self._is_robot_embodiment_query(query):
                # Robot embodiment state is already keyed by arm in the
                # runtime snapshot.  It is not a scene-memory instance and
                # must never be forced to invent a track_id merely because a
                # verifier asks to look at the gripper or end effector.
                query["entity_scope"] = "robot_state"
                query["identity_binding_required"] = False
                query.pop("identity_binding_error", None)
                query.pop("instance_ref", None)
                query.pop("oracle_id", None)
            elif role in binding_roles:
                query["identity_binding_required"] = True
            marked.append(query)
        return marked

    def _is_robot_embodiment_query(
        self,
        query: dict[str, Any],
    ) -> bool:
        explicit_scope = str(
            query.get(
                "entity_scope",
                query.get("identity_scope", ""),
            )
            or ""
        ).strip().lower().replace("-", "_")
        if explicit_scope in {
            "robot",
            "robot_state",
            "embodiment",
        }:
            return True
        object_id, _ = normalize_perception_object_id(
            query.get("object_id", query.get("name", ""))
        )
        if object_id in _ROBOT_EMBODIMENT_QUERY_IDS:
            return True
        prompt_id, _ = normalize_perception_object_id(
            query.get("text_prompt", "")
        )
        prompt_tokens = set(prompt_id.split("_"))
        return (
            "robot" in prompt_tokens
            and bool(
                prompt_tokens
                & {
                    "gripper",
                    "wrist",
                    "tcp",
                    "effector",
                    "arm",
                }
            )
        )

    def _scene_instance_catalog_for_query_payload(
        self,
        scene_memory: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        memory = scene_memory
        if not isinstance(memory, dict):
            memory = self.memory_store.state.working.scene_memory
        if not isinstance(memory, dict):
            return []
        instances = memory.get("instances")
        if not isinstance(instances, list):
            return []
        max_objects = max(1, int(getattr(self.config, "oracle_objects_max_objects", 12)))
        compact: list[dict[str, Any]] = []
        for item in instances:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "") or "").strip().lower()
            stability = str(item.get("stability", "") or "").strip().lower()
            action_geometry_state = str(
                item.get("action_geometry_state", "verified")
                or "verified"
            ).strip().lower()
            position_state = str(
                item.get("position_state", "") or ""
            ).strip().lower()
            if status not in {"visible", "tracked"} or item.get("actionable") is False:
                continue
            if not instance_has_finite_grounding(item):
                continue
            recovery_binding_only = status == "tracked" or stability in {
                "missing_current_frame",
                "geometry_inconsistent_current_frame",
                "relocation_geometry_pending",
                "identity_repair_expired",
            } or action_geometry_state in {
                "relocation_pending",
                "identity_repair_expired",
                "unavailable",
            }
            compact.append(
                {
                    "instance_id": item.get("instance_id"),
                    "track_id": item.get("track_id"),
                    "oracle_id": item.get("oracle_id"),
                    "oracle_source_path": item.get("oracle_source_path"),
                    "class": item.get("class"),
                    "class_aliases": item.get("class_aliases", []),
                    "source_object_id": item.get("source_object_id"),
                    "source_text_prompt": item.get(
                        "source_text_prompt"
                    ),
                    "status": status,
                    "stability": stability,
                    "position_state": position_state,
                    "position_source": item.get(
                        "position_source"
                    ),
                    "last_verified_world_m": item.get(
                        "last_verified_world_m"
                    ),
                    "last_verified_score": item.get(
                        "last_verified_score"
                    ),
                    "last_verified_step": item.get(
                        "last_verified_step"
                    ),
                    "action_geometry_state": action_geometry_state,
                    "action_geometry_confirmation_count": item.get(
                        "action_geometry_confirmation_count",
                        0,
                    ),
                    "missing_steps": item.get("missing_steps", 0),
                    "actionable": bool(item.get("actionable", True)),
                    "recovery_binding_only": recovery_binding_only,
                    "grounded_execution_allowed": not recovery_binding_only,
                    "grounded_execution_scope": (
                        "approach_only"
                        if (
                            position_state
                            == POSITION_MEMORY_VALID
                            and action_geometry_state
                            == "verified"
                        )
                        else "none"
                        if (
                            position_state
                            == POSITION_MOTION_UNCERTAIN
                            or recovery_binding_only
                        )
                        else "full"
                    ),
                    "verified_roles": [
                        dict(role)
                        for role in item.get(
                            "verified_roles",
                            [],
                        )
                        or []
                        if isinstance(role, dict)
                    ],
                    "quality_warnings": list(item.get("quality_warnings", []) or []),
                    "camera": item.get("camera"),
                    "bbox_xyxy": item.get("bbox_xyxy"),
                    "centroid_px": item.get("centroid_px"),
                    "world_m": item.get("world_m"),
                    "top_surface_world_m": item.get("top_surface_world_m"),
                    "approach_world_m": item.get("approach_world_m"),
                    "grasp_world_m": item.get("grasp_world_m"),
                    "contact_world_m": item.get("contact_world_m"),
                    "score": item.get("score"),
                }
            )
            if len(compact) >= max_objects:
                break
        return compact

    def _prepare_agent_identity_bindings(
        self,
        queries: list[dict[str, Any]],
        *,
        scene_instances: list[dict[str, Any]],
        allow_context_binding: bool = False,
    ) -> list[dict[str, Any]]:
        prepared = self._mark_agent_identity_binding_required(
            queries,
            allow_context_binding=allow_context_binding,
        )
        for query in prepared:
            if not bool(query.get("identity_binding_required")):
                continue
            query.pop("identity_binding_error", None)
            instance_ref = str(query.get("instance_ref", "") or "").strip()
            oracle_id = str(query.get("oracle_id", "") or "").strip()
            matches: list[dict[str, Any]] = []
            for candidate in scene_instances:
                candidate_refs = {
                    str(candidate.get("instance_id", "") or "").strip(),
                    str(candidate.get("track_id", "") or "").strip(),
                    str(candidate.get("oracle_id", "") or "").strip(),
                    str(candidate.get("oracle_source_path", "") or "").strip(),
                }
                if instance_ref and instance_ref not in candidate_refs:
                    continue
                if oracle_id and oracle_id != str(candidate.get("oracle_id", "") or "").strip():
                    continue
                if instance_ref or oracle_id:
                    matches.append(candidate)
            if len(matches) == 1:
                candidate = matches[0]
                canonical_ref = str(candidate.get("track_id") or candidate.get("instance_id") or "").strip()
                if canonical_ref:
                    query["instance_ref"] = canonical_ref
                candidate_oracle_id = str(candidate.get("oracle_id", "") or "").strip()
                if candidate_oracle_id:
                    query["oracle_id"] = candidate_oracle_id
                continue
            if not instance_ref and not oracle_id:
                query["identity_binding_error"] = "agent_did_not_return_instance_reference"
            elif not matches:
                query["identity_binding_error"] = "selected_instance_not_visible_or_grounded"
            else:
                query["identity_binding_error"] = "instance_reference_not_unique"
        return prepared

    def _preserve_required_binding_roles(
        self,
        original: list[dict[str, Any]],
        rebound: list[dict[str, Any]],
        *,
        max_objects: int,
        allow_context_binding: bool = False,
    ) -> list[dict[str, Any]]:
        result = [dict(item) for item in rebound if isinstance(item, dict)]
        roles = ("target", "tool", "context") if allow_context_binding else ("target", "tool")
        for role in roles:
            required = [
                dict(item)
                for item in original
                if isinstance(item, dict)
                and bool(item.get("identity_binding_required"))
                and normalize_query_role(item.get("role", item.get("query_role", "context"))) == role
            ]
            returned_count = sum(
                normalize_query_role(item.get("role", item.get("query_role", "context"))) == role
                for item in result
            )
            for item in required[returned_count:]:
                item["identity_binding_error"] = "agent_omitted_required_binding"
                result.append(item)
        while len(result) > max_objects:
            removable_index = next(
                (
                    index
                    for index in range(len(result) - 1, -1, -1)
                    if not bool(result[index].get("identity_binding_required"))
                ),
                None,
            )
            if removable_index is None:
                break
            result.pop(removable_index)
        return result[:max_objects]

    def _bind_scene_memory_focus_with_agent(
        self,
        *,
        scene_memory: dict[str, Any],
        perception_queries: list[dict[str, Any]],
        observation_summary: str,
        current_subtask: str,
        identity_binding_required: bool = True,
        allow_context_binding: bool = False,
        previous_scene_memory: dict[str, Any] | None = None,
        perception_binding_requirement: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        scene_instances = self._scene_instance_catalog_for_query_payload(scene_memory)
        prepared = self._prepare_agent_identity_bindings(
            perception_queries,
            scene_instances=scene_instances,
            allow_context_binding=allow_context_binding,
        )
        has_required_role_query = any(bool(item.get("identity_binding_required")) for item in prepared)
        needs_binding = any(str(item.get("identity_binding_error", "") or "") for item in prepared) or (
            identity_binding_required and not has_required_role_query
        )
        binding_requested = False
        if needs_binding and scene_instances:
            max_objects = max(1, int(getattr(self.config, "observation_preprocess_max_objects", 3)))
            binding_requirement = self._perception_binding_requirement(
                allow_context_binding=allow_context_binding,
                retry_context=perception_binding_requirement,
            )
            binding_failures = [
                str(item.get("identity_binding_error", "") or "").strip()
                for item in prepared
                if str(item.get("identity_binding_error", "") or "").strip()
            ]
            if not has_required_role_query:
                binding_failures.append(
                    (
                        "agent_returned_no_target_tool_or_context_binding"
                        if allow_context_binding
                        else "agent_returned_no_target_or_tool_binding"
                    )
                )
            if binding_failures and not binding_requirement.get(
                "failure_reason"
            ):
                binding_requirement["failure_reason"] = "|".join(
                    binding_failures
                )[:320]
            if prepared and not binding_requirement.get("previous_queries"):
                binding_requirement["previous_queries"] = [
                    dict(item)
                    for item in prepared[:max_objects]
                ]
            rebound = self._normalize_perception_queries_with_api(
                prepared,
                max_objects=max_objects,
                observation_summary=observation_summary,
                scene_instances=scene_instances,
                require_instance_binding=True,
                binding_requirement=binding_requirement,
            )
            rebound = self._preserve_required_binding_roles(
                prepared,
                rebound,
                max_objects=max_objects,
                allow_context_binding=allow_context_binding,
            )
            prepared = self._prepare_agent_identity_bindings(
                rebound,
                scene_instances=scene_instances,
                allow_context_binding=allow_context_binding,
            )
            binding_requested = True
        rebound_scene = self._scene_memory_tracker.rebind_task_focus(
            scene_memory,
            perception_queries=prepared,
            global_task=self.current_instruction,
            current_subtask=current_subtask,
            identity_binding_required=identity_binding_required,
            allow_context_binding=allow_context_binding,
            previous_scene_memory=previous_scene_memory,
        )
        binding_errors = list((rebound_scene.get("task_focus") or {}).get("identity_binding_errors", []) or [])
        binding_event = {
            "binding_requested": binding_requested,
            "candidate_count": len(scene_instances),
            "queries": prepared,
            "binding_errors": binding_errors,
            "allow_context_binding": allow_context_binding,
        }
        self._trace("agent_instance_binding", **binding_event)
        self._dump_rollout_event("agent_instance_binding", **binding_event)
        return rebound_scene, prepared

    def _encode_snapshot_image_b64(self, snapshot: EnvSnapshot | None, *, camera: str) -> str:
        if snapshot is None:
            return ""
        image = getattr(snapshot, f"{camera}_rgb", None)
        if image is None:
            image = getattr(snapshot, "head_rgb", None)
        if image is None:
            return ""
        try:
            pil_image = Image.fromarray(image.astype("uint8"))
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception:
            return ""

    def _attach_perception_query_metadata(self, result: dict[str, Any], query: dict[str, Any]) -> None:
        role = str(query.get("role", query.get("query_role", "context"))).strip().lower().replace("-", "_").replace(" ", "_")
        if role not in {"target", "tool", "context"}:
            role = "context"
        result["query_role"] = role
        result["query_instance_hint"] = str(query.get("instance_hint", query.get("query_instance_hint", ""))).strip()
        result["query_reason"] = str(query.get("reason", query.get("query_reason", ""))).strip()
        result["instance_ref"] = str(query.get("instance_ref", query.get("query_instance_ref", "")) or "").strip()
        result["query_oracle_id"] = str(query.get("oracle_id", query.get("query_oracle_id", "")) or "").strip()
        result["identity_binding_required"] = bool(query.get("identity_binding_required", query.get("_identity_binding_required", False)))
        result["identity_binding_error"] = str(query.get("identity_binding_error", "") or "").strip()
        for key in REFERENCE_QUERY_METADATA_KEYS:
            if key in query:
                result[key] = query.get(key)
        detections = result.get("detections")
        if isinstance(detections, list):
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                detection.setdefault("camera", result.get("camera"))
                detection.setdefault("query_role", result["query_role"])
                detection.setdefault("query_instance_hint", result["query_instance_hint"])
                detection.setdefault("query_reason", result["query_reason"])
                detection.setdefault("instance_ref", result["instance_ref"])
                detection.setdefault("query_oracle_id", result["query_oracle_id"])
                detection.setdefault("identity_binding_required", result["identity_binding_required"])
                detection.setdefault("identity_binding_error", result["identity_binding_error"])
                for key in REFERENCE_QUERY_METADATA_KEYS:
                    if key in result:
                        detection.setdefault(key, result.get(key))

    def _compact_segmentation_result(self, result: dict[str, Any]) -> dict[str, Any]:
        keys = {
            "success",
            "object_id",
            "text_prompt",
            "num_detections",
            "mask_path",
            "bbox_xyxy",
            "centroid_px",
            "area_px",
            "score",
            "backend",
            "camera",
            "env_step",
            "error",
            "grounding_3d",
            "quality",
            "dropped_detections",
            "query_role",
            "query_instance_hint",
            "query_reason",
            *REFERENCE_QUERY_METADATA_KEYS,
            "instance_ref",
            "query_oracle_id",
            "identity_binding_required",
            "identity_binding_error",
            "requested_text_prompt",
            "fallback_text_prompt",
            "text_prompt_fallback_attempted",
            "text_prompt_fallback_used",
            "fallback_error",
        }
        compact = {key: result.get(key) for key in keys if key in result}
        detections = result.get("detections")
        if isinstance(detections, list):
            compact_detections: list[dict[str, Any]] = []
            for detection in detections[:5]:
                if not isinstance(detection, dict):
                    continue
                compact_detection_keys = {
                    "rank",
                    "score",
                    "box_xyxy",
                    "bbox_xyxy",
                    "centroid_px",
                    "area_px",
                    "mask_path",
                    "mask_shape",
                    "camera",
                    "grounding_3d",
                    "quality",
                    "query_role",
                    "query_instance_hint",
                    "query_reason",
                    *REFERENCE_QUERY_METADATA_KEYS,
                    "instance_ref",
                    "query_oracle_id",
                    "identity_binding_required",
                    "identity_binding_error",
                }
                compact_detections.append({key: detection.get(key) for key in compact_detection_keys if key in detection})
            compact["detections"] = compact_detections
        return compact

    def _append_preprocess_summary(self, summary: str, payload: dict[str, Any]) -> str:
        segments = payload.get("segmentation", [])
        if not isinstance(segments, list) or not segments:
            return summary
        scene_memory = payload.get("scene_memory")
        if isinstance(scene_memory, dict) and scene_memory:
            parts = self._scene_memory_summary_parts(scene_memory)
            candidate_summary = self._sam_candidate_summary_parts(segments)
            if candidate_summary:
                parts.append("sam3_candidates=" + ";".join(candidate_summary))
            if parts:
                return f"{summary}; " + "; ".join(parts)
        parts: list[str] = []
        for item in segments:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id", "object"))
            camera = str(item.get("camera", "") or "")
            label = f"{object_id}@{camera}" if camera else object_id
            if item.get("success") and item.get("bbox_xyxy"):
                fields = [
                    f"bbox={item.get('bbox_xyxy')}",
                    f"centroid={item.get('centroid_px')}",
                ]
                grounding = item.get("grounding_3d")
                if isinstance(grounding, dict) and grounding.get("success"):
                    fields.append(f"world={grounding.get('centroid_world')}")
                    fields.append(f"approach={grounding.get('approach_point_world')}")
                fields.append(f"score={item.get('score')}")
                parts.append(f"{label}:" + ",".join(fields))
            else:
                parts.append(f"{label}:not_found")
        if not parts:
            return summary
        return f"{summary}; perception_candidates=" + "; ".join(parts)

    def _oracle_catalog_for_query_payload(self) -> list[dict[str, Any]]:
        snapshot = self.latest_snapshot
        if snapshot is None or not bool(getattr(self.config, "oracle_objects_enabled", False)):
            return []
        catalog = self._oracle_catalog_from_snapshot(snapshot)
        if not isinstance(catalog, list):
            return []
        max_objects = max(1, int(getattr(self.config, "oracle_objects_max_objects", 12)))
        compact: list[dict[str, Any]] = []
        for item in catalog[:max_objects]:
            if not isinstance(item, dict):
                continue
            compact.append(
                {
                    "oracle_id": item.get("oracle_id"),
                    "class_name": item.get("class_name"),
                    "source_path": item.get("source_path"),
                    "actor_name": item.get("actor_name"),
                    "aliases": item.get("aliases", []),
                    "position_world": item.get("position_world"),
                    "world_extent_m": item.get("world_extent_m"),
                }
            )
        return compact

    def _oracle_catalog_from_snapshot(self, snapshot: EnvSnapshot) -> list[Any]:
        catalog = getattr(snapshot, "oracle_objects", None)
        return catalog if isinstance(catalog, list) else []

    def _scene_memory_summary_parts(self, scene_memory: dict[str, Any]) -> list[str]:
        instances = scene_memory.get("instances")
        if not isinstance(instances, list):
            instances = []
        instance_by_id = {
            str(item.get("instance_id")): item
            for item in instances
            if isinstance(item, dict) and item.get("instance_id")
        }
        focus = scene_memory.get("task_focus") if isinstance(scene_memory.get("task_focus"), dict) else {}
        parts: list[str] = []
        if isinstance(focus, dict) and focus:
            target_ids = [str(item) for item in focus.get("target_instances", []) or []]
            tool_ids = [str(item) for item in focus.get("tool_instances", []) or []]
            focus_parts = [
                "target=[" + ",".join(self._format_scene_instance(instance_by_id.get(item), fallback_id=item) for item in target_ids) + "]",
                "tool=[" + ",".join(self._format_scene_instance(instance_by_id.get(item), fallback_id=item) for item in tool_ids) + "]",
            ]
            reason = str(focus.get("reason_summary", "")).strip()
            if reason:
                focus_parts.append("reason=" + self._compact_summary_text(reason, 180))
            parts.append("scene_memory=task_focus " + " ".join(focus_parts))
        visible_ids = [
            str(item.get("instance_id"))
            for item in instances
            if isinstance(item, dict) and item.get("status", "visible") == "visible" and item.get("instance_id")
        ]
        if visible_ids:
            parts.append("scene_instances=[" + ",".join(visible_ids[:10]) + "]")
        uncertainty = scene_memory.get("uncertainty")
        if isinstance(uncertainty, list) and uncertainty:
            parts.append("scene_uncertainty=" + self._compact_summary_text(uncertainty[0], 180))
        return parts

    def _format_scene_instance(self, instance: dict[str, Any] | None, *, fallback_id: str) -> str:
        if not isinstance(instance, dict):
            return fallback_id
        fields = [str(instance.get("instance_id", fallback_id))]
        for key, label in (
            ("track_id", "track"),
            ("camera", "camera"),
            ("status", "status"),
            ("stability", "stability"),
            ("bbox_xyxy", "bbox"),
            ("world_cm", "world_cm"),
            ("approach_world_cm", "approach_cm"),
            ("grasp_world_cm", "grasp_cm"),
            ("operation_pose_candidate_count", "pose_candidates"),
            ("score", "score"),
        ):
            value = instance.get(key)
            if value in (None, "", []):
                continue
            if key == "score":
                try:
                    value = round(float(value), 3)
                except Exception:
                    pass
            fields.append(f"{label}={self._compact_summary_text(value, 80)}")
        return "(" + ",".join(fields) + ")"

    def _sam_candidate_summary_parts(self, segments: list[dict[str, Any]]) -> list[str]:
        parts: list[str] = []
        for item in segments[:4]:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id", "object"))
            camera = str(item.get("camera", "") or "")
            label = f"{object_id}@{camera}" if camera else object_id
            role = str(item.get("query_role", "") or "")
            hint = str(item.get("query_instance_hint", "") or "")
            detections = item.get("detections")
            if isinstance(detections, list):
                count = len(detections)
            else:
                try:
                    count = int(item.get("num_detections", 1 if item.get("success") else 0))
                except Exception:
                    count = 1 if item.get("success") else 0
            fields = [f"{label}:count={count}"]
            if role:
                fields.append(f"role={role}")
            if hint:
                fields.append(f"hint={hint}")
            if not item.get("success"):
                error = item.get("error")
                fields.append("status=not_found")
                if error:
                    fields.append("error=" + self._compact_summary_text(error, 90))
            parts.append(",".join(fields))
        return parts

    def _compact_summary_text(self, value: Any, max_chars: int) -> str:
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        text = " ".join(text.split())
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)] + "..."

    def _ground_segmentation_result(self, *, snapshot: EnvSnapshot, result: dict[str, Any], camera: str) -> dict[str, Any]:
        if not bool(result.get("success", False)):
            return {
                "success": False,
                "object_id": str(result.get("object_id", "object")),
                "camera": camera,
                "error": "segmentation failed; grounding skipped",
                "segmentation_error": str(result.get("error", "")),
            }
        camera_data = RMBenchEnvAdapter.camera_data(snapshot, camera)
        return ground_segmentation_result(
            segmentation=result,
            depth_mm=camera_data.get("depth"),
            intrinsic_cv=camera_data.get("intrinsic_cv"),
            cam2world_gl=camera_data.get("cam2world_gl"),
            camera=camera,
            min_valid_ratio=float(getattr(self.config, "observation_grounding_min_valid_ratio", 0.05)),
            max_points=int(getattr(self.config, "observation_grounding_max_points", 5000)),
            approach_height_m=float(getattr(self.config, "observation_grounding_approach_height_m", 0.08)),
            ee_to_contact_m=float(getattr(snapshot, "ee_to_tcp_m", 0.12)),
            tcp_calibration_by_arm=getattr(snapshot, "tcp_calibration_by_arm", None),
        )

    def _ground_detection_candidates(self, *, snapshot: EnvSnapshot, result: dict[str, Any], camera: str) -> None:
        detections = result.get("detections")
        if not isinstance(detections, list):
            return
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            detection.setdefault("object_id", result.get("object_id"))
            detection.setdefault("text_prompt", result.get("text_prompt"))
            detection.setdefault("camera", result.get("camera", camera))
            if "bbox_xyxy" not in detection and "box_xyxy" in detection:
                detection["bbox_xyxy"] = detection.get("box_xyxy")
            grounding_input = dict(detection)
            grounding_input.setdefault("success", bool(result.get("success", False)))
            detection["grounding_3d"] = self._ground_segmentation_result(
                snapshot=snapshot,
                result=grounding_input,
                camera=str(detection.get("camera", camera)),
            )

    def _filter_grounded_preprocess_candidates(self, result: dict[str, Any]) -> None:
        detections = result.get("detections")
        if isinstance(detections, list):
            kept: list[dict[str, Any]] = []
            dropped: list[dict[str, Any]] = []
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                candidate = self._quality_candidate_from_detection(result=result, detection=detection)
                quality = self._grounded_candidate_quality(candidate)
                detection["quality"] = quality
                if bool(quality.get("actionable", True)):
                    kept.append(detection)
                else:
                    dropped.append(self._compact_dropped_preprocess_candidate(candidate, quality))
            if dropped:
                result["dropped_detections"] = dropped
            result["detections"] = kept
            result["num_detections"] = len(kept)
            if kept:
                self._sync_preprocess_result_to_detection(result, kept[0])
                return
            if detections:
                self._mark_preprocess_result_not_actionable(result)
            return

        quality = self._grounded_candidate_quality(result)
        result["quality"] = quality
        if bool(quality.get("actionable", True)):
            return
        result["dropped_detections"] = [self._compact_dropped_preprocess_candidate(result, quality)]
        result["num_detections"] = 0
        self._mark_preprocess_result_not_actionable(result)

    def _quality_candidate_from_detection(self, *, result: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any]:
        candidate = dict(detection)
        for key in (
            "object_id",
            "text_prompt",
            "backend",
            "camera",
            "env_step",
            "robot_state",
            "query_role",
            "query_instance_hint",
            "query_reason",
            "oracle_aliases",
        ):
            candidate.setdefault(key, result.get(key))
        if "bbox_xyxy" not in candidate and "box_xyxy" in candidate:
            candidate["bbox_xyxy"] = candidate.get("box_xyxy")
        return candidate

    def _grounded_candidate_quality(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return scene_candidate_quality(
            candidate,
            min_candidate_score=float(getattr(self.config, "scene_memory_min_candidate_score", 0.12)),
            max_object_z_extent_m=float(getattr(self.config, "scene_memory_max_object_z_extent_m", 0.18)),
            max_object_xy_extent_m=float(getattr(self.config, "scene_memory_max_object_xy_extent_m", 0.30)),
            max_object_world_z_m=float(getattr(self.config, "scene_memory_max_object_world_z_m", 0.95)),
            robot_state=self._get_robot_state(),
            robot_self_filter_radius_m=float(getattr(self.config, "scene_memory_robot_self_filter_radius_m", 0.08)),
            robot_self_filter_z_margin_m=float(getattr(self.config, "scene_memory_robot_self_filter_z_margin_m", 0.08)),
        )

    def _compact_dropped_preprocess_candidate(self, candidate: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
        dropped: dict[str, Any] = {
            "object_id": candidate.get("object_id"),
            "camera": candidate.get("camera"),
            "rank": candidate.get("rank"),
            "score": candidate.get("score"),
            "bbox_xyxy": candidate.get("bbox_xyxy"),
            "quality": quality,
        }
        mask_path = candidate.get("mask_path")
        if mask_path:
            dropped["mask_path"] = mask_path
        return {key: value for key, value in dropped.items() if value not in (None, "", [])}

    def _sync_preprocess_result_to_detection(self, result: dict[str, Any], detection: dict[str, Any]) -> None:
        for key in ("mask_path", "bbox_xyxy", "centroid_px", "area_px", "score", "camera", "grounding_3d", "quality"):
            if key in detection:
                result[key] = detection.get(key)
        if "bbox_xyxy" not in result and "box_xyxy" in detection:
            result["bbox_xyxy"] = detection.get("box_xyxy")
        result["success"] = True

    def _mark_preprocess_result_not_actionable(self, result: dict[str, Any]) -> None:
        result["success"] = False
        result["error"] = "no actionable grounded SAM detections after quality filtering"
        for key in ("mask_path", "bbox_xyxy", "centroid_px", "area_px", "score", "grounding_3d", "quality"):
            result.pop(key, None)

    def _attach_candidate_robot_state(self, result: dict[str, Any]) -> None:
        robot_state = self._get_robot_state()
        if not robot_state:
            return
        result["robot_state"] = robot_state
        detections = result.get("detections")
        if isinstance(detections, list):
            for detection in detections:
                if isinstance(detection, dict):
                    detection.setdefault("robot_state", robot_state)

    def needs_control_turn(self) -> tuple[bool, ControlSignal]:
        state = self.memory_store.state
        snapshot = self.latest_snapshot
        if snapshot is None:
            return True, ControlSignal.STARTUP
        if self._pure_tool_control_enabled():
            if self._latest_snapshot_task_success():
                return True, ControlSignal.SUCCESS
        elif snapshot.eval_success or snapshot.check_success:
            return True, ControlSignal.SUCCESS
        if state.task.task_finished:
            return False, ControlSignal.SUCCESS
        if state.recovery.pending_action:
            return True, ControlSignal.RECOVERY_PENDING
        if state.active_skill is None:
            return True, ControlSignal.NO_ACTIVE_SKILL
        if state.monitor.status in {"rollout_failed", "rollout_stalled", "waiting_retry", "waiting_reset", "waiting_replan"}:
            return True, ControlSignal.MONITOR_ALERT
        if state.working.steps_since_last_decision >= max(1, self.config.decision_interval):
            return True, ControlSignal.INTERVAL
        return False, ControlSignal.INTERVAL

    def build_control_turn_payload(self, signal: ControlSignal) -> str:
        return build_reasoner_turn_payload(
            global_task=self.current_instruction,
            agent_state=self._reasoner_agent_state(),
            available_skills=self._available_control_skill_summaries(),
            trigger=signal.value,
            memory_harness=self._memory_summarization_harness(),
            runtime_evaluation=self._runtime_evaluation_context(),
        )

    def _reasoner_agent_state(self) -> dict[str, Any]:
        """Return the semantic planner view without private executor geometry.

        Working Memory intentionally retains complete operation-pose candidates so
        the local recovery adapter can select and block them.  Those SE(3) arrays
        are runtime-owned execution data, not planner inputs.  Serializing the
        complete observation-preprocess and Scene Memory payloads here also
        duplicates that geometry and can exceed the model context window.
        """

        state = self.memory_store.to_reasoner_context()
        working = state.get("working")
        if not isinstance(working, dict):
            return state
        compact_working = dict(working)
        observation_preprocess = self.memory_store.state.working.observation_preprocess
        scene_memory = self.memory_store.state.working.scene_memory
        planner_context_mode = self._planner_context_mode()
        compact_working["planner_context_mode"] = planner_context_mode
        if planner_context_mode == COMPACT_PLANNER_CONTEXT_MODE:
            compact_working.pop("recent_observation_summary", None)
            compact_working.pop("observation_preprocess", None)
            compact_working.pop("manipulation_state", None)
            compact_working["scene_memory"] = self._planner_scene_view(
                scene_memory=scene_memory,
                observation_preprocess=observation_preprocess,
            )
        else:
            compact_working["observation_preprocess"] = (
                self._compact_observation_preprocess_for_recovery(
                    observation_preprocess
                )
                if observation_preprocess
                else {}
            )
            compact_working["scene_memory"] = (
                self._compact_scene_memory_for_effect(scene_memory)
                if scene_memory
                else {}
            )
        state["working"] = compact_working
        return state

    def _runtime_evaluation_context(self) -> dict[str, Any]:
        snapshot = self.latest_snapshot
        if snapshot is None:
            environment_success = bool(self._authoritative_environment_success)
            return {
                "environment_success": environment_success,
                "check_success": False,
                "global_task_success": environment_success,
                "global_success_authority": (
                    "environment_eval_success"
                    if environment_success
                    else "environment_not_observed"
                ),
                "max_reward": 0.0,
                "env_step": -1,
                "step_limit": -1,
                "pure_tool_control": self._pure_tool_control_enabled(),
                "reobserve_scene_enabled": bool(
                    self._recovery_dispatcher.reobserve_scene_enabled
                ),
                "grasp_transport_policy": self._grasp_transport_policy(),
                "release_guard_enabled": self._release_guard_enabled(),
                "action_geometry_repair_pending_policy": (
                    self._action_geometry_repair_pending_policy()
                ),
            }
        pure_tool_control = self._pure_tool_control_enabled()
        environment_success = self._latest_snapshot_task_success()
        check_success = bool(snapshot.check_success)
        return {
            "environment_success": environment_success,
            "check_success": check_success,
            "global_task_success": environment_success if pure_tool_control else environment_success or check_success,
            "global_success_authority": (
                "environment_eval_success" if pure_tool_control else "environment_eval_or_check_success"
            ),
            "max_reward": float(snapshot.max_reward),
            "env_step": int(snapshot.step_count),
            "step_limit": int(snapshot.step_limit),
            "pure_tool_control": pure_tool_control,
            "reobserve_scene_enabled": bool(
                self._recovery_dispatcher.reobserve_scene_enabled
            ),
            "grasp_transport_policy": self._grasp_transport_policy(),
            "release_guard_enabled": self._release_guard_enabled(),
            "action_geometry_repair_pending_policy": (
                self._action_geometry_repair_pending_policy()
            ),
        }

    def _available_control_skill_summaries(self) -> list[str]:
        summaries: list[str] = []
        for skill in self.skill_registry.list_skills(refresh=True):
            if str(skill.metadata.get("runtime_role", "")).strip() == "memory_harness":
                continue
            summaries.append(
                f"{skill.name}: {skill.description} | policy={skill.policy_binding} "
                f"| recovery={','.join(skill.recovery_skills) if skill.recovery_skills else 'none'} "
                f"| retry={skill.retry_budget} reset={skill.reset_budget} replan={skill.replan_budget}"
            )
        return summaries

    def _memory_summarization_harness(self) -> dict[str, Any]:
        skill = self.skill_registry.get_skill("observation-memory-summarization", refresh=True)
        if skill is None:
            return {}
        return {
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.body,
        }

    def run_control_turn(self, signal: ControlSignal) -> dict[str, Any]:
        assert self.latest_snapshot is not None
        control_turn_index = self._pure_tool_control_control_turns + (1 if self._pure_tool_control_enabled() else 0)
        payload = RMBenchEnvAdapter.build_reasoner_payload(self.segment_start_snapshot or self.latest_snapshot, self.latest_snapshot)
        control_turn = self.build_control_turn_payload(signal)
        control_payload_bytes = len(control_turn.encode("utf-8"))
        control_scene_memory: dict[str, Any] | None = None
        try:
            control_turn_payload = json.loads(control_turn)
            candidate_scene_memory = (
                control_turn_payload.get("agent_state", {})
                .get("working", {})
                .get("scene_memory", {})
            )
            if isinstance(candidate_scene_memory, dict):
                control_scene_memory = candidate_scene_memory
        except Exception:
            control_scene_memory = None
        self.memory_store.set_monitor_status(phase="reasoning", status="needs_reasoning", note=f"control turn triggered: {signal.value}")
        self.memory_store.add_user_message(control_turn)
        self._trace(
            "control_turn_start",
            trigger=signal.value,
            active_skill=None if self.memory_store.state.active_skill is None else self.memory_store.state.active_skill.skill_name,
            monitor_status=self.memory_store.state.monitor.status,
            recovery_pending=self.memory_store.state.recovery.pending_action,
            control_turn_index=control_turn_index,
            control_payload_bytes=control_payload_bytes,
            runtime_evaluation=self._runtime_evaluation_context(),
        )
        self._dump_rollout_event(
            "control_turn_start",
            trigger=signal.value,
            active_skill=None if self.memory_store.state.active_skill is None else self.memory_store.state.active_skill.skill_name,
            monitor_status=self.memory_store.state.monitor.status,
            recovery_pending=self.memory_store.state.recovery.pending_action,
            control_turn_index=control_turn_index,
            control_payload_bytes=control_payload_bytes,
            runtime_evaluation=self._runtime_evaluation_context(),
        )
        started = time.time()
        if control_scene_memory is not None:
            self._record_gpt_scene_memory_request(
                consumer="control_planner",
                scene_memory=control_scene_memory,
                request_metadata={
                    "control_turn_index": control_turn_index,
                    "trigger": signal.value,
                    "active_skill": (
                        None
                        if self.memory_store.state.active_skill is None
                        else self.memory_store.state.active_skill.skill_name
                    ),
                    "control_payload_bytes": control_payload_bytes,
                },
            )
        prediction = self.control_runtime.predict_planner_step(
            task=control_turn,
            previous_memory_text=self.memory_store.export_reasoner_memory(),
            planner_start_image=payload["planner_start_image"],
            planner_end_image=payload["planner_end_image"],
            planner_state=payload["planner_state"],
        )
        latency_sec = time.time() - started
        if self._pure_tool_control_enabled() and self._pure_tool_control_control_backend_errors:
            previous_error_count = self._pure_tool_control_control_backend_errors
            self._pure_tool_control_control_backend_errors = 0
            recovery_payload = {
                "previous_backend_error_count": previous_error_count,
                "backend_error_count": 0,
                "backend_error_budget": self._pure_tool_control_backend_error_budget(),
            }
            self._trace("pure_tool_control_control_backend_recovered", **recovery_payload)
            self._dump_rollout_event("pure_tool_control_control_backend_recovered", **recovery_payload)
        prediction.setdefault("action_mode", "start" if self.memory_store.state.active_skill is None else "switch")
        prediction.setdefault("note", "unified chat tool loop")
        self.memory_store.set_committed_memory(str(prediction["memory_text"]))
        self.memory_store.add_agent_message(str(prediction))
        self.memory_store.record_decision(
            trigger=signal.value,
            note=f"action_mode={prediction.get('action_mode', '')}, subtask={prediction.get('subtask_text', '')}",
            commit_label=str(prediction.get("commit_label", "")),
        )
        self.segment_start_snapshot = self.latest_snapshot
        if self._pure_tool_control_enabled():
            self._pure_tool_control_control_turns = control_turn_index
            self._pure_tool_control_no_progress_control_turns += 1
        self._trace(
            "control_turn_result",
            trigger=signal.value,
            latency_sec=round(latency_sec, 4),
            action_mode=str(prediction.get("action_mode", "")),
            selected_skill=str(prediction.get("selected_skill", "")),
            subtask_text=str(prediction.get("subtask_text", "")),
            preferred_arm=normalize_preferred_arm(prediction.get("preferred_arm")),
            memory_text=str(prediction.get("memory_text", "")),
            commit_label=str(prediction.get("commit_label", "")),
            semantic_tags=prediction.get("semantic_tags", {}),
            control_turn_index=control_turn_index,
        )
        self._dump_rollout_event(
            "control_turn_result",
            trigger=signal.value,
            latency_sec=round(latency_sec, 4),
            action_mode=str(prediction.get("action_mode", "")),
            selected_skill=str(prediction.get("selected_skill", "")),
            subtask_text=str(prediction.get("subtask_text", "")),
            preferred_arm=normalize_preferred_arm(prediction.get("preferred_arm")),
            memory_text=str(prediction.get("memory_text", "")),
            commit_label=str(prediction.get("commit_label", "")),
            semantic_tags=prediction.get("semantic_tags", {}),
            control_turn_index=control_turn_index,
        )
        return prediction

    def _normalize_skill_name(self, skill_name: str) -> str:
        normalized = str(skill_name).strip().strip("`").strip("\"'")
        normalized = normalized.replace("–", "-").replace("—", "-").replace("_", "-")
        normalized = "-".join(part for part in normalized.split("-") if part)
        return normalized.lower()

    def _resolve_skill_spec(self, skill_name: str):
        normalized_skill_name = self._normalize_skill_name(skill_name)
        skill = self.skill_registry.get_skill(normalized_skill_name, refresh=True)
        if skill is not None:
            return skill.to_skill_spec(normalized_skill_name)
        for candidate in self.skill_registry.list_skills(refresh=True):
            if self._normalize_skill_name(candidate.name) == normalized_skill_name:
                return candidate.to_skill_spec(candidate.name)
        available_skills = [candidate.name for candidate in self.skill_registry.list_skills(refresh=True)]
        raise RuntimeError(
            "No workflow-defined skill matches skill name: "
            f"raw={skill_name!r} normalized={normalized_skill_name!r} "
            f"available={available_skills!r}"
        )

    def _select_action_from_policy(self, *, signal_name: str) -> str:
        recovery = self.memory_store.state.recovery
        active = self.memory_store.state.active_skill
        abort_on_exhausted = True
        signal_actions: tuple[str, ...] = ("abort",)
        if active is not None:
            skill_spec = self._resolve_skill_spec(active.skill_name)
            recovery_policy = getattr(skill_spec, "recovery_policy", None)
            if recovery_policy is not None:
                abort_on_exhausted = bool(getattr(recovery_policy, "abort_on_exhausted", True))
                signal_actions = tuple(getattr(recovery_policy, "signal_actions", {}).get(signal_name, ("abort",)))
        for action in signal_actions:
            if action == "retry" and recovery.retry_used < recovery.retry_budget:
                return "retry"
            if action == "reset" and recovery.reset_used < recovery.reset_budget:
                return "reset"
            if action == "replan" and recovery.replan_used < recovery.replan_budget:
                return "replan"
            if action == "abort":
                return "abort" if abort_on_exhausted else "replan"
        return "abort" if abort_on_exhausted else "replan"

    def _start_or_replace_scoped_skill(
        self,
        *,
        subtask_text: str,
        skill_spec: Any,
        semantic_tags: dict[str, Any] | None = None,
        preferred_arm: str = "either",
        force_new_attempt: bool = False,
    ) -> Any:
        previous = self.memory_store.state.active_skill
        previous_skill_id = "" if previous is None else str(previous.skill_id)
        discarded_evidence_count = len(self.memory_store.state.working.recovery_history)
        run_state = self.memory_store.start_or_replace_skill(
            subtask_text=subtask_text,
            skill_spec=skill_spec,
            semantic_tags=semantic_tags,
            preferred_arm=preferred_arm,
            force_new_attempt=force_new_attempt,
        )
        if str(run_state.skill_id) == previous_skill_id:
            return run_state
        self._pending_action_effect_verification = None
        self._pure_tool_control_empty_plan_turns = 0
        self._grounded_geometry_leases.clear()
        self._partial_grounded_approach_leases.clear()
        payload = {
            "previous_skill_id": previous_skill_id,
            "skill_id": str(run_state.skill_id),
            "active_subtask": str(run_state.instruction),
            "discarded_recovery_evidence_count": discarded_evidence_count,
        }
        self._trace("subtask_evidence_scope_started", **payload)
        self._dump_rollout_event("subtask_evidence_scope_started", **payload)
        return run_state

    def apply_control_decision(self, prediction: dict[str, Any]) -> None:
        action_mode = str(prediction.get("action_mode", "")).strip() or ("start" if self.memory_store.state.active_skill is None else "switch")
        subtask_text = str(prediction.get("subtask_text", "")).strip()
        selected_skill = str(prediction.get("selected_skill", "")).strip() or "monitored-subtask-execution"
        active = self.memory_store.state.active_skill
        preferred_arm = normalize_preferred_arm(prediction.get("preferred_arm"))
        planner_tags = normalize_semantic_tags(prediction.get("semantic_tags"), default_source="planner_vlm")
        if has_semantic_tags(planner_tags):
            self.memory_store.update_runtime_semantic_tags(planner_tags)
        print(f"[img_agent] selected_skill={selected_skill!r} subtask_text={subtask_text!r}", flush=True)
        if action_mode == "finish":
            if self._pure_tool_control_enabled() and not self._latest_snapshot_task_success():
                note = str(prediction.get("note", "planner requested finish without environment success")).strip()
                self.memory_store.record_recovery("pure_tool_control_finish_rejected")
                self.memory_store.set_monitor_status(
                    phase="reasoning",
                    status="needs_reasoning",
                    note=f"planner finish rejected because environment success is false: {note}",
                    env_signal=RUNNING,
                    failure_reason=note,
                    progress_score=0.0,
                )
                self._trace(
                    "control_decision",
                    action_mode=action_mode,
                    rejected=True,
                    reason="pure_tool_control_finish_without_env_success",
                    note=note,
                    eval_success=False if self.latest_snapshot is None else bool(self.latest_snapshot.eval_success),
                    check_success=False if self.latest_snapshot is None else bool(self.latest_snapshot.check_success),
                )
                self._dump_rollout_event(
                    "control_decision",
                    action_mode=action_mode,
                    rejected=True,
                    reason="pure_tool_control_finish_without_env_success",
                    note=note,
                )
                return
            self.memory_store.mark_active_skill_succeeded(note="finished by control loop")
            self.memory_store.mark_task_finished(note=str(prediction.get("note", "control loop finished task")))
            self._trace("control_decision", action_mode=action_mode, note=str(prediction.get("note", "")))
            self._dump_rollout_event("control_decision", action_mode=action_mode, note=str(prediction.get("note", "")))
            return
        if action_mode == "retry":
            self.memory_store.record_recovery("policy_retry")
            if active is not None:
                self.memory_store.update_active_skill_preferred_arm(preferred_arm)
                self.memory_store.retry_active_skill()
                self._trace("control_decision", action_mode=action_mode)
                return
            if not subtask_text:
                note = "planner retry rejected because no active skill or replacement subtask exists"
                self.memory_store.set_monitor_status(
                    phase="reasoning",
                    status="needs_reasoning",
                    note=note,
                    env_signal=RUNNING,
                    failure_reason=note,
                    progress_score=0.0,
                )
                self._trace("control_decision", action_mode=action_mode, rejected=True, reason="retry_without_active_skill_or_subtask")
                self._dump_rollout_event("control_decision", action_mode=action_mode, rejected=True, reason="retry_without_active_skill_or_subtask")
                return
            self.memory_store.record_recovery("policy_retry_promoted_to_start")
            self._trace(
                "control_decision_normalized",
                original_action_mode="retry",
                action_mode="start",
                reason="retry supplied a replacement subtask while no active skill existed",
                subtask_text=subtask_text,
            )
            self._dump_rollout_event(
                "control_decision_normalized",
                original_action_mode="retry",
                action_mode="start",
                reason="retry supplied a replacement subtask while no active skill existed",
                subtask_text=subtask_text,
            )
            action_mode = "start"
        if action_mode == "reset":
            self.memory_store.record_recovery("policy_reset")
            self.executor_runtime.reset()
            self.memory_store.set_monitor_status(phase="recovery", status="waiting_replan", note="executor reset complete")
            self._trace("control_decision", action_mode=action_mode)
            return
        if action_mode == "replan":
            self.memory_store.record_recovery("policy_replan")
            note = str(prediction.get("note", "workflow requested replan"))
            if self._pure_tool_control_enabled():
                active = self.memory_store.state.active_skill
                if active is None and subtask_text:
                    self.memory_store.record_recovery(
                        "pure_tool_control_planner_replan_promoted_to_start"
                    )
                    self._trace(
                        "control_decision_normalized",
                        original_action_mode="replan",
                        action_mode="start",
                        reason=(
                            "planner supplied an executable replacement subtask "
                            "while no active skill existed"
                        ),
                        subtask_text=subtask_text,
                    )
                    self._dump_rollout_event(
                        "control_decision_normalized",
                        original_action_mode="replan",
                        action_mode="start",
                        reason=(
                            "planner supplied an executable replacement subtask "
                            "while no active skill existed"
                        ),
                        subtask_text=subtask_text,
                    )
                    action_mode = "start"
                elif active is not None:
                    self.memory_store.record_recovery("pure_tool_control_planner_replan_deferred")
                    self.memory_store.set_monitor_status(
                        phase="monitoring",
                        status="rollout_active",
                        note="planner replan deferred until after-action verifier resolves the active subtask",
                        env_signal=RUNNING,
                        failure_reason=note,
                        progress_score=0.0,
                    )
                    self._trace(
                        "control_decision",
                        action_mode=action_mode,
                        rejected=True,
                        reason="after_action_verifier_owns_active_subtask_transition",
                        note=note,
                    )
                    self._dump_rollout_event(
                        "control_decision",
                        action_mode=action_mode,
                        rejected=True,
                        reason="after_action_verifier_owns_active_subtask_transition",
                        note=note,
                    )
                    return
                else:
                    self.memory_store.set_monitor_status(
                        phase="reasoning",
                        status="needs_reasoning",
                        note="pure tool-control planner requested another planning turn",
                        env_signal=RUNNING,
                        failure_reason=note,
                        progress_score=0.0,
                    )
                    self._trace("control_decision", action_mode=action_mode, note=str(prediction.get("note", "")))
                    self._dump_rollout_event("control_decision", action_mode=action_mode, note=str(prediction.get("note", "")))
                    return
            else:
                self.memory_store.mark_active_skill_failed(note=note)
                self._trace("control_decision", action_mode=action_mode, note=str(prediction.get("note", "")))
                self._dump_rollout_event("control_decision", action_mode=action_mode, note=str(prediction.get("note", "")))
                return
        if action_mode == "continue" and active is not None:
            self.memory_store.update_active_skill_preferred_arm(preferred_arm)
            if has_semantic_tags(planner_tags):
                self.memory_store.update_active_skill_semantic_tags(planner_tags)
            self.memory_store.record_tool_call("continue_active_skill")
            self.memory_store.set_monitor_status(phase="rollout", status="running", note="continue current skill")
            self._trace(
                "control_decision",
                action_mode=action_mode,
                selected_skill=active.skill_name,
                subtask_text=active.instruction,
                semantic_tags=planner_tags,
                preferred_arm=preferred_arm,
            )
            return
        skill_spec = self._resolve_skill_spec(selected_skill)
        rendered_instruction = skill_spec.render_instruction(subtask_text)
        self._trace(
            "control_decision",
            action_mode=action_mode,
            selected_skill=skill_spec.name,
            subtask_text=subtask_text,
            rendered_instruction=rendered_instruction,
            reset_executor=bool(active is not None and active.instruction != rendered_instruction and self.config.interrupt_on_skill_change),
            semantic_tags=planner_tags,
            preferred_arm=preferred_arm,
        )
        if action_mode in {"start", "recover"} or active is None:
            self._start_or_replace_scoped_skill(
                subtask_text=subtask_text,
                skill_spec=skill_spec,
                semantic_tags=planner_tags,
                preferred_arm=preferred_arm,
                force_new_attempt=active is not None,
            )
            return
        if active.instruction != rendered_instruction or active.skill_name != skill_spec.name or action_mode == "switch":
            if self.config.interrupt_on_skill_change:
                self.executor_runtime.reset()
            self._start_or_replace_scoped_skill(
                subtask_text=subtask_text,
                skill_spec=skill_spec,
                semantic_tags=planner_tags,
                preferred_arm=preferred_arm,
                force_new_attempt=action_mode == "switch",
            )

    def _apply_recovery_policy(self) -> None:
        action, reason = self.memory_store.consume_recovery_policy()
        if not action:
            return
        self.memory_store.record_recovery(f"workflow_policy:{action}:{reason}")
        self._trace("recovery_policy", action=action, reason=reason, robot_state=self._get_robot_state())
        if action == "retry":
            self.memory_store.retry_active_skill()
            return
        if action == "reset":
            self.executor_runtime.reset()
            self.memory_store.set_monitor_status(phase="recovery", status="waiting_replan", note=reason)
            return
        if action == "replan":
            if self._pure_tool_control_enabled():
                self.memory_store.record_recovery("pure_tool_control_legacy_replan_deferred")
                self.memory_store.set_monitor_status(
                    phase="monitoring",
                    status="rollout_active",
                    note="legacy recovery replan deferred until after-action verification",
                    env_signal=RUNNING,
                    failure_reason=reason,
                    progress_score=0.0,
                )
                return
            self.memory_store.mark_active_skill_failed(note=reason)
            return
        if action == "abort":
            if self._pure_tool_control_enabled():
                self.memory_store.record_recovery("pure_tool_control_legacy_abort_deferred")
                self.memory_store.set_monitor_status(
                    phase="monitoring",
                    status="rollout_active",
                    note="legacy recovery abort deferred until after-action verification",
                    env_signal=RUNNING,
                    failure_reason=reason,
                    progress_score=0.0,
                )
                return
            self.memory_store.mark_active_skill_failed(note=reason)
            self.memory_store.mark_task_finished(note=reason)

    def _select_recovery_action(self, *, preferred_action: str, signal_name: str) -> str:
        recovery = self.memory_store.state.recovery
        preferred = str(preferred_action).strip()
        if preferred == "retry" and recovery.retry_used < recovery.retry_budget:
            return "retry"
        if preferred == "reset" and recovery.reset_used < recovery.reset_budget:
            return "reset"
        if preferred == "replan" and recovery.replan_used < recovery.replan_budget:
            return "replan"
        if preferred == "abort":
            return "abort"
        return self._select_action_from_policy(signal_name=signal_name)

    def _pure_tool_control_enabled(self) -> bool:
        return bool(getattr(self.config, "pure_tool_control_enabled", False))

    def _grasp_transport_policy(self) -> str:
        return str(
            getattr(self.config, "grasp_transport_policy", "strict")
            or "strict"
        ).strip().lower().replace("-", "_")

    def _grasp_transport_evidence_only(self) -> bool:
        return self._grasp_transport_policy() == "evidence_only"

    def _release_guard_enabled(self) -> bool:
        return bool(getattr(self.config, "release_guard_enabled", False))

    def _planner_context_mode(self) -> str:
        return normalize_planner_context_mode(
            getattr(self.config, "planner_context_mode", "legacy"),
            field_name="agent.recovery.planner_context_mode",
        )

    def _observation_trace_payload_mode(self) -> str:
        return normalize_trace_payload_mode(
            getattr(
                self.config,
                "observation_trace_payload_mode",
                COMPACT_TRACE_PAYLOAD_MODE,
            ),
            field_name="agent.rollout_dump.trace_payload_mode",
        )

    def _planner_scene_view(
        self,
        *,
        scene_memory: dict[str, Any] | None = None,
        observation_preprocess: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        working = self.memory_store.state.working
        return build_planner_scene_view(
            working.scene_memory
            if scene_memory is None
            else scene_memory,
            manipulation_state=working.manipulation_state,
            observation_preprocess=(
                working.observation_preprocess
                if observation_preprocess is None
                else observation_preprocess
            ),
            max_instances=max(
                8,
                int(
                    getattr(
                        self.config,
                        "observation_preprocess_max_objects",
                        8,
                    )
                ),
            ),
        )

    def _observation_preprocess_finalized_trace_payload(
        self,
        preprocess_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            self._observation_trace_payload_mode()
            != COMPACT_TRACE_PAYLOAD_MODE
        ):
            return dict(preprocess_payload or {})
        scene_memory = preprocess_payload.get("scene_memory")
        planner_scene_view = self._planner_scene_view(
            scene_memory=(
                scene_memory if isinstance(scene_memory, dict) else {}
            ),
            observation_preprocess=preprocess_payload,
        )
        compact = compact_finalized_observation_event(
            preprocess_payload,
            planner_scene_view=planner_scene_view,
        )
        compact["task"] = {
            "global_task": self.current_instruction,
            "current_subtask": str(
                (
                    planner_scene_view.get("task_focus", {}) or {}
                ).get("current_subtask", "")
                or ""
            ),
        }
        compact["robot_state"] = self._compact_robot_state_for_effect(
            self._get_robot_state()
        )
        return compact

    def _safe_observation_preprocess_finalized_trace_payload(
        self,
        preprocess_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep finalized trace projection failures out of rollout control."""

        try:
            return self._observation_preprocess_finalized_trace_payload(
                preprocess_payload
            )
        except Exception as exc:
            return {
                "schema": (
                    "trace/observation_preprocess_finalized_error/v1"
                ),
                "schema_version": 1,
                "projection_error": type(exc).__name__,
                "payload_omitted": True,
                "observation_generation": preprocess_payload.get(
                    "observation_generation"
                ),
                "observation_capture_id": preprocess_payload.get(
                    "observation_capture_id"
                ),
            }

    def _action_geometry_repair_pending_policy(self) -> str:
        return normalize_action_geometry_repair_pending_policy(
            getattr(
                self.config,
                "action_geometry_repair_pending_policy",
                STRICT_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
            )
        )

    def _pure_tool_control_max_control_turns(self) -> int:
        return max(0, int(getattr(self.config, "pure_tool_control_max_control_turns", 64)))

    def _pure_tool_control_max_no_progress_control_turns(self) -> int:
        return max(
            0,
            int(getattr(self.config, "pure_tool_control_max_no_progress_control_turns", 10)),
        )

    def _pure_tool_control_backend_error_budget(self) -> int:
        return max(1, int(getattr(self.config, "pure_tool_control_backend_error_budget", 5)))

    def _pure_tool_control_empty_plan_replan_threshold(self) -> int:
        return max(1, int(getattr(self.config, "pure_tool_control_empty_plan_replan_threshold", 2)))

    def _stop_for_pure_tool_control_turn_budget(self) -> bool:
        if not self._pure_tool_control_enabled() or self._latest_snapshot_task_success():
            return False
        max_no_progress_turns = self._pure_tool_control_max_no_progress_control_turns()
        if (
            max_no_progress_turns > 0
            and self._pure_tool_control_no_progress_control_turns >= max_no_progress_turns
        ):
            if self._pure_tool_control_terminal_failure:
                return True
            reason = (
                "pure_tool_control_max_no_progress_control_turns_exhausted:"
                f"{max_no_progress_turns}"
            )
            self._set_pure_tool_control_terminal_failure(
                reason=reason,
                note=(
                    "pure tool-control made no verifier-confirmed subtask progress within the "
                    "consecutive control-turn allowance"
                ),
                event="pure_tool_control_no_progress_control_turn_budget_exhausted",
                control_turn_count=self._pure_tool_control_control_turns,
                no_progress_control_turn_count=self._pure_tool_control_no_progress_control_turns,
                max_no_progress_control_turns=max_no_progress_turns,
            )
            return True
        max_control_turns = self._pure_tool_control_max_control_turns()
        if max_control_turns <= 0 or self._pure_tool_control_control_turns < max_control_turns:
            return False
        if self._pure_tool_control_terminal_failure:
            return True
        reason = f"pure_tool_control_max_control_turns_exhausted:{max_control_turns}"
        self._pure_tool_control_terminal_failure = reason
        self.memory_store.record_recovery(reason)
        self.memory_store.set_monitor_status(
            phase="reasoning",
            status="rollout_failed",
            note="pure tool-control planner-call budget exhausted before environment success",
            env_signal=RUNNING,
            failure_reason=reason,
            progress_score=0.0,
        )
        payload = {
            "reason": reason,
            "control_turn_count": self._pure_tool_control_control_turns,
            "max_control_turns": max_control_turns,
            "eval_success": False,
        }
        self._trace("pure_tool_control_control_turn_budget_exhausted", **payload)
        self._dump_rollout_event("pure_tool_control_control_turn_budget_exhausted", **payload)
        return True

    def _latest_snapshot_success(self) -> bool:
        snapshot = self.latest_snapshot
        return bool(snapshot is not None and (snapshot.eval_success or snapshot.check_success))

    def _latest_snapshot_task_success(self) -> bool:
        snapshot = self.latest_snapshot
        return bool(
            self._authoritative_environment_success
            or (snapshot is not None and snapshot.eval_success)
        )

    def _defer_pure_tool_control_control_error(
        self,
        *,
        stage: str,
        signal: ControlSignal,
        error: Exception,
    ) -> bool:
        if not self._pure_tool_control_enabled():
            return False
        self._pure_tool_control_control_backend_errors += 1
        error_text = f"{type(error).__name__}: {error}"
        backend_error_budget = self._pure_tool_control_backend_error_budget()
        if self._pure_tool_control_control_backend_errors >= backend_error_budget:
            reason = f"pure_tool_control_control_backend_unavailable:{backend_error_budget}"
            self._set_pure_tool_control_terminal_failure(
                reason=reason,
                note="pure tool-control planner backend error budget exhausted",
                event="pure_tool_control_control_backend_unavailable",
                stage=stage,
                trigger=signal.value,
                error=error_text,
                backend_error_count=self._pure_tool_control_control_backend_errors,
                backend_error_budget=backend_error_budget,
            )
            return True
        note = f"pure tool-control control planner unavailable during {stage}; retrying without ending episode"
        self.memory_store.record_recovery(
            f"pure_tool_control_control_turn_error:{stage}:{self._compact_summary_text(error_text, 240)}"
        )
        self.memory_store.set_last_error(error_text)
        self.memory_store.set_monitor_status(
            phase="reasoning",
            status="needs_reasoning",
            note=note,
            env_signal=RUNNING,
            failure_reason=error_text,
            progress_score=0.0,
        )
        self._debug_recovery_planner_bootstrapped = False
        self._trace(
            "control_turn_error",
            stage=stage,
            trigger=signal.value,
            error=error_text,
            retryable=True,
            task_finished=False,
            backend_error_count=self._pure_tool_control_control_backend_errors,
            backend_error_budget=backend_error_budget,
            robot_state=self._get_robot_state(),
        )
        self._dump_rollout_event(
            "control_turn_error",
            stage=stage,
            trigger=signal.value,
            error=error_text,
            retryable=True,
            task_finished=False,
            backend_error_count=self._pure_tool_control_control_backend_errors,
            backend_error_budget=backend_error_budget,
        )
        return True

    def _set_pure_tool_control_terminal_failure(
        self,
        *,
        reason: str,
        note: str,
        event: str,
        **payload: Any,
    ) -> None:
        if self._pure_tool_control_terminal_failure:
            return
        self._pure_tool_control_terminal_failure = reason
        self._pending_action_effect_verification = None
        self.memory_store.record_recovery(reason)
        self.memory_store.set_monitor_status(
            phase="reasoning",
            status="rollout_failed",
            note=note,
            env_signal=RUNNING,
            failure_reason=reason,
            progress_score=0.0,
        )
        event_payload = {"reason": reason, "eval_success": False, **payload}
        self._trace(event, **event_payload)
        self._dump_rollout_event(event, **event_payload)

    def _repair_unvalidated_pure_tool_control_finish(self) -> bool:
        if not self._pure_tool_control_enabled():
            return False
        if not self.memory_store.state.task.task_finished:
            return False
        if self._latest_snapshot_task_success():
            return False
        note = "pure tool-control task finish rejected because environment success is false"
        self.memory_store.record_recovery("pure_tool_control_unvalidated_finish_reopened")
        self.memory_store.reopen_task(note=note)
        self._debug_recovery_triggered = False
        self._debug_recovery_rounds = 0
        self._debug_recovery_scene_wait_turns = 0
        self._debug_recovery_planner_bootstrapped = False
        self._trace(
            "pure_tool_control_finish_reopened",
            reason=note,
            eval_success=False,
            check_success=False if self.latest_snapshot is None else bool(self.latest_snapshot.check_success),
        )
        return True

    def _forced_recovery_enabled(self) -> bool:
        return bool(getattr(self.config, "debug_recovery_enabled", False)) or self._pure_tool_control_enabled()

    def _forced_recovery_pure_control(self) -> bool:
        return bool(getattr(self.config, "debug_recovery_pure_control", False)) or self._pure_tool_control_enabled()

    def _forced_recovery_trigger_step(self) -> int:
        if self._pure_tool_control_enabled():
            return max(0, int(getattr(self.config, "pure_tool_control_trigger_step", 0)))
        return max(0, int(getattr(self.config, "debug_recovery_trigger_step", 0)))

    def _forced_recovery_wait_for_scene_memory(self) -> bool:
        if self._pure_tool_control_enabled():
            return bool(getattr(self.config, "pure_tool_control_wait_for_scene_memory", True))
        return bool(getattr(self.config, "debug_recovery_wait_for_scene_memory", True))

    def _forced_recovery_max_wait_steps(self) -> int:
        if self._pure_tool_control_enabled():
            return max(0, int(getattr(self.config, "pure_tool_control_max_wait_steps", 10)))
        return max(0, int(getattr(self.config, "debug_recovery_max_wait_steps", 10)))

    def _forced_recovery_retry_budget(self) -> int:
        if self._pure_tool_control_enabled():
            return max(0, int(getattr(self.config, "pure_tool_control_retry_budget", 1)))
        return max(0, int(getattr(self.config, "debug_recovery_retry_budget", 1)))

    def _forced_recovery_max_rounds(self) -> int:
        if self._pure_tool_control_enabled():
            return max(1, int(getattr(self.config, "pure_tool_control_max_rounds", 10)))
        return max(1, int(getattr(self.config, "debug_recovery_max_rounds", 100)))

    def _forced_recovery_repeat_signal(self) -> bool:
        if self._pure_tool_control_enabled():
            return bool(getattr(self.config, "pure_tool_control_repeat_signal", True))
        return bool(getattr(self.config, "debug_recovery_repeat_signal", False))

    def _forced_recovery_bootstrap_with_planner(self) -> bool:
        if self._pure_tool_control_enabled():
            return bool(getattr(self.config, "pure_tool_control_bootstrap_with_planner", True))
        return bool(getattr(self.config, "debug_recovery_bootstrap_with_planner", False))

    def _forced_recovery_skip_vla_rollout(self) -> bool:
        if self._pure_tool_control_enabled():
            return bool(getattr(self.config, "pure_tool_control_skip_vla_rollout", True))
        return bool(getattr(self.config, "debug_recovery_skip_vla_rollout", False))

    def _maybe_bootstrap_debug_recovery_with_planner(self) -> bool:
        if not self._forced_recovery_enabled():
            return False
        if not self._forced_recovery_bootstrap_with_planner():
            return False
        if self.latest_snapshot is None:
            return False
        if self.memory_store.state.task.task_finished:
            return False
        if self.memory_store.state.active_skill is not None:
            return False
        if self._debug_recovery_planner_bootstrapped:
            return False
        trigger_step = self._forced_recovery_trigger_step()
        current_step = int(getattr(self.latest_snapshot, "step_count", 0))
        if current_step < trigger_step:
            return False

        if self._stop_for_pure_tool_control_turn_budget():
            return True

        _, signal = self.needs_control_turn()
        try:
            prediction = self.run_control_turn(signal)
        except Exception as exc:
            if self._defer_pure_tool_control_control_error(stage="planner_bootstrap", signal=signal, error=exc):
                return True
            raise
        self.apply_control_decision(prediction)
        self._debug_recovery_planner_bootstrapped = True
        active = self.memory_store.state.active_skill
        self._trace(
            "debug_recovery_planner_bootstrap",
            control_mode="pure_tool_control" if self._pure_tool_control_enabled() else "debug_recovery",
            trigger=signal.value,
            action_mode=str(prediction.get("action_mode", "")),
            selected_skill=str(prediction.get("selected_skill", "")),
            subtask_text=str(prediction.get("subtask_text", "")),
            active_skill=None if active is None else active.skill_name,
            active_subtask="" if active is None else active.instruction,
        )
        return True

    def _skip_vla_rollout_for_forced_recovery(self) -> bool:
        if not self._forced_recovery_enabled():
            return False
        if not self._forced_recovery_skip_vla_rollout():
            return False
        if not self._pure_tool_control_enabled() and not self._debug_recovery_triggered:
            return False
        active = self.memory_store.state.active_skill
        self.memory_store.set_monitor_status(
            phase="monitoring",
            status="rollout_active",
            note="VLA rollout skipped for pure agent tool-control evaluation",
            env_signal=RUNNING,
            failure_reason="",
        )
        self._trace(
            "vla_rollout_skipped",
            reason="forced_recovery_skip_vla_rollout",
            control_mode="pure_tool_control" if self._pure_tool_control_enabled() else "debug_recovery",
            active_skill=None if active is None else active.skill_name,
            active_subtask="" if active is None else active.instruction,
        )
        return True

    def _maybe_force_debug_recovery(self, task_env: Any) -> bool:
        pure_control = self._forced_recovery_pure_control()
        if self._pure_tool_control_terminal_failure:
            return True
        if self._debug_recovery_triggered and not pure_control:
            return False
        if not self._forced_recovery_enabled():
            return False
        if self.latest_snapshot is None:
            return False
        if self.memory_store.state.task.task_finished:
            return False
        if self._pure_tool_control_enabled():
            if self._latest_snapshot_task_success():
                return False
        elif bool(getattr(self.latest_snapshot, "eval_success", False)) or bool(getattr(self.latest_snapshot, "check_success", False)):
            return False
        if pure_control:
            max_rounds = self._forced_recovery_max_rounds()
            if self._debug_recovery_rounds >= max_rounds:
                if self._pure_tool_control_enabled():
                    self._replan_after_pure_tool_control_round_limit(max_rounds=max_rounds)
                else:
                    self._finish_debug_recovery_pure_control_limit(max_rounds=max_rounds)
                return True
        trigger_step = self._forced_recovery_trigger_step()
        current_step = int(getattr(self.latest_snapshot, "step_count", 0))
        if current_step < trigger_step:
            return False
        if self._forced_recovery_wait_for_scene_memory() and not self._debug_recovery_scene_ready():
            max_wait_steps = self._forced_recovery_max_wait_steps()
            if pure_control:
                binding_errors = self._required_identity_binding_errors()
                if (
                    self._pure_tool_control_enabled()
                    and binding_errors
                    and self._can_retry_identity_binding_query()
                ):
                    active_skill_id = self._active_skill_id()
                    if (
                        active_skill_id
                        != self._identity_binding_retry_skill_id
                    ):
                        self._identity_binding_retry_skill_id = (
                            active_skill_id
                        )
                        self._identity_binding_retry_attempts = 0
                    if (
                        self._identity_binding_retry_attempts
                        < max_wait_steps
                    ):
                        self._identity_binding_retry_attempts += 1
                        retry_attempt = (
                            self._identity_binding_retry_attempts
                        )
                        self._retry_identity_binding_query(
                            binding_errors=binding_errors,
                            retry_attempt=retry_attempt,
                            max_retries=max_wait_steps,
                        )
                        if self._debug_recovery_scene_ready():
                            self._trace(
                                "perception_binding_retry_recovered",
                                retry_attempt=retry_attempt,
                                max_retries=max_wait_steps,
                            )
                            self._debug_recovery_scene_wait_turns = 0
                            self._identity_binding_retry_attempts = 0
                            self._identity_binding_retry_skill_id = ""
                        else:
                            remaining_errors = (
                                self._required_identity_binding_errors()
                                or binding_errors
                            )
                            if retry_attempt >= max_wait_steps:
                                self._trace(
                                    "debug_recovery_wait_timeout",
                                    reason="identity_binding_retries_exhausted",
                                    trigger_step=trigger_step,
                                    current_step=current_step,
                                    max_wait_steps=max_wait_steps,
                                    wait_turn=retry_attempt,
                                    binding_errors=remaining_errors,
                                )
                                self._identity_binding_retry_attempts = 0
                                self._identity_binding_retry_skill_id = ""
                                self._fail_unresolved_identity_binding(
                                    remaining_errors
                                )
                                return True
                            self._trace(
                                "debug_recovery_wait",
                                reason="identity_binding_retry_pending",
                                trigger_step=trigger_step,
                                current_step=current_step,
                                max_wait_steps=max_wait_steps,
                                wait_turn=retry_attempt,
                                binding_errors=remaining_errors,
                            )
                            return True
                    else:
                        self._trace(
                            "debug_recovery_wait_timeout",
                            reason="identity_binding_retries_exhausted",
                            trigger_step=trigger_step,
                            current_step=current_step,
                            max_wait_steps=max_wait_steps,
                            wait_turn=self._identity_binding_retry_attempts,
                            binding_errors=binding_errors,
                        )
                        self._identity_binding_retry_attempts = 0
                        self._identity_binding_retry_skill_id = ""
                        self._fail_unresolved_identity_binding(
                            binding_errors
                        )
                        return True
                elif binding_errors and max_wait_steps == 0:
                    self._trace(
                        "debug_recovery_wait_timeout",
                        reason="identity_binding_retry_budget_zero",
                        trigger_step=trigger_step,
                        current_step=current_step,
                        max_wait_steps=max_wait_steps,
                        wait_turn=0,
                        binding_errors=binding_errors,
                    )
                    self._fail_unresolved_identity_binding(binding_errors)
                    return True
                elif self._debug_recovery_scene_wait_turns < max_wait_steps:
                    self._debug_recovery_scene_wait_turns += 1
                    self._trace(
                        "debug_recovery_wait",
                        reason="scene_memory_not_ready",
                        trigger_step=trigger_step,
                        current_step=current_step,
                        max_wait_steps=max_wait_steps,
                        wait_turn=self._debug_recovery_scene_wait_turns,
                    )
                    return True
                else:
                    self._trace(
                        "debug_recovery_wait_timeout",
                        reason="scene_memory_not_ready",
                        trigger_step=trigger_step,
                        current_step=current_step,
                        max_wait_steps=max_wait_steps,
                        wait_turn=self._debug_recovery_scene_wait_turns,
                    )
                    self._debug_recovery_scene_wait_turns = 0
                    if self._pure_tool_control_enabled() and binding_errors:
                        self._fail_unresolved_identity_binding(binding_errors)
                        return True
            else:
                if current_step < trigger_step + max_wait_steps:
                    self._trace(
                        "debug_recovery_wait",
                        reason="scene_memory_not_ready",
                        trigger_step=trigger_step,
                        current_step=current_step,
                        max_wait_steps=max_wait_steps,
                    )
                    return False
                self._trace(
                    "debug_recovery_wait_timeout",
                    reason="scene_memory_not_ready",
                    trigger_step=trigger_step,
                    current_step=current_step,
                    max_wait_steps=max_wait_steps,
                )
        else:
            self._debug_recovery_scene_wait_turns = 0
            self._identity_binding_retry_attempts = 0
            self._identity_binding_retry_skill_id = ""
        self._debug_recovery_triggered = True
        self._debug_recovery_rounds += 1
        if not self._ensure_debug_recovery_active_skill():
            reason = "planner bootstrap did not create an active skill for forced recovery"
            self.memory_store.record_recovery("forced_recovery_no_active_skill")
            self.memory_store.set_monitor_status(
                phase="reasoning",
                status="needs_reasoning",
                note=reason,
                env_signal=RUNNING,
                failure_reason=reason,
                progress_score=0.0,
            )
            self._trace(
                "debug_recovery_no_active_skill",
                control_mode="pure_tool_control" if self._pure_tool_control_enabled() else "debug_recovery",
                round_index=self._debug_recovery_rounds,
                reason=reason,
                robot_state=self._get_robot_state(),
            )
            self._debug_recovery_planner_bootstrapped = False
            return True
        self._apply_debug_recovery_budget()
        signal_name, reason = self._debug_recovery_signal_for_round(pure_control=pure_control)
        signal = self._make_monitor_signal(
            name=signal_name,
            level="warning",
            reason=reason,
            score=1.0,
            details={"ood_scenario": signal_name, "source": "debug_recovery"},
        )
        self.memory_store.fail_rollout(
            status="rollout_stalled",
            reason=reason,
            env_signal=signal.name,
            progress_score=signal.score,
        )
        debug_scene_trace = (
            {
                "planner_scene_view": self._planner_scene_view(),
                "planner_context_mode": COMPACT_PLANNER_CONTEXT_MODE,
            }
            if self._observation_trace_payload_mode()
            == COMPACT_PLANNER_CONTEXT_MODE
            else {
                "scene_memory": self.memory_store.state.working.scene_memory,
                "planner_context_mode": "legacy",
            }
        )
        self._trace(
            "debug_recovery_trigger",
            round_index=self._debug_recovery_rounds,
            signal_name=signal.name,
            reason=reason,
            active_skill=None if self.memory_store.state.active_skill is None else self.memory_store.state.active_skill.skill_name,
            robot_state=self._get_robot_state(),
            **debug_scene_trace,
        )
        preferred_action = self._dispatch_recovery_tools(task_env, signal)
        if pure_control:
            if self._pure_tool_control_enabled() and preferred_action == _RECOVERY_BACKEND_ERROR:
                self._debug_recovery_rounds = max(0, self._debug_recovery_rounds - 1)
                self._handle_pure_tool_control_recovery_backend_error(signal=signal)
                return True
            self._continue_debug_recovery_pure_control(
                preferred_action=preferred_action,
                signal=signal,
                reason=reason,
            )
            return True
        self.memory_store.set_recovery_policy(
            action=self._select_recovery_action(preferred_action=preferred_action, signal_name=signal.name),
            reason=reason,
        )
        return True

    def _continue_debug_recovery_pure_control(self, *, preferred_action: str, signal: MonitorSignal, reason: str) -> None:
        control_mode = "pure_tool_control" if self._pure_tool_control_enabled() else "debug_pure_control"
        round_index = self._debug_recovery_rounds
        if self._commit_pure_tool_control_environment_success(
            source="recovery_batch",
            round_index=round_index,
        ):
            return
        if (
            self._pure_tool_control_enabled()
            and preferred_action == _RECOVERY_INTERNAL_PROGRESS
        ):
            previous_empty_plan_turns = (
                self._pure_tool_control_empty_plan_turns
            )
            previous_no_progress_turns = (
                self._pure_tool_control_no_progress_control_turns
            )
            self._pure_tool_control_empty_plan_turns = 0
            completion = str(
                self._last_evidence_acquisition_decision.get(
                    "controlled_return_zero_call_reason",
                    "internal_recovery_state_transition_completed",
                )
                or "internal_recovery_state_transition_completed"
            )
            note = (
                "runtime completed a verified internal recovery state "
                "transition; continue the active subtask"
            )
            self.memory_store.record_recovery(
                f"pure_tool_control_internal_recovery_progress:{completion}"
            )
            self.memory_store.set_monitor_status(
                phase="monitoring",
                status="rollout_active",
                note=note,
                env_signal=signal.name,
                failure_reason="",
                progress_score=0.0,
            )
            payload = {
                "authority": "runtime_internal_recovery_reducer",
                "completion": completion,
                "round_index": round_index,
                "previous_empty_plan_count": previous_empty_plan_turns,
                "empty_plan_count": 0,
                "previous_no_progress_control_turn_count": (
                    previous_no_progress_turns
                ),
                "no_progress_control_turn_count": previous_no_progress_turns,
                "next_action": "continue_active_subtask",
            }
            self._trace(
                "pure_tool_control_internal_recovery_progress",
                **payload,
            )
            self._dump_rollout_event(
                "pure_tool_control_internal_recovery_progress",
                **payload,
            )
            return
        if self._pure_tool_control_enabled() and preferred_action == _RECOVERY_EMPTY_PLAN:
            self._pure_tool_control_empty_plan_turns += 1
            threshold = self._pure_tool_control_empty_plan_replan_threshold()
            self.memory_store.record_recovery(
                f"pure_tool_control_empty_recovery_plan:{self._pure_tool_control_empty_plan_turns}/{threshold}"
            )
            if self._pure_tool_control_empty_plan_turns >= threshold:
                self._replan_after_pure_tool_control_empty_plans(
                    threshold=threshold,
                    round_index=round_index,
                )
                return
            note = (
                "valid recovery plan contained no task-progress tools; preserve active subtask and request "
                f"another grounded plan ({self._pure_tool_control_empty_plan_turns}/{threshold})"
            )
            self.memory_store.set_monitor_status(
                phase="monitoring",
                status="rollout_active",
                note=note,
                env_signal=signal.name,
                failure_reason="",
                progress_score=0.0,
            )
            self._trace(
                "pure_tool_control_empty_recovery_plan",
                round_index=round_index,
                empty_plan_count=self._pure_tool_control_empty_plan_turns,
                threshold=threshold,
            )
            self._dump_rollout_event(
                "pure_tool_control_empty_recovery_plan",
                round_index=round_index,
                empty_plan_count=self._pure_tool_control_empty_plan_turns,
                threshold=threshold,
            )
            return
        self._pure_tool_control_empty_plan_turns = 0
        note = f"{control_mode} round {self._debug_recovery_rounds} finished; resolved_control={preferred_action or 'none'}; reason={reason}"
        self.memory_store.record_recovery(f"{control_mode}_round:{self._debug_recovery_rounds}")
        if self._pure_tool_control_enabled() and preferred_action == "subtask_complete":
            completion_note = "after-action verifier confirmed current subtask completion"
            active = self.memory_store.state.active_skill
            completed_subtask = "" if active is None else active.instruction
            self.memory_store.record_recovery("pure_tool_control_subtask_completed")
            self.memory_store.mark_active_skill_succeeded(note=completion_note)
            self._debug_recovery_triggered = False
            self._debug_recovery_planner_bootstrapped = False
            self._debug_recovery_rounds = 0
            self._debug_recovery_scene_wait_turns = 0
            self._pure_tool_control_empty_plan_turns = 0
            self.memory_store.set_monitor_status(
                phase="reasoning",
                status="needs_reasoning",
                note="verified subtask completion committed; request next subtask from planner",
                env_signal=RUNNING,
                failure_reason="",
                progress_score=1.0,
            )
            transition_payload = {
                "authority": "after_action_agent_api_verifier",
                "subtask_status": "completed",
                "completed_subtask": completed_subtask,
                "next_action": "plan_next_subtask",
                "round_index": round_index,
            }
            self._trace("subtask_transition_commit", **transition_payload)
            self._dump_rollout_event("subtask_transition_commit", **transition_payload)
            return
        if self._pure_tool_control_enabled() and preferred_action == "replan":
            self.memory_store.record_recovery("pure_tool_control_replan_requested")
            failure_note = "after-action verifier classified current subtask as failed; request planner replan"
            active = self.memory_store.state.active_skill
            failed_subtask = "" if active is None else active.instruction
            self.memory_store.mark_active_skill_failed(note=failure_note)
            self._debug_recovery_triggered = False
            self._debug_recovery_planner_bootstrapped = False
            self._debug_recovery_rounds = 0
            self._debug_recovery_scene_wait_turns = 0
            self._pure_tool_control_empty_plan_turns = 0
            self.memory_store.set_monitor_status(
                phase="reasoning",
                status="needs_reasoning",
                note=failure_note,
                env_signal=RUNNING,
                failure_reason=failure_note,
                progress_score=0.0,
            )
            transition_payload = {
                "authority": "after_action_agent_api_verifier",
                "subtask_status": "failed",
                "failed_subtask": failed_subtask,
                "next_action": "replan_subtask",
                "round_index": round_index,
            }
            self._trace("subtask_transition_commit", **transition_payload)
            self._dump_rollout_event("subtask_transition_commit", **transition_payload)
            self._trace(
                "debug_recovery_pure_control_round",
                control_mode=control_mode,
                round_index=round_index,
                preferred_action=preferred_action,
                signal_name=signal.name,
                reason=reason,
                robot_state=self._get_robot_state(),
            )
            self._dump_rollout_event(
                "debug_recovery_pure_control_round",
                control_mode=control_mode,
                round_index=round_index,
                preferred_action=preferred_action,
                signal_name=signal.name,
                reason=reason,
                robot_state=self._get_robot_state(),
            )
            return
        elif self._pure_tool_control_enabled() and preferred_action == "abort":
            self.memory_store.record_recovery("pure_tool_control_unexpected_abort_deferred")
            note = (
                "unexpected abort control deferred; preserve active subtask until after-action verification; "
                f"reason={reason}"
            )
        self.memory_store.set_monitor_status(
            phase="monitoring",
            status="rollout_active",
            note=note,
            env_signal=signal.name,
            failure_reason=reason,
            progress_score=signal.score,
        )
        self._trace(
            "debug_recovery_pure_control_round",
            control_mode=control_mode,
            round_index=self._debug_recovery_rounds,
            preferred_action=preferred_action,
            signal_name=signal.name,
            reason=reason,
            robot_state=self._get_robot_state(),
        )
        self._dump_rollout_event(
            "debug_recovery_pure_control_round",
            control_mode=control_mode,
            round_index=self._debug_recovery_rounds,
            preferred_action=preferred_action,
            signal_name=signal.name,
            reason=reason,
            robot_state=self._get_robot_state(),
        )

    def _commit_pure_tool_control_environment_success(
        self,
        *,
        source: str,
        round_index: int | None = None,
    ) -> bool:
        if not self._pure_tool_control_enabled() or not self._latest_snapshot_task_success():
            return False

        state = self.memory_store.state
        state.recovery.pending_action = ""
        state.recovery.pending_reason = ""
        self.memory_store.set_last_error("")
        self._pure_tool_control_terminal_failure = ""
        self._debug_recovery_triggered = False
        self._debug_recovery_planner_bootstrapped = False
        self._debug_recovery_rounds = 0
        self._debug_recovery_scene_wait_turns = 0
        self._pure_tool_control_empty_plan_turns = 0
        self._pure_tool_control_no_progress_control_turns = 0
        self._pending_action_effect_verification = None
        self._clear_recovery_backend_error_streak(stage="environment_success_commit")
        if state.task.task_finished:
            self.memory_store.set_monitor_status(
                phase="finished",
                status="finished",
                note="authoritative RMBench environment success already committed",
                env_signal=TASK_SUCCESS,
                failure_reason="",
                progress_score=1.0,
            )
            return True

        active = state.active_skill
        completed_subtask = "" if active is None else active.instruction
        note = "RMBench eval_success authoritatively completed the global task"
        self.memory_store.record_recovery("pure_tool_control_environment_success")
        if active is not None:
            self.memory_store.mark_active_skill_succeeded(note=note)
        self.memory_store.mark_task_finished(note=note)
        self.memory_store.set_monitor_status(
            phase="finished",
            status="finished",
            note=note,
            env_signal=TASK_SUCCESS,
            failure_reason="",
            progress_score=1.0,
        )
        payload = {
            "authority": "environment_eval_success",
            "source": source,
            "global_task_success": True,
            "subtask_status": "completed",
            "completed_subtask": completed_subtask,
            "next_action": "stop_global_task",
            "round_index": round_index,
        }
        self._trace("global_task_success_commit", **payload)
        self._dump_rollout_event("global_task_success_commit", **payload)
        if completed_subtask:
            self._trace("subtask_transition_commit", **payload)
            self._dump_rollout_event("subtask_transition_commit", **payload)
        return True

    def _record_recovery_backend_error(self, *, stage: str, error: Exception | str) -> None:
        if isinstance(error, Exception):
            error_text = f"{type(error).__name__}: {error}"
        else:
            error_text = str(error)
        self._last_recovery_backend_error = self._compact_summary_text(error_text, 240)
        self._last_recovery_backend_error_stage = str(stage or "recovery_backend")

    def _clear_recovery_backend_error_streak(self, *, stage: str) -> None:
        if not self._pure_tool_control_enabled():
            return
        previous_error_count = self._pure_tool_control_recovery_backend_errors
        self._pure_tool_control_recovery_backend_errors = 0
        self._last_recovery_backend_error = ""
        self._last_recovery_backend_error_stage = ""
        if previous_error_count <= 0:
            return
        payload = {
            "stage": stage,
            "previous_backend_error_count": previous_error_count,
            "backend_error_count": 0,
            "backend_error_budget": self._pure_tool_control_backend_error_budget(),
        }
        self._trace("pure_tool_control_recovery_backend_recovered", **payload)
        self._dump_rollout_event("pure_tool_control_recovery_backend_recovered", **payload)

    def _handle_pure_tool_control_recovery_backend_error(self, *, signal: MonitorSignal) -> None:
        self._pure_tool_control_recovery_backend_errors += 1
        backend_error_budget = self._pure_tool_control_backend_error_budget()
        backend_stage = self._last_recovery_backend_error_stage or "recovery_route"
        backend_error = (
            self._last_recovery_backend_error
            or self._recovery_policy_resolver.last_error
            or "recovery backend unavailable"
        )
        compact_error = self._compact_summary_text(backend_error, 240)
        if self._pure_tool_control_recovery_backend_errors >= backend_error_budget:
            reason = f"pure_tool_control_recovery_backend_unavailable:{backend_error_budget}"
            self._set_pure_tool_control_terminal_failure(
                reason=reason,
                note="pure tool-control recovery backend error budget exhausted",
                event="pure_tool_control_recovery_backend_unavailable",
                signal_name=signal.name,
                backend_stage=backend_stage,
                backend_error=compact_error,
                backend_error_count=self._pure_tool_control_recovery_backend_errors,
                backend_error_budget=backend_error_budget,
                semantic_round_count=self._debug_recovery_rounds,
            )
            return
        note = (
            "recovery backend unavailable; retrying without consuming the active subtask's semantic "
            f"MAX_ROUNDS budget ({self._pure_tool_control_recovery_backend_errors}/{backend_error_budget})"
        )
        self.memory_store.set_monitor_status(
            phase="recovery",
            status="waiting_backend_retry",
            note=note,
            env_signal=signal.name,
            failure_reason=compact_error,
            progress_score=0.0,
        )
        payload = {
            "signal_name": signal.name,
            "backend_stage": backend_stage,
            "backend_error": compact_error,
            "backend_error_count": self._pure_tool_control_recovery_backend_errors,
            "backend_error_budget": backend_error_budget,
            "semantic_round_count": self._debug_recovery_rounds,
        }
        self._trace("pure_tool_control_recovery_backend_retry", **payload)
        self._dump_rollout_event("pure_tool_control_recovery_backend_retry", **payload)

    def _replan_after_pure_tool_control_empty_plans(self, *, threshold: int, round_index: int) -> None:
        note = (
            "recovery planner returned consecutive valid empty plans; runtime requested a new subtask "
            f"after {threshold} empty plans"
        )
        active = self.memory_store.state.active_skill
        failed_subtask = "" if active is None else active.instruction
        self.memory_store.record_recovery("pure_tool_control_empty_plan_replan")
        self.memory_store.mark_active_skill_failed(note=note)
        self._debug_recovery_triggered = False
        self._debug_recovery_planner_bootstrapped = False
        self._debug_recovery_rounds = 0
        self._debug_recovery_scene_wait_turns = 0
        self._pure_tool_control_empty_plan_turns = 0
        self.memory_store.set_monitor_status(
            phase="reasoning",
            status="needs_reasoning",
            note=note,
            env_signal=RUNNING,
            failure_reason=note,
            progress_score=0.0,
        )
        payload = {
            "authority": "runtime_empty_plan_guard",
            "subtask_status": "failed",
            "failed_subtask": failed_subtask,
            "next_action": "replan_subtask",
            "round_index": round_index,
            "empty_plan_threshold": threshold,
        }
        self._trace("subtask_transition_commit", **payload)
        self._dump_rollout_event("subtask_transition_commit", **payload)

    def _debug_recovery_signal_for_round(self, *, pure_control: bool) -> tuple[str, str]:
        if self._pure_tool_control_enabled():
            forced_signal = self._normalize_debug_recovery_signal(
                str(getattr(self.config, "pure_tool_control_signal", TASK_LEVEL_RECOVERY_CONTROL))
            )
            forced_reason = str(getattr(self.config, "pure_tool_control_reason", "pure tool-control ablation")).strip() or "pure tool-control ablation"
        else:
            forced_signal = self._normalize_debug_recovery_signal(str(getattr(self.config, "debug_recovery_signal", "motion_blocked")))
            forced_reason = str(getattr(self.config, "debug_recovery_reason", "debug forced recovery loop")).strip() or "debug forced recovery loop"
        repeat_signal = self._forced_recovery_repeat_signal()
        if not pure_control or repeat_signal or self._debug_recovery_rounds <= 1:
            return forced_signal, forced_reason
        history_tail = list(self.memory_store.state.working.recovery_history[-6:])
        scene_focus = dict((self.memory_store.state.working.scene_memory or {}).get("task_focus", {}) or {})
        reason_parts = [
            f"debug pure recovery task-level control round {self._debug_recovery_rounds}",
            "continue the global instruction using grounded scene memory and robot state",
            "do not repeat clearance motions unless current observation provides new blockage evidence",
            f"initial_debug_signal={forced_signal}",
        ]
        if scene_focus:
            reason_parts.append(f"scene_focus={scene_focus}")
        if history_tail:
            reason_parts.append(f"recent_recovery_history={history_tail}")
        return TASK_LEVEL_RECOVERY_CONTROL, "; ".join(reason_parts)

    def _finish_debug_recovery_pure_control_limit(self, *, max_rounds: int) -> None:
        note = f"debug pure recovery max_rounds reached: {max_rounds}"
        self.memory_store.record_recovery("debug_pure_control_max_rounds")
        self.memory_store.mark_task_finished(note=note)
        self.memory_store.set_monitor_status(
            phase="finished",
            status="finished",
            note=note,
            env_signal="debug_pure_recovery_max_rounds",
            failure_reason=note,
            progress_score=0.0,
        )
        self._trace("debug_recovery_pure_control_stop", reason=note, max_rounds=max_rounds, robot_state=self._get_robot_state())

    def _replan_after_pure_tool_control_round_limit(self, *, max_rounds: int) -> None:
        note = (
            f"pure tool-control max_rounds reached before after-action verifier confirmed completion: "
            f"{max_rounds}; forcing planner replan"
        )
        active = self.memory_store.state.active_skill
        failed_subtask = "" if active is None else active.instruction
        self.memory_store.record_recovery("pure_tool_control_max_rounds_replan")
        self.memory_store.record_recovery("pure_tool_control_subtask_budget_exhausted")
        self.memory_store.mark_active_skill_failed(note=note)
        self._debug_recovery_rounds = 0
        self._debug_recovery_scene_wait_turns = 0
        self._debug_recovery_triggered = False
        self._debug_recovery_planner_bootstrapped = False
        self._pure_tool_control_empty_plan_turns = 0
        self.memory_store.set_monitor_status(
            phase="reasoning",
            status="needs_reasoning",
            note=note,
            env_signal=RUNNING,
            failure_reason=note,
            progress_score=0.0,
        )
        self._trace(
            "pure_tool_control_round_limit_replan",
            reason=note,
            max_rounds=max_rounds,
            robot_state=self._get_robot_state(),
        )
        self._dump_rollout_event(
            "pure_tool_control_round_limit_replan",
            reason=note,
            max_rounds=max_rounds,
            robot_state=self._get_robot_state(),
        )
        transition_payload = {
            "authority": "runtime_budget_guard",
            "subtask_status": "failed",
            "failed_subtask": failed_subtask,
            "next_action": "replan_subtask",
            "max_rounds": max_rounds,
        }
        self._trace("subtask_transition_commit", **transition_payload)
        self._dump_rollout_event("subtask_transition_commit", **transition_payload)

    def _ensure_debug_recovery_active_skill(self) -> bool:
        if self.memory_store.state.active_skill is not None:
            return True
        if self._pure_tool_control_enabled():
            self._trace(
                "debug_recovery_active_skill_missing",
                reason="pure tool-control requires a planner-created active skill",
                control_mode="pure_tool_control",
            )
            return False
        subtask = str(getattr(self.config, "debug_recovery_subtask", "")).strip()
        if not subtask and self._forced_recovery_bootstrap_with_planner():
            self._trace(
                "debug_recovery_active_skill_missing",
                reason="planner bootstrap did not create an active skill",
                control_mode="pure_tool_control" if self._pure_tool_control_enabled() else "debug_recovery",
            )
            return False
        if not subtask:
            subtask = self.current_instruction.strip() or "debug recovery subtask"
        skill_name = str(getattr(self.config, "debug_recovery_skill", "monitored-subtask-execution")).strip() or "monitored-subtask-execution"
        skill_spec = self._resolve_skill_spec(skill_name)
        self._start_or_replace_scoped_skill(subtask_text=subtask, skill_spec=skill_spec)
        self.segment_start_snapshot = self.latest_snapshot
        return True

    def _apply_debug_recovery_budget(self) -> None:
        retry_budget = self._forced_recovery_retry_budget()
        if retry_budget > self.memory_store.state.recovery.retry_budget:
            self.memory_store.state.recovery.retry_budget = retry_budget

    def _can_retry_identity_binding_query(self) -> bool:
        return bool(
            self.latest_snapshot is not None
            and getattr(
                self.config,
                "observation_preprocess_enabled",
                False,
            )
            and getattr(
                self.config,
                "observation_preprocess_auto_objects",
                True,
            )
            and str(
                getattr(
                    self.config,
                    "observation_preprocess_query_url",
                    "",
                )
                or ""
            ).strip()
        )

    def _previous_perception_queries_for_retry(
        self,
    ) -> list[dict[str, Any]]:
        preprocess = self.memory_store.state.working.observation_preprocess
        if not isinstance(preprocess, dict):
            return []
        max_objects = max(
            1,
            int(
                getattr(
                    self.config,
                    "observation_preprocess_max_objects",
                    3,
                )
            ),
        )
        return self._normalize_perception_queries(
            preprocess.get("perception_queries", []),
            max_objects=max_objects,
        )

    def _retry_identity_binding_query(
        self,
        *,
        binding_errors: list[str],
        retry_attempt: int,
        max_retries: int,
    ) -> bool:
        if not self._can_retry_identity_binding_query():
            return False
        snapshot = self.latest_snapshot
        if snapshot is None:
            return False
        retry_context = {
            "failure_reason": "|".join(
                self._compact_summary_text(item, 160)
                for item in binding_errors[:4]
            ),
            "retry_attempt": int(retry_attempt),
            "force_refresh": True,
            "previous_queries": self._previous_perception_queries_for_retry(),
        }
        payload = {
            "retry_attempt": int(retry_attempt),
            "max_retries": int(max_retries),
            "binding_errors": list(binding_errors),
            "previous_queries": list(
                retry_context["previous_queries"]
            ),
            "env_step": int(getattr(snapshot, "step_count", 0)),
        }
        self._trace("perception_binding_retry_start", **payload)
        self._dump_rollout_event(
            "perception_binding_retry_start",
            **payload,
        )
        error = ""
        try:
            self.update_snapshot(
                snapshot,
                force_preprocess=True,
                perception_binding_requirement=retry_context,
            )
        except Exception as exc:
            error = repr(exc)
            self._trace(
                "perception_binding_retry_error",
                retry_attempt=int(retry_attempt),
                max_retries=int(max_retries),
                error=error,
            )
            self._dump_rollout_event(
                "perception_binding_retry_error",
                retry_attempt=int(retry_attempt),
                max_retries=int(max_retries),
                error=error,
            )
        result_payload = {
            "retry_attempt": int(retry_attempt),
            "max_retries": int(max_retries),
            "scene_ready": self._debug_recovery_scene_ready(),
            "binding_errors": self._required_identity_binding_errors(),
            "error": error,
        }
        self._trace(
            "perception_binding_retry_result",
            **result_payload,
        )
        self._dump_rollout_event(
            "perception_binding_retry_result",
            **result_payload,
        )
        return True

    def _debug_recovery_scene_ready(self) -> bool:
        scene_memory = self.memory_store.state.working.scene_memory
        if not isinstance(scene_memory, dict):
            return False
        instances = scene_memory.get("instances")
        if not isinstance(instances, list) or not instances:
            return False
        focus = scene_memory.get("task_focus")
        if not isinstance(focus, dict):
            return True
        if bool(focus.get("identity_binding_required")):
            binding_roles = {
                str(item).strip().lower()
                for item in focus.get(
                    "identity_binding_roles",
                    ("target", "tool"),
                )
                or ()
            }
            bound_instances = bool(
                focus.get("target_instances")
                or focus.get("tool_instances")
                or (
                    "context" in binding_roles
                    and focus.get("context_instances")
                )
            )
            return bound_instances and not bool(
                focus.get("identity_binding_errors")
            )
        return bool(focus.get("target_instances") or focus.get("tool_instances") or instances)

    def _required_identity_binding_errors(self) -> list[str]:
        scene_memory = self.memory_store.state.working.scene_memory
        if not isinstance(scene_memory, dict):
            return []
        focus = scene_memory.get("task_focus")
        if not isinstance(focus, dict) or not bool(focus.get("identity_binding_required")):
            return []
        return [str(item) for item in focus.get("identity_binding_errors", []) or [] if str(item).strip()]

    def _fail_unresolved_identity_binding(self, binding_errors: list[str]) -> None:
        if self._pure_tool_control_terminal_failure:
            return
        self._debug_recovery_scene_wait_turns = 0
        self._identity_binding_retry_attempts = 0
        self._identity_binding_retry_skill_id = ""
        compact_errors = [self._compact_summary_text(item, 160) for item in binding_errors[:4]]
        reason = "pure_tool_control_identity_binding_unresolved:" + "|".join(compact_errors)
        self._pure_tool_control_terminal_failure = reason
        self.memory_store.record_recovery(reason)
        self.memory_store.set_monitor_status(
            phase="reasoning",
            status="rollout_failed",
            note="Agent-selected instance identity remained unresolved; manipulation was not attempted",
            env_signal=RUNNING,
            failure_reason=reason,
            progress_score=0.0,
        )
        payload = {
            "reason": reason,
            "binding_errors": compact_errors,
            "eval_success": False,
        }
        self._trace("pure_tool_control_identity_binding_failed", **payload)
        self._dump_rollout_event("pure_tool_control_identity_binding_failed", **payload)

    def _normalize_debug_recovery_signal(self, signal_name: str) -> str:
        normalized = str(signal_name or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "object_not_visible": OBJECT_NOT_VISIBLE,
            "motion_blocked": MOTION_BLOCKED,
            "grasp_lost": "grasp_lost",
            "scene_drift_detected": SCENE_DRIFT_DETECTED,
            "requires_replan": "requires_replan",
            "stall_detected": STALL_DETECTED,
            "step_budget_exhausted": STEP_BUDGET_EXHAUSTED,
            "needs_recovery_tools": NEEDS_RECOVERY_TOOLS,
            "task_level_recovery_control": TASK_LEVEL_RECOVERY_CONTROL,
            "task_recovery_control": TASK_LEVEL_RECOVERY_CONTROL,
            "pure_tool_control": TASK_LEVEL_RECOVERY_CONTROL,
        }
        return aliases.get(normalized, MOTION_BLOCKED)

    def _monitor_state_semantic_tags(self, signal: MonitorSignal) -> dict[str, Any]:
        state_tags = {
            "object_state": "dropped" if signal.name == "grasp_lost" else "",
            "visibility_state": "object_not_visible" if signal.name == "object_not_visible" else "",
            "gripper_state": "empty" if signal.name == "grasp_lost" else "",
            "motion_state": "stalled" if signal.name in {STALL_DETECTED, STEP_BUDGET_EXHAUSTED} else "blocked" if signal.name == "motion_blocked" else "",
        }
        return normalize_semantic_tags({"state_tags": state_tags, "tag_source": "unknown"})

    def _active_skill_semantic_tags(self) -> dict[str, Any]:
        active = self.memory_store.state.active_skill
        if active is not None:
            return normalize_semantic_tags(active.semantic_tags)
        return normalize_semantic_tags(self.memory_store.state.working.semantic_tags)

    def _recovery_semantic_tags(self, *, signal: MonitorSignal, current_subtask: str) -> dict[str, Any]:
        ood_tags = normalize_semantic_tags(signal.details.get("semantic_tags"), default_source="ood_vlm")
        active_tags = self._active_skill_semantic_tags()
        monitor_tags = self._monitor_state_semantic_tags(signal)
        has_vlm_or_active_tags = has_semantic_tags(ood_tags) or has_semantic_tags(active_tags)

        tags = normalize_semantic_tags({})
        tags = merge_semantic_tags(tags, active_tags, overwrite=True)
        tags = merge_semantic_tags(tags, ood_tags, overwrite=True, merge_task_fields=False, merge_state_fields=True)
        tags = merge_semantic_tags(tags, monitor_tags, overwrite=False, merge_task_fields=False, merge_state_fields=True)
        if not has_vlm_or_active_tags:
            fallback_tags = fallback_semantic_tags(
                global_task=self.current_instruction,
                current_subtask=current_subtask,
                signal_name=signal.name,
            )
            tags = merge_semantic_tags(tags, fallback_tags, overwrite=False)
        return tags

    def _recovery_environment_success_short_circuit(self, task_env: Any, *, stage: str) -> bool:
        if (
            not self._pure_tool_control_enabled()
            or not bool(getattr(task_env, "eval_success", False))
        ):
            return False
        self._authoritative_environment_success = True
        self._pending_action_effect_verification = None
        self._clear_recovery_backend_error_streak(stage="environment_success")
        payload = {
            "authority": "environment_eval_success",
            "stage": stage,
            "environment_success": True,
            "next_action": "stop_global_task",
        }
        self._trace("recovery_environment_success_short_circuit", **payload)
        self._dump_rollout_event("recovery_environment_success_short_circuit", **payload)
        return True

    def _dispatch_recovery_tools(self, task_env: Any, signal: MonitorSignal) -> str:
        if self._recovery_environment_success_short_circuit(task_env, stage="before_recovery_route"):
            return "task_complete"
        if self._pending_action_effect_verification is not None:
            return self._retry_pending_action_effect_verification(signal=signal)
        active = self.memory_store.state.active_skill
        available_tools = self._recovery_dispatcher.available_tools(task_env)
        observation_budget = self._current_stationary_observation_budget()
        observation_tool_suppressed = bool(
            REOBSERVE_SCENE_TOOL in available_tools
            and observation_budget.get(
                "stationary_scene_reobserve_budget_exhausted"
            )
        )
        if observation_tool_suppressed:
            # Do not ask the model for a tool that the runtime will reject.
            # The remaining tools advance the physical rollout.  This state is
            # keyed by physical action and viewpoint, so a real change makes
            # re-observation available again without ending the episode.
            available_tools = [
                tool
                for tool in available_tools
                if tool != REOBSERVE_SCENE_TOOL
            ]
        current_subtask = "" if active is None else active.instruction
        preferred_arm = "either" if active is None else normalize_preferred_arm(active.preferred_arm)
        blocked_grounded_setups = self._blocked_grounded_setup_payload()
        semantic_tags = self._recovery_semantic_tags(signal=signal, current_subtask=current_subtask)
        planner_context_mode = self._planner_context_mode()
        if planner_context_mode == COMPACT_PLANNER_CONTEXT_MODE:
            planning_scene_memory = self._planner_scene_view()
            planning_observation_preprocess = {}
            planning_observation_summary = ""
        else:
            planning_scene_memory = self._compact_scene_memory_for_effect(
                self.memory_store.state.working.scene_memory
            )
            planning_observation_preprocess = self._compact_observation_preprocess_for_recovery(
                self.memory_store.state.working.observation_preprocess
            )
            planning_observation_summary = self._get_env_summary()
        recovery_state = self.memory_store.state.recovery.to_dict()
        recovery_state.update(
            {
                "active_skill_id": self._active_skill_id(),
                "grasp_transport_policy": (
                    self._grasp_transport_policy()
                ),
                "release_guard_enabled": self._release_guard_enabled(),
                "action_geometry_repair_pending_policy": (
                    self._action_geometry_repair_pending_policy()
                ),
                "evidence_acquisition": dict(
                    self._last_evidence_acquisition_decision
                ),
                "stationary_observation_budget": observation_budget,
                "reobserve_scene_available": not observation_tool_suppressed,
                "planner_context_mode": planner_context_mode,
            }
        )
        if planner_context_mode != COMPACT_PLANNER_CONTEXT_MODE:
            recovery_state.update(
                {
                    "preferred_arm": preferred_arm,
                    "blocked_grounded_setups": blocked_grounded_setups,
                    "manipulation_state": public_manipulation_state(
                        self.memory_store.state.working.manipulation_state
                    ),
                }
            )
        route = self._recovery_policy_resolver.resolve(
            signal=signal,
            global_task=self.current_instruction,
            current_subtask=current_subtask,
            recovery_state=recovery_state,
            observation_summary=planning_observation_summary,
            robot_state=self._get_robot_state(),
            recovery_history=list(self.memory_store.state.working.recovery_history),
            available_tools=available_tools,
            semantic_tags=semantic_tags,
            scene_memory=planning_scene_memory,
            observation_preprocess=planning_observation_preprocess,
            preferred_arm=preferred_arm,
            blocked_grounded_setups=blocked_grounded_setups,
        )
        if self._recovery_environment_success_short_circuit(task_env, stage="after_recovery_route"):
            return "task_complete"
        if route is None:
            fallback_action = "retry" if self._pure_tool_control_enabled() else self._select_action_from_policy(signal_name=signal.name)
            route_error = self._recovery_policy_resolver.last_error
            route_error_kind = self._recovery_policy_resolver.last_error_kind
            if self._pure_tool_control_enabled() and route_error_kind == "backend":
                fallback_action = _RECOVERY_BACKEND_ERROR
                self._record_recovery_backend_error(
                    stage="recovery_route",
                    error=route_error or "recovery route backend unavailable",
                )
            if route_error:
                self.memory_store.record_recovery(
                    "recovery_router_unavailable:" + self._compact_summary_text(route_error, 240)
                )
            self._trace(
                "recovery_router",
                signal_name=signal.name,
                reason=signal.reason,
                route_found=False,
                fallback_action=fallback_action,
                route_error=route_error,
                route_error_kind=route_error_kind,
                available_tools=available_tools,
                semantic_tags=semantic_tags,
                scene_memory=planning_scene_memory,
                observation_preprocess=planning_observation_preprocess,
                planner_context_mode=planner_context_mode,
            )
            self._dump_rollout_event(
                "recovery_router",
                signal_name=signal.name,
                reason=signal.reason,
                route_found=False,
                fallback_action=fallback_action,
                route_error=route_error,
                route_error_kind=route_error_kind,
                available_tools=available_tools,
                semantic_tags=semantic_tags,
                scene_memory=planning_scene_memory,
                observation_preprocess=planning_observation_preprocess,
                planner_context_mode=planner_context_mode,
            )
            return fallback_action

        tool_names = [call.tool_name for call in route.plan.tool_calls]
        public_tool_calls = [
            {
                "tool_name": call.tool_name,
                "args": self._public_tool_args(call.args or {}),
            }
            for call in route.plan.tool_calls
        ]
        self._trace(
            "recovery_router",
            signal_name=signal.name,
            ood_scenario=signal.details.get("ood_scenario", signal.name),
            workflow=route.workflow_name,
            post_recovery_intent=route.post_recovery_intent,
            selected_arm=route.selected_arm,
            preferred_arm=preferred_arm,
            tools=tool_names,
            tool_calls=public_tool_calls,
            available_tools=available_tools,
            reason=route.reason,
            semantic_tags=semantic_tags,
            scene_memory=planning_scene_memory,
            observation_preprocess=planning_observation_preprocess,
            blocked_grounded_setups=blocked_grounded_setups,
            observation_summary=planning_observation_summary,
            robot_state=self._get_robot_state(),
            planner_context_mode=planner_context_mode,
            reobserve_scene_suppressed=observation_tool_suppressed,
            stationary_observation_budget=observation_budget,
        )
        self._dump_rollout_event(
            "recovery_router",
            signal_name=signal.name,
            ood_scenario=signal.details.get("ood_scenario", signal.name),
            workflow=route.workflow_name,
            post_recovery_intent=route.post_recovery_intent,
            selected_arm=route.selected_arm,
            preferred_arm=preferred_arm,
            tools=tool_names,
            tool_calls=public_tool_calls,
            available_tools=available_tools,
            reason=route.reason,
            semantic_tags=semantic_tags,
            scene_memory=planning_scene_memory,
            observation_preprocess=planning_observation_preprocess,
            blocked_grounded_setups=blocked_grounded_setups,
            observation_summary=planning_observation_summary,
            robot_state=self._get_robot_state(),
            planner_context_mode=planner_context_mode,
            reobserve_scene_suppressed=observation_tool_suppressed,
            stationary_observation_budget=observation_budget,
        )
        if self._pure_tool_control_enabled() and not route.plan.tool_calls:
            self._clear_recovery_backend_error_streak(stage="valid_empty_recovery_route")
            return _RECOVERY_EMPTY_PLAN
        self._trace("recovery_dispatch", signal_name=signal.name, reason=signal.reason, plan=route.plan.name, workflow=route.workflow_name)
        planner_explicit_open = any(
            call.tool_name == "open_gripper"
            for call in route.plan.tool_calls
        )
        tool_calls = self._with_internal_recovery_context(route.plan.tool_calls)
        tool_calls = self._apply_evidence_acquisition_policy(
            tool_calls,
            planner_explicit_open=planner_explicit_open,
        )
        if not tool_calls:
            internal_completion = str(
                self._pending_internal_recovery_completion or ""
            ).strip()
            self._pending_internal_recovery_completion = ""
            if internal_completion:
                payload = {
                    "signal_name": signal.name,
                    "workflow": route.workflow_name,
                    "plan": route.plan.name,
                    "completion": internal_completion,
                    "decision": dict(
                        self._last_evidence_acquisition_decision
                    ),
                }
                self._trace("recovery_internal_completion", **payload)
                self._dump_rollout_event(
                    "recovery_internal_completion",
                    **payload,
                )
                return (
                    _RECOVERY_INTERNAL_PROGRESS
                    if self._pure_tool_control_enabled()
                    else "retry"
                )
            payload = {
                "signal_name": signal.name,
                "workflow": route.workflow_name,
                "plan": route.plan.name,
                "decision": dict(
                    self._last_evidence_acquisition_decision
                ),
            }
            self._trace("recovery_evidence_acquisition_blocked", **payload)
            self._dump_rollout_event(
                "recovery_evidence_acquisition_blocked",
                **payload,
            )
            if (
                self._last_evidence_acquisition_decision.get(
                    "next_information_action"
                )
                == "perform_bounded_grasp_verification_motion"
            ):
                return "retry"
            if self._pure_tool_control_enabled():
                return _RECOVERY_EMPTY_PLAN
            return "retry"
        pre_effect_evidence = self._capture_action_effect_state()
        guarded_tool_names = [call.tool_name for call in tool_calls]
        if guarded_tool_names != tool_names:
            original_tool_calls = [
                {"tool_name": call.tool_name, "args": self._public_tool_args(call.args or {})}
                for call in route.plan.tool_calls
            ]
            guarded_tool_calls = [
                {"tool_name": call.tool_name, "args": self._public_tool_args(call.args or {})}
                for call in tool_calls
            ]
            self._trace(
                "recovery_guard",
                workflow=route.workflow_name,
                original_tools=tool_names,
                guarded_tools=guarded_tool_names,
                original_tool_calls=original_tool_calls,
                guarded_tool_calls=guarded_tool_calls,
                robot_state=self._get_robot_state(),
            )
            self._dump_rollout_event(
                "recovery_guard",
                workflow=route.workflow_name,
                original_tools=tool_names,
                guarded_tools=guarded_tool_names,
                original_tool_calls=original_tool_calls,
                guarded_tool_calls=guarded_tool_calls,
                robot_state=self._get_robot_state(),
            )
        input_snapshot = self.latest_snapshot
        results = self._recovery_dispatcher.dispatch_batch(
            tool_calls,
            task_env=task_env,
            latest_snapshot=input_snapshot,
        )
        try:
            self._record_selected_operation_candidates(
                calls=tool_calls,
                results=results,
            )
        except Exception:
            # Selection trace is diagnostic only; execution already happened.
            pass
        self._authorize_operation_geometry_refresh(
            calls=tool_calls,
            results=results,
        )
        if self._recovery_dispatcher.latest_environment_success:
            self._authoritative_environment_success = True
        if self._recovery_dispatcher.latest_snapshot is not None:
            self.update_snapshot(
                self._recovery_dispatcher.latest_snapshot,
                force_preprocess=any(
                    call.tool_name == "reobserve_scene"
                    for call in tool_calls
                ),
            )
            if self._recovery_dispatcher.latest_snapshot is not input_snapshot:
                self._pending_recovery_observation = {
                    "step_count": int(self._recovery_dispatcher.latest_snapshot.step_count),
                    "raw": dict(self._recovery_dispatcher.latest_snapshot.raw),
                }
        environment_success = bool(
            self._pure_tool_control_enabled()
            and (
                self._latest_snapshot_task_success()
                or self._recovery_dispatcher.latest_environment_success
            )
        )
        self._record_grounded_setup_outcomes(tool_calls, results)
        post_effect_evidence = self._capture_action_effect_state()
        self.memory_store.record_recovery(
            self._format_recovery_history_entry(route.workflow_name, tool_calls, results)
        )
        if environment_success:
            action_effect = self._record_environment_success_effect(
                route.workflow_name,
                current_subtask=current_subtask,
            )
            self._clear_recovery_backend_error_streak(stage="environment_success_after_recovery_batch")
            return self._complete_recovery_effect_transition(
                workflow_name=route.workflow_name,
                plan_name=route.plan.name,
                current_subtask=current_subtask,
                post_recovery_intent=route.post_recovery_intent,
                expected_outcome=route.plan.expected_outcome,
                signal_reason=signal.reason,
                calls=tool_calls,
                results=results,
                action_effect=action_effect,
                environment_success=True,
            )
        try:
            action_effect = self._verify_action_effect(
                route.workflow_name,
                current_subtask=current_subtask,
                post_recovery_intent=route.post_recovery_intent,
                expected_outcome=route.plan.expected_outcome,
                recovery_reason=route.reason,
                calls=tool_calls,
                results=results,
                pre=pre_effect_evidence,
                post=post_effect_evidence,
            )
        except ActionEffectVerifierBackendError as exc:
            self._pending_action_effect_verification = {
                "skill_id": self._active_skill_id(),
                "workflow_name": route.workflow_name,
                "plan_name": route.plan.name,
                "current_subtask": current_subtask,
                "post_recovery_intent": route.post_recovery_intent,
                "expected_outcome": route.plan.expected_outcome,
                "recovery_reason": route.reason,
                "signal_reason": signal.reason,
                "calls": list(tool_calls),
                "results": list(results),
                "pre": dict(pre_effect_evidence),
                "post": dict(post_effect_evidence),
            }
            self._record_recovery_backend_error(stage="action_effect_verifier", error=exc)
            payload = {
                "workflow": route.workflow_name,
                "plan": route.plan.name,
                "backend_stage": "action_effect_verifier",
                "backend_error": self._last_recovery_backend_error,
                "physical_batch_replay_blocked": True,
            }
            self._trace("action_effect_verification_backend_error", **payload)
            self._dump_rollout_event("action_effect_verification_backend_error", **payload)
            return _RECOVERY_BACKEND_ERROR

        self._clear_recovery_backend_error_streak(stage="action_effect_verifier")
        return self._complete_recovery_effect_transition(
            workflow_name=route.workflow_name,
            plan_name=route.plan.name,
            current_subtask=current_subtask,
            post_recovery_intent=route.post_recovery_intent,
            expected_outcome=route.plan.expected_outcome,
            signal_reason=signal.reason,
            calls=tool_calls,
            results=results,
            action_effect=action_effect,
            environment_success=False,
        )

    def _retry_pending_action_effect_verification(self, *, signal: MonitorSignal) -> str:
        pending = self._pending_action_effect_verification
        if pending is None:
            return "retry"
        active = self.memory_store.state.active_skill
        active_skill_id = "" if active is None else str(active.skill_id)
        pending_skill_id = str(pending.get("skill_id", "") or "")
        if pending_skill_id and pending_skill_id != active_skill_id:
            self._pending_action_effect_verification = None
            payload = {
                "pending_skill_id": pending_skill_id,
                "active_skill_id": active_skill_id,
                "physical_batch_replay_blocked": True,
            }
            self._trace("action_effect_verification_scope_mismatch", **payload)
            self._dump_rollout_event("action_effect_verification_scope_mismatch", **payload)
            return "retry"
        try:
            action_effect = self._verify_action_effect(
                str(pending["workflow_name"]),
                current_subtask=str(pending["current_subtask"]),
                post_recovery_intent=str(pending["post_recovery_intent"]),
                expected_outcome=str(pending["expected_outcome"]),
                recovery_reason=str(pending["recovery_reason"]),
                calls=list(pending["calls"]),
                results=list(pending["results"]),
                pre=dict(pending["pre"]),
                post=dict(pending["post"]),
            )
        except ActionEffectVerifierBackendError as exc:
            self._record_recovery_backend_error(stage="action_effect_verifier", error=exc)
            payload = {
                "workflow": str(pending["workflow_name"]),
                "plan": str(pending["plan_name"]),
                "backend_stage": "action_effect_verifier",
                "backend_error": self._last_recovery_backend_error,
                "physical_batch_replay_blocked": True,
                "retry_signal": signal.name,
            }
            self._trace("action_effect_verification_backend_error", **payload)
            self._dump_rollout_event("action_effect_verification_backend_error", **payload)
            return _RECOVERY_BACKEND_ERROR

        self._pending_action_effect_verification = None
        self._clear_recovery_backend_error_streak(stage="action_effect_verifier")
        return self._complete_recovery_effect_transition(
            workflow_name=str(pending["workflow_name"]),
            plan_name=str(pending["plan_name"]),
            current_subtask=str(pending["current_subtask"]),
            post_recovery_intent=str(pending["post_recovery_intent"]),
            expected_outcome=str(pending["expected_outcome"]),
            signal_reason=str(pending["signal_reason"]),
            calls=list(pending["calls"]),
            results=list(pending["results"]),
            action_effect=action_effect,
            environment_success=False,
        )

    def _complete_recovery_effect_transition(
        self,
        *,
        workflow_name: str,
        plan_name: str,
        current_subtask: str,
        post_recovery_intent: str,
        expected_outcome: str,
        signal_reason: str,
        calls: list[RecoveryToolCall],
        results: list[Any],
        action_effect: dict[str, Any],
        environment_success: bool,
    ) -> str:
        self._update_manipulation_state_from_effect(
            calls=calls,
            results=results,
            action_effect=action_effect,
        )
        self._credit_pure_tool_control_verified_progress(
            action_effect=action_effect,
            calls=calls,
        )
        if environment_success:
            resolved_control = "task_complete"
            transition_reason = "authoritative RMBench eval_success completed the global task"
        else:
            resolved_control, transition_reason = self._resolve_after_action_subtask_control(
                proposed_intent=post_recovery_intent,
                action_effect=action_effect,
            )
        self.memory_store.record_recovery(
            self._format_action_effect_history_entry(
                action_effect,
                calls,
                results,
                resolved_control=resolved_control,
            )
        )
        transition_payload = {
            "authority": (
                "environment_eval_success"
                if environment_success
                else "after_action_agent_api_verifier"
                if self._pure_tool_control_enabled()
                else "legacy_recovery_intent"
            ),
            "verifier_bypassed": environment_success,
            "global_task_success": environment_success,
            "current_subtask": current_subtask,
            "skill_id": self._active_skill_id(),
            "proposed_post_recovery_intent": post_recovery_intent,
            "effect_verified": action_effect.get("effect_verified", "unverified"),
            "subtask_status": action_effect.get("subtask_status", "uncertain"),
            "recommended_control": action_effect.get("recommended_control", "retry"),
            "resolved_control": resolved_control,
            "reason": transition_reason,
        }
        self._trace("subtask_transition_decision", **transition_payload)
        self._dump_rollout_event("subtask_transition_decision", **transition_payload)
        if not environment_success:
            self.memory_store.set_last_error(signal_reason)
        self._record_trace_and_rollout_event(
            "recovery_result",
            {
                "workflow": workflow_name,
                "plan": plan_name,
                "post_recovery_intent": post_recovery_intent,
                "resolved_control": resolved_control,
                "subtask_status": action_effect.get(
                    "subtask_status", "uncertain"
                ),
                "effect_authority": (
                    "environment_eval_success"
                    if environment_success
                    else "after_action_agent_api_verifier"
                ),
                "verifier_bypassed": environment_success,
                "environment_success": environment_success,
                "expected_outcome": expected_outcome,
                "results": self._recovery_results_for_trace(
                    calls,
                    results,
                ),
            },
        )
        return resolved_control

    def _credit_pure_tool_control_verified_progress(
        self,
        *,
        action_effect: dict[str, Any],
        calls: list[RecoveryToolCall],
    ) -> bool:
        """Reset the stall budget only for verified physical subtask progress."""

        if not self._pure_tool_control_enabled():
            return False
        effect_verified = str(
            action_effect.get("effect_verified", "unverified") or "unverified"
        ).strip().lower()
        subtask_status = str(
            action_effect.get("subtask_status", "uncertain") or "uncertain"
        ).strip().lower()
        if effect_verified != "true" or subtask_status not in {
            "in_progress",
            "completed",
        }:
            return False
        physical_tools = [
            str(call.tool_name)
            for call in calls
            if str(call.tool_name).strip()
            and str(call.tool_name).strip() != "reobserve_scene"
        ]
        if not physical_tools:
            return False
        previous_count = self._pure_tool_control_no_progress_control_turns
        if previous_count <= 0:
            return False
        self._pure_tool_control_no_progress_control_turns = 0
        effect_type = str(
            action_effect.get("effect_type", "unknown") or "unknown"
        ).strip().lower()
        self.memory_store.record_recovery(
            "pure_tool_control_verified_progress:"
            f"{effect_type}:{previous_count}->0"
        )
        payload = {
            "authority": "after_action_agent_api_verifier",
            "effect_type": effect_type,
            "subtask_status": subtask_status,
            "physical_tools": physical_tools,
            "previous_no_progress_control_turn_count": previous_count,
            "no_progress_control_turn_count": 0,
        }
        self._trace("pure_tool_control_verified_progress", **payload)
        self._dump_rollout_event(
            "pure_tool_control_verified_progress",
            **payload,
        )
        return True

    def _resolve_after_action_subtask_control(
        self,
        *,
        proposed_intent: str,
        action_effect: dict[str, Any],
    ) -> tuple[str, str]:
        if not self._pure_tool_control_enabled():
            return proposed_intent, "non-pure recovery keeps the existing intent policy"

        effect_verified = str(action_effect.get("effect_verified", "unverified")).strip().lower()
        subtask_status = str(action_effect.get("subtask_status", "uncertain")).strip().lower()
        recommended_control = str(action_effect.get("recommended_control", "retry")).strip().lower()

        if subtask_status == "completed":
            if effect_verified == "true":
                return "subtask_complete", "fresh after-action evidence verified the subtask stop condition"
            return "retry", "completion rejected because the physical effect is not verified"
        if subtask_status == "failed":
            return "replan", "after-action verifier classified the current subtask as failed"
        if subtask_status == "in_progress":
            if recommended_control in {"continue", "retry"}:
                return recommended_control, "after-action verifier kept the current subtask active"
            return "retry", "in-progress subtask cannot be advanced by a replan recommendation"
        return "retry", "subtask status is uncertain; preserve the current subtask and gather more evidence"

    def _capture_action_effect_state(self) -> dict[str, Any]:
        return {
            "robot_state": self._compact_robot_state_for_effect(self._get_robot_state()),
            "scene_memory": self._compact_scene_memory_for_effect(
                self.memory_store.state.working.scene_memory,
                prefer_latest_observation=True,
            ),
            "observation_summary": self._get_env_summary(),
            "runtime_evaluation": self._runtime_evaluation_context(),
            "_runtime_grasp_motion_snapshot": capture_grasp_motion_snapshot(
                observation_preprocess=(
                    self.memory_store.state.working.observation_preprocess
                ),
                manipulation_state=(
                    self.memory_store.state.working.manipulation_state
                ),
                robot_state=self._get_robot_state(),
            ),
        }

    def _record_environment_success_effect(self, workflow_name: str, *, current_subtask: str) -> dict[str, Any]:
        result = {
            "effect_verified": "true",
            "effect_type": "unknown",
            "confidence": 1.0,
            "evidence_summary": (
                "RMBench EnvSnapshot.eval_success became true after the bounded recovery batch; "
                "this environment result is the authoritative global task-success signal."
            ),
            "failure_reason": "",
            "next_constraint": "Stop dispatching physical tools because the environment is terminal.",
            "memory_update": "Authoritative environment success observed; global task completed.",
            "subtask_status": "completed",
            "recommended_control": "continue",
            "global_task_success": True,
            "authority": "environment_eval_success",
            "current_subtask": current_subtask,
        }
        payload = {
            "workflow": workflow_name,
            "authority": "environment_eval_success",
            "verifier_bypassed": True,
            "result": result,
        }
        self._trace("environment_success_effect_commit", **payload)
        self._dump_rollout_event("environment_success_effect_commit", **payload)
        return result

    def _verify_action_effect(
        self,
        workflow_name: str,
        *,
        current_subtask: str,
        post_recovery_intent: str,
        expected_outcome: str,
        recovery_reason: str,
        calls: list[RecoveryToolCall],
        results: list[Any],
        pre: dict[str, Any],
        post: dict[str, Any],
    ) -> dict[str, Any]:
        verifier_pre = dict(pre)
        verifier_post = dict(post)
        grasp_pre_snapshot = verifier_pre.pop(
            "_runtime_grasp_motion_snapshot",
            {},
        )
        grasp_post_snapshot = verifier_post.pop(
            "_runtime_grasp_motion_snapshot",
            {},
        )
        grasp_validation = validate_grasp_motion_effect(
            calls=calls,
            results=results,
            pre=grasp_pre_snapshot,
            post=grasp_post_snapshot,
        )
        if grasp_validation.get("applicable") is not True:
            deferred_grasp_validation = (
                validate_pending_grasp_observation_effect(
                    calls=calls,
                    results=results,
                    post=grasp_post_snapshot,
                )
            )
            if deferred_grasp_validation.get("applicable") is True:
                grasp_validation = deferred_grasp_validation
        ambiguous_return_validation = (
            self._runtime_ambiguous_grasp_return_validation(
                calls=calls,
                results=results,
            )
        )
        payload = {
            "global_task": self.current_instruction,
            "current_subtask": current_subtask,
            "skill_id": self._active_skill_id(),
            "workflow": workflow_name,
            "post_recovery_intent": post_recovery_intent,
            "expected_outcome": expected_outcome,
            "recovery_reason": recovery_reason,
            "last_tool_calls": self._compact_tool_calls_for_effect(calls, results),
            "pre": verifier_pre,
            "post": verifier_post,
            "recent_recovery_history": list(self.memory_store.state.working.recovery_history[-8:]),
        }
        place_validation = self._runtime_place_effect_validation(
            calls=calls,
            results=results,
        )
        if place_validation.get("applicable") is True:
            payload["runtime_place_validation"] = place_validation
        if grasp_validation.get("applicable") is True:
            payload["runtime_grasp_validation"] = grasp_validation
        if ambiguous_return_validation.get("applicable") is True:
            payload["runtime_ambiguous_grasp_return"] = (
                ambiguous_return_validation
            )
        result = self._action_effect_verifier.verify(payload)
        result = apply_grasp_motion_validation(
            result,
            grasp_validation,
            diagnostic_lift_attempted=any(
                is_runtime_grasp_diagnostic_lift_call(call)
                for call in calls
            )
            or pending_grasp_observation_attempted(
                calls=calls,
                post=grasp_post_snapshot,
            ),
        )
        result = apply_grasp_close_boundary_effect(
            result,
            calls=calls,
            results=results,
            grasp_transport_policy=self._grasp_transport_policy(),
        )
        if ambiguous_return_validation.get("applicable") is True:
            result["runtime_ambiguous_grasp_return"] = (
                ambiguous_return_validation
            )
            if ambiguous_return_validation.get("attempt_completed") is True:
                result.update(
                    {
                        "effect_verified": "true",
                        "effect_type": "grasp_recovery",
                        "subtask_status": "in_progress",
                        "recommended_control": "retry",
                        "failure_reason": "",
                        "next_constraint": (
                            "The ambiguous grasp was returned to its recorded "
                            "support path and the arm cleared it; reacquire "
                            "before starting a new grasp attempt."
                        ),
                        "memory_update": (
                            "Exact ambiguous grasp attempt safely returned; "
                            "transport was never authorized."
                        ),
                        "authority": (
                            "runtime_ambiguous_grasp_return_contract"
                        ),
                    }
                )
            else:
                result.update(
                    {
                        "effect_verified": "unverified",
                        "effect_type": "grasp_recovery",
                        "subtask_status": "in_progress",
                        "recommended_control": "retry",
                        "failure_reason": (
                            ambiguous_return_validation.get("reason", "")
                        ),
                        "next_constraint": (
                            "Resume only the persisted exact ambiguous-grasp "
                            "return transaction; do not transport, relower, "
                            "or reopen completed phases."
                        ),
                        "authority": (
                            "runtime_ambiguous_grasp_return_contract"
                        ),
                    }
                )
        if place_validation.get("applicable") is True:
            result["runtime_place_validation"] = place_validation
            if self._release_requires_placement_recovery(place_validation):
                result.update(
                    {
                        "effect_verified": "false",
                        "effect_type": "place",
                        "subtask_status": "failed",
                        "recommended_control": "replan",
                        "failure_reason": (
                            "the released object is detached but its observed "
                            "target error exceeds the validated placement "
                            "tolerance"
                        ),
                        "next_constraint": (
                            "reacquire the released instance and choose a "
                            "currently free/support-valid placement target; "
                            "do not continue observation-only verification"
                        ),
                        "memory_update": (
                            "release completed physically, but deterministic "
                            "placement validation requires bounded regrasp and "
                            "replacement"
                        ),
                    }
                )
            elif self._release_waits_for_stability_only(place_validation):
                result["next_constraint"] = (
                    "keep the released object and robot unchanged; EE clearance "
                    "is already verified, so issue reobserve_scene only until "
                    "runtime_place_validation.verified=true"
                )
            if (
                place_validation.get("release_verification_required") is True
                and place_validation.get("verified") is not True
                and str(result.get("effect_verified", "")).strip().lower()
                == "true"
                and (
                    str(
                        result.get("effect_type", "")
                    ).strip().lower()
                    in {"place", "placement", "release", "open"}
                    or str(
                        result.get("subtask_status", "")
                    ).strip().lower()
                    == "completed"
                )
            ):
                result.update(
                    {
                        "effect_verified": "unverified",
                        "subtask_status": "uncertain",
                        "recommended_control": "retry",
                        "failure_reason": (
                            "runtime placement evidence has not yet confirmed "
                            "target occupancy, detachment, and stability"
                        ),
                        "next_constraint": (
                            (
                                "keep the released object and robot unchanged; "
                                "EE clearance is already verified, so issue "
                                "reobserve_scene only until "
                                "runtime_place_validation.verified=true"
                            )
                            if place_validation.get(
                                "ee_cleared_from_release_pose"
                            )
                            is True
                            else (
                                "keep the released object unchanged, perform "
                                "at most one bounded clearance action, and then "
                                "reobserve until "
                                "runtime_place_validation.verified=true"
                            )
                        ),
                    }
                )
        self._trace("action_effect_verification", workflow=workflow_name, result=result)
        self._dump_rollout_event("action_effect_verification", workflow=workflow_name, result=result)
        return result

    def _runtime_ambiguous_grasp_return_validation(
        self,
        *,
        calls: list[RecoveryToolCall],
        results: list[Any],
    ) -> dict[str, Any]:
        if not any(
            is_runtime_ambiguous_grasp_return_call(call)
            for call in calls
        ):
            return {"applicable": False}
        reduction = reduce_ambiguous_grasp_return_results(
            self.memory_store.state.working.manipulation_state,
            calls=calls,
            results=results,
            post_return_snapshot=self.latest_snapshot,
        )
        pending = [
            {
                "arm": arm,
                "phase": state.get("phase"),
                "status": (
                    state.get("ambiguous_grasp_return", {}).get("status")
                    if isinstance(
                        state.get("ambiguous_grasp_return"),
                        dict,
                    )
                    else None
                ),
                "held_instance_id": state.get("held_instance_id"),
                "grasp_attempt_nonce": state.get("grasp_attempt_nonce"),
            }
            for arm, state in reduction.manipulation_state.items()
            if isinstance(state, dict)
            and str(state.get("phase", "") or "").strip().lower()
            == "ambiguous_grasp_return_pending"
        ]
        return {
            "applicable": True,
            "physical_return_succeeded": (
                reduction.physical_return_succeeded
            ),
            "fresh_snapshot_confirmed": (
                reduction.fresh_snapshot_confirmed
            ),
            "attempt_completed": reduction.attempt_completed,
            "reason": reduction.reason,
            "pending_returns": pending,
        }

    @staticmethod
    def _placement_release_context_from_state(
        state: Any,
    ) -> dict[str, Any] | None:
        """Return prior runtime placement facts for post-release recording.

        This context never authorizes or blocks ``open_gripper``.  It only
        lets the after-action verifier attribute a successful opening to the
        placement pose that runtime had already reached.
        """

        if not isinstance(state, dict):
            return None
        phase = str(state.get("phase", "") or "").strip().lower()
        support_lease = state.get("support_contact_release")
        support_ready = bool(
            state.get("release_ready") is True
            and isinstance(support_lease, dict)
            and support_lease.get("state") == "ready"
        )
        if phase != "place_aligned" and not support_ready:
            return None
        held_instance_id = str(
            state.get("held_instance_id", "") or ""
        ).strip()
        target_id = str(state.get("place_target_id", "") or "").strip()
        if not held_instance_id or not target_id:
            return None
        lease_candidate = (
            support_lease.get("candidate")
            if isinstance(support_lease, dict)
            else None
        )
        candidate = (
            dict(lease_candidate)
            if isinstance(lease_candidate, dict)
            else {
                "candidate_id": state.get("place_candidate_id"),
                "target_id": target_id,
                "held_instance_id": held_instance_id,
                "held_object_target_world_m": state.get(
                    "held_object_target_world_m"
                ),
                "held_extent_m": state.get("held_extent_m"),
                "ee_target_pose": state.get(
                    "release_ee_target_world_m"
                ),
            }
        )
        return {
            "held_instance_id": held_instance_id,
            "target_id": target_id,
            "candidate": candidate,
            "pre_release_validated": bool(
                state.get("pre_release_validated")
            ),
        }

    def _runtime_place_effect_validation(
        self,
        *,
        calls: list[RecoveryToolCall],
        results: list[Any],
        pending_arm: str = "",
    ) -> dict[str, Any]:
        manipulation = self.memory_store.state.working.manipulation_state
        release_call: RecoveryToolCall | None = None
        release_result: Any | None = None
        candidate: dict[str, Any] | None = None
        arm = ""
        held_instance_id = ""
        target_id = ""
        release_ee_target: list[float] | None = None
        pre_release_valid = False
        release_step: int | None = None

        for index, call in enumerate(calls):
            result = results[index] if index < len(results) else None
            if call.tool_name == "move_ee_to_grounded_instance":
                details = getattr(result, "details", {}) or {}
                mode = str(
                    details.get(
                        "operation_action_mode",
                        (call.args or {}).get("_operation_action_mode", ""),
                    )
                    or ""
                ).strip().lower()
                if mode != "place":
                    continue
                arm = self._single_arm_from_call(call)
                held_instance_id = self._recovery_history_instance_id(
                    call,
                    details,
                )
                target_id = str(
                    details.get(
                        "operation_target_id",
                        (call.args or {}).get("target_id", ""),
                    )
                    or ""
                ).strip()
                materialized = self._grounded_instance_for_call(call)
                selected = (
                    materialized.get("selected_operation_candidate")
                    if isinstance(materialized, dict)
                    else None
                )
                if isinstance(selected, dict):
                    candidate = dict(selected)
                release_ee_target = self._xyz_prefix(details.get("target_pose"))
                pre_release_valid = bool(
                    details.get("place_target_revalidated")
                    and details.get("support_valid")
                    and details.get("free")
                )
                continue
            if call.tool_name != "open_gripper":
                continue
            open_arm = self._single_arm_from_call(call)
            call_candidate = (call.args or {}).get("_release_place_candidate")
            release_call = call
            release_result = result
            if isinstance(call_candidate, dict):
                candidate = dict(call_candidate)
                arm = open_arm
                held_instance_id = str(
                    (call.args or {}).get(
                        "release_held_instance_id",
                        held_instance_id,
                    )
                    or held_instance_id
                ).strip()
                target_id = str(
                    (call.args or {}).get("release_target_id", target_id)
                    or target_id
                ).strip()
                release_ee_target = self._xyz_prefix(
                    candidate.get("ee_target_pose")
                )
                validation = (call.args or {}).get(
                    "_release_place_validation"
                )
                pre_release_valid = bool(
                    isinstance(validation, dict)
                    and validation.get("valid") is True
                )
                continue

            # With the Release Guard disabled, planner release IDs are not
            # authorization.  Associate the observed gripper opening with the
            # actual runtime placement context, if one exists, without
            # changing or delaying the command.
            if candidate is not None and open_arm == arm:
                continue
            prior_state = (
                manipulation.get(open_arm)
                if isinstance(manipulation, dict)
                and open_arm in {"left", "right"}
                else None
            )
            state_context = self._placement_release_context_from_state(
                prior_state
            )
            if state_context is None:
                release_call = None
                release_result = None
                continue
            arm = open_arm
            candidate = dict(state_context["candidate"])
            held_instance_id = state_context["held_instance_id"]
            target_id = state_context["target_id"]
            release_ee_target = self._xyz_prefix(
                candidate.get("ee_target_pose")
            )
            pre_release_valid = state_context["pre_release_validated"]

        pending_state: dict[str, Any] | None = None
        if release_call is None and isinstance(manipulation, dict):
            normalized_pending_arm = str(pending_arm or "").strip().lower()
            candidate_arms = (
                (normalized_pending_arm,)
                if normalized_pending_arm in {"left", "right"}
                else ("left", "right")
            )
            for candidate_arm in candidate_arms:
                state = manipulation.get(candidate_arm)
                if (
                    isinstance(state, dict)
                    and state.get("phase") == "release_pending_verification"
                ):
                    pending_state = dict(state)
                    arm = candidate_arm
                    held_instance_id = str(
                        state.get("held_instance_id", "") or ""
                    ).strip()
                    target_id = str(
                        state.get("place_target_id", "") or ""
                    ).strip()
                    release_ee_target = self._xyz_prefix(
                        state.get("release_ee_target_world_m")
                    )
                    pre_release_valid = bool(
                        state.get("pre_release_validated")
                    )
                    try:
                        release_step = int(state.get("updated_step"))
                    except (TypeError, ValueError):
                        release_step = None
                    break

        if release_call is None and pending_state is None:
            return {
                "applicable": bool(candidate is not None),
                "release_verification_required": False,
                "verified": False,
                "arm": arm,
                "held_instance_id": held_instance_id,
                "target_id": target_id,
            }

        if candidate is not None:
            expected_object = self._xyz_prefix(
                candidate.get("held_object_target_world_m")
            )
            extent = self._xyz_prefix(candidate.get("held_extent_m"))
        else:
            expected_object = self._xyz_prefix(
                (pending_state or {}).get("held_object_target_world_m")
            )
            extent = self._xyz_prefix(
                (pending_state or {}).get("held_extent_m")
            )
        current_scene = self.memory_store.state.working.scene_memory
        current_instance = self._scene_instance_by_ref(
            current_scene,
            held_instance_id,
        )
        object_position_is_fresh = (
            self._scene_instance_position_is_fresh(
                current_instance,
                min_env_step=release_step,
            )
        )
        current_object = (
            self._xyz_prefix(
                current_instance.get(
                    "latest_world_m",
                    current_instance.get("world_m"),
                )
            )
            if (
                isinstance(current_instance, dict)
                and object_position_is_fresh
            )
            else None
        )
        tolerance = place_target_tolerance_m(extent)
        object_target_error = (
            sum(
                (current_object[index] - expected_object[index]) ** 2
                for index in range(3)
            )
            ** 0.5
            if current_object is not None and expected_object is not None
            else None
        )
        object_at_target = bool(
            object_target_error is not None
            and object_target_error <= tolerance
        )

        robot_state = self._get_robot_state()
        arm_state = (
            robot_state.get(arm)
            if isinstance(robot_state, dict) and arm in {"left", "right"}
            else None
        )
        gripper_open = gripper_command_state(arm_state) == "open"
        current_ee = (
            self._xyz_prefix(arm_state.get("xyz"))
            if isinstance(arm_state, dict)
            else None
        )
        ee_clearance = (
            sum(
                (current_ee[index] - release_ee_target[index]) ** 2
                for index in range(3)
            )
            ** 0.5
            if current_ee is not None and release_ee_target is not None
            else None
        )
        ee_cleared = bool(
            gripper_open
            and ee_clearance is not None
            and ee_clearance >= 0.02
        )
        detached = bool(gripper_open and ee_cleared)
        stable = self._instance_recently_stable_at_target(
            current_scene,
            instance=current_instance,
            target=expected_object,
            tolerance=tolerance,
            min_env_step=release_step,
        )
        release_executed = bool(
            pending_state is not None
            or (
                release_call is not None
                and release_result is not None
                and getattr(release_result, "success", False)
            )
        )
        object_position_observed = bool(
            current_object is not None
            and expected_object is not None
            and object_target_error is not None
        )
        placement_recovery_required = bool(
            pending_state is not None
            and release_executed
            and pre_release_valid
            and gripper_open
            and detached
            and object_position_observed
            and object_at_target is False
        )
        verified = bool(
            release_executed
            and pre_release_valid
            and gripper_open
            and object_at_target
            and detached
            and stable
        )
        return {
            "applicable": True,
            "release_verification_required": True,
            "verified": verified,
            "arm": arm,
            "held_instance_id": held_instance_id,
            "target_id": target_id,
            "pre_release_validated": pre_release_valid,
            "release_executed": release_executed,
            "gripper_open": gripper_open,
            "object_position_observed": object_position_observed,
            "object_position_fresh": object_position_is_fresh,
            "verified_object_world_m": (
                list(current_object)
                if current_object is not None
                else None
            ),
            "object_at_target": object_at_target,
            "object_target_error_m": (
                None
                if object_target_error is None
                else round(object_target_error, 6)
            ),
            "target_tolerance_m": round(tolerance, 6),
            "ee_cleared_from_release_pose": ee_cleared,
            "ee_release_clearance_m": (
                None if ee_clearance is None else round(ee_clearance, 6)
            ),
            "detachment_verified": detached,
            "stable_across_fresh_observations": stable,
            "placement_recovery_required": placement_recovery_required,
        }

    def _resolve_pending_release_from_runtime(
        self,
        *,
        source: str,
    ) -> dict[str, Any]:
        """Commit a pending release as soon as deterministic evidence is complete.

        Action-effect verification used to be the only place that cleared
        ``release_pending_verification``.  A later Scene Memory update can
        either provide the required second stable sample or prove that the
        detached object is outside the selected target.  Resolve both outcomes
        directly from runtime evidence: successful placement clears pending,
        while an observed target miss becomes a recoverable released-object
        state instead of an observation-only deadlock.
        """

        pending = self._pending_release_verification_states()
        if not pending:
            return {}
        validation = self._runtime_place_effect_validation(
            calls=[],
            results=[],
        )
        if self._release_requires_placement_recovery(validation):
            self._update_manipulation_state_from_effect(
                calls=[],
                results=[],
                action_effect={
                    "effect_verified": "false",
                    "effect_type": "place",
                    "runtime_place_validation": validation,
                },
            )
            return validation
        if validation.get("verified") is not True:
            return validation
        arm = str(validation.get("arm", "") or "").strip().lower()
        state = pending.get(arm)
        if arm not in {"left", "right"} or not isinstance(state, dict):
            return validation
        resolution = {
            "arm": arm,
            "held_instance_id": str(
                validation.get("held_instance_id", "") or ""
            ).strip(),
            "target_id": str(
                validation.get("target_id", "") or ""
            ).strip(),
            "active_skill_id": self._active_skill_id(),
            "source_skill_id": str(
                state.get("source_skill_id", "") or ""
            ).strip(),
            "env_step": (
                int(self.latest_snapshot.step_count)
                if self.latest_snapshot is not None
                else -1
            ),
            "source": str(source or "runtime"),
            "validation": dict(validation),
        }
        self._update_manipulation_state_from_effect(
            calls=[],
            results=[],
            action_effect={
                "effect_verified": "true",
                "effect_type": "release",
                "runtime_place_validation": validation,
            },
        )
        self._recent_release_resolutions[arm] = resolution
        history_entry = (
            "runtime_release_verified:"
            f"arm={arm},instance={resolution['held_instance_id']},"
            f"target={resolution['target_id']},source={resolution['source']}"
        )
        self.memory_store.record_recovery(history_entry)
        self._trace("runtime_release_verification_resolved", **resolution)
        self._dump_rollout_event(
            "runtime_release_verification_resolved",
            **resolution,
        )
        return validation

    def _scene_instance_by_ref(
        self,
        scene_memory: Any,
        instance_ref: str,
    ) -> dict[str, Any] | None:
        if not isinstance(scene_memory, dict):
            return None
        normalized = str(instance_ref or "").strip().lower()
        if not normalized:
            return None
        matches: list[dict[str, Any]] = []
        for instance in scene_memory.get("instances", []) or []:
            if not isinstance(instance, dict):
                continue
            refs = {
                str(instance.get(key, "") or "").strip().lower()
                for key in (
                    "instance_id",
                    "track_id",
                    "oracle_id",
                    "oracle_source_path",
                )
                if str(instance.get(key, "") or "").strip()
            }
            if normalized in refs:
                matches.append(instance)
        return matches[0] if len(matches) == 1 else None

    def _perception_descriptor_for_scene_instance(
        self,
        instance: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(instance, dict):
            return {}
        raw_object_id = str(
            instance.get("source_object_id")
            or instance.get("class")
            or ""
        ).strip()
        object_id, inferred_hint = normalize_perception_object_id(
            raw_object_id
        )
        if not object_id:
            return {}
        text_prompt = str(
            instance.get("source_text_prompt")
            or object_id
        ).strip()
        return {
            "object_id": object_id,
            "text_prompt": text_prompt or object_id,
            "instance_hint": merge_instance_hints(
                instance.get("query_instance_hint", ""),
                inferred_hint,
            ),
            "class_aliases": [
                str(item)
                for item in instance.get("class_aliases", []) or []
                if str(item or "").strip()
            ],
        }

    def _scene_instance_position_is_fresh(
        self,
        instance: dict[str, Any] | None,
        *,
        min_env_step: int | None,
    ) -> bool:
        if not isinstance(instance, dict):
            return False
        position_state = str(
            instance.get("position_state", "") or ""
        ).strip().lower()
        if (
            position_state
            and position_state != POSITION_CURRENT_VERIFIED
        ):
            return False
        status = str(instance.get("status", "") or "").strip().lower()
        if status and status != "visible":
            return False
        if min_env_step is None:
            return True
        raw_last_seen = instance.get(
            "last_verified_step",
            instance.get("last_seen_step"),
        )
        if raw_last_seen is None:
            # Legacy/test scene payloads did not expose freshness metadata.
            return not status or status == "visible"
        try:
            return int(raw_last_seen) >= int(min_env_step)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _position_history_observation_token(
        item: dict[str, Any],
    ) -> tuple[str, str] | None:
        try:
            capture_id = int(item.get("observation_capture_id"))
        except (TypeError, ValueError):
            capture_id = -1
        if capture_id >= 0:
            return "capture", str(capture_id)
        explicit = str(
            item.get("observation_token", "") or ""
        ).strip()
        if explicit:
            return "token", explicit
        mask_path = str(item.get("mask_path", "") or "").strip()
        if mask_path:
            camera = str(item.get("camera", "") or "").strip()
            return "mask", f"{camera}:{mask_path}"
        try:
            return "env_step", str(int(item.get("env_step")))
        except (TypeError, ValueError):
            return None

    def _instance_recently_stable_at_target(
        self,
        scene_memory: Any,
        *,
        instance: dict[str, Any] | None,
        target: list[float] | None,
        tolerance: float,
        min_env_step: int | None = None,
    ) -> bool:
        if (
            not isinstance(scene_memory, dict)
            or not isinstance(instance, dict)
            or target is None
        ):
            return False
        track_id = str(instance.get("track_id", "") or "").strip()
        temporal = scene_memory.get("temporal_memory")
        tracks = (
            temporal.get("tracks", [])
            if isinstance(temporal, dict)
            else []
        )
        track = next(
            (
                item
                for item in tracks
                if isinstance(item, dict)
                and str(item.get("track_id", "") or "").strip() == track_id
            ),
            None,
        )
        history = track.get("history", []) if isinstance(track, dict) else []
        eligible_history: list[dict[str, Any]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            if min_env_step is not None:
                try:
                    item_step = int(item.get("env_step"))
                except (TypeError, ValueError):
                    continue
                if item_step < min_env_step:
                    continue
            eligible_history.append(item)
        distinct_observations: list[dict[str, Any]] = []
        seen_tokens: set[tuple[str, str]] = set()
        for item in reversed(eligible_history):
            token = self._position_history_observation_token(item)
            if token is None or token in seen_tokens:
                continue
            seen_tokens.add(token)
            distinct_observations.append(item)
            if len(distinct_observations) == 2:
                break
        recent = [
            self._xyz_prefix(item.get("world_m"))
            for item in reversed(distinct_observations)
        ]
        if len(recent) != 2 or any(item is None for item in recent):
            return False
        first = recent[0]
        second = recent[1]
        assert first is not None and second is not None
        first_error = sum(
            (first[index] - target[index]) ** 2 for index in range(3)
        ) ** 0.5
        second_error = sum(
            (second[index] - target[index]) ** 2 for index in range(3)
        ) ** 0.5
        drift = sum(
            (second[index] - first[index]) ** 2 for index in range(3)
        ) ** 0.5
        return (
            first_error <= tolerance
            and second_error <= tolerance
            and drift <= min(0.01, tolerance / 2.0)
        )

    def _format_action_effect_history_entry(
        self,
        action_effect: dict[str, Any],
        calls: list[RecoveryToolCall],
        results: list[Any],
        *,
        resolved_control: str = "",
    ) -> str:
        effect = self._history_value(action_effect.get("effect_verified", "unverified")) or "unverified"
        effect_type = self._history_value(action_effect.get("effect_type", "unknown")) or "unknown"
        confidence = self._history_value(action_effect.get("confidence", 0.0))
        if results:
            tool = self._history_value(getattr(results[-1], "tool_name", ""))
        elif calls:
            tool = self._history_value(calls[-1].tool_name)
        else:
            tool = ""
        instance = ""
        for call in reversed(calls):
            instance = self._recovery_history_instance_id(call, {})
            if instance:
                break
        parts = [f"effect={effect}", f"type={effect_type}"]
        skill_id = self._active_skill_id()
        if skill_id:
            parts.append(f"skill={self._history_value(skill_id)}")
        subtask_status = self._history_value(action_effect.get("subtask_status", "uncertain")) or "uncertain"
        recommended_control = self._history_value(action_effect.get("recommended_control", "retry")) or "retry"
        effective_control = self._history_value(resolved_control) or recommended_control
        parts.extend((f"subtask={subtask_status}", f"control={effective_control}"))
        if resolved_control and effective_control != recommended_control:
            parts.append(f"recommended={recommended_control}")
        if tool:
            parts.append(f"tool={tool}")
        if instance:
            parts.append(f"instance={instance}")
        if confidence:
            parts.append(f"confidence={confidence}")
        failure = self._history_value(action_effect.get("failure_reason"))
        if failure:
            parts.append(f"failure={failure}")
        evidence = self._history_value(action_effect.get("evidence_summary"))
        if evidence:
            parts.append(f"evidence={evidence}")
        next_constraint = self._history_value(action_effect.get("next_constraint"))
        if next_constraint:
            parts.append(f"next={next_constraint}")
        memory_update = self._history_value(action_effect.get("memory_update"))
        if memory_update:
            parts.append(f"memory={memory_update}")
        return "action_effect:" + ",".join(parts)

    def _record_selected_operation_candidates(
        self,
        *,
        calls: list[RecoveryToolCall],
        results: list[Any],
    ) -> None:
        """Trace geometry only for candidates that reached the dispatcher.

        Candidate IDs are reusable slots, so the geometry revision is part of
        the identity.  A batch-level set avoids writing the same four poses for
        an approach and its subsequent final grounded move.
        """

        recorded: set[tuple[str, str, str, str, str, str]] = set()
        for dispatch_index, call in enumerate(calls):
            if (
                call.tool_name != "move_ee_to_grounded_instance"
                or dispatch_index >= len(results)
            ):
                continue
            result = results[dispatch_index]
            details = getattr(result, "details", {})
            if isinstance(details, dict) and details.get("skipped") is True:
                continue
            try:
                selection_key, selection_payload, contract_error = (
                    self._selected_operation_candidate_trace_payload(
                        call=call,
                        result=result,
                        dispatch_index=dispatch_index,
                    )
                )
            except Exception as exc:
                # One malformed candidate must not suppress later selections.
                self._record_trace_and_rollout_event(
                    "operation_candidate_selection_trace_error",
                    {
                        "schema": (
                            "trace/operation_candidate_selection_contract_error/v1"
                        ),
                        "schema_version": 1,
                        "reason": "candidate trace projection failed",
                        "error_type": type(exc).__name__,
                        "dispatch_index": dispatch_index,
                    },
                )
                continue
            if contract_error:
                self._record_trace_and_rollout_event(
                    "operation_candidate_selection_trace_error",
                    contract_error,
                )
                continue
            if (
                selection_key is None
                or selection_payload is None
                or selection_key in recorded
            ):
                continue
            recorded.add(selection_key)
            self._record_trace_and_rollout_event(
                "operation_candidate_selected",
                selection_payload,
            )

    def _selected_operation_candidate_trace_payload(
        self,
        *,
        call: RecoveryToolCall,
        result: Any,
        dispatch_index: int,
    ) -> tuple[
        tuple[str, str, str, str, str, str] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        raw_details = getattr(result, "details", {})
        details = dict(raw_details) if isinstance(raw_details, dict) else {}
        materialized = self._grounded_instance_for_call(call)
        candidate = (
            materialized.get("selected_operation_candidate")
            if isinstance(materialized, dict)
            else None
        )
        if not isinstance(candidate, dict):
            return None, None, None
        raw_args = call.args
        args = dict(raw_args) if isinstance(raw_args, dict) else {}
        attempt = args.get("_operation_candidate_attempt")
        if not isinstance(attempt, dict):
            attempt = {}

        candidate_id = str(candidate.get("candidate_id", "") or "").strip()
        reported_candidate_id = str(
            details.get("operation_candidate_id", "") or ""
        ).strip()
        requested_candidate_id = str(
            args.get("_operation_candidate_id", "") or ""
        ).strip()
        geometry = candidate_geometry(candidate)
        revision = candidate_geometry_revision(geometry)
        attempted_candidate_id = str(
            attempt.get("candidate_id", "") or ""
        ).strip()
        attempted_revision = str(
            attempt.get("candidate_geometry_revision", "") or ""
        ).strip()
        mismatches: list[str] = []
        if reported_candidate_id and reported_candidate_id != candidate_id:
            mismatches.append("result_candidate_id")
        if requested_candidate_id and requested_candidate_id != candidate_id:
            mismatches.append("requested_candidate_id")
        if attempted_candidate_id and attempted_candidate_id != candidate_id:
            mismatches.append("attempted_candidate_id")
        if attempted_revision and attempted_revision != revision:
            mismatches.append("candidate_geometry_revision")
        if not candidate_id:
            mismatches.append("missing_materialized_candidate_id")
        required_pose_keys = (
            "object_contact_pose",
            "tcp_pose",
            "ee_target_pose",
            "approach_pose",
        )
        geometry_poses = geometry.get("poses")
        if not isinstance(geometry_poses, dict):
            geometry_poses = {}
        missing_pose_keys = [
            key for key in required_pose_keys if key not in geometry_poses
        ]
        if missing_pose_keys:
            mismatches.append("missing_candidate_geometry")
        if mismatches:
            return (
                None,
                None,
                {
                    "schema": (
                        "trace/operation_candidate_selection_contract_error/v1"
                    ),
                    "schema_version": 1,
                    "reason": (
                        "materialized candidate identity or geometry mismatch"
                    ),
                    "mismatches": mismatches,
                    "dispatch_index": dispatch_index,
                    "materialized_candidate_id": candidate_id,
                    "reported_candidate_id": reported_candidate_id,
                    "requested_candidate_id": requested_candidate_id,
                    "attempted_candidate_id": attempted_candidate_id,
                    "materialized_geometry_revision": revision,
                    "attempted_geometry_revision": attempted_revision,
                    "missing_pose_keys": missing_pose_keys,
                },
            )

        instance_id = str(
            details.get("instance_id", "")
            or self._recovery_history_instance_id(call, details)
        ).strip()
        arm = str(
            details.get("operation_candidate_arm", "")
            or candidate.get("arm", "")
            or self._single_arm_from_call(call)
        ).strip().lower()
        action_mode = str(
            details.get("operation_action_mode", "")
            or candidate.get("action_mode", "")
            or args.get("_operation_action_mode", "")
        ).strip().lower()
        target_id = str(
            details.get("operation_target_id", "")
            or candidate.get("target_id", "")
            or args.get("target_id", "")
        ).strip()
        selection_key = (
            instance_id,
            candidate_id,
            revision,
            arm,
            action_mode,
            target_id,
        )
        observation = args.get("_observation_preprocess")
        if not isinstance(observation, dict):
            observation = {}
        source_camera = str(
            candidate.get("source_camera", "")
            or materialized.get("camera", "")
        ).strip()
        raw_supporting_cameras = materialized.get(
            "supporting_cameras", []
        )
        supporting_cameras = (
            list(raw_supporting_cameras)
            if isinstance(raw_supporting_cameras, (list, tuple, set))
            else []
        )
        selection_payload = {
            "candidate_id": candidate_id,
            "candidate_geometry_revision": revision,
            "instance_id": instance_id,
            "track_id": materialized.get("track_id"),
            "object_id": materialized.get("source_object_id"),
            "source_detection_index": materialized.get("source_rank"),
            "source_candidate_index": candidate.get(
                "source_candidate_index"
            ),
            "source_camera": source_camera,
            "supporting_cameras": supporting_cameras,
            "arm": arm,
            "action_mode": action_mode,
            "target_id": target_id,
            "geometry_source": candidate.get("geometry_source"),
            "priority": candidate.get("priority"),
            "score": candidate.get("score"),
            "observation_generation": attempt.get(
                "observation_generation",
                observation.get("observation_generation"),
            ),
            "observation_capture_id": attempt.get(
                "observation_capture_id",
                observation.get("observation_capture_id"),
            ),
            "dispatch_index": dispatch_index,
            "dispatch_env_step": attempt.get("dispatch_env_step"),
            "result_env_step": details.get("step_count"),
            "object_contact_pose": geometry_poses["object_contact_pose"],
            "tcp_pose": geometry_poses["tcp_pose"],
            "ee_target_pose": geometry_poses["ee_target_pose"],
            "approach_pose": geometry_poses["approach_pose"],
        }
        return selection_key, selection_payload, None

    def _recovery_results_for_trace(
        self,
        calls: list[RecoveryToolCall],
        results: list[Any],
    ) -> list[dict[str, Any]]:
        """Copy execution evidence and attach a trace-only geometry revision."""

        projected: list[dict[str, Any]] = []
        for index, result in enumerate(results):
            raw_details = getattr(result, "details", {})
            details = dict(raw_details) if isinstance(raw_details, dict) else {}
            if index < len(calls) and details.get("skipped") is not True:
                raw_args = calls[index].args
                args = dict(raw_args) if isinstance(raw_args, dict) else {}
                attempt = args.get("_operation_candidate_attempt")
                if isinstance(attempt, dict):
                    candidate_id = str(
                        details.get("operation_candidate_id", "")
                        or attempt.get("candidate_id", "")
                        or args.get("_operation_candidate_id", "")
                    ).strip()
                    revision = str(
                        attempt.get("candidate_geometry_revision", "")
                        or ""
                    ).strip()
                    if candidate_id:
                        details["operation_candidate_id"] = candidate_id
                    if revision:
                        details[
                            "operation_candidate_geometry_revision"
                        ] = revision
            projected.append(
                {
                    "tool_name": str(getattr(result, "tool_name", "")),
                    "success": bool(getattr(result, "success", False)),
                    "message": str(getattr(result, "message", "")),
                    "details": details,
                }
            )
        return projected

    def _compact_tool_calls_for_effect(self, calls: list[RecoveryToolCall], results: list[Any]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for index, call in enumerate(calls):
            result = results[index] if index < len(results) else None
            args = self._public_tool_args(call.args or {})
            item: dict[str, Any] = {"tool_name": call.tool_name, "args": args}
            if result is not None:
                item["result"] = {
                    "success": bool(getattr(result, "success", False)),
                    "message": str(getattr(result, "message", "")),
                    "details": self._compact_result_details(getattr(result, "details", {}) or {}),
                }
            compact.append(item)
        return compact

    def _public_tool_args(self, args: dict[str, Any]) -> dict[str, Any]:
        keep_keys = {
            "arm",
            "role",
            "focus_key",
            "instance_id",
            "instance_ref",
            "action_mode",
            "target_id",
            "release_target_id",
            "point_key",
            "gripper_precondition",
            "complete_transient_cycle",
            "post_contact_clearance",
            "offset_xyz",
            "preserve_height",
            "held_instance_id",
            "held_instance_ref",
            "release_held_instance_id",
            "held_role",
            "held_focus_key",
            "clearance_margin",
            "max_clearance_lift",
            "clearance_steps",
            "max_translation",
            "steps",
            "axis",
            "direction",
            "distance",
            "_guard_reason",
        }
        return {key: value for key, value in dict(args or {}).items() if key in keep_keys}

    def _compact_result_details(self, details: dict[str, Any]) -> dict[str, Any]:
        keep_keys = {
            "arm",
            "instance_id",
            "point_key",
            "target_pose",
            "executed_pose",
            "observed_pose",
            "target_observation_error_m",
            "target_reached",
            "target_reached_tolerance_m",
            "operation_target_id",
            "operation_target_kind",
            "operation_action_mode",
            "operation_candidate_arm",
            "operation_geometry_source",
            "operation_source_candidate_index",
            "operation_tcp_pose",
            "operation_object_contact_pose",
            "held_object_target_world_m",
            "held_object_to_tcp_translation_world_m",
            "place_target_revalidated",
            "support_valid",
            "free",
            "occupied_by",
            "target_geometry_drift_m",
            "release_target_id",
            "release_held_instance_id",
            "release_place_validation",
            "actual_spatial_state_signature",
            "contact_reference_world_m",
            "contact_reference_tolerance_m",
            "requested_translation",
            "max_translation",
            "preserve_height",
            "held_instance_id",
            "target_instance_id",
            "held_bottom_world_z",
            "target_top_world_z",
            "clearance_margin",
            "clearance_lift",
            "steps",
            "gripper_value",
            "step_count",
            "axis",
            "signed_distance",
            "start_pose",
            "observed_displacement_xyz",
            "observed_displacement_m",
            "observed_axis_displacement_m",
            "skipped",
            "batch_halted",
            "batch_halt_reason",
            "failure_reason",
        }
        return {key: value for key, value in dict(details or {}).items() if key in keep_keys}

    def _compact_robot_state_for_effect(self, robot_state: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for key in ("step", "step_limit"):
            if key in robot_state:
                compact[key] = robot_state[key]
        for arm in ("left", "right"):
            arm_state = robot_state.get(arm)
            if isinstance(arm_state, dict):
                compact[arm] = {key: arm_state.get(key) for key in ("xyz", "rpy", "gripper", "gripper_command") if key in arm_state}
        return compact

    def _compact_scene_memory_for_effect(
        self,
        scene_memory: dict[str, Any],
        *,
        prefer_latest_observation: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(scene_memory, dict):
            return {}
        compact_instances: list[dict[str, Any]] = []
        focus = scene_memory.get("task_focus") if isinstance(scene_memory.get("task_focus"), dict) else {}
        relevant_instance_refs: set[str] = set()
        for key in ("target_instances", "tool_instances"):
            values = focus.get(key) if isinstance(focus, dict) else None
            if isinstance(values, list):
                relevant_instance_refs.update(
                    str(item or "").strip().lower()
                    for item in values
                    if str(item or "").strip()
                )
        observation_preprocess = self.memory_store.state.working.observation_preprocess
        if isinstance(observation_preprocess, dict):
            for segment in observation_preprocess.get("segmentation", []) or []:
                if not isinstance(segment, dict):
                    continue
                for key in (
                    "instance_ref",
                    "query_instance_ref",
                    "track_id",
                    "instance_id",
                    "oracle_id",
                    "oracle_source_path",
                ):
                    value = str(segment.get(key, "") or "").strip().lower()
                    if value:
                        relevant_instance_refs.add(value)
        for instance in scene_memory.get("instances", []) or []:
            if not isinstance(instance, dict):
                continue
            identity_refs = {
                str(instance.get(key, "") or "").strip().lower()
                for key in (
                    "instance_id",
                    "track_id",
                    "oracle_id",
                    "oracle_source_path",
                )
                if str(instance.get(key, "") or "").strip()
            }
            if relevant_instance_refs and not relevant_instance_refs.intersection(
                identity_refs
            ):
                continue
            compact_instance = {
                key: instance.get(key)
                for key in (
                    "instance_id",
                    "track_id",
                    "oracle_id",
                    "oracle_source_path",
                    "class",
                    "role",
                    "query_role",
                    "status",
                    "stability",
                    "position_state",
                    "position_source",
                    "last_verified_world_m",
                    "last_verified_score",
                    "last_verified_step",
                    "verified_roles",
                    "score",
                    "bbox_xyxy",
                    "centroid_px",
                    "world_m",
                    "first_observed_world_m",
                    "first_observed_top_surface_world_m",
                    "approach_world_m",
                    "approach_quat_wxyz",
                    "grasp_world_m",
                    "grasp_quat_wxyz",
                    "contact_world_m",
                    "contact_quat_wxyz",
                    "operation_pose_candidate_count",
                    "quality",
                    "missing_steps",
                )
                if key in instance
            }
            latest_world = instance.get("latest_world_m")
            if prefer_latest_observation and isinstance(latest_world, (list, tuple)) and len(latest_world) == 3:
                stable_world = compact_instance.get("world_m")
                compact_instance["stable_world_m"] = stable_world
                compact_instance["latest_world_m"] = latest_world
                compact_instance["world_m"] = latest_world
                compact_instance["position_source"] = "latest_observation"
            compact_instances.append(compact_instance)
        operation_targets = [
            {
                key: item.get(key)
                for key in (
                    "target_id",
                    "action_mode",
                    "held_instance_id",
                    "target_kind",
                    "object_target_world_m",
                    "support_instance_id",
                    "support_evidence",
                    "vacated_by_instance_id",
                    "placement_relation",
                    "reference_region_id",
                    "reference_object_id",
                    "reference_instance_ids",
                    "expected_count",
                    "observed_member_count",
                    "reference_evidence_status",
                    "support_valid",
                    "free",
                    "arm_options",
                    "holding_status",
                    "grasp_transport_policy",
                    "priority",
                )
                if key in item
            }
            for item in scene_memory.get(OPERATION_TARGETS_KEY, []) or []
            if isinstance(item, dict)
        ][:8]
        raw_spatial_state = scene_memory.get(SPATIAL_STATE_KEY)
        spatial_state: dict[str, Any] = {}
        if isinstance(raw_spatial_state, dict):
            spatial_state = {
                key: raw_spatial_state.get(key)
                for key in (
                    "coordinate_source",
                    "signature_resolution_m",
                    "signature",
                    "order_by_x",
                    "order_by_y",
                    "horizontal_overlap_pairs",
                    "support_pairs",
                )
                if key in raw_spatial_state
            }
            spatial_state["instances"] = [
                dict(item)
                for item in raw_spatial_state.get("instances", []) or []
                if isinstance(item, dict)
            ][:12]
        reference_regions = compact_reference_regions(scene_memory)
        return {
            "env_step": scene_memory.get("env_step"),
            "task_focus": focus,
            "instances": compact_instances[:6],
            OPERATION_TARGETS_KEY: operation_targets,
            REFERENCE_REGIONS_KEY: reference_regions,
            SPATIAL_STATE_KEY: spatial_state,
            "manipulation_state": public_manipulation_state(
                scene_memory.get(
                    "manipulation_state",
                    self.memory_store.state.working.manipulation_state,
                )
                or {}
            ),
            "uncertainty": list(scene_memory.get("uncertainty", []) or [])[:4],
            "summary": scene_memory.get("summary", ""),
        }

    def _compact_observation_preprocess_for_recovery(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        compact_segments: list[dict[str, Any]] = []
        segments = payload.get("segmentation", [])
        if not isinstance(segments, list):
            segments = []
        for segment in segments[:8]:
            if not isinstance(segment, dict):
                continue
            detections = segment.get("detections", [])
            if not isinstance(detections, list):
                detections = []
            compact_segment: dict[str, Any] = {
                key: segment.get(key)
                for key in (
                    "success",
                    "object_id",
                    "text_prompt",
                    "camera",
                    "backend",
                    "query_role",
                    "instance_ref",
                    "query_instance_ref",
                    "oracle_id",
                    "oracle_source_path",
                    "identity_binding_required",
                    "identity_binding_error",
                    "query_instance_hint",
                    "query_reason",
                    "num_detections",
                    "score",
                    "bbox_xyxy",
                    "centroid_px",
                    "quality",
                )
                if key in segment
            }
            if "num_detections" not in compact_segment:
                compact_segment["num_detections"] = len(detections)
            if segment.get("error"):
                compact_segment["error"] = self._compact_summary_text(segment.get("error"), 180)
            grounding = self._compact_grounding_for_recovery(segment.get("grounding_3d"))
            if grounding:
                compact_segment["grounding_3d"] = grounding
            compact_detections: list[dict[str, Any]] = []
            for detection in detections[:5]:
                if not isinstance(detection, dict):
                    continue
                compact_detection = {
                    key: detection.get(key)
                    for key in (
                        "rank",
                        "score",
                        "camera",
                        "instance_ref",
                        "query_instance_ref",
                        "oracle_id",
                        "oracle_source_path",
                        "bbox_xyxy",
                        "centroid_px",
                        "quality",
                        "query_role",
                        *REFERENCE_QUERY_METADATA_KEYS,
                    )
                    if key in detection
                }
                detection_grounding = self._compact_grounding_for_recovery(detection.get("grounding_3d"))
                if detection_grounding:
                    compact_detection["grounding_3d"] = detection_grounding
                compact_detections.append(compact_detection)
            if compact_detections:
                compact_segment["detections"] = compact_detections
            dropped = segment.get("dropped_detections")
            if isinstance(dropped, list) and dropped:
                compact_segment["dropped_detections"] = [
                    {
                        key: item.get(key)
                        for key in ("object_id", "camera", "rank", "score", "bbox_xyxy", "quality")
                        if key in item
                    }
                    for item in dropped[:5]
                    if isinstance(item, dict)
                ]
            compact_segments.append(compact_segment)
        return {
            "stage": payload.get("stage", "observation_preprocess"),
            "env_step": payload.get("env_step"),
            "segmentation": compact_segments,
        }

    @staticmethod
    def _compact_grounding_for_recovery(grounding: Any) -> dict[str, Any]:
        if not isinstance(grounding, dict):
            return {}
        return {
            key: grounding.get(key)
            for key in (
                "success",
                "object_id",
                "camera",
                "valid_ratio",
                "centroid_world",
                "top_surface_world",
                "approach_point_world",
                "approach_pose_world",
                "grasp_pose_world",
                "contact_pose_world",
                "object_contact_pose_world",
                "bbox_world_min",
                "bbox_world_max",
            )
            if key in grounding
        }

    def _format_recovery_history_entry(self, workflow_name: str, calls: list[RecoveryToolCall], results: list[Any]) -> str:
        parts: list[str] = []
        last_grounded_point_by_arm: dict[str, str] = {}
        last_grounded_reached_by_arm: dict[str, str] = {}
        for index, result in enumerate(results):
            call = calls[index] if index < len(calls) else RecoveryToolCall(tool_name=getattr(result, "tool_name", ""), args={})
            result_details = getattr(result, "details", {}) or {}
            args = dict(call.args or {})
            tool_name = str(getattr(result, "tool_name", call.tool_name))
            success = bool(getattr(result, "success", False))
            context = self._recovery_tool_history_context(
                call=call,
                result_details=result_details,
                result_message=str(getattr(result, "message", "")),
                result_success=success,
                last_grounded_point_by_arm=last_grounded_point_by_arm,
                last_grounded_reached_by_arm=last_grounded_reached_by_arm,
            )
            if call.tool_name == "move_ee_to_grounded_instance":
                arm = self._single_arm_from_call(call)
                last_grounded_point_by_arm[arm] = self._normalized_grounded_point_key(call)
                reached = self._grounded_move_reached_point(result_details)
                if reached:
                    last_grounded_reached_by_arm[arm] = reached
            elif call.tool_name == "close_gripper":
                arm = self._single_arm_from_call(call)
                last_grounded_point_by_arm.setdefault(arm, "")
                last_grounded_reached_by_arm.setdefault(arm, "")
            segment = f"{tool_name}={success}"
            if context:
                segment += "(" + ",".join(context) + ")"
            parts.append(segment)
        skill_id = self._active_skill_id()
        scope = f"skill={self._history_value(skill_id)};" if skill_id else ""
        return f"tool_recovery:{workflow_name}:" + scope + ";".join(parts)

    def _recovery_tool_history_context(
        self,
        *,
        call: RecoveryToolCall,
        result_details: dict[str, Any],
        result_message: str,
        result_success: bool,
        last_grounded_point_by_arm: dict[str, str],
        last_grounded_reached_by_arm: dict[str, str],
    ) -> list[str]:
        args = dict(call.args or {})
        context: list[str] = []
        arm = self._history_value(result_details.get("arm", args.get("arm")))
        if arm:
            context.append(f"arm={arm}")
        role = self._history_value(args.get("role"))
        if role:
            context.append(f"role={role}")
        instance_id = self._recovery_history_instance_id(call, result_details)
        if instance_id:
            context.append(f"instance={instance_id}")
        point_key = ""
        if call.tool_name == "move_ee_to_grounded_instance":
            point_key = self._normalized_grounded_point_key(call)
            context.append(f"point={point_key}")
            candidate_mode = self._history_value(
                result_details.get(
                    "operation_action_mode",
                    args.get("_operation_action_mode"),
                )
            )
            target_id = self._history_value(
                result_details.get(
                    "operation_target_id",
                    args.get("target_id"),
                )
            )
            if candidate_mode == "place" and target_id:
                context.append(f"target={target_id}")
            if candidate_mode:
                context.append(f"mode={candidate_mode}")
            reached = self._grounded_move_reached_point(result_details)
            if reached:
                context.append(f"reached_point={reached}")
        elif call.tool_name == "close_gripper":
            point_key = last_grounded_point_by_arm.get(str(arm or self._single_arm_from_call(call)), "")
            if point_key:
                context.append(f"after_point={point_key}")
            reached = last_grounded_reached_by_arm.get(str(arm or self._single_arm_from_call(call)), "")
            if reached:
                context.append(f"after_point_reached={reached}")
        elif self._history_value(args.get("point_key")):
            context.append(f"point={self._history_value(args.get('point_key'))}")
        guard_reason = self._history_value(args.get("_guard_reason"))
        if guard_reason:
            context.append(f"guard={guard_reason}")
        steps = self._history_value(result_details.get("steps", args.get("steps")))
        if steps:
            context.append(f"steps={steps}")
        gripper_value = self._history_value(result_details.get("gripper_value"))
        if gripper_value:
            context.append(f"gripper={gripper_value}")
        failure = ""
        if not result_success:
            failure = self._history_value(result_message)
        if result_message and "failed" in result_message.lower():
            failure = self._history_value(result_message)
        if not failure and result_message and str(result_message).lower().startswith(("scene ", "could not", "no ", "target_", "offset_")):
            failure = self._history_value(result_message)
        if failure and "executed " not in failure.lower() and "refreshed " not in failure.lower() and "set " not in failure.lower():
            context.append(f"failure={failure}")
        if call.tool_name == "close_gripper":
            if point_key == "grasp_world_m":
                context.append("grasp_confirmed=unverified")
            elif point_key:
                context.append("grasp_confirmed=false")
            else:
                context.append("grasp_confirmed=unverified")
        if call.tool_name == "reobserve_scene":
            step_count = self._history_value(result_details.get("step_count"))
            if step_count:
                context.append(f"step={step_count}")
        return context

    def _grounded_move_reached_point(self, result_details: dict[str, Any]) -> str:
        target_reached = result_details.get("target_reached")
        if isinstance(target_reached, bool):
            return "true" if target_reached else "false"
        target_pose = result_details.get("target_pose")
        observed_pose = result_details.get("observed_pose", result_details.get("executed_pose"))
        target_xyz = self._xyz_prefix(target_pose)
        observed_xyz = self._xyz_prefix(observed_pose)
        if target_xyz is None or observed_xyz is None:
            return ""
        distance = sum((float(a) - float(b)) ** 2 for a, b in zip(target_xyz, observed_xyz)) ** 0.5
        return "true" if distance <= 0.01 else "false"

    def _xyz_prefix(self, value: Any) -> list[float] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            xyz = [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
        if any(item != item for item in xyz):
            return None
        return xyz

    def _pose7_prefix(self, value: Any) -> list[float] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 7:
            return None
        try:
            pose = [float(value[index]) for index in range(7)]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in pose):
            return None
        quaternion_norm = math.sqrt(
            sum(item * item for item in pose[3:])
        )
        if quaternion_norm <= 1e-8:
            return None
        pose[3:] = [item / quaternion_norm for item in pose[3:]]
        return pose

    def _recovery_history_instance_id(self, call: RecoveryToolCall, result_details: dict[str, Any]) -> str:
        args = dict(call.args or {})
        explicit = self._history_value(result_details.get("instance_id", args.get("instance_id", args.get("instance_ref"))))
        if explicit:
            return explicit
        scene_memory = args.get("_scene_memory")
        if not isinstance(scene_memory, dict):
            return ""
        focus_key = str(args.get("focus_key", "")).strip()
        if not focus_key:
            role = str(args.get("role", "target")).strip().lower()
            focus_key = "tool_instances" if role == "tool" else "target_instances"
        task_focus = scene_memory.get("task_focus")
        if isinstance(task_focus, dict) and bool(task_focus.get("identity_binding_required")):
            return ""
        focused_ids = task_focus.get(focus_key) if isinstance(task_focus, dict) else None
        if isinstance(focused_ids, list):
            normalized_ids = {
                self._history_value(item)
                for item in focused_ids
                if self._history_value(item)
            }
            if len(normalized_ids) == 1:
                return next(iter(normalized_ids))
        return ""

    def _grounded_setup_key(self, call: RecoveryToolCall, result_details: dict[str, Any]) -> tuple[str, str, str] | None:
        if call.tool_name != "move_ee_to_grounded_instance":
            return None
        if (call.args or {}).get("_runtime_post_contact_clearance") is True:
            # Visibility cleanup is not a task-progress setup.  A partial clearance move must not
            # poison the same instance/arm/approach key used by later grounded approaches.
            return None
        instance_id = self._recovery_history_instance_id(call, result_details)
        arm = normalize_physical_arm(result_details.get("arm", (call.args or {}).get("arm")))
        point_key = self._normalized_grounded_point_key(call)
        if not instance_id or arm not in {"left", "right"} or not point_key:
            return None
        return instance_id, arm, point_key

    def _record_grounded_setup_outcomes(self, calls: list[RecoveryToolCall], results: list[Any]) -> None:
        for index, call in enumerate(calls):
            if index >= len(results):
                break
            details = getattr(results[index], "details", {}) or {}
            key = self._grounded_setup_key(call, details)
            if key is None:
                continue
            target_reached = details.get("target_reached")
            attempt = (call.args or {}).get(
                "_operation_candidate_attempt"
            )
            attempt_candidate_id = (
                str(attempt.get("candidate_id", "") or "").strip()
                if isinstance(attempt, dict)
                else ""
            )
            candidate_id = str(
                details.get("operation_candidate_id", "")
                or attempt_candidate_id
            ).strip()
            if key[2] == "approach_world_m":
                self._partial_grounded_approach_leases.pop(
                    (self._active_skill_id(), key[0], key[1]),
                    None,
                )
            if candidate_id:
                action_mode = operation_action_mode(
                    self._normalized_grounded_point_key(call),
                    details.get(
                        "operation_action_mode",
                        (call.args or {}).get("_operation_action_mode"),
                    ),
                )
                candidate_key = (key[0], key[1], action_mode, candidate_id)
                attempt = self._validated_operation_candidate_attempt(
                    call=call,
                    details=details,
                    key=candidate_key,
                )
                if target_reached is True:
                    if attempt is not None:
                        self._operation_candidate_lifecycle.record_success(
                            attempt,
                            executed_target_pose=details.get(
                                "target_pose"
                            ),
                        )
                    if bool(getattr(results[index], "success", False)):
                        self._record_grounded_geometry_lease(
                            call=call,
                            details=details,
                            key=key,
                            later_calls=calls[index + 1 :],
                        )
                elif target_reached is False:
                    if attempt is not None:
                        self._record_operation_candidate_failure(
                            attempt=attempt,
                            details=details,
                        )
                    if bool(getattr(results[index], "success", False)):
                        self._record_partial_grounded_approach_lease(
                            call=call,
                            details=details,
                            key=key,
                            later_calls=calls[index + 1 :],
                        )
                continue
            if target_reached is True:
                self._grounded_setup_failures.pop(key, None)
                self._blocked_grounded_setups.discard(key)
                if bool(getattr(results[index], "success", False)):
                    self._record_grounded_geometry_lease(
                        call=call,
                        details=details,
                        key=key,
                        later_calls=calls[index + 1 :],
                    )
                continue
            if target_reached is not False:
                continue
            try:
                target_error = float(details.get("target_observation_error_m"))
                if target_error != target_error:
                    target_error = None
            except (TypeError, ValueError):
                target_error = None
            previous = self._grounded_setup_failures.get(key, {})
            previous_error = previous.get("last_target_error_m")
            failure_count = int(previous.get("failure_count", 0)) + 1
            if (
                target_error is not None
                and isinstance(previous_error, (int, float))
                and target_error < float(previous_error) - _GROUNDED_SETUP_NO_PROGRESS_EPSILON_M
            ):
                failure_count = 1
            self._grounded_setup_failures[key] = {
                "failure_count": failure_count,
                "last_target_error_m": target_error,
            }
            if failure_count < _GROUNDED_SETUP_FAILURE_THRESHOLD or key in self._blocked_grounded_setups:
                continue
            self._blocked_grounded_setups.add(key)
            instance_id, arm, point_key = key
            history_entry = (
                "grounded_setup_blocked:"
                f"instance={instance_id},arm={arm},point={point_key},failures={failure_count}"
            )
            self.memory_store.record_recovery(history_entry)
            payload = {
                "instance_id": instance_id,
                "arm": arm,
                "point_key": point_key,
                "failure_count": failure_count,
                "last_target_error_m": target_error,
            }
            self._trace("grounded_setup_blocked", **payload)
            self._dump_rollout_event("grounded_setup_blocked", **payload)

    def _record_operation_candidate_failure(
        self,
        *,
        attempt: dict[str, Any],
        details: dict[str, Any],
    ) -> None:
        payload = self._operation_candidate_lifecycle.record_failure(
            attempt,
            target_error_m=details.get(
                "target_observation_error_m"
            ),
            target_id=details.get("operation_target_id"),
            failure_reason="no_progress",
            env_step=(
                int(self.latest_snapshot.step_count)
                if self.latest_snapshot is not None
                else None
            ),
            executed_target_pose=details.get("target_pose"),
        )
        if payload is None:
            return
        history_entry = (
            "operation_candidate_blocked:"
            f"instance={payload['instance_id']},arm={payload['arm']},"
            f"mode={payload['action_mode']},"
            f"point={payload.get('point_key', '')},"
            f"failures={payload['failure_count']},"
            f"geometry={payload['geometry_revision']}"
        )
        target_id = str(payload.get("target_id", "") or "").strip()
        if payload.get("action_mode") == "place" and target_id:
            history_entry += f",target={target_id}"
        self.memory_store.record_recovery(history_entry)
        payload["next_candidate_policy"] = (
            "select_first_unblocked_same_arm_and_mode"
        )
        self._trace("operation_candidate_blocked", **payload)
        self._dump_rollout_event("operation_candidate_blocked", **payload)

    def _validated_operation_candidate_attempt(
        self,
        *,
        call: RecoveryToolCall,
        details: dict[str, Any],
        key: tuple[str, str, str, str],
    ) -> dict[str, Any] | None:
        """Validate the frozen pre-dispatch attempt against adapter output.

        Reconstructing a missing token from post-action Scene Memory would
        assign an old execution failure to newly observed geometry.  Treat a
        missing or inconsistent token as an internal contract violation and
        preserve both versions unchanged.
        """

        attempt = (call.args or {}).get(
            "_operation_candidate_attempt"
        )
        reasons: list[str] = []
        if not isinstance(attempt, dict):
            reasons.append("missing_frozen_attempt")
            scope: list[Any] = []
            frozen_candidate_id = ""
        else:
            raw_scope = attempt.get("scope")
            scope = (
                list(raw_scope)
                if isinstance(raw_scope, (list, tuple))
                else []
            )
            frozen_candidate_id = str(
                attempt.get("candidate_id", "") or ""
            ).strip()
            expected_scope = [key[0], key[1], key[2]]
            if len(scope) != 4 or [str(item) for item in scope[:3]] != expected_scope:
                reasons.append("scope_mismatch")
            if frozen_candidate_id != key[3]:
                reasons.append("candidate_id_mismatch")
            frozen_point_key = self._normalized_grounded_point_key(
                RecoveryToolCall(
                    tool_name=call.tool_name,
                    args={
                        "point_key": attempt.get("point_key")
                    },
                )
            )
            current_point_key = (
                self._normalized_grounded_point_key(call)
            )
            if frozen_point_key != current_point_key:
                reasons.append("point_key_mismatch")
            result_point_key = str(
                details.get("point_key", "") or ""
            ).strip()
            if (
                result_point_key
                and self._normalized_grounded_point_key(
                    RecoveryToolCall(
                        tool_name=call.tool_name,
                        args={"point_key": result_point_key},
                    )
                )
                != frozen_point_key
            ):
                reasons.append("result_point_key_mismatch")
            result_target_id = str(
                details.get("operation_target_id", "") or ""
            ).strip()
            frozen_target_id = (
                str(scope[3] or "").strip()
                if len(scope) == 4
                else ""
            )
            if result_target_id and result_target_id != frozen_target_id:
                reasons.append("target_id_mismatch")
        if not reasons:
            return dict(attempt)
        payload = {
            "instance_id": key[0],
            "arm": key[1],
            "action_mode": key[2],
            "candidate_id": key[3],
            "reasons": reasons,
            "frozen_candidate_id": frozen_candidate_id,
            "frozen_scope": scope,
            "frozen_point_key": (
                attempt.get("point_key")
                if isinstance(attempt, dict)
                else None
            ),
            "result_point_key": details.get("point_key"),
            "result_target_id": details.get("operation_target_id"),
        }
        public_target_id = str(
            details.get(
                "operation_target_id",
                (call.args or {}).get("target_id", ""),
            )
            or (scope[3] if len(scope) == 4 else "")
            or ""
        ).strip()
        public_identity = (
            f",target={public_target_id or 'unknown'}"
            if key[2] == "place"
            else ""
        )
        self.memory_store.record_recovery(
            "operation_candidate_attempt_contract_violation:"
            f"instance={key[0]},arm={key[1]},mode={key[2]},"
            f"reasons={','.join(reasons)}{public_identity}"
        )
        self._trace(
            "operation_candidate_attempt_contract_violation",
            **payload,
        )
        self._dump_rollout_event(
            "operation_candidate_attempt_contract_violation",
            **payload,
        )
        return None

    def _record_partial_grounded_approach_lease(
        self,
        *,
        call: RecoveryToolCall,
        details: dict[str, Any],
        key: tuple[str, str, str],
        later_calls: list[RecoveryToolCall],
    ) -> None:
        """Authorize one exact continuation of a converging, self-occluded approach.

        This is not a general stale-pose exception.  It is created only when a
        structured candidate approach made monotonic observed progress, stopped
        within a small residual that one more bounded step could cover, and the
        post-action scene supplies a camera-ray self-occlusion proof.  The
        capability can only finish the same candidate's approach; it cannot
        authorize contact, grasp, close, or release.
        """

        instance_id, arm, point_key = key
        skill_id = self._active_skill_id()
        if (
            point_key != "approach_world_m"
            or not skill_id
            or len(later_calls) != 1
            or later_calls[0].tool_name != "reobserve_scene"
            or self.latest_snapshot is None
            or self._recovery_gripper_state().get(arm) != "open"
            or self._transport_holding_state(arm) is not None
        ):
            return
        args = dict(call.args or {})
        action_mode = operation_action_mode(
            point_key,
            details.get(
                "operation_action_mode",
                args.get("_operation_action_mode"),
            ),
        )
        candidate_id = str(
            details.get(
                "operation_candidate_id",
                args.get("_operation_candidate_id", ""),
            )
            or ""
        ).strip()
        if action_mode not in {"contact", "grasp"} or not candidate_id:
            return
        instance = self._grounded_instance_for_call(call)
        if instance is None:
            return
        status = str(instance.get("status", "") or "").strip().lower()
        stability = str(instance.get("stability", "") or "").strip().lower()
        action_geometry_state = str(
            instance.get("action_geometry_state", "verified")
            or "verified"
        ).strip().lower()
        if status != "visible" or stability in {
            "geometry_inconsistent_current_frame",
            "relocation_geometry_pending",
            "identity_repair_expired",
        } or action_geometry_state in {
            "relocation_pending",
            "identity_repair_expired",
            "unavailable",
        }:
            return

        raw_history = details.get("target_error_history_m")
        if not isinstance(raw_history, list) or len(raw_history) < 2:
            return
        try:
            error_history = [float(value) for value in raw_history]
            residual = float(details.get("target_observation_error_m"))
            target_tolerance = float(
                details.get("target_reached_tolerance_m", 0.01)
            )
            original_max_translation = float(
                details.get(
                    "max_translation",
                    args.get("max_translation", 0.06),
                )
            )
            executed_steps = int(details.get("executed_steps", -1))
        except (TypeError, ValueError):
            return
        if (
            executed_steps != len(error_history)
            or not all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    *error_history,
                    residual,
                    target_tolerance,
                    original_max_translation,
                )
            )
            or target_tolerance < 0.0
            or original_max_translation <= 0.0
            or residual <= target_tolerance
            or residual > _PARTIAL_APPROACH_CONTINUATION_MAX_RESIDUAL_M
            or residual > original_max_translation + 1e-8
            or abs(error_history[-1] - residual)
            > _GROUNDED_SETUP_NO_PROGRESS_EPSILON_M
            or any(
                current
                >= previous - _GROUNDED_SETUP_NO_PROGRESS_EPSILON_M
                for previous, current in zip(
                    error_history,
                    error_history[1:],
                )
            )
        ):
            return

        approach = self._xyz_prefix(instance.get("approach_world_m"))
        target_point_key = (
            "contact_world_m"
            if action_mode == "contact"
            else "grasp_world_m"
        )
        operation_target = self._xyz_prefix(instance.get(target_point_key))
        target_pose = self._xyz_prefix(details.get("target_pose"))
        observed = self._xyz_prefix(
            details.get("observed_pose", details.get("executed_pose"))
        )
        current_pose = getattr(
            self.latest_snapshot,
            f"{arm}_endpose",
            None,
        )
        current_xyz = self._xyz_prefix(
            current_pose.tolist()
            if hasattr(current_pose, "tolist")
            else current_pose
        )
        approach_target_drift = (
            None
            if approach is None or target_pose is None
            else self._xyz_distance(approach, target_pose)
        )
        observed_pose_drift = (
            None
            if observed is None or current_xyz is None
            else self._xyz_distance(observed, current_xyz)
        )
        observed_residual = (
            None
            if observed is None or approach is None
            else self._xyz_distance(observed, approach)
        )
        if (
            approach is None
            or operation_target is None
            or observed is None
            or approach_target_drift is None
            or approach_target_drift
            > _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
            or observed_pose_drift is None
            or observed_pose_drift
            > _GROUNDED_GEOMETRY_LEASE_MAX_POSE_DRIFT_M
            or observed_residual is None
            or abs(observed_residual - residual)
            > _GROUNDED_SETUP_NO_PROGRESS_EPSILON_M
        ):
            return

        current_scene = self.memory_store.state.working.scene_memory or {}
        current_matches: list[dict[str, Any]] = []
        normalized_instance_id = instance_id.strip().lower()
        for current_instance in current_scene.get("instances", []) or []:
            if not isinstance(current_instance, dict):
                continue
            refs = {
                str(current_instance.get(field, "") or "").strip().lower()
                for field in (
                    "instance_id",
                    "track_id",
                    "oracle_id",
                    "oracle_source_path",
                )
                if str(current_instance.get(field, "") or "").strip()
            }
            if normalized_instance_id in refs:
                current_matches.append(current_instance)
        if len(current_matches) != 1:
            return
        current_instance = current_matches[0]
        current_status = str(
            current_instance.get("status", "") or ""
        ).strip().lower()
        current_stability = str(
            current_instance.get("stability", "") or ""
        ).strip().lower()
        if (
            current_status != "tracked"
            or current_stability
            not in {
                "missing_current_frame",
                "geometry_inconsistent_current_frame",
            }
        ):
            return
        current_candidate = select_operation_pose_candidate(
            current_instance,
            arm=arm,
            action_mode=action_mode,
            requested_candidate_id=candidate_id,
        )
        if current_candidate is None:
            return
        current_materialized = materialize_operation_candidate(
            current_instance,
            current_candidate,
        )
        current_approach = self._xyz_prefix(
            current_materialized.get("approach_world_m")
        )
        current_target = self._xyz_prefix(
            current_materialized.get(target_point_key)
        )
        current_approach_drift = (
            None
            if current_approach is None
            else self._xyz_distance(approach, current_approach)
        )
        current_target_drift = (
            None
            if current_target is None
            else self._xyz_distance(operation_target, current_target)
        )
        if (
            current_approach_drift is None
            or current_approach_drift
            > _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
            or current_target_drift is None
            or current_target_drift
            > _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
        ):
            return
        occlusion_proof = self._camera_ray_occlusion_proof(
            instance=instance,
            arm=arm,
            observed_xyz=observed,
            target_xyz=operation_target,
        )
        if occlusion_proof is None:
            return

        lease_key = (skill_id, instance_id, arm)
        self._partial_grounded_approach_leases[lease_key] = {
            "created_env_step": int(self.latest_snapshot.step_count),
            "observed_xyz": observed,
            "approach_world_m": approach,
            "target_world_m": operation_target,
            "target_point_key": target_point_key,
            "operation_action_mode": action_mode,
            "operation_candidate_id": candidate_id,
            "allowed_current_stability": current_stability,
            "residual_m": residual,
            "original_max_translation_m": original_max_translation,
            "error_history_m": error_history,
            **occlusion_proof,
        }
        payload = {
            "skill_id": skill_id,
            "instance_id": instance_id,
            "arm": arm,
            "created_env_step": int(self.latest_snapshot.step_count),
            "operation_action_mode": action_mode,
            "operation_candidate_id": candidate_id,
            "current_stability": current_stability,
            "residual_m": residual,
            "error_history_m": error_history,
            "camera": occlusion_proof["occlusion_camera"],
            "camera_ray_distance_m": occlusion_proof[
                "camera_ray_distance_m"
            ],
        }
        self._trace(
            "grounded_approach_continuation_lease_created",
            **payload,
        )
        self._dump_rollout_event(
            "grounded_approach_continuation_lease_created",
            **payload,
        )

    def _record_grounded_geometry_lease(
        self,
        *,
        call: RecoveryToolCall,
        details: dict[str, Any],
        key: tuple[str, str, str],
        later_calls: list[RecoveryToolCall],
    ) -> None:
        instance_id, arm, point_key = key
        if point_key != "approach_world_m" or not self._active_skill_id():
            return
        instance = self._grounded_instance_for_call(call)
        if instance is None:
            return
        action_mode = str(
            details.get(
                "operation_action_mode",
                (call.args or {}).get(
                    "_operation_action_mode",
                    instance.get("selected_operation_action_mode"),
                ),
            )
            or ""
        ).strip().lower()
        if action_mode not in {"contact", "grasp"}:
            if self._xyz_prefix(instance.get("contact_world_m")) is not None:
                action_mode = "contact"
            elif self._xyz_prefix(instance.get("grasp_world_m")) is not None:
                action_mode = "grasp"
        if action_mode not in {"contact", "grasp"}:
            return
        status = str(instance.get("status", "") or "").strip().lower()
        stability = str(instance.get("stability", "") or "").strip().lower()
        action_geometry_state = str(
            instance.get("action_geometry_state", "verified")
            or "verified"
        ).strip().lower()
        continued_partial_approach = (
            (call.args or {}).get(
                "_runtime_partial_approach_continuation"
            )
            is True
        )
        allowed_continuation_stabilities = {"missing_current_frame"}
        if action_mode == "grasp":
            allowed_continuation_stabilities.add(
                "geometry_inconsistent_current_frame"
            )
        if (
            status != "visible"
            or stability
            in {
                "geometry_inconsistent_current_frame",
                "relocation_geometry_pending",
                "identity_repair_expired",
            }
            or action_geometry_state
            in {
                "relocation_pending",
                "identity_repair_expired",
                "unavailable",
            }
        ) and not (
            continued_partial_approach
            and status == "tracked"
            and stability in allowed_continuation_stabilities
            and action_geometry_state
            not in {
                "relocation_pending",
                "identity_repair_expired",
                "unavailable",
            }
        ):
            return
        candidate_id = str(
            details.get(
                "operation_candidate_id",
                (call.args or {}).get(
                    "_operation_candidate_id",
                    instance.get("selected_operation_candidate_id"),
                ),
            )
            or ""
        ).strip()
        if action_mode == "grasp" and not candidate_id:
            # Geometry-inconsistent grasp recovery must remain tied to the exact
            # runtime-selected structured candidate that produced the approach.
            return
        for later in later_calls:
            if later.tool_name == "reobserve_scene":
                continue
            if later.tool_name in {
                "contact_displace",
                "lift_ee",
                "move_ee_to_grounded_instance",
                "move_ee_to_pose",
                "move_to_home",
                "retreat_arm",
                "safe_reset_posture",
            } and self._single_arm_from_call(later) in {arm, "both"}:
                return
        if self._recovery_holding_confirmed(arm):
            return
        current_scene = self.memory_store.state.working.scene_memory or {}
        current_matches: list[dict[str, Any]] = []
        for current_instance in current_scene.get("instances", []) or []:
            if not isinstance(current_instance, dict):
                continue
            refs = {
                str(current_instance.get(field, "") or "").strip().lower()
                for field in ("instance_id", "track_id", "oracle_id", "oracle_source_path")
                if str(current_instance.get(field, "") or "").strip()
            }
            if instance_id.strip().lower() in refs:
                current_matches.append(current_instance)
        if len(current_matches) != 1:
            return
        current_instance = current_matches[0]
        current_status = str(
            current_instance.get("status", "") or ""
        ).strip().lower()
        current_stability = str(
            current_instance.get("stability", "") or ""
        ).strip().lower()
        allowed_current_stabilities = {"missing_current_frame"}
        if action_mode == "grasp":
            allowed_current_stabilities.add(
                "geometry_inconsistent_current_frame"
            )
        if (
            current_status != "tracked"
            or current_stability not in allowed_current_stabilities
        ):
            return
        approach = self._xyz_prefix(instance.get("approach_world_m"))
        target_point_key = (
            "contact_world_m"
            if action_mode == "contact"
            else "grasp_world_m"
        )
        target = self._xyz_prefix(instance.get(target_point_key))
        observed = self._xyz_prefix(details.get("observed_pose", details.get("executed_pose")))
        if (
            approach is None
            or target is None
            or observed is None
            or self.latest_snapshot is None
        ):
            return
        if action_mode == "grasp":
            current_candidate = select_operation_pose_candidate(
                current_instance,
                arm=arm,
                action_mode=action_mode,
                requested_candidate_id=candidate_id,
            )
            if current_candidate is None:
                return
            current_materialized = materialize_operation_candidate(
                current_instance,
                current_candidate,
            )
            current_approach = self._xyz_prefix(
                current_materialized.get("approach_world_m")
            )
            current_target = self._xyz_prefix(
                current_materialized.get(target_point_key)
            )
            approach_drift = (
                None
                if current_approach is None
                else self._xyz_distance(approach, current_approach)
            )
            target_drift = (
                None
                if current_target is None
                else self._xyz_distance(target, current_target)
            )
            if (
                approach_drift is None
                or target_drift is None
                or approach_drift
                > _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
                or target_drift
                > _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
            ):
                return
        occlusion_proof = self._camera_ray_occlusion_proof(
            instance=instance,
            arm=arm,
            observed_xyz=observed,
            target_xyz=target,
        )
        if occlusion_proof is None:
            return
        lease_key = (self._active_skill_id(), instance_id, arm)
        self._grounded_geometry_leases[lease_key] = {
            "created_env_step": int(self.latest_snapshot.step_count),
            "observed_xyz": observed,
            "approach_world_m": approach,
            "target_world_m": target,
            "target_point_key": target_point_key,
            "operation_action_mode": action_mode,
            "operation_candidate_id": candidate_id,
            "allowed_current_stability": current_stability,
            **occlusion_proof,
        }
        if action_mode == "contact":
            # Preserve the existing contact-cycle lease payload consumed by
            # _stage_occlusion_geometry_leases.
            self._grounded_geometry_leases[lease_key][
                "contact_world_m"
            ] = target
        payload = {
            "skill_id": lease_key[0],
            "instance_id": instance_id,
            "arm": arm,
            "created_env_step": int(self.latest_snapshot.step_count),
            "operation_action_mode": action_mode,
            "operation_candidate_id": candidate_id,
            "allowed_current_stability": current_stability,
            "source": (
                "partial_approach_continuation"
                if continued_partial_approach
                else "reached_visible_approach"
            ),
            "camera": occlusion_proof["occlusion_camera"],
            "camera_ray_distance_m": occlusion_proof["camera_ray_distance_m"],
        }
        self._trace("grounded_geometry_lease_created", **payload)
        self._dump_rollout_event("grounded_geometry_lease_created", **payload)

    def _camera_ray_occlusion_proof(
        self,
        *,
        instance: dict[str, Any],
        arm: str,
        observed_xyz: list[float],
        target_xyz: list[float],
    ) -> dict[str, Any] | None:
        """Return an action-frame line-of-sight proxy for expected robot self-occlusion.

        The grounded contact point is an EE action-frame target rather than a reconstructed
        photometric surface point.  This is therefore a conservative authorization signal used
        together with visible-to-missing tracking, not a standalone visibility certificate.
        """

        snapshot = self.latest_snapshot
        camera = str(instance.get("camera", "") or "").strip().lower().replace("_camera", "")
        if (
            snapshot is None
            or camera not in {"head", "left", "right", "third"}
            or camera == arm
        ):
            return None
        transform = getattr(snapshot, f"{camera}_cam2world_gl", None)
        try:
            camera_origin = [float(transform[index][3]) for index in range(3)]
        except (TypeError, ValueError, IndexError):
            return None
        if not all(math.isfinite(value) for value in (*camera_origin, *observed_xyz, *target_xyz)):
            return None
        ray = [target_xyz[index] - camera_origin[index] for index in range(3)]
        ray_norm_sq = self._dot_xyz(ray, ray)
        if ray_norm_sq <= 1e-10:
            return None
        from_camera = [observed_xyz[index] - camera_origin[index] for index in range(3)]
        ray_fraction = self._dot_xyz(from_camera, ray) / ray_norm_sq
        if ray_fraction < 0.0 or ray_fraction > 1.05:
            return None
        closest = [camera_origin[index] + ray_fraction * ray[index] for index in range(3)]
        ray_distance = self._xyz_distance(observed_xyz, closest)
        try:
            occluder_radius = float(
                getattr(self.config, "scene_memory_robot_self_filter_radius_m", 0.08)
            )
        except (TypeError, ValueError):
            return None
        if (
            ray_distance is None
            or not math.isfinite(occluder_radius)
            or occluder_radius <= 0.0
            or ray_distance > occluder_radius
        ):
            return None
        return {
            "occlusion_camera": camera,
            "camera_ray_distance_m": round(ray_distance, 6),
            "camera_ray_fraction": round(ray_fraction, 6),
        }

    def _blocked_grounded_setup_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for instance_id, arm, point_key in sorted(self._blocked_grounded_setups):
            details = self._grounded_setup_failures.get((instance_id, arm, point_key), {})
            payload.append(
                {
                    "instance_id": instance_id,
                    "arm": arm,
                    "point_key": point_key,
                    "failure_count": int(details.get("failure_count", _GROUNDED_SETUP_FAILURE_THRESHOLD)),
                    "last_target_error_m": details.get("last_target_error_m"),
                    "status": "blocked_after_repeated_no_progress",
                }
            )
        scene_memory = self.memory_store.state.working.scene_memory
        instances = (
            scene_memory.get("instances", [])
            if isinstance(scene_memory, dict)
            else []
        )
        payload.extend(
            self._operation_candidate_lifecycle.planner_payload(
                instances
            )
        )
        return payload

    def _history_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, (int, bool)):
            return str(value)
        if isinstance(value, (list, tuple)):
            if len(value) > 4:
                return ""
            return "[" + ",".join(self._history_value(item) for item in value) + "]"
        text = str(value).strip()
        if not text:
            return ""
        text = " ".join(text.split())
        return text[:120]

    def _with_internal_recovery_context(self, calls: list[RecoveryToolCall]) -> list[RecoveryToolCall]:
        scene_memory = self.memory_store.state.working.scene_memory
        observation_preprocess = self.memory_store.state.working.observation_preprocess
        operation_modes = self._infer_operation_action_modes(calls)
        enriched: list[RecoveryToolCall] = []
        for index, call in enumerate(calls):
            args = dict(call.args or {})
            args.pop("_runtime_post_contact_clearance", None)
            args.pop("_runtime_failed_grasp_clearance", None)
            args.pop("_runtime_grasp_close_boundary", None)
            args.pop("_runtime_grasp_diagnostic_lift", None)
            args.pop("_runtime_ambiguous_grasp_return", None)
            args.pop("_runtime_occlusion_geometry_lease", None)
            args.pop("_runtime_partial_approach_continuation", None)
            args.pop("_runtime_place_motion_authorized", None)
            args.pop("_runtime_memory_valid_final_action", None)
            args.pop("_runtime_visual_clearance", None)
            args.pop(
                "_runtime_action_geometry_repair_safe_motion",
                None,
            )
            args.pop(
                "_runtime_action_geometry_repair_pending_bypass",
                None,
            )
            args.pop(
                "_runtime_action_geometry_repair_retreat",
                None,
            )
            args.pop("_scene_memory", None)
            args.pop("_observation_preprocess", None)
            args.pop("_operation_action_mode", None)
            args.pop("_operation_candidate_id", None)
            args.pop("_blocked_operation_candidate_ids", None)
            args.pop("_operation_candidate_attempt", None)
            args.pop("_release_place_candidate", None)
            args.pop("_release_place_validation", None)
            args.pop("_runtime_support_contact_release", None)
            args.pop("_support_contact_release_settle_dropped", None)
            args.pop("_actual_spatial_state_signature", None)
            args.pop("_evidence_acquisition", None)
            args.pop("_evidence_situation", None)
            args.pop("_evidence_track_id", None)
            args.pop("_evidence_arm", None)
            args.pop("_evidence_phase", None)
            if call.tool_name == "move_ee_to_grounded_instance":
                args = self._normalize_grounded_identity_args(
                    args,
                    scene_memory=scene_memory,
                )
                # These are runtime-owned snapshots.  Planner-provided private context must
                # never select or authorize geometry.
                args["_scene_memory"] = dict(scene_memory or {})
                args["_observation_preprocess"] = dict(observation_preprocess or {})
                point_key = self._normalized_grounded_point_key(
                    RecoveryToolCall(tool_name=call.tool_name, args=args)
                )
                if not uses_operation_pose_candidate(point_key):
                    enriched.append(
                        RecoveryToolCall(tool_name=call.tool_name, args=args)
                    )
                    continue
                action_mode = operation_modes.get(
                    index,
                    operation_action_mode(point_key),
                )
                args["_operation_action_mode"] = action_mode
                unresolved_call = RecoveryToolCall(tool_name=call.tool_name, args=args)
                instance = self._raw_grounded_instance_for_call(unresolved_call)
                arm = self._single_arm_from_call(unresolved_call)
                if instance is not None and arm in {"left", "right"}:
                    instance_id = str(instance.get("instance_id", "") or "").strip()
                    blocked_ids = self._blocked_candidate_ids(
                        instance_id=instance_id,
                        arm=arm,
                        action_mode=action_mode,
                        instance=instance,
                        requested_target_id=args.get("target_id"),
                        point_key=point_key,
                        offset_xyz=args.get("offset_xyz"),
                        offset_xyz_provided=(
                            "offset_xyz" in args
                        ),
                        preserve_height=args.get(
                            "preserve_height",
                            False,
                        ),
                        target_quat_wxyz=args.get(
                            "target_quat_wxyz",
                            args.get("quat_wxyz"),
                        ),
                        current_pose=self._latest_arm_endpose(arm),
                    )
                    args["_blocked_operation_candidate_ids"] = blocked_ids
                    selected = select_operation_pose_candidate(
                        instance,
                        arm=arm,
                        action_mode=action_mode,
                        blocked_candidate_ids=blocked_ids,
                        requested_target_id=args.get("target_id"),
                    )
                    if selected is not None:
                        args["_operation_candidate_id"] = str(
                            selected.get("candidate_id", "")
                        )
                        observation = args.get(
                            "_observation_preprocess"
                        )
                        if not isinstance(observation, dict):
                            observation = {}
                        attempt_kwargs: dict[str, Any] = {
                            "instance_id": instance_id,
                            "candidate": selected,
                            "instance": instance,
                            "point_key": point_key,
                            "preserve_height": args.get(
                                "preserve_height",
                                False,
                            ),
                            "target_quat_wxyz": args.get(
                                "target_quat_wxyz",
                                args.get("quat_wxyz"),
                            ),
                            "current_pose": (
                                self._latest_arm_endpose(arm)
                            ),
                            "dispatch_env_step": (
                                int(self.latest_snapshot.step_count)
                                if self.latest_snapshot is not None
                                else None
                            ),
                            "observation_generation": observation.get(
                                "observation_generation"
                            ),
                            "observation_capture_id": observation.get(
                                "observation_capture_id"
                            ),
                        }
                        if "offset_xyz" in args:
                            attempt_kwargs["offset_xyz"] = args[
                                "offset_xyz"
                            ]
                        args["_operation_candidate_attempt"] = (
                            self._operation_candidate_lifecycle.bind_attempt(
                                **attempt_kwargs
                            )
                        )
            elif (
                call.tool_name == "contact_displace"
                and args.get("complete_transient_cycle") is True
            ):
                spatial_state = (
                    scene_memory.get(SPATIAL_STATE_KEY)
                    if isinstance(scene_memory, dict)
                    else None
                )
                signature = (
                    str(spatial_state.get("signature", "") or "").strip()
                    if isinstance(spatial_state, dict)
                    else ""
                )
                if signature:
                    args["_actual_spatial_state_signature"] = signature
            enriched.append(RecoveryToolCall(tool_name=call.tool_name, args=args))
        failed_grasp_cleared = stage_failed_grasp_clearance(
            enriched,
            manipulation_state=(
                self.memory_store.state.working.manipulation_state
            ),
            robot_state=self._get_robot_state(),
            grasp_transport_policy=self._grasp_transport_policy(),
            release_guard_enabled=self._release_guard_enabled(),
        )
        grasp_boundary_staged = stage_grasp_verification_boundary(
            failed_grasp_cleared,
            manipulation_state=(
                self.memory_store.state.working.manipulation_state
            ),
            robot_state=self._get_robot_state(),
            grasp_transport_policy=self._grasp_transport_policy(),
            release_guard_enabled=self._release_guard_enabled(),
        )
        approach_continued = (
            self._stage_partial_grounded_approach_continuation(
                grasp_boundary_staged
            )
        )
        transient_guarded = self._stage_transient_contact_cycles(
            approach_continued
        )
        leased = self._stage_occlusion_geometry_leases(transient_guarded)
        visibility_guarded = [
            guarded_call
            for call in leased
            for guarded_call in self._guard_occluded_grounded_instance(call)
        ]
        blocked_guarded = [self._guard_blocked_grounded_setup(call) for call in visibility_guarded]
        return self._apply_recovery_manipulation_guards(self._stage_active_contact_sequences(blocked_guarded))

    def _apply_evidence_acquisition_policy(
        self,
        calls: list[RecoveryToolCall],
        *,
        planner_explicit_open: bool = False,
    ) -> list[RecoveryToolCall]:
        """Bound only stationary observation batches by physical state.

        A batch containing a physical tool is already information-changing:
        RMBench returns a fresh snapshot after the action, so it must not
        consume the stationary retry allowance.  Planner-issued and
        guard-generated observation-only batches meet here after all other
        guards, which makes the budget independent of skill/replan names.
        """

        staged = list(calls)
        # This is a one-dispatch signal.  Clear it before evaluating the
        # current guarded batch so an earlier reducer completion cannot be
        # mistaken for progress by a later empty plan.
        self._pending_internal_recovery_completion = ""
        reobserve_enabled = bool(
            self._recovery_dispatcher.reobserve_scene_enabled
        )
        evidence_only = self._grasp_transport_evidence_only()
        manipulation_state = (
            self.memory_store.state.working.manipulation_state
        )
        pending_return = isinstance(manipulation_state, dict) and any(
            isinstance(state, dict)
            and str(state.get("phase", "") or "").strip().lower()
            == "ambiguous_grasp_return_pending"
            for state in manipulation_state.values()
        )
        preserve_explicit_open = bool(
            planner_explicit_open
            and not self._release_guard_enabled()
        )
        if (
            pending_return
            and not evidence_only
            and not preserve_explicit_open
        ):
            return self._stage_ambiguous_grasp_return(
                evidence_decision=dict(
                    self._last_evidence_acquisition_decision
                ),
                include_reobserve=reobserve_enabled,
            )
        observation_calls = [
            call
            for call in staged
            if call.tool_name == REOBSERVE_SCENE_TOOL
        ]
        if not observation_calls:
            return staged
        physical_calls = [
            call
            for call in staged
            if call.tool_name != REOBSERVE_SCENE_TOOL
        ]
        if physical_calls:
            if reobserve_enabled:
                return staged
            filtered = [
                call
                for call in staged
                if call.tool_name != REOBSERVE_SCENE_TOOL
            ]
            self._record_evidence_acquisition_decision(
                {
                    "allow_reobserve": False,
                    "next_information_action": (
                        "use_fresh_snapshot_from_physical_action"
                    ),
                    "exhausted_reason": "disabled_by_ablation",
                    "evidence_situation": "physical_action_batch",
                    "physical_tools": [
                        call.tool_name for call in physical_calls
                    ],
                    "reobserve_scene_enabled": False,
                }
            )
            return filtered

        observation = observation_calls[0]
        if not reobserve_enabled:
            context, situation, evidence_sufficient = (
                self._evidence_acquisition_context(observation)
            )
            if evidence_only and situation in {
                EvidenceSituation.AMBIGUOUS_GRASP,
                EvidenceSituation.PROVEN_FAILED_GRASP,
            }:
                self._record_evidence_acquisition_decision(
                    {
                        "allow_reobserve": False,
                        "next_information_action": (
                            "continue_without_attachment_guard"
                        ),
                        "exhausted_reason": "disabled_by_ablation",
                        "evidence_situation": situation.value,
                        "reobserve_scene_enabled": False,
                        "grasp_transport_policy": "evidence_only",
                    }
                )
                return []
            if (
                situation is EvidenceSituation.AMBIGUOUS_GRASP
                and self._grasp_diagnostic_lift_evidence_missing(context)
            ):
                decision = self._diagnostic_lift_required_decision(
                    context,
                    reobserve_enabled=False,
                )
                self._record_evidence_acquisition_decision(decision)
                return []
            if situation is EvidenceSituation.AMBIGUOUS_GRASP:
                decision = {
                    "allow_reobserve": False,
                    "next_information_action": (
                        "perform_controlled_return_and_regrasp"
                    ),
                    "exhausted_reason": (
                        "same_state_ambiguous_grasp_reobserve_budget_exhausted"
                    ),
                    "evidence_situation": situation.value,
                    "state_key": (
                        self._evidence_acquisition_policy.state_key(
                            context
                        ).as_dict()
                    ),
                    "stationary_reobserves_used": 0,
                    "stationary_reobserve_limit": 0,
                    "distinct_capture_count": 0,
                    "distinct_capture_limit": 0,
                    "skill_id": self._active_skill_id(),
                    "capture_id": context.capture_id,
                    "physical_action_token": (
                        context.physical_action_token
                    ),
                    "viewpoint_token": context.viewpoint_token,
                    "reobserve_scene_enabled": False,
                    "ablation_zero_observation_budget": True,
                }
                return self._stage_ambiguous_grasp_return(
                    evidence_decision=decision,
                    include_reobserve=False,
                )
            self._record_evidence_acquisition_decision(
                {
                    "allow_reobserve": False,
                    "next_information_action": (
                        "replan_without_stationary_reobserve"
                    ),
                    "exhausted_reason": "disabled_by_ablation",
                    "evidence_situation": "stationary_observation",
                    "reobserve_scene_enabled": False,
                }
            )
            return []

        context, situation, evidence_sufficient = (
            self._evidence_acquisition_context(observation)
        )
        decision = self._evidence_acquisition_policy.decide(
            context,
            situation=situation,
            evidence_sufficient=evidence_sufficient,
        )
        if (
            not evidence_only
            and
            situation is EvidenceSituation.AMBIGUOUS_GRASP
            and self._grasp_diagnostic_lift_evidence_missing(context)
            and decision.get("next_information_action")
            == "perform_controlled_return_and_regrasp"
        ):
            # A return is meaningful only after the bounded diagnostic lift
            # produced attempt-local evidence.  Before that boundary, budget
            # exhaustion asks the next planner turn for the diagnostic lift;
            # it must not manufacture an unauthorized controlled return.
            decision = {
                **decision,
                "allow_reobserve": False,
                "next_information_action": (
                    "perform_bounded_grasp_verification_motion"
                ),
                "exhausted_reason": (
                    "diagnostic_lift_evidence_missing"
                ),
                "diagnostic_lift_required": True,
            }
        decision.update(
            {
                "skill_id": self._active_skill_id(),
                "capture_id": context.capture_id,
                "physical_action_token": context.physical_action_token,
                "viewpoint_token": context.viewpoint_token,
                "reobserve_scene_enabled": True,
            }
        )
        if decision.get("allow_reobserve") is not True:
            if evidence_only and situation in {
                EvidenceSituation.AMBIGUOUS_GRASP,
                EvidenceSituation.PROVEN_FAILED_GRASP,
            }:
                decision.update(
                    {
                        "next_information_action": (
                            "continue_without_attachment_guard"
                        ),
                        "grasp_transport_policy": "evidence_only",
                    }
                )
                self._record_evidence_acquisition_decision(decision)
                return []
            if decision.get("next_information_action") == (
                "perform_controlled_return_and_regrasp"
            ):
                return self._stage_ambiguous_grasp_return(
                    evidence_decision=decision,
                    include_reobserve=True,
                )
            self._record_evidence_acquisition_decision(decision)
            return []
        self._record_evidence_acquisition_decision(decision)
        args = dict(observation.args or {})
        args["_evidence_acquisition"] = dict(decision)
        return [
            RecoveryToolCall(
                tool_name=REOBSERVE_SCENE_TOOL,
                args=args,
            )
        ]

    def _grasp_diagnostic_lift_evidence_missing(
        self,
        context: EvidenceAcquisitionContext,
    ) -> bool:
        arm = str(context.arm or "").strip().lower()
        if arm not in {"left", "right"}:
            return False
        manipulation = (
            self.memory_store.state.working.manipulation_state
        )
        arm_state = (
            manipulation.get(arm)
            if isinstance(manipulation, dict)
            else None
        )
        return bool(
            isinstance(arm_state, dict)
            and str(arm_state.get("phase", "") or "").strip().lower()
            == "grasp_candidate"
            and not isinstance(
                arm_state.get("diagnostic_lift_evidence"),
                dict,
            )
        )

    def _diagnostic_lift_required_decision(
        self,
        context: EvidenceAcquisitionContext,
        *,
        reobserve_enabled: bool,
    ) -> dict[str, Any]:
        return {
            "allow_reobserve": False,
            "next_information_action": (
                "perform_bounded_grasp_verification_motion"
            ),
            "exhausted_reason": "diagnostic_lift_evidence_missing",
            "evidence_situation": EvidenceSituation.AMBIGUOUS_GRASP.value,
            "state_key": (
                self._evidence_acquisition_policy.state_key(context).as_dict()
            ),
            "stationary_reobserves_used": 0,
            "stationary_reobserve_limit": (
                0 if not reobserve_enabled else 1
            ),
            "distinct_capture_count": 0,
            "distinct_capture_limit": 0,
            "skill_id": self._active_skill_id(),
            "capture_id": context.capture_id,
            "physical_action_token": context.physical_action_token,
            "viewpoint_token": context.viewpoint_token,
            "reobserve_scene_enabled": bool(reobserve_enabled),
            "diagnostic_lift_required": True,
        }

    def _stage_ambiguous_grasp_return(
        self,
        *,
        evidence_decision: dict[str, Any],
        include_reobserve: bool,
    ) -> list[RecoveryToolCall]:
        plan = plan_ambiguous_grasp_return(
            manipulation_state=(
                self.memory_store.state.working.manipulation_state
            ),
            robot_state=self._get_robot_state(),
            evidence_decision=evidence_decision,
            include_reobserve=include_reobserve,
        )
        zero_call_reduction = None
        if plan.authorized and not plan.calls:
            zero_call_reduction = reduce_ambiguous_grasp_return_results(
                self.memory_store.state.working.manipulation_state,
                calls=[],
                results=[],
                post_return_snapshot=self.latest_snapshot,
            )
            if zero_call_reduction.attempt_completed:
                self.memory_store.state.working.manipulation_state = {
                    str(arm): dict(state)
                    for arm, state in (
                        zero_call_reduction.manipulation_state
                    ).items()
                    if isinstance(state, dict)
                }
                self._sync_runtime_manipulation_state_to_scene_memory()
                self._pending_internal_recovery_completion = str(
                    zero_call_reduction.reason
                    or "ambiguous_grasp_return_completed"
                )
        decision = {
            **dict(evidence_decision or {}),
            "controlled_return_applicable": plan.applicable,
            "controlled_return_authorized": plan.authorized,
            "controlled_return_reason": plan.reason,
            "controlled_return_arm": plan.arm,
            "controlled_return_held_instance_id": (
                plan.held_instance_id
            ),
            "controlled_return_grasp_attempt_nonce": (
                plan.grasp_attempt_nonce
            ),
            "controlled_return_tools": [
                call.tool_name for call in plan.calls
            ],
            "controlled_return_zero_call_completion": bool(
                zero_call_reduction is not None
                and zero_call_reduction.attempt_completed
            ),
            "controlled_return_zero_call_reason": (
                zero_call_reduction.reason
                if zero_call_reduction is not None
                else ""
            ),
            "reobserve_scene_enabled": bool(include_reobserve),
        }
        self._record_evidence_acquisition_decision(decision)
        return list(plan.calls) if plan.authorized else []

    def _record_evidence_acquisition_decision(
        self,
        decision: dict[str, Any],
    ) -> None:
        payload = dict(decision or {})
        self._last_evidence_acquisition_decision = payload
        self.memory_store.record_recovery(
            "evidence_acquisition_policy:"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
        self._trace("evidence_acquisition_policy", **payload)
        self._dump_rollout_event(
            "evidence_acquisition_policy",
            **payload,
        )

    def _current_stationary_observation_budget(
        self,
    ) -> dict[str, Any]:
        """Return the current scene retry budget without spending it."""

        observation = RecoveryToolCall(
            tool_name=REOBSERVE_SCENE_TOOL,
            args={},
        )
        context, situation, _ = self._evidence_acquisition_context(
            observation
        )
        status = (
            self._evidence_acquisition_policy
            .stationary_scene_budget_status(
                context,
                situation=situation,
            )
        )
        return {
            **status,
            "evidence_situation": situation.value,
            "physical_action_token": context.physical_action_token,
            "viewpoint_token": context.viewpoint_token,
            "camera_set": list(context.camera_set),
        }

    def _evidence_acquisition_context(
        self,
        observation: RecoveryToolCall,
    ) -> tuple[
        EvidenceAcquisitionContext,
        EvidenceSituation,
        bool,
    ]:
        args = dict(observation.args or {})
        manipulation = (
            self.memory_store.state.working.manipulation_state
        )
        states = manipulation if isinstance(manipulation, dict) else {}
        explicit_arm = normalize_physical_arm(
            str(args.get("_evidence_arm", "") or "").strip().lower()
        )
        arm = explicit_arm if explicit_arm in {"left", "right"} else ""
        arm_state = states.get(arm) if arm else None
        if not isinstance(arm_state, dict):
            candidates = [
                (candidate_arm, state)
                for candidate_arm, state in states.items()
                if candidate_arm in {"left", "right"}
                and isinstance(state, dict)
                and str(state.get("phase", "") or "").strip()
                in {
                    "grasp_candidate",
                    "failed_grasp_clearance_pending",
                    "release_pending_verification",
                    "release_recovery_required",
                }
            ]
            if len(candidates) == 1:
                arm, arm_state = candidates[0]
            else:
                arm_state = None
        if not isinstance(arm_state, dict):
            arm_state = {}
        normalized_arm = arm or "scene"
        track_id = str(
            args.get("_evidence_track_id", "")
            or arm_state.get("held_instance_id", "")
            or arm_state.get("released_instance_id", "")
            or "__scene__"
        ).strip()
        phase = str(
            args.get("_evidence_phase", "")
            or arm_state.get("phase", "")
            or "scene_observation"
        ).strip().lower()
        pose = self._evidence_acquisition_ee_pose(
            normalized_arm
        )
        cameras = self._evidence_acquisition_cameras()
        preprocess = (
            self.memory_store.state.working.observation_preprocess
        )
        capture_id = (
            preprocess.get("observation_capture_id")
            if isinstance(preprocess, dict)
            else None
        )
        context = EvidenceAcquisitionContext(
            track_id=track_id,
            arm=normalized_arm,
            manipulation_phase=phase,
            grasp_attempt_nonce=str(
                arm_state.get("grasp_attempt_nonce", "") or ""
            ).strip(),
            ee_pose=pose,
            camera_set=cameras,
            capture_id=capture_id,
            skill_id=self._active_skill_id(),
            physical_action_token=(
                int(self.latest_snapshot.step_count)
                if self.latest_snapshot is not None
                else None
            ),
            viewpoint_token=self._evidence_viewpoint_token(cameras),
        )
        explicit_situation = str(
            args.get("_evidence_situation", "") or ""
        ).strip().lower()
        try:
            situation = EvidenceSituation(explicit_situation)
        except ValueError:
            situation = EvidenceSituation.TRANSIENT_BACKEND_RETRY
        evidence_sufficient = False
        if phase == "release_pending_verification":
            situation = EvidenceSituation.RELEASE_STABILITY
        elif phase == "grasp_candidate":
            validation = self._pending_grasp_runtime_validation(
                arm_state
            )
            if validation.get("verified") is False:
                situation = EvidenceSituation.PROVEN_FAILED_GRASP
            elif validation.get("verified") is True:
                situation = EvidenceSituation.STRONG_GRASP_EVIDENCE
                evidence_sufficient = True
            elif explicit_situation != EvidenceSituation.OCCLUSION.value:
                situation = EvidenceSituation.AMBIGUOUS_GRASP
        return context, situation, evidence_sufficient

    @staticmethod
    def _pending_grasp_runtime_validation(
        arm_state: dict[str, Any],
    ) -> dict[str, Any]:
        diagnostic = arm_state.get("diagnostic_lift_evidence")
        if not isinstance(diagnostic, dict):
            return {}
        validation = diagnostic.get("runtime_grasp_validation")
        return dict(validation) if isinstance(validation, dict) else {}

    def _evidence_acquisition_ee_pose(
        self,
        arm: str,
    ) -> list[float] | None:
        if arm not in {"left", "right"}:
            return None
        robot_state = self._get_robot_state()
        arm_robot = (
            robot_state.get(arm)
            if isinstance(robot_state, dict)
            else None
        )
        if not isinstance(arm_robot, dict):
            return None
        xyz = self._xyz_prefix(arm_robot.get("xyz"))
        quaternion = arm_robot.get("quat_wxyz")
        if xyz is None or not isinstance(quaternion, (list, tuple)):
            return xyz
        try:
            quat = [float(value) for value in quaternion[:4]]
        except (TypeError, ValueError):
            return xyz
        if len(quat) != 4 or not all(math.isfinite(value) for value in quat):
            return xyz
        return [*xyz, *quat]

    def _evidence_acquisition_cameras(self) -> tuple[str, ...]:
        preprocess = (
            self.memory_store.state.working.observation_preprocess
        )
        cameras: set[str] = set()
        segmentation = (
            preprocess.get("segmentation", [])
            if isinstance(preprocess, dict)
            else []
        )
        if isinstance(segmentation, list):
            for item in segmentation:
                if not isinstance(item, dict):
                    continue
                camera = str(item.get("camera", "") or "").strip().lower()
                if camera:
                    cameras.add(camera)
        if not cameras:
            cameras.update(self._observation_preprocess_cameras())
        return tuple(sorted(cameras))

    def _evidence_viewpoint_token(
        self,
        cameras: tuple[str, ...],
    ) -> str:
        snapshot = self.latest_snapshot
        if snapshot is None:
            return ""
        values: list[tuple[str, tuple[int, ...]]] = []
        for camera in cameras:
            transform = getattr(
                snapshot,
                f"{camera}_cam2world_gl",
                None,
            )
            if transform is None:
                continue
            raw = transform.tolist() if hasattr(transform, "tolist") else transform
            try:
                flattened = [
                    float(value)
                    for row in raw
                    for value in row
                ]
            except (TypeError, ValueError):
                continue
            if not flattened or not all(
                math.isfinite(value) for value in flattened
            ):
                continue
            values.append(
                (
                    camera,
                    tuple(int(round(value * 1000.0)) for value in flattened),
                )
            )
        if not values:
            return ""
        return hashlib.sha256(
            json.dumps(values, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

    def _normalize_grounded_identity_args(
        self,
        args: dict[str, Any],
        *,
        scene_memory: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(args or {})
        if not isinstance(scene_memory, dict):
            return normalized

        public_explicit_ref = str(
            normalized.get("instance_id")
            or normalized.get("instance_ref")
            or ""
        ).strip()
        explicit_ref = public_explicit_ref
        if not explicit_ref:
            for alias in (
                "destination_instance_id",
                "destination_instance_ref",
                "target_instance_id",
                "target_instance_ref",
            ):
                alias_ref = str(normalized.get(alias, "") or "").strip()
                if alias_ref:
                    explicit_ref = alias_ref
                    break
        if explicit_ref and not public_explicit_ref:
            canonical = self._canonical_scene_instance_id(
                scene_memory,
                explicit_ref,
            )
            normalized["instance_id"] = canonical or explicit_ref

        arm = normalize_physical_arm(normalized.get("arm"))
        preserve_height = normalized.get("preserve_height") is True
        held_state = self._transport_holding_state(arm)
        held_ref = str(
            normalized.get("held_instance_id")
            or normalized.get("held_instance_ref")
            or (held_state or {}).get("held_instance_id")
            or ""
        ).strip()
        held_instance_id = (
            self._canonical_scene_instance_id(scene_memory, held_ref)
            if held_ref
            else ""
        )
        point_key = self._normalized_grounded_point_key(
            RecoveryToolCall(
                tool_name="move_ee_to_grounded_instance",
                args=normalized,
            )
        )
        requested_mode = str(
            normalized.get("action_mode", "") or ""
        ).strip().lower()
        is_place = (
            requested_mode == "place"
            or point_key == "place_world_m"
        )
        if is_place and held_instance_id:
            # Public placement selects a target group.  Runtime binds the
            # executable geometry to the runtime-authorized held object.
            normalized["action_mode"] = "place"
            normalized["held_instance_id"] = held_instance_id
            normalized["instance_id"] = held_instance_id
            normalized.pop("instance_ref", None)
            return normalized
        if preserve_height and held_instance_id:
            normalized["held_instance_id"] = held_instance_id

        if explicit_ref:
            return normalized

        task_focus = scene_memory.get("task_focus")
        identity_binding_required = bool(
            isinstance(task_focus, dict)
            and task_focus.get("identity_binding_required")
        )
        if not identity_binding_required and not preserve_height:
            return normalized

        focus_key = str(normalized.get("focus_key", "") or "").strip()
        if not focus_key:
            role = str(normalized.get("role", "target") or "target").strip().lower()
            focus_key = (
                "tool_instances"
                if role == "tool"
                else "target_instances"
            )
        candidate_ids = self._focused_scene_instance_ids(
            scene_memory,
            focus_keys=(focus_key,),
        )
        if preserve_height and held_instance_id:
            candidate_ids = [
                item for item in candidate_ids if item != held_instance_id
            ]
            if len(candidate_ids) != 1:
                candidate_ids = [
                    item
                    for item in self._focused_scene_instance_ids(
                        scene_memory,
                        focus_keys=("target_instances", "tool_instances"),
                    )
                    if item != held_instance_id
                ]
        if len(candidate_ids) == 1:
            normalized["instance_id"] = candidate_ids[0]
        return normalized

    def _focused_scene_instance_ids(
        self,
        scene_memory: dict[str, Any],
        *,
        focus_keys: tuple[str, ...],
    ) -> list[str]:
        task_focus = scene_memory.get("task_focus")
        if not isinstance(task_focus, dict):
            return []
        resolved: list[str] = []
        for focus_key in focus_keys:
            values = task_focus.get(focus_key)
            if not isinstance(values, list):
                continue
            for value in values:
                canonical = self._canonical_scene_instance_id(
                    scene_memory,
                    str(value or ""),
                )
                if canonical and canonical not in resolved:
                    resolved.append(canonical)
        return resolved

    def _canonical_scene_instance_id(
        self,
        scene_memory: dict[str, Any],
        instance_ref: str,
    ) -> str:
        normalized_ref = str(instance_ref or "").strip().lower()
        if not normalized_ref:
            return ""
        matches: list[str] = []
        for instance in scene_memory.get("instances", []) or []:
            if not isinstance(instance, dict):
                continue
            identity_refs = {
                str(instance.get(key, "") or "").strip().lower()
                for key in (
                    "instance_id",
                    "track_id",
                    "oracle_id",
                    "oracle_source_path",
                )
                if str(instance.get(key, "") or "").strip()
            }
            if normalized_ref not in identity_refs:
                continue
            instance_id = str(instance.get("instance_id", "") or "").strip()
            if instance_id and instance_id not in matches:
                matches.append(instance_id)
        return matches[0] if len(matches) == 1 else ""

    def _infer_operation_action_modes(
        self,
        calls: list[RecoveryToolCall],
    ) -> dict[int, str]:
        modes: dict[int, str] = {}
        for index, call in enumerate(calls):
            if call.tool_name != "move_ee_to_grounded_instance":
                continue
            point_key = self._normalized_grounded_point_key(call)
            explicit_mode = str(
                (call.args or {}).get("action_mode", "") or ""
            ).strip().lower()
            if explicit_mode in SUPPORTED_OPERATION_ACTION_MODES:
                modes[index] = explicit_mode
                continue
            if point_key in {
                "contact_world_m",
                "grasp_world_m",
                "place_world_m",
            }:
                modes[index] = operation_action_mode(point_key)
                continue
            if point_key != "approach_world_m":
                continue
            arm = self._single_arm_from_call(call)
            inferred = "grasp"
            for later in calls[index + 1 :]:
                if later.tool_name == "reobserve_scene":
                    break
                if self._single_arm_from_call(later) != arm:
                    continue
                if later.tool_name == "contact_displace":
                    inferred = "contact"
                    break
                if later.tool_name != "move_ee_to_grounded_instance":
                    continue
                later_key = self._normalized_grounded_point_key(later)
                later_explicit_mode = str(
                    (later.args or {}).get("action_mode", "") or ""
                ).strip().lower()
                if later_explicit_mode in SUPPORTED_OPERATION_ACTION_MODES:
                    inferred = later_explicit_mode
                    break
                if later_key in {
                    "contact_world_m",
                    "grasp_world_m",
                    "place_world_m",
                }:
                    inferred = operation_action_mode(later_key)
                    break
            modes[index] = inferred
        return modes

    def _stage_transient_contact_cycles(self, calls: list[RecoveryToolCall]) -> list[RecoveryToolCall]:
        """Return a transient contact arm to its grounded approach before re-observation.

        A transient contact cycle is not complete while the end effector remains at the contact
        pose.  Besides leaving the mechanism actuated, that posture can occlude the same target and
        make every later grounded retry fail the temporal-geometry guard.  Only a cycle explicitly
        marked in structured tool args is completed here; unmarked calls keep their existing
        contact semantics.
        """

        staged: list[RecoveryToolCall] = []
        grounded_contact_by_arm: dict[str, RecoveryToolCall] = {}
        for index, raw_call in enumerate(calls):
            call = raw_call
            arm = self._single_arm_from_call(call)
            if call.tool_name == "move_ee_to_grounded_instance":
                if arm in {"left", "right"} and self._normalized_grounded_point_key(call) == "contact_world_m":
                    grounded_contact_by_arm[arm] = call
                elif arm in {"left", "right"}:
                    grounded_contact_by_arm.pop(arm, None)
                staged.append(call)
                continue

            if call.tool_name != "contact_displace" or arm not in {"left", "right"}:
                if call.tool_name in {
                    "lift_ee",
                    "move_ee_to_pose",
                    "move_to_home",
                    "retreat_arm",
                    "safe_reset_posture",
                }:
                    if arm == "both":
                        grounded_contact_by_arm.clear()
                    else:
                        grounded_contact_by_arm.pop(arm, None)
                staged.append(call)
                continue

            if not self._requests_complete_transient_contact_cycle(call):
                staged.append(call)
                continue

            setup = grounded_contact_by_arm.get(arm)
            instance = self._grounded_instance_for_call(setup) if setup is not None else None
            approach = self._xyz_prefix(instance.get("approach_world_m")) if instance is not None else None
            contact = self._xyz_prefix(instance.get("contact_world_m")) if instance is not None else None
            if (
                setup is None
                or approach is None
                or contact is None
                or not self._contact_displacement_follows_approach(call, approach=approach, contact=contact)
            ):
                staged.append(
                    RecoveryToolCall(
                        tool_name="reobserve_scene",
                        args={
                            "_guard_reason": (
                                "blocked transient contact cycle without a same-batch grounded contact setup; "
                                "refresh grounding before another count-sensitive actuation"
                            )
                        },
                    )
                )
                grounded_contact_by_arm.pop(arm, None)
                continue

            contact_args = dict(call.args or {})
            contact_args["complete_transient_cycle"] = True
            staged.append(RecoveryToolCall(tool_name=call.tool_name, args=contact_args))
            if not self._has_grounded_transient_release(calls[index + 1 :], setup=setup):
                staged.append(
                    self._grounded_transient_release_call(
                        setup,
                        actuation=call,
                        approach=approach,
                        contact=contact,
                    )
                )
            grounded_contact_by_arm.pop(arm, None)
        return self._stage_transient_visual_clearance(staged)

    def _requests_complete_transient_contact_cycle(self, call: RecoveryToolCall) -> bool:
        return (call.args or {}).get("complete_transient_cycle") is True

    def _has_grounded_transient_release(
        self,
        later_calls: list[RecoveryToolCall],
        *,
        setup: RecoveryToolCall,
    ) -> bool:
        setup_arm = self._single_arm_from_call(setup)
        setup_instance = self._recovery_history_instance_id(setup, {})
        for call in later_calls:
            if call.tool_name == "reobserve_scene":
                return False
            if self._is_grounded_transient_release(
                call,
                setup_arm=setup_arm,
                setup_instance=setup_instance,
            ):
                return True
        return False

    def _is_grounded_transient_release(
        self,
        call: RecoveryToolCall,
        *,
        setup_arm: str,
        setup_instance: str,
    ) -> bool:
        if call.tool_name != "move_ee_to_grounded_instance":
            return False
        if self._single_arm_from_call(call) != setup_arm:
            return False
        if self._normalized_grounded_point_key(call) != "approach_world_m":
            return False
        if self._recovery_history_instance_id(call, {}) != setup_instance:
            return False
        offset = self._xyz_prefix((call.args or {}).get("offset_xyz", [0.0, 0.0, 0.0]))
        return offset is not None and sum(value * value for value in offset) <= 1e-10

    def _grounded_transient_release_call(
        self,
        setup: RecoveryToolCall,
        *,
        actuation: RecoveryToolCall,
        approach: list[float],
        contact: list[float],
    ) -> RecoveryToolCall:
        args = dict(setup.args or {})
        for key in (
            "clearance_margin",
            "clearance_steps",
            "complete_transient_cycle",
            "gripper_precondition",
            "held_focus_key",
            "held_instance_id",
            "held_instance_ref",
            "held_role",
            "max_clearance_lift",
            "offset_xyz",
            "preserve_height",
        ):
            args.pop(key, None)
        args["point_key"] = "approach_world_m"
        try:
            requested_translation = float(args.get("max_translation", 0.0))
        except (TypeError, ValueError):
            requested_translation = 0.0
        if not math.isfinite(requested_translation) or requested_translation < 0.0:
            requested_translation = 0.0
        grounded_release_distance = sum(
            (approach[index] - contact[index]) ** 2 for index in range(3)
        ) ** 0.5
        actuation_displacement = self._contact_displacement_vector(actuation)
        if actuation_displacement is not None:
            grounded_release_distance += sum(value * value for value in actuation_displacement) ** 0.5
        args["max_translation"] = max(
            requested_translation,
            grounded_release_distance,
        )
        if "steps" not in args and "steps" in (actuation.args or {}):
            args["steps"] = (actuation.args or {}).get("steps")
        args["_guard_reason"] = (
            "complete one transient contact cycle by releasing to the same grounded approach before re-observation"
        )
        return RecoveryToolCall(tool_name="move_ee_to_grounded_instance", args=args)

    def _stage_transient_visual_clearance(self, calls: list[RecoveryToolCall]) -> list[RecoveryToolCall]:
        """Move a released transient-contact arm out of the target camera ray.

        Physical release and visual clearance are deliberately separate motions: the first move
        follows the grounded contact normal back to the approach pose, and only then may the arm
        move laterally.  The lateral direction comes from current camera/contact geometry and is
        never inferred from task language, an object name, or a fixed world axis.
        """

        if not bool(
            getattr(
                self.config,
                "enable_automatic_self_occlusion_visual_clearance",
                True,
            )
        ):
            return list(calls)

        staged: list[RecoveryToolCall] = []
        grounded_contact_by_arm: dict[str, RecoveryToolCall] = {}
        pending_cycle_by_arm: dict[
            str,
            tuple[RecoveryToolCall, RecoveryToolCall, list[float], list[float]],
        ] = {}
        for index, call in enumerate(calls):
            arm = self._single_arm_from_call(call)
            if call.tool_name == "move_ee_to_grounded_instance":
                point_key = self._normalized_grounded_point_key(call)
                if arm in {"left", "right"} and point_key == "contact_world_m":
                    grounded_contact_by_arm[arm] = call

            if (
                call.tool_name == "contact_displace"
                and arm in {"left", "right"}
                and self._requests_complete_transient_contact_cycle(call)
            ):
                setup = grounded_contact_by_arm.get(arm)
                instance = self._grounded_instance_for_call(setup) if setup is not None else None
                approach = self._xyz_prefix(instance.get("approach_world_m")) if instance is not None else None
                contact = self._xyz_prefix(instance.get("contact_world_m")) if instance is not None else None
                if setup is not None and approach is not None and contact is not None:
                    pending_cycle_by_arm[arm] = (setup, call, approach, contact)

            staged.append(call)

            pending = pending_cycle_by_arm.get(arm)
            if pending is not None:
                setup, actuation, approach, contact = pending
                setup_instance = self._recovery_history_instance_id(setup, {})
                if self._is_grounded_transient_release(
                    call,
                    setup_arm=arm,
                    setup_instance=setup_instance,
                ):
                    if not self._has_explicit_post_contact_clearance(calls[index + 1 :], arm=arm):
                        clearance = self._grounded_transient_visual_clearance_call(
                            setup,
                            actuation=actuation,
                            approach=approach,
                            contact=contact,
                        )
                        if clearance is not None:
                            staged.append(clearance)
                    pending_cycle_by_arm.pop(arm, None)

            if call.tool_name == "reobserve_scene":
                pending_cycle_by_arm.clear()
                grounded_contact_by_arm.clear()
            elif call.tool_name in {
                "move_ee_to_pose",
                "move_to_home",
                "retreat_arm",
                "safe_reset_posture",
            }:
                if arm == "both":
                    pending_cycle_by_arm.clear()
                    grounded_contact_by_arm.clear()
                elif arm in {"left", "right"}:
                    pending_cycle_by_arm.pop(arm, None)
                    grounded_contact_by_arm.pop(arm, None)
        return staged

    def _has_explicit_post_contact_clearance(
        self,
        later_calls: list[RecoveryToolCall],
        *,
        arm: str,
    ) -> bool:
        for call in later_calls:
            if call.tool_name == "reobserve_scene":
                return False
            if self._single_arm_from_call(call) != arm:
                continue
            if call.tool_name in {"lift_ee", "move_ee_to_pose", "move_to_home", "retreat_arm", "safe_reset_posture"}:
                return True
            if call.tool_name != "move_ee_to_grounded_instance":
                continue
            if self._normalized_grounded_point_key(call) != "approach_world_m":
                continue
            offset = self._xyz_prefix((call.args or {}).get("offset_xyz", [0.0, 0.0, 0.0]))
            if offset is not None and sum(value * value for value in offset) > 1e-10:
                return True
        return False

    def _grounded_transient_visual_clearance_call(
        self,
        setup: RecoveryToolCall,
        *,
        actuation: RecoveryToolCall,
        approach: list[float],
        contact: list[float],
    ) -> RecoveryToolCall | None:
        instance = self._grounded_instance_for_call(setup)
        arm = self._single_arm_from_call(setup)
        if instance is None or arm not in {"left", "right"}:
            return None
        offset = self._transient_visual_clearance_offset(
            instance=instance,
            arm=arm,
            approach=approach,
            contact=contact,
        )
        if offset is None:
            return None

        args = dict(setup.args or {})
        for key in (
            "clearance_margin",
            "clearance_steps",
            "complete_transient_cycle",
            "gripper_precondition",
            "held_focus_key",
            "held_instance_id",
            "held_instance_ref",
            "held_role",
            "max_clearance_lift",
            "preserve_height",
        ):
            args.pop(key, None)
        clearance_distance = sum(value * value for value in offset) ** 0.5
        args.update(
            {
                "point_key": "approach_world_m",
                "offset_xyz": offset,
                "max_translation": clearance_distance,
                "post_contact_clearance": "camera_tangent",
                "_runtime_post_contact_clearance": True,
                "_guard_reason": (
                    "restore target visibility after transient release using current camera/contact geometry"
                ),
            }
        )
        try:
            setup_steps = int((setup.args or {}).get("steps", 1))
        except (TypeError, ValueError):
            setup_steps = 1
        try:
            actuation_steps = int((actuation.args or {}).get("steps", 1))
        except (TypeError, ValueError):
            actuation_steps = 1
        args["steps"] = max(2, setup_steps, actuation_steps)
        return RecoveryToolCall(tool_name="move_ee_to_grounded_instance", args=args)

    def _transient_visual_clearance_offset(
        self,
        *,
        instance: dict[str, Any],
        arm: str,
        approach: list[float],
        contact: list[float],
    ) -> list[float] | None:
        snapshot = self.latest_snapshot
        camera = str(instance.get("camera", "") or "").strip().lower().replace("_camera", "")
        # A wrist camera on the arm being cleared changes extrinsics during the move, so its
        # current-frame ray is not a valid predictor for the post-motion view.
        if snapshot is None or camera not in {"head", "left", "right"} or camera == arm:
            return None
        transform = getattr(snapshot, f"{camera}_cam2world_gl", None)
        try:
            camera_origin = [float(transform[index][3]) for index in range(3)]
            camera_right = [float(transform[index][0]) for index in range(3)]
        except (TypeError, ValueError, IndexError):
            return None
        if not all(math.isfinite(value) for value in (*camera_origin, *camera_right)):
            return None

        normal = self._normalized_xyz_vector(
            [approach[index] - contact[index] for index in range(3)]
        )
        view = self._normalized_xyz_vector(
            [contact[index] - camera_origin[index] for index in range(3)]
        )
        if normal is None or view is None:
            return None
        tangent = self._normalized_xyz_vector(self._cross_xyz(view, normal))
        if tangent is None:
            camera_right_tangent = self._reject_xyz(camera_right, normal)
            tangent = self._normalized_xyz_vector(camera_right_tangent)
        if tangent is None:
            return None

        selected_pose = getattr(snapshot, f"{arm}_endpose", None)
        selected_xyz = self._xyz_prefix(
            selected_pose.tolist() if hasattr(selected_pose, "tolist") else selected_pose
        )
        if selected_xyz is None:
            return None
        precontact_delta = [selected_xyz[index] - approach[index] for index in range(3)]
        precontact_side = self._dot_xyz(precontact_delta, tangent)
        if abs(precontact_side) > 1e-6:
            if precontact_side < 0.0:
                tangent = [-value for value in tangent]
        else:
            robot_state = self._get_robot_state()
            other_arm = "right" if arm == "left" else "left"
            other_xyz = self._xyz_prefix((robot_state.get(other_arm) or {}).get("xyz")) if isinstance(robot_state, dict) else None
            if other_xyz is None:
                return None
            positive = [approach[index] + tangent[index] for index in range(3)]
            negative = [approach[index] - tangent[index] for index in range(3)]
            positive_clearance = self._xyz_distance(positive, other_xyz)
            negative_clearance = self._xyz_distance(negative, other_xyz)
            if positive_clearance is None or negative_clearance is None:
                return None
            if negative_clearance > positive_clearance:
                tangent = [-value for value in tangent]

        quality = instance.get("quality") if isinstance(instance.get("quality"), dict) else {}
        extent = self._xyz_prefix(quality.get("world_extent_m"))
        try:
            max_clearance = float(getattr(self.config, "scene_memory_robot_self_filter_radius_m", 0.08))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(max_clearance) or max_clearance <= 0.0:
            return None
        if extent is not None and all(value >= 0.0 for value in extent):
            clearance_distance = sum(abs(tangent[index]) * extent[index] for index in range(3))
        else:
            tangent_delta = self._reject_xyz(precontact_delta, normal)
            clearance_distance = sum(value * value for value in tangent_delta) ** 0.5
        clearance_distance = min(clearance_distance, max_clearance)
        if not math.isfinite(clearance_distance) or clearance_distance <= 1e-4:
            return None
        return [round(tangent[index] * clearance_distance, 6) for index in range(3)]

    @staticmethod
    def _dot_xyz(left: list[float], right: list[float]) -> float:
        return sum(left[index] * right[index] for index in range(3))

    @classmethod
    def _reject_xyz(cls, vector: list[float], axis: list[float]) -> list[float]:
        projection = cls._dot_xyz(vector, axis)
        return [vector[index] - projection * axis[index] for index in range(3)]

    @staticmethod
    def _cross_xyz(left: list[float], right: list[float]) -> list[float]:
        return [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]

    @staticmethod
    def _normalized_xyz_vector(vector: list[float]) -> list[float] | None:
        if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
            return None
        norm = sum(value * value for value in vector) ** 0.5
        if norm <= 1e-8:
            return None
        return [value / norm for value in vector]

    @staticmethod
    def _xyz_distance(left: list[float], right: list[float]) -> float | None:
        if len(left) != 3 or len(right) != 3:
            return None
        if not all(math.isfinite(value) for value in (*left, *right)):
            return None
        return sum((left[index] - right[index]) ** 2 for index in range(3)) ** 0.5

    def _stage_partial_grounded_approach_continuation(
        self,
        calls: list[RecoveryToolCall],
    ) -> list[RecoveryToolCall]:
        """Consume one lease to finish only the same structured approach."""

        staged = list(calls)
        if self.latest_snapshot is None:
            self._partial_grounded_approach_leases.clear()
            return staged
        skill_id = self._active_skill_id()
        env_step = int(self.latest_snapshot.step_count)
        for lease_key, lease in list(
            self._partial_grounded_approach_leases.items()
        ):
            try:
                lease_step = int(lease.get("created_env_step", -1))
            except (AttributeError, TypeError, ValueError):
                lease_step = -1
            if lease_key[0] != skill_id or lease_step != env_step:
                self._partial_grounded_approach_leases.pop(
                    lease_key,
                    None,
                )
        if not staged or not self._partial_grounded_approach_leases:
            return staged
        setup = staged[0]
        if setup.tool_name != "move_ee_to_grounded_instance":
            if any(call.tool_name != "reobserve_scene" for call in staged):
                for lease_key in list(
                    self._partial_grounded_approach_leases
                ):
                    if lease_key[0] == skill_id:
                        self._partial_grounded_approach_leases.pop(
                            lease_key,
                            None,
                        )
            return staged
        instance_id = self._recovery_history_instance_id(setup, {})
        if not instance_id:
            return staged
        matching_keys = [
            lease_key
            for lease_key in self._partial_grounded_approach_leases
            if lease_key[0] == skill_id
            and lease_key[1] == instance_id
        ]
        if not matching_keys:
            return staged
        leases = [
            self._partial_grounded_approach_leases.pop(
                lease_key,
            )
            for lease_key in matching_keys
        ]
        if len(matching_keys) != 1:
            return staged
        lease_key = matching_keys[0]
        lease = leases[0]
        arm = self._single_arm_from_call(setup)
        if (
            arm not in {"left", "right"}
            or arm != lease_key[2]
            or len(staged) != 2
            or staged[1].tool_name != "reobserve_scene"
            or self._normalized_grounded_point_key(setup)
            != "approach_world_m"
        ):
            return staged

        args = dict(setup.args or {})
        action_mode = operation_action_mode(
            "approach_world_m",
            args.get("_operation_action_mode"),
        )
        candidate_id = str(
            args.get("_operation_candidate_id", "") or ""
        ).strip()
        if (
            action_mode
            != str(
                lease.get("operation_action_mode", "") or ""
            ).strip().lower()
            or not candidate_id
            or candidate_id
            != str(
                lease.get("operation_candidate_id", "") or ""
            ).strip()
        ):
            return staged
        instance = self._grounded_instance_for_call(setup)
        if instance is None:
            return staged
        status = str(instance.get("status", "") or "").strip().lower()
        stability = str(
            instance.get("stability", "") or ""
        ).strip().lower()
        if (
            status != "tracked"
            or stability
            != str(
                lease.get("allowed_current_stability", "") or ""
            ).strip().lower()
        ):
            return staged

        setup_offset = self._xyz_prefix(
            args.get("offset_xyz", [0.0, 0.0, 0.0])
        )
        try:
            max_translation = float(
                args.get("max_translation", 0.06)
            )
            raw_steps = float(args.get("steps", 1))
            steps = int(raw_steps)
            original_max_translation = float(
                lease.get("original_max_translation_m")
            )
        except (TypeError, ValueError):
            return staged
        if (
            setup_offset is None
            or sum(value * value for value in setup_offset) > 1e-10
            or args.get("preserve_height", False) is not False
            or not math.isfinite(max_translation)
            or max_translation <= 0.0
            or not math.isfinite(raw_steps)
            or raw_steps != float(steps)
            or steps < 1
            or not math.isfinite(original_max_translation)
            or max_translation
            > original_max_translation
            + _GROUNDED_SETUP_NO_PROGRESS_EPSILON_M
            or self._recovery_gripper_state().get(arm) != "open"
            or self._transport_holding_state(arm) is not None
        ):
            return staged

        approach = self._xyz_prefix(
            lease.get("approach_world_m")
        )
        operation_target = self._xyz_prefix(
            lease.get("target_world_m")
        )
        current_approach = self._xyz_prefix(
            instance.get("approach_world_m")
        )
        target_point_key = str(
            lease.get("target_point_key", "") or ""
        ).strip()
        current_target = self._xyz_prefix(
            instance.get(target_point_key)
        )
        current_pose = getattr(
            self.latest_snapshot,
            f"{arm}_endpose",
            None,
        )
        current_xyz = self._xyz_prefix(
            current_pose.tolist()
            if hasattr(current_pose, "tolist")
            else current_pose
        )
        observed_xyz = self._xyz_prefix(lease.get("observed_xyz"))
        pose_drift = (
            None
            if current_xyz is None or observed_xyz is None
            else self._xyz_distance(current_xyz, observed_xyz)
        )
        residual = (
            None
            if current_xyz is None or approach is None
            else self._xyz_distance(current_xyz, approach)
        )
        approach_drift = (
            None
            if approach is None or current_approach is None
            else self._xyz_distance(approach, current_approach)
        )
        target_drift = (
            None
            if operation_target is None or current_target is None
            else self._xyz_distance(operation_target, current_target)
        )
        if (
            pose_drift is None
            or pose_drift
            > _GROUNDED_GEOMETRY_LEASE_MAX_POSE_DRIFT_M
            or residual is None
            or residual
            > _PARTIAL_APPROACH_CONTINUATION_MAX_RESIDUAL_M
            or max_translation * steps
            + _GROUNDED_SETUP_NO_PROGRESS_EPSILON_M
            < residual
            or approach_drift is None
            or approach_drift
            > _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
            or target_drift is None
            or target_drift
            > _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
        ):
            return staged
        occlusion_proof = self._camera_ray_occlusion_proof(
            instance=instance,
            arm=arm,
            observed_xyz=current_xyz,
            target_xyz=operation_target,
        )
        if occlusion_proof is None:
            return staged

        reason = (
            "one-shot partial-approach continuation: observed motion "
            "converged monotonically before same-candidate self-occlusion; "
            "finish only the exact bounded approach and then reobserve"
        )
        args["_runtime_partial_approach_continuation"] = True
        existing_reason = str(
            args.get("_guard_reason", "") or ""
        ).strip()
        args["_guard_reason"] = (
            f"{existing_reason}; {reason}"
            if existing_reason
            else reason
        )
        staged[0] = RecoveryToolCall(
            tool_name=setup.tool_name,
            args=args,
        )
        payload = {
            "skill_id": skill_id,
            "instance_id": instance_id,
            "arm": arm,
            "created_env_step": int(
                lease.get("created_env_step", -1)
            ),
            "operation_action_mode": action_mode,
            "operation_candidate_id": candidate_id,
            "current_stability": stability,
            "residual_m": residual,
            "max_translation_m": max_translation,
            "steps": steps,
            "pose_drift_m": pose_drift,
            "approach_drift_m": approach_drift,
            "target_drift_m": target_drift,
            "authorized_call_indices": [0],
        }
        self._trace(
            "grounded_approach_continuation_lease_consumed",
            **payload,
        )
        self._dump_rollout_event(
            "grounded_approach_continuation_lease_consumed",
            **payload,
        )
        return staged

    def _stage_grasp_occlusion_geometry_lease(
        self,
        calls: list[RecoveryToolCall],
    ) -> list[RecoveryToolCall]:
        """Consume one exact approach-bound lease for a self-occluded grasp.

        This capability is intentionally narrower than a general stale-pose
        exception.  It only permits the immediate
        ``grasp_world_m -> close_gripper -> reobserve_scene`` transaction for
        the same skill, instance, arm, candidate, environment step, and cached
        stable geometry that produced a reached approach.
        """

        staged = list(calls)
        if not staged:
            return staged
        setup = staged[0]
        if (
            setup.tool_name != "move_ee_to_grounded_instance"
            or self._normalized_grounded_point_key(setup)
            != "grasp_world_m"
        ):
            return staged
        arm = self._single_arm_from_call(setup)
        instance_id = self._recovery_history_instance_id(setup, {})
        if arm not in {"left", "right"} or not instance_id:
            return staged
        lease_key = (self._active_skill_id(), instance_id, arm)
        lease = self._grounded_geometry_leases.get(lease_key)
        if (
            not isinstance(lease, dict)
            or str(
                lease.get("operation_action_mode", "") or ""
            ).strip().lower()
            != "grasp"
        ):
            return staged
        # A matching grasp proposal consumes the one-shot capability even when
        # the rest of the proposed transaction is malformed.
        self._grounded_geometry_leases.pop(lease_key, None)
        if (
            len(staged) != 3
            or staged[1].tool_name != "close_gripper"
            or self._single_arm_from_call(staged[1]) != arm
            or staged[2].tool_name != "reobserve_scene"
        ):
            return staged
        instance = self._grounded_instance_for_call(setup)
        if instance is None or self.latest_snapshot is None:
            return staged
        status = str(instance.get("status", "") or "").strip().lower()
        stability = str(
            instance.get("stability", "") or ""
        ).strip().lower()
        if (
            status not in {"tracked", "visible"}
            or stability
            not in {
                "missing_current_frame",
                "geometry_inconsistent_current_frame",
            }
            or stability
            != str(
                lease.get("allowed_current_stability", "") or ""
            ).strip().lower()
        ):
            return staged
        args = dict(setup.args or {})
        candidate_id = str(
            args.get("_operation_candidate_id", "") or ""
        ).strip()
        if (
            not candidate_id
            or candidate_id
            != str(
                lease.get("operation_candidate_id", "") or ""
            ).strip()
            or operation_action_mode(
                self._normalized_grounded_point_key(setup),
                args.get("_operation_action_mode"),
            )
            != "grasp"
        ):
            return staged
        try:
            same_env_step = int(
                lease.get("created_env_step", -1)
            ) == int(self.latest_snapshot.step_count)
        except (TypeError, ValueError):
            same_env_step = False
        current_pose = getattr(
            self.latest_snapshot,
            f"{arm}_endpose",
            None,
        )
        current_xyz = self._xyz_prefix(
            current_pose.tolist()
            if hasattr(current_pose, "tolist")
            else current_pose
        )
        observed_xyz = self._xyz_prefix(lease.get("observed_xyz"))
        pose_drift = (
            None
            if current_xyz is None or observed_xyz is None
            else self._xyz_distance(current_xyz, observed_xyz)
        )
        approach = self._xyz_prefix(lease.get("approach_world_m"))
        target = self._xyz_prefix(lease.get("target_world_m"))
        current_approach = self._xyz_prefix(
            instance.get("approach_world_m")
        )
        current_target = self._xyz_prefix(
            instance.get("grasp_world_m")
        )
        approach_drift = (
            None
            if approach is None or current_approach is None
            else self._xyz_distance(approach, current_approach)
        )
        target_drift = (
            None
            if target is None or current_target is None
            else self._xyz_distance(target, current_target)
        )
        target_travel = (
            None
            if approach is None or target is None
            else self._xyz_distance(approach, target)
        )
        try:
            setup_max_translation = float(
                args.get("max_translation", 0.06)
            )
        except (TypeError, ValueError):
            setup_max_translation = float("nan")
        setup_offset = self._xyz_prefix(
            args.get("offset_xyz", [0.0, 0.0, 0.0])
        )
        if (
            not same_env_step
            or pose_drift is None
            or pose_drift
            > _GROUNDED_GEOMETRY_LEASE_MAX_POSE_DRIFT_M
            or approach_drift is None
            or approach_drift
            > _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
            or target_drift is None
            or target_drift
            > _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
            or target_travel is None
            or not math.isfinite(setup_max_translation)
            or setup_max_translation + 1e-8 < target_travel
            or setup_offset is None
            or sum(value * value for value in setup_offset) > 1e-10
            or args.get("preserve_height", False) is not False
            or self._recovery_gripper_state().get(arm) != "open"
        ):
            return staged

        reason = (
            "one-shot grasp geometry lease: a freshly reached grounded "
            "approach caused expected self-occlusion; complete only the "
            "same-candidate bounded grasp and close before re-observation"
        )
        setup_args = dict(setup.args or {})
        setup_args["_runtime_occlusion_geometry_lease"] = True
        existing_reason = str(
            setup_args.get("_guard_reason", "") or ""
        ).strip()
        setup_args["_guard_reason"] = (
            f"{existing_reason}; {reason}"
            if existing_reason
            else reason
        )
        close_args = dict(staged[1].args or {})
        close_reason = str(
            close_args.get("_guard_reason", "") or ""
        ).strip()
        close_args["_guard_reason"] = (
            f"{close_reason}; {reason}" if close_reason else reason
        )
        staged[0] = RecoveryToolCall(
            tool_name=setup.tool_name,
            args=setup_args,
        )
        staged[1] = RecoveryToolCall(
            tool_name=staged[1].tool_name,
            args=close_args,
        )
        payload = {
            "skill_id": lease_key[0],
            "instance_id": instance_id,
            "arm": arm,
            "created_env_step": int(
                lease.get("created_env_step", -1)
            ),
            "operation_action_mode": "grasp",
            "operation_candidate_id": candidate_id,
            "current_stability": stability,
            "pose_drift_m": pose_drift,
            "approach_drift_m": approach_drift,
            "target_drift_m": target_drift,
            "authorized_call_indices": [0, 1],
        }
        self._trace("grounded_geometry_lease_consumed", **payload)
        self._dump_rollout_event(
            "grounded_geometry_lease_consumed",
            **payload,
        )
        return staged

    def _stage_occlusion_geometry_leases(
        self,
        calls: list[RecoveryToolCall],
    ) -> list[RecoveryToolCall]:
        """Authorize one bounded operation after a reached approach self-occludes its target.

        The lease is created only from a successful, freshly grounded approach.
        It is bound to the active skill, instance, arm, environment step,
        selected candidate, and observed EE pose, and is consumed before
        dispatch.  Contact retains its original visible-to-missing-only
        contract.  Grasp additionally accepts a proven action-induced
        geometry-inconsistent frame, but only for the exact one-shot grasp
        transaction staged above.
        """

        staged = self._stage_grasp_occlusion_geometry_lease(
            list(calls)
        )
        for index, setup in enumerate(staged):
            if setup.tool_name != "move_ee_to_grounded_instance":
                continue
            if self._normalized_grounded_point_key(setup) != "contact_world_m":
                continue
            arm = self._single_arm_from_call(setup)
            instance_id = self._recovery_history_instance_id(setup, {})
            if arm not in {"left", "right"} or not instance_id:
                continue
            lease_key = (self._active_skill_id(), instance_id, arm)
            lease = self._grounded_geometry_leases.pop(lease_key, None)
            if lease is None or self.latest_snapshot is None:
                continue
            # Seeing a matching contact proposal consumes the capability regardless of current
            # visibility or plan shape.  A planner cannot probe and reuse it repeatedly.
            instance = self._grounded_instance_for_call(setup)
            if instance is None:
                continue
            status = str(instance.get("status", "") or "").strip().lower()
            stability = str(instance.get("stability", "") or "").strip().lower()
            if status != "tracked" or stability != "missing_current_frame":
                continue
            # A lease authorizes one atomic transaction, never a suffix of a larger physical
            # plan.  Starting at index zero also rules out another arm changing the scene first.
            if index != 0:
                continue
            try:
                same_env_step = int(lease.get("created_env_step", -1)) == int(
                    self.latest_snapshot.step_count
                )
            except (TypeError, ValueError):
                same_env_step = False
            current_pose = getattr(self.latest_snapshot, f"{arm}_endpose", None)
            current_xyz = self._xyz_prefix(
                current_pose.tolist() if hasattr(current_pose, "tolist") else current_pose
            )
            observed_xyz = self._xyz_prefix(lease.get("observed_xyz"))
            pose_drift = (
                None
                if current_xyz is None or observed_xyz is None
                else self._xyz_distance(current_xyz, observed_xyz)
            )
            approach = self._xyz_prefix(lease.get("approach_world_m"))
            contact = self._xyz_prefix(lease.get("contact_world_m"))
            current_approach = self._xyz_prefix(instance.get("approach_world_m"))
            current_contact = self._xyz_prefix(instance.get("contact_world_m"))
            approach_drift = (
                None
                if approach is None or current_approach is None
                else self._xyz_distance(approach, current_approach)
            )
            contact_drift = (
                None
                if contact is None or current_contact is None
                else self._xyz_distance(contact, current_contact)
            )
            if (
                not same_env_step
                or pose_drift is None
                or pose_drift > _GROUNDED_GEOMETRY_LEASE_MAX_POSE_DRIFT_M
                or approach is None
                or contact is None
                or approach_drift is None
                or contact_drift is None
                or approach_drift > _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
                or contact_drift > _GROUNDED_GEOMETRY_LEASE_MAX_POINT_DRIFT_M
            ):
                continue
            setup_args = dict(setup.args or {})
            setup_offset = self._xyz_prefix(
                setup_args.get("offset_xyz", [0.0, 0.0, 0.0])
            )
            contact_travel = self._xyz_distance(approach, contact)
            try:
                setup_max_translation = float(setup_args.get("max_translation", 0.06))
            except (TypeError, ValueError):
                continue
            if (
                setup_offset is None
                or sum(value * value for value in setup_offset) > 1e-10
                or setup_args.get("preserve_height", False) is not False
                or contact_travel is None
                or not math.isfinite(setup_max_translation)
                or setup_max_translation + 1e-8 < contact_travel
            ):
                continue

            # After transient-cycle staging, the only legal shapes are:
            #   contact, unique structured actuation, grounded release, reobserve
            #   contact, unique structured actuation, grounded release,
            #       runtime-generated camera clearance, reobserve
            if len(staged) not in {4, 5} or staged[-1].tool_name != "reobserve_scene":
                continue
            gripper_state = self._recovery_gripper_state()
            if any(
                self._required_gripper_transition(candidate, gripper_state)
                for candidate in staged[:-1]
            ):
                continue
            actuation = staged[1]
            release = staged[2]
            if (
                actuation.tool_name != "contact_displace"
                or self._single_arm_from_call(actuation) != arm
                or not self._requests_complete_transient_contact_cycle(actuation)
                or not self._contact_displacement_follows_approach(
                    actuation,
                    approach=approach,
                    contact=contact,
                )
                or not self._is_grounded_transient_release(
                    release,
                    setup_arm=arm,
                    setup_instance=instance_id,
                )
            ):
                continue
            release_args = dict(release.args or {})
            try:
                release_max_translation = float(
                    release_args.get("max_translation", 0.06)
                )
            except (TypeError, ValueError):
                continue
            actuation_displacement = self._contact_displacement_vector(actuation)
            required_release_travel = (
                None
                if contact_travel is None or actuation_displacement is None
                else contact_travel
                + sum(value * value for value in actuation_displacement) ** 0.5
            )
            if (
                release_args.get("preserve_height", False) is not False
                or required_release_travel is None
                or not math.isfinite(release_max_translation)
                or release_max_translation + 1e-8 < required_release_travel
            ):
                continue
            authorized_grounded_indices = {0, 2}
            if len(staged) == 5:
                clearance = staged[3]
                clearance_offset = self._xyz_prefix(
                    (clearance.args or {}).get("offset_xyz", [0.0, 0.0, 0.0])
                )
                if (
                    clearance.tool_name != "move_ee_to_grounded_instance"
                    or self._single_arm_from_call(clearance) != arm
                    or self._recovery_history_instance_id(clearance, {}) != instance_id
                    or self._normalized_grounded_point_key(clearance) != "approach_world_m"
                    or (clearance.args or {}).get("_runtime_post_contact_clearance") is not True
                    or clearance_offset is None
                    or sum(value * value for value in clearance_offset) <= 1e-10
                ):
                    continue
                authorized_grounded_indices.add(3)

            reason = (
                "one-shot geometry lease: a freshly reached grounded approach caused expected "
                "self-occlusion; complete the bounded contact, release, clearance, and reobserve cycle"
            )
            for candidate_index in sorted(authorized_grounded_indices):
                candidate = staged[candidate_index]
                args = dict(candidate.args or {})
                args["_runtime_occlusion_geometry_lease"] = True
                existing_reason = str(args.get("_guard_reason", "") or "").strip()
                args["_guard_reason"] = f"{existing_reason}; {reason}" if existing_reason else reason
                staged[candidate_index] = RecoveryToolCall(
                    tool_name=candidate.tool_name,
                    args=args,
                )
            payload = {
                "skill_id": lease_key[0],
                "instance_id": instance_id,
                "arm": arm,
                "created_env_step": int(lease.get("created_env_step", -1)),
                "pose_drift_m": pose_drift,
                "approach_drift_m": approach_drift,
                "contact_drift_m": contact_drift,
                "authorized_call_indices": sorted(authorized_grounded_indices),
            }
            self._trace("grounded_geometry_lease_consumed", **payload)
            self._dump_rollout_event("grounded_geometry_lease_consumed", **payload)
            break
        return staged

    def _self_occlusion_visual_clearance_call(
        self,
        call: RecoveryToolCall,
        *,
        instance: dict[str, Any],
    ) -> RecoveryToolCall | None:
        """Build a bounded camera-tangent clearance from a proven local occlusion."""

        if not bool(
            getattr(
                self.config,
                "enable_automatic_self_occlusion_visual_clearance",
                True,
            )
        ):
            return None

        arm = self._single_arm_from_call(call)
        if arm not in {"left", "right"} or self.latest_snapshot is None:
            return None
        point_key = self._normalized_grounded_point_key(call)
        action_mode = operation_action_mode(
            point_key,
            (call.args or {}).get("_operation_action_mode"),
        )
        target_point_key = (
            f"{action_mode}_world_m"
            if action_mode in {"contact", "grasp"}
            else point_key
        )
        approach = self._xyz_prefix(instance.get("approach_world_m"))
        target = self._xyz_prefix(instance.get(target_point_key))
        current_pose = getattr(
            self.latest_snapshot,
            f"{arm}_endpose",
            None,
        )
        current_pose_values = (
            current_pose.tolist()
            if hasattr(current_pose, "tolist")
            else current_pose
        )
        current_xyz = self._xyz_prefix(current_pose_values)
        if (
            approach is None
            or target is None
            or current_xyz is None
            or not isinstance(current_pose_values, (list, tuple))
            or len(current_pose_values) < 7
        ):
            return None
        approach_drift = self._xyz_distance(current_xyz, approach)
        if (
            approach_drift is None
            or approach_drift
            > _GROUNDED_VISUAL_CLEARANCE_MAX_APPROACH_DRIFT_M
        ):
            return None
        if (
            self._camera_ray_occlusion_proof(
                instance=instance,
                arm=arm,
                observed_xyz=current_xyz,
                target_xyz=target,
            )
            is None
        ):
            return None
        offset = self._transient_visual_clearance_offset(
            instance=instance,
            arm=arm,
            approach=approach,
            contact=target,
        )
        if offset is None:
            return None
        clearance_distance = sum(
            value * value for value in offset
        ) ** 0.5
        if (
            not math.isfinite(clearance_distance)
            or clearance_distance <= 1e-4
        ):
            return None
        try:
            target_pose = [
                float(value) for value in current_pose_values[:7]
            ]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in target_pose):
            return None
        for index in range(3):
            target_pose[index] += offset[index]
        return RecoveryToolCall(
            tool_name="move_ee_to_pose",
            args={
                "arm": arm,
                "target_pose": target_pose,
                "max_translation": clearance_distance,
                "steps": 3,
                "_runtime_visual_clearance": True,
                "_guard_reason": (
                    "restore target visibility with a bounded camera-tangent "
                    "clearance after a grounded setup caused local self-occlusion"
                ),
            },
        )

    def _runtime_place_motion_authorization(
        self,
        call: RecoveryToolCall,
    ) -> tuple[dict[str, Any] | None, str]:
        """Validate exact dynamic place geometry independently of held-object visibility.

        A transported object is commonly absent from the current camera frame
        because it is inside the gripper.  Place motion is therefore authorized
        by the runtime holding capability, held-object-to-TCP propagation, and
        fresh support/occupancy validation—not by requiring the held object to
        be visually reacquired.
        """

        if call.tool_name != "move_ee_to_grounded_instance":
            return None, "place requires a grounded motion tool"
        args = dict(call.args or {})
        point_key = self._normalized_grounded_point_key(call)
        action_mode = operation_action_mode(
            point_key,
            args.get(
                "_operation_action_mode",
                args.get("action_mode"),
            ),
        )
        if (
            action_mode != "place"
            or point_key
            not in {"approach_world_m", "place_world_m"}
        ):
            return None, "place motion requires a place approach or placement point"
        arm = self._single_arm_from_call(call)
        holding_state = self._transport_holding_state(arm)
        raw_instance = self._raw_grounded_instance_for_call(call)
        if arm not in {"left", "right"}:
            return None, "place motion requires one physical arm"
        if holding_state is None:
            return None, "the selected arm has no transport-authorized held object"
        if raw_instance is None:
            return None, "the held object is absent from current scene memory"
        if self._recovery_gripper_state().get(arm) != "closed":
            return None, "the selected arm has no maintained closing command"
        held_instance_id = str(
            holding_state.get("held_instance_id", "") or ""
        ).strip()
        call_instance_id = str(
            raw_instance.get("instance_id", "") or ""
        ).strip()
        target_id = str(args.get("target_id", "") or "").strip()
        candidate_id = str(
            args.get("_operation_candidate_id", "") or ""
        ).strip()
        if not held_instance_id or held_instance_id != call_instance_id:
            return None, "the requested object differs from the transport-authorized held object"
        if not target_id or not candidate_id:
            return None, "the place request lacks a resolved target or candidate"
        offset = self._xyz_prefix(
            args.get("offset_xyz", [0.0, 0.0, 0.0])
        )
        if (
            offset is None
            or sum(value * value for value in offset) > 1e-10
            or args.get("preserve_height", False) is not False
        ):
            return None, "the place request changes the validated candidate offset or height"
        selected = select_operation_pose_candidate(
            raw_instance,
            arm=arm,
            action_mode="place",
            requested_candidate_id=candidate_id,
            requested_target_id=target_id,
        )
        if selected is None:
            return None, "the requested place candidate is unavailable for this arm and target"
        if (
            str(selected.get("candidate_id", "") or "").strip()
            != candidate_id
            or str(selected.get("target_id", "") or "").strip()
            != target_id
            or str(
                selected.get("held_instance_id", "") or ""
            ).strip()
            != held_instance_id
            or selected.get(
                "post_release_verification_required"
            )
            is not True
        ):
            return None, "the resolved place candidate does not match the held object and target"
        scene_memory = args.get("_scene_memory")
        validation = validate_place_candidate(
            scene_memory if isinstance(scene_memory, dict) else {},
            held_instance=raw_instance,
            candidate=selected,
        )
        if validation.get("valid") is not True:
            return None, f"place target validation failed: {validation}"
        return {
            "arm": arm,
            "held_instance_id": held_instance_id,
            "point_key": point_key,
            "target_id": target_id,
            "candidate_id": candidate_id,
            "target_kind": selected.get("target_kind"),
            "holding_status": selected.get("holding_status"),
            "support_valid": validation.get("support_valid"),
            "free": validation.get("free"),
            "target_geometry_drift_m": validation.get(
                "target_geometry_drift_m"
            ),
        }, ""

    def _relocation_pending_reobserve_call(
        self,
        *,
        arm: str,
        instance_id: str,
        track_id: str,
        status: str,
        stability: str,
        policy: str,
        reason_prefix: str = "",
    ) -> RecoveryToolCall:
        reason = (
            "relocated action geometry is awaiting confirmation from two "
            "independent clean observations; do not execute an unconfirmed "
            "contact pose "
            f"instance={instance_id},track={track_id},status={status},"
            f"stability={stability},policy={policy}; reobserve before a "
            "final descent or contact"
        )
        if reason_prefix:
            reason = f"{reason_prefix}; {reason}"
        return RecoveryToolCall(
            tool_name="reobserve_scene",
            args={
                "_evidence_situation": (
                    EvidenceSituation.TEMPORAL_IDENTITY_REPAIR.value
                ),
                "_evidence_track_id": track_id or instance_id,
                "_evidence_arm": arm,
                "_evidence_phase": "relocation_pending",
                "_guard_reason": reason,
            },
        )

    def _repair_pending_safe_motion_call(
        self,
        call: RecoveryToolCall,
        *,
        instance: dict[str, Any],
        arm: str,
        instance_id: str,
        track_id: str,
    ) -> tuple[RecoveryToolCall, dict[str, Any]] | None:
        if arm not in {"left", "right"} or self.latest_snapshot is None:
            return None
        current_pose = getattr(
            self.latest_snapshot,
            f"{arm}_endpose",
            None,
        )
        pose_values = (
            current_pose.tolist()
            if hasattr(current_pose, "tolist")
            else current_pose
        )
        safe_motion = plan_repair_pending_safe_motion(
            instance=instance,
            current_pose=pose_values,
            gripper_state=self._recovery_gripper_state().get(arm),
            ee_to_tcp_m=getattr(
                self.latest_snapshot,
                "ee_to_tcp_m",
                0.12,
            ),
            approach_clearance_m=getattr(
                self.config,
                "observation_grounding_approach_height_m",
                0.08,
            ),
        )
        if safe_motion is None:
            return None
        payload = {
            "policy": SAFE_MOTION_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
            "arm": arm,
            "instance_id": instance_id,
            "track_id": track_id,
            "requested_point_key": self._normalized_grounded_point_key(
                call
            ),
            **safe_motion.as_dict(),
        }
        return (
            RecoveryToolCall(
                tool_name="move_ee_to_pose",
                args={
                    "arm": arm,
                    "target_pose": list(safe_motion.target_pose),
                    "max_translation": safe_motion.max_translation_m,
                    "steps": 3,
                    "_runtime_action_geometry_repair_safe_motion": True,
                    "_evidence_track_id": track_id or instance_id,
                    "_guard_reason": (
                        "runtime-derived lift-first non-contact motion from "
                        "current object position; quarantined approach/grasp/"
                        "contact poses were not used"
                    ),
                },
            ),
            payload,
        )

    def _guard_occluded_grounded_instance(
        self,
        call: RecoveryToolCall,
    ) -> list[RecoveryToolCall]:
        if call.tool_name != "move_ee_to_grounded_instance":
            return [call]
        instance = self._grounded_instance_for_call(call)
        if instance is None:
            return [call]
        status = str(instance.get("status", "") or "").strip().lower()
        stability = str(instance.get("stability", "") or "").strip().lower()
        action_geometry_state = str(
            instance.get("action_geometry_state", "verified")
            or "verified"
        ).strip().lower()
        position_state = str(
            instance.get("position_state", "") or ""
        ).strip().lower()
        point_key = self._normalized_grounded_point_key(call)
        action_mode = operation_action_mode(
            point_key,
            (call.args or {}).get("_operation_action_mode"),
        )
        arm = self._single_arm_from_call(call)
        instance_id = str(
            instance.get("instance_id", "") or ""
        ).strip()
        track_id = str(
            instance.get("track_id", "") or ""
        ).strip()
        if position_state == POSITION_MOTION_UNCERTAIN:
            return [
                RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_evidence_track_id": track_id or instance_id,
                        "_evidence_arm": arm,
                        "_guard_reason": (
                            "grounded instance may have moved after an "
                            "unverified physical event; historical world "
                            "coordinates are non-executable "
                            f"instance={instance_id},track={track_id},"
                            f"position_state={position_state}; reacquire "
                            "the object before grounded motion"
                        )
                    },
                )
            ]
        if action_mode == "place":
            authorization, rejection_reason = self._runtime_place_motion_authorization(
                call
            )
            if authorization is not None:
                args = dict(call.args or {})
                args["_runtime_place_motion_authorized"] = True
                existing_reason = str(
                    args.get("_guard_reason", "") or ""
                ).strip()
                reason = (
                    "runtime-authorized place geometry: transport holding "
                    "state and current support/occupancy validation replace "
                    "held-object visibility"
                )
                args["_guard_reason"] = (
                    f"{existing_reason}; {reason}"
                    if existing_reason
                    else reason
                )
                authorized_call = RecoveryToolCall(
                    tool_name=call.tool_name,
                    args=args,
                )
                payload = {
                    **authorization,
                    "held_track_status": status,
                    "held_track_stability": stability,
                }
                self._trace(
                    "runtime_place_motion_authorized",
                    **payload,
                )
                self._dump_rollout_event(
                    "runtime_place_motion_authorized",
                    **payload,
                )
                return [authorized_call]
            return [
                RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_evidence_track_id": track_id or instance_id,
                        "_evidence_arm": arm,
                        "_guard_reason": rejection_reason,
                    },
                )
            ]
        if action_geometry_state == "relocation_pending":
            policy = self._action_geometry_repair_pending_policy()
            if (
                policy
                == DISABLED_ACTION_GEOMETRY_REPAIR_PENDING_POLICY
            ):
                args = dict(call.args or {})
                args[
                    "_runtime_action_geometry_repair_pending_bypass"
                ] = True
                reason = (
                    "action-geometry repair-pending grounded-motion guard "
                    "disabled by explicit experiment policy; all unrelated "
                    "runtime guards remain active"
                )
                existing_reason = str(
                    args.get("_guard_reason", "") or ""
                ).strip()
                args["_guard_reason"] = (
                    f"{existing_reason}; {reason}"
                    if existing_reason
                    else reason
                )
                payload = {
                    "policy": policy,
                    "arm": arm,
                    "instance_id": instance_id,
                    "track_id": track_id,
                    "point_key": point_key,
                    "action_mode": action_mode,
                }
                self._trace(
                    "relocation_pending_guard_bypassed",
                    **payload,
                )
                self._dump_rollout_event(
                    "relocation_pending_guard_bypassed",
                    **payload,
                )
                return [
                    RecoveryToolCall(
                        tool_name=call.tool_name,
                        args=args,
                    )
                ]
            if (
                policy
                == SAFE_MOTION_ACTION_GEOMETRY_REPAIR_PENDING_POLICY
            ):
                safe_motion = self._repair_pending_safe_motion_call(
                    call,
                    instance=instance,
                    arm=arm,
                    instance_id=instance_id,
                    track_id=track_id,
                )
                if safe_motion is not None:
                    safe_call, payload = safe_motion
                    self._trace(
                        "relocation_pending_safe_motion_authorized",
                        **payload,
                    )
                    self._dump_rollout_event(
                        "relocation_pending_safe_motion_authorized",
                        **payload,
                    )
                    return [
                        safe_call,
                        self._relocation_pending_reobserve_call(
                            arm=arm,
                            instance_id=instance_id,
                            track_id=track_id,
                            status=status,
                            stability=stability,
                            policy=policy,
                            reason_prefix=(
                                "safe high motion completed; collect fresh "
                                "geometry before contact"
                            ),
                        ),
                    ]
            return [
                self._relocation_pending_reobserve_call(
                    arm=arm,
                    instance_id=instance_id,
                    track_id=track_id,
                    status=status,
                    stability=stability,
                    policy=policy,
                )
            ]
        if action_geometry_state == "unavailable":
            return [
                RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_evidence_situation": (
                            EvidenceSituation.TEMPORAL_IDENTITY_REPAIR.value
                        ),
                        "_evidence_track_id": track_id or instance_id,
                        "_evidence_arm": arm,
                        "_evidence_phase": "action_geometry_unavailable",
                        "_guard_reason": (
                            "the object identity/position may remain in memory, "
                            "but no verified executable action geometry exists; "
                            f"instance={instance_id},track={track_id}; acquire "
                            "fresh grounded geometry before moving toward it"
                        ),
                    },
                )
            ]
        call_args = dict(call.args or {})
        memory_valid_final = authorize_memory_valid_final_grounded_action(
            enabled=bool(
                getattr(
                    self.config,
                    "allow_memory_valid_final_grounded_action",
                    False,
                )
            ),
            raw_instance=self._raw_grounded_instance_for_call(call),
            materialized_instance=instance,
            arm=arm,
            point_key=point_key,
            action_mode=action_mode,
            candidate_id=call_args.get(
                "_operation_candidate_id",
                "",
            ),
            blocked_candidate_ids=call_args.get(
                "_blocked_operation_candidate_ids"
            ),
            requested_target_id=call_args.get("target_id"),
            offset_xyz=call_args.get("offset_xyz", [0.0, 0.0, 0.0]),
            preserve_height=call_args.get("preserve_height", False),
        )
        if memory_valid_final is not None:
            args = dict(call.args or {})
            args["_runtime_memory_valid_final_action"] = True
            reason = (
                "exact final grounded action authorized from physically "
                "valid world memory; current visibility is not required, "
                "but close/attachment/transport contracts remain active"
            )
            existing_reason = str(
                args.get("_guard_reason", "") or ""
            ).strip()
            args["_guard_reason"] = (
                f"{existing_reason}; {reason}"
                if existing_reason
                else reason
            )
            authorized = RecoveryToolCall(
                tool_name=call.tool_name,
                args=args,
            )
            self._trace(
                "memory_valid_final_grounded_action_authorized",
                **memory_valid_final,
            )
            self._dump_rollout_event(
                "memory_valid_final_grounded_action_authorized",
                **memory_valid_final,
            )
            return [authorized]
        if (
            position_state == POSITION_MEMORY_VALID
            and point_key == "approach_world_m"
            and action_mode in {"contact", "grasp"}
            and action_geometry_state == "verified"
            and self._xyz_prefix(
                instance.get("approach_world_m")
            )
            is not None
        ):
            args = dict(call.args or {})
            args["_runtime_memory_valid_approach"] = True
            reason = (
                "safe approach authorized from a physically valid "
                "remembered world position; final descent/contact still "
                "requires a current observation or a one-shot proven "
                "self-occlusion lease"
            )
            existing_reason = str(
                args.get("_guard_reason", "") or ""
            ).strip()
            args["_guard_reason"] = (
                f"{existing_reason}; {reason}"
                if existing_reason
                else reason
            )
            authorized = RecoveryToolCall(
                tool_name=call.tool_name,
                args=args,
            )
            payload = {
                "instance_id": instance_id,
                "track_id": track_id,
                "point_key": point_key,
                "action_mode": action_mode,
                "position_state": position_state,
                "position_source": instance.get(
                    "position_source"
                ),
            }
            self._trace(
                "memory_valid_approach_authorized",
                **payload,
            )
            self._dump_rollout_event(
                "memory_valid_approach_authorized",
                **payload,
            )
            return [authorized]
        if action_geometry_state == "identity_repair_expired":
            return [
                RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_evidence_situation": EvidenceSituation.OCCLUSION.value,
                        "_evidence_track_id": track_id or instance_id,
                        "_evidence_arm": arm,
                        "_evidence_phase": "identity_repair_expired",
                        "_guard_reason": (
                            "post-release identity/action-geometry repair lease "
                            "expired; stale grounded geometry remains forbidden "
                            f"instance={instance_id},track={track_id},"
                            f"status={status},stability={stability},"
                            f"action_geometry_state={action_geometry_state}; "
                            "replan identity recovery instead of repeating "
                            "grounded motion"
                        )
                    },
                )
            ]
        if (
            (call.args or {}).get(
                "_runtime_partial_approach_continuation"
            )
            is True
            and self._normalized_grounded_point_key(call)
            == "approach_world_m"
            and status == "tracked"
            and stability
            in {
                "missing_current_frame",
                "geometry_inconsistent_current_frame",
            }
        ):
            return [call]
        if (call.args or {}).get(
            "_runtime_occlusion_geometry_lease"
        ) is True:
            if (
                status in {"tracked", "visible"}
                and (
                    stability == "missing_current_frame"
                    or (
                        stability
                        == "geometry_inconsistent_current_frame"
                        and point_key == "grasp_world_m"
                        and action_mode == "grasp"
                    )
                )
            ):
                return [call]
        if status != "tracked" and stability not in {
            "missing_current_frame",
            "geometry_inconsistent_current_frame",
            "relocation_geometry_pending",
            "identity_repair_expired",
        }:
            return [call]
        if stability == "relocation_geometry_pending":
            guard_reason = (
                "relocated action geometry is awaiting confirmation from two clean observations; "
                f"do not execute an unconfirmed pose instance={instance_id},track={track_id},"
                f"status={status},stability={stability}; reobserve before another grounded move"
            )
        elif stability == "geometry_inconsistent_current_frame":
            guard_reason = (
                "grounded instance has inconsistent current-frame geometry; do not execute a stale pose "
                f"instance={instance_id},track={track_id},status={status},stability={stability}; "
                "retreat or reobserve before another grounded move"
            )
        else:
            guard_reason = (
                "grounded instance is temporarily occluded; do not execute a stale pose "
                f"instance={instance_id},track={track_id},status={status},stability={stability}; "
                "retreat or reobserve before another grounded move"
            )
        reobserve = RecoveryToolCall(
            tool_name="reobserve_scene",
            args={
                "_evidence_situation": (
                    EvidenceSituation.TEMPORAL_IDENTITY_REPAIR.value
                    if stability == "relocation_geometry_pending"
                    else EvidenceSituation.OCCLUSION.value
                ),
                "_evidence_track_id": track_id or instance_id,
                "_evidence_arm": arm,
                "_evidence_phase": (
                    "relocation_pending"
                    if stability == "relocation_geometry_pending"
                    else "self_occluded_grounded_instance"
                ),
                "_guard_reason": guard_reason,
            },
        )
        if stability == "relocation_geometry_pending":
            return [reobserve]
        clearance = self._self_occlusion_visual_clearance_call(
            call,
            instance=instance,
        )
        return [clearance, reobserve] if clearance is not None else [reobserve]

    def _guard_blocked_grounded_setup(self, call: RecoveryToolCall) -> RecoveryToolCall:
        key = self._grounded_setup_key(call, {})
        if key is None or key not in self._blocked_grounded_setups:
            return call
        instance = self._raw_grounded_instance_for_call(call)
        if (
            instance is not None
            and uses_operation_pose_candidate(key[2])
            and operation_pose_candidates(instance)
        ):
            # Structured candidates are blocked individually.  A historical
            # single-pose failure must not suppress untried candidates.
            return call
        instance_id, arm, point_key = key
        return RecoveryToolCall(
            tool_name="reobserve_scene",
            args={
                "_guard_reason": (
                    "blocked repeated no-progress grounded setup "
                    f"instance={instance_id},arm={arm},point={point_key}; select another arm or strategy"
                )
            },
        )

    def _stage_active_contact_sequences(self, calls: list[RecoveryToolCall]) -> list[RecoveryToolCall]:
        staged = list(calls)
        transparent_tools = {"close_gripper", "open_gripper", "reobserve_scene"}
        for index, call in enumerate(staged):
            if call.tool_name != "move_ee_to_grounded_instance":
                continue
            if self._normalized_grounded_point_key(call) != "contact_world_m":
                continue
            args = dict(call.args or {})
            if bool(args.get("preserve_height", False)):
                continue
            offset = self._xyz_prefix(args.get("offset_xyz", [0.0, 0.0, 0.0]))
            if offset is None or sum(value * value for value in offset) > 1e-10:
                continue
            instance = self._grounded_instance_for_call(call)
            if instance is None:
                continue
            approach = self._xyz_prefix(instance.get("approach_world_m"))
            contact = self._xyz_prefix(instance.get("contact_world_m"))
            if approach is None or contact is None:
                continue
            arm = self._single_arm_from_call(call)
            active_contact: RecoveryToolCall | None = None
            for candidate in staged[index + 1 :]:
                if candidate.tool_name in transparent_tools:
                    continue
                if candidate.tool_name == "contact_displace" and self._single_arm_from_call(candidate) == arm:
                    active_contact = candidate
                break
            if active_contact is None or not self._contact_displacement_follows_approach(
                active_contact,
                approach=approach,
                contact=contact,
            ):
                continue
            displacement = self._contact_displacement_vector(active_contact)
            if displacement is None:
                continue
            displacement_norm = sum(value * value for value in displacement) ** 0.5
            if displacement_norm <= 1e-8:
                continue
            args["point_key"] = "contact_world_m"
            args["offset_xyz"] = [
                -value - (value / displacement_norm) * _MAX_GROUNDED_CONTACT_OVERTRAVEL_M
                for value in displacement
            ]
            guard_reason = (
                "staged active contact from a displacement-matched pre-contact offset with compliance margin "
                "instead of commanding full contact before inward displacement"
            )
            existing_reason = str(args.get("_guard_reason", "")).strip()
            args["_guard_reason"] = f"{existing_reason}; {guard_reason}" if existing_reason else guard_reason
            staged[index] = RecoveryToolCall(tool_name=call.tool_name, args=args)
        return staged

    def _guard_recently_resolved_release_clearance(
        self,
        calls: list[RecoveryToolCall],
    ) -> list[RecoveryToolCall]:
        if not self._recent_release_resolutions:
            return calls
        current_step = (
            int(self.latest_snapshot.step_count)
            if self.latest_snapshot is not None
            else -1
        )
        relevant = {
            arm: resolution
            for arm, resolution in self._recent_release_resolutions.items()
            if abs(
                current_step
                - int(resolution.get("env_step", current_step))
            )
            <= 1
        }
        if not relevant:
            self._recent_release_resolutions.clear()
            return calls

        clearance_tools = {
            "lift_ee",
            "move_ee_to_pose",
            "move_to_home",
            "retreat_arm",
            "safe_reset_posture",
        }
        blocked_arms: set[str] = set()
        for call in calls:
            arm = self._single_arm_from_call(call)
            selected_arms = (
                set(relevant)
                if arm == "both"
                else {arm}
                if arm in relevant
                else set()
            )
            if not selected_arms:
                continue
            released_object_move = False
            if call.tool_name == "move_ee_to_grounded_instance":
                instance_id = self._recovery_history_instance_id(
                    call,
                    {},
                )
                released_object_move = any(
                    instance_id
                    and instance_id
                    == str(
                        relevant[selected_arm].get(
                            "held_instance_id",
                            "",
                        )
                        or ""
                    ).strip()
                    for selected_arm in selected_arms
                )
            if call.tool_name in clearance_tools or released_object_move:
                blocked_arms.update(selected_arms)

        if blocked_arms:
            resolutions = {
                arm: self._recent_release_resolutions.pop(arm)
                for arm in sorted(blocked_arms)
                if arm in self._recent_release_resolutions
            }
            payload = {
                "arms": sorted(blocked_arms),
                "blocked_tools": [
                    call.tool_name
                    for call in calls
                    if call.tool_name != "reobserve_scene"
                ],
                "resolutions": resolutions,
                "reason": (
                    "runtime already verified release target, EE clearance, "
                    "detachment, and stability before dispatch"
                ),
            }
            self._trace(
                "resolved_release_clearance_batch_blocked",
                **payload,
            )
            self._dump_rollout_event(
                "resolved_release_clearance_batch_blocked",
                **payload,
            )
            return [
                RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_guard_reason": (
                            "blocked stale post-release clearance because "
                            "deterministic runtime verification already "
                            "resolved the release"
                        )
                    },
                )
            ]

        touched_arms = {
            arm
            for call in calls
            if call.tool_name != "reobserve_scene"
            for arm in (
                set(relevant)
                if self._single_arm_from_call(call) == "both"
                else {self._single_arm_from_call(call)}
            )
            if arm in relevant
        }
        if touched_arms or any(
            call.tool_name == "reobserve_scene" for call in calls
        ):
            for arm in touched_arms or set(relevant):
                self._recent_release_resolutions.pop(arm, None)
        return calls

    @staticmethod
    def _release_waits_for_stability_only(
        validation: dict[str, Any],
    ) -> bool:
        """Return whether placement is complete except for a fresh stability sample."""

        return bool(
            validation.get("applicable") is True
            and validation.get("release_verification_required") is True
            and validation.get("verified") is not True
            and validation.get("pre_release_validated") is True
            and validation.get("release_executed") is True
            and validation.get("gripper_open") is True
            and validation.get("object_at_target") is True
            and validation.get("ee_cleared_from_release_pose") is True
            and validation.get("detachment_verified") is True
            and validation.get("stable_across_fresh_observations") is not True
        )

    @staticmethod
    def _release_requires_placement_recovery(
        validation: dict[str, Any],
    ) -> bool:
        """Return whether a detached release has an observed target miss."""

        return bool(
            validation.get("applicable") is True
            and validation.get("release_verification_required") is True
            and validation.get("verified") is not True
            and validation.get("pre_release_validated") is True
            and validation.get("release_executed") is True
            and validation.get("gripper_open") is True
            and validation.get("object_position_observed") is True
            and validation.get("object_at_target") is False
            and validation.get("ee_cleared_from_release_pose") is True
            and validation.get("detachment_verified") is True
            and validation.get("placement_recovery_required") is True
        )

    def _guard_pending_release_verification(
        self,
        calls: list[RecoveryToolCall],
    ) -> list[RecoveryToolCall]:
        """Keep a released object unchanged while collecting stability evidence.

        Once the object is at its validated target, the gripper is open, the
        object is detached, and the EE has already cleared the release pose,
        another physical action cannot add verification evidence.  It can only
        disturb the placement.  Replace such a batch with a forced fresh
        observation.  If the object has already drifted outside the target
        tolerance, this guard deliberately does not apply so a later recovery
        batch may regrasp and replace it.
        """

        pending = self._pending_release_verification_states()
        if not pending or not calls:
            return calls
        waiting: dict[str, dict[str, Any]] = {}
        for arm in sorted(pending):
            validation = self._runtime_place_effect_validation(
                calls=[],
                results=[],
                pending_arm=arm,
            )
            if self._release_waits_for_stability_only(validation):
                waiting[arm] = validation
        if not waiting:
            return calls
        physical_calls = [
            call for call in calls if call.tool_name != "reobserve_scene"
        ]
        if not physical_calls:
            return calls

        payload = {
            "arms": sorted(waiting),
            "blocked_tools": [call.tool_name for call in physical_calls],
            "validations": waiting,
            "reason": (
                "released object is already at its validated target, detached, "
                "and clear of the EE; only a fresh stability observation remains"
            ),
        }
        self._trace("pending_release_physical_batch_blocked", **payload)
        self._dump_rollout_event(
            "pending_release_physical_batch_blocked",
            **payload,
        )
        return [
            RecoveryToolCall(
                tool_name="reobserve_scene",
                args={
                    "_guard_reason": (
                        "blocked physical action during release verification "
                        "because target, detachment, and EE clearance are "
                        "already satisfied; collect a fresh stability "
                        "observation without moving the robot"
                    ),
                    "_release_verification_arms": sorted(waiting),
                },
            )
        ]

    def _repair_pending_instance_for_arm(
        self,
        arm: str,
    ) -> dict[str, Any] | None:
        if arm not in {"left", "right"}:
            return None
        manipulation = (
            self.memory_store.state.working.manipulation_state
        )
        arm_state = (
            manipulation.get(arm)
            if isinstance(manipulation, dict)
            else None
        )
        if not isinstance(arm_state, dict):
            return None
        if str(arm_state.get("phase", "") or "").strip().lower() != (
            "release_recovery_required"
        ):
            return None
        instance_ref = str(
            arm_state.get(
                "released_instance_id",
                arm_state.get("held_instance_id", ""),
            )
            or ""
        ).strip()
        if not instance_ref:
            return None
        instance = self._scene_instance_by_ref(
            self.memory_store.state.working.scene_memory,
            instance_ref,
        )
        if not isinstance(instance, dict):
            return None
        if str(
            instance.get("action_geometry_state", "") or ""
        ).strip().lower() != "relocation_pending":
            return None
        return instance

    def _block_repair_pending_call(
        self,
        *,
        call: RecoveryToolCall,
        arm: str,
        instance: dict[str, Any],
        reason: str,
    ) -> RecoveryToolCall:
        instance_id = str(
            instance.get("instance_id", "") or ""
        ).strip()
        track_id = str(instance.get("track_id", "") or "").strip()
        payload = {
            "policy": SAFE_MOTION_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
            "arm": arm,
            "instance_id": instance_id,
            "track_id": track_id,
            "blocked_tool": call.tool_name,
            "reason": reason,
        }
        self._trace(
            "relocation_pending_unsafe_motion_blocked",
            **payload,
        )
        self._dump_rollout_event(
            "relocation_pending_unsafe_motion_blocked",
            **payload,
        )
        return self._relocation_pending_reobserve_call(
            arm=arm,
            instance_id=instance_id,
            track_id=track_id,
            status=str(instance.get("status", "") or ""),
            stability=str(instance.get("stability", "") or ""),
            policy=SAFE_MOTION_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
            reason_prefix=reason,
        )

    def _guard_repair_pending_safe_motion_tool(
        self,
        call: RecoveryToolCall,
    ) -> RecoveryToolCall:
        if self._action_geometry_repair_pending_policy() != (
            SAFE_MOTION_ACTION_GEOMETRY_REPAIR_PENDING_POLICY
        ):
            return call
        arm = self._single_arm_from_call(call)
        instance = self._repair_pending_instance_for_arm(arm)
        if instance is None:
            return call
        args = dict(call.args or {})
        if call.tool_name in {"reobserve_scene", "open_gripper"}:
            return call
        if call.tool_name == "move_ee_to_pose":
            if (
                args.get(
                    "_runtime_action_geometry_repair_safe_motion"
                )
                is True
                or args.get("_runtime_visual_clearance") is True
            ):
                return call
            return self._block_repair_pending_call(
                call=call,
                arm=arm,
                instance=instance,
                reason=(
                    "blocked planner-authored absolute pose while action "
                    "geometry is pending; only runtime-derived high motion "
                    "or visual clearance is allowed"
                ),
            )
        if call.tool_name in {"retreat_arm", "lift_ee"}:
            current_pose = (
                getattr(self.latest_snapshot, f"{arm}_endpose", None)
                if self.latest_snapshot is not None
                else None
            )
            pose_values = (
                current_pose.tolist()
                if hasattr(current_pose, "tolist")
                else current_pose
            )
            axis = (
                "z"
                if call.tool_name == "lift_ee"
                else args.get("axis", "x")
            )
            direction = (
                "positive"
                if call.tool_name == "lift_ee"
                else args.get("direction", "negative")
            )
            if repair_pending_retreat_is_safe(
                instance=instance,
                current_pose=pose_values,
                gripper_state=(
                    self._recovery_gripper_state().get(arm)
                ),
                axis=axis,
                direction=direction,
                distance_m=args.get("distance", 0.03),
                maximum_distance_m=0.05,
            ):
                args[
                    "_runtime_action_geometry_repair_retreat"
                ] = True
                reason = (
                    "bounded open-gripper retreat does not reduce clearance "
                    "from the pending object"
                )
                existing_reason = str(
                    args.get("_guard_reason", "") or ""
                ).strip()
                args["_guard_reason"] = (
                    f"{existing_reason}; {reason}"
                    if existing_reason
                    else reason
                )
                return RecoveryToolCall(
                    tool_name=call.tool_name,
                    args=args,
                )
            return self._block_repair_pending_call(
                call=call,
                arm=arm,
                instance=instance,
                reason=(
                    "retreat was not proven to preserve or increase "
                    "clearance from the pending object"
                ),
            )
        return self._block_repair_pending_call(
            call=call,
            arm=arm,
            instance=instance,
            reason=(
                "repair-pending safe_motion permits only bounded retreat, "
                "runtime-derived high motion, visual clearance, and "
                "observation; descent, contact, and close remain blocked"
            ),
        )

    def _apply_recovery_manipulation_guards(self, calls: list[RecoveryToolCall]) -> list[RecoveryToolCall]:
        self._resolve_pending_release_from_runtime(
            source="pre_dispatch",
        )
        if self._release_guard_enabled():
            calls = self._guard_recently_resolved_release_clearance(calls)
            calls = self._guard_pending_release_verification(calls)
            calls = self._drop_redundant_support_contact_settle(calls)
        guarded: list[RecoveryToolCall] = []
        gripper_state = self._recovery_gripper_state()
        last_grounded_point_key_by_arm: dict[str, str] = {}
        last_grounded_call_by_arm: dict[str, RecoveryToolCall] = {}
        prior_physical_arms: set[str] = set()
        for raw_call in calls:
            raw_arm = self._single_arm_from_call(raw_call)
            policy_guarded = (
                self._guard_repair_pending_safe_motion_tool(raw_call)
            )
            if (
                raw_call.tool_name != "reobserve_scene"
                and policy_guarded.tool_name == "reobserve_scene"
            ):
                guarded.append(policy_guarded)
                break
            call = policy_guarded
            if self._release_guard_enabled():
                call = self._guard_open_gripper_while_holding(
                    policy_guarded,
                    place_setup=last_grounded_call_by_arm.get(raw_arm),
                    prior_physical_arms=prior_physical_arms,
                )
            if (
                raw_call.tool_name == "open_gripper"
                and call.tool_name == "reobserve_scene"
            ):
                guarded.append(call)
                break
            call = self._guard_cross_arm_target_occupancy(call)
            call = self._guard_grounded_contact_overtravel(call, last_grounded_call_by_arm)
            required_gripper_state = self._required_gripper_transition(call, gripper_state)
            if required_gripper_state:
                arm = self._single_arm_from_call(call)
                gripper_tool = "open_gripper" if required_gripper_state == "open" else "close_gripper"
                guarded.append(
                    RecoveryToolCall(
                        tool_name=gripper_tool,
                        args={
                            "arm": arm,
                            "_guard_reason": f"planner requested gripper_precondition={required_gripper_state}",
                        },
                    )
                )
                self._update_guard_gripper_state(
                    RecoveryToolCall(tool_name=gripper_tool, args={"arm": arm}),
                    gripper_state,
                    required_gripper_state,
                )
            if self._is_unsafe_close_after_approach(call, last_grounded_point_key_by_arm):
                guarded.append(
                    RecoveryToolCall(
                        tool_name="reobserve_scene",
                        args={
                            "_guard_reason": "blocked close_gripper after approach-only grounded move; request grasp_world_m/contact_world_m first",
                        },
                    )
                )
                break
            guarded.append(call)
            if call.tool_name == "move_ee_to_grounded_instance":
                arm = self._single_arm_from_call(call)
                last_grounded_point_key_by_arm[arm] = self._normalized_grounded_point_key(call)
                last_grounded_call_by_arm[arm] = call
            elif call.tool_name in {
                "lift_ee",
                "move_ee_to_pose",
                "move_to_home",
                "retreat_arm",
                "safe_reset_posture",
            }:
                arm = self._single_arm_from_call(call)
                if arm == "both":
                    last_grounded_call_by_arm.clear()
                else:
                    last_grounded_call_by_arm.pop(arm, None)
            if call.tool_name == "open_gripper":
                self._update_guard_gripper_state(call, gripper_state, "open")
            elif call.tool_name == "close_gripper":
                self._update_guard_gripper_state(call, gripper_state, "closed")
            physical_arm = self._single_arm_from_call(call)
            if (
                call.tool_name != "reobserve_scene"
                and physical_arm in {"left", "right"}
            ):
                prior_physical_arms.add(physical_arm)
            if call.tool_name == "reobserve_scene":
                break
        return guarded

    def _support_contact_release_authorization(
        self,
        *,
        arm: str,
        holding_state: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if arm not in {"left", "right"} or not isinstance(
            holding_state,
            dict,
        ):
            return None
        held_instance = self._scene_instance_by_ref(
            self.memory_store.state.working.scene_memory,
            str(holding_state.get("held_instance_id", "") or "").strip(),
        )
        robot_state = self._get_robot_state()
        calibration = (
            (
                getattr(
                    self.latest_snapshot,
                    "tcp_calibration_by_arm",
                    {},
                )
                or {}
            ).get(arm)
            if self.latest_snapshot is not None
            else None
        )
        return authorize_support_contact_release(
            scene_memory=self.memory_store.state.working.scene_memory,
            held_instance=held_instance,
            arm=arm,
            arm_state=holding_state,
            robot_arm_state=(
                robot_state.get(arm)
                if isinstance(robot_state, dict)
                else None
            ),
            calibration=calibration,
            gripper_closed=(
                self._recovery_gripper_state().get(arm) == "closed"
            ),
            limits=self._support_contact_limits(),
        )

    def _drop_redundant_support_contact_settle(
        self,
        calls: list[RecoveryToolCall],
    ) -> list[RecoveryToolCall]:
        holding_states = {
            arm: self._transport_holding_state(arm)
            for arm in ("left", "right")
        }
        authorizations = {
            arm: self._support_contact_release_authorization(
                arm=arm,
                holding_state=holding_states[arm],
            )
            for arm in ("left", "right")
        }
        filtered, dropped = drop_redundant_support_contact_settles(
            calls,
            authorizations=authorizations,
            holding_states=holding_states,
            limits=self._support_contact_limits(),
        )
        for payload in dropped:
            self._trace(
                "redundant_support_contact_settle_dropped",
                **payload,
            )
            self._dump_rollout_event(
                "redundant_support_contact_settle_dropped",
                **payload,
            )
        return filtered

    def _guard_open_gripper_while_holding(
        self,
        call: RecoveryToolCall,
        *,
        place_setup: RecoveryToolCall | None = None,
        prior_physical_arms: set[str] | None = None,
    ) -> RecoveryToolCall:
        if call.tool_name != "open_gripper":
            return call
        if is_runtime_failed_grasp_clearance_call(call):
            return call
        arm = self._single_arm_from_call(call)
        selected_arms = (
            ("left", "right")
            if arm == "both"
            else (arm,)
            if arm in {"left", "right"}
            else ()
        )
        holding_states = [
            state
            for selected_arm in selected_arms
            if (state := self._transport_holding_state(selected_arm)) is not None
        ]
        if not holding_states:
            manipulation = (
                self.memory_store.state.working.manipulation_state
            )
            pending_arms = [
                selected_arm
                for selected_arm in selected_arms
                if isinstance(manipulation, dict)
                and isinstance(manipulation.get(selected_arm), dict)
                and str(
                    manipulation[selected_arm].get("phase", "") or ""
                ).strip().lower()
                in {
                    "grasp_candidate",
                    "failed_grasp_clearance_pending",
                }
            ]
            if pending_arms:
                if self._grasp_transport_evidence_only():
                    return call
                return RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_guard_reason": (
                            "blocked open_gripper for a pending grasp because "
                            "only runtime-proven failure clearance may release it"
                        )
                    },
                )
            return call
        if len(selected_arms) == 1 and len(holding_states) == 1:
            provisional_state = holding_states[0]
            if (
                provisional_state.get("holding_confirmed") is not True
                and str(
                    provisional_state.get("grasp_transport_policy", "")
                    or ""
                ).strip().lower()
                == "evidence_only"
            ):
                provisional_args = dict(call.args or {})
                provisional_place_intent = bool(
                    provisional_args.get("release_target_id")
                    or provisional_args.get("target_id")
                    or place_setup is not None
                    or provisional_state.get("phase") == "place_aligned"
                )
                if not provisional_place_intent:
                    # The planner may explicitly abandon an assumed grasp.
                    # Do not convert that choice into automatic observation or
                    # fixed reverse-ingress recovery.
                    return call
        release_ref = str(
            (call.args or {}).get("release_held_instance_id", "") or ""
        ).strip().lower()
        held_ids = {
            str(state.get("held_instance_id", "") or "").strip().lower()
            for state in holding_states
            if str(state.get("held_instance_id", "") or "").strip()
        }
        if len(selected_arms) == 1 and release_ref and release_ref in held_ids:
            holding_state = holding_states[0]
            support_contact_state = holding_state.get(
                "support_contact_release"
            )
            args = dict(call.args or {})
            release_target_id = str(
                args.get("release_target_id", args.get("target_id", ""))
                or holding_state.get("place_target_id", "")
                or (
                    support_contact_state.get("target_id", "")
                    if isinstance(support_contact_state, dict)
                    else ""
                )
                or ""
            ).strip()
            setup_mode = ""
            if place_setup is not None:
                setup_mode = operation_action_mode(
                    self._normalized_grounded_point_key(place_setup),
                    (place_setup.args or {}).get(
                        "_operation_action_mode",
                        (place_setup.args or {}).get("action_mode"),
                    ),
                )
            placement_release = bool(
                release_target_id
                or setup_mode == "place"
                or holding_state.get("phase") == "place_aligned"
            )
            if not placement_release:
                return RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_guard_reason": (
                            "blocked confirmed-hold release without a runtime-"
                            "validated place target or support-contact state"
                        )
                    },
                )
            if any(
                physical_arm != selected_arms[0]
                for physical_arm in (prior_physical_arms or set())
            ):
                return RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_guard_reason": (
                            "blocked placement release after another arm moved "
                            "within the same static scene snapshot; refresh "
                            "occupancy before releasing"
                        )
                    },
                )
            if place_setup is None:
                support_authorization = (
                    self._support_contact_release_authorization(
                        arm=selected_arms[0],
                        holding_state=holding_state,
                    )
                )
                if support_authorization is not None:
                    candidate, validation = support_authorization
                    candidate_target_id = str(
                        candidate.get("target_id", "") or ""
                    ).strip()
                    if prior_physical_arms:
                        return RecoveryToolCall(
                            tool_name="reobserve_scene",
                            args={
                                "_guard_reason": (
                                    "blocked support-contact release after an "
                                    "additional physical action in the same "
                                    "batch; refresh support evidence first"
                                )
                            },
                        )
                    if (
                        release_target_id
                        and release_target_id != candidate_target_id
                    ):
                        return RecoveryToolCall(
                            tool_name="reobserve_scene",
                            args={
                                "_guard_reason": (
                                    "blocked support-contact release because "
                                    "the persisted target identity changed"
                                )
                            },
                        )
                    args["release_target_id"] = candidate_target_id
                    args["_release_place_candidate"] = dict(candidate)
                    args["_release_place_validation"] = dict(validation)
                    args["_runtime_support_contact_release"] = True
                    return RecoveryToolCall(
                        tool_name=call.tool_name,
                        args=args,
                    )
            if (
                place_setup is None
                or setup_mode != "place"
                or self._normalized_grounded_point_key(place_setup)
                != "place_world_m"
            ):
                return RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_guard_reason": (
                            "blocked placement release without a same-batch final "
                            "place_world_m setup; refresh and revalidate the public target_id"
                        )
                    },
                )
            setup_target_id = str(
                (place_setup.args or {}).get("target_id", "") or ""
            ).strip()
            if not release_target_id:
                release_target_id = setup_target_id
            if not release_target_id or setup_target_id != release_target_id:
                return RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_guard_reason": (
                            "blocked placement release because release_target_id "
                            "does not match the final place setup"
                        )
                    },
                )
            held_instance = self._raw_grounded_instance_for_call(place_setup)
            arm_name = selected_arms[0]
            if held_instance is None:
                return RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_guard_reason": (
                            "blocked placement release because the held instance "
                            "could not be rebound in current scene memory"
                        )
                    },
                )
            selected = select_operation_pose_candidate(
                held_instance,
                arm=arm_name,
                action_mode="place",
                requested_candidate_id=(place_setup.args or {}).get(
                    "_operation_candidate_id"
                ),
                requested_target_id=release_target_id,
            )
            validation = (
                validate_place_candidate(
                    (place_setup.args or {}).get("_scene_memory", {}),
                    held_instance=held_instance,
                    candidate=selected,
                )
                if selected is not None
                else {"valid": False}
            )
            if validation.get("valid") is not True:
                return RecoveryToolCall(
                    tool_name="reobserve_scene",
                    args={
                        "_guard_reason": (
                            "blocked placement release because the target is no "
                            "longer free/support-valid under current geometry"
                        )
                    },
                )
            args["release_target_id"] = release_target_id
            args["_release_place_candidate"] = dict(selected)
            args["_release_place_validation"] = dict(validation)
            return RecoveryToolCall(tool_name=call.tool_name, args=args)
        return RecoveryToolCall(
            tool_name="reobserve_scene",
            args={
                "_guard_reason": (
                    "blocked open_gripper because runtime has a verifier-confirmed held "
                    "instance or a provisionally transport-authorized held instance; "
                    "continue carry/place, or explicitly provide the matching "
                    "release_held_instance_id when a verified support state justifies release"
                )
            },
        )

    def _guard_cross_arm_target_occupancy(self, call: RecoveryToolCall) -> RecoveryToolCall:
        if call.tool_name != "move_ee_to_grounded_instance":
            return call
        arm = self._single_arm_from_call(call)
        if arm not in {"left", "right"}:
            return call
        instance = self._grounded_instance_for_call(call)
        if instance is None:
            return call
        point_key = self._normalized_grounded_point_key(call)
        target = self._xyz_prefix(instance.get(point_key))
        offset = self._xyz_prefix((call.args or {}).get("offset_xyz", [0.0, 0.0, 0.0]))
        if target is None or offset is None:
            return call
        target = [target[index] + offset[index] for index in range(3)]
        other_arm = "right" if arm == "left" else "left"
        robot_state = self._get_robot_state()
        other_xyz = self._xyz_prefix((robot_state.get(other_arm) or {}).get("xyz")) if isinstance(robot_state, dict) else None
        if other_xyz is None:
            return call
        clearance = sum((target[index] - other_xyz[index]) ** 2 for index in range(3)) ** 0.5
        if clearance >= _MIN_DUAL_ARM_TARGET_CLEARANCE_M:
            return call
        instance_id = self._recovery_history_instance_id(call, {})
        return RecoveryToolCall(
            tool_name="reobserve_scene",
            args={
                "_guard_reason": (
                    f"blocked {arm} grounded move to {instance_id or 'target'} because {other_arm} EE occupies "
                    f"the target envelope at {clearance:.4g}m; retreat and verify the occupying arm first"
                )
            },
        )

    def _guard_grounded_contact_overtravel(
        self,
        call: RecoveryToolCall,
        last_grounded_call_by_arm: dict[str, RecoveryToolCall],
    ) -> RecoveryToolCall:
        if call.tool_name != "contact_displace":
            return call
        arm = self._single_arm_from_call(call)
        if arm not in {"left", "right"}:
            return call
        grounded_call = last_grounded_call_by_arm.get(arm)
        if grounded_call is None or self._normalized_grounded_point_key(grounded_call) != "contact_world_m":
            return call
        grounded_args = dict(grounded_call.args or {})
        if bool(grounded_args.get("preserve_height", False)):
            return call
        grounded_offset = self._xyz_prefix(grounded_args.get("offset_xyz", [0.0, 0.0, 0.0]))
        if grounded_offset is None or sum(value * value for value in grounded_offset) > 1e-10:
            return call

        instance = self._grounded_instance_for_call(grounded_call)
        if instance is None:
            return call
        approach = self._xyz_prefix(instance.get("approach_world_m"))
        contact = self._xyz_prefix(instance.get("contact_world_m"))
        if approach is None or contact is None:
            return call

        args = dict(call.args or {})
        axis = str(args.get("axis", "z")).strip().lower()
        direction = str(args.get("direction", "negative")).strip().lower()
        if axis not in {"x", "y", "z"} or direction not in {"positive", "negative"}:
            return call
        try:
            distance = float(args.get("distance", 0.02))
        except (TypeError, ValueError):
            return call
        if distance != distance or distance <= 0.0:
            return call

        if not self._contact_displacement_follows_approach(call, approach=approach, contact=contact):
            return call

        args["_contact_reference_world_m"] = list(contact)
        args["_contact_reference_tolerance_m"] = _MAX_GROUNDED_CONTACT_OVERTRAVEL_M
        if distance > _MAX_GROUNDED_CONTACT_OVERTRAVEL_M:
            args["distance"] = _MAX_GROUNDED_CONTACT_OVERTRAVEL_M
            guard_reason = (
                "capped same-direction displacement after grounded contact pose "
                f"from {distance:.4g}m to {_MAX_GROUNDED_CONTACT_OVERTRAVEL_M:.4g}m"
            )
            existing_reason = str(args.get("_guard_reason", "")).strip()
            args["_guard_reason"] = f"{existing_reason}; {guard_reason}" if existing_reason else guard_reason
        return RecoveryToolCall(tool_name=call.tool_name, args=args)

    def _contact_displacement_follows_approach(
        self,
        call: RecoveryToolCall,
        *,
        approach: list[float],
        contact: list[float],
    ) -> bool:
        displacement = self._contact_displacement_vector(call)
        if displacement is None:
            return False
        contact_direction = [contact[index] - approach[index] for index in range(3)]
        contact_norm = sum(value * value for value in contact_direction) ** 0.5
        displacement_norm = sum(value * value for value in displacement) ** 0.5
        if contact_norm <= 1e-8 or displacement_norm <= 1e-8:
            return False
        alignment = sum(
            contact_direction[index] * displacement[index] for index in range(3)
        ) / (contact_norm * displacement_norm)
        return alignment >= _MIN_GROUNDED_CONTACT_ALIGNMENT_COSINE

    def _contact_displacement_vector(self, call: RecoveryToolCall) -> list[float] | None:
        args = dict(call.args or {})
        axis = str(args.get("axis", "z")).strip().lower()
        direction = str(args.get("direction", "negative")).strip().lower()
        if axis not in {"x", "y", "z"} or direction not in {"positive", "negative"}:
            return None
        try:
            distance = float(args.get("distance", 0.02))
        except (TypeError, ValueError):
            return None
        if distance != distance or distance <= 0.0:
            return None
        displacement = [0.0, 0.0, 0.0]
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        displacement[axis_index] = distance if direction == "positive" else -distance
        return displacement

    def _raw_grounded_instance_for_call(self, call: RecoveryToolCall) -> dict[str, Any] | None:
        scene_memory = (call.args or {}).get("_scene_memory")
        if not isinstance(scene_memory, dict):
            return None
        instance_ref = self._recovery_history_instance_id(call, {}).strip().lower()
        if not instance_ref:
            return None
        matches: list[dict[str, Any]] = []
        for instance in scene_memory.get("instances", []) or []:
            if not isinstance(instance, dict):
                continue
            identity_refs = {
                str(instance.get(key, "") or "").strip().lower()
                for key in ("instance_id", "track_id", "oracle_id", "oracle_source_path")
                if str(instance.get(key, "") or "").strip()
            }
            if instance_ref in identity_refs:
                matches.append(instance)
        return matches[0] if len(matches) == 1 else None

    def _grounded_instance_for_call(self, call: RecoveryToolCall) -> dict[str, Any] | None:
        instance = self._raw_grounded_instance_for_call(call)
        if instance is None:
            return None
        arm = self._single_arm_from_call(call)
        if (
            arm not in {"left", "right"}
            or not uses_operation_pose_candidate(
                self._normalized_grounded_point_key(call)
            )
        ):
            return instance
        args = dict(call.args or {})
        action_mode = operation_action_mode(
            self._normalized_grounded_point_key(call),
            args.get("_operation_action_mode"),
        )
        selected = select_operation_pose_candidate(
            instance,
            arm=arm,
            action_mode=action_mode,
            blocked_candidate_ids=args.get("_blocked_operation_candidate_ids"),
            requested_candidate_id=args.get("_operation_candidate_id"),
            requested_target_id=args.get("target_id"),
        )
        return materialize_operation_candidate(instance, selected)

    def _blocked_candidate_ids(
        self,
        *,
        instance_id: str,
        arm: str,
        action_mode: str,
        instance: dict[str, Any],
        requested_target_id: Any = None,
        point_key: Any = None,
        offset_xyz: Any = None,
        offset_xyz_provided: bool = False,
        preserve_height: Any = False,
        target_quat_wxyz: Any = None,
        current_pose: Any = None,
    ) -> list[str]:
        lifecycle_kwargs: dict[str, Any] = {
            "instance_id": instance_id,
            "arm": arm,
            "action_mode": action_mode,
            "candidates": operation_pose_candidates(instance),
            "instance": instance,
            "requested_target_id": requested_target_id,
            "point_key": point_key,
            "preserve_height": preserve_height,
            "target_quat_wxyz": target_quat_wxyz,
            "current_pose": current_pose,
        }
        if offset_xyz_provided:
            lifecycle_kwargs["offset_xyz"] = offset_xyz
        blocked, revalidated = (
            self._operation_candidate_lifecycle.blocked_candidate_ids(
                **lifecycle_kwargs
            )
        )
        for payload in revalidated:
            history_entry = (
                "operation_candidate_revalidated:"
                f"instance={payload['instance_id']},arm={payload['arm']},"
                f"mode={payload['action_mode']},"
                f"point={payload.get('point_key', '')},"
                f"prior_geometry={payload['prior_geometry_revision']},"
                f"geometry={payload['geometry_revision']}"
            )
            target_id = str(
                payload.get("target_id", "") or ""
            ).strip()
            if payload.get("action_mode") == "place" and target_id:
                history_entry += f",target={target_id}"
            self.memory_store.record_recovery(history_entry)
            self._trace(
                "operation_candidate_revalidated",
                **payload,
            )
            self._dump_rollout_event(
                "operation_candidate_revalidated",
                **payload,
            )
        return blocked

    def _latest_arm_endpose(self, arm: str) -> Any:
        if self.latest_snapshot is None or arm not in {"left", "right"}:
            return None
        return getattr(
            self.latest_snapshot,
            f"{arm}_endpose",
            None,
        )

    def _is_unsafe_close_after_approach(self, call: RecoveryToolCall, last_grounded_point_key_by_arm: dict[str, str]) -> bool:
        if call.tool_name != "close_gripper":
            return False
        arm = self._single_arm_from_call(call)
        point_key = last_grounded_point_key_by_arm.get(arm, "")
        return point_key == "approach_world_m"

    def _normalized_grounded_point_key(self, call: RecoveryToolCall) -> str:
        raw = str((call.args or {}).get("point_key", "approach_world_m")).strip().lower()
        aliases = {
            "approach": "approach_world_m",
            "approach_point_world": "approach_world_m",
            "approach_world": "approach_world_m",
            "grasp": "grasp_world_m",
            "grasp_world": "grasp_world_m",
            "grasp_point_world": "grasp_world_m",
            "contact": "contact_world_m",
            "contact_world": "contact_world_m",
            "contact_point_world": "contact_world_m",
            "place": "place_world_m",
            "place_world": "place_world_m",
            "place_point_world": "place_world_m",
            "top_surface": "top_surface_world_m",
            "top_surface_world": "top_surface_world_m",
            "centroid": "world_m",
            "centroid_world": "world_m",
            "world": "world_m",
            "world_m": "world_m",
        }
        return aliases.get(raw, raw)

    def _required_gripper_transition(self, call: RecoveryToolCall, gripper_state: dict[str, str]) -> str:
        if call.tool_name in {"open_gripper", "close_gripper"}:
            return ""
        args = dict(call.args or {})
        required = str(
            args.get("gripper_precondition", args.get("requires_gripper_state", args.get("required_gripper_state", "")))
        ).strip().lower()
        aliases = {"opened": "open", "close": "closed", "close_gripper": "closed"}
        required = aliases.get(required, required)
        if required not in {"open", "closed"}:
            return ""
        arm = self._single_arm_from_call(call)
        if arm not in {"left", "right", "both"}:
            return ""
        selected_arms = ("left", "right") if arm == "both" else (arm,)
        if required == "open" and any(
            self._recovery_holding_authorized(selected_arm)
            for selected_arm in selected_arms
        ):
            return ""
        if all(gripper_state.get(selected_arm) == required for selected_arm in selected_arms):
            return ""
        return required

    def _requires_open_gripper_precondition(self, call: RecoveryToolCall, gripper_state: dict[str, str]) -> bool:
        return self._required_gripper_transition(call, gripper_state) == "open"

    def _update_guard_gripper_state(self, call: RecoveryToolCall, gripper_state: dict[str, str], value: str) -> None:
        arm = self._single_arm_from_call(call)
        if arm == "both":
            for key in ("left", "right"):
                gripper_state[key] = value
            return
        if arm in {"left", "right"}:
            gripper_state[arm] = value

    def _single_arm_from_call(self, call: RecoveryToolCall) -> str:
        arm = str((call.args or {}).get("arm") or "").strip().lower()
        aliases = {
            "all": "both",
            "dual": "both",
            "left_arm": "left",
            "right_arm": "right",
        }
        return normalize_physical_arm(aliases.get(arm, arm))

    def _annotate_recovery_call(self, call: RecoveryToolCall, reason: str) -> RecoveryToolCall:
        args = dict(call.args or {})
        args["_guard_reason"] = reason
        return RecoveryToolCall(tool_name=call.tool_name, args=args)

    def _recovery_gripper_state(self) -> dict[str, str]:
        robot_state = self._get_robot_state()
        return {arm: gripper_command_state(robot_state.get(arm)) for arm in ("left", "right")}

    def _scene_events_from_recovery_effect(
        self,
        *,
        calls: list[RecoveryToolCall],
        results: list[Any],
        action_effect: dict[str, Any],
        previous_manipulation_state: dict[str, Any],
        updated_manipulation_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        step = (
            int(self.latest_snapshot.step_count)
            if self.latest_snapshot is not None
            else -1
        )
        effect_verified = str(
            action_effect.get(
                "effect_verified",
                "unverified",
            )
            or "unverified"
        ).strip().lower()
        effect_type = str(
            action_effect.get("effect_type", "unknown")
            or "unknown"
        ).strip().lower()
        last_operation_ref_by_arm: dict[
            str,
            tuple[str, str],
        ] = {}
        events: list[dict[str, Any]] = []

        def motion_uncertain(
            instance_ref: str,
            *,
            source: str,
            reason: str,
        ) -> None:
            if not instance_ref:
                return
            events.append(
                {
                    "instance_ref": instance_ref,
                    "position_state": POSITION_MOTION_UNCERTAIN,
                    "source": source,
                    "reason": reason,
                    "env_step": step,
                }
            )

        for index, call in enumerate(calls):
            result = (
                results[index]
                if index < len(results)
                else None
            )
            details = (
                dict(getattr(result, "details", {}) or {})
                if result is not None
                else {}
            )
            arm = self._single_arm_from_call(call)
            if call.tool_name == "move_ee_to_grounded_instance":
                point_key = self._normalized_grounded_point_key(
                    call
                )
                if point_key not in {
                    "grasp_world_m",
                    "contact_world_m",
                    "place_world_m",
                }:
                    continue
                instance_ref = (
                    self._recovery_history_instance_id(
                        call,
                        details,
                    )
                )
                if arm in {"left", "right"} and instance_ref:
                    last_operation_ref_by_arm[arm] = (
                        instance_ref,
                        point_key,
                    )
                if any(
                    details.get(key) is True
                    for key in (
                        "collision",
                        "stalled",
                        "valid_contact",
                    )
                ) and details.get("target_reached") is not True:
                    motion_uncertain(
                        instance_ref,
                        source="unverified_grounded_contact",
                        reason=(
                            "grounded motion reported contact, collision, "
                            "or stall without a verified object effect"
                        ),
                    )
                continue

            if call.tool_name == "close_gripper" and arm in {
                "left",
                "right",
            }:
                instance_ref, _ = last_operation_ref_by_arm.get(
                    arm,
                    ("", ""),
                )
                motion_uncertain(
                    instance_ref,
                    source="grasp_closure",
                    reason=(
                        "gripper closure may have displaced the object "
                        "until grasp coupling is verified"
                    ),
                )
                continue

            if call.tool_name == "contact_displace" and arm in {
                "left",
                "right",
            }:
                instance_ref, _ = last_operation_ref_by_arm.get(
                    arm,
                    ("", ""),
                )
                if effect_verified != "true":
                    motion_uncertain(
                        instance_ref,
                        source="unverified_contact_displacement",
                        reason=(
                            "contact displacement occurred without a "
                            "verified resulting object state"
                        ),
                    )
                continue

            if call.tool_name == "open_gripper":
                selected_arms = (
                    ("left", "right")
                    if arm == "both"
                    else (arm,)
                    if arm in {"left", "right"}
                    else ()
                )
                for selected_arm in selected_arms:
                    previous_state = (
                        previous_manipulation_state.get(
                            selected_arm
                        )
                    )
                    instance_ref = str(
                        (call.args or {}).get(
                            "release_held_instance_id",
                            "",
                        )
                        or (
                            previous_state.get(
                                "held_instance_id",
                                "",
                            )
                            if isinstance(
                                previous_state,
                                dict,
                            )
                            else ""
                        )
                        or ""
                    ).strip()
                    previous_phase = str(
                        previous_state.get("phase", "")
                        if isinstance(previous_state, dict)
                        else ""
                    ).strip().lower()
                    if (
                        previous_phase == "grasp_candidate"
                        and not bool(
                            previous_state.get("holding_confirmed")
                            if isinstance(previous_state, dict)
                            else False
                        )
                    ):
                        # Opening after an unverified grasp clears the attempt;
                        # it is not evidence that a held object was released.
                        # The earlier closure uncertainty remains until a clean
                        # visual observation reacquires the object.
                        continue
                    motion_uncertain(
                        instance_ref,
                        source="object_release",
                        reason=(
                            "released object position remains uncertain "
                            "until post-release validation succeeds"
                        ),
                    )

        # A confirmed/provisionally authorized attachment supersedes the
        # closure uncertainty with a current TCP-propagated position.
        events.extend(
            self._runtime_scene_position_events(
                updated_manipulation_state
            )
        )

        runtime_validation = action_effect.get(
            "runtime_place_validation"
        )
        if not isinstance(runtime_validation, dict):
            runtime_validation = {}
        if runtime_validation.get("verified") is True:
            instance_ref = str(
                runtime_validation.get(
                    "held_instance_id",
                    "",
                )
                or ""
            ).strip()
            release_arm, release_state = next(
                (
                    (str(arm), state)
                    for arm, state in (
                        previous_manipulation_state.items()
                    )
                    if (
                        isinstance(state, dict)
                        and str(
                            state.get("held_instance_id", "") or ""
                        ).strip()
                        == instance_ref
                    )
                ),
                ("", {}),
            )
            verified_world = self._xyz_prefix(
                runtime_validation.get(
                    "verified_object_world_m"
                )
            )
            if verified_world is None:
                verified_world = self._xyz_prefix(
                    release_state.get(
                        "held_object_target_world_m"
                    )
                )
            operation_target_world = self._xyz_prefix(
                release_state.get("held_object_target_world_m")
            )
            events.append(
                {
                    "instance_ref": instance_ref,
                    "position_state": POSITION_CURRENT_VERIFIED,
                    "world_m": verified_world,
                    "confidence": 1.0,
                    "tolerance_m": runtime_validation.get(
                        "target_tolerance_m"
                    ),
                    "operation_target_world_m": (
                        operation_target_world
                    ),
                    "operation_target_tolerance_m": (
                        runtime_validation.get("target_tolerance_m")
                    ),
                    "operation_target_id": str(
                        runtime_validation.get("target_id", "") or ""
                    ).strip(),
                    "arm": str(
                        runtime_validation.get("arm", release_arm) or ""
                    ).strip().lower(),
                    "source": "verified_post_release_position",
                    "reason": (
                        "release, detachment, target occupancy, and "
                        "stability were verified"
                    ),
                    "env_step": step,
                    "verified_role": "placed_object",
                    "effect_type": (
                        effect_type
                        if effect_type != "unknown"
                        else "place"
                    ),
                }
            )

        if effect_verified == "true":
            verified_refs: set[str] = set()
            if effect_type == "grasp":
                verified_refs.update(
                    str(state.get("held_instance_id", "") or "")
                    for state in updated_manipulation_state.values()
                    if (
                        isinstance(state, dict)
                        and state.get("holding_confirmed")
                        is True
                    )
                )
            else:
                verified_refs.update(
                    instance_ref
                    for instance_ref, point_key in (
                        last_operation_ref_by_arm.values()
                    )
                    if point_key in {
                        "contact_world_m",
                        "grasp_world_m",
                    }
                )
            role = (
                effect_type
                if effect_type not in {"", "unknown"}
                else "verified_interaction_target"
            )
            for instance_ref in sorted(
                ref for ref in verified_refs if ref
            ):
                events.append(
                    {
                        "instance_ref": instance_ref,
                        "verified_role": role,
                        "effect_type": effect_type,
                        "source": "verified_physical_effect",
                        "env_step": step,
                    }
                )
        return events

    def _update_manipulation_state_from_effect(
        self,
        *,
        calls: list[RecoveryToolCall],
        results: list[Any],
        action_effect: dict[str, Any],
    ) -> None:
        previous = {
            str(arm): dict(value)
            for arm, value in dict(
                self.memory_store.state.working.manipulation_state
            ).items()
            if isinstance(value, dict)
        }
        updated = {
            arm: dict(value)
            for arm, value in previous.items()
        }
        last_grounded_grasp_by_arm: dict[str, dict[str, Any]] = {}
        successful_lifts_by_arm: dict[str, dict[str, Any]] = {}
        effect_arms: set[str] = set()
        runtime_place_validation = action_effect.get(
            "runtime_place_validation"
        )
        if not isinstance(runtime_place_validation, dict):
            runtime_place_validation = {}
        step = (
            int(self.latest_snapshot.step_count)
            if self.latest_snapshot is not None
            else -1
        )
        updated = apply_failed_grasp_clearance_results(
            updated,
            calls=calls,
            results=results,
            env_step=step,
            fresh_physical_snapshot_satisfies_observation=(
                not self._recovery_dispatcher.reobserve_scene_enabled
            ),
        )
        ambiguous_return_reduction = None
        if any(
            is_runtime_ambiguous_grasp_return_call(call)
            for call in calls
        ):
            ambiguous_return_reduction = (
                reduce_ambiguous_grasp_return_results(
                    updated,
                    calls=calls,
                    results=results,
                    post_return_snapshot=self.latest_snapshot,
                )
            )
            updated = {
                str(arm): dict(state)
                for arm, state in (
                    ambiguous_return_reduction.manipulation_state
                ).items()
                if isinstance(state, dict)
            }
        release_recovery_transition: dict[str, Any] = {}
        diagnostic_grasp_lifts: list[dict[str, Any]] = []
        support_contact_release_transitions: list[dict[str, Any]] = []

        for index, call in enumerate(calls):
            result = results[index] if index < len(results) else None
            arm = self._single_arm_from_call(call)
            state_arms = (
                ("left", "right")
                if arm == "both"
                else (arm,)
                if arm in {"left", "right"}
                else ()
            )
            release_context_before_clear: dict[str, dict[str, Any]] = {}
            if (
                call.tool_name == "open_gripper"
                and not self._release_guard_enabled()
            ):
                for state_arm in state_arms:
                    state_context = (
                        self._placement_release_context_from_state(
                            updated.get(state_arm)
                        )
                    )
                    if state_context is not None:
                        release_context_before_clear[state_arm] = (
                            state_context
                        )
            if is_runtime_failed_grasp_clearance_call(call):
                continue
            if is_runtime_ambiguous_grasp_return_call(call):
                continue
            if call.tool_name != "reobserve_scene":
                for state_arm in state_arms:
                    if (
                        isinstance(updated.get(state_arm), dict)
                        and isinstance(
                            updated[state_arm].get(
                                "support_contact_release"
                            ),
                            dict,
                        )
                    ):
                        updated[state_arm] = (
                            clear_support_contact_release_state(
                                updated[state_arm]
                            )
                        )
            success = bool(result is not None and getattr(result, "success", False))
            if not success:
                continue
            if call.tool_name == "move_ee_to_grounded_instance":
                details = getattr(result, "details", {}) or {}
                operation_mode = str(
                    details.get(
                        "operation_action_mode",
                        (call.args or {}).get("_operation_action_mode", ""),
                    )
                    or ""
                ).strip().lower()
                point_key = self._normalized_grounded_point_key(call)
                if (
                    arm in {"left", "right"}
                    and operation_mode == "place"
                    and point_key == "place_world_m"
                    and details.get("target_reached") is not False
                ):
                    arm_state = updated.get(arm)
                    held_instance_id = self._recovery_history_instance_id(
                        call,
                        details,
                    )
                    if (
                        isinstance(arm_state, dict)
                        and manipulation_state_allows_transport(
                            arm_state,
                            active_grasp_transport_policy=(
                                self._grasp_transport_policy()
                            ),
                        )
                        and str(
                            arm_state.get("held_instance_id", "") or ""
                        ).strip()
                        == held_instance_id
                    ):
                        materialized = self._grounded_instance_for_call(call)
                        candidate = (
                            materialized.get(
                                "selected_operation_candidate"
                            )
                            if isinstance(materialized, dict)
                            else None
                        )
                        updated[arm] = {
                            **arm_state,
                            "phase": "place_aligned",
                            "place_target_id": str(
                                details.get(
                                    "operation_target_id",
                                    (call.args or {}).get("target_id", ""),
                                )
                                or ""
                            ).strip(),
                            "place_candidate_id": str(
                                details.get("operation_candidate_id", "")
                                or ""
                            ).strip(),
                            "held_object_target_world_m": (
                                dict(candidate).get(
                                    "held_object_target_world_m"
                                )
                                if isinstance(candidate, dict)
                                else details.get(
                                    "held_object_target_world_m"
                                )
                            ),
                            "held_extent_m": (
                                dict(candidate).get("held_extent_m")
                                if isinstance(candidate, dict)
                                else None
                            ),
                            "release_ee_target_world_m": self._xyz_prefix(
                                details.get("target_pose")
                            ),
                            "pre_release_validated": bool(
                                details.get("place_target_revalidated")
                                and details.get("support_valid")
                                and details.get("free")
                            ),
                            "updated_step": step,
                            "evidence": (
                                "reached_runtime_selected_place_pose"
                            ),
                        }
                if (
                    arm in {"left", "right"}
                    and operation_mode == "grasp"
                    and point_key
                    in {"grasp_world_m", "contact_world_m"}
                    and details.get("target_reached") is not False
                ):
                    instance_id = self._recovery_history_instance_id(
                        call,
                        details,
                    )
                    if instance_id:
                        last_grounded_grasp_by_arm[arm] = {
                            "instance_id": instance_id,
                            "point_key": point_key,
                            "operation_action_mode": operation_mode,
                            "operation_candidate_id": str(
                                details.get(
                                    "operation_candidate_id",
                                    (call.args or {}).get(
                                        "_operation_candidate_id",
                                        "",
                                    ),
                                )
                                or ""
                            ).strip(),
                            "occlusion_geometry_lease": bool(
                                (call.args or {}).get(
                                    "_runtime_occlusion_geometry_lease"
                                )
                            ),
                            "call": call,
                            "details": dict(details),
                        }
                continue
            if call.tool_name == "open_gripper":
                selected_arms = (
                    ("left", "right")
                    if arm == "both"
                    else (arm,)
                    if arm in {"left", "right"}
                    else ()
                )
                for selected_arm in selected_arms:
                    args = dict(call.args or {})
                    prior_state = updated.get(selected_arm)
                    prior_phase = str(
                        (
                            prior_state.get("phase", "")
                            if isinstance(prior_state, dict)
                            else ""
                        )
                        or ""
                    ).strip().lower()
                    if (
                        runtime_place_validation.get("verified") is True
                        and runtime_place_validation.get("arm")
                        == selected_arm
                    ):
                        updated.pop(selected_arm, None)
                        continue
                    if prior_phase in {
                        "release_pending_verification",
                        "release_recovery_required",
                    }:
                        # Repeating an explicit open cannot erase an existing
                        # post-release obligation or restart its freshness
                        # baseline.  The common validation transition below
                        # still resolves verified placements and target misses.
                        continue
                    release_target_id = str(
                        args.get("release_target_id", "") or ""
                    ).strip()
                    release_candidate = args.get(
                        "_release_place_candidate"
                    )
                    pre_release_validated = bool(
                        (
                            args.get("_release_place_validation")
                            or {}
                        ).get("valid")
                    )
                    if not release_target_id or not isinstance(
                        release_candidate,
                        dict,
                    ):
                        state_context = (
                            release_context_before_clear.get(selected_arm)
                            or self._placement_release_context_from_state(
                                prior_state
                            )
                        )
                        if state_context is None:
                            updated.pop(selected_arm, None)
                            continue
                        # Public release IDs are optional attribution hints
                        # when the Guard is disabled.  Record the actual
                        # runtime placement context instead of trusting a
                        # missing or mismatched planner label.
                        release_target_id = state_context["target_id"]
                        release_candidate = dict(
                            state_context["candidate"]
                        )
                        pre_release_validated = state_context[
                            "pre_release_validated"
                        ]
                    held_instance_id = str(
                        (
                            prior_state.get("held_instance_id", "")
                            if isinstance(prior_state, dict)
                            else ""
                        )
                        or args.get("release_held_instance_id", "")
                        or ""
                    ).strip()
                    updated[selected_arm] = {
                        "phase": "release_pending_verification",
                        "held_instance_id": held_instance_id,
                        "holding_confirmed": False,
                        "pregrasp_object_world_m": (
                            prior_state.get(
                                "pregrasp_object_world_m"
                            )
                            if isinstance(prior_state, dict)
                            else None
                        ),
                        "pregrasp_object_contact_world_m": (
                            prior_state.get(
                                "pregrasp_object_contact_world_m"
                            )
                            if isinstance(prior_state, dict)
                            else None
                        ),
                        **(
                            {
                                "held_object_perception_descriptor": dict(
                                    prior_state.get(
                                        "held_object_perception_descriptor"
                                    )
                                )
                            }
                            if (
                                isinstance(prior_state, dict)
                                and isinstance(
                                    prior_state.get(
                                        "held_object_perception_descriptor"
                                    ),
                                    dict,
                                )
                            )
                            else {}
                        ),
                        "place_target_id": release_target_id,
                        "place_candidate_id": release_candidate.get(
                            "candidate_id"
                        ),
                        "held_object_target_world_m": release_candidate.get(
                            "held_object_target_world_m"
                        ),
                        "held_extent_m": release_candidate.get(
                            "held_extent_m"
                        ),
                        "release_ee_target_world_m": self._xyz_prefix(
                            release_candidate.get("ee_target_pose")
                        ),
                        "pre_release_validated": bool(
                            pre_release_validated
                        ),
                        "source_skill_id": self._active_skill_id(),
                        "updated_step": step,
                        "evidence": (
                            "gripper_opened_at_revalidated_place_target;"
                            "awaiting_detachment_and_stability"
                        ),
                    }
                continue
            if call.tool_name == "contact_displace" and arm in {
                "left",
                "right",
            }:
                arm_state = updated.get(arm)
                scene_memory = self.memory_store.state.working.scene_memory
                held_instance = self._scene_instance_by_ref(
                    scene_memory,
                    str(
                        (arm_state or {}).get("held_instance_id", "")
                        or ""
                    ).strip(),
                )
                robot_state = self._get_robot_state()
                calibration = (
                    (
                        getattr(
                            self.latest_snapshot,
                            "tcp_calibration_by_arm",
                            {},
                        )
                        or {}
                    ).get(arm)
                    if self.latest_snapshot is not None
                    else None
                )
                release_evidence = (
                    create_support_contact_release_evidence(
                        scene_memory=scene_memory,
                        held_instance=held_instance,
                        arm=arm,
                        arm_state=arm_state,
                        robot_arm_state=(
                            robot_state.get(arm)
                            if isinstance(robot_state, dict)
                            else None
                        ),
                        calibration=calibration,
                        gripper_closed=(
                            self._recovery_gripper_state().get(arm)
                            == "closed"
                        ),
                        call_args=call.args,
                        result_details=dict(
                            getattr(result, "details", {}) or {}
                        ),
                        limits=self._support_contact_limits(),
                    )
                    if (
                        isinstance(arm_state, dict)
                        and manipulation_state_allows_transport(
                            arm_state,
                            active_grasp_transport_policy=(
                                self._grasp_transport_policy()
                            ),
                        )
                    )
                    else None
                )
                if release_evidence is not None:
                    candidate, validation, metrics = release_evidence
                    updated[arm] = install_support_contact_release_state(
                        arm_state=arm_state,
                        candidate=candidate,
                        validation=validation,
                        metrics=metrics,
                        env_step=step,
                    )
                    support_contact_release_transitions.append(
                        {
                            "arm": arm,
                            "held_instance_id": str(
                                arm_state.get("held_instance_id", "")
                                or ""
                            ).strip(),
                            "target_id": candidate.get("target_id"),
                            "target_kind": candidate.get("target_kind"),
                            "support_instance_id": candidate.get(
                                "support_instance_id"
                            ),
                            **metrics,
                            "target_geometry_drift_m": validation.get(
                                "target_geometry_drift_m"
                            ),
                        }
                    )
                continue
            diagnostic_grasp_motion = bool(
                arm in {"left", "right"}
                and (
                    call.tool_name == "lift_ee"
                    or is_runtime_grasp_diagnostic_lift_call(call)
                )
            )
            if arm in {"left", "right"} and (
                call.tool_name == "close_gripper"
                or diagnostic_grasp_motion
            ):
                effect_arms.add(arm)
            if diagnostic_grasp_motion:
                successful_lifts_by_arm[arm] = dict(
                    getattr(result, "details", {}) or {}
                )
            if call.tool_name != "close_gripper" or arm not in {"left", "right"}:
                continue
            grasp_setup = last_grounded_grasp_by_arm.get(arm, {})
            instance_id = str(
                grasp_setup.get("instance_id", "") or ""
            ).strip()
            if not instance_id:
                continue
            scene_instance = self._scene_instance_by_ref(
                self.memory_store.state.working.scene_memory,
                instance_id,
            )
            held_object_perception_descriptor = (
                self._perception_descriptor_for_scene_instance(
                    scene_instance
                )
            )
            setup_call = grasp_setup.get("call")
            materialized = (
                self._grounded_instance_for_call(setup_call)
                if isinstance(setup_call, RecoveryToolCall)
                else None
            )
            selected_candidate = (
                materialized.get("selected_operation_candidate")
                if isinstance(materialized, dict)
                else None
            )
            pregrasp_object_world = (
                self._xyz_prefix(
                    materialized.get(
                        "latest_world_m",
                        materialized.get("world_m"),
                    )
                )
                if isinstance(materialized, dict)
                else None
            ) or (
                self._xyz_prefix(
                    scene_instance.get(
                        "latest_world_m",
                        scene_instance.get("world_m"),
                    )
                )
                if isinstance(scene_instance, dict)
                else None
            )
            object_contact_world = (
                self._xyz_prefix(
                    selected_candidate.get("object_contact_pose")
                )
                if isinstance(selected_candidate, dict)
                else None
            )
            if object_contact_world is None:
                object_contact_world = pregrasp_object_world
            robot_state = self._get_robot_state()
            close_details = dict(
                getattr(result, "details", {}) or {}
            )
            close_pose = self._pose7_prefix(
                close_details.get("observed_pose")
            )
            if close_pose is None:
                observed_poses = close_details.get(
                    "observed_poses_by_arm"
                )
                close_pose = self._pose7_prefix(
                    observed_poses.get(arm)
                    if isinstance(observed_poses, dict)
                    else None
                )
            close_robot_arm_state = (
                {
                    "xyz": close_pose[:3],
                    "quat_wxyz": close_pose[3:],
                }
                if close_pose is not None
                else (
                    robot_state.get(arm)
                    if isinstance(robot_state, dict)
                    else None
                )
            )
            calibration = (
                (
                    getattr(
                        self.latest_snapshot,
                        "tcp_calibration_by_arm",
                        {},
                    )
                    or {}
                ).get(arm)
                if self.latest_snapshot is not None
                else None
            )
            attachment = capture_held_object_to_tcp_attachment(
                object_world_m=pregrasp_object_world,
                robot_arm_state=close_robot_arm_state,
                calibration=calibration,
            )
            grasp_candidate_id = str(
                grasp_setup.get("operation_candidate_id", "") or ""
            ).strip()
            grasp_attempt = grasp_attempt_metadata(
                arm=arm,
                held_instance_id=instance_id,
                grasp_candidate_id=grasp_candidate_id,
                env_step=(
                    close_details.get("step_count")
                    if close_details.get("step_count") is not None
                    else step
                ),
            )
            pending_attachment = bind_pending_grasp_attachment(
                attachment,
                grasp_candidate_id=grasp_candidate_id,
                grasp_attempt=grasp_attempt,
            )
            close_state = {
                "phase": "grasp_candidate",
                "held_instance_id": instance_id,
                "holding_confirmed": False,
                "transport_authorized": False,
                "operation_action_mode": str(
                    grasp_setup.get("operation_action_mode", "") or ""
                ).strip().lower(),
                "grasp_candidate_id": grasp_candidate_id,
                **grasp_attempt,
                "grasp_occlusion_geometry_lease": bool(
                    grasp_setup.get("occlusion_geometry_lease")
                ),
                "grasp_ee_target_world_m": (
                    self._xyz_prefix(
                        (
                            grasp_setup.get("details")
                            if isinstance(
                                grasp_setup.get("details"),
                                dict,
                            )
                            else {}
                        ).get("target_pose")
                    )
                    or (
                        self._xyz_prefix(
                            selected_candidate.get("ee_target_pose")
                        )
                        if isinstance(selected_candidate, dict)
                        else None
                    )
                ),
                "grasp_approach_world_m": (
                    self._xyz_prefix(
                        selected_candidate.get("approach_pose")
                    )
                    if isinstance(selected_candidate, dict)
                    else (
                        self._xyz_prefix(
                            materialized.get("approach_world_m")
                        )
                        if isinstance(materialized, dict)
                        else None
                    )
                ),
                "pregrasp_object_world_m": pregrasp_object_world,
                "pregrasp_object_contact_world_m": object_contact_world,
                **(
                    {
                        "held_object_perception_descriptor": (
                            held_object_perception_descriptor
                        )
                    }
                    if held_object_perception_descriptor
                    else {}
                ),
                **(
                    {
                        "held_object_to_tcp_attachment": (
                            pending_attachment
                        )
                    }
                    if pending_attachment is not None
                    else {}
                ),
                "source_skill_id": self._active_skill_id(),
                "updated_step": step,
                "evidence": "closed_after_reached_grounded_grasp",
            }
            updated[arm] = apply_close_time_grasp_transport_policy(
                close_state,
                grasp_transport_policy=self._grasp_transport_policy(),
            )

        effect_verified = str(
            action_effect.get("effect_verified", "unverified") or "unverified"
        ).strip().lower()
        effect_type = str(
            action_effect.get("effect_type", "unknown") or "unknown"
        ).strip().lower()
        runtime_grasp_validation = action_effect.get(
            "runtime_grasp_validation"
        )
        if not isinstance(runtime_grasp_validation, dict):
            runtime_grasp_validation = {}
        updated = merge_pending_grasp_validation(
            updated,
            validation=runtime_grasp_validation,
            env_step=step,
        )
        runtime_validation_arm = str(
            runtime_grasp_validation.get("arm", "") or ""
        ).strip().lower()
        if (
            runtime_grasp_validation.get("applicable") is True
            and runtime_validation_arm in {"left", "right"}
        ):
            effect_arms.add(runtime_validation_arm)
        if effect_verified == "true" and effect_type == "grasp":
            for arm in effect_arms:
                arm_state = updated.get(arm)
                if not isinstance(arm_state, dict):
                    continue
                held_instance_id = str(
                    arm_state.get("held_instance_id", "") or ""
                ).strip()
                if not held_instance_id:
                    continue
                matched_runtime_validation = (
                    matched_grasp_motion_validation(
                        action_effect,
                        arm=arm,
                        arm_state=arm_state,
                    )
                )
                if (
                    matched_runtime_validation is None
                    or matched_runtime_validation.get("verified") is not True
                ):
                    continue
                verified_attachment = (
                    promote_verified_grasp_attachment(
                        arm_state,
                        validation=matched_runtime_validation,
                    )
                )
                attachment_valid = verified_attachment is not None
                updated[arm] = {
                    **arm_state,
                    "phase": "holding",
                    "holding_confirmed": True,
                    "transport_authorized": attachment_valid,
                    **(
                        {
                            "held_object_to_tcp_attachment": (
                                verified_attachment
                            )
                        }
                        if verified_attachment is not None
                        else {}
                    ),
                    "updated_step": step,
                    "evidence": (
                        "after_action_verified_grasp_coupling"
                        if attachment_valid
                        else (
                            "after_action_verified_grasp_coupling;"
                            "transport_blocked_missing_attachment"
                        )
                    ),
                }

        for arm, lift_details in successful_lifts_by_arm.items():
            arm_state = updated.get(arm)
            if (
                not isinstance(arm_state, dict)
                or arm_state.get("holding_confirmed") is True
            ):
                continue
            diagnostic_evidence = (
                self._diagnostic_grasp_lift_evidence(
                    arm=arm,
                    arm_state=arm_state,
                    lift_details=lift_details,
                    action_effect=action_effect,
                )
            )
            if diagnostic_evidence is None:
                continue
            updated[arm] = {
                **arm_state,
                "holding_confirmed": False,
                "transport_authorized": False,
                "diagnostic_lift_evidence": diagnostic_evidence,
                "updated_step": step,
                "evidence": (
                    "bounded_diagnostic_lift_completed;"
                    "attachment_not_verified_transport_forbidden"
                ),
            }
            diagnostic_grasp_lifts.append(
                {
                    "arm": arm,
                    "held_instance_id": str(
                        arm_state.get("held_instance_id", "") or ""
                    ),
                    "grasp_candidate_id": str(
                        arm_state.get("grasp_candidate_id", "") or ""
                    ),
                    **diagnostic_evidence,
                }
            )

        if self._release_requires_placement_recovery(
            runtime_place_validation
        ):
            recovery_arm = str(
                runtime_place_validation.get("arm", "") or ""
            ).strip().lower()
            recovery_state = updated.get(recovery_arm)
            if (
                isinstance(recovery_state, dict)
                and recovery_state.get("phase")
                == "release_pending_verification"
            ):
                released_instance_id = str(
                    runtime_place_validation.get(
                        "held_instance_id",
                        recovery_state.get("held_instance_id", ""),
                    )
                    or ""
                ).strip()
                updated[recovery_arm] = {
                    **recovery_state,
                    "phase": "release_recovery_required",
                    "released_instance_id": released_instance_id,
                    "holding_confirmed": False,
                    "object_target_error_m": (
                        runtime_place_validation.get(
                            "object_target_error_m"
                        )
                    ),
                    "target_tolerance_m": (
                        runtime_place_validation.get(
                            "target_tolerance_m"
                        )
                    ),
                    "placement_failure_reason": (
                        "released_object_outside_target_tolerance"
                    ),
                    "updated_step": step,
                    "evidence": (
                        "release_detached_but_observed_outside_target;"
                        "bounded_regrasp_and_replacement_required"
                    ),
                }
                release_recovery_transition = {
                    "arm": recovery_arm,
                    "released_instance_id": released_instance_id,
                    "target_id": str(
                        runtime_place_validation.get("target_id", "")
                        or recovery_state.get("place_target_id", "")
                        or ""
                    ).strip(),
                    "object_target_error_m": (
                        runtime_place_validation.get(
                            "object_target_error_m"
                        )
                    ),
                    "target_tolerance_m": (
                        runtime_place_validation.get(
                            "target_tolerance_m"
                        )
                    ),
                    "source_skill_id": str(
                        recovery_state.get("source_skill_id", "") or ""
                    ).strip(),
                }
        elif runtime_place_validation.get("verified") is True:
            verified_arm = str(
                runtime_place_validation.get("arm", "") or ""
            ).strip().lower()
            verified_state = updated.get(verified_arm)
            if (
                isinstance(verified_state, dict)
                and verified_state.get("phase")
                == "release_pending_verification"
            ):
                updated.pop(verified_arm, None)

        scene_events = self._scene_events_from_recovery_effect(
            calls=calls,
            results=results,
            action_effect=action_effect,
            previous_manipulation_state=previous,
            updated_manipulation_state=updated,
        )
        if updated == previous:
            self._apply_scene_runtime_events(scene_events)
            if ambiguous_return_reduction is not None:
                self._trace(
                    "ambiguous_grasp_return_reduction",
                    physical_return_succeeded=(
                        ambiguous_return_reduction.physical_return_succeeded
                    ),
                    fresh_snapshot_confirmed=(
                        ambiguous_return_reduction.fresh_snapshot_confirmed
                    ),
                    attempt_completed=(
                        ambiguous_return_reduction.attempt_completed
                    ),
                    reason=ambiguous_return_reduction.reason,
                )
            return
        self.memory_store.state.working.manipulation_state = updated
        if scene_events:
            self._apply_scene_runtime_events(scene_events)
        else:
            self._sync_runtime_manipulation_state_to_scene_memory()
        if ambiguous_return_reduction is not None:
            reduction_payload = {
                "physical_return_succeeded": (
                    ambiguous_return_reduction.physical_return_succeeded
                ),
                "fresh_snapshot_confirmed": (
                    ambiguous_return_reduction.fresh_snapshot_confirmed
                ),
                "attempt_completed": (
                    ambiguous_return_reduction.attempt_completed
                ),
                "reason": ambiguous_return_reduction.reason,
            }
            self._trace(
                "ambiguous_grasp_return_reduction",
                **reduction_payload,
            )
            self._dump_rollout_event(
                "ambiguous_grasp_return_reduction",
                **reduction_payload,
            )
        for transition in diagnostic_grasp_lifts:
            self.memory_store.record_recovery(
                "diagnostic_grasp_lift_requires_verification:"
                f"arm={transition['arm']},"
                f"instance={transition['held_instance_id']},"
                f"lift_m={transition['observed_lift_m']}"
            )
            self._trace(
                "diagnostic_grasp_lift_requires_verification",
                **transition,
            )
            self._dump_rollout_event(
                "diagnostic_grasp_lift_requires_verification",
                **transition,
            )
        for transition in support_contact_release_transitions:
            self.memory_store.record_recovery(
                "support_contact_release_ready:"
                f"arm={transition['arm']},"
                f"instance={transition['held_instance_id']},"
                f"target={transition['target_id']}"
            )
            self._trace("support_contact_release_ready", **transition)
            self._dump_rollout_event(
                "support_contact_release_ready",
                **transition,
            )
        if release_recovery_transition:
            history_entry = (
                "runtime_release_recovery_required:"
                f"arm={release_recovery_transition['arm']},"
                "instance="
                f"{release_recovery_transition['released_instance_id']},"
                f"target={release_recovery_transition['target_id']},"
                "error_m="
                f"{release_recovery_transition['object_target_error_m']},"
                "tolerance_m="
                f"{release_recovery_transition['target_tolerance_m']}"
            )
            self.memory_store.record_recovery(history_entry)
            self._trace(
                "runtime_release_recovery_required",
                **release_recovery_transition,
            )
            self._dump_rollout_event(
                "runtime_release_recovery_required",
                **release_recovery_transition,
            )
        payload = {
            "previous": previous,
            "current": updated,
            "effect_verified": effect_verified,
            "effect_type": effect_type,
            "source_skill_id": self._active_skill_id(),
        }
        self._trace("manipulation_state_update", **payload)
        self._dump_rollout_event("manipulation_state_update", **payload)

    def _diagnostic_grasp_lift_evidence(
        self,
        *,
        arm: str,
        arm_state: dict[str, Any],
        lift_details: dict[str, Any],
        action_effect: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Record a bounded reverse-ingress probe without granting transport.

        A retained ground pose is not fresh negative evidence.  Therefore an
        exact reached-grasp hypothesis may remain under verification after a
        real bounded motion when the same track is occluded by the closing arm.
        End-effector motion alone is never evidence that the object moved with
        the TCP, so this record cannot create carry/place capability.
        """

        if (
            normalize_physical_arm(arm) not in {"left", "right"}
            or str(arm_state.get("phase", "") or "").strip().lower()
            != "grasp_candidate"
            or str(
                arm_state.get("operation_action_mode", "grasp") or "grasp"
            ).strip().lower()
            != "grasp"
            or not str(
                arm_state.get("grasp_candidate_id", "") or ""
            ).strip()
            or not isinstance(
                arm_state.get("held_object_to_tcp_attachment"),
                dict,
            )
        ):
            return None

        observed_displacement = self._xyz_prefix(
            lift_details.get("observed_displacement_xyz")
        )
        if observed_displacement is None:
            # Compatibility with historical lift_ee results that predate the
            # generic pose-target displacement fields.
            axis = str(
                lift_details.get("axis", "z") or "z"
            ).strip().lower()
            if axis not in {"x", "y", "z"}:
                return None
            observed_raw = lift_details.get(
                "observed_axis_displacement_m"
            )
            if observed_raw is None:
                per_arm = lift_details.get(
                    "observed_axis_displacements_m"
                )
                if isinstance(per_arm, dict):
                    observed_raw = per_arm.get(arm)
            try:
                observed_axis_displacement = float(observed_raw)
            except (TypeError, ValueError):
                return None
            observed_displacement = [0.0, 0.0, 0.0]
            observed_displacement[{"x": 0, "y": 1, "z": 2}[axis]] = (
                observed_axis_displacement
            )
        observed_motion = sum(
            value * value for value in observed_displacement
        ) ** 0.5

        start_pose = self._xyz_prefix(lift_details.get("start_pose"))
        target_pose = self._xyz_prefix(lift_details.get("target_pose"))
        if start_pose is not None and target_pose is not None:
            planned_displacement = [
                target_pose[index] - start_pose[index]
                for index in range(3)
            ]
        else:
            # Legacy lift requests carry a signed axis displacement rather
            # than a target pose.
            axis = str(
                lift_details.get("axis", "z") or "z"
            ).strip().lower()
            if axis not in {"x", "y", "z"}:
                return None
            try:
                signed_distance = float(
                    lift_details.get(
                        "signed_distance",
                        lift_details.get("distance", 0.0),
                    )
                )
            except (TypeError, ValueError):
                return None
            planned_displacement = [0.0, 0.0, 0.0]
            planned_displacement[{"x": 0, "y": 1, "z": 2}[axis]] = (
                signed_distance
            )
        planned_motion = sum(
            value * value for value in planned_displacement
        ) ** 0.5
        direction_cosine = (
            sum(
                planned_displacement[index]
                * observed_displacement[index]
                for index in range(3)
            )
            / (planned_motion * observed_motion)
            if planned_motion > 1e-8 and observed_motion > 1e-8
            else -1.0
        )
        if (
            not math.isfinite(observed_motion)
            or observed_motion < _DIAGNOSTIC_GRASP_MIN_LIFT_M
            or observed_motion > _DIAGNOSTIC_GRASP_MAX_MOTION_M
            or not math.isfinite(planned_motion)
            or planned_motion <= 0.0
            or direction_cosine
            < _DIAGNOSTIC_GRASP_MIN_DIRECTION_COSINE
            or self._recovery_gripper_state().get(arm) != "closed"
        ):
            return None
        postlift_ee_world = self._xyz_prefix(
            lift_details.get("observed_pose")
        )
        if postlift_ee_world is None:
            robot_state = self._get_robot_state()
            postlift_ee_world = self._xyz_prefix(
                (robot_state.get(arm) or {}).get("xyz")
                if isinstance(robot_state, dict)
                else None
            )
        preprocess = (
            self.memory_store.state.working.observation_preprocess
        )
        lift_reference = {
            # Keep historical field names for persisted-state and trace
            # compatibility; both now describe a direction-agnostic motion.
            "observed_lift_m": round(observed_motion, 6),
            "observed_motion_m": round(observed_motion, 6),
            "observed_displacement_xyz_m": observed_displacement,
            "minimum_lift_m": _DIAGNOSTIC_GRASP_MIN_LIFT_M,
            "minimum_motion_m": _DIAGNOSTIC_GRASP_MIN_LIFT_M,
            "planned_displacement_xyz_m": planned_displacement,
            "observed_planned_direction_cosine": round(
                direction_cosine,
                6,
            ),
            "lift_env_step": (
                int(self.latest_snapshot.step_count)
                if self.latest_snapshot is not None
                else -1
            ),
            "lift_observation_generation": int(
                (preprocess or {}).get("observation_generation", -1)
                if isinstance(preprocess, dict)
                else -1
            ),
            **(
                {
                    "postlift_ee_world_m": postlift_ee_world,
                    "prelift_ee_world_m": [
                        postlift_ee_world[index]
                        - observed_displacement[index]
                        for index in range(3)
                    ],
                }
                if postlift_ee_world is not None
                else {}
            ),
        }

        runtime_validation = matched_grasp_motion_validation(
            action_effect,
            arm=arm,
            arm_state=arm_state,
        )
        if (
            isinstance(runtime_validation, dict)
            and runtime_validation.get("verified") is False
        ):
            return {
                **lift_reference,
                "track_status": "observed_by_multiple_views",
                "track_stability": "stationary_during_lift",
                "fresh_visible_negative_evidence": True,
                "occlusion_source": "none",
                "verifier_effect": str(
                    action_effect.get("effect_verified", "unverified")
                    or "unverified"
                ).strip().lower(),
                "positive_attachment_evidence": False,
                "negative_attachment_evidence": True,
                "transport_authorized": False,
                "runtime_grasp_validation": runtime_validation,
            }
        if isinstance(runtime_validation, dict):
            return {
                **lift_reference,
                "track_status": "multiview_motion_ambiguous",
                "track_stability": "attachment_not_proven",
                "fresh_visible_negative_evidence": False,
                "occlusion_source": "none",
                "verifier_effect": str(
                    action_effect.get("effect_verified", "unverified")
                    or "unverified"
                ).strip().lower(),
                "positive_attachment_evidence": False,
                "negative_attachment_evidence": False,
                "transport_authorized": False,
                "runtime_grasp_validation": runtime_validation,
            }

        held_instance_id = str(
            arm_state.get("held_instance_id", "") or ""
        ).strip()
        held_instance = self._scene_instance_by_ref(
            self.memory_store.state.working.scene_memory,
            held_instance_id,
        )
        if not isinstance(held_instance, dict):
            return None
        status = str(
            held_instance.get("status", "") or ""
        ).strip().lower()
        stability = str(
            held_instance.get("stability", "") or ""
        ).strip().lower()
        if (
            status != "tracked"
            or stability
            not in {
                "missing_current_frame",
                "geometry_inconsistent_current_frame",
            }
        ):
            # A visible object provides direct positive/negative evidence and
            # must be handled by the normal verifier path.
            return None

        occlusion_source = ""
        occlusion_proof: dict[str, Any] = {}
        if arm_state.get("grasp_occlusion_geometry_lease") is True:
            occlusion_source = "consumed_exact_grasp_geometry_lease"
        else:
            observed_pose = lift_details.get("observed_pose")
            observed_xyz = self._xyz_prefix(observed_pose)
            if observed_xyz is None:
                robot_state = self._get_robot_state()
                arm_robot_state = (
                    (robot_state.get(arm) or {})
                    if isinstance(robot_state, dict)
                    else {}
                )
                observed_xyz = self._xyz_prefix(
                    arm_robot_state.get("xyz")
                )
            target_xyz = self._xyz_prefix(
                arm_state.get("grasp_ee_target_world_m")
            )
            proof = (
                self._camera_ray_occlusion_proof(
                    instance=held_instance,
                    arm=arm,
                    observed_xyz=observed_xyz,
                    target_xyz=target_xyz,
                )
                if observed_xyz is not None and target_xyz is not None
                else None
            )
            if proof is None:
                return None
            occlusion_source = "post_diagnostic_motion_camera_ray_proof"
            occlusion_proof = dict(proof)

        verifier_effect = str(
            action_effect.get("effect_verified", "unverified")
            or "unverified"
        ).strip().lower()
        return {
            **lift_reference,
            "track_status": status,
            "track_stability": stability,
            "fresh_visible_negative_evidence": (
                stability
                == "geometry_inconsistent_current_frame"
            ),
            "occlusion_source": occlusion_source,
            "verifier_effect": verifier_effect,
            "positive_attachment_evidence": False,
            "transport_authorized": False,
            **(
                {"runtime_grasp_validation": runtime_validation}
                if runtime_validation is not None
                else {}
            ),
            **occlusion_proof,
        }

    def _confirmed_holding_state(self, arm: str) -> dict[str, Any] | None:
        normalized_arm = normalize_physical_arm(arm)
        if normalized_arm not in {"left", "right"}:
            return None
        manipulation = self.memory_store.state.working.manipulation_state
        if isinstance(manipulation, dict):
            arm_state = manipulation.get(normalized_arm)
            if (
                isinstance(arm_state, dict)
                and bool(arm_state.get("holding_confirmed"))
                and str(arm_state.get("held_instance_id", "") or "").strip()
            ):
                return dict(arm_state)
        scene_memory = self.memory_store.state.working.scene_memory or {}
        scene_manipulation = (
            scene_memory.get("manipulation_state")
            if isinstance(scene_memory, dict)
            else None
        )
        if isinstance(scene_manipulation, dict):
            arm_state = scene_manipulation.get(normalized_arm)
            if (
                isinstance(arm_state, dict)
                and bool(arm_state.get("holding_confirmed"))
                and str(arm_state.get("held_instance_id", "") or "").strip()
            ):
                return dict(arm_state)
        return None

    def _transport_holding_state(
        self,
        arm: str,
    ) -> dict[str, Any] | None:
        """Return a hold authorized by the active carry/place policy.

        Strict mode accepts only a verifier-confirmed attachment. Evidence-only
        mode may also accept an exact runtime-authored provisional attachment;
        switching back to strict immediately revokes that provisional authority.
        """

        normalized_arm = normalize_physical_arm(arm)
        if normalized_arm not in {"left", "right"}:
            return None
        for manipulation in (
            self.memory_store.state.working.manipulation_state,
            (
                self.memory_store.state.working.scene_memory or {}
            ).get("manipulation_state"),
        ):
            if not isinstance(manipulation, dict):
                continue
            arm_state = manipulation.get(normalized_arm)
            if manipulation_state_allows_transport(
                arm_state,
                active_grasp_transport_policy=(
                    self._grasp_transport_policy()
                ),
            ):
                return dict(arm_state)
        return None

    def _recovery_holding_confirmed(self, arm: str) -> bool:
        if self._confirmed_holding_state(arm) is not None:
            return True
        for item in self.memory_store.state.working.recovery_history[-8:]:
            text = str(item).lower()
            if f"{arm}" in text and "holding_confirmed" in text:
                return True
        return False

    def _recovery_holding_authorized(self, arm: str) -> bool:
        return self._transport_holding_state(arm) is not None

    def _build_vla_execution_request(self, snapshot: EnvSnapshot, subtask_instruction: str):
        payload = self._vla_instruction_builder.build(
            global_task=self.current_instruction,
            current_subtask=subtask_instruction,
            committed_memory=self.memory_store.export_executor_memory(),
            retry_count=self.memory_store.state.recovery.retry_used,
            recovered=bool(self.memory_store.state.recovery.last_action),
            recovery_reason=self.memory_store.state.recovery.pending_reason or self.memory_store.state.working.last_error,
            stage=self.memory_store.state.monitor.phase,
            extra={
                "active_skill_name": self.memory_store.state.active_skill.skill_name if self.memory_store.state.active_skill is not None else "",
                "monitor_status": self.memory_store.state.monitor.status,
            },
        )
        instruction_text = self._vla_instruction_builder.render_text(payload)
        return build_vla_execution_request(
            instruction_text=instruction_text,
            instruction_payload={
                "global_task": payload.global_task,
                "current_subtask": payload.current_subtask,
                "committed_memory": payload.committed_memory,
                "execution_context": {
                    "retry_count": payload.execution_context.retry_count,
                    "recovered": payload.execution_context.recovered,
                    "recovery_reason": payload.execution_context.recovery_reason,
                    "stage": payload.execution_context.stage,
                    "extra": dict(payload.execution_context.extra),
                },
            },
            observation=snapshot.raw,
            current_subtask=subtask_instruction,
            global_task=self.current_instruction,
            committed_memory=self.memory_store.export_executor_memory(),
        )

    def _make_monitor_signal(self, *, name: str, level: str = "info", reason: str = "", score: float = 0.0, details: dict[str, Any] | None = None) -> MonitorSignal:
        return make_signal(name, level=level, reason=reason, score=score, details=details)

    def _decide_handoff(self, *, signal: MonitorSignal, active_subtask: str) -> str:
        decision = self._handoff_policy.decide(
            signals=[signal],
            recovery_pending=bool(self.memory_store.state.recovery.pending_action),
            active_subtask=active_subtask,
        )
        return decision.target

    def update_monitor_after_rollout_step(self, task_env: Any, snapshot: EnvSnapshot) -> bool:
        active = self.memory_store.state.active_skill
        success_signal = self._progress_monitor.detect_success(snapshot)
        progress = RMBenchEnvAdapter.measure_progress(self.previous_snapshot, snapshot)
        if bool(progress.get("made_progress")):
            self.memory_store.reset_stall_count()
        else:
            self.memory_store.increment_stall_count()
        self.memory_store.increment_step_counters()
        progress["success"] = success_signal is not None
        progress["task_success"] = bool(success_signal is not None and success_signal.name == TASK_SUCCESS)
        if active is None:
            inactive_signal = self._make_monitor_signal(name=INVALID_ACTION_PATTERN, level="warning", reason="rollout stopped without active skill", score=float(progress.get("progress_score", 0.0)))
            self.memory_store.complete_rollout(env_signal=inactive_signal.name, note=inactive_signal.reason, progress_score=inactive_signal.score)
            self._trace("monitor_signal", signal=inactive_signal.name, progress_score=inactive_signal.score, reason=inactive_signal.reason)
            return True
        ood_signals = self._ood_detector.detect(
            snapshot=snapshot,
            action_chunk=None,
            current_subtask=active.instruction,
            observation_summary=self._get_env_summary(),
            monitor_status=self.memory_store.state.monitor.status,
            recovery_state=self.memory_store.state.recovery.to_dict(),
            execution_context={
                "progress_score": float(progress.get("progress_score", 0.0)),
                "step_count": getattr(snapshot, "step_count", 0),
                "step_limit": getattr(snapshot, "step_limit", 0),
                "task_success": progress["task_success"],
            },
        )
        if ood_signals:
            ood_signal = ood_signals[0]
            self.memory_store.fail_rollout(status="rollout_stalled", reason=ood_signal.reason, env_signal=ood_signal.name, progress_score=ood_signal.score)
            recovery_signal = self._make_monitor_signal(
                name=NEEDS_RECOVERY_TOOLS,
                level="warning",
                reason=ood_signal.reason,
                score=ood_signal.score,
                details={"ood_signal": ood_signal.name},
            )
            if "semantic_tags" in ood_signal.details:
                recovery_signal.details["semantic_tags"] = ood_signal.details["semantic_tags"]
            target = self._decide_handoff(signal=recovery_signal, active_subtask=active.instruction)
            self._trace(
                "monitor_signal",
                signal=ood_signal.name,
                progress_score=ood_signal.score,
                reason=ood_signal.reason,
                handoff_target=target,
                semantic_tags=ood_signal.details.get("semantic_tags", {}),
                observation_summary=self._get_env_summary(),
                robot_state=self._get_robot_state(),
            )
            if target == "vla":
                self.memory_store.set_monitor_status(
                    phase="monitoring",
                    status="rollout_active",
                    note="OOD detected but handoff kept on VLA",
                    env_signal=RUNNING,
                    progress_score=float(progress.get("progress_score", 0.0)),
                    failure_reason="",
                )
                return False
            preferred_action = self._dispatch_recovery_tools(task_env, ood_signal)
            self.memory_store.set_recovery_policy(action=self._select_recovery_action(preferred_action=preferred_action, signal_name=ood_signal.name), reason=ood_signal.reason)
            return True
        assessment = self._progress_monitor.assess_step(
            previous_snapshot=self.previous_snapshot,
            current_snapshot=progress,
            steps_used=active.steps_used,
            max_steps=min(active.max_steps, self.config.max_steps_per_skill),
            stall_count=self.memory_store.state.working.stall_count,
            stall_patience=self.config.stall_patience,
        )
        signal = assessment.signal
        target = self._decide_handoff(signal=signal if signal.name != STALL_DETECTED else self._make_monitor_signal(name=NEEDS_RECOVERY_TOOLS, level="warning", reason=signal.reason, score=signal.score), active_subtask=active.instruction)
        self._trace(
            "monitor_signal",
            signal=signal.name,
            progress_score=assessment.progress_score,
            reason=signal.reason,
            handoff_target=target,
            observation_summary=self._get_env_summary(),
            robot_state=self._get_robot_state(),
        )
        if signal.name == TASK_SUCCESS:
            self.memory_store.complete_rollout(env_signal=signal.name, note=signal.reason, progress_score=signal.score)
            self.memory_store.mark_active_skill_succeeded(note=signal.reason)
            self.memory_store.mark_task_finished(note="task completed")
            return True
        if signal.name == "subtask_success":
            self.memory_store.complete_rollout(env_signal=signal.name, note=signal.reason, progress_score=signal.score)
            self.memory_store.mark_active_skill_succeeded(note=signal.reason)
            if self._pure_tool_control_enabled() and not self._latest_snapshot_task_success():
                self.memory_store.record_recovery("pure_tool_control_subtask_success_continue")
                self._debug_recovery_planner_bootstrapped = False
                self._debug_recovery_rounds = 0
                self._debug_recovery_scene_wait_turns = 0
                self.memory_store.set_monitor_status(
                    phase="reasoning",
                    status="needs_reasoning",
                    note="subtask success observed; global task still not complete",
                    env_signal=RUNNING,
                    failure_reason="",
                    progress_score=signal.score,
                )
                return True
            self.memory_store.mark_task_finished(note="task completed")
            return True
        if signal.name == STEP_BUDGET_EXHAUSTED:
            self.memory_store.fail_rollout(status="rollout_failed", reason=signal.reason, env_signal=signal.name, progress_score=signal.score)
            if target == "vla":
                self.memory_store.set_monitor_status(phase="monitoring", status="rollout_active", note="handoff kept on VLA", env_signal=RUNNING, progress_score=assessment.progress_score, failure_reason="")
                return False
            preferred_action = ""
            if target == "recovery_tools":
                preferred_action = self._dispatch_recovery_tools(task_env, signal)
            self.memory_store.set_recovery_policy(action=self._select_recovery_action(preferred_action=preferred_action, signal_name=signal.name), reason=signal.reason)
            return True
        if signal.name == STALL_DETECTED:
            self.memory_store.fail_rollout(status="rollout_stalled", reason=signal.reason, env_signal=signal.name, progress_score=signal.score)
            if target == "vla":
                self.memory_store.set_monitor_status(phase="monitoring", status="rollout_active", note="handoff kept on VLA", env_signal=RUNNING, progress_score=assessment.progress_score, failure_reason="")
                return False
            preferred_action = ""
            if target == "recovery_tools":
                preferred_action = self._dispatch_recovery_tools(task_env, signal)
            self.memory_store.set_recovery_policy(action=self._select_recovery_action(preferred_action=preferred_action, signal_name=signal.name), reason=signal.reason)
            return True
        self.memory_store.set_monitor_status(phase="monitoring", status="rollout_active", note=signal.reason, env_signal=signal.name, progress_score=assessment.progress_score, failure_reason="")
        return False

    def execute_monitored_rollout(self, task_env: Any, snapshot: EnvSnapshot) -> None:
        active = self.memory_store.state.active_skill
        if active is None:
            return
        import numpy as np
        self.memory_store.start_rollout()
        vla_request = self._build_vla_execution_request(snapshot, active.instruction)
        self._trace("vla_request", subtask=active.instruction, robot_state=RMBenchEnvAdapter.robot_state(snapshot))
        started = time.time()
        actions = self.executor_runtime.predict_action_chunk(
            observation=vla_request.observation,
            task="",
            subtask=vla_request.instruction_text,
            memory="",
        )
        latency_sec = time.time() - started
        actions = np.asarray(actions, dtype=np.float32)
        self._trace("vla_response", latency_sec=round(latency_sec, 4), action_chunk_shape=list(actions.shape))
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        total_actions = int(actions.shape[0])
        for action_index, action in enumerate(actions):
            if self._env_at_step_limit(task_env):
                self._trace(
                    "rollout_chunk_stop",
                    reason="environment_step_limit",
                    actions_remaining=max(0, total_actions - action_index),
                )
                break
            task_env.take_action(action, action_type="qpos")
            new_observation = task_env.get_obs()
            new_snapshot = RMBenchEnvAdapter.from_env(
                task_env,
                new_observation,
                oracle_objects_enabled=bool(
                    getattr(self.config, "oracle_objects_enabled", False)
                ),
            )
            self.update_snapshot(new_snapshot)
            if self.update_monitor_after_rollout_step(task_env, new_snapshot):
                break
            if self._env_at_step_limit(task_env):
                self._trace("rollout_chunk_stop", reason="environment_step_limit", actions_remaining=max(0, total_actions - action_index - 1))
                break

    def _env_at_step_limit(self, task_env: Any) -> bool:
        try:
            step_limit = int(getattr(task_env, "step_lim", 0) or 0)
            step_count = int(getattr(task_env, "take_action_cnt", 0) or 0)
        except Exception:
            return False
        return step_limit > 0 and step_count >= step_limit

    def _retain_observation_after_no_environment_action(
        self,
        *,
        task_env: Any,
        snapshot: EnvSnapshot,
        reason: str,
    ) -> None:
        if (
            self._pending_recovery_observation is not None
            or self._pure_tool_control_terminal_failure
            or self._latest_snapshot_task_success()
        ):
            return
        try:
            current_step = int(getattr(task_env, "take_action_cnt", -1))
        except (TypeError, ValueError):
            return
        if current_step != int(snapshot.step_count):
            return
        self._pending_recovery_observation = {
            "step_count": int(snapshot.step_count),
            "raw": dict(snapshot.raw),
        }
        self._trace(
            "recovery_observation_handoff_retained",
            cached_step=int(snapshot.step_count),
            reason=reason,
        )

    def run_step(self, task_env: Any, observation: dict[str, Any]) -> None:
        snapshot = RMBenchEnvAdapter.from_env(
            task_env,
            observation,
            oracle_objects_enabled=bool(
                getattr(self.config, "oracle_objects_enabled", False)
            ),
        )
        if not self.current_instruction:
            self.set_instruction(snapshot.instruction or task_env.get_instruction())
        self.update_snapshot(snapshot)
        if self._commit_pure_tool_control_environment_success(source="run_step"):
            return
        self._repair_unvalidated_pure_tool_control_finish()
        bootstrap_attempted = self._maybe_bootstrap_debug_recovery_with_planner()
        if self._pure_tool_control_enabled() and bootstrap_attempted:
            self._retain_observation_after_no_environment_action(
                task_env=task_env,
                snapshot=snapshot,
                reason="planner_bootstrap_completed_without_environment_action",
            )
            return
        if self._maybe_force_debug_recovery(task_env):
            self._retain_observation_after_no_environment_action(
                task_env=task_env,
                snapshot=snapshot,
                reason="forced_recovery_returned_without_environment_action",
            )
            return
        self._trace(
            "run_step",
            trigger_phase="pre_decision",
            active_skill=None if self.memory_store.state.active_skill is None else self.memory_store.state.active_skill.skill_name,
            monitor_status=self.memory_store.state.monitor.status,
            recovery_pending=self.memory_store.state.recovery.pending_action,
        )
        if self.memory_store.state.task.task_finished:
            return
        if self.memory_store.state.recovery.pending_action:
            self._apply_recovery_policy()
        need_decision, signal = self.needs_control_turn()
        self._trace("decision_check", need_decision=need_decision, trigger=signal.value)
        if need_decision and not self.memory_store.state.task.task_finished:
            if self._stop_for_pure_tool_control_turn_budget():
                return
            try:
                prediction = self.run_control_turn(signal)
            except Exception as exc:
                if self._defer_pure_tool_control_control_error(stage="control_turn", signal=signal, error=exc):
                    return
                raise
            self.apply_control_decision(prediction)
        active = self.memory_store.state.active_skill
        if active is None or self.memory_store.state.task.task_finished:
            return
        if self._skip_vla_rollout_for_forced_recovery():
            return
        self.execute_monitored_rollout(task_env, snapshot)

    def current_status(self) -> dict[str, Any]:
        state = self.memory_store.state
        active = state.active_skill
        pure_tool_control = self._pure_tool_control_enabled()
        task_finish_validated = bool(
            state.task.task_finished
            and (not pure_tool_control or self._latest_snapshot_task_success())
        )
        return {
            "active_skill": None if state.active_skill is None else state.active_skill.skill_name,
            "preferred_arm": "either" if active is None else active.preferred_arm,
            "monitor_phase": state.monitor.phase,
            "monitor_status": state.monitor.status,
            "recovery_pending": state.recovery.pending_action,
            "task_finished": state.task.task_finished,
            "pure_tool_control": pure_tool_control,
            "task_finish_validated": task_finish_validated,
            "environment_success": self._latest_snapshot_task_success(),
            "terminal_failure": bool(self._pure_tool_control_terminal_failure),
            "terminal_failure_reason": self._pure_tool_control_terminal_failure,
            "control_turn_count": self._pure_tool_control_control_turns,
            "max_control_turns": self._pure_tool_control_max_control_turns(),
            "no_progress_control_turn_count": self._pure_tool_control_no_progress_control_turns,
            "max_no_progress_control_turns": self._pure_tool_control_max_no_progress_control_turns(),
            "semantic_round_index": self._debug_recovery_rounds,
            "max_semantic_rounds_per_active_subtask": self._forced_recovery_max_rounds(),
            "perception_condition": (
                "oracle" if bool(getattr(self.config, "oracle_objects_enabled", False)) else "no_oracle"
            ),
            "oracle_objects_enabled": bool(getattr(self.config, "oracle_objects_enabled", False)),
            "control_backend_error_count": self._pure_tool_control_control_backend_errors,
            "recovery_backend_error_count": self._pure_tool_control_recovery_backend_errors,
            "recovery_backend_error_stage": self._last_recovery_backend_error_stage,
            "action_effect_verification_pending": self._pending_action_effect_verification is not None,
            "reobserve_scene_enabled": bool(
                self._recovery_dispatcher.reobserve_scene_enabled
            ),
            "grasp_transport_policy": self._grasp_transport_policy(),
            "release_guard_enabled": self._release_guard_enabled(),
            "action_geometry_repair_pending_policy": (
                self._action_geometry_repair_pending_policy()
            ),
            "last_evidence_acquisition_decision": dict(
                self._last_evidence_acquisition_decision
            ),
            "backend_error_budget": self._pure_tool_control_backend_error_budget(),
            "empty_plan_count": self._pure_tool_control_empty_plan_turns,
            "empty_plan_replan_threshold": self._pure_tool_control_empty_plan_replan_threshold(),
            "blocked_grounded_setups": self._blocked_grounded_setup_payload(),
        }
