from __future__ import annotations

import math
import re
from typing import Any

from ..operation_candidates import (
    operation_pose_candidates as validated_operation_pose_candidates,
)
from .query_normalization import (
    normalize_entity_scope,
    normalize_perception_object_id,
)
from .position_identity_contract import (
    OPERATION_GEOMETRY_ROTATION_CHANGE_RAD,
    OPERATION_GEOMETRY_TRANSLATION_CHANGE_M,
    POSITION_ONLY_ACTION_GEOMETRY_QUARANTINE_LEASE_KEY,
    VERIFIED_OPERATION_TARGET_ANCHOR_KEY,
    eligible_anchor_matches,
    executable_candidate_sets_consistent,
    operation_target_anchor_from_event,
    verified_position_anchors,
)
from ..relational_place_targets import (
    REFERENCE_REGIONS_KEY,
    ReferenceRegionTracker,
)


DEFAULT_GRASP_CLEARANCE_M = 0.02
DEFAULT_TEMPORAL_WINDOW_SIZE = 8
DEFAULT_TEMPORAL_ACTION_GEOMETRY_DISTANCE_M = 0.04
RELOCATION_ACTION_GEOMETRY_CONFIRMATIONS = 2
STRONG_GEOMETRY_DISTANCE_M = 0.035
STRONG_GEOMETRY_IOU = 0.6
CENTROID_MATCH_DISTANCE_PX = 60.0
BOUND_REFINEMENT_MIN_CONTAINMENT = 0.85
BOUND_REFINEMENT_MAX_AREA_RATIO = 0.65
BOUND_REFINEMENT_MAX_EXTENT_RATIO = 0.75
MIN_SCENE_CANDIDATE_SCORE = 0.12
MAX_MANIPULATION_OBJECT_Z_EXTENT_M = 0.18
MAX_MANIPULATION_OBJECT_XY_EXTENT_M = 0.30
MAX_MANIPULATION_OBJECT_WORLD_Z_M = 0.95
POSITION_CURRENT_VERIFIED = "current_verified"
POSITION_MEMORY_VALID = "memory_valid"
POSITION_MOTION_UNCERTAIN = "motion_uncertain"
VALID_POSITION_STATES = {
    POSITION_CURRENT_VERIFIED,
    POSITION_MEMORY_VALID,
    POSITION_MOTION_UNCERTAIN,
}
MULTIVIEW_WORLD_MERGE_DISTANCE_M = 0.02
BOUND_IDENTITY_VARIANT_TOP_DISTANCE_M = 0.02
BOUND_IDENTITY_VARIANT_MIN_XY_CONTAINMENT = 0.80


class SceneMemoryTracker:
    def __init__(
        self,
        *,
        max_missing_steps: int = 20,
        stable_distance_m: float = 0.08,
        temporal_action_geometry_distance_m: float = DEFAULT_TEMPORAL_ACTION_GEOMETRY_DISTANCE_M,
        temporal_window_size: int = DEFAULT_TEMPORAL_WINDOW_SIZE,
        min_candidate_score: float = MIN_SCENE_CANDIDATE_SCORE,
        max_object_z_extent_m: float = MAX_MANIPULATION_OBJECT_Z_EXTENT_M,
        max_object_xy_extent_m: float = MAX_MANIPULATION_OBJECT_XY_EXTENT_M,
        max_object_world_z_m: float = MAX_MANIPULATION_OBJECT_WORLD_Z_M,
        robot_self_filter_radius_m: float = 0.08,
        robot_self_filter_z_margin_m: float = 0.08,
        enable_executable_candidate_temporal_consistency: bool = True,
    ) -> None:
        self.max_missing_steps = max(1, int(max_missing_steps))
        self.stable_distance_m = max(0.0, float(stable_distance_m))
        self.temporal_action_geometry_distance_m = max(0.0, float(temporal_action_geometry_distance_m))
        self._temporal_tracker = TemporalInstanceTracker(
            window_size=temporal_window_size,
            max_missing_steps=self.max_missing_steps,
            stable_distance_m=self.stable_distance_m,
            temporal_action_geometry_distance_m=self.temporal_action_geometry_distance_m,
            min_candidate_score=min_candidate_score,
            max_object_z_extent_m=max_object_z_extent_m,
            max_object_xy_extent_m=max_object_xy_extent_m,
            max_object_world_z_m=max_object_world_z_m,
            robot_self_filter_radius_m=robot_self_filter_radius_m,
            robot_self_filter_z_margin_m=robot_self_filter_z_margin_m,
            enable_executable_candidate_temporal_consistency=(
                enable_executable_candidate_temporal_consistency
            ),
        )
        self._reference_region_tracker = ReferenceRegionTracker(
            max_missing_steps=self.max_missing_steps,
        )

    def reset(self) -> None:
        self._temporal_tracker.reset()
        self._reference_region_tracker.reset()

    def update(
        self,
        *,
        segmentation: list[dict[str, Any]],
        env_step: int,
        global_task: str,
        current_subtask: str,
        identity_relocation_leases: list[dict[str, Any]] | None = None,
        position_events: list[dict[str, Any]] | None = None,
        observation_capture_id: int | None = None,
    ) -> dict[str, Any]:
        # A reference-set query describes a spatial relation, not a physical
        # object that may be grasped.  Its aggregate/member masks are consumed
        # by ReferenceRegionTracker below and must not create ordinary scene
        # tracks (or later appear as grasp/occupancy candidates).
        object_segmentation = [
            item
            for item in segmentation
            if not (
                isinstance(item, dict)
                and normalize_entity_scope(item.get("entity_scope"))
                == "reference_set"
            )
        ]
        candidates = extract_scene_candidates(
            object_segmentation,
            multiview_action_geometry_distance_m=(
                self.temporal_action_geometry_distance_m
            ),
        )
        temporal_memory = self._temporal_tracker.update(
            candidates=candidates,
            env_step=env_step,
            identity_relocation_leases=identity_relocation_leases,
            position_events=position_events,
            observation_capture_id=observation_capture_id,
        )
        merged = self._instances_from_tracks(temporal_memory.get("tracks", []), env_step=env_step)
        reference_regions = self._reference_region_tracker.update(
            segmentation=segmentation,
            instances=merged,
            env_step=env_step,
            current_subtask=current_subtask,
            position_events=position_events,
        )
        task_focus = build_task_focus(
            instances=merged,
            global_task=global_task,
            current_subtask=current_subtask,
            perception_queries=segmentation,
        )
        uncertainty = build_uncertainty(merged)
        return {
            "env_step": int(env_step),
            "instances": merged,
            "task_focus": task_focus,
            "uncertainty": uncertainty,
            "summary": summarize_scene_memory(merged, task_focus, uncertainty),
            "temporal_memory": temporal_memory,
            REFERENCE_REGIONS_KEY: reference_regions,
        }

    def apply_runtime_events(
        self,
        scene_memory: dict[str, Any],
        *,
        events: list[dict[str, Any]],
        env_step: int,
    ) -> dict[str, Any]:
        """Apply physical-state facts without fabricating a visual observation.

        Runtime manipulation results arrive after the post-action camera frame
        has already been processed.  Applying them directly to the temporal
        tracker keeps the authoritative position state synchronized for the
        very next guarded action while preserving the distinction between a
        physical event and a camera observation.
        """

        if not events:
            return dict(scene_memory or {})
        tracks = self._temporal_tracker.apply_runtime_events(
            events,
            env_step=env_step,
        )
        rebound = dict(scene_memory or {})
        instances = self._instances_from_tracks(
            tracks,
            env_step=env_step,
        )
        temporal = dict(rebound.get("temporal_memory") or {})
        temporal["tracks"] = tracks
        temporal["num_active_tracks"] = len(tracks)
        rebound["env_step"] = int(env_step)
        rebound["instances"] = instances
        self._reference_region_tracker.apply_runtime_events(
            events,
            env_step=env_step,
        )
        rebound[REFERENCE_REGIONS_KEY] = self._reference_region_tracker.update(
            segmentation=[],
            instances=instances,
            env_step=env_step,
            current_subtask=str(
                (rebound.get("task_focus") or {}).get(
                    "current_subtask",
                    "",
                )
            ),
        )
        rebound["temporal_memory"] = temporal
        uncertainty = build_uncertainty(instances)
        rebound["uncertainty"] = uncertainty
        focus = (
            dict(rebound.get("task_focus") or {})
            if isinstance(rebound.get("task_focus"), dict)
            else {}
        )
        rebound["summary"] = summarize_scene_memory(
            instances,
            focus,
            uncertainty,
        )
        return rebound

    def rebind_task_focus(
        self,
        scene_memory: dict[str, Any],
        *,
        perception_queries: list[dict[str, Any]],
        global_task: str,
        current_subtask: str,
        identity_binding_required: bool = False,
        allow_context_binding: bool = False,
        previous_scene_memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Recompute focus after the Agent API selects exact stable identities."""
        rebound = dict(scene_memory)
        instances = [item for item in scene_memory.get("instances", []) if isinstance(item, dict)]
        binding_roles = {"target", "tool"}
        if allow_context_binding:
            binding_roles.add("context")
        task_focus = build_task_focus(
            instances=instances,
            global_task=global_task,
            current_subtask=current_subtask,
            perception_queries=perception_queries,
            binding_roles=binding_roles,
        )
        if identity_binding_required:
            task_focus["identity_binding_required"] = True
            has_bound_role_query = any(
                normalize_role(query.get("query_role", query.get("role"))) in binding_roles
                and query_requires_instance_binding(query)
                for query in perception_queries
            )
            temporal_focus = self._temporally_retained_task_focus(
                instances=instances,
                current_subtask=current_subtask,
                perception_queries=perception_queries,
                previous_scene_memory=previous_scene_memory,
            )
            if temporal_focus is not None and (not has_bound_role_query or task_focus.get("identity_binding_errors")):
                task_focus = temporal_focus
            elif not has_bound_role_query:
                task_focus["identity_binding_errors"] = [
                    (
                        "agent_returned_no_target_tool_or_context_binding"
                        if allow_context_binding
                        else "agent_returned_no_target_or_tool_binding"
                    )
                ]
            if not task_focus.get("identity_binding_roles"):
                task_focus["identity_binding_roles"] = [
                    role
                    for role, focus_key in (
                        ("target", "target_instances"),
                        ("tool", "tool_instances"),
                        ("context", "context_instances"),
                    )
                    if task_focus.get(focus_key)
                ]
        uncertainty = list(scene_memory.get("uncertainty", []) or [])
        rebound["task_focus"] = task_focus
        rebound["summary"] = summarize_scene_memory(instances, task_focus, uncertainty)
        return rebound

    def _temporally_retained_task_focus(
        self,
        *,
        instances: list[dict[str, Any]],
        current_subtask: str,
        perception_queries: list[dict[str, Any]],
        previous_scene_memory: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(previous_scene_memory, dict):
            return None
        previous_focus = previous_scene_memory.get("task_focus")
        if not isinstance(previous_focus, dict) or not bool(previous_focus.get("identity_binding_required")):
            return None
        if previous_focus.get("identity_binding_errors"):
            return None
        normalized_subtask = str(current_subtask or "").strip()
        if normalized_subtask != str(previous_focus.get("current_subtask", "") or "").strip():
            return None

        previous_instances = {
            str(item.get("instance_id", "") or "").strip(): item
            for item in previous_scene_memory.get("instances", []) or []
            if isinstance(item, dict) and str(item.get("instance_id", "") or "").strip()
        }
        current_by_track = {
            str(item.get("track_id", "") or "").strip(): item
            for item in instances
            if str(item.get("track_id", "") or "").strip()
        }
        retained_by_role: dict[str, list[dict[str, Any]]] = {"target": [], "tool": []}
        previous_identity_refs: set[str] = set()
        has_occluded_identity = False
        for role, focus_key in (("target", "target_instances"), ("tool", "tool_instances")):
            for previous_instance_id in previous_focus.get(focus_key, []) or []:
                previous_instance = previous_instances.get(str(previous_instance_id))
                if not isinstance(previous_instance, dict):
                    return None
                track_id = str(previous_instance.get("track_id", "") or "").strip()
                current_instance = current_by_track.get(track_id)
                if not track_id or not isinstance(current_instance, dict):
                    return None
                try:
                    missing_steps = int(current_instance.get("missing_steps", 0) or 0)
                except (TypeError, ValueError):
                    return None
                if (
                    (
                        missing_steps > self.max_missing_steps
                        and current_instance.get(
                            "position_state"
                        )
                        != POSITION_MEMORY_VALID
                    )
                    or not is_actionable_instance(current_instance)
                    or not instance_has_finite_grounding(current_instance)
                ):
                    return None
                status = str(current_instance.get("status", "") or "").strip().lower()
                stability = str(current_instance.get("stability", "") or "").strip().lower()
                if status == "tracked" or stability == "missing_current_frame":
                    has_occluded_identity = True
                retained_by_role[role].append(current_instance)
                previous_identity_refs.update(instance_identity_refs(previous_instance))
                previous_identity_refs.update(instance_identity_refs(current_instance))
        if not has_occluded_identity or not any(retained_by_role.values()):
            return None

        requested_refs = {
            ref
            for query in perception_queries
            if normalize_role(query.get("query_role", query.get("role"))) in {"target", "tool"}
            for ref in (query_instance_ref(query), query_oracle_id(query))
            if ref
        }
        if requested_refs and not requested_refs.issubset(previous_identity_refs):
            return None

        target_ids = [str(item.get("instance_id", "")) for item in retained_by_role["target"]]
        tool_ids = [str(item.get("instance_id", "")) for item in retained_by_role["tool"]]
        focused_tracks = [
            str(item.get("track_id", ""))
            for role in ("target", "tool")
            for item in retained_by_role[role]
        ]
        return {
            "current_subtask": normalized_subtask,
            "target_instances": target_ids,
            "tool_instances": tool_ids,
            "reason_summary": (
                "retained Agent-committed stable identity through bounded current-frame occlusion; "
                f"tracks={','.join(focused_tracks)}"
            ),
            "source": "temporal_identity_persistence",
            "identity_binding_required": True,
            "identity_binding_errors": [],
            "temporal_binding_retained": True,
            "temporarily_occluded_tracks": focused_tracks,
        }

    def _instances_from_tracks(self, tracks: list[dict[str, Any]], *, env_step: int) -> list[dict[str, Any]]:
        instances: list[dict[str, Any]] = []
        for track in sorted(tracks, key=lambda item: str(item.get("track_id", ""))):
            class_name = normalize_class(track.get("class", "object"))
            if not class_name:
                continue
            track_id = str(track.get("track_id", "") or "").strip()
            if not track_id:
                continue
            world_m = track.get("stable_world_m", track.get("world_m"))
            top_surface_world_m = track.get("stable_top_surface_world_m", track.get("top_surface_world_m"))
            approach_world_m = track.get("stable_approach_world_m", track.get("approach_world_m"))
            grasp_world_m = track.get("stable_grasp_world_m", track.get("grasp_world_m"))
            contact_world_m = track.get("stable_contact_world_m", track.get("contact_world_m"))
            instance = {
                    # The runtime identity is the stable tracker identity.  Do
                    # not derive semantic spatial labels from geometric order.
                    "instance_id": track_id,
                    "track_id": track_id,
                    "oracle_id": track.get("oracle_id"),
                    "oracle_source_path": track.get("oracle_source_path"),
                    "class": class_name,
                    "class_aliases": track.get("class_aliases", []),
                    "status": track.get("status", "tracked"),
                    "stability": track.get("stability", "tracked"),
                    "score": round_float(track.get("score")),
                    "position_state": track.get(
                        "position_state",
                        POSITION_MOTION_UNCERTAIN,
                    ),
                    "position_source": track.get("position_source"),
                    "position_state_reason": track.get(
                        "position_state_reason"
                    ),
                    "position_state_updated_step": track.get(
                        "position_state_updated_step"
                    ),
                    "last_verified_world_m": round_list(
                        track.get("last_verified_world_m")
                    ),
                    "last_verified_score": round_float(
                        track.get("last_verified_score")
                    ),
                    "last_verified_step": track.get(
                        "last_verified_step"
                    ),
                    "last_verified_source": (
                        dict(track.get("last_verified_source"))
                        if isinstance(
                            track.get("last_verified_source"),
                            dict,
                        )
                        else track.get("last_verified_source")
                    ),
                    "position_tolerance_m": round_float(
                        track.get("position_tolerance_m")
                    ),
                    VERIFIED_OPERATION_TARGET_ANCHOR_KEY: (
                        dict(
                            track.get(
                                VERIFIED_OPERATION_TARGET_ANCHOR_KEY
                            )
                        )
                        if isinstance(
                            track.get(
                                VERIFIED_OPERATION_TARGET_ANCHOR_KEY
                            ),
                            dict,
                        )
                        else None
                    ),
                    "last_motion_event": (
                        dict(track.get("last_motion_event"))
                        if isinstance(
                            track.get("last_motion_event"),
                            dict,
                        )
                        else None
                    ),
                    "verified_roles": [
                        dict(item)
                        for item in track.get("verified_roles", []) or []
                        if isinstance(item, dict)
                    ],
                    "verified_identity_tokens": list(
                        track.get("verified_identity_tokens", []) or []
                    ),
                    "bbox_xyxy": track.get("bbox_xyxy"),
                    "centroid_px": track.get("centroid_px"),
                    "camera": track.get("camera"),
                    "supporting_cameras": list(
                        track.get("supporting_cameras", []) or []
                    ),
                    "multiview_support_count": int(
                        track.get("multiview_support_count", 0)
                        or 0
                    ),
                    "world_m": round_list(world_m),
                    "world_cm": round_list(scale_list(world_m, 100.0), digits=2),
                    "first_observed_world_m": round_list(track.get("first_observed_world_m")),
                    "first_observed_world_cm": round_list(
                        scale_list(track.get("first_observed_world_m"), 100.0), digits=2
                    ),
                    "first_observed_top_surface_world_m": round_list(
                        track.get("first_observed_top_surface_world_m")
                    ),
                    "top_surface_world_m": round_list(top_surface_world_m),
                    "top_surface_world_cm": round_list(scale_list(top_surface_world_m, 100.0), digits=2),
                    "approach_world_m": round_list(approach_world_m),
                    "approach_world_cm": round_list(scale_list(approach_world_m, 100.0), digits=2),
                    "approach_quat_wxyz": round_list(track.get("approach_quat_wxyz")),
                    "grasp_world_m": round_list(grasp_world_m),
                    "grasp_world_cm": round_list(scale_list(grasp_world_m, 100.0), digits=2),
                    "grasp_quat_wxyz": round_list(track.get("grasp_quat_wxyz")),
                    "contact_world_m": round_list(contact_world_m),
                    "contact_world_cm": round_list(scale_list(contact_world_m, 100.0), digits=2),
                    "contact_quat_wxyz": round_list(track.get("contact_quat_wxyz")),
                    "operation_pose_candidates": [
                        dict(item)
                        for item in track.get("operation_pose_candidates", [])
                        if isinstance(item, dict)
                    ],
                    "operation_pose_candidate_provenance": {
                        str(candidate_id): dict(provenance)
                        for candidate_id, provenance in dict(
                            track.get(
                                "operation_pose_candidate_provenance",
                                {},
                            )
                            or {}
                        ).items()
                        if isinstance(provenance, dict)
                    },
                    "operation_pose_candidate_count": len(
                        track.get("operation_pose_candidates", []) or []
                    ),
                    "latest_world_m": round_list(track.get("world_m")),
                    "latest_world_cm": round_list(scale_list(track.get("world_m"), 100.0), digits=2),
                    "source_object_id": track.get("source_object_id"),
                    "source_text_prompt": track.get("source_text_prompt"),
                    "source_rank": track.get("source_rank"),
                    "identity_granularity": track.get(
                        "identity_granularity",
                        "instance",
                    ),
                    "identity_refinement_history": [
                        dict(item)
                        for item in track.get(
                            "identity_refinement_history",
                            [],
                        )
                        if isinstance(item, dict)
                    ],
                    "identity_relocation_history": [
                        dict(item)
                        for item in track.get(
                            "identity_relocation_history",
                            [],
                        )
                        if isinstance(item, dict)
                    ],
                    "action_geometry_state": track.get(
                        "action_geometry_state",
                        "verified",
                    ),
                    "action_geometry_pending_scope": (
                        dict(
                            track.get(
                                "relocation_action_geometry_scope"
                            )
                        )
                        if isinstance(
                            track.get(
                                "relocation_action_geometry_scope"
                            ),
                            dict,
                        )
                        else None
                    ),
                    "action_geometry_confirmation_count": len(
                        track.get(
                            "relocation_action_geometry_repair_samples",
                            [],
                        )
                        or []
                    ),
                    "action_geometry_repair_history": [
                        dict(item)
                        for item in track.get(
                            "action_geometry_repair_history",
                            [],
                        )
                        if isinstance(item, dict)
                    ],
                    "query_role": track.get("query_role"),
                    "query_instance_hint": track.get("query_instance_hint"),
                    "query_reason": track.get("query_reason"),
                    "quality": track.get("quality"),
                    "quality_warnings": list(track.get("quality_warnings", []) or []),
                    "actionable": bool(track.get("actionable", True)),
                    "first_seen_step": track.get("first_seen_step"),
                    "last_seen_step": track.get("last_seen_step", int(env_step)),
                    "missing_steps": track.get("missing_steps", 0),
                    "visible_count": track.get("visible_count", 0),
                    "history_length": track.get("history_length", 0),
                }
            identity_repair_lease = track.get(
                "relocation_identity_repair_lease"
            )
            if (
                isinstance(identity_repair_lease, dict)
                and identity_repair_lease.get("state")
                == "active"
            ):
                instance["identity_repair_lease"] = dict(
                    identity_repair_lease
                )
            identity_repair_expiration = track.get(
                "identity_repair_expiration"
            )
            if isinstance(identity_repair_expiration, dict):
                instance["identity_repair_expiration"] = dict(
                    identity_repair_expiration
                )
            instances.append(instance)
        return instances


class TemporalInstanceTracker:
    def __init__(
        self,
        *,
        window_size: int,
        max_missing_steps: int,
        stable_distance_m: float,
        temporal_action_geometry_distance_m: float,
        min_candidate_score: float,
        max_object_z_extent_m: float,
        max_object_xy_extent_m: float,
        max_object_world_z_m: float,
        robot_self_filter_radius_m: float,
        robot_self_filter_z_margin_m: float,
        enable_executable_candidate_temporal_consistency: bool = True,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.max_missing_steps = max(1, int(max_missing_steps))
        self.stable_distance_m = max(0.01, float(stable_distance_m))
        self.temporal_action_geometry_distance_m = max(0.0, float(temporal_action_geometry_distance_m))
        self.min_candidate_score = max(0.0, float(min_candidate_score))
        self.max_object_z_extent_m = max(0.0, float(max_object_z_extent_m))
        self.max_object_xy_extent_m = max(0.0, float(max_object_xy_extent_m))
        self.max_object_world_z_m = max(0.0, float(max_object_world_z_m))
        self.robot_self_filter_radius_m = max(0.0, float(robot_self_filter_radius_m))
        self.robot_self_filter_z_margin_m = max(0.0, float(robot_self_filter_z_margin_m))
        self.enable_executable_candidate_temporal_consistency = bool(
            enable_executable_candidate_temporal_consistency
        )
        self._tracks: dict[str, dict[str, Any]] = {}
        self._frames: list[dict[str, Any]] = []
        self._next_track_index = 1

    def reset(self) -> None:
        self._tracks = {}
        self._frames = []
        self._next_track_index = 1

    def apply_runtime_events(
        self,
        events: list[dict[str, Any]],
        *,
        env_step: int,
    ) -> list[dict[str, Any]]:
        for event in events:
            if not isinstance(event, dict):
                continue
            instance_ref = str(
                event.get(
                    "instance_ref",
                    event.get("track_id", ""),
                )
                or ""
            ).strip()
            if not instance_ref:
                continue
            matching_track_ids = [
                track_id
                for track_id, track in self._tracks.items()
                if instance_ref in instance_identity_refs(track)
            ]
            if len(matching_track_ids) != 1:
                continue
            track = self._tracks[matching_track_ids[0]]
            self._apply_runtime_event_to_track(
                track,
                event,
                env_step=env_step,
            )
        return [
            self._track_snapshot(track)
            for track in sorted(
                self._tracks.values(),
                key=track_sort_key,
            )
        ]

    def _apply_runtime_event_to_track(
        self,
        track: dict[str, Any],
        event: dict[str, Any],
        *,
        env_step: int,
    ) -> None:
        step = int(event.get("env_step", env_step))
        source = str(
            event.get("source", "runtime_physical_event") or
            "runtime_physical_event"
        ).strip()
        verified_role = str(
            event.get("verified_role", "") or ""
        ).strip().lower()
        if verified_role:
            role_history = [
                dict(item)
                for item in track.get("verified_roles", []) or []
                if isinstance(item, dict)
            ]
            role_record = {
                "role": verified_role,
                "effect_type": str(
                    event.get("effect_type", verified_role) or
                    verified_role
                ).strip().lower(),
                "env_step": step,
                "source": source,
            }
            if role_record not in role_history:
                role_history.append(role_record)
            track["verified_roles"] = role_history[-8:]
            locked_tokens = set(
                str(item)
                for item in track.get(
                    "verified_identity_tokens",
                    [],
                )
                or []
                if str(item)
            )
            locked_tokens.update(track_identity_tokens(track))
            track["verified_identity_tokens"] = sorted(
                locked_tokens
            )

        state = str(event.get("position_state", "") or "").strip()
        if state not in VALID_POSITION_STATES:
            return
        event_reason = str(
            event.get("reason", source) or source
        ).strip()
        if state == POSITION_MOTION_UNCERTAIN:
            last_motion_event = track.get("last_motion_event")
            try:
                last_motion_step = int(
                    last_motion_event.get("env_step", -1)
                    if isinstance(last_motion_event, dict)
                    else -1
                )
            except (TypeError, ValueError):
                last_motion_step = -1
            if (
                isinstance(last_motion_event, dict)
                and last_motion_step == step
                and str(
                    last_motion_event.get("source", "") or ""
                ).strip()
                == source
                and str(
                    last_motion_event.get("reason", "") or ""
                ).strip()
                == event_reason
            ):
                # The same physical manipulation boundary may be projected
                # into several perception refreshes at one environment step.
                # It invalidates pre-boundary geometry once; replaying it must
                # not erase clean repair samples collected afterwards.
                return
        if source in {
            "verified_tcp_attachment",
            "provisional_tcp_attachment",
            "verified_post_release_position",
        }:
            # A new manipulation boundary supersedes the operation target
            # from the preceding release.  A verified release may install a
            # replacement anchor below.
            track.pop(VERIFIED_OPERATION_TARGET_ANCHOR_KEY, None)
            self._invalidate_operation_geometry(
                track,
                clear_history=False,
            )
        track["position_state"] = state
        track["position_source"] = source
        track["position_state_reason"] = event_reason
        track["position_state_updated_step"] = step
        if state == POSITION_MOTION_UNCERTAIN:
            track.pop(VERIFIED_OPERATION_TARGET_ANCHOR_KEY, None)
            self._invalidate_operation_geometry(
                track,
                clear_history=True,
            )
            track["last_motion_event"] = {
                "env_step": step,
                "source": source,
                "reason": track["position_state_reason"],
            }
            return

        world = event.get("world_m")
        if is_number_list(world, length=3):
            current_world = [float(value) for value in world]
            prior_world = track.get(
                "stable_world_m",
                track.get("world_m"),
            )
            displacement = world_distance_m(
                current_world,
                prior_world,
            )
            if (
                displacement is not None
                and displacement > 0.005
                and source
                in {
                    "verified_tcp_attachment",
                    "provisional_tcp_attachment",
                    "verified_post_release_position",
                }
            ):
                # Samples before an actual manipulation belong to the old
                # object pose.  Keeping them in the averaging window would
                # pull the current coordinate back toward the table location.
                track["history"] = []
                track["history_length"] = 0
            track["world_m"] = round_list(current_world)
            track["stable_world_m"] = round_list(current_world)
            if state == POSITION_CURRENT_VERIFIED:
                track["last_verified_world_m"] = round_list(
                    current_world
                )
        confidence = finite_float(event.get("confidence"))
        if confidence is not None:
            track["last_verified_score"] = round_float(confidence)
        tolerance = finite_float(event.get("tolerance_m"))
        if tolerance is not None and tolerance > 0.0:
            track["position_tolerance_m"] = round_float(
                min(self.stable_distance_m, tolerance)
            )
        if state == POSITION_CURRENT_VERIFIED:
            track["last_verified_step"] = step
            track["last_verified_source"] = {
                "kind": source,
                "camera": event.get("camera"),
                "supporting_cameras": list(
                    event.get("supporting_cameras", []) or []
                ),
            }
            if source == "verified_post_release_position":
                operation_anchor = operation_target_anchor_from_event(
                    event,
                    env_step=step,
                    max_tolerance_m=self.stable_distance_m,
                )
                if operation_anchor is not None:
                    track[VERIFIED_OPERATION_TARGET_ANCHOR_KEY] = (
                        operation_anchor
                    )

    def _invalidate_operation_geometry(
        self,
        track: dict[str, Any],
        *,
        clear_history: bool,
    ) -> None:
        """Revoke geometry derived before a physical manipulation boundary."""

        reference_offsets = self._track_action_geometry_offsets(
            track
        )
        reference_candidates = [
            dict(item)
            for item in track.get(
                "operation_pose_candidates",
                [],
            )
            if isinstance(item, dict)
        ]
        if reference_offsets or reference_candidates:
            track[
                "last_verified_observation_geometry_reference"
            ] = {
                "offsets": reference_offsets,
                "operation_pose_candidates": reference_candidates,
                "source": "pre_manipulation_verified_geometry",
            }
        if clear_history:
            track["history"] = []
            track["history_length"] = 0
        for key in (
            "top_surface_world_m",
            "stable_top_surface_world_m",
            "approach_world_m",
            "approach_quat_wxyz",
            "stable_approach_world_m",
            "grasp_world_m",
            "grasp_quat_wxyz",
            "stable_grasp_world_m",
            "contact_world_m",
            "contact_quat_wxyz",
            "stable_contact_world_m",
        ):
            track[key] = None
        track["operation_pose_candidates"] = []
        track["operation_pose_candidate_provenance"] = {}
        for key in (
            "relocation_action_geometry_reference_offsets",
            "relocation_action_geometry_initial_warning",
            "relocation_action_geometry_last_warning",
            "relocation_action_geometry_repair_samples",
            "relocation_action_geometry_scope",
            "relocation_action_geometry_retained_provenance",
            "relocation_identity_repair_lease",
            "identity_repair_expiration",
        ):
            track.pop(key, None)
        track["action_geometry_state"] = "unavailable"

    def update(
        self,
        *,
        candidates: list[dict[str, Any]],
        env_step: int,
        identity_relocation_leases: list[dict[str, Any]] | None = None,
        position_events: list[dict[str, Any]] | None = None,
        observation_capture_id: int | None = None,
    ) -> dict[str, Any]:
        step = int(env_step)
        try:
            capture_id = (
                None
                if observation_capture_id is None
                else int(observation_capture_id)
            )
        except (TypeError, ValueError):
            capture_id = None
        if capture_id is not None and capture_id < 0:
            capture_id = None
        self.apply_runtime_events(
            list(position_events or []),
            env_step=step,
        )
        normalized: list[dict[str, Any]] = []
        dropped_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            if not has_candidate_evidence(candidate):
                continue
            quality = scene_candidate_quality(
                candidate,
                min_candidate_score=self.min_candidate_score,
                max_object_z_extent_m=self.max_object_z_extent_m,
                max_object_xy_extent_m=self.max_object_xy_extent_m,
                max_object_world_z_m=self.max_object_world_z_m,
                robot_state=candidate.get("robot_state"),
                robot_self_filter_radius_m=self.robot_self_filter_radius_m,
                robot_self_filter_z_margin_m=self.robot_self_filter_z_margin_m,
            )
            candidate_with_quality = dict(candidate)
            if capture_id is not None:
                candidate_with_quality[
                    "_observation_capture_id"
                ] = capture_id
            candidate_with_quality["quality"] = quality
            if not quality["actionable"]:
                dropped_candidates.append(compact_dropped_candidate(candidate_with_quality))
                continue
            normalized.append(self._normalize_candidate(candidate_with_quality))
        matches, association_rejections, identity_binding_outcomes = (
            self._associate_candidates(
                normalized,
                env_step=step,
                identity_relocation_leases=identity_relocation_leases,
            )
        )
        matched_track_ids = set(matches.values())

        frame_candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(normalized):
            association_rejection = association_rejections.get(index)
            if association_rejection is not None:
                warning, rejected_track_id = association_rejection
                rejected = dict(candidate)
                quality = dict(rejected.get("quality") or {})
                quality["warnings"] = dedupe_preserving_order(
                    [*list(quality.get("warnings", []) or []), warning]
                )
                quality["actionable"] = False
                rejected["quality"] = quality
                if rejected_track_id:
                    rejected["_rejected_track_id"] = rejected_track_id
                dropped_candidates.append(compact_dropped_candidate(rejected))
                continue
            track_id = matches.get(index)
            if track_id is None:
                conflict = self._role_conflict_with_memory(candidate)
                if conflict:
                    candidate_with_warning = dict(candidate)
                    quality = dict(candidate_with_warning.get("quality") or {})
                    warnings = list(quality.get("warnings", []) or [])
                    warnings.append(conflict)
                    quality["warnings"] = warnings
                    quality["actionable"] = False
                    candidate_with_warning["quality"] = quality
                    dropped_candidates.append(compact_dropped_candidate(candidate_with_warning))
                    continue
                track_id = self._spawn_track(candidate, env_step=step)
            else:
                track = self._tracks[track_id]
                if track.get("action_geometry_state") == "unavailable":
                    observation_reference = track.get(
                        "last_verified_observation_geometry_reference"
                    )
                    reference_offsets = (
                        observation_reference.get("offsets")
                        if isinstance(observation_reference, dict)
                        else None
                    )
                    candidate_offsets = (
                        self._candidate_action_geometry_offsets(
                            candidate
                        )
                    )
                    reference_warning = (
                        self._action_geometry_warning_between_offsets(
                            candidate_offsets,
                            reference_offsets,
                        )
                        if isinstance(reference_offsets, dict)
                        else ""
                    )
                    reference_candidate_consistency = (
                        executable_candidate_sets_consistent(
                            observation_reference.get(
                                "operation_pose_candidates"
                            ),
                            candidate.get(
                                "_operation_pose_candidates"
                            ),
                            max_translation_m=(
                                OPERATION_GEOMETRY_TRANSLATION_CHANGE_M
                            ),
                            max_rotation_rad=(
                                OPERATION_GEOMETRY_ROTATION_CHANGE_RAD
                            ),
                        )
                        if (
                            self.enable_executable_candidate_temporal_consistency
                            and isinstance(observation_reference, dict)
                        )
                        else None
                    )
                    matches_prior_verified_geometry = bool(
                        isinstance(observation_reference, dict)
                        and not reference_warning
                        and reference_candidate_consistency is not False
                        and (
                            bool(reference_offsets)
                            or (
                                self.enable_executable_candidate_temporal_consistency
                                and reference_candidate_consistency is True
                            )
                        )
                    )
                    if not matches_prior_verified_geometry:
                        candidate = (
                            self._begin_relocation_geometry_quarantine(
                                track_id,
                                candidate,
                                env_step=step,
                                warning=(
                                    reference_warning
                                    or (
                                        "motion_uncertain_reacquisition:"
                                        "clean_geometry_confirmation_required"
                                    )
                                ),
                                force_observed_geometry_rebuild=True,
                            )
                        )
                        frame_candidates.append(
                            compact_candidate_for_memory(
                                candidate,
                                track_id=track_id,
                            )
                        )
                        continue
                relocation_evidence = candidate.get(
                    "_identity_relocation_evidence"
                )
                refresh_scope = (
                    self._action_geometry_refresh_scope(
                        relocation_evidence
                    )
                    if isinstance(relocation_evidence, dict)
                    else None
                )
                geometry_warning = (
                    self._temporal_action_geometry_warning(
                        track_id,
                        candidate,
                        action_geometry_scope=refresh_scope,
                    )
                )
                requires_observed_geometry_rebuild = bool(
                    isinstance(relocation_evidence, dict)
                    and relocation_evidence.get(
                        "rebuild_action_geometry_from_observations"
                    )
                    is True
                )
                if (
                    requires_observed_geometry_rebuild
                    and self._tracks[track_id].get(
                        "action_geometry_state"
                    )
                    != "relocation_pending"
                ):
                    candidate = (
                        self._begin_relocation_geometry_quarantine(
                            track_id,
                            candidate,
                            env_step=step,
                            warning=(
                                geometry_warning
                                or (
                                    "verified_operation_target_relocation:"
                                    "clean_geometry_confirmation_required"
                                )
                            ),
                        )
                    )
                    frame_candidates.append(
                        compact_candidate_for_memory(
                            candidate,
                            track_id=track_id,
                        )
                    )
                    continue
                if geometry_warning:
                    if isinstance(
                        candidate.get(
                            "_identity_relocation_evidence"
                        ),
                        dict,
                    ):
                        candidate = (
                            self._begin_relocation_geometry_quarantine(
                                track_id,
                                candidate,
                                env_step=step,
                                warning=geometry_warning,
                            )
                        )
                        frame_candidates.append(
                            compact_candidate_for_memory(
                                candidate,
                                track_id=track_id,
                            )
                        )
                        continue
                    if (
                        self._tracks[track_id].get(
                            "action_geometry_state"
                        )
                        == "relocation_pending"
                    ):
                        candidate = (
                            self._retain_relocation_geometry_quarantine(
                                track_id,
                                candidate,
                                env_step=step,
                                warning=geometry_warning,
                            )
                        )
                        frame_candidates.append(
                            compact_candidate_for_memory(
                                candidate,
                                track_id=track_id,
                            )
                        )
                        continue
                    rejected = dict(candidate)
                    quality = dict(rejected.get("quality") or {})
                    quality["warnings"] = dedupe_preserving_order(
                        [
                            *list(quality.get("warnings", []) or []),
                            geometry_warning,
                        ]
                    )
                    quality["actionable"] = False
                    rejected["quality"] = quality
                    rejected["_rejected_track_id"] = track_id
                    dropped_candidates.append(
                        compact_dropped_candidate(rejected)
                    )
                    self._mark_geometry_inconsistent(
                        track_id,
                        env_step=step,
                        warning=geometry_warning,
                    )
                    continue
                if (
                    self._tracks[track_id].get(
                        "action_geometry_state"
                    )
                    == "relocation_pending"
                ):
                    candidate = (
                        self._confirm_relocated_action_geometry(
                            track_id,
                            candidate,
                            env_step=step,
                        )
                    )
                else:
                    self._update_track(
                        track_id,
                        candidate,
                        env_step=step,
                    )
            frame_candidates.append(compact_candidate_for_memory(candidate, track_id=track_id))

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            if track_id not in {item.get("track_id") for item in frame_candidates}:
                missing_steps = max(1, step - int(track.get("last_seen_step", step)))
                if missing_steps > self.max_missing_steps:
                    # Visibility loss is not evidence of object motion.  A
                    # physically unchallenged verified position remains in
                    # structured memory even after the short visual tracking
                    # window; only an explicit physical event may invalidate
                    # it.
                    if track.get("position_state") in {
                        POSITION_CURRENT_VERIFIED,
                        POSITION_MEMORY_VALID,
                    } and track.get(
                        "action_geometry_state"
                    ) != "identity_repair_expired":
                        track["status"] = "tracked"
                        track["stability"] = "missing_current_frame"
                        track["missing_steps"] = missing_steps
                        track["position_state"] = POSITION_MEMORY_VALID
                        track["position_source"] = (
                            "retained_verified_memory"
                        )
                        track["position_state_reason"] = (
                            "not observed; no physical motion event recorded"
                        )
                        track["position_state_updated_step"] = step
                        continue
                    expiration = track.get(
                        "identity_repair_expiration"
                    )
                    if (
                        track.get("action_geometry_state")
                        == "identity_repair_expired"
                        and isinstance(expiration, dict)
                    ):
                        try:
                            expired_step = int(
                                expiration.get(
                                    "env_step",
                                    step,
                                )
                            )
                        except (TypeError, ValueError):
                            expired_step = step
                        # Retain only the non-executable terminal audit state
                        # for one ordinary missing window after lease expiry.
                        # This lets the planner observe the explicit replan
                        # condition without reviving any stale action pose.
                        if (
                            step - expired_step
                            <= self.max_missing_steps
                        ):
                            track["status"] = "tracked"
                            track["stability"] = (
                                "identity_repair_expired"
                            )
                            track["missing_steps"] = (
                                missing_steps
                            )
                            continue
                    del self._tracks[track_id]
                    continue
                track["status"] = "tracked"
                track["stability"] = (
                    "identity_repair_expired"
                    if track.get("action_geometry_state")
                    == "identity_repair_expired"
                    else "missing_current_frame"
                )
                track["missing_steps"] = missing_steps
                if track.get("position_state") in {
                    POSITION_CURRENT_VERIFIED,
                    POSITION_MEMORY_VALID,
                }:
                    track["position_state"] = POSITION_MEMORY_VALID
                    track["position_source"] = (
                        "retained_verified_memory"
                    )
                    track["position_state_reason"] = (
                        "not observed; no physical motion event recorded"
                    )
                    track["position_state_updated_step"] = step

        self._frames.append({"env_step": step, "candidates": frame_candidates})
        self._frames = self._frames[-self.window_size :]
        return {
            "window_size": self.window_size,
            "frames": list(self._frames),
            "tracks": [self._track_snapshot(track) for track in sorted(self._tracks.values(), key=track_sort_key)],
            "num_visible_candidates": len(frame_candidates),
            "num_active_tracks": len(self._tracks),
            "dropped_candidates": dropped_candidates,
            "identity_binding_outcomes": identity_binding_outcomes,
        }

    def _normalize_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        aliases = candidate_class_aliases(candidate)
        class_name = aliases[0] if aliases else normalize_class(candidate.get("object_id", "object")) or "object"
        normalized = dict(candidate)
        normalized["_class_name"] = class_name
        normalized["_class_aliases"] = aliases or [class_name]
        normalized["_world_m"] = candidate_world(candidate)
        normalized["_top_surface_world_m"] = candidate_top_surface(candidate)
        normalized["_approach_world_m"] = candidate_approach(candidate)
        normalized["_approach_quat_wxyz"] = candidate_grounded_quat(candidate, "approach_pose_world")
        normalized["_grasp_world_m"] = candidate_grasp(candidate)
        normalized["_grasp_quat_wxyz"] = candidate_grounded_quat(candidate, "grasp_pose_world")
        normalized["_contact_world_m"] = candidate_contact(candidate)
        normalized["_contact_quat_wxyz"] = candidate_grounded_quat(candidate, "contact_pose_world")
        normalized["_operation_pose_candidates"] = candidate_operation_pose_candidates(candidate)
        normalized["_bbox_xyxy"] = parse_bbox(candidate.get("bbox_xyxy"))
        normalized["_centroid_px"] = parse_centroid(candidate.get("centroid_px"))
        normalized["_source_instance_key"] = candidate_source_instance_key(candidate)
        return normalized

    def _associate_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        env_step: int,
        identity_relocation_leases: list[dict[str, Any]] | None = None,
    ) -> tuple[
        dict[int, str],
        dict[int, tuple[str, str | None]],
        list[dict[str, Any]],
    ]:
        """Associate observations while treating explicit instance refs as constraints.

        A non-Oracle identity-bound query may update only the already-established
        track named by ``instance_ref``. Each query/camera result must contribute
        at most one geometrically compatible candidate. When several cameras
        independently produce a unique observation, the strongest observation
        updates the track and the others are treated as redundant. Every rejected
        bound candidate is prevented from matching another track or spawning a
        new identity. Unbound discovery and Oracle actor-ID association retain
        the existing greedy behavior.
        """
        matches: dict[int, str] = {}
        rejections: dict[int, tuple[str, str | None]] = {}
        outcomes: list[dict[str, Any]] = []
        assigned_tracks: set[str] = set()
        reserved_tracks: set[str] = set()
        relocation_leases = self._identity_relocation_lease_map(
            identity_relocation_leases,
            env_step=env_step,
        )
        bound_groups: dict[str, dict[str, list[int]]] = {}
        unbound_indices: list[int] = []
        for candidate_index, candidate in enumerate(candidates):
            instance_ref = identity_bound_candidate_ref(candidate)
            if instance_ref:
                query_key = identity_bound_candidate_query_key(candidate)
                bound_groups.setdefault(instance_ref, {}).setdefault(query_key, []).append(
                    candidate_index
                )
            else:
                unbound_indices.append(candidate_index)

        for instance_ref, query_groups in bound_groups.items():
            candidate_indices = [
                candidate_index
                for group_indices in query_groups.values()
                for candidate_index in group_indices
            ]
            referenced_track_ids = [
                track_id
                for track_id, track in self._tracks.items()
                if instance_ref in instance_identity_refs(track)
            ]
            if len(referenced_track_ids) != 1:
                status = (
                    "unresolved_reference"
                    if not referenced_track_ids
                    else "ambiguous_reference"
                )
                warning = f"bound_identity_{status}:{instance_ref}"
                for candidate_index in candidate_indices:
                    rejections[candidate_index] = (warning, None)
                outcomes.append(
                    {
                        "instance_ref": instance_ref,
                        "status": status,
                        "candidate_count": len(candidate_indices),
                        "compatible_candidate_count": 0,
                    }
                )
                continue

            track_id = referenced_track_ids[0]
            reserved_tracks.add(track_id)
            relocation_lease = relocation_leases.get(instance_ref)
            track = self._tracks[track_id]
            repair_lease = bool(
                relocation_lease is not None
                and track.get("action_geometry_state")
                == "relocation_pending"
            )
            if (
                track.get("action_geometry_state")
                == "identity_repair_expired"
                and relocation_lease is None
            ):
                warning = (
                    "bound_identity_repair_expired:"
                    f"{instance_ref}"
                )
                for candidate_index in candidate_indices:
                    rejections[candidate_index] = (
                        warning,
                        track_id,
                    )
                outcomes.append(
                    {
                        "instance_ref": instance_ref,
                        "track_id": track_id,
                        "status": "identity_repair_expired",
                        "candidate_count": len(candidate_indices),
                        "compatible_candidate_count": 0,
                    }
                )
                continue
            if relocation_lease is not None:
                if repair_lease:
                    self._record_identity_repair_lease_attempt(
                        track_id
                    )
                target_world = relocation_lease["target_world_m"]
                tolerance_m = relocation_lease["tolerance_m"]
                viable_relocations: list[
                    tuple[float, float, int, str]
                ] = []
                near_target_candidate_indices: set[int] = set()
                ambiguous_query_count = 0
                for query_key, group_indices in query_groups.items():
                    near_target: list[tuple[float, float, int]] = []
                    for candidate_index in group_indices:
                        distance_m = world_distance_m(
                            candidates[candidate_index].get("_world_m"),
                            target_world,
                        )
                        if (
                            distance_m is None
                            or distance_m > tolerance_m
                        ):
                            continue
                        near_target_candidate_indices.add(
                            candidate_index
                        )
                        try:
                            candidate_score = float(
                                candidates[candidate_index].get(
                                    "score",
                                    0.0,
                                )
                                or 0.0
                            )
                        except (TypeError, ValueError):
                            candidate_score = 0.0
                        near_target.append(
                            (
                                distance_m,
                                -candidate_score,
                                candidate_index,
                            )
                        )
                    near_target.sort()
                    relocation_hypotheses = (
                        cluster_bound_identity_observation_variants(
                            [
                                (
                                    distance_m,
                                    negative_score,
                                    candidate_index,
                                    query_key,
                                )
                                for (
                                    distance_m,
                                    negative_score,
                                    candidate_index,
                                ) in near_target
                            ],
                            candidates=candidates,
                        )
                    )
                    if len(relocation_hypotheses) == 1:
                        (
                            distance_m,
                            negative_score,
                            candidate_index,
                            _,
                        ) = relocation_hypotheses[0][0]
                        viable_relocations.append(
                            (
                                distance_m,
                                negative_score,
                                candidate_index,
                                query_key,
                            )
                        )
                    elif len(relocation_hypotheses) > 1:
                        ambiguous_query_count += 1

                viable_relocations.sort()
                if (
                    viable_relocations
                    and ambiguous_query_count == 0
                    and track_id not in assigned_tracks
                ):
                    (
                        distance_m,
                        _,
                        matched_index,
                        _,
                    ) = viable_relocations[0]
                    prior_world = self._tracks[track_id].get(
                        "stable_world_m",
                        self._tracks[track_id].get("world_m"),
                    )
                    prior_target_error = world_distance_m(
                        prior_world,
                        target_world,
                    )
                    position_only_geometry_warning = ""
                    if (
                        relocation_lease.get(
                            POSITION_ONLY_ACTION_GEOMETRY_QUARANTINE_LEASE_KEY
                        )
                        is True
                        and track.get("action_geometry_state")
                        != "relocation_pending"
                    ):
                        position_only_geometry_warning = (
                            self._temporal_action_geometry_warning(
                                track_id,
                                candidates[matched_index],
                                action_geometry_scope=(
                                    self._action_geometry_refresh_scope(
                                        relocation_lease
                                    )
                                ),
                            )
                        )
                    position_only_release_boundary = bool(
                        position_only_geometry_warning
                    )
                    needs_relocation_evidence = (
                        prior_target_error is None
                        or prior_target_error > tolerance_m
                        or track.get("action_geometry_state")
                        == "identity_repair_expired"
                        or position_only_release_boundary
                    )
                    if needs_relocation_evidence:
                        relocation_evidence = {
                            "instance_ref": instance_ref,
                            "target_world_m": round_list(target_world),
                            "candidate_target_error_m": round(
                                float(distance_m),
                                6,
                            ),
                            "tolerance_m": round(
                                float(tolerance_m),
                                6,
                            ),
                            "prior_world_m": round_list(prior_world),
                            "prior_target_error_m": (
                                None
                                if prior_target_error is None
                                else round(
                                    float(prior_target_error),
                                    6,
                                )
                            ),
                            "source": str(
                                relocation_lease.get(
                                    "source",
                                    "runtime_manipulation_state",
                                )
                                or "runtime_manipulation_state"
                            ),
                            "arm": str(
                                relocation_lease.get("arm", "")
                                or ""
                            ),
                            "action_modes": [
                                str(mode)
                                for mode in relocation_lease.get(
                                    "action_modes",
                                    [],
                                )
                                or []
                                if str(mode).strip()
                            ],
                            "release_step": relocation_lease.get(
                                "release_step"
                            ),
                            "rebuild_action_geometry_from_observations": (
                                relocation_lease.get(
                                    "rebuild_action_geometry_from_observations"
                                )
                                is True
                            ),
                        }
                        if position_only_release_boundary:
                            relocation_evidence.update(
                                {
                                    "reason": (
                                        "release_position_verification_with_"
                                        "quarantined_action_geometry"
                                    ),
                                    "action_geometry_warning": (
                                        position_only_geometry_warning
                                    ),
                                }
                            )
                        candidates[matched_index][
                            "_identity_relocation_evidence"
                        ] = relocation_evidence
                    matches[matched_index] = track_id
                    assigned_tracks.add(track_id)
                    for candidate_index in candidate_indices:
                        if candidate_index == matched_index:
                            continue
                        rejections[candidate_index] = (
                            (
                                (
                                    "bound_identity_repair_redundant:"
                                    f"{instance_ref}"
                                )
                                if (
                                    repair_lease
                                    and candidate_index
                                    in near_target_candidate_indices
                                )
                                else (
                                    "bound_identity_repair_outside_lease:"
                                    f"{instance_ref}"
                                )
                                if repair_lease
                                else (
                                    "bound_identity_relocation_redundant:"
                                    f"{instance_ref}"
                                )
                            ),
                            track_id,
                        )
                    outcomes.append(
                        {
                            "instance_ref": instance_ref,
                            "track_id": track_id,
                            "status": "matched",
                            "candidate_count": len(candidate_indices),
                            "compatible_candidate_count": len(
                                viable_relocations
                            ),
                            "query_group_count": len(query_groups),
                            "compatible_query_count": len(
                                viable_relocations
                            ),
                            "ambiguous_query_count": (
                                ambiguous_query_count
                            ),
                            "query_error_count": 0,
                            "matched_candidate_index": matched_index,
                            "matched_by": (
                                "post_release_identity_repair_lease"
                                if repair_lease
                                else (
                                    "release_position_only_geometry_quarantine"
                                    if position_only_release_boundary
                                    else (
                                        "operation_target_relocation"
                                        if needs_relocation_evidence
                                        else "operation_target_continuity"
                                    )
                                )
                            ),
                            "relocation_target_world_m": round_list(
                                target_world
                            ),
                            "relocation_target_error_m": round(
                                float(distance_m),
                                6,
                            ),
                            "relocation_tolerance_m": round(
                                float(tolerance_m),
                                6,
                            ),
                        }
                    )
                    continue

                recovery_observation = (
                    self._unique_relocation_recovery_observation(
                        query_groups=query_groups,
                        candidates=candidates,
                        relocation_lease=relocation_lease,
                    )
                    if (
                        not repair_lease
                        and ambiguous_query_count == 0
                    )
                    else {}
                )
                recovery_matched_index = recovery_observation.get(
                    "matched_index"
                )
                if (
                    isinstance(recovery_matched_index, int)
                    and track_id not in assigned_tracks
                ):
                    matched_index = recovery_matched_index
                    candidate_world = candidates[matched_index].get(
                        "_world_m"
                    )
                    anchor = dict(
                        recovery_observation.get("anchor") or {}
                    )
                    target_error_m = world_distance_m(
                        candidate_world,
                        target_world,
                    )
                    prior_world = self._tracks[track_id].get(
                        "stable_world_m",
                        self._tracks[track_id].get("world_m"),
                    )
                    prior_candidate_error = world_distance_m(
                        prior_world,
                        candidate_world,
                    )
                    candidates[matched_index][
                        "_identity_relocation_evidence"
                    ] = {
                        "instance_ref": instance_ref,
                        "target_world_m": list(candidate_world),
                        "candidate_target_error_m": 0.0,
                        "tolerance_m": float(
                            anchor.get("tolerance_m", tolerance_m)
                        ),
                        "prior_world_m": round_list(prior_world),
                        "prior_target_error_m": (
                            None
                            if prior_candidate_error is None
                            else round(
                                float(prior_candidate_error),
                                6,
                            )
                        ),
                        "source": (
                            "runtime_identity_recovery_anchor"
                        ),
                        "anchor_source": anchor.get("source"),
                        "committed_target_world_m": round_list(
                            target_world
                        ),
                        "committed_target_error_m": (
                            None
                            if target_error_m is None
                            else round(float(target_error_m), 6)
                        ),
                        "arm": str(
                            relocation_lease.get("arm", "") or ""
                        ),
                        "release_step": relocation_lease.get(
                            "release_step"
                        ),
                    }
                    matches[matched_index] = track_id
                    assigned_tracks.add(track_id)
                    recovery_eligible_indices = set(
                        recovery_observation.get(
                            "eligible_indices",
                            set(),
                        )
                        or set()
                    )
                    for candidate_index in candidate_indices:
                        if candidate_index == matched_index:
                            continue
                        rejections[candidate_index] = (
                            (
                                "bound_identity_recovery_redundant:"
                                f"{instance_ref}"
                                if candidate_index
                                in recovery_eligible_indices
                                else (
                                    "bound_identity_recovery_outside_anchors:"
                                    f"{instance_ref}"
                                )
                            ),
                            track_id,
                        )
                    outcomes.append(
                        {
                            "instance_ref": instance_ref,
                            "track_id": track_id,
                            "status": "matched_recovery_observation",
                            "candidate_count": len(candidate_indices),
                            "compatible_candidate_count": 1,
                            "query_group_count": len(query_groups),
                            "compatible_query_count": 1,
                            "ambiguous_query_count": 0,
                            "query_error_count": 0,
                            "matched_candidate_index": matched_index,
                            "matched_by": (
                                "runtime_manipulation_history_anchor"
                            ),
                            "relocation_target_world_m": round_list(
                                target_world
                            ),
                            "relocation_target_error_m": (
                                None
                                if target_error_m is None
                                else round(float(target_error_m), 6)
                            ),
                            "relocation_tolerance_m": round(
                                float(tolerance_m),
                                6,
                            ),
                            "recovery_anchor_source": anchor.get(
                                "source"
                            ),
                            "recovery_anchor_world_m": round_list(
                                anchor.get("world_m")
                            ),
                            "recovery_anchor_error_m": round(
                                float(anchor.get("distance_m", 0.0)),
                                6,
                            ),
                            "recovery_observation_hypothesis_count": (
                                recovery_observation.get(
                                    "observation_hypothesis_count",
                                    1,
                                )
                            ),
                        }
                    )
                    continue

                recovery_ambiguous = bool(
                    recovery_observation.get("ambiguous")
                )
                warning = (
                    (
                        "bound_identity_repair_ambiguous_candidates:"
                        f"{instance_ref}"
                    )
                    if repair_lease and ambiguous_query_count
                    else (
                        "bound_identity_repair_target_not_observed:"
                        f"{instance_ref}"
                    )
                    if repair_lease
                    else (
                        "bound_identity_recovery_ambiguous_candidates:"
                        f"{instance_ref}"
                    )
                    if recovery_ambiguous
                    else (
                        "bound_identity_relocation_ambiguous_target_candidates:"
                        f"{instance_ref}"
                    )
                    if ambiguous_query_count
                    else (
                        "bound_identity_relocation_target_not_observed:"
                        f"{instance_ref}"
                    )
                )
                for candidate_index in candidate_indices:
                    rejections[candidate_index] = (warning, track_id)
                outcomes.append(
                    {
                        "instance_ref": instance_ref,
                        "track_id": track_id,
                        "status": (
                            "ambiguous_relocation_repair_candidates"
                            if repair_lease
                            and ambiguous_query_count
                            else "relocation_repair_target_not_observed"
                            if repair_lease
                            else "ambiguous_relocation_target_candidates"
                            if ambiguous_query_count
                            else "ambiguous_recovery_anchor_candidates"
                            if recovery_ambiguous
                            else "relocation_target_not_observed"
                        ),
                        "candidate_count": len(candidate_indices),
                        "compatible_candidate_count": len(
                            viable_relocations
                        ),
                        "query_group_count": len(query_groups),
                        "compatible_query_count": len(
                            viable_relocations
                        ),
                        "ambiguous_query_count": (
                            ambiguous_query_count
                            + int(
                                recovery_observation.get(
                                    "ambiguous_query_count",
                                    0,
                                )
                                or 0
                            )
                        ),
                        "query_error_count": 0,
                        "relocation_target_world_m": round_list(
                            target_world
                        ),
                        "relocation_tolerance_m": round(
                            float(tolerance_m),
                            6,
                        ),
                    }
                )
                continue

            compatible_by_query: dict[
                str,
                list[tuple[float, int]],
            ] = {}
            compatible_candidate_count = 0
            query_error_count = 0
            for query_key, group_indices in query_groups.items():
                explicit_errors = dedupe_preserving_order(
                    [
                        str(candidates[index].get("identity_binding_error", "") or "").strip()
                        for index in group_indices
                        if str(candidates[index].get("identity_binding_error", "") or "").strip()
                    ]
                )
                if explicit_errors:
                    warning = f"bound_identity_query_error:{explicit_errors[0]}"
                    for candidate_index in group_indices:
                        rejections[candidate_index] = (warning, track_id)
                    query_error_count += 1
                    continue

                compatible: list[tuple[float, int]] = []
                for candidate_index in group_indices:
                    score = self._association_score(
                        self._tracks[track_id],
                        candidates[candidate_index],
                    )
                    if score is not None:
                        compatible.append((score, candidate_index))
                compatible.sort(key=lambda item: (-item[0], item[1]))
                compatible_by_query[query_key] = compatible
                compatible_candidate_count += len(compatible)
                compatible_indices = {
                    candidate_index
                    for _, candidate_index in compatible
                }
                for candidate_index in group_indices:
                    if candidate_index not in compatible_indices:
                        rejections[candidate_index] = (
                            f"bound_identity_candidate_mismatch:{instance_ref}",
                            track_id,
                        )

            position_anchors = self._verified_position_anchors(track)
            viable_observations: list[
                tuple[float, int, str]
            ] = []
            ambiguous_query_count = 0
            matched_by = "identity_and_geometry"
            anchor_matches_by_candidate: dict[
                int,
                list[dict[str, Any]],
            ] = {}
            if position_anchors:
                world_eligible: list[
                    tuple[float, float, int, str]
                ] = []
                for query_key, compatible in compatible_by_query.items():
                    for score, candidate_index in compatible:
                        anchor_matches = eligible_anchor_matches(
                            candidates[candidate_index].get("_world_m"),
                            position_anchors,
                        )
                        refinement_evidence = candidates[
                            candidate_index
                        ].get(
                            "_identity_refinement_evidence"
                        )
                        if (
                            not anchor_matches
                            and not isinstance(
                                refinement_evidence,
                                dict,
                            )
                        ):
                            rejections[candidate_index] = (
                                (
                                    "bound_identity_outside_verified_position:"
                                    f"{instance_ref}"
                                ),
                                track_id,
                            )
                            continue
                        if anchor_matches:
                            anchor_matches_by_candidate[
                                candidate_index
                            ] = anchor_matches
                            distance = float(
                                anchor_matches[0]["distance_m"]
                            )
                        else:
                            distance = min(
                                world_distance_m(
                                    candidates[candidate_index].get(
                                        "_world_m"
                                    ),
                                    item.get("world_m"),
                                )
                                or float("inf")
                                for item in position_anchors
                            )
                        world_eligible.append(
                            (
                                distance,
                                -score,
                                candidate_index,
                                query_key,
                            )
                        )
                world_eligible.sort()
                observation_hypotheses = (
                    cluster_bound_identity_observation_variants(
                        world_eligible,
                        candidates=candidates,
                    )
                )
                if len(observation_hypotheses) == 1:
                    # Check every view variant before choosing a representative.
                    # Otherwise a slightly closer robot-contaminated mask can
                    # discard a clean view and only fail the temporal check
                    # after the clean observation is no longer available.
                    observation_cluster = observation_hypotheses[0]
                    preferred_indices, geometry_rejections = (
                        self._temporally_consistent_observation_indices(
                            track_id,
                            [item[2] for item in observation_cluster],
                            candidates=candidates,
                        )
                    )
                    preferred_cluster = [
                        item
                        for item in observation_cluster
                        if item[2] in preferred_indices
                    ]
                    (
                        distance,
                        negative_score,
                        candidate_index,
                        query_key,
                    ) = preferred_cluster[0]
                    viable_observations = [
                        (
                            -negative_score,
                            candidate_index,
                            query_key,
                        )
                    ]
                    selected_anchor_matches = (
                        anchor_matches_by_candidate.get(
                            candidate_index,
                            [],
                        )
                    )
                    matched_by = (
                        "verified_operation_target"
                        if selected_anchor_matches
                        and selected_anchor_matches[0].get("kind")
                        == "verified_operation_target"
                        else "verified_world_position"
                    )
                    for redundant in observation_cluster:
                        redundant_index = redundant[2]
                        if redundant_index == candidate_index:
                            continue
                        rejections[redundant_index] = (
                            (
                                geometry_rejections.get(redundant_index)
                                or (
                                    "bound_identity_redundant_observation_variant:"
                                    f"{instance_ref}"
                                )
                            ),
                            track_id,
                        )
                elif len(observation_hypotheses) > 1:
                    ambiguous_query_count = 1
                    warning = (
                        f"bound_identity_ambiguous_candidates:{instance_ref}:"
                        f"{len(observation_hypotheses)}"
                    )
                    for _, _, candidate_index, _ in world_eligible:
                        rejections[candidate_index] = (
                            warning,
                            track_id,
                        )
            else:
                # Without a physically valid world anchor, retain the
                # fail-closed per-query contract.  Image overlap may establish
                # compatibility, but it cannot choose among multiple same-view
                # objects.
                for query_key, compatible in compatible_by_query.items():
                    if len(compatible) == 1:
                        score, candidate_index = compatible[0]
                        viable_observations.append(
                            (score, candidate_index, query_key)
                        )
                    elif len(compatible) > 1:
                        ambiguous_query_count += 1
                        warning = (
                            f"bound_identity_ambiguous_candidates:{instance_ref}:"
                            f"{len(compatible)}"
                        )
                        for _, candidate_index in compatible:
                            rejections[candidate_index] = (
                                warning,
                                track_id,
                            )

            # Several cameras may each contribute one identity-compatible
            # observation while a physical event has temporarily invalidated
            # the world-position anchor.  Prefer a view whose action geometry
            # agrees with the track before the normal association score chooses
            # a representative.  Do not resolve a genuine same-query ambiguity,
            # and preserve the existing single/all-inconsistent fail-closed path.
            if (
                len(viable_observations) > 1
                and ambiguous_query_count == 0
            ):
                preferred_indices, geometry_rejections = (
                    self._temporally_consistent_observation_indices(
                        track_id,
                        [item[1] for item in viable_observations],
                        candidates=candidates,
                    )
                )
                viable_observations = [
                    item
                    for item in viable_observations
                    if item[1] in preferred_indices
                ]
                for rejected_index, warning in geometry_rejections.items():
                    rejections[rejected_index] = (warning, track_id)

            viable_observations.sort(
                key=lambda item: (-item[0], item[1], item[2])
            )
            if (
                viable_observations
                and ambiguous_query_count == 0
                and track_id not in assigned_tracks
            ):
                matched_index = viable_observations[0][1]
                refinement_evidence = candidates[matched_index].get(
                    "_identity_refinement_evidence"
                )
                selected_anchor_matches = (
                    anchor_matches_by_candidate.get(
                        matched_index,
                        [],
                    )
                )
                operation_target_match = next(
                    (
                        item
                        for item in selected_anchor_matches
                        if item.get("kind")
                        == "verified_operation_target"
                    ),
                    None,
                )
                observation_match = next(
                    (
                        item
                        for item in selected_anchor_matches
                        if item.get("kind")
                        == "verified_observation"
                    ),
                    None,
                )
                if (
                    operation_target_match is not None
                    and observation_match is None
                ):
                    prior_world = track.get(
                        "stable_world_m",
                        track.get("world_m"),
                    )
                    target_world = operation_target_match.get(
                        "world_m"
                    )
                    candidates[matched_index][
                        "_identity_relocation_evidence"
                    ] = {
                        "instance_ref": instance_ref,
                        "target_world_m": round_list(target_world),
                        "candidate_target_error_m": round(
                            float(
                                operation_target_match["distance_m"]
                            ),
                            6,
                        ),
                        "tolerance_m": round(
                            float(
                                operation_target_match["tolerance_m"]
                            ),
                            6,
                        ),
                        "prior_world_m": round_list(prior_world),
                        "prior_target_error_m": (
                            None
                            if (
                                prior_target_error := world_distance_m(
                                    prior_world,
                                    target_world,
                                )
                            )
                            is None
                            else round(float(prior_target_error), 6)
                        ),
                        "source": (
                            "verified_operation_target_anchor"
                        ),
                        "reason": (
                            "candidate uniquely matched a verified "
                            "operation target outside the last observed "
                            "centroid gate"
                        ),
                        "target_id": operation_target_match.get(
                            "target_id"
                        ),
                        "arm": operation_target_match.get("arm"),
                        "release_step": operation_target_match.get(
                            "verified_step"
                        ),
                        "rebuild_action_geometry_from_observations": True,
                    }
                matches[matched_index] = track_id
                assigned_tracks.add(track_id)
                for _, redundant_index, _ in viable_observations[1:]:
                    rejections[redundant_index] = (
                        f"bound_identity_redundant_observation:{instance_ref}",
                        track_id,
                    )
                outcomes.append(
                    {
                        "instance_ref": instance_ref,
                        "track_id": track_id,
                        "status": "matched",
                        "candidate_count": len(candidate_indices),
                        "compatible_candidate_count": compatible_candidate_count,
                        "query_group_count": len(query_groups),
                        "compatible_query_count": len(viable_observations),
                        "ambiguous_query_count": ambiguous_query_count,
                        "query_error_count": query_error_count,
                        "matched_candidate_index": matched_index,
                        "matched_by": (
                            "aggregate_member_refinement"
                            if isinstance(refinement_evidence, dict)
                            else matched_by
                        ),
                        **(
                            {
                                "observation_hypothesis_count": len(
                                    observation_hypotheses
                                ),
                                "redundant_observation_variant_count": max(
                                    0,
                                    len(world_eligible)
                                    - len(observation_hypotheses),
                                ),
                            }
                            if position_anchors
                            else {}
                        ),
                        **(
                            {
                                "verified_position_anchor_world_m": (
                                    round_list(
                                        selected_anchor_matches[0].get(
                                            "world_m"
                                        )
                                    )
                                ),
                                "verified_position_tolerance_m": round(
                                    float(
                                        selected_anchor_matches[0][
                                            "tolerance_m"
                                        ]
                                    ),
                                    6,
                                ),
                                "verified_position_anchor_kind": (
                                    selected_anchor_matches[0].get(
                                        "kind"
                                    )
                                ),
                                "matched_position_error_m": round(
                                    float(
                                        selected_anchor_matches[0][
                                            "distance_m"
                                        ]
                                    ),
                                    6,
                                ),
                            }
                            if selected_anchor_matches
                            else {}
                        ),
                        **(
                            {
                                "refinement_evidence": dict(
                                    refinement_evidence
                                )
                            }
                            if isinstance(refinement_evidence, dict)
                            else {}
                        ),
                    }
                )
                continue

            if track_id in assigned_tracks:
                status = "track_already_claimed"
                warning = f"bound_identity_track_already_claimed:{instance_ref}"
                for _, candidate_index, _ in viable_observations:
                    rejections[candidate_index] = (warning, track_id)
            elif ambiguous_query_count:
                status = "ambiguous_candidates"
                warning = (
                    f"bound_identity_ambiguous_candidates:{instance_ref}"
                )
                for _, candidate_index, _ in viable_observations:
                    rejections[candidate_index] = (
                        warning,
                        track_id,
                    )
            elif query_error_count == len(query_groups):
                status = "query_error"
            else:
                status = "no_compatible_candidate"
            outcomes.append(
                {
                    "instance_ref": instance_ref,
                    "track_id": track_id,
                    "status": status,
                    "candidate_count": len(candidate_indices),
                    "compatible_candidate_count": compatible_candidate_count,
                    "query_group_count": len(query_groups),
                    "compatible_query_count": len(viable_observations),
                    "ambiguous_query_count": ambiguous_query_count,
                    "query_error_count": query_error_count,
                }
            )

        pairs: list[tuple[float, int, str]] = []
        for candidate_index in unbound_indices:
            candidate = candidates[candidate_index]
            overlapping_bound_track = next(
                (
                    track_id
                    for track_id in sorted(reserved_tracks)
                    if self._association_score(self._tracks[track_id], candidate) is not None
                ),
                None,
            )
            if overlapping_bound_track is not None:
                rejections[candidate_index] = (
                    f"candidate_overlaps_identity_bound_track:{overlapping_bound_track}",
                    overlapping_bound_track,
                )
                continue
            for track_id, track in self._tracks.items():
                if track_id in reserved_tracks:
                    continue
                score = self._association_score(track, candidate)
                if score is not None:
                    pairs.append((score, candidate_index, track_id))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        assigned_candidates: set[int] = set(matches)
        for score, candidate_index, track_id in pairs:
            if candidate_index in assigned_candidates or track_id in assigned_tracks:
                continue
            assigned_candidates.add(candidate_index)
            assigned_tracks.add(track_id)
            matches[candidate_index] = track_id
        return matches, rejections, outcomes

    def _identity_relocation_lease_map(
        self,
        leases: list[dict[str, Any]] | None,
        *,
        env_step: int,
    ) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        if isinstance(leases, list):
            for item in leases:
                if not isinstance(item, dict):
                    continue
                instance_ref = str(
                    item.get(
                        "instance_ref",
                        item.get("track_id", ""),
                    )
                    or ""
                ).strip()
                target_world = item.get("target_world_m")
                if (
                    not instance_ref
                    or not is_number_list(
                        target_world,
                        length=3,
                    )
                ):
                    continue
                try:
                    tolerance_m = float(
                        item.get("tolerance_m", 0.04)
                    )
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(tolerance_m):
                    continue
                recovery_anchors: list[dict[str, Any]] = []
                for raw_anchor in item.get(
                    "recovery_anchors",
                    [],
                ) or []:
                    if not isinstance(raw_anchor, dict):
                        continue
                    anchor_world = raw_anchor.get("world_m")
                    if not is_number_list(anchor_world, length=3):
                        continue
                    try:
                        anchor_tolerance = float(
                            raw_anchor.get("tolerance_m", tolerance_m)
                        )
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(anchor_tolerance):
                        continue
                    recovery_anchors.append(
                        {
                            "world_m": [
                                float(value)
                                for value in anchor_world
                            ],
                            "tolerance_m": min(
                                0.08,
                                max(0.015, anchor_tolerance),
                            ),
                            "source": str(
                                raw_anchor.get(
                                    "source",
                                    "runtime_manipulation_history",
                                )
                                or "runtime_manipulation_history"
                            ),
                        }
                    )
                normalized[instance_ref] = {
                    **dict(item),
                    "instance_ref": instance_ref,
                    "target_world_m": [
                        float(value) for value in target_world
                    ],
                    # A relocation lease is an identity constraint, not a
                    # broad nearest-neighbour escape hatch.
                    "tolerance_m": min(
                        0.08,
                        max(0.015, tolerance_m),
                    ),
                    "recovery_anchors": recovery_anchors,
                    "arm": str(
                        item.get("arm", "") or ""
                    ).strip().lower(),
                    "action_modes": sorted(
                        {
                            str(mode).strip().lower()
                            for mode in item.get(
                                "action_modes",
                                [],
                            )
                            or []
                            if str(mode).strip()
                        }
                    ),
                }

        # Runtime events are applied before association in this same update.
        # A clearance authorization issued from the preceding snapshot must
        # never restore its old position anchor after an unverified physical
        # event has made that position uncertain.
        for instance_ref, lease in list(normalized.items()):
            # Only the generic clearance-refresh authorization is stale after
            # a same-update unknown-motion event.  A release-verification
            # lease is different: it is the runtime-owned evidence that lets
            # us locate the just-released object at either the committed
            # target or a recorded recovery anchor, and must remain active
            # precisely while the ordinary position state is uncertain.
            if str(lease.get("source", "") or "") != (
                "blocked_operation_geometry_refresh_after_clearance"
            ):
                continue
            matching_tracks = [
                track
                for track in self._tracks.values()
                if instance_ref in instance_identity_refs(track)
            ]
            if (
                len(matching_tracks) == 1
                and str(
                    matching_tracks[0].get(
                        "position_state",
                        "",
                    )
                    or ""
                ).strip().lower()
                == POSITION_MOTION_UNCERTAIN
            ):
                normalized.pop(instance_ref, None)

        for track_id, track in self._tracks.items():
            if (
                track.get("action_geometry_state")
                != "relocation_pending"
            ):
                continue
            repair_lease = track.get(
                "relocation_identity_repair_lease"
            )
            if not isinstance(repair_lease, dict):
                continue
            refs = instance_identity_refs(track) or {track_id}
            authoritative_external = any(
                ref in normalized
                and str(
                    normalized[ref].get("source", "") or ""
                )
                != "post_release_action_geometry_repair"
                for ref in refs
            )
            if (
                not authoritative_external
                and self._identity_repair_lease_expired(
                    repair_lease,
                    env_step=env_step,
                )
            ):
                self._expire_relocation_geometry_repair(
                    track_id,
                    env_step=env_step,
                    reason=(
                        "post_release_identity_repair_lease_expired"
                    ),
                )
                for ref in refs:
                    if (
                        ref in normalized
                        and str(
                            normalized[ref].get(
                                "source",
                                "",
                            )
                            or ""
                        )
                        == "post_release_action_geometry_repair"
                    ):
                        normalized.pop(ref, None)
                continue
            for ref in refs:
                normalized.setdefault(
                    ref,
                    {
                        **dict(repair_lease),
                        "instance_ref": ref,
                    },
                )
        return normalized

    @staticmethod
    def _unique_relocation_recovery_observation(
        *,
        query_groups: dict[str, list[int]],
        candidates: list[dict[str, Any]],
        relocation_lease: dict[str, Any],
    ) -> dict[str, Any]:
        """Find one physical observation near runtime-owned history anchors.

        This fallback is evaluated only after the committed relocation target
        has not produced a unique match.  It can therefore report that the
        same object is observably elsewhere, while preserving fail-closed
        identity behavior when two physical hypotheses remain plausible.
        """

        anchors = [
            dict(item)
            for item in relocation_lease.get(
                "recovery_anchors",
                [],
            )
            or []
            if isinstance(item, dict)
        ]
        if not anchors:
            return {}

        eligible_indices: set[int] = set()
        candidate_anchor: dict[int, dict[str, Any]] = {}
        representatives: list[tuple[float, float, int, str]] = []
        ambiguous_query_count = 0
        for query_key, group_indices in query_groups.items():
            eligible: list[tuple[float, float, int, str]] = []
            for candidate_index in group_indices:
                world = candidates[candidate_index].get("_world_m")
                anchor_matches: list[
                    tuple[float, float, dict[str, Any]]
                ] = []
                for anchor in anchors:
                    distance_m = world_distance_m(
                        world,
                        anchor.get("world_m"),
                    )
                    try:
                        tolerance_m = float(
                            anchor.get("tolerance_m")
                        )
                    except (TypeError, ValueError):
                        continue
                    if (
                        distance_m is None
                        or not math.isfinite(tolerance_m)
                        or tolerance_m <= 0.0
                        or distance_m > tolerance_m
                    ):
                        continue
                    anchor_matches.append(
                        (
                            distance_m / tolerance_m,
                            distance_m,
                            anchor,
                        )
                    )
                if not anchor_matches:
                    continue
                normalized_distance, distance_m, anchor = min(
                    anchor_matches,
                    key=lambda item: (item[0], item[1]),
                )
                try:
                    score = float(
                        candidates[candidate_index].get("score", 0.0)
                        or 0.0
                    )
                except (TypeError, ValueError):
                    score = 0.0
                eligible_indices.add(candidate_index)
                candidate_anchor[candidate_index] = {
                    "distance_m": distance_m,
                    "tolerance_m": float(
                        anchor.get("tolerance_m")
                    ),
                    "source": str(
                        anchor.get(
                            "source",
                            "runtime_manipulation_history",
                        )
                        or "runtime_manipulation_history"
                    ),
                    "world_m": list(anchor.get("world_m") or []),
                }
                eligible.append(
                    (
                        normalized_distance,
                        -score,
                        candidate_index,
                        query_key,
                    )
                )
            hypotheses = cluster_bound_identity_observation_variants(
                sorted(eligible),
                candidates=candidates,
            )
            if len(hypotheses) == 1:
                representatives.append(hypotheses[0][0])
            elif len(hypotheses) > 1:
                ambiguous_query_count += 1

        cross_query_hypotheses = (
            cluster_bound_identity_observation_variants(
                sorted(representatives),
                candidates=candidates,
            )
            if representatives
            else []
        )
        if (
            ambiguous_query_count > 0
            or len(cross_query_hypotheses) > 1
        ):
            return {
                "ambiguous": True,
                "eligible_indices": eligible_indices,
                "ambiguous_query_count": ambiguous_query_count,
                "observation_hypothesis_count": len(
                    cross_query_hypotheses
                ),
            }
        if len(cross_query_hypotheses) != 1:
            return {
                "ambiguous": False,
                "eligible_indices": eligible_indices,
                "ambiguous_query_count": 0,
                "observation_hypothesis_count": 0,
            }
        matched_index = cross_query_hypotheses[0][0][2]
        return {
            "ambiguous": False,
            "matched_index": matched_index,
            "eligible_indices": eligible_indices,
            "ambiguous_query_count": 0,
            "observation_hypothesis_count": 1,
            "anchor": candidate_anchor[matched_index],
        }

    @staticmethod
    def _identity_repair_lease_expired(
        lease: dict[str, Any],
        *,
        env_step: int,
    ) -> bool:
        try:
            expires_step = int(
                lease.get("expires_step", env_step)
            )
            attempts = int(
                lease.get("observation_attempts", 0)
            )
            max_attempts = int(
                lease.get("max_observation_attempts", 1)
            )
        except (TypeError, ValueError):
            return True
        return bool(
            int(env_step) > expires_step
            or attempts >= max(1, max_attempts)
        )

    def _record_identity_repair_lease_attempt(
        self,
        track_id: str,
    ) -> None:
        track = self._tracks[track_id]
        lease = track.get(
            "relocation_identity_repair_lease"
        )
        if not isinstance(lease, dict):
            return
        updated = dict(lease)
        updated["observation_attempts"] = int(
            updated.get("observation_attempts", 0)
            or 0
        ) + 1
        track["relocation_identity_repair_lease"] = updated

    def _expire_relocation_geometry_repair(
        self,
        track_id: str,
        *,
        env_step: int,
        reason: str,
    ) -> None:
        track = self._tracks[track_id]
        lease = track.get(
            "relocation_identity_repair_lease"
        )
        expiration = {
            "env_step": int(env_step),
            "reason": str(reason or "identity_repair_expired"),
        }
        if isinstance(lease, dict):
            expiration.update(
                {
                    "target_world_m": round_list(
                        lease.get("target_world_m")
                    ),
                    "tolerance_m": round_float(
                        lease.get("tolerance_m")
                    ),
                    "started_step": lease.get(
                        "started_step"
                    ),
                    "observation_attempts": lease.get(
                        "observation_attempts"
                    ),
                }
            )
        for key in (
            "top_surface_world_m",
            "approach_world_m",
            "approach_quat_wxyz",
            "grasp_world_m",
            "grasp_quat_wxyz",
            "contact_world_m",
            "contact_quat_wxyz",
            "stable_top_surface_world_m",
            "stable_approach_world_m",
            "stable_grasp_world_m",
            "stable_contact_world_m",
        ):
            track[key] = None
        track["operation_pose_candidates"] = []
        track["operation_pose_candidate_provenance"] = {}
        track["status"] = "tracked"
        track["stability"] = "identity_repair_expired"
        track["action_geometry_state"] = (
            "identity_repair_expired"
        )
        track["identity_repair_expiration"] = expiration
        track["quality_warnings"] = dedupe_preserving_order(
            [
                *list(track.get("quality_warnings", []) or []),
                (
                    "relocation_action_geometry_repair_expired:"
                    f"{expiration['reason']}"
                ),
            ]
        )
        for key in (
            "relocation_identity_repair_lease",
            "relocation_action_geometry_repair_samples",
            "relocation_action_geometry_scope",
            "relocation_action_geometry_retained_provenance",
        ):
            track.pop(key, None)

    def _verified_position_anchor_and_tolerance(
        self,
        track: dict[str, Any],
    ) -> tuple[list[float] | None, float | None]:
        anchors = self._verified_position_anchors(track)
        if not anchors:
            return None, None
        primary = anchors[0]
        return list(primary["world_m"]), float(
            primary["tolerance_m"]
        )

    def _verified_position_anchors(
        self,
        track: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return verified_position_anchors(
            track,
            default_tolerance_m=STRONG_GEOMETRY_DISTANCE_M,
            max_tolerance_m=self.stable_distance_m,
        )

    def _association_score(self, track: dict[str, Any], candidate: dict[str, Any]) -> float | None:
        if (
            track.get("action_geometry_state")
            == "identity_repair_expired"
        ):
            return None
        bound_instance_ref = identity_bound_candidate_ref(candidate)
        if bound_instance_ref and bound_instance_ref not in instance_identity_refs(track):
            return None
        track_source_key = str(track.get("source_instance_key", "") or "")
        candidate_source_key = str(candidate.get("_source_instance_key", "") or "")
        if track_source_key and candidate_source_key:
            if track_source_key != candidate_source_key:
                return None
            return 1000.0
        track_aliases = track_identity_tokens(track)
        candidate_aliases = candidate_identity_tokens(candidate)
        alias_overlap = bool(track_aliases & candidate_aliases)
        verified_identity_tokens = {
            normalize_class(item)
            for item in track.get(
                "verified_identity_tokens",
                [],
            )
            or []
            if normalize_class(item)
        }
        if (
            verified_identity_tokens
            and not identity_tokens_semantically_overlap(
                verified_identity_tokens,
                candidate_aliases,
            )
        ):
            # Interaction-verified identity (for example an object that
            # actually produced a press/open/insert effect) is stronger than a
            # later single semantic mask.  Keep the observation available for
            # discovery, but never let it silently relabel this track.
            return None
        world_distance = world_distance_m(candidate.get("_world_m"), track.get("stable_world_m", track.get("world_m")))
        bound_anchor_matches = (
            eligible_anchor_matches(
                candidate.get("_world_m"),
                self._verified_position_anchors(track),
            )
            if bound_instance_ref
            else []
        )
        image_geometry_allowed = same_or_unknown_camera(candidate.get("camera"), track.get("camera"))
        iou = bbox_iou(candidate.get("_bbox_xyxy"), track.get("bbox_xyxy")) if image_geometry_allowed else None
        centroid_distance = point_distance_2d(candidate.get("_centroid_px"), track.get("centroid_px")) if image_geometry_allowed else None
        refinement_evidence = (
            bound_identity_refinement_evidence(track, candidate)
            if bound_instance_ref and not alias_overlap
            else None
        )
        if refinement_evidence is not None:
            candidate["_identity_refinement_evidence"] = refinement_evidence
        strong_geometry = (
            (world_distance is not None and world_distance <= STRONG_GEOMETRY_DISTANCE_M)
            or (iou is not None and iou >= STRONG_GEOMETRY_IOU)
            or refinement_evidence is not None
            or bool(bound_anchor_matches)
        )
        if not alias_overlap and explicit_role_conflict(track.get("query_role"), candidate.get("query_role")):
            return None
        if not alias_overlap and not strong_geometry:
            return None

        geometry_supported = strong_geometry
        if world_distance is not None and world_distance <= max(self.stable_distance_m, STRONG_GEOMETRY_DISTANCE_M):
            geometry_supported = True
        if iou is not None and iou >= 0.15:
            geometry_supported = True
        if centroid_distance is not None and centroid_distance <= CENTROID_MATCH_DISTANCE_PX:
            geometry_supported = True
        if not geometry_supported:
            return None

        score = 0.0
        if alias_overlap:
            score += 2.0
        if bound_anchor_matches:
            score += 3.0 / (
                1.0
                + float(bound_anchor_matches[0]["distance_m"])
                / max(
                    float(
                        bound_anchor_matches[0]["tolerance_m"]
                    ),
                    1e-9,
                )
            )
        if refinement_evidence is not None:
            score += 3.0 + float(
                refinement_evidence.get("candidate_containment", 0.0)
            )
        if world_distance is not None:
            if world_distance > max(self.stable_distance_m * 1.5, 0.12) and not strong_geometry:
                return None
            score += 4.0 * max(0.0, 1.0 - world_distance / max(self.stable_distance_m, 1e-6))
        if iou is not None:
            score += 2.0 * iou
        if centroid_distance is not None:
            score += max(0.0, 1.0 - centroid_distance / CENTROID_MATCH_DISTANCE_PX)
        if normalize_role(track.get("query_role")) == normalize_role(candidate.get("query_role")):
            score += 0.25
        return score if score >= 0.5 else None

    def _role_conflict_with_memory(self, candidate: dict[str, Any]) -> str:
        for track_id, track in self._tracks.items():
            if not isinstance(track, dict):
                continue
            if track.get("status") not in {"visible", "tracked"}:
                continue
            verified_tokens = {
                normalize_class(item)
                for item in track.get(
                    "verified_identity_tokens",
                    [],
                )
                or []
                if normalize_class(item)
            }
            candidate_tokens = candidate_identity_tokens(
                candidate
            )
            anchor_matches = eligible_anchor_matches(
                candidate.get("_world_m"),
                self._verified_position_anchors(track),
            )
            verified_distance = (
                float(anchor_matches[0]["distance_m"])
                if anchor_matches
                else None
            )
            tolerance = (
                float(anchor_matches[0]["tolerance_m"])
                if anchor_matches
                else None
            )
            if (
                verified_tokens
                and not identity_tokens_semantically_overlap(
                    verified_tokens,
                    candidate_tokens,
                )
                and verified_distance is not None
                and tolerance is not None
                and verified_distance <= tolerance
            ):
                return (
                    "verified_identity_conflict:"
                    f"{track_id}"
                )
            if not explicit_role_conflict(track.get("query_role"), candidate.get("query_role")):
                continue
            if track_identity_tokens(track) & candidate_identity_tokens(candidate):
                continue
            world_distance = world_distance_m(candidate.get("_world_m"), track.get("stable_world_m", track.get("world_m")))
            image_geometry_allowed = same_or_unknown_camera(candidate.get("camera"), track.get("camera"))
            iou = bbox_iou(candidate.get("_bbox_xyxy"), track.get("bbox_xyxy")) if image_geometry_allowed else None
            if (world_distance is not None and world_distance <= STRONG_GEOMETRY_DISTANCE_M) or (
                iou is not None and iou >= STRONG_GEOMETRY_IOU
            ):
                return f"role_conflict_with_current_{normalize_role(track.get('query_role'))}:{track.get('track_id')}"
        return ""

    def _spawn_track(self, candidate: dict[str, Any], *, env_step: int) -> str:
        track_id = f"track_{self._next_track_index:04d}"
        self._next_track_index += 1
        self._tracks[track_id] = {
            "track_id": track_id,
            "class": candidate.get("_class_name", "object"),
            "class_aliases": list(candidate.get("_class_aliases", [])),
            "first_seen_step": int(env_step),
            "first_observed_world_m": round_list(candidate.get("_world_m")),
            "first_observed_top_surface_world_m": round_list(candidate.get("_top_surface_world_m")),
            "visible_count": 0,
            "history": [],
        }
        self._update_track(track_id, candidate, env_step=env_step, is_new=True)
        return track_id

    def _temporal_action_geometry_warning(
        self,
        track_id: str,
        candidate: dict[str, Any],
        *,
        include_observation_reference: bool = False,
        action_geometry_scope: dict[str, Any] | None = None,
    ) -> str:
        if self.temporal_action_geometry_distance_m <= 0.0:
            return ""
        track = self._tracks[track_id]
        candidate_offsets = self._candidate_action_geometry_offsets(
            candidate
        )
        candidate_offsets = self._offsets_for_action_geometry_scope(
            candidate_offsets,
            action_geometry_scope,
        )
        if (
            track.get("action_geometry_state")
            == "relocation_pending"
        ):
            reference_offsets = track.get(
                "relocation_action_geometry_reference_offsets"
            )
        else:
            reference_offsets = self._track_action_geometry_offsets(
                track
            )
        observation_reference = track.get(
            "last_verified_observation_geometry_reference"
        )
        relocation_pending = (
            track.get("action_geometry_state")
            == "relocation_pending"
        )
        if (
            include_observation_reference
            and not relocation_pending
            and not reference_offsets
            and isinstance(observation_reference, dict)
        ):
            reference_offsets = observation_reference.get(
                "offsets"
            )
        if not isinstance(reference_offsets, dict):
            return ""
        reference_offsets = self._offsets_for_action_geometry_scope(
            reference_offsets,
            action_geometry_scope,
        )
        warning = self._action_geometry_warning_between_offsets(
            candidate_offsets,
            reference_offsets,
        )
        if warning:
            return warning
        if relocation_pending:
            # Pending repair is compared against its own first clean sample
            # in _confirm_relocated_action_geometry, never against the
            # invalidated pre-manipulation executable set.
            return ""
        reference_candidates = track.get(
            "operation_pose_candidates"
        )
        if (
            include_observation_reference
            and not relocation_pending
            and not reference_candidates
            and isinstance(observation_reference, dict)
        ):
            reference_candidates = observation_reference.get(
                "operation_pose_candidates"
            )
        reference_candidates = self._operation_candidates_for_scope(
            reference_candidates,
            action_geometry_scope,
        )
        candidate_operation_candidates = (
            self._operation_candidates_for_scope(
                candidate.get("_operation_pose_candidates"),
                action_geometry_scope,
            )
        )
        if self.enable_executable_candidate_temporal_consistency:
            consistency = executable_candidate_sets_consistent(
                reference_candidates,
                candidate_operation_candidates,
                max_translation_m=(
                    OPERATION_GEOMETRY_TRANSLATION_CHANGE_M
                ),
                max_rotation_rad=(
                    OPERATION_GEOMETRY_ROTATION_CHANGE_RAD
                ),
            )
            if consistency is False:
                return (
                    "temporal_action_geometry_inconsistent:"
                    "executable_candidate_pose_changed"
                )
        return ""

    @staticmethod
    def _action_geometry_refresh_scope(
        value: Any,
    ) -> dict[str, Any] | None:
        source = value if isinstance(value, dict) else {}
        arm = str(source.get("arm", "") or "").strip().lower()
        modes = sorted(
            {
                str(mode).strip().lower()
                for mode in source.get("action_modes", []) or []
                if str(mode).strip()
            }
        )
        if arm not in {"left", "right"} or not modes:
            return None
        return {"arm": arm, "action_modes": modes}

    @staticmethod
    def _operation_candidates_for_scope(
        candidates: Any,
        scope: dict[str, Any] | None,
        *,
        invert: bool = False,
    ) -> list[dict[str, Any]]:
        items = [
            dict(item)
            for item in candidates or []
            if isinstance(item, dict)
        ]
        if not isinstance(scope, dict):
            return items
        arm = str(scope.get("arm", "") or "").strip().lower()
        modes = {
            str(mode).strip().lower()
            for mode in scope.get("action_modes", []) or []
            if str(mode).strip()
        }
        selected: list[dict[str, Any]] = []
        for item in items:
            in_scope = bool(
                str(item.get("arm", "") or "").strip().lower()
                == arm
                and str(
                    item.get("action_mode", "") or ""
                ).strip().lower()
                in modes
            )
            if in_scope != invert:
                selected.append(item)
        return selected

    @staticmethod
    def _offsets_for_action_geometry_scope(
        offsets: Any,
        scope: dict[str, Any] | None,
    ) -> dict[str, list[float]]:
        parsed = dict(offsets) if isinstance(offsets, dict) else {}
        if not isinstance(scope, dict):
            return parsed
        modes = {
            str(mode).strip().lower()
            for mode in scope.get("action_modes", []) or []
            if str(mode).strip()
        }
        names: set[str] = set()
        if "grasp" in modes:
            names.update({"top_surface", "approach", "grasp"})
        if "contact" in modes:
            names.update({"top_surface", "approach", "contact"})
        if "place" in modes:
            names.add("top_surface")
        return {
            name: value
            for name, value in parsed.items()
            if name in names
        }

    def _temporally_consistent_observation_indices(
        self,
        track_id: str,
        candidate_indices: list[int],
        *,
        candidates: list[dict[str, Any]],
    ) -> tuple[set[int], dict[int, str]]:
        """Filter view variants only when at least one has clean geometry.

        A single candidate must continue through the ordinary update path so
        existing quarantine/geometry-inconsistent bookkeeping remains intact.
        Likewise, if every view is inconsistent, retain the original choices
        and let the normal fail-closed path reject the selected representative.
        """

        unique_indices = list(dict.fromkeys(candidate_indices))
        if len(unique_indices) <= 1:
            return set(unique_indices), {}
        warnings = {
            candidate_index: warning
            for candidate_index in unique_indices
            if (
                warning := self._temporal_action_geometry_warning(
                    track_id,
                    candidates[candidate_index],
                    include_observation_reference=True,
                )
            )
        }
        clean_indices = {
            candidate_index
            for candidate_index in unique_indices
            if candidate_index not in warnings
        }
        if not clean_indices:
            return set(unique_indices), {}
        return clean_indices, warnings


    @staticmethod
    def _candidate_action_geometry_offsets(
        candidate: dict[str, Any],
    ) -> dict[str, list[float]]:
        return candidate_action_geometry_offsets(candidate)

    @staticmethod
    def _track_action_geometry_offsets(
        track: dict[str, Any],
    ) -> dict[str, list[float]]:
        origin = track.get(
            "stable_world_m",
            track.get("world_m"),
        )
        if not is_number_list(origin, length=3):
            return {}
        offsets: dict[str, list[float]] = {}
        for name, track_key in (
            ("top_surface", "stable_top_surface_world_m"),
            ("approach", "stable_approach_world_m"),
            ("grasp", "stable_grasp_world_m"),
            ("contact", "stable_contact_world_m"),
        ):
            offset = relative_xyz(
                track.get(track_key),
                origin,
            )
            if offset is not None:
                offsets[name] = offset
        return offsets

    @staticmethod
    def _sample_action_geometry_offsets(
        sample: dict[str, Any],
    ) -> dict[str, list[float]]:
        origin = sample.get("world_m")
        if not is_number_list(origin, length=3):
            return {}
        offsets: dict[str, list[float]] = {}
        for name, sample_key in (
            ("top_surface", "top_surface_world_m"),
            ("approach", "approach_world_m"),
            ("grasp", "grasp_world_m"),
            ("contact", "contact_world_m"),
        ):
            offset = relative_xyz(
                sample.get(sample_key),
                origin,
            )
            if offset is not None:
                offsets[name] = offset
        return offsets

    def _action_geometry_warning_between_offsets(
        self,
        candidate_offsets: dict[str, list[float]],
        reference_offsets: dict[str, list[float]],
    ) -> str:
        jumps: list[tuple[str, float]] = []
        for name in (
            "top_surface",
            "approach",
            "grasp",
            "contact",
        ):
            jump = world_distance_m(
                candidate_offsets.get(name),
                reference_offsets.get(name),
            )
            if jump is not None:
                jumps.append((name, jump))
        if not jumps:
            return ""
        name, max_jump = max(
            jumps,
            key=lambda item: item[1],
        )
        if (
            max_jump
            <= self.temporal_action_geometry_distance_m
        ):
            return ""
        return (
            "temporal_action_geometry_inconsistent:"
            f"{name}_relative_jump={max_jump:.3f}m>"
            f"{self.temporal_action_geometry_distance_m:.3f}m"
        )

    @staticmethod
    def _candidate_observation_token(
        candidate: dict[str, Any],
        *,
        env_step: int,
    ) -> str:
        try:
            capture_id = int(
                candidate.get("_observation_capture_id")
            )
        except (TypeError, ValueError):
            capture_id = -1
        if capture_id >= 0:
            return f"capture:{capture_id}"
        mask_path = str(
            candidate.get("mask_path", "") or ""
        ).strip()
        if mask_path:
            return (
                f"{candidate.get('camera', '')}:"
                f"{mask_path}"
            )
        return repr(
            (
                int(env_step),
                str(candidate.get("camera", "") or ""),
                round_list(
                    candidate.get("_bbox_xyxy"),
                    digits=2,
                ),
                round_list(
                    candidate.get("_centroid_px"),
                    digits=2,
                ),
            )
        )

    @staticmethod
    def _position_only_relocation_candidate(
        candidate: dict[str, Any],
        *,
        warning: str,
        retained_track: dict[str, Any] | None = None,
        action_geometry_scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        position_only = dict(candidate)
        if (
            isinstance(retained_track, dict)
            and isinstance(action_geometry_scope, dict)
        ):
            # A refresh lease invalidates only the arm/mode that exhausted
            # its runtime candidates.  Shared scalar geometry and candidates
            # for the other arm/modes retain their prior provenance while the
            # scoped replacement waits for independent confirmation.
            for candidate_key, track_key in (
                ("_top_surface_world_m", "top_surface_world_m"),
                ("_approach_world_m", "approach_world_m"),
                ("_approach_quat_wxyz", "approach_quat_wxyz"),
                ("_grasp_world_m", "grasp_world_m"),
                ("_grasp_quat_wxyz", "grasp_quat_wxyz"),
                ("_contact_world_m", "contact_world_m"),
                ("_contact_quat_wxyz", "contact_quat_wxyz"),
            ):
                position_only[candidate_key] = retained_track.get(
                    track_key
                )
            position_only["_operation_pose_candidates"] = (
                TemporalInstanceTracker._operation_candidates_for_scope(
                    retained_track.get(
                        "operation_pose_candidates",
                        [],
                    ),
                    action_geometry_scope,
                    invert=True,
                )
            )
        else:
            for key in (
                "_top_surface_world_m",
                "_approach_world_m",
                "_approach_quat_wxyz",
                "_grasp_world_m",
                "_grasp_quat_wxyz",
                "_contact_world_m",
                "_contact_quat_wxyz",
            ):
                position_only[key] = None
            position_only["_operation_pose_candidates"] = []
        quality = dict(position_only.get("quality") or {})
        prior_warnings = [
            str(item)
            for item in list(
                quality.get("warnings", []) or []
            )
            if not str(item).startswith(
                "relocation_action_geometry_"
            )
        ]
        quality["warnings"] = dedupe_preserving_order(
            [*prior_warnings, warning]
        )
        position_only["quality"] = quality
        return position_only

    def _begin_relocation_geometry_quarantine(
        self,
        track_id: str,
        candidate: dict[str, Any],
        *,
        env_step: int,
        warning: str,
        force_observed_geometry_rebuild: bool = False,
    ) -> dict[str, Any]:
        track = self._tracks[track_id]
        relocation_evidence = candidate.get(
            "_identity_relocation_evidence"
        )
        rebuild_from_observations = bool(
            force_observed_geometry_rebuild
            or (
                isinstance(relocation_evidence, dict)
                and relocation_evidence.get(
                    "rebuild_action_geometry_from_observations"
                )
                is True
            )
        )
        action_geometry_scope = (
            self._action_geometry_refresh_scope(
                relocation_evidence
            )
            if rebuild_from_observations
            else None
        )
        retained_candidate_provenance = {
            str(candidate_id): dict(provenance)
            for candidate_id, provenance in dict(
                track.get(
                    "operation_pose_candidate_provenance",
                    {},
                )
                or {}
            ).items()
            if isinstance(provenance, dict)
            and any(
                str(item.get("candidate_id", "") or "").strip()
                == str(candidate_id)
                for item in self._operation_candidates_for_scope(
                    track.get("operation_pose_candidates", []),
                    action_geometry_scope,
                    invert=True,
                )
            )
        }
        existing_reference_offsets = track.get(
            "relocation_action_geometry_reference_offsets"
        )
        reference_offsets = (
            {}
            if rebuild_from_observations
            else dict(existing_reference_offsets)
            if isinstance(existing_reference_offsets, dict)
            else self._track_action_geometry_offsets(track)
        )
        identity_repair_lease: dict[str, Any] | None = None
        if isinstance(relocation_evidence, dict):
            target_world = relocation_evidence.get(
                "target_world_m"
            )
            try:
                tolerance_m = float(
                    relocation_evidence.get(
                        "tolerance_m",
                        0.04,
                    )
                )
            except (TypeError, ValueError):
                tolerance_m = float("nan")
            if (
                is_number_list(target_world, length=3)
                and math.isfinite(tolerance_m)
            ):
                identity_repair_lease = {
                    "state": "active",
                    "instance_ref": track_id,
                    "target_world_m": [
                        float(value)
                        for value in target_world
                    ],
                    "tolerance_m": min(
                        0.08,
                        max(0.015, tolerance_m),
                    ),
                    "started_step": int(env_step),
                    "expires_step": (
                        int(env_step)
                        + self.max_missing_steps
                    ),
                    "observation_attempts": 0,
                    "max_observation_attempts": (
                        self.max_missing_steps
                    ),
                    "source": (
                        "post_release_action_geometry_repair"
                    ),
                    "origin_source": str(
                        relocation_evidence.get(
                            "source",
                            "",
                        )
                        or ""
                    ),
                    "arm": str(
                        relocation_evidence.get("arm", "")
                        or ""
                    ),
                    "action_modes": list(
                        action_geometry_scope.get(
                            "action_modes",
                            [],
                        )
                    )
                    if isinstance(action_geometry_scope, dict)
                    else [],
                    "release_step": (
                        relocation_evidence.get(
                            "release_step"
                        )
                    ),
                }
        quarantine_warning = (
            "relocation_action_geometry_quarantined:"
            f"{warning}"
        )
        position_only = (
            self._position_only_relocation_candidate(
                candidate,
                warning=quarantine_warning,
                retained_track=track,
                action_geometry_scope=action_geometry_scope,
            )
        )
        self._update_track(
            track_id,
            position_only,
            env_step=env_step,
        )
        repair_samples: list[dict[str, Any]] = []
        if (
            rebuild_from_observations
            and self._candidate_action_geometry_offsets(candidate)
        ):
            sample = compact_candidate_for_track(
                candidate,
                env_step=env_step,
            )
            sample["observation_token"] = (
                self._candidate_observation_token(
                    candidate,
                    env_step=env_step,
                )
            )
            repair_samples.append(sample)
        track = self._tracks[track_id]
        if isinstance(action_geometry_scope, dict):
            track["operation_pose_candidate_provenance"] = dict(
                retained_candidate_provenance
            )
        track.update(
            {
                "action_geometry_state": (
                    "relocation_pending"
                ),
                "relocation_action_geometry_reference_offsets": (
                    reference_offsets
                ),
                "relocation_action_geometry_initial_warning": (
                    warning
                ),
                "relocation_action_geometry_last_warning": (
                    warning
                ),
                "relocation_action_geometry_repair_samples": (
                    repair_samples
                ),
                "relocation_action_geometry_scope": (
                    dict(action_geometry_scope)
                    if isinstance(action_geometry_scope, dict)
                    else None
                ),
                "relocation_action_geometry_retained_provenance": (
                    retained_candidate_provenance
                ),
                "stability": (
                    "relocation_geometry_pending"
                ),
            }
        )
        if identity_repair_lease is not None:
            track["relocation_identity_repair_lease"] = (
                identity_repair_lease
            )
        track.pop("identity_repair_expiration", None)
        return position_only

    def _retain_relocation_geometry_quarantine(
        self,
        track_id: str,
        candidate: dict[str, Any],
        *,
        env_step: int,
        warning: str,
    ) -> dict[str, Any]:
        quarantine_warning = (
            "relocation_action_geometry_quarantined:"
            f"{warning}"
        )
        track = self._tracks[track_id]
        action_geometry_scope = track.get(
            "relocation_action_geometry_scope"
        )
        retained_provenance = dict(
            track.get(
                "relocation_action_geometry_retained_provenance",
                {},
            )
            or {}
        )
        position_only = (
            self._position_only_relocation_candidate(
                candidate,
                warning=quarantine_warning,
                retained_track=track,
                action_geometry_scope=(
                    action_geometry_scope
                    if isinstance(action_geometry_scope, dict)
                    else None
                ),
            )
        )
        self._update_track(
            track_id,
            position_only,
            env_step=env_step,
        )
        track = self._tracks[track_id]
        if isinstance(action_geometry_scope, dict):
            track["operation_pose_candidate_provenance"] = (
                retained_provenance
            )
        track["action_geometry_state"] = (
            "relocation_pending"
        )
        track[
            "relocation_action_geometry_last_warning"
        ] = warning
        track[
            "relocation_action_geometry_repair_samples"
        ] = []
        track["stability"] = (
            "relocation_geometry_pending"
        )
        return position_only

    def _confirm_relocated_action_geometry(
        self,
        track_id: str,
        candidate: dict[str, Any],
        *,
        env_step: int,
    ) -> dict[str, Any]:
        track = self._tracks[track_id]
        action_geometry_scope = track.get(
            "relocation_action_geometry_scope"
        )
        if not isinstance(action_geometry_scope, dict):
            action_geometry_scope = None
        retained_provenance = dict(
            track.get(
                "relocation_action_geometry_retained_provenance",
                {},
            )
            or {}
        )
        candidate_offsets = (
            self._offsets_for_action_geometry_scope(
                self._candidate_action_geometry_offsets(
                    candidate
                ),
                action_geometry_scope,
            )
        )
        if not candidate_offsets:
            return self._retain_relocation_geometry_quarantine(
                track_id,
                candidate,
                env_step=env_step,
                warning=(
                    "temporal_action_geometry_inconsistent:"
                    "no_comparable_action_geometry"
                ),
            )

        token = self._candidate_observation_token(
            candidate,
            env_step=env_step,
        )
        repair_samples = [
            dict(item)
            for item in track.get(
                "relocation_action_geometry_repair_samples",
                [],
            )
            if isinstance(item, dict)
        ]
        if (
            repair_samples
            and str(
                repair_samples[-1].get(
                    "observation_token",
                    "",
                )
                or ""
            )
            == token
        ):
            pending_warning = (
                "relocation_action_geometry_confirmation_pending:"
                f"{len(repair_samples)}/"
                f"{RELOCATION_ACTION_GEOMETRY_CONFIRMATIONS}"
            )
            position_only = (
                self._position_only_relocation_candidate(
                    candidate,
                    warning=pending_warning,
                    retained_track=track,
                    action_geometry_scope=action_geometry_scope,
                )
            )
            self._update_track(
                track_id,
                position_only,
                env_step=env_step,
            )
            track = self._tracks[track_id]
            if action_geometry_scope is not None:
                track[
                    "operation_pose_candidate_provenance"
                ] = retained_provenance
            track[
                "relocation_action_geometry_repair_samples"
            ] = repair_samples
            track["action_geometry_state"] = (
                "relocation_pending"
            )
            track["stability"] = (
                "relocation_geometry_pending"
            )
            return position_only

        sample = compact_candidate_for_track(
            candidate,
            env_step=env_step,
        )
        sample["observation_token"] = token
        if repair_samples:
            prior_offsets = (
                self._offsets_for_action_geometry_scope(
                    self._sample_action_geometry_offsets(
                        repair_samples[-1]
                    ),
                    action_geometry_scope,
                )
            )
            confirmation_warning = (
                self._action_geometry_warning_between_offsets(
                    candidate_offsets,
                    prior_offsets,
                )
            )
            executable_geometry_consistent = None
            if self.enable_executable_candidate_temporal_consistency:
                executable_geometry_consistent = (
                    executable_candidate_sets_consistent(
                        self._operation_candidates_for_scope(
                            repair_samples[-1].get(
                                "operation_pose_candidates"
                            ),
                            action_geometry_scope,
                        ),
                        self._operation_candidates_for_scope(
                            candidate.get(
                                "_operation_pose_candidates"
                            ),
                            action_geometry_scope,
                        ),
                        max_translation_m=(
                            OPERATION_GEOMETRY_TRANSLATION_CHANGE_M
                        ),
                        max_rotation_rad=(
                            OPERATION_GEOMETRY_ROTATION_CHANGE_RAD
                        ),
                    )
                )
            if (
                confirmation_warning
                or executable_geometry_consistent is False
            ):
                repair_samples = [sample]
            else:
                repair_samples.append(sample)
        else:
            repair_samples = [sample]

        if (
            len(repair_samples)
            < RELOCATION_ACTION_GEOMETRY_CONFIRMATIONS
        ):
            pending_warning = (
                "relocation_action_geometry_confirmation_pending:"
                f"{len(repair_samples)}/"
                f"{RELOCATION_ACTION_GEOMETRY_CONFIRMATIONS}"
            )
            position_only = (
                self._position_only_relocation_candidate(
                    candidate,
                    warning=pending_warning,
                    retained_track=track,
                    action_geometry_scope=action_geometry_scope,
                )
            )
            self._update_track(
                track_id,
                position_only,
                env_step=env_step,
            )
            track = self._tracks[track_id]
            if action_geometry_scope is not None:
                track[
                    "operation_pose_candidate_provenance"
                ] = retained_provenance
            track[
                "relocation_action_geometry_repair_samples"
            ] = repair_samples
            track["action_geometry_state"] = (
                "relocation_pending"
            )
            track["stability"] = (
                "relocation_geometry_pending"
            )
            return position_only

        initial_warning = str(
            track.get(
                "relocation_action_geometry_initial_warning",
                "",
            )
            or ""
        )
        first_sample = repair_samples[-2]
        repair_history = [
            dict(item)
            for item in track.get(
                "action_geometry_repair_history",
                [],
            )
            if isinstance(item, dict)
        ]
        repair_history.append(
            {
                "env_step": int(env_step),
                "source": (
                    "post_relocation_clean_observations"
                ),
                "confirmation_count": (
                    RELOCATION_ACTION_GEOMETRY_CONFIRMATIONS
                ),
                "observation_tokens": [
                    str(
                        item.get(
                            "observation_token",
                            "",
                        )
                        or ""
                    )
                    for item in repair_samples[-2:]
                ],
                "initial_warning": initial_warning,
            }
        )
        track["history"] = [dict(first_sample)]
        verified_candidate = dict(candidate)
        if action_geometry_scope is not None:
            retained_candidates = (
                self._operation_candidates_for_scope(
                    track.get("operation_pose_candidates", []),
                    action_geometry_scope,
                    invert=True,
                )
            )
            refreshed_candidates = (
                self._operation_candidates_for_scope(
                    candidate.get(
                        "_operation_pose_candidates",
                        [],
                    ),
                    action_geometry_scope,
                )
            )
            verified_candidate[
                "_operation_pose_candidates"
            ] = [*retained_candidates, *refreshed_candidates]
        self._update_track(
            track_id,
            verified_candidate,
            env_step=env_step,
        )
        track = self._tracks[track_id]
        if action_geometry_scope is not None:
            current_provenance = dict(
                track.get(
                    "operation_pose_candidate_provenance",
                    {},
                )
                or {}
            )
            current_provenance.update(retained_provenance)
            track["operation_pose_candidate_provenance"] = (
                current_provenance
            )
        track["action_geometry_state"] = "verified"
        track["action_geometry_repair_history"] = (
            repair_history[-4:]
        )
        for key in (
            "relocation_action_geometry_reference_offsets",
            "relocation_action_geometry_initial_warning",
            "relocation_action_geometry_last_warning",
            "relocation_action_geometry_repair_samples",
            "relocation_action_geometry_scope",
            "relocation_action_geometry_retained_provenance",
            "relocation_identity_repair_lease",
        ):
            track.pop(key, None)
        track.pop("identity_repair_expiration", None)
        return verified_candidate

    def _mark_geometry_inconsistent(self, track_id: str, *, env_step: int, warning: str) -> None:
        track = self._tracks[track_id]
        missing_steps = max(1, int(env_step) - int(track.get("last_seen_step", env_step)))
        if missing_steps > self.max_missing_steps:
            del self._tracks[track_id]
            return
        track["status"] = "tracked"
        track["stability"] = "geometry_inconsistent_current_frame"
        track["missing_steps"] = missing_steps
        prior_warnings = [
            item
            for item in list(track.get("quality_warnings", []) or [])
            if not str(item).startswith("temporal_action_geometry_inconsistent:")
        ]
        track["quality_warnings"] = dedupe_preserving_order(
            [*prior_warnings, warning]
        )
        if track.get("position_state") in {
            POSITION_CURRENT_VERIFIED,
            POSITION_MEMORY_VALID,
        }:
            track["position_state"] = POSITION_MEMORY_VALID
            track["position_source"] = (
                "retained_verified_memory"
            )
            track["position_state_reason"] = (
                "current candidate was rejected and did not overwrite "
                "the last verified position"
            )
            track["position_state_updated_step"] = int(env_step)
        track["last_rejected_step"] = int(env_step)
        track["rejected_geometry_count"] = int(track.get("rejected_geometry_count", 0)) + 1

    def _update_track(self, track_id: str, candidate: dict[str, Any], *, env_step: int, is_new: bool = False) -> None:
        track = self._tracks[track_id]
        relocation_evidence = candidate.get(
            "_identity_relocation_evidence"
        )
        if isinstance(relocation_evidence, dict):
            relocation_history = list(
                track.get("identity_relocation_history", []) or []
            )
            relocation_history.append(
                {
                    "env_step": int(env_step),
                    **dict(relocation_evidence),
                }
            )
            track["identity_relocation_history"] = (
                relocation_history[-4:]
            )
            track["last_relocated_step"] = int(env_step)
            # Geometry collected before a commanded manipulation belongs to
            # the old pose. Do not average it into the post-release pose.
            track["history"] = []
        refinement_evidence = candidate.get(
            "_identity_refinement_evidence"
        )
        if isinstance(refinement_evidence, dict):
            lineage = list(track.get("identity_refinement_history", []) or [])
            lineage.append(
                {
                    "env_step": int(env_step),
                    "source_class": track.get("class"),
                    "source_class_aliases": list(
                        track.get("class_aliases", []) or []
                    ),
                    "source_world_m": round_list(
                        track.get("stable_world_m", track.get("world_m"))
                    ),
                    "source_bbox_xyxy": round_list(
                        track.get("bbox_xyxy"),
                        digits=2,
                    ),
                    "refined_class": candidate.get(
                        "_class_name",
                        "object",
                    ),
                    "refined_world_m": round_list(
                        candidate.get("_world_m")
                    ),
                    **dict(refinement_evidence),
                }
            )
            track["identity_refinement_history"] = lineage[-4:]
            track["identity_granularity"] = "refined_member"
            track["class"] = candidate.get("_class_name", "object")
            track["class_aliases"] = list(
                candidate.get("_class_aliases", [])
            )
            # Aggregate and member centroids are different semantic frames.
            # Keeping the aggregate samples would bias the executable member
            # pose, so refinement starts a fresh geometry history while the
            # lineage above preserves the audit trail.
            track["history"] = []
            track["first_observed_world_m"] = round_list(
                candidate.get("_world_m")
            )
            track["first_observed_top_surface_world_m"] = round_list(
                candidate.get("_top_surface_world_m")
            )
        merge_identity = (
            should_merge_candidate_identity(track, candidate)
            or isinstance(relocation_evidence, dict)
        )
        if merge_identity:
            track["class_aliases"] = merge_class_aliases(track.get("class_aliases", []), candidate.get("_class_aliases", []))
        previous_world = track.get("stable_world_m", track.get("world_m"))
        current_world = candidate.get("_world_m")
        distance = world_distance_m(current_world, previous_world)
        if is_new:
            stability = "new"
        elif distance is None:
            stability = "tracked"
        elif distance <= self.stable_distance_m:
            stability = "stable"
        else:
            stability = "updated"

        sample = compact_candidate_for_track(candidate, env_step=env_step)
        history = list(track.get("history", []))
        history.append(sample)
        history = history[-self.window_size :]
        operation_candidates: list[dict[str, Any]] = []
        operation_candidate_provenance: dict[str, dict[str, int]] = {}
        for item in candidate.get("_operation_pose_candidates", []) or []:
            if not isinstance(item, dict):
                continue
            operation_candidates.append(dict(item))
            candidate_id = str(
                item.get("candidate_id", "") or ""
            ).strip()
            if not candidate_id:
                continue
            provenance = {
                "geometry_observation_env_step": int(env_step),
            }
            capture_id = _integer_or_none(
                candidate.get("_observation_capture_id")
            )
            if capture_id is not None:
                provenance[
                    "geometry_observation_capture_id"
                ] = capture_id
            operation_candidate_provenance[candidate_id] = provenance

        track.update(
            {
                "status": "visible",
                "stability": stability,
                "last_seen_step": int(env_step),
                "missing_steps": 0,
                "visible_count": int(track.get("visible_count", 0)) + 1,
                "history": history,
                "history_length": len(history),
                "score": round_float(candidate.get("score")),
                "bbox_xyxy": candidate.get("_bbox_xyxy"),
                "centroid_px": candidate.get("_centroid_px"),
                "camera": candidate.get("camera"),
                "supporting_cameras": list(
                    candidate.get(
                        "_supporting_cameras",
                        [],
                    )
                    or []
                ),
                "multiview_support_count": int(
                    candidate.get(
                        "_multiview_support_count",
                        0,
                    )
                    or 0
                ),
                "world_m": round_list(current_world),
                "top_surface_world_m": round_list(candidate.get("_top_surface_world_m")),
                "approach_world_m": round_list(candidate.get("_approach_world_m")),
                "approach_quat_wxyz": round_list(candidate.get("_approach_quat_wxyz")),
                "grasp_world_m": round_list(candidate.get("_grasp_world_m")),
                "grasp_quat_wxyz": round_list(candidate.get("_grasp_quat_wxyz")),
                "contact_world_m": round_list(candidate.get("_contact_world_m")),
                "contact_quat_wxyz": round_list(candidate.get("_contact_quat_wxyz")),
                "operation_pose_candidates": operation_candidates,
                "operation_pose_candidate_provenance": (
                    operation_candidate_provenance
                ),
                "source_rank": candidate.get("rank"),
                "quality": candidate.get("quality"),
                "quality_warnings": list((candidate.get("quality") or {}).get("warnings", []) or []),
                "actionable": bool((candidate.get("quality") or {}).get("actionable", True)),
                "mask_path": candidate.get("mask_path"),
                "source_instance_key": candidate.get("_source_instance_key"),
                "oracle_id": candidate.get("oracle_id"),
                "oracle_source_path": candidate.get("oracle_source_path"),
            }
        )
        if (
            operation_candidates
            or self._candidate_action_geometry_offsets(candidate)
        ):
            track["action_geometry_state"] = "verified"
            track.pop(
                "last_verified_observation_geometry_reference",
                None,
            )
        if merge_identity:
            incoming_role = candidate.get("query_role")
            next_role = merge_track_query_role(track.get("query_role"), incoming_role)
            track["source_object_id"] = candidate.get("object_id")
            track["source_text_prompt"] = candidate.get("text_prompt")
            track["query_role"] = next_role
            if normalize_role(next_role) == normalize_role(incoming_role):
                track["query_instance_hint"] = candidate.get("query_instance_hint")
                track["query_reason"] = candidate.get("query_reason")
        self._update_stable_points(track)
        self._record_current_verified_position(
            track,
            candidate,
            env_step=env_step,
        )

    def _record_current_verified_position(
        self,
        track: dict[str, Any],
        candidate: dict[str, Any],
        *,
        env_step: int,
    ) -> None:
        world = candidate.get("_world_m")
        if not is_number_list(world, length=3):
            return
        relocation_evidence = candidate.get(
            "_identity_relocation_evidence"
        )
        tolerance = None
        if isinstance(relocation_evidence, dict):
            tolerance = finite_float(
                relocation_evidence.get("tolerance_m")
            )
        if tolerance is None:
            tolerance = finite_float(
                track.get("position_tolerance_m")
            )
        if tolerance is None or tolerance <= 0.0:
            tolerance = STRONG_GEOMETRY_DISTANCE_M
        supporting_cameras = dedupe_preserving_order(
            [
                str(camera)
                for camera in (
                    list(
                        candidate.get(
                            "_supporting_cameras",
                            [],
                        )
                        or []
                    )
                    + [candidate.get("camera")]
                )
                if str(camera or "").strip()
            ]
        )
        track.update(
            {
                "position_state": POSITION_CURRENT_VERIFIED,
                "position_source": (
                    "multiview_world_observation"
                    if len(supporting_cameras) > 1
                    else "current_world_observation"
                ),
                "position_state_reason": (
                    "current calibrated world-coordinate observation accepted"
                ),
                "position_state_updated_step": int(env_step),
                "last_verified_world_m": round_list(world),
                "last_verified_score": round_float(
                    candidate.get("score")
                ),
                "last_verified_step": int(env_step),
                "last_verified_source": {
                    "kind": (
                        "multiview_world_observation"
                        if len(supporting_cameras) > 1
                        else "current_world_observation"
                    ),
                    "camera": candidate.get("camera"),
                    "supporting_cameras": supporting_cameras,
                    "backend": candidate.get("backend"),
                },
                "position_tolerance_m": round_float(
                    min(self.stable_distance_m, tolerance)
                ),
            }
        )

    def _update_stable_points(self, track: dict[str, Any]) -> None:
        history = list(track.get("history", []))
        for source_key, target_key in (
            ("world_m", "stable_world_m"),
            ("top_surface_world_m", "stable_top_surface_world_m"),
            ("approach_world_m", "stable_approach_world_m"),
            ("contact_world_m", "stable_contact_world_m"),
        ):
            track[target_key] = round_list(weighted_average_xyz([item.get(source_key) for item in history]))
        grounded_grasp = weighted_average_xyz([item.get("grasp_world_m") for item in history])
        if is_number_list(grounded_grasp, length=3):
            track["stable_grasp_world_m"] = round_list(grounded_grasp)
        else:
            top_surface = track.get("stable_top_surface_world_m")
            if is_number_list(top_surface, length=3):
                track["stable_grasp_world_m"] = round_list(
                    [top_surface[0], top_surface[1], top_surface[2] + DEFAULT_GRASP_CLEARANCE_M]
                )
            else:
                track["stable_grasp_world_m"] = None

    def _track_snapshot(self, track: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            key: value
            for key, value in track.items()
            if key not in {"history"}
        }
        snapshot["history"] = list(track.get("history", []))
        return snapshot


def extract_scene_candidates(
    segmentation: list[dict[str, Any]],
    *,
    multiview_action_geometry_distance_m: float = (
        DEFAULT_TEMPORAL_ACTION_GEOMETRY_DISTANCE_M
    ),
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for query_index, item in enumerate(segmentation):
        if not isinstance(item, dict) or not item.get("success"):
            continue
        identity_binding_query_key = f"scene_query_{query_index:04d}"
        detections = item.get("detections")
        if isinstance(detections, list) and detections:
            raw_candidates = [det for det in detections if isinstance(det, dict)]
        else:
            raw_candidates = [item]
        for raw in raw_candidates:
            candidate = dict(raw)
            candidate.setdefault("object_id", item.get("object_id"))
            candidate.setdefault("text_prompt", item.get("text_prompt"))
            candidate.setdefault("camera", item.get("camera"))
            candidate.setdefault("backend", item.get("backend"))
            candidate.setdefault("env_step", item.get("env_step"))
            candidate.setdefault("robot_state", item.get("robot_state"))
            candidate.setdefault("query_role", item.get("query_role"))
            candidate.setdefault("query_instance_hint", item.get("query_instance_hint"))
            candidate.setdefault("query_reason", item.get("query_reason"))
            candidate.setdefault("instance_ref", item.get("instance_ref", item.get("query_instance_ref")))
            candidate.setdefault("identity_binding_required", item.get("identity_binding_required"))
            candidate.setdefault("identity_binding_error", item.get("identity_binding_error"))
            candidate.setdefault("_identity_binding_query_key", identity_binding_query_key)
            candidate.setdefault("oracle_aliases", item.get("oracle_aliases"))
            candidate.setdefault("oracle_id", item.get("oracle_id"))
            candidate.setdefault("oracle_source_path", item.get("oracle_source_path"))
            candidate.setdefault("oracle_class_name", item.get("oracle_class_name"))
            if "bbox_xyxy" not in candidate and "box_xyxy" in candidate:
                candidate["bbox_xyxy"] = candidate.get("box_xyxy")
            candidate["grounding_3d"] = select_candidate_grounding(candidate, item)
            if "score" not in candidate:
                candidate["score"] = item.get("score")
            key = (
                *candidate_identity_binding_partition(candidate),
                normalized_camera_name(candidate.get("camera")),
                *candidate_key(candidate),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return merge_duplicate_candidates(
        candidates,
        multiview_action_geometry_distance_m=(
            multiview_action_geometry_distance_m
        ),
    )


def has_candidate_evidence(candidate: dict[str, Any]) -> bool:
    if candidate_world(candidate) is not None:
        return True
    if parse_bbox(candidate.get("bbox_xyxy")) is not None:
        return True
    if parse_centroid(candidate.get("centroid_px")) is not None:
        return True
    if candidate.get("mask_path"):
        return True
    return False


def scene_candidate_quality(
    candidate: dict[str, Any],
    *,
    min_candidate_score: float = MIN_SCENE_CANDIDATE_SCORE,
    max_object_z_extent_m: float = MAX_MANIPULATION_OBJECT_Z_EXTENT_M,
    max_object_xy_extent_m: float = MAX_MANIPULATION_OBJECT_XY_EXTENT_M,
    max_object_world_z_m: float = MAX_MANIPULATION_OBJECT_WORLD_Z_M,
    robot_state: dict[str, Any] | None = None,
    robot_self_filter_radius_m: float = 0.08,
    robot_self_filter_z_margin_m: float = 0.08,
) -> dict[str, Any]:
    warnings: list[str] = []
    reject = False
    is_oracle = str(candidate.get("backend", "") or "").strip().lower() == "oracle_simulator"
    min_candidate_score = max(0.0, float(min_candidate_score))
    max_object_z_extent_m = max(0.0, float(max_object_z_extent_m))
    max_object_xy_extent_m = max(0.0, float(max_object_xy_extent_m))
    max_object_world_z_m = max(0.0, float(max_object_world_z_m))
    robot_self_filter_radius_m = max(0.0, float(robot_self_filter_radius_m))
    robot_self_filter_z_margin_m = max(0.0, float(robot_self_filter_z_margin_m))

    raw_score = candidate.get("score")
    score = finite_float(raw_score)
    if not is_oracle and raw_score is not None and score is not None and score < min_candidate_score:
        warnings.append(f"sam_score_below_min:{score:.3f}<{min_candidate_score:.3f}")
        reject = True

    grounding = candidate.get("grounding_3d")
    extents = grounding_extents_m(grounding)
    world_z_max = grounding_world_z_max_m(grounding)
    if not is_oracle and extents is not None:
        x_extent, y_extent, z_extent = extents
        if max_object_z_extent_m > 0.0 and z_extent > max_object_z_extent_m:
            warnings.append(f"z_extent_too_large:{z_extent:.3f}m")
            reject = True
        if max_object_xy_extent_m > 0.0 and max(x_extent, y_extent) > max_object_xy_extent_m:
            warnings.append(f"xy_extent_too_large:{max(x_extent, y_extent):.3f}m")
            reject = True
    if not is_oracle and max_object_world_z_m > 0.0 and world_z_max is not None and world_z_max > max_object_world_z_m:
        warnings.append(f"world_z_above_workspace:{world_z_max:.3f}>{max_object_world_z_m:.3f}m")
        reject = True
    robot_overlap = robot_self_overlap_warning(
        candidate,
        robot_state=robot_state,
        radius_m=robot_self_filter_radius_m,
        z_margin_m=robot_self_filter_z_margin_m,
    )
    if not is_oracle and robot_overlap:
        warnings.append(robot_overlap)
        reject = True

    return {
        "actionable": not reject,
        "warnings": warnings,
        "score": None if score is None else round(score, 6),
        "world_extent_m": None if extents is None else round_list(extents),
        "world_z_max_m": None if world_z_max is None else round_float(world_z_max),
    }


def robot_self_overlap_warning(
    candidate: dict[str, Any],
    *,
    robot_state: dict[str, Any] | None,
    radius_m: float,
    z_margin_m: float,
) -> str:
    if radius_m <= 0.0 or not isinstance(robot_state, dict):
        return ""
    candidate_point = candidate_world(candidate)
    if candidate_point is None:
        return ""
    candidate_z = float(candidate_point[2])
    for arm in ("left", "right"):
        arm_state = robot_state.get(arm)
        if not isinstance(arm_state, dict):
            continue
        ee_xyz = arm_state.get("xyz")
        if not is_number_list(ee_xyz, length=3):
            continue
        ee = [float(item) for item in ee_xyz]
        distance = world_distance_m(candidate_point, ee)
        if distance is None or distance > radius_m:
            continue
        if z_margin_m > 0.0 and abs(candidate_z - ee[2]) > z_margin_m:
            continue
        return f"robot_self_geometry:{arm}_ee_distance={distance:.3f}m<={radius_m:.3f}m"
    return ""


def candidate_class_aliases(candidate: dict[str, Any]) -> list[str]:
    values = [
        candidate.get("oracle_class_name"),
        candidate.get("object_id"),
        candidate.get("source_object_id"),
    ]
    aliases = candidate.get("oracle_aliases")
    if isinstance(aliases, list):
        values.extend(aliases)
    aliases = [normalize_class(value) for value in values]
    return dedupe_preserving_order([item for item in aliases if item])


def merge_duplicate_candidates(
    candidates: list[dict[str, Any]],
    *,
    multiview_action_geometry_distance_m: float = (
        DEFAULT_TEMPORAL_ACTION_GEOMETRY_DISTANCE_M
    ),
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=candidate_quality_sort_key):
        candidate = with_candidate_support_observation(candidate)
        duplicate_index = find_duplicate_candidate(
            merged,
            candidate,
            multiview_action_geometry_distance_m=(
                multiview_action_geometry_distance_m
            ),
        )
        if duplicate_index is None:
            merged.append(candidate)
            continue
        existing = merged[duplicate_index]
        existing_score = score_value(existing.get("score"))
        candidate_score = score_value(candidate.get("score"))
        supporting_observations = merge_candidate_support_observations(
            existing,
            candidate,
        )
        if candidate_score > existing_score:
            representative = dict(candidate)
        else:
            representative = dict(existing)
        representative["_supporting_observations"] = (
            supporting_observations
        )
        representative["_supporting_cameras"] = (
            dedupe_preserving_order(
                [
                    str(item.get("camera", "") or "")
                    for item in supporting_observations
                    if str(item.get("camera", "") or "").strip()
                ]
            )
        )
        representative["_multiview_support_count"] = len(
            representative["_supporting_cameras"]
        )
        merged[duplicate_index] = representative
    return merged


def find_duplicate_candidate(
    existing: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    multiview_action_geometry_distance_m: float = (
        DEFAULT_TEMPORAL_ACTION_GEOMETRY_DISTANCE_M
    ),
) -> int | None:
    candidate_source_key = candidate_source_instance_key(candidate)
    candidate_world_value = candidate_world(candidate)
    candidate_bbox = parse_bbox(candidate.get("bbox_xyxy"))
    candidate_binding_partition = candidate_identity_binding_partition(candidate)
    for index, item in enumerate(existing):
        if candidate_identity_binding_partition(item) != candidate_binding_partition:
            continue
        item_source_key = candidate_source_instance_key(item)
        if candidate_source_key and item_source_key:
            if candidate_source_key == item_source_key:
                return index
            continue
        distance = world_distance_m(candidate_world_value, candidate_world(item))
        bound_identity = (
            candidate_binding_partition
            and candidate_binding_partition[0]
            == "identity_bound"
        )
        candidate_camera = normalized_camera_name(
            candidate.get("camera")
        )
        item_camera = normalized_camera_name(
            item.get("camera")
        )
        if bound_identity:
            # Bound detections from the same camera are alternative physical
            # hypotheses and must stay separate.  Calibrated detections from
            # different cameras may be fused only when their world positions
            # agree tightly.
            if (
                candidate_camera
                and item_camera
                and candidate_camera != item_camera
                and distance is not None
                and distance
                <= MULTIVIEW_WORLD_MERGE_DISTANCE_M
            ):
                geometry_consistent = (
                    candidate_action_geometry_consistency(
                        candidate,
                        item,
                        max_offset_jump_m=(
                            multiview_action_geometry_distance_m
                        ),
                    )
                )
                if geometry_consistent is False:
                    continue
                return index
        elif distance is not None and distance <= 0.035:
            return index
        if (
            not bound_identity
            and same_or_unknown_camera(
                candidate.get("camera"),
                item.get("camera"),
            )
        ):
            iou = bbox_iou(candidate_bbox, parse_bbox(item.get("bbox_xyxy")))
            if iou is not None and iou >= 0.6:
                return index
    return None


def normalized_camera_name(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("_camera", "")
    )


def with_candidate_support_observation(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(candidate)
    observations = [
        dict(item)
        for item in candidate.get(
            "_supporting_observations",
            [],
        )
        or []
        if isinstance(item, dict)
    ]
    observations.append(
        {
            "camera": candidate.get("camera"),
            "score": round_float(candidate.get("score")),
            "world_m": round_list(candidate_world(candidate)),
            "bbox_xyxy": round_list(
                parse_bbox(candidate.get("bbox_xyxy")),
                digits=2,
            ),
            "mask_path": candidate.get("mask_path"),
        }
    )
    enriched["_supporting_observations"] = (
        dedupe_candidate_support_observations(observations)
    )
    enriched["_supporting_cameras"] = (
        dedupe_preserving_order(
            [
                str(item.get("camera", "") or "")
                for item in enriched[
                    "_supporting_observations"
                ]
                if str(item.get("camera", "") or "").strip()
            ]
        )
    )
    enriched["_multiview_support_count"] = len(
        enriched["_supporting_cameras"]
    )
    return enriched


def merge_candidate_support_observations(
    first: dict[str, Any],
    second: dict[str, Any],
) -> list[dict[str, Any]]:
    return dedupe_candidate_support_observations(
        [
            *[
                dict(item)
                for item in first.get(
                    "_supporting_observations",
                    [],
                )
                or []
                if isinstance(item, dict)
            ],
            *[
                dict(item)
                for item in second.get(
                    "_supporting_observations",
                    [],
                )
                or []
                if isinstance(item, dict)
            ],
        ]
    )


def dedupe_candidate_support_observations(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in observations:
        key = (
            normalized_camera_name(item.get("camera")),
            str(item.get("mask_path", "") or ""),
            repr(item.get("bbox_xyxy")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
    return deduped


def candidate_quality_sort_key(candidate: dict[str, Any]) -> tuple[int, float, int]:
    has_world = 0 if candidate_world(candidate) is not None else 1
    return has_world, -score_value(candidate.get("score")), int(candidate.get("rank", 9999) or 9999)


def select_candidate_grounding(candidate: dict[str, Any], parent: dict[str, Any]) -> Any:
    candidate_grounding = candidate.get("grounding_3d")
    if isinstance(candidate_grounding, dict) and candidate_grounding.get("success"):
        return candidate_grounding
    parent_grounding = parent.get("grounding_3d")
    if isinstance(parent_grounding, dict) and parent_grounding.get("success"):
        return parent_grounding
    if "grounding_3d" in candidate:
        return candidate_grounding
    return parent_grounding


def candidate_key(candidate: dict[str, Any]) -> tuple[str, str]:
    source_key = candidate_source_instance_key(candidate)
    if source_key:
        return "source_instance", source_key
    class_name = normalize_class(candidate.get("object_id", "object"))
    camera = str(candidate.get("camera", "") or "")
    mask_path = str(candidate.get("mask_path", "") or "")
    if mask_path:
        return class_name, mask_path
    bbox = candidate.get("bbox_xyxy")
    if isinstance(bbox, (list, tuple)):
        return class_name, camera + ":" + ",".join(str(round_float(item, digits=2)) for item in bbox)
    return class_name, camera + ":" + str(candidate.get("rank", len(mask_path)))


def candidate_source_instance_key(candidate: dict[str, Any]) -> str:
    oracle_id = str(candidate.get("oracle_id", "") or "").strip()
    source_path = str(candidate.get("oracle_source_path", "") or "").strip()
    if oracle_id:
        return "oracle_id:" + oracle_id
    if source_path:
        return "oracle_path:" + source_path
    return ""


def same_or_unknown_camera(a: Any, b: Any) -> bool:
    camera_a = str(a or "").strip().lower().replace("_camera", "")
    camera_b = str(b or "").strip().lower().replace("_camera", "")
    return not camera_a or not camera_b or camera_a == camera_b


def normalize_class(value: Any) -> str:
    text, _ = normalize_perception_object_id(value)
    if text:
        return text
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def track_sort_key(track: dict[str, Any]) -> tuple[int, float, float, str]:
    world = track.get("stable_world_m", track.get("world_m"))
    if is_number_list(world, length=3):
        return 0, float(world[0]), float(world[1]), str(track.get("track_id", ""))
    bbox = parse_bbox(track.get("bbox_xyxy"))
    if bbox is not None:
        center_x = 0.5 * (bbox[0] + bbox[2])
        center_y = 0.5 * (bbox[1] + bbox[3])
        return 1, center_x, center_y, str(track.get("track_id", ""))
    return 2, 0.0, 0.0, str(track.get("track_id", ""))


def candidate_world(candidate: dict[str, Any]) -> list[float] | None:
    grounding = candidate.get("grounding_3d")
    if isinstance(grounding, dict) and grounding.get("success"):
        value = grounding.get("centroid_world")
        if is_number_list(value, length=3):
            return [float(item) for item in value]
    return None


def candidate_approach(candidate: dict[str, Any]) -> list[float] | None:
    grounding = candidate.get("grounding_3d")
    if isinstance(grounding, dict) and grounding.get("success"):
        pose = grounding.get("approach_pose_world")
        if is_number_list(pose, length=7):
            return [float(item) for item in pose[:3]]
        value = grounding.get("approach_point_world")
        if is_number_list(value, length=3):
            return [float(item) for item in value]
    return None


def candidate_top_surface(candidate: dict[str, Any]) -> list[float] | None:
    grounding = candidate.get("grounding_3d")
    if isinstance(grounding, dict) and grounding.get("success"):
        value = grounding.get("top_surface_world")
        if is_number_list(value, length=3):
            return [float(item) for item in value]
    return None


def candidate_grasp(candidate: dict[str, Any]) -> list[float] | None:
    grounding = candidate.get("grounding_3d")
    if isinstance(grounding, dict) and grounding.get("success"):
        pose = grounding.get("grasp_pose_world")
        if is_number_list(pose, length=7):
            return [float(item) for item in pose[:3]]
        value = grounding.get("grasp_point_world")
        if is_number_list(value, length=3):
            return [float(item) for item in value]
    top_surface = candidate_top_surface(candidate)
    if top_surface is not None:
        return [top_surface[0], top_surface[1], top_surface[2] + DEFAULT_GRASP_CLEARANCE_M]
    return candidate_world(candidate)


def candidate_contact(candidate: dict[str, Any]) -> list[float] | None:
    grounding = candidate.get("grounding_3d")
    if isinstance(grounding, dict) and grounding.get("success"):
        value = grounding.get("contact_point_world")
        if is_number_list(value, length=3):
            return [float(item) for item in value]
    top_surface = candidate_top_surface(candidate)
    if top_surface is not None:
        return top_surface
    return candidate_world(candidate)


def candidate_grounded_quat(candidate: dict[str, Any], pose_key: str) -> list[float] | None:
    grounding = candidate.get("grounding_3d")
    if not isinstance(grounding, dict) or not grounding.get("success"):
        return None
    pose = grounding.get(pose_key)
    if not is_number_list(pose, length=7):
        return None
    quat = [float(item) for item in pose[3:7]]
    norm = math.sqrt(sum(item * item for item in quat))
    if not math.isfinite(norm) or norm <= 1e-8:
        return None
    return [item / norm for item in quat]


def candidate_operation_pose_candidates(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    grounding = candidate.get("grounding_3d")
    if not isinstance(grounding, dict) or not grounding.get("success"):
        return []
    return validated_operation_pose_candidates(grounding)


def parse_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except Exception:
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    return bbox


def parse_centroid(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        centroid = [float(value[0]), float(value[1])]
    except Exception:
        return None
    if not all(math.isfinite(item) for item in centroid):
        return None
    return centroid


def bbox_iou(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    intersection = width * height
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    if union <= 0.0:
        return None
    return intersection / union


def bound_identity_refinement_evidence(
    track: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    """Prove that one bound detection is a member-level refinement.

    A broad discovery mask may cover several adjacent objects.  A later
    instance-specific query can legitimately return one strict subset whose
    centroid is farther from the aggregate centroid than the ordinary matching
    threshold.  This path is geometry-only: the candidate must be uniquely
    selected by the caller, nearly contained by the previous image footprint,
    materially smaller in image and 3D extent, and still lie inside the prior
    observed volume.  It does not depend on task names, colors, object counts,
    or language keywords.
    """

    if not same_or_unknown_camera(
        candidate.get("camera"),
        track.get("camera"),
    ):
        return None
    candidate_bbox = candidate.get("_bbox_xyxy")
    track_bbox = parse_bbox(track.get("bbox_xyxy"))
    containment = bbox_containment_ratio(
        candidate_bbox,
        track_bbox,
    )
    area_ratio = bbox_area_ratio(candidate_bbox, track_bbox)
    if (
        containment is None
        or containment < BOUND_REFINEMENT_MIN_CONTAINMENT
        or area_ratio is None
        or area_ratio > BOUND_REFINEMENT_MAX_AREA_RATIO
    ):
        return None

    track_quality = (
        track.get("quality")
        if isinstance(track.get("quality"), dict)
        else {}
    )
    candidate_quality = (
        candidate.get("quality")
        if isinstance(candidate.get("quality"), dict)
        else {}
    )
    track_extent = track_quality.get("world_extent_m")
    candidate_extent = candidate_quality.get("world_extent_m")
    if not is_number_list(track_extent, length=3) or not is_number_list(
        candidate_extent,
        length=3,
    ):
        return None
    horizontal_extent_ratios = [
        float(candidate_extent[index])
        / max(float(track_extent[index]), 1e-6)
        for index in (0, 1)
    ]
    if min(horizontal_extent_ratios) > BOUND_REFINEMENT_MAX_EXTENT_RATIO:
        return None

    candidate_world_value = candidate.get("_world_m")
    track_world_value = track.get(
        "stable_world_m",
        track.get("world_m"),
    )
    if not point_within_observed_extent(
        candidate_world_value,
        center=track_world_value,
        extent=track_extent,
    ):
        return None
    return {
        "candidate_containment": round(float(containment), 6),
        "candidate_area_ratio": round(float(area_ratio), 6),
        "horizontal_extent_ratios": round_list(
            horizontal_extent_ratios,
            digits=6,
        ),
        "source": "strict_subregion_of_bound_observation",
    }


def bbox_containment_ratio(
    inner: list[float] | None,
    outer: list[float] | None,
) -> float | None:
    if inner is None or outer is None:
        return None
    inner_area = bbox_area(inner)
    if inner_area is None or inner_area <= 0.0:
        return None
    left = max(inner[0], outer[0])
    top = max(inner[1], outer[1])
    right = min(inner[2], outer[2])
    bottom = min(inner[3], outer[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    return intersection / inner_area


def bbox_area_ratio(
    numerator: list[float] | None,
    denominator: list[float] | None,
) -> float | None:
    numerator_area = bbox_area(numerator)
    denominator_area = bbox_area(denominator)
    if (
        numerator_area is None
        or denominator_area is None
        or denominator_area <= 0.0
    ):
        return None
    return numerator_area / denominator_area


def bbox_area(value: list[float] | None) -> float | None:
    if value is None:
        return None
    return max(0.0, value[2] - value[0]) * max(
        0.0,
        value[3] - value[1],
    )


def point_within_observed_extent(
    point: Any,
    *,
    center: Any,
    extent: Any,
    margin_m: float = 0.01,
) -> bool:
    if (
        not is_number_list(point, length=3)
        or not is_number_list(center, length=3)
        or not is_number_list(extent, length=3)
    ):
        return False
    margin = max(0.0, float(margin_m))
    return all(
        abs(float(point[index]) - float(center[index]))
        <= 0.5 * float(extent[index]) + margin
        for index in range(3)
    )


def point_distance_2d(a: Any, b: Any) -> float | None:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != 2 or len(b) != 2:
        return None
    try:
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
    except Exception:
        return None
    if not all(math.isfinite(item) for item in (ax, ay, bx, by)):
        return None
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def relative_xyz(point: Any, origin: Any) -> list[float] | None:
    if not is_number_list(point, length=3) or not is_number_list(origin, length=3):
        return None
    return [float(point[index]) - float(origin[index]) for index in range(3)]


def weighted_average_xyz(values: list[Any]) -> list[float] | None:
    points = [list(map(float, item)) for item in values if is_number_list(item, length=3)]
    if not points:
        return None
    total_weight = 0.0
    accum = [0.0, 0.0, 0.0]
    for index, point in enumerate(points):
        weight = float(index + 1)
        total_weight += weight
        for dim in range(3):
            accum[dim] += point[dim] * weight
    if total_weight <= 0.0:
        return None
    return [value / total_weight for value in accum]


def compact_candidate_for_track(candidate: dict[str, Any], *, env_step: int) -> dict[str, Any]:
    return {
        "env_step": int(env_step),
        "observation_capture_id": candidate.get(
            "_observation_capture_id"
        ),
        "object_id": candidate.get("object_id"),
        "camera": candidate.get("camera"),
        "supporting_cameras": list(
            candidate.get("_supporting_cameras", []) or []
        ),
        "query_role": candidate.get("query_role"),
        "score": round_float(candidate.get("score")),
        "bbox_xyxy": round_list(candidate.get("_bbox_xyxy"), digits=2),
        "centroid_px": round_list(candidate.get("_centroid_px"), digits=2),
        "world_m": round_list(candidate.get("_world_m")),
        "top_surface_world_m": round_list(candidate.get("_top_surface_world_m")),
        "approach_world_m": round_list(candidate.get("_approach_world_m")),
        "approach_quat_wxyz": round_list(candidate.get("_approach_quat_wxyz")),
        "grasp_world_m": round_list(candidate.get("_grasp_world_m")),
        "grasp_quat_wxyz": round_list(candidate.get("_grasp_quat_wxyz")),
        "contact_world_m": round_list(candidate.get("_contact_world_m")),
        "contact_quat_wxyz": round_list(candidate.get("_contact_quat_wxyz")),
        "operation_pose_candidates": [
            dict(item)
            for item in candidate.get("_operation_pose_candidates", [])
            if isinstance(item, dict)
        ],
        "mask_path": candidate.get("mask_path"),
    }


def compact_candidate_for_memory(candidate: dict[str, Any], *, track_id: str) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "observation_capture_id": candidate.get(
            "_observation_capture_id"
        ),
        "object_id": candidate.get("object_id"),
        "camera": candidate.get("camera"),
        "query_role": candidate.get("query_role"),
        "score": round_float(candidate.get("score")),
        "bbox_xyxy": round_list(candidate.get("_bbox_xyxy"), digits=2),
        "centroid_px": round_list(candidate.get("_centroid_px"), digits=2),
        "world_m": round_list(candidate.get("_world_m")),
        "quality": candidate.get("quality"),
        "mask_path": candidate.get("mask_path"),
    }


def compact_dropped_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    quality = candidate.get("quality") if isinstance(candidate.get("quality"), dict) else {}
    compact = {
        "object_id": candidate.get("object_id"),
        "camera": candidate.get("camera"),
        "query_role": candidate.get("query_role"),
        "score": round_float(candidate.get("score")),
        "bbox_xyxy": round_list(parse_bbox(candidate.get("bbox_xyxy")), digits=2),
        "centroid_px": round_list(parse_centroid(candidate.get("centroid_px")), digits=2),
        "world_m": round_list(candidate_world(candidate)),
        "quality": quality,
        "mask_path": candidate.get("mask_path"),
    }
    rejected_track_id = candidate.get("_rejected_track_id")
    if rejected_track_id:
        compact["track_id"] = rejected_track_id
    return compact


def merge_class_aliases(existing: Any, incoming: Any) -> list[str]:
    existing_values = existing if isinstance(existing, list) else []
    incoming_values = incoming if isinstance(incoming, list) else []
    return dedupe_preserving_order(
        [str(item) for item in existing_values + incoming_values if str(item or "").strip()]
    )


def track_identity_tokens(track: dict[str, Any]) -> set[str]:
    values = [track.get("class"), track.get("source_object_id")]
    aliases = track.get("class_aliases")
    if isinstance(aliases, list):
        values.extend(aliases)
    return {normalize_class(item) for item in values if normalize_class(item)}


def candidate_identity_tokens(candidate: dict[str, Any]) -> set[str]:
    values = [candidate.get("_class_name"), candidate.get("object_id"), candidate.get("source_object_id")]
    aliases = candidate.get("_class_aliases")
    if isinstance(aliases, list):
        values.extend(aliases)
    return {normalize_class(item) for item in values if normalize_class(item)}


def identity_tokens_semantically_overlap(
    first: set[str],
    second: set[str],
) -> bool:
    """Match exact aliases or noun-like components without color hardcoding."""

    if first & second:
        return True
    non_identity_components = {
        "object",
        "item",
        "selected",
        "target",
        "component",
        "thing",
    }
    first_components = {
        component
        for token in first
        for component in token.split("_")
        if (
            len(component) >= 3
            and component not in non_identity_components
        )
    }
    second_components = {
        component
        for token in second
        for component in token.split("_")
        if (
            len(component) >= 3
            and component not in non_identity_components
        )
    }
    return bool(first_components & second_components)


def explicit_role_conflict(existing_role: Any, incoming_role: Any) -> bool:
    existing = normalize_role(existing_role)
    incoming = normalize_role(incoming_role)
    return existing in {"target", "tool"} and incoming in {"target", "tool"} and existing != incoming


def should_merge_candidate_identity(track: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if not track.get("class_aliases"):
        return True
    if track_identity_tokens(track) & candidate_identity_tokens(candidate):
        return True
    return not explicit_role_conflict(track.get("query_role"), candidate.get("query_role"))


def merge_track_query_role(existing_role: Any, incoming_role: Any) -> str:
    existing = normalize_role(existing_role)
    incoming = normalize_role(incoming_role)
    if incoming == "context":
        return existing
    if existing == "context":
        return incoming
    if existing == incoming:
        return existing
    return existing


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def build_task_focus(
    *,
    instances: list[dict[str, Any]],
    global_task: str,
    current_subtask: str,
    perception_queries: list[dict[str, Any]],
    binding_roles: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    accepted_binding_roles = (
        {"target", "tool"}
        if binding_roles is None
        else {
            normalize_role(role)
            for role in binding_roles
            if normalize_role(role) in {"target", "tool", "context"}
        }
    )
    focus_pool = [
        item for item in instances
        if item.get("status") in {"visible", "tracked"} and is_actionable_instance(item)
    ]
    target_queries = [query for query in perception_queries if normalize_role(query.get("query_role", query.get("role"))) == "target"]
    tool_queries = [query for query in perception_queries if normalize_role(query.get("query_role", query.get("role"))) == "tool"]
    context_queries = [query for query in perception_queries if normalize_role(query.get("query_role", query.get("role"))) == "context"]
    target_instances = select_instances_for_queries(focus_pool, target_queries)
    tool_instances = select_instances_for_queries(focus_pool, tool_queries)
    context_instances = select_instances_for_queries(focus_pool, context_queries)
    queries_by_role = {
        "target": target_queries,
        "tool": tool_queries,
        "context": context_queries,
    }
    validated_queries = [
        query
        for role in ("target", "tool", "context")
        if role in accepted_binding_roles
        for query in queries_by_role[role]
    ]
    binding_queries = [
        query
        for query in validated_queries
        if query_requires_instance_binding(query)
    ]
    binding_errors: list[str] = []
    for query in validated_queries:
        matches = matching_instances_for_query(focus_pool, query)
        requires_binding = query_requires_instance_binding(query)
        if normalize_entity_scope(query.get("entity_scope")) == "reference_set":
            continue
        if len(matches) == 1:
            continue
        object_id = str(query.get("object_id", "object") or "object")
        explicit_error = str(query.get("identity_binding_error", "") or "").strip()
        if explicit_error:
            binding_errors.append(f"{object_id}:{explicit_error}")
        elif len(matches) > 1 and (query_instance_ref(query) or query_oracle_id(query)):
            binding_errors.append(f"{object_id}:ambiguous_instance_reference")
        elif len(matches) > 1:
            binding_errors.append(f"{object_id}:ambiguous_instance_candidates:{len(matches)}")
        elif query_instance_ref(query) or query_oracle_id(query):
            binding_errors.append(f"{object_id}:unresolved_instance_reference")
        elif requires_binding:
            binding_errors.append(f"{object_id}:missing_instance_reference")
    reason_summary = build_reason_summary(
        target_instances=target_instances,
        tool_instances=tool_instances,
        perception_queries=perception_queries,
    )
    return {
        "current_subtask": str(current_subtask or "").strip(),
        "target_instances": [str(item.get("instance_id")) for item in target_instances],
        "tool_instances": [str(item.get("instance_id")) for item in tool_instances],
        "context_instances": [str(item.get("instance_id")) for item in context_instances],
        "reason_summary": reason_summary,
        "source": "vlm_perception_queries",
        "identity_binding_required": bool(binding_queries or binding_errors),
        "identity_binding_errors": dedupe_preserving_order(binding_errors),
        "identity_binding_roles": sorted(
            {
                normalize_role(query.get("query_role", query.get("role")))
                for query in binding_queries
            }
        ),
    }


def normalize_role(value: Any) -> str:
    role = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if role in {"target", "tool", "context"}:
        return role
    return "context"


def select_instances_for_queries(
    instances: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        if normalize_entity_scope(query.get("entity_scope")) == "reference_set":
            # Reference sets are exposed through scene_memory.reference_regions
            # and operation targets, never through manipulable task focus.
            continue
        candidates = matching_instances_for_query(instances, query)
        # Only an exact, unique candidate may advance task focus. Natural
        # language hints and deterministic sorting are not identity bindings.
        if len(candidates) != 1:
            continue
        candidate = candidates[0]
        instance_id = str(candidate.get("instance_id", ""))
        if not instance_id or instance_id in seen:
            continue
        selected.append(candidate)
        seen.add(instance_id)
    return selected


def matching_instances_for_query(instances: list[dict[str, Any]], query: dict[str, Any]) -> list[dict[str, Any]]:
    instance_ref = query_instance_ref(query)
    oracle_id = query_oracle_id(query)
    if instance_ref or oracle_id:
        exact_matches: list[dict[str, Any]] = []
        for instance in instances:
            if not is_actionable_instance(instance):
                continue
            if query_requires_instance_binding(query) and (
                instance.get("status") not in {"visible", "tracked"}
                or not instance_has_finite_grounding(instance)
            ):
                continue
            if instance_ref and instance_ref not in instance_identity_refs(instance):
                continue
            if oracle_id and oracle_id != str(instance.get("oracle_id", "") or "").strip():
                continue
            exact_matches.append(instance)
        return exact_matches
    if query_requires_instance_binding(query):
        return []
    object_id = normalize_class(query.get("object_id", ""))
    text_prompt = normalize_class(query.get("text_prompt", ""))
    matches: list[dict[str, Any]] = []
    for instance in instances:
        if not is_actionable_instance(instance):
            continue
        identity = track_identity_tokens(instance)
        if object_id and object_id in identity:
            matches.append(instance)
            continue
        if text_prompt and text_prompt in identity:
            matches.append(instance)
    return matches


def query_instance_ref(query: dict[str, Any]) -> str:
    return str(query.get("instance_ref", query.get("query_instance_ref", "")) or "").strip()


def query_oracle_id(query: dict[str, Any]) -> str:
    explicit_query_id = str(query.get("query_oracle_id", "") or "").strip()
    if explicit_query_id:
        return explicit_query_id
    if query_requires_instance_binding(query):
        return str(query.get("oracle_id", "") or "").strip()
    return ""


def query_requires_instance_binding(query: dict[str, Any]) -> bool:
    return bool(query.get("identity_binding_required", query.get("_identity_binding_required", False)))


def identity_bound_candidate_ref(candidate: dict[str, Any]) -> str:
    """Return the stable track constraint for a non-Oracle bound observation."""
    if str(candidate.get("backend", "") or "").strip().lower() == "oracle_simulator":
        return ""
    if not query_requires_instance_binding(candidate):
        return ""
    return query_instance_ref(candidate)


def identity_bound_candidate_query_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("_identity_binding_query_key", "") or "").strip() or "scene_query"


def candidate_identity_binding_partition(candidate: dict[str, Any]) -> tuple[str, ...]:
    """Partition by stable identity while allowing calibrated view fusion."""
    instance_ref = identity_bound_candidate_ref(candidate)
    if instance_ref:
        return "identity_bound", instance_ref
    return "unbound", ""


def instance_identity_refs(instance: dict[str, Any]) -> set[str]:
    return {
        value
        for value in (
            str(instance.get("instance_id", "") or "").strip(),
            str(instance.get("track_id", "") or "").strip(),
            str(instance.get("oracle_id", "") or "").strip(),
            str(instance.get("oracle_source_path", "") or "").strip(),
        )
        if value
    }


def instance_has_finite_grounding(instance: dict[str, Any]) -> bool:
    return any(
        is_number_list(instance.get(key), length=3)
        for key in ("world_m", "approach_world_m", "grasp_world_m", "contact_world_m")
    )


def build_reason_summary(
    *,
    target_instances: list[dict[str, Any]],
    tool_instances: list[dict[str, Any]],
    perception_queries: list[dict[str, Any]],
) -> str:
    if not target_instances and not tool_instances:
        return "No VLM-designated target/tool instance is available from current perception."
    parts = []
    if target_instances:
        parts.append("target=" + ",".join(str(item.get("instance_id")) for item in target_instances))
    if tool_instances:
        parts.append("tool=" + ",".join(str(item.get("instance_id")) for item in tool_instances))
    reasons = [str(item.get("query_reason", item.get("reason", ""))).strip() for item in perception_queries]
    reasons = [item for item in reasons if item]
    if reasons:
        parts.append("vlm_reason=" + reasons[0])
    parts.append("selected from VLM perception-query roles and stable 3D instance binding")
    return "; ".join(parts)


def build_uncertainty(instances: list[dict[str, Any]]) -> list[str]:
    uncertainty: list[str] = []
    for item in instances:
        warnings = item.get("quality_warnings")
        if isinstance(warnings, list):
            for warning in warnings:
                if warning:
                    uncertainty.append(f"{item.get('instance_id')} quality warning: {warning}.")
    for item in instances:
        score = item.get("score")
        try:
            score_value = float(score)
        except Exception:
            continue
        if score_value < 0.3:
            uncertainty.append(f"{item.get('instance_id')} has low SAM score {score_value:.2f}; use geometry/history before action.")
    for item in instances:
        stability = str(
            item.get("stability", "") or ""
        ).strip().lower()
        position_state = str(
            item.get("position_state", "") or ""
        ).strip().lower()
        action_geometry_state = str(
            item.get("action_geometry_state", "verified")
            or "verified"
        ).strip().lower()
        if position_state == POSITION_MOTION_UNCERTAIN:
            uncertainty.append(
                f"{item.get('instance_id')} may have moved after an "
                "unverified physical event; its historical position is "
                "non-executable until reacquired."
            )
        elif action_geometry_state == "identity_repair_expired":
            uncertainty.append(
                f"{item.get('instance_id')} post-release identity/action-geometry repair "
                "lease expired; stale grounded geometry is unavailable and the task must replan."
            )
        elif action_geometry_state == "relocation_pending":
            uncertainty.append(
                f"{item.get('instance_id')} relocated position is visible, but action geometry "
                "is quarantined until two clean observations agree."
            )
        elif item.get("status") == "tracked":
            if stability == "geometry_inconsistent_current_frame":
                uncertainty.append(
                    f"{item.get('instance_id')} current detection has inconsistent action geometry; "
                    "the last physically valid position remains in memory."
                )
            elif position_state == POSITION_MEMORY_VALID:
                uncertainty.append(
                    f"{item.get('instance_id')} is currently occluded, but "
                    "its last verified world position remains valid because "
                    "no object-motion event has occurred."
                )
            else:
                uncertainty.append(
                    f"{item.get('instance_id')} is not visible in the current frame; retained from recent memory."
                )
    return uncertainty[:6]


def summarize_scene_memory(instances: list[dict[str, Any]], task_focus: dict[str, Any], uncertainty: list[str]) -> str:
    visible = [item for item in instances if item.get("status") == "visible"]
    names = [str(item.get("instance_id")) for item in visible[:8]]
    target = ",".join(task_focus.get("target_instances", []) or [])
    tool = ",".join(task_focus.get("tool_instances", []) or [])
    parts = [f"scene_instances={names}"]
    if target or tool:
        parts.append(f"task_focus=target:{target or '-'} tool:{tool or '-'}")
    if uncertainty:
        parts.append(f"uncertainty={uncertainty[0]}")
    return "; ".join(parts)


def is_actionable_instance(instance: dict[str, Any]) -> bool:
    quality = instance.get("quality")
    if isinstance(quality, dict) and quality.get("actionable") is False:
        return False
    if instance.get("actionable") is False:
        return False
    return True


def world_distance_m(a: Any, b: Any) -> float | None:
    if not is_number_list(a, length=3) or not is_number_list(b, length=3):
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def candidate_action_geometry_offsets(
    candidate: dict[str, Any],
) -> dict[str, list[float]]:
    origin = candidate_world(candidate)
    if not is_number_list(origin, length=3):
        return {}
    offsets: dict[str, list[float]] = {}
    for name, value in (
        ("top_surface", candidate_top_surface(candidate)),
        ("approach", candidate_approach(candidate)),
        ("grasp", candidate_grasp(candidate)),
        ("contact", candidate_contact(candidate)),
    ):
        offset = relative_xyz(value, origin)
        if offset is not None:
            offsets[name] = offset
    return offsets


def candidate_action_geometry_consistency(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    max_offset_jump_m: float,
) -> bool | None:
    """Compare cross-view shape geometry before irreversible early fusion.

    ``None`` preserves legacy fusion when neither view exposes comparable
    action geometry.  ``False`` keeps both full candidates for the track-aware
    association stage instead of losing the lower-scoring clean observation.
    """

    try:
        threshold = float(max_offset_jump_m)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(threshold) or threshold <= 0.0:
        return None
    first_offsets = candidate_action_geometry_offsets(first)
    second_offsets = candidate_action_geometry_offsets(second)
    common = sorted(set(first_offsets) & set(second_offsets))
    if not common:
        return None
    jumps = [
        world_distance_m(first_offsets[name], second_offsets[name])
        for name in common
    ]
    finite_jumps = [jump for jump in jumps if jump is not None]
    if not finite_jumps:
        return None
    return max(finite_jumps) <= threshold


def cluster_bound_identity_observation_variants(
    eligible: list[tuple[float, float, int, str]],
    *,
    candidates: list[dict[str, Any]],
) -> list[list[tuple[float, float, int, str]]]:
    """Group redundant masks without merging distinct physical hypotheses.

    Identity-bound SAM output may contain both a clean object mask and a wider
    mask that contains the same object plus part of its support.  Candidate
    count alone would treat those nested observations as two objects and make
    the verified track disappear.  Two observations are variants only when
    their grounded XY footprints are strongly nested and their observed top
    surfaces agree in world coordinates.  Missing 3-D bounds, merely nearby
    centroids, or two adjacent objects therefore remain ambiguous.
    """

    clusters: list[list[tuple[float, float, int, str]]] = []
    for observation in eligible:
        candidate_index = observation[2]
        candidate = candidates[candidate_index]
        matching_cluster = next(
            (
                cluster
                for cluster in clusters
                if any(
                    bound_identity_observation_variant(
                        candidate,
                        candidates[member[2]],
                    )
                    for member in cluster
                )
            ),
            None,
        )
        if matching_cluster is None:
            clusters.append([observation])
        else:
            matching_cluster.append(observation)
    for cluster in clusters:
        cluster.sort()
    clusters.sort(key=lambda cluster: cluster[0])
    return clusters


def bound_identity_observation_variant(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    first_top = first.get("_top_surface_world_m")
    second_top = second.get("_top_surface_world_m")
    top_distance = world_distance_m(first_top, second_top)
    if (
        top_distance is None
        or top_distance > BOUND_IDENTITY_VARIANT_TOP_DISTANCE_M
    ):
        return False
    first_bounds = candidate_world_xy_bounds(first)
    second_bounds = candidate_world_xy_bounds(second)
    if first_bounds is None or second_bounds is None:
        return False
    first_min, first_max = first_bounds
    second_min, second_max = second_bounds
    intersection = max(
        0.0,
        min(first_max[0], second_max[0])
        - max(first_min[0], second_min[0]),
    ) * max(
        0.0,
        min(first_max[1], second_max[1])
        - max(first_min[1], second_min[1]),
    )
    first_area = max(0.0, first_max[0] - first_min[0]) * max(
        0.0,
        first_max[1] - first_min[1],
    )
    second_area = max(0.0, second_max[0] - second_min[0]) * max(
        0.0,
        second_max[1] - second_min[1],
    )
    smaller_area = min(first_area, second_area)
    if smaller_area <= 0.0:
        return False
    return (
        intersection / smaller_area
        >= BOUND_IDENTITY_VARIANT_MIN_XY_CONTAINMENT
    )


def candidate_world_xy_bounds(
    candidate: dict[str, Any],
) -> tuple[list[float], list[float]] | None:
    grounding = candidate.get("grounding_3d")
    if not isinstance(grounding, dict) or grounding.get("success") is not True:
        return None
    lower = grounding.get("bbox_world_min")
    upper = grounding.get("bbox_world_max")
    if not is_number_list(lower, length=3) or not is_number_list(
        upper,
        length=3,
    ):
        return None
    minimum = [
        min(float(lower[index]), float(upper[index]))
        for index in range(2)
    ]
    maximum = [
        max(float(lower[index]), float(upper[index]))
        for index in range(2)
    ]
    if any(maximum[index] <= minimum[index] for index in range(2)):
        return None
    return minimum, maximum


def is_number_list(value: Any, *, length: int) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except Exception:
        return False


def scale_list(values: list[float] | None, scale: float) -> list[float] | None:
    if values is None:
        return None
    return [float(item) * float(scale) for item in values]


def round_list(values: Any, *, digits: int = 6) -> list[float] | None:
    if values is None:
        return None
    return [round_float(item, digits=digits) for item in values]


def round_float(value: Any, *, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def grounding_extents_m(grounding: Any) -> list[float] | None:
    if not isinstance(grounding, dict) or not grounding.get("success"):
        return None
    bbox_min = grounding.get("bbox_world_min")
    bbox_max = grounding.get("bbox_world_max")
    if not is_number_list(bbox_min, length=3) or not is_number_list(bbox_max, length=3):
        return None
    return [abs(float(high) - float(low)) for low, high in zip(bbox_min, bbox_max)]


def grounding_world_z_max_m(grounding: Any) -> float | None:
    if not isinstance(grounding, dict) or not grounding.get("success"):
        return None
    bbox_max = grounding.get("bbox_world_max")
    if is_number_list(bbox_max, length=3):
        return float(bbox_max[2])
    candidates = [
        grounding.get("top_surface_world"),
        grounding.get("approach_point_world"),
        grounding.get("centroid_world"),
    ]
    zs = [float(value[2]) for value in candidates if is_number_list(value, length=3)]
    return max(zs) if zs else None


def score_value(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return score
