from __future__ import annotations

from typing import Any

import numpy as np

from roboharn_evo.agent.environment import capture_env_snapshot
from roboharn_evo.agent.gripper_state import normalized_gripper_target
from roboharn_evo.agent.grounded_target_pose_contract import (
    resolve_grounded_target_pose,
)
from roboharn_evo.agent.operation_candidates import (
    materialize_operation_candidate,
    normalize_grounded_point_key,
    operation_action_mode,
    operation_pose_candidates,
    select_operation_pose_candidate,
    uses_operation_pose_candidate,
    validate_place_candidate,
)

from .recovery_adapter import RecoveryCapabilities, RecoveryExecutionResult
from .tool_specs import RECOVERY_TOOLS, RecoveryToolCall, RecoveryToolResult


_GRIPPER_TOOLS = {
    "open_gripper",
    "close_gripper",
}

_EE_TOOLS = {
    "contact_displace",
    "move_ee_to_pose",
    "move_ee_to_grounded_instance",
    "retreat_arm",
    "lift_ee",
    "move_to_home",
    "safe_reset_posture",
}

_EE_TARGET_REACHED_TOLERANCE_M = 0.01
_EE_TARGET_REACHED_TOLERANCE_RAD = 0.1
_MAX_EE_CONTROL_STEPS = 20
_DUAL_ARM_MIN_EE_SEPARATION_M = 0.15
_INACTIVE_ARM_MAX_DRIFT_M = 0.005


class RMBenchRecoveryAdapter:
    def __init__(
        self,
        task_env: Any,
        *,
        oracle_objects_enabled: bool = False,
        complete_grounded_goals: bool = False,
        dual_arm_min_ee_separation_m: float = _DUAL_ARM_MIN_EE_SEPARATION_M,
        inactive_arm_max_drift_m: float = _INACTIVE_ARM_MAX_DRIFT_M,
    ) -> None:
        validated_thresholds: dict[str, float] = {}
        for label, value in (
            ("dual_arm_min_ee_separation_m", dual_arm_min_ee_separation_m),
            ("inactive_arm_max_drift_m", inactive_arm_max_drift_m),
        ):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} must be finite and positive") from exc
            if not np.isfinite(numeric_value) or numeric_value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
            validated_thresholds[label] = numeric_value
        self.task_env = task_env
        self.oracle_objects_enabled = bool(oracle_objects_enabled)
        self.complete_grounded_goals = bool(complete_grounded_goals)
        self.dual_arm_min_ee_separation_m = validated_thresholds[
            "dual_arm_min_ee_separation_m"
        ]
        self.inactive_arm_max_drift_m = validated_thresholds[
            "inactive_arm_max_drift_m"
        ]

    def capabilities(self) -> RecoveryCapabilities:
        supported_tools: set[str] = set()
        motion_modes: set[str] = set()
        gripper_api = "none"
        notes: dict[str, Any] = {}

        has_get_obs = hasattr(self.task_env, "get_obs")
        has_take_action = hasattr(self.task_env, "take_action")
        dual_arm = bool(getattr(self.task_env, "is_dual_arm", False))
        active_arm_control = not dual_arm or bool(
            getattr(self.task_env, "supports_active_arm_actions", False)
        )
        if has_get_obs:
            supported_tools.add("reobserve_scene")
        else:
            notes["missing_get_obs"] = True

        if has_take_action and has_get_obs and active_arm_control:
            if self._can_probe_gripper_indices():
                supported_tools.update(_GRIPPER_TOOLS)
                motion_modes.add("qpos")
                gripper_api = "qpos_vector"
            else:
                notes["gripper_tools_disabled"] = "robot jointState APIs are unavailable"
            supported_tools.update(_EE_TOOLS)
            motion_modes.add("ee")
        elif not has_take_action:
            notes["missing_take_action"] = True
        elif not active_arm_control:
            notes["motion_tools_disabled"] = (
                "dual-arm environment lacks active-arm action masking"
            )
        else:
            notes["motion_tools_disabled"] = "get_obs is required to verify post-action state"

        if dual_arm:
            notes["dual_arm_min_ee_separation_m"] = (
                self.dual_arm_min_ee_separation_m
            )
            notes["inactive_arm_max_drift_m"] = self.inactive_arm_max_drift_m

        supported_tools &= RECOVERY_TOOLS
        return RecoveryCapabilities(
            supported_tools=supported_tools,
            motion_modes=motion_modes,
            gripper_api=gripper_api,
            notes=notes,
        )

    def execute(self, call: RecoveryToolCall, latest_snapshot: Any | None) -> RecoveryExecutionResult:
        tool_name = call.tool_name
        try:
            if tool_name == "reobserve_scene":
                return self.reobserve_scene()
            if self._environment_success(latest_snapshot):
                return self._terminal_skip(tool_name, latest_snapshot)
            if tool_name == "open_gripper":
                return self.open_gripper(latest_snapshot, call.args)
            if tool_name == "close_gripper":
                return self.close_gripper(latest_snapshot, call.args)
            if tool_name == "move_ee_to_pose":
                return self.move_ee_to_pose(latest_snapshot, call.args)
            if tool_name == "move_ee_to_grounded_instance":
                return self.move_ee_to_grounded_instance(latest_snapshot, call.args)
            if tool_name == "contact_displace":
                return self.contact_displace(latest_snapshot, call.args)
            if tool_name == "retreat_arm":
                return self.retreat_arm(latest_snapshot, call.args)
            if tool_name == "lift_ee":
                return self.lift_ee(latest_snapshot, call.args)
            if tool_name == "move_to_home":
                return self.move_to_home(latest_snapshot, call.args)
            if tool_name == "safe_reset_posture":
                return self.safe_reset_posture(latest_snapshot, call.args)
            return self._failure(tool_name, "recovery tool not supported by RMBench adapter")
        except Exception as exc:
            return self._failure(tool_name, f"recovery tool failed: {exc}")

    def reobserve_scene(self) -> RecoveryExecutionResult:
        snapshot = self._refresh_snapshot()
        return RecoveryExecutionResult(
            result=RecoveryToolResult(
                tool_name="reobserve_scene",
                success=True,
                message="refreshed scene observation",
                details={"step_count": snapshot.step_count},
            ),
            latest_snapshot=snapshot,
        )

    def open_gripper(self, latest_snapshot: Any | None, args: dict[str, Any] | None = None) -> RecoveryExecutionResult:
        normalized_args = args or {}
        result = self._execute_gripper_qpos(
            "open_gripper",
            latest_snapshot,
            normalized_args,
            gripper_value=1.0,
        )
        release_target_id = str(
            normalized_args.get("release_target_id", "") or ""
        ).strip()
        if release_target_id:
            validation = normalized_args.get("_release_place_validation")
            candidate = normalized_args.get("_release_place_candidate")
            result.result.details.update(
                {
                    "release_target_id": release_target_id,
                    "release_held_instance_id": normalized_args.get(
                        "release_held_instance_id"
                    ),
                    "release_place_validation": (
                        dict(validation) if isinstance(validation, dict) else {}
                    ),
                    "release_place_candidate": (
                        dict(candidate) if isinstance(candidate, dict) else {}
                    ),
                }
            )
        return result

    def close_gripper(self, latest_snapshot: Any | None, args: dict[str, Any] | None = None) -> RecoveryExecutionResult:
        return self._execute_gripper_qpos("close_gripper", latest_snapshot, args or {}, gripper_value=0.0)

    def move_ee_to_pose(self, latest_snapshot: Any | None, args: dict[str, Any]) -> RecoveryExecutionResult:
        arm = self._requested_single_arm(args, tool_name="move_ee_to_pose")
        if isinstance(arm, RecoveryExecutionResult):
            return arm
        target_pose = self._target_pose_from_args(args, latest_snapshot, arm=arm, tool_name="move_ee_to_pose")
        if isinstance(target_pose, RecoveryExecutionResult):
            return target_pose
        max_translation = self._validated_distance(
            args,
            default=0.06,
            maximum=0.12,
            tool_name="move_ee_to_pose",
            arg_name="max_translation",
        )
        if isinstance(max_translation, RecoveryExecutionResult):
            return max_translation
        steps = self._validated_steps(args, tool_name="move_ee_to_pose")
        if isinstance(steps, RecoveryExecutionResult):
            return steps
        return self._execute_ee_pose_target(
            tool_name="move_ee_to_pose",
            latest_snapshot=latest_snapshot,
            arm=arm,
            target_pose=target_pose,
            max_translation=max_translation,
            steps=steps,
            message="executed bounded EE move to validated target pose",
        )

    def move_ee_to_grounded_instance(self, latest_snapshot: Any | None, args: dict[str, Any]) -> RecoveryExecutionResult:
        arm = self._requested_single_arm(args, tool_name="move_ee_to_grounded_instance")
        if isinstance(arm, RecoveryExecutionResult):
            return arm
        instance = self._resolve_scene_instance(args, tool_name="move_ee_to_grounded_instance")
        if isinstance(instance, RecoveryExecutionResult):
            return instance
        point_key = normalize_grounded_point_key(args.get("point_key", "approach_world_m"))
        action_mode = operation_action_mode(
            point_key,
            args.get("_operation_action_mode"),
        )
        all_candidates = (
            operation_pose_candidates(instance)
            if uses_operation_pose_candidate(point_key)
            else []
        )
        mode_candidates = [
            item
            for item in all_candidates
            if str(item.get("arm", "")).strip().lower() == arm
            and str(item.get("action_mode", "")).strip().lower() == action_mode
        ]
        selected_candidate = (
            select_operation_pose_candidate(
                instance,
                arm=arm,
                action_mode=action_mode,
                blocked_candidate_ids=args.get("_blocked_operation_candidate_ids"),
                requested_candidate_id=args.get("_operation_candidate_id"),
                requested_target_id=args.get("target_id"),
            )
            if all_candidates
            else None
        )
        if all_candidates and selected_candidate is None:
            return self._failure(
                "move_ee_to_grounded_instance",
                "no unblocked operation-pose candidate is available for requested arm and action mode",
                {
                    "instance_id": instance.get("instance_id"),
                    "arm": arm,
                    "point_key": point_key,
                    "operation_action_mode": action_mode,
                    "operation_target_id": str(args.get("target_id", "") or ""),
                    "available_operation_candidate_ids": [
                        item.get("candidate_id") for item in mode_candidates
                    ],
                    "blocked_operation_candidate_ids": list(
                        args.get("_blocked_operation_candidate_ids", []) or []
                    ),
                },
            )
        place_validation: dict[str, Any] = {}
        if action_mode == "place":
            target_id = str(args.get("target_id", "") or "").strip()
            if not target_id:
                return self._failure(
                    "move_ee_to_grounded_instance",
                    "place mode requires a public target_id",
                    {"instance_id": instance.get("instance_id"), "arm": arm},
                )
            if selected_candidate is None:
                return self._failure(
                    "move_ee_to_grounded_instance",
                    "place target has no executable candidate for the requested arm",
                    {
                        "instance_id": instance.get("instance_id"),
                        "arm": arm,
                        "operation_target_id": target_id,
                    },
                )
            place_validation = validate_place_candidate(
                args.get("_scene_memory", {}),
                held_instance=instance,
                candidate=selected_candidate,
            )
            if place_validation.get("valid") is not True:
                return self._failure(
                    "move_ee_to_grounded_instance",
                    "place target failed current occupancy/support revalidation",
                    {
                        "instance_id": instance.get("instance_id"),
                        "arm": arm,
                        "operation_target_id": target_id,
                        **place_validation,
                    },
                )
        selected_instance = materialize_operation_candidate(instance, selected_candidate)
        preserve_height = args.get("preserve_height", False)
        if not isinstance(preserve_height, bool):
            return self._failure(
                "move_ee_to_grounded_instance",
                "preserve_height must be a boolean",
                {"preserve_height": preserve_height},
            )
        if action_mode == "place" and preserve_height:
            return self._failure(
                "move_ee_to_grounded_instance",
                "place candidates already contain a clearance approach; preserve_height is not valid in place mode",
                {
                    "operation_target_id": args.get("target_id"),
                    "point_key": point_key,
                },
            )
        current_snapshot = latest_snapshot
        clearance_details: dict[str, Any] = {}
        if preserve_height:
            clearance = self._prepare_grounded_carry_clearance(
                latest_snapshot=current_snapshot,
                args=args,
                arm=arm,
                tool_name="move_ee_to_grounded_instance",
            )
            if isinstance(clearance, RecoveryExecutionResult):
                return clearance
            current_snapshot, clearance_details = clearance
        target_pose = self._target_pose_from_grounded_instance(
            args=args,
            latest_snapshot=current_snapshot,
            arm=arm,
            tool_name="move_ee_to_grounded_instance",
            instance=selected_instance,
        )
        if isinstance(target_pose, RecoveryExecutionResult):
            return target_pose
        max_translation = self._validated_distance(
            args,
            default=0.06,
            maximum=0.12,
            tool_name="move_ee_to_grounded_instance",
            arg_name="max_translation",
        )
        if isinstance(max_translation, RecoveryExecutionResult):
            return max_translation
        steps = self._validated_steps(args, tool_name="move_ee_to_grounded_instance")
        if isinstance(steps, RecoveryExecutionResult):
            return steps
        result = self._execute_ee_pose_target(
            tool_name="move_ee_to_grounded_instance",
            latest_snapshot=current_snapshot,
            arm=arm,
            target_pose=target_pose,
            max_translation=max_translation,
            steps=steps,
            message="executed bounded EE move to grounded scene-memory instance",
        )
        result.result.details.update(
            {
                "instance_id": str(instance.get("instance_id", "")),
                "point_key": point_key,
                "preserve_height": preserve_height,
                **place_validation,
                **clearance_details,
            }
        )
        if selected_candidate is not None:
            result.result.details.update(
                {
                    "operation_candidate_id": str(selected_candidate.get("candidate_id", "")),
                    "operation_target_id": str(selected_candidate.get("target_id", "") or ""),
                    "operation_target_kind": selected_candidate.get("target_kind"),
                    "operation_action_mode": action_mode,
                    "operation_candidate_arm": arm,
                    "operation_geometry_source": selected_candidate.get("geometry_source"),
                    "operation_source_candidate_index": selected_candidate.get(
                        "source_candidate_index"
                    ),
                    "operation_tcp_pose": selected_candidate.get("tcp_pose"),
                    "operation_object_contact_pose": selected_candidate.get(
                        "object_contact_pose"
                    ),
                    "held_object_target_world_m": selected_candidate.get(
                        "held_object_target_world_m"
                    ),
                    "held_object_to_tcp_translation_world_m": selected_candidate.get(
                        "held_object_to_tcp_translation_world_m"
                    ),
                }
            )
        return result

    def contact_displace(self, latest_snapshot: Any | None, args: dict[str, Any]) -> RecoveryExecutionResult:
        arm = self._requested_arm(args, tool_name="contact_displace")
        if isinstance(arm, RecoveryExecutionResult):
            return arm
        contact_precondition = self._validate_contact_reference(latest_snapshot, args=args, arm=arm)
        if isinstance(contact_precondition, RecoveryExecutionResult):
            return contact_precondition
        axis = self._validated_axis(args, tool_name="contact_displace", default="z")
        if isinstance(axis, RecoveryExecutionResult):
            return axis
        direction = self._validated_direction(args, tool_name="contact_displace", default="negative")
        if isinstance(direction, RecoveryExecutionResult):
            return direction
        distance = self._validated_distance(args, default=0.02, maximum=0.04, tool_name="contact_displace")
        if isinstance(distance, RecoveryExecutionResult):
            return distance
        steps = self._validated_steps(args, tool_name="contact_displace")
        if isinstance(steps, RecoveryExecutionResult):
            return steps
        signed_distance = -distance if direction == "negative" else distance
        result = self._execute_ee_displacement(
            tool_name="contact_displace",
            latest_snapshot=latest_snapshot,
            arm=arm,
            axis=axis,
            signed_distance=signed_distance,
            steps=steps,
            message="executed bounded EE contact displacement",
        )
        signature = str(
            args.get("_actual_spatial_state_signature", "") or ""
        ).strip()
        if signature:
            result.result.details["actual_spatial_state_signature"] = signature
        return result

    def _validate_contact_reference(
        self,
        latest_snapshot: Any | None,
        *,
        args: dict[str, Any],
        arm: str,
    ) -> RecoveryExecutionResult | None:
        raw_reference = args.get("_contact_reference_world_m")
        if raw_reference is None:
            return None
        reference = self._xyz_array(raw_reference)
        if reference is None:
            return self._failure(
                "contact_displace",
                "grounded contact reference must be a finite 3D point",
                {"contact_reference_world_m": raw_reference},
            )
        if arm not in {"left", "right"}:
            return self._failure(
                "contact_displace",
                "grounded contact reference requires one selected arm",
                {"arm": arm, "contact_reference_world_m": self._round_array(reference)},
            )
        try:
            tolerance = float(args.get("_contact_reference_tolerance_m", 0.01))
        except (TypeError, ValueError):
            tolerance = float("nan")
        if not np.isfinite(tolerance) or tolerance <= 0.0 or tolerance > 0.05:
            return self._failure(
                "contact_displace",
                "grounded contact reference tolerance must be within (0, 0.05] meters",
                {"contact_reference_tolerance_m": args.get("_contact_reference_tolerance_m")},
            )
        observed_pose = self._snapshot_pose(latest_snapshot, arm, "contact_displace")
        if isinstance(observed_pose, RecoveryExecutionResult):
            return observed_pose
        error = float(np.linalg.norm(reference - observed_pose[:3]))
        if error <= tolerance:
            return None
        return self._failure(
            "contact_displace",
            "grounded contact precondition failed: observed EE pose is outside contact tolerance",
            {
                "arm": arm,
                "contact_reference_world_m": self._round_array(reference),
                "contact_reference_tolerance_m": tolerance,
                "observed_pose": self._round_array(observed_pose),
                "target_observation_error_m": error,
            },
        )

    def retreat_arm(self, latest_snapshot: Any | None, args: dict[str, Any]) -> RecoveryExecutionResult:
        arm = self._requested_arm(args, tool_name="retreat_arm")
        if isinstance(arm, RecoveryExecutionResult):
            return arm
        axis = self._validated_axis(args, tool_name="retreat_arm", default="x")
        if isinstance(axis, RecoveryExecutionResult):
            return axis
        distance = self._validated_distance(args, default=0.03, maximum=0.05, tool_name="retreat_arm")
        if isinstance(distance, RecoveryExecutionResult):
            return distance
        steps = self._validated_steps(args, tool_name="retreat_arm")
        if isinstance(steps, RecoveryExecutionResult):
            return steps
        signed_by_arm: dict[str, float] | None = None
        if "direction" in args:
            direction = self._validated_direction(
                args,
                tool_name="retreat_arm",
                default="negative",
            )
            if isinstance(direction, RecoveryExecutionResult):
                return direction
            signed_distance = -distance if direction == "negative" else distance
        elif bool(getattr(self.task_env, "is_dual_arm", False)):
            axis_index = {"x": 0, "y": 1, "z": 2}[axis]
            left_pose = self._snapshot_pose(latest_snapshot, "left", "retreat_arm")
            if isinstance(left_pose, RecoveryExecutionResult):
                return left_pose
            right_pose = self._snapshot_pose(latest_snapshot, "right", "retreat_arm")
            if isinstance(right_pose, RecoveryExecutionResult):
                return right_pose
            separation_on_axis = float(
                left_pose[axis_index] - right_pose[axis_index]
            )
            if abs(separation_on_axis) <= 1e-4:
                return self._failure(
                    "retreat_arm",
                    "direction is required when the arms are not separated on the requested axis",
                    {"arm": arm, "axis": axis},
                )
            left_outward_sign = 1.0 if separation_on_axis > 0.0 else -1.0
            outward_signs = {
                "left": left_outward_sign,
                "right": -left_outward_sign,
            }
            signed_by_arm = {
                selected_arm: outward_signs[selected_arm] * distance
                for selected_arm in self._selected_arms(arm)
            }
            signed_distance = next(iter(signed_by_arm.values()))
        elif axis == "x":
            # Preserve the legacy single-arm default when no direction is given.
            signed_distance = -distance
        else:
            return self._failure(
                "retreat_arm",
                "direction is required when retreat axis is not x",
                {"arm": arm, "axis": axis},
            )
        return self._execute_ee_displacement(
            tool_name="retreat_arm",
            latest_snapshot=latest_snapshot,
            arm=arm,
            axis=axis,
            signed_distance=signed_distance,
            signed_distance_by_arm=signed_by_arm,
            steps=steps,
            message="executed bounded EE retreat displacement",
        )

    def lift_ee(self, latest_snapshot: Any | None, args: dict[str, Any] | None = None) -> RecoveryExecutionResult:
        args = args or {}
        arm = self._requested_arm(args, tool_name="lift_ee")
        if isinstance(arm, RecoveryExecutionResult):
            return arm
        distance = self._validated_distance(args, default=0.03, maximum=0.05, tool_name="lift_ee")
        if isinstance(distance, RecoveryExecutionResult):
            return distance
        steps = self._validated_steps(args, tool_name="lift_ee")
        if isinstance(steps, RecoveryExecutionResult):
            return steps
        return self._execute_ee_displacement(
            tool_name="lift_ee",
            latest_snapshot=latest_snapshot,
            arm=arm,
            axis="z",
            signed_distance=distance,
            steps=steps,
            message="executed bounded EE lift displacement",
        )

    def move_to_home(self, latest_snapshot: Any | None, args: dict[str, Any]) -> RecoveryExecutionResult:
        arm = self._requested_arm(args, tool_name="move_to_home")
        if isinstance(arm, RecoveryExecutionResult):
            return arm
        max_translation = self._validated_distance(
            args,
            default=0.05,
            maximum=0.08,
            tool_name="move_to_home",
            arg_name="max_translation",
        )
        if isinstance(max_translation, RecoveryExecutionResult):
            return max_translation
        steps = self._validated_steps(args, tool_name="move_to_home")
        if isinstance(steps, RecoveryExecutionResult):
            return steps
        return self._execute_ee_home_step(
            tool_name="move_to_home",
            latest_snapshot=latest_snapshot,
            arm=arm,
            max_translation=max_translation,
            steps=steps,
            message="executed bounded EE move toward original pose",
        )

    def safe_reset_posture(self, latest_snapshot: Any | None, args: dict[str, Any]) -> RecoveryExecutionResult:
        arm = self._requested_arm(args, tool_name="safe_reset_posture")
        if isinstance(arm, RecoveryExecutionResult):
            return arm
        max_translation = self._validated_distance(
            args,
            default=0.025,
            maximum=0.04,
            tool_name="safe_reset_posture",
            arg_name="max_translation",
        )
        if isinstance(max_translation, RecoveryExecutionResult):
            return max_translation
        steps = self._validated_steps(args, tool_name="safe_reset_posture")
        if isinstance(steps, RecoveryExecutionResult):
            return steps
        return self._execute_ee_home_step(
            tool_name="safe_reset_posture",
            latest_snapshot=latest_snapshot,
            arm=arm,
            max_translation=max_translation,
            steps=steps,
            message="executed bounded EE safe reset posture step",
        )

    def _execute_gripper_qpos(self, tool_name: str, latest_snapshot: Any | None, args: dict[str, Any], *, gripper_value: float) -> RecoveryExecutionResult:
        qpos = self._snapshot_qpos(latest_snapshot, tool_name)
        if isinstance(qpos, RecoveryExecutionResult):
            return qpos
        arm = self._requested_arm(args, tool_name=tool_name)
        if isinstance(arm, RecoveryExecutionResult):
            return arm
        gripper_indices = self._gripper_indices(qpos, arm=arm, tool_name=tool_name)
        if isinstance(gripper_indices, RecoveryExecutionResult):
            return gripper_indices
        if not gripper_indices:
            return self._failure(tool_name, "could not infer gripper slots from RMBench robot jointState", {"qpos_dim": int(qpos.size), "arm": arm})
        action = qpos.copy()
        for index in gripper_indices:
            action[index] = gripper_value
        result = self._take_qpos_action(
            action,
            tool_name,
            latest_snapshot=latest_snapshot,
            active_arm=arm,
        )
        if not result.result.success:
            return result
        result.result.message = f"set {arm} gripper qpos slots {gripper_indices} to {gripper_value}"
        observed_poses: dict[str, list[float]] = {}
        for selected_arm in self._selected_arms(arm):
            observed_pose = self._snapshot_pose(
                result.latest_snapshot,
                selected_arm,
                tool_name,
            )
            if isinstance(observed_pose, RecoveryExecutionResult):
                continue
            observed_poses[selected_arm] = self._round_array(observed_pose)
        result.result.details.update(
            {
                "arm": arm,
                "gripper_indices": gripper_indices,
                "gripper_value": gripper_value,
                "observed_poses_by_arm": observed_poses,
                **(
                    {"observed_pose": next(iter(observed_poses.values()))}
                    if len(observed_poses) == 1
                    else {}
                ),
            }
        )
        return result

    def _execute_ee_displacement(
        self,
        *,
        tool_name: str,
        latest_snapshot: Any | None,
        arm: str,
        axis: str,
        signed_distance: float,
        steps: int,
        message: str,
        signed_distance_by_arm: dict[str, float] | None = None,
    ) -> RecoveryExecutionResult:
        current_snapshot = latest_snapshot
        selected_arms = self._selected_arms(arm)
        initial_poses: dict[str, np.ndarray] = {}
        for selected_arm in selected_arms:
            initial_pose = self._snapshot_pose(current_snapshot, selected_arm, tool_name)
            if isinstance(initial_pose, RecoveryExecutionResult):
                return initial_pose
            initial_poses[selected_arm] = initial_pose.copy()
        signed_distances = (
            {
                selected_arm: float(signed_distance_by_arm[selected_arm])
                for selected_arm in selected_arms
            }
            if signed_distance_by_arm is not None
            else {selected_arm: signed_distance for selected_arm in selected_arms}
        )
        step_distances = {
            selected_arm: distance / max(1, int(steps))
            for selected_arm, distance in signed_distances.items()
        }
        result: RecoveryExecutionResult | None = None
        executed_steps = 0
        for _ in range(max(1, int(steps))):
            state = self._ee_state(current_snapshot, tool_name)
            if isinstance(state, RecoveryExecutionResult):
                return state
            axis_index = {"x": 0, "y": 1, "z": 2}[axis]
            for selected_arm in selected_arms:
                state[f"{selected_arm}_pose"][axis_index] += step_distances[
                    selected_arm
                ]
            result = self._take_ee_action(
                state,
                tool_name,
                latest_snapshot=current_snapshot,
                active_arm=arm,
            )
            current_snapshot = result.latest_snapshot
            executed_steps += 1
            if not result.result.success:
                return result
            if self._environment_success(current_snapshot):
                break
        if result is None:
            return self._failure(tool_name, "no EE displacement step was executed")
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        if result.result.success:
            observed_poses: dict[str, list[float]] = {}
            observed_displacements: dict[str, list[float]] = {}
            observed_distances: dict[str, float] = {}
            observed_axis_displacements: dict[str, float] = {}
            for selected_arm in selected_arms:
                observed_pose = self._snapshot_pose(result.latest_snapshot, selected_arm, tool_name)
                if isinstance(observed_pose, RecoveryExecutionResult):
                    continue
                displacement = observed_pose[:3] - initial_poses[selected_arm][:3]
                observed_poses[selected_arm] = self._round_array(observed_pose)
                observed_displacements[selected_arm] = self._round_array(displacement)
                observed_distances[selected_arm] = float(np.linalg.norm(displacement))
                observed_axis_displacements[selected_arm] = float(displacement[axis_index])
            result.result.message = message
            result.result.details.update(
                {
                    "arm": arm,
                    "selected_arms": selected_arms,
                    "axis": axis,
                    "signed_distance": signed_distance,
                    "signed_distances_by_arm": signed_distances,
                    "steps": int(steps),
                    "executed_steps": executed_steps,
                    "step_distances_by_arm": step_distances,
                    **(
                        {"step_distance": next(iter(step_distances.values()))}
                        if len(selected_arms) == 1
                        else {}
                    ),
                    "environment_success": self._environment_success(result.latest_snapshot),
                    "start_poses": {
                        selected_arm: self._round_array(initial_poses[selected_arm]) for selected_arm in selected_arms
                    },
                    "observed_poses": observed_poses,
                    "observed_displacements_xyz": observed_displacements,
                    "observed_displacements_m": observed_distances,
                    "observed_axis_displacements_m": observed_axis_displacements,
                }
            )
            if len(selected_arms) == 1:
                selected_arm = selected_arms[0]
                result.result.details.update(
                    {
                        "start_pose": self._round_array(initial_poses[selected_arm]),
                        "observed_pose": observed_poses.get(selected_arm),
                        "observed_displacement_xyz": observed_displacements.get(selected_arm),
                        "observed_displacement_m": observed_distances.get(selected_arm),
                        "observed_axis_displacement_m": observed_axis_displacements.get(selected_arm),
                    }
                )
        return result

    def _execute_ee_home_step(
        self,
        *,
        tool_name: str,
        latest_snapshot: Any | None,
        arm: str,
        max_translation: float,
        steps: int,
        message: str,
    ) -> RecoveryExecutionResult:
        current_snapshot = latest_snapshot
        selected_arms = self._selected_arms(arm)
        applied: dict[str, float] = {}
        result: RecoveryExecutionResult | None = None
        executed_steps = 0
        for _ in range(max(1, int(steps))):
            state = self._ee_state(current_snapshot, tool_name)
            if isinstance(state, RecoveryExecutionResult):
                return state
            step_applied = 0.0
            for selected_arm in selected_arms:
                current_pose = state[f"{selected_arm}_pose"]
                target_pose = self._original_pose(selected_arm, tool_name)
                if isinstance(target_pose, RecoveryExecutionResult):
                    return target_pose
                delta = target_pose[:3] - current_pose[:3]
                distance = float(np.linalg.norm(delta))
                if distance > max_translation and distance > 0.0:
                    delta = delta / distance * max_translation
                current_pose[:3] = current_pose[:3] + delta
                applied[selected_arm] = applied.get(selected_arm, 0.0) + float(np.linalg.norm(delta))
                step_applied = max(step_applied, float(np.linalg.norm(delta)))
            result = self._take_ee_action(
                state,
                tool_name,
                latest_snapshot=current_snapshot,
                active_arm=arm,
            )
            current_snapshot = result.latest_snapshot
            executed_steps += 1
            if not result.result.success:
                return result
            if self._environment_success(current_snapshot):
                break
            if step_applied <= 1e-6:
                break
        if result is None:
            return self._failure(tool_name, "no EE home step was executed")
        if result.result.success:
            result.result.message = message
            result.result.details.update(
                {
                    "arm": arm,
                    "selected_arms": selected_arms,
                    "max_translation": max_translation,
                    "applied_translation": applied,
                    "steps": int(steps),
                    "executed_steps": executed_steps,
                    "environment_success": self._environment_success(result.latest_snapshot),
                }
            )
        return result

    def _execute_ee_pose_target(
        self,
        *,
        tool_name: str,
        latest_snapshot: Any | None,
        arm: str,
        target_pose: np.ndarray,
        max_translation: float,
        steps: int,
        message: str,
    ) -> RecoveryExecutionResult:
        current_snapshot = latest_snapshot
        requested_translation = 0.0
        executed_pose = target_pose.copy()
        executed_steps = 0
        target_error_history: list[float] = []
        orientation_error_history: list[float] = []
        complete_goal = self.complete_grounded_goals and tool_name == "move_ee_to_grounded_instance"
        step_budget = _MAX_EE_CONTROL_STEPS if complete_goal else max(1, int(steps))
        no_progress_steps = 0
        stop_reason = "step_budget"
        result: RecoveryExecutionResult | None = None
        initial_pose: np.ndarray | None = None
        for _ in range(step_budget):
            if (complete_goal and getattr(current_snapshot, "step_limit", 0) > 0
                    and current_snapshot.step_count >= current_snapshot.step_limit):
                stop_reason = "episode_step_limit"
                break
            state = self._ee_state(current_snapshot, tool_name)
            if isinstance(state, RecoveryExecutionResult):
                return state
            current_pose = state[f"{arm}_pose"]
            if initial_pose is None:
                initial_pose = current_pose.copy()
            bounded_pose = target_pose.copy()
            delta = bounded_pose[:3] - current_pose[:3]
            distance = float(np.linalg.norm(delta))
            orientation_before = self._pose_orientation_error(target_pose, current_pose)
            requested_translation = max(requested_translation, distance)
            if distance > max_translation and distance > 0.0:
                bounded_pose[:3] = current_pose[:3] + delta / distance * max_translation
            state[f"{arm}_pose"] = bounded_pose
            result = self._take_ee_action(
                state,
                tool_name,
                latest_snapshot=current_snapshot,
                active_arm=arm,
            )
            current_snapshot = result.latest_snapshot
            executed_pose = bounded_pose.copy()
            executed_steps += 1
            if not result.result.success:
                return result
            observed_pose = self._snapshot_pose(current_snapshot, arm, tool_name)
            if isinstance(observed_pose, RecoveryExecutionResult):
                return observed_pose
            target_error = float(np.linalg.norm(target_pose[:3] - observed_pose[:3]))
            orientation_error = self._pose_orientation_error(target_pose, observed_pose)
            target_error_history.append(target_error)
            orientation_error_history.append(orientation_error)
            if self._environment_success(current_snapshot):
                stop_reason = "environment_success"
                break
            if (target_error <= _EE_TARGET_REACHED_TOLERANCE_M
                    and (not complete_goal or orientation_error <= _EE_TARGET_REACHED_TOLERANCE_RAD)):
                stop_reason = "target_reached"
                break
            if complete_goal:
                progressing = distance - target_error > 1e-4 or orientation_before - orientation_error > 1e-3
                no_progress_steps = 0 if progressing else no_progress_steps + 1
                if no_progress_steps >= 3:
                    stop_reason = "no_progress"
                    break
        if result is None:
            return self._failure(tool_name, "no EE pose-target step was executed",
                                 {"local_goal_stop_reason": stop_reason})
        if result.result.success:
            observed_pose = self._snapshot_pose(result.latest_snapshot, arm, tool_name)
            observed_details: dict[str, Any] = {}
            if not isinstance(observed_pose, RecoveryExecutionResult):
                target_observation_error = float(np.linalg.norm(target_pose[:3] - observed_pose[:3]))
                target_orientation_error = self._pose_orientation_error(target_pose, observed_pose)
                observed_displacement = (
                    observed_pose[:3] - initial_pose[:3]
                    if initial_pose is not None
                    else None
                )
                observed_details = {
                    "observed_pose": self._round_array(observed_pose),
                    "target_observation_error_m": target_observation_error,
                    "target_reached": (target_observation_error <= _EE_TARGET_REACHED_TOLERANCE_M
                                       and (not complete_goal or target_orientation_error <= _EE_TARGET_REACHED_TOLERANCE_RAD)),
                    "target_reached_tolerance_m": _EE_TARGET_REACHED_TOLERANCE_M,
                    **({"target_orientation_error_rad": target_orientation_error,
                        "target_orientation_tolerance_rad": _EE_TARGET_REACHED_TOLERANCE_RAD}
                       if complete_goal else {}),
                    **(
                        {
                            "start_pose": self._round_array(initial_pose),
                            "observed_displacement_xyz": self._round_array(
                                observed_displacement
                            ),
                            "observed_displacement_m": float(
                                np.linalg.norm(observed_displacement)
                            ),
                        }
                        if initial_pose is not None
                        and observed_displacement is not None
                        else {}
                    ),
                }
            result.result.message = message
            result.result.details.update(
                {
                    "arm": arm,
                    "target_pose": self._round_array(target_pose),
                    "executed_pose": self._round_array(executed_pose),
                    "requested_translation": requested_translation,
                    "max_translation": float(max_translation),
                    "steps": int(steps),
                    "executed_steps": executed_steps,
                    "target_error_history_m": [round(value, 6) for value in target_error_history],
                    **({"local_goal_control": True, "local_goal_step_budget": step_budget,
                        "local_goal_stop_reason": stop_reason,
                        "target_orientation_error_history_rad": [round(v, 6) for v in orientation_error_history]}
                       if complete_goal else {}),
                    "environment_success": self._environment_success(result.latest_snapshot),
                    **observed_details,
                }
            )
        return result

    @staticmethod
    def _pose_orientation_error(target: np.ndarray, observed: np.ndarray) -> float:
        target_quat, observed_quat = target[3:7], observed[3:7]
        cosine = abs(float(np.dot(target_quat, observed_quat))) / float(
            np.linalg.norm(target_quat) * np.linalg.norm(observed_quat)
        )
        return float(2.0 * np.arccos(np.clip(cosine, 0.0, 1.0)))

    def _target_pose_from_args(
        self,
        args: dict[str, Any],
        latest_snapshot: Any | None,
        *,
        arm: str,
        tool_name: str,
    ) -> np.ndarray | RecoveryExecutionResult:
        raw_pose = args.get("target_pose")
        if raw_pose is not None:
            pose = self._pose_array(raw_pose)
            if pose is None:
                return self._failure(tool_name, "target_pose must be a finite 7D xyz+quat_wxyz pose")
            return pose

        raw_xyz = args.get("target_xyz", args.get("position"))
        if raw_xyz is None:
            return self._failure(tool_name, "move_ee_to_pose requires target_pose or target_xyz")
        xyz = self._xyz_array(raw_xyz)
        if xyz is None:
            return self._failure(tool_name, "target_xyz must be a finite 3D world position", {"target_xyz": raw_xyz})

        raw_quat = args.get("target_quat_wxyz", args.get("quat_wxyz"))
        if raw_quat is None or str(raw_quat).strip().lower() in {"", "preserve", "current"}:
            quat = self._snapshot_pose(latest_snapshot, arm, tool_name)
            if isinstance(quat, RecoveryExecutionResult):
                return quat
            quat = quat[3:7]
        else:
            quat = self._quat_array(raw_quat)
            if quat is None:
                return self._failure(tool_name, "target_quat_wxyz must be a finite 4D quaternion", {"target_quat_wxyz": raw_quat})
        return np.concatenate([xyz, quat]).astype(np.float32)

    def _target_pose_from_grounded_instance(
        self,
        *,
        args: dict[str, Any],
        latest_snapshot: Any | None,
        arm: str,
        tool_name: str,
        instance: dict[str, Any] | None = None,
    ) -> np.ndarray | RecoveryExecutionResult:
        resolved_instance: dict[str, Any] | RecoveryExecutionResult
        if instance is None:
            resolved_instance = self._resolve_scene_instance(args, tool_name=tool_name)
            if isinstance(resolved_instance, RecoveryExecutionResult):
                return resolved_instance
            instance = resolved_instance
        contract_args: dict[str, Any] = {
            "instance": instance,
            "point_key": args.get("point_key", "approach_world_m"),
            "preserve_height": args.get("preserve_height", False),
            "current_pose": None,
        }
        # Preserve ``dict.get`` alias semantics, including the distinction
        # between an omitted value and an explicitly supplied ``None``.
        for key in ("offset_xyz", "target_quat_wxyz", "quat_wxyz"):
            if key in args:
                contract_args[key] = args[key]

        resolution = resolve_grounded_target_pose(**contract_args)
        if (
            resolution.error is not None
            and resolution.error.code == "invalid_current_pose"
        ):
            current_pose = self._snapshot_pose(
                latest_snapshot,
                arm,
                tool_name,
            )
            if isinstance(current_pose, RecoveryExecutionResult):
                return current_pose
            contract_args["current_pose"] = current_pose
            resolution = resolve_grounded_target_pose(**contract_args)

        if resolution.error is not None:
            details = dict(resolution.error.details)
            # Keep the adapter's established public failure payload while the
            # contract retains richer machine-readable diagnostics.
            if resolution.error.code == "missing_grounded_point":
                details.pop("point_key", None)
            elif resolution.error.code == "missing_grounded_quaternion":
                details.pop("grounded_quaternion_key", None)
            return self._failure(
                tool_name,
                resolution.error.message,
                details,
            )

        assert resolution.target_pose is not None
        return resolution.target_pose

    def _prepare_grounded_carry_clearance(
        self,
        *,
        latest_snapshot: Any | None,
        args: dict[str, Any],
        arm: str,
        tool_name: str,
    ) -> tuple[Any | None, dict[str, Any]] | RecoveryExecutionResult:
        # Aperture is not attachment evidence; this layer resolves carry geometry.
        target_instance = self._resolve_scene_instance(args, tool_name=tool_name)
        if isinstance(target_instance, RecoveryExecutionResult):
            return target_instance
        held_args: dict[str, Any] = {
            "_scene_memory": args.get("_scene_memory"),
            "role": str(args.get("held_role", "tool") or "tool"),
            "focus_key": str(args.get("held_focus_key", "tool_instances") or "tool_instances"),
        }
        held_instance_id = str(args.get("held_instance_id", args.get("held_instance_ref", "")) or "").strip()
        if held_instance_id:
            held_args["instance_id"] = held_instance_id
        held_instance = self._resolve_scene_instance(held_args, tool_name=tool_name)
        if isinstance(held_instance, RecoveryExecutionResult):
            return held_instance
        if str(held_instance.get("instance_id", "")) == str(target_instance.get("instance_id", "")):
            return self._failure(tool_name, "held and target scene instances must be different")

        held_world = self._instance_current_world(held_instance)
        if held_world is None:
            return self._failure(tool_name, "held scene instance has no finite current world position")
        held_quality = held_instance.get("quality") if isinstance(held_instance.get("quality"), dict) else {}
        held_extent = self._xyz_array(held_quality.get("world_extent_m"))
        if held_extent is not None:
            held_bottom_z = float(held_world[2] - abs(float(held_extent[2])) / 2.0)
        else:
            held_top = self._instance_world_z_max(held_instance)
            if held_top is None or held_top < float(held_world[2]):
                return self._failure(tool_name, "held scene instance has no finite vertical extent for carry clearance")
            held_bottom_z = float(held_world[2] - (held_top - float(held_world[2])))

        target_top_z = self._instance_world_z_max(target_instance)
        if target_top_z is None:
            return self._failure(tool_name, "target scene instance has no finite top surface for carry clearance")
        clearance_margin = self._validated_distance(
            args,
            default=0.015,
            maximum=0.05,
            tool_name=tool_name,
            arg_name="clearance_margin",
        )
        if isinstance(clearance_margin, RecoveryExecutionResult):
            return clearance_margin
        max_clearance_lift = self._validated_distance(
            args,
            default=0.08,
            maximum=0.12,
            tool_name=tool_name,
            arg_name="max_clearance_lift",
        )
        if isinstance(max_clearance_lift, RecoveryExecutionResult):
            return max_clearance_lift
        lift_distance = max(0.0, target_top_z + clearance_margin - held_bottom_z)
        if lift_distance > max_clearance_lift + 1e-6:
            return self._failure(
                tool_name,
                "required carry clearance lift exceeds configured maximum",
                {
                    "required_lift": lift_distance,
                    "max_clearance_lift": max_clearance_lift,
                    "held_bottom_world_z": held_bottom_z,
                    "target_top_world_z": target_top_z,
                },
            )

        current_snapshot = latest_snapshot
        if lift_distance > 1e-4:
            clearance_steps = self._validated_steps(
                {"steps": args.get("clearance_steps", 2)},
                tool_name=tool_name,
            )
            if isinstance(clearance_steps, RecoveryExecutionResult):
                return clearance_steps
            lift_result = self._execute_ee_displacement(
                tool_name=tool_name,
                latest_snapshot=current_snapshot,
                arm=arm,
                axis="z",
                signed_distance=lift_distance,
                steps=clearance_steps,
                message="raised held object to validated carry clearance",
            )
            if not lift_result.result.success:
                return lift_result
            current_snapshot = lift_result.latest_snapshot

        return current_snapshot, {
            "held_instance_id": held_instance.get("instance_id"),
            "target_instance_id": target_instance.get("instance_id"),
            "held_bottom_world_z": round(held_bottom_z, 6),
            "target_top_world_z": round(target_top_z, 6),
            "clearance_margin": round(float(clearance_margin), 6),
            "clearance_lift": round(float(lift_distance), 6),
        }

    def _instance_current_world(self, instance: dict[str, Any]) -> np.ndarray | None:
        for key in ("latest_world_m", "world_m"):
            value = self._xyz_array(instance.get(key))
            if value is not None:
                return value
        return None

    def _instance_world_z_max(self, instance: dict[str, Any]) -> float | None:
        quality = instance.get("quality") if isinstance(instance.get("quality"), dict) else {}
        raw_value = quality.get("world_z_max_m")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            return value
        top_surface = self._xyz_array(instance.get("top_surface_world_m"))
        if top_surface is not None:
            return float(top_surface[2])
        return None

    def _ee_state(self, latest_snapshot: Any | None, tool_name: str) -> dict[str, Any] | RecoveryExecutionResult:
        left_pose = self._snapshot_pose(latest_snapshot, "left", tool_name)
        if isinstance(left_pose, RecoveryExecutionResult):
            return left_pose
        left_gripper = self._snapshot_gripper(latest_snapshot, "left", tool_name)
        if isinstance(left_gripper, RecoveryExecutionResult):
            return left_gripper
        state: dict[str, Any] = {"left_pose": left_pose, "left_gripper": left_gripper}
        if bool(getattr(self.task_env, "is_dual_arm", False)):
            right_pose = self._snapshot_pose(latest_snapshot, "right", tool_name)
            if isinstance(right_pose, RecoveryExecutionResult):
                return right_pose
            right_gripper = self._snapshot_gripper(latest_snapshot, "right", tool_name)
            if isinstance(right_gripper, RecoveryExecutionResult):
                return right_gripper
            state.update({"right_pose": right_pose, "right_gripper": right_gripper})
        return state

    def _dual_arm_action_supported(self) -> bool:
        return not bool(getattr(self.task_env, "is_dual_arm", False)) or bool(
            getattr(self.task_env, "supports_active_arm_actions", False)
        )

    def _dual_arm_separation_audit(
        self,
        *,
        state: dict[str, Any],
        latest_snapshot: Any | None,
        active_arm: str,
        tool_name: str,
    ) -> dict[str, Any] | RecoveryExecutionResult:
        if not bool(getattr(self.task_env, "is_dual_arm", False)):
            return {}
        left_start = self._snapshot_pose(latest_snapshot, "left", tool_name)
        if isinstance(left_start, RecoveryExecutionResult):
            return left_start
        right_start = self._snapshot_pose(latest_snapshot, "right", tool_name)
        if isinstance(right_start, RecoveryExecutionResult):
            return right_start
        left_target = (
            np.asarray(state["left_pose"], dtype=np.float32)
            if active_arm in {"left", "both"}
            else left_start
        )
        right_target = (
            np.asarray(state["right_pose"], dtype=np.float32)
            if active_arm in {"right", "both"}
            else right_start
        )
        relative_start = left_start[:3] - right_start[:3]
        relative_delta = (
            left_target[:3]
            - left_start[:3]
            - (right_target[:3] - right_start[:3])
        )
        denominator = float(np.dot(relative_delta, relative_delta))
        interpolation = (
            0.0
            if denominator <= 1e-12
            else float(
                np.clip(
                    -np.dot(relative_start, relative_delta) / denominator,
                    0.0,
                    1.0,
                )
            )
        )
        current_separation = float(np.linalg.norm(relative_start))
        target_separation = float(
            np.linalg.norm(left_target[:3] - right_target[:3])
        )
        minimum_separation = float(
            np.linalg.norm(relative_start + interpolation * relative_delta)
        )
        required = self.dual_arm_min_ee_separation_m
        permitted_floor = min(required, current_separation)
        unsafe = minimum_separation < permitted_floor - 1e-6
        if current_separation < required:
            unsafe = unsafe or target_separation <= current_separation + 1e-4
        audit = {
            "active_arm": active_arm,
            "current_ee_separation_m": current_separation,
            "target_ee_separation_m": target_separation,
            "minimum_swept_ee_separation_m": minimum_separation,
            "required_ee_separation_m": required,
        }
        if unsafe:
            return self._failure(
                tool_name,
                "dual-arm separation guard rejected the requested EE motion",
                audit,
            )
        return audit

    def _inactive_arm_poses(
        self,
        latest_snapshot: Any | None,
        *,
        active_arm: str,
        tool_name: str,
    ) -> dict[str, np.ndarray] | RecoveryExecutionResult:
        if not bool(getattr(self.task_env, "is_dual_arm", False)):
            return {}
        inactive = (
            {"right"}
            if active_arm == "left"
            else {"left"}
            if active_arm == "right"
            else set()
        )
        result: dict[str, np.ndarray] = {}
        for arm in inactive:
            pose = self._snapshot_pose(latest_snapshot, arm, tool_name)
            if isinstance(pose, RecoveryExecutionResult):
                return pose
            result[arm] = pose
        return result

    def _inactive_arm_drift(
        self,
        *,
        start_poses: dict[str, np.ndarray],
        latest_snapshot: Any | None,
        tool_name: str,
    ) -> dict[str, float] | RecoveryExecutionResult:
        drift: dict[str, float] = {}
        for arm, start in start_poses.items():
            observed = self._snapshot_pose(latest_snapshot, arm, tool_name)
            if isinstance(observed, RecoveryExecutionResult):
                return observed
            drift[arm] = float(np.linalg.norm(observed[:3] - start[:3]))
        return drift

    def _take_ee_action(
        self,
        state: dict[str, Any],
        tool_name: str,
        *,
        latest_snapshot: Any | None,
        active_arm: str,
    ) -> RecoveryExecutionResult:
        if self._environment_success(latest_snapshot):
            return self._terminal_skip(tool_name, latest_snapshot)
        if not hasattr(self.task_env, "take_action"):
            return self._failure(tool_name, "task_env.take_action is unavailable")
        if not self._dual_arm_action_supported():
            return self._failure(
                tool_name,
                "dual-arm environment lacks active-arm action masking",
                {"active_arm": active_arm},
            )
        separation = self._dual_arm_separation_audit(
            state=state,
            latest_snapshot=latest_snapshot,
            active_arm=active_arm,
            tool_name=tool_name,
        )
        if isinstance(separation, RecoveryExecutionResult):
            return separation
        inactive_start = self._inactive_arm_poses(
            latest_snapshot,
            active_arm=active_arm,
            tool_name=tool_name,
        )
        if isinstance(inactive_start, RecoveryExecutionResult):
            return inactive_start
        action = self._build_ee_action(state)
        if bool(getattr(self.task_env, "is_dual_arm", False)):
            self.task_env.take_action(
                action.astype(np.float32),
                action_type="ee",
                active_arm=active_arm,
            )
        else:
            self.task_env.take_action(action.astype(np.float32), action_type="ee")
        snapshot = self._refresh_snapshot()
        inactive_drift = self._inactive_arm_drift(
            start_poses=inactive_start,
            latest_snapshot=snapshot,
            tool_name=tool_name,
        )
        if isinstance(inactive_drift, RecoveryExecutionResult):
            return inactive_drift
        excessive_drift = {
            arm: value
            for arm, value in inactive_drift.items()
            if value > self.inactive_arm_max_drift_m
        }
        if excessive_drift:
            return RecoveryExecutionResult(
                result=RecoveryToolResult(
                    tool_name=tool_name,
                    success=False,
                    message="inactive arm moved during a single-arm EE action",
                    details={
                        **separation,
                        "inactive_arm_drift_m": inactive_drift,
                        "inactive_arm_max_drift_m": (
                            self.inactive_arm_max_drift_m
                        ),
                    },
                ),
                latest_snapshot=snapshot,
            )
        return RecoveryExecutionResult(
            result=RecoveryToolResult(
                tool_name=tool_name,
                success=True,
                message="executed EE recovery action",
                details={
                    "step_count": snapshot.step_count,
                    "ee_action_dim": int(action.size),
                    "environment_success": self._environment_success(snapshot),
                    **separation,
                    "inactive_arm_drift_m": inactive_drift,
                    "inactive_arm_max_drift_m": self.inactive_arm_max_drift_m,
                },
            ),
            latest_snapshot=snapshot,
        )

    def _build_ee_action(self, state: dict[str, Any]) -> np.ndarray:
        left = np.concatenate([state["left_pose"], np.array([state["left_gripper"]], dtype=np.float32)])
        if bool(getattr(self.task_env, "is_dual_arm", False)):
            right = np.concatenate([state["right_pose"], np.array([state["right_gripper"]], dtype=np.float32)])
            return np.concatenate([left, right]).astype(np.float32)
        return left.astype(np.float32)

    def _snapshot_pose(self, latest_snapshot: Any | None, arm: str, tool_name: str) -> np.ndarray | RecoveryExecutionResult:
        if latest_snapshot is None:
            return self._failure(tool_name, "latest snapshot unavailable for EE recovery")
        pose = getattr(latest_snapshot, f"{arm}_endpose", None)
        if pose is None:
            return self._failure(tool_name, f"latest snapshot has no {arm}_endpose")
        pose_array = np.asarray(pose, dtype=np.float32).copy()
        if pose_array.shape != (7,):
            return self._failure(tool_name, f"{arm}_endpose must be a 7D xyz+quat pose", {"shape": pose_array.shape})
        if not np.all(np.isfinite(pose_array)):
            return self._failure(tool_name, f"{arm}_endpose contains non-finite values")
        quat_norm = float(np.linalg.norm(pose_array[3:]))
        if quat_norm <= 1e-6:
            return self._failure(tool_name, f"{arm}_endpose quaternion has near-zero norm")
        pose_array[3:] = pose_array[3:] / quat_norm
        return pose_array

    def _snapshot_gripper(self, latest_snapshot: Any | None, arm: str, tool_name: str) -> float | RecoveryExecutionResult:
        if latest_snapshot is None:
            return self._failure(tool_name, "latest snapshot unavailable for gripper preservation")
        if arm in latest_snapshot.gripper_command_by_arm:
            return normalized_gripper_target(latest_snapshot.gripper_command_by_arm[arm])
        raw = getattr(latest_snapshot, "raw", {}) or {}
        endpose = raw.get("endpose", {}) if isinstance(raw, dict) else {}
        if arm == "left":
            raw_value = endpose.get("left_gripper", endpose.get("gripper"))
        else:
            raw_value = endpose.get("right_gripper")
        if raw_value is not None:
            value = float(raw_value)
            if np.isfinite(value):
                return float(np.clip(value, 0.0, 1.0))
        qpos = self._snapshot_qpos(latest_snapshot, tool_name)
        if isinstance(qpos, RecoveryExecutionResult):
            return qpos
        indices = self._gripper_indices(qpos, arm=arm, tool_name=tool_name)
        if isinstance(indices, RecoveryExecutionResult):
            return indices
        if len(indices) != 1:
            return self._failure(tool_name, "could not identify a single gripper value to preserve", {"arm": arm, "indices": indices})
        return float(np.clip(qpos[indices[0]], 0.0, 1.0))

    def _original_pose(self, arm: str, tool_name: str) -> np.ndarray | RecoveryExecutionResult:
        robot = getattr(self.task_env, "robot", None)
        if robot is None:
            return self._failure(tool_name, "task_env.robot is unavailable for original EE pose")
        attr_name = f"{arm}_original_pose"
        if hasattr(robot, attr_name):
            pose = self._pose_array(getattr(robot, attr_name))
            if pose is not None:
                return pose
        fn_name = f"get_{arm}_orig_endpose"
        fn = getattr(robot, fn_name, None)
        if callable(fn):
            pose = self._pose_array(fn())
            if pose is not None:
                return pose
        return self._failure(tool_name, f"robot original EE pose is unavailable for {arm} arm")

    def _pose_array(self, pose: Any) -> np.ndarray | None:
        if hasattr(pose, "p") and hasattr(pose, "q"):
            pose = list(pose.p) + list(pose.q)
        pose_array = np.asarray(pose, dtype=np.float32)
        if pose_array.shape != (7,) or not np.all(np.isfinite(pose_array)):
            return None
        quat_norm = float(np.linalg.norm(pose_array[3:]))
        if quat_norm <= 1e-6:
            return None
        pose_array = pose_array.copy()
        pose_array[3:] = pose_array[3:] / quat_norm
        return pose_array

    def _xyz_array(self, value: Any) -> np.ndarray | None:
        try:
            xyz = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            return None
        return xyz.copy()

    def _quat_array(self, value: Any) -> np.ndarray | None:
        try:
            quat = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            return None
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-6:
            return None
        return quat / norm

    def _round_array(self, value: Any) -> list[float]:
        return [round(float(item), 6) for item in np.asarray(value, dtype=np.float64).reshape(-1)]

    def _resolve_scene_instance(self, args: dict[str, Any], *, tool_name: str) -> dict[str, Any] | RecoveryExecutionResult:
        scene_memory = args.get("_scene_memory")
        if not isinstance(scene_memory, dict):
            return self._failure(tool_name, "scene_memory context is unavailable for grounded recovery")
        instances = scene_memory.get("instances")
        if not isinstance(instances, list) or not instances:
            return self._failure(tool_name, "scene_memory contains no instances for grounded recovery")

        def identity_refs(item: dict[str, Any]) -> set[str]:
            return {
                str(item.get(key, "") or "").strip().lower()
                for key in ("instance_id", "track_id", "oracle_id", "oracle_source_path")
                if str(item.get(key, "") or "").strip()
            }

        instance_ref = str(args.get("instance_id") or args.get("instance_ref") or "").strip()
        if instance_ref:
            normalized_ref = instance_ref.lower()
            matches = [
                item for item in instances
                if isinstance(item, dict) and normalized_ref in identity_refs(item)
            ]
            if len(matches) == 1:
                return dict(matches[0])
            return self._failure(
                tool_name,
                "explicit scene instance reference did not resolve uniquely",
                {
                    "instance_ref": instance_ref,
                    "match_count": len(matches),
                },
            )

        focus_key = str(args.get("focus_key", "")).strip().lower()
        if not focus_key:
            role = str(args.get("role", "target")).strip().lower()
            focus_key = "tool_instances" if role == "tool" else "target_instances"
        task_focus = scene_memory.get("task_focus", {})
        identity_binding_required = bool(
            isinstance(task_focus, dict) and task_focus.get("identity_binding_required")
        )
        if identity_binding_required:
            return self._failure(
                tool_name,
                "identity-bound grounded recovery requires an explicit instance_id or instance_ref",
                {"instance_ref": "", "focus_key": focus_key},
            )
        focused_ids = task_focus.get(focus_key) if isinstance(task_focus, dict) else None
        if isinstance(focused_ids, list):
            normalized_ids = {
                str(focused_id or "").strip().lower()
                for focused_id in focused_ids
                if str(focused_id or "").strip()
            }
            if len(normalized_ids) != 1:
                return self._failure(
                    tool_name,
                    "scene-memory focus is ambiguous; provide an explicit instance_id or instance_ref",
                    {
                        "instance_ref": "",
                        "focus_key": focus_key,
                        "focused_instance_count": len(normalized_ids),
                    },
                )
            focused_ref = next(iter(normalized_ids))
            matches = [
                item
                for item in instances
                if isinstance(item, dict) and focused_ref in identity_refs(item)
            ]
            if len(matches) == 1:
                return dict(matches[0])
            return self._failure(
                tool_name,
                "scene-memory focus did not resolve uniquely",
                {
                    "instance_ref": "",
                    "focus_key": focus_key,
                    "focused_instance_ref": focused_ref,
                    "match_count": len(matches),
                },
            )
        return self._failure(
            tool_name,
            "could not resolve scene instance for grounded recovery",
            {"instance_ref": instance_ref, "focus_key": focus_key},
        )

    def _selected_arms(self, arm: str) -> list[str]:
        if arm == "both":
            return ["left", "right"] if bool(getattr(self.task_env, "is_dual_arm", False)) else ["left"]
        return [arm]

    def _requested_arm(self, args: dict[str, Any], *, tool_name: str) -> str | RecoveryExecutionResult:
        if "arm" not in args or not str(args.get("arm") or "").strip():
            return self._failure(tool_name, "arm is required for physical recovery tools")
        arm = str(args.get("arm")).strip().lower()
        aliases = {"all": "both", "dual": "both", "left_arm": "left", "right_arm": "right"}
        arm = aliases.get(arm, arm)
        if arm not in {"left", "right", "both"}:
            return self._failure(tool_name, "arm must be left, right, or both", {"arm": args.get("arm")})
        if arm == "right" and not bool(getattr(self.task_env, "is_dual_arm", False)):
            return self._failure(tool_name, "right arm requested for non-dual-arm RMBench env", {"arm": arm})
        return arm

    def _requested_single_arm(self, args: dict[str, Any], *, tool_name: str) -> str | RecoveryExecutionResult:
        arm = self._requested_arm(args, tool_name=tool_name)
        if isinstance(arm, RecoveryExecutionResult):
            return arm
        if arm == "both":
            return self._failure(tool_name, "grounded or absolute EE pose tools require arm to be left or right", {"arm": arm})
        return arm

    def _validated_axis(self, args: dict[str, Any], *, tool_name: str, default: str) -> str | RecoveryExecutionResult:
        axis = str(args.get("axis", default)).strip().lower()
        if axis not in {"x", "y", "z"}:
            return self._failure(tool_name, "axis must be x, y, or z", {"axis": args.get("axis")})
        return axis

    def _validated_direction(self, args: dict[str, Any], *, tool_name: str, default: str) -> str | RecoveryExecutionResult:
        direction = str(args.get("direction", default)).strip().lower()
        aliases = {"+": "positive", "pos": "positive", "-": "negative", "neg": "negative"}
        direction = aliases.get(direction, direction)
        if direction not in {"positive", "negative"}:
            return self._failure(tool_name, "direction must be positive or negative", {"direction": args.get("direction")})
        return direction

    def _validated_distance(
        self,
        args: dict[str, Any],
        *,
        default: float,
        maximum: float,
        tool_name: str,
        arg_name: str = "distance",
    ) -> float | RecoveryExecutionResult:
        raw_value = args.get(arg_name, default)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return self._failure(tool_name, f"{arg_name} must be a finite positive number", {arg_name: raw_value})
        if not np.isfinite(value) or value <= 0.0:
            return self._failure(tool_name, f"{arg_name} must be a finite positive number", {arg_name: raw_value})
        return min(value, maximum)

    def _validated_steps(self, args: dict[str, Any], *, tool_name: str) -> int | RecoveryExecutionResult:
        raw_value = args.get("steps", 1)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return self._failure(tool_name, "steps must be a positive integer", {"steps": raw_value})
        if value <= 0:
            return self._failure(tool_name, "steps must be a positive integer", {"steps": raw_value})
        return min(value, _MAX_EE_CONTROL_STEPS)

    def _take_qpos_action(
        self,
        action: np.ndarray,
        tool_name: str,
        *,
        latest_snapshot: Any | None,
        active_arm: str,
    ) -> RecoveryExecutionResult:
        if self._environment_success(latest_snapshot):
            return self._terminal_skip(tool_name, latest_snapshot)
        if not hasattr(self.task_env, "take_action"):
            return self._failure(tool_name, "task_env.take_action is unavailable")
        if not self._dual_arm_action_supported():
            return self._failure(
                tool_name,
                "dual-arm environment lacks active-arm action masking",
                {"active_arm": active_arm},
            )
        inactive_start = self._inactive_arm_poses(
            latest_snapshot,
            active_arm=active_arm,
            tool_name=tool_name,
        )
        if isinstance(inactive_start, RecoveryExecutionResult):
            return inactive_start
        if bool(getattr(self.task_env, "is_dual_arm", False)):
            self.task_env.take_action(
                action.astype(np.float32),
                action_type="qpos",
                active_arm=active_arm,
            )
        else:
            self.task_env.take_action(action.astype(np.float32), action_type="qpos")
        snapshot = self._refresh_snapshot()
        inactive_drift = self._inactive_arm_drift(
            start_poses=inactive_start,
            latest_snapshot=snapshot,
            tool_name=tool_name,
        )
        if isinstance(inactive_drift, RecoveryExecutionResult):
            return inactive_drift
        if any(
            value > self.inactive_arm_max_drift_m
            for value in inactive_drift.values()
        ):
            return RecoveryExecutionResult(
                result=RecoveryToolResult(
                    tool_name=tool_name,
                    success=False,
                    message="inactive arm moved during a single-arm qpos action",
                    details={
                        "active_arm": active_arm,
                        "inactive_arm_drift_m": inactive_drift,
                        "inactive_arm_max_drift_m": (
                            self.inactive_arm_max_drift_m
                        ),
                    },
                ),
                latest_snapshot=snapshot,
            )
        return RecoveryExecutionResult(
            result=RecoveryToolResult(
                tool_name=tool_name,
                success=True,
                message="executed qpos recovery action",
                details={
                    "step_count": snapshot.step_count,
                    "qpos_dim": int(action.size),
                    "environment_success": self._environment_success(snapshot),
                    "active_arm": active_arm,
                    "inactive_arm_drift_m": inactive_drift,
                    "inactive_arm_max_drift_m": self.inactive_arm_max_drift_m,
                },
            ),
            latest_snapshot=snapshot,
        )

    def _environment_success(self, snapshot: Any | None) -> bool:
        return bool(getattr(snapshot, "eval_success", False)) or bool(getattr(self.task_env, "eval_success", False))

    @staticmethod
    def _terminal_skip(tool_name: str, latest_snapshot: Any | None) -> RecoveryExecutionResult:
        return RecoveryExecutionResult(
            result=RecoveryToolResult(
                tool_name=tool_name,
                success=False,
                message="skipped physical recovery action after authoritative environment success",
                details={
                    "skipped": True,
                    "terminal_skip": True,
                    "skip_reason": "environment_eval_success",
                    "authority": "environment_eval_success",
                    "environment_success": True,
                },
            ),
            latest_snapshot=latest_snapshot,
        )

    def _refresh_snapshot(self) -> Any:
        if not hasattr(self.task_env, "get_obs"):
            raise RuntimeError("task_env.get_obs is unavailable")
        observation = self.task_env.get_obs()
        return capture_env_snapshot(
            self.task_env,
            observation,
            oracle_objects_enabled=self.oracle_objects_enabled,
        )

    def _snapshot_qpos(self, latest_snapshot: Any | None, tool_name: str) -> np.ndarray | RecoveryExecutionResult:
        if latest_snapshot is None:
            return self._failure(tool_name, "latest snapshot unavailable for qpos recovery")
        qpos = getattr(latest_snapshot, "joint_vector", None)
        if qpos is None:
            return self._failure(tool_name, "latest snapshot has no joint_vector")
        qpos_array = np.asarray(qpos, dtype=np.float32)
        if qpos_array.ndim != 1 or qpos_array.size == 0:
            return self._failure(tool_name, "latest snapshot joint_vector is not a non-empty 1D vector")
        return qpos_array

    def _can_probe_gripper_indices(self) -> bool:
        robot = getattr(self.task_env, "robot", None)
        if robot is None:
            return False
        if not callable(getattr(robot, "get_left_arm_jointState", None)):
            return False
        if bool(getattr(self.task_env, "is_dual_arm", False)) and not callable(getattr(robot, "get_right_arm_jointState", None)):
            return False
        return True

    def _gripper_indices(self, qpos: np.ndarray, *, arm: str, tool_name: str) -> list[int] | RecoveryExecutionResult:
        robot = getattr(self.task_env, "robot", None)
        if robot is None:
            return self._failure(tool_name, "task_env.robot is unavailable for gripper index probing")

        left_jointstate_fn = getattr(robot, "get_left_arm_jointState", None)
        if not callable(left_jointstate_fn):
            return self._failure(tool_name, "robot.get_left_arm_jointState is unavailable")
        left_arm_dim = len(left_jointstate_fn()) - 1
        if left_arm_dim < 0:
            return self._failure(tool_name, "left jointState does not include a gripper slot")
        indices: list[int] = []
        if arm in {"left", "both"}:
            indices.append(left_arm_dim)

        if bool(getattr(self.task_env, "is_dual_arm", False)):
            right_jointstate_fn = getattr(robot, "get_right_arm_jointState", None)
            if not callable(right_jointstate_fn):
                return self._failure(tool_name, "robot.get_right_arm_jointState is unavailable for dual-arm env")
            right_arm_dim = len(right_jointstate_fn()) - 1
            if right_arm_dim < 0:
                return self._failure(tool_name, "right jointState does not include a gripper slot")
            if arm in {"right", "both"}:
                indices.append(left_arm_dim + 1 + right_arm_dim)
        elif arm == "both":
            indices = [left_arm_dim]

        qpos_dim = int(qpos.size)
        invalid_indices = [index for index in indices if index < 0 or index >= qpos_dim]
        if invalid_indices:
            return self._failure(
                tool_name,
                "computed gripper index is outside qpos vector",
                {"arm": arm, "indices": indices, "qpos_dim": qpos_dim},
            )
        return indices

    def _failure(self, tool_name: str, message: str, details: dict[str, Any] | None = None) -> RecoveryExecutionResult:
        return RecoveryExecutionResult(
            result=RecoveryToolResult(
                tool_name=tool_name,
                success=False,
                message=message,
                details=dict(details or {}),
            )
        )
