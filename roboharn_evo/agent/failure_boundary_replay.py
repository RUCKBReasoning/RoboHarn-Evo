"""Capture and restore one quiescent RMBench failure boundary."""

from __future__ import annotations

import copy
import functools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roboharn_evo.agent.state import (
    AgentState,
    MonitorSnapshot,
    RecoveryPolicyState,
    RoleMemory,
    SkillRunState,
    TaskMemory,
    TaskPlanItem,
    WorkingMemory,
)

BOUNDARY_SCHEMA = "roboharn_evo/rmbench_failure_boundary/v1"

_AGENT_REPLAY_COMPONENTS = (
    ("_scene_memory_tracker", "scene_memory_tracker"),
    ("_operation_candidate_lifecycle", "operation_candidate_lifecycle"),
    (
        "_operation_geometry_refresh_contract",
        "operation_geometry_refresh_contract",
    ),
    ("_evidence_acquisition_policy", "evidence_acquisition_policy"),
)


class FailureBoundaryReplayError(RuntimeError):
    """The requested boundary cannot be captured or restored faithfully."""


class ActionPrefixRecorder:
    """Record environment-advancing RMBench ``take_action`` calls."""

    def __init__(self, task_env: Any) -> None:
        self.task_env = task_env
        self._original_take_action = None
        self._had_take_action_attribute = False
        self._take_action_attribute = None
        self._original_get_obs = None
        self._had_get_obs_attribute = False
        self._get_obs_attribute = None
        self._pending_observation_calls = 0
        self._inside_take_action = False
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._records)

    @property
    def observation_calls_after_last_action(self) -> int:
        return self._pending_observation_calls

    def start(self) -> None:
        if self._original_take_action is not None:
            raise RuntimeError("action-prefix recorder is already active")
        original = getattr(self.task_env, "take_action", None)
        if not callable(original):
            raise FailureBoundaryReplayError(
                "task environment exposes no take_action method"
            )
        instance_values = getattr(self.task_env, "__dict__", {})
        self._had_take_action_attribute = "take_action" in instance_values
        self._take_action_attribute = instance_values.get("take_action")
        self._original_take_action = original
        original_get_obs = getattr(self.task_env, "get_obs", None)
        if callable(original_get_obs):
            self._had_get_obs_attribute = "get_obs" in instance_values
            self._get_obs_attribute = instance_values.get("get_obs")
            self._original_get_obs = original_get_obs

            @functools.wraps(original_get_obs)
            def observed(*args, **kwargs):
                result = original_get_obs(*args, **kwargs)
                if not self._inside_take_action:
                    self._pending_observation_calls += 1
                return result

            self.task_env.get_obs = observed

        @functools.wraps(original)
        def recorded(action, *args, **kwargs):
            before = int(getattr(self.task_env, "take_action_cnt", 0))
            frozen_action = np.asarray(action).copy()
            observations_before = self._pending_observation_calls
            self._pending_observation_calls = 0
            was_inside_take_action = self._inside_take_action
            self._inside_take_action = True
            action_type = kwargs.get(
                "action_type",
                args[0] if args else "qpos",
            )
            active_arm = kwargs.get(
                "active_arm",
                args[1] if len(args) > 1 else None,
            )
            try:
                return original(action, *args, **kwargs)
            finally:
                self._inside_take_action = was_inside_take_action
                after = int(getattr(self.task_env, "take_action_cnt", before))
                if after > before:
                    self._records.append(
                        {
                            "index": len(self._records),
                            "env_step_before": before,
                            "env_step_after": after,
                            "observation_calls_before": observations_before,
                            "action_type": str(action_type),
                            "active_arm": (
                                None if active_arm is None else str(active_arm)
                            ),
                            "action_dtype": str(frozen_action.dtype),
                            "action_shape": list(frozen_action.shape),
                            "action": _plain(frozen_action),
                        }
                    )
                else:
                    self._pending_observation_calls += observations_before

        self.task_env.take_action = recorded

    def stop(self) -> None:
        if self._original_take_action is None:
            return
        if self._had_take_action_attribute:
            self.task_env.take_action = self._take_action_attribute
        else:
            delattr(self.task_env, "take_action")
        if self._original_get_obs is not None:
            if self._had_get_obs_attribute:
                self.task_env.get_obs = self._get_obs_attribute
            else:
                delattr(self.task_env, "get_obs")
        self._original_take_action = None
        self._had_take_action_attribute = False
        self._take_action_attribute = None
        self._original_get_obs = None
        self._had_get_obs_attribute = False
        self._get_obs_attribute = None
        self._inside_take_action = False


@dataclass(frozen=True, slots=True)
class FailureBoundary:
    task: str
    seed: int
    instruction: str
    label: str
    reason: str
    action_prefix: list[dict[str, Any]]
    observation_calls_after_last_action: int
    environment: dict[str, Any]
    agent: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BOUNDARY_SCHEMA,
            "task": self.task,
            "seed": self.seed,
            "instruction": self.instruction,
            "boundary": {
                "label": self.label,
                "reason": self.reason,
                "env_step": self.environment["task_state"].get(
                    "take_action_cnt",
                    0,
                ),
            },
            "action_prefix": copy.deepcopy(self.action_prefix),
            "observation_calls_after_last_action": (
                self.observation_calls_after_last_action
            ),
            "environment": copy.deepcopy(self.environment),
            "agent": copy.deepcopy(self.agent),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FailureBoundary:
        payload = dict(value)
        boundary = payload.get("boundary")
        action_prefix = payload.get("action_prefix")
        trailing_observations = payload.get("observation_calls_after_last_action")
        environment = payload.get("environment")
        agent = payload.get("agent")
        if (
            payload.get("schema") not in {BOUNDARY_SCHEMA, "tcm/rmbench_failure_boundary/v1"}
            or not all(
                isinstance(item, Mapping) for item in (boundary, environment, agent)
            )
            or not isinstance(action_prefix, list)
        ):
            raise FailureBoundaryReplayError("failure boundary JSON is invalid")
        seed = payload.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise FailureBoundaryReplayError(
                "failure boundary seed must be non-negative"
            )
        normalized_prefix = _normalize_action_prefix(action_prefix)
        normalized_trailing_observations = _nonnegative_int(
            trailing_observations,
            "observation_calls_after_last_action",
        )
        normalized_environment = copy.deepcopy(dict(environment))
        _require_action_prefix_covers_boundary(
            normalized_prefix,
            normalized_environment,
        )
        return cls(
            task=_required_text(payload.get("task"), "task"),
            seed=seed,
            instruction=_required_text(
                payload.get("instruction"),
                "instruction",
            ),
            label=_required_text(boundary.get("label"), "boundary.label"),
            reason=str(boundary.get("reason", "") or "").strip(),
            action_prefix=normalized_prefix,
            observation_calls_after_last_action=(normalized_trailing_observations),
            environment=normalized_environment,
            agent=copy.deepcopy(dict(agent)),
        )


def save_failure_boundary(
    path: str | Path,
    boundary: FailureBoundary,
) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"failure boundary already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(boundary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def load_failure_boundary(path: str | Path) -> FailureBoundary:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FailureBoundaryReplayError("cannot read failure boundary JSON") from exc
    if not isinstance(payload, Mapping):
        raise FailureBoundaryReplayError("failure boundary must be one JSON object")
    return FailureBoundary.from_dict(payload)


def capture_failure_boundary(
    task_env: Any,
    model: Any,
    *,
    task: str,
    seed: int,
    instruction: str,
    label: str,
    reason: str = "",
    action_prefix: Sequence[Mapping[str, Any]] | None = None,
    observation_calls_after_last_action: int = 0,
) -> FailureBoundary:
    agent = _resolve_agent(model)
    _require_quiescent_agent(agent)
    normalized_prefix = _normalize_action_prefix(list(action_prefix or []))
    environment = _capture_environment(task_env)
    _require_action_prefix_covers_boundary(normalized_prefix, environment)
    return FailureBoundary(
        task=_required_text(task, "task"),
        seed=_nonnegative_int(seed, "seed"),
        instruction=_required_text(instruction, "instruction"),
        label=_required_text(label, "label"),
        reason=str(reason or "").strip(),
        action_prefix=normalized_prefix,
        observation_calls_after_last_action=_nonnegative_int(
            observation_calls_after_last_action,
            "observation_calls_after_last_action",
        ),
        environment=environment,
        agent=_capture_agent(agent),
    )


def restore_failure_boundary(
    task_env: Any,
    model: Any,
    boundary: FailureBoundary | Mapping[str, Any],
    *,
    expected_task: str,
    expected_seed: int,
    expected_instruction: str,
    position_tolerance_m: float = 1e-6,
    joint_tolerance: float = 1e-6,
) -> dict[str, Any]:
    typed = _as_boundary(boundary)
    if typed.task != _required_text(expected_task, "expected_task"):
        raise FailureBoundaryReplayError("failure boundary task mismatch")
    if typed.seed != _nonnegative_int(expected_seed, "expected_seed"):
        raise FailureBoundaryReplayError("failure boundary seed mismatch")
    if typed.instruction != _required_text(
        expected_instruction,
        "expected_instruction",
    ):
        raise FailureBoundaryReplayError("failure boundary instruction mismatch")

    prefix_report = replay_action_prefix(
        task_env,
        typed.action_prefix,
        observation_calls_after_last_action=(typed.observation_calls_after_last_action),
    )
    report = compare_environment_to_boundary(
        task_env,
        typed,
        position_tolerance_m=position_tolerance_m,
        joint_tolerance=joint_tolerance,
    )
    if not report["matches"]:
        restore_gate = _physics_restore_gate(
            task_env,
            typed,
            strict_tolerance=max(float(position_tolerance_m), float(joint_tolerance)),
        )
        if not restore_gate["allowed"]:
            gate_reasons = (
                restore_gate["logical_mismatches"] + restore_gate["topology_mismatches"]
            )
            raise FailureBoundaryReplayError(
                "action-prefix replay differs from the captured failure boundary: "
                + "; ".join(gate_reasons[:6] or report["mismatches"][:6])
            )
        pre_correction_error = float(report["max_numeric_error"])
        _restore_simulator_state(task_env, typed.environment["state_summary"])
        report = compare_environment_to_boundary(
            task_env,
            typed,
            position_tolerance_m=position_tolerance_m,
            joint_tolerance=joint_tolerance,
        )
        if not report["matches"]:
            raise FailureBoundaryReplayError(
                "simulator state correction did not reconstruct the captured "
                "failure boundary: " + "; ".join(report["mismatches"][:6])
            )
        report["physics_correction"] = {
            "applied": True,
            "method": "explicit_captured_state",
            "pre_correction_max_numeric_error": pre_correction_error,
            "post_correction_max_numeric_error": report["max_numeric_error"],
        }
    else:
        report["physics_correction"] = {
            "applied": False,
            "method": "action_prefix_exact",
            "pre_correction_max_numeric_error": report["max_numeric_error"],
            "post_correction_max_numeric_error": report["max_numeric_error"],
        }
    agent = _resolve_agent(model)
    _restore_agent(agent, typed.agent)
    _restore_recovery_observation_handoff(
        agent,
        task_env,
        typed.agent,
    )
    restored_agent = _capture_agent(agent)
    if restored_agent != typed.agent:
        sections = sorted(
            key
            for key in set(restored_agent) | set(typed.agent)
            if restored_agent.get(key) != typed.agent.get(key)
        )
        raise FailureBoundaryReplayError(
            "restored Agent state differs from the captured failure boundary: "
            + ", ".join(sections[:6])
        )
    report["agent_state_matches"] = True
    report["action_prefix_replay"] = prefix_report
    return report


def _physics_restore_gate(
    task_env: Any,
    boundary: FailureBoundary,
    *,
    strict_tolerance: float,
) -> dict[str, Any]:
    current = _capture_environment(task_env)
    wanted = boundary.environment
    logical_mismatches: list[str] = []
    logical_metrics = {"max_numeric_error": 0.0}
    for section in ("task_state", "robot_cache"):
        _compare_values(
            current.get(section),
            wanted.get(section),
            path=section,
            tolerance=strict_tolerance,
            mismatches=logical_mismatches,
            metrics=logical_metrics,
        )
    topology_mismatches: list[str] = []
    topology_metrics = {"max_numeric_error": 0.0}
    _compare_values(
        _physics_topology_projection(current.get("state_summary")),
        _physics_topology_projection(wanted.get("state_summary")),
        path="state_summary",
        tolerance=0.0,
        mismatches=topology_mismatches,
        metrics=topology_metrics,
    )
    return {
        "allowed": not logical_mismatches and not topology_mismatches,
        "logical_mismatches": logical_mismatches,
        "topology_mismatches": topology_mismatches,
        "max_numeric_error": max(
            logical_metrics["max_numeric_error"],
            topology_metrics["max_numeric_error"],
        ),
    }


def _physics_topology_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"invalid": True}
    actors = value.get("actors")
    articulations = value.get("articulations")
    if not isinstance(actors, list) or not isinstance(articulations, list):
        return {"invalid": True}
    return {
        "actors": [
            {
                "name": actor.get("name") if isinstance(actor, Mapping) else None,
                "fields": sorted(actor) if isinstance(actor, Mapping) else [],
            }
            for actor in actors
        ],
        "articulations": [
            {
                "name": articulation.get("name")
                if isinstance(articulation, Mapping)
                else None,
                "fields": sorted(articulation)
                if isinstance(articulation, Mapping)
                else [],
                "joints": [
                    {
                        "name": joint.get("name")
                        if isinstance(joint, Mapping)
                        else None,
                        "fields": sorted(joint) if isinstance(joint, Mapping) else [],
                    }
                    for joint in (
                        articulation.get("joints", [])
                        if isinstance(articulation, Mapping)
                        and isinstance(articulation.get("joints"), list)
                        else []
                    )
                ],
            }
            for articulation in articulations
        ],
    }


def _restore_simulator_state(task_env: Any, payload: Mapping[str, Any]) -> None:
    scene = getattr(task_env, "scene", None)
    if scene is None:
        raise FailureBoundaryReplayError("task environment has no simulator scene")
    actors = _scene_items(scene, "get_all_actors")
    articulations = _scene_items(scene, "get_all_articulations")
    actor_records = payload.get("actors")
    articulation_records = payload.get("articulations")
    if not isinstance(actor_records, list) or len(actor_records) != len(actors):
        raise FailureBoundaryReplayError("captured actor topology cannot be restored")
    if not isinstance(articulation_records, list) or len(articulation_records) != len(
        articulations
    ):
        raise FailureBoundaryReplayError(
            "captured articulation topology cannot be restored"
        )
    for actor, record in zip(actors, actor_records, strict=True):
        if not isinstance(record, Mapping) or _name(actor) != record.get("name"):
            raise FailureBoundaryReplayError(
                "captured actor identity cannot be restored"
            )
        _restore_actor_state(actor, record)
    for articulation, record in zip(articulations, articulation_records, strict=True):
        if not isinstance(record, Mapping) or _name(articulation) != record.get("name"):
            raise FailureBoundaryReplayError(
                "captured articulation identity cannot be restored"
            )
        _restore_articulation_state(articulation, record)


def _restore_actor_state(actor: Any, record: Mapping[str, Any]) -> None:
    setter = getattr(actor, "set_pose", None)
    getter = getattr(actor, "get_pose", None)
    if not callable(setter) or not callable(getter):
        raise FailureBoundaryReplayError("captured actor pose cannot be restored")
    setter(_restored_pose(getter(), record.get("pose")))
    velocity_fields = (
        ("linear_velocity", "set_linear_velocity"),
        ("angular_velocity", "set_angular_velocity"),
    )
    if any(field in record for field, _ in velocity_fields):
        component = _dynamic_actor_component(actor)
        if component is None:
            raise FailureBoundaryReplayError(
                "captured actor velocity cannot be restored"
            )
        for field, setter_name in velocity_fields:
            if field not in record:
                continue
            velocity_setter = getattr(component, setter_name, None)
            if not callable(velocity_setter):
                raise FailureBoundaryReplayError(
                    "captured actor velocity cannot be restored"
                )
            velocity_setter(np.asarray(_vector(record[field], length=3)))


def _restore_articulation_state(
    articulation: Any,
    record: Mapping[str, Any],
) -> None:
    pose_getter = getattr(articulation, "get_root_pose", None)
    operations = (
        ("set_root_pose", _restored_pose(pose_getter(), record.get("root_pose"))),
        ("set_qpos", np.asarray(_vector(record.get("qpos")))),
        ("set_qvel", np.asarray(_vector(record.get("qvel")))),
        (
            "set_root_linear_velocity",
            np.asarray(_vector(record.get("root_linear_velocity"), length=3)),
        ),
        (
            "set_root_angular_velocity",
            np.asarray(_vector(record.get("root_angular_velocity"), length=3)),
        ),
    )
    for setter_name, value in operations:
        setter = getattr(articulation, setter_name, None)
        if not callable(setter):
            raise FailureBoundaryReplayError(
                f"captured articulation state lacks {setter_name}"
            )
        setter(value)
    joints = list(articulation.get_active_joints())
    joint_records = record.get("joints")
    if not isinstance(joint_records, list) or len(joint_records) != len(joints):
        raise FailureBoundaryReplayError("captured joint topology cannot be restored")
    for joint, joint_record in zip(joints, joint_records, strict=True):
        if not isinstance(joint_record, Mapping) or _name(joint) != joint_record.get(
            "name"
        ):
            raise FailureBoundaryReplayError(
                "captured joint identity cannot be restored"
            )
        for field, setter_name in (
            ("drive_target", "set_drive_target"),
            ("drive_velocity_target", "set_drive_velocity_target"),
        ):
            setter = getattr(joint, setter_name, None)
            if not callable(setter):
                raise FailureBoundaryReplayError(
                    "captured joint drive state cannot be restored"
                )
            values = _vector(joint_record.get(field))
            setter(values[0] if len(values) == 1 else np.asarray(values))


def _restored_pose(current: Any, payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        raise FailureBoundaryReplayError("captured pose is invalid")
    position = np.asarray(_vector(payload.get("p"), length=3))
    quaternion = np.asarray(_vector(payload.get("q"), length=4))
    try:
        return type(current)(position, quaternion)
    except Exception as exc:
        raise FailureBoundaryReplayError(
            "captured pose type cannot be reconstructed"
        ) from exc


def _dynamic_actor_component(actor: Any) -> Any | None:
    getter = getattr(actor, "get_components", None)
    if not callable(getter):
        return None
    for component in getter():
        if callable(getattr(component, "get_linear_velocity", None)) and callable(
            getattr(component, "get_angular_velocity", None)
        ):
            return component
    return None


def compare_environment_to_boundary(
    task_env: Any,
    boundary: FailureBoundary | Mapping[str, Any],
    *,
    position_tolerance_m: float = 1e-6,
    joint_tolerance: float = 1e-6,
) -> dict[str, Any]:
    typed = _as_boundary(boundary)
    current = _capture_environment(task_env)
    wanted = typed.environment
    mismatches: list[str] = []
    metrics = {"max_numeric_error": 0.0}
    tolerance = max(float(position_tolerance_m), float(joint_tolerance))
    for section in ("state_summary", "task_state", "robot_cache"):
        _compare_values(
            current.get(section),
            wanted.get(section),
            path=section,
            tolerance=tolerance,
            mismatches=mismatches,
            metrics=metrics,
        )
    return {
        "matches": not mismatches,
        "mismatches": mismatches,
        **metrics,
    }


def replay_action_prefix(
    task_env: Any,
    action_prefix: Sequence[Mapping[str, Any]],
    *,
    observation_calls_after_last_action: int = 0,
) -> dict[str, Any]:
    records = _normalize_action_prefix(list(action_prefix))
    trailing_observations = _nonnegative_int(
        observation_calls_after_last_action,
        "observation_calls_after_last_action",
    )
    start_step = int(getattr(task_env, "take_action_cnt", 0))
    observations_replayed = 0
    has_video_path = hasattr(task_env, "eval_video_path")
    video_path = getattr(task_env, "eval_video_path", None)
    if has_video_path:
        task_env.eval_video_path = None
    try:
        for record in records:
            observations_replayed += _replay_observations(
                task_env,
                record["observation_calls_before"],
            )
            before = int(getattr(task_env, "take_action_cnt", 0))
            if before != record["env_step_before"]:
                raise FailureBoundaryReplayError(
                    "action-prefix replay started from a different environment step"
                )
            try:
                dtype = np.dtype(record["action_dtype"])
                shape = tuple(record["action_shape"])
                action = np.asarray(record["action"], dtype=dtype).reshape(shape)
            except (TypeError, ValueError) as exc:
                raise FailureBoundaryReplayError(
                    "action-prefix record has an invalid action vector"
                ) from exc
            task_env.take_action(
                action,
                action_type=record["action_type"],
                active_arm=record["active_arm"],
            )
            after = int(getattr(task_env, "take_action_cnt", before))
            if after != record["env_step_after"]:
                raise FailureBoundaryReplayError(
                    "action-prefix replay advanced to a different environment step"
                )
        observations_replayed += _replay_observations(
            task_env,
            trailing_observations,
        )
    finally:
        if has_video_path:
            task_env.eval_video_path = video_path
    return {
        "actions_replayed": len(records),
        "observation_calls_replayed": observations_replayed,
        "start_env_step": start_step,
        "end_env_step": int(getattr(task_env, "take_action_cnt", start_step)),
    }


def _normalize_action_prefix(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise FailureBoundaryReplayError("action prefix must be a list")
    normalized: list[dict[str, Any]] = []
    previous_after = 0
    for expected_index, raw_record in enumerate(value):
        if not isinstance(raw_record, Mapping):
            raise FailureBoundaryReplayError("action-prefix record must be a mapping")
        record = dict(raw_record)
        index = _nonnegative_int(record.get("index"), "action-prefix index")
        if index != expected_index:
            raise FailureBoundaryReplayError(
                "action-prefix indices must be contiguous and ordered"
            )
        before = _nonnegative_int(
            record.get("env_step_before"),
            "action-prefix env_step_before",
        )
        after = _nonnegative_int(
            record.get("env_step_after"),
            "action-prefix env_step_after",
        )
        if before != previous_after or after <= before:
            raise FailureBoundaryReplayError(
                "action-prefix environment steps must form one advancing prefix"
            )
        observation_calls_before = _nonnegative_int(
            record.get("observation_calls_before"),
            "action-prefix observation_calls_before",
        )
        action_type = _required_text(
            record.get("action_type"),
            "action-prefix action_type",
        ).lower()
        if action_type not in {"qpos", "ee"}:
            raise FailureBoundaryReplayError(
                "action-prefix action_type must be qpos or ee"
            )
        active_arm = record.get("active_arm")
        if active_arm is not None:
            active_arm = _required_text(
                active_arm,
                "action-prefix active_arm",
            ).lower()
            if active_arm not in {"left", "right", "both"}:
                raise FailureBoundaryReplayError(
                    "action-prefix active_arm must be left, right, both, or null"
                )
        shape = record.get("action_shape")
        if not isinstance(shape, list) or not shape:
            raise FailureBoundaryReplayError(
                "action-prefix action_shape must be a non-empty list"
            )
        normalized_shape = [
            _nonnegative_int(item, "action-prefix action_shape item") for item in shape
        ]
        if any(item == 0 for item in normalized_shape):
            raise FailureBoundaryReplayError(
                "action-prefix action_shape cannot contain zero"
            )
        try:
            dtype = np.dtype(
                _required_text(
                    record.get("action_dtype"),
                    "action-prefix action_dtype",
                )
            )
        except TypeError as exc:
            raise FailureBoundaryReplayError(
                "action-prefix action_dtype must be numeric"
            ) from exc
        if dtype.kind not in {"f", "i", "u"}:
            raise FailureBoundaryReplayError(
                "action-prefix action_dtype must be numeric"
            )
        try:
            action = np.asarray(record.get("action"), dtype=dtype).reshape(
                tuple(normalized_shape)
            )
        except (TypeError, ValueError) as exc:
            raise FailureBoundaryReplayError(
                "action-prefix action does not match its dtype and shape"
            ) from exc
        if not np.all(np.isfinite(action)):
            raise FailureBoundaryReplayError(
                "action-prefix action must contain finite numbers"
            )
        normalized.append(
            {
                "index": index,
                "env_step_before": before,
                "env_step_after": after,
                "observation_calls_before": observation_calls_before,
                "action_type": action_type,
                "active_arm": active_arm,
                "action_dtype": dtype.name,
                "action_shape": normalized_shape,
                "action": _plain(action),
            }
        )
        previous_after = after
    return normalized


def _replay_observations(task_env: Any, count: int) -> int:
    if count == 0:
        return 0
    get_obs = getattr(task_env, "get_obs", None)
    if not callable(get_obs):
        raise FailureBoundaryReplayError(
            "action-prefix replay requires the recorded get_obs calls"
        )
    for _ in range(count):
        get_obs()
    return count


def _require_action_prefix_covers_boundary(
    action_prefix: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
) -> None:
    task_state = environment.get("task_state")
    if not isinstance(task_state, Mapping):
        raise FailureBoundaryReplayError("failure boundary has no readable task state")
    boundary_step = _nonnegative_int(
        task_state.get("take_action_cnt"),
        "failure boundary take_action_cnt",
    )
    prefix_step = int(action_prefix[-1]["env_step_after"]) if action_prefix else 0
    if boundary_step != prefix_step:
        raise FailureBoundaryReplayError(
            "action prefix does not cover every environment action before the "
            "failure boundary"
        )


def run_paired_failure_continuations(
    boundary: FailureBoundary | Mapping[str, Any],
    *,
    conditions: Sequence[str],
    environment_factory: Callable[[str], Any],
    model_factory: Callable[[str], Any],
    continue_episode: Callable[[Any, Any, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run independent conditions from one restored boundary."""

    typed = _as_boundary(boundary)
    names = [str(item or "").strip() for item in conditions]
    if not names or any(not item for item in names) or len(set(names)) != len(names):
        raise ValueError("paired continuation conditions must be unique and non-empty")
    results = []
    for condition in names:
        environment = environment_factory(condition)
        model = model_factory(condition)
        try:
            match = restore_failure_boundary(
                environment,
                model,
                typed,
                expected_task=typed.task,
                expected_seed=typed.seed,
                expected_instruction=typed.instruction,
            )
            start_step = int(getattr(environment, "take_action_cnt", 0))
            outcome = dict(continue_episode(environment, model, condition))
            end_step = int(getattr(environment, "take_action_cnt", start_step))
            results.append(
                {
                    "condition": condition,
                    "boundary_match": match,
                    "success": bool(outcome.get("success", False)),
                    "additional_environment_actions": max(
                        0,
                        end_step - start_step,
                    ),
                    "outcome": _plain(outcome),
                }
            )
        finally:
            close = getattr(environment, "close_env", None)
            if callable(close):
                close()
    return {
        "schema": "roboharn_evo/rmbench_failure_boundary_comparison/v1",
        "task": typed.task,
        "seed": typed.seed,
        "boundary": {"label": typed.label, "reason": typed.reason},
        "results": results,
    }


def _capture_environment(task_env: Any) -> dict[str, Any]:
    scene = getattr(task_env, "scene", None)
    if scene is None:
        raise FailureBoundaryReplayError("task environment has no simulator scene")
    return {
        "state_summary": _physics_summary(scene),
        "task_state": _capture_task_state(task_env),
        "robot_cache": _capture_robot_cache(task_env),
    }


def _physics_summary(scene: Any) -> dict[str, Any]:
    actors = _scene_items(scene, "get_all_actors")
    articulations = _scene_items(scene, "get_all_articulations")
    return {
        "actors": [
            {
                "name": _name(actor),
                "pose": _pose(actor.get_pose()),
                **_actor_velocity(actor),
            }
            for actor in actors
        ],
        "articulations": [
            _articulation_summary(articulation) for articulation in articulations
        ],
    }


def _articulation_summary(value: Any) -> dict[str, Any]:
    joints = list(value.get_active_joints())
    return {
        "name": _name(value),
        "root_pose": _pose(value.get_root_pose()),
        "root_linear_velocity": _vector(value.get_root_linear_velocity()),
        "root_angular_velocity": _vector(value.get_root_angular_velocity()),
        "qpos": _vector(value.get_qpos()),
        "qvel": _vector(value.get_qvel()),
        "joints": [
            {
                "name": _name(joint),
                "drive_target": _vector(joint.get_drive_target()),
                "drive_velocity_target": _vector(joint.get_drive_velocity_target()),
            }
            for joint in joints
        ],
    }


def _actor_velocity(actor: Any) -> dict[str, Any]:
    getter = getattr(actor, "get_components", None)
    if not callable(getter):
        return {}
    for component in getter():
        if callable(getattr(component, "get_linear_velocity", None)) and callable(
            getattr(component, "get_angular_velocity", None)
        ):
            return {
                "linear_velocity": _vector(component.get_linear_velocity()),
                "angular_velocity": _vector(component.get_angular_velocity()),
            }
    return {}


def _capture_task_state(task_env: Any) -> dict[str, Any]:
    result = {}
    for name, value in vars(task_env).items():
        if name.startswith("_") or _is_runtime_object_field(name):
            continue
        try:
            result[name] = _plain(value)
        except (FailureBoundaryReplayError, RecursionError):
            continue
    return result


def _is_runtime_object_field(name: str) -> bool:
    normalized = str(name).strip().lower()
    if normalized in {
        "scene",
        "engine",
        "renderer",
        "viewer",
        "robot",
        "now_obs",
        "raw_head_pcl",
        "real_head_pcl",
        "real_head_pcl_color",
        "world_pcd",
        "eval_video_path",
        "eval_video_save_dir",
        "save_dir",
        "file_path",
        "suc",
        "test_num",
    }:
        return True
    return any(
        token in normalized
        for token in (
            "camera",
            "renderer",
            "ffmpeg",
            "video",
            "rollout_dir",
            "trace_file",
        )
    )


def _capture_robot_cache(task_env: Any) -> dict[str, Any]:
    robot = getattr(task_env, "robot", None)
    if robot is None:
        return {}
    return {
        name: _plain(getattr(robot, name))
        for name in ("left_gripper_val", "right_gripper_val")
        if hasattr(robot, name)
    }


def _capture_agent(agent: Any) -> dict[str, Any]:
    store = getattr(agent, "memory_store", None)
    state = getattr(store, "state", None)
    if state is None or not callable(getattr(state, "to_dict", None)):
        raise FailureBoundaryReplayError("model has no serializable Agent state")
    runtime_fields = (
        "_debug_recovery_triggered",
        "_pure_tool_control_control_turns",
        "_pure_tool_control_no_progress_control_turns",
        "_debug_recovery_rounds",
        "_debug_recovery_scene_wait_turns",
        "_identity_binding_retry_attempts",
        "_identity_binding_retry_skill_id",
        "_debug_recovery_planner_bootstrapped",
        "_pure_tool_control_control_backend_errors",
        "_pure_tool_control_empty_plan_turns",
        "_pure_tool_control_recovery_backend_errors",
        "_last_recovery_backend_error",
        "_last_recovery_backend_error_stage",
        "_pure_tool_control_terminal_failure",
        "_authoritative_environment_success",
        "_observation_preprocess_generation",
        "_observation_capture_generation",
        "_pending_internal_recovery_completion",
    )
    components = {}
    for attribute, label in _AGENT_REPLAY_COMPONENTS:
        component = getattr(agent, attribute, None)
        if component is None:
            continue
        exporter = getattr(component, "export_replay_state", None)
        if not callable(exporter):
            raise FailureBoundaryReplayError(
                f"Agent component cannot export replay state: {label}"
            )
        components[label] = _plain(exporter())
    return {
        "memory_state": _plain(state.to_dict()),
        "memory_counters": {
            name: int(getattr(store, name))
            for name in ("_plan_counter", "_rollout_counter")
            if hasattr(store, name)
        },
        "runtime_counters": {
            name: _plain(getattr(agent, name))
            for name in runtime_fields
            if hasattr(agent, name)
        },
        "blocked_grounded_setups": [
            list(item)
            for item in sorted(
                getattr(agent, "_blocked_grounded_setups", set()) or set()
            )
        ],
        "grounded_setup_failures": _encode_keyed_mapping(
            getattr(agent, "_grounded_setup_failures", {}) or {}
        ),
        "grounded_geometry_leases": _encode_keyed_mapping(
            getattr(agent, "_grounded_geometry_leases", {}) or {}
        ),
        "partial_grounded_approach_leases": _encode_keyed_mapping(
            getattr(agent, "_partial_grounded_approach_leases", {}) or {}
        ),
        "recent_release_resolutions": _plain(
            getattr(agent, "_recent_release_resolutions", {}) or {}
        ),
        "last_evidence_acquisition_decision": _plain(
            getattr(agent, "_last_evidence_acquisition_decision", {}) or {}
        ),
        "recovery_observation_handoff_pending": isinstance(
            getattr(agent, "_pending_recovery_observation", None),
            Mapping,
        ),
        "components": components,
    }


def _restore_agent(agent: Any, payload: Mapping[str, Any]) -> None:
    store = getattr(agent, "memory_store", None)
    memory_state = payload.get("memory_state")
    if store is None or not isinstance(memory_state, Mapping):
        raise FailureBoundaryReplayError("failure boundary Agent state is invalid")
    store.state = _agent_state_from_dict(memory_state)
    memory_counters = payload.get("memory_counters", {})
    if not isinstance(memory_counters, Mapping):
        raise FailureBoundaryReplayError("failure boundary memory counters are invalid")
    for name, value in memory_counters.items():
        if hasattr(store, name):
            setattr(store, name, int(value))
    for name, value in dict(payload.get("runtime_counters", {})).items():
        if hasattr(agent, name):
            setattr(agent, name, copy.deepcopy(value))
    agent._blocked_grounded_setups = {
        tuple(str(part) for part in item)
        for item in payload.get("blocked_grounded_setups", []) or []
        if isinstance(item, list)
    }
    for field in (
        "grounded_setup_failures",
        "grounded_geometry_leases",
        "partial_grounded_approach_leases",
    ):
        setattr(agent, f"_{field}", _decode_keyed_mapping(payload.get(field)))
    agent._recent_release_resolutions = copy.deepcopy(
        dict(payload.get("recent_release_resolutions", {}) or {})
    )
    agent._last_evidence_acquisition_decision = copy.deepcopy(
        dict(payload.get("last_evidence_acquisition_decision", {}) or {})
    )
    components = payload.get("components", {})
    if not isinstance(components, Mapping):
        raise FailureBoundaryReplayError(
            "failure boundary Agent component state is invalid"
        )
    for attribute, label in _AGENT_REPLAY_COMPONENTS:
        if label not in components:
            continue
        component = getattr(agent, attribute, None)
        importer = getattr(component, "restore_replay_state", None)
        if not callable(importer) or not isinstance(
            components[label],
            Mapping,
        ):
            raise FailureBoundaryReplayError(
                f"Agent component cannot restore replay state: {label}"
            )
        importer(copy.deepcopy(dict(components[label])))
    agent._pending_action_effect_verification = None
    agent._pending_recovery_observation = None
    agent.current_instruction = store.state.task.global_task


def _restore_recovery_observation_handoff(
    agent: Any,
    task_env: Any,
    payload: Mapping[str, Any],
) -> None:
    pending = payload.get("recovery_observation_handoff_pending", False)
    if not isinstance(pending, bool):
        raise FailureBoundaryReplayError(
            "failure boundary recovery observation marker is invalid"
        )
    if not pending:
        return
    raw = getattr(task_env, "now_obs", None)
    if not isinstance(raw, Mapping):
        raise FailureBoundaryReplayError(
            "action-prefix replay did not reconstruct the pending observation"
        )
    agent._pending_recovery_observation = {
        "step_count": int(getattr(task_env, "take_action_cnt", 0)),
        "raw": copy.deepcopy(dict(raw)),
    }


def _require_quiescent_agent(agent: Any) -> None:
    if getattr(agent, "_pending_action_effect_verification", None) is not None:
        raise FailureBoundaryReplayError(
            "failure boundary must be captured after effect verification finishes"
        )
    runtime = getattr(agent, "_hpk_runtime", None)
    if runtime is not None and getattr(
        runtime,
        "has_pending_action_transition",
        False,
    ):
        raise FailureBoundaryReplayError(
            "failure boundary must be captured after the pending HPK action "
            "transition finishes"
        )


def _agent_state_from_dict(value: Mapping[str, Any]) -> AgentState:
    role = dict(value.get("role", {}) or {})
    task = dict(value.get("task", {}) or {})
    working = dict(value.get("working", {}) or {})
    monitor = dict(value.get("monitor", {}) or {})
    recovery = dict(value.get("recovery", {}) or {})
    active = value.get("active_skill")
    plan = [TaskPlanItem(**dict(item)) for item in task.pop("plan", [])]
    return AgentState(
        role=RoleMemory(**role),
        task=TaskMemory(plan=plan, **task),
        working=WorkingMemory(**working),
        active_skill=(None if active is None else SkillRunState(**dict(active))),
        monitor=MonitorSnapshot(**monitor),
        recovery=RecoveryPolicyState(**recovery),
        decision_count=int(value.get("decision_count", 0)),
        recovery_attempts=int(value.get("recovery_attempts", 0)),
    )


def _compare_values(
    actual: Any,
    wanted: Any,
    *,
    path: str,
    tolerance: float,
    mismatches: list[str],
    metrics: dict[str, float],
) -> None:
    if isinstance(wanted, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(wanted):
            mismatches.append(f"{path} fields changed")
            return
        for key in wanted:
            _compare_values(
                actual[key],
                wanted[key],
                path=f"{path}.{key}",
                tolerance=tolerance,
                mismatches=mismatches,
                metrics=metrics,
            )
        return
    if isinstance(wanted, list):
        if not isinstance(actual, list) or len(actual) != len(wanted):
            mismatches.append(f"{path} length changed")
            return
        if path.endswith(".q") and len(wanted) == 4:
            first = np.asarray(actual, dtype=float)
            second = np.asarray(wanted, dtype=float)
            error = float(
                min(
                    np.max(np.abs(first - second)),
                    np.max(np.abs(first + second)),
                )
            )
            metrics["max_numeric_error"] = max(
                metrics["max_numeric_error"],
                error,
            )
            if error > tolerance:
                mismatches.append(f"{path} changed")
            return
        for index, value in enumerate(wanted):
            _compare_values(
                actual[index],
                value,
                path=f"{path}[{index}]",
                tolerance=tolerance,
                mismatches=mismatches,
                metrics=metrics,
            )
        return
    if (
        isinstance(wanted, (int, float))
        and not isinstance(wanted, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        error = abs(float(actual) - float(wanted))
        metrics["max_numeric_error"] = max(
            metrics["max_numeric_error"],
            error,
        )
        if not math.isfinite(error) or error > tolerance:
            mismatches.append(f"{path} changed")
        return
    if actual != wanted:
        mismatches.append(f"{path} changed")


def _encode_keyed_mapping(value: Mapping[Any, Any]) -> list[dict[str, Any]]:
    records = [
        {
            "key": _plain(list(key) if isinstance(key, tuple) else [key]),
            "value": _plain(item),
        }
        for key, item in value.items()
    ]
    return sorted(
        records,
        key=lambda item: json.dumps(item["key"], sort_keys=True),
    )


def _decode_keyed_mapping(value: Any) -> dict[tuple[str, ...], Any]:
    if not value:
        return {}
    if not isinstance(value, list):
        raise FailureBoundaryReplayError("keyed Agent runtime state is invalid")
    result = {}
    for item in value:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("key"),
            list,
        ):
            raise FailureBoundaryReplayError("keyed Agent runtime record is invalid")
        result[tuple(str(part) for part in item["key"])] = copy.deepcopy(
            item.get("value")
        )
    return result


def _scene_items(scene: Any, getter_name: str) -> list[Any]:
    getter = getattr(scene, getter_name, None)
    if not callable(getter):
        raise FailureBoundaryReplayError(f"simulator scene lacks {getter_name}")
    return list(getter())


def _name(value: Any) -> str:
    getter = getattr(value, "get_name", None)
    return str(getter() if callable(getter) else getattr(value, "name", "") or "")


def _pose(value: Any) -> dict[str, list[float]]:
    return {
        "p": _vector(getattr(value, "p", None), length=3),
        "q": _vector(getattr(value, "q", None), length=4),
    }


def _vector(value: Any, *, length: int | None = None) -> list[float]:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise FailureBoundaryReplayError("numeric simulator state is invalid") from exc
    if length is not None and array.size != length:
        raise FailureBoundaryReplayError("numeric simulator state has wrong length")
    if not np.all(np.isfinite(array)):
        raise FailureBoundaryReplayError("numeric simulator state must be finite")
    return [float(item) for item in array]


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise FailureBoundaryReplayError("boundary contains a non-finite number")
        return value
    raise FailureBoundaryReplayError(
        f"boundary contains unsupported state type: {type(value).__name__}"
    )


def _resolve_agent(model: Any) -> Any:
    session = getattr(model, "session", None)
    agent = getattr(session, "agent", None) if session is not None else None
    if agent is None:
        agent = getattr(model, "agent", model)
    if getattr(agent, "memory_store", None) is None:
        raise FailureBoundaryReplayError("model exposes no RoboHarn-Evo Agent")
    return agent


def _as_boundary(
    value: FailureBoundary | Mapping[str, Any],
) -> FailureBoundary:
    return (
        value
        if isinstance(value, FailureBoundary)
        else FailureBoundary.from_dict(value)
    )


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise FailureBoundaryReplayError(f"{label} must be non-empty")
    return text


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FailureBoundaryReplayError(f"{label} must be a non-negative integer")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FailureBoundaryReplayError(
            f"{label} must be a finite non-negative number"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise FailureBoundaryReplayError(
            f"{label} must be a finite non-negative number"
        )
    return result


__all__ = [
    "ActionPrefixRecorder",
    "BOUNDARY_SCHEMA",
    "FailureBoundary",
    "FailureBoundaryReplayError",
    "capture_failure_boundary",
    "compare_environment_to_boundary",
    "load_failure_boundary",
    "replay_action_prefix",
    "restore_failure_boundary",
    "run_paired_failure_continuations",
    "save_failure_boundary",
]
