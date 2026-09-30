"""Evolving-only runtime seam for private action-effect transitions.

The dispatcher owns physical execution.  This module owns one ephemeral action
session around an already-guarded recovery batch and persists its exact call
boundaries before producing the typed transition.  Static/off callers get
``None`` before nonce generation or any runtime hook is touched.
"""

from __future__ import annotations

import copy
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from roboharn_evo.agent.hpk.action_transition import (
    ActionEffectTransitionV1,
    build_action_effect_transition,
    public_transition_projection,
)
from roboharn_evo.agent.hpk.effect_extractor import AbstractEffectExtractor
from roboharn_evo.agent.hpk.schemas import stable_content_id, validate_content_id
from roboharn_evo.agent.recovery.tool_dispatcher import RecoveryDispatchObservation
from roboharn_evo.agent.recovery.tool_specs import RecoveryToolCall


ACTION_ATTEMPT_PREPARED_EVENT = "hpk_action_attempt_prepared_private"
ACTION_TOOL_BOUNDARY_EVENT = "hpk_action_tool_boundary_private"
ACTION_TOOL_RESULT_BUNDLE_EVENT = "hpk_action_tool_result_bundle_private"
ACTION_EFFECT_TRANSITION_EVENT = "action_effect_transition"
PUBLIC_ACTION_EFFECT_TRANSITION_EVENT = "hpk_action_effect_transition_public"

PHYSICAL_RECOVERY_TOOLS = frozenset(
    {
        "close_gripper",
        "contact_displace",
        "lift_ee",
        "move_ee_to_grounded_instance",
        "move_ee_to_pose",
        "move_to_home",
        "open_gripper",
        "retreat_arm",
        "safe_reset_posture",
    }
)

_SEMANTIC_TERMINALS = {
    "grasp": "close_gripper",
    "place": "open_gripper",
    "contact": "contact_displace",
}
_SEMANTIC_PHASE_BY_TOOL = {
    ("grasp", "move_ee_to_pose"): "grasp_diagnostic",
    ("grasp", "close_gripper"): "grasp_close",
    ("grasp", "lift_ee"): "grasp_lift",
    ("grasp", "retreat_arm"): "grasp_retreat",
    ("place", "move_ee_to_pose"): "place_clearance",
    ("place", "open_gripper"): "place_release",
    ("place", "lift_ee"): "place_settle",
    ("place", "retreat_arm"): "place_retreat",
    ("contact", "contact_displace"): "contact_displace",
    ("contact", "retreat_arm"): "contact_retreat",
}
_SEMANTIC_TAILS = {
    "grasp": ("grasp_lift", "grasp_diagnostic", "grasp_retreat"),
    "place": ("place_settle", "place_clearance", "place_retreat"),
    "contact": ("contact_retreat",),
}

PrivateEventSink = Callable[[str, dict[str, Any]], None]
PublicEventSink = Callable[[str, dict[str, Any]], None]
NonceFactory = Callable[[], str]


class HPKTransitionRuntimeError(RuntimeError):
    """Raised when an evolving transition boundary cannot be trusted."""


@dataclass(frozen=True, slots=True)
class DispatchBoundaryRecord:
    phase: str
    dispatch_index: int
    tool_name: str
    physical: bool
    tool_call_ref: str
    tool_result_ref: str | None
    snapshot_before: dict[str, Any] | None
    snapshot_after: dict[str, Any] | None
    result: dict[str, Any] | None
    environment_success: bool
    halt_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "dispatch_index": self.dispatch_index,
            "tool_name": self.tool_name,
            "physical": self.physical,
            "tool_call_ref": self.tool_call_ref,
            "tool_result_ref": self.tool_result_ref,
            "snapshot_before": copy.deepcopy(self.snapshot_before),
            "snapshot_after": copy.deepcopy(self.snapshot_after),
            "result": copy.deepcopy(self.result),
            "environment_success": self.environment_success,
            "halt_reason": self.halt_reason,
        }


def evolving_transition_enabled(hpk_runtime: Any) -> bool:
    return bool(
        hpk_runtime is not None
        and getattr(hpk_runtime, "evolving_enabled", False) is True
    )


def _default_nonce() -> str:
    return "afkattempt_" + uuid.uuid4().hex


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 16:
        raise HPKTransitionRuntimeError(
            "private transition payload nesting is too deep"
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HPKTransitionRuntimeError(
                "private transition payload contains a non-finite number"
            )
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth=depth + 1) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist(), depth=depth + 1)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict(), depth=depth + 1)
    raise HPKTransitionRuntimeError(
        f"private transition payload contains unsupported {type(value).__name__}"
    )


def _snapshot_boundary(snapshot: Any | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    payload: dict[str, Any] = {}
    for key in (
        "step_count",
        "step_limit",
        "eval_success",
        "check_success",
        "max_reward",
        "joint_vector",
        "left_endpose",
        "right_endpose",
    ):
        if hasattr(snapshot, key):
            payload[key] = _jsonable(getattr(snapshot, key))
    return payload or None


def _snapshot_step(snapshot: Any | None) -> int | None:
    raw = getattr(snapshot, "step_count", None)
    return (
        raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else None
    )


def _call_payload(call: RecoveryToolCall) -> dict[str, Any]:
    return {
        "tool_name": str(call.tool_name),
        "args": _jsonable(dict(call.args or {})),
    }


def _result_payload(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "tool_name": str(getattr(value, "tool_name", "")),
        "success": bool(getattr(value, "success", False)),
        "message": str(getattr(value, "message", "")),
        "details": _jsonable(getattr(value, "details", {}) or {}),
    }


def _deterministic_runtime_validation(
    action_effect: Mapping[str, Any],
) -> dict[str, Any]:
    trusted_keys = (
        "grasp_validation",
        "runtime_grasp_validation",
        "place_validation",
        "runtime_place_validation",
        "attachment_state",
        "robot_state",
        "scene_memory_delta",
    )
    return {
        key: copy.deepcopy(action_effect[key])
        for key in trusted_keys
        if isinstance(action_effect.get(key), Mapping)
    }


def _prepared_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _jsonable(dict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, Mapping):
            return _jsonable(dict(mapped))
    raise HPKTransitionRuntimeError("prepare_action_transition must return a mapping")


def _normalize_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    binding = copy.deepcopy(dict(value))
    selected_entry = binding.get(
        "selected_hpk_entry_id", binding.get("selected_entry_id")
    )
    selected_geometry = binding.get(
        "geometric_strategy_id", binding.get("selected_geometric_strategy_id")
    )
    retrieved = binding.get("retrieved_hpk_entry_ids")
    if not isinstance(retrieved, list):
        retrieved = [selected_entry] if selected_entry else []
    raw_group = binding.get("semantic_attempt_group")
    semantic_group = (
        copy.deepcopy(dict(raw_group)) if isinstance(raw_group, Mapping) else {}
    )
    raw_indices = semantic_group.get("physical_dispatch_indices", [])
    group_indices = (
        [
            int(index)
            for index in raw_indices
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0
        ]
        if isinstance(raw_indices, list)
        else []
    )
    normalized = {
        "condition_id": binding.get("condition_id"),
        "task_strategy_id": binding.get("task_strategy_id"),
        "retrieved_hpk_entry_ids": list(retrieved),
        "selected_hpk_entry_id": selected_entry,
        "geometric_strategy_id": selected_geometry,
        "operation": binding.get("operation"),
        "arm": binding.get("arm"),
        "target_role": binding.get("target_role"),
        "target_relation": binding.get("target_relation"),
        "selected_candidate_private_ref": binding.get("selected_candidate_private_ref"),
        "candidate_geometry_features": binding.get("candidate_geometry_features"),
        "geometric_compliance": binding.get("geometric_compliance", "unverified"),
        "realization_status": binding.get(
            "realization_status",
            binding.get("strategy_realization_status", "unknown"),
        ),
        "target_identity_status": binding.get("target_identity_status", "unknown"),
        "expected_effect": binding.get("expected_effect"),
        "verifier_conflicts": list(binding.get("verifier_conflicts", []) or []),
        "oracle_derived": bool(binding.get("oracle_derived", False)),
        "expert_derived": bool(binding.get("expert_derived", False)),
        "_hpk_selection_token": binding.get("_hpk_selection_token"),
        "_hpk_runtime_binding_id": binding.get("_hpk_runtime_binding_id"),
        "semantic_attempt_group": {
            "grouped": semantic_group.get("grouped") is True,
            "physical_dispatch_indices": group_indices,
        },
    }
    public_usage_audit_id = binding.get("public_usage_audit_id")
    private_ranking_audit_id = binding.get("private_ranking_audit_id")
    if (public_usage_audit_id is None) != (private_ranking_audit_id is None):
        raise HPKTransitionRuntimeError(
            "public usage and private ranking audit IDs must be paired"
        )
    if public_usage_audit_id is not None:
        validate_content_id(
            public_usage_audit_id,
            prefix="afkaudit",
            path="public usage audit ID",
        )
        validate_content_id(
            private_ranking_audit_id,
            prefix="afkprivrank",
            path="private ranking audit ID",
        )
        normalized["public_usage_audit_id"] = public_usage_audit_id
        normalized["private_ranking_audit_id"] = private_ranking_audit_id
    exploration_public = binding.get("safe_exploration_public_audit_id")
    exploration_private = binding.get("safe_exploration_private_audit_id")
    if (exploration_public is None) != (exploration_private is None):
        raise HPKTransitionRuntimeError(
            "safe exploration public/private audit IDs must be paired"
        )
    if exploration_public is not None:
        validate_content_id(
            exploration_public,
            prefix="afkexplore",
            path="safe exploration public audit ID",
        )
        validate_content_id(
            exploration_private,
            prefix="afkprivexplore",
            path="safe exploration private audit ID",
        )
        normalized["safe_exploration_public_audit_id"] = exploration_public
        normalized["safe_exploration_private_audit_id"] = exploration_private
    return normalized


def _physical_indices(calls: Sequence[RecoveryToolCall]) -> tuple[int, ...]:
    return tuple(
        index
        for index, call in enumerate(calls)
        if str(call.tool_name) in PHYSICAL_RECOVERY_TOOLS
    )


def prove_semantic_attempt_group(
    *,
    calls: Sequence[RecoveryToolCall],
    binding: Mapping[str, Any],
) -> bool:
    """Recompute one strict semantic chain from typed calls and binding."""

    operation = str(binding.get("operation", "") or "").strip().lower()
    execution_side = str(binding.get("arm", "") or "").strip().lower()
    token = str(binding.get("_hpk_selection_token", "") or "")
    terminal = _SEMANTIC_TERMINALS.get(operation)
    if (
        terminal is None
        or execution_side not in {"left", "right"}
        or not token
        or not isinstance(binding.get("expected_effect"), Mapping)
    ):
        return False
    physical = [call for call in calls if call.tool_name in PHYSICAL_RECOVERY_TOOLS]
    if not physical:
        return False
    phases: list[str] = []
    terminals: list[int] = []
    for index, call in enumerate(physical):
        args = call.args or {}
        if str(args.get("arm", "") or "").strip().lower() != execution_side:
            return False
        call_binding = args.get("_hpk_usage_binding")
        if (
            not isinstance(call_binding, Mapping)
            or call_binding.get("_hpk_selection_token") != token
        ):
            return False
        tool_name = str(call.tool_name)
        marker = args.get("_hpk_semantic_phase")
        if (
            tool_name in {"move_ee_to_grounded_instance", "move_ee_to_pose"}
            and marker is None
        ):
            phase = "candidate_move"
        else:
            if not isinstance(marker, Mapping):
                return False
            expected_phase = _SEMANTIC_PHASE_BY_TOOL.get((operation, tool_name))
            expected_keys = {
                "schema",
                "marker_id",
                "selection_token",
                "operation",
                "arm",
                "tool_name",
                "phase",
            }
            if expected_phase is None or set(marker) != expected_keys:
                return False
            identity = dict(marker)
            marker_id = identity.pop("marker_id")
            if identity != {
                "schema": "roboharn_evo/hpk/semantic_phase/v1",
                "selection_token": token,
                "operation": operation,
                "arm": execution_side,
                "tool_name": tool_name,
                "phase": expected_phase,
            } or marker_id != stable_content_id("afkphase", identity):
                return False
            phase = expected_phase
        phases.append(phase)
        if tool_name == terminal:
            terminals.append(index)
    if len(terminals) != 1:
        return False
    terminal_index = terminals[0]
    if any(phase != "candidate_move" for phase in phases[:terminal_index]):
        return False
    tail = phases[terminal_index + 1 :]
    if any(phase == "candidate_move" for phase in tail):
        return False
    allowed_tail = _SEMANTIC_TAILS[operation]
    return tail == [phase for phase in allowed_tail if phase in tail]


class EvolvingActionTransitionSession:
    """One guarded recovery batch and its immutable evolving transition."""

    def __init__(
        self,
        *,
        hpk_runtime: Any,
        episode_id: str | int,
        seed: int,
        nonce: str,
        calls: Sequence[RecoveryToolCall],
        pre_effect_state: Mapping[str, Any],
        snapshot_before: Any | None,
        context: Mapping[str, Any],
        private_sink: PrivateEventSink,
        public_sink: PublicEventSink,
    ) -> None:
        self.hpk_runtime = hpk_runtime
        self.episode_id = episode_id
        self.seed = int(seed)
        self.action_attempt_nonce = str(nonce)
        self.calls = tuple(
            RecoveryToolCall(
                tool_name=str(call.tool_name),
                args=copy.deepcopy(dict(call.args or {})),
            )
            for call in calls
        )
        self.pre_effect_state = _jsonable(dict(pre_effect_state))
        self.snapshot_before = snapshot_before
        self.context = _jsonable(dict(context))
        self.private_sink = private_sink
        self.public_sink = public_sink
        self.physical_indices = _physical_indices(self.calls)
        self._records: dict[int, DispatchBoundaryRecord] = {}
        self._before_seen: set[int] = set()
        self._completed = False
        self._sealed = False
        self._sealed_result_ref: str | None = None
        self._sealed_result_bundle: list[dict[str, Any]] | None = None

        prepare = getattr(hpk_runtime, "prepare_action_transition", None)
        accept = getattr(hpk_runtime, "accept_action_transition", None)
        if not callable(prepare) or not callable(accept):
            raise HPKTransitionRuntimeError(
                "evolving HPK runtime requires prepare_action_transition and "
                "accept_action_transition hooks"
            )
        prepared = prepare(
            action_attempt_nonce=self.action_attempt_nonce,
            episode_id=self.episode_id,
            seed=self.seed,
            calls=tuple(
                RecoveryToolCall(
                    tool_name=call.tool_name,
                    args=copy.deepcopy(call.args),
                )
                for call in self.calls
            ),
            pre_effect_state=copy.deepcopy(self.pre_effect_state),
            context=copy.deepcopy(self.context),
        )
        prepared_payload = _prepared_mapping(prepared)
        self.binding = _normalize_binding(prepared_payload)
        self.runtime_binding_id = str(
            self.binding.get("_hpk_runtime_binding_id", "") or ""
        ).strip()
        validate_content_id(
            self.runtime_binding_id,
            prefix="afkruntime",
            path="prepared runtime_binding_id",
        )
        prepared_nonce = str(
            prepared_payload.get("action_attempt_nonce", "") or ""
        ).strip()
        if prepared_nonce and prepared_nonce != self.action_attempt_nonce:
            raise HPKTransitionRuntimeError(
                "evolving runtime changed the prepared action_attempt_nonce"
            )
        call_payloads = [_call_payload(call) for call in self.calls]
        self.tool_call_ref = stable_content_id(
            "afktcall",
            {
                "episode_id": self.episode_id,
                "action_attempt_nonce": self.action_attempt_nonce,
                "runtime_binding_id": self.runtime_binding_id,
                "guarded_calls": call_payloads,
            },
        )
        # Validate the complete pre-dispatch binding before physical execution.
        self._build_transition(
            tool_result_ref=None,
            physical_action_executed=False,
            motion_status="unknown",
            realization_status="unknown",
            env_step_before=_snapshot_step(snapshot_before),
            env_step_after=None,
            post_effect_state=None,
            effect_observation_scope="missing",
            observed_effect=None,
            verifier_sources=[],
            verifier_conflicts=self.binding["verifier_conflicts"],
            infrastructure_valid=True,
        )
        self.private_sink(
            ACTION_ATTEMPT_PREPARED_EVENT,
            {
                "schema": "trace/hpk_action_attempt_prepared/v2",
                "event_id": self.tool_call_ref,
                "episode_id": self.episode_id,
                "seed": self.seed,
                "action_attempt_nonce": self.action_attempt_nonce,
                "runtime_binding_id": self.runtime_binding_id,
                "guarded_calls": call_payloads,
                "physical_dispatch_indices": list(self.physical_indices),
                "binding": copy.deepcopy(self.binding),
                "snapshot_before": _snapshot_boundary(snapshot_before),
            },
        )

    @property
    def dispatch_observations(self) -> tuple[DispatchBoundaryRecord, ...]:
        return tuple(self._records[index] for index in sorted(self._records))

    def observe_dispatch(self, observation: RecoveryDispatchObservation) -> None:
        if self._completed:
            raise HPKTransitionRuntimeError(
                "cannot append dispatch evidence after transition finalization"
            )
        index = int(observation.dispatch_index)
        if index < 0 or index >= len(self.calls):
            raise HPKTransitionRuntimeError("dispatcher reported an invalid call index")
        expected_call = self.calls[index]
        if str(expected_call.tool_name) != str(
            observation.call.tool_name
        ) or _call_payload(expected_call) != _call_payload(observation.call):
            raise HPKTransitionRuntimeError(
                "dispatcher call does not match the frozen guarded batch"
            )
        call_ref = stable_content_id(
            "afktcall",
            {
                "tool_call_bundle_ref": self.tool_call_ref,
                "dispatch_index": index,
                "call": _call_payload(observation.call),
            },
        )
        before = _snapshot_boundary(observation.snapshot_before)
        after = _snapshot_boundary(observation.snapshot_after)
        result = _result_payload(observation.result)
        result_ref = None
        if result is not None:
            result_ref = stable_content_id(
                "afktresult",
                {
                    "tool_call_ref": call_ref,
                    "phase": str(observation.phase),
                    "result": result,
                    "snapshot_after": after,
                },
            )
        phase = str(observation.phase)
        if phase == "before":
            if index in self._before_seen or index in self._records:
                raise HPKTransitionRuntimeError("dispatcher repeated a before boundary")
            self._before_seen.add(index)
        elif phase == "after":
            if index not in self._before_seen or index in self._records:
                raise HPKTransitionRuntimeError(
                    "dispatcher after boundary has no unique before boundary"
                )
        elif phase == "skipped":
            if index in self._before_seen or index in self._records:
                raise HPKTransitionRuntimeError(
                    "skipped dispatcher call must not have an executed boundary"
                )
        else:
            raise HPKTransitionRuntimeError(
                f"unsupported dispatcher observation phase: {phase!r}"
            )
        payload = {
            "schema": "trace/hpk_action_tool_boundary/v2",
            "event_id": result_ref or call_ref,
            "tool_call_bundle_ref": self.tool_call_ref,
            "action_attempt_nonce": self.action_attempt_nonce,
            "runtime_binding_id": self.runtime_binding_id,
            "phase": phase,
            "dispatch_index": index,
            "call": _call_payload(observation.call),
            "result": result,
            "snapshot_before": before,
            "snapshot_after": after,
            "environment_success": bool(observation.environment_success),
            "halt_reason": str(observation.halt_reason or ""),
        }
        self.private_sink(ACTION_TOOL_BOUNDARY_EVENT, payload)
        record_dispatch = getattr(self.hpk_runtime, "record_dispatch_observation", None)
        if callable(record_dispatch):
            record_dispatch(copy.deepcopy(payload))
        if phase in {"after", "skipped"}:
            self._records[index] = DispatchBoundaryRecord(
                phase=phase,
                dispatch_index=index,
                tool_name=str(observation.call.tool_name),
                physical=index in self.physical_indices,
                tool_call_ref=call_ref,
                tool_result_ref=result_ref,
                snapshot_before=before,
                snapshot_after=after,
                result=result,
                environment_success=bool(observation.environment_success),
                halt_reason=str(observation.halt_reason or ""),
            )

    def _execution_status(self) -> tuple[bool, str]:
        records = [self._records.get(index) for index in self.physical_indices]
        if not records or any(record is None for record in records):
            return False, "unknown"
        executed = False
        failed = False
        for record in records:
            assert record is not None
            result = record.result or {}
            details = result.get("details")
            details = details if isinstance(details, dict) else {}
            skipped = record.phase == "skipped" or details.get("skipped") is True
            success = result.get("success") is True
            partial_steps = details.get("executed_steps")
            executed = bool(
                executed
                or not skipped
                and (
                    success
                    or isinstance(partial_steps, int)
                    and not isinstance(partial_steps, bool)
                    and partial_steps > 0
                )
            )
            if skipped or not success or details.get("target_reached") is False:
                failed = True
        if failed:
            return executed, "failed_before_effect"
        return executed, "completed" if executed else "unknown"

    def _effect_observation_scope(
        self, post_effect_state: Mapping[str, Any] | None
    ) -> str:
        records = [self._records.get(index) for index in self.physical_indices]
        ordered_records = [self._records[index] for index in sorted(self._records)]
        fresh = _has_fresh_effect_observation(
            ordered_records,
            env_step_before=(
                _step_from_boundary(records[0].snapshot_before)
                if records and records[0] is not None
                else _snapshot_step(self.snapshot_before)
            ),
        )
        if len(records) > 1:
            group = self.binding["semantic_attempt_group"]
            if not (
                group["grouped"] is True
                and group["physical_dispatch_indices"] == list(self.physical_indices)
                and all(record is not None for record in records)
                and records[0].phase == "after"
                and records[-1].phase == "after"
                and records[0].snapshot_before is not None
                and records[-1].snapshot_after is not None
                and post_effect_state is not None
                and self.binding["expected_effect"] is not None
                and fresh
            ):
                return "batch_unseparated"
            return "independent"
        if (
            len(records) == 1
            and records[0] is not None
            and records[0].phase == "after"
            and records[0].snapshot_before is not None
            and records[0].snapshot_after is not None
            and post_effect_state is not None
            and fresh
        ):
            return "independent"
        return "missing"

    def _build_transition(self, **outcome: Any) -> ActionEffectTransitionV1:
        return build_action_effect_transition(
            episode_id=self.episode_id,
            action_attempt_nonce=self.action_attempt_nonce,
            condition_id=self.binding["condition_id"],
            task_strategy_id=self.binding["task_strategy_id"],
            retrieved_hpk_entry_ids=self.binding["retrieved_hpk_entry_ids"],
            selected_hpk_entry_id=self.binding["selected_hpk_entry_id"],
            geometric_strategy_id=self.binding["geometric_strategy_id"],
            operation=self.binding["operation"],
            arm=self.binding["arm"],
            target_role=self.binding["target_role"],
            target_relation=self.binding["target_relation"],
            selected_candidate_private_ref=self.binding[
                "selected_candidate_private_ref"
            ],
            candidate_geometry_features=self.binding["candidate_geometry_features"],
            geometric_compliance=self.binding["geometric_compliance"],
            tool_call_ref=self.tool_call_ref,
            pre_effect_state=copy.deepcopy(self.pre_effect_state),
            expected_effect=self.binding["expected_effect"],
            target_identity_status=self.binding["target_identity_status"],
            oracle_derived=self.binding["oracle_derived"],
            expert_derived=self.binding["expert_derived"],
            **outcome,
        )

    def _seal_result_bundle(self) -> tuple[str, list[dict[str, Any]]]:
        if self._sealed:
            assert self._sealed_result_ref is not None
            assert self._sealed_result_bundle is not None
            return self._sealed_result_ref, copy.deepcopy(self._sealed_result_bundle)
        if set(self._records) != set(range(len(self.calls))):
            raise HPKTransitionRuntimeError(
                "dispatcher did not close every frozen tool-call boundary"
            )
        result_bundle = [
            self._records[index].to_dict() for index in sorted(self._records)
        ]
        tool_result_ref = stable_content_id(
            "afktresult",
            {
                "tool_call_ref": self.tool_call_ref,
                "action_attempt_nonce": self.action_attempt_nonce,
                "runtime_binding_id": self.runtime_binding_id,
                "dispatch_results": result_bundle,
            },
        )
        self.private_sink(
            ACTION_TOOL_RESULT_BUNDLE_EVENT,
            {
                "schema": "trace/hpk_action_tool_result_bundle/v2",
                "event_id": tool_result_ref,
                "tool_call_ref": self.tool_call_ref,
                "action_attempt_nonce": self.action_attempt_nonce,
                "runtime_binding_id": self.runtime_binding_id,
                "dispatch_results": result_bundle,
            },
        )
        self._sealed = True
        self._sealed_result_ref = tool_result_ref
        self._sealed_result_bundle = copy.deepcopy(result_bundle)
        return tool_result_ref, result_bundle

    def finalize(
        self,
        *,
        action_effect: Mapping[str, Any],
        post_effect_state: Mapping[str, Any] | None,
        infrastructure_valid: bool = True,
    ) -> ActionEffectTransitionV1 | None:
        if self._completed:
            raise HPKTransitionRuntimeError(
                "action transition session was already finalized"
            )
        self._seal_result_bundle()
        take_pending = getattr(self.hpk_runtime, "take_pending_action_transition", None)
        pending = take_pending() if callable(take_pending) else None
        segments = list(getattr(pending, "segments", []) or []) + [self]
        if pending is not None and (
            str(getattr(pending, "nonce", "")) != self.action_attempt_nonce
            or str(getattr(getattr(pending, "selection", None), "token", ""))
            != str(self.binding.get("_hpk_selection_token", "") or "")
        ):
            raise HPKTransitionRuntimeError(
                "pending semantic continuation changed its nonce or selection"
            )
        effect_payload = _jsonable(dict(action_effect))
        if _should_defer_semantic_attempt(
            segments=segments,
            binding=segments[0].binding,
            action_effect=effect_payload,
            infrastructure_valid=bool(infrastructure_valid),
        ):
            defer = getattr(self.hpk_runtime, "defer_action_transition", None)
            selection_token = str(
                segments[0].binding.get("_hpk_selection_token", "") or ""
            )
            if not callable(defer) or not selection_token:
                raise HPKTransitionRuntimeError(
                    "runtime cannot retain a typed pending semantic action"
                )
            started, latest = _segment_step_bounds(segments)
            defer(
                nonce=self.action_attempt_nonce,
                selection_token=selection_token,
                segments=segments,
                env_step_started=started,
                env_step_latest=latest,
                post_effect_state=post_effect_state,
            )
            self._completed = True
            return None
        transition = _finalize_semantic_segments(
            segments=segments,
            action_effect=effect_payload,
            post_effect_state=post_effect_state,
            infrastructure_valid=bool(infrastructure_valid),
        )
        for segment in segments:
            segment._completed = True
        return transition


def _has_fresh_effect_observation(
    records: Sequence[DispatchBoundaryRecord],
    *,
    env_step_before: int | None,
) -> bool:
    """Bind effect evidence to a successful, strictly newer dispatch snapshot."""

    if env_step_before is None or not records:
        return False
    explicit_observations = [
        record for record in records if record.tool_name == "reobserve_scene"
    ]
    evidence = explicit_observations[-1] if explicit_observations else records[-1]
    result = evidence.result or {}
    after = _step_from_boundary(evidence.snapshot_after)
    return bool(
        evidence.phase == "after"
        and result.get("success") is True
        and after is not None
        and after > env_step_before
    )


def _segment_step_bounds(
    segments: Sequence[EvolvingActionTransitionSession],
) -> tuple[int, int]:
    physical = [
        segment._records[index]
        for segment in segments
        for index in segment.physical_indices
        if index in segment._records
    ]
    ordered = [
        segment._records[index]
        for segment in segments
        for index in sorted(segment._records)
    ]
    before = (
        _step_from_boundary(physical[0].snapshot_before)
        if physical
        else _snapshot_step(segments[0].snapshot_before)
    )
    after = _step_from_boundary(ordered[-1].snapshot_after) if ordered else None
    if before is None or after is None:
        raise HPKTransitionRuntimeError(
            "pending semantic action requires explicit step boundaries"
        )
    return before, after


def _should_defer_semantic_attempt(
    *,
    segments: Sequence[EvolvingActionTransitionSession],
    binding: Mapping[str, Any],
    action_effect: Mapping[str, Any],
    infrastructure_valid: bool,
) -> bool:
    """Recognize only exact deterministic-runtime pending boundaries."""

    allow_defer = getattr(
        segments[0].hpk_runtime if segments else None,
        "allow_pending_semantic_defer",
        None,
    )
    if (
        not infrastructure_valid
        or not callable(allow_defer)
        or not allow_defer(segment_count=len(segments))
        or binding.get("target_identity_status") != "bound"
        or binding.get("geometric_compliance") is not True
        or not isinstance(binding.get("expected_effect"), Mapping)
        or not isinstance(binding.get("semantic_attempt_group"), Mapping)
        or binding["semantic_attempt_group"].get("grouped") is not True
    ):
        return False
    segment = segments[0]
    operation = str(binding.get("operation", "") or "").strip().lower()
    if operation == "grasp":
        exact_boundary = any(
            call.tool_name == "close_gripper"
            and (call.args or {}).get("_runtime_grasp_close_boundary") is True
            for call in segment.calls
        )
        validation = action_effect.get("runtime_grasp_validation")
        resolved = isinstance(validation, Mapping) and isinstance(
            validation.get("verified"), bool
        )
        return exact_boundary and not resolved
    if operation == "place":
        exact_release = any(call.tool_name == "open_gripper" for call in segment.calls)
        validation = action_effect.get("runtime_place_validation")
        return bool(
            exact_release
            and isinstance(validation, Mapping)
            and validation.get("applicable") is True
            and validation.get("release_verification_required") is True
            and validation.get("verified") is not True
            and validation.get("placement_recovery_required") is not True
        )
    return False


def _aggregate_refs(
    segments: Sequence[EvolvingActionTransitionSession],
) -> tuple[str, str]:
    first = segments[0]
    call_refs = [segment.tool_call_ref for segment in segments]
    result_refs = [segment._seal_result_bundle()[0] for segment in segments]
    if len(segments) == 1:
        return call_refs[0], result_refs[0]
    call_ref = stable_content_id(
        "afktcall",
        {
            "episode_id": first.episode_id,
            "action_attempt_nonce": first.action_attempt_nonce,
            "runtime_binding_id": first.runtime_binding_id,
            "segment_tool_call_refs": call_refs,
        },
    )
    result_ref = stable_content_id(
        "afktresult",
        {
            "tool_call_ref": call_ref,
            "action_attempt_nonce": first.action_attempt_nonce,
            "runtime_binding_id": first.runtime_binding_id,
            "segment_tool_result_refs": result_refs,
        },
    )
    return call_ref, result_ref


def _combined_records(
    segments: Sequence[EvolvingActionTransitionSession],
) -> tuple[
    tuple[RecoveryToolCall, ...],
    tuple[DispatchBoundaryRecord, ...],
    tuple[int, ...],
]:
    calls: list[RecoveryToolCall] = []
    records: list[DispatchBoundaryRecord] = []
    physical: list[int] = []
    offset = 0
    for segment in segments:
        calls.extend(segment.calls)
        physical.extend(offset + index for index in segment.physical_indices)
        for index in sorted(segment._records):
            record = segment._records[index]
            records.append(
                DispatchBoundaryRecord(
                    phase=record.phase,
                    dispatch_index=offset + record.dispatch_index,
                    tool_name=record.tool_name,
                    physical=record.physical,
                    tool_call_ref=record.tool_call_ref,
                    tool_result_ref=record.tool_result_ref,
                    snapshot_before=copy.deepcopy(record.snapshot_before),
                    snapshot_after=copy.deepcopy(record.snapshot_after),
                    result=copy.deepcopy(record.result),
                    environment_success=record.environment_success,
                    halt_reason=record.halt_reason,
                )
            )
        offset += len(segment.calls)
    return tuple(calls), tuple(records), tuple(physical)


def _aggregate_execution_status(
    records: Sequence[DispatchBoundaryRecord],
    physical_indices: Sequence[int],
) -> tuple[bool, str]:
    by_index = {record.dispatch_index: record for record in records}
    selected = [by_index.get(index) for index in physical_indices]
    if not selected or any(record is None for record in selected):
        return False, "unknown"
    executed = False
    failed = False
    for record in selected:
        assert record is not None
        result = record.result or {}
        details = result.get("details")
        details = details if isinstance(details, Mapping) else {}
        skipped = record.phase == "skipped" or details.get("skipped") is True
        success = result.get("success") is True
        partial_steps = details.get("executed_steps")
        executed = bool(
            executed
            or not skipped
            and (
                success
                or isinstance(partial_steps, int)
                and not isinstance(partial_steps, bool)
                and partial_steps > 0
            )
        )
        if skipped or not success or details.get("target_reached") is False:
            failed = True
    if failed:
        return executed, "failed_before_effect"
    return executed, "completed" if executed else "unknown"


def _finalize_semantic_segments(
    *,
    segments: Sequence[EvolvingActionTransitionSession],
    action_effect: Mapping[str, Any],
    post_effect_state: Mapping[str, Any] | None,
    infrastructure_valid: bool,
) -> ActionEffectTransitionV1:
    if not segments:
        raise HPKTransitionRuntimeError("semantic action has no dispatch segment")
    first = segments[0]
    if any(
        segment.hpk_runtime is not first.hpk_runtime
        or segment.episode_id != first.episode_id
        or segment.action_attempt_nonce != first.action_attempt_nonce
        or segment.runtime_binding_id != first.runtime_binding_id
        for segment in segments
    ):
        raise HPKTransitionRuntimeError(
            "semantic action segments do not share runtime, episode, and nonce"
        )
    calls, records, physical_indices = _combined_records(segments)
    binding = copy.deepcopy(first.binding)
    selection_token = str(binding.get("_hpk_selection_token", "") or "")
    prove = getattr(first.hpk_runtime, "prove_semantic_attempt_group", None)
    grouped = bool(
        prove(calls=calls, selection_token=selection_token)
        if selection_token and callable(prove)
        else len(segments) == 1
        and isinstance(binding.get("semantic_attempt_group"), Mapping)
        and binding["semantic_attempt_group"].get("grouped") is True
    )
    binding["semantic_attempt_group"] = {
        "grouped": grouped,
        "physical_dispatch_indices": list(physical_indices),
    }
    physical_executed, motion_status = _aggregate_execution_status(
        records, physical_indices
    )
    realization = str(binding.get("realization_status", "unknown") or "unknown")
    semantic_terminal = _SEMANTIC_TERMINALS.get(
        str(binding.get("operation", "") or "").strip().lower()
    )
    terminal_executed = bool(
        semantic_terminal and any(call.tool_name == semantic_terminal for call in calls)
    )
    if motion_status != "completed" and realization != "violated":
        realization = "unknown"
    post_state = (
        None if post_effect_state is None else _jsonable(dict(post_effect_state))
    )
    env_step_before, env_step_after = _segment_step_bounds(segments)
    fresh = _has_fresh_effect_observation(records, env_step_before=env_step_before)
    # Approach/clearance without the operation's terminal tool has not yet
    # attempted the grasp/place/contact effect.  Keep its scope missing so it
    # cannot support or oppose that effect, even if a model emits a boolean.
    if (
        terminal_executed
        and (grouped or len(physical_indices) == 1)
        and post_state is not None
        and binding.get("expected_effect") is not None
        and fresh
    ):
        effect_scope = "independent"
    elif len(physical_indices) > 1 and not grouped:
        effect_scope = "batch_unseparated"
    else:
        effect_scope = "missing"
    expected = binding.get("expected_effect")
    observed = None
    verifier_sources: list[str] = []
    effect_payload = _jsonable(dict(action_effect))
    if expected is not None:
        observed_record = AbstractEffectExtractor().extract(
            expected,
            pre_effect_state=first.pre_effect_state,
            post_effect_state=post_state,
            runtime_validation=_deterministic_runtime_validation(effect_payload),
            verifier_result=effect_payload,
            motion_status=motion_status,
        )
        observed = observed_record.to_dict()
        verifier_sources = list(observed.get("verifier_sources", []))
    conflicts = list(binding.get("verifier_conflicts", []) or [])
    raw_conflicts = effect_payload.get("verifier_conflicts")
    if isinstance(raw_conflicts, list):
        conflicts.extend(str(item) for item in raw_conflicts if str(item))
    conflicts = list(dict.fromkeys(conflicts))
    tool_call_ref, tool_result_ref = _aggregate_refs(segments)
    transition = build_action_effect_transition(
        episode_id=first.episode_id,
        action_attempt_nonce=first.action_attempt_nonce,
        condition_id=binding.get("condition_id"),
        task_strategy_id=binding.get("task_strategy_id"),
        retrieved_hpk_entry_ids=binding.get("retrieved_hpk_entry_ids", []),
        selected_hpk_entry_id=binding.get("selected_hpk_entry_id"),
        geometric_strategy_id=binding.get("geometric_strategy_id"),
        operation=binding.get("operation"),
        arm=binding.get("arm"),
        target_role=binding.get("target_role"),
        target_relation=binding.get("target_relation"),
        selected_candidate_private_ref=binding.get("selected_candidate_private_ref"),
        candidate_geometry_features=binding.get("candidate_geometry_features"),
        geometric_compliance=binding.get("geometric_compliance", "unverified"),
        tool_call_ref=tool_call_ref,
        tool_result_ref=tool_result_ref,
        physical_action_executed=physical_executed,
        motion_status=motion_status,
        realization_status=realization,
        env_step_before=env_step_before,
        env_step_after=env_step_after,
        pre_effect_state=copy.deepcopy(first.pre_effect_state),
        post_effect_state=post_state,
        effect_observation_scope=effect_scope,
        target_identity_status=binding.get("target_identity_status", "unknown"),
        expected_effect=expected,
        observed_effect=observed,
        verifier_sources=verifier_sources,
        verifier_conflicts=conflicts,
        infrastructure_valid=bool(infrastructure_valid),
        oracle_derived=bool(binding.get("oracle_derived", False)),
        expert_derived=bool(binding.get("expert_derived", False)),
    )
    public = public_transition_projection(transition)
    first.private_sink(
        ACTION_EFFECT_TRANSITION_EVENT,
        {
            "schema": "trace/action_effect_transition/v2",
            "event_id": transition.stable_id,
            "runtime_binding_id": first.runtime_binding_id,
            "transition": transition.to_dict(),
        },
    )
    first.public_sink(PUBLIC_ACTION_EFFECT_TRANSITION_EVENT, public)
    aggregate_results = [record.to_dict() for record in records]
    first.hpk_runtime.accept_action_transition(
        transition=transition,
        public_record=copy.deepcopy(public),
        dispatch_observations=tuple(copy.deepcopy(aggregate_results)),
    )
    return transition


def finalize_pending_action_transition(
    *,
    hpk_runtime: Any,
    reason: str,
    post_effect_state: Mapping[str, Any] | None = None,
    infrastructure_valid: bool = True,
) -> ActionEffectTransitionV1 | None:
    """Close a pending attempt as unverified before mismatch/end/reset."""

    take = getattr(hpk_runtime, "take_pending_action_transition", None)
    pending = take() if callable(take) else None
    if pending is None:
        return None
    detail = str(reason or "pending_semantic_attempt_closed").strip()
    effect = {
        "effect_verified": "unverified",
        "effect_type": str(getattr(pending, "operation", "unknown") or "unknown"),
        "verifier_conflicts": [detail],
    }
    return _finalize_semantic_segments(
        segments=tuple(pending.segments),
        action_effect=effect,
        post_effect_state=(
            post_effect_state
            if post_effect_state is not None
            else getattr(pending, "post_effect_state", None)
        ),
        infrastructure_valid=bool(infrastructure_valid),
    )


def _step_from_boundary(value: Mapping[str, Any] | None) -> int | None:
    if not isinstance(value, Mapping):
        return None
    step = value.get("step_count")
    return (
        step
        if isinstance(step, int) and not isinstance(step, bool) and step >= 0
        else None
    )


def begin_evolving_action_transition(
    *,
    hpk_runtime: Any,
    episode_id: str | int,
    seed: int,
    calls: Sequence[RecoveryToolCall],
    pre_effect_state: Mapping[str, Any],
    snapshot_before: Any | None,
    context: Mapping[str, Any],
    private_sink: PrivateEventSink,
    public_sink: PublicEventSink,
    nonce_factory: NonceFactory | None = None,
) -> EvolvingActionTransitionSession | None:
    """Create a session only for an explicit evolving runtime and physical batch."""

    if not evolving_transition_enabled(hpk_runtime):
        return None
    frozen_calls = tuple(calls)
    continuation_nonce: str | None = None
    pending_conflict: str | None = None
    continuation = getattr(hpk_runtime, "pending_action_continuation", None)
    if callable(continuation):
        continuation_nonce, pending_conflict = continuation(
            calls=frozen_calls,
            episode_id=episode_id,
            snapshot_before=snapshot_before,
        )
    if pending_conflict:
        finalize_pending_action_transition(
            hpk_runtime=hpk_runtime,
            reason=pending_conflict,
            post_effect_state=pre_effect_state,
        )
    if not _physical_indices(frozen_calls) and continuation_nonce is None:
        return None
    factory = nonce_factory or _default_nonce
    nonce = str(continuation_nonce or factory() or "").strip()
    if not nonce:
        raise HPKTransitionRuntimeError(
            "action transition nonce factory returned an empty value"
        )
    return EvolvingActionTransitionSession(
        hpk_runtime=hpk_runtime,
        episode_id=episode_id,
        seed=seed,
        nonce=nonce,
        calls=frozen_calls,
        pre_effect_state=pre_effect_state,
        snapshot_before=snapshot_before,
        context=context,
        private_sink=private_sink,
        public_sink=public_sink,
    )


__all__ = [
    "ACTION_ATTEMPT_PREPARED_EVENT",
    "ACTION_EFFECT_TRANSITION_EVENT",
    "ACTION_TOOL_BOUNDARY_EVENT",
    "ACTION_TOOL_RESULT_BUNDLE_EVENT",
    "HPKTransitionRuntimeError",
    "DispatchBoundaryRecord",
    "EvolvingActionTransitionSession",
    "PHYSICAL_RECOVERY_TOOLS",
    "PUBLIC_ACTION_EFFECT_TRANSITION_EVENT",
    "begin_evolving_action_transition",
    "evolving_transition_enabled",
    "finalize_pending_action_transition",
]
