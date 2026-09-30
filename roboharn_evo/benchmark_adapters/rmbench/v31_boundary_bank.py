"""RMBench-local capture helpers for the HPK v3.1 utility gate.

The utility experiment needs multiple *real* continuation states at the
boundary between a verified grasp and a subsequent realization.  Re-running
an episode from step zero for every boundary is unnecessary: RMBench is
already quiescent when one Agent control turn returns, so one rollout can
capture each distinct Runtime-confirmed holding transaction exactly once.

This module deliberately knows nothing about a task name, object class,
support type, seed policy, or language template.  It observes only the typed
manipulation state that the existing Runtime already uses to authorize
transport.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SCHEMA = "roboharn_evo/rmbench/v31/failure_boundary_bank"
FAILURE_BOUNDARY_CATEGORIES_V31 = frozenset(
    {"goal_inconsistency", "realization_failure"}
)


def validate_failure_category_coverage_v31(categories: Iterable[Any]) -> None:
    """Require the final Gate to exercise both preregistered failure classes."""

    observed = frozenset(str(value or "").strip() for value in categories)
    if observed != FAILURE_BOUNDARY_CATEGORIES_V31:
        raise ValueError(
            "the complete v3.1 Gate must cover goal_inconsistency and "
            "realization_failure exactly"
        )


def _positive_int(value: Any, *, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0 or value > maximum:
        raise ValueError(f"{label} must be between 1 and {maximum}")
    return value


def _non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _bounded_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    result = " ".join(value.strip().split())
    if not result or len(result) > 300:
        raise ValueError(f"{label} must be non-empty and at most 300 characters")
    return result


@dataclass(frozen=True, slots=True)
class RMBenchV31BoundaryBankConfig:
    """Strict capture-bank configuration owned by the RMBench experiment."""

    output_dir: Path
    max_boundaries: int = 4
    min_boundaries: int = 1
    capture_at_or_after_env_step: int = 0
    stop_after_bank_full: bool = False
    label_prefix: str = "confirmed holding boundary"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "RMBenchV31BoundaryBankConfig":
        if not isinstance(value, Mapping):
            raise TypeError("failure-boundary capture-bank config must be a mapping")
        allowed = {
            "mode",
            "output_dir",
            "max_boundaries",
            "min_boundaries",
            "capture_at_or_after_env_step",
            "stop_after_bank_full",
            "label_prefix",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "failure-boundary capture-bank config has unknown fields: "
                f"{sorted(unknown)}"
            )
        if str(value.get("mode", "") or "").strip().lower() != "capture_bank":
            raise ValueError("capture-bank mode must be capture_bank")
        raw_output = value.get("output_dir")
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ValueError("capture_bank.output_dir must be an explicit path")
        output_dir = Path(raw_output).expanduser().resolve()
        max_boundaries = _positive_int(
            value.get("max_boundaries", 4),
            label="capture_bank.max_boundaries",
            maximum=20,
        )
        min_boundaries = _positive_int(
            value.get("min_boundaries", 1),
            label="capture_bank.min_boundaries",
            maximum=20,
        )
        if min_boundaries > max_boundaries:
            raise ValueError("capture_bank.min_boundaries cannot exceed max_boundaries")
        stop_after = value.get("stop_after_bank_full", False)
        if not isinstance(stop_after, bool):
            raise TypeError("capture_bank.stop_after_bank_full must be a boolean")
        return cls(
            output_dir=output_dir,
            max_boundaries=max_boundaries,
            min_boundaries=min_boundaries,
            capture_at_or_after_env_step=_non_negative_int(
                value.get("capture_at_or_after_env_step", 0),
                label="capture_bank.capture_at_or_after_env_step",
            ),
            stop_after_bank_full=stop_after,
            label_prefix=_bounded_text(
                value.get("label_prefix", "confirmed holding boundary"),
                label="capture_bank.label_prefix",
            ),
        )

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "mode": "capture_bank",
            "schema": _SCHEMA,
            "typed_config": self,
            "output_dir": self.output_dir,
            "max_boundaries": self.max_boundaries,
            "min_boundaries": self.min_boundaries,
            "capture_at_or_after_env_step": self.capture_at_or_after_env_step,
            "stop_after_bank_full": self.stop_after_bank_full,
            "label_prefix": self.label_prefix,
        }


@dataclass(frozen=True, slots=True)
class ConfirmedHoldingBoundaryV31:
    """One unique Runtime-authorized object-holding transaction."""

    active_skill_ref: str
    arm: str
    held_instance_ref: str
    grasp_attempt_nonce: str
    phase: str

    @property
    def signature(self) -> tuple[str, str, str, str]:
        return (
            self.active_skill_ref,
            self.arm,
            self.held_instance_ref,
            self.grasp_attempt_nonce,
        )

    def to_private_dict(self) -> dict[str, str]:
        return {
            "active_skill_ref": self.active_skill_ref,
            "arm": self.arm,
            "held_instance_ref": self.held_instance_ref,
            "grasp_attempt_nonce": self.grasp_attempt_nonce,
            "phase": self.phase,
        }


def _active_skill_ref(agent: Any) -> str:
    memory_store = getattr(agent, "memory_store", None)
    state = getattr(memory_store, "state", None)
    active = getattr(state, "active_skill", None)
    return str(getattr(active, "skill_id", "") or "").strip()


def _confirmed_holding_boundary(
    *,
    active_skill_ref: str,
    manipulation_state: Any,
) -> ConfirmedHoldingBoundaryV31 | None:
    if not isinstance(manipulation_state, Mapping):
        return None
    if not active_skill_ref:
        return None

    confirmed: list[ConfirmedHoldingBoundaryV31] = []
    for arm in ("left", "right"):
        arm_state = manipulation_state.get(arm)
        if not isinstance(arm_state, Mapping):
            continue
        if (
            arm_state.get("holding_confirmed") is not True
            or arm_state.get("transport_authorized") is not True
        ):
            continue
        held_ref = str(arm_state.get("held_instance_id", "") or "").strip()
        nonce = str(arm_state.get("grasp_attempt_nonce", "") or "").strip()
        phase = str(arm_state.get("phase", "") or "").strip()
        if not held_ref or not nonce or not phase:
            continue
        confirmed.append(
            ConfirmedHoldingBoundaryV31(
                active_skill_ref=active_skill_ref,
                arm=arm,
                held_instance_ref=held_ref,
                grasp_attempt_nonce=nonce,
                phase=phase,
            )
        )
    return confirmed[0] if len(confirmed) == 1 else None


def confirmed_holding_boundary_from_memory_state_v31(
    memory_state: Mapping[str, Any],
) -> ConfirmedHoldingBoundaryV31 | None:
    """Read the same boundary from a serialized failure-boundary memory state."""

    if not isinstance(memory_state, Mapping):
        return None
    working = memory_state.get("working")
    if not isinstance(working, Mapping):
        return None
    return _confirmed_holding_boundary(
        active_skill_ref=str(working.get("active_skill_id", "") or "").strip(),
        manipulation_state=working.get("manipulation_state"),
    )


def confirmed_holding_boundary_v31(model: Any) -> ConfirmedHoldingBoundaryV31 | None:
    """Return a boundary only when exactly one arm has verified attachment.

    Both predicates are required because a provisional/evidence-only grasp is
    not a valid starting state for the shared staged-carry realization.
    """

    session = getattr(model, "session", None)
    agent = getattr(session, "agent", None)
    memory_store = getattr(agent, "memory_store", None)
    state = getattr(memory_store, "state", None)
    working = getattr(state, "working", None)
    return _confirmed_holding_boundary(
        active_skill_ref=_active_skill_ref(agent),
        manipulation_state=getattr(working, "manipulation_state", None),
    )


def _failure_evidence(
    *,
    no_progress_count: Any,
    empty_plan_count: Any,
    blocked_count: int,
    grounded_failure_count: int,
    monitor_status: Any,
) -> dict[str, Any] | None:
    counts: dict[str, int] = {}
    for label, value in (
        ("no_progress_control_turn_count", no_progress_count),
        ("empty_plan_count", empty_plan_count),
        ("blocked_grounded_setup_count", blocked_count),
        ("grounded_setup_failure_count", grounded_failure_count),
    ):
        counts[label] = (
            int(value)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
            else 0
        )
    if not any(counts.values()):
        return None
    status = " ".join(str(monitor_status or "").strip().split())
    return {**counts, "monitor_status": status or "unavailable"}


def recoverable_failure_boundary_v31(
    model: Any,
) -> tuple[ConfirmedHoldingBoundaryV31, dict[str, Any]] | None:
    """Return a strict held-object boundary only after Runtime failure evidence."""

    holding = confirmed_holding_boundary_v31(model)
    if holding is None:
        return None
    session = getattr(model, "session", None)
    agent = getattr(session, "agent", None)
    memory_store = getattr(agent, "memory_store", None)
    state = getattr(memory_store, "state", None)
    monitor = getattr(state, "monitor", None)
    evidence = _failure_evidence(
        no_progress_count=getattr(
            agent, "_pure_tool_control_no_progress_control_turns", 0
        ),
        empty_plan_count=getattr(agent, "_pure_tool_control_empty_plan_turns", 0),
        blocked_count=len(getattr(agent, "_blocked_grounded_setups", ()) or ()),
        grounded_failure_count=len(
            getattr(agent, "_grounded_setup_failures", {}) or {}
        ),
        monitor_status=getattr(monitor, "status", ""),
    )
    return None if evidence is None else (holding, evidence)


def recoverable_failure_boundary_from_agent_payload_v31(
    agent_payload: Mapping[str, Any],
) -> tuple[ConfirmedHoldingBoundaryV31, dict[str, Any]] | None:
    """Validate the same failure predicate from one serialized boundary."""

    if not isinstance(agent_payload, Mapping):
        return None
    memory_state = agent_payload.get("memory_state")
    holding = confirmed_holding_boundary_from_memory_state_v31(memory_state)
    if holding is None:
        return None
    counters = agent_payload.get("runtime_counters")
    counters = counters if isinstance(counters, Mapping) else {}
    memory_working = (
        memory_state.get("working") if isinstance(memory_state, Mapping) else None
    )
    monitor_status = ""
    if isinstance(memory_state, Mapping):
        monitor = memory_state.get("monitor")
        if isinstance(monitor, Mapping):
            monitor_status = monitor.get("status", "")
    evidence = _failure_evidence(
        no_progress_count=counters.get(
            "_pure_tool_control_no_progress_control_turns", 0
        ),
        empty_plan_count=counters.get("_pure_tool_control_empty_plan_turns", 0),
        blocked_count=len(agent_payload.get("blocked_grounded_setups", ()) or ()),
        grounded_failure_count=len(
            agent_payload.get("grounded_setup_failures", ()) or ()
        ),
        monitor_status=(
            monitor_status
            if monitor_status
            else (
                memory_working.get("monitor_status", "")
                if isinstance(memory_working, Mapping)
                else ""
            )
        ),
    )
    return None if evidence is None else (holding, evidence)


def classify_failure_boundary_v31(
    agent_payload: Mapping[str, Any],
    *,
    held_instance_ref: str,
    expected_target_ref: str,
) -> str:
    """Classify a held-object failure from existing candidate lifecycle facts."""

    held_ref = str(held_instance_ref or "").strip()
    target_ref = str(expected_target_ref or "").strip()
    if not held_ref or not target_ref:
        raise ValueError("held object and expected target refs must be explicit")
    components = agent_payload.get("components")
    components = components if isinstance(components, Mapping) else {}
    lifecycle = components.get("operation_candidate_lifecycle")
    lifecycle = lifecycle if isinstance(lifecycle, Mapping) else {}
    records = lifecycle.get("records")
    records = records if isinstance(records, list) else []
    place_failures: list[tuple[int, str]] = []
    for value in records:
        if not isinstance(value, Mapping):
            continue
        scope = value.get("scope")
        if (
            not isinstance(scope, list)
            or len(scope) < 4
            or str(scope[0] or "").strip() != held_ref
            or str(scope[2] or "").strip().casefold() != "place"
        ):
            continue
        failure_count = value.get("failure_count")
        if (
            isinstance(failure_count, bool)
            or not isinstance(failure_count, int)
            or failure_count <= 0
        ):
            continue
        step = value.get("env_step")
        step = step if isinstance(step, int) and not isinstance(step, bool) else -1
        place_failures.append((step, str(scope[3] or "").strip()))
    if not place_failures:
        return "realization_failure"
    latest_step = max(value[0] for value in place_failures)
    latest_targets = {
        value[1] for value in place_failures if value[0] == latest_step and value[1]
    }
    return (
        "goal_inconsistency"
        if any(value != target_ref for value in latest_targets)
        else "realization_failure"
    )


def capture_bank_boundary_path(
    config: RMBenchV31BoundaryBankConfig,
    *,
    seed: int,
    ordinal: int,
    env_step: int,
) -> Path:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("ordinal must be a non-negative integer")
    if isinstance(env_step, bool) or not isinstance(env_step, int) or env_step < 0:
        raise ValueError("env_step must be a non-negative integer")
    return config.output_dir / (
        f"boundary_seed_{seed:06d}_index_{ordinal:02d}_step_{env_step:04d}.json"
    )


__all__ = [
    "ConfirmedHoldingBoundaryV31",
    "RMBenchV31BoundaryBankConfig",
    "capture_bank_boundary_path",
    "classify_failure_boundary_v31",
    "confirmed_holding_boundary_from_memory_state_v31",
    "confirmed_holding_boundary_v31",
    "recoverable_failure_boundary_from_agent_payload_v31",
    "recoverable_failure_boundary_v31",
]
