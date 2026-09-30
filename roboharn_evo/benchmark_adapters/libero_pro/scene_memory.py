"""Minimal structured Scene Memory for the LIBERO Agent loop."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from roboharn_evo.benchmark_adapters.base import EpisodeState, NeutralObservation
from roboharn_evo.benchmark_adapters.libero_pro.effect_verifier import (
    LiberoDeterministicEffectVerifier,
    LiberoEffectVerdict,
)
from roboharn_evo.benchmark_adapters.libero_pro.perception import (
    LiberoSAM3PerceptionClient,
    PerceptionFrame,
)


@dataclass(slots=True)
class LiberoSceneMemory:
    """Only semantic physical state; no pose, action vector, or stable identity."""

    overall_task: str = ""
    revision: int = 0
    target_description: str = "unknown"
    target_visibility: str = "unknown"
    support_state: str = "unknown"
    held_state: str = "unknown"
    last_action: str = "none"
    last_effect_verdict: str = "unverified"
    last_effect_observation: str = "none"
    last_observed_step: int = 0

    def reset(self, task: str) -> None:
        self.overall_task = " ".join(str(task).strip().split())
        self.revision = 0
        self.target_description = "unknown"
        self.target_visibility = "unknown"
        self.support_state = "unknown"
        self.held_state = "unknown"
        self.last_action = "none"
        self.last_effect_verdict = "unverified"
        self.last_effect_observation = "none"
        self.last_observed_step = 0

    def _physical_state(self) -> tuple[str, str, str]:
        return (
            self.target_visibility,
            self.support_state,
            self.held_state,
        )

    def observe_before_action(
        self,
        *,
        frame: PerceptionFrame,
        observation: NeutralObservation,
        action: str,
    ) -> None:
        before = self._physical_state()
        target_queries = tuple(
            value for value in frame.queries if value.role == "target"
        )
        if target_queries:
            self.target_description = target_queries[0].text_prompt
        target_detections = frame.detections_for(role="target")
        if target_detections:
            self.target_visibility = "visible"
        elif target_queries and not frame.failures:
            self.target_visibility = "not visible"
        else:
            self.target_visibility = "unknown"
        aperture = _gripper_aperture(observation)
        if action == "grasp" and aperture is not None and aperture >= 0.05:
            if self.held_state in {"unknown", "not held"}:
                self.held_state = "not held"
                if self.target_visibility == "visible":
                    self.support_state = "supported"
        elif (
            action == "place"
            and aperture is not None
            and aperture <= 0.03
            and self.held_state in {"unknown", "possibly held", "held"}
        ):
            self.held_state = "possibly held"
            self.support_state = "carried"
        self.last_action = action
        self.last_observed_step = frame.step
        if self._physical_state() != before:
            self.revision += 1

    def apply_effect(self, verdict: LiberoEffectVerdict, *, step: int) -> None:
        before = self._physical_state()
        self.last_action = verdict.action
        self.last_effect_verdict = verdict.verdict
        self.last_effect_observation = verdict.observed_effect
        self.last_observed_step = step
        if verdict.action == "grasp":
            if verdict.verdict == "support":
                self.held_state = "held"
                self.support_state = "carried"
                self.target_visibility = "visible"
            elif verdict.verdict == "oppose":
                self.held_state = "not held"
                self.support_state = "supported"
        elif verdict.action == "place" and verdict.verdict == "support":
            self.held_state = "not held"
            self.support_state = "supported"
            self.target_visibility = "visible"
        if self._physical_state() != before:
            self.revision += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall task": self.overall_task,
            "revision": self.revision,
            "target": {
                "description": self.target_description,
                "visibility": self.target_visibility,
                "support state": self.support_state,
                "held state": self.held_state,
            },
            "last action": {
                "type": self.last_action,
                "effect verdict": self.last_effect_verdict,
                "observed effect": self.last_effect_observation,
            },
            "last observed step": self.last_observed_step,
        }

    def render_for_planner(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _gripper_aperture(observation: NeutralObservation) -> float | None:
    try:
        value = np.asarray(
            observation.raw["robot0_gripper_qpos"],
            dtype=np.float64,
        ).reshape(-1)
    except (KeyError, TypeError, ValueError):
        return None
    if value.size != 2 or not np.isfinite(value).all():
        return None
    return float(abs(value[0] - value[1]))


class LiberoPerceptionMemoryRuntime:
    """Orchestrate query generation, SAM3, memory, and effect verification."""

    profile = "perception_memory"

    def __init__(
        self,
        *,
        perception: LiberoSAM3PerceptionClient,
        verifier: LiberoDeterministicEffectVerifier | None = None,
    ) -> None:
        self.perception = perception
        self.verifier = verifier or LiberoDeterministicEffectVerifier()
        self.memory = LiberoSceneMemory()
        self._before_frame: PerceptionFrame | None = None
        self._before_observation: NeutralObservation | None = None
        self._queries = ()
        self._active_action = "other"
        self._verdict_counts = {"support": 0, "oppose": 0, "unverified": 0}
        self.turns = 0

    def reset_episode(
        self,
        *,
        task: str,
        initial_observation: NeutralObservation,
        initial_state: EpisodeState,
    ) -> None:
        del initial_observation, initial_state
        self.perception.reset()
        self.memory.reset(task)
        self._before_frame = None
        self._before_observation = None
        self._queries = ()
        self._active_action = "other"
        self._verdict_counts = {"support": 0, "oppose": 0, "unverified": 0}
        self.turns = 0

    def planner_context(self) -> Mapping[str, Any]:
        return self.memory.to_dict()

    def planner_memory_text(self, previous_memory_text: str) -> str:
        previous = " ".join(str(previous_memory_text).strip().split())
        structured = self.memory.render_for_planner()
        if previous:
            return previous + "\nStructured scene memory: " + structured
        return "Structured scene memory: " + structured

    def observe_before_action(
        self,
        *,
        task: str,
        subtask: str,
        memory_text: str,
        semantic_tags: Mapping[str, Any],
        observation: NeutralObservation,
        state: EpisodeState,
    ) -> Mapping[str, Any]:
        action = str(semantic_tags.get("subtask_type", "other") or "other").strip()
        if not action:
            action = "other"
        try:
            queries = self.perception.generate_queries(
                task=task,
                subtask=subtask,
                memory=memory_text,
                observation=observation,
            )
            frame = self.perception.observe(
                observation=observation,
                queries=queries,
                step=state.step_count,
                phase="before_action",
            )
        except Exception as exc:  # noqa: BLE001 - unknown keeps baseline available
            queries = ()
            frame = PerceptionFrame(
                step=state.step_count,
                phase="before_action",
                queries=(),
                detections=(),
                failures=(f"query:{type(exc).__name__}",),
                sam3_calls=0,
                elapsed_ms=0,
            )
        self.memory.observe_before_action(
            frame=frame,
            observation=observation,
            action=action,
        )
        self._before_frame = frame
        self._before_observation = observation
        self._queries = queries
        self._active_action = action
        self.turns += 1
        return {
            "profile": self.profile,
            "perception": frame.to_public_dict(),
            "scene_memory": self.memory.to_dict(),
        }

    def observe_after_action(
        self,
        *,
        observation: NeutralObservation,
        state: EpisodeState,
    ) -> Mapping[str, Any]:
        before_frame = self._before_frame
        before_observation = self._before_observation
        if before_frame is None or before_observation is None:
            raise RuntimeError("before-action perception is missing")
        try:
            after_frame = self.perception.observe(
                observation=observation,
                queries=self._queries,
                step=state.step_count,
                phase="after_action",
            )
        except Exception as exc:  # noqa: BLE001 - verifier must abstain
            after_frame = PerceptionFrame(
                step=state.step_count,
                phase="after_action",
                queries=tuple(self._queries),
                detections=(),
                failures=(f"sam3:{type(exc).__name__}",),
                sam3_calls=0,
                elapsed_ms=0,
            )
        verdict = self.verifier.verify(
            action=self._active_action,
            before_observation=before_observation,
            after_observation=observation,
            before_frame=before_frame,
            after_frame=after_frame,
        )
        self._verdict_counts[verdict.verdict] += 1
        self.memory.apply_effect(verdict, step=state.step_count)
        return {
            "profile": self.profile,
            "perception": after_frame.to_public_dict(),
            "effect": verdict.to_dict(),
            "scene_memory": self.memory.to_dict(),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "turns": self.turns,
            "query_calls": self.perception.query_calls,
            "sam3_calls": self.perception.sam3_calls,
            "verdicts": dict(self._verdict_counts),
            "final_scene_memory": self.memory.to_dict(),
        }


__all__ = ["LiberoPerceptionMemoryRuntime", "LiberoSceneMemory"]
