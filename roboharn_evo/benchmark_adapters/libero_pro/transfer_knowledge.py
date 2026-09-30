
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.family_router import FamilyRoutingConfig, KnowledgeFamilyRouter
from roboharn_evo.agent.hpk.family_store import (
    KnowledgeFamilyStoreError,
    load_knowledge_family_catalog,
)
from roboharn_evo.agent.hpk.hierarchical_knowledge import ActionKnowledgeV3
from roboharn_evo.agent.hpk.hierarchical_retriever import (
    ActionKnowledgeQuery,
    AgentApiHierarchicalRetrievalBackend,
    HierarchicalHPKRetrievalRuntime,
    VLMActionKnowledgeRetriever,
    VLMSubtaskKnowledgeRetriever,
    hierarchical_retrieval_output_json_schema,
)
from roboharn_evo.agent.hpk.hierarchical_store import (
    load_hierarchical_store,
    merge_action_units,
    save_hierarchical_store,
)
from roboharn_evo.agent.hpk.schemas import HPKValidationError, reject_private_transferable
from roboharn_evo.agent.hpk.semantic_consolidator import VLMKnowledgeConsolidator
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    HierarchicalReflectionBackend,
    _strict_response,
)


class LiberoTransferKnowledgeError(ValueError):
    """A source Store cannot support the requested transfer experiment."""


_ACTION_PROMPT_GROUNDING_INSTRUCTIONS = """You match reusable geometric Action Knowledge to the current robot subtask.

The knowledge may come from another benchmark, object category, robot, or controller. Select it only when the action type, object affordance, held/support state, geometric relations, and expected effect are compatible. Reject knowledge that is specific to an incompatible task topology or object geometry.

If compatible, write one concise target-domain action guidance statement. Preserve the current subtask and target exactly. Ground only relative geometry supported by the selected knowledge and current semantic state. Use selected_knowledge_index only to identify the input record; never copy an index into the guidance. Do not output coordinates, poses, action vectors, object IDs, candidate IDs, file paths, or a new task plan. Return strict JSON only."""


def action_prompt_grounding_json_schema() -> dict[str, Any]:
    return hierarchical_retrieval_output_json_schema("hpk_v3_action_prompt_grounding")


def _text(value: Any, *, label: str) -> str:
    text = " ".join(str(value).strip().split())
    if not text:
        raise LiberoTransferKnowledgeError(f"{label} must be non-empty")
    return text


class LiberoActionKnowledgePromptRuntime:
    """Ground semantic Action Knowledge into a native-policy prompt."""

    def __init__(
        self,
        *,
        action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
        backend: HierarchicalReflectionBackend,
        source_domain: str,
        family_router: KnowledgeFamilyRouter | None = None,
    ) -> None:
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must expose complete")
        self.action_knowledge = tuple(
            value if isinstance(value, ActionKnowledgeV3) else ActionKnowledgeV3(value)
            for value in action_knowledge
            if (
                value["status"]
                if isinstance(value, ActionKnowledgeV3)
                else value.get("status")
            )
            == "supported"
        )
        if not self.action_knowledge:
            raise LiberoTransferKnowledgeError(
                "Action Knowledge prompt runtime requires supported knowledge"
            )
        self._backend = backend
        self.source_domain = _text(source_domain, label="source_domain")
        if family_router is not None and not isinstance(
            family_router, KnowledgeFamilyRouter
        ):
            raise TypeError("family_router must be KnowledgeFamilyRouter or None")
        self._family_router = family_router
        self._decisions = 0
        self._applied_states: set[tuple[str, int]] = set()

    def reset_episode(self) -> None:
        self._decisions = 0
        self._applied_states.clear()

    def _adopted_decision(
        self,
        *,
        action: str,
        subtask: str,
        knowledge: ActionKnowledgeV3,
        guidance: str,
        reason: str,
        retrieval_audit: Mapping[str, Any] | None = None,
        selected_source_index: int | None = None,
    ) -> dict[str, Any]:
        expected = knowledge["expected_effect"]
        audit = {
            **dict(retrieval_audit or {}),
            "source_domain": self.source_domain,
            "action": action,
            "retrieval_reason": reason,
            "retrieved_knowledge": knowledge.to_dict(),
            "grounded_action_guidance": guidance,
            "grounding_reused": False,
        }
        if retrieval_audit is not None:
            audit.update(
                {
                    "selected_atomic_knowledge": selected_source_index,
                    "knowledge_selected": selected_source_index is not None,
                    "final_behavior_changed": True,
                }
            )
        return {
            "prompt_before": subtask,
            "prompt_after": (
                subtask
                + "\n\nReusable action guidance: "
                + guidance
                + "\nExpected effect: "
                + expected["physical_effect"]
                + "\nVerify: "
                + expected["verification_observation"]
            ),
            "knowledge_adopted": True,
            "audit": audit,
        }

    def ground_executor_prompt(
        self,
        *,
        task: str,
        subtask: str,
        memory: str,
        semantic_tags: Mapping[str, Any],
        semantic_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        task_text = _text(task, label="task")
        subtask_text = _text(subtask, label="subtask")
        if not isinstance(semantic_tags, Mapping) or not isinstance(
            semantic_state, Mapping
        ):
            raise TypeError("semantic tags and state must be objects")
        action = str(semantic_tags.get("subtask_type", "")).strip().casefold()
        scene_memory = semantic_state.get("scene memory")
        scene_revision: int | None = None
        held_state = "unknown"
        if isinstance(scene_memory, Mapping):
            raw_revision = scene_memory.get("revision")
            if (
                not isinstance(raw_revision, bool)
                and isinstance(raw_revision, int)
                and raw_revision >= 0
            ):
                scene_revision = raw_revision
            target_state = scene_memory.get("target")
            if isinstance(target_state, Mapping):
                held_state = (
                    str(target_state.get("held state", "unknown") or "unknown")
                    .strip()
                    .casefold()
                )
        self._decisions += 1
        if action == "grasp" and held_state == "held":
            return {
                "prompt_before": subtask_text,
                "prompt_after": subtask_text,
                "knowledge_adopted": False,
                "audit": {
                    "source_domain": self.source_domain,
                    "action": action,
                    "retrieval_reason": (
                        "grasp knowledge suppressed because fresh scene memory "
                        "already verifies the target is held"
                    ),
                    "knowledge_suppressed": True,
                    "suppression_reason": "target already held",
                    "retrieved_knowledge": None,
                    "grounded_action_guidance": None,
                },
            }
        route_audit: dict[str, Any] | None = None
        source_indices = tuple(
            index
            for index, value in enumerate(self.action_knowledge)
            if value["condition"]["action"] == action
        )
        candidates = tuple(self.action_knowledge[index] for index in source_indices)
        if self._family_router is not None and action in {"grasp", "place", "contact"}:
            target_state = (
                scene_memory.get("target")
                if isinstance(scene_memory, Mapping)
                else None
            )
            object_description = "the current target object"
            support_relation = None
            if isinstance(target_state, Mapping):
                raw_description = target_state.get("description")
                if isinstance(raw_description, str) and raw_description.strip():
                    object_description = " ".join(raw_description.strip().split())
                raw_support = target_state.get("support state")
                if isinstance(raw_support, str) and raw_support.strip():
                    support_relation = " ".join(raw_support.strip().split())
            route = self._family_router.route_action(
                self.action_knowledge,
                ActionKnowledgeQuery(
                    action=action,
                    object_description=object_description,
                    held_state=held_state,
                    support_relation=support_relation,
                ),
            )
            candidates = route.knowledge
            source_indices = route.source_indices
            route_audit = route.audit

        def merged_audit(value: Mapping[str, Any]) -> dict[str, Any]:
            return {**dict(route_audit or {}), **dict(value)}

        if not candidates:
            return {
                "prompt_before": subtask_text,
                "prompt_after": subtask_text,
                "knowledge_adopted": False,
                "audit": merged_audit(
                    {
                        "source_domain": self.source_domain,
                        "action": action or None,
                        "retrieval_reason": "no supported knowledge for the current action type",
                        "retrieved_knowledge": None,
                        "grounded_action_guidance": None,
                        "knowledge_selected": False,
                        "final_behavior_changed": False,
                    }
                ),
            }
        state_key = None if scene_revision is None else (action, scene_revision)
        if state_key is not None and state_key in self._applied_states:
            return {
                "prompt_before": subtask_text,
                "prompt_after": subtask_text,
                "knowledge_adopted": False,
                "audit": {
                    "source_domain": self.source_domain,
                    "action": action,
                    "retrieval_reason": (
                        "Action Knowledge was already applied once without a "
                        "verified physical state change"
                    ),
                    "knowledge_suppressed": True,
                    "suppression_reason": "once per scene state",
                    "retrieved_knowledge": None,
                    "grounded_action_guidance": None,
                },
            }
        payload = {
            "source_domain": self.source_domain,
            "current": {
                "task": task_text,
                "subtask": subtask_text,
                "memory": " ".join(str(memory).strip().split()),
                "semantic_tags": dict(semantic_tags),
                "semantic_state": dict(semantic_state),
            },
            "supported_action_knowledge": [
                {"source_index": index, "knowledge": value.to_dict()}
                for index, value in enumerate(candidates)
            ],
        }
        try:
            completion = self._backend.complete(
                instructions=_ACTION_PROMPT_GROUNDING_INSTRUCTIONS,
                input_text=json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
                images=(),
                output_schema=action_prompt_grounding_json_schema(),
                schema_name="hpk_v3_action_prompt_grounding",
            )
            result = _strict_response(completion.output)
            if set(result) != {
                "selected_knowledge_index",
                "grounded_action_guidance",
                "reason",
            }:
                raise LiberoTransferKnowledgeError(
                    "Action Knowledge response fields mismatch"
                )
            selected = result["selected_knowledge_index"]
            guidance = result["grounded_action_guidance"]
            reason = _text(result["reason"], label="retrieval reason")
            if selected is None:
                if guidance is not None:
                    raise LiberoTransferKnowledgeError(
                        "rejected Action Knowledge must not return guidance"
                    )
                return {
                    "prompt_before": subtask_text,
                    "prompt_after": subtask_text,
                    "knowledge_adopted": False,
                    "audit": merged_audit(
                        {
                            "source_domain": self.source_domain,
                            "action": action,
                            "retrieval_reason": reason,
                            "retrieved_knowledge": None,
                            "grounded_action_guidance": None,
                            "knowledge_selected": False,
                            "final_behavior_changed": False,
                        }
                    ),
                }
            if (
                isinstance(selected, bool)
                or not isinstance(selected, int)
                or not 0 <= selected < len(candidates)
            ):
                raise LiberoTransferKnowledgeError(
                    "selected Action Knowledge index is invalid"
                )
            guidance_text = _text(guidance, label="grounded_action_guidance")
            reject_private_transferable(
                guidance_text,
                path="grounded_action_guidance",
            )
            knowledge = candidates[selected]
            if state_key is not None:
                self._applied_states.add(state_key)
            return self._adopted_decision(
                action=action,
                subtask=subtask_text,
                knowledge=knowledge,
                guidance=guidance_text,
                reason=reason,
                retrieval_audit=route_audit,
                selected_source_index=source_indices[selected],
            )
        except Exception as exc:  # noqa: BLE001 - retrieval failure retains baseline
            return {
                "prompt_before": subtask_text,
                "prompt_after": subtask_text,
                "knowledge_adopted": False,
                "audit": merged_audit(
                    {
                        "source_domain": self.source_domain,
                        "action": action,
                        "retrieval_reason": "action retrieval failed; baseline prompt retained",
                        "retrieval_error": type(exc).__name__,
                        "retrieved_knowledge": None,
                        "grounded_action_guidance": None,
                        "knowledge_selected": False,
                        "final_behavior_changed": False,
                    }
                ),
            }


@dataclass(frozen=True, slots=True)
class LiberoTaskActionRuntimes:
    task: HierarchicalHPKRetrievalRuntime
    action: LiberoActionKnowledgePromptRuntime


def _family_router(
    *,
    store_root: str | Path,
    task_knowledge: Sequence[Mapping[str, Any]],
    action_knowledge: Sequence[Mapping[str, Any]],
    backend: HierarchicalReflectionBackend,
    exhaustive_threshold: int,
) -> KnowledgeFamilyRouter:
    catalog = None
    catalog_error = None
    try:
        catalog = load_knowledge_family_catalog(
            store_root,
            task_knowledge=task_knowledge,
            action_knowledge=action_knowledge,
        )
    except KnowledgeFamilyStoreError as exc:
        catalog_error = type(exc).__name__
    return KnowledgeFamilyRouter(
        backend,
        catalog=catalog,
        config=FamilyRoutingConfig(exhaustive_threshold=exhaustive_threshold),
        catalog_error=catalog_error,
    )


def build_transfer_task_only_runtime(
    *,
    store_root: str | Path,
    planner_url: str,
    timeout_sec: int = 600,
    family_exhaustive_threshold: int = 24,
) -> HierarchicalHPKRetrievalRuntime:
    """Load only the Task layer from one shared read-only transfer Store."""

    tasks, actions = load_hierarchical_store(store_root)
    if not tasks:
        raise LiberoTransferKnowledgeError("Task-only runtime requires Task Knowledge")
    backend = AgentApiHierarchicalRetrievalBackend(
        planner_url,
        timeout_sec=timeout_sec,
    )
    return HierarchicalHPKRetrievalRuntime(
        mode="full",
        task_knowledge=tasks,
        action_knowledge=(),
        subtask_retriever=VLMSubtaskKnowledgeRetriever(backend),
        action_retriever=VLMActionKnowledgeRetriever(backend),
        family_router=_family_router(
            store_root=store_root,
            task_knowledge=tasks,
            action_knowledge=actions,
            backend=backend,
            exhaustive_threshold=family_exhaustive_threshold,
        ),
    )


def build_action_only_runtime(
    *,
    store_root: str | Path,
    planner_url: str,
    source_domain: str,
    timeout_sec: int = 600,
    family_exhaustive_threshold: int = 24,
) -> LiberoActionKnowledgePromptRuntime:
    """Load only Action Knowledge; the planner receives no Task Knowledge."""

    tasks, actions = load_hierarchical_store(store_root)
    if not actions:
        raise LiberoTransferKnowledgeError(
            "Action-only runtime requires Action Knowledge"
        )
    backend = AgentApiHierarchicalRetrievalBackend(
        planner_url,
        timeout_sec=timeout_sec,
    )
    return LiberoActionKnowledgePromptRuntime(
        action_knowledge=actions,
        backend=backend,
        family_router=_family_router(
            store_root=store_root,
            task_knowledge=tasks,
            action_knowledge=actions,
            backend=backend,
            exhaustive_threshold=family_exhaustive_threshold,
        ),
        source_domain=source_domain,
    )


def build_task_action_runtimes(
    *,
    store_root: str | Path,
    planner_url: str,
    source_domain: str,
    timeout_sec: int = 600,
    family_exhaustive_threshold: int = 24,
) -> LiberoTaskActionRuntimes:
    tasks, actions = load_hierarchical_store(store_root)
    if not tasks or not actions:
        raise LiberoTransferKnowledgeError(
            "Task+Action runtime requires both knowledge layers"
        )
    task_backend = AgentApiHierarchicalRetrievalBackend(
        planner_url,
        timeout_sec=timeout_sec,
    )
    action_backend = AgentApiHierarchicalRetrievalBackend(
        planner_url,
        timeout_sec=timeout_sec,
    )
    task_family_router = _family_router(
        store_root=store_root,
        task_knowledge=tasks,
        action_knowledge=actions,
        backend=task_backend,
        exhaustive_threshold=family_exhaustive_threshold,
    )
    action_family_router = _family_router(
        store_root=store_root,
        task_knowledge=tasks,
        action_knowledge=actions,
        backend=action_backend,
        exhaustive_threshold=family_exhaustive_threshold,
    )
    task_runtime = HierarchicalHPKRetrievalRuntime(
        mode="full",
        task_knowledge=tasks,
        action_knowledge=(),
        subtask_retriever=VLMSubtaskKnowledgeRetriever(task_backend),
        action_retriever=VLMActionKnowledgeRetriever(task_backend),
        family_router=task_family_router,
    )
    return LiberoTaskActionRuntimes(
        task=task_runtime,
        action=LiberoActionKnowledgePromptRuntime(
            action_knowledge=actions,
            backend=action_backend,
            source_domain=source_domain,
            family_router=action_family_router,
        ),
    )


@dataclass(frozen=True, slots=True)
class LiberoTransferStoreBundle:
    root: Path
    rmbench_task_store: Path
    rmbench_task_action_store: Path
    libero_native_store: Path
    rmbench_task_count: int
    rmbench_action_count: int
    libero_task_count: int
    libero_action_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "stores": {
                "rmbench_task": str(self.rmbench_task_store),
                "rmbench_task_action": str(self.rmbench_task_action_store),
                "libero_native": str(self.libero_native_store),
            },
            "counts": {
                "rmbench_task": self.rmbench_task_count,
                "rmbench_action": self.rmbench_action_count,
                "libero_task": self.libero_task_count,
                "libero_action": self.libero_action_count,
            },
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiberoTransferKnowledgeError(f"cannot read {path}") from exc
    if not isinstance(value, dict):
        raise LiberoTransferKnowledgeError(f"{path.name} must contain one object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_transferable(values: tuple[Any, ...], *, label: str) -> None:
    for index, value in enumerate(values):
        try:
            reject_private_transferable(value.to_dict(), path=f"{label}[{index}]")
        except HPKValidationError as exc:
            raise LiberoTransferKnowledgeError(
                f"{label}[{index}] contains runtime-private data"
            ) from exc


def _reload_exact(
    root: Path,
    *,
    expected_tasks: tuple[Any, ...],
    expected_actions: tuple[Any, ...],
) -> None:
    tasks, actions = load_hierarchical_store(root)
    if tuple(value.to_dict() for value in tasks) != tuple(
        value.to_dict() for value in expected_tasks
    ):
        raise LiberoTransferKnowledgeError("published Task Knowledge changed on reload")
    if tuple(value.to_dict() for value in actions) != tuple(
        value.to_dict() for value in expected_actions
    ):
        raise LiberoTransferKnowledgeError(
            "published Action Knowledge changed on reload"
        )


def build_augmented_action_store(
    *,
    base_store_root: str | Path,
    atomic_knowledge_paths: Sequence[str | Path],
    action: str,
    output_root: str | Path,
    backend: HierarchicalReflectionBackend,
    source_label: str,
) -> Path:
    """Add VLM-consolidated cross-domain actions without changing the source.

    The only deterministic selection is the caller-declared action vocabulary
    value.  Object affordance and geometric equivalence remain model decisions.
    """

    action_name = _text(action, label="action").casefold()
    if action_name not in {"contact", "grasp", "place"}:
        raise LiberoTransferKnowledgeError("action must be contact, grasp, or place")
    source = Path(base_store_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise LiberoTransferKnowledgeError(
            "augmented Store output already exists; refusing to overwrite"
        )
    if not atomic_knowledge_paths:
        raise LiberoTransferKnowledgeError(
            "at least one atomic knowledge input is required"
        )
    base_tasks, base_actions = load_hierarchical_store(source)
    selected: list[ActionKnowledgeV3] = []
    resolved_inputs: list[Path] = []
    for raw_path in atomic_knowledge_paths:
        path = Path(raw_path).expanduser().resolve(strict=True)
        payload = _load_json(path)
        raw_actions = payload.get("action_knowledge")
        if not isinstance(raw_actions, list):
            raise LiberoTransferKnowledgeError(
                f"{path.name} lacks an action_knowledge array"
            )
        resolved_inputs.append(path)
        selected.extend(
            typed
            for value in raw_actions
            if (typed := ActionKnowledgeV3(value))["condition"]["action"] == action_name
        )
    if not selected:
        raise LiberoTransferKnowledgeError(
            f"atomic inputs contain no {action_name} Action Knowledge"
        )
    _validate_transferable(tuple(selected), label="atomic_action")
    consolidated = VLMKnowledgeConsolidator(backend).consolidate(
        task_knowledge=(),
        action_knowledge=selected,
    )
    if consolidated.task_knowledge:
        raise LiberoTransferKnowledgeError(
            "action-only consolidation unexpectedly returned Task Knowledge"
        )
    if any(
        value["condition"]["action"] != action_name
        for value in consolidated.action_knowledge
    ):
        raise LiberoTransferKnowledgeError(
            "action consolidation changed the declared action type"
        )
    augmented_actions = (*base_actions, *consolidated.action_knowledge)
    _validate_transferable(base_tasks, label="base_task")
    _validate_transferable(tuple(augmented_actions), label="augmented_action")
    output.mkdir(parents=True)
    save_hierarchical_store(
        output,
        task_knowledge=base_tasks,
        action_knowledge=augmented_actions,
    )
    _reload_exact(
        output,
        expected_tasks=base_tasks,
        expected_actions=tuple(augmented_actions),
    )
    raw_response = consolidated.raw_response
    if isinstance(raw_response, Mapping):
        _write_json(
            output / "semantic_consolidation_response.json",
            dict(raw_response),
        )
    else:
        (output / "semantic_consolidation_response.json").write_text(
            str(raw_response).rstrip() + "\n",
            encoding="utf-8",
        )
    _write_json(
        output / "source_provenance.json",
        {
            "schema": "roboharn_evo/libero_cross_domain_action_source/v1",
            "source_label": _text(source_label, label="source_label"),
            "base_store": str(source),
            "atomic_inputs": [str(path) for path in resolved_inputs],
            "selected_action": action_name,
            "atomic_units": len(selected),
            "consolidated_units": len(consolidated.action_knowledge),
            "source_records_changed": False,
        },
    )
    return output


def build_transfer_store_bundle(
    *,
    rmbench_store_root: str | Path,
    libero_expert_root: str | Path,
    output_root: str | Path,
) -> LiberoTransferStoreBundle:
    """Build three immutable experiment inputs without changing either source."""

    rmbench_source = Path(rmbench_store_root).expanduser().resolve(strict=True)
    libero_source = Path(libero_expert_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise LiberoTransferKnowledgeError(
            "transfer Store bundle output already exists; refusing to overwrite"
        )

    rmbench_tasks, rmbench_actions = load_hierarchical_store(rmbench_source)
    if not rmbench_tasks or not rmbench_actions:
        raise LiberoTransferKnowledgeError(
            "RMBench source must contain Task and Action Knowledge"
        )
    if not any(
        value["status"] == "supported" and value["condition"]["action"] == "grasp"
        for value in rmbench_actions
    ):
        raise LiberoTransferKnowledgeError(
            "RMBench source contains no supported grasp knowledge"
        )
    _validate_transferable(rmbench_tasks, label="rmbench_task")
    _validate_transferable(rmbench_actions, label="rmbench_action")

    libero_tasks, existing_libero_actions = load_hierarchical_store(
        libero_source / "store"
    )
    if not libero_tasks:
        raise LiberoTransferKnowledgeError(
            "LIBERO expert source contains no Task Knowledge"
        )
    if existing_libero_actions:
        raise LiberoTransferKnowledgeError(
            "LIBERO expert source must remain the frozen Task-only Store"
        )
    atomic = _load_json(libero_source / "atomic_knowledge.json")
    consolidation = _load_json(libero_source / "semantic_consolidation_response.json")
    raw_actions = atomic.get("action_knowledge")
    action_groups = consolidation.get("action_groups")
    if not isinstance(raw_actions, list) or not isinstance(action_groups, list):
        raise LiberoTransferKnowledgeError(
            "LIBERO source lacks atomic actions or semantic action groups"
        )
    libero_actions = merge_action_units(
        tuple(ActionKnowledgeV3(value) for value in raw_actions),
        groups=action_groups,
    )
    if not libero_actions:
        raise LiberoTransferKnowledgeError(
            "LIBERO source produced no consolidated Action Knowledge"
        )
    _validate_transferable(libero_tasks, label="libero_task")
    _validate_transferable(libero_actions, label="libero_action")

    output.mkdir(parents=True)
    rmbench_task_store = save_hierarchical_store(
        output / "rmbench_task",
        task_knowledge=rmbench_tasks,
        action_knowledge=(),
    )
    rmbench_task_action_store = save_hierarchical_store(
        output / "rmbench_task_action",
        task_knowledge=rmbench_tasks,
        action_knowledge=rmbench_actions,
    )
    libero_native_store = save_hierarchical_store(
        output / "libero_native",
        task_knowledge=libero_tasks,
        action_knowledge=libero_actions,
    )
    _reload_exact(
        rmbench_task_store,
        expected_tasks=rmbench_tasks,
        expected_actions=(),
    )
    _reload_exact(
        rmbench_task_action_store,
        expected_tasks=rmbench_tasks,
        expected_actions=rmbench_actions,
    )
    _reload_exact(
        libero_native_store,
        expected_tasks=libero_tasks,
        expected_actions=libero_actions,
    )
    _write_json(
        output / "source_provenance.json",
        {
            "schema": "roboharn_evo/libero_cross_benchmark_store_bundle/v1",
            "sources": {
                "rmbench": {
                    "benchmark": "RMBench",
                    "kind": "three ground-truth manipulation trajectories",
                    "store": str(rmbench_source),
                },
                "libero_native": {
                    "benchmark": "LIBERO-PRO",
                    "kind": "one real expert trajectory",
                    "reflection_root": str(libero_source),
                },
            },
            "knowledge_constraints": {
                "absolute_poses": False,
                "action_vectors": False,
                "candidate_ids": False,
                "object_instance_ids": False,
                "source_records_changed": False,
            },
        },
    )
    result = LiberoTransferStoreBundle(
        root=output,
        rmbench_task_store=rmbench_task_store,
        rmbench_task_action_store=rmbench_task_action_store,
        libero_native_store=libero_native_store,
        rmbench_task_count=len(rmbench_tasks),
        rmbench_action_count=len(rmbench_actions),
        libero_task_count=len(libero_tasks),
        libero_action_count=len(libero_actions),
    )
    _write_json(output / "build_result.json", result.to_dict())
    return result


__all__ = [
    "LiberoActionKnowledgePromptRuntime",
    "LiberoTaskActionRuntimes",
    "LiberoTransferKnowledgeError",
    "LiberoTransferStoreBundle",
    "action_prompt_grounding_json_schema",
    "build_action_only_runtime",
    "build_augmented_action_store",
    "build_task_action_runtimes",
    "build_transfer_store_bundle",
    "build_transfer_task_only_runtime",
]
