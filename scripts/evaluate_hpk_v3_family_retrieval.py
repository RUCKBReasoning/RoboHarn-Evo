from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from roboharn_evo.agent.hpk.family_router import FamilyRoutingConfig, KnowledgeFamilyRouter
from roboharn_evo.agent.hpk.hierarchical_knowledge import ActionKnowledgeV3, SubtaskKnowledgeV3
from roboharn_evo.agent.hpk.hierarchical_retriever import (
    ActionKnowledgeQuery,
    AgentApiHierarchicalRetrievalBackend,
    SubtaskKnowledgeQuery,
    VLMActionKnowledgeRetriever,
    VLMSubtaskKnowledgeRetriever,
)
from roboharn_evo.agent.hpk.knowledge_family import (
    KnowledgeFamilyCatalogV1,
    validate_catalog_against_knowledge,
)
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import HierarchicalBackendCompletion

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "eval_result/hpk/v3/family_retrieval/metrics.json"


class SavedResponseBackend:
    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self._responses = [copy.deepcopy(dict(value)) for value in responses]
        self.outputs: list[dict[str, Any]] = []

    def complete(self, **_kwargs: Any) -> HierarchicalBackendCompletion:
        if not self._responses:
            raise RuntimeError("saved retrieval responses are exhausted")
        output = self._responses.pop(0)
        self.outputs.append(copy.deepcopy(output))
        return HierarchicalBackendCompletion(output=output)

    def assert_consumed(self) -> None:
        if self._responses:
            raise RuntimeError("saved retrieval responses contain unused values")


class RecordingBackend:
    def __init__(self, backend: Any, *, checkpoint_path: Path | None = None) -> None:
        self._backend = backend
        self._checkpoint_path = checkpoint_path
        self.outputs: list[dict[str, Any] | str] = []

    def complete(self, **kwargs: Any) -> HierarchicalBackendCompletion:
        completion = self._backend.complete(**kwargs)
        output = completion.output
        self.outputs.append(
            copy.deepcopy(dict(output)) if isinstance(output, Mapping) else output
        )
        if self._checkpoint_path is not None:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_path.write_text(
                json.dumps(self.outputs, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return completion


def _load_corpus(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "task_knowledge",
        "action_knowledge",
        "knowledge_families",
        "cases",
    }:
        raise ValueError(f"invalid retrieval corpus fields in {path}")
    tasks = tuple(SubtaskKnowledgeV3(value) for value in payload["task_knowledge"])
    actions = tuple(ActionKnowledgeV3(value) for value in payload["action_knowledge"])
    catalog = validate_catalog_against_knowledge(
        payload["knowledge_families"],
        task_knowledge=tasks,
        action_knowledge=actions,
    )
    cases = payload["cases"]
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence) or not cases:
        raise ValueError("retrieval corpus cases must be a non-empty array")
    return {
        "task_knowledge": tasks,
        "action_knowledge": actions,
        "knowledge_families": catalog,
        "cases": [copy.deepcopy(dict(value)) for value in cases],
    }


def shuffled_family_membership(
    catalog: KnowledgeFamilyCatalogV1,
) -> KnowledgeFamilyCatalogV1:
    """Rotate memberships only within knowledge type and exact Action partitions."""

    payload = catalog.to_dict()

    def rotate(families: list[dict[str, Any]], indices: list[int]) -> None:
        if len(indices) < 2:
            return
        memberships = [families[index]["member_indices"] for index in indices]
        rotated = memberships[1:] + memberships[:1]
        for index, members in zip(indices, rotated, strict=True):
            families[index]["member_indices"] = members

    rotate(payload["task_families"], list(range(len(payload["task_families"]))))
    for action in ("grasp", "place", "contact"):
        rotate(
            payload["action_families"],
            [
                index
                for index, family in enumerate(payload["action_families"])
                if family["action"] == action
            ],
        )
    return KnowledgeFamilyCatalogV1(payload)


def _selected_source_index(
    selected: Any,
    values: Sequence[Any],
    source_indices: Sequence[int],
) -> int | None:
    if selected is None:
        return None
    selected_payload = selected.knowledge.to_dict()
    for source_index, value in zip(source_indices, values, strict=True):
        if value.to_dict() == selected_payload:
            return int(source_index)
    return None


def _combine_usage(*audits: Mapping[str, Any] | None) -> dict[str, Any]:
    calls = 0
    characters = 0
    latency = 0.0
    tokens: int | None = None
    for audit in audits:
        if audit is None:
            continue
        calls += int(audit.get("retrieval_calls", 0) or 0)
        characters += int(audit.get("retrieval_input_characters", 0) or 0)
        latency += float(audit.get("retrieval_latency_ms", 0.0) or 0.0)
        value = audit.get("retrieval_input_tokens")
        if isinstance(value, int) and not isinstance(value, bool):
            tokens = value if tokens is None else tokens + value
    return {
        "retrieval_calls": calls,
        "retrieval_input_characters": characters,
        "retrieval_input_tokens": tokens,
        "retrieval_latency_ms": round(latency, 3),
    }


def _case_query(case: Mapping[str, Any]) -> Any:
    kind = case.get("kind")
    query = case.get("query")
    if not isinstance(query, Mapping):
        raise TypeError("case query must be an object")
    if kind == "task":
        return SubtaskKnowledgeQuery(
            overall_goal=query["overall_goal"],
            task_state=query["task_state"],
            relevant_relations=tuple(query.get("relevant_relations", [])),
        )
    if kind == "action":
        return ActionKnowledgeQuery(
            action=query["action"],
            object_description=query["object_description"],
            held_state=query.get("held_state"),
            support_relation=query.get("support_relation"),
            target_relation=query.get("target_relation"),
            intended_effect=query.get("intended_effect"),
            candidate_geometry=tuple(query.get("candidate_geometry", [])),
        )
    raise ValueError("case kind must be task or action")


def _expected(case: Mapping[str, Any], field: str) -> Any:
    value = case.get(field)
    if (
        field == "expected_atomic_index"
        and value is not None
        and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
    ):
        raise ValueError("expected_atomic_index must be null or non-negative")
    return value


def _evaluate_exhaustive(
    *,
    case: Mapping[str, Any],
    tasks: Sequence[SubtaskKnowledgeV3],
    actions: Sequence[ActionKnowledgeV3],
    backend: Any,
) -> dict[str, Any]:
    query = _case_query(case)
    retrieval_error = None
    if case["kind"] == "task":
        retriever = VLMSubtaskKnowledgeRetriever(backend)
        try:
            selected = retriever.retrieve(
                tasks,
                query,
                baseline_subtask=case["baseline_subtask"],
            )
        except Exception as exc:  # noqa: BLE001 - a failed retrieval is a metric row
            selected = None
            retrieval_error = type(exc).__name__
        supported = tuple(value for value in tasks if value["status"] == "supported")
        source_indices = tuple(
            index for index, value in enumerate(tasks) if value["status"] == "supported"
        )
        selected_index = _selected_source_index(selected, supported, source_indices)
        behavior_after = (
            case["baseline_subtask"] if selected is None else selected.grounded_subtask
        )
        candidate_index = None
    else:
        retriever = VLMActionKnowledgeRetriever(backend)
        try:
            selected = retriever.retrieve(actions, query)
        except Exception as exc:  # noqa: BLE001 - a failed retrieval is a metric row
            selected = None
            retrieval_error = type(exc).__name__
        eligible = tuple(
            value
            for value in actions
            if value["status"] == "supported"
            and value["condition"]["action"] == query.action
        )
        source_indices = tuple(
            index
            for index, value in enumerate(actions)
            if value["status"] == "supported"
            and value["condition"]["action"] == query.action
        )
        selected_index = _selected_source_index(selected, eligible, source_indices)
        candidate_index = (
            None
            if selected is None
            else next(
                index
                for index, value in enumerate(query.candidate_geometry)
                if dict(value) == selected.selected_candidate_geometry
            )
        )
        behavior_after = 0 if candidate_index is None else candidate_index
        rank_before = [
            {"rank": index, "geometry": copy.deepcopy(dict(value))}
            for index, value in enumerate(query.candidate_geometry)
        ]
        ranked_indices = (
            list(range(len(query.candidate_geometry)))
            if candidate_index is None
            else [candidate_index]
            + [
                index
                for index in range(len(query.candidate_geometry))
                if index != candidate_index
            ]
        )
        rank_after = [
            {
                "rank": rank,
                "geometry": copy.deepcopy(dict(query.candidate_geometry[index])),
            }
            for rank, index in enumerate(ranked_indices)
        ]
    usage = _combine_usage(retriever.last_call_audit)
    return {
        "selected_atomic_index": selected_index,
        "selected_atomic_knowledge": (
            None if selected is None else selected.knowledge.to_dict()
        ),
        "selected_candidate_geometry_index": candidate_index,
        "behavior_before": case.get("baseline_subtask", 0),
        "behavior_after": behavior_after,
        "behavior_changed": behavior_after != case.get("baseline_subtask", 0),
        "retrieval_error": retrieval_error,
        **(
            {"rank_before": rank_before, "rank_after": rank_after}
            if case["kind"] == "action"
            else {}
        ),
        **usage,
    }


def _evaluate_family(
    *,
    mode: str,
    case: Mapping[str, Any],
    tasks: Sequence[SubtaskKnowledgeV3],
    actions: Sequence[ActionKnowledgeV3],
    catalog: KnowledgeFamilyCatalogV1,
    backend: Any,
    config: FamilyRoutingConfig,
) -> dict[str, Any]:
    query = _case_query(case)
    router = KnowledgeFamilyRouter(backend, catalog=catalog, config=config)
    summary_direct_guidance = None
    retrieval_error = None
    if case["kind"] == "task":
        route = router.route_task(
            tasks,
            query,
            baseline_subtask=case["baseline_subtask"],
        )
        if mode == "family-summary-only":
            selected = None
            final_audit = None
            selected_families = route.audit["selected_family_indices"]
            if selected_families:
                summary_direct_guidance = catalog.task_families[selected_families[0]][
                    "strategy_summary"
                ]
            behavior_after = summary_direct_guidance or case["baseline_subtask"]
        else:
            retriever = VLMSubtaskKnowledgeRetriever(backend)
            try:
                selected = retriever.retrieve(
                    route.knowledge,
                    query,
                    baseline_subtask=case["baseline_subtask"],
                )
            except Exception as exc:  # noqa: BLE001 - score the failed retrieval
                selected = None
                retrieval_error = type(exc).__name__
            final_audit = retriever.last_call_audit
            behavior_after = (
                case["baseline_subtask"]
                if selected is None
                else selected.grounded_subtask
            )
        candidate_index = None
    else:
        route = router.route_action(actions, query)
        if mode == "family-summary-only":
            selected = None
            final_audit = None
            candidate_index = None
            selected_families = route.audit["selected_family_indices"]
            if selected_families:
                summary_direct_guidance = catalog.action_families[selected_families[0]][
                    "strategy_summary"
                ]
        else:
            retriever = VLMActionKnowledgeRetriever(backend)
            try:
                selected = retriever.retrieve(route.knowledge, query)
            except Exception as exc:  # noqa: BLE001 - score the failed retrieval
                selected = None
                retrieval_error = type(exc).__name__
            final_audit = retriever.last_call_audit
            candidate_index = (
                None
                if selected is None
                else next(
                    index
                    for index, value in enumerate(query.candidate_geometry)
                    if dict(value) == selected.selected_candidate_geometry
                )
            )
        behavior_after = 0 if candidate_index is None else candidate_index
    selected_index = _selected_source_index(
        selected,
        route.knowledge,
        route.source_indices,
    )
    usage = _combine_usage(route.audit, final_audit)
    return {
        "selected_atomic_index": selected_index,
        "selected_atomic_knowledge": (
            None if selected is None else selected.knowledge.to_dict()
        ),
        "selected_candidate_geometry_index": candidate_index,
        "behavior_before": case.get("baseline_subtask", 0),
        "behavior_after": behavior_after,
        "behavior_changed": behavior_after != case.get("baseline_subtask", 0),
        "retrieval_error": retrieval_error,
        "family_path": route.audit["exhaustive_or_family_path"],
        "selected_family_indices": route.audit["selected_family_indices"],
        "selected_family_summaries": route.audit["selected_family_summaries"],
        "atomic_shortlist_indices": list(route.source_indices),
        "eligible_atomic_count": route.audit["eligible_atomic_count"],
        "atomic_count_after_shortlist": route.audit["atomic_count_after_shortlist"],
        "member_shortlist_performed": route.audit["member_shortlist_performed"],
        "family_summary_direct_guidance": summary_direct_guidance,
        **(
            {
                "rank_before": [
                    {"rank": index, "geometry": copy.deepcopy(dict(value))}
                    for index, value in enumerate(query.candidate_geometry)
                ],
                "rank_after": [
                    {
                        "rank": rank,
                        "geometry": copy.deepcopy(
                            dict(query.candidate_geometry[index])
                        ),
                    }
                    for rank, index in enumerate(
                        list(range(len(query.candidate_geometry)))
                        if candidate_index is None
                        else [candidate_index]
                        + [
                            index
                            for index in range(len(query.candidate_geometry))
                            if index != candidate_index
                        ]
                    )
                ],
            }
            if case["kind"] == "action"
            else {}
        ),
        **usage,
    }


def _score_row(case: Mapping[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    expected_atomic = _expected(case, "expected_atomic_index")
    expected_candidate = case.get("expected_candidate_geometry_index")
    expected_subtask = case.get("expected_grounded_subtask")
    selected_atomic = row["selected_atomic_index"]
    correct = selected_atomic == expected_atomic
    expected_null = expected_atomic is None
    selected_null = selected_atomic is None
    if case["kind"] == "task":
        grounded_subtask_exact_match = (
            None
            if expected_subtask is None
            else row["behavior_after"] == expected_subtask
        )
        # The held-in atomic label defines which reusable subtask is correct.
        # A grounded rewrite may paraphrase that subtask, so exact wording is a
        # separate diagnostic rather than the semantic accuracy authority.
        behavior_correct = correct
    else:
        grounded_subtask_exact_match = None
        behavior_correct = (
            row["selected_candidate_geometry_index"] == expected_candidate
        )
    should_change = bool(case.get("behavior_change_expected", False))
    helpful = bool(
        should_change and correct and behavior_correct and row["behavior_changed"]
    )
    harmful = bool(row["behavior_changed"] and not (correct and behavior_correct))
    expected_families = set(case.get("expected_family_indices", []))
    selected_families = set(row.get("selected_family_indices", []))
    shortlist = set(row.get("atomic_shortlist_indices", []))
    family_path = row.get("family_path")
    row.update(
        {
            "final_selection_correct": correct,
            "expected_null": expected_null,
            "selected_null": selected_null,
            "behavior_correct": behavior_correct,
            "grounded_subtask_exact_match": grounded_subtask_exact_match,
            "task_subtask_correct": (
                behavior_correct if case["kind"] == "task" else None
            ),
            "action_candidate_compliant": (
                behavior_correct if case["kind"] == "action" else None
            ),
            "helpful_behavior_change": helpful,
            "harmful_behavior_change": harmful,
            "family_recall": (
                None
                if not expected_families
                or "selected_family_indices" not in row
                or family_path != "family"
                else bool(expected_families.intersection(selected_families))
            ),
            "atomic_candidate_recall": (
                None
                if expected_atomic is None
                or "atomic_shortlist_indices" not in row
                or family_path != "family"
                else expected_atomic in shortlist
            ),
            "atomic_recall_at_8": (
                None
                if expected_atomic is None
                or "atomic_shortlist_indices" not in row
                or family_path != "family"
                or len(shortlist) > 8
                else expected_atomic in shortlist
            ),
        }
    )
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def rate(field: str) -> float | None:
        values = [value[field] for value in rows if value.get(field) is not None]
        return None if not values else mean(bool(value) for value in values)

    expected_null_rows = [row for row in rows if row["expected_null"]]
    selected_null_rows = [row for row in rows if row["selected_null"]]
    true_nulls = sum(row["expected_null"] for row in selected_null_rows)
    return {
        "case_count": len(rows),
        "family_recall_case_count": sum(
            row.get("family_recall") is not None for row in rows
        ),
        "family_recall_at_2": rate("family_recall"),
        "atomic_candidate_recall_case_count": sum(
            row.get("atomic_candidate_recall") is not None for row in rows
        ),
        "atomic_candidate_recall": rate("atomic_candidate_recall"),
        "atomic_recall_at_8_case_count": sum(
            row.get("atomic_recall_at_8") is not None for row in rows
        ),
        "atomic_recall_at_8": rate("atomic_recall_at_8"),
        "final_selection_accuracy": rate("final_selection_correct"),
        "null_recall": (
            None
            if not expected_null_rows
            else mean(row["selected_null"] for row in expected_null_rows)
        ),
        "null_precision": (
            None if not selected_null_rows else true_nulls / len(selected_null_rows)
        ),
        "behavior_accuracy": rate("behavior_correct"),
        "task_subtask_accuracy": rate("task_subtask_correct"),
        "action_candidate_compliance": rate("action_candidate_compliant"),
        "helpful_behavior_change_count": sum(
            row["helpful_behavior_change"] for row in rows
        ),
        "harmful_behavior_change_count": sum(
            row["harmful_behavior_change"] for row in rows
        ),
        "retrieval_calls": sum(row["retrieval_calls"] for row in rows),
        "retrieval_input_characters": sum(
            row["retrieval_input_characters"] for row in rows
        ),
        "retrieval_input_tokens": (
            None
            if any(row["retrieval_input_tokens"] is None for row in rows)
            else sum(row["retrieval_input_tokens"] for row in rows)
        ),
        "retrieval_latency_ms": round(
            sum(row["retrieval_latency_ms"] for row in rows), 3
        ),
    }


def evaluate_corpora(
    corpora: Mapping[int, Path],
    *,
    backend: Any,
    config: FamilyRoutingConfig | None = None,
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    active_config = config or FamilyRoutingConfig()
    allowed_methods = (
        "exhaustive",
        "family-routed",
        "shuffled-family",
        "family-summary-only",
    )
    active_methods = tuple(methods or allowed_methods)
    if (
        not active_methods
        or len(active_methods) != len(set(active_methods))
        or any(method not in allowed_methods for method in active_methods)
    ):
        raise ValueError("evaluation methods must be unique supported method names")
    scales: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for requested_size, path in sorted(corpora.items()):
        corpus = _load_corpus(path)
        tasks = corpus["task_knowledge"]
        actions = corpus["action_knowledge"]
        actual_size = len(tasks) + len(actions)
        if actual_size != requested_size:
            raise ValueError(
                f"corpus {path} contains {actual_size} units, expected {requested_size}"
            )
        catalog = corpus["knowledge_families"]
        shuffled = shuffled_family_membership(catalog)
        validate_catalog_against_knowledge(
            shuffled,
            task_knowledge=tasks,
            action_knowledge=actions,
        )
        rows = []
        for case in corpus["cases"]:
            name = str(case.get("case_name", "") or "").strip()
            if not name:
                raise ValueError("every retrieval case needs case_name")
            method_rows: dict[str, dict[str, Any]] = {}
            if "exhaustive" in active_methods:
                method_rows["exhaustive"] = _evaluate_exhaustive(
                    case=case,
                    tasks=tasks,
                    actions=actions,
                    backend=backend,
                )
            if "family-routed" in active_methods:
                method_rows["family-routed"] = _evaluate_family(
                    mode="family-routed",
                    case=case,
                    tasks=tasks,
                    actions=actions,
                    catalog=catalog,
                    backend=backend,
                    config=active_config,
                )
            if "shuffled-family" in active_methods:
                method_rows["shuffled-family"] = _evaluate_family(
                    mode="shuffled-family",
                    case=case,
                    tasks=tasks,
                    actions=actions,
                    catalog=shuffled,
                    backend=backend,
                    config=active_config,
                )
            if "family-summary-only" in active_methods:
                method_rows["family-summary-only"] = _evaluate_family(
                    mode="family-summary-only",
                    case=case,
                    tasks=tasks,
                    actions=actions,
                    catalog=catalog,
                    backend=backend,
                    config=active_config,
                )
            for method, row in method_rows.items():
                row.update(
                    {
                        "scale": requested_size,
                        "case_name": name,
                        "kind": case["kind"],
                        "method": method,
                        "expected_atomic_index": case.get("expected_atomic_index"),
                        "query": copy.deepcopy(case["query"]),
                    }
                )
                _score_row(case, row)
                rows.append(row)
                all_rows.append(row)
        by_method = {
            method: _aggregate([row for row in rows if row["method"] == method])
            for method in active_methods
        }
        exhaustive_metrics = by_method.get("exhaustive")
        family_metrics = by_method.get("family-routed")
        exhaustive_chars = (
            None
            if exhaustive_metrics is None
            else exhaustive_metrics["retrieval_input_characters"]
        )
        family_chars = (
            None
            if family_metrics is None
            else family_metrics["retrieval_input_characters"]
        )
        exhaustive_tokens = (
            None
            if exhaustive_metrics is None
            else exhaustive_metrics["retrieval_input_tokens"]
        )
        family_tokens = (
            None if family_metrics is None else family_metrics["retrieval_input_tokens"]
        )
        character_reduction = (
            None
            if exhaustive_chars in (None, 0) or family_chars is None
            else 1.0 - family_chars / exhaustive_chars
        )
        token_reduction = (
            None
            if exhaustive_tokens in (None, 0) or family_tokens is None
            else 1.0 - family_tokens / exhaustive_tokens
        )
        scales[str(requested_size)] = {
            "atomic_unit_count": actual_size,
            "task_unit_count": len(tasks),
            "action_unit_count": len(actions),
            "metrics": by_method,
            "family_input_character_reduction": character_reduction,
            "family_input_token_reduction": token_reduction,
            "family_input_reduction": (
                token_reduction if token_reduction is not None else character_reduction
            ),
            "family_input_reduction_measure": (
                "provider input tokens"
                if token_reduction is not None
                else "serialized input characters"
            ),
            "rows": rows,
        }

    final_scale = scales.get("300")
    family_metrics = [
        row
        for row in all_rows
        if row["method"] == "family-routed" and row.get("family_recall") is not None
    ]
    family_rows = [row for row in all_rows if row["method"] == "family-routed"]
    atomic_at_8_metrics = [
        row for row in family_rows if row.get("atomic_recall_at_8") is not None
    ]
    exhaustive_metrics = [row for row in all_rows if row["method"] == "exhaustive"]
    final_family_accuracy = (
        None
        if not family_rows
        else mean(row["final_selection_correct"] for row in family_rows)
    )
    final_exhaustive_accuracy = (
        None
        if not exhaustive_metrics
        else mean(row["final_selection_correct"] for row in exhaustive_metrics)
    )
    comparative_methods_present = bool(family_rows and exhaustive_metrics)
    gates = {
        "family_recall_at_2_at_least_095": (
            bool(family_metrics)
            and mean(row["family_recall"] for row in family_metrics) >= 0.95
        ),
        "atomic_recall_at_8_at_least_090": (
            bool(atomic_at_8_metrics)
            and mean(row["atomic_recall_at_8"] for row in atomic_at_8_metrics) >= 0.90
        ),
        "final_accuracy_drop_at_most_003": (
            comparative_methods_present
            and final_exhaustive_accuracy is not None
            and final_family_accuracy is not None
            and final_exhaustive_accuracy - final_family_accuracy <= 0.03
        ),
        "input_token_reduction_at_300_at_least_060": (
            final_scale is not None
            and final_scale["family_input_token_reduction"] is not None
            and final_scale["family_input_token_reduction"] >= 0.60
        ),
        "harmful_behavior_change_not_increased": sum(
            row["harmful_behavior_change"]
            for row in all_rows
            if row["method"] == "family-routed"
        )
        <= sum(
            row["harmful_behavior_change"]
            for row in all_rows
            if row["method"] == "exhaustive"
        )
        if comparative_methods_present
        else False,
    }
    return {
        "routing_config": active_config.to_dict(),
        "atomic_prompt_representation": "compact semantic cards v1",
        "evaluated_methods": list(active_methods),
        "scales": scales,
        "retrieval_gates": gates,
        "retrieval_gates_evaluable": comparative_methods_present,
        "retrieval_go": all(gates.values()),
    }


def compare_incremental_and_full_rebuild_reports(
    incremental: Mapping[str, Any],
    full_rebuild: Mapping[str, Any],
) -> dict[str, Any]:
    def rows(
        report: Mapping[str, Any],
    ) -> dict[tuple[str, str, str], Mapping[str, Any]]:
        result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        scales = report.get("scales")
        if not isinstance(scales, Mapping):
            raise TypeError("retrieval report is missing scales")
        for scale, payload in scales.items():
            if not isinstance(payload, Mapping):
                raise TypeError("retrieval scale payload must be an object")
            for row in payload.get("rows", []):
                if row.get("method") != "family-routed":
                    continue
                key = (str(scale), str(row.get("case_name")), str(row.get("kind")))
                if key in result:
                    raise ValueError("retrieval report contains a duplicate case")
                result[key] = row
        return result

    incremental_rows = rows(incremental)
    rebuild_rows = rows(full_rebuild)
    if set(incremental_rows) != set(rebuild_rows) or not incremental_rows:
        raise ValueError("incremental and full-rebuild retrieval cases differ")
    comparisons = []
    for key in sorted(incremental_rows):
        first = incremental_rows[key]
        second = rebuild_rows[key]
        agreed = (
            first.get("selected_atomic_knowledge")
            == second.get("selected_atomic_knowledge")
            and first.get("behavior_after") == second.get("behavior_after")
            and first.get("selected_candidate_geometry_index")
            == second.get("selected_candidate_geometry_index")
        )
        comparisons.append(
            {
                "scale": key[0],
                "case_name": key[1],
                "kind": key[2],
                "agreed": agreed,
            }
        )
    agreement = mean(value["agreed"] for value in comparisons)
    return {
        "case_count": len(comparisons),
        "final_retrieval_agreement": agreement,
        "agreement_at_least_090": agreement >= 0.90,
        "comparisons": comparisons,
    }


def _parse_corpus(value: str) -> tuple[int, Path]:
    raw_size, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("corpus must be SIZE=PATH")
    try:
        size = int(raw_size)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("corpus size must be an integer") from exc
    if size <= 0:
        raise argparse.ArgumentTypeError("corpus size must be positive")
    return size, Path(raw_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        type=_parse_corpus,
        required=True,
        help="repeat as SIZE=PATH for 20, 50, 100, and 300 unit corpora",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-responses", type=Path)
    source.add_argument("--call-agent-api", action="store_true")
    parser.add_argument("--planner-url", default="http://127.0.0.1:9104/plan")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--response-path", type=Path)
    parser.add_argument("--audit-output-dir", type=Path)
    parser.add_argument("--compare-with-full-rebuild", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "exhaustive",
            "family-routed",
            "shuffled-family",
            "family-summary-only",
        ),
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    corpora = dict(args.corpus)
    if not set(corpora).issubset({20, 50, 100, 300}):
        raise ValueError("corpus sizes must be selected from 20, 50, 100, and 300")
    saved = None
    if args.call_agent_api:
        backend = AgentApiHierarchicalRetrievalBackend(args.planner_url)
    else:
        responses = json.loads(args.model_responses.read_text(encoding="utf-8"))
        if not isinstance(responses, list):
            raise ValueError("model responses must be a JSON array")
        saved = SavedResponseBackend(responses)
        backend = saved
    recording = RecordingBackend(backend, checkpoint_path=args.response_path)
    report = evaluate_corpora(
        corpora,
        backend=recording,
        methods=args.methods,
    )
    report["evaluation_authority"] = (
        "live semantic model"
        if args.call_agent_api
        else "controlled held-in response oracle"
    )
    report["semantic_model_validated"] = bool(args.call_agent_api)
    report["input_reduction_measure"] = (
        "provider input tokens when available; serialized input characters otherwise"
    )
    report["formal_go_claim"] = bool(
        args.call_agent_api
        and report["retrieval_go"]
        and set(corpora) == {20, 50, 100, 300}
        and all(
            metrics["retrieval_input_tokens"] is not None
            for scale in report["scales"].values()
            for metrics in scale["metrics"].values()
            if metrics["retrieval_calls"]
        )
    )
    if args.compare_with_full_rebuild is not None:
        full_rebuild = json.loads(
            args.compare_with_full_rebuild.read_text(encoding="utf-8")
        )
        report["incremental_vs_full_rebuild"] = (
            compare_incremental_and_full_rebuild_reports(report, full_rebuild)
        )
    else:
        report["incremental_vs_full_rebuild"] = None
    report["formal_go_claim"] = bool(
        report["formal_go_claim"]
        and report["incremental_vs_full_rebuild"] is not None
        and report["incremental_vs_full_rebuild"]["agreement_at_least_090"]
    )
    if saved is not None:
        saved.assert_consumed()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.response_path is not None:
        args.response_path.parent.mkdir(parents=True, exist_ok=True)
        args.response_path.write_text(
            json.dumps(recording.outputs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.audit_output_dir is not None:
        args.audit_output_dir.mkdir(parents=True, exist_ok=True)
        largest_scale = report["scales"][str(max(corpora))]
        for kind in ("task", "action"):
            candidates = [
                row
                for row in largest_scale["rows"]
                if row["kind"] == kind
                and row["expected_atomic_index"] is not None
                and row["method"] in {"exhaustive", "family-routed"}
            ]
            by_method = {row["method"]: row for row in candidates}
            if set(by_method) != {"exhaustive", "family-routed"}:
                raise ValueError(f"missing positive {kind} audit case")
            audit = {
                "evaluation_authority": report["evaluation_authority"],
                "semantic_model_validated": report["semantic_model_validated"],
                "scale": largest_scale["atomic_unit_count"],
                "case_name": candidates[0]["case_name"],
                "query": candidates[0]["query"],
                "exhaustive": by_method["exhaustive"],
                "family_routed": by_method["family-routed"],
            }
            (
                args.audit_output_dir / f"real_{kind}_exhaustive_vs_family.json"
            ).write_text(
                json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(report["retrieval_gates"], sort_keys=True))


if __name__ == "__main__":
    main()
