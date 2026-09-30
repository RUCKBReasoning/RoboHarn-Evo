from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.family_store import (
    load_knowledge_family_catalog,
    save_hierarchical_store_with_catalog,
)
from roboharn_evo.agent.hpk.hierarchical_retriever import (
    AgentApiHierarchicalRetrievalBackend,
)
from roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store
from roboharn_evo.agent.hpk.incremental_maintainer import (
    IncrementalKnowledgeMaintainer,
    load_atomic_knowledge,
)
from roboharn_evo.agent.hpk.knowledge_family import VLMKnowledgeFamilyCatalogBuilder
from roboharn_evo.agent.hpk.semantic_consolidator import VLMKnowledgeConsolidator
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    HierarchicalBackendCompletion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE_ROOT = REPO_ROOT / "eval_result/hpk/v3/family_catalog/merged_store"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "eval_result/hpk/v3/incremental_maintenance/store"


class SavedResponseSequenceBackend:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = copy.deepcopy(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> HierarchicalBackendCompletion:
        self.calls.append(copy.deepcopy(kwargs))
        if not self._responses:
            raise RuntimeError("saved model response sequence is exhausted")
        return HierarchicalBackendCompletion(output=self._responses.pop(0))

    def assert_consumed(self) -> None:
        if self._responses:
            raise RuntimeError("saved model response sequence contains unused values")


class RecordingBackend:
    def __init__(self, backend: Any, *, checkpoint_path: Path | None = None) -> None:
        self._backend = backend
        self._checkpoint_path = checkpoint_path
        self.responses: list[dict[str, Any] | str] = []
        self.calls: list[dict[str, Any]] = []
        self.audits: list[dict[str, Any] | None] = []

    def complete(self, **kwargs: Any) -> HierarchicalBackendCompletion:
        self.calls.append(copy.deepcopy(kwargs))
        completion = self._backend.complete(**kwargs)
        self.audits.append(
            copy.deepcopy(dict(completion.audit))
            if isinstance(completion.audit, Mapping)
            else None
        )
        output = completion.output
        self.responses.append(
            copy.deepcopy(dict(output)) if isinstance(output, dict) else output
        )
        if self._checkpoint_path is not None:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_path.write_text(
                json.dumps(self.responses, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return completion

    def usage_summary(self) -> dict[str, Any]:
        input_tokens: list[int] = []
        for audit in self.audits:
            if audit is None:
                continue
            sources = [audit]
            for key in ("usage", "provider_usage"):
                nested = audit.get(key)
                if isinstance(nested, Mapping):
                    sources.append(nested)
            token_value = next(
                (
                    source[key]
                    for source in sources
                    for key in ("input_tokens", "prompt_tokens")
                    if isinstance(source.get(key), int)
                    and not isinstance(source[key], bool)
                ),
                None,
            )
            if token_value is not None:
                input_tokens.append(token_value)
        return {
            "model_calls": len(self.calls),
            "model_input_characters": sum(
                len(str(call.get("input_text", ""))) for call in self.calls
            ),
            "model_input_tokens": (
                sum(input_tokens) if len(input_tokens) == len(self.calls) else None
            ),
        }


def maintain_family_store(
    *,
    store_root: Path,
    atomic_knowledge_path: Path,
    output_root: Path,
    backend: Any,
    report_path: Path | None = None,
    response_path: Path | None = None,
) -> dict[str, Any]:
    tasks, actions = load_hierarchical_store(store_root)
    catalog = load_knowledge_family_catalog(
        store_root,
        task_knowledge=tasks,
        action_knowledge=actions,
    )
    new_tasks, new_actions = load_atomic_knowledge(atomic_knowledge_path)
    recording = RecordingBackend(backend, checkpoint_path=response_path)
    result = IncrementalKnowledgeMaintainer(
        recording,
        consolidator=VLMKnowledgeConsolidator(recording),
    ).maintain(
        task_knowledge=tasks,
        action_knowledge=actions,
        catalog=catalog,
        new_task_knowledge=new_tasks,
        new_action_knowledge=new_actions,
        output_root=output_root,
    )
    report = {
        "source_store": str(store_root),
        "new_atomic_knowledge": str(atomic_knowledge_path),
        "output_store": str(output_root),
        "counts_before": {
            "task_knowledge": len(tasks),
            "action_knowledge": len(actions),
            "task_families": len(catalog.task_families),
            "action_families": len(catalog.action_families),
        },
        "counts_after": {
            "task_knowledge": len(result.task_knowledge),
            "action_knowledge": len(result.action_knowledge),
            "task_families": len(result.catalog.task_families),
            "action_families": len(result.catalog.action_families),
        },
        "maintenance": result.audit,
        "model_call_schemas": [call["schema_name"] for call in recording.calls],
        "model_usage": recording.usage_summary(),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if response_path is not None:
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps(recording.responses, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _evidence_totals(tasks: Any, actions: Any) -> dict[str, dict[str, int]]:
    task = {key: 0 for key in ("support", "oppose", "unverified")}
    action = {
        key: 0
        for key in (
            "support",
            "oppose",
            "unverified",
            "independent_verified_trials",
        )
    }
    for value in tasks:
        for key in task:
            task[key] += value["evidence_summary"][key]
    for value in actions:
        for key in action:
            action[key] += value["evidence_summary"][key]
    return {"task": task, "action": action}


def rebuild_family_store(
    *,
    store_root: Path,
    atomic_knowledge_path: Path,
    output_root: Path,
    backend: Any,
    report_path: Path | None = None,
    response_path: Path | None = None,
) -> dict[str, Any]:
    """Full consolidation + catalog rebuild control for incremental maintenance."""

    tasks, actions = load_hierarchical_store(store_root)
    new_tasks, new_actions = load_atomic_knowledge(atomic_knowledge_path)
    recording = RecordingBackend(backend, checkpoint_path=response_path)
    consolidated = VLMKnowledgeConsolidator(recording).consolidate(
        task_knowledge=(*tasks, *new_tasks),
        action_knowledge=(*actions, *new_actions),
    )
    catalog = (
        VLMKnowledgeFamilyCatalogBuilder(recording)
        .build(
            task_knowledge=consolidated.task_knowledge,
            action_knowledge=consolidated.action_knowledge,
        )
        .catalog
    )
    save_hierarchical_store_with_catalog(
        output_root,
        task_knowledge=consolidated.task_knowledge,
        action_knowledge=consolidated.action_knowledge,
        catalog=catalog,
    )

    expected_existing = _evidence_totals(tasks, actions)
    expected_new = _evidence_totals(new_tasks, new_actions)
    expected_after = {
        kind: {
            key: expected_existing[kind][key] + expected_new[kind][key]
            for key in expected_existing[kind]
        }
        for kind in expected_existing
    }
    actual_after = _evidence_totals(
        consolidated.task_knowledge,
        consolidated.action_knowledge,
    )
    evidence_error = sum(
        abs(expected_after[kind][key] - actual_after[kind][key])
        for kind in expected_after
        for key in expected_after[kind]
    )
    if evidence_error:
        raise ValueError("full rebuild did not conserve evidence")
    report = {
        "operation": "full rebuild control",
        "source_store": str(store_root),
        "new_atomic_knowledge": str(atomic_knowledge_path),
        "output_store": str(output_root),
        "counts_before": {
            "task_knowledge": len(tasks),
            "action_knowledge": len(actions),
        },
        "new_counts": {
            "task_knowledge": len(new_tasks),
            "action_knowledge": len(new_actions),
        },
        "counts_after": {
            "task_knowledge": len(consolidated.task_knowledge),
            "action_knowledge": len(consolidated.action_knowledge),
            "task_families": len(catalog.task_families),
            "action_families": len(catalog.action_families),
        },
        "model_call_schemas": [call["schema_name"] for call in recording.calls],
        "model_usage": recording.usage_summary(),
        "evidence": {
            "existing": expected_existing,
            "new": expected_new,
            "expected_after": expected_after,
            "actual_after": actual_after,
            "evidence_conservation_error": evidence_error,
        },
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument(
        "--operation",
        choices=("incremental", "full-rebuild"),
        default="incremental",
    )
    parser.add_argument("--atomic-knowledge", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model-responses",
        type=Path,
        help="saved ordered strict JSON responses for reproducible maintenance",
    )
    source.add_argument(
        "--call-gpt",
        action="store_true",
        help="perform the required text-only GPT calls with no retry",
    )
    parser.add_argument("--gpt-url", default="http://127.0.0.1:9104")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--response-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    saved_backend = None
    if args.call_gpt:
        backend = AgentApiHierarchicalRetrievalBackend(
            args.gpt_url,
            timeout_sec=600,
        )
    else:
        responses = json.loads(args.model_responses.read_text(encoding="utf-8"))
        if not isinstance(responses, list) or not all(
            isinstance(value, dict) for value in responses
        ):
            raise ValueError("model responses must be a JSON array of objects")
        saved_backend = SavedResponseSequenceBackend(responses)
        backend = saved_backend
    operation = (
        maintain_family_store
        if args.operation == "incremental"
        else rebuild_family_store
    )
    report = operation(
        store_root=args.store_root,
        atomic_knowledge_path=args.atomic_knowledge,
        output_root=args.output_root,
        backend=backend,
        report_path=args.report_path,
        response_path=args.response_path,
    )
    if saved_backend is not None:
        saved_backend.assert_consumed()
    print(
        json.dumps(
            report.get("maintenance", report),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
