#!/usr/bin/env python3
"""Build a frozen, task-only ESI SubtaskKnowledgeV3 store."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.agent.hpk.hierarchical_retriever import (  # noqa: E402
    AgentApiHierarchicalRetrievalBackend,
)
from roboharn_evo.agent.hpk.semantic_consolidator import (  # noqa: E402
    VLMKnowledgeConsolidator,
)
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (  # noqa: E402
    HierarchicalBackendCompletion,
)
from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    ESIKnowledgeError,
    ESITaskConsolidationBackend,
    ESITaskReflection,
    ESITraceRecord,
    VLMESITaskReflector,
    build_shuffle_manifest,
    compile_trace_reflection,
    consolidate_task_knowledge,
    load_split_manifest,
    save_frozen_task_store,
    save_shuffle_manifest,
    validate_store_split_provenance,
)


class _QueueBackend:
    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self._responses = [dict(item) for item in responses]

    def complete(self, **_kwargs) -> HierarchicalBackendCompletion:
        if not self._responses:
            raise RuntimeError("precomputed ESI response queue is empty")
        return HierarchicalBackendCompletion(output=self._responses.pop(0))

    def require_empty(self) -> None:
        if self._responses:
            raise RuntimeError("unused precomputed ESI responses remain")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile correct ESI source traces into frozen Task Knowledge."
    )
    parser.add_argument(
        "--source-traces",
        type=Path,
        action="append",
        required=True,
        help="Repeat to consolidate multiple source-task bundles jointly.",
    )
    parser.add_argument(
        "--source-collection-receipt",
        type=Path,
        action="append",
        default=[],
        help="Receipt paired by position with each --source-traces bundle.",
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--planner-url")
    parser.add_argument(
        "--reflection-responses",
        type=Path,
        action="append",
        default=[],
        help="Repeat to provide precomputed reflections from every source bundle.",
    )
    parser.add_argument("--consolidation-response", type=Path)
    parser.add_argument("--shuffle-seed", type=int, default=20260825)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.planner_url and not args.reflection_responses:
        parser.error("reflection requires --planner-url or --reflection-responses")
    if args.consolidation_response is None and not args.planner_url:
        parser.error("live consolidation requires --planner-url")
    if args.source_collection_receipt and len(args.source_collection_receipt) != len(
        args.source_traces
    ):
        parser.error(
            "repeat --source-collection-receipt once per --source-traces bundle"
        )
    return args


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records = []
    for line_number, line in enumerate(
        path.expanduser().resolve().read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ESIKnowledgeError(
                f"{path.name} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise ESIKnowledgeError(f"{path.name} line {line_number} must be an object")
        records.append(dict(value))
    return tuple(records)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(value), ensure_ascii=False, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _receipt_source_refs(path: Path) -> set[str]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ESIKnowledgeError("cannot read source collection receipt") from exc
    if not isinstance(value, Mapping):
        raise ESIKnowledgeError("source collection receipt must be one object")
    expected = {
        "schema",
        "status",
        "required_correct",
        "attempt_cap",
        "attempts",
        "correct_trace_count",
        "automatic_retries",
    }
    if (
        set(value) != expected
        or value.get("schema") not in {"roboharn_evo/esi_bench/source_collection_receipt/v1", "tcm/esi_bench/source_collection_receipt/v1"}
        or value.get("status") != "completed"
        or value.get("automatic_retries") != 0
    ):
        raise ESIKnowledgeError("source collection receipt is not completed")
    required = value.get("required_correct")
    attempts = value.get("attempts")
    if (
        isinstance(required, bool)
        or not isinstance(required, int)
        or required < 1
        or not isinstance(attempts, list)
    ):
        raise ESIKnowledgeError("source collection receipt counts are invalid")
    refs = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ESIKnowledgeError("source collection attempt must be one object")
        if attempt.get("benchmark_correct") is True:
            if attempt.get("rollout_exit_code") != 0:
                raise ESIKnowledgeError("correct source attempt has a failed rollout")
            summary_exit_code = attempt.get("summary_exit_code")
            if summary_exit_code != 0:
                if isinstance(summary_exit_code, bool) or not isinstance(
                    summary_exit_code, int
                ):
                    raise ESIKnowledgeError(
                        "correct source attempt lacks a materialization result"
                    )
                continue
            ref = " ".join(str(attempt.get("instance_ref") or "").strip().split())
            if not ref:
                raise ESIKnowledgeError("correct source attempt has no instance_ref")
            refs.append(ref)
    if (
        len(refs) != required
        or len(set(refs)) != len(refs)
        or value.get("correct_trace_count") != len(refs)
    ):
        raise ESIKnowledgeError("source collection receipt trace count mismatch")
    return set(refs)


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"knowledge output root is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(args.split_manifest)
    source_refs = {item.instance_ref for item in manifest.source}
    heldout_questions = tuple(item.question_text for item in manifest.heldout)
    traces = tuple(
        ESITraceRecord.from_mapping(item)
        for path in args.source_traces
        for item in _read_jsonl(path)
    )
    if len({item.source_ref for item in traces}) != len(traces):
        raise ESIKnowledgeError(
            "source trace file contains duplicate source_ref values"
        )
    trace_refs = {item.source_ref for item in traces}
    if args.source_collection_receipt:
        receipt_ref_sets = tuple(
            _receipt_source_refs(path) for path in args.source_collection_receipt
        )
        if sum(len(values) for values in receipt_ref_sets) != len(
            set().union(*receipt_ref_sets)
        ):
            raise ESIKnowledgeError(
                "source collection receipts contain overlapping source refs"
            )
        expected_trace_refs = set().union(*receipt_ref_sets)
    else:
        expected_trace_refs = source_refs
    if not expected_trace_refs.issubset(source_refs):
        raise ESIKnowledgeError("source collection receipt contains non-source refs")
    if trace_refs != expected_trace_refs:
        missing = sorted(expected_trace_refs - trace_refs)
        extra = sorted(trace_refs - expected_trace_refs)
        raise ESIKnowledgeError(
            f"source trace/split mismatch: missing={missing}, extra={extra}"
        )
    correct_source_refs = {item.source_ref for item in traces if item.benchmark_correct}

    precomputed_reflections: dict[str, ESITaskReflection] = {}
    if args.reflection_responses:
        for path in args.reflection_responses:
            for item in _read_jsonl(path):
                reflection = ESITaskReflection.from_mapping(item)
                if reflection.source_ref in precomputed_reflections:
                    raise ESIKnowledgeError(
                        "duplicate precomputed reflection source_ref"
                    )
                precomputed_reflections[reflection.source_ref] = reflection
        if not set(precomputed_reflections).issubset(correct_source_refs):
            raise ESIKnowledgeError(
                "precomputed reflections contain an unknown source trace"
            )
        if not args.planner_url and set(precomputed_reflections) != correct_source_refs:
            raise ESIKnowledgeError(
                "offline reflections must cover exactly the correct source traces"
            )
    reflection_backend = (
        AgentApiHierarchicalRetrievalBackend(args.planner_url)
        if args.planner_url
        else None
    )
    if reflection_backend is not None:
        reflector = VLMESITaskReflector(
            reflection_backend,
            include_images=False,
        )
    else:
        reflector = None

    compiled = []
    flat_lessons = []
    raw_reflections = []
    rejected = []
    for trace in traces:
        if not trace.benchmark_correct:
            rejected.append(
                {
                    "source_ref": trace.source_ref,
                    "reason": "source episode was not benchmark-correct",
                }
            )
            continue
        try:
            reflection = (
                precomputed_reflections[trace.source_ref]
                if trace.source_ref in precomputed_reflections
                else reflector.reflect(trace)
            )
            units, lesson = compile_trace_reflection(
                trace,
                reflection,
                heldout_questions=heldout_questions,
            )
        except Exception as exc:
            rejected.append(
                {
                    "source_ref": trace.source_ref,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        raw_reflections.append(reflection.to_dict())
        compiled.extend(units)
        flat_lessons.append(lesson)
        for item in units:
            if item.knowledge["evidence_summary"]["support"] == 0:
                rejected.append(
                    {
                        "source_ref": item.source_ref,
                        "reason": "subtask-to-observation attribution unverified",
                        "knowledge": item.knowledge.to_dict(),
                    }
                )

    _write_jsonl(output / "reflection_outputs.jsonl", raw_reflections)
    _write_jsonl(output / "rejected_or_unverified.jsonl", rejected)
    (output / "reflection_stage_audit.json").write_text(
        json.dumps(
            {
                "source_trace_count": len(traces),
                "correct_source_trace_count": len(correct_source_refs),
                "reflection_output_count": len(raw_reflections),
                "compiled_candidate_count": len(compiled),
                "attribution_supported_candidate_count": sum(
                    item.knowledge["evidence_summary"]["support"] > 0
                    for item in compiled
                ),
                "single_source_lifecycle": "candidate",
                "rejected_or_unverified_count": len(rejected),
                "store_write_performed_at_stage": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.consolidation_response is not None:
        payload = json.loads(
            args.consolidation_response.expanduser()
            .resolve()
            .read_text(encoding="utf-8")
        )
        if not isinstance(payload, Mapping):
            raise ESIKnowledgeError("consolidation response must be one object")
        queue_backend = _QueueBackend((payload,))
        consolidation_backend = queue_backend
    else:
        queue_backend = None
        consolidation_backend = ESITaskConsolidationBackend(reflection_backend)
    consolidated = consolidate_task_knowledge(
        VLMKnowledgeConsolidator(consolidation_backend),
        compiled,
        flat_lessons=flat_lessons,
        source_answers=tuple(trace.final_answer for trace in traces),
        heldout_questions=heldout_questions,
    )
    if queue_backend is not None:
        queue_backend.require_empty()

    _write_jsonl(
        output / "lifecycle_candidates.jsonl",
        [item.to_audit_dict() for item in consolidated.candidates],
    )
    shuffle_written = False
    if consolidated.store is not None:
        save_frozen_task_store(output, consolidated.store, overwrite=args.overwrite)
        validate_store_split_provenance(consolidated.store, manifest)
        if len(consolidated.store.knowledge) >= 2:
            shuffle = build_shuffle_manifest(
                len(consolidated.store.knowledge),
                seed=args.shuffle_seed,
            )
            shuffle_path = output / "shuffle_manifest.json"
            if shuffle_path.exists() and args.overwrite:
                shuffle_path.unlink()
            save_shuffle_manifest(shuffle_path, shuffle)
            shuffle_written = True
    (output / "consolidation_response.json").write_text(
        json.dumps(
            consolidated.raw_response,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "scientific_scope": "ESI RQ2 Task Knowledge only",
        "source_bundle_count": len(args.source_traces),
        "source_trace_count": len(traces),
        "compiled_candidate_count": len(compiled),
        "attribution_supported_candidate_count": sum(
            item.knowledge["evidence_summary"]["support"] > 0 for item in compiled
        ),
        "candidate_knowledge_count": sum(
            item.lifecycle_status == "candidate" for item in consolidated.candidates
        ),
        "development_supported_knowledge_count": sum(
            item.lifecycle_status == "development-supported"
            for item in consolidated.candidates
        ),
        "frozen_task_knowledge_count": (
            0 if consolidated.store is None else len(consolidated.store.knowledge)
        ),
        "flat_lesson_count": (
            0 if consolidated.store is None else len(consolidated.store.flat_lessons)
        ),
        "evaluation_store_written": consolidated.store is not None,
        "shuffle_manifest_written": shuffle_written,
        "rejected_or_unverified_count": len(rejected),
        "action_knowledge_count": 0,
        "heldout_store_update_enabled": False,
    }
    (output / "build_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
