#!/usr/bin/env python3
"""Audit whether ESI Task Knowledge adds information beyond an official prompt."""

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

from roboharn_evo.agent.hpk.hierarchical_knowledge import SubtaskKnowledgeV3  # noqa: E402
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (  # noqa: E402
    HierarchicalBackendCompletion,
)
from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    AgentApiESIEvaluatedModel,
    ESIKnowledgeError,
    VLMTaskKnowledgeNoveltyAuditor,
)


class _AgentApiNoveltyBackend:
    def __init__(self, service_url: str) -> None:
        self._transport = AgentApiESIEvaluatedModel(
            service_url,
            model="gpt-5.5",
            reasoning_effort="xhigh",
        )
        self.health = self._transport.preflight()
        self.call_audits: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> HierarchicalBackendCompletion:
        if kwargs.get("images"):
            raise ESIKnowledgeError("novelty audit is text-only")
        output, _raw, _status = self._transport.generate_json(
            contents=[kwargs["input_text"]],
            system_instruction=kwargs["instructions"],
            response_schema=kwargs["output_schema"],
            schema_name=kwargs["schema_name"],
            max_output_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            fallback=None,
        )
        self.call_audits.append(dict(self._transport.last_call_audit or {}))
        return HierarchicalBackendCompletion(output=output)


class _QueueBackend:
    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self._responses = [dict(item) for item in responses]

    def complete(self, **_kwargs: Any) -> HierarchicalBackendCompletion:
        if not self._responses:
            raise ESIKnowledgeError("precomputed novelty response queue is empty")
        return HierarchicalBackendCompletion(output=self._responses.pop(0))

    def require_empty(self) -> None:
        if self._responses:
            raise ESIKnowledgeError("unused precomputed novelty responses remain")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a semantic novelty audit over ESI Task Knowledge."
    )
    parser.add_argument("--task-knowledge-jsonl", type=Path, required=True)
    parser.add_argument(
        "--official-prompt-file",
        type=Path,
        action="append",
        required=True,
        help="Repeat when one knowledge pool spans multiple official task prompts.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--judge-label", required=True)
    parser.add_argument("--planner-url")
    parser.add_argument("--judge-responses", type=Path)
    args = parser.parse_args()
    if bool(args.planner_url) == bool(args.judge_responses):
        parser.error("select exactly one judge source")
    return args


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records = []
    for line_number, line in enumerate(
        path.expanduser().resolve().read_text(encoding="utf-8").splitlines(), 1
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


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"novelty output root is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prompts = tuple(
        path.expanduser().resolve().read_text(encoding="utf-8")
        for path in args.official_prompt_file
    )
    prompt = "\n\n--- NEXT OFFICIAL PROMPT ---\n\n".join(
        value.strip() for value in prompts
    )
    knowledge = tuple(
        SubtaskKnowledgeV3(item) for item in _read_jsonl(args.task_knowledge_jsonl)
    )
    if not knowledge:
        raise ESIKnowledgeError("novelty audit requires Task Knowledge")
    if args.judge_responses is not None:
        backend = _QueueBackend(_read_jsonl(args.judge_responses))
    else:
        backend = _AgentApiNoveltyBackend(args.planner_url)
    auditor = VLMTaskKnowledgeNoveltyAuditor(backend)
    records = []
    for source_index, item in enumerate(knowledge):
        payload = item.to_dict()
        records.append(
            {
                "source_index": source_index,
                "semantic_content": {
                    "condition": payload["condition"],
                    "subtask_strategy": payload["subtask_strategy"],
                },
                "judgment": auditor.audit(
                    official_prompt=prompt,
                    knowledge=item,
                ).to_dict(),
            }
        )
    if isinstance(backend, _QueueBackend):
        backend.require_empty()
        call_audits = []
        health_model = None
    else:
        call_audits = backend.call_audits
        health_model = backend.health.get("model")
    (output / "official_prompt.txt").write_text(prompt, encoding="utf-8")
    (output / "novelty_audit.jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )
    counts = {
        label: sum(item["judgment"]["classification"] == label for item in records)
        for label in ("fully covered", "partially novel", "substantially novel")
    }
    (output / "summary.json").write_text(
        json.dumps(
            {
                "scientific_scope": "offline ESI Task Knowledge novelty audit",
                "judge_label": " ".join(args.judge_label.strip().split()),
                "knowledge_count": len(records),
                "official_prompt_count": len(prompts),
                "classification_counts": counts,
                "keyword_overlap_heuristic_used": False,
                "health_model": health_model,
                "external_call_audits": call_audits,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
