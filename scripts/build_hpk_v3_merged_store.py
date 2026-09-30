from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.hierarchical_store import (
    load_hierarchical_store,
    save_hierarchical_store,
)
from roboharn_evo.agent.hpk.semantic_consolidator import VLMKnowledgeConsolidator
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    ExistingGPTMultimodalReflectorBackend,
    HierarchicalBackendCompletion,
)
from roboharn_evo.agent.reflector.multimodal_transport import (
    ImageEgressAuthorization,
    OpenAICompatibleMultimodalTransport,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "eval_result/hpk/v3/rmbench_gt_hierarchical_generic"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "eval_result/hpk/v3/merged_store"
DEFAULT_REPORT_PATH = REPO_ROOT / "eval_result/hpk/v3/merged_store_report.json"
DEFAULT_RESPONSE_PATH = (
    REPO_ROOT / "eval_result/hpk/v3/semantic_consolidation_response.json"
)


class _SavedConsolidationBackend:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def complete(self, **_: Any) -> HierarchicalBackendCompletion:
        return HierarchicalBackendCompletion(output=self._response)


def _load_inputs(
    input_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    task_units: list[dict[str, Any]] = []
    action_units: list[dict[str, Any]] = []
    sources = sorted(input_root.glob("*/atomic_knowledge.json"))
    if not sources:
        raise FileNotFoundError(f"no atomic knowledge found under {input_root}")
    source_names: list[str] = []
    for source in sources:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if set(payload) != {"task_knowledge", "action_knowledge"}:
            raise ValueError(f"unexpected atomic knowledge fields in {source}")
        task_units.extend(payload["task_knowledge"])
        action_units.extend(payload["action_knowledge"])
        source_names.append(source.parent.name)
    return task_units, action_units, source_names


def build_merged_store(
    *,
    input_root: Path,
    output_root: Path,
    report_path: Path,
    consolidation_response: dict[str, Any] | None = None,
    consolidator: VLMKnowledgeConsolidator | None = None,
    response_path: Path | None = None,
) -> dict[str, Any]:
    if (consolidation_response is None) == (consolidator is None):
        raise ValueError(
            "provide exactly one saved consolidation response or live consolidator"
        )
    task_units, action_units, source_names = _load_inputs(input_root)
    active = consolidator
    if active is None:
        assert consolidation_response is not None
        active = VLMKnowledgeConsolidator(
            _SavedConsolidationBackend(consolidation_response)
        )
    consolidation = active.consolidate(
        task_knowledge=task_units,
        action_knowledge=action_units,
    )
    if response_path is not None:
        if not isinstance(consolidation.raw_response, dict):
            raise TypeError("semantic consolidation response must be a JSON object")
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps(consolidation.raw_response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    save_hierarchical_store(
        output_root,
        task_knowledge=consolidation.task_knowledge,
        action_knowledge=consolidation.action_knowledge,
    )
    merged_tasks, merged_actions = load_hierarchical_store(output_root)
    report = {
        "source": "VLM semantic consolidation of real hierarchical outputs",
        "source_collections": source_names,
        "store": {
            "task_knowledge": "task_knowledge.jsonl",
            "action_knowledge": "action_knowledge.jsonl",
        },
        "counts": {
            "task_units_before": len(task_units),
            "task_units_after": len(merged_tasks),
            "action_units_before": len(action_units),
            "action_units_after": len(merged_actions),
        },
        "task_knowledge": [
            {
                "subtask": item["subtask_strategy"]["subtask"],
                "evidence_summary": item["evidence_summary"],
                "status": item["status"],
            }
            for item in merged_tasks
        ],
        "action_knowledge": [
            {
                "action": item["condition"]["action"],
                "evidence_summary": item["evidence_summary"],
                "status": item["status"],
            }
            for item in merged_actions
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--consolidation-response",
        type=Path,
        help="saved strict JSON returned by one authorized semantic consolidation call",
    )
    source.add_argument(
        "--call-gpt",
        action="store_true",
        help="perform one GPT semantic consolidation call with no retry",
    )
    parser.add_argument("--gpt-url", default="http://127.0.0.1:9104")
    parser.add_argument("--response-path", type=Path, default=DEFAULT_RESPONSE_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    response = (
        None
        if args.consolidation_response is None
        else json.loads(args.consolidation_response.read_text(encoding="utf-8"))
    )
    consolidator = None
    if args.call_gpt:
        authorization = ImageEgressAuthorization.operator_granted(
            assertion="HPK v3 semantic knowledge consolidation",
            scope="hpk_v3_atomic_knowledge_text_only",
        )
        transport = OpenAICompatibleMultimodalTransport(
            service_url=args.gpt_url,
            model="gpt-5.5",
            reasoning_effort="xhigh",
            timeout_sec=600,
            max_images=16,
        )
        transport.capability_preflight(authorization=authorization)
        consolidator = VLMKnowledgeConsolidator(
            ExistingGPTMultimodalReflectorBackend(
                transport,
                authorization=authorization,
            )
        )
    report = build_merged_store(
        input_root=args.input_root,
        output_root=args.output_root,
        report_path=args.report_path,
        consolidation_response=response,
        consolidator=consolidator,
        response_path=args.response_path if args.call_gpt else None,
    )
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
