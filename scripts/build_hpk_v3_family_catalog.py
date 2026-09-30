from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.family_store import save_hierarchical_store_with_catalog
from roboharn_evo.agent.hpk.hierarchical_retriever import (
    AgentApiHierarchicalRetrievalBackend,
)
from roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store
from roboharn_evo.agent.hpk.knowledge_family import VLMKnowledgeFamilyCatalogBuilder
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    HierarchicalBackendCompletion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE_ROOT = REPO_ROOT / "eval_result/hpk/v3/merged_store"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "eval_result/hpk/v3/family_catalog/merged_store"


class _SavedCatalogBackend:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def complete(self, **_: Any) -> HierarchicalBackendCompletion:
        return HierarchicalBackendCompletion(output=self._response)


def build_family_catalog(
    *,
    store_root: Path,
    output_root: Path,
    builder: VLMKnowledgeFamilyCatalogBuilder,
    response_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    tasks, actions = load_hierarchical_store(store_root)
    result = builder.build(task_knowledge=tasks, action_knowledge=actions)
    save_hierarchical_store_with_catalog(
        output_root,
        task_knowledge=tasks,
        action_knowledge=actions,
        catalog=result.catalog,
    )
    if response_path is not None:
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps(result.raw_response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    report = {
        "source_store": str(store_root),
        "output_store": str(output_root),
        "supported_task_knowledge": sum(
            value["status"] == "supported" for value in tasks
        ),
        "supported_action_knowledge": sum(
            value["status"] == "supported" for value in actions
        ),
        "task_family_count": len(result.catalog.task_families),
        "action_family_count": len(result.catalog.action_families),
        "action_family_counts": {
            action: sum(
                family["action"] == action for family in result.catalog.action_families
            )
            for action in ("grasp", "place", "contact")
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--catalog-response",
        type=Path,
        help="saved strict JSON from an authorized catalog-builder call",
    )
    source.add_argument(
        "--call-gpt",
        action="store_true",
        help="perform one text-only GPT catalog-builder call with no retry",
    )
    parser.add_argument("--gpt-url", default="http://127.0.0.1:9104")
    parser.add_argument("--response-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.call_gpt:
        backend = AgentApiHierarchicalRetrievalBackend(
            args.gpt_url,
            timeout_sec=600,
        )
    else:
        response = json.loads(args.catalog_response.read_text(encoding="utf-8"))
        backend = _SavedCatalogBackend(response)
    report = build_family_catalog(
        store_root=args.store_root,
        output_root=args.output_root,
        builder=VLMKnowledgeFamilyCatalogBuilder(backend),
        response_path=args.response_path,
        report_path=args.report_path,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
