"""Consolidate explicit cross-domain Action Knowledge into a frozen Store."""

from __future__ import annotations

import argparse
from pathlib import Path

from roboharn_evo.agent.hpk.hierarchical_retriever import (
    AgentApiHierarchicalRetrievalBackend,
)
from roboharn_evo.benchmark_adapters.libero_pro.transfer_knowledge import (
    build_augmented_action_store,
    build_transfer_store_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-store-root", type=Path, required=True)
    parser.add_argument(
        "--atomic-knowledge",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--action", choices=("contact", "grasp", "place"), required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--libero-expert-root", type=Path)
    parser.add_argument("--bundle-output-root", type=Path)
    parser.add_argument("--planner-url", default="http://127.0.0.1:9104/plan")
    parser.add_argument("--timeout-sec", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.libero_expert_root is None) != (args.bundle_output_root is None):
        raise ValueError(
            "--libero-expert-root and --bundle-output-root must be provided together"
        )
    output = build_augmented_action_store(
        base_store_root=args.base_store_root,
        atomic_knowledge_paths=args.atomic_knowledge,
        action=args.action,
        output_root=args.output_root,
        backend=AgentApiHierarchicalRetrievalBackend(
            args.planner_url,
            timeout_sec=args.timeout_sec,
        ),
        source_label=args.source_label,
    )
    print(output)
    if args.libero_expert_root is not None:
        bundle = build_transfer_store_bundle(
            rmbench_store_root=output,
            libero_expert_root=args.libero_expert_root,
            output_root=args.bundle_output_root,
        )
        print(bundle.root)


if __name__ == "__main__":
    main()
