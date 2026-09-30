#!/usr/bin/env python3
"""Run the offline, candidate-only Self-Evolution Phase A bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from roboharn_evo.agent.reflector.bootstrap import (
    BootstrapError,
    BootstrapResourceLimits,
    bootstrap_expert_trajectories,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read exactly the RMBench expert episodes listed by a content-addressed "
            "manifest and write an offline candidate-only Phase A audit."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="explicit roboharn_evo/expert_bootstrap_manifest/v1 JSON file",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help=(
            "fresh absolute run directory under /tmp or "
            "RoboHarn-Evo/eval_result/self_evolution/phase_a"
        ),
    )
    defaults = BootstrapResourceLimits()
    parser.add_argument(
        "--max-entries",
        type=int,
        default=defaults.max_entries,
        help="maximum manifest entries (full pools require an explicit override)",
    )
    parser.add_argument(
        "--max-total-artifacts",
        type=int,
        default=defaults.max_total_artifacts,
        help="maximum manifest artifact processing nodes",
    )
    parser.add_argument(
        "--max-total-declared-bytes",
        type=int,
        default=defaults.max_total_declared_bytes,
        help="maximum total bytes declared by unique artifact descriptors",
    )
    parser.add_argument(
        "--max-total-frames",
        type=int,
        default=defaults.max_total_frames,
        help="maximum cumulative HDF5 frames",
    )
    parser.add_argument(
        "--max-total-evidence-entries",
        type=int,
        default=defaults.max_total_evidence_entries,
        help="maximum cumulative logical evidence-index records",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        limits = BootstrapResourceLimits(
            max_entries=args.max_entries,
            max_total_artifacts=args.max_total_artifacts,
            max_total_declared_bytes=args.max_total_declared_bytes,
            max_total_frames=args.max_total_frames,
            max_total_evidence_entries=args.max_total_evidence_entries,
        )
        result = bootstrap_expert_trajectories(
            manifest_path=args.manifest,
            output_root=args.output_root,
            resource_limits=limits,
        )
    except (BootstrapError, ValueError) as exc:
        print(f"Phase A bootstrap failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
