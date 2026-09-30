#!/usr/bin/env python3
"""Validate or probe services referenced by RMBench's deploy policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


# This script is commonly launched from the checkout during preflight.  Keep
# that read-only validation from producing bytecode artifacts in source trees.
sys.dont_write_bytecode = True


if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.rmbench.paths import default_deploy_config  # noqa: E402
from roboharn_evo.services.preflight import (  # noqa: E402
    DEFAULT_PI_OPENPI_REPOSITORY,
    probe_rmbench_services,
    static_service_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate RMBench service URLs or probe their /health endpoints."
    )
    parser.add_argument("--config", type=Path, default=default_deploy_config())
    parser.add_argument(
        "--probe",
        action="store_true",
        help="perform GET /health requests; without this flag, validation is network-free",
    )
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    parser.add_argument(
        "--segmentation-artifact-dir",
        type=Path,
        default=None,
        help=(
            "AgentTools/SAM3 shared artifact directory; defaults to "
            "ROBOHARN_EVO_SEGMENTATION_ARTIFACT_DIR or the canonical RoboHarn-Evo eval_result path"
        ),
    )
    parser.add_argument(
        "--pi-openpi-repo",
        type=Path,
        default=DEFAULT_PI_OPENPI_REPOSITORY,
        help="Expected RoboHarn-Evo code-only OpenPI source directory.",
    )
    parser.add_argument(
        "--pi-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Expected external read-only PI0.5 checkpoint directory; when set, "
            "the health identity must match it exactly"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    identity_arguments = {
        "segmentation_artifact_dir": args.segmentation_artifact_dir,
        "pi_openpi_repo": args.pi_openpi_repo,
        "pi_checkpoint_dir": args.pi_checkpoint_dir,
    }
    payload = (
        probe_rmbench_services(
            args.config,
            timeout_sec=args.timeout_sec,
            **identity_arguments,
        )
        if args.probe
        else static_service_report(args.config, **identity_arguments)
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("valid", payload.get("healthy", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
