#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboharn_evo.benchmark_adapters.rmbench.formal_q1 import (  # noqa: E402
    current_git_commit,
    write_preparation,
)


DEFAULT_PROTOCOL = REPO_ROOT / "benchmarks/rmbench/experiments/hpk_formal_q1_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("/path/to/RMBench"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--code-commit",
        default="",
        help="defaults to the current full Git HEAD",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = args.code_commit.strip() or current_git_commit(REPO_ROOT)
    output = write_preparation(
        protocol_path=args.protocol,
        repo_root=REPO_ROOT,
        asset_root=args.asset_root,
        output_root=args.output_root,
        code_commit=commit,
    )
    print(output)


if __name__ == "__main__":
    main()
