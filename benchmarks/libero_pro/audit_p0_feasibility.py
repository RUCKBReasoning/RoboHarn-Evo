"""Validate and summarize a LIBERO P0 read-only feasibility matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roboharn_evo.benchmark_adapters.libero_pro.p0_feasibility import (
    audit_p0_feasibility_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(
        audit_p0_feasibility_matrix(args.root),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
