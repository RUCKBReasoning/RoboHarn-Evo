from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboharn_evo.benchmark_adapters.rmbench.formal_runner import run_cell  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one preregistered RMBench HPK Q1 episode without retry.")
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "benchmarks/rmbench/experiments/hpk_formal_q1_v1.yaml",
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=(
            REPO_ROOT
            / "benchmarks/rmbench/policy/roboharn_evo/scripts/run_gpt55_pure_tool_control_8way.sh"
        ),
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("/path/to/RMBench/assets"),
    )
    parser.add_argument("--agent-api-base-url", required=True)
    parser.add_argument("--sam3-service-url", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--expected-agent-concurrency", type=int, default=1)
    parser.add_argument("--agent-service-log", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cell(
        protocol_path=args.protocol,
        preparation_root=args.preparation_root,
        source_root=args.source_root,
        experiment_root=args.experiment_root,
        cell_id=args.cell_id,
        launcher=args.launcher,
        assets_root=args.assets_root,
        agent_api_base_url=args.agent_api_base_url,
        sam3_service_url=args.sam3_service_url,
        gpu=args.gpu,
        expected_agent_concurrency=args.expected_agent_concurrency,
        agent_service_log=args.agent_service_log,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] == "infrastructure_invalid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
