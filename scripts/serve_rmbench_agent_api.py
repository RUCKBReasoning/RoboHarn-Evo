from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


# Direct execution must not create ``__pycache__`` in this source checkout.
sys.dont_write_bytecode = True


if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))


def _split_provider_args(argv: Sequence[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--provider",
        choices=("dashscope", "openai"),
        default="dashscope",
        help="upstream provider (default: DashScope qwen3.6-plus)",
    )
    args, remaining = parser.parse_known_args(argv)
    return str(args.provider), remaining


def main(argv: Sequence[str] | None = None) -> int:
    provider, remaining = _split_provider_args(argv)
    if "-h" in remaining or "--help" in remaining:
        print(
            "RMBench Agent API provider selection:\n"
            "  --provider {dashscope,openai}\n"
            "    dashscope: Qwen/DashScope service (default)\n"
            "    openai: OpenAI-compatible or Codex-account gateway\n\n"
            "Pass --provider PROVIDER --help to inspect that backend.\n"
            "Selected provider options:"
        )
    if provider == "dashscope":
        from roboharn_evo.services.agent_api.qwen_planner import main as serve
    else:
        from roboharn_evo.services.agent_api.openai_planner import main as serve
    serve(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
