"""Run RPent's unchanged Codex tool loop against one native RMBench worker."""
from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from rpent.utils.http_rpc import HttpRpcClient
from rpent.utils.logging import init_output_dir

from .planner import ChatGPTPlanner, account_home
from .prompts import SYSTEM, user_prompt
from .toolkit import RMBenchToolkit


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output_dir"])
    init_output_dir(output)
    memory = Path(config["memory_dir"])
    toolkit = RMBenchToolkit(HttpRpcClient(config["env_url"]), output_dir=output, memory_dir=memory,
                             bootstrap=config["phase"] == "bootstrap")
    user_message = user_prompt(memory_dir=memory, bootstrap=config["phase"] == "bootstrap",
                               task=config["task"], budget=config["max_turns"])
    (output / "planner_prompts.json").write_text(json.dumps(
        {"system": SYSTEM, "user": user_message}, indent=2, ensure_ascii=False))
    with account_home(config["auth_file"]) as account:
        planner = ChatGPTPlanner(account_home=account, output_dir=str(output), repo_root=output,
                                 model="gpt-5.5", timeout_s=config["wall_timeout_sec"],
                                 output_path=output / "codex.txt")
        result = planner.solve(system_prompt=SYSTEM,
                               user_message=user_message,
                               toolkit=toolkit, max_turns=config["max_turns"])
    payload = {name: getattr(result, name) for name in ("finish_result", "messages", "stats", "error")}
    if planner.budget_recorder is not None:
        payload["stats"].update(model_responses=planner.budget_recorder.model_responses,
                                model_response_budget=config["max_turns"],
                                budget_exhausted=planner.budget_recorder.budget_exhausted)
    payload["billing"] = {"source": "ChatGPT subscription", "api_price_estimate": None,
                          "note": "Token usage is measured; account quota has no per-run RMB invoice here."}
    (output / "planner_result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return int(bool(result.error))


if __name__ == "__main__":
    raise SystemExit(main())
