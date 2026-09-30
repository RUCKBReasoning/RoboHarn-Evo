#!/usr/bin/env python3
"""One account/MCP/three-image probe; no simulator or physical action tools."""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpent-root", type=Path, default=Path("/path/to/HarnessVLA"))
    parser.add_argument("--auth-file", type=Path, default=Path("/path/to/provider/auth.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response-budget", type=int, default=8)
    parser.add_argument("--expect-budget-stop", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.rpent_root))
    from PIL import Image
    from rpent.tools.toolkit import Toolkit
    from rpent.utils.logging import init_output_dir
    from roboharn_evo.benchmark_adapters.rmbench.harness_vla.planner import ChatGPTPlanner, account_home
    from roboharn_evo.benchmark_adapters.rmbench.harness_vla.toolkit import spec
    init_output_dir(args.output)
    toolkit = Toolkit()
    # Only the observation probe and the stock finish tool are exposed.
    toolkit._tools = {"finish": toolkit._tools["finish"]}
    calls = []

    def observe():
        calls.append("observe_probe")
        result = {"image_order": ["view A", "view B", "view C"]}
        for rgb, key in zip([(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                            ["_image_bytes", "_image_cam_bytes", "_image_wrist_bytes"]):
            image = io.BytesIO()
            Image.new("RGB", (64, 64), rgb).save(image, format="PNG")
            result[key] = image.getvalue()
        return result

    toolkit.add_tool("observe_probe", spec("observe_probe", "Read three test camera images.", {}), observe)
    with account_home(args.auth_file) as account:
        planner = ChatGPTPlanner(account_home=account, output_dir=str(args.output), repo_root=args.output,
                                 model="gpt-5.5", timeout_s=300, output_path=args.output / "codex.txt")
        result = planner.solve(system_prompt="You are testing a robot observation interface. No robot exists in this test.",
                               user_message="Call observe_probe exactly once. Inspect all three images. Then call finish with status success and summary containing only the three dominant colors in image order. Do not guess without observing.",
                               toolkit=toolkit, max_turns=args.response_budget)
    report = {"model": "gpt-5.5", "reasoning_effort": "high", "auth": "ChatGPT account",
              "observation_calls": len(calls), "images_sent": len(calls) * 3,
              "finish": result.finish_result, "error": result.error, "stats": result.stats}
    colors = re.findall(r"\b(red|green|blue)\b", str((result.finish_result or {}).get("summary", "")).lower())
    report["passed"] = not result.error and len(calls) == 1 and colors == ["red", "green", "blue"]
    if planner.budget_recorder is not None:
        report["model_responses"] = planner.budget_recorder.model_responses
        report["budget_exhausted"] = planner.budget_recorder.budget_exhausted
        if args.expect_budget_stop:
            report["passed"] = (report["budget_exhausted"] and not result.error
                                and report["model_responses"] <= args.response_budget)
    (args.output / "probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
