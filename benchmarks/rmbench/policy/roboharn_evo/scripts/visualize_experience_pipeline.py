#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the RoboHarn-Evo Experience continuous-learning pipeline.")
    parser.add_argument("--experience-root", type=Path, default=Path("policy/roboharn_evo/skills/experience"))
    parser.add_argument("--output-dir", type=Path, default=Path("policy/roboharn_evo/skills/experience/retrieval-index/visualization"))
    parser.add_argument("--title", default="RoboHarn-Evo Experience Continuous-Learning Pipeline")
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def metadata_value(text: str, key: str) -> str:
    pattern = re.compile(rf"^\s*-\s*{re.escape(key)}:\s*`?([^`\n]+)`?\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip().lower() if match else ""


def lesson_status_counts(lesson_dir: Path) -> dict[str, int]:
    counts = {"candidate": 0, "accepted": 0, "deprecated": 0, "unknown": 0}
    if not lesson_dir.exists():
        return counts
    for path in lesson_dir.glob("*.md"):
        if "template" in path.name:
            continue
        status = metadata_value(path.read_text(encoding="utf-8"), "Status")
        if status in counts:
            counts[status] += 1
        else:
            counts["unknown"] += 1
    return counts


def collect_summary(root: Path) -> dict[str, Any]:
    raw_dir = root / "raw-traces"
    case_dir = root / "case-summaries"
    lesson_dir = root / "learned-lessons"
    index_dir = root / "retrieval-index"
    case_count = len([path for path in case_dir.glob("*.md") if "template" not in path.name]) if case_dir.exists() else 0
    return {
        "experience_root": str(root),
        "raw_traces": {
            "recovery_trials": count_jsonl(raw_dir / "recovery_trials.jsonl"),
            "ood_trials": count_jsonl(raw_dir / "ood_trials.jsonl"),
            "reentry_trials": count_jsonl(raw_dir / "reentry_trials.jsonl"),
        },
        "case_summaries": case_count,
        "lessons": lesson_status_counts(lesson_dir),
        "avoid_patterns": count_jsonl(index_dir / "avoid_patterns.jsonl"),
        "regression_payloads": count_jsonl(index_dir / "recovery_regression_payloads.jsonl"),
        "lesson_eval_results": count_jsonl(index_dir / "lesson_eval_results.jsonl"),
        "retrieval_index_records": count_jsonl(index_dir / "index.jsonl"),
    }


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def node(x: int, y: int, w: int, h: int, title: str, lines: list[str], *, fill: str) -> str:
    line_svg = []
    line_svg.append(f'<text x="{x + 18}" y="{y + 32}" class="title">{esc(title)}</text>')
    for index, line in enumerate(lines):
        line_svg.append(f'<text x="{x + 18}" y="{y + 62 + index * 22}" class="body">{esc(line)}</text>')
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="node" fill="{fill}"/>',
            *line_svg,
        ]
    )


def arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="arrow" marker-end="url(#arrowhead)"/>'


def render_svg(summary: dict[str, Any], title: str) -> str:
    raw = summary["raw_traces"]
    lessons = summary["lessons"]
    width = 1320
    height = 760
    nodes = [
        node(
            40,
            120,
            230,
            150,
            "Online Runtime",
            [
                "append factual traces",
                "no online promotion",
                "robot/VLA/recovery events",
            ],
            fill="#e8f1ff",
        ),
        node(
            330,
            90,
            250,
            210,
            "Raw Ledgers",
            [
                f"recovery_trials: {raw['recovery_trials']}",
                f"ood_trials: {raw['ood_trials']}",
                f"reentry_trials: {raw['reentry_trials']}",
                "source of truth",
            ],
            fill="#eef7ed",
        ),
        node(
            640,
            90,
            250,
            210,
            "Case Compression",
            [
                f"case summaries: {summary['case_summaries']}",
                "factual summaries",
                "no policy rule here",
            ],
            fill="#fff4df",
        ),
        node(
            950,
            70,
            290,
            250,
            "Lessons / Avoids",
            [
                f"candidate: {lessons['candidate']}",
                f"accepted: {lessons['accepted']}",
                f"deprecated: {lessons['deprecated']}",
                f"avoid patterns: {summary['avoid_patterns']}",
            ],
            fill="#f5edff",
        ),
        node(
            330,
            410,
            250,
            180,
            "Regression Suite",
            [
                f"payloads: {summary['regression_payloads']}",
                f"eval results: {summary['lesson_eval_results']}",
                "promotion gate",
            ],
            fill="#f0f4f8",
        ),
        node(
            640,
            410,
            250,
            180,
            "Retrieval Index",
            [
                f"records: {summary['retrieval_index_records']}",
                "accepted lessons",
                "verified avoid patterns",
            ],
            fill="#eaf7f6",
        ),
        node(
            950,
            410,
            290,
            180,
            "Runtime /recover",
            [
                "retrieved_experience",
                "planner chooses workflow/tools",
                "runtime validates calls",
            ],
            fill="#ffeef0",
        ),
    ]
    arrows = [
        arrow(270, 195, 330, 195),
        arrow(580, 195, 640, 195),
        arrow(890, 195, 950, 195),
        arrow(1070, 320, 455, 410),
        arrow(580, 500, 640, 500),
        arrow(890, 500, 950, 500),
        arrow(765, 410, 1095, 320),
    ]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#52606d"/>
  </marker>
  <style>
    .bg {{ fill: #ffffff; }}
    .heading {{ font: 700 28px Arial, sans-serif; fill: #1f2933; }}
    .subtitle {{ font: 15px Arial, sans-serif; fill: #52606d; }}
    .node {{ stroke: #bcccdc; stroke-width: 1.4; }}
    .title {{ font: 700 18px Arial, sans-serif; fill: #1f2933; }}
    .body {{ font: 14px Arial, sans-serif; fill: #334e68; }}
    .arrow {{ stroke: #52606d; stroke-width: 2.2; fill: none; }}
    .caption {{ font: 13px Arial, sans-serif; fill: #627d98; }}
  </style>
</defs>
<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>
<text x="40" y="48" class="heading">{esc(title)}</text>
<text x="40" y="76" class="subtitle">online facts -> offline lessons/evaluation -> accepted retrieval context for dynamic recovery</text>
{chr(10).join(arrows)}
{chr(10).join(nodes)}
<text x="40" y="700" class="caption">Experience root: {esc(summary['experience_root'])}</text>
<text x="40" y="724" class="caption">Note: Experience is advisory context only; workflow/tool/args/capability validation remains authoritative.</text>
</svg>
"""


def render_markdown(summary: dict[str, Any], svg_path: Path) -> str:
    raw = summary["raw_traces"]
    lessons = summary["lessons"]
    return f"""# Experience Pipeline Visualization

![Experience pipeline]({svg_path.name})

## Counts

- recovery_trials: `{raw['recovery_trials']}`
- ood_trials: `{raw['ood_trials']}`
- reentry_trials: `{raw['reentry_trials']}`
- case_summaries: `{summary['case_summaries']}`
- candidate_lessons: `{lessons['candidate']}`
- accepted_lessons: `{lessons['accepted']}`
- deprecated_lessons: `{lessons['deprecated']}`
- avoid_patterns: `{summary['avoid_patterns']}`
- regression_payloads: `{summary['regression_payloads']}`
- lesson_eval_results: `{summary['lesson_eval_results']}`
- retrieval_index_records: `{summary['retrieval_index_records']}`
"""


def main() -> None:
    args = parse_args()
    root = args.experience_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = collect_summary(root)
    svg_path = output_dir / "experience_pipeline.svg"
    summary_path = output_dir / "experience_pipeline_summary.json"
    markdown_path = output_dir / "experience_pipeline.md"
    svg_path.write_text(render_svg(summary, args.title), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary, svg_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "svg": str(svg_path),
                "summary": str(summary_path),
                "markdown": str(markdown_path),
                "counts": summary,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
