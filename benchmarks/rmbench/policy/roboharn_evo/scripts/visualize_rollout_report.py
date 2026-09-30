from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any


IMPORTANT_EVENTS = {
    "episode_start",
    "episode_progress",
    "episode_interrupt",
    "episode_agent_terminal_failure",
    "episode_exception",
    "episode_end",
    "instruction_set",
    "observation_preprocess",
    "scene_memory_update",
    "control_turn_start",
    "control_turn_result",
    "control_decision",
    "pure_tool_control_control_turn_budget_exhausted",
    "vla_request",
    "vla_response",
    "monitor_signal",
    "recovery_router",
    "recovery_dispatch",
    "operation_candidate_selected",
    "recovery_result",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a human-readable HTML report for one RoboHarn-Evo/RMBench rollout.")
    parser.add_argument("--input", type=Path, required=True, help="Eval run dir, rollout dir, trace JSONL, or events JSONL.")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path. Defaults to <run>/rollout_report.html.")
    parser.add_argument("--max-events", type=int, default=300)
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--embed-images", action="store_true", help="Embed images as base64 data URLs instead of relative paths.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def find_run_dir(input_path: Path) -> Path:
    path = input_path.expanduser().resolve()
    if path.is_file():
        name = path.name
        if name in {"events.jsonl", "meta.json"}:
            return path.parent.parent if path.parent.name.startswith("episode_") and path.parent.name.endswith("_rollout") else path.parent
        return path.parent
    return path


def find_trace_files(run_dir: Path, input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_file() and path.suffix == ".jsonl":
        return [path]
    traces = sorted(run_dir.glob("episode_*_agent_trace.jsonl"))
    traces.extend(sorted(run_dir.glob("*agent_trace.jsonl")))
    rollout_events = sorted(run_dir.glob("episode_*_rollout/events.jsonl"))
    rollout_events.extend(sorted(run_dir.glob("events.jsonl")))
    return _dedupe_paths(traces + rollout_events)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def find_rollout_dirs(run_dir: Path, input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_dir() and path.name.startswith("episode_") and path.name.endswith("_rollout"):
        return [path]
    return sorted(run_dir.glob("episode_*_rollout"))


def read_meta(run_dir: Path, rollout_dirs: list[Path]) -> dict[str, Any]:
    for candidate in [*(rollout / "meta.json" for rollout in rollout_dirs), run_dir / "meta.json"]:
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
    return {}


def collect_videos(run_dir: Path, rollout_dirs: list[Path]) -> list[Path]:
    videos: list[Path] = []
    videos.extend(sorted(run_dir.glob("episode*.mp4")))
    for rollout in rollout_dirs:
        videos.extend(sorted((rollout / "video").glob("*.mp4")))
    return _dedupe_paths(videos)


def collect_images(run_dir: Path, rollout_dirs: list[Path], max_images: int) -> list[Path]:
    images: list[Path] = []
    for rollout in rollout_dirs:
        for camera in ("head", "left", "right"):
            camera_dir = rollout / camera
            if camera_dir.exists():
                images.extend(sorted(camera_dir.glob("step_*.png")))
    frames_dir = run_dir / "frames"
    if frames_dir.exists():
        images.extend(sorted(frames_dir.glob("*.png")))
    return images[:max_images]


def summarize_events(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        event = str(record.get("event", "unknown"))
        counts[event] = counts.get(event, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def event_step(record: dict[str, Any]) -> Any:
    return record.get("env_step", record.get("step", ""))


def compact_value(value: Any, max_chars: int = 260) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def event_summary(record: dict[str, Any]) -> str:
    event = str(record.get("event", ""))
    if event == "control_turn_result":
        return compact_value({key: record.get(key) for key in ("action_mode", "selected_skill", "subtask_text", "commit_label") if key in record})
    if event == "control_decision":
        return compact_value({key: record.get(key) for key in ("action_mode", "selected_skill", "subtask_text", "rendered_instruction") if key in record})
    if event == "observation_preprocess":
        return compact_value({"segmentation": record.get("segmentation", []), "scene_memory": record.get("scene_memory")}, 520)
    if event == "scene_memory_update":
        return compact_value(record.get("scene_memory", {}), 520)
    if event == "operation_candidate_selected":
        return compact_value(
            {
                key: record.get(key)
                for key in (
                    "candidate_id",
                    "candidate_geometry_revision",
                    "instance_id",
                    "target_id",
                    "arm",
                    "action_mode",
                    "source_camera",
                    "object_contact_pose",
                    "tcp_pose",
                    "ee_target_pose",
                    "approach_pose",
                )
                if key in record
            },
            520,
        )
    if event == "monitor_signal":
        return compact_value({key: record.get(key) for key in ("signal", "progress_score", "reason", "handoff_target") if key in record})
    if event in {"recovery_router", "recovery_result", "episode_end", "episode_progress", "episode_exception"}:
        skip = {"event", "timestamp"}
        return compact_value({key: value for key, value in record.items() if key not in skip}, 520)
    if event == "vla_request":
        return compact_value({key: record.get(key) for key in ("subtask", "robot_state") if key in record}, 420)
    if event == "vla_response":
        return compact_value({key: record.get(key) for key in ("latency_sec", "action_chunk_shape") if key in record})
    return compact_value({key: value for key, value in record.items() if key not in {"event", "timestamp"}}, 420)


def image_src(path: Path, output_path: Path, embed: bool) -> str:
    if embed:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    try:
        return path.resolve().relative_to(output_path.parent.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def video_src(path: Path, output_path: Path) -> str:
    try:
        return path.resolve().relative_to(output_path.parent.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_html(
    *,
    run_dir: Path,
    output_path: Path,
    trace_files: list[Path],
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    videos: list[Path],
    images: list[Path],
    embed_images: bool,
    max_events: int,
) -> str:
    counts = summarize_events(records)
    important = [record for record in records if str(record.get("event", "")) in IMPORTANT_EVENTS]
    if not important:
        important = records
    important = important[:max_events]
    title = f"RoboHarn-Evo Rollout Report - {run_dir.name}"
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2933; background: #f6f7f9; }
    header { background: #17212b; color: white; padding: 18px 24px; }
    main { padding: 18px 24px 32px; }
    section { background: white; border: 1px solid #d9dee5; border-radius: 8px; padding: 14px 16px; margin: 14px 0; }
    h1 { font-size: 22px; margin: 0 0 6px; }
    h2 { font-size: 16px; margin: 0 0 12px; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f2f4f7; padding: 10px; border-radius: 6px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #e4e8ee; padding: 7px 8px; vertical-align: top; text-align: left; }
    th { background: #f0f3f6; position: sticky; top: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
    .thumb { border: 1px solid #d9dee5; border-radius: 6px; overflow: hidden; background: #fbfcfd; }
    .thumb img { width: 100%; display: block; }
    .thumb div { padding: 6px 8px; font-size: 12px; color: #52606d; overflow-wrap: anywhere; }
    video { max-width: 100%; border-radius: 6px; background: #111; }
    .pill { display: inline-block; padding: 2px 7px; border-radius: 999px; background: #eef2f7; margin: 2px; font-size: 12px; }
    .event-observation_preprocess, .event-scene_memory_update { background: #f7fbff; }
    .event-monitor_signal { background: #fffaf0; }
    .event-recovery_router, .event-recovery_result { background: #fff5f5; }
    .event-control_turn_result, .event-control_decision { background: #f3fff8; }
    """
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{css}</style></head><body>",
        f"<header><h1>{html.escape(title)}</h1><div>{html.escape(str(run_dir))}</div></header><main>",
    ]
    html_parts.append("<section><h2>Episode</h2>")
    html_parts.append(f"<pre>{html.escape(json.dumps(meta, ensure_ascii=False, indent=2) if meta else '{}')}</pre>")
    html_parts.append("</section>")
    html_parts.append("<section><h2>Inputs</h2>")
    html_parts.append("<div>" + "".join(f"<span class='pill'>{html.escape(path.name)}</span>" for path in trace_files) + "</div>")
    html_parts.append("</section>")
    if videos:
        html_parts.append("<section><h2>Videos</h2><div class='grid'>")
        for video in videos:
            src = html.escape(video_src(video, output_path))
            html_parts.append(f"<div><video controls src='{src}'></video><div><code>{html.escape(str(video))}</code></div></div>")
        html_parts.append("</div></section>")
    html_parts.append("<section><h2>Event Counts</h2><div>")
    html_parts.append("".join(f"<span class='pill'>{html.escape(key)}: {value}</span>" for key, value in counts.items()))
    html_parts.append("</div></section>")
    html_parts.append("<section><h2>Timeline</h2><table><thead><tr><th>#</th><th>step</th><th>event</th><th>summary</th></tr></thead><tbody>")
    for idx, record in enumerate(important):
        event = str(record.get("event", "unknown"))
        html_parts.append(
            f"<tr class='event-{html.escape(event)}'><td>{idx}</td><td>{html.escape(str(event_step(record)))}</td>"
            f"<td><code>{html.escape(event)}</code></td><td>{html.escape(event_summary(record))}</td></tr>"
        )
    html_parts.append("</tbody></table></section>")
    if images:
        html_parts.append("<section><h2>Saved Frames</h2><div class='grid'>")
        for image in images:
            src = html.escape(image_src(image, output_path, embed_images))
            html_parts.append(f"<div class='thumb'><img src='{src}'><div><code>{html.escape(str(image))}</code></div></div>")
        html_parts.append("</div></section>")
    html_parts.append("</main></body></html>")
    return "\n".join(html_parts)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    run_dir = find_run_dir(input_path)
    rollout_dirs = find_rollout_dirs(run_dir, input_path)
    trace_files = find_trace_files(run_dir, input_path)
    records: list[dict[str, Any]] = []
    for path in trace_files:
        records.extend(load_jsonl(path))
    output_path = args.output.expanduser().resolve() if args.output else run_dir / "rollout_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_html(
        run_dir=run_dir,
        output_path=output_path,
        trace_files=trace_files,
        records=records,
        meta=read_meta(run_dir, rollout_dirs),
        videos=collect_videos(run_dir, rollout_dirs),
        images=collect_images(run_dir, rollout_dirs, max_images=args.max_images),
        embed_images=bool(args.embed_images),
        max_events=max(1, int(args.max_events)),
    )
    output_path.write_text(html_text, encoding="utf-8")
    print(json.dumps({"report": str(output_path), "events": len(records), "trace_files": [str(path) for path in trace_files]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
