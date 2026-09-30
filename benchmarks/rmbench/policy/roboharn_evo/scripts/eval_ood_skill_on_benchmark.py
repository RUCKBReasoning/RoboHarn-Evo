#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import yaml

from policy.roboharn_evo.models.agent_api_ood_adapter import AgentApiOODAdapter, AgentApiOODConfig


DEFAULT_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "monitoring" / "ood-detection" / "SKILL.md"
_SKILL_FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<frontmatter>.*?)\n---\s*\n?(?P<body>.*)\Z", re.DOTALL)

DEFAULT_OOD_PROMPT = """You are executing the local RoboHarn-Evo skill `{skill_name}`.

Skill description:
{skill_description}

Skill instructions:
{skill_body}

Benchmark evaluation wrapper rules:
- Return JSON only.
- Do not output markdown or prose outside JSON.
- Do not use hidden benchmark annotation labels because they are not part of deployment-time input.

Skill payload: {{skill_payload}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ood-detection skill on benchmark samples using the /ood endpoint.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ood-url", type=str, required=True)
    parser.add_argument("--input-mode", type=str, default="key_frame", choices=["video", "key_frame", "frames"])
    parser.add_argument("--include-wrist", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--model-tag", type=str, default="ood-eval")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-request-payload", action="store_true")
    parser.add_argument("--sample-id", type=str)
    parser.add_argument("--key-frame", type=Path)
    parser.add_argument("--global-task", type=str)
    parser.add_argument("--current-subtask", type=str, default="")
    parser.add_argument("--step-count", type=int, default=0)
    parser.add_argument("--skill-path", type=Path, default=DEFAULT_SKILL_PATH)
    return parser.parse_args()


def load_skill_prompt(skill_path: Path) -> dict[str, str]:
    resolved_path = skill_path.expanduser().resolve()
    raw_text = resolved_path.read_text(encoding="utf-8")
    match = _SKILL_FRONTMATTER_RE.match(raw_text)
    if not match:
        raise ValueError(f"SKILL.md must start with YAML frontmatter: {resolved_path}")
    metadata = yaml.safe_load(match.group("frontmatter")) or {}
    name = str(metadata.get("name", "")).strip()
    description = str(metadata.get("description", "")).strip()
    body = match.group("body").strip()
    if name != "ood-detection":
        raise ValueError(f"Expected skill name 'ood-detection' in {resolved_path}, got {name!r}")
    if not description:
        raise ValueError(f"Skill description is required in {resolved_path}")
    if not body:
        raise ValueError(f"Skill body is required in {resolved_path}")
    return {
        "name": name,
        "description": description,
        "body": body,
        "path": str(resolved_path),
    }


def build_prompt_template(skill: dict[str, str]) -> str:
    return DEFAULT_OOD_PROMPT.format(
        skill_name=skill["name"],
        skill_description=skill["description"],
        skill_body=skill["body"],
    )


    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    raise ValueError(f"Unsupported image type for key frame: {path}")


def encode_image_media(path_str: str | None, *, role: str) -> dict[str, Any] | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    mime_type = guess_mime_type(path)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": "base64",
        "mime_type": mime_type,
        "data": encoded,
        "role": role,
    }


def build_visual_inputs(record: dict[str, Any], input_mode: str, include_wrist: bool) -> dict[str, Any]:
    normalized_mode = "key_frame" if input_mode == "video" else input_mode
    payload: dict[str, Any] = {
        "input_mode": normalized_mode,
        "video_path_agent": record.get("video_path_agent"),
        "key_frame_path": record.get("key_frame_path"),
    }
    if include_wrist:
        payload["video_path_wrist"] = record.get("video_path_wrist")
        payload["image_dir_wrist"] = record.get("image_dir_wrist")
    if normalized_mode == "frames":
        payload["image_dir_agent"] = record.get("image_dir_agent")
        if include_wrist:
            payload["image_dir_wrist"] = record.get("image_dir_wrist")
    return payload


def build_media(record: dict[str, Any], input_mode: str, include_wrist: bool) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    key_frame = encode_image_media(record.get("key_frame_path"), role="key_frame")
    if key_frame is not None:
        media.append(key_frame)
    if include_wrist:
        wrist_candidate = record.get("key_frame_path")
        if wrist_candidate:
            wrist_path = Path(str(wrist_candidate)).parent.parent / "wrist" / Path(str(wrist_candidate)).name
            wrist_media = encode_image_media(str(wrist_path), role="wrist_key_frame")
            if wrist_media is not None:
                media.append(wrist_media)
    return media


def build_skill_payload(record: dict[str, Any], input_mode: str, include_wrist: bool) -> dict[str, Any]:
    annotated_step = record.get("annotated_step")
    num_steps = record.get("num_steps")
    try:
        step_count = int(annotated_step) if annotated_step is not None else 0
    except Exception:
        step_count = 0
    try:
        step_limit = int(num_steps) if num_steps is not None else 0
    except Exception:
        step_limit = 0
    return {
        "selected_skill": "ood-detection",
        "benchmark_mode": True,
        "sample_id": record.get("sample_id"),
        "model_name": record.get("model_name"),
        "task_suite_name": record.get("task_suite_name"),
        "task_id": record.get("task_id"),
        "task_description": record.get("task_description"),
        "episode_idx": record.get("episode_idx"),
        "annotated_step": record.get("annotated_step"),
        "timestamp": record.get("timestamp"),
        "global_task": record.get("task_description", ""),
        "current_subtask": record.get("current_subtask", ""),
        "monitor_status": "rollout_failure",
        "recovery_state": {},
        "execution_context": {
            "benchmark_mode": True,
            "episode_idx": record.get("episode_idx"),
            "annotated_step": record.get("annotated_step"),
            "num_steps": record.get("num_steps"),
            "success": record.get("success"),
            "input_mode": input_mode,
        },
        "step_count": step_count,
        "step_limit": step_limit,
        "observation_summary": "",
        "visual_inputs": build_visual_inputs(record, input_mode=input_mode, include_wrist=include_wrist),
    }


def build_single_record(args: argparse.Namespace) -> dict[str, Any]:
    if args.key_frame is None:
        raise ValueError("--key-frame is required when no --manifest is provided")
    if not args.global_task:
        raise ValueError("--global-task is required when no --manifest is provided")
    return {
        "sample_id": args.sample_id or args.key_frame.stem,
        "model_name": None,
        "task_suite_name": None,
        "task_id": None,
        "task_description": args.global_task,
        "episode_idx": None,
        "annotated_step": args.step_count,
        "timestamp": None,
        "error_category": None,
        "target_error_category": None,
        "s_stage": None,
        "d_stage": None,
        "video_path_agent": None,
        "video_path_wrist": None,
        "image_dir_agent": str(args.key_frame.parent),
        "image_dir_wrist": None,
        "key_frame_path": str(args.key_frame.resolve()),
        "action_json_path": None,
        "ood_json_path": None,
        "success": None,
        "num_steps": args.step_limit,
        "initial_state": None,
        "has_wrist_view": False,
        "preferred_input_mode": "key_frame",
        "annotation_source": "manual_single_sample",
        "notes": "",
        "current_subtask": args.current_subtask,
    }


def coerce_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "OOD_scenario": response.get("OOD_scenario"),
        "reason": response.get("reason", ""),
        "confidence": response.get("confidence", 0.0),
    }


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    summary_path = args.summary.resolve()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
        records = load_manifest(manifest_path)
        if args.sample_id:
            records = [record for record in records if record.get("sample_id") == args.sample_id]
        if args.start_index > 0:
            records = records[args.start_index :]
        if args.limit > 0:
            records = records[: args.limit]
    else:
        manifest_path = None
        records = [build_single_record(args)]

    skill = load_skill_prompt(args.skill_path)
    prompt_template = build_prompt_template(skill)

    adapter = AgentApiOODAdapter(
        AgentApiOODConfig(
            server_url=args.ood_url,
            timeout_sec=args.timeout_sec,
            prompt_template=prompt_template,
            auth_token="",
            auth_header="Authorization",
            extra_headers={},
            extra_body={},
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0
    scenario_counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as f:
        for index, record in enumerate(records):
            skill_payload = build_skill_payload(record, input_mode=args.input_mode, include_wrist=args.include_wrist)
            media = build_media(record, input_mode=args.input_mode, include_wrist=args.include_wrist)
            started = time.time()
            status = "ok"
            error: str | None = None
            response_payload: dict[str, Any] = {}
            if not media:
                status = "error"
                error = "No usable key_frame media found for sample"
                failed += 1
            else:
                try:
                    response_payload = adapter.evaluate_ood(
                        skill_payload=json.dumps(skill_payload, ensure_ascii=False),
                        media=media,
                    )
                    response_payload = coerce_response(response_payload)
                    scenario = str(response_payload.get("OOD_scenario") or "")
                    if scenario:
                        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                except Exception as exc:
                    status = "error"
                    error = str(exc)
                    failed += 1
            latency_sec = round(time.time() - started, 4)
            result_record = {
                "sample_id": record.get("sample_id"),
                "manifest_index": (args.start_index + index) if args.manifest is not None else index,
                "model_tag": args.model_tag,
                "task_description": record.get("task_description"),
                "episode_idx": record.get("episode_idx"),
                "annotated_step": record.get("annotated_step"),
                "error_category": record.get("error_category"),
                "target_error_category": record.get("target_error_category"),
                "s_stage": record.get("s_stage"),
                "d_stage": record.get("d_stage"),
                "input_mode": "key_frame" if args.input_mode == "video" else args.input_mode,
                "video_path_agent": record.get("video_path_agent"),
                "video_path_wrist": record.get("video_path_wrist") if args.include_wrist else None,
                "key_frame_path": record.get("key_frame_path"),
                "skill": {
                    "name": skill["name"],
                    "path": skill["path"],
                    "description": skill["description"],
                },
                "request_payload": {
                    **(skill_payload if args.save_request_payload else {
                        "selected_skill": skill_payload.get("selected_skill"),
                        "global_task": skill_payload.get("global_task"),
                        "current_subtask": skill_payload.get("current_subtask"),
                        "monitor_status": skill_payload.get("monitor_status"),
                        "execution_context": skill_payload.get("execution_context"),
                        "step_count": skill_payload.get("step_count"),
                        "step_limit": skill_payload.get("step_limit"),
                        "visual_inputs": skill_payload.get("visual_inputs"),
                    }),
                    "media_count": len(media),
                    "media_roles": [item.get("role") for item in media],
                },
                "response": response_payload,
                "status": status,
                "latency_sec": latency_sec,
                "error": error,
            }
            f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            processed += 1

    summary = {
        "status": "ok",
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "output_path": str(output_path),
        "model_tag": args.model_tag,
        "input_mode": "key_frame" if args.input_mode == "video" else args.input_mode,
        "skill": {
            "name": skill["name"],
            "path": skill["path"],
            "description": skill["description"],
        },
        "total_samples": len(records),
        "processed_samples": processed,
        "failed_samples": failed,
        "scenario_counts": scenario_counts,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
