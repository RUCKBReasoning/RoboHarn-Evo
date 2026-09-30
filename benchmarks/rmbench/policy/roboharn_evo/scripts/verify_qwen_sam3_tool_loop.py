from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image

from policy.roboharn_evo.agent.components.agent_tools.agent_tools import AgentTools, build_tool_from_func
from policy.roboharn_evo.agent.environment import EnvSnapshot
from policy.roboharn_evo.agent.paths import eval_result_dir
from policy.roboharn_evo.agent.perception.sam3_client import SAM3SegmentationClient


SYSTEM_PROMPT = """You are a robot manipulation planner with tool access.
Your job in this verification run is to prove that you can use the segmentation tool before deciding.

Rules:
- First call AgentTools___segment_object on the target object.
- After the tool result returns, read its bbox_xyxy, centroid_px, mask_path, score, env_step, and robot_state.
- Then call final_decision exactly once.
- Do not answer with free-form text outside tool calls.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Qwen tool-calling can drive SAM3 through AgentTools___segment_object.")
    parser.add_argument("--image-path", required=True, help="Current head-camera image to segment.")
    parser.add_argument("--task", default="press the button three times")
    parser.add_argument("--object-id", default="button")
    parser.add_argument("--text-prompt", default="", help="SAM3 text prompt. Defaults to object-id.")
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--sam3-service-url", default=os.getenv("ROBOHARN_EVO_SAM3_SERVICE_URL", "http://127.0.0.1:9301"))
    parser.add_argument("--output-dir", default=str(eval_result_dir("tcm_qwen_sam3_tool_loop")))
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def load_rgb_array(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def image_to_data_url(image: np.ndarray) -> str:
    from io import BytesIO

    buffer = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def build_snapshot(image: np.ndarray, *, task: str) -> EnvSnapshot:
    joint_vector = np.zeros(14, dtype=np.float32)
    left_endpose = np.zeros(7, dtype=np.float32)
    right_endpose = np.zeros(7, dtype=np.float32)
    left_endpose[3] = 1.0
    right_endpose[3] = 1.0
    return EnvSnapshot(
        raw={
            "endpose": {"left_gripper": 1.0, "right_gripper": 1.0},
            "joint_action": {"vector": joint_vector.tolist()},
        },
        head_rgb=image,
        left_rgb=image,
        right_rgb=image,
        joint_vector=joint_vector,
        left_endpose=left_endpose,
        right_endpose=right_endpose,
        step_count=0,
        step_limit=0,
        eval_success=False,
        check_success=False,
        max_reward=0.0,
        instruction=task,
    )


class ToolLoopHarness:
    def __init__(self, *, image: np.ndarray, task: str, sam3_service_url: str, artifact_dir: Path) -> None:
        self.latest_snapshot = build_snapshot(image, task=task)
        self.artifact_dir = artifact_dir
        self._agent_tools = AgentTools(memory_manager=_DummyMemoryManager(), agent_card=None, agent_instance=self)
        self._sam3_service_url = sam3_service_url

    def _get_env_summary(self) -> str:
        return f"verification snapshot: task={self.latest_snapshot.instruction}, step={self.latest_snapshot.step_count}"

    def _get_robot_state(self) -> dict[str, Any]:
        return {
            "left": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0], "gripper": 1.0},
            "right": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0], "gripper": 1.0},
        }

    def segment_object(self, args: dict[str, Any]) -> str:
        args = dict(args)
        args.setdefault("backend", "sam3")
        args.setdefault("camera", "head")
        args.setdefault("service_url", self._sam3_service_url)
        if "confidence_threshold" in args:
            threshold = args.pop("confidence_threshold")
            if threshold is not None:
                os.environ["ROBOHARN_EVO_SAM3_CONFIDENCE_THRESHOLD"] = str(threshold)
        return _run_async(self._agent_tools.segment_object(**args))


class _DummyMemoryManager:
    def __init__(self) -> None:
        self.runtime_tree = type("_RuntimeTree", (), {"current_task_node": None})()

    def add_robot_image(self, *, image_data_url: str, frame_id: int) -> None:
        return

    def compress_current_memory(self, drop_n: int = 1) -> None:
        return


def _run_async(coro) -> str:
    import asyncio

    return str(asyncio.run(coro))


def final_decision_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "final_decision",
            "description": "Finalize the verification after reading the SAM3 tool observation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "used_segmentation": {"type": "boolean"},
                    "object_id": {"type": "string"},
                    "bbox_xyxy": {"type": "array", "items": {"type": "number"}},
                    "centroid_px": {"type": "array", "items": {"type": "number"}},
                    "mask_path": {"type": "string"},
                    "score": {"type": "number"},
                    "next_subtask": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["used_segmentation", "object_id", "next_subtask", "reason"],
            },
        },
    }


def compact_tool_result(result_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(result_text)
    except json.JSONDecodeError:
        return {"raw": result_text[:1000]}
    if not isinstance(payload, dict):
        return {"raw": result_text[:1000]}
    keep = {
        "success",
        "object_id",
        "text_prompt",
        "num_detections",
        "mask_path",
        "bbox_xyxy",
        "centroid_px",
        "area_px",
        "score",
        "backend",
        "camera",
        "env_step",
        "robot_state",
        "error",
    }
    compact = {key: payload.get(key) for key in keep if key in payload}
    detections = payload.get("detections")
    if isinstance(detections, list):
        compact["detections"] = detections[:3]
    return compact


def assert_sam3_health(base_url: str) -> dict[str, Any]:
    try:
        health = SAM3SegmentationClient(base_url=base_url, timeout_sec=10).health()
    except Exception as exc:
        raise RuntimeError(
            "SAM3 service is not reachable. Start it in a separate terminal first, "
            f"or pass the correct --sam3-service-url. url={base_url} error={exc}"
        ) from exc
    if health.get("status") != "ok":
        raise RuntimeError(f"SAM3 health check failed: {health}")
    return health


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb_array(args.image_path)

    sam3_health = assert_sam3_health(args.sam3_service_url)
    harness = ToolLoopHarness(
        image=image,
        task=args.task,
        sam3_service_url=args.sam3_service_url,
        artifact_dir=output_dir,
    )

    segment_schema = build_tool_from_func(AgentTools.segment_object, service_name="AgentTools").openai_format
    tools = [segment_schema, final_decision_schema()]

    prompt_object = args.text_prompt.strip() or args.object_id
    user_text = (
        f"Task: {args.task}\n"
        f"Target object_id: {args.object_id}\n"
        f"Preferred SAM3 text_prompt: {prompt_object}\n"
        "Use the segmentation tool first, then finalize."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
            ],
        },
    ]

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout_sec)
    transcript: list[dict[str, Any]] = [
        {"event": "sam3_health", "payload": sam3_health},
        {"event": "request", "task": args.task, "object_id": args.object_id, "text_prompt": prompt_object},
    ]
    final_payload: dict[str, Any] | None = None

    for round_idx in range(args.max_rounds):
        extra_body = {"enable_thinking": True} if args.enable_thinking else None
        completion_kwargs: dict[str, Any] = {
            "model": args.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if extra_body is not None:
            completion_kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**completion_kwargs)
        assistant_msg = response.choices[0].message
        assistant_dump = assistant_msg.model_dump(exclude_none=True)
        messages.append(assistant_dump)
        transcript.append({"event": "assistant", "round": round_idx, "message": assistant_dump})

        if not assistant_msg.tool_calls:
            break

        for tool_call in assistant_msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}
            if tool_name == "AgentTools___segment_object":
                tool_args.setdefault("object_id", args.object_id)
                tool_args.setdefault("text_prompt", prompt_object)
                if args.confidence_threshold is not None:
                    tool_args["confidence_threshold"] = args.confidence_threshold
                result_text = harness.segment_object(tool_args)
                transcript.append(
                    {
                        "event": "tool_result",
                        "tool": tool_name,
                        "args": tool_args,
                        "result": compact_tool_result(result_text),
                    }
                )
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})
                continue
            if tool_name == "final_decision":
                final_payload = tool_args
                transcript.append({"event": "final_decision", "payload": final_payload})
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"status": "ok"}, ensure_ascii=False)})
                break
            result_text = json.dumps({"error": f"unknown tool: {tool_name}"}, ensure_ascii=False)
            transcript.append({"event": "tool_result", "tool": tool_name, "args": tool_args, "result": {"error": result_text}})
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})
        if final_payload is not None:
            break

    transcript_path = output_dir / "qwen_sam3_tool_loop_transcript.json"
    with transcript_path.open("w", encoding="utf-8") as handle:
        json.dump(transcript, handle, ensure_ascii=False, indent=2)

    if final_payload is None:
        raise RuntimeError(f"Qwen did not call final_decision. Transcript: {transcript_path}")
    if not bool(final_payload.get("used_segmentation")):
        raise RuntimeError(f"Qwen final_decision did not acknowledge segmentation. Transcript: {transcript_path}")

    print(json.dumps({"success": True, "transcript_path": str(transcript_path), "final_decision": final_payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
