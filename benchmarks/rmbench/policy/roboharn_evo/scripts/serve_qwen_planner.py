from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only in minimal test envs
    OpenAI = None  # type: ignore[assignment]

from roboharn_evo.agent.arm_contract import normalize_physical_arm, normalize_preferred_arm, normalize_selected_arm
from roboharn_evo.agent.experience import normalize_semantic_tags
from roboharn_evo.agent.paths import roboharn_skills_dir
from roboharn_evo.agent.perception.query_normalization import (
    matching_perception_query_index,
    merge_instance_hints,
    merge_perception_query,
    normalized_relation_metadata,
    normalize_perception_object_id,
    normalize_query_role,
)


SYSTEM_PROMPT = """You are the planner VLM in a long-horizon robot manipulation system.
You receive:
- the global task instruction
- the previous committed memory
- two images: segment start frame and segment end frame
- a numeric state summary vector

Return JSON only with this schema:
{
  \"commit_label\": \"no_update | subtask_complete | state_change\",
  \"memory_text\": \"one concise sentence describing committed task state after this segment\",
  \"selected_skill\": \"workflow skill name to execute next; for normal executor-facing steps use monitored-subtask-execution\",
  \"subtask_text\": \"the next concrete executor-facing instruction\",
  \"preferred_arm\": \"left | right | either\",
  \"semantic_tags\": {
    \"task_family\": \"pick_and_place | open_drawer | close_drawer | articulated_object | tool_use | other\",
    \"subtask_type\": \"grasp | place | open | close | move | align | reobserve | recover | other\",
    \"state_tags\": {
      \"object_state\": \"\",
      \"visibility_state\": \"\",
      \"gripper_state\": \"\",
      \"motion_state\": \"\"
    }
  }
}

Rules:
- Use exactly one commit label from: no_update, subtask_complete, state_change.
- If there is no task-relevant state update, set commit_label to no_update and keep memory consistent.
- memory_text must describe current committed state, not a future plan.
- selected_skill must be a workflow-defined skill name. For ordinary single-step execution, default to monitored-subtask-execution.
- subtask_text must be the concrete instruction passed into that skill.
- Select preferred_arm from current visual geometry and robot state. Use either only when neither arm has a meaningful advantage. Do not derive it from wording conventions in subtask_text.
- semantic_tags is optional. If present, use only the exact taxonomy values shown above; leave uncertain state tag values as empty strings.
- Do not output explanations, markdown, or extra keys.
"""

OOD_SYSTEM_PROMPT = """You are the OOD detector VLM in a robot manipulation system.
You receive a structured payload describing the current rollout state.

Return JSON only with this schema:
{
  \"OOD_scenario\": \"none | object_not_visible | motion_blocked | grasp_lost | scene_drift_detected | requires_replan\",
  \"reason\": \"short evidence-based explanation\",
  \"confidence\": 0.0,
  \"semantic_tags\": {
    \"task_family\": \"pick_and_place | open_drawer | close_drawer | articulated_object | tool_use | other\",
    \"subtask_type\": \"grasp | place | open | close | move | align | reobserve | recover | other\",
    \"state_tags\": {
      \"object_state\": \"\",
      \"visibility_state\": \"visible | partially_visible | object_not_visible | occluded | out_of_view | visible_object | target_visible | other\",
      \"gripper_state\": \"open | closed | partially_closed | holding_object | empty | slipping | other\",
      \"motion_state\": \"moving | stable | stalled | blocked | no_progress | overshot | other\"
    }
  }
}

Rules:
- You must directly output the OOD classification in `OOD_scenario`.
- Do not output only a summary for the runtime to classify later.
- If there is insufficient evidence for an OOD diagnosis, return `OOD_scenario` as `none`.
- semantic_tags is optional. Prefer image-grounded state_tags when visible; use empty strings for uncertain state tags.
- Do not output markdown, prose outside JSON, or extra keys.
"""


ALLOWED_COMMIT_LABELS = {"no_update", "subtask_complete", "state_change"}
ALLOWED_ACTION_MODES = {"start", "continue", "retry", "reset", "replan", "recover", "switch", "finish"}
ALLOWED_OOD_SCENARIOS = {"none", "object_not_visible", "motion_blocked", "grasp_lost", "scene_drift_detected", "requires_replan"}
ALLOWED_RECOVERY_INTENTS = {"retry", "replan", "abort"}
ALLOWED_ACTION_EFFECT_VALUES = {"true", "false", "unverified"}
ALLOWED_SUBTASK_STATUS_VALUES = {"in_progress", "completed", "failed", "uncertain"}
ALLOWED_VERIFIER_CONTROL_VALUES = {"continue", "retry", "replan"}
ARM_BOUND_RECOVERY_TOOLS = {
    "close_gripper",
    "contact_displace",
    "lift_ee",
    "move_ee_to_grounded_instance",
    "move_ee_to_pose",
    "open_gripper",
}

RECOVERY_SYSTEM_PROMPT = """You are the recovery planner in a robot manipulation system.
You receive a structured recovery payload containing:
- monitor/OOD signal and reason
- current subtask and observation summary
- recovery-router SKILL harness
- available recovery workflow SKILL harnesses
- recovery primitive/tool guides
- runtime allowed tool names

Return JSON only with this schema:
{
  "recovery_workflow": "one workflow name from available workflows",
  "selected_arm": "left | right | none",
  "tool_calls": [
    {"tool_name": "runtime tool name", "args": {}, "reason": "short evidence-based reason"}
  ],
  "post_recovery_intent": "retry | replan | abort",
  "reason": "short explanation for selected workflow and plan",
  "stop_condition": "condition for stopping recovery"
}

Rules:
- When recovery_state.planner_context_mode=compact_v1, scene_memory is the single planner_scene_view/compact_v1 projection; observation_summary and observation_preprocess may be empty by design. Do not reconstruct duplicate scene copies.
- Do not output markdown, prose outside JSON, or extra keys.
- Use only tool names supplied in available_tools.
- Generate a situation-specific plan; reference plans in SKILL text are examples, not mandatory fixed sequences.
- Use grounded end-effector tools only when scene_memory.instances contains finite world_m or approach_world_m evidence for the selected instance; do not invent coordinates.
- Prefer move_ee_to_grounded_instance with an existing instance_id, role, or focus_key over copying coordinates into tool args.
- Use selected_arm for every task-progress tool call and make its args.arm match. Set selected_arm=none for re-observation or clearance-only plans. Respect recovery_payload.preferred_arm; clearance of a previously used arm may still use that other arm.
- Treat recovery_payload.blocked_grounded_setups as runtime evidence. Do not retry a legacy identical blocked setup or a scope with all_candidates_blocked=true. If runtime_will_select_next=true, the same public instance/arm/action_mode may be retried and runtime will select another private pose candidate; never emit a candidate ID.
- EE recovery tools may use args.steps for bounded multi-step control. Keep steps small and purposeful; runtime caps unsafe values.
- It is allowed to return an empty tool_calls list when replanning is the right recovery action.
- `post_recovery_intent` is a pre-action proposal only. The after-action verifier will decide the actual subtask transition from fresh evidence, so do not use this field to claim completion.
"""

PERCEPTION_QUERY_SYSTEM_PROMPT = """You are the perception-query and instance-binding planner for a robot manipulation agent.
Given the global task, current subtask, committed memory, observation summary, robot state, optional current image, optional scene_instances, and optional oracle_objects, choose the task-relevant objects and bind target/tool queries to stable candidate identities when candidates are available.

Return JSON only with this schema:
{
  "queries": [
    {
      "object_id": "short stable object name",
      "text_prompt": "visual text prompt for segmentation",
      "role": "target | tool | context",
      "entity_scope": "single_instance | reference_set",
      "placement_relation": "center_of, only for a reference_set that defines the requested placement",
      "expected_count": "optional visible/semantic member count, or omit",
      "instance_hint": "optional disambiguation such as left/right/middle/color/ordinal, or empty string",
      "instance_ref": "exact instance_id or track_id from scene_instances, or empty string",
      "oracle_id": "exact oracle_id from oracle_objects when applicable, or empty string",
      "reason": "short reason"
    }
  ]
}

Rules:
- Return at most 3 queries.
- Infer task roles from the instruction and memory. Python will only transport your role fields; it will not infer target/tool semantics.
- When binding_requirement.required_any_roles is non-empty, return at least one semantically supported query whose role is in that list. A context-only response does not satisfy a target/tool requirement.
- When binding_requirement.force_refresh=true, treat previous_queries as rejected output, use failure_reason as correction feedback, and produce a fresh query set rather than repeating the rejected roles.
- Do not relabel a reference object merely to satisfy the role contract. If no supported required-role query can be selected confidently, return an empty query list so the runtime can perform a bounded retry.
- Use role=target for the object or part whose state matters for task success.
- Use role=tool for an object that should be manipulated mainly to affect a target object.
- Use role=context for supporting references, obstacles, or uncertain but potentially relevant objects.
- When the current subtask explicitly asks to place an object at the center of, or between the members of, a visual reference set, emit one context query for that set with entity_scope=reference_set and placement_relation=center_of. Set expected_count only when the instruction states the count. This is a semantic relation request; do not calculate or invent coordinates.
- For ordinary independently manipulable objects use entity_scope=single_instance and omit placement_relation. For a reference_set, make text_prompt a short member-level segmentation prompt (for example, the singular category of one member); entity_scope and expected_count request all matching members. Do not phrase text_prompt as a plural scene description or bind one arbitrary member with instance_ref.
- Prefer concrete visible manipulable objects or object parts.
- Use short English text prompts suitable for image segmentation.
- Treat segmentation and simulator catalogs as candidate generation only. You are the authority for semantic instance selection; Python only validates exact identifiers and grounded evidence.
- Respect instance_binding_phase. During candidate_discovery, scene_instances is a partial catalog: preserve a task-supported target/tool query for a required object that has no matching candidate yet, leave instance_ref empty, and let segmentation create candidates. Never bind it to an unrelated existing instance merely because the catalog is non-empty.
- During catalog_selection or post_detection_selection, choose an exact advertised identity for every target/tool query when require_instance_binding=true; return no such query if the intended identity is still ambiguous.
- If scene_instances contains the intended target/tool, set instance_ref to one exact advertised track_id (preferred) or instance_id. Never invent or rewrite an identifier.
- If oracle_objects contains the intended target/tool, set oracle_id to its exact advertised oracle_id. When no scene instance exists yet, instance_ref may also be that exact oracle_id.
- When require_instance_binding=true, every target/tool query must contain an exact candidate reference. This strict rule applies to catalog_selection and post_detection_selection, not candidate_discovery. If the evidence is ambiguous, return no target/tool query instead of guessing.
- Use geometry, images, task state, and memory to interpret relational language. Do not assume list order identifies an instance.
- Do not invent objects outside the supplied catalogs unless the task requires a missing object to be segmented.
- Do not include background, robot arm, table, or vague words unless they are the task target.
- If no object should be segmented, return {"queries": []}.
- Do not output markdown, prose outside JSON, or extra keys.
"""

PERCEPTION_QUERY_NORMALIZATION_SKILL_PATH = (
    roboharn_skills_dir() / "perception" / "perception-query-normalization" / "SKILL.md"
)


def load_perception_query_normalization_prompt() -> str:
    try:
        return Path(PERCEPTION_QUERY_NORMALIZATION_SKILL_PATH).read_text(encoding="utf-8")
    except OSError:
        return """You normalize robot perception queries. Return JSON only with {"queries": [...]}. Do not output extra keys."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve DashScope Qwen planner VLM over HTTP for RoboHarn-Evo deployment.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--model", type=str, default="qwen3.6-plus")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def build_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed; install it before starting the Qwen planner service")
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        payload = json.loads(stripped[first : last + 1])
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Failed to parse planner JSON output: {text}")


def normalize_prediction(payload: dict[str, Any], previous_memory_text: str) -> dict[str, Any]:
    commit_label = str(payload.get("commit_label", "")).strip()
    if commit_label not in ALLOWED_COMMIT_LABELS:
        commit_label = "no_update"

    memory_text = str(payload.get("memory_text", previous_memory_text)).strip()
    selected_skill = str(payload.get("selected_skill", "")).strip() or "monitored-subtask-execution"
    subtask_text = str(payload.get("subtask_text", "")).strip()

    if not memory_text:
        memory_text = previous_memory_text
    if not subtask_text:
        raise ValueError(f"Planner output missing subtask_text: {payload}")

    result = {
        "commit_label": commit_label,
        "memory_text": memory_text,
        "selected_skill": selected_skill,
        "subtask_text": subtask_text,
        "preferred_arm": normalize_preferred_arm(payload.get("preferred_arm")),
    }
    action_mode = str(payload.get("action_mode", "")).strip().lower()
    if action_mode in ALLOWED_ACTION_MODES:
        result["action_mode"] = action_mode
    if isinstance(payload.get("semantic_tags"), dict):
        result["semantic_tags"] = normalize_semantic_tags(payload.get("semantic_tags"), default_source="planner_vlm")
    return result


def normalize_ood_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    scenario = str(payload.get("OOD_scenario", "none")).strip().lower().replace("-", "_").replace(" ", "_")
    if scenario not in ALLOWED_OOD_SCENARIOS:
        scenario = "none"
    reason = str(payload.get("reason", "")).strip()
    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    result = {
        "OOD_scenario": scenario,
        "reason": reason,
        "confidence": confidence,
    }
    if isinstance(payload.get("semantic_tags"), dict):
        result["semantic_tags"] = normalize_semantic_tags(payload.get("semantic_tags"), default_source="ood_vlm")
    return result


def normalize_recovery_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    workflow = str(payload.get("recovery_workflow", "")).strip()
    intent = str(payload.get("post_recovery_intent", "replan")).strip().lower().replace("same_subtask", "").strip("_")
    if intent not in ALLOWED_RECOVERY_INTENTS:
        intent = "replan"
    tool_calls_payload = payload.get("tool_calls", [])
    if tool_calls_payload is None:
        tool_calls_payload = []
    if not isinstance(tool_calls_payload, list):
        raise ValueError(f"Recovery output tool_calls must be a list: {payload}")
    tool_calls: list[dict[str, Any]] = []
    for item in tool_calls_payload:
        if not isinstance(item, dict):
            raise ValueError(f"Recovery tool call must be an object: {item!r}")
        tool_name = str(item.get("tool_name", "")).strip()
        if not tool_name:
            raise ValueError(f"Recovery tool call missing tool_name: {item!r}")
        args = item.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ValueError(f"Recovery tool call args must be an object: {item!r}")
        tool_calls.append({
            "tool_name": tool_name,
            "args": args,
            "reason": str(item.get("reason", "")).strip(),
        })
    selected_arm = normalize_selected_arm(payload.get("selected_arm"))
    if "selected_arm" not in payload:
        progress_arms = {
            normalize_physical_arm((item.get("args") or {}).get("arm"))
            for item in tool_calls
            if item.get("tool_name") in ARM_BOUND_RECOVERY_TOOLS
        }
        progress_arms.discard("")
        if len(progress_arms) == 1:
            selected_arm = next(iter(progress_arms))
    return {
        "recovery_workflow": workflow,
        "selected_arm": selected_arm,
        "tool_calls": tool_calls,
        "post_recovery_intent": intent,
        "reason": str(payload.get("reason", "")).strip(),
        "stop_condition": str(payload.get("stop_condition", "")).strip(),
    }


def normalize_action_effect_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    effect = str(payload.get("effect_verified", payload.get("effect", "unverified"))).strip().lower()
    if effect not in ALLOWED_ACTION_EFFECT_VALUES:
        effect = "unverified"
    subtask_status = str(payload.get("subtask_status", "uncertain")).strip().lower().replace("-", "_").replace(" ", "_")
    if subtask_status not in ALLOWED_SUBTASK_STATUS_VALUES:
        subtask_status = "uncertain"
    recommended_control = str(payload.get("recommended_control", "retry")).strip().lower().replace("-", "_").replace(" ", "_")
    if recommended_control not in ALLOWED_VERIFIER_CONTROL_VALUES:
        recommended_control = "retry"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "effect_verified": effect,
        "effect_type": " ".join(str(payload.get("effect_type", "unknown") or "unknown").strip().split())[:120],
        "confidence": confidence,
        "evidence_summary": " ".join(str(payload.get("evidence_summary", "") or "").strip().split())[:240],
        "failure_reason": " ".join(str(payload.get("failure_reason", "") or "").strip().split())[:240],
        "next_constraint": " ".join(str(payload.get("next_constraint", "") or "").strip().split())[:240],
        "memory_update": " ".join(str(payload.get("memory_update", "") or "").strip().split())[:240],
        "subtask_status": subtask_status,
        "recommended_control": recommended_control,
    }


def validate_perception_queries(payload: dict[str, Any], *, default_max_queries: int = 3) -> dict[str, Any]:
    raw_queries = payload.get("queries", [])
    if raw_queries is None:
        raw_queries = []
    if not isinstance(raw_queries, list):
        raise ValueError(f"Perception query output must contain a list: {payload}")
    queries: list[dict[str, str]] = []
    for item in raw_queries:
        if not isinstance(item, dict):
            continue
        raw_object_id = str(item.get("object_id", item.get("name", ""))).strip()
        object_id, inferred_hint = normalize_perception_object_id(raw_object_id)
        text_prompt = str(item.get("text_prompt", raw_object_id or object_id)).strip()
        role = normalize_query_role(item.get("role", item.get("query_role", "context")))
        if not object_id:
            continue
        query = {
            "object_id": object_id,
            "text_prompt": text_prompt or object_id,
            "role": role,
            "instance_hint": merge_instance_hints(item.get("instance_hint", item.get("query_instance_hint", "")), inferred_hint),
            "reason": str(item.get("reason", "")).strip(),
        }
        query.update(normalized_relation_metadata(item))
        instance_ref = str(item.get("instance_ref", item.get("track_id", item.get("instance_id", ""))) or "").strip()
        oracle_id = str(item.get("oracle_id", "") or "").strip()
        if instance_ref and query.get("entity_scope") != "reference_set":
            query["instance_ref"] = instance_ref[:160]
        if oracle_id and query.get("entity_scope") != "reference_set":
            query["oracle_id"] = oracle_id[:160]
        merge_index = matching_perception_query_index(queries, query)
        if merge_index is not None:
            queries[merge_index] = merge_perception_query(
                queries[merge_index],
                query,
            )
            continue
        queries.append(query)
        if len(queries) >= max(1, int(default_max_queries)):
            break
    return {"queries": queries}


def build_planner_messages(
    payload: dict[str, Any],
    *,
    planner_prompt_mode: str = "legacy_duplicate",
) -> tuple[list[dict[str, Any]], str]:
    """Build the two-view planner request without silently dropping task context.

    ``AgentApiPlannerAdapter`` supplies an already-rendered planner prompt containing
    the task, committed memory, and state vector.  Local models with a bounded
    context window can opt into using that rendered prompt as the single copy of
    dynamic context.  That mode fails closed when the rendered prompt is absent,
    rather than silently serving a request with missing task information.

    The second return value is the previous memory used by the response normalizer.
    """

    task = str(payload["task"])
    previous_memory_text = str(payload.get("previous_memory_text", ""))
    planner_state = payload.get("planner_state", [])
    planner_start_image_b64 = str(payload["planner_start_image_b64"])
    planner_end_image_b64 = str(payload["planner_end_image_b64"])
    provided_prompt = str(payload.get("prompt", "")).strip()
    prompt = provided_prompt or SYSTEM_PROMPT

    if planner_prompt_mode not in {"legacy_duplicate", "rendered_system_once"}:
        raise ValueError(f"Unsupported planner prompt mode: {planner_prompt_mode!r}")
    if planner_prompt_mode == "rendered_system_once" and not provided_prompt:
        raise ValueError(
            "planner_prompt_mode=rendered_system_once requires an adapter-rendered prompt"
        )

    if planner_prompt_mode == "rendered_system_once":
        rendered_task = task.strip()
        rendered_memory = previous_memory_text.strip()
        rendered_state = json.dumps(planner_state, ensure_ascii=False)
        missing_rendered_fields: list[str] = []
        if not rendered_task or rendered_task not in provided_prompt:
            missing_rendered_fields.append("task")
        if rendered_memory and rendered_memory not in provided_prompt:
            missing_rendered_fields.append("previous_memory_text")
        if rendered_state not in provided_prompt:
            missing_rendered_fields.append("planner_state")
        if missing_rendered_fields:
            raise ValueError(
                "planner_prompt_mode=rendered_system_once prompt is missing rendered "
                "dynamic fields: " + ", ".join(missing_rendered_fields)
            )
        user_text = (
            "Compare the segment start and segment end frames using the task, "
            "committed memory, and state summary already supplied in the system instruction."
        )
    else:
        user_text = (
            f"Task instruction: {task}\n"
            f"Previous committed memory: {previous_memory_text}\n"
            f"State summary vector: {json.dumps(planner_state, ensure_ascii=False)}"
        )
        try:
            structured_task = extract_json_object(task)
        except Exception:
            structured_task = {}
        if isinstance(structured_task.get("memory_harness"), dict):
            user_text += "\nMemory harness: " + json.dumps(
                structured_task["memory_harness"], ensure_ascii=False
            )

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{planner_start_image_b64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{planner_end_image_b64}"},
                },
            ],
        },
    ]
    return messages, previous_memory_text

class QwenPlannerHandler(BaseHTTPRequestHandler):
    client: OpenAI | None = None
    model: str = "qwen3.6-plus"
    timeout_sec: int = 120
    enable_thinking: bool = False
    planner_prompt_mode: str = "legacy_duplicate"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"status": "ok", "model": self.model})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/plan":
            self._handle_plan()
            return
        if self.path == "/ood":
            self._handle_ood()
            return
        if self.path == "/recover":
            self._handle_recover()
            return
        if self.path == "/perception_queries":
            self._handle_perception_queries()
            return
        if self.path == "/normalize_perception_queries":
            self._handle_normalize_perception_queries()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def _handle_plan(self) -> None:
        if self.client is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "client is not initialized")
            return

        body_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(body_length)
        payload = json.loads(raw_body.decode("utf-8"))

        messages, previous_memory_text = build_planner_messages(
            payload,
            planner_prompt_mode=self.planner_prompt_mode,
        )
        prediction = self._run_completion(messages=messages, normalizer=lambda payload: normalize_prediction(payload, previous_memory_text))
        self._send_json(prediction)

    def _handle_ood(self) -> None:
        if self.client is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "client is not initialized")
            return

        body_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(body_length)
        payload = json.loads(raw_body.decode("utf-8"))
        skill_payload = payload.get("skill_payload", {})
        media = payload.get("media", [])
        prompt = str(payload.get("prompt", "")).strip() or OOD_SYSTEM_PROMPT
        content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(skill_payload, ensure_ascii=False)}
        ]
        for item in media:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                continue
            if item.get("source") != "base64":
                continue
            mime_type = str(item.get("mime_type", "image/jpeg"))
            data = str(item.get("data", ""))
            if not data:
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{data}"},
                }
            )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": content,
            },
        ]
        prediction = self._run_completion(messages=messages, normalizer=normalize_ood_prediction)
        self._send_json(prediction)

    def _handle_recover(self) -> None:
        if self.client is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "client is not initialized")
            return

        body_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(body_length)
        payload = json.loads(raw_body.decode("utf-8"))
        recovery_payload = payload.get("recovery_payload", {})
        media = payload.get("media", [])
        prompt = str(payload.get("prompt", "")).strip() or RECOVERY_SYSTEM_PROMPT
        content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(recovery_payload, ensure_ascii=False)}
        ]
        for item in media:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                continue
            if item.get("source") != "base64":
                continue
            mime_type = str(item.get("mime_type", "image/jpeg"))
            data = str(item.get("data", ""))
            if not data:
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{data}"},
                }
            )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": content,
            },
        ]
        if recovery_payload.get("mode") == "action_effect_verification":
            normalizer = normalize_action_effect_prediction
        else:
            normalizer = normalize_recovery_prediction
        prediction = self._run_completion(messages=messages, normalizer=normalizer)
        self._send_json(prediction)

    def _handle_perception_queries(self) -> None:
        if self.client is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "client is not initialized")
            return

        body_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(body_length)
        payload = json.loads(raw_body.decode("utf-8"))
        prompt = str(payload.get("prompt", "")).strip() or PERCEPTION_QUERY_SYSTEM_PROMPT
        query_payload = {
            "global_task": payload.get("global_task", ""),
            "current_subtask": payload.get("current_subtask", ""),
            "committed_memory": payload.get("committed_memory", ""),
            "observation_summary": payload.get("observation_summary", ""),
            "robot_state": payload.get("robot_state", {}),
            "active_skill": payload.get("active_skill", ""),
            "max_queries": payload.get("max_queries", 3),
            "camera": payload.get("camera", ""),
            "cameras": payload.get("cameras", []),
            "oracle_objects": payload.get("oracle_objects", []),
            "scene_instances": payload.get("scene_instances", []),
            "require_instance_binding": bool(payload.get("require_instance_binding", False)),
            "instance_binding_phase": payload.get(
                "instance_binding_phase",
                (
                    "post_detection_selection"
                    if bool(payload.get("require_instance_binding", False))
                    else "candidate_discovery"
                ),
            ),
            "binding_requirement": payload.get("binding_requirement", {}),
        }
        image_b64 = str(payload.get("image_b64", "")).strip()
        image_b64_by_camera = payload.get("image_b64_by_camera", {})
        image_items: list[dict[str, Any]] = []
        if isinstance(image_b64_by_camera, dict):
            for camera, camera_image_b64 in image_b64_by_camera.items():
                encoded = str(camera_image_b64 or "").strip()
                if not encoded:
                    continue
                image_items.append({"type": "text", "text": f"camera={str(camera)}"})
                image_items.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        if not image_items and image_b64:
            image_items.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
        if image_items:
            user_content: str | list[dict[str, Any]] = [{"type": "text", "text": json.dumps(query_payload, ensure_ascii=False)}]
            user_content.extend(image_items)
        else:
            user_content = json.dumps(query_payload, ensure_ascii=False)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ]
        prediction = self._run_completion(
            messages=messages,
            normalizer=lambda item: validate_perception_queries(
                item,
                default_max_queries=int(payload.get("max_queries", 3) or 3),
            ),
        )
        self._send_json(prediction)

    def _handle_normalize_perception_queries(self) -> None:
        if self.client is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "client is not initialized")
            return

        body_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(body_length)
        payload = json.loads(raw_body.decode("utf-8"))
        prompt = str(payload.get("prompt", "")).strip() or load_perception_query_normalization_prompt()
        max_queries = int(payload.get("max_queries", 3) or 3)
        normalization_payload = {
            "global_task": payload.get("global_task", ""),
            "current_subtask": payload.get("current_subtask", ""),
            "committed_memory": payload.get("committed_memory", ""),
            "observation_summary": payload.get("observation_summary", ""),
            "raw_queries": payload.get("raw_queries", payload.get("queries", [])),
            "oracle_objects": payload.get("oracle_objects", []),
            "scene_instances": payload.get("scene_instances", []),
            "require_instance_binding": bool(payload.get("require_instance_binding", False)),
            "instance_binding_phase": payload.get(
                "instance_binding_phase",
                (
                    "post_detection_selection"
                    if bool(payload.get("require_instance_binding", False))
                    else "candidate_discovery"
                ),
            ),
            "binding_requirement": payload.get("binding_requirement", {}),
            "max_queries": max_queries,
        }
        image_b64 = str(payload.get("image_b64", "")).strip()
        image_b64_by_camera = payload.get("image_b64_by_camera", {})
        image_items: list[dict[str, Any]] = []
        if isinstance(image_b64_by_camera, dict):
            for camera, camera_image_b64 in image_b64_by_camera.items():
                encoded = str(camera_image_b64 or "").strip()
                if not encoded:
                    continue
                image_items.append({"type": "text", "text": f"camera={str(camera)}"})
                image_items.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        if not image_items and image_b64:
            image_items.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
        if image_items:
            user_content: str | list[dict[str, Any]] = [
                {"type": "text", "text": json.dumps(normalization_payload, ensure_ascii=False)}
            ]
            user_content.extend(image_items)
        else:
            user_content = json.dumps(normalization_payload, ensure_ascii=False)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ]
        prediction = self._run_completion(
            messages=messages,
            normalizer=lambda item: validate_perception_queries(item, default_max_queries=max_queries),
        )
        self._send_json(prediction)

    def _run_completion(self, *, messages: list[dict[str, Any]], normalizer):
        extra_body: dict[str, Any] = {}
        if self.enable_thinking:
            extra_body["enable_thinking"] = True
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            extra_body=extra_body,
            timeout=self.timeout_sec,
        )
        text = response.choices[0].message.content or ""
        return normalizer(extract_json_object(text))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    args = parse_args()
    QwenPlannerHandler.client = build_client()
    QwenPlannerHandler.model = args.model
    QwenPlannerHandler.timeout_sec = args.timeout_sec
    QwenPlannerHandler.enable_thinking = bool(args.enable_thinking)
    server = ThreadingHTTPServer((args.host, args.port), QwenPlannerHandler)
    print(f"[qwen-planner] listening on http://{args.host}:{args.port} model={args.model}")
    server.serve_forever()


if __name__ == "__main__":
    main()
