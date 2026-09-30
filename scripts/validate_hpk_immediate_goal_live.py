from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboharn_evo.agent.hpk.goal_consistency import build_subtask_goal_contract_v31
from roboharn_evo.agent.hpk.hierarchical_knowledge import SubtaskGoalContractV31, subtask_goal_contract_json_schema
from roboharn_evo.agent.hpk.hierarchical_retriever import SubtaskKnowledgeQuery, VLMSubtaskKnowledgeRetriever
from roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store
from roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex
from roboharn_evo.agent.hpk.rgb_maintenance import object_schema
from roboharn_evo.agent.hpk.rgb_retrieval import RGBRetrievalContext, RuntimeMultimodalBackend
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import _strict_response
from roboharn_evo.agent.reflector.multimodal_transport import MultimodalImage
from roboharn_evo.services.agent_api.qwen_planner import _build_planner_message_bundle, normalize_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--job", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.source_root / "source_trajectories.json").read_text())
    tasks, actions = load_hierarchical_store(args.source_root)
    evidence = RGBEvidenceIndex.load(args.source_root)
    backend = RuntimeMultimodalBackend("http://127.0.0.1:19135", model="gpt-5.5", reasoning_effort="xhigh", max_images=32, timeout_sec=1200)
    text_field = {"type": "string"}
    schema = object_schema({"commit_label": {"type": "string", "enum": ["no_update", "subtask_complete", "state_change"]}, "memory_text": text_field, "selected_skill": text_field, "subtask_text": text_field, "preferred_arm": {"type": "string", "enum": ["left", "right", "either"]}, "subtask_goal": subtask_goal_contract_json_schema()})
    reports = []
    for job_name in args.job:
        job = args.source_root / "integration_runs" / job_name
        directory = args.output / job_name
        directory.mkdir()
        request_roots = [job / name / "request_00000" for name in ("hpk_rgb_retrieval", "afk_rgb_retrieval")]
        request_roots = [path for path in request_roots if (path / "request.json").is_file()]
        if len(request_roots) != 1:
            raise ValueError(f"expected exactly one initial retrieval request in {job}")
        request_root = request_roots[0]
        request = json.loads((request_root / "request.json").read_text())
        event_path, = (job / "rollout").glob("episode_*/events.jsonl")
        records = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
        observation = next(record for record in records if record["event"] == "observation_preprocess")
        router = next(record for record in records if record["event"] == "recovery_router")
        requested_action = next(call["args"]["action_mode"] for call in router["tool_calls"] if "action_mode" in call["args"])
        image_bytes = (request_root / request["images"][0]).read_bytes()
        (directory / "current.png").write_bytes(image_bytes)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {"task": manifest["instruction"], "previous_memory_text": request["input"]["query"]["task_state"], "planner_state": observation["robot_state"], "planner_start_image_b64": encoded, "planner_end_image_b64": encoded, "hpk_goal_consistency_enabled": True}
        messages, previous_memory, _ = _build_planner_message_bundle(payload)
        input_text = messages[1]["content"][0]["text"]
        (directory / "planner_request.json").write_text(json.dumps({"instructions": messages[0]["content"], "input_text": input_text, "images": ["current.png", "current.png"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        images = tuple(MultimodalImage(evidence_id=f"current observation {index}", mime_type="image/png", content=image_bytes, detail="high") for index in range(2))
        completion = backend.complete(instructions=messages[0]["content"], input_text=input_text, images=images, output_schema=schema, schema_name="hpk_v31_immediate_planner_step")
        raw = _strict_response(completion.output)
        (directory / "planner_response.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        prediction = normalize_prediction(raw, previous_memory, hpk_goal_consistency_enabled=True)
        contract = SubtaskGoalContractV31(prediction["subtask_goal"])
        build_subtask_goal_contract_v31(contract, operation=requested_action)
        context = RGBRetrievalContext(evidence, task_knowledge=tasks, action_knowledge=actions, max_images=32, camera_names=("head_camera",))
        context.set_current(images[:1], audit_root=directory / "retrieval")
        selected = VLMSubtaskKnowledgeRetriever(backend, rgb_context=context).retrieve(tasks, SubtaskKnowledgeQuery(**request["input"]["query"]), baseline_subtask=prediction["subtask_text"], goal_contract=contract, require_goal_contract=True)
        selected_contract = contract if selected is None else selected.subtask_goal
        build_subtask_goal_contract_v31(selected_contract, operation=requested_action)
        reports.append({"job": job_name, "actual_requested_operation": requested_action, "planner_subtask": prediction["subtask_text"], "planner_goal": contract.to_dict(), "knowledge_selected": selected is not None, "retrieved_goal": selected_contract.to_dict(), "robot_motion_executed": False})
        (args.output / "results.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(reports[-1], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
