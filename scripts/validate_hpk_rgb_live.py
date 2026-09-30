from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.materialize_hpk_rgb_sources import materialize_episode
from roboharn_evo.agent.hpk.family_store import load_knowledge_family_catalog, save_hierarchical_store_with_catalog
from roboharn_evo.agent.hpk.hierarchical_retriever import ActionKnowledgeQuery, SubtaskKnowledgeQuery, VLMActionKnowledgeRetriever, VLMSubtaskKnowledgeRetriever
from roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store
from roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex
from roboharn_evo.agent.hpk.rgb_maintenance import RGBKnowledgeReviewer
from roboharn_evo.agent.hpk.rgb_retrieval import RGBRetrievalContext, RuntimeMultimodalBackend
from roboharn_evo.agent.reflector.multimodal_transport import MultimodalImage


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--query-episode", type=int, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:19135")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    tasks, actions = load_hierarchical_store(args.store_root)
    stored = RGBEvidenceIndex.load(args.store_root)
    manifest = json.loads((args.source_root / "source_trajectories.json").read_text())
    model = manifest["config"]["model"]
    if args.query_episode not in {item["episode"] for item in manifest["source_trajectories"]}:
        raise ValueError("query trajectory must belong to the authorized source manifest")
    report = json.loads((args.store_root / "store_build_report.json").read_text())
    source_events = {}
    for episode in report["source_episodes"]:
        _, source_index = materialize_episode(args.source_root, episode)
        source_events.update({record["execution_event"]: record for record in source_index.records})
    # 原始执行记录由同一次真实反思补充，保留已经维护的 verdict 与适用性。
    records = []
    for record in stored.records:
        event = source_events[record["execution_event"]]
        records.append({**record, "execution": event["execution"], "tested_knowledge": event["tested_knowledge"]})
    evidence = RGBEvidenceIndex(stored.root, records)
    evidence.validate_knowledge(tasks, actions)
    catalog = load_knowledge_family_catalog(args.store_root, task_knowledge=tasks, action_knowledge=actions)
    relocated = args.output_root / "relocated_package"
    save_hierarchical_store_with_catalog(relocated, task_knowledge=tasks, action_knowledge=actions, catalog=catalog, evidence_index=evidence)
    backend = RuntimeMultimodalBackend(args.service_url, model=model["model"], reasoning_effort=model["reasoning_effort"], max_images=32, timeout_sec=1200)
    query_atomic, query_index = materialize_episode(args.source_root, args.query_episode)
    package = json.loads((args.source_root / f"episode_{args.query_episode}" / "trajectory_knowledge_package.json").read_text())
    baseline = package["task_strategy"]["subtasks"][0]["subtask"]
    task_query = SubtaskKnowledgeQuery(**query_atomic["task_knowledge"][0]["condition"])
    action = query_atomic["action_knowledge"][0]
    action_query = ActionKnowledgeQuery(**action["condition"], intended_effect=action["expected_effect"]["physical_effect"], candidate_geometry=(action["geometric_strategy"],))
    results = []
    for name, directory in (("original", args.store_root), ("relocated", relocated)):
        loaded_tasks, loaded_actions = load_hierarchical_store(directory)
        index = evidence if name == "original" else RGBEvidenceIndex.load(directory)
        context = RGBRetrievalContext(index, task_knowledge=loaded_tasks, action_knowledge=loaded_actions, max_images=32, camera_names=("head_camera",))
        for kind in ("task", "action"):
            record = query_index.for_knowledge(kind, 0)[0]
            observation = next(item for item in record["before"] if item["camera"] == "head_camera")
            image = MultimodalImage(evidence_id="current source observation", mime_type="image/png", content=query_index.image_path(observation["path"]).read_bytes(), detail="high")
            context.set_current((image,), audit_root=args.output_root / name / kind)
            if kind == "task":
                retriever = VLMSubtaskKnowledgeRetriever(backend, rgb_context=context)
                match = retriever.retrieve(loaded_tasks, task_query, baseline_subtask=baseline)
            else:
                retriever = VLMActionKnowledgeRetriever(backend, rgb_context=context)
                match = retriever.retrieve(loaded_actions, action_query)
            matched = None if match is None else {"knowledge": match.knowledge.to_dict(), "reason": match.reason, "decision": match.grounded_subtask if kind == "task" else match.selected_candidate_geometry}
            result = {"package": name, "knowledge_type": kind, "match": matched, "audit": retriever.last_call_audit, "query_source": str(args.source_root / f"episode_{args.query_episode}"), "candidate_source": "executed geometry from the real source trajectory"}
            results.append(result)
            write_json(args.output_root / "retrieval_results.json", results)
            print(json.dumps({"package": name, "kind": kind, "selected": match is not None}), flush=True)
    reviewer = RGBKnowledgeReviewer(backend, max_images=32, camera_names=("head_camera",))
    reviews = []
    for enabled in (True, False):
        name = "rgb" if enabled else "without_rgb"
        _, _, review = reviewer.review(kind="action", knowledge_index=0, knowledge=actions[0], evidence=evidence, maintenance_rgb_enabled=enabled, output_root=args.output_root / "paired_maintenance" / name)
        reviews.append({"configuration": name, "decision": review["decision"], "reason": review["reason"], "knowledge": review["knowledge"], "requests": len(review["calls"])})
        print(json.dumps(reviews[-1]), flush=True)
    first = json.loads((args.output_root / "paired_maintenance/rgb/call_000/request.json").read_text())
    second = json.loads((args.output_root / "paired_maintenance/without_rgb/call_000/request.json").read_text())
    if first["input"] != second["input"] or first["instructions"] != second["instructions"]:
        raise ValueError("paired maintenance must share its initial text and review rules")
    if not first["images"] or second["images"]:
        raise ValueError("paired maintenance RGB configuration did not control image transfer")
    write_json(args.output_root / "validation_report.json", {"model": model, "relocated_package": str(relocated), "retrieval_calls_completed": len(results), "paired_maintenance": reviews, "shared_initial_text_and_rules": True})


if __name__ == "__main__":
    main()
