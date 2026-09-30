from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.materialize_hpk_rgb_sources import materialize_episode
from scripts.maintain_hpk_v3_family_store import RecordingBackend
from roboharn_evo.agent.hpk.family_store import save_hierarchical_store_with_catalog
from roboharn_evo.agent.hpk.hierarchical_knowledge import ActionKnowledgeV3, SubtaskKnowledgeV3
from roboharn_evo.agent.hpk.incremental_maintainer import IncrementalKnowledgeMaintainer
from roboharn_evo.agent.hpk.knowledge_family import VLMKnowledgeFamilyCatalogBuilder
from roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex
from roboharn_evo.agent.hpk.rgb_maintenance import RGBKnowledgeReviewer
from roboharn_evo.agent.hpk.rgb_retrieval import RuntimeMultimodalBackend
from roboharn_evo.agent.hpk.semantic_consolidator import VLMKnowledgeConsolidator


def build_store(*, source_root: Path, service_url: str, max_images: int, episodes: list[int] | None = None, output_root: Path | None = None, consolidation_response: Path | None = None):
    manifest = json.loads((source_root / "source_trajectories.json").read_text())
    fixed_episodes = [item["episode"] for item in manifest["source_trajectories"]]
    episodes = fixed_episodes if episodes is None else episodes
    if len(episodes) != len(set(episodes)) or not set(episodes) <= set(fixed_episodes):
        raise ValueError("store inputs must belong to the fixed source manifest")
    store_root = source_root if output_root is None else output_root
    store_root.mkdir(parents=True, exist_ok=True)
    if len(episodes) < 2:
        raise ValueError("source store maintenance requires multiple trajectories")
    for episode in episodes:
        if not (source_root / f"episode_{episode}/reflection_response.json").is_file():
            raise FileNotFoundError(source_root / f"episode_{episode}/reflection_response.json")
    output = store_root / "build"
    output.mkdir(exist_ok=consolidation_response is not None)
    sources = [materialize_episode(source_root, episode) for episode in episodes]
    model = manifest["config"]["model"]
    live_backend = RuntimeMultimodalBackend(service_url, model=model["model"], reasoning_effort=model["reasoning_effort"], max_images=max_images, timeout_sec=1200)
    backend = RecordingBackend(live_backend, checkpoint_path=output / "model_responses.json")
    tasks, actions = [], []
    evidence = RGBEvidenceIndex(source_root, ())
    for atomic, source_evidence in sources[:-1]:
        evidence = evidence.append(source_evidence, task_offset=len(tasks), action_offset=len(actions))
        tasks.extend(SubtaskKnowledgeV3(value) for value in atomic["task_knowledge"])
        actions.extend(ActionKnowledgeV3(value) for value in atomic["action_knowledge"])
    consolidator = VLMKnowledgeConsolidator(backend)
    consolidated = (
        consolidator.consolidate(task_knowledge=tasks, action_knowledge=actions, evidence_index=evidence)
        if consolidation_response is None
        else consolidator.materialize_response(task_knowledge=tasks, action_knowledge=actions, evidence_index=evidence, response=json.loads(consolidation_response.read_text()))
    )
    (output / "semantic_groups.json").write_text(json.dumps(consolidated.raw_response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    catalog_result = VLMKnowledgeFamilyCatalogBuilder(backend).build(task_knowledge=consolidated.task_knowledge, action_knowledge=consolidated.action_knowledge)
    save_hierarchical_store_with_catalog(output / "bootstrap", task_knowledge=consolidated.task_knowledge, action_knowledge=consolidated.action_knowledge, catalog=catalog_result.catalog, evidence_index=consolidated.evidence_index)
    incoming, incoming_evidence = sources[-1]
    maintained = IncrementalKnowledgeMaintainer(
        backend, consolidator=consolidator,
        rgb_reviewer=RGBKnowledgeReviewer(backend, max_images=max_images, camera_names=("head_camera",)),
    ).maintain(
        task_knowledge=consolidated.task_knowledge, action_knowledge=consolidated.action_knowledge,
        catalog=catalog_result.catalog,
        new_task_knowledge=incoming["task_knowledge"], new_action_knowledge=incoming["action_knowledge"],
        evidence_index=consolidated.evidence_index, new_evidence_index=incoming_evidence,
        maintenance_rgb_enabled=True, review_output_root=output / "maintenance", output_root=store_root,
    )
    report = {
        "source_episodes": episodes,
        "bootstrap_episodes": episodes[:-1],
        "incremental_episode": episodes[-1],
        "model": model["model"], "reasoning_effort": model["reasoning_effort"],
        "service_url": service_url, "historical_rgb_cameras": ["head_camera"],
        "task_knowledge": len(maintained.task_knowledge), "action_knowledge": len(maintained.action_knowledge),
        "task_skills": len(maintained.catalog.task_families), "action_skills": len(maintained.catalog.action_families),
        "maintenance": maintained.audit, "model_usage": backend.usage_summary(),
        "reused_consolidation_response": None if consolidation_response is None else str(consolidation_response),
    }
    (store_root / "store_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:19135")
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--consolidation-response", type=Path)
    args = parser.parse_args()
    report = build_store(source_root=args.source_root.resolve(strict=True), service_url=args.service_url, max_images=args.max_images, episodes=args.episodes, output_root=args.output_root, consolidation_response=args.consolidation_response)
    print(json.dumps({key: report[key] for key in ("source_episodes", "task_knowledge", "action_knowledge", "task_skills", "action_skills")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
