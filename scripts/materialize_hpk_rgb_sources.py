from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboharn_evo.agent.hpk.hierarchical_knowledge import TrajectoryKnowledgePackageV3, atomize_package
from roboharn_evo.agent.hpk.rgb_evidence import bind_reflection_evidence


def materialize_episode(root: Path, episode: int):
    directory = root / f"episode_{episode}"
    response = json.loads((directory / "reflection_response.json").read_text())
    observations = json.loads((directory / "source_observations.json").read_text())
    package = TrajectoryKnowledgePackageV3(response["package"])
    atomic, evidence = bind_reflection_evidence(root=root, source_name=f"episode_{episode}", source_observations=observations, package=package, atomic_knowledge=atomize_package(package), bindings=response["evidence_bindings"])
    evidence.validate_knowledge(atomic["task_knowledge"], atomic["action_knowledge"])
    evidence.save(filename=f"episode_{episode}/evidence_index.jsonl")
    (directory / "trajectory_knowledge_package.json").write_text(json.dumps(package.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "atomic_knowledge.json").write_text(json.dumps(atomic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return atomic, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for episode in args.episodes:
        atomic, evidence = materialize_episode(args.source_root.resolve(strict=True), episode)
        print(json.dumps({"episode": episode, "task_knowledge": len(atomic["task_knowledge"]), "action_knowledge": len(atomic["action_knowledge"]), "execution_events": len(evidence.records)}), flush=True)


if __name__ == "__main__":
    main()
