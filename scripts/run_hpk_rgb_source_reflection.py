from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_rmbench_hpk_v31_gate_with_services import _wait_health
from scripts.maintain_hpk_v3_family_store import RecordingBackend
from roboharn_evo.agent.hpk.hierarchical_knowledge import TrajectoryKnowledgePackageV3, atomize_package
from roboharn_evo.agent.hpk.rgb_evidence import bind_reflection_evidence
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import ExistingGPTMultimodalReflectorBackend, VLMHierarchicalReflector, build_hierarchical_reflection_input
from roboharn_evo.agent.reflector.multimodal_transport import ImageEgressAuthorization, MultimodalImage, OpenAICompatibleMultimodalTransport


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_episode_request(root: Path, episode: int, *, review: bool):
    directory = root / f"episode_{episode}"
    if (directory / "trajectory_knowledge_package.json").exists() and not review:
        raise FileExistsError(directory / "trajectory_knowledge_package.json")
    review_package = json.loads((directory / "reflection_response.json").read_text())["package"] if review else None
    if review_package is not None:
        TrajectoryKnowledgePackageV3(review_package)
    payload = json.loads((directory / "reflector_input.json").read_text())
    observations = json.loads((directory / "source_observations.json").read_text())
    payload["key_observations"] = [f"Observation at position {index}: {observations['operations'][item['annotation_index']]['annotation']}" for index, item in enumerate(observations["observations"])]
    for chunk, source in zip(payload["ordered_action_chunks"], observations["chunks"], strict=True):
        chunk["visual_observations"] = [f"Before image at position {source['before_observation']}; after image at position {source['after_observation']}; subsequent observations can confirm delayed effects."]
    paths = [item["images"][observations["reflection_camera"]] for item in observations["observations"]]
    images = [MultimodalImage(evidence_id=f"observation {ordinal}", mime_type="image/png", content=(root / path).read_bytes(), detail="high") for ordinal, path in enumerate(paths)]
    build_hierarchical_reflection_input(instruction=payload["instruction"], ordered_action_chunks=payload["ordered_action_chunks"], key_observations=payload["key_observations"], available_state=payload["available_state"], image_count=len(images))
    return payload, observations, paths, images, review_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--port", type=int, default=19164)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--start-service", action="store_true")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--bind-existing-package", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    source_manifest = json.loads((root / "source_trajectories.json").read_text())
    selected = {source["episode"] for source in source_manifest["source_trajectories"]}
    if len(set(args.episodes)) != len(args.episodes) or not set(args.episodes) <= selected:
        raise ValueError("requested episodes must belong to the fixed source manifest")
    prepared = {episode: prepare_episode_request(root, episode, review=args.review) for episode in args.episodes}
    if any(len(item[3]) > args.max_images for item in prepared.values()):
        raise ValueError("source images exceed the configured service capacity")
    service_root = root / "services" / f"gpt55_rgb_{args.port}"
    service_root.mkdir(parents=True, exist_ok=True)
    workdir = REPO_ROOT / "eval_result" / "hpk" / "rgb_services" / f"gpt55_{args.port}" / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(workdir), NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    service_url = f"http://127.0.0.1:{args.port}"
    service = None
    log = None
    if args.start_service:
        command = [
            sys.executable, "-u", str(REPO_ROOT / "scripts/serve_rmbench_agent_api.py"),
            "--provider", "openai", "--backend", "codex-account",
            "--host", "127.0.0.1", "--port", str(args.port),
            "--model", args.model, "--reasoning-effort", args.reasoning_effort,
            "--responses-max-images", str(args.max_images), "--timeout-sec", "1200",
            "--max-retries", "0", "--max-concurrent-requests", "1",
            "--codex-bin", os.environ.get("CODEX_BIN", "codex"),
            "--codex-auth-file", os.environ["ROBOHARN_EVO_PROVIDER_AUTH_FILE"],
            "--config-file", os.environ["ROBOHARN_EVO_PROVIDER_CONFIG_FILE"],
            "--codex-runtime-root", str(workdir.parent / "runtime"),
            "--codex-workdir", str(workdir),
        ]
        log = (service_root / "service.log").open("ab")
        service = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        health = _wait_health(service, service_url, timeout_sec=90)
        write_json(service_root / "service_identity.json", {"pid": service.pid, "health": health, "command": command})
    try:
        authorization = ImageEgressAuthorization.operator_granted(assertion="Cover Blocks source trajectory RGB reflection", scope="fixed source trajectories and operation observations")
        transport = OpenAICompatibleMultimodalTransport(service_url=service_url, model=args.model, reasoning_effort=args.reasoning_effort, timeout_sec=1200, max_images=args.max_images)
        transport.capability_preflight(authorization=authorization)
        write_json(service_root / "capability_report.json", transport.last_preflight_audit)
        for episode in args.episodes:
            episode_root = root / f"episode_{episode}"
            reflector = VLMHierarchicalReflector(RecordingBackend(ExistingGPTMultimodalReflectorBackend(transport, authorization=authorization), checkpoint_path=episode_root / "model_response.json"))
            package_path = episode_root / "trajectory_knowledge_package.json"
            payload, source_observations, image_paths, images, review_package = prepared[episode]
            write_json(episode_root / "reflector_input.json", payload)
            write_json(episode_root / "reflection_request.json", {"model": args.model, "reasoning_effort": args.reasoning_effort, "service_url": service_url, "input": payload, "images": image_paths, "review_package": review_package})
            print(json.dumps({"episode": episode, "status": "reflection requested", "images": len(images)}), flush=True)
            result = reflector.reflect(instruction=payload["instruction"], ordered_action_chunks=payload["ordered_action_chunks"], key_observations=payload["key_observations"], available_state=payload["available_state"], images=images, bind_rgb_evidence=True, review_package=review_package, bind_existing_package=args.bind_existing_package)
            write_json(episode_root / "reflection_response.json", result.to_dict())
            write_json(episode_root / "reflection_transport_audit.json", transport.last_multimodal_completion_audit)
            atomic, evidence = bind_reflection_evidence(root=root, source_name=f"episode_{episode}", source_observations=source_observations, package=result.package, atomic_knowledge=atomize_package(result.package), bindings=result.evidence_bindings)
            evidence.validate_knowledge(atomic["task_knowledge"], atomic["action_knowledge"])
            evidence.save(filename=f"episode_{episode}/evidence_index.jsonl")
            write_json(package_path, result.package.to_dict())
            write_json(episode_root / "atomic_knowledge.json", atomic)
            print(json.dumps({"episode": episode, "status": "completed", "task_knowledge": len(atomic["task_knowledge"]), "action_knowledge": len(atomic["action_knowledge"]), "execution_events": len(evidence.records)}), flush=True)
    finally:
        if service is not None:
            service.terminate()
            service.wait(timeout=30)
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
