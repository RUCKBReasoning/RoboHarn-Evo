from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_hpk_v3_gt_hierarchical_outputs import (  # noqa: E402
    _load_prepared_episode,
    _prepare_episode,
    _write_json,
)
from scripts.build_hpk_v3_merged_store import build_merged_store  # noqa: E402
from roboharn_evo.agent.hpk.hierarchical_knowledge import (  # noqa: E402
    HPKV3ValidationError,
    atomize_package,
)
from roboharn_evo.agent.hpk.semantic_consolidator import (  # noqa: E402
    VLMKnowledgeConsolidator,
)
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (  # noqa: E402
    ExistingGPTMultimodalReflectorBackend,
    VLMHierarchicalReflector,
)
from roboharn_evo.agent.reflector.multimodal_transport import (  # noqa: E402
    ImageEgressAuthorization,
    OpenAICompatibleMultimodalTransport,
)
from roboharn_evo.benchmark_adapters.rmbench.formal_q1 import (  # noqa: E402
    load_protocol,
    validate_benchmark_inputs,
)
from roboharn_evo.benchmark_adapters.rmbench.formal_source import (  # noqa: E402
    FormalSourceError,
    materialize_method_views,
    select_reflection_evidence,
)


DEFAULT_PROTOCOL = REPO_ROOT / "benchmarks/rmbench/experiments/hpk_formal_q1_v1.yaml"


def _instruction(data_root: Path, episodes: list[int], instruction_type: str) -> str:
    values: list[str] = []
    for episode in episodes:
        payload = json.loads(
            (data_root / "instructions" / f"episode{episode}.json").read_text(
                encoding="utf-8"
            )
        )
        candidates = payload.get(instruction_type)
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise FormalSourceError(
                f"episode {episode} must contain one {instruction_type} instruction"
            )
        values.append(str(candidates[0]).strip())
    if len(set(values)) != 1 or not values[0]:
        raise FormalSourceError(
            "source trajectories do not share one frozen instruction"
        )
    return values[0]


def _transport(gpt_url: str, authorization: ImageEgressAuthorization, max_images: int):
    transport = OpenAICompatibleMultimodalTransport(
        service_url=gpt_url,
        model="gpt-5.5",
        reasoning_effort="xhigh",
        timeout_sec=600,
        max_images=max_images,
    )
    transport.capability_preflight(authorization=authorization)
    return transport


def prepare_task(
    *,
    protocol: dict,
    task: dict,
    data_root: Path,
    output_root: Path,
) -> None:
    episodes = list(protocol["source"]["episode_indices"])
    instruction = _instruction(data_root, episodes, protocol["instruction_type"])
    output_root.mkdir(parents=True, exist_ok=False)
    prepared: list[dict] = []
    for episode in episodes:
        chunks, descriptions, images = _prepare_episode(
            episode=episode,
            instruction=instruction,
            available_state=(),
            data_root=data_root,
            output_root=output_root,
            sam_url=None,
            sam_concept="task-relevant objects from the benchmark instruction",
            sam_input_root=output_root / "unused_sam_inputs",
            sam_mask_root=output_root / "unused_sam_masks",
        )
        prepared.append(
            {
                "episode": episode,
                "seed": task["source_seeds"][episode],
                "action_chunks": len(chunks),
                "visual_boundaries": len(images),
                "service_calls": 0,
            }
        )
        assert len(descriptions) == len(images)
    _write_json(
        output_root / "preparation_summary.json",
        {
            "task": task["name"],
            "instruction": instruction,
            "source_trajectories": prepared,
            "rgb_channel_correction": "RMBench HDF5 historical BGR encoding corrected",
            "external_model_calls": 0,
            "sam3_calls": 0,
        },
    )


def reflect_task(
    *,
    protocol: dict,
    task: dict,
    data_root: Path,
    output_root: Path,
    gpt_url: str,
    max_images: int,
) -> None:
    episodes = list(protocol["source"]["episode_indices"])
    instruction = _instruction(data_root, episodes, protocol["instruction_type"])
    authorization = ImageEgressAuthorization.operator_granted(
        assertion="RMBench formal Q1 source trajectory reflection",
        scope="rmbench_q1_frozen_gt_source_pool",
    )
    transport = _transport(gpt_url, authorization, max_images)
    reflector = VLMHierarchicalReflector(
        ExistingGPTMultimodalReflectorBackend(
            transport,
            authorization=authorization,
        )
    )
    completed: list[dict] = []
    rejected = 0
    for episode in episodes:
        chunks, descriptions, images = _load_prepared_episode(
            output_root,
            episode,
            expected_instruction=instruction,
            expected_available_state=(),
        )
        selected_descriptions, selected_images, selected_indices = (
            select_reflection_evidence(
                descriptions,
                images,
                max_images=max_images,
            )
        )
        episode_root = output_root / f"episode_{episode}"
        try:
            result = reflector.reflect(
                instruction=instruction,
                ordered_action_chunks=chunks,
                key_observations=selected_descriptions,
                available_state=(),
                images=selected_images,
            )
        except HPKV3ValidationError as exc:
            rejected += 1
            raw = reflector.last_raw_response
            if raw is not None:
                _write_json(episode_root / "rejected_raw_reflector_output.json", raw)
            _write_json(
                episode_root / "reflection_error.json",
                {
                    "status": "rejected by local HPK v3 validation",
                    "reason": str(exc),
                    "automatic_retry": False,
                },
            )
            completed.append(
                {
                    "episode": episode,
                    "status": "rejected",
                    "reflection_images": len(selected_images),
                    "selected_visual_boundaries": selected_indices,
                }
            )
            continue
        package = result.package.to_dict()
        atomized = atomize_package(result.package)
        _write_json(episode_root / "trajectory_knowledge_package.json", package)
        _write_json(episode_root / "atomic_knowledge.json", atomized)
        completed.append(
            {
                "episode": episode,
                "status": "accepted",
                "action_chunks": len(chunks),
                "reflection_images": len(selected_images),
                "selected_visual_boundaries": selected_indices,
                "subtasks": len(package["task_strategy"]["subtasks"]),
                "task_knowledge_units": len(atomized["task_knowledge"]),
                "action_knowledge_units": len(atomized["action_knowledge"]),
            }
        )
    _write_json(
        output_root / "reflection_summary.json",
        {
            "task": task["name"],
            "instruction": instruction,
            "model": "GPT-5.5 xhigh",
            "external_model_calls": len(episodes),
            "sam3_calls": 0,
            "automatic_retries": 0,
            "accepted": len(episodes) - rejected,
            "rejected": rejected,
            "trajectories": completed,
        },
    )
    if rejected:
        raise FormalSourceError(
            f"{rejected} of 10 source trajectories failed strict reflection"
        )


def consolidate_task(
    *,
    task: dict,
    output_root: Path,
    gpt_url: str,
    max_images: int,
) -> None:
    sources = [f"episode_{index}" for index in range(10)]
    missing = [
        source
        for source in sources
        if not (output_root / source / "atomic_knowledge.json").is_file()
    ]
    if missing:
        raise FormalSourceError(f"source package set is incomplete: {missing}")
    authorization = ImageEgressAuthorization.operator_granted(
        assertion="RMBench formal Q1 source semantic consolidation",
        scope="rmbench_q1_frozen_atomic_source_pool",
    )
    transport = _transport(gpt_url, authorization, max_images)
    consolidator = VLMKnowledgeConsolidator(
        ExistingGPTMultimodalReflectorBackend(
            transport,
            authorization=authorization,
        )
    )
    consolidated = output_root / "consolidated_store"
    build_merged_store(
        input_root=output_root,
        output_root=consolidated,
        report_path=output_root / "consolidation_report.json",
        consolidator=consolidator,
        response_path=output_root / "consolidation_response.json",
    )
    materialize_method_views(
        consolidated_store=consolidated,
        output_root=output_root / "method_views",
        source_collections=sources,
    )
    _write_json(
        output_root / "source_complete.json",
        {
            "task": task["name"],
            "source_trajectory_count": 10,
            "reflection_calls": 10,
            "consolidation_calls": 1,
            "sam3_calls": 0,
            "automatic_retries": 0,
            "method_views": ["off", "flat", "task", "action", "full"],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "reflect", "consolidate"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("/path/to/RMBench"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpt-url", default="http://127.0.0.1:9104")
    parser.add_argument("--max-images", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    inputs = validate_benchmark_inputs(
        protocol,
        repo_root=REPO_ROOT,
        asset_root=args.asset_root,
    )
    task = next((item for item in protocol["tasks"] if item["name"] == args.task), None)
    resolved = next((item for item in inputs if item["task"] == args.task), None)
    if task is None or resolved is None:
        raise FormalSourceError(f"task {args.task!r} is not in the frozen Q1 protocol")
    data_root = Path(resolved["data_root"])
    task_output = args.output_root / args.task
    if args.stage == "prepare":
        prepare_task(
            protocol=protocol,
            task=task,
            data_root=data_root,
            output_root=task_output,
        )
    elif args.stage == "reflect":
        reflect_task(
            protocol=protocol,
            task=task,
            data_root=data_root,
            output_root=task_output,
            gpt_url=args.gpt_url,
            max_images=args.max_images,
        )
    else:
        consolidate_task(
            task=task,
            output_root=task_output,
            gpt_url=args.gpt_url,
            max_images=args.max_images,
        )


if __name__ == "__main__":
    main()
