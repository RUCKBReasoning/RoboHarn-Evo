#!/usr/bin/env python3
"""Run discovery, validation, or one explicit action in copied LIBERO-PRO."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.libero_pro.integration import (  # noqa: E402
    PRO_SUITES,
    build_environment,
    discover_suites,
    discover_tasks,
    dry_run_provenance,
    reset_to_initial_state,
    task_by_id,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RoboHarn-Evo LIBERO-PRO runner. Discovery and dry-run do not import "
            "MuJoCo, robosuite, a renderer, or the copied simulator package."
        )
    )
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--list-suites", action="store_true")
    operation.add_argument("--list-tasks", action="store_true")
    operation.add_argument("--dry-run", action="store_true")
    operation.add_argument("--run-agent", action="store_true")
    operation.add_argument("--policy-preflight", action="store_true")
    operation.add_argument(
        "--policy-inference-preflight",
        action="store_true",
        help="run one policy inference from a real reset without env.step",
    )
    operation.add_argument(
        "--native-policy-off",
        action="store_true",
        help="run one LIBERO-native policy episode with HPK disabled",
    )
    operation.add_argument(
        "--agent-loop-off",
        action="store_true",
        help="run the RoboHarn-Evo high-level planner loop with native policy and HPK off",
    )
    operation.add_argument(
        "--agent-loop-task-hpk",
        action="store_true",
        help="run the same Agent loop with read-only HPK v3 Task Knowledge",
    )
    operation.add_argument(
        "--agent-loop-action-hpk",
        action="store_true",
        help="run the same Agent loop with read-only HPK v3 Action Knowledge",
    )
    operation.add_argument(
        "--agent-loop-task-action-hpk",
        action="store_true",
        help="run the Agent loop with read-only Task and Action Knowledge",
    )
    operation.add_argument(
        "--build-expert-task-store",
        action="store_true",
        help="reflect one real LIBERO expert trajectory into a task-only v3 Store",
    )
    operation.add_argument("--smoke-reset", action="store_true")
    operation.add_argument(
        "--action-json",
        metavar="JSON",
        help="execute exactly one caller-supplied native 7D LIBERO action",
    )
    parser.add_argument("--suite", default=PRO_SUITES[0])
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "libero_pro",
    )
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--custom-assets-root", type=Path)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--policy-checkpoint-dir", type=Path)
    parser.add_argument("--policy-source-ref")
    parser.add_argument("--policy-license-id")
    parser.add_argument("--policy-config-name", default="pi05_libero")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--policy-replan-steps", type=int, default=5)
    parser.add_argument("--policy-settle-steps", type=int, default=10)
    parser.add_argument("--planner-server-url")
    parser.add_argument("--planner-timeout-sec", type=int, default=600)
    parser.add_argument("--agent-max-planner-calls", type=int, default=8)
    parser.add_argument(
        "--agent-loop-profile",
        choices=("lite", "perception_memory"),
        default="lite",
        help="retain the lite loop or add SAM3/Scene Memory/effect verification",
    )
    parser.add_argument("--sam3-server-url")
    parser.add_argument("--perception-artifact-root", type=Path)
    parser.add_argument("--perception-timeout-sec", type=int, default=120)
    parser.add_argument("--perception-max-queries", type=int, default=3)
    parser.add_argument(
        "--perception-confidence-threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument("--hpk-task-store-root", type=Path)
    parser.add_argument("--hpk-source-domain", default="LIBERO-PRO")
    parser.add_argument(
        "--hpk-family-exhaustive-threshold",
        type=int,
        default=24,
        help=(
            "use exhaustive HPK retrieval at or below this eligible-unit count; "
            "smaller development values exercise the read-only Family route"
        ),
    )
    parser.add_argument("--expert-dataset-root", type=Path)
    parser.add_argument("--expert-episode-index", type=int, default=0)
    parser.add_argument("--reflection-window-steps", type=int, default=10)
    parser.add_argument("--reflection-max-images", type=int, default=16)
    parser.add_argument(
        "--mujoco-gl",
        choices=("osmesa", "egl", "glfw"),
        default="osmesa",
        help="renderer backend for explicit simulator operations (default: CPU osmesa)",
    )
    parser.add_argument(
        "--json-provenance",
        action="store_true",
        help="emit machine-readable JSON for discovery/dry-run",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.suite not in discover_suites(pro_only=False):
        parser.error(f"unknown suite: {args.suite}")
    for name in ("horizon", "camera_height", "camera_width"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.policy_replan_steps <= 0:
        parser.error("--policy-replan-steps must be positive")
    if args.policy_settle_steps < 0:
        parser.error("--policy-settle-steps must be non-negative")
    if args.planner_timeout_sec <= 0:
        parser.error("--planner-timeout-sec must be positive")
    if args.agent_max_planner_calls <= 0:
        parser.error("--agent-max-planner-calls must be positive")
    if args.hpk_family_exhaustive_threshold <= 0:
        parser.error("--hpk-family-exhaustive-threshold must be positive")
    if args.perception_timeout_sec <= 0:
        parser.error("--perception-timeout-sec must be positive")
    if not 1 <= args.perception_max_queries <= 3:
        parser.error("--perception-max-queries must be within [1, 3]")
    if not 0.0 <= args.perception_confidence_threshold <= 1.0:
        parser.error("--perception-confidence-threshold must be within [0, 1]")
    if args.expert_episode_index < 0:
        parser.error("--expert-episode-index must be non-negative")
    if args.reflection_window_steps <= 0:
        parser.error("--reflection-window-steps must be positive")
    if args.reflection_max_images <= 0:
        parser.error("--reflection-max-images must be positive")
    if args.config is not None:
        args.config = args.config.expanduser().resolve()
        if not args.config.is_file():
            parser.error(f"--config does not exist: {args.config}")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.assets_root is not None:
        args.assets_root = args.assets_root.expanduser().resolve()
    if args.custom_assets_root is not None:
        args.custom_assets_root = args.custom_assets_root.expanduser().resolve()
    if args.policy_checkpoint_dir is not None:
        args.policy_checkpoint_dir = args.policy_checkpoint_dir.expanduser().resolve()
    if args.hpk_task_store_root is not None:
        args.hpk_task_store_root = args.hpk_task_store_root.expanduser().resolve()
    if args.expert_dataset_root is not None:
        args.expert_dataset_root = args.expert_dataset_root.expanduser().resolve()
    if args.perception_artifact_root is not None:
        args.perception_artifact_root = (
            args.perception_artifact_root.expanduser().resolve()
        )
    operation_name = (
        "--policy-preflight"
        if args.policy_preflight
        else "--policy-inference-preflight"
        if args.policy_inference_preflight
        else "--native-policy-off"
        if args.native_policy_off
        else "--agent-loop-off"
        if args.agent_loop_off
        else "--agent-loop-task-hpk"
        if args.agent_loop_task_hpk
        else "--agent-loop-action-hpk"
        if args.agent_loop_action_hpk
        else "--agent-loop-task-action-hpk"
        if args.agent_loop_task_action_hpk
        else "--build-expert-task-store"
        if args.build_expert_task_store
        else "simulator operation"
    )
    if (
        args.policy_preflight
        or args.policy_inference_preflight
        or args.native_policy_off
        or args.agent_loop_off
        or args.agent_loop_task_hpk
        or args.agent_loop_action_hpk
        or args.agent_loop_task_action_hpk
    ):
        for field_name in (
            "policy_checkpoint_dir",
            "policy_source_ref",
            "policy_license_id",
        ):
            if not getattr(args, field_name):
                parser.error(
                    operation_name + " requires --" + field_name.replace("_", "-")
                )
    if (
        args.native_policy_off
        or args.policy_inference_preflight
        or args.agent_loop_off
        or args.agent_loop_task_hpk
        or args.agent_loop_action_hpk
        or args.agent_loop_task_action_hpk
        or args.build_expert_task_store
    ) and args.output_dir.exists():
        parser.error(
            operation_name
            + " requires a new --output-dir so an earlier run cannot be overwritten"
        )
    if (
        args.agent_loop_off
        or args.agent_loop_task_hpk
        or args.agent_loop_action_hpk
        or args.agent_loop_task_action_hpk
        or args.build_expert_task_store
    ) and not str(args.planner_server_url or "").strip():
        parser.error(operation_name + " requires --planner-server-url")
    if (
        args.agent_loop_task_hpk
        or args.agent_loop_action_hpk
        or args.agent_loop_task_action_hpk
    ) and args.hpk_task_store_root is None:
        parser.error("HPK Agent loop requires --hpk-task-store-root")
    if (
        args.agent_loop_task_hpk
        or args.agent_loop_action_hpk
        or args.agent_loop_task_action_hpk
    ) and not str(args.hpk_source_domain).strip():
        parser.error("HPK Agent loop requires non-empty --hpk-source-domain")
    if args.build_expert_task_store and args.expert_dataset_root is None:
        parser.error("--build-expert-task-store requires --expert-dataset-root")
    agent_loop = bool(
        args.agent_loop_off
        or args.agent_loop_task_hpk
        or args.agent_loop_action_hpk
        or args.agent_loop_task_action_hpk
    )
    if args.agent_loop_profile == "perception_memory":
        if not agent_loop:
            parser.error("perception_memory profile requires an Agent loop mode")
        if not str(args.sam3_server_url or "").strip():
            parser.error("perception_memory profile requires --sam3-server-url")
    elif args.sam3_server_url or args.perception_artifact_root is not None:
        parser.error("SAM3/perception options require perception_memory profile")


def _print(payload: Any, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                print(
                    f"{item.get('task_id', '-'):>3}  {item.get('suite', '')}/"
                    f"{item.get('name', '')}  ::  {item.get('instruction', '')}"
                )
            else:
                print(item)
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _native_action_contract(*, policy_passthrough: bool = False) -> Any:
    from roboharn_evo.benchmark_adapters.libero_pro import LiberoProActionContract

    return LiberoProActionContract(
        action_type="libero",
        shape=(7,),
        lower_bounds=(-1.0,) * 7,
        upper_bounds=(1.0,) * 7,
        bounds_handling=(
            "native_controller" if policy_passthrough else "adapter_validate"
        ),
        control_mode="OSC_POSE",
        translation_mode="normalized delta; robosuite output scale +/-0.05 m",
        rotation_representation="normalized delta axis-angle; output scale +/-0.5 rad",
        reference_frame="native robosuite OSC_POSE controller frame; no conversion",
        gripper_convention="-1=open, +1=close",
    )


def _capability_gate() -> tuple[str, ...]:
    from roboharn_evo.benchmark_adapters.libero_pro import (
        CANONICAL_CAMERA_KEYS,
        CANONICAL_PROPRIOCEPTION_KEYS,
        LiberoProCapabilities,
        assess_roboharn_agent_compatibility,
    )

    capabilities = LiberoProCapabilities(
        camera_keys=CANONICAL_CAMERA_KEYS,
        proprioception_keys=CANONICAL_PROPRIOCEPTION_KEYS,
        action=_native_action_contract(),
    )
    return assess_roboharn_agent_compatibility(capabilities).missing


def _parse_action(parser: argparse.ArgumentParser, raw: str) -> list[float]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        parser.error(f"--action-json is invalid JSON: {exc}")
    if not isinstance(value, list) or len(value) != 7:
        parser.error("--action-json must be a JSON list with exactly 7 numbers")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        parser.error("--action-json must contain only numbers")
    return [float(item) for item in value]


def _observed_value_contract(value: Any) -> dict[str, Any]:
    """Describe one observed array without serializing simulator state."""

    shape_value = getattr(value, "shape", None)
    shape = None
    if shape_value is not None:
        try:
            shape = [int(size) for size in shape_value]
        except (TypeError, ValueError):
            shape = None
    dtype_value = getattr(value, "dtype", None)
    return {
        "shape": shape,
        "dtype": None if dtype_value is None else str(dtype_value),
    }


def _observation_contract_payload(observation: Any) -> dict[str, Any]:
    """Project a neutral observation into non-oracle contract metadata."""

    return {
        "cameras": {
            key: _observed_value_contract(value)
            for key, value in observation.cameras.items()
        },
        "proprioception": {
            key: _observed_value_contract(value)
            for key, value in observation.proprioception.items()
        },
        "public_raw_keys": sorted(observation.raw),
        "capabilities": dict(observation.capabilities),
    }


def _run_simulator(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    provenance: dict[str, Any],
    *,
    policy_backend: Any | None = None,
    policy_report: dict[str, Any] | None = None,
    planner_backend: Any | None = None,
    task_hpk_runtime: Any | None = None,
    action_hpk_runtime: Any | None = None,
    perception_memory_runtime: Any | None = None,
) -> int:
    if args.assets_root is None:
        parser.error(
            "simulator operation requires --assets-root because the external "
            "LIBERO-PRO assets were not legally vendored"
        )
    if args.custom_assets_root is not None:
        os.environ["LIBERO_PRO_CUSTOM_ASSETS_ROOT"] = str(args.custom_assets_root)
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_dir = args.output_dir / ".liberopro"
    task = task_by_id(args.suite, args.task_id)

    env = build_environment(
        task,
        seed=args.seed,
        assets_root=args.assets_root,
        runtime_config_dir=runtime_config_dir,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        camera_depths=args.depth,
        horizon=args.horizon,
    )
    try:
        raw_observation = reset_to_initial_state(
            env,
            task,
            init_state_id=args.init_state_id,
        )
        from roboharn_evo.benchmark_adapters import ActionRequest
        from roboharn_evo.benchmark_adapters.libero_pro import LiberoProAdapter

        adapter = LiberoProAdapter(
            env,
            instruction=task.instruction,
            action_contract=_native_action_contract(
                policy_passthrough=(
                    args.native_policy_off
                    or args.agent_loop_off
                    or args.agent_loop_task_hpk
                    or args.agent_loop_action_hpk
                    or args.agent_loop_task_action_hpk
                )
            ),
            step_limit=args.horizon,
        )
        observation = adapter.reset(raw_observation)
        assert observation is not None
        result_payload: dict[str, Any] = {
            "reset": "passed",
            "instruction": observation.instruction,
            "camera_keys": sorted(observation.cameras),
            "proprioception_keys": sorted(observation.proprioception),
            "observation_contract": _observation_contract_payload(observation),
            "legacy_imgagent_capability_gaps": list(
                adapter.roboharn_capability_report(observation).missing
            ),
            "roboharn_agent_capability_gaps": (
                []
                if (
                    args.agent_loop_off
                    or args.agent_loop_task_hpk
                    or args.agent_loop_action_hpk
                    or args.agent_loop_task_action_hpk
                )
                else list(adapter.roboharn_capability_report(observation).missing)
            ),
        }
        if args.action_json is not None:
            action = _parse_action(parser, args.action_json)
            result = adapter.execute(ActionRequest(action=action, action_type="libero"))
            result_payload["action"] = {
                "executions": 1,
                "step_before": result.step_before,
                "step_after": result.step_after,
                "benchmark_success": result.episode_state.benchmark_success,
                "terminated": result.episode_state.terminated,
                "truncated": result.episode_state.truncated,
                "reward": result.episode_state.reward,
            }
            post_contract = _observation_contract_payload(result.post_observation)
            result_payload["post_observation_contract"] = post_contract
            result_payload["post_observation_contract_matches_reset"] = (
                post_contract == result_payload["observation_contract"]
            )
        elif args.policy_inference_preflight:
            if policy_backend is None or policy_report is None:
                raise RuntimeError("native policy backend was not preflighted")
            action_chunk = policy_backend.predict_native_action_chunk(
                observation,
                prompt=observation.instruction,
            )
            provenance["policy"] = policy_report
            provenance["hpk"] = {
                "mode": "off",
                "persistent_store_read": False,
                "persistent_store_write": False,
            }
            result_payload["policy_inference_preflight"] = {
                "policy_calls": 1,
                "motion_executed": False,
                "action_chunk_shape": list(action_chunk.shape),
                "action_chunk": action_chunk.tolist(),
            }
        elif (
            args.agent_loop_off
            or args.agent_loop_task_hpk
            or args.agent_loop_action_hpk
            or args.agent_loop_task_action_hpk
        ):
            if policy_backend is None or policy_report is None:
                raise RuntimeError("native policy backend was not preflighted")
            if planner_backend is None:
                raise RuntimeError("RoboHarn-Evo planner backend was not configured")
            from roboharn_evo.agent.benchmark_neutral_loop import (
                BenchmarkAgentLoopConfig,
                RoboHarnBenchmarkAgentLoop,
            )
            from roboharn_evo.benchmark_adapters.libero_pro import LiberoProAgentBridge

            trace_path = args.output_dir / "agent_loop_trace.jsonl"
            with trace_path.open("x", encoding="utf-8") as trace:

                def record_agent_event(event: Any) -> None:
                    payload = dict(event)
                    if payload.get("event") == "episode_start":
                        payload["benchmark_context"] = {
                            "suite": args.suite,
                            "task_id": args.task_id,
                            "init_state_id": args.init_state_id,
                            "seed": args.seed,
                            "action_horizon": args.horizon,
                            "planner_call_limit": args.agent_max_planner_calls,
                            "settle_steps": args.policy_settle_steps,
                            "replan_steps": args.policy_replan_steps,
                            "camera_height": args.camera_height,
                            "camera_width": args.camera_width,
                            "policy_config_name": args.policy_config_name,
                            "policy_device": args.policy_device,
                            "loop_profile": args.agent_loop_profile,
                        }
                    trace.write(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    trace.flush()

                agent_loop = RoboHarnBenchmarkAgentLoop(
                    planner=planner_backend,
                    executor=policy_backend,
                    adapter=adapter,
                    bridge=LiberoProAgentBridge(
                        settle_steps=args.policy_settle_steps,
                        replan_steps=args.policy_replan_steps,
                    ),
                    config=BenchmarkAgentLoopConfig(
                        max_actions=args.horizon,
                        max_planner_calls=args.agent_max_planner_calls,
                    ),
                    task_hpk_runtime=task_hpk_runtime,
                    action_hpk_runtime=action_hpk_runtime,
                    perception_memory_runtime=perception_memory_runtime,
                    event_sink=record_agent_event,
                )
                agent_result = agent_loop.run_episode(
                    task=observation.instruction,
                    initial_observation=observation,
                )
            provenance["policy"] = policy_report
            task_enabled = bool(
                args.agent_loop_task_hpk or args.agent_loop_task_action_hpk
            )
            action_enabled = bool(
                args.agent_loop_action_hpk or args.agent_loop_task_action_hpk
            )
            store_enabled = task_enabled or action_enabled
            provenance["hpk"] = {
                "mode": (
                    "full"
                    if task_enabled and action_enabled
                    else "task_only"
                    if task_enabled
                    else "action_only"
                    if action_enabled
                    else "off"
                ),
                "task_knowledge_enabled": task_enabled,
                "action_knowledge_enabled": action_enabled,
                "persistent_store_read": store_enabled,
                "persistent_store_write": False,
                "store_root": (
                    str(args.hpk_task_store_root) if store_enabled else None
                ),
                "source_domain": (
                    str(args.hpk_source_domain).strip() if store_enabled else None
                ),
                "loop_profile": args.agent_loop_profile,
                "perception_memory_enabled": (
                    args.agent_loop_profile == "perception_memory"
                ),
                "family_routing": (
                    {
                        "exhaustive_threshold": args.hpk_family_exhaustive_threshold,
                        "catalog_expected": True,
                    }
                    if store_enabled
                    else None
                ),
            }
            provenance["agent"]["end_to_end_supported"] = True
            provenance["agent"]["capability_gate"] = (
                "passed through additive benchmark-neutral Agent bridge"
            )
            result_payload["roboharn_agent_loop"] = agent_result.to_dict()
            result_payload["agent_loop_trace"] = str(trace_path)
        elif args.native_policy_off:
            if policy_backend is None or policy_report is None:
                raise RuntimeError("native policy backend was not preflighted")
            from roboharn_evo.benchmark_adapters.libero_pro import (
                run_libero_native_off_episode,
            )

            trace_path = args.output_dir / "native_policy_trace.jsonl"
            with trace_path.open("x", encoding="utf-8") as trace:

                def record_action(record: Any) -> None:
                    trace.write(
                        json.dumps(
                            {
                                "schema": "roboharn_evo/libero_native_action/v1",
                                **record.to_dict(),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    trace.flush()

                rollout = run_libero_native_off_episode(
                    adapter=adapter,
                    policy=policy_backend,
                    initial_observation=observation,
                    prompt=observation.instruction,
                    max_steps=args.horizon,
                    replan_steps=args.policy_replan_steps,
                    settle_steps=args.policy_settle_steps,
                    action_observer=record_action,
                )
            provenance["policy"] = policy_report
            provenance["hpk"] = {
                "mode": "off",
                "persistent_store_read": False,
                "persistent_store_write": False,
            }
            result_payload["native_policy_off"] = rollout.to_dict()
            result_payload["native_policy_trace"] = str(trace_path)
        provenance["simulator"] = result_payload
        provenance["simulator_imported"] = True
        provenance_path = args.output_dir / "provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _print(provenance, json_output=True)
        return 0
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    if args.list_suites:
        _print(list(discover_suites()), json_output=args.json_provenance)
        return 0
    if args.list_tasks:
        tasks = [task.as_json() for task in discover_tasks(args.suite)]
        _print(tasks, json_output=args.json_provenance)
        return 0

    provenance = dry_run_provenance(
        suite=args.suite,
        task_id=args.task_id,
        seed=args.seed,
        config=args.config,
        output_dir=args.output_dir,
        assets_root=args.assets_root,
    )
    provenance["run"].update(
        {
            "task_id": args.task_id,
            "init_state_id": args.init_state_id,
            "horizon": args.horizon,
            "camera_height": args.camera_height,
            "camera_width": args.camera_width,
            "depth_enabled": bool(args.depth),
            "policy_settle_steps": args.policy_settle_steps,
            "policy_replan_steps": args.policy_replan_steps,
            "agent_max_planner_calls": args.agent_max_planner_calls,
        }
    )
    if args.dry_run:
        _print(provenance, json_output=bool(args.json_provenance))
        return 0
    if args.run_agent:
        missing = _capability_gate()
        provenance["agent"]["missing_capabilities"] = list(missing)
        _print(provenance, json_output=True)
        print(
            "RoboHarn-Evo Agent bridge refused: LIBERO-PRO does not expose the RMBench "
            f"dual-arm/qpos contract. Missing: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    if args.build_expert_task_store:
        from roboharn_evo.benchmark_adapters.libero_pro import (
            build_task_only_store_from_expert,
        )

        result = build_task_only_store_from_expert(
            dataset_root=args.expert_dataset_root,
            episode_index=args.expert_episode_index,
            output_dir=args.output_dir,
            planner_url=str(args.planner_server_url),
            timeout_sec=args.planner_timeout_sec,
            window_steps=args.reflection_window_steps,
            max_images=args.reflection_max_images,
        )
        _print(result.to_dict(), json_output=True)
        return 0
    policy_backend = None
    policy_report = None
    planner_backend = None
    task_hpk_runtime = None
    action_hpk_runtime = None
    perception_memory_runtime = None
    if (
        args.policy_preflight
        or args.policy_inference_preflight
        or args.native_policy_off
        or args.agent_loop_off
        or args.agent_loop_task_hpk
        or args.agent_loop_action_hpk
        or args.agent_loop_task_action_hpk
    ):
        from roboharn_evo.benchmark_adapters.libero_pro import (
            LiberoPi05PolicyBackend,
            LiberoPi05PolicyConfig,
        )

        backend = LiberoPi05PolicyBackend(
            LiberoPi05PolicyConfig(
                checkpoint_dir=args.policy_checkpoint_dir,
                source_ref=args.policy_source_ref,
                license_id=args.policy_license_id,
                config_name=args.policy_config_name,
                device=args.policy_device,
            )
        )
        policy_backend = backend
        policy_report = backend.preflight()
    if (
        args.agent_loop_off
        or args.agent_loop_task_hpk
        or args.agent_loop_action_hpk
        or args.agent_loop_task_action_hpk
    ):
        from roboharn_evo.benchmark_adapters.libero_pro import LIBERO_AGENT_PLANNER_PROMPT
        from roboharn_evo.models.backend_factory import build_planner_backend

        if args.agent_loop_task_action_hpk:
            from roboharn_evo.benchmark_adapters.libero_pro import (
                build_task_action_runtimes,
            )

            runtimes = build_task_action_runtimes(
                store_root=args.hpk_task_store_root,
                planner_url=str(args.planner_server_url),
                source_domain=str(args.hpk_source_domain),
                timeout_sec=args.planner_timeout_sec,
                family_exhaustive_threshold=(args.hpk_family_exhaustive_threshold),
            )
            task_hpk_runtime = runtimes.task
            action_hpk_runtime = runtimes.action
        elif args.agent_loop_action_hpk:
            from roboharn_evo.benchmark_adapters.libero_pro import (
                build_action_only_runtime,
            )

            action_hpk_runtime = build_action_only_runtime(
                store_root=args.hpk_task_store_root,
                planner_url=str(args.planner_server_url),
                source_domain=str(args.hpk_source_domain),
                timeout_sec=args.planner_timeout_sec,
                family_exhaustive_threshold=(args.hpk_family_exhaustive_threshold),
            )
        elif args.agent_loop_task_hpk:
            from roboharn_evo.benchmark_adapters.libero_pro import (
                build_transfer_task_only_runtime,
            )

            task_hpk_runtime = build_transfer_task_only_runtime(
                store_root=args.hpk_task_store_root,
                planner_url=str(args.planner_server_url),
                timeout_sec=args.planner_timeout_sec,
                family_exhaustive_threshold=(args.hpk_family_exhaustive_threshold),
            )

        planner_config = {
            "backend": "agent_api",
            "agent_api": {
                "server_url": str(args.planner_server_url),
                "timeout_sec": args.planner_timeout_sec,
                "prompt_template": LIBERO_AGENT_PLANNER_PROMPT,
                "auth_token": "",
                "auth_header": "Authorization",
                "extra_headers": {},
                "extra_body": {},
            },
        }
        planner_backend = (
            build_planner_backend(planner_config, hpk_runtime=task_hpk_runtime)
            if task_hpk_runtime is not None
            else build_planner_backend(planner_config)
        )
        if args.agent_loop_profile == "perception_memory":
            from roboharn_evo.benchmark_adapters.libero_pro import (
                LiberoPerceptionMemoryRuntime,
                LiberoSAM3PerceptionClient,
            )

            perception_root = (
                args.perception_artifact_root
                if args.perception_artifact_root is not None
                else args.output_dir / "perception"
            )
            perception_memory_runtime = LiberoPerceptionMemoryRuntime(
                perception=LiberoSAM3PerceptionClient(
                    planner_service_url=str(args.planner_server_url),
                    sam3_service_url=str(args.sam3_server_url),
                    artifact_root=perception_root,
                    timeout_sec=args.perception_timeout_sec,
                    max_queries=args.perception_max_queries,
                    confidence_threshold=args.perception_confidence_threshold,
                )
            )
    if args.policy_preflight:
        _print(policy_report, json_output=True)
        return 0
    return _run_simulator(
        parser,
        args,
        provenance,
        policy_backend=policy_backend,
        policy_report=policy_report,
        planner_backend=planner_backend,
        task_hpk_runtime=task_hpk_runtime,
        action_hpk_runtime=action_hpk_runtime,
        perception_memory_runtime=perception_memory_runtime,
    )


if __name__ == "__main__":
    raise SystemExit(main())
