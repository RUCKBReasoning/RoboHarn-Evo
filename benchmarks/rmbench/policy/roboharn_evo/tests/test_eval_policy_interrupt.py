from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import script.eval_policy as eval_module

from policy.roboharn_evo.scripts.check_pure_tool_control_early_stop import analyze_trace


class ObservationHandoffTest(unittest.TestCase):
    def test_recovery_observation_is_used_before_environment_capture(self) -> None:
        class FakeEnv:
            take_action_cnt = 4

            def __init__(self) -> None:
                self.get_obs_calls = 0

            def get_obs(self) -> dict:
                self.get_obs_calls += 1
                return {"source": "environment"}

        cached = {"source": "recovery"}
        queued = [cached, None]
        model = SimpleNamespace(
            consume_recovery_observation=lambda _env: queued.pop(0),
        )
        env = FakeEnv()
        with tempfile.TemporaryDirectory(
            prefix="roboharn_observation_handoff_"
        ) as directory:
            trace = Path(directory) / "trace.jsonl"
            first = eval_module._acquire_eval_observation(
                env,
                model,
                trace,
                episode_id=0,
                seed=100002,
            )
            second = eval_module._acquire_eval_observation(
                env,
                model,
                trace,
                episode_id=0,
                seed=100002,
            )
            records = [
                json.loads(line)
                for line in trace.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(first, cached)
        self.assertEqual(second, {"source": "environment"})
        self.assertEqual(env.get_obs_calls, 1)
        self.assertEqual(records[0]["event"], "observation_acquire_cache_hit")
        self.assertEqual(records[1]["event"], "observation_acquire_start")
        self.assertEqual(records[2]["event"], "observation_acquire_end")

    def test_no_progress_budget_is_a_capability_failure(self) -> None:
        validity = eval_module._classify_episode_validity(
            success=False,
            interrupted=False,
            episode_error=None,
            agent_status={
                "terminal_failure": True,
                "terminal_failure_reason": (
                    "pure_tool_control_max_no_progress_control_turns_exhausted:4"
                ),
            },
        )

        self.assertEqual(validity["label"], "valid_task_failure")
        self.assertTrue(validity["benchmark_denominator_eligible"])


class FailureBoundaryReplayConfigTest(unittest.TestCase):
    def test_capture_and_restore_configuration_are_explicit(self) -> None:
        self.assertEqual(
            eval_module._resolve_failure_boundary_replay_config({}),
            {"mode": "off"},
        )
        with tempfile.TemporaryDirectory(
            prefix="roboharn_failure_boundary_config_"
        ) as directory:
            path = Path(directory) / "boundary.json"
            capture = eval_module._resolve_failure_boundary_replay_config(
                {
                    "agent": {
                        "failure_boundary_replay": {
                            "mode": "capture",
                            "path": str(path),
                            "capture_at_or_after_env_step": 12,
                            "stop_after_capture": True,
                        }
                    }
                }
            )
            restore = eval_module._resolve_failure_boundary_replay_config(
                {
                    "agent": {
                        "failure_boundary_replay": {
                            "mode": "restore",
                            "path": str(path),
                        }
                    }
                }
            )

        self.assertEqual(capture["mode"], "capture")
        self.assertEqual(capture["capture_at_or_after_env_step"], 12)
        self.assertTrue(capture["stop_after_capture"])
        self.assertEqual(restore, {"mode": "restore", "path": path.resolve()})

    def test_capture_bank_configuration_and_summary_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="roboharn_failure_boundary_bank_config_"
        ) as directory:
            output_dir = Path(directory) / "bank"
            config = eval_module._resolve_failure_boundary_replay_config(
                {
                    "agent": {
                        "failure_boundary_replay": {
                            "mode": "capture_bank",
                            "output_dir": str(output_dir),
                            "min_boundaries": 2,
                            "max_boundaries": 4,
                            "capture_at_or_after_env_step": 5,
                            "stop_after_bank_full": True,
                            "label_prefix": "verified transport boundary",
                        }
                    }
                }
            )
            summary = eval_module._failure_boundary_episode_summary(
                config=config,
                restored_boundary=None,
                captured_boundary=None,
                captured_boundaries=[
                    {"path": str(output_dir / "first.json"), "env_step": 6},
                    {"path": str(output_dir / "second.json"), "env_step": 20},
                ],
                task_name="generic task",
                seed=4,
                instruction="complete the task",
                result="Fail",
                success=False,
                failure_reason="failure_boundary_bank_full",
                final_env_step=20,
                max_reward=0.0,
            )

        self.assertEqual(config["mode"], "capture_bank")
        self.assertEqual(config["output_dir"], output_dir.resolve())
        self.assertEqual(config["min_boundaries"], 2)
        self.assertEqual(config["max_boundaries"], 4)
        self.assertEqual(summary["capture_count"], 2)
        self.assertEqual(summary["requested"]["max_boundaries"], 4)
        self.assertEqual(len(summary["boundaries"]), 2)

    def test_restore_helper_checks_exact_task_seed_and_instruction(self) -> None:
        boundary = SimpleNamespace(
            label="failed effect",
            reason="effect not verified",
            environment={"task_state": {"take_action_cnt": 12}},
        )
        report = {"matches": True, "mismatches": []}
        with (
            patch(
                "roboharn_evo.agent.failure_boundary_replay.load_failure_boundary",
                return_value=boundary,
            ) as load,
            patch(
                "roboharn_evo.agent.failure_boundary_replay.restore_failure_boundary",
                return_value=report,
            ) as restore,
        ):
            result = eval_module._restore_failure_boundary_before_episode(
                object(),
                object(),
                {"mode": "restore", "path": Path("/tmp/boundary.json")},
                task_name="generic task",
                seed=4,
                instruction="perform the task",
            )

        load.assert_called_once_with(Path("/tmp/boundary.json"))
        self.assertEqual(restore.call_args.kwargs["expected_task"], "generic task")
        self.assertEqual(restore.call_args.kwargs["expected_seed"], 4)
        self.assertEqual(
            restore.call_args.kwargs["expected_instruction"], "perform the task"
        )
        self.assertEqual(result["boundary_env_step"], 12)

    def test_restore_summary_reports_only_post_boundary_actions(self) -> None:
        summary = eval_module._failure_boundary_episode_summary(
            config={"mode": "restore"},
            restored_boundary={
                "path": "/tmp/boundary.json",
                "label": "after failed grasp",
                "reason": "attachment was not verified",
                "boundary_env_step": 12,
                "match": {"matches": True, "mismatches": []},
            },
            captured_boundary=None,
            task_name="generic task",
            seed=7,
            instruction="complete the task",
            result="Success",
            success=True,
            failure_reason="",
            final_env_step=19,
            max_reward=1.0,
        )

        self.assertEqual(summary["mode"], "restore")
        self.assertEqual(summary["outcome"]["additional_environment_actions"], 7)
        self.assertTrue(summary["boundary"]["state_match"]["matches"])

    def test_eval_loop_restores_before_first_action_and_writes_readable_result(
        self,
    ) -> None:
        def decorator(_policy_name: str, function_name: str):
            if function_name == "eval":

                def evaluate(task_env, _model, _observation):
                    task_env.take_action_cnt += 2
                    task_env.eval_success = True

                return evaluate
            if function_name == "reset_model":
                return lambda _model: None
            raise AssertionError(function_name)

        def restore(task_env, *_args, **_kwargs):
            task_env.take_action_cnt = 4
            return {
                "path": "/tmp/boundary.json",
                "label": "after failed grasp",
                "reason": "attachment was not verified",
                "boundary_env_step": 4,
                "match": {"matches": True, "mismatches": []},
            }

        with (
            tempfile.TemporaryDirectory(
                prefix="roboharn_failure_boundary_eval_"
            ) as directory,
            patch.object(
                eval_module,
                "eval_function_decorator",
                side_effect=decorator,
            ),
            patch.object(
                eval_module,
                "generate_episode_descriptions",
                return_value=[{"seen": ["test instruction"]}],
            ),
            patch.object(
                eval_module,
                "_restore_failure_boundary_before_episode",
                side_effect=restore,
            ),
            patch.object(
                eval_module,
                "_finalize_rollout_visualization",
                return_value=None,
            ),
        ):
            save_dir = Path(directory)
            result = eval_module.eval_policy(
                "fake_task",
                FakeTaskEnv(),
                {
                    "task_name": "fake_task",
                    "task_config": "fake_config",
                    "policy_name": "fake_policy",
                    "ckpt_setting": "fake_ckpt",
                    "clear_cache_freq": 1,
                    "render_freq": 0,
                },
                object(),
                7,
                test_num=1,
                instruction_type="seen",
                save_dir=save_dir,
                failure_boundary_replay={
                    "mode": "restore",
                    "path": Path("/tmp/boundary.json"),
                },
            )
            summary = json.loads(
                (
                    save_dir / "episode_0000_rollout" / "failure_boundary_result.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(result[1], 1)
        self.assertEqual(summary["outcome"]["final_env_step"], 6)
        self.assertEqual(summary["outcome"]["additional_environment_actions"], 2)

    def test_eval_capture_records_every_environment_action_from_step_zero(
        self,
    ) -> None:
        recorded_prefix = None
        trailing_observations = None

        def decorator(_policy_name: str, function_name: str):
            if function_name == "eval":

                def evaluate(task_env, _model, _observation):
                    task_env.take_action(
                        [0.25],
                        action_type="qpos",
                        active_arm="left",
                    )

                return evaluate
            if function_name == "reset_model":
                return lambda _model: None
            raise AssertionError(function_name)

        def capture(_task_env, _model, _config, **kwargs):
            nonlocal recorded_prefix, trailing_observations
            recorded_prefix = kwargs["action_prefix"]
            trailing_observations = kwargs["observation_calls_after_last_action"]
            return {
                "path": "/tmp/boundary.json",
                "label": "captured boundary",
                "reason": "test",
                "env_step": 1,
                "actions_recorded": len(recorded_prefix),
            }

        with (
            tempfile.TemporaryDirectory(
                prefix="roboharn_failure_boundary_capture_eval_"
            ) as directory,
            patch.object(
                eval_module,
                "eval_function_decorator",
                side_effect=decorator,
            ),
            patch.object(
                eval_module,
                "generate_episode_descriptions",
                return_value=[{"seen": ["test instruction"]}],
            ),
            patch.object(
                eval_module,
                "_capture_failure_boundary_after_turn",
                side_effect=capture,
            ),
            patch.object(
                eval_module,
                "_finalize_rollout_visualization",
                return_value=None,
            ),
        ):
            eval_module.eval_policy(
                "fake_task",
                FakeTaskEnv(),
                {
                    "task_name": "fake_task",
                    "task_config": "fake_config",
                    "policy_name": "fake_policy",
                    "ckpt_setting": "fake_ckpt",
                    "clear_cache_freq": 1,
                    "render_freq": 0,
                },
                object(),
                7,
                test_num=1,
                instruction_type="seen",
                save_dir=Path(directory),
                failure_boundary_replay={
                    "mode": "capture",
                    "path": Path(directory) / "boundary.json",
                    "capture_at_or_after_env_step": 1,
                    "stop_after_capture": True,
                    "label": "captured boundary",
                },
            )

        self.assertIsNotNone(recorded_prefix)
        self.assertEqual(len(recorded_prefix), 1)
        self.assertEqual(recorded_prefix[0]["env_step_before"], 0)
        self.assertEqual(recorded_prefix[0]["env_step_after"], 1)
        self.assertEqual(recorded_prefix[0]["observation_calls_before"], 1)
        self.assertEqual(recorded_prefix[0]["action_type"], "qpos")
        self.assertEqual(recorded_prefix[0]["active_arm"], "left")
        self.assertEqual(trailing_observations, 0)

    def test_eval_capture_bank_saves_one_confirmed_holding_transaction(
        self,
    ) -> None:
        captured_paths = []

        def decorator(_policy_name: str, function_name: str):
            if function_name == "eval":

                def evaluate(task_env, _model, _observation):
                    task_env.take_action(
                        [0.25],
                        action_type="qpos",
                        active_arm="left",
                    )

                return evaluate
            if function_name == "reset_model":
                return lambda _model: None
            raise AssertionError(function_name)

        def capture(_task_env, _model, config, **kwargs):
            captured_paths.append(config["path"])
            return {
                "path": str(config["path"]),
                "label": config["label"],
                "reason": "confirmed holding test",
                "env_step": 1,
                "actions_recorded": len(kwargs["action_prefix"]),
            }

        arm_state = {
            "phase": "holding_confirmed",
            "held_instance_id": "object-ref",
            "holding_confirmed": True,
            "transport_authorized": True,
            "grasp_attempt_nonce": "attempt-ref",
        }
        memory_state = SimpleNamespace(
            active_skill=SimpleNamespace(skill_id="skill-ref"),
            working=SimpleNamespace(
                manipulation_state={"left": arm_state},
            ),
        )
        agent = SimpleNamespace(
            memory_store=SimpleNamespace(state=memory_state),
            _pure_tool_control_no_progress_control_turns=1,
            _pure_tool_control_empty_plan_turns=0,
            _blocked_grounded_setups=set(),
            _grounded_setup_failures={},
        )
        model = SimpleNamespace(session=SimpleNamespace(agent=agent))

        with (
            tempfile.TemporaryDirectory(
                prefix="roboharn_failure_boundary_bank_eval_"
            ) as directory,
            patch.object(
                eval_module,
                "eval_function_decorator",
                side_effect=decorator,
            ),
            patch.object(
                eval_module,
                "generate_episode_descriptions",
                return_value=[{"seen": ["test instruction"]}],
            ),
            patch.object(
                eval_module,
                "_capture_failure_boundary_after_turn",
                side_effect=capture,
            ),
            patch.object(
                eval_module,
                "_finalize_rollout_visualization",
                return_value=None,
            ),
        ):
            root = Path(directory)
            config = eval_module._resolve_failure_boundary_replay_config(
                {
                    "agent": {
                        "failure_boundary_replay": {
                            "mode": "capture_bank",
                            "output_dir": str(root / "bank"),
                            "min_boundaries": 1,
                            "max_boundaries": 1,
                            "stop_after_bank_full": True,
                        }
                    }
                }
            )
            eval_module.eval_policy(
                "fake_task",
                FakeTaskEnv(),
                {
                    "task_name": "fake_task",
                    "task_config": "fake_config",
                    "policy_name": "fake_policy",
                    "ckpt_setting": "fake_ckpt",
                    "clear_cache_freq": 1,
                    "render_freq": 0,
                },
                model,
                7,
                test_num=1,
                instruction_type="seen",
                save_dir=root,
                failure_boundary_replay=config,
            )
            summary = json.loads(
                (
                    root / "episode_0000_rollout" / "failure_boundary_result.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(len(captured_paths), 1)
        self.assertEqual(
            captured_paths[0].name,
            "boundary_seed_000007_index_00_step_0001.json",
        )
        self.assertEqual(summary["mode"], "capture_bank")
        self.assertEqual(summary["capture_count"], 1)
        self.assertEqual(
            summary["boundaries"][0]["holding_boundary"]["held_instance_ref"],
            "object-ref",
        )


class RuntimeProvenanceArchiveTest(unittest.TestCase):
    @staticmethod
    def _manifest() -> dict:
        return {
            "schema_version": 1,
            "git_head": "a" * 40,
            "git_dirty_for_runtime_paths": True,
            "tracked_runtime_diff_sha256": "b" * 64,
            "runtime_tree_sha256": "c" * 64,
            "runtime_file_count": 3,
            "runtime_paths": ["policy/roboharn_evo/agent"],
            "runtime_file_hashes_sha256": {
                "policy/roboharn_evo/agent/runtime.py": "d" * 64,
            },
            "secrets_recorded": False,
        }

    def test_nonformal_legacy_manifest_is_validated_archived_and_summarized(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_eval_provenance_") as directory:
            root = Path(directory)
            source = root / "source.json"
            save_dir = root / "result"
            save_dir.mkdir()
            source.write_text(json.dumps(self._manifest()) + "\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "ROBOHARN_EVO_FORMAL_PROTOCOL": "0",
                    "ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION": "0",
                    "ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH": str(source),
                },
                clear=True,
            ):
                summary = eval_module._archive_runtime_provenance(save_dir)

            archive = save_dir / "runtime_provenance.json"
            self.assertEqual(archive.read_bytes(), source.read_bytes())
            self.assertFalse(summary["formal_protocol"])
            self.assertEqual(summary["formal_protocol_version"], 0)
            self.assertTrue(summary["recorded"])
            self.assertEqual(summary["archive_file"], "runtime_provenance.json")
            self.assertEqual(summary["runtime_tree_sha256"], "c" * 64)
        self.assertEqual(len(summary["manifest_sha256"]), 64)

    def test_service_identity_is_summarized_into_episode_provenance(self) -> None:
        manifest = self._manifest()
        manifest["agent_service_identity"] = {
            "service_url": "http://127.0.0.1:9105",
            "backend": "openai",
            "provider": "local-vllm",
            "model": "Qwen/Qwen3.5-397B-A17B-FP8",
            "api_mode": "chat",
            "reasoning_effort": "",
            "thinking_mode": "enabled",
            "fallback_enabled": False,
        }
        manifest["agent_service_identity_sha256"] = "e" * 64
        with tempfile.TemporaryDirectory(
            prefix="roboharn_eval_service_identity_"
        ) as directory:
            root = Path(directory)
            source = root / "source.json"
            save_dir = root / "result"
            save_dir.mkdir()
            source.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "ROBOHARN_EVO_FORMAL_PROTOCOL": "0",
                    "ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION": "0",
                    "ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH": str(source),
                },
                clear=True,
            ):
                summary = eval_module._archive_runtime_provenance(save_dir)

        self.assertEqual(summary["agent_service_identity_sha256"], "e" * 64)
        self.assertEqual(
            summary["agent_service_identity"]["service_url"],
            "http://127.0.0.1:9105",
        )
        self.assertIs(summary["agent_service_identity"]["fallback_enabled"], False)

    def test_formal_protocol_rejects_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="roboharn_eval_provenance_missing_"
        ) as directory:
            with patch.dict(
                os.environ,
                {
                    "ROBOHARN_EVO_FORMAL_PROTOCOL": "1",
                    "ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION": "3",
                },
                clear=True,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "requires a runtime provenance manifest"
                ):
                    eval_module._archive_runtime_provenance(Path(directory))

    def test_formal_protocol_requires_current_explicit_version(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="roboharn_eval_protocol_version_"
        ) as directory:
            for version in ("0", "1", "2", "4"):
                with (
                    self.subTest(version=version),
                    patch.dict(
                        os.environ,
                        {
                            "ROBOHARN_EVO_FORMAL_PROTOCOL": "1",
                            "ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION": version,
                        },
                        clear=True,
                    ),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION=3",
                    ):
                        eval_module._archive_runtime_provenance(Path(directory))

    def test_nonformal_generic_eval_can_run_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="roboharn_eval_provenance_generic_"
        ) as directory:
            with patch.dict(os.environ, {}, clear=True):
                summary = eval_module._archive_runtime_provenance(Path(directory))

        self.assertEqual(summary, {"formal_protocol": False, "recorded": False})


class FakeTaskEnv:
    def __init__(self) -> None:
        self.plan_success = True
        self.eval_success = False
        self.max_reward = 0.0
        self.step_lim = 10
        self.take_action_cnt = 0
        self.eval_video_path = None
        self.render_freq = 0
        self.test_num = 0

    def setup_demo(self, **_kwargs) -> None:
        self.eval_success = False
        self.take_action_cnt = 0

    def play_once(self) -> dict:
        return {"info": {}}

    def close_env(self, **_kwargs) -> None:
        return None

    def check_success(self) -> bool:
        return True

    def set_instruction(self, *, instruction: str) -> None:
        self.instruction = instruction

    def get_obs(self) -> dict:
        return {}

    def take_action(self, _action, action_type="qpos", active_arm=None) -> None:
        del action_type, active_arm
        if self.take_action_cnt < self.step_lim and not self.eval_success:
            self.take_action_cnt += 1


class EvalPolicyInterruptTest(unittest.TestCase):
    def _run_eval(
        self,
        fake_eval,
        *,
        model=None,
        runtime_provenance: dict | None = None,
    ) -> tuple[BaseException | None, object, list[dict], dict]:
        old_decorator = eval_module.eval_function_decorator
        old_descriptions = eval_module.generate_episode_descriptions
        old_finalize = eval_module._finalize_rollout_visualization

        def fake_decorator(_policy_name: str, function_name: str):
            if function_name == "eval":
                return fake_eval
            if function_name == "reset_model":
                return lambda _model: None
            raise AssertionError(function_name)

        try:
            eval_module.eval_function_decorator = fake_decorator
            eval_module.generate_episode_descriptions = lambda *_args, **_kwargs: [
                {"seen": ["test instruction"]}
            ]
            eval_module._finalize_rollout_visualization = lambda *_args, **_kwargs: None

            with tempfile.TemporaryDirectory(prefix="roboharn_eval_interrupt_") as temp_dir:
                save_dir = Path(temp_dir)
                args = {
                    "task_name": "fake_task",
                    "task_config": "fake_config",
                    "policy_name": "fake_policy",
                    "ckpt_setting": "fake_ckpt",
                    "clear_cache_freq": 1,
                    "render_freq": 0,
                }
                if runtime_provenance is not None:
                    args["runtime_provenance"] = dict(runtime_provenance)

                raised = None
                result = None
                try:
                    result = eval_module.eval_policy(
                        "fake_task",
                        FakeTaskEnv(),
                        args,
                        object() if model is None else model,
                        100000,
                        test_num=1,
                        instruction_type="seen",
                        save_dir=save_dir,
                    )
                except BaseException as error:
                    raised = error

                trace_path = save_dir / "episode_0000_agent_trace.jsonl"
                records = [
                    json.loads(line)
                    for line in trace_path.read_text(encoding="utf-8").splitlines()
                ]
                report = analyze_trace(trace_path)
                return raised, result, records, report
        finally:
            eval_module.eval_function_decorator = old_decorator
            eval_module.generate_episode_descriptions = old_descriptions
            eval_module._finalize_rollout_visualization = old_finalize

    def test_renderer_contract_failure_is_not_retried_or_reclassified_as_seed_error(
        self,
    ) -> None:
        class RendererFailureEnv(FakeTaskEnv):
            def __init__(self) -> None:
                super().__init__()
                self.setup_seeds: list[int] = []
                self.close_calls = 0

            def setup_demo(self, **kwargs) -> None:
                self.setup_seeds.append(kwargs["seed"])
                raise eval_module.RendererDeviceContractError(
                    "strict renderer-device validation failed",
                    stage="device_validation",
                    provenance={
                        "schema": "rmbench/renderer_device_binding/v1",
                        "validation_passed": False,
                    },
                )

            def close_env(self, **_kwargs) -> None:
                self.close_calls += 1

        environment = RendererFailureEnv()

        def fake_decorator(_policy_name: str, function_name: str):
            if function_name == "eval":
                return lambda *_args, **_kwargs: None
            if function_name == "reset_model":
                return lambda _model: None
            raise AssertionError(function_name)

        args = {
            "task_name": "fake_task",
            "policy_name": "fake_policy",
            "clear_cache_freq": 1,
            "render_freq": 7,
        }
        with (
            patch.object(
                eval_module,
                "eval_function_decorator",
                side_effect=fake_decorator,
            ),
            self.assertRaises(eval_module.RendererDeviceContractError) as raised,
        ):
            eval_module.eval_policy(
                "fake_task",
                environment,
                args,
                object(),
                100000,
                test_num=1,
                instruction_type="seen",
            )

        self.assertEqual(raised.exception.stage, "device_validation")
        self.assertEqual(environment.setup_seeds, [100000])
        self.assertEqual(environment.close_calls, 1)
        self.assertEqual(environment.test_num, 0)
        self.assertEqual(environment.suc, 0)
        self.assertEqual(args["render_freq"], 0)

    def test_sigterm_writes_interrupt_and_terminal_episode_end(self) -> None:
        old_handler = signal.getsignal(signal.SIGTERM)
        old_signal_name = eval_module._TERMINATION_SIGNAL_NAME

        def fake_eval(_task_env, _model, _observation) -> None:
            os.kill(os.getpid(), signal.SIGTERM)

        try:
            signal.signal(signal.SIGTERM, eval_module._handle_termination_signal)
            eval_module._TERMINATION_SIGNAL_NAME = ""
            raised, _result, records, report = self._run_eval(fake_eval)
        finally:
            signal.signal(signal.SIGTERM, old_handler)
            eval_module._TERMINATION_SIGNAL_NAME = old_signal_name

        self.assertIsInstance(raised, KeyboardInterrupt)
        interrupt = next(
            record for record in records if record.get("event") == "episode_interrupt"
        )
        episode_end = next(
            record for record in records if record.get("event") == "episode_end"
        )
        self.assertEqual(interrupt["reason"], "external_sigterm")
        self.assertEqual(interrupt["interrupt_signal"], "sigterm")
        self.assertEqual(episode_end["failure_reason"], "external_sigterm")
        self.assertFalse(episode_end["success"])
        self.assertFalse(episode_end["natural_episode_end"])
        self.assertEqual(episode_end["episode_validity"]["label"], "user_interrupted")
        self.assertFalse(
            episode_end["episode_validity"]["benchmark_denominator_eligible"]
        )
        self.assertEqual(
            [issue["type"] for issue in report["bad_issues"]], ["episode_interrupted"]
        )

    def test_formal_protocol_version_is_identical_at_episode_boundaries(self) -> None:
        def fake_eval(task_env, _model, _observation) -> None:
            task_env.take_action_cnt += 1
            task_env.eval_success = True

        raised, _result, records, _report = self._run_eval(
            fake_eval,
            runtime_provenance={
                "formal_protocol": True,
                "formal_protocol_version": 3,
            },
        )

        self.assertIsNone(raised)
        episode_start = next(
            record for record in records if record["event"] == "episode_start"
        )
        episode_end = next(
            record for record in records if record["event"] == "episode_end"
        )
        self.assertTrue(episode_start["formal_protocol"])
        self.assertTrue(episode_end["formal_protocol"])
        self.assertEqual(episode_start["formal_protocol_version"], 3)
        self.assertEqual(episode_end["formal_protocol_version"], 3)

    def test_exception_writes_terminal_episode_end_before_reraise(self) -> None:
        def fake_eval(_task_env, _model, _observation) -> None:
            raise RuntimeError("test failure")

        raised, _result, records, report = self._run_eval(fake_eval)

        self.assertIsInstance(raised, RuntimeError)
        exception = next(
            record for record in records if record.get("event") == "episode_exception"
        )
        episode_end = next(
            record for record in records if record.get("event") == "episode_end"
        )
        self.assertEqual(exception["error"], "RuntimeError('test failure')")
        self.assertEqual(episode_end["failure_reason"], "RuntimeError('test failure')")
        self.assertFalse(episode_end["success"])
        self.assertFalse(episode_end["natural_episode_end"])
        self.assertEqual(
            episode_end["episode_validity"]["label"], "incomplete_or_corrupt"
        )
        self.assertEqual(
            [issue["type"] for issue in report["bad_issues"]], ["episode_exception"]
        )

    def test_evolving_missing_session_fails_before_any_policy_step(self) -> None:
        policy_called = False

        def fake_eval(_task_env, _model, _observation) -> None:
            nonlocal policy_called
            policy_called = True

        model = SimpleNamespace(
            hpk_runtime=SimpleNamespace(evolving_enabled=True),
        )
        raised, _result, records, _report = self._run_eval(
            fake_eval,
            model=model,
        )

        self.assertIsInstance(raised, RuntimeError)
        self.assertIn("requires the RoboHarn-Evo Agent session", str(raised))
        self.assertFalse(policy_called)
        exception = next(
            record for record in records if record.get("event") == "episode_exception"
        )
        self.assertIn("requires the RoboHarn-Evo Agent session", exception["error"])

    def test_unvalidated_pure_tool_finish_is_rejected_until_environment_success(
        self,
    ) -> None:
        calls = 0

        class FakeAgent:
            def current_status(self) -> dict:
                return {
                    "pure_tool_control": True,
                    "task_finished": True,
                    "task_finish_validated": False,
                    "monitor_status": "agent_claimed_finish",
                }

        def fake_eval(task_env, _model, _observation) -> None:
            nonlocal calls
            calls += 1
            task_env.take_action_cnt += 1
            if calls == 2:
                task_env.eval_success = True

        model = SimpleNamespace(session=SimpleNamespace(agent=FakeAgent()))
        raised, result, records, report = self._run_eval(fake_eval, model=model)

        self.assertIsNone(raised)
        self.assertEqual(result[1], 1)
        rejected = [
            record
            for record in records
            if record.get("event") == "episode_agent_finish_rejected"
        ]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["step"], 1)
        episode_start = next(
            record for record in records if record.get("event") == "episode_start"
        )
        episode_end = next(
            record for record in records if record.get("event") == "episode_end"
        )
        for record in (episode_start, episode_end):
            self.assertEqual(record["task_name"], "fake_task")
            self.assertEqual(record["task_config"], "fake_config")
            self.assertEqual(record["policy_name"], "fake_policy")
            self.assertEqual(record["ckpt_setting"], "fake_ckpt")
        self.assertTrue(episode_end["success"])
        self.assertEqual(episode_end["total_steps"], 2)
        self.assertTrue(episode_end["natural_episode_end"])
        self.assertEqual(episode_end["episode_validity"]["label"], "valid_success")
        self.assertTrue(episode_end["episode_validity"]["task_success_numerator"])
        self.assertEqual(report["bad_issues"], [])

    def test_agent_terminal_failure_ends_episode_without_false_success(self) -> None:
        class FakeAgent:
            def current_status(self) -> dict:
                return {
                    "pure_tool_control": True,
                    "task_finished": False,
                    "task_finish_validated": False,
                    "terminal_failure": True,
                    "terminal_failure_reason": "pure_tool_control_max_control_turns_exhausted:8",
                    "control_turn_count": 8,
                    "max_control_turns": 8,
                    "monitor_status": "rollout_failed",
                }

        def fake_eval(task_env, _model, _observation) -> None:
            task_env.take_action_cnt += 1

        model = SimpleNamespace(session=SimpleNamespace(agent=FakeAgent()))
        raised, result, records, report = self._run_eval(fake_eval, model=model)

        self.assertIsNone(raised)
        self.assertEqual(result[1], 0)
        terminal = next(
            record
            for record in records
            if record.get("event") == "episode_agent_terminal_failure"
        )
        self.assertEqual(
            terminal["reason"], "pure_tool_control_max_control_turns_exhausted:8"
        )
        episode_end = next(
            record for record in records if record.get("event") == "episode_end"
        )
        self.assertFalse(episode_end["success"])
        self.assertEqual(
            episode_end["failure_reason"],
            "pure_tool_control_max_control_turns_exhausted:8",
        )
        self.assertEqual(episode_end["episode_validity"]["label"], "valid_task_failure")
        self.assertTrue(
            episode_end["episode_validity"]["benchmark_denominator_eligible"]
        )
        self.assertEqual(report["bad_issues"], [])

    def test_backend_budget_terminal_is_classified_as_infrastructure_invalid(
        self,
    ) -> None:
        class FakeAgent:
            def current_status(self) -> dict:
                return {
                    "pure_tool_control": True,
                    "task_finished": False,
                    "task_finish_validated": False,
                    "terminal_failure": True,
                    "terminal_failure_reason": "pure_tool_control_recovery_backend_unavailable:5",
                    "recovery_backend_error_count": 5,
                    "backend_error_budget": 5,
                    "monitor_status": "rollout_failed",
                }

        def fake_eval(task_env, _model, _observation) -> None:
            task_env.take_action_cnt += 1

        model = SimpleNamespace(session=SimpleNamespace(agent=FakeAgent()))
        raised, result, records, report = self._run_eval(fake_eval, model=model)

        self.assertIsNone(raised)
        self.assertEqual(result[1], 0)
        episode_end = next(
            record for record in records if record.get("event") == "episode_end"
        )
        self.assertEqual(
            episode_end["episode_validity"]["label"], "infrastructure_invalid"
        )
        self.assertFalse(
            episode_end["episode_validity"]["benchmark_denominator_eligible"]
        )
        self.assertFalse(episode_end["episode_validity"]["task_success_numerator"])
        self.assertEqual(report["bad_issues"], [])

    def test_unknown_terminal_failure_fails_closed_for_manual_artifact_audit(
        self,
    ) -> None:
        class FakeAgent:
            def current_status(self) -> dict:
                return {
                    "pure_tool_control": True,
                    "task_finished": False,
                    "terminal_failure": True,
                    "terminal_failure_reason": "untyped_service_failure",
                }

        def fake_eval(task_env, _model, _observation) -> None:
            task_env.take_action_cnt += 1

        model = SimpleNamespace(session=SimpleNamespace(agent=FakeAgent()))
        raised, _result, records, _report = self._run_eval(fake_eval, model=model)

        self.assertIsNone(raised)
        episode_end = next(
            record for record in records if record.get("event") == "episode_end"
        )
        self.assertEqual(
            episode_end["episode_validity"]["label"], "incomplete_or_corrupt"
        )
        self.assertIn(
            "requires artifact audit", episode_end["episode_validity"]["reason"]
        )


class EvalPolicyExactSeedTest(unittest.TestCase):
    @staticmethod
    def _decorator(_policy_name: str, function_name: str):
        if function_name == "eval":
            return lambda task_env, _model, _observation: setattr(
                task_env, "eval_success", True
            )
        if function_name == "reset_model":
            return lambda _model: None
        raise AssertionError(function_name)

    @staticmethod
    def _args(*, exact: bool) -> dict:
        return {
            "task_name": "fake_task",
            "task_config": "fake_config",
            "policy_name": "fake_policy",
            "ckpt_setting": "fake_ckpt",
            "clear_cache_freq": 1,
            "render_freq": 7,
            "eval_exact_seed_fail_closed": exact,
        }

    def test_exact_seed_unstable_expert_demo_fails_without_advancing(self) -> None:
        class UnstableEnv(FakeTaskEnv):
            def __init__(self) -> None:
                super().__init__()
                self.setup_seeds: list[int] = []
                self.close_calls = 0

            def setup_demo(self, **kwargs) -> None:
                self.setup_seeds.append(kwargs["seed"])
                raise eval_module.UnStableError("unstable test fixture")

            def close_env(self, **_kwargs) -> None:
                self.close_calls += 1

        environment = UnstableEnv()
        with (
            patch.object(
                eval_module,
                "eval_function_decorator",
                side_effect=self._decorator,
            ),
            self.assertRaisesRegex(
                eval_module.ExactSeedValidationError,
                "seed=0, reason=unstable_expert_demo",
            ),
        ):
            eval_module.eval_policy(
                "fake_task",
                environment,
                self._args(exact=True),
                object(),
                0,
                test_num=1,
                instruction_type="seen",
            )

        self.assertEqual(environment.setup_seeds, [0])
        self.assertEqual(environment.close_calls, 1)

    def test_exact_seed_unsuccessful_expert_demo_fails_without_advancing(self) -> None:
        class UnsuccessfulEnv(FakeTaskEnv):
            def __init__(self) -> None:
                super().__init__()
                self.plan_success = False
                self.setup_seeds: list[int] = []

            def setup_demo(self, **kwargs) -> None:
                self.setup_seeds.append(kwargs["seed"])

            def check_success(self) -> bool:
                return False

        environment = UnsuccessfulEnv()
        with (
            patch.object(
                eval_module,
                "eval_function_decorator",
                side_effect=self._decorator,
            ),
            self.assertRaisesRegex(
                eval_module.ExactSeedValidationError,
                "seed=0, reason=expert_demo_unsuccessful",
            ),
        ):
            eval_module.eval_policy(
                "fake_task",
                environment,
                self._args(exact=True),
                object(),
                0,
                test_num=1,
                instruction_type="seen",
            )

        self.assertEqual(environment.setup_seeds, [0])

    def test_default_mode_still_advances_past_an_unstable_seed(self) -> None:
        class AdvanceOnceEnv(FakeTaskEnv):
            def __init__(self) -> None:
                super().__init__()
                self.setup_seeds: list[int] = []
                self._unstable_emitted = False

            def setup_demo(self, **kwargs) -> None:
                self.setup_seeds.append(kwargs["seed"])
                if not self._unstable_emitted:
                    self._unstable_emitted = True
                    raise eval_module.UnStableError("advance once")
                super().setup_demo(**kwargs)

        environment = AdvanceOnceEnv()
        with (
            patch.object(
                eval_module,
                "eval_function_decorator",
                side_effect=self._decorator,
            ),
            patch.object(
                eval_module,
                "generate_episode_descriptions",
                return_value=[{"seen": ["test instruction"]}],
            ),
            patch.object(
                eval_module,
                "_finalize_rollout_visualization",
                return_value=None,
            ),
        ):
            result = eval_module.eval_policy(
                "fake_task",
                environment,
                self._args(exact=False),
                object(),
                0,
                test_num=1,
                instruction_type="seen",
            )

        self.assertEqual(environment.setup_seeds, [0, 1, 1])
        self.assertEqual(result[1], 1)


class EvalPolicyConfigTest(unittest.TestCase):
    def test_default_start_seed_preserves_launcher_sharding(self) -> None:
        self.assertEqual(eval_module._resolve_start_seed({"seed": 0}), 100000)
        self.assertEqual(eval_module._resolve_start_seed({"seed": 1}), 200000)

    def test_explicit_start_seed_overrides_launcher_sharding(self) -> None:
        config = {"seed": 0, "eval": {"start_seed": 100001}}

        self.assertEqual(eval_module._resolve_start_seed(config), 100001)

    def test_start_seed_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            eval_module._resolve_start_seed({"seed": 0, "eval": {"start_seed": -1}})

    def test_exact_seed_fail_closed_is_explicit_and_strictly_boolean(self) -> None:
        self.assertFalse(eval_module._resolve_exact_seed_fail_closed({"seed": 0}))
        self.assertTrue(
            eval_module._resolve_exact_seed_fail_closed(
                {"seed": 0, "eval": {"exact_seed_fail_closed": True}}
            )
        )
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            eval_module._resolve_exact_seed_fail_closed(
                {"seed": 0, "eval": {"exact_seed_fail_closed": 1}}
            )

    def test_optional_eval_step_limit_is_resolved(self) -> None:
        self.assertIsNone(eval_module._resolve_eval_step_limit({"seed": 0}))
        self.assertEqual(
            eval_module._resolve_eval_step_limit(
                {"seed": 0, "eval": {"step_limit": "150"}}
            ),
            150,
        )

    def test_eval_step_limit_must_be_positive(self) -> None:
        for value in (0, -1, True, "not-an-integer"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "must be a positive integer"),
            ):
                eval_module._resolve_eval_step_limit(
                    {"seed": 0, "eval": {"step_limit": value}}
                )

    def test_eval_step_limit_caps_but_never_extends_task_default(self) -> None:
        capped_env = SimpleNamespace(step_lim=700)
        capped = eval_module._apply_eval_step_limit(capped_env, 150)
        self.assertEqual(capped_env.step_lim, 150)
        self.assertEqual(
            capped,
            {"requested": 150, "task_default": 700, "effective": 150},
        )

        stricter_env = SimpleNamespace(step_lim=100)
        stricter = eval_module._apply_eval_step_limit(stricter_env, 150)
        self.assertEqual(stricter_env.step_lim, 100)
        self.assertEqual(stricter["effective"], 100)


class EvalPolicyHPKEvolvingFinalizationTest(unittest.TestCase):
    def test_hierarchical_read_only_utility_gate_is_explicit_and_strict(self) -> None:
        self.assertFalse(eval_module._hierarchical_hpk_read_only_utility_gate({}))
        self.assertTrue(
            eval_module._hierarchical_hpk_read_only_utility_gate(
                {}, model=SimpleNamespace(rmbench_formal_method="task")
            )
        )
        self.assertTrue(
            eval_module._hierarchical_hpk_read_only_utility_gate(
                {"rmbench_formal": {"config_path": "frozen_cell.json"}}
            )
        )
        self.assertTrue(
            eval_module._hierarchical_hpk_read_only_utility_gate(
                {"agent": {"hpk_v3": {"rmbench_read_only_utility_gate": True}}}
            )
        )
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            eval_module._hierarchical_hpk_read_only_utility_gate(
                {"agent": {"hpk_v3": {"rmbench_read_only_utility_gate": "yes"}}}
            )

    def test_hierarchical_v3_finalizes_the_two_file_store_once(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_hpk_v3_finalize_") as directory:
            trace = Path(directory) / "trace.jsonl"
            calls: list[str] = []

            class Model:
                hpk_runtime = SimpleNamespace(
                    hierarchical_full_enabled=True,
                    online_store_enabled=True,
                )

                @staticmethod
                def finalize_hpk_v3_episode(**kwargs) -> dict:
                    calls.append("finalize")
                    self.assertEqual(kwargs["result"], "Success")
                    self.assertEqual(kwargs["rollout_dir"], directory)
                    return {
                        "status": "action knowledge updated",
                        "evidence_count": 1,
                        "task_knowledge_count": 2,
                        "action_knowledge_count": 3,
                        "semantic_consolidation_performed": True,
                        "evidence": [],
                    }

            result = eval_module._finalize_hierarchical_hpk_episode(
                Model(),
                rollout_dir=Path(directory),
                trace_file=trace,
                episode_id=1,
                seed=2,
                result="Success",
            )

            self.assertEqual(calls, ["finalize"])
            self.assertEqual(result["evidence_count"], 1)
            record = json.loads(trace.read_text(encoding="utf-8"))
            self.assertEqual(record["event"], "hpk_v3_episode_update")
            self.assertEqual(record["episode_id"], 1)
            self.assertEqual(record["seed"], 2)
            self.assertNotIn("sha256", trace.read_text(encoding="utf-8"))

    def test_off_mode_does_not_touch_finalization_or_require_output(self) -> None:
        model = SimpleNamespace(hpk_runtime=None)

        result = eval_module._finalize_evolving_hpk_episode(
            model,
            rollout_dir=None,
            trace_file=None,
            episode_id=0,
            result="Fail",
            natural_episode_end=True,
            episode_validity={"label": "valid_task_failure"},
        )

        self.assertIsNone(result)

    def test_evolving_end_is_sealed_then_finalized_once_with_frozen_time(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_hpk_eval_finalize_") as directory:
            root = Path(directory)
            rollout_dir = root / "rollout"
            rollout_dir.mkdir()
            public_trace = rollout_dir / "hpk_public_audit.jsonl"
            private_trace = rollout_dir / "hpk_private_trace.jsonl"
            private_trace.write_bytes(b"")
            public_trace.write_text(
                json.dumps(
                    {
                        "event": "episode_start",
                        "timestamp": 100.0,
                        "episode_id": 0,
                        "seed": 7,
                        "env_step": -1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            main_trace = root / "episode_agent_trace.jsonl"
            calls: list[str] = []

            class Finalization:
                @staticmethod
                def to_dict() -> dict:
                    return {"status": "abstained", "child": None}

            class SnapshotRef:
                @staticmethod
                def identity() -> dict:
                    return {
                        "snapshot_id": "afksnap_" + "a" * 64,
                        "manifest_sha256": "b" * 64,
                    }

            coordinated = SimpleNamespace(
                finalization=Finalization(),
                next_snapshot_ref=SnapshotRef(),
                advanced=False,
            )
            receipt = SimpleNamespace(
                path=rollout_dir / "hpk_finalization_receipt.json",
                sha256="c" * 64,
                payload={"receipt_id": "afkfinal_" + "d" * 64},
            )

            class Model:
                hpk_runtime = SimpleNamespace(evolving_enabled=True)
                hpk_trace_paths = {
                    "public_trace": str(public_trace),
                    "private_transition_trace": str(private_trace),
                }

                @staticmethod
                def write_hpk_public_episode_event(event, payload) -> None:
                    calls.append("episode_end")
                    self.assertEqual(event, "episode_end")
                    with public_trace.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "event": event,
                                    "timestamp": 123.25,
                                    "seed": 7,
                                    "env_step": 5,
                                    **payload,
                                }
                            )
                            + "\n"
                        )

                @staticmethod
                def publish_hpk_rollout_import_manifest(path):
                    calls.append("manifest")
                    self.assertEqual(calls, ["episode_end", "manifest"])
                    manifest = Path(path)
                    manifest.write_text("{}\n", encoding="utf-8")
                    return SimpleNamespace(
                        path=manifest,
                        sha256="e" * 64,
                        payload={
                            "public_trace": {
                                "sha256": hashlib.sha256(
                                    public_trace.read_bytes()
                                ).hexdigest()
                            }
                        },
                    )

                @staticmethod
                def finalize_hpk_episode(**kwargs):
                    calls.append("finalize")
                    self.assertEqual(
                        kwargs["created_at"],
                        "1970-01-01T00:02:03.250000Z",
                    )
                    self.assertEqual(kwargs["trace_path"], str(public_trace))
                    self.assertEqual(kwargs["expected_manifest_sha256"], "e" * 64)
                    self.assertEqual(
                        kwargs["expected_trace_sha256"],
                        hashlib.sha256(public_trace.read_bytes()).hexdigest(),
                    )
                    self.assertEqual(
                        kwargs["receipt_path"],
                        str(receipt.path.resolve()),
                    )
                    return SimpleNamespace(
                        coordinated_result=coordinated,
                        receipt=receipt,
                        semantic_result=None,
                    )

            summary = eval_module._finalize_evolving_hpk_episode(
                Model(),
                rollout_dir=rollout_dir,
                trace_file=main_trace,
                episode_id=0,
                result="Fail",
                natural_episode_end=True,
                episode_validity={
                    "label": "valid_task_failure",
                    "reason": "valid task failure",
                },
            )

            self.assertEqual(calls, ["episode_end", "manifest", "finalize"])
            self.assertFalse(summary["advanced"])
            self.assertEqual(summary["finalization"]["status"], "abstained")
            recorded = [
                json.loads(line)
                for line in main_trace.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(recorded[-1]["event"], "hpk_episode_finalization")

    def test_frozen_time_rejects_duplicate_episode_end(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_hpk_duplicate_end_") as directory:
            trace = Path(directory) / "public.jsonl"
            records = [
                {
                    "event": "episode_end",
                    "timestamp": float(index + 1),
                    "episode_id": 0,
                }
                for index in range(2)
            ]
            trace.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "exactly one episode_end"):
                eval_module._frozen_hpk_created_at(
                    trace,
                    expected_episode_id=0,
                )


if __name__ == "__main__":
    unittest.main()
