from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from policy.roboharn_evo.scripts.record_runtime_provenance import (
    DEFAULT_RUNTIME_PATHS,
    EFFECTIVE_CONFIG_ENV_KEYS,
    RUNTIME_PROVENANCE_SCHEMA,
    SOURCE_IDENTITY_MODE,
    build_runtime_provenance,
    canonical_json_sha256,
    decode_json_object,
    effective_config_from_environment,
    load_external_evidence,
    load_service_identity,
    parse_external_evidence_specs,
    roboharn_eval_result_root,
    validate_output_path,
    validate_runtime_provenance,
    write_runtime_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT.parents[1]
SCRIPT = REPO_ROOT / "policy" / "roboharn_evo" / "scripts" / "record_runtime_provenance.py"


def formal_environment(*, seed: int = 123, gpu: int = 4) -> dict[str, str]:
    return {
        "TASK_NAME": "swap_blocks",
        "TASK_CONFIG": "demo_clean",
        "INSTRUCTION_SET": "rmbench_original",
        "POLICY_NAME": "policy.roboharn_evo.deploy_policy",
        "PERCEPTION_CONDITION": "no_oracle",
        "EVAL_START_SEEDS": str(seed),
        "NUM_WORKERS": "1",
        "GPU_IDS": str(gpu),
        "AGENT_API_BASE_URL": "http://127.0.0.1:9104",
        "SAM3_SERVICE_URL": "http://127.0.0.1:9311",
        "EXPECTED_AGENT_MODEL": "gpt-5.5",
        "EXPECTED_AGENT_API_MODE": "responses_compat",
        "EXPECTED_REASONING_EFFORT": "xhigh",
        "EXPECTED_RESPONSE_STORAGE": "account_default",
        "EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS": "4",
        "MAX_ROUNDS": "10",
        "MAX_CONTROL_TURNS": "64",
        "MAX_NO_PROGRESS_CONTROL_TURNS": "10",
        "EVAL_STEP_LIMIT": "150",
        "MAX_OBJECTS": "8",
        "N_PER_WORKER": "1",
        "SEED_OFFSET": "0",
        "NON_FORMAL_DIAGNOSTIC": "0",
        "ROBOHARN_EVO_FORMAL_PROTOCOL": "1",
        "ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION": "3",
        "REQUIRE_EXPLICIT_EVAL_START_SEEDS": "1",
        "REQUIRE_SAM3_PREFLIGHT": "1",
        "SKIP_PREFLIGHT": "0",
        "SKIP_AGENT_IDENTITY_CHECK": "0",
        "REQUIRE_AGENT_INFERENCE_PREFLIGHT": "1",
        "RECORD_RUNTIME_PROVENANCE": "1",
    }


def project_fixture(directory: str) -> tuple[Path, tuple[str, ...], Path]:
    project = Path(directory) / "RoboHarn-Evo"
    repo_root = project / "benchmarks" / "rmbench"
    first = repo_root / "policy" / "roboharn_evo" / "agent" / "runtime.py"
    second = repo_root / "script" / "eval_policy.py"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("RUNTIME = 'content'\n", encoding="utf-8")
    second.write_text("def evaluate(): return True\n", encoding="utf-8")
    output = project / "eval_result" / "rmbench" / "tests" / "runtime.json"
    paths = ("policy/roboharn_evo/agent/runtime.py", "script/eval_policy.py")
    return repo_root, paths, output


class RuntimeProvenanceTest(unittest.TestCase):
    def test_default_paths_cover_complete_formal_chain_and_four_seeds(self) -> None:
        required = {
            "tcm/agent",
            "tcm/models",
            "tcm/resources",
            "tcm/utils",
            "benchmarks/rmbench/policy/roboharn_evo/scripts/launch_gpt55_formal4x20_from_seed_manifest.sh",
            "benchmarks/rmbench/policy/roboharn_evo/scripts/run_gpt55_parallel4_isolated.py",
            "benchmarks/rmbench/policy/roboharn_evo/scripts/run_gpt55_six_tasks_sequential.sh",
            "benchmarks/rmbench/policy/roboharn_evo/scripts/run_gpt55_pure_tool_control_8way.sh",
            "benchmarks/rmbench/policy/roboharn_evo/scripts/check_pure_tool_control_early_stop.py",
            "benchmarks/rmbench/policy/roboharn_evo/scripts/serve_openai_planner.py",
            "benchmarks/rmbench/policy/roboharn_evo/scripts/serve_sam3_segmentation.py",
            "benchmarks/rmbench/script/eval_policy.py",
            "benchmarks/rmbench/policy/roboharn_evo/deploy_policy.py",
            "benchmarks/rmbench/policy/roboharn_evo/deploy_policy.yml",
            "benchmarks/rmbench/data/data/rearrange_blocks/demo_clean/seed.txt",
            "benchmarks/rmbench/data/data/swap_blocks/demo_clean/seed.txt",
            "benchmarks/rmbench/data/data/swap_T/demo_clean/seed.txt",
            "benchmarks/rmbench/data/data/battery_try/demo_clean/seed.txt",
        }
        self.assertTrue(required.issubset(DEFAULT_RUNTIME_PATHS))
        self.assertNotIn(
            "benchmarks/rmbench/policy/roboharn_evo/agent",
            DEFAULT_RUNTIME_PATHS,
        )
        self.assertNotIn(
            "benchmarks/rmbench/policy/roboharn_evo/models",
            DEFAULT_RUNTIME_PATHS,
        )
        self.assertEqual(
            [path for path in DEFAULT_RUNTIME_PATHS if not (PROJECT_ROOT / path).exists()],
            [],
        )

    def test_content_manifest_records_path_hash_size_and_no_git_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_evo_content_provenance_") as directory:
            repo_root, paths, _output = project_fixture(directory)
            payload = build_runtime_provenance(
                repo_root,
                paths,
                environ=formal_environment(),
            )
            validated = validate_runtime_provenance(
                payload,
                repo_root,
                expected_effective_config=effective_config_from_environment(
                    formal_environment()
                ),
            )

        self.assertEqual(validated["schema"], RUNTIME_PROVENANCE_SCHEMA)
        self.assertEqual(validated["source_identity_mode"], SOURCE_IDENTITY_MODE)
        self.assertEqual(validated["schema_version"], 2)
        self.assertFalse(validated["secrets_recorded"])
        self.assertNotIn("git_head", validated)
        self.assertNotIn("git_dirty_for_runtime_paths", validated)
        self.assertNotIn("tracked_runtime_diff_sha256", validated)
        entries = validated["runtime_content_manifest"]["entries"]
        self.assertEqual(
            [entry["path"] for entry in entries],
            ["policy/roboharn_evo/agent/runtime.py", "script/eval_policy.py"],
        )
        self.assertTrue(all(set(entry) == {"path", "sha256", "size_bytes"} for entry in entries))
        self.assertEqual(
            validated["runtime_content_manifest"]["aggregate_sha256"],
            canonical_json_sha256(entries),
        )
        self.assertEqual(validated["runtime_tree_sha256"], canonical_json_sha256(entries))
        self.assertEqual(
            validated["effective_config_sha256"],
            canonical_json_sha256(validated["effective_config"]),
        )

    def test_effective_config_is_strict_typed_and_ignores_secret_environment(self) -> None:
        environment = formal_environment(seed=7, gpu=6)
        environment.update(
            {
                "OPENAI_API_KEY": "must-never-be-recorded",
                "AUTHORIZATION": "Bearer must-never-be-recorded",
                "UNRELATED_VALUE": "ignored",
            }
        )
        config = effective_config_from_environment(environment)

        self.assertEqual(config["task_name"], "swap_blocks")
        self.assertEqual(config["eval_start_seed"], 7)
        self.assertEqual(config["gpu_ids"], [6])
        self.assertEqual(config["num_workers"], 1)
        self.assertIs(config["pure_tool_control"], True)
        self.assertIs(config["formal_protocol"], True)
        self.assertEqual(config["formal_protocol_version"], 3)
        self.assertNotIn("must-never-be-recorded", json.dumps(config))
        self.assertFalse(any(key in config for key in environment if key not in EFFECTIVE_CONFIG_ENV_KEYS))

    def test_effective_config_supports_typed_multi_seed_and_nonformal_modes(self) -> None:
        environment = formal_environment()
        environment["EVAL_START_SEEDS"] = "1,2"
        environment["NUM_WORKERS"] = "2"
        environment["GPU_IDS"] = "4,5"
        multi_seed = effective_config_from_environment(environment)
        self.assertEqual(multi_seed["eval_start_seeds"], [1, 2])
        self.assertIsNone(multi_seed["eval_start_seed"])

        environment = formal_environment()
        environment["NON_FORMAL_DIAGNOSTIC"] = "1"
        environment["ROBOHARN_EVO_FORMAL_PROTOCOL"] = "0"
        environment["ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION"] = "0"
        nonformal = effective_config_from_environment(environment)
        self.assertIs(nonformal["non_formal_diagnostic"], True)
        self.assertIs(nonformal["formal_protocol"], False)
        self.assertEqual(nonformal["formal_protocol_version"], 0)

        environment = formal_environment()
        environment["AGENT_API_BASE_URL"] = "http://user:secret@127.0.0.1:9104"
        with self.assertRaisesRegex(ValueError, "without credentials"):
            effective_config_from_environment(environment)

    def test_effective_config_hash_binds_runtime_budgets_and_timeouts(self) -> None:
        baseline = effective_config_from_environment(formal_environment())
        environment = formal_environment()
        environment.update(
            {
                "MAX_WAIT_STEPS": "3",
                "RETRY_BUDGET": "2",
                "BACKEND_ERROR_BUDGET": "6",
                "EMPTY_PLAN_REPLAN_THRESHOLD": "3",
                "PLANNER_TIMEOUT_SEC": "1200",
                "QUERY_TIMEOUT_SEC": "600",
                "SHUTDOWN_GRACE_SEC": "45",
                "HEARTBEAT_SEC": "30",
                "INTERRUPT_ESCALATION_POLICY": "ctrl_c_only",
            }
        )
        changed = effective_config_from_environment(environment)

        self.assertEqual(changed["max_wait_steps"], 3)
        self.assertEqual(changed["retry_budget"], 2)
        self.assertEqual(changed["planner_timeout_sec"], 1200.0)
        self.assertEqual(changed["interrupt_escalation_policy"], "ctrl_c_only")
        self.assertNotEqual(
            canonical_json_sha256(baseline),
            canonical_json_sha256(changed),
        )

    def test_validator_rejects_git_fields_content_tamper_and_config_tamper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_evo_validate_provenance_") as directory:
            repo_root, paths, _output = project_fixture(directory)
            payload = build_runtime_provenance(
                repo_root,
                paths,
                environ=formal_environment(),
            )
            with_git = copy.deepcopy(payload)
            with_git["git_head"] = "fabricated"
            with self.assertRaisesRegex(ValueError, "Git identity"):
                validate_runtime_provenance(with_git, repo_root)

            bad_config = copy.deepcopy(payload)
            bad_config["effective_config"]["max_rounds"] = 11
            with self.assertRaisesRegex(ValueError, "effective_config"):
                validate_runtime_provenance(bad_config, repo_root)

            (repo_root / paths[0]).write_text("RUNTIME = 'changed'\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content (size|hash) changed"):
                validate_runtime_provenance(payload, repo_root)

    def test_strict_decoder_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        for raw in (b'{"key":1,"key":2}', b'{"key":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_json_object(raw, label="test manifest")

    def test_formal_required_runtime_contract_cannot_be_self_declared_away(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_evo_required_provenance_") as directory:
            repo_root, paths, _output = project_fixture(directory)
            payload = build_runtime_provenance(
                repo_root,
                (paths[0],),
                environ=formal_environment(),
            )
            with self.assertRaisesRegex(ValueError, "required formal runtime files"):
                validate_runtime_provenance(
                    payload,
                    repo_root,
                    required_runtime_paths=(paths[1],),
                )

    def test_service_identity_is_preserved_with_exact_byte_hash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_evo_service_identity_") as directory:
            repo_root, paths, _output = project_fixture(directory)
            identity_path = Path(directory) / "agent_service_identity.json"
            raw = b'{"status":"ok","model":"gpt-5.5","fallback_enabled":false}\n'
            identity_path.write_bytes(raw)
            payload = build_runtime_provenance(
                repo_root,
                paths,
                service_identity_path=identity_path,
                environ=formal_environment(),
            )
            tampered = copy.deepcopy(payload)
            tampered["agent_service_identity"]["model"] = "tampered"
            with self.assertRaisesRegex(ValueError, "canonical_sha256"):
                validate_runtime_provenance(tampered, repo_root)

        self.assertEqual(payload["agent_service_identity"]["status"], "ok")
        self.assertEqual(
            payload["agent_service_identity_sha256"],
            hashlib.sha256(raw).hexdigest(),
        )
        self.assertEqual(
            payload["agent_service_identity_bytes_sha256"],
            hashlib.sha256(raw).hexdigest(),
        )
        self.assertEqual(
            payload["agent_service_identity_hash_mode"],
            "exact_source_json_bytes",
        )
        self.assertEqual(
            payload["agent_service_identity_canonical_sha256"],
            canonical_json_sha256(payload["agent_service_identity"]),
        )

    def test_service_and_external_evidence_reject_secret_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_evo_evidence_secret_") as directory:
            path = Path(directory) / "identity.json"
            path.write_text('{"api_key":"must-not-be-archived"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "credential-like field"):
                load_service_identity(path)
            with self.assertRaisesRegex(ValueError, "credential-like field"):
                load_external_evidence(path, label="scheduler")

    def test_external_evidence_can_be_immutable_at_record_and_mutable_at_archive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_evo_mutable_evidence_") as directory:
            repo_root, paths, _output = project_fixture(directory)
            evidence = Path(directory) / "scheduler_manifest.json"
            evidence.write_text('{"state":"planned"}\n', encoding="utf-8")
            payload = build_runtime_provenance(
                repo_root,
                paths,
                external_evidence_specs=(f"scheduler_manifest={evidence}",),
                environ=formal_environment(),
            )
            evidence.write_text('{"state":"completed"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "content changed"):
                validate_runtime_provenance(payload, repo_root)
            validate_runtime_provenance(
                payload,
                repo_root,
                verify_external_evidence=False,
                expected_effective_config=effective_config_from_environment(
                    formal_environment()
                ),
            )

    def test_external_evidence_spec_parser_remains_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate external evidence label"):
            parse_external_evidence_specs(
                ("report=/tmp/one.json", "report=/tmp/two.json")
            )
        with self.assertRaisesRegex(ValueError, "path must be absolute"):
            parse_external_evidence_specs(("report=relative.json",))
        with self.assertRaisesRegex(ValueError, "must be a .json file"):
            parse_external_evidence_specs(("report=/tmp/report.txt",))

    def test_output_is_confined_to_project_eval_result_rmbench(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_evo_output_boundary_") as directory:
            repo_root, paths, output = project_fixture(directory)
            payload = build_runtime_provenance(
                repo_root,
                paths,
                environ=formal_environment(),
            )
            self.assertEqual(
                roboharn_eval_result_root(repo_root),
                Path(directory) / "RoboHarn-Evo" / "eval_result" / "rmbench",
            )
            self.assertEqual(validate_output_path(repo_root, output), output)
            written = write_runtime_provenance(repo_root, output, payload)
            self.assertEqual(written, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)

            outside = Path(directory) / "outside.json"
            with self.assertRaisesRegex(ValueError, "inside"):
                write_runtime_provenance(repo_root, outside, payload)
            self.assertFalse(outside.exists())

    def test_self_evolution_output_is_nonformal_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_evo_self_evolution_boundary_") as directory:
            repo_root, paths, _output = project_fixture(directory)
            self_evolution_output = (
                Path(directory)
                / "RoboHarn-Evo"
                / "eval_result"
                / "self_evolution"
                / "phase_b"
                / "runtime_provenance.json"
            )
            formal_payload = build_runtime_provenance(
                repo_root,
                paths,
                environ=formal_environment(),
            )
            with self.assertRaisesRegex(ValueError, "allowed result root"):
                write_runtime_provenance(
                    repo_root,
                    self_evolution_output,
                    formal_payload,
                )
            self.assertFalse(self_evolution_output.exists())

            nonformal_environment = formal_environment()
            nonformal_environment.update(
                {
                    "NON_FORMAL_DIAGNOSTIC": "1",
                    "ROBOHARN_EVO_FORMAL_PROTOCOL": "0",
                    "ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION": "0",
                }
            )
            nonformal_payload = build_runtime_provenance(
                repo_root,
                paths,
                environ=nonformal_environment,
            )
            self.assertEqual(
                validate_output_path(
                    repo_root,
                    self_evolution_output,
                    formal_protocol=False,
                ),
                self_evolution_output,
            )
            written = write_runtime_provenance(
                repo_root,
                self_evolution_output,
                nonformal_payload,
            )
            self.assertEqual(written, self_evolution_output)
            self.assertEqual(
                json.loads(self_evolution_output.read_text(encoding="utf-8")),
                nonformal_payload,
            )

            unrelated_output = (
                Path(directory)
                / "RoboHarn-Evo"
                / "eval_result"
                / "unrelated"
                / "runtime_provenance.json"
            )
            with self.assertRaisesRegex(ValueError, "allowed result root"):
                write_runtime_provenance(
                    repo_root,
                    unrelated_output,
                    nonformal_payload,
                )
            self.assertFalse(unrelated_output.exists())

    def test_cli_writes_inside_eval_result_and_reports_content_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_evo_provenance_cli_") as directory:
            repo_root, paths, output = project_fixture(directory)
            environment = dict(os.environ)
            environment.update(formal_environment())
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--repo-root",
                    str(repo_root),
                    "--output",
                    str(output),
                    "--path",
                    paths[0],
                ],
                cwd=repo_root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary["source_identity_mode"], "content")
        self.assertEqual(summary["runtime_file_count"], 1)
        self.assertNotIn("git_head", summary)
        self.assertEqual(payload["runtime_paths"], [paths[0]])

    def test_cli_rejects_output_outside_eval_boundary_without_writing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_evo_provenance_cli_outside_") as directory:
            repo_root, paths, _output = project_fixture(directory)
            outside = Path(directory) / "outside.json"
            environment = dict(os.environ)
            environment.update(formal_environment())
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--repo-root",
                    str(repo_root),
                    "--output",
                    str(outside),
                    "--path",
                    paths[0],
                ],
                cwd=repo_root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be a file inside", result.stderr)
            self.assertFalse(outside.exists())

    def test_default_runtime_set_is_recordable_without_git(self) -> None:
        payload = build_runtime_provenance(
            PROJECT_ROOT,
            DEFAULT_RUNTIME_PATHS,
            environ=formal_environment(),
        )

        self.assertEqual(payload["runtime_paths"], list(DEFAULT_RUNTIME_PATHS))
        self.assertGreater(payload["runtime_file_count"], len(DEFAULT_RUNTIME_PATHS))
        self.assertEqual(len(payload["runtime_tree_sha256"]), 64)
        recorded = set(payload["runtime_file_hashes_sha256"])
        self.assertIn("tcm/agent/runtime.py", recorded)
        self.assertIn("tcm/models/backend_factory.py", recorded)
        self.assertIn(
            "benchmarks/rmbench/policy/roboharn_evo/deploy_policy.py",
            recorded,
        )
        self.assertFalse(
            any(
                path.startswith("benchmarks/rmbench/policy/roboharn_evo/agent/")
                or path.startswith("benchmarks/rmbench/policy/roboharn_evo/models/")
                for path in recorded
            )
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("def _git(", source)


if __name__ == "__main__":
    unittest.main()
