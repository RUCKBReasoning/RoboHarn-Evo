from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from policy.roboharn_evo.scripts.summarize_object_info_ablation import (
    EXPECTED_COVER_BLOCKS_PHASES,
    aggregate,
    build_acceptance_summary,
    classify_cover_blocks_subtask,
    collect_runs,
    evaluate_cover_blocks_phase_order,
    parse_agent_trace,
)


SUCCESS_SUBTASKS = [
    "Pick up the left lid and place it over the leftmost green block.",
    "Pick up the middle lid and place it over the middle red block.",
    "Pick up the right lid and place it over the rightmost blue block.",
    "Pick up the middle lid covering the red block and place it away to uncover the red block first.",
    "Pick up the left lid covering the green block and place it away to uncover the green block next.",
    "Pick up the right lid covering the blue block and place it away to uncover the blue block.",
]
FORMAL_RUNTIME_MANIFEST = {
    "schema_version": 1,
    "git_head": "1" * 40,
    "runtime_tree_sha256": "2" * 64,
    "tracked_runtime_diff_sha256": "3" * 64,
    "secrets_recorded": False,
}
FORMAL_RUNTIME_MANIFEST_TEXT = (
    json.dumps(FORMAL_RUNTIME_MANIFEST, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
FORMAL_RUNTIME_PROVENANCE = {
    "manifest_sha256": hashlib.sha256(FORMAL_RUNTIME_MANIFEST_TEXT.encode("utf-8")).hexdigest(),
    "runtime_tree_sha256": FORMAL_RUNTIME_MANIFEST["runtime_tree_sha256"],
    "git_head": FORMAL_RUNTIME_MANIFEST["git_head"],
    "tracked_runtime_diff_sha256": FORMAL_RUNTIME_MANIFEST["tracked_runtime_diff_sha256"],
    "archive_file": "runtime_provenance.json",
}


class ObjectInfoAblationSummaryTest(unittest.TestCase):
    def _write_trace(
        self,
        root: Path,
        records: list[dict],
        *,
        manifest_text: str = FORMAL_RUNTIME_MANIFEST_TEXT,
    ) -> Path:
        (root / "runtime_provenance.json").write_text(
            manifest_text,
            encoding="utf-8",
        )
        path = root / "episode_0000_agent_trace.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        return path

    def _write_success_run(
        self,
        root: Path,
        name: str,
        records: list[dict],
        *,
        manifest_text: str = FORMAL_RUNTIME_MANIFEST_TEXT,
    ) -> Path:
        run_dir = root / "oracle" / name
        run_dir.mkdir(parents=True)
        episode_end = records[-1]
        seed = int(episode_end.get("seed", 100001))
        (run_dir / "_result.txt").write_text(
            "Success Rate: 1.0\nReward: 1.0\n",
            encoding="utf-8",
        )
        (run_dir / "eval_log.txt").write_text(
            f"episode_id=0, seed={seed}, instruction=cover blocks, result=Success, "
            "steps=30, reward=1.0, failure_reason=\n",
            encoding="utf-8",
        )
        self._write_trace(run_dir, records, manifest_text=manifest_text)
        return run_dir

    def _successful_records(self) -> list[dict]:
        records: list[dict] = [
            {
                "event": "episode_start",
                "timestamp": 10.0,
                "episode_id": 0,
                "seed": 100001,
                "task_name": "cover_blocks",
                "task_config": "demo_clean",
                "policy_name": "policy.roboharn_evo.deploy_policy",
                "ckpt_setting": "gpt55_formal_test",
                "formal_protocol": True,
                "formal_protocol_version": 3,
                "runtime_provenance": dict(FORMAL_RUNTIME_PROVENANCE),
            }
        ]
        for index, subtask in enumerate(SUCCESS_SUBTASKS):
            records.extend(
                [
                    {"event": "observation_preprocess", "env_step": index},
                    {
                        "event": "control_turn_result",
                        "env_step": index,
                        "subtask_text": subtask,
                        "latency_sec": float(index + 1),
                    },
                    {
                        "event": "recovery_result",
                        "env_step": index,
                        "results": [{"tool_name": "reobserve_scene", "success": True}],
                    },
                    {
                        "event": "action_effect_verification",
                        "env_step": index,
                        "result": {"effect_verified": "true", "effect_type": "move"},
                    },
                ]
            )
        records.append(
            {
                "event": "episode_end",
                "timestamp": 40.0,
                "episode_id": 0,
                "seed": 100001,
                "task_name": "cover_blocks",
                "task_config": "demo_clean",
                "policy_name": "policy.roboharn_evo.deploy_policy",
                "ckpt_setting": "gpt55_formal_test",
                "success": True,
                "environment_success": True,
                "total_steps": 30,
                "max_reward": 1.0,
                "perception_condition": "oracle",
                "semantic_round_index": 0,
                "max_semantic_rounds_per_active_subtask": 10,
                "max_control_turns": 64,
                "max_no_progress_control_turns": 10,
                "backend_error_budget": 5,
                "pure_tool_control": True,
                "formal_protocol": True,
                "formal_protocol_version": 3,
                "runtime_provenance": dict(FORMAL_RUNTIME_PROVENANCE),
                "episode_validity": {
                    "label": "valid_success",
                    "benchmark_denominator_eligible": True,
                    "task_success_numerator": True,
                },
            }
        )
        return records

    def test_cover_blocks_phase_order(self) -> None:
        result = evaluate_cover_blocks_phase_order(SUCCESS_SUBTASKS)

        self.assertTrue(result["passed"])
        self.assertEqual(result["matched"], list(EXPECTED_COVER_BLOCKS_PHASES))
        self.assertEqual(result["unexpected"], [])

    def test_uncover_phase_uses_primary_color_before_context_colors(self) -> None:
        subtasks = [
            *SUCCESS_SUBTASKS[:3],
            "Uncover the red block first without disturbing the covered green and blue blocks.",
            "Uncover the green block next without disturbing the uncovered red block or covered blue block.",
            "Uncover the blue block without disturbing the uncovered green and red blocks.",
        ]

        self.assertEqual(classify_cover_blocks_subtask(subtasks[4]), "uncover_green")
        self.assertEqual(classify_cover_blocks_subtask(subtasks[5]), "uncover_blue")
        self.assertEqual(
            classify_cover_blocks_subtask("No further action is needed; keep all blocks uncovered."),
            "",
        )
        self.assertTrue(evaluate_cover_blocks_phase_order(subtasks)["passed"])

    def test_script_runs_by_path_outside_repository(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_object_info_ablation.py"
        with tempfile.TemporaryDirectory(prefix="tcm_p0_cli_") as directory:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=directory,
                capture_output=True,
                check=False,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--require-acceptance", result.stdout)

    def test_successful_trace_passes_all_p0_gates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_trace_") as directory:
            trace = self._write_trace(Path(directory), self._successful_records())

            report = parse_agent_trace(trace)

        self.assertTrue(report["acceptance_passed"])
        self.assertTrue(report["trace_complete"])
        self.assertTrue(report["episode_start_present"])
        self.assertTrue(report["strict_passed"])
        self.assertTrue(report["phase_order"]["passed"])
        self.assertEqual(report["vla_request_count"], 0)
        self.assertEqual(report["gpt_endpoint_call_count_inferred"], 30)
        self.assertEqual(report["tool_call_count"], 6)
        self.assertEqual(report["wall_duration_sec"], 30.0)
        self.assertEqual(report["failure_stage"], "")
        self.assertEqual(report["episode_id"], 0)
        self.assertEqual(report["seed"], 100001)
        self.assertEqual(report["episode_validity"]["label"], "valid_success")
        self.assertTrue(report["episode_validity"]["runtime_label_matches_trace_audit"])
        self.assertEqual(report["perception_condition"], "oracle")
        self.assertEqual(report["max_semantic_rounds_per_active_subtask"], 10)

    def test_malformed_jsonl_trace_is_incomplete_or_corrupt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_malformed_trace_") as directory:
            trace = self._write_trace(Path(directory), self._successful_records())
            lines = trace.read_text(encoding="utf-8").splitlines()
            lines.insert(1, '{"event":"control_turn_result",')
            trace.write_text("\n".join(lines) + "\n", encoding="utf-8")

            report = parse_agent_trace(trace)

        self.assertEqual(report["malformed_jsonl_line_count"], 1)
        self.assertFalse(report["trace_complete"])
        self.assertFalse(report["strict_passed"])
        self.assertEqual(report["episode_validity"]["trace_audit_label"], "incomplete_or_corrupt")
        self.assertFalse(report["episode_validity"]["benchmark_denominator_eligible"])

    def test_missing_or_conflicting_runtime_validity_label_is_raw_only(self) -> None:
        cases = {
            "missing": None,
            "mismatch": "infrastructure_invalid",
        }
        for case_name, runtime_label in cases.items():
            with self.subTest(case=case_name), tempfile.TemporaryDirectory(
                prefix=f"tcm_p0_runtime_label_{case_name}_"
            ) as directory:
                records = self._successful_records()
                if runtime_label is None:
                    records[-1].pop("episode_validity")
                else:
                    records[-1]["episode_validity"]["label"] = runtime_label
                trace = self._write_trace(Path(directory), records)

                report = parse_agent_trace(trace)
                validity = report["episode_validity"]

            self.assertEqual(validity["trace_audit_label"], "valid_success")
            self.assertFalse(validity["benchmark_denominator_eligible"])
            self.assertFalse(validity["task_success_numerator"])
            self.assertFalse(validity["runtime_label_matches_trace_audit"])
            self.assertTrue(validity["eligibility_exclusion_reasons"])
            self.assertEqual(
                report["failure_stage"],
                f"runtime_validity_label_{'missing_or_invalid' if case_name == 'missing' else 'mismatch'}",
            )

    def test_aggregate_excludes_missing_and_conflicting_runtime_labels(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_runtime_label_aggregate_") as directory:
            root = Path(directory)
            valid_records = self._successful_records()
            missing_records = self._successful_records()
            mismatch_records = self._successful_records()
            for seed, records in zip((100001, 100002, 100003), (valid_records, missing_records, mismatch_records)):
                records[0]["seed"] = seed
                records[-1]["seed"] = seed
            missing_records[-1].pop("episode_validity")
            mismatch_records[-1]["episode_validity"]["label"] = "infrastructure_invalid"
            self._write_success_run(root, "valid", valid_records)
            self._write_success_run(root, "missing", missing_records)
            self._write_success_run(root, "mismatch", mismatch_records)

            summary = aggregate(collect_runs(root))["oracle"]

        self.assertEqual(summary["num_episodes"], 3)
        self.assertEqual(summary["num_capability_eligible"], 1)
        self.assertEqual(summary["runtime_validity_contract_counts"], {"matched": 1, "missing": 1, "mismatch": 1})
        self.assertEqual(summary["formal_eligibility_counts"], {"eligible": 1, "excluded": 2})
        self.assertEqual(summary["validity_counts"]["valid_success"], 3)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["formal_exclusion_counts"]["runtime_validity_label_missing_or_invalid"], 1)
        self.assertEqual(summary["formal_exclusion_counts"]["runtime_validity_label_mismatch"], 1)

    def test_duplicate_formal_identity_excludes_every_duplicate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_duplicate_formal_identity_") as directory:
            root = Path(directory)
            self._write_success_run(root, "duplicate_a", self._successful_records())
            self._write_success_run(root, "duplicate_b", self._successful_records())

            runs = collect_runs(root)
            acceptance = build_acceptance_summary(runs)
            summary = aggregate(runs)["oracle"]

        self.assertEqual(summary["num_formal_base_eligible"], 2)
        self.assertEqual(summary["num_capability_eligible"], 0)
        self.assertIsNone(summary["success_rate"])
        self.assertEqual(summary["duplicate_identity_episode_count"], 2)
        self.assertEqual(
            summary["duplicate_identity_keys"],
            [
                {
                    "task_name": "cover_blocks",
                    "task_config": "demo_clean",
                    "condition": "oracle",
                    "seed": 100001,
                    "count": 2,
                }
            ],
        )
        self.assertEqual(summary["formal_exclusion_counts"]["duplicate_identity"], 2)
        self.assertEqual(summary["runtime_tree_sha256_counts"], {"2" * 64: 2})
        self.assertFalse(summary["formal_aggregate_formed"])
        self.assertEqual(acceptance["failed_count"], 2)
        self.assertEqual(
            {item["failure_stage"] for item in acceptance["failed"]},
            {"duplicate_identity"},
        )

    def test_mixed_runtime_trees_do_not_form_one_formal_aggregate(self) -> None:
        alternate_manifest = {
            **FORMAL_RUNTIME_MANIFEST,
            "runtime_tree_sha256": "4" * 64,
        }
        alternate_manifest_text = (
            json.dumps(alternate_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        alternate_provenance = {
            **FORMAL_RUNTIME_PROVENANCE,
            "manifest_sha256": hashlib.sha256(
                alternate_manifest_text.encode("utf-8")
            ).hexdigest(),
            "runtime_tree_sha256": "4" * 64,
        }
        with tempfile.TemporaryDirectory(prefix="tcm_p0_mixed_runtime_tree_") as directory:
            root = Path(directory)
            first_records = self._successful_records()
            second_records = self._successful_records()
            second_records[0]["seed"] = 100002
            second_records[-1]["seed"] = 100002
            second_records[0]["runtime_provenance"] = dict(alternate_provenance)
            second_records[-1]["runtime_provenance"] = dict(alternate_provenance)
            self._write_success_run(root, "tree_a", first_records)
            self._write_success_run(
                root,
                "tree_b",
                second_records,
                manifest_text=alternate_manifest_text,
            )

            runs = collect_runs(root)
            summary = aggregate(runs)["oracle"]
            acceptance = build_acceptance_summary(runs)

        self.assertEqual(summary["attempted_success_rate"], 1.0)
        self.assertEqual(summary["num_formal_base_eligible"], 2)
        self.assertEqual(summary["num_capability_eligible"], 0)
        self.assertIsNone(summary["success_rate"])
        self.assertEqual(
            summary["runtime_tree_sha256_counts"],
            {"2" * 64: 1, "4" * 64: 1},
        )
        self.assertTrue(summary["mixed_runtime_tree_sha256"])
        self.assertFalse(summary["formal_aggregate_formed"])
        self.assertEqual(
            summary["formal_aggregate_exclusion_reason"],
            "mixed_runtime_tree_sha256",
        )
        self.assertEqual(summary["formal_exclusion_counts"]["mixed_runtime_tree_sha256"], 2)
        self.assertEqual(acceptance["failed_count"], 2)
        self.assertEqual(
            {item["failure_stage"] for item in acceptance["failed"]},
            {"mixed_runtime_tree_sha256"},
        )

    def test_unversioned_legacy_eight_turn_artifact_is_inferred_as_protocol_v1(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_legacy_protocol_v1_") as directory:
            root = Path(directory)
            records = self._successful_records()
            records[0].pop("formal_protocol_version")
            records[-1].pop("formal_protocol_version")
            records[-1]["max_control_turns"] = 8
            records[-1].pop("max_no_progress_control_turns")
            self._write_success_run(root, "legacy_v1", records)

            runs = collect_runs(root)
            metadata = runs[0]["episodes"][0]["trace_metrics"]["formal_metadata"]
            summary = aggregate(runs)["oracle"]

        self.assertTrue(metadata["complete"])
        self.assertEqual(metadata["formal_protocol_version"], 1)
        self.assertEqual(metadata["formal_protocol_version_source"], "legacy_inferred")
        self.assertFalse(metadata["episode_start_formal_protocol_version_recorded"])
        self.assertFalse(metadata["episode_end_formal_protocol_version_recorded"])
        self.assertFalse(metadata["formal_protocol_version_boundary_present"])
        self.assertTrue(metadata["formal_protocol_version_boundary_consistent"])
        self.assertFalse(metadata["max_no_progress_control_turns_required"])
        self.assertTrue(metadata["max_no_progress_control_turns_matches_protocol"])
        self.assertEqual(summary["formal_protocol_version_counts"], {"1": 1})
        self.assertEqual(summary["num_capability_eligible"], 1)

    def test_protocol_v3_requires_matching_explicit_episode_boundary_versions(self) -> None:
        cases = {
            "missing_end": "formal_protocol_version_missing_at_episode_boundary",
            "mismatch": "formal_protocol_version_start_end_mismatch",
            "missing_both": "formal_protocol_version_missing_for_nonlegacy_budget",
            "unsupported": "formal_protocol_version_unsupported:4",
            "explicit_zero": "formal_protocol_version_must_be_positive_integer",
        }
        for case_name, expected_error in cases.items():
            with self.subTest(case=case_name), tempfile.TemporaryDirectory(
                prefix=f"tcm_p0_protocol_version_{case_name}_"
            ) as directory:
                root = Path(directory)
                records = self._successful_records()
                if case_name == "missing_end":
                    records[-1].pop("formal_protocol_version")
                elif case_name == "mismatch":
                    records[-1]["formal_protocol_version"] = 1
                elif case_name == "missing_both":
                    records[0].pop("formal_protocol_version")
                    records[-1].pop("formal_protocol_version")
                elif case_name == "unsupported":
                    records[0]["formal_protocol_version"] = 4
                    records[-1]["formal_protocol_version"] = 4
                else:
                    records[0]["formal_protocol_version"] = 0
                    records[-1]["formal_protocol_version"] = 0
                    records[-1]["max_control_turns"] = 8
                self._write_success_run(root, case_name, records)

                runs = collect_runs(root)
                metadata = runs[0]["episodes"][0]["trace_metrics"]["formal_metadata"]

            self.assertFalse(metadata["complete"])
            self.assertIn(expected_error, metadata["errors"])

    def test_v1_and_v2_do_not_form_one_formal_aggregate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_mixed_protocol_versions_") as directory:
            root = Path(directory)
            v2_records = self._successful_records()
            v2_records[0]["formal_protocol_version"] = 2
            v2_records[-1]["formal_protocol_version"] = 2
            v2_records[-1]["max_no_progress_control_turns"] = 4
            v1_records = self._successful_records()
            v1_records[0]["seed"] = 100002
            v1_records[-1]["seed"] = 100002
            v1_records[0].pop("formal_protocol_version")
            v1_records[-1].pop("formal_protocol_version")
            v1_records[-1]["max_control_turns"] = 8
            v1_records[-1].pop("max_no_progress_control_turns")
            self._write_success_run(root, "protocol_v2", v2_records)
            self._write_success_run(root, "protocol_v1", v1_records)

            runs = collect_runs(root)
            summary = aggregate(runs)["oracle"]
            acceptance = build_acceptance_summary(runs)

        self.assertEqual(summary["formal_protocol_version_counts"], {"1": 1, "2": 1})
        self.assertTrue(summary["mixed_formal_protocol_version"])
        self.assertEqual(summary["num_formal_base_eligible"], 2)
        self.assertEqual(summary["num_capability_eligible"], 0)
        self.assertEqual(
            summary["formal_aggregate_exclusion_reason"],
            "mixed_formal_protocol_version",
        )
        self.assertEqual(
            summary["formal_exclusion_counts"]["mixed_formal_protocol_version"],
            2,
        )
        self.assertEqual(
            {item["failure_stage"] for item in acceptance["failed"]},
            {"mixed_formal_protocol_version"},
        )

    def test_formal_metadata_rejects_protocol_budget_and_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_formal_metadata_") as directory:
            root = Path(directory)
            records = self._successful_records()
            records[0]["task_name"] = ""
            records[0]["seed"] = 999999
            records[0]["formal_protocol"] = False
            records[-1]["task_config"] = "demo_clean_mismatch"
            records[-1]["policy_name"] = "policy.other"
            records[-1]["ckpt_setting"] = "other_ckpt"
            records[-1]["perception_condition"] = "no_oracle"
            records[-1]["pure_tool_control"] = False
            records[-1]["max_semantic_rounds_per_active_subtask"] = 9
            records[-1]["max_control_turns"] = 0
            records[-1].pop("max_no_progress_control_turns")
            records[-1].pop("backend_error_budget")
            records[-1]["runtime_provenance"]["runtime_tree_sha256"] = "9" * 64
            run_dir = self._write_success_run(root, "invalid_metadata", records)

            runs = collect_runs(root)
            summary = aggregate(runs)["oracle"]
            metadata = runs[0]["episodes"][0]["trace_metrics"]["formal_metadata"]
            report = runs[0]["episodes"][0]["trace_metrics"]
            acceptance = build_acceptance_summary(runs)

        self.assertFalse(metadata["complete"])
        self.assertIn("task_name_missing_at_episode_boundary", metadata["errors"])
        self.assertIn("task_config_start_end_mismatch", metadata["errors"])
        self.assertIn("policy_name_start_end_mismatch", metadata["errors"])
        self.assertIn("ckpt_setting_start_end_mismatch", metadata["errors"])
        self.assertIn("seed_missing_or_inconsistent", metadata["errors"])
        self.assertIn("perception_condition_mismatch", metadata["errors"])
        self.assertIn("episode_start_formal_protocol_not_true", metadata["errors"])
        self.assertIn("pure_tool_control_not_true", metadata["errors"])
        self.assertIn("max_semantic_rounds_must_equal_10", metadata["errors"])
        self.assertIn("max_control_turns_must_equal_64_for_protocol_v3", metadata["errors"])
        self.assertIn(
            "max_no_progress_control_turns_must_equal_10_for_protocol_v3",
            metadata["errors"],
        )
        self.assertIn("backend_error_budget_missing_or_invalid", metadata["errors"])
        self.assertIn("runtime_provenance_start_end_mismatch", metadata["errors"])
        self.assertEqual(summary["num_capability_eligible"], 0)
        self.assertEqual(summary["formal_metadata_counts"], {"complete": 0, "incomplete": 1})
        self.assertEqual(summary["formal_eligibility_counts"], {"eligible": 0, "excluded": 1})
        self.assertEqual(report["failure_stage"], "formal_metadata_incomplete")
        self.assertEqual(
            acceptance["failed"][0]["formal_metadata"]["errors"],
            metadata["errors"],
        )
        self.assertIn(
            "formal_metadata:max_control_turns_must_equal_64_for_protocol_v3",
            summary["formal_exclusion_counts"],
        )

    def test_formal_metadata_rejects_missing_or_tampered_provenance_archive(self) -> None:
        cases = ("missing", "tampered", "secrets_recorded")
        for case_name in cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory(
                prefix=f"tcm_p0_provenance_archive_{case_name}_"
            ) as directory:
                root = Path(directory)
                records = self._successful_records()
                secrets_manifest_text = ""
                if case_name == "secrets_recorded":
                    secrets_manifest_text = json.dumps(
                        {**FORMAL_RUNTIME_MANIFEST, "secrets_recorded": True},
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ) + "\n"
                    manifest_hash = hashlib.sha256(secrets_manifest_text.encode("utf-8")).hexdigest()
                    records[0]["runtime_provenance"]["manifest_sha256"] = manifest_hash
                    records[-1]["runtime_provenance"]["manifest_sha256"] = manifest_hash
                run_dir = self._write_success_run(root, case_name, records)
                archive = run_dir / "runtime_provenance.json"
                if case_name == "missing":
                    archive.unlink()
                elif case_name == "tampered":
                    archive.write_text(
                        json.dumps({**FORMAL_RUNTIME_MANIFEST, "runtime_tree_sha256": "f" * 64}) + "\n",
                        encoding="utf-8",
                    )
                else:
                    archive.write_text(secrets_manifest_text, encoding="utf-8")

                runs = collect_runs(root)
                summary = aggregate(runs)["oracle"]
                metadata = runs[0]["episodes"][0]["trace_metrics"]["formal_metadata"]

            self.assertFalse(metadata["complete"])
            self.assertEqual(summary["num_capability_eligible"], 0)
            if case_name == "missing":
                self.assertIn("runtime_provenance_archive_missing", metadata["errors"])
            else:
                if case_name == "tampered":
                    self.assertIn("runtime_provenance_manifest_sha256_mismatch", metadata["errors"])
                    self.assertIn("runtime_provenance_archive_payload_mismatch", metadata["errors"])
                else:
                    self.assertNotIn("runtime_provenance_manifest_sha256_mismatch", metadata["errors"])
                    self.assertIn(
                        "runtime_provenance_archive_secrets_recorded_not_false",
                        metadata["errors"],
                    )

    def test_formal_metadata_rejects_unsafe_or_escaping_provenance_archive(self) -> None:
        cases = ("unsafe_basename", "symlink_escape")
        for case_name in cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory(
                prefix=f"tcm_p0_provenance_path_{case_name}_"
            ) as directory:
                root = Path(directory)
                records = self._successful_records()
                archive_file = (
                    "../runtime_provenance.json"
                    if case_name == "unsafe_basename"
                    else "runtime_provenance_link.json"
                )
                records[0]["runtime_provenance"]["archive_file"] = archive_file
                records[-1]["runtime_provenance"]["archive_file"] = archive_file
                run_dir = self._write_success_run(root, case_name, records)
                if case_name == "symlink_escape":
                    outside_archive = root / "outside_runtime_provenance.json"
                    outside_archive.write_text(FORMAL_RUNTIME_MANIFEST_TEXT, encoding="utf-8")
                    (run_dir / archive_file).symlink_to(outside_archive)

                runs = collect_runs(root)
                metadata = runs[0]["episodes"][0]["trace_metrics"]["formal_metadata"]

            self.assertFalse(metadata["complete"])
            if case_name == "unsafe_basename":
                self.assertIn(
                    "runtime_provenance_archive_file_is_not_a_safe_basename",
                    metadata["errors"],
                )
            else:
                self.assertIn("runtime_provenance_archive_escapes_run_dir", metadata["errors"])

    def test_backend_error_budget_zero_is_recorded_formal_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_backend_budget_zero_") as directory:
            root = Path(directory)
            records = self._successful_records()
            records[-1]["backend_error_budget"] = 0
            self._write_success_run(root, "budget_zero", records)

            runs = collect_runs(root)
            metadata = runs[0]["episodes"][0]["trace_metrics"]["formal_metadata"]

        self.assertTrue(metadata["backend_error_budget_recorded"])
        self.assertTrue(metadata["complete"])

    def test_environment_terminal_effect_is_not_inferred_as_gpt_verifier_call(self) -> None:
        records = self._successful_records()
        last_effect_index = max(
            index
            for index, record in enumerate(records)
            if record.get("event") == "action_effect_verification"
        )
        records[last_effect_index] = {
            "event": "environment_success_effect_commit",
            "authority": "environment_eval_success",
            "verifier_bypassed": True,
        }
        last_result = next(
            record
            for record in reversed(records)
            if record.get("event") == "recovery_result"
        )
        last_result["results"].append(
            {
                "tool_name": "reobserve_scene",
                "success": False,
                "details": {"terminal_skip": True},
            }
        )
        with tempfile.TemporaryDirectory(prefix="tcm_p0_terminal_trace_") as directory:
            trace = self._write_trace(Path(directory), records)
            report = parse_agent_trace(trace)

        self.assertEqual(report["gpt_endpoint_call_count_inferred"], 29)
        self.assertEqual(report["gpt_endpoint_call_breakdown_inferred"]["effect_verification_via_recover"], 5)
        self.assertEqual(report["effect_verdicts"], {"true": 5})
        self.assertEqual(report["environment_success_effect_count"], 1)
        self.assertEqual(report["terminal_tool_skip_count"], 1)

    def test_vla_request_fails_acceptance_even_when_environment_succeeds(self) -> None:
        records = self._successful_records()
        records.insert(-1, {"event": "vla_request", "env_step": 29})
        with tempfile.TemporaryDirectory(prefix="tcm_p0_vla_") as directory:
            trace = self._write_trace(Path(directory), records)

            report = parse_agent_trace(trace)

        self.assertFalse(report["acceptance_passed"])
        self.assertEqual(report["failure_stage"], "unexpected_vla_request")

    def test_missing_episode_start_fails_complete_trace_gate(self) -> None:
        records = self._successful_records()[1:]
        with tempfile.TemporaryDirectory(prefix="tcm_p0_missing_start_") as directory:
            trace = self._write_trace(Path(directory), records)

            report = parse_agent_trace(trace)

        self.assertFalse(report["trace_complete"])
        self.assertFalse(report["acceptance_passed"])
        self.assertEqual(report["failure_stage"], "missing_episode_start")

    def test_collect_and_aggregate_exposes_trace_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_summary_") as directory:
            root = Path(directory)
            run_dir = root / "oracle" / "run_0"
            run_dir.mkdir(parents=True)
            (run_dir / "_result.txt").write_text("Success Rate: 1.0\nReward: 1.0\n", encoding="utf-8")
            (run_dir / "eval_log.txt").write_text(
                "episode_id=0, seed=100001, instruction=cover blocks, result=Success, steps=30, reward=1.0, failure_reason=\n",
                encoding="utf-8",
            )
            self._write_trace(run_dir, self._successful_records())

            runs = collect_runs(root)
            summary = aggregate(runs)
            acceptance = build_acceptance_summary(runs)

        self.assertEqual(len(runs), 1)
        trace_metrics = runs[0]["episodes"][0]["trace_metrics"]
        self.assertTrue(trace_metrics["acceptance_passed"])
        self.assertEqual(summary["oracle"]["success_rate"], 1.0)
        self.assertEqual(summary["oracle"]["attempted_success_rate"], 1.0)
        self.assertEqual(summary["oracle"]["validity_counts"]["valid_success"], 1)
        self.assertEqual(summary["oracle"]["num_capability_eligible"], 1)
        self.assertEqual(summary["oracle"]["trace_coverage_rate"], 1.0)
        self.assertEqual(summary["oracle"]["acceptance_rate"], 1.0)
        self.assertEqual(summary["oracle"]["strict_pass_rate"], 1.0)
        self.assertEqual(summary["oracle"]["mean_gpt_endpoint_calls_inferred"], 30.0)
        self.assertEqual(summary["oracle"]["mean_tool_calls"], 6.0)
        self.assertEqual(summary["oracle"]["formal_protocol_version_counts"], {"3": 1})
        self.assertFalse(summary["oracle"]["mixed_formal_protocol_version"])
        self.assertTrue(acceptance["all_passed"])
        self.assertEqual(acceptance["matched_episode_count"], 1)
        formal_metadata = runs[0]["episodes"][0]["trace_metrics"]["formal_metadata"]
        self.assertTrue(formal_metadata["complete"])
        self.assertEqual(formal_metadata["formal_protocol_version"], 3)
        self.assertEqual(formal_metadata["formal_protocol_version_source"], "explicit")

    def test_infrastructure_invalid_is_excluded_from_capability_denominator(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_validity_aggregate_") as directory:
            root = Path(directory)
            success_dir = root / "oracle" / "success"
            invalid_dir = root / "oracle" / "invalid"
            success_dir.mkdir(parents=True)
            invalid_dir.mkdir(parents=True)

            (success_dir / "_result.txt").write_text("Success Rate: 1.0\nReward: 1.0\n", encoding="utf-8")
            (success_dir / "eval_log.txt").write_text(
                "episode_id=0, seed=100001, instruction=cover blocks, result=Success, "
                "steps=30, reward=1.0, failure_reason=\n",
                encoding="utf-8",
            )
            self._write_trace(success_dir, self._successful_records())

            invalid_records = self._successful_records()
            invalid_records[0]["seed"] = 100002
            invalid_records.insert(
                -1,
                {
                    "event": "pure_tool_control_recovery_backend_unavailable",
                    "reason": "pure_tool_control_recovery_backend_unavailable:5",
                },
            )
            invalid_records[-1].update(
                {
                    "seed": 100002,
                    "success": False,
                    "environment_success": False,
                    "max_reward": 0.0,
                    "failure_reason": "pure_tool_control_recovery_backend_unavailable:5",
                    "terminal_failure": True,
                    "terminal_failure_reason": "pure_tool_control_recovery_backend_unavailable:5",
                    "episode_validity": {
                        "label": "infrastructure_invalid",
                        "benchmark_denominator_eligible": False,
                        "task_success_numerator": False,
                    },
                }
            )
            (invalid_dir / "_result.txt").write_text("Success Rate: 0.0\nReward: 0.0\n", encoding="utf-8")
            (invalid_dir / "eval_log.txt").write_text(
                "episode_id=0, seed=100002, instruction=cover blocks, result=Fail, steps=30, "
                "reward=0.0, failure_reason=pure_tool_control_recovery_backend_unavailable:5\n",
                encoding="utf-8",
            )
            self._write_trace(invalid_dir, invalid_records)

            runs = collect_runs(root)
            summary = aggregate(runs)["oracle"]

        self.assertEqual(summary["num_episodes"], 2)
        self.assertEqual(summary["attempted_success_rate"], 0.5)
        self.assertEqual(summary["num_capability_eligible"], 1)
        self.assertEqual(summary["num_valid_success"], 1)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["validity_counts"]["valid_success"], 1)
        self.assertEqual(summary["validity_counts"]["infrastructure_invalid"], 1)

    def test_mismatched_trace_does_not_satisfy_episode_acceptance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_mismatch_") as directory:
            root = Path(directory)
            run_dir = root / "oracle" / "run_0"
            run_dir.mkdir(parents=True)
            (run_dir / "_result.txt").write_text("Success Rate: 1.0\nReward: 1.0\n", encoding="utf-8")
            (run_dir / "eval_log.txt").write_text(
                "episode_id=0, seed=100002, instruction=cover blocks, result=Success, steps=30, reward=1.0, failure_reason=\n",
                encoding="utf-8",
            )
            self._write_trace(run_dir, self._successful_records())

            runs = collect_runs(root)
            summary = aggregate(runs)
            acceptance = build_acceptance_summary(runs)

        self.assertNotIn("trace_metrics", runs[0]["episodes"][0])
        self.assertEqual(summary["oracle"]["trace_coverage_rate"], 0.0)
        self.assertEqual(summary["oracle"]["acceptance_rate"], 0.0)
        self.assertEqual(summary["oracle"]["failure_stages"], {"missing_trace": 1})
        self.assertFalse(acceptance["all_passed"])
        self.assertEqual(acceptance["failed"][0]["failure_stage"], "missing_trace")
        self.assertEqual(len(acceptance["orphan_traces"]), 1)

    def test_one_trace_cannot_cover_duplicate_episode_results(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_duplicate_") as directory:
            root = Path(directory)
            run_dir = root / "oracle" / "run_0"
            run_dir.mkdir(parents=True)
            (run_dir / "_result.txt").write_text("Success Rate: 1.0\nReward: 1.0\n", encoding="utf-8")
            eval_line = (
                "episode_id=0, seed=100001, instruction=cover blocks, result=Success, "
                "steps=30, reward=1.0, failure_reason=\n"
            )
            (run_dir / "eval_log.txt").write_text(eval_line * 2, encoding="utf-8")
            self._write_trace(run_dir, self._successful_records())

            runs = collect_runs(root)
            summary = aggregate(runs)
            acceptance = build_acceptance_summary(runs)

        self.assertEqual(summary["oracle"]["num_episodes"], 2)
        self.assertEqual(summary["oracle"]["trace_coverage_rate"], 0.5)
        self.assertEqual(summary["oracle"]["acceptance_rate"], 0.5)
        self.assertFalse(acceptance["all_passed"])
        self.assertEqual(acceptance["matched_episode_count"], 1)
        self.assertEqual(acceptance["failed"][0]["failure_stage"], "missing_trace")

    def test_interrupted_worker_uses_finalize_artifact_to_find_trace(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_p0_interrupt_") as directory:
            root = Path(directory)
            log_dir = root / "oracle"
            run_dir = root / "eval_result" / "interrupted_run"
            log_dir.mkdir(parents=True)
            run_dir.mkdir(parents=True)
            worker_log = log_dir / "worker_1_gpu_3_seed_1_e100002.log"
            records = [
                {
                    "event": "episode_start",
                    "timestamp": 10.0,
                    "episode_id": 0,
                    "seed": 100002,
                    "task_name": "cover_blocks",
                    "instruction": "cover blocks",
                },
                {
                    "event": "episode_interrupt",
                    "timestamp": 20.0,
                    "episode_id": 0,
                    "seed": 100002,
                    "reason": "external_sigterm",
                    "interrupt_signal": "sigterm",
                },
                {
                    "event": "episode_end",
                    "timestamp": 20.0,
                    "episode_id": 0,
                    "seed": 100002,
                    "success": False,
                    "environment_success": False,
                    "total_steps": 15,
                    "max_reward": 0.3,
                    "failure_reason": "external_sigterm",
                },
            ]
            self._write_trace(run_dir, records)
            log_records = [
                *records,
                {
                    "event": "rollout_report_finalize",
                    "report": str(run_dir / "rollout_report.html"),
                },
            ]
            worker_log.write_text(
                "\n".join("[eval] " + json.dumps(record) for record in log_records) + "\n",
                encoding="utf-8",
            )

            runs = collect_runs(root)
            acceptance = build_acceptance_summary(runs)

        self.assertEqual(len(runs), 1)
        episode = runs[0]["episodes"][0]
        self.assertEqual(episode["seed"], 100002)
        self.assertEqual(episode["trace_metrics"]["failure_stage"], "external_sigterm")
        self.assertEqual(episode["episode_validity"]["label"], "user_interrupted")
        self.assertFalse(episode["episode_validity"]["benchmark_denominator_eligible"])
        self.assertEqual(episode["trace_metrics"]["strict_issue_count"], 1)
        self.assertEqual(acceptance["matched_episode_count"], 1)
        self.assertEqual(acceptance["failed"][0]["failure_stage"], "external_sigterm")


if __name__ == "__main__":
    unittest.main()
