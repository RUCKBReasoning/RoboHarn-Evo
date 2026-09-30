from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from policy.roboharn_evo.scripts.record_agent_contract_manifest import (
    build_agent_contract_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "policy" / "roboharn_evo" / "scripts" / "record_agent_contract_manifest.py"


class AgentContractManifestTest(unittest.TestCase):
    def test_current_shared_contract_is_complete_and_secret_free(self) -> None:
        payload = build_agent_contract_manifest(
            repo_root=REPO_ROOT,
            perception_condition="no_oracle",
            primary_cameras=["head"],
            verification_cameras=["third"],
            max_objects=8,
            instruction_set="rmbench_original",
        )

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(len(payload["shared_model_role_contract"]["required_endpoints"]), 5)
        self.assertEqual(
            payload["shared_model_role_contract"]["recover_modes"],
            ["recovery_planning", "action_effect_verification"],
        )
        self.assertTrue(
            payload["shared_model_role_contract"][
                "same_prompt_and_normalizer_path_for_gpt55_and_local_qwen"
            ]
        )
        self.assertFalse(
            payload["shared_model_role_contract"]["qwen_specific_task_semantics"]
        )
        self.assertEqual(payload["tool_contract"]["public_operation_modes"], ["contact", "grasp", "place"])
        self.assertGreater(payload["skill_bundle"]["skill_file_count"], 10)
        self.assertEqual(payload["observation_contract"]["primary_camera_order"], ["head"])
        self.assertEqual(payload["observation_contract"]["verification_camera_order"], ["third"])
        self.assertFalse(payload["observation_contract"]["oracle_objects_enabled"])
        self.assertFalse(payload["instruction_contract"]["task_instruction_modified_for_qwen"])
        self.assertEqual(len(payload["contract_composite_sha256"]), 64)
        self.assertFalse(payload["secrets_recorded"])

    def test_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roboharn_evo_agent_contract_") as directory:
            output = Path(directory) / "contract.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--repo-root",
                    str(REPO_ROOT),
                    "--output",
                    str(output),
                    "--perception-condition",
                    "no_oracle",
                    "--primary-camera",
                    "head",
                    "--verification-camera",
                    "third",
                    "--max-objects",
                    "8",
                    "--instruction-set",
                    "rmbench_original",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["instruction_contract"]["instruction_set"], "rmbench_original")
        self.assertEqual(payload["observation_contract"]["max_objects"], 8)

    def test_rejects_invalid_observation_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_objects"):
            build_agent_contract_manifest(
                repo_root=REPO_ROOT,
                perception_condition="no_oracle",
                primary_cameras=["head"],
                verification_cameras=["third"],
                max_objects=0,
                instruction_set="rmbench_original",
            )


if __name__ == "__main__":
    unittest.main()
