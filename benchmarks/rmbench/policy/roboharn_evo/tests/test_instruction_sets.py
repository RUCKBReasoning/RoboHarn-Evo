from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
DESCRIPTION_UTILS = REPO_ROOT / "description" / "utils"
if str(DESCRIPTION_UTILS) not in sys.path:
    sys.path.insert(0, str(DESCRIPTION_UTILS))

from generate_episode_instructions import (  # noqa: E402
    ORIGINAL_INSTRUCTION_SET,
    describe_instruction_source,
    generate_episode_descriptions,
    load_task_instructions,
    resolve_task_instruction_path,
)


CUSTOM_SET = "contract_clarified_v1"
TASK_NAME = "blocks_ranking_try"


class InstructionSetTest(unittest.TestCase):
    def test_original_instruction_remains_the_default(self) -> None:
        payload = load_task_instructions(TASK_NAME)

        self.assertEqual(ORIGINAL_INSTRUCTION_SET, "rmbench_original")
        self.assertEqual(
            payload["unseen"],
            [
                "There is a button and three colored cubes arranged in a random row "
                "on the table. Each time the cubes are rearranged, the arm presses "
                "the button until the arrangement is successful."
            ],
        )

    def test_custom_instruction_exposes_the_complete_task_contract(self) -> None:
        results = generate_episode_descriptions(
            TASK_NAME,
            [{}],
            1,
            instruction_set=CUSTOM_SET,
        )
        instruction = results[0]["unseen"][0]

        self.assertIn("The button only tests the current cube order", instruction)
        self.assertIn("rearrange the cubes into a different, untested", instruction)
        self.assertIn("Do not press the button again unless", instruction)
        self.assertNotIn("left arm", instruction.lower())
        self.assertNotIn("right arm", instruction.lower())

    def test_custom_instruction_provenance_is_content_addressed(self) -> None:
        metadata = describe_instruction_source(TASK_NAME, CUSTOM_SET)
        task_path = REPO_ROOT / metadata["task_file"]
        manifest_path = REPO_ROOT / metadata["manifest_file"]

        self.assertEqual(metadata["instruction_set_id"], CUSTOM_SET)
        self.assertEqual(metadata["source"], "researcher_authored")
        self.assertEqual(
            metadata["task_file_sha256"],
            hashlib.sha256(task_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            metadata["manifest_file_sha256"],
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest["tasks"][TASK_NAME]
        original_path = REPO_ROOT / base["base_file"]
        self.assertEqual(
            base["base_file_sha256"],
            hashlib.sha256(original_path.read_bytes()).hexdigest(),
        )

    def test_custom_set_fails_closed_for_undeclared_tasks(self) -> None:
        with self.assertRaisesRegex(KeyError, "does not declare task"):
            resolve_task_instruction_path("press_button", CUSTOM_SET)

    def test_catalog_identifiers_reject_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "instruction_set"):
            resolve_task_instruction_path(TASK_NAME, "../contract_clarified_v1")
        with self.assertRaisesRegex(ValueError, "task_name"):
            resolve_task_instruction_path("../blocks_ranking_try", CUSTOM_SET)


if __name__ == "__main__":
    unittest.main()
