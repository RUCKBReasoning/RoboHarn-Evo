from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
DESCRIPTION_UTILS = REPO_ROOT / "description" / "utils"
if str(DESCRIPTION_UTILS) not in sys.path:
    sys.path.insert(0, str(DESCRIPTION_UTILS))

from generate_episode_instructions import load_task_instructions  # noqa: E402


INSTRUCTION_SET = "contract_clarified_six_tasks_v1"
FORMAL_TASKS = {
    "rearrange_blocks",
    "swap_blocks",
    "press_button",
    "place_block_mat",
    "put_back_block",
    "cover_blocks",
}


class SixTaskInstructionSetTest(unittest.TestCase):
    def test_all_six_task_instructions_load_and_match_the_manifest(self) -> None:
        manifest_path = (
            REPO_ROOT
            / "description"
            / "task_instruction_sets"
            / INSTRUCTION_SET
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(set(manifest["tasks"]), FORMAL_TASKS)
        for task_name in sorted(FORMAL_TASKS):
            payload = load_task_instructions(task_name, INSTRUCTION_SET)
            self.assertEqual(payload["instruction_set_id"], INSTRUCTION_SET)
            self.assertEqual(payload["task_name"], task_name)
            self.assertEqual(payload["seen"], payload["unseen"])
            self.assertEqual(len(payload["seen"]), 1)
            self.assertIn("The task is complete", payload["seen"][0])

    def test_each_instruction_keeps_its_defining_requirement(self) -> None:
        expected_fragments = {
            "rearrange_blocks": "press and release the button exactly once",
            "swap_blocks": "initially empty tray as temporary space",
            "press_button": "separate complete press followed by release",
            "place_block_mat": "back onto its own remembered original blue mat",
            "put_back_block": "while the block remains in the center",
            "cover_blocks": "place that same lid back at its own original starting position",
        }

        for task_name, fragment in expected_fragments.items():
            instruction = load_task_instructions(task_name, INSTRUCTION_SET)["unseen"][0]
            self.assertIn(fragment, instruction)


if __name__ == "__main__":
    unittest.main()
