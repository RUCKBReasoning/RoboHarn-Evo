from __future__ import annotations

import unittest

from policy.roboharn_evo.scripts.visualize_rollout_contact_sheet import scene_memory_focus_boxes, segmentation_boxes


class RolloutContactSheetVisualizationTest(unittest.TestCase):
    def test_draws_sam_candidates_and_scene_memory_focus_separately(self) -> None:
        records = [
            {
                "event": "observation_preprocess",
                "env_step": 3,
                "segmentation": [
                    {
                        "success": True,
                        "object_id": "lid",
                        "camera": "head",
                        "detections": [
                            {"bbox_xyxy": [220, 100, 280, 140], "score": 0.91},
                            {"bbox_xyxy": [70, 100, 120, 140], "score": 0.41},
                        ],
                    }
                ],
            },
            {
                "event": "scene_memory_update",
                "env_step": 3,
                "scene_memory": {
                    "instances": [
                        {
                            "instance_id": "lid_left",
                            "camera": "head",
                            "bbox_xyxy": [70, 100, 120, 140],
                            "score": 0.41,
                        }
                    ],
                    "task_focus": {
                        "tool_instances": ["lid_left"],
                        "target_instances": [],
                    },
                },
            },
        ]

        candidate_boxes = segmentation_boxes(records, "head")
        focus_boxes = scene_memory_focus_boxes(records, "head")

        self.assertEqual(len(candidate_boxes), 2)
        self.assertEqual(candidate_boxes[0]["style"], "candidate")
        self.assertEqual(len(focus_boxes), 1)
        self.assertEqual(focus_boxes[0]["label"], "SELECTED tool:lid_left")
        self.assertEqual(focus_boxes[0]["style"], "selected_tool")


if __name__ == "__main__":
    unittest.main()
