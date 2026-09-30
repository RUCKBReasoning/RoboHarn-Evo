from __future__ import annotations

import unittest

from policy.roboharn_evo.scripts.serve_qwen_planner import validate_perception_queries


class PerceptionQueriesTest(unittest.TestCase):
    def test_validate_perception_queries_deduplicates_exact_object_ids_and_caps(self) -> None:
        result = validate_perception_queries(
            {
                "queries": [
                    {"object_id": "block", "text_prompt": "red block", "role": "target", "instance_hint": "left", "reason": "active target"},
                    {"object_id": "block", "text_prompt": "cube"},
                    {"object_id": "button", "role": "invalid-role"},
                    {"object_id": "drawer handle"},
                    {"object_id": "extra"},
                ]
            }
        )

        self.assertEqual(
            result,
            {
                "queries": [
                    {"object_id": "block", "text_prompt": "red block", "role": "target", "instance_hint": "left", "reason": "active target"},
                    {"object_id": "button", "text_prompt": "button", "role": "context", "instance_hint": "", "reason": ""},
                    {"object_id": "drawer_handle", "text_prompt": "drawer handle", "role": "context", "instance_hint": "", "reason": ""},
                ]
            },
        )

    def test_schema_validation_does_not_move_instance_descriptors_out_of_object_id(self) -> None:
        result = validate_perception_queries(
            {
                "queries": [
                    {"object_id": "left brown lid", "text_prompt": "brown lid", "role": "tool", "reason": "left-to-right order"},
                    {"object_id": "lid", "text_prompt": "lid", "role": "context", "instance_hint": "middle"},
                    {"object_id": "right block", "role": "target"},
                ]
            }
        )

        self.assertEqual(
            result,
            {
                "queries": [
                    {
                        "object_id": "left_brown_lid",
                        "text_prompt": "brown lid",
                        "role": "tool",
                        "instance_hint": "",
                        "reason": "left-to-right order",
                    },
                    {"object_id": "lid", "text_prompt": "lid", "role": "context", "instance_hint": "middle", "reason": ""},
                    {"object_id": "right_block", "text_prompt": "right block", "role": "target", "instance_hint": "", "reason": ""},
                ]
            },
        )

    def test_schema_validation_does_not_merge_semantically_related_names(self) -> None:
        result = validate_perception_queries(
            {
                "queries": [
                    {"object_id": "lid", "role": "context", "instance_hint": "middle"},
                    {"object_id": "left lid", "role": "tool", "reason": "active tool"},
                ]
            }
        )

        self.assertEqual(
            result,
            {
                "queries": [
                    {"object_id": "lid", "text_prompt": "lid", "role": "context", "instance_hint": "middle", "reason": ""},
                    {"object_id": "left_lid", "text_prompt": "left lid", "role": "tool", "instance_hint": "", "reason": "active tool"},
                ]
            },
        )

    def test_schema_preserves_agent_selected_stable_identity_fields(self) -> None:
        result = validate_perception_queries(
            {
                "queries": [
                    {
                        "object_id": "component",
                        "text_prompt": "component",
                        "role": "target",
                        "instance_ref": "track_0042",
                        "oracle_id": "components_7",
                        "reason": "selected from advertised candidates",
                    }
                ]
            }
        )

        self.assertEqual(result["queries"][0]["instance_ref"], "track_0042")
        self.assertEqual(result["queries"][0]["oracle_id"], "components_7")

    def test_schema_keeps_distinct_instances_of_the_same_category(self) -> None:
        result = validate_perception_queries(
            {
                "queries": [
                    {"object_id": "component", "role": "target", "instance_ref": "track_a"},
                    {"object_id": "component", "role": "target", "instance_ref": "track_b"},
                ]
            }
        )

        self.assertEqual(
            [item["instance_ref"] for item in result["queries"]],
            ["track_a", "track_b"],
        )

    def test_schema_keeps_distinct_unbound_instance_hints_for_discovery(self) -> None:
        result = validate_perception_queries(
            {
                "queries": [
                    {
                        "object_id": "movable_object",
                        "text_prompt": "blue movable object",
                        "role": "target",
                        "instance_hint": "blue",
                    },
                    {
                        "object_id": "movable_object",
                        "text_prompt": "red movable object",
                        "role": "tool",
                        "instance_hint": "red",
                    },
                ]
            }
        )

        self.assertEqual(
            [item["instance_hint"] for item in result["queries"]],
            ["blue", "red"],
        )
        self.assertEqual(
            [item["role"] for item in result["queries"]],
            ["target", "tool"],
        )

    def test_schema_preserves_reference_set_relation_without_single_binding(self) -> None:
        result = validate_perception_queries(
            {
                "queries": [
                    {
                        "object_id": "mats",
                        "text_prompt": "the complete set of four mats",
                        "role": "context",
                        "entity_scope": "reference_set",
                        "placement_relation": "center_of",
                        "expected_count": 4,
                        "instance_ref": "track_0001",
                        "oracle_id": "mats_0",
                    }
                ]
            }
        )

        query = result["queries"][0]
        self.assertEqual(query["entity_scope"], "reference_set")
        self.assertEqual(query["placement_relation"], "center_of")
        self.assertEqual(query["expected_count"], 4)
        self.assertNotIn("instance_ref", query)
        self.assertNotIn("oracle_id", query)


if __name__ == "__main__":
    unittest.main()
