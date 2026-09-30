from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from policy.roboharn_evo.agent.components.agent_tools.agent_tools import AgentTools, build_tool_from_func


class SAM2SegmentationToolTest(unittest.TestCase):
    def test_segment_object_tool_schema_exposes_prompt_fields(self) -> None:
        tool = build_tool_from_func(AgentTools.segment_object, service_name="AgentTools")

        self.assertEqual(tool.tool_name, "segment_object")
        self.assertEqual(tool.parameters["required"], ["object_id"])
        properties = tool.parameters["properties"]
        self.assertIn("backend", properties)
        self.assertIn("text_prompt", properties)
        self.assertIn("bbox_xyxy", properties)
        self.assertIn("point_coords", properties)
        self.assertIn("point_labels", properties)

    def test_parse_optional_json_array(self) -> None:
        parser = AgentTools.__new__(AgentTools)

        self.assertIsNone(parser._parse_optional_json_array("", "bbox_xyxy"))
        self.assertEqual(parser._parse_optional_json_array("[1, 2, 3, 4]", "bbox_xyxy"), [1, 2, 3, 4])
        self.assertEqual(parser._parse_optional_json_array("[[10, 20]]", "point_coords"), [[10, 20]])
        with self.assertRaises(ValueError):
            parser._parse_optional_json_array('"not-array"', "bbox_xyxy")

    def test_sam2_client_payload_uses_paths(self) -> None:
        from policy.roboharn_evo.agent.perception.sam2_client import SAM2SegmentationClient

        captured = {}

        def fake_request(self, method, path, payload=None):
            captured["method"] = method
            captured["path"] = path
            captured["payload"] = payload
            return {"success": True}

        with mock.patch.object(SAM2SegmentationClient, "_request", fake_request):
            client = SAM2SegmentationClient(base_url="http://127.0.0.1:9201")
            result = client.segment_image(
                image_path=Path("/tmp/input.png"),
                object_id="button",
                bbox_xyxy=[1, 2, 3, 4],
                output_dir=Path("/tmp/masks"),
            )

        self.assertEqual(result, {"success": True})
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/segment_image")
        self.assertEqual(captured["payload"]["image_path"], "/tmp/input.png")
        self.assertEqual(captured["payload"]["output_dir"], "/tmp/masks")

    def test_sam3_client_payload_uses_text_prompt(self) -> None:
        from policy.roboharn_evo.agent.perception.sam3_client import SAM3SegmentationClient

        captured = {}

        def fake_request(self, method, path, payload=None):
            captured["method"] = method
            captured["path"] = path
            captured["payload"] = payload
            return {"success": True}

        with mock.patch.object(SAM3SegmentationClient, "_request", fake_request):
            client = SAM3SegmentationClient(base_url="http://127.0.0.1:9301")
            result = client.segment_image(
                image_path=Path("/tmp/input.png"),
                object_id="button",
                text_prompt="button",
                output_dir=Path("/tmp/masks"),
                top_k=2,
            )

        self.assertEqual(result, {"success": True})
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/segment_image")
        self.assertEqual(captured["payload"]["text_prompt"], "button")
        self.assertEqual(captured["payload"]["top_k"], 2)


if __name__ == "__main__":
    unittest.main()
