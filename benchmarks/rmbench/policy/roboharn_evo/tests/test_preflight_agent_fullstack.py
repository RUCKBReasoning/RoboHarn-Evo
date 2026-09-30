from __future__ import annotations

import base64
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib

from policy.roboharn_evo.scripts import preflight_agent_fullstack as preflight


def _png_pixel(red: int, green: int, blue: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = bytes((0, red, green, blue))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


class _FakeTransport:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.invalid_mode = ""

    def request(
        self,
        *,
        url: str,
        timeout_sec: float,
        payload: dict | None = None,
        raw_body: bytes | None = None,
    ) -> preflight.HttpResult:
        del timeout_sec
        path = "/" + url.split("/", 3)[-1]
        if raw_body is not None:
            self.records.append({"path": path, "mode": "malformed_json_negative"})
            body = json.dumps(
                {
                    "error": "The request body is not valid JSON.",
                    "error_category": "malformed_json",
                }
            ).encode("utf-8")
            return preflight.HttpResult(status=400, body=body, latency_ms=0.25)
        assert payload is not None
        mode = self._mode(payload)
        image_payloads = self._image_payloads(payload)
        self.records.append(
            {
                "path": path,
                "mode": mode,
                "image_payloads": image_payloads,
                "payload": payload,
            }
        )
        response_payload = (
            {"unexpected": "schema"}
            if mode == self.invalid_mode
            else self._response(mode)
        )
        return preflight.HttpResult(
            status=200,
            body=json.dumps(response_payload).encode("utf-8"),
            latency_ms=1.25,
        )

    @staticmethod
    def _mode(payload: dict) -> str:
        if "planner_start_image_b64" in payload:
            return "planner"
        if "skill_payload" in payload:
            return "ood_detection"
        if "recovery_payload" in payload:
            if payload["recovery_payload"].get("mode") == "action_effect_verification":
                return "action_effect_verification"
            return "recovery_planning"
        if "raw_queries" in payload:
            return "perception_query_normalization"
        return "perception_query_generation"

    @staticmethod
    def _image_payloads(payload: dict) -> list[tuple[str, bytes]]:
        if "planner_start_image_b64" in payload:
            return [
                ("head", base64.b64decode(payload["planner_start_image_b64"])),
                ("third", base64.b64decode(payload["planner_end_image_b64"])),
            ]
        if isinstance(payload.get("media"), list):
            return [
                (str(item.get("camera", "")), base64.b64decode(item["data"]))
                for item in payload["media"]
                if isinstance(item, dict) and item.get("type") == "image"
            ]
        if isinstance(payload.get("image_b64_by_camera"), dict):
            return [
                (str(camera), base64.b64decode(encoded))
                for camera, encoded in payload["image_b64_by_camera"].items()
            ]
        return []

    @staticmethod
    def _response(mode: str) -> dict:
        if mode == "planner":
            return {
                "commit_label": "no_update",
                "memory_text": "No task-relevant state change has occurred.",
                "selected_skill": "monitored-subtask-execution",
                "subtask_text": "Observe the visible movable block.",
                "preferred_arm": "either",
            }
        if mode == "ood_detection":
            return {
                "OOD_scenario": "none",
                "reason": "Both supplied views contain usable visual evidence.",
                "confidence": 0.9,
            }
        if mode == "recovery_planning":
            return {
                "recovery_workflow": "reobserve-scene",
                "selected_arm": "none",
                "tool_calls": [
                    {
                        "tool_name": "reobserve_scene",
                        "args": {},
                        "reason": "Acquire one fresh observation.",
                    }
                ],
                "post_recovery_intent": "replan",
                "reason": "The preflight requests an observation-only recovery.",
                "stop_condition": "Stop after one fresh observation.",
            }
        if mode == "action_effect_verification":
            return {
                "effect_verified": "unverified",
                "effect_type": "observation_refresh",
                "confidence": 0.5,
                "evidence_summary": "The images are available but no physical effect is expected.",
                "failure_reason": "",
                "next_constraint": "Use the observation for replanning.",
                "memory_update": "A fresh observation was acquired.",
                "subtask_status": "in_progress",
                "recommended_control": "replan",
            }
        return {
            "queries": [
                {
                    "object_id": "movable_block",
                    "text_prompt": "movable block",
                    "role": "target",
                    "instance_hint": "",
                    "reason": "The current subtask refers to the block.",
                }
            ]
        }

class FullstackPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = _FakeTransport()
        self.base_url = "http://127.0.0.1:19105"

    def _image_files(self, root: Path) -> tuple[Path, Path, bytes, bytes]:
        head_bytes = _png_pixel(255, 0, 0)
        third_bytes = _png_pixel(0, 0, 255)
        head_path = root / "head.png"
        third_path = root / "third.png"
        head_path.write_bytes(head_bytes)
        third_path.write_bytes(third_bytes)
        return head_path, third_path, head_bytes, third_bytes

    def test_exercises_all_roles_with_ordered_images_and_writes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            head_path, third_path, head_bytes, third_bytes = self._image_files(root)
            output_path = root / "preflight.json"

            with mock.patch.object(preflight, "_http_request", side_effect=self.transport.request):
                report = preflight.run_preflight(
                    gateway_base_url=self.base_url,
                    head_image_path=head_path,
                    third_image_path=third_path,
                    output_json_path=output_path,
                    timeout_sec=5,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["probe_count"], 7)
            self.assertEqual(report["passed_probe_count"], 7)
            self.assertEqual(report["fallback_count"], 0)
            self.assertEqual(
                [(item["endpoint"], item["mode"]) for item in report["probes"]],
                [
                    ("/plan", "planner"),
                    ("/ood", "ood_detection"),
                    ("/recover", "recovery_planning"),
                    ("/recover", "action_effect_verification"),
                    ("/perception_queries", "perception_query_generation"),
                    ("/normalize_perception_queries", "perception_query_normalization"),
                    ("/plan", "malformed_json_negative"),
                ],
            )
            for probe_record in report["probes"][:-1]:
                self.assertEqual(probe_record["request_image_count"], 2)
                self.assertEqual(
                    [item["camera"] for item in probe_record["request_images"]],
                    ["head", "third"],
                )
                self.assertEqual(
                    [item["order"] for item in probe_record["request_images"]],
                    [0, 1],
                )
                self.assertEqual(len(probe_record["response_sha256"]), 64)
                self.assertNotIn("response", probe_record)

            positive_requests = self.transport.records[:-1]
            self.assertEqual(len(positive_requests), 6)
            for request_record in positive_requests:
                self.assertEqual(
                    request_record["image_payloads"],
                    [("head", head_bytes), ("third", third_bytes)],
                )
            self.assertEqual(self.transport.records[-1]["mode"], "malformed_json_negative")
            action_effect_request = next(
                item
                for item in positive_requests
                if item["mode"] == "action_effect_verification"
            )
            self.assertEqual(
                action_effect_request["payload"]["recovery_payload"]["verification_skill"]["body"],
                preflight.ACTION_EFFECT_SKILL_PATH.read_text(encoding="utf-8"),
            )
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), report)

    def test_long_planner_probe_matches_agent_payload_shape_and_reports_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            head_path, third_path, _, _ = self._image_files(root)
            images = [
                preflight.ImageInput.load(camera="head", path=head_path),
                preflight.ImageInput.load(camera="third", path=third_path),
            ]

            planner = preflight.build_probe_specs(
                images,
                planner_task_min_bytes=53_342,
            )[0]

            self.assertEqual(planner.endpoint, "/plan")
            self.assertEqual(
                set(planner.payload),
                {
                    "task",
                    "previous_memory_text",
                    "planner_state",
                    "planner_start_image_b64",
                    "planner_end_image_b64",
                    "prompt",
                },
            )
            self.assertEqual(planner.payload["prompt"].count(planner.payload["task"]), 1)
            self.assertIn("memory_harness", json.loads(planner.payload["task"]))
            request_bytes, non_image_bytes = preflight._payload_byte_counts(planner.payload)
            task_bytes = len(planner.payload["task"].encode("utf-8"))
            self.assertGreaterEqual(task_bytes, 53_342)
            self.assertLess(task_bytes, 54_000)
            self.assertGreater(non_image_bytes, task_bytes * 2)
            self.assertGreater(request_bytes, non_image_bytes)
            assert planner.request_metadata is not None
            self.assertEqual(
                planner.request_metadata[
                    "planner_task_actual_utf8_byte_count"
                ],
                task_bytes,
            )
            self.assertTrue(
                planner.request_metadata["planner_rendered_prompt_present"]
            )
            self.assertGreater(
                planner.request_metadata["planner_diagnostic_context_record_count"],
                0,
            )

    def test_long_planner_preflight_checks_health_contract_and_records_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            head_path, third_path, _, _ = self._image_files(root)
            output_path = root / "long-preflight.json"
            health_result = preflight.HttpResult(
                status=200,
                body=json.dumps(
                    {
                        "planner_prompt_mode": "rendered_system_once",
                        "planner_max_output_tokens": 8192,
                    }
                ).encode("utf-8"),
                latency_ms=0.5,
            )

            with mock.patch.object(
                preflight,
                "_http_get",
                return_value=health_result,
            ), mock.patch.object(
                preflight,
                "_http_request",
                side_effect=self.transport.request,
            ):
                report = preflight.run_preflight(
                    gateway_base_url=self.base_url,
                    head_image_path=head_path,
                    third_image_path=third_path,
                    output_json_path=output_path,
                    timeout_sec=5,
                    include_malformed_json_probe=False,
                    planner_task_min_bytes=53_342,
                    expected_planner_prompt_mode="rendered_system_once",
                    expected_planner_max_output_tokens=8192,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["probe_count"], 7)
            self.assertEqual(
                (report["probes"][0]["endpoint"], report["probes"][0]["mode"]),
                ("/health", "planner_context_contract"),
            )
            planner_record = report["probes"][1]
            self.assertEqual(planner_record["request_image_count"], 2)
            self.assertGreaterEqual(
                planner_record["planner_task_actual_utf8_byte_count"],
                53_342,
            )
            self.assertEqual(
                report["planner_payload"]["actual_task_utf8_byte_count"],
                planner_record["planner_task_actual_utf8_byte_count"],
            )
            self.assertTrue(report["planner_payload"]["rendered_prompt_present"])

    def test_planner_context_health_mismatch_fails_closed(self) -> None:
        health_result = preflight.HttpResult(
            status=200,
            body=json.dumps(
                {
                    "planner_prompt_mode": "duplicated_dynamic_context",
                    "planner_max_output_tokens": 32768,
                }
            ).encode("utf-8"),
            latency_ms=0.5,
        )
        with mock.patch.object(preflight, "_http_get", return_value=health_result):
            record = preflight.run_planner_context_contract_probe(
                base_url=self.base_url,
                timeout_sec=5,
                expected_prompt_mode="rendered_system_once",
                expected_max_output_tokens=8192,
            )
        self.assertEqual(record["status"], "failed")
        self.assertIn("planner_prompt_mode", record["error"])
        self.assertIn("planner_max_output_tokens", record["error"])

    def test_planner_payload_limit_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            head_path, third_path, _, _ = self._image_files(root)
            images = [
                preflight.ImageInput.load(camera="head", path=head_path),
                preflight.ImageInput.load(camera="third", path=third_path),
            ]
            for invalid in (-1, preflight.MAX_PLANNER_TASK_MIN_BYTES + 1):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    preflight.build_probe_specs(
                        images,
                        planner_task_min_bytes=invalid,
                    )

    def test_transport_opener_disables_environment_proxies(self) -> None:
        proxy_handlers = [
            handler
            for handler in preflight._NO_PROXY_OPENER.handlers
            if isinstance(handler, preflight.request.ProxyHandler)
        ]
        # ``build_opener`` omits an empty ProxyHandler from the final handler
        # list.  In either representation, no handler may carry proxy routes.
        self.assertEqual(proxy_handlers, [])

    def test_schema_failure_is_reported_and_cli_returns_nonzero(self) -> None:
        self.transport.invalid_mode = "ood_detection"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            head_path, third_path, _, _ = self._image_files(root)
            output_path = root / "failed.json"

            with mock.patch.object(preflight, "_http_request", side_effect=self.transport.request):
                exit_code = preflight.main(
                    [
                        "--gateway-base-url",
                        self.base_url,
                        "--head-image",
                        str(head_path),
                        "--third-image",
                        str(third_path),
                        "--output-json",
                        str(output_path),
                        "--timeout-sec",
                        "5",
                        "--skip-malformed-json-probe",
                    ]
                )

            self.assertEqual(exit_code, 1)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(report["failed_probe_count"], 1)
            self.assertEqual(report["failures"][0]["endpoint"], "/ood")
            self.assertIn("schema validation", report["failures"][0]["error"])

    def test_rejects_nonlocal_or_decorated_gateway_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            head_path, third_path, _, _ = self._image_files(root)
            forbidden = (
                "http://example.com:9105",
                "http://127.0.0.2:9105",
                "http://user:pass@127.0.0.1:9105",
                "http://127.0.0.1:9105?",
                "http://127.0.0.1:9105?remote=true",
                "http://127.0.0.1:9105#",
                "http://127.0.0.1:9105#fragment",
            )
            for base_url in forbidden:
                with self.subTest(base_url=base_url), self.assertRaises(ValueError):
                    preflight.run_preflight(
                        gateway_base_url=base_url,
                        head_image_path=head_path,
                        third_image_path=third_path,
                        output_json_path=root / "unused.json",
                        timeout_sec=5,
                    )

    def test_accepts_explicit_loopback_gateway_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            head_path, third_path, _, _ = self._image_files(root)
            for base_url in (
                "http://localhost:9105",
                "http://127.0.0.1:9105",
                "http://[::1]:9105",
            ):
                with self.subTest(base_url=base_url), mock.patch.object(
                    preflight,
                    "_http_request",
                    side_effect=self.transport.request,
                ):
                    report = preflight.run_preflight(
                        gateway_base_url=base_url,
                        head_image_path=head_path,
                        third_image_path=third_path,
                        output_json_path=root / f"{len(self.transport.records)}.json",
                        timeout_sec=5,
                        include_malformed_json_probe=False,
                    )
                    self.assertTrue(report["passed"])

    def test_semantic_validators_reject_empty_or_task_free_outputs(self) -> None:
        with self.assertRaisesRegex(preflight.SchemaValidationError, "must not be empty"):
            preflight.validate_ood_response(
                {"OOD_scenario": "none", "reason": "", "confidence": 0.9}
            )
        with self.assertRaisesRegex(preflight.SchemaValidationError, "task-related"):
            preflight.validate_ood_response(
                {"OOD_scenario": "none", "reason": "Everything is fine.", "confidence": 0.9}
            )

        recovery = self.transport._response("recovery_planning")
        invalid_recovery_cases = (
            ({**recovery, "tool_calls": []}, "at least one recovery action"),
            (
                {
                    **recovery,
                    "tool_calls": [
                        {
                            "tool_name": "retreat_arm",
                            "args": {},
                            "reason": "Move away.",
                        }
                    ],
                },
                "contain reobserve_scene",
            ),
            ({**recovery, "reason": ""}, "must not be empty"),
            ({**recovery, "stop_condition": ""}, "must not be empty"),
        )
        for invalid, expected_error in invalid_recovery_cases:
            with self.subTest(expected_error=expected_error), self.assertRaisesRegex(
                preflight.SchemaValidationError,
                expected_error,
            ):
                preflight.validate_recovery_response(invalid)

        effect = self.transport._response("action_effect_verification")
        for key in ("evidence_summary", "next_constraint"):
            invalid = dict(effect)
            invalid[key] = ""
            with self.subTest(key=key), self.assertRaisesRegex(
                preflight.SchemaValidationError,
                "must not be empty",
            ):
                preflight.validate_action_effect_response(invalid)

        no_target = {
            "queries": [
                {
                    "object_id": "table",
                    "text_prompt": "table",
                    "role": "context",
                    "instance_hint": "",
                    "reason": "Support context.",
                }
            ]
        }
        with self.assertRaisesRegex(preflight.SchemaValidationError, "at least one target"):
            preflight.validate_perception_query_response(no_target)
        with self.assertRaisesRegex(preflight.SchemaValidationError, "at least one target"):
            preflight.validate_perception_normalization_response(
                {
                    "queries": [
                        {
                            "object_id": "table",
                            "text_prompt": "table",
                            "role": "context",
                            "instance_hint": "",
                            "reason": "Support context.",
                        }
                    ]
                }
            )

    def test_malformed_json_probe_requires_exact_400_category(self) -> None:
        cases = (
            (
                preflight.HttpResult(
                    status=500,
                    body=json.dumps({"error_category": "malformed_json"}).encode(),
                    latency_ms=1.0,
                ),
                "HTTP 400",
            ),
            (
                preflight.HttpResult(
                    status=400,
                    body=json.dumps({"error_category": "invalid_request"}).encode(),
                    latency_ms=1.0,
                ),
                "error_category=malformed_json",
            ),
            (
                preflight.HttpResult(
                    status=None,
                    body=b"",
                    latency_ms=1.0,
                    transport_error=ConnectionResetError("closed"),
                ),
                "transport",
            ),
        )
        for result, expected_error in cases:
            with self.subTest(expected_error=expected_error), mock.patch.object(
                preflight,
                "_http_request",
                return_value=result,
            ):
                record = preflight.run_malformed_json_probe(
                    base_url=self.base_url,
                    timeout_sec=5,
                )
                self.assertEqual(record["status"], "failed")
                self.assertIn(expected_error, record["error"])

        accepted = preflight.HttpResult(
            status=400,
            body=json.dumps({"error_category": "malformed_json"}).encode(),
            latency_ms=1.0,
        )
        with mock.patch.object(preflight, "_http_request", return_value=accepted):
            record = preflight.run_malformed_json_probe(
                base_url=self.base_url,
                timeout_sec=5,
            )
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["error_category"], "malformed_json")


if __name__ == "__main__":
    unittest.main()
