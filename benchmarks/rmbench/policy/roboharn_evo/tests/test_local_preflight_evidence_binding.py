from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from policy.roboharn_evo.scripts.bind_local_fullstack_preflight_evidence import (
    EvidenceBindingError,
    build_binding,
)


MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
ENGINE = "http://127.0.0.1:8000/v1"
GATEWAY = "http://127.0.0.1:9105"
HEAD_SHA256 = "c" * 64
THIRD_SHA256 = "d" * 64
PLANNER_TASK_MIN_BYTES = 53_342
PLANNER_TASK_ACTUAL_BYTES = 53_500
PLANNER_NON_IMAGE_BYTES = 110_000
PLANNER_REQUEST_BYTES = 120_000
PLANNER_PROMPT_BYTES = 55_000
SERVING_FRAMEWORK = "vLLM 0.26.1-test"


class LocalPreflightEvidenceBindingTest(unittest.TestCase):
    def _fixtures(self, root: Path) -> dict[str, Path]:
        runtime = root / "local_serving_runtime_identity.json"
        runtime.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "vllm": {"version_output": SERVING_FRAMEWORK},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runtime_sha = hashlib.sha256(runtime.read_bytes()).hexdigest()
        models_sha = "a" * 64
        models_identity_sha = "e" * 64
        health_payload = {
            "status": "ok",
            "local_only": True,
            "model": MODEL,
            "service_url": GATEWAY,
            "base_url": ENGINE,
            "serving_framework": SERVING_FRAMEWORK,
            "planner_prompt_mode": "rendered_system_once",
            "planner_max_output_tokens": 8192,
            "fallback_enabled": False,
            "response_storage": "disabled",
            "upstream_model_identity": {
                "verified": True,
                "expected_model": MODEL,
                "models_response_sha256": models_sha,
                "models_identity_sha256": models_identity_sha,
            },
            "serving_runtime_identity": {
                "path": str(runtime),
                "sha256": runtime_sha,
            },
        }
        before = root / "before.json"
        after = root / "after.json"
        serialized_health = json.dumps(health_payload, indent=2, sort_keys=True) + "\n"
        before.write_text(serialized_health, encoding="utf-8")
        after.write_text(serialized_health, encoding="utf-8")

        modality = root / "modality.json"
        modality.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "model": MODEL,
                    "engine_base_url": ENGINE,
                    "fallback_count": 0,
                    "errors": [],
                    "model_check": {
                        "exact_model_found": True,
                        "response_sha256": "b" * 64,
                        "identity_sha256": models_identity_sha,
                    },
                    "requests": [
                        {
                            "request_mode": "text_only",
                            "status": "passed",
                            "fallback_count": 0,
                            "image_sha256": [],
                        },
                        {
                            "request_mode": "head_image",
                            "status": "passed",
                            "fallback_count": 0,
                            "image_sha256": [HEAD_SHA256],
                        },
                        {
                            "request_mode": "head_then_third_images",
                            "status": "passed",
                            "fallback_count": 0,
                            "image_sha256": [HEAD_SHA256, THIRD_SHA256],
                        },
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        fullstack = root / "fullstack.json"
        modes = (
            "planner_context_contract",
            "planner",
            "ood_detection",
            "recovery_planning",
            "action_effect_verification",
            "perception_query_generation",
            "perception_query_normalization",
            "malformed_json_negative",
        )
        source_images = [
            {"order": 0, "camera": "head", "sha256": HEAD_SHA256},
            {"order": 1, "camera": "third", "sha256": THIRD_SHA256},
        ]
        probes = [
            {"mode": mode, "status": "passed", "fallback_count": 0}
            for mode in modes
        ]
        probes[0].update(
            {
                "request_image_count": 0,
                "request_images": [],
                "expected_planner_prompt_mode": "rendered_system_once",
                "planner_prompt_mode": "rendered_system_once",
                "expected_planner_max_output_tokens": 8192,
                "planner_max_output_tokens": 8192,
            }
        )
        probes[1].update(
            {
                "request_image_count": 2,
                "request_images": source_images,
                "planner_task_requested_min_utf8_byte_count": PLANNER_TASK_MIN_BYTES,
                "planner_task_actual_utf8_byte_count": PLANNER_TASK_ACTUAL_BYTES,
                "planner_payload_actual_non_image_byte_count": PLANNER_NON_IMAGE_BYTES,
                "planner_payload_actual_request_body_byte_count": PLANNER_REQUEST_BYTES,
                "planner_rendered_prompt_byte_count": PLANNER_PROMPT_BYTES,
                "planner_diagnostic_context_record_count": 100,
                "planner_rendered_prompt_present": True,
            }
        )
        fullstack.write_text(
            json.dumps(
                {
                    "passed": True,
                    "gateway_base_url": GATEWAY,
                    "fallback_count": 0,
                    "probe_count": len(modes),
                    "passed_probe_count": len(modes),
                    "failed_probe_count": 0,
                    "failures": [],
                    "source_images": source_images,
                    "planner_payload": {
                        "requested_min_task_utf8_byte_count": PLANNER_TASK_MIN_BYTES,
                        "actual_task_utf8_byte_count": PLANNER_TASK_ACTUAL_BYTES,
                        "actual_non_image_byte_count": PLANNER_NON_IMAGE_BYTES,
                        "actual_request_body_byte_count": PLANNER_REQUEST_BYTES,
                        "request_image_count": 2,
                        "rendered_prompt_present": True,
                    },
                    "planner_context_contract": {
                        "expected_prompt_mode": "rendered_system_once",
                        "expected_max_output_tokens": 8192,
                    },
                    "probes": probes,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "runtime": runtime,
            "before": before,
            "after": after,
            "modality": modality,
            "fullstack": fullstack,
        }

    def _build(
        self,
        paths: dict[str, Path],
        *,
        planner_task_min_bytes: int = PLANNER_TASK_MIN_BYTES,
        planner_max_output_tokens: int = 8192,
    ):
        return build_binding(
            gateway_health_before_path=paths["before"],
            gateway_health_after_path=paths["after"],
            modality_report_path=paths["modality"],
            fullstack_report_path=paths["fullstack"],
            expected_model=MODEL,
            expected_engine_base_url=ENGINE,
            expected_gateway_base_url=GATEWAY,
            expected_head_image_sha256=HEAD_SHA256,
            expected_third_image_sha256=THIRD_SHA256,
            expected_planner_task_min_bytes=planner_task_min_bytes,
            expected_planner_max_output_tokens=planner_max_output_tokens,
        )

    def test_binds_reports_to_unchanged_gateway_and_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_binding_") as directory:
            paths = self._fixtures(Path(directory))
            binding, runtime_path = self._build(paths)

        self.assertEqual(binding["status"], "passed")
        self.assertEqual(binding["schema_version"], 3)
        self.assertTrue(
            binding["gateway_service_identity"]["unchanged_during_preflights"]
        )
        self.assertEqual(
            binding["gateway_service_identity"]["planner_prompt_mode"],
            "rendered_system_once",
        )
        self.assertEqual(
            binding["gateway_service_identity"]["planner_max_output_tokens"],
            8192,
        )
        self.assertEqual(
            binding["gateway_service_identity"]["serving_framework"],
            SERVING_FRAMEWORK,
        )
        self.assertEqual(
            binding["agent_fullstack_preflight"][
                "planner_task_requested_min_utf8_byte_count"
            ],
            PLANNER_TASK_MIN_BYTES,
        )
        self.assertEqual(
            binding["agent_fullstack_preflight"]["planner_task_actual_utf8_byte_count"],
            PLANNER_TASK_ACTUAL_BYTES,
        )
        self.assertTrue(
            binding["agent_fullstack_preflight"]["planner_rendered_prompt_present"]
        )
        self.assertEqual(
            binding["agent_fullstack_preflight"]["planner_request_image_count"],
            2,
        )
        self.assertEqual(runtime_path, paths["runtime"])
        self.assertEqual(binding["fallback_count"], 0)

    def test_rejects_gateway_change_during_preflights(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_gateway_change_") as directory:
            paths = self._fixtures(Path(directory))
            payload = json.loads(paths["after"].read_text(encoding="utf-8"))
            payload["max_concurrent_requests"] = 2
            paths["after"].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceBindingError, "identity changed"):
                self._build(paths)

    def test_rejects_legacy_planner_prompt_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_legacy_prompt_") as directory:
            paths = self._fixtures(Path(directory))
            for key in ("before", "after"):
                payload = json.loads(paths[key].read_text(encoding="utf-8"))
                payload["planner_prompt_mode"] = "legacy_duplicate"
                paths[key].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceBindingError, "planner_prompt_mode"):
                self._build(paths)

    def test_rejects_unbounded_planner_output_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_planner_budget_") as directory:
            paths = self._fixtures(Path(directory))
            for key in ("before", "after"):
                payload = json.loads(paths[key].read_text(encoding="utf-8"))
                payload["planner_max_output_tokens"] = 32768
                paths[key].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceBindingError, "planner_max_output_tokens"):
                self._build(paths)

    def test_accepts_explicit_nondefault_planner_output_cap(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_planner_cap_") as directory:
            paths = self._fixtures(Path(directory))
            for key in ("before", "after"):
                payload = json.loads(paths[key].read_text(encoding="utf-8"))
                payload["planner_max_output_tokens"] = 4096
                paths[key].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            payload = json.loads(paths["fullstack"].read_text(encoding="utf-8"))
            payload["planner_context_contract"]["expected_max_output_tokens"] = 4096
            context_probe = next(
                item
                for item in payload["probes"]
                if item["mode"] == "planner_context_contract"
            )
            context_probe["expected_planner_max_output_tokens"] = 4096
            context_probe["planner_max_output_tokens"] = 4096
            paths["fullstack"].write_text(json.dumps(payload) + "\n", encoding="utf-8")

            binding, _ = self._build(paths, planner_max_output_tokens=4096)

        self.assertEqual(
            binding["gateway_service_identity"]["planner_max_output_tokens"],
            4096,
        )

    def test_rejects_short_planner_task_probe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_short_task_") as directory:
            paths = self._fixtures(Path(directory))
            payload = json.loads(paths["fullstack"].read_text(encoding="utf-8"))
            payload["planner_payload"]["actual_task_utf8_byte_count"] = 53_341
            paths["fullstack"].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceBindingError, "expected at least 53342"):
                self._build(paths)

    def test_rejects_missing_rendered_prompt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_bad_long_probe_") as directory:
            paths = self._fixtures(Path(directory))
            payload = json.loads(paths["fullstack"].read_text(encoding="utf-8"))
            payload["planner_payload"]["rendered_prompt_present"] = False
            paths["fullstack"].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceBindingError, "rendered_prompt_present"):
                self._build(paths)

    def test_rejects_planner_probe_without_two_bound_images(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_bad_images_") as directory:
            paths = self._fixtures(Path(directory))
            payload = json.loads(paths["fullstack"].read_text(encoding="utf-8"))
            planner_probe = next(
                item for item in payload["probes"] if item["mode"] == "planner"
            )
            planner_probe["request_images"] = planner_probe["request_images"][:1]
            paths["fullstack"].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceBindingError, "head and third"):
                self._build(paths)

    def test_accepts_different_raw_models_response_with_same_stable_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_stale_report_") as directory:
            paths = self._fixtures(Path(directory))
            payload = json.loads(paths["modality"].read_text(encoding="utf-8"))
            payload["model_check"]["response_sha256"] = "f" * 64
            paths["modality"].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            binding, _ = self._build(paths)
        self.assertEqual(
            binding["gateway_service_identity"]["modality_models_response_sha256"],
            "f" * 64,
        )

    def test_rejects_report_from_different_stable_models_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_stale_report_") as directory:
            paths = self._fixtures(Path(directory))
            payload = json.loads(paths["modality"].read_text(encoding="utf-8"))
            payload["model_check"]["identity_sha256"] = "f" * 64
            paths["modality"].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceBindingError, "stable identity SHA256"):
                self._build(paths)

    def test_rejects_runtime_identity_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_runtime_change_") as directory:
            paths = self._fixtures(Path(directory))
            paths["runtime"].write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(EvidenceBindingError, "identity SHA256"):
                self._build(paths)

    def test_rejects_serving_framework_runtime_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen_preflight_framework_mismatch_") as directory:
            paths = self._fixtures(Path(directory))
            for key in ("before", "after"):
                payload = json.loads(paths[key].read_text(encoding="utf-8"))
                payload["serving_framework"] = "vLLM different-version"
                paths[key].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                EvidenceBindingError,
                "serving_framework does not match.*vllm.version_output",
            ):
                self._build(paths)

    def test_rejects_missing_or_blank_serving_framework(self) -> None:
        for value in (None, "", "   "):
            with self.subTest(value=value), tempfile.TemporaryDirectory(
                prefix="qwen_preflight_missing_framework_"
            ) as directory:
                paths = self._fixtures(Path(directory))
                for key in ("before", "after"):
                    payload = json.loads(paths[key].read_text(encoding="utf-8"))
                    if value is None:
                        payload.pop("serving_framework")
                    else:
                        payload["serving_framework"] = value
                    paths[key].write_text(json.dumps(payload) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    EvidenceBindingError,
                    "serving_framework must be a non-empty string",
                ):
                    self._build(paths)

    def test_rejects_missing_or_blank_runtime_vllm_version(self) -> None:
        for value in (None, "", "   "):
            with self.subTest(value=value), tempfile.TemporaryDirectory(
                prefix="qwen_preflight_missing_runtime_version_"
            ) as directory:
                paths = self._fixtures(Path(directory))
                runtime = json.loads(paths["runtime"].read_text(encoding="utf-8"))
                if value is None:
                    runtime["vllm"].pop("version_output")
                else:
                    runtime["vllm"]["version_output"] = value
                paths["runtime"].write_text(
                    json.dumps(runtime) + "\n", encoding="utf-8"
                )
                runtime_sha256 = hashlib.sha256(paths["runtime"].read_bytes()).hexdigest()
                for key in ("before", "after"):
                    payload = json.loads(paths[key].read_text(encoding="utf-8"))
                    payload["serving_runtime_identity"]["sha256"] = runtime_sha256
                    paths[key].write_text(json.dumps(payload) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    EvidenceBindingError,
                    "vllm.version_output must be a non-empty string",
                ):
                    self._build(paths)


if __name__ == "__main__":
    unittest.main()
