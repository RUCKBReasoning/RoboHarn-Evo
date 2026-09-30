from __future__ import annotations

import base64
import io
import json
import tempfile
import time
import unittest
from email.message import Message
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from policy.roboharn_evo.scripts.openai_responses_compat import ResponsesCompatError
from policy.roboharn_evo.scripts.serve_openai_planner import (
    ModelOutputError,
    OpenAIPlannerHandler,
    _classify_business_exception,
    _codex_error_summary,
    _materialize_codex_messages,
    _resolve_runtime_options,
    _safe_url_for_reporting,
    _validate_codex_account_backend,
    _validate_runtime_configuration,
    _verify_local_upstream_model,
    parse_args,
)
from policy.roboharn_evo.scripts.serve_qwen_planner import (
    QwenPlannerHandler,
    build_planner_messages,
)


class OpenAIPlannerCodexAccountTest(unittest.TestCase):
    def test_openai_backend_remains_default(self) -> None:
        args = parse_args([])

        self.assertEqual(args.backend, "openai")
        self.assertIsNone(args.temperature)
        self.assertEqual(args.thinking_mode, "provider_default")
        self.assertEqual(args.planner_prompt_mode, "legacy_duplicate")
        self.assertEqual(args.planner_max_output_tokens, 0)

    def test_rendered_system_planner_prompt_keeps_one_dynamic_context_copy(self) -> None:
        marker = "UNIQUE_RUNTIME_CONTEXT_MARKER"
        trace = marker + ("|track=0003,world_m=[0.1,-0.2,0.7],verified=true" * 1200)
        task = json.dumps(
            {
                "instruction": "put the block back",
                "memory_harness": {"trace": trace},
            }
        )
        payload = {
            "task": task,
            "previous_memory_text": f"previous {marker}",
            "planner_state": [1.0, 2.0],
            "planner_start_image_b64": "AAA=",
            "planner_end_image_b64": "BBB=",
            "prompt": (
                f"Rendered task={task}\n"
                f"Rendered memory=previous {marker}\n"
                "Rendered state=[1.0, 2.0]"
            ),
        }

        messages, previous_memory = build_planner_messages(
            payload,
            planner_prompt_mode="rendered_system_once",
        )
        legacy_messages, _ = build_planner_messages(payload)

        self.assertEqual(previous_memory, f"previous {marker}")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], payload["prompt"])
        user_content = messages[1]["content"]
        self.assertNotIn(marker, user_content[0]["text"])
        self.assertEqual(
            len([item for item in user_content if item["type"] == "image_url"]),
            2,
        )
        # The marker occurs twice in the rendered prompt because the test adapter
        # put it in task and memory.  The gateway must not append either again.
        self.assertEqual(json.dumps(messages).count(marker), 2)
        rendered_bytes = len(json.dumps(messages).encode("utf-8"))
        legacy_bytes = len(json.dumps(legacy_messages).encode("utf-8"))
        self.assertGreater(legacy_bytes, 100_000)
        self.assertGreater(legacy_bytes - rendered_bytes, 50_000)

    def test_legacy_planner_prompt_mapping_remains_default(self) -> None:
        marker = "LEGACY_DYNAMIC_MARKER"
        task = json.dumps({"memory_harness": {"trace": marker}})
        payload = {
            "task": task,
            "previous_memory_text": marker,
            "planner_state": [0.0],
            "planner_start_image_b64": "AAA=",
            "planner_end_image_b64": "BBB=",
            "prompt": f"Rendered prompt {task} memory={marker}",
        }

        messages, _ = build_planner_messages(payload)

        self.assertGreater(json.dumps(messages).count(marker), 2)

    def test_rendered_system_mode_requires_adapter_rendered_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an adapter-rendered prompt"):
            build_planner_messages(
                {
                    "task": "test",
                    "planner_start_image_b64": "AAA=",
                    "planner_end_image_b64": "BBB=",
                },
                planner_prompt_mode="rendered_system_once",
            )

    def test_rendered_system_mode_rejects_missing_dynamic_fields(self) -> None:
        base_payload = {
            "task": "  move the blue block  ",
            "previous_memory_text": "  gripper is open  ",
            "planner_state": [1.0, 2.0],
            "planner_start_image_b64": "AAA=",
            "planner_end_image_b64": "BBB=",
        }
        cases = {
            "task": "gripper is open\n[1.0, 2.0]",
            "previous_memory_text": "move the blue block\n[1.0, 2.0]",
            "planner_state": "move the blue block\ngripper is open",
        }
        for field, prompt in cases.items():
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, rf"dynamic fields:.*{field}"
            ):
                build_planner_messages(
                    {**base_payload, "prompt": prompt},
                    planner_prompt_mode="rendered_system_once",
                )

    def test_rendered_system_mode_matches_adapter_stripping_and_allows_empty_memory(self) -> None:
        messages, previous_memory = build_planner_messages(
            {
                "task": "  move the blue block  ",
                "previous_memory_text": "   ",
                "planner_state": [1.0, 2.0],
                "planner_start_image_b64": "AAA=",
                "planner_end_image_b64": "BBB=",
                "prompt": "task=move the blue block\nstate=[1.0, 2.0]",
            },
            planner_prompt_mode="rendered_system_once",
        )

        self.assertEqual(previous_memory, "   ")
        self.assertIn("task=move the blue block", messages[0]["content"])

    def test_local_chat_identity_and_sampling_arguments_are_explicit(self) -> None:
        args = parse_args(
            [
                "--provider-name",
                "local-vllm",
                "--model-source",
                "modelscope",
                "--model-revision",
                "abc123",
                "--quantization",
                "fp8",
                "--serving-framework",
                "vllm-1.2.3",
                "--context-limit",
                "65536",
                "--temperature",
                "0.6",
                "--top-p",
                "0.95",
                "--top-k",
                "20",
                "--min-p",
                "0.0",
                "--thinking-mode",
                "enabled",
                "--upstream-models-response-sha256",
                "d" * 64,
                "--upstream-model-verified",
                "--agent-contract-path",
                "/tmp/contract.json",
                "--agent-contract-manifest-sha256",
                "e" * 64,
                "--serving-runtime-identity-path",
                "/tmp/local_serving_runtime_identity.json",
                "--serving-runtime-identity-sha256",
                "f" * 64,
            ]
        )

        self.assertEqual(args.provider_name, "local-vllm")
        self.assertEqual(args.model_source, "modelscope")
        self.assertEqual(args.model_revision, "abc123")
        self.assertEqual(args.quantization, "fp8")
        self.assertEqual(args.serving_framework, "vllm-1.2.3")
        self.assertEqual(args.context_limit, 65536)
        self.assertEqual(args.temperature, 0.6)
        self.assertEqual(args.top_p, 0.95)
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.min_p, 0.0)
        self.assertEqual(args.thinking_mode, "enabled")
        self.assertEqual(args.upstream_models_response_sha256, "d" * 64)
        self.assertTrue(args.upstream_model_verified)
        self.assertEqual(args.agent_contract_path, "/tmp/contract.json")
        self.assertEqual(args.agent_contract_manifest_sha256, "e" * 64)
        self.assertEqual(
            args.serving_runtime_identity_path,
            "/tmp/local_serving_runtime_identity.json",
        )
        self.assertEqual(args.serving_runtime_identity_sha256, "f" * 64)

    def test_serving_runtime_identity_requires_absolute_content_addressed_json(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_serving_runtime_identity_") as directory:
            identity_path = Path(directory) / "runtime.json"
            identity_path.write_text(
                '{"schema_version":1,"vllm":{"version":"test"}}\n',
                encoding="utf-8",
            )
            import hashlib

            digest = hashlib.sha256(identity_path.read_bytes()).hexdigest()
            common = [
                "--config-file",
                "",
                "--base-url",
                "http://127.0.0.1:8000/v1",
                "--provider-name",
                "local-vllm",
                "--model",
                "local-model",
                "--api-mode",
                "chat",
            ]
            args = parse_args(
                common
                + [
                    "--serving-runtime-identity-path",
                    str(identity_path),
                    "--serving-runtime-identity-sha256",
                    digest,
                ]
            )
            _validate_runtime_configuration(args, _resolve_runtime_options(args))

            missing_hash = parse_args(
                common + ["--serving-runtime-identity-path", str(identity_path)]
            )
            with self.assertRaisesRegex(ValueError, "must be supplied together"):
                _validate_runtime_configuration(
                    missing_hash, _resolve_runtime_options(missing_hash)
                )

            relative = parse_args(
                common
                + [
                    "--serving-runtime-identity-path",
                    "runtime.json",
                    "--serving-runtime-identity-sha256",
                    digest,
                ]
            )
            with self.assertRaisesRegex(ValueError, "must be absolute"):
                _validate_runtime_configuration(relative, _resolve_runtime_options(relative))

    def test_health_omits_optional_runtime_identity_unless_configured(self) -> None:
        handler = object.__new__(OpenAIPlannerHandler)
        handler.path = "/health"
        handler.backend = "openai"
        handler.local_only = False
        handler.serving_runtime_identity_path = ""
        handler.serving_runtime_identity_sha256 = ""
        handler._send_json = mock.Mock()

        handler.do_GET()

        payload = handler._send_json.call_args.args[0]
        self.assertNotIn("serving_runtime_identity", payload)
        self.assertEqual(payload["planner_prompt_mode"], "legacy_duplicate")
        self.assertEqual(payload["planner_max_output_tokens"], 0)

        handler.serving_runtime_identity_path = "/tmp/runtime.json"
        handler.serving_runtime_identity_sha256 = "f" * 64
        handler.planner_prompt_mode = "rendered_system_once"
        handler.planner_max_output_tokens = 8192
        handler.do_GET()
        payload = handler._send_json.call_args.args[0]
        self.assertEqual(
            payload["serving_runtime_identity"],
            {"path": "/tmp/runtime.json", "sha256": "f" * 64},
        )
        self.assertEqual(payload["planner_prompt_mode"], "rendered_system_once")
        self.assertEqual(payload["planner_max_output_tokens"], 8192)

    def test_validate_codex_account_checks_metadata_without_returning_tokens(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_codex_auth_test_") as directory:
            root = Path(directory)
            auth_file = root / "auth.json"
            auth_file.write_text(
                json.dumps(
                    {
                        "auth_mode": "chatgpt",
                        "tokens": {"access_token": "sensitive-test-value"},
                    }
                ),
                encoding="utf-8",
            )

            result = _validate_codex_account_backend(
                codex_bin="/bin/true",
                auth_file=str(auth_file),
                workdir=str(root),
            )

        self.assertEqual(result, "/bin/true")
        self.assertNotIn("sensitive-test-value", result)

    def test_materialize_codex_messages_extracts_embedded_images(self) -> None:
        image_bytes = b"\x89PNG\r\n\x1a\n"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": "Return JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect the image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ]

        with tempfile.TemporaryDirectory(prefix="tcm_codex_image_test_") as directory:
            prompt, image_paths = _materialize_codex_messages(
                messages,
                temp_dir=Path(directory),
            )

            self.assertEqual(len(image_paths), 1)
            self.assertEqual(image_paths[0].read_bytes(), image_bytes)

        self.assertIn("SYSTEM:\nReturn JSON.", prompt)
        self.assertIn("[attached image 1]", prompt)
        self.assertNotIn(encoded, prompt)

    def test_codex_error_summary_redacts_bearer_and_api_keys(self) -> None:
        summary = _codex_error_summary(
            "Error: Authorization: Bearer abc.def.ghi, sk-secretvalue1234, and "
            "eyJheader1234.eyJpayload1234.signature1234 were rejected"
        )

        self.assertNotIn("abc.def.ghi", summary)
        self.assertNotIn("sk-secretvalue1234", summary)
        self.assertNotIn("eyJpayload1234", summary)
        self.assertIn("[REDACTED", summary)

    def test_codex_account_completion_uses_native_app_server_mapping(self) -> None:
        handler = object.__new__(OpenAIPlannerHandler)
        handler.backend = "codex-account"
        handler.codex_workdir = "/tmp"
        handler.model = "gpt-5.5"
        handler.reasoning_effort = "xhigh"
        handler.disable_response_storage = True
        handler.timeout_sec = 600
        captured: dict[str, object] = {}

        def fake_request(request: object, **_: object) -> object:
            captured["request"] = request
            return SimpleNamespace(text='{"queries": []}', usage=None)

        with mock.patch.object(handler, "_run_prepared_codex_request", side_effect=fake_request):
            result = handler._run_codex_account_completion(
                messages=[
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": '{"task":"test"}'},
                ],
                normalizer=lambda payload: payload,
            )

        request = captured["request"]
        self.assertEqual(request.model, "gpt-5.5")
        self.assertEqual(request.effort, "xhigh")
        self.assertIsNone(request.output_schema)
        self.assertIn("Return JSON only.", request.developer_instructions)
        self.assertEqual(request.user_input, [{"type": "text", "text": '{"task":"test"}'}])
        self.assertEqual(result, {"queries": []})

    def test_local_chat_completion_preserves_system_multiview_and_sampling(self) -> None:
        handler = object.__new__(OpenAIPlannerHandler)
        handler.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=mock.Mock(
                        return_value=SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content='{"queries": []}'))]
                        )
                    )
                )
            )
        )
        handler.model = "Qwen/Qwen3.5-397B-A17B-FP8"
        handler.timeout_sec = 600
        handler.response_format_json = True
        handler.reasoning_effort = ""
        handler.max_output_tokens = 32768
        handler.planner_max_output_tokens = 0
        handler.temperature = 0.6
        handler.top_p = 0.95
        handler.top_k = 20
        handler.min_p = 0.0
        handler.thinking_mode = "enabled"
        handler.disable_response_storage = False
        messages = [
            {"role": "system", "content": "Return JSON only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare both views."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB="}},
                ],
            },
        ]

        result = handler._run_chat_completion(messages=messages, normalizer=lambda payload: payload)

        request = handler.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "Qwen/Qwen3.5-397B-A17B-FP8")
        self.assertEqual(request["messages"], messages)
        self.assertEqual(request["messages"][0]["role"], "system")
        image_items = [
            item
            for item in request["messages"][1]["content"]
            if item.get("type") == "image_url"
        ]
        self.assertEqual(len(image_items), 2)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["max_completion_tokens"], 32768)
        self.assertEqual(request["temperature"], 0.6)
        self.assertEqual(request["top_p"], 0.95)
        self.assertEqual(
            request["extra_body"],
            {
                "top_k": 20,
                "min_p": 0.0,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        self.assertNotIn("reasoning_effort", request)
        self.assertNotIn("store", request)
        self.assertEqual(result, {"queries": []})

    def test_plan_route_can_use_smaller_output_budget_than_other_routes(self) -> None:
        create = mock.Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"queries": []}')
                    )
                ],
                usage=None,
            )
        )
        handler = object.__new__(OpenAIPlannerHandler)
        handler.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        handler.model = "Qwen/Qwen3.5-397B-A17B-FP8"
        handler.timeout_sec = 600
        handler.response_format_json = True
        handler.reasoning_effort = ""
        handler.max_output_tokens = 32768
        handler.planner_max_output_tokens = 8192
        handler.temperature = None
        handler.top_p = None
        handler.top_k = None
        handler.min_p = None
        handler.thinking_mode = "enabled"
        handler.disable_response_storage = False

        handler._current_request_mode = "planner"
        handler._run_chat_completion(
            messages=[{"role": "user", "content": "plan"}],
            normalizer=lambda payload: payload,
        )
        self.assertEqual(create.call_args.kwargs["max_completion_tokens"], 8192)

        handler._current_request_mode = "ood_detection"
        handler._run_chat_completion(
            messages=[{"role": "user", "content": "ood"}],
            normalizer=lambda payload: payload,
        )
        self.assertEqual(create.call_args.kwargs["max_completion_tokens"], 32768)


class OpenAIPlannerHardeningTest(unittest.TestCase):
    @staticmethod
    def _bare_business_handler(path: str = "/plan") -> OpenAIPlannerHandler:
        handler = object.__new__(OpenAIPlannerHandler)
        handler.path = path
        handler.client = object()
        handler.model = "Qwen/Qwen3.5-397B-A17B-FP8"
        handler.provider_name = "local-vllm"
        handler.backend = "openai"
        handler.api_mode = "chat"
        handler.max_retries = 0
        handler.request_queue_timeout_sec = 0.01
        handler._prepare_business_request_replay = mock.Mock()
        handler._send_error_json = mock.Mock()
        handler._emit_completion_audit = mock.Mock()
        return handler

    def test_explicit_different_model_does_not_inherit_codex_reasoning(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_provider_config_test_") as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                'model = "gpt-5.5"\nmodel_reasoning_effort = "xhigh"\n',
                encoding="utf-8",
            )
            qwen_args = parse_args(
                [
                    "--config-file",
                    str(config_path),
                    "--model",
                    "Qwen/Qwen3.5-397B-A17B-FP8",
                    "--api-mode",
                    "chat",
                ]
            )
            gpt_args = parse_args(
                ["--config-file", str(config_path), "--model", "gpt-5.5"]
            )

            qwen_options = _resolve_runtime_options(qwen_args)
            gpt_options = _resolve_runtime_options(gpt_args)

        self.assertEqual(qwen_options["reasoning_effort"], "")
        self.assertEqual(gpt_options["reasoning_effort"], "xhigh")

    def test_provider_specific_controls_are_rejected_outside_chat(self) -> None:
        cases = (
            ("--top-k", "20"),
            ("--min-p", "0.1"),
            ("--thinking-mode", "enabled"),
        )
        for option, value in cases:
            with self.subTest(option=option):
                args = parse_args(
                    [
                        "--config-file",
                        "",
                        "--api-mode",
                        "responses",
                        option,
                        value,
                    ]
                )
                options = _resolve_runtime_options(args)
                with self.assertRaisesRegex(ValueError, "require --api-mode chat"):
                    _validate_runtime_configuration(args, options)

    def test_codex_account_rejects_controls_it_cannot_apply(self) -> None:
        args = parse_args(
            [
                "--config-file",
                "",
                "--backend",
                "codex-account",
                "--temperature",
                "0",
            ]
        )
        options = _resolve_runtime_options(args)

        with self.assertRaisesRegex(ValueError, "not supported by --backend codex-account"):
            _validate_runtime_configuration(args, options)

    def test_local_only_requires_loopback_and_content_addressed_manifests(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_local_only_test_") as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            model = "Qwen/Qwen3.5-397B-A17B-FP8"
            model_manifest = root / "model_identity.json"
            model_manifest.write_text(
                json.dumps(
                    {
                        "model_directory": str(model_dir.resolve()),
                        "source": {"repo_id": model},
                    }
                ),
                encoding="utf-8",
            )
            contract = root / "agent_contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "shared_model_role_contract": {
                            "required_endpoints": [
                                "/plan",
                                "/ood",
                                "/recover",
                                "/perception_queries",
                                "/normalize_perception_queries",
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            import hashlib

            common = [
                "--local-only",
                "--backend",
                "openai",
                "--config-file",
                "",
                "--base-url",
                "http://127.0.0.1:8000/v1",
                "--provider-name",
                "local-vllm",
                "--model",
                model,
                "--api-mode",
                "chat",
                "--max-retries",
                "0",
                "--disable-response-storage",
                "--model-artifact-path",
                str(model_dir),
                "--model-artifact-manifest-path",
                str(model_manifest),
                "--model-artifact-manifest-sha256",
                hashlib.sha256(model_manifest.read_bytes()).hexdigest(),
                "--agent-contract-path",
                str(contract),
                "--agent-contract-manifest-sha256",
                hashlib.sha256(contract.read_bytes()).hexdigest(),
            ]
            args = parse_args(common)
            options = _resolve_runtime_options(args)
            _validate_runtime_configuration(args, options)
            self.assertEqual(options["base_url"], "http://127.0.0.1:8000/v1")

            remote = list(common)
            remote[remote.index("http://127.0.0.1:8000/v1")] = "https://example.com/v1"
            remote_args = parse_args(remote)
            with self.assertRaisesRegex(ValueError, "loopback"):
                _validate_runtime_configuration(
                    remote_args, _resolve_runtime_options(remote_args)
                )

    def test_local_upstream_verification_is_exact_and_hashes_response(self) -> None:
        body = json.dumps(
            {"object": "list", "data": [{"id": "local-model"}]},
            separators=(",", ":"),
        ).encode("utf-8")

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return None

            def getcode(self):
                return self.status

            def read(self):
                return body

        opener = mock.Mock()
        opener.open.return_value = Response()
        digest = _verify_local_upstream_model(
            base_url="http://127.0.0.1:8000/v1",
            model="local-model",
            timeout_sec=5,
            opener=opener,
        )
        import hashlib

        self.assertEqual(digest, hashlib.sha256(body).hexdigest())
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/models")

    def test_health_url_reporting_removes_credentials_and_query(self) -> None:
        self.assertEqual(
            _safe_url_for_reporting("https://user:secret@example.com:8443/v1?token=x"),
            "https://example.com:8443/v1",
        )

    def test_responses_completion_forwards_supported_sampling_and_usage(self) -> None:
        create = mock.Mock(
            return_value=SimpleNamespace(
                output_text='{"queries": []}',
                usage=SimpleNamespace(input_tokens=11, output_tokens=4, total_tokens=15),
            )
        )
        handler = object.__new__(OpenAIPlannerHandler)
        handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
        handler.model = "provider-model"
        handler.timeout_sec = 30
        handler.reasoning_effort = ""
        handler.max_output_tokens = 256
        handler.temperature = 0.2
        handler.top_p = 0.9
        handler.disable_response_storage = False

        result = handler._run_responses_completion(
            messages=[
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "test"},
            ],
            normalizer=lambda payload: payload,
        )

        request = create.call_args.kwargs
        self.assertEqual(request["temperature"], 0.2)
        self.assertEqual(request["top_p"], 0.9)
        self.assertEqual(request["max_output_tokens"], 256)
        self.assertEqual(handler._last_completion_usage["total_tokens"], 15)
        self.assertEqual(result, {"queries": []})

    def test_business_request_preparser_rejects_malformed_json(self) -> None:
        handler = object.__new__(OpenAIPlannerHandler)
        headers = Message()
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = "1"
        handler.headers = headers
        handler.rfile = io.BytesIO(b"{")
        handler.max_request_bytes = 1024
        handler.connection = SimpleNamespace(
            gettimeout=lambda: None,
            settimeout=lambda _: None,
        )

        with self.assertRaises(ResponsesCompatError) as error:
            handler._prepare_business_request_replay()

        self.assertEqual(error.exception.code, "invalid_json")

    def test_business_exception_status_mapping_is_fail_closed(self) -> None:
        class ServiceUnavailableError(RuntimeError):
            status_code = 503

        cases = (
            (
                ResponsesCompatError("bad JSON", code="invalid_json"),
                HTTPStatus.BAD_REQUEST,
                "malformed_json",
            ),
            (TimeoutError("slow"), HTTPStatus.GATEWAY_TIMEOUT, "upstream_timeout"),
            (
                ServiceUnavailableError("service unavailable"),
                HTTPStatus.SERVICE_UNAVAILABLE,
                "upstream_unavailable",
            ),
            (
                RuntimeError("CUDA out of memory"),
                HTTPStatus.SERVICE_UNAVAILABLE,
                "upstream_oom",
            ),
            (
                ModelOutputError("invalid"),
                HTTPStatus.BAD_GATEWAY,
                "invalid_upstream_output",
            ),
            (RuntimeError("other"), HTTPStatus.BAD_GATEWAY, "upstream_error"),
        )
        for error, expected_status, expected_category in cases:
            with self.subTest(error=type(error).__name__):
                status, category, _ = _classify_business_exception(error)
                self.assertEqual(status, expected_status)
                self.assertEqual(category, expected_category)

    def test_upstream_request_shape_rejection_is_safe_and_distinct(self) -> None:
        class UpstreamRequestError(RuntimeError):
            def __init__(self, status_code: int) -> None:
                super().__init__(
                    "request body included Bearer secret-token and private prompt text"
                )
                self.status_code = status_code

        for upstream_status in (400, 413, 422):
            with self.subTest(upstream_status=upstream_status):
                status, category, message = _classify_business_exception(
                    UpstreamRequestError(upstream_status)
                )
                self.assertEqual(status, HTTPStatus.BAD_GATEWAY)
                self.assertEqual(category, "upstream_rejected_request")
                self.assertEqual(
                    message,
                    f"The upstream model rejected the request with HTTP {upstream_status}.",
                )
                self.assertNotIn("secret-token", message)
                self.assertNotIn("private prompt", message)

        status, category, _ = _classify_business_exception(
            UpstreamRequestError(429)
        )
        self.assertEqual(status, HTTPStatus.BAD_GATEWAY)
        self.assertEqual(category, "upstream_error")

    def test_business_timeout_response_and_audit_declare_no_fallback(self) -> None:
        handler = self._bare_business_handler("/recover")
        handler._prepare_business_request_replay.side_effect = TimeoutError("slow")

        with mock.patch.object(OpenAIPlannerHandler, "request_semaphore", None):
            handler.do_POST()

        status, payload = handler._send_error_json.call_args.args
        self.assertEqual(status, HTTPStatus.GATEWAY_TIMEOUT)
        self.assertEqual(payload["error_category"], "upstream_timeout")
        self.assertIs(payload["fallback_used"], False)
        audit = handler._emit_completion_audit.call_args.kwargs
        self.assertEqual(audit["endpoint"], "/recover")
        self.assertEqual(audit["status"], HTTPStatus.GATEWAY_TIMEOUT)
        self.assertEqual(audit["error_category"], "upstream_timeout")

    def test_all_five_business_endpoints_emit_success_audits(self) -> None:
        endpoints = (
            "/plan",
            "/ood",
            "/recover",
            "/perception_queries",
            "/normalize_perception_queries",
        )
        with mock.patch.object(OpenAIPlannerHandler, "request_semaphore", None), mock.patch.object(
            QwenPlannerHandler, "do_POST", return_value=None
        ):
            for endpoint in endpoints:
                with self.subTest(endpoint=endpoint):
                    handler = self._bare_business_handler(endpoint)
                    handler.do_POST()
                    audit = handler._emit_completion_audit.call_args.kwargs
                    self.assertEqual(audit["endpoint"], endpoint)
                    self.assertIs(audit["success"], True)
                    self.assertEqual(audit["status"], HTTPStatus.OK)
                    self.assertIsNone(audit["error_category"])

    def test_structured_audit_contains_only_identity_outcome_and_usage(self) -> None:
        handler = object.__new__(OpenAIPlannerHandler)
        handler.provider_name = "local-vllm"
        handler.model = "Qwen/Qwen3.5-397B-A17B-FP8"
        handler.backend = "openai"
        handler.api_mode = "chat"
        handler.max_retries = 0
        handler._last_completion_usage = {"input_tokens": 10, "output_tokens": 3}

        with mock.patch("builtins.print") as print_mock:
            handler._emit_completion_audit(
                endpoint="/plan",
                started_at=time.monotonic(),
                success=True,
                status=HTTPStatus.OK,
                error_category=None,
            )

        line = print_mock.call_args.args[0]
        payload = json.loads(line.removeprefix("[openai-planner-audit] "))
        self.assertEqual(payload["endpoint"], "/plan")
        self.assertEqual(payload["provider"], "local-vllm")
        self.assertEqual(payload["usage"]["output_tokens"], 3)
        self.assertIs(payload["fallback_used"], False)
        self.assertEqual(payload["retry_count"], 0)
        self.assertNotIn("messages", payload)
        self.assertNotIn("prompt", payload)
        self.assertNotIn("images", payload)

    def test_chat_upstream_failure_never_calls_another_transport(self) -> None:
        handler = object.__new__(OpenAIPlannerHandler)
        handler.client = object()
        handler.backend = "openai"
        handler.api_mode = "chat"
        handler._run_chat_completion = mock.Mock(side_effect=RuntimeError("upstream failed"))
        handler._run_responses_completion = mock.Mock()
        handler._run_codex_account_completion = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "upstream failed"):
            handler._run_completion(messages=[], normalizer=lambda payload: payload)

        handler._run_chat_completion.assert_called_once()
        handler._run_responses_completion.assert_not_called()
        handler._run_codex_account_completion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
