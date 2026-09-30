from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import tempfile
import unittest
import urllib.request

from policy.roboharn_evo.scripts.preflight_openai_vlm_modalities import (
    _NO_PROXY_OPENER,
    PreflightError,
    execute,
    parse_args,
    parse_generated_json,
    run_preflight,
)


MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self._body


class RecordingTransport:
    def __init__(
        self,
        *,
        model: str = MODEL,
        bad_mode: str | None = None,
        pure_text_visual: bool = False,
        swap_multi: bool = False,
    ) -> None:
        self.model = model
        self.bad_mode = bad_mode
        self.pure_text_visual = pure_text_visual
        self.swap_multi = swap_multi
        self.calls: list[dict] = []

    def __call__(self, request, timeout: float) -> FakeResponse:
        body = None if request.data is None else json.loads(request.data.decode("utf-8"))
        self.calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "body": body,
                "timeout": timeout,
            }
        )
        if request.full_url.endswith("/models"):
            return FakeResponse({"object": "list", "data": [{"id": self.model}]})
        user_content = body["messages"][1]["content"]
        if isinstance(user_content, str):
            mode = "text_only"
        else:
            image_count = sum(part.get("type") == "image_url" for part in user_content)
            mode = "head_image" if image_count == 1 else "head_then_third_images"
        generated_mode = "wrong" if mode == self.bad_mode else mode
        if mode == "text_only":
            generated = {"modality": generated_mode, "arithmetic_result": 42}
        else:
            head_facts = {
                "image_index": 1,
                "blue_block_count": 4,
                "red_circle_horizontal_relation": "left_of_blue_blocks",
            }
            third_facts = {
                "image_index": 2,
                "blue_block_count": 4,
                "red_circle_horizontal_relation": "right_of_blue_blocks",
            }
            if self.pure_text_visual:
                head_facts = {
                    "image_index": 1,
                    "blue_block_count": 0,
                    "red_circle_horizontal_relation": "not_visible",
                }
                third_facts = {
                    "image_index": 2,
                    "blue_block_count": 0,
                    "red_circle_horizontal_relation": "not_visible",
                }
            observations = [head_facts]
            if mode == "head_then_third_images":
                observations.append(third_facts)
                if self.swap_multi:
                    observations = [
                        {**third_facts, "image_index": 1},
                        {**head_facts, "image_index": 2},
                    ]
            generated = {"modality": generated_mode, "observations": observations}
        content = json.dumps(generated)
        return FakeResponse(
            {
                "id": f"cmpl-{mode}",
                "object": "chat.completion",
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )


class ModalityPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="vlm_modality_preflight_")
        root = Path(self.temporary.name)
        self.head_bytes = b"\x89PNG\r\n\x1a\nhead-payload"
        self.third_bytes = b"\xff\xd8\xffthird-payload"
        self.head = root / "head.png"
        self.third = root / "third.jpg"
        self.head.write_bytes(self.head_bytes)
        self.third.write_bytes(self.third_bytes)
        self.output = root / "report.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def args(self, **overrides) -> argparse.Namespace:
        values = {
            "base_url": "http://127.0.0.1:8000/v1/",
            "model": MODEL,
            "head_image": self.head,
            "third_image": self.third,
            "expected_head_blue_block_count": 4,
            "expected_head_red_circle_relation": "left_of_blue_blocks",
            "expected_third_blue_block_count": 4,
            "expected_third_red_circle_relation": "right_of_blue_blocks",
            "output": self.output,
            "timeout_sec": 12.0,
            "max_tokens": 512,
            "response_format_json_object": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_models_and_all_three_request_payloads(self) -> None:
        transport = RecordingTransport()
        report = run_preflight(self.args(), opener=transport)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["fallback_count"], 0)
        self.assertEqual(len(transport.calls), 4)
        self.assertEqual(transport.calls[0]["method"], "GET")
        self.assertEqual(transport.calls[0]["url"], "http://127.0.0.1:8000/v1/models")
        self.assertTrue(report["model_check"]["exact_model_found"])

        payloads = [call["body"] for call in transport.calls[1:]]
        for payload in payloads:
            self.assertEqual(payload["model"], MODEL)
            self.assertEqual(payload["temperature"], 0.6)
            self.assertEqual(payload["top_p"], 0.95)
            self.assertEqual(payload["top_k"], 20)
            self.assertEqual(payload["min_p"], 0.0)
            self.assertEqual(payload["max_tokens"], 512)
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
            self.assertNotIn("response_format", payload)

        prompt_text = "\n".join(
            part
            for payload in payloads
            for message in payload["messages"]
            for part in (
                [message["content"]]
                if isinstance(message["content"], str)
                else [
                    item["text"]
                    for item in message["content"]
                    if item.get("type") == "text"
                ]
            )
        )
        self.assertNotIn("expected", prompt_text.lower())
        self.assertNotIn('"blue_block_count": 4', prompt_text)

        self.assertIsInstance(payloads[0]["messages"][1]["content"], str)
        one_image_content = payloads[1]["messages"][1]["content"]
        two_image_content = payloads[2]["messages"][1]["content"]
        self.assertEqual([part["type"] for part in one_image_content], ["text", "image_url"])
        self.assertEqual(
            [part["type"] for part in two_image_content],
            ["text", "image_url", "image_url"],
        )
        expected_head = "data:image/png;base64," + base64.b64encode(self.head_bytes).decode("ascii")
        expected_third = "data:image/jpeg;base64," + base64.b64encode(self.third_bytes).decode("ascii")
        self.assertEqual(one_image_content[1]["image_url"]["url"], expected_head)
        self.assertEqual(two_image_content[1]["image_url"]["url"], expected_head)
        self.assertEqual(two_image_content[2]["image_url"]["url"], expected_third)
        self.assertEqual(
            report["requests"][2]["image_order"],
            ["head", "third"],
        )
        self.assertEqual(report["requests"][2]["image_count"], 2)
        self.assertNotIn(expected_head, json.dumps(report))
        self.assertNotIn("messages", json.dumps(report))
        self.assertEqual(report["requests"][0]["usage"]["total_tokens"], 15)

    def test_cli_accepts_engine_and_model_id_aliases(self) -> None:
        args = parse_args(
            [
                "--engine-base-url",
                "http://127.0.0.1:9000/v1",
                "--model-id",
                MODEL,
                "--head-image",
                str(self.head),
                "--third-image",
                str(self.third),
                "--expected-head-blue-block-count",
                "4",
                "--expected-head-red-circle-relation",
                "left_of_blue_blocks",
                "--expected-third-blue-block-count",
                "4",
                "--expected-third-red-circle-relation",
                "right_of_blue_blocks",
                "--output",
                str(self.output),
            ]
        )

        self.assertEqual(args.base_url, "http://127.0.0.1:9000/v1")
        self.assertEqual(args.model, MODEL)
        self.assertEqual(args.max_tokens, 512)
        self.assertFalse(args.response_format_json_object)
        self.assertEqual(args.expected_head_blue_block_count, 4)

    def test_optional_json_response_format_is_sent_to_all_modes(self) -> None:
        transport = RecordingTransport()
        run_preflight(
            self.args(response_format_json_object=True, max_tokens=77),
            opener=transport,
        )

        for call in transport.calls[1:]:
            self.assertEqual(call["body"]["response_format"], {"type": "json_object"})
            self.assertEqual(call["body"]["max_tokens"], 77)

    def test_missing_exact_model_fails_closed_before_chat_requests(self) -> None:
        transport = RecordingTransport(model="not-the-requested-model")

        with self.assertRaisesRegex(PreflightError, "exact required model id"):
            run_preflight(self.args(), opener=transport)

        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0]["method"], "GET")

    def test_one_bad_modality_makes_atomic_report_fail(self) -> None:
        transport = RecordingTransport(bad_mode="head_image")

        exit_code = execute(self.args(), opener=transport)
        report = json.loads(self.output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(transport.calls), 4)
        self.assertEqual(
            [item["status"] for item in report["requests"]],
            ["passed", "failed", "passed"],
        )
        self.assertEqual(report["errors"][0]["request_mode"], "head_image")
        self.assertFalse(any(self.output.parent.glob(f".{self.output.name}.*.tmp")))

    def test_text_only_visual_guess_cannot_pass_image_probes(self) -> None:
        transport = RecordingTransport(pure_text_visual=True)

        report = run_preflight(self.args(), opener=transport)

        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            [item["status"] for item in report["requests"]],
            ["passed", "failed", "failed"],
        )
        self.assertTrue(all("expected fact" in item["error"] for item in report["errors"]))

    def test_swapped_head_and_third_facts_fail_ordered_multi_image_probe(self) -> None:
        transport = RecordingTransport(swap_multi=True)

        report = run_preflight(self.args(), opener=transport)

        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            [item["status"] for item in report["requests"]],
            ["passed", "passed", "failed"],
        )
        self.assertIn("image 1", report["errors"][0]["error"])

    def test_rejects_nonlocal_or_decorated_engine_urls(self) -> None:
        forbidden = (
            "http://example.com:8000/v1",
            "http://127.0.0.2:8000/v1",
            "http://user:pass@127.0.0.1:8000/v1",
            "http://127.0.0.1:8000/v1?",
            "http://127.0.0.1:8000/v1?remote=true",
            "http://127.0.0.1:8000/v1#",
            "http://127.0.0.1:8000/v1#fragment",
        )
        for base_url in forbidden:
            with self.subTest(base_url=base_url), self.assertRaises(PreflightError):
                run_preflight(self.args(base_url=base_url), opener=RecordingTransport())

    def test_accepts_all_three_explicit_loopback_hostnames(self) -> None:
        for base_url in (
            "http://localhost:8000/v1",
            "http://127.0.0.1:8000/v1",
            "http://[::1]:8000/v1",
        ):
            with self.subTest(base_url=base_url):
                report = run_preflight(
                    self.args(base_url=base_url),
                    opener=RecordingTransport(),
                )
                self.assertEqual(report["status"], "passed")

    def test_transport_opener_has_no_proxy_routes(self) -> None:
        proxy_handlers = [
            handler
            for handler in _NO_PROXY_OPENER.handlers
            if isinstance(handler, urllib.request.ProxyHandler)
        ]
        self.assertEqual(proxy_handlers, [])

    def test_setup_failure_writes_failure_report_and_returns_nonzero(self) -> None:
        missing = self.head.parent / "missing.png"
        transport = RecordingTransport()

        exit_code = execute(self.args(head_image=missing), opener=transport)
        report = json.loads(self.output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(transport.calls, [])
        self.assertEqual(report["errors"][0]["stage"], "setup_or_model_check")

    def test_generated_json_allows_fences_or_single_extracted_object(self) -> None:
        self.assertEqual(
            parse_generated_json('```json\n{"modality":"text_only","ok":true}\n```'),
            {"modality": "text_only", "ok": True},
        )
        self.assertEqual(
            parse_generated_json('result: {"modality":"text_only","ok":true} done'),
            {"modality": "text_only", "ok": True},
        )
        with self.assertRaisesRegex(PreflightError, "exactly one JSON object"):
            parse_generated_json('{"ok":true} {"ok":true}')


if __name__ == "__main__":
    unittest.main()
