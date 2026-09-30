from __future__ import annotations

import base64
import http.client
import json
from types import SimpleNamespace
import threading
import unittest

from openai import OpenAI
from openai.types.responses import ResponseTextDeltaEvent, ResponseTextDoneEvent

from policy.roboharn_evo.scripts.codex_app_server_client import CodexAppServerTimeoutError
from policy.roboharn_evo.scripts.openai_responses_compat import (
    ResponsesCompatError,
    build_response_object,
    prepare_responses_request,
)
from policy.roboharn_evo.scripts.serve_openai_planner import OpenAIPlannerHandler


class _FakeAppServer:
    def __init__(self, *, text: str = '{"action":"move"}') -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []
        self.is_running = True
        self.error: Exception | None = None

    def run_turn(self, input_items: object, **kwargs: object) -> object:
        self.calls.append({"input_items": input_items, **kwargs})
        if self.error is not None:
            raise self.error
        callback = kwargs.get("on_delta")
        if callable(callback):
            midpoint = max(1, len(self.text) // 2)
            callback(self.text[:midpoint])
            callback(self.text[midpoint:])
        return SimpleNamespace(
            text=self.text,
            usage=SimpleNamespace(
                input_tokens=11,
                cached_input_tokens=2,
                output_tokens=5,
                reasoning_output_tokens=1,
                total_tokens=16,
            ),
        )


class _ServerFixture:
    def __init__(self, *, concurrency: int = 4, queue_timeout: float = 0.05) -> None:
        self.backend = _FakeAppServer()

        class Handler(OpenAIPlannerHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

        Handler.backend = "codex-account"
        Handler.client = object()
        Handler.model = "gpt-5.5"
        Handler.reasoning_effort = "xhigh"
        Handler.timeout_sec = 10
        Handler.codex_workdir = "/tmp"
        Handler.codex_app_server = self.backend
        Handler.max_concurrent_requests = concurrency
        Handler.request_queue_timeout_sec = queue_timeout
        Handler.max_request_bytes = 4 * 1024 * 1024
        Handler.request_semaphore = threading.BoundedSemaphore(concurrency)
        Handler.request_body_semaphore = threading.BoundedSemaphore(concurrency)
        self.handler_class = Handler

        from http.server import ThreadingHTTPServer

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def post(self, payload: object) -> tuple[int, dict[str, str], bytes]:
        body = json.dumps(payload).encode("utf-8")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            "/v1/responses",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        raw = response.read()
        headers = {key.lower(): value for key, value in response.getheaders()}
        status = response.status
        connection.close()
        return status, headers, raw

    def get(self, path: str = "/v1/responses") -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", path)
        response = connection.getresponse()
        raw = response.read()
        headers = {key.lower(): value for key, value in response.getheaders()}
        status = response.status
        connection.close()
        return status, headers, raw


class ResponsesCompatMappingTest(unittest.TestCase):
    def test_instructions_and_roles_remain_above_user_input(self) -> None:
        request = prepare_responses_request(
            {
                "model": "gpt-5.5",
                "instructions": "TOP",
                "input": [
                    {"role": "system", "content": "SYS"},
                    {"role": "developer", "content": "DEV"},
                    {"role": "assistant", "content": "PRIOR"},
                    {"role": "user", "content": "FOLLOWUP"},
                ],
            },
            expected_model="gpt-5.5",
            default_effort="xhigh",
        )

        self.assertEqual(request.effort, "xhigh")
        self.assertIn("TOP", request.developer_instructions)
        self.assertIn("[SYSTEM]\nSYS", request.developer_instructions)
        self.assertIn("[DEVELOPER]\nDEV", request.developer_instructions)
        self.assertEqual(
            request.user_input,
            [
                {
                    "type": "text",
                    "text": "<prior_message role='assistant'>\nPRIOR\n</prior_message>",
                },
                {"type": "text", "text": "FOLLOWUP"},
            ],
        )

    def test_json_schema_is_mapped_to_app_server_output_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": ["move", "stop"]}},
            "required": ["action"],
            "additionalProperties": False,
        }
        format_value = {
            "type": "json_schema",
            "name": "robot_action",
            "strict": True,
            "schema": schema,
        }
        request = prepare_responses_request(
            {
                "model": "gpt-5.5",
                "input": "Choose one action.",
                "text": {"format": format_value},
            },
            expected_model="gpt-5.5",
            default_effort=None,
        )

        self.assertEqual(request.output_schema, schema)
        self.assertEqual(request.text_config["format"], format_value)

    def test_json_object_does_not_become_app_server_output_schema(self) -> None:
        format_value = {"type": "json_object"}
        request = prepare_responses_request(
            {
                "model": "gpt-5.5",
                "instructions": "Return one JSON object only.",
                "input": "Choose one action.",
                "text": {"format": format_value},
            },
            expected_model="gpt-5.5",
            default_effort=None,
        )

        self.assertIsNone(request.output_schema)
        self.assertEqual(request.text_config["format"], format_value)

    def test_embedded_image_becomes_native_app_server_image_input(self) -> None:
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
        url = f"data:image/png;base64,{png}"
        request = prepare_responses_request(
            {
                "model": "gpt-5.5",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe it."},
                            {"type": "input_image", "image_url": url, "detail": "high"},
                        ],
                    }
                ],
            },
            expected_model="gpt-5.5",
            default_effort=None,
        )

        self.assertEqual(
            request.user_input,
            [
                {"type": "text", "text": "Describe it."},
                {"type": "image", "url": url, "detail": "high"},
            ],
        )

    def test_remote_image_and_tools_are_explicitly_rejected(self) -> None:
        with self.assertRaises(ResponsesCompatError) as image_error:
            prepare_responses_request(
                {
                    "model": "gpt-5.5",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": "https://example.invalid/image.png",
                                }
                            ],
                        }
                    ],
                },
                expected_model="gpt-5.5",
                default_effort=None,
            )
        self.assertEqual(image_error.exception.code, "unsupported_image_url")

        with self.assertRaises(ResponsesCompatError) as tool_error:
            prepare_responses_request(
                {
                    "model": "gpt-5.5",
                    "input": "test",
                    "tools": [{"type": "function", "name": "unsafe"}],
                },
                expected_model="gpt-5.5",
                default_effort=None,
            )
        self.assertEqual(tool_error.exception.param, "tools")

    def test_response_envelope_contains_sdk_output_shape(self) -> None:
        request = prepare_responses_request(
            {"model": "gpt-5.5", "input": "test"},
            expected_model="gpt-5.5",
            default_effort="high",
        )
        response = build_response_object(
            request,
            text="done",
            usage={
                "inputTokens": 3,
                "cachedInputTokens": 1,
                "outputTokens": 2,
                "reasoningOutputTokens": 1,
                "totalTokens": 5,
            },
        )
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["output_text"], "done")
        self.assertEqual(response["output"][0]["content"][0]["text"], "done")
        self.assertEqual(response["usage"]["total_tokens"], 5)


class ResponsesCompatHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _ServerFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_nonstreaming_endpoint_and_openai_sdk(self) -> None:
        client = OpenAI(
            api_key="local-test-key",
            base_url=f"http://127.0.0.1:{self.fixture.port}/v1",
            max_retries=0,
            timeout=5,
        )
        response = client.responses.create(
            model="gpt-5.5",
            instructions="Return JSON.",
            input="Move.",
            reasoning={"effort": "high"},
            store=False,
        )

        self.assertEqual(response.output_text, '{"action":"move"}')
        self.assertEqual(response.output[0].role, "assistant")
        self.assertEqual(len(self.fixture.backend.calls), 1)
        call = self.fixture.backend.calls[0]
        self.assertEqual(call["developer_instructions"], "Return JSON.")
        self.assertEqual(call["effort"], "high")

    def test_json_object_is_not_forwarded_as_strict_output_schema(self) -> None:
        status, _, raw = self.fixture.post(
            {
                "model": "gpt-5.5",
                "instructions": "Return one JSON object only.",
                "input": "Move.",
                "text": {"format": {"type": "json_object"}},
            }
        )

        self.assertEqual(status, 200, raw.decode("utf-8"))
        self.assertEqual(len(self.fixture.backend.calls), 1)
        self.assertIsNone(self.fixture.backend.calls[0]["output_schema"])

    def test_streaming_sse_event_lifecycle(self) -> None:
        status, headers, raw = self.fixture.post(
            {"model": "gpt-5.5", "input": "Move.", "stream": True}
        )
        self.assertEqual(status, 200)
        self.assertTrue(headers["content-type"].startswith("text/event-stream"))
        self.assertNotIn("content-length", headers)

        blocks = [block for block in raw.decode("utf-8").strip().split("\n\n") if block]
        events: list[tuple[str, dict[str, object]]] = []
        for block in blocks:
            lines = block.splitlines()
            event_type = lines[0].removeprefix("event: ")
            payload = json.loads(lines[1].removeprefix("data: "))
            events.append((event_type, payload))
        event_types = [event_type for event_type, _ in events]
        self.assertEqual(
            event_types,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [payload["sequence_number"] for _, payload in events],
            list(range(len(events))),
        )
        deltas = "".join(
            str(payload["delta"])
            for event_type, payload in events
            if event_type == "response.output_text.delta"
        )
        completed = events[-1][1]["response"]
        self.assertEqual(deltas, '{"action":"move"}')
        self.assertEqual(completed["output_text"], deltas)
        for event_type, payload in events:
            if event_type == "response.output_text.delta":
                ResponseTextDeltaEvent.model_validate(payload)
            elif event_type == "response.output_text.done":
                ResponseTextDoneEvent.model_validate(payload)
        self.assertNotIn(b"[DONE]", raw)

    def test_streaming_endpoint_is_consumable_by_openai_sdk(self) -> None:
        client = OpenAI(
            api_key="local-test-key",
            base_url=f"http://127.0.0.1:{self.fixture.port}/v1",
            max_retries=0,
            timeout=5,
        )
        stream = client.responses.create(model="gpt-5.5", input="Move.", stream=True)
        with stream:
            events = list(stream)

        self.assertEqual(events[0].type, "response.created")
        self.assertEqual(events[-1].type, "response.completed")
        self.assertEqual(events[-1].response.output_text, '{"action":"move"}')

    def test_invalid_request_uses_openai_error_envelope(self) -> None:
        status, _, raw = self.fixture.post(
            {"model": "gpt-5.5", "input": "test", "tools": [{"type": "function"}]}
        )
        payload = json.loads(raw)
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["param"], "tools")
        self.assertEqual(len(self.fixture.backend.calls), 0)

    def test_store_true_is_rejected_before_starting_a_turn(self) -> None:
        status, _, raw = self.fixture.post(
            {"model": "gpt-5.5", "input": "test", "store": True}
        )

        payload = json.loads(raw)
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["param"], "store")
        self.assertEqual(payload["error"]["code"], "unsupported_parameter")
        self.assertEqual(len(self.fixture.backend.calls), 0)

    def test_unhashable_detail_and_reasoning_summary_are_invalid_requests(self) -> None:
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
        cases = (
            (
                {
                    "model": "gpt-5.5",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{png}",
                                    "detail": [],
                                }
                            ],
                        }
                    ],
                },
                "input[0].content[0].detail",
            ),
            (
                {
                    "model": "gpt-5.5",
                    "input": "test",
                    "reasoning": {"summary": {}},
                },
                "reasoning.summary",
            ),
        )

        for request, expected_param in cases:
            with self.subTest(param=expected_param):
                status, _, raw = self.fixture.post(request)
                payload = json.loads(raw)
                self.assertEqual(status, 400)
                self.assertEqual(payload["error"]["type"], "invalid_request_error")
                self.assertEqual(payload["error"]["param"], expected_param)
        self.assertEqual(len(self.fixture.backend.calls), 0)

    def test_get_responses_returns_json_method_not_allowed(self) -> None:
        status, headers, raw = self.fixture.get()

        self.assertEqual(status, 405)
        self.assertTrue(headers["content-type"].startswith("application/json"))
        payload = json.loads(raw)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["param"], None)
        self.assertEqual(payload["error"]["code"], "method_not_allowed")
        self.assertEqual(len(self.fixture.backend.calls), 0)

    def test_app_server_timeout_maps_to_gateway_timeout(self) -> None:
        self.fixture.backend.error = CodexAppServerTimeoutError(
            "Codex turn exceeded the test timeout"
        )

        status, headers, raw = self.fixture.post({"model": "gpt-5.5", "input": "test"})

        payload = json.loads(raw)
        self.assertEqual(status, 504)
        self.assertTrue(headers["content-type"].startswith("application/json"))
        self.assertEqual(payload["error"]["type"], "server_error")
        self.assertEqual(payload["error"]["param"], None)
        self.assertEqual(len(self.fixture.backend.calls), 1)

    def test_concurrency_limit_returns_retryable_openai_error(self) -> None:
        semaphore = self.fixture.handler_class.request_semaphore
        self.assertIsNotNone(semaphore)
        acquired = [semaphore.acquire(timeout=0.1) for _ in range(self.fixture.handler_class.max_concurrent_requests)]
        self.assertTrue(all(acquired))
        try:
            status, _, raw = self.fixture.post({"model": "gpt-5.5", "input": "test"})
        finally:
            for _ in acquired:
                semaphore.release()

        payload = json.loads(raw)
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["type"], "server_error")
        self.assertEqual(payload["error"]["code"], "server_overloaded")
        self.assertEqual(len(self.fixture.backend.calls), 0)

    def test_request_body_admission_is_bounded_separately(self) -> None:
        semaphore = self.fixture.handler_class.request_body_semaphore
        self.assertIsNotNone(semaphore)
        acquired = [
            semaphore.acquire(timeout=0.1)
            for _ in range(self.fixture.handler_class.max_concurrent_requests)
        ]
        self.assertTrue(all(acquired))
        try:
            status, _, raw = self.fixture.post({"model": "gpt-5.5", "input": "test"})
        finally:
            for _ in acquired:
                semaphore.release()

        payload = json.loads(raw)
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "server_overloaded")
        self.assertEqual(len(self.fixture.backend.calls), 0)


if __name__ == "__main__":
    unittest.main()
