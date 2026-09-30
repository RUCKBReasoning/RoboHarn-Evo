from __future__ import annotations

import io
import itertools
import threading
import unittest
from unittest import mock

from policy.roboharn_evo.scripts import codex_app_server_client as app_server_module
from policy.roboharn_evo.scripts.codex_app_server_client import (
    CodexAppServerClient,
    CodexAppServerClosedError,
    CodexAppServerProtocolError,
    CodexAppServerRPCError,
    CodexAppServerTimeoutError,
)


class CodexAppServerClientTest(unittest.TestCase):
    @staticmethod
    def _mark_fake_transport_running(client: CodexAppServerClient) -> mock.Mock:
        process = mock.Mock()
        process.poll.return_value = None
        client._process = process  # noqa: SLF001 - transport-generation fixture
        client._transport_failed_event = threading.Event()  # noqa: SLF001
        client._initialized = True  # noqa: SLF001
        return process

    def test_four_interleaved_ephemeral_turns_are_correlated(self) -> None:
        client = CodexAppServerClient()
        self._mark_fake_transport_running(client)
        thread_counter = itertools.count(1)
        barrier = threading.Barrier(4)
        allocation_lock = threading.Lock()

        def fake_rpc(method: str, params: dict[str, object], **_: object) -> object:
            if method == "thread/start":
                with allocation_lock:
                    index = next(thread_counter)
                return {"thread": {"id": f"thread-{index}"}}
            if method != "turn/start":
                return {}
            thread_id = str(params["threadId"])
            turn_id = thread_id.replace("thread", "turn")
            barrier.wait(timeout=2)
            client._handle_notification(  # noqa: SLF001 - protocol-order regression fixture
                "item/agentMessage/delta",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "itemId": f"item-{thread_id}",
                    "delta": f"delta-{thread_id}",
                },
            )
            client._handle_notification(  # noqa: SLF001
                "item/completed",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "completedAtMs": 1,
                    "item": {
                        "id": f"item-{thread_id}",
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": f"final-{thread_id}",
                    },
                },
            )
            client._handle_notification(  # noqa: SLF001
                "thread/tokenUsage/updated",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "tokenUsage": {
                        "last": {
                            "inputTokens": 1,
                            "cachedInputTokens": 0,
                            "outputTokens": 1,
                            "reasoningOutputTokens": 0,
                            "totalTokens": 2,
                        },
                        "total": {
                            "inputTokens": 10,
                            "cachedInputTokens": 2,
                            "outputTokens": 8,
                            "reasoningOutputTokens": 3,
                            "totalTokens": 18,
                        },
                    },
                },
            )
            client._handle_notification(  # noqa: SLF001
                "turn/completed",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "turn": {
                        "id": turn_id,
                        "items": [],
                        "status": "completed",
                        "durationMs": 5,
                    },
                },
            )
            return {"turn": {"id": turn_id}}

        results: list[object] = []
        failures: list[BaseException] = []

        def worker() -> None:
            try:
                results.append(
                    client.run_turn(
                        "test",
                        model="gpt-5.5",
                        developer_instructions="trusted",
                        cwd="/tmp",
                        effort="xhigh",
                        timeout=3,
                    )
                )
            except BaseException as exc:  # pragma: no cover - assertion reports captured failure
                failures.append(exc)

        with mock.patch.object(client, "start", return_value=client), mock.patch.object(
            client, "_rpc_call", side_effect=fake_rpc
        ), mock.patch.object(client, "_delete_thread"):
            workers = [threading.Thread(target=worker) for _ in range(4)]
            for worker_thread in workers:
                worker_thread.start()
            for worker_thread in workers:
                worker_thread.join(timeout=5)

        self.assertEqual(failures, [])
        self.assertEqual(len(results), 4)
        self.assertEqual(
            {result.text for result in results},
            {f"final-thread-{index}" for index in range(1, 5)},
        )
        self.assertTrue(all(result.usage.total_tokens == 18 for result in results))
        self.assertTrue(all(result.usage.reasoning_output_tokens == 3 for result in results))

    def test_completed_turn_without_agent_message_is_protocol_error(self) -> None:
        with self.assertRaises(CodexAppServerProtocolError):
            CodexAppServerClient._select_final_text({})  # noqa: SLF001

        with self.assertRaises(CodexAppServerProtocolError):
            CodexAppServerClient._select_final_text(  # noqa: SLF001
                {"commentary": ("commentary", "must-not-leak")}
            )

    def test_callback_cancel_discards_queued_deltas(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        seen: list[str] = []

        def callback(delta: str) -> None:
            seen.append(delta)
            if delta == "first":
                entered.set()
                release.wait(timeout=2)

        state = app_server_module._TurnState("thread", mock.Mock(), callback)  # noqa: SLF001
        state.set_item_phase(
            {"id": "final", "type": "agentMessage", "phase": "final_answer"}
        )
        state.add_delta("final", "first")
        self.assertTrue(entered.wait(timeout=1))
        state.add_delta("final", "second-after-cancel")
        state.cancel_callbacks(join_timeout=0)
        release.set()
        state.cancel_callbacks(join_timeout=1)

        self.assertEqual(seen, ["first"])

    def test_missing_completed_phase_cannot_downgrade_commentary(self) -> None:
        state = app_server_module._TurnState("thread", mock.Mock(), None)  # noqa: SLF001
        state.set_item_phase(
            {"id": "commentary", "type": "agentMessage", "phase": "commentary"}
        )
        state.add_delta("commentary", "must-not-leak")
        state.add_message(
            {"id": "commentary", "type": "agentMessage", "text": "must-not-leak"}
        )

        self.assertEqual(state.messages["commentary"][0], "commentary")
        with self.assertRaises(CodexAppServerProtocolError):
            CodexAppServerClient._select_final_text(state.messages)  # noqa: SLF001

    def test_rpc_rejects_a_stale_process_generation_before_write(self) -> None:
        client = CodexAppServerClient()
        self._mark_fake_transport_running(client)
        stale_process = mock.Mock()

        with mock.patch.object(client, "_send_message") as send_message:
            with self.assertRaises(CodexAppServerClosedError):
                client._rpc_call(  # noqa: SLF001 - generation isolation regression
                    "turn/start",
                    {"threadId": "thread-old", "input": []},
                    timeout=0.1,
                    expected_process=stale_process,
                )

        send_message.assert_not_called()
        self.assertEqual(client._pending, {})  # noqa: SLF001

    def test_primary_rpc_timeout_marks_generation_unhealthy(self) -> None:
        client = CodexAppServerClient()
        process = self._mark_fake_transport_running(client)
        write_queue = app_server_module.queue.Queue()
        client._write_queue = write_queue  # noqa: SLF001
        writer = threading.Thread(
            target=client._writer_loop,  # noqa: SLF001
            args=(
                process,
                client._transport_failed_event,  # noqa: SLF001
                io.StringIO(),
                write_queue,
            ),
            daemon=True,
        )
        client._writer_thread = writer  # noqa: SLF001
        writer.start()

        with mock.patch.object(client, "_signal_process"):
            with self.assertRaises(CodexAppServerTimeoutError):
                client._rpc_call(  # noqa: SLF001
                    "thread/start",
                    {"ephemeral": True},
                    timeout=0.1,
                    expected_process=process,
                )

        self.assertFalse(client.is_running)
        writer.join(timeout=1)
        self.assertFalse(writer.is_alive())

    def test_synchronous_turn_rejection_does_not_defer_cleanup(self) -> None:
        client = CodexAppServerClient()
        self._mark_fake_transport_running(client)

        def fake_rpc(method: str, params: dict[str, object], **_: object) -> object:
            if method == "thread/start":
                return {"thread": {"id": "thread-rejected"}}
            if method == "turn/start":
                raise CodexAppServerRPCError(method, "badRequest")
            return {}

        with mock.patch.object(client, "start", return_value=client), mock.patch.object(
            client, "_rpc_call", side_effect=fake_rpc
        ), mock.patch.object(client, "_defer_turn_cleanup") as defer_cleanup, mock.patch.object(
            client, "_delete_thread"
        ) as delete_thread:
            with self.assertRaises(CodexAppServerRPCError):
                client.run_turn(
                    "test",
                    model="gpt-5.5",
                    developer_instructions="trusted",
                    cwd="/tmp",
                    timeout=3,
                )

        defer_cleanup.assert_not_called()
        delete_thread.assert_called_once()

    def test_stream_exposes_only_final_answer_deltas(self) -> None:
        client = CodexAppServerClient()
        self._mark_fake_transport_running(client)

        def fake_rpc(method: str, params: dict[str, object], **_: object) -> object:
            if method == "thread/start":
                return {"thread": {"id": "thread-stream"}}
            if method != "turn/start":
                return {}

            thread_id = str(params["threadId"])
            turn_id = "turn-stream"
            commentary_item_id = "item-commentary"
            final_item_id = "item-final"

            # Deltas can arrive before item/completed reveals their phase.
            client._handle_notification(  # noqa: SLF001 - protocol-order regression fixture
                "item/agentMessage/delta",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "itemId": commentary_item_id,
                    "delta": "private-commentary",
                },
            )
            client._handle_notification(  # noqa: SLF001
                "item/completed",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "item": {
                        "id": commentary_item_id,
                        "type": "agentMessage",
                        "phase": "commentary",
                        "text": "private-commentary",
                    },
                },
            )
            client._handle_notification(  # noqa: SLF001
                "item/agentMessage/delta",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "itemId": final_item_id,
                    "delta": "public-",
                },
            )
            client._handle_notification(  # noqa: SLF001
                "item/completed",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "item": {
                        "id": final_item_id,
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": "public-answer",
                    },
                },
            )
            client._handle_notification(  # noqa: SLF001
                "item/agentMessage/delta",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "itemId": final_item_id,
                    "delta": "answer",
                },
            )
            client._handle_notification(  # noqa: SLF001
                "turn/completed",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "turn": {
                        "id": turn_id,
                        "items": [],
                        "status": "completed",
                    },
                },
            )
            return {"turn": {"id": turn_id}}

        streamed: list[str] = []
        with mock.patch.object(client, "start", return_value=client), mock.patch.object(
            client, "_rpc_call", side_effect=fake_rpc
        ), mock.patch.object(client, "_delete_thread"):
            result = client.run_turn(
                "test",
                model="gpt-5.5",
                developer_instructions="trusted",
                cwd="/tmp",
                timeout=3,
                on_delta=streamed.append,
            )

        self.assertEqual(result.text, "public-answer")
        self.assertEqual(result.deltas, ("public-", "answer"))
        self.assertEqual(streamed, ["public-", "answer"])
        self.assertNotIn("private-commentary", "".join(result.deltas))
        self.assertNotIn("private-commentary", "".join(streamed))

    def test_reasoning_summary_is_forwarded_to_turn_start(self) -> None:
        client = CodexAppServerClient()
        self._mark_fake_transport_running(client)
        thread_start_params: dict[str, object] = {}
        turn_start_params: dict[str, object] = {}

        def fake_rpc(method: str, params: dict[str, object], **_: object) -> object:
            if method == "thread/start":
                thread_start_params.update(params)
                return {"thread": {"id": "thread-summary"}}
            if method != "turn/start":
                return {}

            turn_start_params.update(params)
            thread_id = str(params["threadId"])
            turn_id = "turn-summary"
            client._handle_notification(  # noqa: SLF001 - protocol-order regression fixture
                "item/completed",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "item": {
                        "id": "item-final",
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": "done",
                    },
                },
            )
            client._handle_notification(  # noqa: SLF001
                "turn/completed",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "turn": {
                        "id": turn_id,
                        "items": [],
                        "status": "completed",
                    },
                },
            )
            return {"turn": {"id": turn_id}}

        with mock.patch.object(client, "start", return_value=client), mock.patch.object(
            client, "_rpc_call", side_effect=fake_rpc
        ), mock.patch.object(client, "_delete_thread"):
            result = client.run_turn(
                "test",
                model="gpt-5.5",
                developer_instructions="trusted",
                cwd="/tmp",
                effort="xhigh",
                summary="detailed",
                timeout=3,
            )

        self.assertEqual(result.text, "done")
        self.assertEqual(turn_start_params["summary"], "detailed")
        self.assertEqual(turn_start_params["effort"], "xhigh")
        self.assertNotIn("summary", thread_start_params)


if __name__ == "__main__":
    unittest.main()
