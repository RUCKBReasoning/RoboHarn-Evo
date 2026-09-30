from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import re
import signal
import shutil
import subprocess
import threading
import time
from typing import Any, TextIO


DeltaCallback = Callable[[str], None]


class CodexAppServerError(RuntimeError):
    """Base error for the persistent Codex app-server client."""


class CodexAppServerClosedError(CodexAppServerError):
    """Raised when the app-server transport is unavailable or already closed."""


class CodexAppServerTimeoutError(CodexAppServerError):
    """Raised when an RPC or model turn exceeds its deadline."""


class CodexAppServerProtocolError(CodexAppServerError):
    """Raised when app-server returns an invalid protocol payload."""


class CodexStreamConsumerClosedError(CodexAppServerError):
    """Raised when a downstream streaming consumer disconnects or stalls."""


class CodexAppServerRPCError(CodexAppServerError):
    """A JSON-RPC error with deliberately non-sensitive diagnostics."""

    def __init__(self, method: str, code: int | str | None) -> None:
        self.method = method
        self.code = code
        code_text = str(code) if isinstance(code, (int, str)) else "unknown"
        super().__init__(f"Codex app-server request {method!r} failed (code={code_text})")


class CodexTurnFailedError(CodexAppServerError):
    """Raised when app-server reports a failed model turn."""

    def __init__(self, thread_id: str, turn_id: str, error_code: str) -> None:
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.error_code = error_code
        super().__init__(f"Codex turn failed (code={error_code})")


class CodexTurnInterruptedError(CodexAppServerError):
    """Raised when a model turn finishes with interrupted status."""

    def __init__(self, thread_id: str, turn_id: str) -> None:
        self.thread_id = thread_id
        self.turn_id = turn_id
        super().__init__("Codex turn was interrupted")


@dataclass(frozen=True, slots=True)
class CodexTokenUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    total_tokens: int
    model_context_window: int | None = None


@dataclass(frozen=True, slots=True)
class CodexTurnResult:
    thread_id: str
    turn_id: str
    status: str
    text: str
    deltas: tuple[str, ...]
    usage: CodexTokenUsage | None
    duration_ms: int | None = None


@dataclass(slots=True)
class _PendingCall:
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error_code: int | str | None = None
    transport_closed: bool = False


@dataclass(slots=True)
class _WriteRequest:
    encoded: str
    event: threading.Event = field(default_factory=threading.Event)
    failed: bool = False


_CALLBACK_END = object()
_WRITER_END = object()


class _TurnState:
    def __init__(
        self,
        thread_id: str,
        process: subprocess.Popen[str],
        callback: DeltaCallback | None,
    ) -> None:
        self.thread_id = thread_id
        self.process = process
        self.turn_id: str | None = None
        self.done = threading.Event()
        self.usage_received = threading.Event()
        self.callback_failed = threading.Event()
        self.lock = threading.Lock()
        self.status = "inProgress"
        self.duration_ms: int | None = None
        self.error_code = "turn_failed"
        self.transport_closed = False
        self.protocol_error = False
        self.deltas: list[tuple[str, str]] = []
        self.delta_by_item: dict[str, list[str]] = {}
        self.item_phases: dict[str, str | None] = {}
        self.emitted_delta_count: dict[str, int] = {}
        self.messages: dict[str, tuple[str | None, str]] = {}
        self.usage_payload: dict[str, Any] | None = None
        self._callback = callback
        self._callback_queue: queue.Queue[object] | None = None
        self._callback_thread: threading.Thread | None = None
        self._callback_finished = False
        if callback is not None:
            self._callback_queue = queue.Queue(maxsize=256)
            self._callback_thread = threading.Thread(
                target=self._callback_loop,
                name=f"codex-delta-{thread_id[:8]}",
                daemon=True,
            )
            self._callback_thread.start()

    def _callback_loop(self) -> None:
        callback_queue = self._callback_queue
        callback = self._callback
        if callback_queue is None or callback is None:
            return
        while True:
            item = callback_queue.get()
            if item is _CALLBACK_END:
                return
            if self.callback_failed.is_set():
                return
            try:
                callback(str(item))
            except Exception:
                self.callback_failed.set()
                return

    def _enqueue_delta(self, delta: str) -> None:
        callback_queue = self._callback_queue
        if (
            callback_queue is None
            or self.callback_failed.is_set()
            or self._callback_finished
        ):
            return
        try:
            callback_queue.put_nowait(delta)
        except queue.Full:
            self.callback_failed.set()

    def add_delta(self, item_id: str, delta: str) -> None:
        should_emit = False
        with self.lock:
            self.deltas.append((item_id, delta))
            self.delta_by_item.setdefault(item_id, []).append(delta)
            should_emit = self.item_phases.get(item_id) == "final_answer"
            if should_emit:
                self.emitted_delta_count[item_id] = len(self.delta_by_item[item_id])
        if should_emit:
            self._enqueue_delta(delta)

    def set_item_phase(self, item: Mapping[str, Any]) -> str | None:
        if item.get("type") != "agentMessage":
            return None
        item_id = item.get("id")
        phase = item.get("phase")
        if not isinstance(item_id, str):
            return None
        safe_phase = phase if isinstance(phase, str) else None
        pending: list[str] = []
        with self.lock:
            existing_phase = self.item_phases.get(item_id)
            if existing_phase is not None and safe_phase is None:
                effective_phase = existing_phase
            elif (
                existing_phase is not None
                and safe_phase is not None
                and existing_phase != safe_phase
            ):
                self.protocol_error = True
                effective_phase = existing_phase
            else:
                effective_phase = safe_phase
            self.item_phases[item_id] = effective_phase
            if effective_phase == "final_answer":
                deltas = self.delta_by_item.get(item_id, [])
                emitted = self.emitted_delta_count.get(item_id, 0)
                pending = list(deltas[emitted:])
                self.emitted_delta_count[item_id] = len(deltas)
        for delta in pending:
            self._enqueue_delta(delta)
        return effective_phase

    def add_message(self, item: Mapping[str, Any]) -> None:
        if item.get("type") != "agentMessage":
            return
        item_id = item.get("id")
        text = item.get("text")
        if not isinstance(item_id, str) or not isinstance(text, str):
            return
        effective_phase = self.set_item_phase(item)
        with self.lock:
            self.messages[item_id] = (effective_phase, text)

    def flush_item_deltas(self, item_id: str) -> tuple[str, ...]:
        with self.lock:
            if self.item_phases.get(item_id) not in {None, "final_answer"}:
                raise CodexAppServerProtocolError(
                    "Codex attempted to expose a non-final agent message"
                )
            deltas = list(self.delta_by_item.get(item_id, []))
            emitted = self.emitted_delta_count.get(item_id, 0)
            pending = list(deltas[emitted:])
            self.emitted_delta_count[item_id] = len(deltas)
        for delta in pending:
            self._enqueue_delta(delta)
        return tuple(deltas)

    def finish_callbacks(self, *, join_timeout: float = 5.0) -> bool:
        callback_queue: queue.Queue[object] | None
        should_signal = False
        with self.lock:
            if self._callback_finished:
                callback_thread = self._callback_thread
                callback_queue = self._callback_queue
            else:
                self._callback_finished = True
                should_signal = True
                callback_thread = self._callback_thread
                callback_queue = self._callback_queue
        if (
            should_signal
            and callback_queue is not None
            and not self.callback_failed.is_set()
        ):
            try:
                callback_queue.put(
                    _CALLBACK_END,
                    timeout=max(0.0, min(join_timeout, 1.0)),
                )
            except queue.Full:
                self.callback_failed.set()
        if self.callback_failed.is_set():
            self.cancel_callbacks(join_timeout=join_timeout)
            return False
        if (
            callback_thread is not None
            and callback_thread is not threading.current_thread()
            and join_timeout > 0
        ):
            callback_thread.join(timeout=join_timeout)
        if callback_thread is not None and callback_thread.is_alive():
            self.callback_failed.set()
            return False
        return not self.callback_failed.is_set()

    def cancel_callbacks(self, *, join_timeout: float = 0.5) -> bool:
        """Stop callback delivery and discard every delta not already in-flight."""

        self.callback_failed.set()
        with self.lock:
            self._callback_finished = True
            callback_thread = self._callback_thread
            callback_queue = self._callback_queue
        if callback_queue is not None:
            try:
                while True:
                    callback_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                callback_queue.put_nowait(_CALLBACK_END)
            except queue.Full:  # pragma: no cover - queue was just drained
                pass
        if (
            callback_thread is not None
            and callback_thread is not threading.current_thread()
            and join_timeout > 0
        ):
            callback_thread.join(timeout=join_timeout)
        return callback_thread is None or not callback_thread.is_alive()


_AUTH_ENV_KEYS = (
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
    "CHAT_API_KEY",
    "CHAT_API_BASE_URL",
)

_DISABLED_THREAD_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "multi_agent",
    "browser_use",
    "computer_use",
    "in_app_browser",
    "image_generation",
    "hooks",
    "plugins",
    "remote_plugin",
    "tool_suggest",
    "goals",
)

_SAFE_CODEX_ERROR_CODES = {
    "activeTurnNotSteerable",
    "badRequest",
    "contextWindowExceeded",
    "cyberPolicy",
    "httpConnectionFailed",
    "internalServerError",
    "other",
    "responseStreamConnectionFailed",
    "responseStreamDisconnected",
    "responseTooManyFailedAttempts",
    "sandboxError",
    "serverOverloaded",
    "sessionBudgetExceeded",
    "threadRollbackFailed",
    "unauthorized",
    "usageLimitExceeded",
}

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
)


def _sanitize_diagnostic(text: str) -> str:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    for pattern in _SECRET_PATTERNS:
        clean = pattern.sub("[REDACTED]", clean)
    clean = "".join(char for char in clean if char == "\t" or ord(char) >= 32)
    return clean[:1000]


def _safe_turn_error_code(error: Any) -> str:
    if not isinstance(error, Mapping):
        return "turn_failed"
    info = error.get("codexErrorInfo")
    if isinstance(info, str) and info in _SAFE_CODEX_ERROR_CODES:
        return info
    if isinstance(info, Mapping):
        for key in info:
            if isinstance(key, str) and key in _SAFE_CODEX_ERROR_CODES:
                return key
    return "turn_failed"


def _positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


class CodexAppServerClient:
    """Thread-safe persistent client for ``codex app-server`` over stdio JSONL.

    ``run_turn`` creates a fresh ephemeral thread for every request. The app-server
    process and authenticated ChatGPT/Codex transport are reused until ``close``.
    """

    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        codex_home: str | os.PathLike[str] | None = None,
        process_cwd: str | os.PathLike[str] = "/tmp",
        startup_timeout: float = 30.0,
        rpc_timeout: float = 30.0,
        interrupt_timeout: float = 5.0,
        client_name: str = "rmbench_tcm",
        client_title: str = "RMBench RoboHarn-Evo planner",
        client_version: str = "1.0.0",
    ) -> None:
        self._codex_bin = codex_bin
        self._codex_home = str(Path(codex_home).expanduser().resolve()) if codex_home else None
        self._process_cwd = str(Path(process_cwd).expanduser().resolve())
        self._startup_timeout = max(0.1, float(startup_timeout))
        self._rpc_timeout = max(0.1, float(rpc_timeout))
        self._interrupt_timeout = max(0.1, float(interrupt_timeout))
        self._client_info = {
            "name": str(client_name),
            "title": str(client_title),
            "version": str(client_version),
        }

        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._write_queue: queue.Queue[object] | None = None
        self._transport_failed_event: threading.Event | None = None
        self._initialized = False
        self._closing = False
        self._closed = False

        self._lifecycle_lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._turns_lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, _PendingCall] = {}
        self._turns_by_id: dict[str, _TurnState] = {}
        self._starting_by_thread: dict[str, _TurnState] = {}
        self._stderr_tail: deque[str] = deque(maxlen=50)

    @property
    def is_running(self) -> bool:
        process = self._process
        transport_failed = self._transport_failed_event
        return bool(
            self._initialized
            and not self._closing
            and process is not None
            and process.poll() is None
            and transport_failed is not None
            and not transport_failed.is_set()
        )

    def start(self) -> CodexAppServerClient:
        with self._lifecycle_lock:
            if self._closed:
                raise CodexAppServerClosedError("Codex app-server client is closed")
            if self.is_running:
                return self
            if self._process is not None:
                # A previous app-server may have exited (or lost its protocol
                # transport) between requests. Reap that generation here so the
                # next request can transparently initialize a fresh one.
                old_process = self._process
                stopped = self._stop_process_generation(old_process, timeout=2.0)
                self._initialized = False
                self._fail_all_waiters()
                self._clear_turn_routes()
                if not stopped:
                    raise CodexAppServerClosedError(
                        "The previous Codex app-server generation could not be reaped"
                    )
                self._process = None
                self._stdout_thread = None
                self._stderr_thread = None
                self._writer_thread = None
                self._write_queue = None
                self._transport_failed_event = None

            resolved_bin = shutil.which(self._codex_bin)
            if resolved_bin is None:
                candidate = Path(self._codex_bin).expanduser()
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    resolved_bin = str(candidate.resolve())
            if resolved_bin is None:
                raise CodexAppServerError("Codex app-server executable was not found")
            if not Path(self._process_cwd).is_dir():
                raise CodexAppServerError("Codex app-server process directory is unavailable")
            if self._codex_home is not None and not Path(self._codex_home).is_dir():
                raise CodexAppServerError("CODEX_HOME is unavailable")

            child_env = os.environ.copy()
            for variable in _AUTH_ENV_KEYS:
                child_env.pop(variable, None)
            if self._codex_home is not None:
                child_env["CODEX_HOME"] = self._codex_home

            command = [
                resolved_bin,
                "app-server",
                "--listen",
                "stdio://",
            ]
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    cwd=self._process_cwd,
                    env=child_env,
                    start_new_session=True,
                )
            except OSError as exc:
                raise CodexAppServerError("Failed to start Codex app-server") from exc

            if process.stdin is None or process.stdout is None or process.stderr is None:
                process.kill()
                process.wait()
                raise CodexAppServerError("Codex app-server stdio transport is unavailable")

            self._process = process
            transport_failed = threading.Event()
            self._transport_failed_event = transport_failed
            # The HTTP adapter admits at most four model requests, so eight
            # entries leave room for cleanup/control RPCs without permitting an
            # unbounded number of image-sized JSON strings to accumulate.
            write_queue: queue.Queue[object] = queue.Queue(maxsize=8)
            self._write_queue = write_queue
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                args=(process, transport_failed, process.stdin, write_queue),
                name="codex-app-server-writer",
                daemon=True,
            )
            self._stdout_thread = threading.Thread(
                target=self._stdout_loop,
                args=(process, transport_failed, process.stdout),
                name="codex-app-server-stdout",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._stderr_loop,
                args=(process.stderr,),
                name="codex-app-server-stderr",
                daemon=True,
            )
            self._writer_thread.start()
            self._stdout_thread.start()
            self._stderr_thread.start()

            try:
                self._rpc_call(
                    "initialize",
                    {"clientInfo": self._client_info},
                    timeout=self._startup_timeout,
                    require_initialized=False,
                    expected_process=process,
                )
                self._send_message({"method": "initialized", "params": {}})
                self._initialized = True
                if (
                    transport_failed.is_set()
                    or process.poll() is not None
                    or self._stdout_thread is None
                    or not self._stdout_thread.is_alive()
                ):
                    self._initialized = False
                    raise CodexAppServerClosedError(
                        "Codex app-server stopped during initialization"
                    )
            except Exception:
                self._abort_process()
                raise
            return self

    def run_turn(
        self,
        input_items: str | Sequence[Mapping[str, Any]],
        *,
        model: str,
        developer_instructions: str,
        cwd: str | os.PathLike[str],
        effort: str | None = None,
        summary: str | None = None,
        output_schema: Mapping[str, Any] | None = None,
        timeout: float = 600.0,
        on_delta: DeltaCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> CodexTurnResult:
        """Run one isolated turn and optionally stream assistant text deltas."""

        with self._lifecycle_lock:
            self.start()
            process = self._process
            if process is None:
                raise CodexAppServerClosedError("Codex app-server is not running")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(developer_instructions, str):
            raise TypeError("developer_instructions must be a string")
        request_cwd = Path(cwd).expanduser().resolve()
        if not request_cwd.is_dir():
            raise ValueError("cwd must be an existing directory")
        if effort is not None and (not isinstance(effort, str) or not effort.strip()):
            raise ValueError("effort must be a non-empty string when provided")
        if summary is not None and (
            not isinstance(summary, str)
            or summary not in {"auto", "concise", "detailed", "none"}
        ):
            raise ValueError("summary must be auto, concise, detailed, or none")
        normalized_input = self._normalize_input(input_items)
        if output_schema is not None:
            self._validate_json_value(output_schema, "output_schema")

        total_timeout = max(0.1, float(timeout))
        deadline = time.monotonic() + total_timeout
        thread_result = self._rpc_call(
            "thread/start",
            {
                "ephemeral": True,
                "model": model.strip(),
                "modelProvider": "openai",
                "developerInstructions": developer_instructions,
                "cwd": str(request_cwd),
                "sandbox": "read-only",
                "approvalPolicy": "never",
                "config": {
                    "web_search": "disabled",
                    "features": {feature: False for feature in _DISABLED_THREAD_FEATURES},
                },
            },
            timeout=self._request_timeout(deadline),
            expected_process=process,
        )
        thread_id = self._extract_nested_id(thread_result, "thread")
        with self._lifecycle_lock:
            if self._process is not process or not self.is_running:
                raise CodexAppServerClosedError(
                    "Codex app-server restarted after creating the request thread"
                )
            state = _TurnState(thread_id, process, on_delta)
            with self._turns_lock:
                self._starting_by_thread[thread_id] = state
            if not self.is_running:
                self._remove_turn_state(state)
                state.cancel_callbacks(join_timeout=0.5)
                raise CodexAppServerClosedError(
                    "Codex app-server stopped before starting the model turn"
                )

        turn_id = ""
        defer_cleanup = False
        try:
            turn_params: dict[str, Any] = {
                "threadId": thread_id,
                "input": normalized_input,
            }
            if effort is not None:
                turn_params["effort"] = effort.strip()
            if summary is not None:
                turn_params["summary"] = summary
            if output_schema is not None:
                turn_params["outputSchema"] = dict(output_schema)

            turn_rpc_timeout = self._request_timeout(deadline)
            defer_cleanup = True
            try:
                turn_result = self._rpc_call(
                    "turn/start",
                    turn_params,
                    timeout=turn_rpc_timeout,
                    expected_process=process,
                )
            except CodexAppServerRPCError:
                # A synchronous RPC rejection is definitive: no model turn was
                # created, so retaining this route or resetting the shared
                # app-server later would only harm unrelated requests.
                if state.turn_id is None:
                    defer_cleanup = False
                raise
            turn_id = self._extract_nested_id(turn_result, "turn")
            self._bind_turn_state(state, turn_id)

            while not state.done.is_set():
                if (
                    state.callback_failed.is_set()
                    or (cancel_event is not None and cancel_event.is_set())
                ):
                    raise CodexStreamConsumerClosedError(
                        "The downstream streaming consumer disconnected or stalled"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CodexAppServerTimeoutError(
                        f"Codex turn exceeded the {total_timeout:g}-second timeout"
                    )
                state.done.wait(timeout=min(0.1, remaining))

            # Usage normally precedes turn/completed on the ordered JSONL stream.
            # This short grace also covers versions that emit it immediately after.
            if not state.usage_received.is_set():
                state.usage_received.wait(timeout=min(0.25, max(0.0, deadline - time.monotonic())))

            with state.lock:
                status = state.status
                transport_closed = state.transport_closed
                protocol_error = state.protocol_error
                error_code = state.error_code
                duration_ms = state.duration_ms
                messages = dict(state.messages)
                usage_payload = dict(state.usage_payload) if state.usage_payload is not None else None

            if protocol_error:
                raise CodexAppServerProtocolError("Codex app-server emitted an invalid turn event")
            if transport_closed:
                raise CodexAppServerClosedError("Codex app-server stopped during a turn")
            if status == "interrupted":
                raise CodexTurnInterruptedError(thread_id, turn_id)
            if status != "completed":
                raise CodexTurnFailedError(thread_id, turn_id, error_code)

            item_id, text = self._select_final_message(messages)
            deltas = state.flush_item_deltas(item_id)
            if not state.finish_callbacks(join_timeout=min(5.0, max(0.1, deadline - time.monotonic()))):
                raise CodexStreamConsumerClosedError(
                    "The downstream streaming consumer disconnected or stalled"
                )
            return CodexTurnResult(
                thread_id=thread_id,
                turn_id=turn_id,
                status=status,
                text=text,
                deltas=deltas,
                usage=self._parse_usage(usage_payload),
                duration_ms=duration_ms,
            )
        except CodexAppServerTimeoutError:
            state.cancel_callbacks(join_timeout=0.5)
            if state.turn_id is not None and not state.done.is_set():
                self._interrupt_turn(state, wait_for_completion=True)
            raise
        except Exception:
            state.cancel_callbacks(join_timeout=0.5)
            if state.turn_id is not None and not state.done.is_set():
                self._interrupt_turn(state, wait_for_completion=True)
            raise
        finally:
            if not state.done.is_set() and defer_cleanup:
                self._defer_turn_cleanup(state)
            else:
                self._remove_turn_state(state)
                if state.done.is_set():
                    state.finish_callbacks()
                else:
                    state.cancel_callbacks()
                if self._process is state.process and self.is_running:
                    self._delete_thread(thread_id, expected_process=state.process)

    def close(self, *, timeout: float = 5.0) -> None:
        """Interrupt active turns, close stdin, then terminate only if needed."""

        close_timeout = max(0.0, float(timeout))
        deadline = time.monotonic() + close_timeout
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closing = True
            with self._turns_lock:
                active_states = list(
                    {
                        id(state): state
                        for state in [
                            *self._turns_by_id.values(),
                            *self._starting_by_thread.values(),
                        ]
                    }.values()
                )
            for state in active_states:
                state.cancel_callbacks(join_timeout=0)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._interrupt_turn(
                    state,
                    allow_closing=True,
                    wait_for_completion=False,
                    rpc_timeout=min(self._interrupt_timeout, remaining),
                )

            process = self._process
            if process is not None:
                writer_stopped = self._writer_thread is None
                write_queue = self._write_queue
                if write_queue is not None:
                    try:
                        write_queue.put_nowait(_WRITER_END)
                    except queue.Full:
                        pass
                writer = self._writer_thread
                if writer is not None and writer is not threading.current_thread():
                    writer.join(timeout=max(0.0, min(0.5, deadline - time.monotonic())))
                    writer_stopped = not writer.is_alive()
                stdin = process.stdin
                if writer_stopped and stdin is not None and not stdin.closed:
                    try:
                        stdin.close()
                    except OSError:
                        pass
                elif not writer_stopped:
                    self._signal_process(process, signal.SIGTERM)
                wait_timeout = max(0.0, deadline - time.monotonic())
                try:
                    process.wait(timeout=wait_timeout)
                except subprocess.TimeoutExpired:
                    self._terminate_process(
                        process,
                        timeout=min(2.0, max(0.1, wait_timeout)),
                    )
                self._stop_process_generation(process, timeout=0.5)

            self._initialized = False
            self._closed = True
            self._fail_all_waiters()
            self._join_reader_threads(timeout=0.5)

    def __enter__(self) -> CodexAppServerClient:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _normalize_input(
        input_items: str | Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(input_items, str):
            return [{"type": "text", "text": input_items}]
        if isinstance(input_items, (bytes, bytearray)) or not isinstance(input_items, Sequence):
            raise TypeError("input_items must be text or a sequence of input objects")
        normalized: list[dict[str, Any]] = []
        for item in input_items:
            if not isinstance(item, Mapping):
                raise TypeError("each input item must be an object")
            normalized.append(dict(item))
        if not normalized:
            raise ValueError("input_items must not be empty")
        CodexAppServerClient._validate_json_value(normalized, "input_items")
        return normalized

    @staticmethod
    def _validate_json_value(value: Any, field_name: str) -> None:
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be valid JSON") from exc

    @staticmethod
    def _extract_nested_id(result: Any, key: str) -> str:
        if not isinstance(result, Mapping):
            raise CodexAppServerProtocolError("Codex app-server returned an invalid RPC result")
        nested = result.get(key)
        identifier = nested.get("id") if isinstance(nested, Mapping) else None
        if not isinstance(identifier, str) or not identifier:
            raise CodexAppServerProtocolError("Codex app-server response is missing an identifier")
        return identifier

    def _request_timeout(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CodexAppServerTimeoutError("Codex turn deadline elapsed before the request started")
        return min(self._rpc_timeout, remaining)

    def _rpc_call(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
        require_initialized: bool = True,
        allow_closing: bool = False,
        expected_process: subprocess.Popen[str] | None = None,
    ) -> Any:
        deadline = time.monotonic() + max(0.1, float(timeout))
        with self._lifecycle_lock:
            if self._closed or (self._closing and not allow_closing):
                raise CodexAppServerClosedError("Codex app-server client is closed")
            if require_initialized and not self._initialized:
                raise CodexAppServerClosedError("Codex app-server is not initialized")
            process = self._process
            if process is None or process.poll() is not None:
                raise CodexAppServerClosedError("Codex app-server is not running")
            if expected_process is not None and process is not expected_process:
                raise CodexAppServerClosedError(
                    "Codex app-server generation changed during the request"
                )

            with self._pending_lock:
                request_id = self._next_id
                self._next_id += 1
                pending = _PendingCall()
                self._pending[request_id] = pending
            try:
                encoded = self._encode_message(
                    {"method": method, "id": request_id, "params": dict(params)}
                )
                write_request = self._queue_write(encoded, process)
            except Exception:
                with self._pending_lock:
                    self._pending.pop(request_id, None)
                raise

        try:
            self._await_write(write_request, process, deadline, method=method)
        except Exception:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise

        remaining = deadline - time.monotonic()
        if remaining <= 0 or not pending.event.wait(remaining):
            with self._pending_lock:
                self._pending.pop(request_id, None)
            if method in {"initialize", "thread/start", "turn/start"}:
                self._fail_transport_generation(process, terminate=True)
            raise CodexAppServerTimeoutError(f"Codex app-server request {method!r} timed out")
        if pending.transport_closed:
            raise CodexAppServerClosedError("Codex app-server stopped while awaiting a response")
        if pending.error_code is not None:
            raise CodexAppServerRPCError(method, pending.error_code)
        return pending.result

    @staticmethod
    def _encode_message(message: Mapping[str, Any]) -> str:
        try:
            return json.dumps(
                message,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise CodexAppServerProtocolError("Codex app-server request is not valid JSON") from exc

    def _queue_write(
        self,
        encoded: str,
        process: subprocess.Popen[str],
    ) -> _WriteRequest:
        write_queue = self._write_queue
        if self._process is not process or write_queue is None:
            raise CodexAppServerClosedError("Codex app-server writer is unavailable")
        request = _WriteRequest(encoded=encoded)
        try:
            write_queue.put_nowait(request)
        except queue.Full as exc:
            raise CodexAppServerTimeoutError(
                "Codex app-server writer queue is full"
            ) from exc
        return request

    def _await_write(
        self,
        request: _WriteRequest,
        process: subprocess.Popen[str],
        deadline: float,
        *,
        method: str,
    ) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not request.event.wait(remaining):
            self._fail_transport_generation(process, terminate=True)
            raise CodexAppServerTimeoutError(
                f"Codex app-server write for request {method!r} timed out"
            )
        if request.failed:
            raise CodexAppServerClosedError("Codex app-server stdin closed unexpectedly")

    def _send_message(
        self,
        message: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            raise CodexAppServerClosedError("Codex app-server stdin is unavailable")
        deadline = time.monotonic() + (
            self._rpc_timeout if timeout is None else max(0.1, float(timeout))
        )
        request = self._queue_write(self._encode_message(message), process)
        self._await_write(request, process, deadline, method=str(message.get("method", "notification")))

    def _mark_transport_failed(self, process: subprocess.Popen[str]) -> None:
        if self._process is not process:
            return
        self._initialized = False
        transport_failed = self._transport_failed_event
        if transport_failed is not None:
            transport_failed.set()

    def _fail_transport_generation(
        self,
        process: subprocess.Popen[str],
        *,
        terminate: bool,
    ) -> None:
        # Serialize against lazy restart so a late timeout from an old process
        # can never fail waiters belonging to the new generation.
        with self._lifecycle_lock:
            if self._process is not process:
                return
            self._mark_transport_failed(process)
            self._fail_all_waiters()
            if terminate and process.poll() is None:
                self._signal_process(process, signal.SIGTERM)

    def _writer_loop(
        self,
        process: subprocess.Popen[str],
        transport_failed: threading.Event,
        stream: TextIO,
        write_queue: queue.Queue[object],
    ) -> None:
        write_failed = False
        try:
            while not transport_failed.is_set():
                try:
                    item = write_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is _WRITER_END:
                    return
                if not isinstance(item, _WriteRequest):
                    continue
                try:
                    stream.write(item.encoded + "\n")
                    stream.flush()
                except (BrokenPipeError, OSError, ValueError):
                    item.failed = True
                    write_failed = True
                    transport_failed.set()
                finally:
                    item.event.set()
                if write_failed:
                    return
        finally:
            while True:
                try:
                    pending_write = write_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(pending_write, _WriteRequest):
                    pending_write.failed = True
                    pending_write.event.set()
            if write_failed and self._process is process:
                self._mark_transport_failed(process)
                self._fail_all_waiters()
                if process.poll() is None and not self._closing:
                    self._signal_process(process, signal.SIGTERM)

    def _stdout_loop(
        self,
        process: subprocess.Popen[str],
        transport_failed: threading.Event,
        stream: TextIO,
    ) -> None:
        protocol_error = False
        try:
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    protocol_error = True
                    return
                if not isinstance(message, Mapping):
                    protocol_error = True
                    return
                if self._process is not process:
                    return
                self._dispatch_message(message)
        finally:
            transport_failed.set()
            # Ignore a late EOF from an already-reaped generation. Otherwise
            # mark the transport unhealthy immediately and fail its in-flight
            # work; start() will create a new generation on the next request.
            if self._process is process:
                self._initialized = False
                self._fail_all_waiters(protocol_error=protocol_error)
                if process.poll() is None and not self._closing:
                    self._signal_process(process, signal.SIGTERM)

    def _stderr_loop(self, stream: TextIO) -> None:
        for raw_line in stream:
            clean = _sanitize_diagnostic(raw_line.strip())
            if clean:
                self._stderr_tail.append(clean)

    def _dispatch_message(self, message: Mapping[str, Any]) -> None:
        method = message.get("method")
        if isinstance(method, str):
            if "id" in message:
                self._reject_server_request(message.get("id"))
            else:
                params = message.get("params")
                self._handle_notification(method, params if isinstance(params, Mapping) else {})
            return
        request_id = message.get("id")
        if not isinstance(request_id, int):
            return
        with self._pending_lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        error = message.get("error")
        if isinstance(error, Mapping):
            code = error.get("code")
            pending.error_code = code if isinstance(code, (int, str)) else "rpc_error"
        elif "result" in message:
            pending.result = message.get("result")
        else:
            pending.error_code = "invalid_response"
        pending.event.set()

    def _reject_server_request(self, request_id: Any) -> None:
        if not isinstance(request_id, (int, str)):
            return
        try:
            self._send_message(
                {
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": "This client does not support server-initiated requests",
                    },
                }
            )
        except CodexAppServerError:
            return

    def _handle_notification(self, method: str, params: Mapping[str, Any]) -> None:
        state = self._state_for_notification(params)
        if state is None:
            return
        if method == "item/started":
            item = params.get("item")
            if isinstance(item, Mapping):
                state.set_item_phase(item)
            return
        if method == "item/agentMessage/delta":
            item_id = params.get("itemId")
            delta = params.get("delta")
            if isinstance(item_id, str) and isinstance(delta, str):
                state.add_delta(item_id, delta)
            else:
                with state.lock:
                    state.protocol_error = True
            return
        if method == "item/completed":
            item = params.get("item")
            if isinstance(item, Mapping):
                state.add_message(item)
            return
        if method == "thread/tokenUsage/updated":
            token_usage = params.get("tokenUsage")
            if isinstance(token_usage, Mapping):
                with state.lock:
                    state.usage_payload = dict(token_usage)
                state.usage_received.set()
            return
        if method == "error":
            if not bool(params.get("willRetry", False)):
                with state.lock:
                    state.error_code = _safe_turn_error_code(params.get("error"))
            return
        if method == "turn/completed":
            turn = params.get("turn")
            if not isinstance(turn, Mapping):
                with state.lock:
                    state.protocol_error = True
                state.done.set()
                return
            items = turn.get("items")
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
                for item in items:
                    if isinstance(item, Mapping):
                        state.add_message(item)
            status = turn.get("status")
            duration_ms = turn.get("durationMs")
            error = turn.get("error")
            with state.lock:
                state.status = status if isinstance(status, str) else "failed"
                state.duration_ms = _positive_int(duration_ms) if duration_ms is not None else None
                if error is not None:
                    state.error_code = _safe_turn_error_code(error)
            state.done.set()

    def _state_for_notification(self, params: Mapping[str, Any]) -> _TurnState | None:
        thread_id = params.get("threadId")
        turn_id = params.get("turnId")
        if not isinstance(turn_id, str):
            turn = params.get("turn")
            if isinstance(turn, Mapping) and isinstance(turn.get("id"), str):
                turn_id = turn.get("id")
        if not isinstance(thread_id, str):
            return None
        with self._turns_lock:
            state = self._turns_by_id.get(turn_id) if isinstance(turn_id, str) else None
            if state is None:
                state = self._starting_by_thread.get(thread_id)
            if state is not None and isinstance(turn_id, str) and turn_id:
                if state.turn_id is None:
                    state.turn_id = turn_id
                    self._turns_by_id[turn_id] = state
                    self._starting_by_thread.pop(thread_id, None)
                elif state.turn_id != turn_id:
                    with state.lock:
                        state.protocol_error = True
            return state

    def _bind_turn_state(self, state: _TurnState, turn_id: str) -> None:
        with self._turns_lock:
            if state.turn_id is not None and state.turn_id != turn_id:
                with state.lock:
                    state.protocol_error = True
                raise CodexAppServerProtocolError("Codex app-server returned inconsistent turn identifiers")
            state.turn_id = turn_id
            self._turns_by_id[turn_id] = state
            self._starting_by_thread.pop(state.thread_id, None)

    def _remove_turn_state(self, state: _TurnState) -> None:
        with self._turns_lock:
            self._starting_by_thread.pop(state.thread_id, None)
            if state.turn_id is not None and self._turns_by_id.get(state.turn_id) is state:
                self._turns_by_id.pop(state.turn_id, None)

    def _defer_turn_cleanup(self, state: _TurnState) -> None:
        """Keep routing a cancelled turn until it terminates or its transport is reset."""

        def reap() -> None:
            state.cancel_callbacks(join_timeout=0.5)
            retry_interval = max(5.0, self._interrupt_timeout * 2.0)
            while not state.done.is_set():
                if self._process is not state.process or not self.is_running:
                    with state.lock:
                        state.transport_closed = True
                    state.done.set()
                    break
                # Keep the route registered and retry a best-effort interrupt.
                # A single abandoned HTTP stream must never reset the shared
                # app-server and kill unrelated healthy turns.
                if state.turn_id is None:
                    if self._delete_thread(
                        state.thread_id,
                        expected_process=state.process,
                    ):
                        with state.lock:
                            state.status = "interrupted"
                        state.done.set()
                        break
                else:
                    interrupted = self._interrupt_turn(
                        state,
                        wait_for_completion=False,
                    )
                    if interrupted:
                        state.done.wait(timeout=min(self._interrupt_timeout, retry_interval))
                    if not state.done.is_set() and self._delete_thread(
                        state.thread_id,
                        expected_process=state.process,
                    ):
                        with state.lock:
                            state.status = "interrupted"
                        state.done.set()
                        break
                state.done.wait(timeout=retry_interval)
            self._remove_turn_state(state)
            state.cancel_callbacks(join_timeout=0.5)
            if (
                state.done.is_set()
                and self._process is state.process
                and self.is_running
            ):
                self._delete_thread(
                    state.thread_id,
                    expected_process=state.process,
                )

        threading.Thread(
            target=reap,
            name=f"codex-reap-{state.thread_id[:8]}",
            daemon=True,
        ).start()

    def _interrupt_turn(
        self,
        state: _TurnState,
        *,
        allow_closing: bool = False,
        wait_for_completion: bool = False,
        rpc_timeout: float | None = None,
    ) -> bool:
        turn_id = state.turn_id
        if turn_id is None:
            return False
        try:
            self._rpc_call(
                "turn/interrupt",
                {"threadId": state.thread_id, "turnId": turn_id},
                timeout=(
                    self._interrupt_timeout
                    if rpc_timeout is None
                    else max(0.1, float(rpc_timeout))
                ),
                allow_closing=allow_closing,
                expected_process=state.process,
            )
        except CodexAppServerError:
            return False
        if wait_for_completion and not state.done.is_set():
            state.done.wait(timeout=self._interrupt_timeout)
        return True

    def _delete_thread(
        self,
        thread_id: str,
        *,
        allow_closing: bool = False,
        expected_process: subprocess.Popen[str] | None = None,
    ) -> bool:
        try:
            self._rpc_call(
                "thread/delete",
                {"threadId": thread_id},
                timeout=self._interrupt_timeout,
                allow_closing=allow_closing,
                expected_process=expected_process,
            )
            return True
        except CodexAppServerError:
            return False

    def _fail_all_waiters(self, *, protocol_error: bool = False) -> None:
        with self._pending_lock:
            pending_calls = list(self._pending.values())
            self._pending.clear()
        for pending in pending_calls:
            pending.transport_closed = True
            pending.event.set()
        with self._turns_lock:
            states = list(
                {id(state): state for state in [
                    *self._turns_by_id.values(),
                    *self._starting_by_thread.values(),
                ]}.values()
            )
        for state in states:
            with state.lock:
                state.transport_closed = not protocol_error
                state.protocol_error = state.protocol_error or protocol_error
            state.done.set()
            state.cancel_callbacks(join_timeout=0)

    def _clear_turn_routes(self) -> None:
        with self._turns_lock:
            self._turns_by_id.clear()
            self._starting_by_thread.clear()

    def _abort_process(self) -> None:
        process = self._process
        stopped = True
        if process is not None:
            stopped = self._stop_process_generation(process, timeout=2.0)
        self._fail_all_waiters()
        self._clear_turn_routes()
        self._initialized = False
        if stopped:
            self._process = None
            self._stdout_thread = None
            self._stderr_thread = None
            self._writer_thread = None
            self._write_queue = None
            self._transport_failed_event = None

    @staticmethod
    def _signal_process(process: subprocess.Popen[str], sig: int) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif sig == signal.SIGKILL:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            # The process may have exited between poll() and signal delivery.
            return
        except OSError:
            try:
                if sig == signal.SIGKILL:
                    process.kill()
                else:
                    process.terminate()
            except OSError:
                return

    @classmethod
    def _terminate_process(
        cls,
        process: subprocess.Popen[str],
        *,
        timeout: float,
    ) -> None:
        if process.poll() is not None:
            return
        cls._signal_process(process, signal.SIGTERM)
        try:
            process.wait(timeout=max(0.1, timeout))
        except subprocess.TimeoutExpired:
            cls._signal_process(process, signal.SIGKILL)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                return

    @staticmethod
    def _signal_process_group(process: subprocess.Popen[str], sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif process.poll() is None:
                if sig == signal.SIGKILL:
                    process.kill()
                else:
                    process.terminate()
        except (OSError, ProcessLookupError):
            return

    def _stop_process_generation(
        self,
        process: subprocess.Popen[str],
        *,
        timeout: float,
    ) -> bool:
        self._terminate_process(process, timeout=timeout)
        if self._join_reader_threads(timeout=timeout):
            return True
        # If a wrapper exited before its native child, the direct Popen may be
        # reaped while descendants still hold the stdio pipes. The saved PGID
        # remains valid while those group members exist.
        self._signal_process_group(process, signal.SIGTERM)
        if self._join_reader_threads(timeout=1.0):
            return True
        self._signal_process_group(process, signal.SIGKILL)
        return self._join_reader_threads(timeout=2.0)

    def _join_reader_threads(self, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        readers: list[threading.Thread] = []
        for reader in (self._writer_thread, self._stdout_thread, self._stderr_thread):
            if reader is None or reader is threading.current_thread():
                continue
            readers.append(reader)
            reader.join(timeout=max(0.0, deadline - time.monotonic()))
        return all(not reader.is_alive() for reader in readers)

    @staticmethod
    def _select_final_message(
        messages: Mapping[str, tuple[str | None, str]],
    ) -> tuple[str, str]:
        final_messages = [
            (item_id, text)
            for item_id, (phase, text) in messages.items()
            if phase == "final_answer"
        ]
        if final_messages:
            return final_messages[-1]
        legacy_messages = [
            (item_id, text)
            for item_id, (phase, text) in messages.items()
            if phase is None
        ]
        if legacy_messages:
            return legacy_messages[-1]
        raise CodexAppServerProtocolError(
            "Codex completed the turn without a final agent message"
        )

    @staticmethod
    def _select_final_text(
        messages: Mapping[str, tuple[str | None, str]],
    ) -> str:
        return CodexAppServerClient._select_final_message(messages)[1]

    @staticmethod
    def _parse_usage(payload: Mapping[str, Any] | None) -> CodexTokenUsage | None:
        if payload is None:
            return None
        total = payload.get("total")
        if not isinstance(total, Mapping):
            total = payload.get("last")
        if not isinstance(total, Mapping):
            return None
        context_window = payload.get("modelContextWindow")
        return CodexTokenUsage(
            input_tokens=_positive_int(total.get("inputTokens")),
            cached_input_tokens=_positive_int(total.get("cachedInputTokens")),
            output_tokens=_positive_int(total.get("outputTokens")),
            reasoning_output_tokens=_positive_int(total.get("reasoningOutputTokens")),
            total_tokens=_positive_int(total.get("totalTokens")),
            model_context_window=(
                _positive_int(context_window) if context_window is not None else None
            ),
        )


__all__ = [
    "CodexAppServerClient",
    "CodexAppServerClosedError",
    "CodexAppServerError",
    "CodexAppServerProtocolError",
    "CodexAppServerRPCError",
    "CodexAppServerTimeoutError",
    "CodexStreamConsumerClosedError",
    "CodexTokenUsage",
    "CodexTurnFailedError",
    "CodexTurnInterruptedError",
    "CodexTurnResult",
    "DeltaCallback",
]
