"""RPent's original Codex planner, pinned to ChatGPT authentication and high."""
from __future__ import annotations

from dataclasses import replace
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from rpent.planner.codex import CodexPlanner, _get


class BudgetRecorder:
    """RPent's recorder plus an enforceable completed-response budget.

The stock Codex backend displays max_turns but does not stop at that value.
Usage-update events count model responses, not physical controller steps or
reasoning text fragments. Interrupting never restarts a turn or an episode.
"""
    def __init__(self, recorder, state):
        self.recorder, self.state = recorder, state
        self.seen_usage = set()
        self.model_responses = 0
        self.budget_exhausted = False

    def __getattr__(self, name):
        return getattr(self.recorder, name)

    def observe(self, event):
        text = self.recorder.observe(event)
        if _get(event, "method") == "thread/tokenUsage/updated":
            payload = _get(event, "payload")
            update = _get(payload, "token_usage")
            total = _get(update, "total")
            if total is not None:
                # New SDK wraps usage in {last, total}; upstream RPent expects
                # a flat object and otherwise reports zero tokens forever.
                self.recorder._set_usage(total)
            usage = tuple(sorted(self.recorder.usage.items()))
            if usage not in self.seen_usage and any(value for _, value in usage):
                self.seen_usage.add(usage)
                self.model_responses += 1
            if self.model_responses >= self.recorder.max_turns and not self.budget_exhausted:
                self.budget_exhausted = True
                turn = self.state.get("turn")
                if turn is not None:
                    turn.interrupt()
        return text


class ChatGPTPlanner(CodexPlanner):
    def __init__(self, *, account_home, **kwargs):
        super().__init__(**kwargs)
        self.account_home = str(account_home)
        self._base_url = None
        self._api_key = None
        self.budget_recorder = None

    def _run_session(self, prompt, output_path, raw_stream_path, last_message_path, recorder, state, mcp_url, input_queue=None):
        self.budget_recorder = BudgetRecorder(recorder, state)
        return super()._run_session(prompt, output_path, raw_stream_path, last_message_path,
                                    self.budget_recorder, state, mcp_url, input_queue)

    def _build_config(self, mcp_url):
        config = super()._build_config(mcp_url)
        env = dict(config.env)
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY", "CODEX_BASE_URL", "CODEX_ACCESS_TOKEN"):
            env.pop(key, None)
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            if key in env:
                value = env[key].strip()
                if not value:
                    env.pop(key)
                elif "://" not in value:
                    env[key] = "http://" + value
        env["CODEX_HOME"] = self.account_home
        overrides = {
            "model_provider": "openai", "model_reasoning_effort": "high",
            "forced_login_method": "chatgpt", "web_search": "disabled",
            "mcp_servers.rpent.tool_timeout_sec": min(self._timeout_s, 3700),
            "features.shell_tool": False, "features.unified_exec": False,
            "features.multi_agent": False, "features.apps": False,
            "features.hooks": False, "features.remote_plugin": False,
            "features.goals": False,
        }
        # Current Codex reserves the built-in OpenAI provider. Keep its native
        # transport reconnect/retry behavior; do not replace account auth with
        # a custom relay provider or retry entire robot turns after failure.
        return replace(config, env=env, config_overrides=config.config_overrides + tuple(
            f"{key}={json.dumps(value)}" for key, value in overrides.items()))


@contextmanager
def account_home(auth_file):
    # Source credentials stay outside the repository and are never logged.
    source = Path(auth_file).expanduser()
    metadata = json.loads(source.read_text())
    if not isinstance(metadata.get("tokens"), dict):
        raise ValueError("ChatGPT account credentials required; API-key fallback is not allowed")
    # Simulator TMPDIR points inside shareable rollout artifacts. Credentials
    # must not follow it, even transiently or if a worker later crashes.
    with tempfile.TemporaryDirectory(prefix="rpent-account-", dir="/tmp") as directory:
        destination = Path(directory) / "auth.json"
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
        yield Path(directory)
