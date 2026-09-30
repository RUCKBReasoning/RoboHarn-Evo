from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from policy.roboharn_evo.scripts.check_pure_tool_control_early_stop import analyze_trace


class PureToolControlTraceCheckTest(unittest.TestCase):
    def _raw_trace(self, lines: list[str]) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="tcm_trace_check_"))
        path = directory / "episode_0000_agent_trace.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.addCleanup(lambda: directory.rmdir())
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def _trace(self, records: list[dict]) -> Path:
        return self._raw_trace([json.dumps(record) for record in records])

    def test_incomplete_trace_is_not_reported_as_pass(self) -> None:
        path = self._trace(
            [
                {"event": "episode_start"},
                {"event": "debug_recovery_trigger", "env_step": 0},
            ]
        )

        report = analyze_trace(path)
        allowed = analyze_trace(path, allow_incomplete=True)

        self.assertEqual([item["type"] for item in report["bad_issues"]], ["incomplete_trace"])
        self.assertEqual(allowed["bad_issues"], [])

    def test_failed_episode_with_task_finished_is_bad(self) -> None:
        path = self._trace(
            [
                {"event": "episode_start"},
                {"event": "episode_agent_finished", "task_finished": True, "step": 3},
                {"event": "episode_end", "task_finished": True, "success": False, "total_steps": 3},
            ]
        )

        report = analyze_trace(path)

        self.assertEqual(
            [item["type"] for item in report["bad_issues"]],
            ["episode_agent_finished", "episode_end_failed_with_task_finished"],
        )

    def test_environment_success_is_complete(self) -> None:
        path = self._trace(
            [
                {"event": "episode_start"},
                {"event": "episode_end", "task_finished": True, "success": True, "total_steps": 20},
            ]
        )

        report = analyze_trace(path)

        self.assertEqual(report["bad_issues"], [])

    def test_external_interrupt_is_distinct_from_agent_finish(self) -> None:
        path = self._trace(
            [
                {"event": "episode_start"},
                {
                    "event": "episode_interrupt",
                    "reason": "external_sigterm",
                    "interrupt_signal": "sigterm",
                    "step": 12,
                },
                {
                    "event": "episode_end",
                    "success": False,
                    "task_finished": False,
                    "failure_reason": "external_sigterm",
                    "total_steps": 12,
                },
            ]
        )

        report = analyze_trace(path)

        self.assertEqual([item["type"] for item in report["bad_issues"]], ["episode_interrupted"])
        self.assertEqual(report["bad_issues"][0]["event"]["interrupt_signal"], "sigterm")

    def test_nonempty_malformed_jsonl_line_is_counted_and_fails_closed(self) -> None:
        path = self._raw_trace(
            [
                json.dumps({"event": "episode_start"}),
                '{"event":"control_turn_result",',
                json.dumps({"event": "episode_end", "success": True}),
            ]
        )

        report = analyze_trace(path)
        allowed = analyze_trace(path, allow_incomplete=True)

        self.assertEqual(report["malformed_jsonl_line_count"], 1)
        self.assertEqual(report["malformed_jsonl_line_numbers"], [2])
        self.assertIn("malformed_jsonl", [item["type"] for item in report["bad_issues"]])
        self.assertIn("malformed_jsonl", [item["type"] for item in allowed["bad_issues"]])

    def test_invalid_utf8_jsonl_line_is_counted_as_malformed(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="tcm_trace_check_invalid_utf8_"))
        path = directory / "episode_0000_agent_trace.jsonl"
        self.addCleanup(lambda: directory.rmdir())
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        path.write_bytes(
            b'{"event":"episode_start"}\n'
            b'{"event":"bad","value":"\xff"}\n'
            b'{"event":"episode_end","success":true}\n'
        )

        report = analyze_trace(path)

        self.assertEqual(report["malformed_jsonl_line_count"], 1)
        self.assertEqual(report["malformed_jsonl_line_numbers"], [2])
        self.assertIn("malformed_jsonl", [item["type"] for item in report["bad_issues"]])


if __name__ == "__main__":
    unittest.main()
