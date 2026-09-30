from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from policy.roboharn_evo.scripts.record_gpu_memory_gate import build_report, parse_inventory


class GpuMemoryGateTest(unittest.TestCase):
    def test_parse_inventory_preserves_selected_memory_values(self) -> None:
        rows = parse_inventory(
            "0, GPU-a, RTX PRO 5000 Blackwell, 73415, 12000, 61415\n"
            "7, GPU-h, RTX PRO 5000 Blackwell, 73415, 60000, 13415\n"
        )

        self.assertEqual(rows[1]["index"], 7)
        self.assertEqual(rows[1]["memory_free_mib"], 13415)

    @mock.patch("policy.roboharn_evo.scripts.record_gpu_memory_gate.subprocess.run")
    def test_gate_passes_and_records_read_only_result(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="7, GPU-h, RTX PRO 5000 Blackwell, 73415, 60000, 13415\n",
            stderr="",
        )

        report = build_report(
            gpu_index=7,
            minimum_free_mib=12288,
            nvidia_smi_bin="nvidia-smi",
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["observed_free_mib"], 13415)
        self.assertTrue(report["read_only_check"])
        self.assertFalse(report["processes_stopped"])

    @mock.patch("policy.roboharn_evo.scripts.record_gpu_memory_gate.subprocess.run")
    def test_gate_fails_below_threshold(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="7, GPU-h, RTX PRO 5000 Blackwell, 73415, 62000, 11415\n",
            stderr="",
        )

        report = build_report(
            gpu_index=7,
            minimum_free_mib=12288,
            nvidia_smi_bin="nvidia-smi",
        )

        self.assertEqual(report["status"], "failed")
        self.assertLess(report["headroom_mib"], 0)


if __name__ == "__main__":
    unittest.main()
