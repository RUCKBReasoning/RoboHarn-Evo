from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "smoke_renderer_device_binding.py"
)
SPEC = importlib.util.spec_from_file_location("renderer_device_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
renderer_smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer_smoke
SPEC.loader.exec_module(renderer_smoke)


class RendererDeviceSmokeScriptTest(unittest.TestCase):
    def test_parse_gpu_inventory_normalizes_pci_and_rejects_duplicates(self) -> None:
        inventory = renderer_smoke.parse_gpu_inventory(
            "4, GPU-abcd, 00000000:4D:00.0\n7, GPU-ef01, 0000:AF:00.0\n"
        )

        self.assertEqual(inventory[4]["uuid"], "GPU-abcd")
        self.assertEqual(inventory[4]["pci_bus_id"], "0000:4d:00.0")
        self.assertEqual(inventory[7]["pci_bus_id"], "0000:af:00.0")
        with self.assertRaisesRegex(renderer_smoke.SmokeError, "duplicate GPU index"):
            renderer_smoke.parse_gpu_inventory(
                "4, GPU-abcd, 0000:4d:00.0\n4, GPU-other, 0000:5e:00.0\n"
            )

    def test_child_environment_derives_exact_pci_alias_and_uuid_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            provenance_path = Path(temporary_directory) / "renderer.json"
            environment = renderer_smoke.build_child_environment(
                {
                    "KEEP_ME": "yes",
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "RMBENCH_RENDER_DEVICE": "cuda:7",
                },
                gpu={
                    "index": 4,
                    "uuid": "GPU-abcd",
                    "pci_bus_id": "00000000:4D:00.0",
                },
                provenance_path=provenance_path,
            )

        self.assertEqual(environment["KEEP_ME"], "yes")
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-abcd")
        expected = {
            "RMBENCH_RENDER_DEVICE": "pci:0000:4d:00.0",
            "RMBENCH_RENDER_DEVICE_STRICT": "1",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": "0000:4d:00.0",
            "RMBENCH_EXPECTED_PHYSICAL_GPU": "4",
            "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH": str(provenance_path),
        }
        self.assertEqual(
            {name: environment[name] for name in renderer_smoke.RENDER_ENV_NAMES},
            expected,
        )

    def test_parent_selects_exact_inventory_record_and_reexecutes_child(self) -> None:
        child_result = {
            "schema": renderer_smoke.SCHEMA,
            "ok": True,
            "physical_gpu_index": 4,
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(child_result) + "\n", stderr=""
        )
        parser = renderer_smoke.build_argument_parser()
        args = parser.parse_args(["--gpu-index", "4"])
        inventory = {
            4: {
                "index": 4,
                "uuid": "GPU-abcd",
                "pci_bus_id": "0000:4d:00.0",
            }
        }

        with mock.patch.object(
            renderer_smoke, "query_gpu_inventory", return_value=inventory
        ), mock.patch.object(
            renderer_smoke.subprocess, "run", return_value=completed
        ) as run_mock, mock.patch("builtins.print") as print_mock:
            return_code = renderer_smoke.run_parent(args)

        self.assertEqual(return_code, 0)
        command = run_mock.call_args.args[0]
        environment = run_mock.call_args.kwargs["env"]
        self.assertIn("--_child", command)
        self.assertIn("GPU-abcd", command)
        self.assertIn("0000:4d:00.0", command)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-abcd")
        self.assertEqual(environment["RMBENCH_EXPECTED_PHYSICAL_GPU"], "4")
        emitted = json.loads(print_mock.call_args.args[0])
        self.assertTrue(emitted["ok"])

    def test_parent_fails_before_child_when_requested_gpu_is_absent(self) -> None:
        parser = renderer_smoke.build_argument_parser()
        args = parser.parse_args(["--gpu-index", "4"])
        with mock.patch.object(
            renderer_smoke,
            "query_gpu_inventory",
            return_value={0: {"index": 0, "uuid": "GPU-zero", "pci_bus_id": "0000:01:00.0"}},
        ), mock.patch.object(renderer_smoke.subprocess, "run") as run_mock:
            with self.assertRaisesRegex(renderer_smoke.SmokeError, "is absent"):
                renderer_smoke.run_parent(args)
        run_mock.assert_not_called()

    def test_parent_rejects_nonpositive_timeout_before_gpu_query(self) -> None:
        parser = renderer_smoke.build_argument_parser()
        args = parser.parse_args(["--gpu-index", "4", "--timeout-sec", "0"])
        with mock.patch.object(renderer_smoke, "query_gpu_inventory") as query_mock:
            with self.assertRaisesRegex(renderer_smoke.SmokeError, "positive finite"):
                renderer_smoke.run_parent(args)
        query_mock.assert_not_called()

    def test_extract_child_result_ignores_native_log_lines(self) -> None:
        expected = {"schema": renderer_smoke.SCHEMA, "ok": True}
        output = "native Vulkan diagnostic\n" + json.dumps(expected) + "\n"
        self.assertEqual(renderer_smoke._extract_child_result(output), expected)


if __name__ == "__main__":
    unittest.main()
