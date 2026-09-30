from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from policy.roboharn_evo.scripts import record_local_serving_runtime_identity as runtime_identity


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "policy" / "roboharn_evo" / "scripts" / "record_local_serving_runtime_identity.py"


def _python_probe_payload() -> dict:
    return {
        "python": {
            "executable": sys.executable,
            "real_executable": str(Path(sys.executable).resolve()),
            "implementation": "CPython",
            "version": "3.11.9",
            "version_info": [3, 11, 9, "final", 0],
        },
        "distributions": [
            {"lookup_name": "vllm", "name": "vllm", "version": "0.10.2.dev0"},
            {"lookup_name": "torch", "name": "torch", "version": "2.8.0+cu129"},
            {
                "lookup_name": "transformers",
                "name": "transformers",
                "version": "4.57.0.dev0",
            },
            {"lookup_name": "openai", "name": "openai", "version": "1.109.1"},
        ],
        "vllm": {
            "executable": "/opt/qwen/bin/vllm",
            "executable_sha256": "a" * 64,
        },
        "torch": {"cuda_build_version": "12.9"},
    }


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class LocalServingRuntimeIdentityTest(unittest.TestCase):
    def test_embedded_target_python_probe_parses(self) -> None:
        compile(runtime_identity._PYTHON_PROBE, "embedded_runtime_identity_probe.py", "exec")

    def test_records_exact_runtime_gpu_and_server_identity_without_environment(self) -> None:
        python_stdout = json.dumps(_python_probe_payload())
        nvidia_stdout = (
            "1, GPU-b, NVIDIA RTX PRO 5000 Blackwell, 73415, 580.126.09\n"
            "0, GPU-a, NVIDIA RTX PRO 5000 Blackwell, 73415, 580.126.09\n"
        )
        with mock.patch.object(
            runtime_identity,
            "_run_checked",
            side_effect=[
                _completed(python_stdout),
                _completed("0.10.2.dev0\n"),
                _completed(nvidia_stdout),
            ],
        ) as run_checked, mock.patch.object(
            runtime_identity.shutil, "which", return_value="/usr/bin/nvidia-smi"
        ), mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "must-not-be-recorded", "HF_TOKEN": "also-secret"},
        ):
            payload = runtime_identity.build_local_serving_runtime_identity(
                python_executable=Path(sys.executable),
                server_args=["--tensor-parallel-size", "8", "--enable-prefix-caching"],
                runtime_settings=["VLLM_USE_FLASHINFER_SAMPLER=0"],
            )

        self.assertEqual(run_checked.call_count, 3)
        python_command = run_checked.call_args_list[0].args[0]
        self.assertEqual(python_command[:3], [str(Path(sys.executable).absolute()), "-I", "-c"])
        self.assertEqual(
            run_checked.call_args_list[0].kwargs["timeout_seconds"],
            runtime_identity.DEFAULT_PYTHON_PROBE_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            run_checked.call_args_list[1].args[0],
            ["/opt/qwen/bin/vllm", "--version"],
        )
        self.assertEqual(
            run_checked.call_args_list[1].kwargs["timeout_seconds"],
            runtime_identity.DEFAULT_VLLM_VERSION_TIMEOUT_SECONDS,
        )
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["python"]["executable"], sys.executable)
        self.assertEqual(payload["python"]["real_executable"], str(Path(sys.executable).resolve()))
        self.assertEqual(payload["distributions"]["required_lookup_names"],
                         ["vllm", "torch", "transformers", "openai"])
        versions = {
            item["lookup_name"]: (item["name"], item["version"])
            for item in payload["distributions"]["installed"]
        }
        self.assertEqual(versions["vllm"], ("vllm", "0.10.2.dev0"))
        self.assertEqual(payload["vllm"]["executable_sha256"], "a" * 64)
        self.assertEqual(payload["vllm"]["version_output"], "0.10.2.dev0")
        self.assertEqual(payload["torch"]["cuda_build_version"], "12.9")
        self.assertEqual(
            payload["probe_timeouts_seconds"],
            {"python_identity": 300, "vllm_version": 300, "nvidia_smi": 30},
        )
        self.assertEqual(payload["nvidia_smi"]["driver_version"], "580.126.09")
        self.assertEqual([gpu["index"] for gpu in payload["nvidia_smi"]["gpus"]], [0, 1])
        self.assertEqual(payload["nvidia_smi"]["gpus"][0]["memory_total_mib"], 73415)
        self.assertEqual(
            payload["server_args"],
            ["--tensor-parallel-size", "8", "--enable-prefix-caching"],
        )
        self.assertEqual(
            payload["runtime_settings"],
            {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        )
        self.assertFalse(payload["secrets_recorded"])
        self.assertFalse(payload["model_server_started"])
        rendered = json.dumps(payload)
        self.assertNotIn("must-not-be-recorded", rendered)
        self.assertNotIn("also-secret", rendered)

    def test_absent_nvidia_smi_is_recorded_without_failing_cpu_host(self) -> None:
        with mock.patch.object(
            runtime_identity,
            "_run_checked",
            side_effect=[
                _completed(json.dumps(_python_probe_payload())),
                _completed("0.10.2.dev0\n"),
            ],
        ), mock.patch.object(runtime_identity.shutil, "which", return_value=None):
            payload = runtime_identity.build_local_serving_runtime_identity(
                python_executable=Path(sys.executable), server_args=[]
            )

        self.assertEqual(
            payload["nvidia_smi"],
            {"available": False, "reason": "executable_not_found"},
        )

    def test_missing_required_distribution_fails_closed(self) -> None:
        probe = _python_probe_payload()
        probe["distributions"] = probe["distributions"][:-1]
        with mock.patch.object(
            runtime_identity, "_run_checked", return_value=_completed(json.dumps(probe))
        ), mock.patch.object(runtime_identity.shutil, "which", return_value=None):
            with self.assertRaisesRegex(runtime_identity.RuntimeIdentityError, "missing=.*openai"):
                runtime_identity.build_local_serving_runtime_identity(
                    python_executable=Path(sys.executable), server_args=[]
                )

    def test_malformed_or_duplicate_probe_json_fails_closed(self) -> None:
        with mock.patch.object(
            runtime_identity,
            "_run_checked",
            return_value=_completed('{"python":{},"python":{}}'),
        ):
            with self.assertRaisesRegex(runtime_identity.RuntimeIdentityError, "duplicate JSON key"):
                runtime_identity.build_local_serving_runtime_identity(
                    python_executable=Path(sys.executable), server_args=[]
                )

    def test_vllm_version_failure_stops_before_gpu_probe(self) -> None:
        with mock.patch.object(
            runtime_identity,
            "_run_checked",
            side_effect=[
                _completed(json.dumps(_python_probe_payload())),
                runtime_identity.RuntimeIdentityError("vllm --version probe timed out"),
            ],
        ) as run_checked, mock.patch.object(
            runtime_identity.shutil, "which"
        ) as nvidia_which:
            with self.assertRaisesRegex(runtime_identity.RuntimeIdentityError, "timed out"):
                runtime_identity.build_local_serving_runtime_identity(
                    python_executable=Path(sys.executable), server_args=[]
                )

        self.assertEqual(run_checked.call_count, 2)
        nvidia_which.assert_not_called()

    def test_invalid_vllm_executable_sha256_fails_closed(self) -> None:
        probe = _python_probe_payload()
        probe["vllm"]["executable_sha256"] = "not-a-sha256"
        with mock.patch.object(
            runtime_identity, "_run_checked", return_value=_completed(json.dumps(probe))
        ), mock.patch.object(runtime_identity.shutil, "which", return_value=None):
            with self.assertRaisesRegex(
                runtime_identity.RuntimeIdentityError,
                "vllm.executable_sha256",
            ):
                runtime_identity.build_local_serving_runtime_identity(
                    python_executable=Path(sys.executable), server_args=[]
                )

    def test_forbidden_secret_server_arguments_are_rejected_before_probe(self) -> None:
        forbidden = (
            ["--api-key", "sk-secret-value"],
            ["--authorization=Bearer secret"],
            ["HF_TOKEN=hf_secretvalue"],
            ["bearer opaque-secret"],
        )
        for server_args in forbidden:
            with self.subTest(server_args=server_args), mock.patch.object(
                runtime_identity, "_run_checked"
            ) as run_checked:
                with self.assertRaisesRegex(runtime_identity.RuntimeIdentityError, "credential"):
                    runtime_identity.build_local_serving_runtime_identity(
                        python_executable=Path(sys.executable), server_args=server_args
                    )
                run_checked.assert_not_called()

    def test_runtime_settings_are_narrowly_allowlisted_before_probe(self) -> None:
        invalid = (
            ["HF_TOKEN=must-not-be-recorded"],
            ["VLLM_USE_FLASHINFER_SAMPLER=2"],
            ["VLLM_USE_FLASHINFER_SAMPLER"],
            ["VLLM_USE_FLASHINFER_SAMPLER=0=extra"],
            [
                "VLLM_USE_FLASHINFER_SAMPLER=0",
                "VLLM_USE_FLASHINFER_SAMPLER=1",
            ],
        )
        for runtime_settings in invalid:
            with self.subTest(runtime_settings=runtime_settings), mock.patch.object(
                runtime_identity, "_run_checked"
            ) as run_checked:
                with self.assertRaises(runtime_identity.RuntimeIdentityError):
                    runtime_identity.build_local_serving_runtime_identity(
                        python_executable=Path(sys.executable),
                        server_args=[],
                        runtime_settings=runtime_settings,
                    )
                run_checked.assert_not_called()

    def test_atomic_writer_uses_same_directory_replace_and_private_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_serving_identity_atomic_") as directory:
            output = Path(directory) / "nested" / "identity.json"
            real_replace = os.replace
            calls: list[tuple[str, str]] = []

            def tracking_replace(source: str, destination: str | Path) -> None:
                calls.append((source, str(destination)))
                real_replace(source, destination)

            with mock.patch.object(runtime_identity.os, "replace", side_effect=tracking_replace):
                runtime_identity.write_json_atomic(output, {"schema_version": 1})

            self.assertEqual(len(calls), 1)
            self.assertEqual(Path(calls[0][0]).parent, output.parent)
            self.assertEqual(Path(calls[0][1]), output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"schema_version": 1})
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    def test_atomic_writer_rejects_nested_secret_fields_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_serving_identity_secret_") as directory:
            output = Path(directory) / "identity.json"
            with self.assertRaisesRegex(runtime_identity.RuntimeIdentityError, "api_key"):
                runtime_identity.write_json_atomic(
                    output,
                    {"runtime": {"credentials": [{"api_key": "must-not-be-recorded"}]}},
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    def test_cli_probe_failure_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_serving_identity_cli_") as directory:
            output = Path(directory) / "identity.json"
            output.write_text("old-valid-identity\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--python",
                    "/bin/false",
                    "--output",
                    str(output),
                    "--server-arg=--tensor-parallel-size",
                    "--server-arg=8",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "old-valid-identity\n")
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    def test_relative_python_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(runtime_identity.RuntimeIdentityError, "absolute"):
            runtime_identity.build_local_serving_runtime_identity(
                python_executable=Path("python"), server_args=[]
            )

    def test_target_virtualenv_python_symlink_is_not_resolved_before_execution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_serving_identity_venv_") as directory:
            python_link = Path(directory) / "python"
            python_link.symlink_to(Path(sys.executable).resolve())
            probe = _python_probe_payload()
            probe["python"]["executable"] = str(python_link)
            probe["python"]["real_executable"] = str(Path(sys.executable).resolve())
            with mock.patch.object(
                runtime_identity,
                "_run_checked",
                side_effect=[_completed(json.dumps(probe)), _completed("0.10.2.dev0\n")],
            ) as run_checked, mock.patch.object(runtime_identity.shutil, "which", return_value=None):
                payload = runtime_identity.build_local_serving_runtime_identity(
                    python_executable=python_link, server_args=[]
                )

            self.assertEqual(run_checked.call_args_list[0].args[0][0], str(python_link))
            self.assertEqual(payload["python"]["executable"], str(python_link))
            self.assertEqual(
                payload["python"]["real_executable"], str(Path(sys.executable).resolve())
            )

    def test_custom_bounded_timeouts_reach_the_two_separate_probes(self) -> None:
        with mock.patch.object(
            runtime_identity,
            "_run_checked",
            side_effect=[
                _completed(json.dumps(_python_probe_payload())),
                _completed("0.10.2.dev0\n"),
            ],
        ) as run_checked, mock.patch.object(runtime_identity.shutil, "which", return_value=None):
            payload = runtime_identity.build_local_serving_runtime_identity(
                python_executable=Path(sys.executable),
                server_args=[],
                python_probe_timeout_seconds=420,
                vllm_version_timeout_seconds=900,
            )

        self.assertEqual(run_checked.call_args_list[0].kwargs["timeout_seconds"], 420)
        self.assertEqual(run_checked.call_args_list[1].kwargs["timeout_seconds"], 900)
        self.assertEqual(payload["probe_timeouts_seconds"]["python_identity"], 420)
        self.assertEqual(payload["probe_timeouts_seconds"]["vllm_version"], 900)

    def test_out_of_range_timeouts_fail_before_any_probe(self) -> None:
        cases = (
            {"python_probe_timeout_seconds": 0},
            {"python_probe_timeout_seconds": 1801},
            {"vllm_version_timeout_seconds": 0},
            {"vllm_version_timeout_seconds": 1801},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), mock.patch.object(
                runtime_identity, "_run_checked"
            ) as run_checked:
                with self.assertRaisesRegex(runtime_identity.RuntimeIdentityError, "between 1 and 1800"):
                    runtime_identity.build_local_serving_runtime_identity(
                        python_executable=Path(sys.executable),
                        server_args=[],
                        **overrides,
                    )
                run_checked.assert_not_called()

    def test_subprocess_timeout_is_secret_free_and_fail_closed(self) -> None:
        timeout = subprocess.TimeoutExpired(
            cmd=["/opt/qwen/bin/vllm", "--version"],
            timeout=17,
            output="must-not-be-reported",
            stderr="also-secret",
        )
        with mock.patch.object(runtime_identity.subprocess, "run", side_effect=timeout):
            with self.assertRaisesRegex(
                runtime_identity.RuntimeIdentityError,
                "timed out after 17 seconds",
            ) as captured:
                runtime_identity._run_checked(
                    ["/opt/qwen/bin/vllm", "--version"],
                    label="vllm --version probe",
                    timeout_seconds=17,
                )

        self.assertNotIn("must-not-be-reported", str(captured.exception))
        self.assertNotIn("also-secret", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
