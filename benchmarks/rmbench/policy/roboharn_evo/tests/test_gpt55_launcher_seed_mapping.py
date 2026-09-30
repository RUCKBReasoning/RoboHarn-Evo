from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOHARN_EVO_PROJECT_ROOT = REPO_ROOT.parents[1]
FORMAL_TEST_RUNTIME_ROOT = (
    ROBOHARN_EVO_PROJECT_ROOT / "eval_result" / "rmbench" / "copied_test_runtime"
)
RUNNER = REPO_ROOT / "policy" / "roboharn_evo" / "scripts" / "run_gpt55_pure_tool_control_8way.sh"
WRAPPER = REPO_ROOT / "policy" / "roboharn_evo" / "scripts" / "run_gpt55_object_info_ablation_8way.sh"


class Gpt55LauncherSeedMappingTest(unittest.TestCase):
    def _run(
        self,
        *,
        extra_args: tuple[str, ...] = (),
        **overrides: str,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="tcm_seed_launcher_") as directory:
            env = os.environ.copy()
            env.update(
                {
                    "DETACH": "0",
                    "RMBENCH_DETACHED_CHILD": "0",
                    "NUM_WORKERS": "4",
                    "GPU_IDS": "2,3,4,5",
                    "N_PER_WORKER": "1",
                    "EVAL_START_SEEDS": "100001,100002,100003,100004",
                    "TASK_NAME": "synthetic_task",
                    "NON_FORMAL_DIAGNOSTIC": "1",
                    "SKIP_PREFLIGHT": "1",
                    "PREFLIGHT_ONLY": "1",
                    "RECORD_RUNTIME_PROVENANCE": "0",
                    "LOG_ROOT": directory,
                    **overrides,
                }
            )
            return subprocess.run(
                ["bash", str(RUNNER), *extra_args],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                check=False,
                text=True,
            )

    def _run_with_fake_preflight(
        self,
        *,
        inference_json: str,
        inference_exit_code: int = 0,
        health_json: str | None = None,
        extra_args: tuple[str, ...] = (),
        **overrides: str,
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        FORMAL_TEST_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="tcm_preflight_launcher_",
            dir=FORMAL_TEST_RUNTIME_ROOT,
        ) as directory, tempfile.TemporaryDirectory(
            prefix="tcm_readonly_assets_"
        ) as assets_directory:
            root = Path(directory)
            assets_root = Path(assets_directory)
            (assets_root / "embodiments").mkdir()
            (assets_root / "objects").mkdir()
            fake_bin = root / "bin"
            fake_bin.mkdir()
            curl_log = root / "curl.log"
            fake_curl = fake_bin / "curl"
            fake_curl.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
url="${!#}"
printf '%s\\n' "$url" >> "${FAKE_CURL_LOG}"
case "$url" in
  */health)
    printf '%s' "${FAKE_HEALTH_JSON}"
    ;;
  */perception_queries)
    if [[ "${FAKE_INFERENCE_EXIT_CODE}" != "0" ]]; then
      exit "${FAKE_INFERENCE_EXIT_CODE}"
    fi
    printf '%s' "${FAKE_INFERENCE_JSON}"
    ;;
  *)
    exit 22
    ;;
esac
""",
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "DETACH": "0",
                    "RMBENCH_DETACHED_CHILD": "0",
                    "NUM_WORKERS": "1",
                    "GPU_IDS": "0",
                    "N_PER_WORKER": "1",
                    "EVAL_START_SEEDS": "100001",
                    "TASK_NAME": "synthetic_task",
                    "PERCEPTION_CONDITION": "oracle",
                    "NON_FORMAL_DIAGNOSTIC": "0",
                    "SKIP_PREFLIGHT": "0",
                    "PREFLIGHT_ONLY": "1",
                    "RECORD_RUNTIME_PROVENANCE": "1",
                    "RUNTIME_PROVENANCE_PATH": str(root / "runtime_provenance.json"),
                    "RMBENCH_OUTPUT_ROOT": str(root),
                    "RMBENCH_ASSETS_ROOT": str(assets_root),
                    "REQUIRE_SAM3_PREFLIGHT": "0",
                    "AGENT_INFERENCE_PREFLIGHT_TIMEOUT_SEC": "5",
                    "LOG_ROOT": str(root / "logs"),
                    "PATH": f"{fake_bin}:{env.get('PATH', '')}",
                    "FAKE_CURL_LOG": str(curl_log),
                    "FAKE_HEALTH_JSON": health_json
                    or (
                        '{"status":"ok","model":"gpt-5.5",'
                        '"api_mode":"responses_compat","reasoning_effort":"xhigh",'
                        '"response_storage":"account_default","timeout_sec":600,'
                        '"max_concurrent_requests":4,"app_server_running":true}'
                    ),
                    "FAKE_INFERENCE_JSON": inference_json,
                    "FAKE_INFERENCE_EXIT_CODE": str(inference_exit_code),
                    **overrides,
                }
            )
            result = subprocess.run(
                ["bash", str(RUNNER), *extra_args],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                check=False,
                text=True,
            )
            calls = curl_log.read_text(encoding="utf-8") if curl_log.exists() else ""
            return result, calls

    def _run_with_fake_worker(
        self,
        *,
        formal: bool,
        capture_renderer_env: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        runtime_parent = FORMAL_TEST_RUNTIME_ROOT if formal else None
        if runtime_parent is not None:
            runtime_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="tcm_worker_env_launcher_",
            dir=runtime_parent,
        ) as directory, tempfile.TemporaryDirectory(
            prefix="tcm_readonly_assets_"
        ) as assets_directory:
            root = Path(directory)
            assets_root = Path(assets_directory)
            (assets_root / "embodiments").mkdir()
            (assets_root / "objects").mkdir()
            fake_bin = root / "bin"
            fake_bin.mkdir()
            env_log = root / "worker_env.log"
            curl_log = root / "curl.log"
            conda_sh = root / "conda.sh"
            conda_sh.write_text("conda() { :; }\n", encoding="utf-8")

            fake_python = fake_bin / "python"
            fake_python.write_text(
                f"""#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
  if [[ "$arg" == "script/eval_policy.py" ]]; then
    printf 'formal=%s\\n' "${{ROBOHARN_EVO_FORMAL_PROTOCOL-}}" > "${{FAKE_WORKER_ENV_LOG}}"
    printf 'protocol_version=%s\\n' "${{ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION-}}" >> "${{FAKE_WORKER_ENV_LOG}}"
    printf 'provenance=%s\\n' "${{ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH-}}" >> "${{FAKE_WORKER_ENV_LOG}}"
    if [[ "${{FAKE_CAPTURE_RENDERER_ENV-0}}" == "1" ]]; then
      printf 'render_device=%s\\n' "${{RMBENCH_RENDER_DEVICE-}}" >> "${{FAKE_WORKER_ENV_LOG}}"
      printf 'render_strict=%s\\n' "${{RMBENCH_RENDER_DEVICE_STRICT-}}" >> "${{FAKE_WORKER_ENV_LOG}}"
      printf 'render_cuda_id=%s\\n' "${{RMBENCH_EXPECTED_RENDER_CUDA_ID-}}" >> "${{FAKE_WORKER_ENV_LOG}}"
      printf 'render_pci=%s\\n' "${{RMBENCH_EXPECTED_RENDER_PCI_BUS_ID-}}" >> "${{FAKE_WORKER_ENV_LOG}}"
      printf 'render_physical_gpu=%s\\n' "${{RMBENCH_EXPECTED_PHYSICAL_GPU-}}" >> "${{FAKE_WORKER_ENV_LOG}}"
      printf 'render_cuda_visible_devices=%s\\n' "${{CUDA_VISIBLE_DEVICES-}}" >> "${{FAKE_WORKER_ENV_LOG}}"
      printf 'render_cuda_selector=%s\\n' "${{RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN-}}" >> "${{FAKE_WORKER_ENV_LOG}}"
      printf 'render_provenance=%s\\n' "${{RMBENCH_RENDER_DEVICE_PROVENANCE_PATH-}}" >> "${{FAKE_WORKER_ENV_LOG}}"
    fi
    exit 0
  fi
done
exec {sys.executable!s} "$@"
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            fake_curl = fake_bin / "curl"
            fake_curl.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
url="${!#}"
printf '%s\n' "$url" >> "${FAKE_CURL_LOG}"
case "$url" in
  */health)
    printf '%s' '{"status":"ok","model":"gpt-5.5","api_mode":"responses_compat","reasoning_effort":"xhigh","response_storage":"account_default","timeout_sec":600,"max_concurrent_requests":4,"app_server_running":true}'
    ;;
  */perception_queries)
    printf '%s' '{"queries":[{"object_id":"generic_item","text_prompt":"generic item","role":"target"}]}'
    ;;
  *)
    exit 22
    ;;
esac
""",
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)

            provenance_path = root / "runtime_provenance.json"
            env = os.environ.copy()
            env.update(
                {
                    "CONDA_SH": str(conda_sh),
                    "DETACH": "0",
                    "RMBENCH_DETACHED_CHILD": "0",
                    "NUM_WORKERS": "1",
                    "GPU_IDS": "0",
                    "N_PER_WORKER": "1",
                    "EVAL_START_SEEDS": "100001",
                    "TASK_NAME": "synthetic_task",
                    "PERCEPTION_CONDITION": "oracle" if formal else "",
                    "NON_FORMAL_DIAGNOSTIC": "0" if formal else "1",
                    "SKIP_PREFLIGHT": "0" if formal else "1",
                    "PREFLIGHT_ONLY": "0",
                    "RECORD_RUNTIME_PROVENANCE": "1" if formal else "0",
                    "RUNTIME_PROVENANCE_PATH": str(provenance_path),
                    "RMBENCH_OUTPUT_ROOT": str(root),
                    "RMBENCH_ASSETS_ROOT": str(assets_root),
                    "REQUIRE_SAM3_PREFLIGHT": "0",
                    "WORKER_START_DELAY_SEC": "0",
                    "HEARTBEAT_SEC": "0",
                    "LOG_ROOT": str(root / "logs"),
                    "PATH": f"{fake_bin}:{env.get('PATH', '')}",
                    "FAKE_CURL_LOG": str(curl_log),
                    "FAKE_WORKER_ENV_LOG": str(env_log),
                    **(
                        {
                            "FAKE_CAPTURE_RENDERER_ENV": "1",
                            "RMBENCH_RENDER_DEVICE": "cuda:0",
                            "RMBENCH_RENDER_DEVICE_STRICT": "1",
                            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
                            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": "00000000:14:00.0",
                            "RMBENCH_EXPECTED_PHYSICAL_GPU": "4",
                            "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN": "GPU-abcd",
                            "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH": str(
                                root / "renderer_device_provenance.json"
                            ),
                        }
                        if capture_renderer_env
                        else {}
                    ),
                }
            )
            result = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                check=False,
                text=True,
            )
            worker_env = env_log.read_text(encoding="utf-8") if env_log.exists() else ""
            return result, worker_env

    def _run_with_four_fake_workers_isolated(
        self,
    ) -> tuple[subprocess.CompletedProcess[str], list[dict[str, str]], Path]:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="tcm_worker_isolation_launcher_"
        )
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        fake_bin = root / "bin"
        fake_bin.mkdir()
        env_log = root / "worker_env.jsonl"
        conda_sh = root / "conda.sh"
        conda_sh.write_text("conda() { :; }\n", encoding="utf-8")

        fake_python = fake_bin / "python"
        fake_python.write_text(
            f"""#!{sys.executable}
import json
import os
import sys

if "script/eval_policy.py" not in sys.argv:
    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])

def option_value(name):
    index = sys.argv.index(name)
    return sys.argv[index + 1]

payload = {{
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "sam3_service_url_env": os.environ.get("ROBOHARN_EVO_SAM3_SERVICE_URL", ""),
    "sam3_service_url_arg": option_value(
        "--agent.observation_preprocess.segmentation.service_url"
    ),
    "segmentation_artifact_dir": os.environ.get(
        "ROBOHARN_EVO_SEGMENTATION_ARTIFACT_DIR", ""
    ),
    "eval_start_seed": option_value("--eval.start_seed"),
}}
line = json.dumps(payload, sort_keys=True) + "\\n"
fd = os.open(os.environ["FAKE_WORKER_ENV_LOG"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
try:
    os.write(fd, line.encode("utf-8"))
finally:
    os.close(fd)
""",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        artifact_root = root / "segmentation-artifacts"
        env = os.environ.copy()
        env.update(
            {
                "CONDA_SH": str(conda_sh),
                "DETACH": "0",
                "RMBENCH_DETACHED_CHILD": "0",
                "NUM_WORKERS": "4",
                "GPU_IDS": "2,3,4,5",
                "N_PER_WORKER": "1",
                "EVAL_START_SEEDS": "100001,100002,100003,100004",
                "TASK_NAME": "synthetic_task",
                "NON_FORMAL_DIAGNOSTIC": "1",
                "SKIP_PREFLIGHT": "1",
                "PREFLIGHT_ONLY": "0",
                "RECORD_RUNTIME_PROVENANCE": "0",
                "REQUIRE_SAM3_PREFLIGHT": "0",
                "WORKER_START_DELAY_SEC": "0",
                "HEARTBEAT_SEC": "0",
                "LOG_ROOT": str(root / "logs"),
                "SEGMENTATION_ARTIFACT_ROOT": str(artifact_root),
                "SAM3_BASE_PORT": "9311",
                "PATH": f"{fake_bin}:{env.get('PATH', '')}",
                "FAKE_WORKER_ENV_LOG": str(env_log),
            }
        )
        result = subprocess.run(
            ["bash", str(RUNNER)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            check=False,
            text=True,
        )
        payloads = (
            [json.loads(line) for line in env_log.read_text(encoding="utf-8").splitlines()]
            if env_log.exists()
            else []
        )
        return result, payloads, artifact_root

    def test_ctrl_c_only_forwards_sigint_and_never_escalates_worker_signal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_ctrl_c_only_launcher_") as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            ready_path = root / "worker.ready"
            done_path = root / "worker.done"
            signal_path = root / "worker.signals"
            conda_sh = root / "conda.sh"
            conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
            fake_python = fake_bin / "python"
            fake_python.write_text(
                f"""#!{sys.executable}
import os
from pathlib import Path
import signal
import sys
import time

if "script/eval_policy.py" not in sys.argv:
    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])

ready = Path(os.environ["FAKE_WORKER_READY"])
done = Path(os.environ["FAKE_WORKER_DONE"])
signals = Path(os.environ["FAKE_WORKER_SIGNALS"])

def record(name):
    with signals.open("a", encoding="utf-8") as handle:
        handle.write(name + "\\n")

def on_sigint(signum, frame):
    record("SIGINT")
    time.sleep(2.0)
    done.write_text("natural exit after SIGINT\\n", encoding="utf-8")
    raise SystemExit(130)

def on_sigterm(signum, frame):
    record("SIGTERM")
    done.write_text("unexpected SIGTERM\\n", encoding="utf-8")
    raise SystemExit(143)

signal.signal(signal.SIGINT, on_sigint)
signal.signal(signal.SIGTERM, on_sigterm)
ready.write_text(str(os.getpid()) + "\\n", encoding="utf-8")
deadline = time.monotonic() + 8.0
while time.monotonic() < deadline:
    time.sleep(0.1)
done.write_text("natural timeout exit\\n", encoding="utf-8")
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "CONDA_SH": str(conda_sh),
                    "DETACH": "0",
                    "RMBENCH_DETACHED_CHILD": "0",
                    "NUM_WORKERS": "1",
                    "GPU_IDS": "0",
                    "N_PER_WORKER": "1",
                    "EVAL_START_SEEDS": "100001",
                    "TASK_NAME": "synthetic_task",
                    "NON_FORMAL_DIAGNOSTIC": "1",
                    "SKIP_PREFLIGHT": "1",
                    "PREFLIGHT_ONLY": "0",
                    "RECORD_RUNTIME_PROVENANCE": "0",
                    "REQUIRE_SAM3_PREFLIGHT": "0",
                    "WORKER_START_DELAY_SEC": "0",
                    "HEARTBEAT_SEC": "0",
                    "SHUTDOWN_GRACE_SEC": "1",
                    "INTERRUPT_ESCALATION_POLICY": "ctrl_c_only",
                    "LOG_ROOT": str(root / "logs"),
                    "PATH": f"{fake_bin}:{env.get('PATH', '')}",
                    "FAKE_WORKER_READY": str(ready_path),
                    "FAKE_WORKER_DONE": str(done_path),
                    "FAKE_WORKER_SIGNALS": str(signal_path),
                }
            )
            process = subprocess.Popen(
                ["bash", str(RUNNER)],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5.0
                while not ready_path.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
                self.assertTrue(ready_path.exists(), "fake worker did not start")
                os.kill(process.pid, signal.SIGINT)
                stdout, stderr = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 130, stdout + stderr)
                self.assertIn("forwarding SIGINT to owned workers", stderr)
                self.assertIn("no SIGTERM or SIGKILL was sent", stderr)

                deadline = time.monotonic() + 5.0
                while not done_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(done_path.exists(), "fake worker did not exit naturally")
                observed_signals = signal_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(observed_signals, ["SIGINT"])
            finally:
                if process.poll() is None:
                    os.kill(process.pid, signal.SIGINT)
                    process.communicate(timeout=10)

    def test_maps_one_explicit_seed_to_each_worker(self) -> None:
        result = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[protocol-mode] NON-FORMAL DIAGNOSTIC", result.stdout)
        self.assertIn(
            "[agent-plan] base_url=http://127.0.0.1:9104 api_mode=responses_compat "
            "response_storage=account_default max_concurrent_requests=4",
            result.stdout,
        )
        self.assertIn(
            "[budget-plan] max_rounds=10 max_control_turns=64 "
            "max_no_progress_control_turns=10",
            result.stdout,
        )
        self.assertIn(
            "[interrupt-plan] escalation_policy=term_then_kill shutdown_grace_sec=90",
            result.stdout,
        )
        self.assertIn(
            "[instruction-plan] instruction_set=rmbench_original "
            "instruction_type=unseen task=synthetic_task",
            result.stdout,
        )
        for worker, (gpu, seed) in enumerate(zip((2, 3, 4, 5), range(100001, 100005))):
            self.assertIn(f"[seed-plan worker {worker}] gpu={gpu} eval_start_seed={seed}", result.stdout)

    def test_nonformal_explicit_budget_override_is_preserved(self) -> None:
        result = self._run(MAX_ROUNDS="6", MAX_CONTROL_TURNS="0")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[protocol-mode] NON-FORMAL DIAGNOSTIC", result.stdout)
        self.assertIn("[budget-plan] max_rounds=6 max_control_turns=0", result.stdout)

    def test_rejects_unknown_interrupt_escalation_policy(self) -> None:
        result = self._run(INTERRUPT_ESCALATION_POLICY="unsafe-policy")

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be term_then_kill or ctrl_c_only", result.stderr)

    def test_rejects_an_alternate_or_donor_repo_root(self) -> None:
        result = self._run(REPO_ROOT="/tmp/alternate-rmbench")

        self.assertEqual(result.returncode, 2)
        self.assertIn("may not select a donor checkout", result.stderr)

    def test_formal_legacy_gateway_identity_override_is_preserved(self) -> None:
        result, calls = self._run_with_fake_preflight(
            inference_json=(
                '{"queries":[{"object_id":"generic_item",'
                '"text_prompt":"generic item","role":"target"}]}'
            ),
            health_json=(
                '{"status":"ok","model":"gpt-5.5","api_mode":"responses",'
                '"reasoning_effort":"xhigh","response_storage":"disabled",'
                '"timeout_sec":600,"max_concurrent_requests":2}'
            ),
            AGENT_API_BASE_URL="http://127.0.0.1:9103",
            EXPECTED_AGENT_API_MODE="responses",
            EXPECTED_RESPONSE_STORAGE="disabled",
            EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS="2",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("http://127.0.0.1:9103/health", calls)
        self.assertIn("[protocol-mode] FORMAL", result.stdout)
        self.assertIn("protocol_version=3", result.stdout)
        self.assertIn(
            "[agent-plan] base_url=http://127.0.0.1:9103 api_mode=responses "
            "response_storage=disabled max_concurrent_requests=2",
            result.stdout,
        )

    def test_provider_neutral_qwen_chat_identity_requires_no_fake_reasoning(self) -> None:
        health = {
            "status": "ok",
            "backend": "openai",
            "model": "Qwen/Qwen3.5-397B-A17B-FP8",
            "api_mode": "chat",
            "provider": "local-vllm",
            "reasoning_effort": "",
            "thinking_mode": "enabled",
            "response_storage": "disabled",
            "timeout_sec": 600,
            "max_concurrent_requests": 1,
            "max_retries": 0,
            "fallback_enabled": False,
            "app_server_running": None,
            "sampling": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
            },
            "upstream_model_identity": {
                "verified": True,
                "expected_model": "Qwen/Qwen3.5-397B-A17B-FP8",
            },
        }
        expected_subset = {
            "backend": "openai",
            "provider": "local-vllm",
            "thinking_mode": "enabled",
            "max_retries": 0,
            "fallback_enabled": False,
            "sampling": health["sampling"],
            "upstream_model_identity": health["upstream_model_identity"],
        }
        result, calls = self._run_with_fake_preflight(
            inference_json=(
                '{"queries":[{"object_id":"generic_item",'
                '"text_prompt":"generic item","role":"target"}]}'
            ),
            health_json=json.dumps(health),
            AGENT_API_BASE_URL="http://127.0.0.1:9105",
            EXPECTED_AGENT_MODEL="Qwen/Qwen3.5-397B-A17B-FP8",
            EXPECTED_AGENT_API_MODE="chat",
            EXPECTED_REASONING_EFFORT="",
            EXPECTED_RESPONSE_STORAGE="disabled",
            EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS="1",
            EXPECTED_AGENT_IDENTITY_JSON=json.dumps(expected_subset),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("http://127.0.0.1:9105/health", calls)
        self.assertIn("reasoning_effort=not_applicable", result.stdout)

    def test_formal_allows_explicit_environment_step_limit_cap(self) -> None:
        result, calls = self._run_with_fake_preflight(
            inference_json=(
                '{"queries":[{"object_id":"generic_item",'
                '"text_prompt":"generic item","role":"target"}]}'
            ),
            extra_args=("--eval.step_limit", "150"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[protocol-mode] FORMAL", result.stdout)
        self.assertIn("http://127.0.0.1:9104/perception_queries", calls)

    def test_rejects_duplicate_explicit_seeds(self) -> None:
        result = self._run(EVAL_START_SEEDS="100001,100002,100002,100004")

        self.assertEqual(result.returncode, 2)
        self.assertIn("duplicate canonical seed", result.stderr)
        self.assertIn("canonical 100002", result.stderr)

    def test_rejects_duplicate_seeds_after_canonical_integer_normalization(self) -> None:
        result = self._run(EVAL_START_SEEDS="0100000,100000,100001,100002")

        self.assertEqual(result.returncode, 2)
        self.assertIn("100000 conflicts with 0100000", result.stderr)
        self.assertIn("canonical 100000", result.stderr)

    def test_emits_canonical_explicit_seed_values(self) -> None:
        result = self._run(EVAL_START_SEEDS="0100001,00100002,000100003,0000100004")

        self.assertEqual(result.returncode, 0, result.stderr)
        for worker, seed in enumerate(range(100001, 100005)):
            self.assertIn(f"[seed-plan worker {worker}]", result.stdout)
            self.assertIn(f"eval_start_seed={seed}", result.stdout)

    def test_rejects_multiple_episodes_per_explicit_seed_worker(self) -> None:
        result = self._run(N_PER_WORKER="2")

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires N_PER_WORKER=1", result.stderr)

    def test_requires_explicit_task_name(self) -> None:
        result = self._run(TASK_NAME="")

        self.assertEqual(result.returncode, 2)
        self.assertIn("TASK_NAME must be supplied explicitly", result.stderr)

    def test_rejects_invalid_instruction_set_identifier(self) -> None:
        result = self._run(INSTRUCTION_SET="../contract_clarified_v1")

        self.assertEqual(result.returncode, 2)
        self.assertIn("INSTRUCTION_SET must be", result.stderr)

    def test_emits_explicit_custom_instruction_set(self) -> None:
        result = self._run(INSTRUCTION_SET="contract_clarified_v1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "[instruction-plan] instruction_set=contract_clarified_v1",
            result.stdout,
        )

    def test_requires_explicit_environment_seeds(self) -> None:
        result = self._run(EVAL_START_SEEDS="")

        self.assertEqual(result.returncode, 2)
        self.assertIn("EVAL_START_SEEDS must be supplied explicitly", result.stderr)

    def test_default_preflight_requires_real_structured_inference(self) -> None:
        result, calls = self._run_with_fake_preflight(
            inference_json=(
                '{"queries":[{"object_id":"generic_item",'
                '"text_prompt":"generic item","role":"target"}]}'
            )
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[protocol-mode] FORMAL", result.stdout)
        self.assertIn("http://127.0.0.1:9104/health", calls)
        self.assertIn("http://127.0.0.1:9104/perception_queries", calls)
        self.assertIn("[agent-inference-preflight]", result.stdout)
        self.assertIn("outer_calls=1", result.stdout)
        self.assertIn("latency_ms=", result.stdout)
        self.assertIn("queries=1", result.stdout)
        self.assertIn('"runtime_provenance":', result.stdout)
        self.assertIn("Preflight complete; no eval workers launched.", result.stdout)

    def test_formal_health_rejects_nonhealthy_status_before_real_inference(self) -> None:
        result, calls = self._run_with_fake_preflight(
            inference_json=(
                '{"queries":[{"object_id":"generic_item",'
                '"text_prompt":"generic item","role":"target"}]}'
            ),
            health_json=(
                '{"status":"degraded","model":"gpt-5.5",'
                '"api_mode":"responses_compat","reasoning_effort":"xhigh",'
                '"response_storage":"account_default","timeout_sec":600,'
                '"max_concurrent_requests":4}'
            ),
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("status='degraded', expected ok or healthy", result.stderr)
        self.assertIn("http://127.0.0.1:9104/health", calls)
        self.assertNotIn("http://127.0.0.1:9104/perception_queries", calls)

    def test_formal_health_rejects_stopped_app_server_when_field_is_present(self) -> None:
        result, calls = self._run_with_fake_preflight(
            inference_json=(
                '{"queries":[{"object_id":"generic_item",'
                '"text_prompt":"generic item","role":"target"}]}'
            ),
            health_json=(
                '{"status":"ok","model":"gpt-5.5",'
                '"api_mode":"responses_compat","reasoning_effort":"xhigh",'
                '"response_storage":"account_default","timeout_sec":600,'
                '"max_concurrent_requests":4,"app_server_running":false}'
            ),
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("app_server_running=False, expected true when present", result.stderr)
        self.assertNotIn("http://127.0.0.1:9104/perception_queries", calls)

    def test_preflight_rejects_empty_inference_result(self) -> None:
        result, calls = self._run_with_fake_preflight(inference_json='{"queries":[]}')

        self.assertEqual(result.returncode, 1)
        self.assertIn("http://127.0.0.1:9104/perception_queries", calls)
        self.assertIn("did not return a non-empty queries array", result.stderr)
        self.assertIn("stage=response_validation", result.stderr)

    def test_preflight_rejects_unstructured_inference_result(self) -> None:
        result, _ = self._run_with_fake_preflight(
            inference_json='{"queries":[{"reason":"missing identity"}]}'
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("returned no structured object query", result.stderr)

    def test_preflight_rejects_inference_transport_failure(self) -> None:
        result, calls = self._run_with_fake_preflight(
            inference_json="",
            inference_exit_code=28,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("http://127.0.0.1:9104/perception_queries", calls)
        self.assertIn("stage=request", result.stderr)
        self.assertIn("curl_exit=28", result.stderr)
        self.assertIn("could not complete a real structured inference", result.stderr)

    def test_formal_protocol_rejects_budget_and_gate_opt_outs(self) -> None:
        good_inference = (
            '{"queries":[{"object_id":"generic_item",'
            '"text_prompt":"generic item","role":"target"}]}'
        )
        cases = (
            ({"MAX_ROUNDS": "6"}, "requires MAX_ROUNDS=10, MAX_CONTROL_TURNS=64"),
            ({"MAX_CONTROL_TURNS": "9"}, "requires MAX_ROUNDS=10, MAX_CONTROL_TURNS=64"),
            (
                {"MAX_NO_PROGRESS_CONTROL_TURNS": "5"},
                "MAX_NO_PROGRESS_CONTROL_TURNS=10",
            ),
            ({"SKIP_PREFLIGHT": "1"}, "forbids SKIP_PREFLIGHT=1"),
            ({"SKIP_AGENT_IDENTITY_CHECK": "1"}, "forbids SKIP_AGENT_IDENTITY_CHECK=1"),
            (
                {"REQUIRE_AGENT_INFERENCE_PREFLIGHT": "0"},
                "requires REQUIRE_AGENT_INFERENCE_PREFLIGHT=1",
            ),
            (
                {"RECORD_RUNTIME_PROVENANCE": "0"},
                "requires RECORD_RUNTIME_PROVENANCE=1",
            ),
            (
                {"REQUIRE_EXPLICIT_EVAL_START_SEEDS": "0"},
                "requires REQUIRE_EXPLICIT_EVAL_START_SEEDS=1",
            ),
        )
        for overrides, expected_error in cases:
            with self.subTest(overrides=overrides):
                result, calls = self._run_with_fake_preflight(
                    inference_json=good_inference,
                    **overrides,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)
                self.assertEqual(calls, "")

    def test_formal_protocol_requires_canonical_policy_name(self) -> None:
        result, calls = self._run_with_fake_preflight(
            inference_json=(
                '{"queries":[{"object_id":"generic_item",'
                '"text_prompt":"generic item","role":"target"}]}'
            ),
            POLICY_NAME="policy.other.deploy_policy",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "Formal protocol requires POLICY_NAME=policy.roboharn_evo.deploy_policy",
            result.stderr,
        )
        self.assertEqual(calls, "")

    def test_formal_protocol_requires_bound_perception_condition_and_sam3_gate(self) -> None:
        good_inference = (
            '{"queries":[{"object_id":"generic_item",'
            '"text_prompt":"generic item","role":"target"}]}'
        )
        cases = (
            (
                {"PERCEPTION_CONDITION": ""},
                "requires PERCEPTION_CONDITION=oracle or no_oracle",
            ),
            (
                {"PERCEPTION_CONDITION": "other"},
                "requires PERCEPTION_CONDITION=oracle or no_oracle",
            ),
            (
                {"PERCEPTION_CONDITION": "no_oracle", "REQUIRE_SAM3_PREFLIGHT": "0"},
                "Formal no_oracle protocol requires REQUIRE_SAM3_PREFLIGHT=1",
            ),
            (
                {"PERCEPTION_CONDITION": "oracle", "REQUIRE_SAM3_PREFLIGHT": "1"},
                "Formal oracle protocol requires REQUIRE_SAM3_PREFLIGHT=0",
            ),
        )
        for overrides, expected_error in cases:
            with self.subTest(overrides=overrides):
                result, calls = self._run_with_fake_preflight(
                    inference_json=good_inference,
                    **overrides,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)
                self.assertEqual(calls, "")

    def test_formal_nooracle_condition_requires_and_executes_sam3_preflight(self) -> None:
        result, calls = self._run_with_fake_preflight(
            inference_json=(
                '{"queries":[{"object_id":"generic_item",'
                '"text_prompt":"generic item","role":"target"}]}'
            ),
            PERCEPTION_CONDITION="no_oracle",
            REQUIRE_SAM3_PREFLIGHT="1",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[protocol-mode] FORMAL perception_condition=no_oracle", result.stdout)
        self.assertIn("http://127.0.0.1:9301/health", calls)

    def test_nonformal_opt_out_is_explicitly_marked(self) -> None:
        result = self._run(
            MAX_ROUNDS="3",
            MAX_CONTROL_TURNS="0",
            REQUIRE_AGENT_INFERENCE_PREFLIGHT="0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[protocol-mode] NON-FORMAL DIAGNOSTIC", result.stdout)
        self.assertIn("opt_out=NON_FORMAL_DIAGNOSTIC=1", result.stdout)

    def test_rejects_ambiguous_nonformal_opt_out_value(self) -> None:
        result = self._run(NON_FORMAL_DIAGNOSTIC="yes")

        self.assertEqual(result.returncode, 2)
        self.assertIn("NON_FORMAL_DIAGNOSTIC must be 0 or 1", result.stderr)

    def test_rejects_empty_expected_identity_fields_in_all_modes(self) -> None:
        for field_name in (
            "EXPECTED_AGENT_MODEL",
            "EXPECTED_AGENT_API_MODE",
            "EXPECTED_REASONING_EFFORT",
            "EXPECTED_RESPONSE_STORAGE",
            "EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS",
        ):
            with self.subTest(field_name=field_name):
                result = self._run(**{field_name: ""})
                self.assertEqual(result.returncode, 2)
                self.assertIn(f"{field_name} must be non-empty", result.stderr)

    def test_formal_rejects_protected_extra_arg_override_spellings(self) -> None:
        good_inference = (
            '{"queries":[{"object_id":"generic_item",'
            '"text_prompt":"generic item","role":"target"}]}'
        )
        cases = (
            (("---config", "other.yml"), "config"),
            (("--ckpt_setting=other",), "ckpt_setting"),
            (("--instruction_set", "other_set"), "instruction_set"),
            (("----instruction_type", "seen"), "instruction_type"),
            (("---task_name", "other_task"), "task_name"),
            (("-----task_config=other_config",), "task_config"),
            (("--eval.start_seed=999999",), "eval.start_seed"),
            (("--eval", "{'start_seed': 999999}"), "eval"),
            (("----eval.test_num", "2"), "eval.test_num"),
            (("--policy_name=other.policy",), "policy_name"),
            (("seed=7",), "seed"),
            (
                ("--planner.agent_api.server_url", "http://127.0.0.1:9999/plan"),
                "planner.agent_api.server_url",
            ),
            (
                ("---ood.agent_api.server_url=http://127.0.0.1:9999/ood",),
                "ood.agent_api.server_url",
            ),
            (
                ("--recovery.agent_api.server_url", "http://127.0.0.1:9999/recover"),
                "recovery.agent_api.server_url",
            ),
            (
                ("---agent.pure_tool_control.enabled", "False"),
                "agent.pure_tool_control.enabled",
            ),
            (("--agent", "{'pure_tool_control': {'enabled': False}}"), "agent"),
            (
                ("------agent.pure_tool_control.max_rounds=99",),
                "agent.pure_tool_control.max_rounds",
            ),
            (
                ("+agent.pure_tool_control.max_control_turns", "99"),
                "agent.pure_tool_control.max_control_turns",
            ),
            (
                ("--agent.observation_preprocess.enabled", "False"),
                "agent.observation_preprocess.enabled",
            ),
            (
                (
                    "--agent.observation_preprocess.query_url",
                    "http://127.0.0.1:9999/perception_queries",
                ),
                "agent.observation_preprocess.query_url",
            ),
            (
                (
                    "---agent.observation_preprocess.normalization_url=http://127.0.0.1:9999/normalize",
                ),
                "agent.observation_preprocess.normalization_url",
            ),
            (
                (
                    "--agent.observation_preprocess.segmentation.service_url",
                    "http://127.0.0.1:9999",
                ),
                "agent.observation_preprocess.segmentation.service_url",
            ),
            (
                ("--agent.observation_preprocess.oracle_objects", "{'enabled': False}"),
                "agent.observation_preprocess.oracle_objects",
            ),
        )
        for extra_args, expected_key in cases:
            with self.subTest(extra_args=extra_args):
                result, calls = self._run_with_fake_preflight(
                    inference_json=good_inference,
                    extra_args=extra_args,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    f"Formal protocol forbids EXTRA_ARGS override of {expected_key}",
                    result.stderr,
                )
                self.assertEqual(calls, "")

    def test_formal_allows_condition_specific_oracle_override(self) -> None:
        result, _ = self._run_with_fake_preflight(
            inference_json=(
                '{"queries":[{"object_id":"generic_item",'
                '"text_prompt":"generic item","role":"target"}]}'
            ),
            extra_args=(
                "--agent.observation_preprocess.oracle_objects.enabled",
                "True",
                "--agent.observation_preprocess.oracle_objects.include_all",
                "True",
                "--agent.observation_preprocess.oracle_objects.max_objects",
                "12",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[protocol-mode] FORMAL", result.stdout)

    def test_formal_allows_declared_reobserve_ablation(self) -> None:
        result, _ = self._run_with_fake_preflight(
            inference_json=(
                '{"queries":[{"object_id":"generic_item",'
                '"text_prompt":"generic item","role":"target"}]}'
            ),
            extra_args=(
                "--agent.recovery.enable_reobserve",
                "False",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[protocol-mode] FORMAL", result.stdout)

    def test_worker_receives_derived_formal_protocol_and_provenance_env(self) -> None:
        result, worker_env = self._run_with_fake_worker(formal=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("formal=1\n", worker_env)
        self.assertIn("protocol_version=3\n", worker_env)
        self.assertRegex(worker_env, r"provenance=.+/runtime_provenance\.json\n")

    def test_nonformal_worker_receives_zero_protocol_and_empty_provenance_path(self) -> None:
        result, worker_env = self._run_with_fake_worker(formal=False)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(worker_env, "formal=0\nprotocol_version=0\nprovenance=\n")
        self.assertIn("[protocol-mode] NON-FORMAL DIAGNOSTIC", result.stdout)

    def test_worker_receives_explicit_renderer_device_contract(self) -> None:
        result, worker_env = self._run_with_fake_worker(
            formal=False,
            capture_renderer_env=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("render_device=cuda:0\n", worker_env)
        self.assertIn("render_strict=1\n", worker_env)
        self.assertIn("render_cuda_id=0\n", worker_env)
        self.assertIn("render_pci=00000000:14:00.0\n", worker_env)
        self.assertIn("render_physical_gpu=4\n", worker_env)
        self.assertIn("render_cuda_visible_devices=GPU-abcd\n", worker_env)
        self.assertIn("render_cuda_selector=GPU-abcd\n", worker_env)
        self.assertRegex(
            worker_env,
            r"render_provenance=.+/renderer_device_provenance\.json\n",
        )

    def test_four_workers_receive_unique_sam3_urls_and_artifact_directories(self) -> None:
        result, payloads, artifact_root = self._run_with_four_fake_workers_isolated()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(payloads), 4)
        payloads.sort(key=lambda item: int(item["eval_start_seed"]))
        expected_urls = [f"http://127.0.0.1:{port}" for port in range(9311, 9315)]
        observed_urls = [item["sam3_service_url_env"] for item in payloads]
        self.assertEqual(observed_urls, expected_urls)
        self.assertEqual(
            [item["sam3_service_url_arg"] for item in payloads], expected_urls
        )
        self.assertEqual(
            [item["cuda_visible_devices"] for item in payloads],
            ["2", "3", "4", "5"],
        )

        artifact_dirs = [Path(item["segmentation_artifact_dir"]) for item in payloads]
        self.assertEqual(len(set(artifact_dirs)), 4)
        for worker, (artifact_dir, seed) in enumerate(
            zip(artifact_dirs, range(100001, 100005), strict=True)
        ):
            self.assertTrue(artifact_dir.is_dir())
            self.assertEqual(artifact_dir.parent, artifact_root)
            self.assertIn(f"worker_{worker}_", artifact_dir.name)
            self.assertIn(f"_e{seed}", artifact_dir.name)

    def test_segmentation_artifact_root_must_be_absolute(self) -> None:
        result = self._run(SEGMENTATION_ARTIFACT_ROOT="relative/shared-artifacts")

        self.assertEqual(result.returncode, 2)
        self.assertIn("SEGMENTATION_ARTIFACT_ROOT must be an absolute path", result.stderr)

    def test_detached_launchers_preserve_protocol_mode_inputs(self) -> None:
        runner_source = RUNNER.read_text(encoding="utf-8")
        wrapper_source = WRAPPER.read_text(encoding="utf-8")

        self.assertIn(
            "NON_FORMAL_DIAGNOSTIC ROBOHARN_EVO_FORMAL_PROTOCOL ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH",
            runner_source,
        )
        self.assertIn("PERCEPTION_CONDITION", runner_source)
        self.assertIn('--repo-root "${ROBOHARN_EVO_PROJECT_ROOT}"', runner_source)
        self.assertNotIn('--repo-root "${REPO_ROOT}"', runner_source)
        self.assertIn('INSTRUCTION_SET="${INSTRUCTION_SET:-rmbench_original}"', runner_source)
        self.assertIn('--instruction_set "${INSTRUCTION_SET}"', runner_source)
        self.assertIn(
            'ROBOHARN_EVO_FORMAL_PROTOCOL="${ROBOHARN_EVO_FORMAL_PROTOCOL}"',
            runner_source,
        )
        self.assertIn(
            'ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION="${ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION}"',
            runner_source,
        )
        self.assertIn(
            'ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH="${ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH}"',
            runner_source,
        )
        for name in (
            "RMBENCH_RENDER_DEVICE",
            "RMBENCH_RENDER_DEVICE_STRICT",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID",
            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID",
            "RMBENCH_EXPECTED_PHYSICAL_GPU",
            "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH",
            "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN",
        ):
            self.assertIn(name, runner_source)
        self.assertLess(
            runner_source.index('"${EXTRA_ARGS[@]}" \\'),
            runner_source.index('"${FORMAL_PERCEPTION_OVERRIDE_ARGS[@]}" \\'),
        )
        self.assertIn("NON_FORMAL_DIAGNOSTIC", wrapper_source)
        self.assertIn("INSTRUCTION_SET", wrapper_source)
        self.assertIn('PERCEPTION_CONDITION="${condition}"', wrapper_source)
        self.assertIn("PERCEPTION_CONDITION", wrapper_source)

    def test_synthetic_inference_preflight_omits_oracle_object_payload(self) -> None:
        runner_source = RUNNER.read_text(encoding="utf-8")

        self.assertNotIn('"oracle_objects": []', runner_source)


if __name__ == "__main__":
    unittest.main()
