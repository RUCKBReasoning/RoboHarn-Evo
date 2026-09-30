from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = (
    REPO_ROOT
    / "policy"
    / "roboharn_evo"
    / "scripts"
    / "run_gpt55_six_tasks_sequential.sh"
)


def test_four_task_two_seed_dry_run_is_strictly_ordered() -> None:
    env = os.environ.copy()
    env.update(
        {
            "TASKS_CSV": (
                "rearrange_blocks,swap_blocks,swap_T,battery_try"
            ),
            "EVAL_START_SEEDS_CSV": "0,1",
            "GPU_ID": "4",
            "RUN_LABEL": "test_seq4x2",
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "4 tasks x 2 seeds = 8 runs" in result.stdout
    rows = [
        line for line in result.stdout.splitlines() if line.startswith("[")
    ]
    assert len(rows) == 8
    expected = [
        ("rearrange_blocks", 0),
        ("rearrange_blocks", 1),
        ("swap_blocks", 0),
        ("swap_blocks", 1),
        ("swap_T", 0),
        ("swap_T", 1),
        ("battery_try", 0),
        ("battery_try", 1),
    ]
    for row, (task, seed) in zip(rows, expected, strict=True):
        assert f"task={task}" in row
        assert f"eval_start_seed={seed}" in row


def test_sequential_launcher_rejects_an_alternate_or_donor_repo_root(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "REPO_ROOT": str(tmp_path / "alternate-rmbench"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "may not select a donor checkout" in result.stderr


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _fake_sequential_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    # The adapted launcher must reject any alternate/donor REPO_ROOT.  Tests
    # therefore use the copied benchmark itself read-only and replace only the
    # runner/checker processes with temporary fixtures.
    repo = REPO_ROOT

    runner = tmp_path / "fake_runner.py"
    _write_executable(
        runner,
        """#!/usr/bin/env bash
set -u
state="${FAKE_STATE_DIR}"
seed="${EVAL_START_SEEDS}"
eval_result_root=""
while (( $# > 0 )); do
  if [[ "$1" == "--output_root" && $# -ge 2 ]]; then
    eval_result_root="$2"
    shift 2
    continue
  fi
  shift
done
if [[ -z "${eval_result_root}" ]]; then
  echo "fake runner did not receive --output_root" >&2
  exit 2
fi
mkdir -p "${state}"
	printf 'started\\n' > "${state}/started_${seed}"
	if [[ "${FAKE_CAPTURE_RENDERER_ENV:-0}" == "1" ]]; then
	  {
	    printf 'device=%s\\n' "${RMBENCH_RENDER_DEVICE-}"
	    printf 'strict=%s\\n' "${RMBENCH_RENDER_DEVICE_STRICT-}"
	    printf 'cuda_id=%s\\n' "${RMBENCH_EXPECTED_RENDER_CUDA_ID-}"
	    printf 'pci=%s\\n' "${RMBENCH_EXPECTED_RENDER_PCI_BUS_ID-}"
	    printf 'physical_gpu=%s\\n' "${RMBENCH_EXPECTED_PHYSICAL_GPU-}"
	    printf 'cuda_visible_device_token=%s\\n' "${RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN-}"
	    printf 'provenance=%s\\n' "${RMBENCH_RENDER_DEVICE_PROVENANCE_PATH-}"
	  } > "${state}/renderer_env_${seed}"
	fi
	worker_pid=""
	host_identity() {
	  local stat_line stat_tail
	  local -a fields
	  IFS= read -r stat_line < /proc/self/stat
	  stat_tail="${stat_line##*) }"
	  read -r -a fields <<< "${stat_tail}"
	  printf '%s\\t%s\\t%s' "${stat_line%% *}" "${fields[1]}" "${fields[19]}"
	}
	write_marker() {
	  local path="$1"
	  local status="$2"
	  local reason="$3"
	  local identity runner_host runner_ppid runner_start
	  [[ -n "${path}" ]] || return 0
	  identity="$(host_identity)"
	  IFS=$'\\t' read -r self_host runner_host runner_start <<< "${identity}"
	  IFS= read -r stat_line < "/proc/${runner_host}/stat"
	  stat_tail="${stat_line##*) }"
	  read -r -a fields <<< "${stat_tail}"
	  runner_ppid="${fields[1]}"
	  runner_start="${fields[19]}"
	  mkdir -p "$(dirname "${path}")"
	  {
	    printf 'schema_version\\t1\\n'
	    printf 'batch_token\\t%s\\n' "${ROBOHARN_EVO_CONTROL_BATCH_TOKEN}"
	    printf 'run_token\\t%s\\n' "${ROBOHARN_EVO_CONTROL_RUN_TOKEN}"
	    printf 'runner_pid\\t%s\\n' "${runner_host}"
	    printf 'status\\t%s\\n' "${status}"
	    printf 'reason\\t%s\\n' "${reason}"
	  } > "${path}"
	}
	publish_ready() {
	  local identity runner_host runner_ppid runner_start
	  local worker_host worker_stat worker_tail
	  local -a worker_fields
	  identity="$(host_identity)"
	  IFS=$'\\t' read -r self_host runner_host runner_start <<< "${identity}"
	  IFS= read -r stat_line < "/proc/${runner_host}/stat"
	  stat_tail="${stat_line##*) }"
	  read -r -a fields <<< "${stat_tail}"
	  runner_ppid="${fields[1]}"
	  runner_start="${fields[19]}"
	  local local_worker="${worker_pid}"
	  local -a host_children
	  IFS= read -r stat_line < /proc/self/stat
	  local self_host="${stat_line%% *}"
	  read -r -a host_children < "/proc/${self_host}/task/${self_host}/children"
	  if (( local_worker <= ${#host_children[@]} )); then
	    worker_host="${host_children[local_worker - 1]}"
	  else
	    worker_host="${host_children[0]}"
	  fi
	  IFS= read -r worker_stat < "/proc/${worker_host}/stat"
	  worker_tail="${worker_stat##*) }"
	  read -r -a worker_fields <<< "${worker_tail}"
	  mkdir -p "$(dirname "${ROBOHARN_EVO_SKIP_READY_FILE}")"
	  {
	    printf 'schema_version\\t1\\n'
	    printf 'batch_token\\t%s\\n' "${ROBOHARN_EVO_CONTROL_BATCH_TOKEN}"
	    printf 'run_token\\t%s\\n' "${ROBOHARN_EVO_CONTROL_RUN_TOKEN}"
	    printf 'boot_id\\t%s\\n' "${ROBOHARN_EVO_CONTROL_BOOT_ID}"
	    printf 'owner_pid\\t%s\\n' "${ROBOHARN_EVO_CONTROL_OWNER_PID}"
	    printf 'owner_start_ticks\\t%s\\n' "${ROBOHARN_EVO_CONTROL_OWNER_START_TICKS}"
	    printf 'runner_pid\\t%s\\n' "${runner_host}"
	    printf 'runner_start_ticks\\t%s\\n' "${runner_start}"
	    printf 'runner_ppid\\t%s\\n' "${runner_ppid}"
	    printf 'interrupt_policy\\tctrl_c_only\\n'
	    printf 'worker_count\\t1\\n'
	    printf 'worker\\t%s\\t%s\\t%s\\n' \
	      "${worker_host}" "${worker_fields[19]}" "${worker_fields[1]}"
	  } > "${ROBOHARN_EVO_SKIP_READY_FILE}"
	}
	stop_runner() {
	  local exit_code="${1:-130}"
	  if [[ -n "${worker_pid}" ]]; then
	    kill -INT "${worker_pid}" 2>/dev/null || true
	    if [[ "${FAKE_IGNORE_INT:-0}" == "1" ]]; then
	      sleep 0.2
	      if kill -0 "${worker_pid}" 2>/dev/null; then
	        write_marker "${ROBOHARN_EVO_SKIP_CLEANUP_FAILED_FILE}" cleanup_failed worker_ignored_sigint
	        printf '%s\\n' "${worker_pid}" > "${state}/orphan_local_pid_${seed}"
	        exit 75
	      fi
	    fi
	    wait "${worker_pid}" 2>/dev/null || true
	  fi
	  if [[ -f "${ROBOHARN_EVO_SKIP_ACCEPTED_FILE:-}" ]]; then
	    write_marker "${ROBOHARN_EVO_SKIP_CLEANUP_COMPLETE_FILE}" cleanup_complete operator_skip_current
	  fi
	  printf '%s\\n' "${exit_code}" > "${state}/terminated_${seed}"
  exit "${exit_code}"
}
trap 'stop_runner 130' INT
trap 'stop_runner 143' TERM
if [[ "${seed}" == "0" ]]; then
  /usr/bin/env --default-signal=INT python3 -c '
import os, signal, sys, time
from pathlib import Path
state = Path(os.environ["FAKE_STATE_DIR"])
seed = os.environ["EVAL_START_SEEDS"]
def stop(_signum, _frame):
    (state / f"worker_interrupted_{seed}").write_text("interrupted\\n")
    raise SystemExit(130)
if os.environ.get("FAKE_IGNORE_INT") == "1":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
else:
    signal.signal(signal.SIGINT, stop)
while True:
    time.sleep(0.05)
	' &
	  worker_pid=$!
	  publish_ready
	  wait "${worker_pid}"
fi
result_dir="${eval_result_root}/${TASK_NAME}/${POLICY_NAME}/${TASK_CONFIG}/${BASE_CKPT}_w0_g${GPU_IDS}_s0_e${seed}"
mkdir -p "${result_dir}"
printf 'completed\\n' > "${state}/completed_${seed}"
""",
    )

    checker = tmp_path / "fake_checker.py"
    _write_executable(
        checker,
        """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

state = Path(os.environ["FAKE_STATE_DIR"])
(state / "checker_called").write_text("called\\n")
raise SystemExit(0)
""",
    )
    return repo, runner, checker


def _safe_runtime_environment(
    repo: Path,
    batch_name: str,
    tmp_path: Path,
) -> dict[str, str]:
    project_root = repo.parents[1]
    output_root = project_root / "eval_result" / "rmbench"
    assets_root = tmp_path / "readonly-assets"
    (assets_root / "embodiments").mkdir(parents=True, exist_ok=True)
    (assets_root / "objects").mkdir(parents=True, exist_ok=True)
    return {
        "ROBOHARN_EVO_PROJECT_ROOT": str(project_root),
        "RMBENCH_OUTPUT_ROOT": str(output_root),
        "RMBENCH_ASSETS_ROOT": str(assets_root),
        "BATCH_LOG_ROOT": str(
            output_root
            / "copied_test_runtime"
            / tmp_path.parent.name
            / tmp_path.name
            / batch_name
        ),
    }


def _wait_for(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def test_skip_current_marks_skipped_and_continues_with_strict_errors_enabled(
    tmp_path: Path,
) -> None:
    repo, runner, checker = _fake_sequential_repo(tmp_path)
    state = tmp_path / "state"
    runtime_env = _safe_runtime_environment(repo, "batch", tmp_path)
    batch = Path(runtime_env["BATCH_LOG_ROOT"])
    env = os.environ.copy()
    env.update(
        {
            **runtime_env,
            "REPO_ROOT": str(repo),
            "RUNNER": str(runner),
            "CHECKER": str(checker),
            "CHECKER_PYTHON": sys.executable,
            "TASKS_CSV": "rearrange_blocks",
            "EVAL_START_SEEDS_CSV": "0,1",
            "BATCH_STAMP": "skip_test",
            "RUN_LABEL": "skip_test",
            "BATCH_LOG_ROOT": str(batch),
            "FAKE_STATE_DIR": str(state),
            "CONTINUE_ON_ERROR": "0",
            "PERCEPTION_CONDITION": "oracle",
        }
    )

    batch_proc = subprocess.Popen(
        ["bash", str(LAUNCHER), "--foreground"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for(state / "started_0")
        current = batch / "control" / "current.tsv"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if current.exists() and current.read_text().rstrip().endswith("\tready"):
                break
            time.sleep(0.05)
        else:
            raise AssertionError("current rollout was not published as ready")
        control = subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                "--skip-current",
                "--batch-log-root",
                str(batch),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert control.returncode == 0, control.stderr
        assert "Skip accepted" in control.stdout
        _wait_for(state / "terminated_0")
        _wait_for(state / "worker_interrupted_0")
        _wait_for(state / "started_1")
        stdout, _ = batch_proc.communicate(timeout=10)
    finally:
        if batch_proc.poll() is None:
            batch_proc.send_signal(signal.SIGINT)
            batch_proc.wait(timeout=10)

    assert batch_proc.returncode == 0, stdout
    rows = (batch / "batch_status.tsv").read_text().splitlines()
    assert len(rows) == 3
    assert "\t0\t130\t-1\tskipped\t" in rows[1]
    assert "operator_skip_current" in rows[1]
    assert "\t1\t0\t0\tcomplete\t" in rows[2]
    assert (state / "checker_called").exists()
    assert "Skipped runs: 1" in stdout

    outcome = json.loads(rows[1].split("\t", 8)[8])
    token = outcome["token"]
    repeated = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--skip-current",
            "--batch-log-root",
            str(batch),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode == 4
    assert token not in repeated.stdout


def test_sequential_launcher_forwards_renderer_contract_with_per_run_provenance(
    tmp_path: Path,
) -> None:
    repo, runner, checker = _fake_sequential_repo(tmp_path)
    state = tmp_path / "state"
    runtime_env = _safe_runtime_environment(repo, "batch", tmp_path)
    batch = Path(runtime_env["BATCH_LOG_ROOT"])
    env = os.environ.copy()
    env.update(
        {
            **runtime_env,
            "REPO_ROOT": str(repo),
            "RUNNER": str(runner),
            "CHECKER": str(checker),
            "CHECKER_PYTHON": sys.executable,
            "TASKS_CSV": "rearrange_blocks",
            "EVAL_START_SEEDS_CSV": "1",
            "BATCH_STAMP": "renderer_env_test",
            "RUN_LABEL": "renderer_env_test",
            "BATCH_LOG_ROOT": str(batch),
            "FAKE_STATE_DIR": str(state),
            "FAKE_CAPTURE_RENDERER_ENV": "1",
            "PERCEPTION_CONDITION": "oracle",
            "RMBENCH_RENDER_DEVICE": "cuda:0",
            "RMBENCH_RENDER_DEVICE_STRICT": "1",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
            "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": "00000000:14:00.0",
            "RMBENCH_EXPECTED_PHYSICAL_GPU": "4",
            "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN": "GPU-abcd",
            "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH": str(
                batch / "placeholder.json"
            ),
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--foreground"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    captured = (state / "renderer_env_1").read_text(encoding="utf-8")
    assert "device=cuda:0\n" in captured
    assert "strict=1\n" in captured
    assert "cuda_id=0\n" in captured
    assert "pci=00000000:14:00.0\n" in captured
    assert "physical_gpu=4\n" in captured
    assert "cuda_visible_device_token=GPU-abcd\n" in captured
    assert (
        f"provenance={batch}/rearrange_blocks/episode_000001/"
        "renderer_device_provenance.json\n"
    ) in captured


def test_ctrl_c_stops_batch_without_starting_next_rollout(tmp_path: Path) -> None:
    repo, runner, checker = _fake_sequential_repo(tmp_path)
    state = tmp_path / "state"
    runtime_env = _safe_runtime_environment(repo, "batch", tmp_path)
    batch = Path(runtime_env["BATCH_LOG_ROOT"])
    env = os.environ.copy()
    env.update(
        {
            **runtime_env,
            "REPO_ROOT": str(repo),
            "RUNNER": str(runner),
            "CHECKER": str(checker),
            "CHECKER_PYTHON": sys.executable,
            "TASKS_CSV": "rearrange_blocks",
            "EVAL_START_SEEDS_CSV": "0,1",
            "BATCH_STAMP": "interrupt_test",
            "RUN_LABEL": "interrupt_test",
            "BATCH_LOG_ROOT": str(batch),
            "FAKE_STATE_DIR": str(state),
            "CONTINUE_ON_ERROR": "0",
            "PERCEPTION_CONDITION": "oracle",
        }
    )

    batch_proc = subprocess.Popen(
        ["bash", str(LAUNCHER), "--foreground"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _wait_for(state / "started_0")
    batch_proc.send_signal(signal.SIGINT)
    stdout, _ = batch_proc.communicate(timeout=10)

    assert batch_proc.returncode == 130, stdout
    assert (state / "terminated_0").exists()
    assert not (state / "started_1").exists()
    assert "no later rollout was started" in stdout


def test_skip_current_without_an_active_supported_batch_is_safe(tmp_path: Path) -> None:
    control = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--skip-current",
            "--batch-log-root",
            str(tmp_path / "missing_batch"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert control.returncode == 4
    assert "No live compatible sequential batch" in control.stderr


def test_skip_current_is_rejected_before_eval_worker_is_ready(tmp_path: Path) -> None:
    repo, runner, checker = _fake_sequential_repo(tmp_path)
    state = tmp_path / "state"
    runtime_env = _safe_runtime_environment(repo, "batch", tmp_path)
    batch = Path(runtime_env["BATCH_LOG_ROOT"])
    delayed_runner = tmp_path / "delayed_runner.sh"
    _write_executable(
        delayed_runner,
        """#!/usr/bin/env bash
set -u
mkdir -p "${FAKE_STATE_DIR}"
printf 'started\n' > "${FAKE_STATE_DIR}/started_${EVAL_START_SEEDS}"
trap 'exit 143' TERM
while [[ ! -f "${FAKE_STATE_DIR}/allow_exit" ]]; do sleep 0.05; done
exit 0
""",
    )
    env = os.environ.copy()
    env.update(
        {
            **runtime_env,
            "REPO_ROOT": str(repo),
            "RUNNER": str(delayed_runner),
            "CHECKER": str(checker),
            "CHECKER_PYTHON": sys.executable,
            "TASKS_CSV": "rearrange_blocks",
            "EVAL_START_SEEDS_CSV": "0",
            "BATCH_STAMP": "pre_ready_test",
            "RUN_LABEL": "pre_ready_test",
            "BATCH_LOG_ROOT": str(batch),
            "FAKE_STATE_DIR": str(state),
            "PERCEPTION_CONDITION": "oracle",
        }
    )
    proc = subprocess.Popen(
        ["bash", str(LAUNCHER), "--foreground"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for(state / "started_0")
        current = batch / "control" / "current.tsv"
        _wait_for(current)
        assert current.read_text().rstrip().endswith("\tstarting")
        control = subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                "--skip-current",
                "--batch-log-root",
                str(batch),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert control.returncode == 3
        assert "still preparing" in control.stderr
        assert proc.poll() is None
        assert not (state / "terminated_0").exists()
        (state / "allow_exit").write_text("go\n")
        proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)


def test_accepted_skip_never_advances_when_worker_ignores_sigint(
    tmp_path: Path,
) -> None:
    repo, runner, checker = _fake_sequential_repo(tmp_path)
    state = tmp_path / "state"
    runtime_env = _safe_runtime_environment(repo, "batch", tmp_path)
    batch = Path(runtime_env["BATCH_LOG_ROOT"])
    env = os.environ.copy()
    env.update(
        {
            **runtime_env,
            "REPO_ROOT": str(repo),
            "RUNNER": str(runner),
            "CHECKER": str(checker),
            "CHECKER_PYTHON": sys.executable,
            "TASKS_CSV": "rearrange_blocks",
            "EVAL_START_SEEDS_CSV": "0,1",
            "BATCH_STAMP": "ignore_int_test",
            "RUN_LABEL": "ignore_int_test",
            "BATCH_LOG_ROOT": str(batch),
            "FAKE_STATE_DIR": str(state),
            "FAKE_IGNORE_INT": "1",
            "CONTINUE_ON_ERROR": "1",
            "PERCEPTION_CONDITION": "oracle",
        }
    )
    proc = subprocess.Popen(
        ["bash", str(LAUNCHER), "--foreground"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    worker_host_pid: int | None = None
    try:
        _wait_for(state / "started_0")
        current = batch / "control" / "current.tsv"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if current.exists() and current.read_text().rstrip().endswith("\tready"):
                break
            time.sleep(0.05)
        else:
            raise AssertionError("runner never became ready")
        run_token = current.read_text().split("\t")[2]
        ready = batch / "control" / "requests" / run_token / "runner_ready.tsv"
        worker_row = next(
            line for line in ready.read_text().splitlines() if line.startswith("worker\t")
        )
        worker_host_pid = int(worker_row.split("\t")[1])
        control = subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                "--skip-current",
                "--batch-log-root",
                str(batch),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert control.returncode == 0
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.05)
        assert proc.poll() is not None, "batch did not terminate after cleanup failure"
        # The failure mode intentionally leaves the fake worker alive to
        # prove that no next rollout is started.  Close the inherited stdout
        # pipe before communicate(), then terminate the fixture worker.
        if worker_host_pid is not None:
            try:
                os.kill(worker_host_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        stdout = (batch / "batch.log").read_text(encoding="utf-8")
        if proc.stdout is not None:
            proc.stdout.close()
        assert proc.returncode != 0
        assert "refusing to start any later rollout" in stdout
        assert not (state / "started_1").exists()
        assert not (state / "checker_called").exists()
        assert "SKIPPED" not in stdout
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)
        if worker_host_pid is not None:
            try:
                os.kill(worker_host_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
