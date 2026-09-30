from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

import pytest

from policy.roboharn_evo.scripts import run_gpt55_parallel4_isolated as parallel4


REPO_ROOT = Path(__file__).resolve().parents[3]
PARALLEL4 = (
    REPO_ROOT / "policy" / "roboharn_evo" / "scripts" / "run_gpt55_parallel4_isolated.py"
)
SEQUENTIAL = (
    REPO_ROOT / "policy" / "roboharn_evo" / "scripts" / "run_gpt55_six_tasks_sequential.sh"
)
RUNNER = (
    REPO_ROOT / "policy" / "roboharn_evo" / "scripts" / "run_gpt55_pure_tool_control_8way.sh"
)
CHECKER = (
    REPO_ROOT / "policy" / "roboharn_evo" / "scripts" / "check_pure_tool_control_early_stop.py"
)

RENDERER_ENV_NAMES = (
    "RMBENCH_RENDER_DEVICE",
    "RMBENCH_RENDER_DEVICE_STRICT",
    "RMBENCH_EXPECTED_RENDER_CUDA_ID",
    "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID",
    "RMBENCH_EXPECTED_PHYSICAL_GPU",
    "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH",
)


def _parallel_runtime_env(
    tmp_path: Path,
    *,
    batch_name: str,
) -> tuple[dict[str, str], Path]:
    repo_root = REPO_ROOT
    project_root = repo_root.parents[1]
    output_root = project_root / "eval_result" / "rmbench"
    batch_root = (
        output_root
        / "copied_test_runtime"
        / tmp_path.parent.name
        / tmp_path.name
        / batch_name
    )
    assets_root = tmp_path / "readonly-assets"
    sam3_repo = tmp_path / "readonly-sam3"
    sam3_checkpoint = tmp_path / "readonly-sam3ckpt" / "sam3.pt"
    sam3_bpe = sam3_repo / "sam3" / "assets" / "bpe.txt.gz"
    causalwam_root = tmp_path / "readonly-causalwam"
    for directory in (
        assets_root / "embodiments",
        assets_root / "objects",
        sam3_bpe.parent,
        causalwam_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    sam3_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    sam3_checkpoint.write_bytes(b"test checkpoint")
    sam3_bpe.write_bytes(b"test bpe")
    return (
        {
            "REPO_ROOT": str(repo_root),
            "ROBOHARN_EVO_PROJECT_ROOT": str(project_root),
            "RMBENCH_OUTPUT_ROOT": str(output_root),
            "RMBENCH_ASSETS_ROOT": str(assets_root),
            "ROBOHARN_EVO_SAM3_REPO": str(sam3_repo),
            "ROBOHARN_EVO_SAM3_CHECKPOINT": str(sam3_checkpoint),
            "ROBOHARN_EVO_SAM3_BPE_PATH": str(sam3_bpe),
            "CAUSALWAM_ROOT": str(causalwam_root),
        },
        batch_root,
    )


def _install_capturing_tmux(tmp_path: Path) -> tuple[Path, Path]:
    """Install a tmux stand-in that records the real new-session argv."""

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    capture_path = tmp_path / "tmux-new-session.argv"
    tmux = fake_bin / "tmux"
    tmux.write_text(
        """#!/bin/sh
set -eu
case "${1-}" in
  has-session)
    exit 1
    ;;
  new-session)
    printf '%s\n' "$@" > "${FAKE_TMUX_CAPTURE}"
    exit 0
    ;;
  *)
    printf 'unexpected fake tmux command: %s\n' "${1-}" >&2
    exit 64
    ;;
esac
""",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    return fake_bin, capture_path


def _tmux_child_command(capture_path: Path) -> list[str]:
    argv = capture_path.read_text(encoding="utf-8").splitlines()
    assert argv[0] == "new-session"
    assert argv[-1]
    return shlex.split(argv[-1])


def _environment_assignments(command: list[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for token in command:
        if "=" not in token:
            continue
        name, value = token.split("=", 1)
        if name.replace("_", "A").isalnum() and name[0].isalpha():
            assignments[name] = value
    return assignments


def _assert_exact_renderer_assignments(
    command: list[str], expected: dict[str, str]
) -> None:
    assignments = _environment_assignments(command)
    assert {name: assignments.get(name) for name in RENDERER_ENV_NAMES} == expected
    for name in RENDERER_ENV_NAMES:
        assert sum(token.startswith(f"{name}=") for token in command) == 1


def _renderer_contract(provenance_path: Path) -> dict[str, str]:
    return {
        "RMBENCH_RENDER_DEVICE": "cuda:0",
        "RMBENCH_RENDER_DEVICE_STRICT": "1",
        "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
        "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": "00000000:14:00.0",
        "RMBENCH_EXPECTED_PHYSICAL_GPU": "4",
        "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH": str(provenance_path),
    }


def test_parallel4_detach_reconstructs_exact_six_key_slot_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin, capture_path = _install_capturing_tmux(tmp_path)
    runtime_env, batch_root = _parallel_runtime_env(
        tmp_path,
        batch_name="parallel batch with spaces",
    )
    env = os.environ.copy()
    env.pop("JOB_ASSIGNMENT_MODE", None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', os.defpath)}",
            "FAKE_TMUX_CAPTURE": str(capture_path),
            "REPO_ROOT": str(REPO_ROOT),
            "PARALLEL_SLOTS": "4",
            "GPU_IDS": "4,5,6,7",
            "SAM3_GPU_IDS": "0,1,2,3",
            "SAM3_BASE_PORT": "9311",
            "TASKS_CSV": "rearrange_blocks",
            "EVAL_START_SEEDS_CSV": "0,1,2,3",
            "BATCH_STAMP": "renderer_detach_parallel4",
            "BATCH_SESSION": "renderer_detach_parallel4",
            "RMBENCH_RENDER_DEVICE": "pci:auto",
            "RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS": (
                f"preexisting_identity={tmp_path / 'identity.json'}"
            ),
            **runtime_env,
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PARALLEL4),
            "--detach",
            "--batch-log-root",
            str(batch_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    detached_command = _tmux_child_command(capture_path)
    assignments = _environment_assignments(detached_command)
    assert assignments["RMBENCH_RENDER_DEVICE"] == "pci:auto"
    assert assignments["JOB_ASSIGNMENT_MODE"] == "round_robin"
    assert assignments["PYTHONPATH"] == os.pathsep.join(
        (runtime_env["ROBOHARN_EVO_PROJECT_ROOT"], runtime_env["REPO_ROOT"])
    )
    assert assignments["PYTHONDONTWRITEBYTECODE"] == "1"
    assert assignments["RMBENCH_ROOT"] == runtime_env["REPO_ROOT"]
    assert assignments["RMBENCH_OUTPUT_ROOT"] == runtime_env[
        "RMBENCH_OUTPUT_ROOT"
    ]
    assert assignments["ROBOHARN_EVO_OUTPUT_ROOT"] == runtime_env["RMBENCH_OUTPUT_ROOT"]
    assert assignments["RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS"] == (
        f"preexisting_identity={tmp_path / 'identity.json'}"
    )
    assert detached_command[-3:] == [
        "--foreground",
        "--batch-log-root",
        str(batch_root),
    ]

    # Recreate the detached scheduler environment from the command that tmux
    # actually received.  Deliberately seed stale values for the five fields
    # that the child scheduler derives from its host GPU inventory; launch_slots
    # must overwrite every one of them rather than trusting tmux-server state.
    with monkeypatch.context() as context:
        for name in RENDERER_ENV_NAMES:
            context.setenv(name, f"stale-{name.lower()}")
        for name, value in assignments.items():
            context.setenv(name, value)

        config = parallel4.load_config(str(batch_root))
        jobs = parallel4.build_jobs(config)
        created: list[Any] = []

        class FakeProcess:
            next_pid = 810_000

            def __init__(self, command: list[str], child_env: dict[str, str]) -> None:
                type(self).next_pid += 1
                self.pid = type(self).next_pid
                self.command = command
                self.env = child_env

            def poll(self) -> None:
                return None

        def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
            process = FakeProcess(command, dict(kwargs["env"]))
            created.append(process)
            return process

        context.setattr(parallel4.subprocess, "Popen", fake_popen)
        context.setattr(parallel4, "proc_start_ticks", lambda pid: pid + 1_000_000)
        scheduler = parallel4.Scheduler(config, jobs)
        scheduler.prepare()
        scheduler.gpu_inventory = {
            gpu: {
                "uuid": f"GPU-{gpu}",
                "pci_bus_id": f"00000000:{gpu + 16:02X}:00.0",
            }
            for gpu in range(8)
        }
        scheduler.launch_slots()

        try:
            assert len(created) == 4
            for slot, process in enumerate(created):
                slot_root = batch_root / "slots" / f"slot_{slot}"
                expected = {
                    "RMBENCH_RENDER_DEVICE": f"pci:0000:{slot + 20:02x}:00.0",
                    "RMBENCH_RENDER_DEVICE_STRICT": "1",
                    "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
                    "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": (
                        f"0000:{slot + 20:02x}:00.0"
                    ),
                    "RMBENCH_EXPECTED_PHYSICAL_GPU": str(slot + 4),
                    "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH": str(
                        slot_root / "renderer_device_provenance.pending.json"
                    ),
                }
                assert {
                    name: process.env.get(name) for name in RENDERER_ENV_NAMES
                } == expected
                assert process.env["RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN"] == (
                    f"GPU-{slot + 4}"
                )

                manifest_contract = scheduler.manifest["slots"][slot][
                    "renderer_device_contract"
                ]
                assert manifest_contract["per_job_provenance"] == [
                    {
                        "job_index": slot,
                        "task": "rearrange_blocks",
                        "eval_start_seed": slot,
                        "path": str(
                            slot_root
                            / "rearrange_blocks"
                            / f"episode_{slot:06d}"
                            / "renderer_device_provenance.json"
                        ),
                    }
                ]
                assert manifest_contract["expected_gpu_uuid"] == f"GPU-{slot + 4}"
                assert manifest_contract["cuda_visible_device_token"] == (
                    f"GPU-{slot + 4}"
                )
        finally:
            for owned in scheduler.slot_processes:
                owned.log_handle.close()


def test_parallel4_detach_propagates_task_affinity_assignment(
    tmp_path: Path,
) -> None:
    fake_bin, capture_path = _install_capturing_tmux(tmp_path)
    runtime_env, batch_root = _parallel_runtime_env(
        tmp_path,
        batch_name="parallel-task-affinity",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', os.defpath)}",
            "FAKE_TMUX_CAPTURE": str(capture_path),
            "REPO_ROOT": str(REPO_ROOT),
            "PARALLEL_SLOTS": "4",
            "GPU_IDS": "4,5,6,7",
            "SAM3_GPU_IDS": "0,1,2,3",
            "TASKS_CSV": "rearrange_blocks,swap_blocks,swap_T,battery_try",
            "EVAL_START_SEEDS_CSV": "0,1",
            "JOB_ASSIGNMENT_MODE": "task_affinity",
            "BATCH_STAMP": "task_affinity_detach",
            "BATCH_SESSION": "task_affinity_detach",
            "RMBENCH_RENDER_DEVICE": "pci:auto",
            **runtime_env,
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PARALLEL4),
            "--detach",
            "--batch-log-root",
            str(batch_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assignments = _environment_assignments(_tmux_child_command(capture_path))
    assert assignments["JOB_ASSIGNMENT_MODE"] == "task_affinity"


def test_sequential_detach_serializes_exact_renderer_contract(
    tmp_path: Path,
) -> None:
    fake_bin, capture_path = _install_capturing_tmux(tmp_path)
    output_root = REPO_ROOT.parents[1] / "eval_result" / "rmbench"
    batch_root = (
        output_root
        / "copied_test_runtime"
        / tmp_path.parent.name
        / tmp_path.name
        / "sequential batch"
    )
    assets_root = tmp_path / "readonly-assets"
    (assets_root / "embodiments").mkdir(parents=True)
    (assets_root / "objects").mkdir()
    expected = _renderer_contract(
        batch_root / "renderer evidence" / "sequential provenance.json"
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', os.defpath)}",
            "FAKE_TMUX_CAPTURE": str(capture_path),
            "REPO_ROOT": str(REPO_ROOT),
            "ROBOHARN_EVO_PROJECT_ROOT": str(REPO_ROOT.parents[1]),
            "RMBENCH_OUTPUT_ROOT": str(output_root),
            "RMBENCH_ASSETS_ROOT": str(assets_root),
            "RUNNER": str(RUNNER),
            "CHECKER": str(CHECKER),
            "CHECKER_PYTHON": sys.executable,
            "TASKS_CSV": "rearrange_blocks",
            "EVAL_START_SEEDS_CSV": "0",
            "BATCH_STAMP": "renderer_detach_sequential",
            "RUN_LABEL": "renderer_detach_sequential",
            "BATCH_LOG_ROOT": str(batch_root),
            "BATCH_SESSION": "renderer_detach_sequential",
            "PERCEPTION_CONDITION": "oracle",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN": "GPU-abcd",
            **expected,
        }
    )

    result = subprocess.run(
        ["/bin/bash", str(SEQUENTIAL), "--detach"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    command = _tmux_child_command(capture_path)
    _assert_exact_renderer_assignments(command, expected)
    assert _environment_assignments(command)[
        "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN"
    ] == "GPU-abcd"
    assignments = _environment_assignments(command)
    assert assignments["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert assignments["RMBENCH_SEQUENTIAL_CHILD"] == "1"
    assert assignments["PYTHONPATH"] == os.pathsep.join(
        (str(REPO_ROOT.parents[1]), str(REPO_ROOT))
    )
    assert assignments["PYTHONDONTWRITEBYTECODE"] == "1"
    assert assignments["RMBENCH_OUTPUT_ROOT"] == str(output_root)
    assert assignments["ROBOHARN_EVO_OUTPUT_ROOT"] == str(output_root)
    assert assignments["RMBENCH_ASSETS_ROOT"] == str(assets_root)
    assert command[-2:] == [str(SEQUENTIAL), "--foreground"]
    shutil.rmtree(batch_root)


def test_8way_detach_serializes_exact_renderer_contract(tmp_path: Path) -> None:
    fake_bin, capture_path = _install_capturing_tmux(tmp_path)
    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text("# detach test fixture\n", encoding="utf-8")
    expected = _renderer_contract(
        tmp_path / "renderer evidence" / "8way provenance.json"
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', os.defpath)}",
            "FAKE_TMUX_CAPTURE": str(capture_path),
            "REPO_ROOT": str(REPO_ROOT),
            "CONDA_SH": str(conda_sh),
            "DETACH": "1",
            "RMBENCH_DETACHED_CHILD": "0",
            "DETACH_SESSION": "renderer_detach_8way",
            "NUM_WORKERS": "1",
            "GPU_IDS": "4",
            "TASK_NAME": "synthetic_task",
            "TASK_CONFIG": "demo_clean",
            "PERCEPTION_CONDITION": "oracle",
            "NON_FORMAL_DIAGNOSTIC": "1",
            "SKIP_PREFLIGHT": "1",
            "PREFLIGHT_ONLY": "1",
            "RECORD_RUNTIME_PROVENANCE": "0",
            "REQUIRE_SAM3_PREFLIGHT": "0",
            "EVAL_START_SEEDS": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN": "GPU-abcd",
            "RMBENCH_RAY_TRACING_DENOISER": "none",
            "RMBENCH_SLOW_ACTION_TRACE_SEC": "120",
            "LOG_ROOT": str(tmp_path / "8way logs"),
            "SEGMENTATION_ARTIFACT_ROOT": str(tmp_path / "8way segmentation"),
            **expected,
        }
    )

    result = subprocess.run(
        ["/bin/bash", str(RUNNER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    command = _tmux_child_command(capture_path)
    _assert_exact_renderer_assignments(command, expected)
    assignments = _environment_assignments(command)
    assert assignments["RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN"] == "GPU-abcd"
    assert assignments["RMBENCH_RAY_TRACING_DENOISER"] == "none"
    assert assignments["RMBENCH_SLOW_ACTION_TRACE_SEC"] == "120"
    assert assignments["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert assignments["PYTHONPATH"] == os.pathsep.join(
        (str(REPO_ROOT.parents[1]), str(REPO_ROOT))
    )
    assert assignments["PYTHONDONTWRITEBYTECODE"] == "1"
    assert assignments["RMBENCH_ROOT"] == str(REPO_ROOT)
    assert assignments["RMBENCH_DETACHED_CHILD"] == "1"
    assert assignments["DETACH"] == "0"
    assert command[-1] == str(RUNNER)
