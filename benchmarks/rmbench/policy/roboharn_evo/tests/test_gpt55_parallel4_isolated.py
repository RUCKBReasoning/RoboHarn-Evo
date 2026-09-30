from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
from typing import Any

import pytest

from policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated import (
    ConfigError,
    GpuLockSet,
    OwnedProcess,
    RUNTIME_CACHE_ENVIRONMENT_LAYOUT,
    Scheduler,
    build_jobs,
    inspect_causalwam_process,
    load_config,
    preflight_gpus,
    preflight_ports,
    preflight_gpu_occupancy,
    write_plan,
    run_foreground,
)
from policy.roboharn_evo.scripts.record_runtime_provenance import (
    build_runtime_provenance,
    validate_runtime_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEDULER = (
    REPO_ROOT
    / "policy"
    / "roboharn_evo"
    / "scripts"
    / "run_gpt55_parallel4_isolated.py"
)


class _FakeProcess:
    next_pid = 700_000

    def __init__(self, command: list[str], env: dict[str, str]) -> None:
        type(self).next_pid += 1
        self.pid = type(self).next_pid
        self.command = command
        self.env = env
        self.returncode: int | None = None
        self.signals: list[signal.Signals] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def send_signal(self, sig: signal.Signals) -> None:
        self.signals.append(sig)
        if self.returncode is None:
            self.returncode = 128 + int(sig)


class _DelayedSigintProcess:
    """Process double that ignores SIGINT until the fake clock releases it."""

    next_pid = 710_000

    def __init__(self) -> None:
        type(self).next_pid += 1
        self.pid = type(self).next_pid
        self.returncode: int | None = None
        self.signals: list[signal.Signals] = []
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            raise AssertionError("wait_owned must not block in wait() before poll reports exit")
        return self.returncode

    def send_signal(self, sig: signal.Signals) -> None:
        self.signals.append(sig)

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


def _sam3_health_payload(
    scheduler: Scheduler,
    slot: int,
    process: _FakeProcess,
) -> dict[str, Any]:
    expected = scheduler.manifest["sam3_services"][slot]
    return {
        "status": "ok",
        "device": "cuda",
        "cuda_available": True,
        "cuda_current_device": 0,
        "cuda_visible_devices": expected["cuda_visible_device_token"],
        "cuda_device_order": "PCI_BUS_ID",
        "instance_id": expected["instance_id"],
        "pid": process.pid,
        "output_root": expected["output_root"],
    }


def _scheduler_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Any, list[Any]]:
    repo_root = REPO_ROOT
    project_root = repo_root.parents[1]
    output_root = project_root / "eval_result" / "rmbench"
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
    for name, value in {
        "REPO_ROOT": str(repo_root),
        "ROBOHARN_EVO_PROJECT_ROOT": str(project_root),
        "RMBENCH_OUTPUT_ROOT": str(output_root),
        "RMBENCH_ASSETS_ROOT": str(assets_root),
        "ROBOHARN_EVO_SAM3_REPO": str(sam3_repo),
        "ROBOHARN_EVO_SAM3_CHECKPOINT": str(sam3_checkpoint),
        "ROBOHARN_EVO_SAM3_BPE_PATH": str(sam3_bpe),
        "CAUSALWAM_ROOT": str(causalwam_root),
        "PARALLEL_SLOTS": "4",
        "GPU_IDS": "4,5,6,7",
        "SAM3_GPU_IDS": "0,1,2,3",
        "SAM3_BASE_PORT": "9311",
        "TASKS_CSV": "rearrange_blocks,swap_blocks",
        "EVAL_START_SEEDS_CSV": "0,1",
        "ALLOW_SHARED_GPU": "0",
        "BATCH_STAMP": "mock_lifecycle",
        "BATCH_LOG_ROOT": str(
            output_root
            / "copied_test_runtime"
            / tmp_path.parent.name
            / tmp_path.name
            / "parallel-batch"
        ),
    }.items():
        monkeypatch.setenv(name, value)
    config = load_config(None)
    return config, build_jobs(config)


def _capture_fake_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_FakeProcess]:
    created: list[_FakeProcess] = []

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        process = _FakeProcess(command, dict(kwargs["env"]))
        created.append(process)
        return process

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.proc_start_ticks",
        lambda pid: pid + 1_000_000,
    )
    return created


def test_runtime_temp_root_is_short_batch_scoped_and_inside_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _scheduler_config(monkeypatch, tmp_path)
    batch_token = hashlib.sha256(
        os.fsencode(str(config.batch_log_root.resolve()))
    ).hexdigest()[:12]

    assert config.runtime_temp_root == (
        config.output_root / "tmp" / batch_token
    ).resolve()
    short_slot_temp = config.runtime_temp_root / "s0"
    old_nested_temp = (
        config.batch_log_root / "slots" / "slot_0" / "runtime_caches" / "tmp"
    )
    assert short_slot_temp.is_relative_to(config.output_root / "tmp")
    assert len(str(short_slot_temp)) < len(str(old_nested_temp))
    assert len(str(short_slot_temp)) < 100

    monkeypatch.setenv(
        "BATCH_LOG_ROOT",
        str(config.batch_log_root.parent / "parallel-batch-2"),
    )
    other_config = load_config(None)
    assert other_config.runtime_temp_root != config.runtime_temp_root
    assert other_config.runtime_temp_root.parent == config.runtime_temp_root.parent


def _fake_gpu_inventory() -> dict[int, dict[str, str]]:
    return {
        gpu: {
            "uuid": f"GPU-{gpu}",
            "pci_bus_id": f"00000000:{gpu + 16:02X}:00.0",
        }
        for gpu in range(8)
    }


def _valid_renderer_provenance(
    *,
    eval_gpu: int,
    expected_visible_token: str,
    expected_pci_bus_id: str,
) -> dict[str, Any]:
    normalized_pci = expected_pci_bus_id.lower()
    if normalized_pci.startswith("00000000:"):
        normalized_pci = "0000:" + normalized_pci.removeprefix("00000000:")
    return {
        "schema": "rmbench/renderer_device_binding/v1",
        "mode": "explicit",
        "strict": True,
        "validation_passed": True,
        "requested_device": f"pci:{normalized_pci}",
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_device_tokens": [expected_visible_token],
        "expected_physical_gpu": str(eval_gpu),
        "expected_logical_cuda_id": 0,
        "requested_device_info": {
            "cuda_id": 0,
            "is_cuda": True,
            "can_render": True,
            "pci_bus_id_normalized": normalized_pci,
        },
        "selected_device_info": {
            "cuda_id": 0,
            "is_cuda": True,
            "can_render": True,
            "pci_bus_id_normalized": normalized_pci,
        },
    }


def test_load_config_rejects_an_alternate_or_donor_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REPO_ROOT", str(tmp_path / "alternate-rmbench"))
    with pytest.raises(ConfigError, match="may not select a donor checkout"):
        load_config(None)


def _write_slot_renderer_audit_inputs(
    scheduler: Scheduler,
    slot: int,
    *,
    status: str = "complete",
    provenance_mode: str = "valid",
) -> None:
    slot_record = scheduler.manifest["slots"][slot]
    slot_jobs = [job for job in scheduler.jobs if job.slot == slot]
    status_path = Path(slot_record["batch_log_root"]) / "batch_status.tsv"
    lines = ["task\teval_start_seed\tstatus"]
    lines.extend(
        f"{job.task}\t{job.eval_start_seed}\t{status}" for job in slot_jobs
    )
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if status == "skipped" or provenance_mode == "missing":
        return
    paths_by_job = {
        item["job_index"]: Path(item["path"])
        for item in slot_record["renderer_device_contract"]["per_job_provenance"]
    }
    expected_pci = slot_record["renderer_device_contract"]["expected_pci_bus_id"]
    expected_visible_token = slot_record["renderer_device_contract"][
        "cuda_visible_device_token"
    ]
    for job in slot_jobs:
        path = paths_by_job[job.job_index]
        path.parent.mkdir(parents=True, exist_ok=True)
        if provenance_mode == "invalid_json":
            path.write_text("{not valid json\n", encoding="utf-8")
        else:
            provenance = _valid_renderer_provenance(
                eval_gpu=job.eval_gpu,
                expected_visible_token=expected_visible_token,
                expected_pci_bus_id=expected_pci,
            )
            if provenance_mode == "wrong_requested_alias":
                provenance["requested_device"] = "pci:0000:ff:00.0"
            path.write_text(
                json.dumps(provenance)
                + "\n",
                encoding="utf-8",
            )


def _run_dry_run(**overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("JOB_ASSIGNMENT_MODE", None)
    env.update(
        {
            "REPO_ROOT": str(REPO_ROOT),
            "ROBOHARN_EVO_PROJECT_ROOT": str(REPO_ROOT.parents[1]),
            "RMBENCH_OUTPUT_ROOT": str(
                REPO_ROOT.parents[1] / "eval_result" / "rmbench"
            ),
            "PARALLEL_SLOTS": "4",
            "GPU_IDS": "4,5,6,7",
            "SAM3_GPU_IDS": "0,1,2,3",
            "SAM3_BASE_PORT": "9311",
            "TASKS_CSV": "rearrange_blocks,swap_blocks",
            "EVAL_START_SEEDS_CSV": "0,1",
            **overrides,
        }
    )
    return subprocess.run(
        [sys.executable, str(SCHEDULER), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )


def _parse_plan_rows(stdout: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("[job "):
            continue
        fields: dict[str, str] = {}
        for token in line.replace("[", "").replace("]", "").split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value
        rows.append(fields)
    return rows


def test_dry_run_maps_four_jobs_to_four_static_isolated_slots() -> None:
    result = _run_dry_run()

    assert result.returncode == 0, result.stderr
    rows = _parse_plan_rows(result.stdout)
    assert len(rows) == 4, result.stdout
    assert [row["job_index"] for row in rows] == ["0", "1", "2", "3"]
    assert [row["slot"] for row in rows] == ["0", "1", "2", "3"]
    assert [row["slot_ordinal"] for row in rows] == ["0", "0", "0", "0"]
    assert [row["eval_gpu"] for row in rows] == ["4", "5", "6", "7"]
    assert [row["sam3_gpu"] for row in rows] == ["0", "1", "2", "3"]
    assert [row["sam3_url"] for row in rows] == [
        "http://127.0.0.1:9311",
        "http://127.0.0.1:9312",
        "http://127.0.0.1:9313",
        "http://127.0.0.1:9314",
    ]
    assert [(row["task"], row["eval_start_seed"]) for row in rows] == [
        ("rearrange_blocks", "0"),
        ("rearrange_blocks", "1"),
        ("swap_blocks", "0"),
        ("swap_blocks", "1"),
    ]


def test_task_affinity_keeps_every_tasks_seeds_on_one_slot() -> None:
    result = _run_dry_run(
        TASKS_CSV="rearrange_blocks,swap_blocks,swap_T,battery_try",
        EVAL_START_SEEDS_CSV="0,1,2",
        JOB_ASSIGNMENT_MODE="task_affinity",
    )

    assert result.returncode == 0, result.stderr
    rows = _parse_plan_rows(result.stdout)
    assert len(rows) == 12
    assert [row["slot"] for row in rows] == [
        "0", "0", "0",
        "1", "1", "1",
        "2", "2", "2",
        "3", "3", "3",
    ]
    assert [row["slot_ordinal"] for row in rows] == [
        "0", "1", "2",
        "0", "1", "2",
        "0", "1", "2",
        "0", "1", "2",
    ]
    task_slots: dict[str, set[str]] = {}
    for row in rows:
        task_slots.setdefault(row["task"], set()).add(row["slot"])
    assert all(len(slots) == 1 for slots in task_slots.values())


def test_default_job_assignment_mode_remains_round_robin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JOB_ASSIGNMENT_MODE", raising=False)
    config = load_config(None)

    assert config.job_assignment_mode == "round_robin"


def test_rejects_unknown_job_assignment_mode() -> None:
    result = _run_dry_run(JOB_ASSIGNMENT_MODE="task_majorish")

    assert result.returncode == 2
    assert "JOB_ASSIGNMENT_MODE must be round_robin or task_affinity" in result.stderr


@pytest.mark.parametrize(
    ("overrides", "error_fragment"),
    [
        ({"GPU_IDS": "4,5,5,7"}, "duplicate"),
        ({"SAM3_GPU_IDS": "0,1,1,3"}, "duplicate"),
        ({"GPU_IDS": "4,5,6"}, "4"),
        ({"SAM3_GPU_IDS": "0,1,2"}, "4"),
        ({"SAM3_BASE_PORT": "0"}, "port"),
        ({"SAM3_BASE_PORT": "65533"}, "port"),
    ],
)
def test_rejects_duplicate_or_incomplete_slot_resources(
    overrides: dict[str, str], error_fragment: str
) -> None:
    result = _run_dry_run(**overrides)

    assert result.returncode == 2
    assert error_fragment.lower() in result.stderr.lower()


def test_rejects_cross_pool_gpu_overlap_by_default() -> None:
    result = _run_dry_run(
        GPU_IDS="4,5,6,7",
        SAM3_GPU_IDS="0,1,2,4",
    )

    assert result.returncode == 2
    assert "overlap" in result.stderr.lower() or "shared" in result.stderr.lower()


def test_diagnostic_shared_gpu_opt_in_does_not_change_slot_mapping() -> None:
    result = _run_dry_run(
        GPU_IDS="4,5,6,7",
        SAM3_GPU_IDS="0,1,2,4",
        ALLOW_SHARED_GPU="1",
    )

    assert result.returncode == 0, result.stderr
    rows = _parse_plan_rows(result.stdout)
    assert [row["eval_gpu"] for row in rows] == ["4", "5", "6", "7"]
    assert [row["sam3_gpu"] for row in rows] == ["0", "1", "2", "4"]


def test_parallel_scheduler_rejects_non_auto_renderer_device_policy() -> None:
    result = _run_dry_run(RMBENCH_RENDER_DEVICE="cuda:0")

    assert result.returncode == 2
    assert "pci:auto" in result.stderr


def test_skip_slot_requires_batch_log_root() -> None:
    result = subprocess.run(
        [sys.executable, str(SCHEDULER), "--skip-slot", "2"],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "batch-log-root" in result.stderr.lower()


def test_dry_run_is_read_only_even_with_explicit_batch_root(tmp_path: Path) -> None:
    batch_root = (
        REPO_ROOT.parents[1]
        / "eval_result"
        / "rmbench"
        / "copied_test_runtime"
        / tmp_path.name
        / "must-not-be-created"
    )
    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO_ROOT),
            "ROBOHARN_EVO_PROJECT_ROOT": str(REPO_ROOT.parents[1]),
            "RMBENCH_OUTPUT_ROOT": str(
                REPO_ROOT.parents[1] / "eval_result" / "rmbench"
            ),
            "PARALLEL_SLOTS": "4",
            "GPU_IDS": "4,5,6,7",
            "SAM3_GPU_IDS": "0,1,2,3",
            "SAM3_BASE_PORT": "9311",
            "TASKS_CSV": "rearrange_blocks,swap_blocks",
            "EVAL_START_SEEDS_CSV": "0,1",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCHEDULER),
            "--dry-run",
            "--batch-log-root",
            str(batch_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not batch_root.exists()
    assert not (batch_root / "parallel_manifest.json").exists()
    batch_token = hashlib.sha256(
        os.fsencode(str(batch_root.resolve()))
    ).hexdigest()[:12]
    assert not (
        REPO_ROOT.parents[1]
        / "eval_result"
        / "rmbench"
        / "tmp"
        / batch_token
    ).exists()


def test_dry_run_does_not_touch_gpu_locks_or_nvidia_smi(tmp_path: Path) -> None:
    lock_root = (
        REPO_ROOT.parents[1]
        / "eval_result"
        / "rmbench"
        / "copied_test_runtime"
        / tmp_path.name
        / "must-not-be-created"
    )
    marker_bin = tmp_path / "bin"
    marker_bin.mkdir()
    marker = tmp_path / "nvidia-smi-was-called"
    fake_smi = marker_bin / "nvidia-smi"
    fake_smi.write_text(
        f"#!/bin/sh\ntouch {marker}\nexit 99\n",
        encoding="utf-8",
    )
    fake_smi.chmod(0o755)
    result = _run_dry_run(
        GPU_LOCK_ROOT=str(lock_root),
        PATH=f"{marker_bin}:{os.environ.get('PATH', '')}",
    )

    assert result.returncode == 0, result.stderr
    assert not lock_root.exists()
    assert not marker.exists()


def test_source_defines_complete_manifest_schema() -> None:
    source = SCHEDULER.read_text(encoding="utf-8")
    required_columns = (
        "job_index",
        "slot",
        "slot_ordinal",
        "task",
        "eval_start_seed",
        "eval_gpu",
        "sam3_gpu",
        "sam3_url",
    )

    for column in required_columns:
        assert column in source


def test_parallel_plan_tsv_has_stable_complete_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "PARALLEL_SLOTS": "4",
        "GPU_IDS": "4,5,6,7",
        "SAM3_GPU_IDS": "0,1,2,3",
        "SAM3_BASE_PORT": "9311",
        "TASKS_CSV": "rearrange_blocks,swap_blocks",
        "EVAL_START_SEEDS_CSV": "0,1",
    }.items():
        monkeypatch.setenv(name, value)
    config = load_config(None)
    jobs = build_jobs(config)
    plan_path = tmp_path / "parallel_plan.tsv"

    write_plan(plan_path, jobs)

    lines = plan_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == [
        "job_index",
        "slot",
        "slot_ordinal",
        "task",
        "eval_start_seed",
        "eval_gpu",
        "sam3_gpu",
        "sam3_url",
    ]
    assert len(lines) == 5
    assert lines[1].split("\t") == [
        "0",
        "0",
        "0",
        "rearrange_blocks",
        "0",
        "4",
        "0",
        "http://127.0.0.1:9311",
    ]


def test_preflight_rejects_an_occupied_sam3_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        pytest.skip("test sandbox forbids creating sockets")
    listener.bind(("127.0.0.1", 0))
    occupied_port = listener.getsockname()[1]
    if occupied_port > 65532:
        listener.close()
        pytest.skip("kernel selected a port too close to 65535 for a four-slot range")
    listener.listen(1)
    for name, value in {
        "PARALLEL_SLOTS": "4",
        "GPU_IDS": "4,5,6,7",
        "SAM3_GPU_IDS": "0,1,2,3",
        "SAM3_BASE_PORT": str(occupied_port),
        "TASKS_CSV": "rearrange_blocks",
        "EVAL_START_SEEDS_CSV": "0",
    }.items():
        monkeypatch.setenv(name, value)
    config = load_config(None)

    try:
        with pytest.raises(RuntimeError, match=rf"port {occupied_port}.*occupied"):
            preflight_ports(config)
    finally:
        listener.close()


def test_scheduler_launches_four_sam3_processes_with_isolated_gpu_identity_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    created = _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    assert scheduler.manifest["config"]["runtime_temp_root"] == str(
        config.runtime_temp_root
    )
    scheduler.prepare()
    scheduler.gpu_inventory = _fake_gpu_inventory()
    monkeypatch.setattr(scheduler, "wait_for_sam3_health", lambda: None)

    scheduler.launch_sam3()

    assert len(created) == 4
    assert [process.env["CUDA_VISIBLE_DEVICES"] for process in created] == [
        "GPU-0",
        "GPU-1",
        "GPU-2",
        "GPU-3",
    ]
    assert all(
        process.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" for process in created
    )
    for slot, process in enumerate(created):
        assert process.env["PYTHONPATH"] == os.pathsep.join(
            (str(config.roboharn_project_root), str(config.repo_root))
        )
        assert process.env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert process.env["RMBENCH_OUTPUT_ROOT"] == str(config.output_root)
        assert process.env["ROBOHARN_EVO_OUTPUT_ROOT"] == str(config.output_root)
        assert process.env["RMBENCH_ASSETS_ROOT"] == str(config.assets_root)
        expected_temp_root = config.runtime_temp_root / f"sam{slot}"
        assert process.env["TMPDIR"] == str(expected_temp_root)
        assert process.env["TMP"] == str(expected_temp_root)
        assert process.env["TEMP"] == str(expected_temp_root)
        assert expected_temp_root.is_dir()
        assert process.command[process.command.index("--port") + 1] == str(9311 + slot)
        output_root = Path(process.command[process.command.index("--output-root") + 1])
        assert output_root == (
            config.batch_log_root / "segmentation_artifacts" / f"slot_{slot}"
        ).resolve()
        instance_id = process.command[process.command.index("--instance-id") + 1]
        assert instance_id == f"{scheduler.token}:slot_{slot}"
        service = scheduler.manifest["sam3_services"][slot]
        assert service["gpu_uuid"] == f"GPU-{slot}"
        assert service["cuda_visible_device_token"] == f"GPU-{slot}"
    assert len({item["output_root"] for item in scheduler.manifest["sam3_services"]}) == 4
    assert len({process.env["TMPDIR"] for process in created}) == 4


def test_sam3_health_accepts_exact_uuid_bound_cuda_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    config = config.__class__(
        **{
            **config.__dict__,
            "sam3_health_interval_sec": 0.0,
            "sam3_real_probe": False,
        }
    )
    created = _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    scheduler.gpu_inventory = _fake_gpu_inventory()

    def fake_request(url: str, timeout: float) -> dict[str, Any]:
        del timeout
        slot = int(url.rsplit(":", 1)[1].split("/", 1)[0]) - config.sam3_base_port
        return _sam3_health_payload(scheduler, slot, created[slot])

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.json_request",
        fake_request,
    )

    scheduler.launch_sam3()

    assert all(
        service["health_verified"] is True
        and service["health_checks"]
        and all(service["health_checks"].values())
        for service in scheduler.manifest["sam3_services"]
    )
    ready_bytes = scheduler.sam3_ready_manifest_path.read_bytes()
    ready = json.loads(ready_bytes)
    assert ready["schema"] == "tcm/sam3_scheduler_readiness/v1"
    assert ready["state"] == "sam3_ready"
    assert [service["sam3_gpu_id"] for service in ready["services"]] == [
        0,
        1,
        2,
        3,
    ]
    assert [service["sam3_gpu_uuid"] for service in ready["services"]] == [
        "GPU-0",
        "GPU-1",
        "GPU-2",
        "GPU-3",
    ]
    assert "token" not in ready_bytes.decode("utf-8").lower()

    scheduler.manifest["slots"].append({"slot": 0, "state": "running"})
    scheduler.persist("running")
    assert scheduler.sam3_ready_manifest_path.read_bytes() == ready_bytes

    provenance = build_runtime_provenance(
        config.repo_root,
        ("policy/roboharn_evo/scripts/run_gpt55_parallel4_isolated.py",),
        external_evidence_specs=(
            "sam3_scheduler_ready_manifest="
            f"{scheduler.sam3_ready_manifest_path}",
        ),
        environ={
            "TASK_NAME": "rearrange_blocks",
            "PERCEPTION_CONDITION": "no_oracle",
            "EVAL_START_SEEDS": "0",
            "GPU_IDS": "4",
            "SAM3_SERVICE_URL": "http://127.0.0.1:9311",
        },
    )
    embedded = provenance["external_evidence"][
        "sam3_scheduler_ready_manifest"
    ]
    assert embedded["json"] == ready
    assert embedded["path"] == str(scheduler.sam3_ready_manifest_path.resolve())
    assert len(embedded["bytes_sha256"]) == 64
    assert len(embedded["canonical_sha256"]) == 64

    validated = validate_runtime_provenance(
        provenance,
        config.repo_root,
        verify_external_evidence=False,
        expected_effective_config=provenance["effective_config"],
    )
    assert validated["external_evidence"][
        "sam3_scheduler_ready_manifest"
    ]["json"] == ready

    tampered = json.loads(json.dumps(provenance))
    tampered["external_evidence"]["sam3_scheduler_ready_manifest"]["json"][
        "state"
    ] = "tampered"
    with pytest.raises(ValueError, match="canonical_sha256"):
        validate_runtime_provenance(
            tampered,
            config.repo_root,
            verify_external_evidence=False,
            expected_effective_config=provenance["effective_config"],
        )


def test_sam3_health_rejects_wrong_visible_gpu_uuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    config = config.__class__(
        **{
            **config.__dict__,
            "sam3_health_interval_sec": 0.0,
            "sam3_real_probe": False,
        }
    )
    created = _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    scheduler.gpu_inventory = _fake_gpu_inventory()

    def fake_request(url: str, timeout: float) -> dict[str, Any]:
        del timeout
        slot = int(url.rsplit(":", 1)[1].split("/", 1)[0]) - config.sam3_base_port
        payload = _sam3_health_payload(scheduler, slot, created[slot])
        if slot == 2:
            payload["cuda_visible_devices"] = "GPU-wrong"
        return payload

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.json_request",
        fake_request,
    )

    with pytest.raises(ConfigError, match=r"slot 2 health identity mismatch"):
        scheduler.launch_sam3()

    assert scheduler.manifest["sam3_services"][2]["health_checks"][
        "cuda_visible_device_token"
    ] is False


def test_sam3_forward_probes_cover_every_slot_and_persist_ready_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    scheduler.manifest["sam3_services"] = [
        {
            "slot": slot,
            "url": f"http://127.0.0.1:{9311 + slot}",
            "output_root": str(
                (
                    config.batch_log_root
                    / "segmentation_artifacts"
                    / f"slot_{slot}"
                ).resolve()
            ),
            "state": "health_verified",
        }
        for slot in range(4)
    ]
    calls: list[tuple[str, dict[str, Any], float]] = []

    def fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        calls.append((url, payload, timeout))
        return {"success": True, "num_detections": 1}

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.json_post_request",
        fake_post,
    )

    scheduler.run_sam3_inference_probes()

    assert {url for url, _, _ in calls} == {
        f"http://127.0.0.1:{port}/segment_image" for port in range(9311, 9315)
    }
    assert len(calls) == 4
    for url, payload, timeout in calls:
        slot = int(url.rsplit(":", 1)[1].split("/", 1)[0]) - 9311
        assert payload["object_id"] == f"startup_probe_slot_{slot}"
        assert Path(payload["image_path"]).is_file()
        assert Path(payload["output_dir"]) == (
            config.batch_log_root
            / "segmentation_artifacts"
            / f"slot_{slot}"
            / "_startup_probe"
        ).resolve()
        assert timeout == config.sam3_probe_timeout_sec
    for service in scheduler.manifest["sam3_services"]:
        assert service["state"] == "ready"
        assert service["inference_probe"]["success"] is True
        assert service["inference_probe"]["num_detections"] == 1


def test_sam3_forward_probe_failure_is_fail_closed_and_identifies_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    scheduler.manifest["sam3_services"] = [
        {
            "slot": slot,
            "url": f"http://127.0.0.1:{9311 + slot}",
            "output_root": str(
                (
                    config.batch_log_root
                    / "segmentation_artifacts"
                    / f"slot_{slot}"
                ).resolve()
            ),
            "state": "health_verified",
        }
        for slot in range(4)
    ]

    def fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        del payload, timeout
        if ":9313/" in url:
            return {"success": False, "error": "synthetic inference failure"}
        return {"success": True, "num_detections": 0}

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.json_post_request",
        fake_post,
    )

    with pytest.raises(RuntimeError, match=r"slot 2:.*synthetic inference failure"):
        scheduler.run_sam3_inference_probes()

    assert scheduler.manifest["sam3_services"][2]["state"] == "health_verified"
    assert "inference_probe" not in scheduler.manifest["sam3_services"][2]
    assert all(service["state"] != "ready" for service in scheduler.manifest["sam3_services"])


def test_scheduler_launches_slots_with_exact_static_job_subset_and_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate --detach, whose outer scheduler needs this variable to launch
    # its service pool.  It must not leak into any delegated rollout slot.
    monkeypatch.setenv("SAM3_BASE_PORT", "9311")
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    created = _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    scheduler.gpu_inventory = _fake_gpu_inventory()

    scheduler.launch_slots()

    assert len(created) == 4
    # The actual task-major grid has four jobs for this fixture, so each slot
    # gets exactly one job.  Build expectations from the immutable Job plan to
    # avoid duplicating scheduler assignment logic in prose.
    expected_jobs = {
        slot: ",".join(
            f"{job.task}:{job.eval_start_seed}" for job in jobs if job.slot == slot
        )
        for slot in range(4)
    }
    for slot, process in enumerate(created):
        slot_root = config.batch_log_root / "slots" / f"slot_{slot}"
        assert process.env["GPU_ID"] == str(4 + slot)
        assert process.env["PYTHONPATH"] == os.pathsep.join(
            (str(config.roboharn_project_root), str(config.repo_root))
        )
        assert process.env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert process.env["RMBENCH_ROOT"] == str(config.repo_root)
        assert process.env["RMBENCH_ASSETS_ROOT"] == str(config.assets_root)
        assert process.env["RMBENCH_OUTPUT_ROOT"] == str(config.output_root)
        assert process.env["ROBOHARN_EVO_OUTPUT_ROOT"] == str(config.output_root)
        assert process.env["ROBOHARN_EVO_WORKSPACE_ROOT"] == str(
            slot_root / "runtime_workspace"
        )
        expected_temp_root = config.runtime_temp_root / f"s{slot}"
        assert process.env["TMPDIR"] == str(expected_temp_root)
        assert process.env["TMP"] == str(expected_temp_root)
        assert process.env["TEMP"] == str(expected_temp_root)
        assert expected_temp_root.is_dir()
        for variable, relative_path in RUNTIME_CACHE_ENVIRONMENT_LAYOUT.items():
            assert process.env[variable] == str(
                slot_root / "runtime_caches" / relative_path
            )
        assert process.env[
            "RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS"
        ].splitlines()[-1] == (
            "sam3_scheduler_ready_manifest="
            f"{scheduler.sam3_ready_manifest_path}"
        )
        assert process.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert process.env["RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN"] == f"GPU-{4 + slot}"
        expected_pci = f"0000:{4 + slot + 16:02x}:00.0"
        assert process.env["RMBENCH_RENDER_DEVICE"] == f"pci:{expected_pci}"
        assert process.env["RMBENCH_RENDER_DEVICE_STRICT"] == "1"
        assert process.env["RMBENCH_EXPECTED_RENDER_CUDA_ID"] == "0"
        assert process.env["RMBENCH_EXPECTED_RENDER_PCI_BUS_ID"] == expected_pci
        assert process.env["RMBENCH_EXPECTED_PHYSICAL_GPU"] == str(4 + slot)
        assert Path(process.env["RMBENCH_RENDER_DEVICE_PROVENANCE_PATH"]) == (
            config.batch_log_root
            / "slots"
            / f"slot_{slot}"
            / "renderer_device_provenance.pending.json"
        ).resolve()
        assert process.env["SAM3_SERVICE_URL"] == f"http://127.0.0.1:{9311 + slot}"
        assert "SAM3_BASE_PORT" not in process.env
        assert process.env["RUN_JOBS_CSV"] == expected_jobs[slot]
        assert process.env["BATCH_DETACH"] == "0"
        assert process.env["CONTINUE_ON_ERROR"] == "0"
        artifact_root = Path(process.env["SEGMENTATION_ARTIFACT_ROOT"])
        assert artifact_root == (
            config.batch_log_root / "segmentation_artifacts" / f"slot_{slot}"
        ).resolve()
        contract = scheduler.manifest["slots"][slot]["renderer_device_contract"]
        assert contract["provenance_scope"] == "per_task_seed_run_log"
        assert contract["cuda_device_order"] == "PCI_BUS_ID"
        assert contract["expected_gpu_uuid"] == f"GPU-{4 + slot}"
        assert contract["cuda_visible_device_token"] == f"GPU-{4 + slot}"
        assert contract["requested_render_alias"] == f"pci:{expected_pci}"
        assert contract["expected_pci_bus_id"] == expected_pci
        assert contract["per_job_provenance"] == [
            {
                "job_index": job.job_index,
                "task": job.task,
                "eval_start_seed": job.eval_start_seed,
                "path": str(
                    config.batch_log_root
                    / "slots"
                    / f"slot_{slot}"
                    / job.task
                    / f"episode_{job.eval_start_seed:06d}"
                    / "renderer_device_provenance.json"
                ),
            }
            for job in jobs
            if job.slot == slot
        ]
    assert len({process.env["SEGMENTATION_ARTIFACT_ROOT"] for process in created}) == 4
    assert len({process.env["TMPDIR"] for process in created}) == 4


def test_scheduler_refuses_slot_launch_without_host_gpu_pci_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()

    with pytest.raises(ConfigError, match=r"missing.*PCI identity.*GPU 4"):
        scheduler.launch_slots()


def test_gpu_preflight_returns_uuid_and_pci_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _jobs = _scheduler_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.run_nvidia_smi",
        lambda arguments: (
            "\n".join(
                f"{gpu}, GPU-{gpu}, 00000000:{gpu + 16:02X}:00.0"
                for gpu in range(8)
            )
            + "\n"
            if arguments[0].startswith("--query-gpu=")
            else pytest.fail(arguments)
        ),
    )

    inventory = preflight_gpus(config)

    assert inventory[4] == {
        "uuid": "GPU-4",
        "pci_bus_id": "0000:14:00.0",
    }
    assert set(inventory) == set(range(8))


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            "4, [N/A], 00000000:14:00.0\n",
            "no usable UUID",
        ),
        (
            "4, GPU-duplicate, 00000000:14:00.0\n"
            "5, GPU-duplicate, 00000000:15:00.0\n",
            "duplicate GPU UUID",
        ),
    ],
)
def test_gpu_preflight_rejects_unusable_or_duplicate_uuid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: str,
    message: str,
) -> None:
    config, _jobs = _scheduler_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.run_nvidia_smi",
        lambda _arguments: rows,
    )

    with pytest.raises(ConfigError, match=message):
        preflight_gpus(config)


def test_sam3_launch_refuses_missing_host_gpu_uuid_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()

    with pytest.raises(ConfigError, match=r"missing.*GPU UUID.*physical GPU 0"):
        scheduler.launch_sam3()


def test_renderer_provenance_audit_accepts_complete_job_with_matching_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    scheduler.gpu_inventory = _fake_gpu_inventory()
    scheduler.launch_slots()
    _write_slot_renderer_audit_inputs(scheduler, 0)

    audit = scheduler.audit_slot_renderer_provenance(
        0, scheduler.manifest["slots"][0]
    )

    assert audit["passed"] is True
    assert audit["errors"] == []
    assert len(audit["entries"]) == 1
    entry = audit["entries"][0]
    assert entry["job_index"] == jobs[0].job_index
    assert entry["task"] == jobs[0].task
    assert entry["eval_start_seed"] == jobs[0].eval_start_seed
    assert entry["status"] == "complete"
    assert entry["provenance_required"] is True
    assert entry["passed"] is True
    assert entry["errors"] == []
    assert entry["checks"] and all(entry["checks"].values())
    assert len(entry["file_sha256"]) == 64


@pytest.mark.parametrize(
    "provenance_mode", ["missing", "invalid_json", "wrong_requested_alias"]
)
def test_monitor_maps_missing_or_invalid_renderer_provenance_to_rc76(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provenance_mode: str,
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    created = _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    scheduler.gpu_inventory = _fake_gpu_inventory()
    scheduler.launch_slots()
    _write_slot_renderer_audit_inputs(
        scheduler,
        0,
        provenance_mode=provenance_mode,
    )
    created[0].returncode = 0
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.sleep",
        lambda _seconds: None,
    )

    result = scheduler.monitor()

    record = scheduler.manifest["slots"][0]
    assert result == 76
    assert record["returncode"] == 0
    assert record["effective_returncode"] == 76
    assert record["state"] == "failed"
    assert record["renderer_device_audit"]["passed"] is False
    assert record["renderer_device_audit"]["errors"]


def test_renderer_provenance_audit_does_not_require_file_for_skipped_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    scheduler.gpu_inventory = _fake_gpu_inventory()
    scheduler.launch_slots()
    _write_slot_renderer_audit_inputs(scheduler, 0, status="skipped")

    audit = scheduler.audit_slot_renderer_provenance(
        0, scheduler.manifest["slots"][0]
    )

    assert audit["passed"] is True
    assert audit["errors"] == []
    assert audit["entries"] == [
        {
            "job_index": jobs[0].job_index,
            "task": jobs[0].task,
            "eval_start_seed": jobs[0].eval_start_seed,
            "status": "skipped",
            "provenance_required": False,
            "passed": True,
        }
    ]


def test_monitor_completes_when_all_ledgers_and_renderer_provenance_are_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    created = _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    scheduler.gpu_inventory = _fake_gpu_inventory()
    scheduler.launch_slots()
    for slot in range(config.slots):
        _write_slot_renderer_audit_inputs(scheduler, slot)
        created[slot].returncode = 0
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.sleep",
        lambda _seconds: None,
    )

    result = scheduler.monitor()

    assert result == 0
    assert [record["state"] for record in scheduler.manifest["slots"]] == [
        "complete"
    ] * 4
    assert [
        record["effective_returncode"] for record in scheduler.manifest["slots"]
    ] == [0] * 4
    assert all(
        record["renderer_device_audit"]["passed"] is True
        for record in scheduler.manifest["slots"]
    )


def test_one_slot_failure_sends_sigint_to_every_other_live_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    created = _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    scheduler.gpu_inventory = _fake_gpu_inventory()
    scheduler.launch_slots()
    created[0].returncode = 7
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.sleep",
        lambda _seconds: None,
    )

    result = scheduler.monitor()

    assert result == 7
    assert created[0].signals == []
    for process in created[1:]:
        assert process.signals == [signal.SIGINT]
        assert process.returncode == 130


def test_shutdown_orders_slot_sigint_before_sam3_sigint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    created = _capture_fake_processes(monkeypatch)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    scheduler.gpu_inventory = _fake_gpu_inventory()
    monkeypatch.setattr(scheduler, "wait_for_sam3_health", lambda: None)
    scheduler.launch_sam3()
    sam3_processes = list(created)
    scheduler.gpu_inventory = _fake_gpu_inventory()
    scheduler.launch_slots()
    slot_processes = created[4:]
    order: list[str] = []
    original_signal_owned = scheduler.signal_owned

    def record_signal(items: list[Any], sig: signal.Signals) -> None:
        order.extend(item.kind for item in items)
        original_signal_owned(items, sig)

    monkeypatch.setattr(scheduler, "signal_owned", record_signal)

    scheduler.shutdown("interrupted")

    assert order == ["slot"] * 4 + ["sam3"] * 4
    assert all(process.signals == [signal.SIGINT] for process in slot_processes)
    assert all(process.signals == [signal.SIGINT] for process in sam3_processes)
    assert scheduler.manifest["state"] == "interrupted"


def test_shutdown_restores_process_signal_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()
    before = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }

    scheduler.shutdown("interrupted")

    assert {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    } == before


def test_wait_owned_grace_reports_live_process_without_escalating_or_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _DelayedSigintProcess()
    log_path = tmp_path / "owned.log"
    log_handle = log_path.open("w", encoding="utf-8")
    owned = OwnedProcess(
        kind="slot",
        slot=0,
        process=process,  # type: ignore[arg-type]
        log_path=log_path,
        log_handle=log_handle,
        start_ticks=123,
    )
    now = [0.0]
    events: list[tuple[str, Any]] = []

    def fake_monotonic() -> float:
        return now[0]

    def fake_sleep(seconds: float) -> None:
        now[0] += seconds
        if now[0] >= 0.05 and process.returncode is None:
            process.returncode = 130
            events.append(("process_exit", now[0]))

    def on_grace_expired(live: list[OwnedProcess]) -> None:
        events.append(("grace_expired", [item.process.pid for item in live]))
        assert process.poll() is None
        assert not log_handle.closed

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.monotonic",
        fake_monotonic,
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.sleep",
        fake_sleep,
    )

    Scheduler.signal_owned([owned], signal.SIGINT)
    Scheduler.wait_owned(
        [owned],
        grace_sec=0.02,
        on_grace_expired=on_grace_expired,
    )
    events.append(("subsequent_resource", now[0]))

    assert events[0] == ("grace_expired", [process.pid])
    assert events[1][0] == "process_exit"
    assert events[2][0] == "subsequent_resource"
    assert process.signals == [signal.SIGINT]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.wait_calls == 1
    assert log_handle.closed


def test_wait_owned_grace_callback_is_emitted_once_while_sigint_exit_is_slow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _DelayedSigintProcess()
    log_path = tmp_path / "slow-owned.log"
    log_handle = log_path.open("w", encoding="utf-8")
    owned = OwnedProcess(
        kind="sam3",
        slot=2,
        process=process,  # type: ignore[arg-type]
        log_path=log_path,
        log_handle=log_handle,
        start_ticks=456,
    )
    now = [0.0]
    grace_snapshots: list[list[int]] = []

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.monotonic",
        lambda: now[0],
    )

    def fake_sleep(seconds: float) -> None:
        now[0] += seconds
        if now[0] >= 0.20:
            process.returncode = 130

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.sleep",
        fake_sleep,
    )

    Scheduler.signal_owned([owned], signal.SIGINT)
    Scheduler.wait_owned(
        [owned],
        grace_sec=0.01,
        on_grace_expired=lambda live: grace_snapshots.append(
            [item.process.pid for item in live]
        ),
    )

    assert grace_snapshots == [[process.pid]]
    assert process.signals == [signal.SIGINT]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.returncode == 130
    assert log_handle.closed


def test_wait_owned_keeps_waiting_when_grace_callback_raises_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    process = _DelayedSigintProcess()
    log_path = tmp_path / "callback-interrupted-owned.log"
    log_handle = log_path.open("w", encoding="utf-8")
    owned = OwnedProcess(
        kind="slot",
        slot=1,
        process=process,  # type: ignore[arg-type]
        log_path=log_path,
        log_handle=log_handle,
        start_ticks=789,
    )
    now = [0.0]
    callback_calls: list[list[int]] = []

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.monotonic",
        lambda: now[0],
    )

    def fake_sleep(seconds: float) -> None:
        now[0] += seconds
        if now[0] >= 0.03:
            process.returncode = 130

    def interrupted_callback(live: list[OwnedProcess]) -> None:
        callback_calls.append([item.process.pid for item in live])
        raise KeyboardInterrupt("synthetic manifest interruption")

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.sleep",
        fake_sleep,
    )

    Scheduler.signal_owned([owned], signal.SIGINT)
    Scheduler.wait_owned(
        [owned],
        grace_sec=0.01,
        on_grace_expired=interrupted_callback,
    )

    assert callback_calls == [[process.pid]]
    assert process.returncode == 130
    assert process.signals == [signal.SIGINT]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.wait_calls == 1
    assert log_handle.closed
    assert "continuing to wait with resources held" in capsys.readouterr().err


def test_shutdown_final_drain_waits_for_all_children_when_stop_paths_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    scheduler = Scheduler(config, jobs)
    scheduler.prepare()

    slot_process = _DelayedSigintProcess()
    sam_process = _DelayedSigintProcess()
    slot_log_path = tmp_path / "failed-stop-slot.log"
    sam_log_path = tmp_path / "failed-stop-sam3.log"
    slot_log_handle = slot_log_path.open("w", encoding="utf-8")
    sam_log_handle = sam_log_path.open("w", encoding="utf-8")
    scheduler.slot_processes = [
        OwnedProcess(
            kind="slot",
            slot=0,
            process=slot_process,  # type: ignore[arg-type]
            log_path=slot_log_path,
            log_handle=slot_log_handle,
            start_ticks=1001,
        )
    ]
    scheduler.sam_processes = [
        OwnedProcess(
            kind="sam3",
            slot=0,
            process=sam_process,  # type: ignore[arg-type]
            log_path=sam_log_path,
            log_handle=sam_log_handle,
            start_ticks=1002,
        )
    ]
    now = [0.0]
    final_persist_observations: list[tuple[int | None, int | None]] = []

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.monotonic",
        lambda: now[0],
    )

    def fake_sleep(seconds: float) -> None:
        now[0] += seconds
        if now[0] >= 0.25:
            slot_process.returncode = 130
        if now[0] >= 0.50:
            sam_process.returncode = 130

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.sleep",
        fake_sleep,
    )

    def fail_slot_stop() -> None:
        raise RuntimeError("synthetic slot stop/persist failure")

    def fail_sam3_stop() -> None:
        raise KeyboardInterrupt("synthetic SAM3 stop interruption")

    monkeypatch.setattr(scheduler, "stop_slots", fail_slot_stop)
    monkeypatch.setattr(scheduler, "stop_sam3", fail_sam3_stop)
    original_persist = scheduler.persist

    def observe_final_persist(state: str | None = None) -> None:
        if state == "interrupted":
            final_persist_observations.append(
                (slot_process.poll(), sam_process.poll())
            )
        original_persist(state)

    monkeypatch.setattr(scheduler, "persist", observe_final_persist)

    with pytest.raises(
        RuntimeError, match="parallel scheduler cleanup encountered an error"
    ) as exc_info:
        scheduler.shutdown("interrupted")

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert final_persist_observations == [(130, 130)]
    assert slot_process.signals == [signal.SIGINT]
    assert sam_process.signals == [signal.SIGINT]
    assert slot_process.terminate_calls == 0
    assert sam_process.terminate_calls == 0
    assert slot_process.kill_calls == 0
    assert sam_process.kill_calls == 0
    assert slot_process.wait_calls == 1
    assert sam_process.wait_calls == 1
    assert slot_log_handle.closed
    assert sam_log_handle.closed
    assert scheduler.manifest["state"] == "interrupted"


def test_gpu_lock_set_excludes_other_batches_and_releases_without_unlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "gpu-locks"
    first = GpuLockSet(root, [3, 1], "first")
    records = first.acquire()

    assert [record["gpu_id"] for record in records] == [1, 3]
    assert [gpu for gpu, _fd in first.acquired] == [1, 3]
    for gpu in (1, 3):
        metadata = json.loads((root / f"gpu_{gpu}.lock").read_text(encoding="utf-8"))
        assert metadata["token"] == "first"
        assert metadata["pid"] == os.getpid()
        # Some test sandboxes translate Python's namespace PID and deny the
        # corresponding /proc entry; flock remains the authoritative lease.
        assert "start_ticks" in metadata
        assert "boot_id" in metadata

    second = GpuLockSet(root, [1], "second")
    with pytest.raises(ConfigError, match=r"GPU 1 has a (?:live|held) cross-batch lock"):
        second.acquire()
    assert second.acquired == []

    first.release()
    # The stable inode remains; flock ownership, not unlinking metadata, is
    # what makes stale locks safely reclaimable without inode races.
    assert (root / "gpu_1.lock").is_file()
    second_records = second.acquire()
    assert second_records[0]["owner"]["token"] == "second"
    second.release()


def test_gpu_lock_partial_acquire_failure_rolls_back_sorted_prefix(tmp_path: Path) -> None:
    root = tmp_path / "gpu-locks"
    blocker = GpuLockSet(root, [2], "blocker")
    blocker.acquire()
    contender = GpuLockSet(root, [3, 1, 2], "contender")

    with pytest.raises(ConfigError, match=r"GPU 2"):
        contender.acquire()

    assert contender.acquired == []
    probe = GpuLockSet(root, [1], "probe")
    probe.acquire()
    probe.release()
    blocker.release()


def test_gpu_lock_file_descriptor_is_not_inheritable(tmp_path: Path) -> None:
    locks = GpuLockSet(tmp_path / "gpu-locks", [0], "owner")
    locks.acquire()
    try:
        assert os.get_inheritable(locks.acquired[0][1]) is False
    finally:
        locks.release()


def test_gpu_lock_release_does_not_overwrite_successor_metadata(tmp_path: Path) -> None:
    root = tmp_path / "gpu-locks"
    owner = GpuLockSet(root, [0], "owner")
    owner.acquire()
    lock_path = root / "gpu_0.lock"
    owner.release()
    successor = GpuLockSet(root, [0], "successor")
    successor.acquire()
    before = lock_path.read_text(encoding="utf-8")

    owner.release()

    assert lock_path.read_text(encoding="utf-8") == before
    assert json.loads(before)["token"] == "successor"
    successor.release()


def test_gpu_occupancy_rejects_compute_and_graphics_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _jobs = _scheduler_config(monkeypatch, tmp_path)

    def fake_smi(arguments: list[str]) -> str:
        query = " ".join(arguments)
        if "query-gpu" in query:
            return "0, GPU-0\n1, GPU-1\n2, GPU-2\n3, GPU-3\n4, GPU-4\n5, GPU-5\n6, GPU-6\n7, GPU-7\n"
        if "query-compute-apps" in query:
            return "GPU-4, 111, compute-job\n"
        if arguments and arguments[0] == "pmon":
            gpu = arguments[arguments.index("-i") + 1]
            return (
                "# gpu pid type sm mem enc dec command\n5 222 G - - - - graphics-job\n"
                if gpu == "5"
                else "# no active graphics clients\n"
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.run_nvidia_smi",
        fake_smi,
    )

    with pytest.raises(ConfigError, match=r"external compute/graphics.*111.*222"):
        preflight_gpu_occupancy(config)


def test_gpu_occupancy_diagnostic_override_records_external_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOW_OCCUPIED_GPU", "1")
    config, _jobs = _scheduler_config(monkeypatch, tmp_path)

    def fake_smi(arguments: list[str]) -> str:
        query = " ".join(arguments)
        if "query-gpu" in query:
            return "\n".join(f"{gpu}, GPU-{gpu}" for gpu in range(8)) + "\n"
        if "query-compute-apps" in query:
            return "GPU-4, 111, compute-job\n"
        if arguments and arguments[0] == "pmon":
            return "# no active graphics clients\n"
        raise AssertionError(arguments)

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.run_nvidia_smi",
        fake_smi,
    )

    result = preflight_gpu_occupancy(config)

    assert result["passed"] is True
    assert result["allow_occupied_gpu"] is True
    assert result["occupied_gpu_ids"] == [4]
    assert result["processes_by_gpu"]["4"][0]["pid"] == 111


def _fake_proc_identity(
    proc_root: Path,
    pid: int,
    *,
    argv: list[str],
    cwd: Path,
    executable: Path,
    start_ticks: int = 12345,
) -> None:
    proc_dir = proc_root / str(pid)
    proc_dir.mkdir(parents=True)
    # proc_start_ticks indexes field 22 overall, or offset 19 after state.
    stat_tail = ["S", *("0" for _ in range(18)), str(start_ticks), *("0" for _ in range(4))]
    (proc_dir / "stat").write_text(
        f"{pid} (python worker) " + " ".join(stat_tail) + "\n",
        encoding="utf-8",
    )
    (proc_dir / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")
    (proc_dir / "cwd").symlink_to(cwd, target_is_directory=True)
    (proc_dir / "exe").symlink_to(executable)


def test_causalwam_identity_requires_real_code_path_under_exact_root(tmp_path: Path) -> None:
    root = tmp_path / "CausalWAM"
    script = root / "FastWAM" / "experiments" / "worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("# test\n", encoding="utf-8")
    executable = tmp_path / "python"
    executable.write_text("", encoding="utf-8")
    proc_root = tmp_path / "proc"
    _fake_proc_identity(
        proc_root,
        411,
        argv=[str(executable), str(script), "--task", "demo"],
        cwd=root,
        executable=executable,
    )

    identity = inspect_causalwam_process(411, root, proc_root=proc_root)

    assert identity["verified"] is True
    assert identity["matched_code_path"] == str(script.resolve())
    assert identity["reason"] == "exact_repo_code_path"
    assert len(identity["cmdline_sha256"]) == 64


def test_causalwam_name_substring_without_repo_code_path_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "CausalWAM"
    root.mkdir()
    outside = tmp_path / "outside" / "worker.py"
    outside.parent.mkdir()
    outside.write_text("# test\n", encoding="utf-8")
    executable = tmp_path / "python"
    executable.write_text("", encoding="utf-8")
    proc_root = tmp_path / "proc"
    _fake_proc_identity(
        proc_root,
        412,
        argv=[str(executable), str(outside), "--name", "CausalWAM worker"],
        cwd=tmp_path,
        executable=executable,
    )

    identity = inspect_causalwam_process(412, root, proc_root=proc_root)

    assert identity["verified"] is False
    assert identity["reason"] == "no_causalwam_code_path_in_cmdline"


def test_gpu_occupancy_ignores_only_explicit_verified_causalwam_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_OCCUPANCY_ALLOWED_CAUSALWAM_PIDS_CSV", "411")
    config, _jobs = _scheduler_config(monkeypatch, tmp_path)
    verified_identity = {
        "verified": True,
        "reason": "explicit_pid_and_exact_repo_code_path",
        "pid": 411,
        "start_ticks": 12345,
        "matched_code_path": "/repo/CausalWAM/FastWAM/worker.py",
        "cmdline_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.inspect_causalwam_process",
        lambda pid, root: verified_identity if pid == 411 else pytest.fail((pid, root)),
    )

    def fake_smi(arguments: list[str]) -> str:
        query = " ".join(arguments)
        if "query-gpu" in query:
            return "\n".join(f"{gpu}, GPU-{gpu}" for gpu in range(8)) + "\n"
        if "query-compute-apps" in query:
            return "GPU-4, 411, python\n"
        if arguments and arguments[0] == "pmon":
            return "# no active graphics clients\n"
        raise AssertionError(arguments)

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.run_nvidia_smi", fake_smi
    )

    result = preflight_gpu_occupancy(config)

    assert result["passed"] is True
    assert result["blocking_processes_by_gpu"] == {}
    ignored = result["ignored_causalwam_processes_by_gpu"]["4"][0]
    assert ignored["pid"] == 411
    assert ignored["occupancy_disposition"] == "ignored_explicit_causalwam"
    assert ignored["causalwam_identity"] == verified_identity


@pytest.mark.parametrize("configured_pid", ["", "411"])
def test_gpu_occupancy_rejects_unlisted_or_identity_mismatched_causalwam(
    configured_pid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_OCCUPANCY_ALLOWED_CAUSALWAM_PIDS_CSV", configured_pid)
    config, _jobs = _scheduler_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.inspect_causalwam_process",
        lambda pid, root: {
            "verified": False,
            "reason": "no_causalwam_code_path_in_cmdline",
            "pid": pid,
        },
    )

    def fake_smi(arguments: list[str]) -> str:
        query = " ".join(arguments)
        if "query-gpu" in query:
            return "\n".join(f"{gpu}, GPU-{gpu}" for gpu in range(8)) + "\n"
        if "query-compute-apps" in query:
            return "GPU-4, 411, CausalWAM-worker\n"
        if arguments and arguments[0] == "pmon":
            return "# no active graphics clients\n"
        raise AssertionError(arguments)

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.run_nvidia_smi", fake_smi
    )

    expected = (
        "blocked_identity_mismatch"
        if configured_pid
        else "blocked_not_verified_causalwam"
    )
    with pytest.raises(ConfigError, match=expected):
        preflight_gpu_occupancy(config)


def test_gpu_occupancy_auto_ignores_verified_causalwam_without_pid_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GPU_OCCUPANCY_ALLOWED_CAUSALWAM_PIDS_CSV", raising=False)
    config, _jobs = _scheduler_config(monkeypatch, tmp_path)
    verified_identity = {
        "verified": True,
        "reason": "explicit_pid_and_exact_repo_code_path",
        "pid": 411,
        "start_ticks": 12345,
        "matched_code_path": "/repo/CausalWAM/FastWAM/scripts/train.py",
        "cmdline_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.inspect_causalwam_process",
        lambda pid, root: verified_identity if pid == 411 else pytest.fail((pid, root)),
    )

    def fake_smi(arguments: list[str]) -> str:
        query = " ".join(arguments)
        if "query-gpu" in query:
            return "\n".join(f"{gpu}, GPU-{gpu}" for gpu in range(8)) + "\n"
        if "query-compute-apps" in query:
            return "GPU-4, 411, python\n"
        if arguments and arguments[0] == "pmon":
            return "# no active graphics clients\n"
        raise AssertionError(arguments)

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.run_nvidia_smi", fake_smi
    )

    result = preflight_gpu_occupancy(config)

    assert result["passed"] is True
    assert result["blocking_processes_by_gpu"] == {}
    assert result["causalwam_exception_policy"]["auto_verify_all_occupants"] is True
    ignored = result["ignored_causalwam_processes_by_gpu"]["4"][0]
    assert ignored["occupancy_disposition"] == "ignored_auto_verified_causalwam"


def test_gpu_occupancy_can_disable_auto_verified_causalwam_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOW_VERIFIED_CAUSALWAM_GPU_OCCUPANCY", "0")
    config, _jobs = _scheduler_config(monkeypatch, tmp_path)

    def fake_smi(arguments: list[str]) -> str:
        query = " ".join(arguments)
        if "query-gpu" in query:
            return "\n".join(f"{gpu}, GPU-{gpu}" for gpu in range(8)) + "\n"
        if "query-compute-apps" in query:
            return "GPU-4, 411, python\n"
        if arguments and arguments[0] == "pmon":
            return "# no active graphics clients\n"
        raise AssertionError(arguments)

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.run_nvidia_smi", fake_smi
    )
    with pytest.raises(ConfigError, match="blocked_not_allowlisted"):
        preflight_gpu_occupancy(config)


def test_gpu_occupancy_parse_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _jobs = _scheduler_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.run_nvidia_smi",
        lambda _arguments: "malformed inventory row",
    )

    with pytest.raises(ConfigError, match="parse"):
        preflight_gpu_occupancy(config)


def test_foreground_occupancy_failure_releases_locks_without_masking_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    lock_root = tmp_path / "host-gpu-locks"
    config = config.__class__(**{**config.__dict__, "gpu_lock_root": lock_root})
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_paths",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_ports",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_agent",
        lambda _config: {"status": "ok"},
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_gpus",
        lambda _config: _fake_gpu_inventory(),
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_gpu_occupancy",
        lambda _config: (_ for _ in ()).throw(ConfigError("synthetic occupancy failure")),
    )

    with pytest.raises(ConfigError, match="synthetic occupancy failure"):
        run_foreground(config, jobs)

    manifest = json.loads(
        (config.batch_log_root / "parallel_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "failed"
    assert manifest["gpu_occupancy_preflight"]["passed"] is False
    assert "synthetic occupancy failure" in manifest["gpu_occupancy_preflight"]["error"]
    probe = GpuLockSet(lock_root, list(range(8)), "probe")
    probe.acquire()
    probe.release()


def test_foreground_retries_failed_final_drain_before_releasing_gpu_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    process = _DelayedSigintProcess()
    log_path = tmp_path / "slow-foreground-child.log"
    log_handle = log_path.open("w", encoding="utf-8")
    events: list[tuple[str, Any]] = []
    now = [0.0]
    drain_calls = [0]

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_paths",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_ports",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_agent",
        lambda _config: {"status": "ok"},
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_gpus",
        lambda _config: _fake_gpu_inventory(),
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_gpu_occupancy",
        lambda _config: {"passed": True},
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.GpuLockSet.acquire",
        lambda _self: events.append(("lock_acquired", process.poll())) or [],
    )

    def observe_release(_self: GpuLockSet) -> None:
        events.append(("lock_released", process.poll()))

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.GpuLockSet.release",
        observe_release,
    )

    def launch_slow_child(scheduler: Scheduler) -> None:
        scheduler.sam_processes.append(
            OwnedProcess(
                kind="sam3",
                slot=0,
                process=process,  # type: ignore[arg-type]
                log_path=log_path,
                log_handle=log_handle,
                start_ticks=4242,
            )
        )
        events.append(("child_launched", process.poll()))

    monkeypatch.setattr(Scheduler, "launch_sam3", launch_slow_child)

    def fail_launch_slots(_scheduler: Scheduler) -> None:
        raise ValueError("synthetic launch failure")

    monkeypatch.setattr(Scheduler, "launch_slots", fail_launch_slots)

    def fail_first_shutdown(_scheduler: Scheduler, _state: str) -> None:
        events.append(("shutdown_failed", process.poll()))
        raise RuntimeError("synthetic shutdown failure")

    monkeypatch.setattr(Scheduler, "shutdown", fail_first_shutdown)
    original_drain = Scheduler.drain_owned_before_unlock

    def fail_then_drain(scheduler: Scheduler) -> None:
        drain_calls[0] += 1
        events.append((f"drain_{drain_calls[0]}", process.poll()))
        if drain_calls[0] == 1:
            raise KeyboardInterrupt("synthetic first final-drain interruption")
        original_drain(scheduler)

    monkeypatch.setattr(Scheduler, "drain_owned_before_unlock", fail_then_drain)
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.monotonic",
        lambda: now[0],
    )

    def release_child_after_retry(seconds: float) -> None:
        now[0] += seconds
        # The first 0.25 s sleep follows the failed final drain.  Keeping the
        # child alive through that sleep proves the outer pre-unlock loop must
        # enter and execute a second drain attempt.
        if now[0] >= 0.50 and process.returncode is None:
            process.returncode = 130
            events.append(("child_exited", process.returncode))

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.time.sleep",
        release_child_after_retry,
    )

    with pytest.raises(RuntimeError, match="synthetic shutdown failure"):
        run_foreground(config, jobs)

    assert drain_calls == [2]
    assert events.index(("drain_1", None)) < events.index(("drain_2", None))
    assert events.index(("drain_2", None)) < events.index(("child_exited", 130))
    assert events[-1] == ("lock_released", 130)
    assert process.signals == [signal.SIGINT]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.wait_calls == 1
    assert log_handle.closed


def test_foreground_installs_and_restores_sigint_shutdown_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _scheduler_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_paths",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_ports",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.preflight_agent",
        lambda _config: {"status": "ok"},
    )
    installed: list[tuple[int, Any]] = []
    current_handlers: dict[int, Any] = {
        signal.SIGINT: "original-int",
        signal.SIGTERM: "original-term",
        signal.SIGHUP: "original-hup",
    }

    def fake_signal(sig: int, handler: Any) -> Any:
        previous = current_handlers[sig]
        current_handlers[sig] = handler
        installed.append((sig, handler))
        return previous

    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.signal.signal",
        fake_signal,
    )
    monkeypatch.setattr(
        "policy.roboharn_evo.scripts.run_gpt55_parallel4_isolated.GpuLockSet.acquire",
        lambda _self: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert run_foreground(config, jobs) == 130
    assert any(sig == signal.SIGINT and callable(handler) for sig, handler in installed)
    assert current_handlers == {
        signal.SIGINT: "original-int",
        signal.SIGTERM: "original-term",
        signal.SIGHUP: "original-hup",
    }
