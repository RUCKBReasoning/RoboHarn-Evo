import os
import sys
import subprocess
import json
import hashlib
import math
import signal
import tempfile
import time

from benchmarks.rmbench.envs import CONFIGS_PATH
from benchmarks.rmbench.envs.renderer_device_contract import RendererDeviceContractError
from benchmarks.rmbench.envs.utils.create_actor import UnStableError
from benchmarks.rmbench.paths import (
    benchmark_root,
    output_root as configured_output_root,
    resolve_asset_reference,
    task_config_root,
)

import numpy as np
from pathlib import Path
import traceback

import yaml
from datetime import datetime, timezone
import importlib
import importlib.util
import argparse
import contextlib

from benchmarks.rmbench.description.utils.generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
BENCHMARK_ROOT = benchmark_root()


DEBUG_MIGRATION_SMOKE_CKPT = "debug_migration_smoke"
_TERMINATION_SIGNAL_NAME = ""
_FORMAL_PROTOCOL_ENV = "ROBOHARN_EVO_FORMAL_PROTOCOL"
_FORMAL_PROTOCOL_VERSION_ENV = "ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION"
_RUNTIME_PROVENANCE_PATH_ENV = "ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH"
_CURRENT_FORMAL_PROTOCOL_VERSION = 3


class ExactSeedValidationError(RuntimeError):
    """Raised when an opt-in exact seed fails expert validation."""


def _handle_termination_signal(signum, _frame):
    global _TERMINATION_SIGNAL_NAME
    try:
        _TERMINATION_SIGNAL_NAME = signal.Signals(signum).name.lower()
    except (TypeError, ValueError):
        _TERMINATION_SIGNAL_NAME = f"signal_{signum}"
    raise KeyboardInterrupt


def _install_termination_signal_handlers():
    for signal_name in ("SIGHUP", "SIGTERM"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            signal.signal(signum, _handle_termination_signal)


def _is_debug_migration_smoke(args):
    return str(args.get("ckpt_setting", "")) == DEBUG_MIGRATION_SMOKE_CKPT


def _resolve_test_num(args, usr_args):
    if _is_debug_migration_smoke(args):
        return 1
    eval_cfg = usr_args.get("eval", {})
    if isinstance(eval_cfg, dict) and eval_cfg.get("test_num") is not None:
        return max(1, int(eval_cfg["test_num"]))
    return 100


def _resolve_start_seed(usr_args):
    eval_cfg = usr_args.get("eval", {})
    if isinstance(eval_cfg, dict) and eval_cfg.get("start_seed") is not None:
        start_seed = int(eval_cfg["start_seed"])
    else:
        start_seed = 100000 * (1 + int(usr_args["seed"]))
    if start_seed < 0:
        raise ValueError(f"eval.start_seed must be non-negative, got {start_seed}")
    return start_seed


def _resolve_exact_seed_fail_closed(usr_args):
    """Return the explicit eval-level exact-seed policy.

    The default is deliberately false so all existing launchers preserve their
    historical behavior of advancing past an invalid expert seed.  Paired
    integration smokes opt in with ``--eval.exact_seed_fail_closed True``.
    """

    eval_cfg = usr_args.get("eval", {})
    raw_value = (
        eval_cfg.get("exact_seed_fail_closed", False)
        if isinstance(eval_cfg, dict)
        else False
    )
    if not isinstance(raw_value, bool):
        raise ValueError("eval.exact_seed_fail_closed must be a boolean")
    return raw_value


def _resolve_eval_step_limit(usr_args):
    """Return an optional positive per-episode environment-step cap."""
    eval_cfg = usr_args.get("eval", {})
    if not isinstance(eval_cfg, dict):
        return None
    raw_value = eval_cfg.get("step_limit")
    if raw_value is None or str(raw_value).strip() == "":
        return None
    if isinstance(raw_value, bool):
        raise ValueError("eval.step_limit must be a positive integer")
    try:
        step_limit = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"eval.step_limit must be a positive integer, got {raw_value!r}"
        ) from exc
    if step_limit <= 0:
        raise ValueError(
            f"eval.step_limit must be a positive integer, got {step_limit}"
        )
    return step_limit


def _apply_eval_step_limit(task_env, requested_step_limit, *, mode="cap"):
    """Cap the native horizon by default; explicit custom-budget runs may override."""
    if mode not in {"cap", "override"}:
        raise ValueError("eval.step_limit_mode must be cap or override")
    if mode == "override" and requested_step_limit is None:
        raise ValueError("eval.step_limit_mode=override requires eval.step_limit")
    raw_configured_limit = getattr(task_env, "step_lim", None)
    try:
        configured_step_limit = int(raw_configured_limit)
    except (TypeError, ValueError):
        configured_step_limit = 0

    if requested_step_limit is None:
        effective_step_limit = configured_step_limit
    elif mode == "override":
        effective_step_limit = int(requested_step_limit)
    elif configured_step_limit > 0:
        effective_step_limit = min(configured_step_limit, int(requested_step_limit))
    else:
        effective_step_limit = int(requested_step_limit)

    if effective_step_limit <= 0:
        raise ValueError(
            "the effective environment step limit must be positive; "
            f"configured={raw_configured_limit!r}, requested={requested_step_limit!r}"
        )
    task_env.step_lim = effective_step_limit
    return {
        "requested": requested_step_limit,
        "task_default": configured_step_limit if configured_step_limit > 0 else None,
        "effective": effective_step_limit,
        "mode": mode,
    }


def class_decorator(task_name):
    envs_module = importlib.import_module(f"benchmarks.rmbench.envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def _deep_update_config(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update_config(target[key], value)
        else:
            target[key] = value
    return target


def _merge_task_config_overrides(task_args, usr_args):
    """Apply CLI/deploy overrides that belong to the RMBench task config."""
    applied = {}
    for key, value in usr_args.items():
        if key not in task_args or value is None:
            continue
        if isinstance(value, dict) and isinstance(task_args.get(key), dict):
            _deep_update_config(task_args[key], value)
        else:
            task_args[key] = value
        applied[key] = task_args[key]
    return applied


def get_camera_config(camera_type):
    camera_config_path = task_config_root() / "_camera_config.yml"
    assert os.path.isfile(camera_config_path), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def _write_agent_trace_event(trace_file: Path | None, event: str, **payload):
    record = {
        "event": event,
        "timestamp": time.time(),
        **payload,
    }
    print("[eval] " + json.dumps(record, ensure_ascii=False), flush=True)
    if trace_file is None:
        return
    try:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        with trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[eval] trace_write_warning={e}", flush=True)


def _is_sha256(value):
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _load_content_provenance_module():
    """Load the copied canonical recorder without importing the heavy policy package."""

    recorder_path = (
        BENCHMARK_ROOT / "policy" / "roboharn_evo" / "scripts" / "record_runtime_provenance.py"
    ).resolve()
    if not recorder_path.is_file():
        raise RuntimeError(
            f"canonical runtime provenance recorder is missing: {recorder_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "_roboharn_evo_rmbench_content_runtime_provenance",
        recorder_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load canonical runtime provenance recorder: {recorder_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_archive_runtime_provenance(archive_path: Path, manifest_bytes: bytes):
    """Atomically replace the archive path itself, never a symlink target."""

    archive_path = Path(archive_path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=archive_path.parent,
            prefix=f".{archive_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(manifest_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, archive_path)
    except OSError as exc:
        raise RuntimeError(
            f"cannot archive runtime provenance at {archive_path}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _archive_runtime_provenance(save_dir: Path):
    protocol_value = str(os.environ.get(_FORMAL_PROTOCOL_ENV, "0") or "0").strip()
    if protocol_value not in {"0", "1"}:
        raise ValueError(
            f"{_FORMAL_PROTOCOL_ENV} must be 0 or 1, got {protocol_value!r}"
        )
    formal_protocol = protocol_value == "1"
    protocol_version_value = str(
        os.environ.get(_FORMAL_PROTOCOL_VERSION_ENV, "0") or "0"
    ).strip()
    try:
        formal_protocol_version = int(protocol_version_value)
    except ValueError as exc:
        raise ValueError(
            f"{_FORMAL_PROTOCOL_VERSION_ENV} must be an integer, "
            f"got {protocol_version_value!r}"
        ) from exc
    if formal_protocol and formal_protocol_version != _CURRENT_FORMAL_PROTOCOL_VERSION:
        raise RuntimeError(
            "formal protocol requires "
            f"{_FORMAL_PROTOCOL_VERSION_ENV}={_CURRENT_FORMAL_PROTOCOL_VERSION}, "
            f"got {formal_protocol_version}"
        )
    if not formal_protocol and formal_protocol_version != 0:
        raise RuntimeError(
            f"non-formal execution requires {_FORMAL_PROTOCOL_VERSION_ENV}=0 or unset, "
            f"got {formal_protocol_version}"
        )
    source_value = str(os.environ.get(_RUNTIME_PROVENANCE_PATH_ENV, "") or "").strip()
    if not source_value:
        if formal_protocol:
            raise RuntimeError("formal protocol requires a runtime provenance manifest")
        return {
            "formal_protocol": False,
            "recorded": False,
        }

    source = Path(source_value).expanduser().resolve()
    try:
        manifest_bytes = source.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"cannot read runtime provenance manifest: {source}"
        ) from exc
    provenance_module = _load_content_provenance_module()
    try:
        payload = provenance_module.decode_json_object(
            manifest_bytes,
            label=f"runtime provenance manifest {source}",
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"runtime provenance manifest is not strict UTF-8 JSON: {source}: {exc}"
        ) from exc

    schema = str(payload.get("schema", "") or "").strip()
    if schema:
        if schema not in {provenance_module.RUNTIME_PROVENANCE_SCHEMA, "tcm/rmbench_runtime_provenance/content/v2"}:
            raise RuntimeError(f"unsupported runtime provenance schema: {schema!r}")
        project_root = BENCHMARK_ROOT.parents[1].resolve()
        declared_root_value = str(payload.get("repository_root", "") or "").strip()
        declared_root = Path(declared_root_value).expanduser().resolve()
        if formal_protocol:
            if declared_root != project_root:
                raise RuntimeError(
                    "formal content runtime provenance must be rooted at the "
                    f"RoboHarn-Evo project: {declared_root} vs {project_root}"
                )
            provenance_root = project_root
        else:
            copied_root = BENCHMARK_ROOT.resolve()
            if declared_root not in {project_root, copied_root}:
                raise RuntimeError(
                    "content runtime provenance root must be the RoboHarn-Evo project "
                    f"or copied benchmark: {declared_root}"
                )
            provenance_root = declared_root
        try:
            provenance_module.validate_output_path(
                provenance_root,
                source,
                formal_protocol=formal_protocol,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"invalid runtime provenance source path: {exc}"
            ) from exc
        expected_config = provenance_module.effective_config_from_environment(
            os.environ
        )
        try:
            validated = provenance_module.validate_runtime_provenance(
                payload,
                provenance_root,
                verify_files=True,
                # The scheduler freezes service evidence before starting a
                # slot.  Validate the embedded bytes/hash binding here; do
                # not turn later external-file mutation into a rollout rule.
                verify_external_evidence=False,
                expected_effective_config=expected_config,
                required_runtime_paths=(
                    provenance_module.DEFAULT_RUNTIME_PATHS if formal_protocol else ()
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid content runtime provenance: {exc}") from exc
        content_manifest = validated["runtime_content_manifest"]
        recorded_config = validated["effective_config"]
        try:
            archive_path = provenance_module.validate_output_path(
                provenance_root,
                save_dir / "runtime_provenance.json",
                formal_protocol=formal_protocol,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"invalid runtime provenance archive path: {exc}"
            ) from exc
        _atomic_archive_runtime_provenance(archive_path, manifest_bytes)
        summary = {
            "formal_protocol": formal_protocol,
            "formal_protocol_version": formal_protocol_version,
            "recorded": True,
            "archive_file": archive_path.name,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "schema": schema,
            "source_identity_mode": validated["source_identity_mode"],
            "runtime_tree_sha256": validated["runtime_tree_sha256"],
            "runtime_file_count": validated["runtime_file_count"],
            "effective_config_sha256": validated["effective_config_sha256"],
            "task": recorded_config["task_name"],
            "eval_start_seed": recorded_config["eval_start_seed"],
            "eval_start_seeds": recorded_config["eval_start_seeds"],
            "content_hash_algorithm": content_manifest["hash_algorithm"],
        }
        if "agent_service_identity_sha256" in validated:
            summary["agent_service_identity_sha256"] = validated[
                "agent_service_identity_sha256"
            ]
            summary["agent_service_identity_canonical_sha256"] = validated[
                "agent_service_identity_canonical_sha256"
            ]
        if "external_evidence" in validated:
            summary["external_evidence_bytes_sha256"] = {
                label: record["bytes_sha256"]
                for label, record in sorted(validated["external_evidence"].items())
            }
        return summary

    if formal_protocol:
        raise RuntimeError(
            "formal protocol requires the canonical content-based runtime "
            "provenance schema; legacy Git provenance is not accepted"
        )

    required_hashes = (
        "runtime_tree_sha256",
        "tracked_runtime_diff_sha256",
    )
    invalid_hashes = [
        key for key in required_hashes if not _is_sha256(payload.get(key))
    ]
    if invalid_hashes:
        raise RuntimeError(
            "runtime provenance manifest has invalid required hashes: "
            + ", ".join(invalid_hashes)
        )
    git_head = str(payload.get("git_head", "") or "").strip()
    if not git_head:
        raise RuntimeError("runtime provenance manifest is missing git_head")
    try:
        runtime_file_count = int(payload.get("runtime_file_count", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "runtime provenance manifest has invalid runtime_file_count"
        ) from exc
    if runtime_file_count <= 0:
        raise RuntimeError("runtime provenance manifest contains no runtime files")
    if payload.get("secrets_recorded") is not False:
        raise RuntimeError(
            "runtime provenance manifest does not affirm secrets_recorded=false"
        )

    service_identity_summary = None
    service_identity = payload.get("agent_service_identity")
    service_identity_sha256 = payload.get("agent_service_identity_sha256")
    if service_identity is not None or service_identity_sha256 is not None:
        if not isinstance(service_identity, dict):
            raise RuntimeError(
                "runtime provenance agent_service_identity must be an object"
            )
        if not _is_sha256(service_identity_sha256):
            raise RuntimeError(
                "runtime provenance has invalid agent_service_identity_sha256"
            )
        service_identity_summary = {
            key: service_identity.get(key)
            for key in (
                "service_url",
                "backend",
                "provider",
                "model",
                "api_mode",
                "reasoning_effort",
                "thinking_mode",
                "fallback_enabled",
            )
        }

    archive_path = save_dir / "runtime_provenance.json"
    _atomic_archive_runtime_provenance(archive_path, manifest_bytes)
    summary = {
        "formal_protocol": formal_protocol,
        "formal_protocol_version": formal_protocol_version,
        "recorded": True,
        "archive_file": archive_path.name,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "runtime_tree_sha256": str(payload["runtime_tree_sha256"]).lower(),
        "tracked_runtime_diff_sha256": str(
            payload["tracked_runtime_diff_sha256"]
        ).lower(),
        "git_head": git_head,
        "git_dirty_for_runtime_paths": bool(
            payload.get("git_dirty_for_runtime_paths", False)
        ),
        "runtime_file_count": runtime_file_count,
    }
    if service_identity_summary is not None:
        summary["agent_service_identity_sha256"] = str(service_identity_sha256).lower()
        summary["agent_service_identity"] = service_identity_summary
    return summary


def _extract_agent_status(model):
    status = {}
    try:
        if hasattr(model, "session") and hasattr(model.session, "agent"):
            status = model.session.agent.current_status()
        elif hasattr(model, "memory_store"):
            state = model.memory_store.state
            status = {
                "active_skill": None
                if state.active_skill is None
                else state.active_skill.skill_name,
                "monitor_phase": state.monitor.phase,
                "monitor_status": state.monitor.status,
                "recovery_pending": state.recovery.pending_action,
                "task_finished": state.task.task_finished,
            }
    except Exception:
        pass
    return status


_INFRASTRUCTURE_TERMINAL_REASON_PREFIXES = (
    "pure_tool_control_control_backend_unavailable:",
    "pure_tool_control_recovery_backend_unavailable:",
    "pure_tool_control_resource_quota_exhausted:",
)
_CAPABILITY_TERMINAL_REASON_PREFIXES = (
    "pure_tool_control_max_control_turns_exhausted:",
    "pure_tool_control_max_no_progress_control_turns_exhausted:",
    "pure_tool_control_identity_binding_unresolved:",
)


def _acquire_eval_observation(
    task_env,
    model,
    trace_file: Path | None,
    *,
    episode_id: int,
    seed: int,
):
    consumer = getattr(model, "consume_recovery_observation", None)
    cached = None
    if callable(consumer):
        try:
            cached = consumer(task_env)
        except Exception as exc:
            _write_agent_trace_event(
                trace_file,
                "observation_handoff_error",
                episode_id=episode_id,
                seed=seed,
                step=int(getattr(task_env, "take_action_cnt", -1)),
                error=f"{type(exc).__name__}: {exc}",
            )
    if isinstance(cached, dict):
        _write_agent_trace_event(
            trace_file,
            "observation_acquire_cache_hit",
            episode_id=episode_id,
            seed=seed,
            step=int(getattr(task_env, "take_action_cnt", -1)),
            source="recovery_handoff",
        )
        return cached

    step = int(getattr(task_env, "take_action_cnt", -1))
    _write_agent_trace_event(
        trace_file,
        "observation_acquire_start",
        episode_id=episode_id,
        seed=seed,
        step=step,
        source="environment",
    )
    observation = task_env.get_obs()
    _write_agent_trace_event(
        trace_file,
        "observation_acquire_end",
        episode_id=episode_id,
        seed=seed,
        step=int(getattr(task_env, "take_action_cnt", -1)),
        source="environment",
    )
    return observation


def _classify_episode_validity(
    *,
    success: bool,
    interrupted: bool,
    episode_error: BaseException | None,
    agent_status: dict,
) -> dict:
    """Assign the mutually exclusive benchmark-attempt label from structured runtime state."""

    terminal_reason = str(agent_status.get("terminal_failure_reason", "") or "")
    if interrupted:
        label = "user_interrupted"
        reason = "The episode received an explicit external/user interrupt."
    elif episode_error is not None:
        label = "incomplete_or_corrupt"
        reason = "The eval loop raised an unhandled exception."
    elif success:
        label = "valid_success"
        reason = "RMBench eval_success ended the episode without interruption or an unhandled exception."
    elif terminal_reason.startswith(_INFRASTRUCTURE_TERMINAL_REASON_PREFIXES):
        label = "infrastructure_invalid"
        reason = f"A typed Agent API backend budget terminated the episode: {terminal_reason}"
    elif bool(
        agent_status.get("terminal_failure", False)
    ) and not terminal_reason.startswith(_CAPABILITY_TERMINAL_REASON_PREFIXES):
        label = "incomplete_or_corrupt"
        reason = f"An unclassified runtime terminal failure requires artifact audit: {terminal_reason or 'unknown'}"
    else:
        label = "valid_task_failure"
        reason = "The episode ended cleanly without RMBench eval_success or a blocking infrastructure failure."
    return {
        "label": label,
        "benchmark_denominator_eligible": label
        in {"valid_success", "valid_task_failure"},
        "task_success_numerator": label == "valid_success",
        "reason": reason,
    }


def _make_rollout_video(rollout_dir: Path, camera_name: str, fps: int = 10) -> None:
    from roboharn_evo.agent.paths import resolve_writable_path
    from roboharn_evo.agent.rollout_video import encode_rollout_video

    output_file = resolve_writable_path(rollout_dir / "video" / f"{camera_name}.mp4")
    encode_rollout_video(rollout_dir / camera_name, output_file, fps=fps)


def _start_continuous_rollout_video(
    task_env,
    rollout_dir: Path | None,
    trace_file: Path | None,
    *,
    episode_id: int,
    seed: int,
    fps: int = 30,
):
    if rollout_dir is None or not callable(
        getattr(task_env, "_set_eval_video_frame_callback", None)
    ):
        return None

    from roboharn_evo.agent.rollout_video import ContinuousRolloutVideoRecorder

    timestep = float(getattr(task_env, "simulation_timestep", 1 / 250))
    if not math.isfinite(timestep) or timestep <= 0:
        raise ValueError(f"invalid simulator timestep for video: {timestep!r}")
    simulation_hz = int(round(1.0 / timestep))
    if not math.isclose(
        simulation_hz * timestep,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "continuous video requires an integral simulator frequency, got "
            f"timestep={timestep!r}"
        )

    recorder = ContinuousRolloutVideoRecorder(
        rollout_dir,
        fps=fps,
        simulation_hz=simulation_hz,
    )
    recorder.attach(task_env)
    _write_agent_trace_event(
        trace_file,
        "continuous_rollout_video_start",
        episode_id=episode_id,
        seed=seed,
        fps=fps,
        simulation_hz=simulation_hz,
        cameras=["head", "left", "right", "third"],
    )
    return recorder


def _close_continuous_rollout_video(
    recorder,
    trace_file: Path | None,
    *,
    episode_id: int,
    seed: int,
) -> dict | None:
    if recorder is None:
        return None
    try:
        summary = recorder.close()
    except Exception as exc:
        _write_agent_trace_event(
            trace_file,
            "continuous_rollout_video_finalize_error",
            episode_id=episode_id,
            seed=seed,
            error=repr(exc),
        )
        return None
    event = (
        "continuous_rollout_video_finalize"
        if summary.get("complete") is True
        else "continuous_rollout_video_finalize_error"
    )
    _write_agent_trace_event(
        trace_file,
        event,
        episode_id=episode_id,
        seed=seed,
        **summary,
    )
    return summary


def _close_eval_video_writer(
    task_env, trace_file: Path | None, *, episode_id: int, seed: int, reason: str
) -> None:
    ffmpeg = getattr(task_env, "eval_video_ffmpeg", None)
    if ffmpeg is None:
        return
    error_messages = []
    write_error = getattr(task_env, "eval_video_write_error", None)
    if write_error is not None:
        error_messages.append(f"frame_write={write_error}")
    try:
        stdin = getattr(ffmpeg, "stdin", None)
        if stdin is not None and not stdin.closed:
            stdin.close()
    except Exception as e:
        error_messages.append(f"stdin_close={repr(e)}")
    try:
        ffmpeg.wait(timeout=15)
    except subprocess.TimeoutExpired:
        error_messages.append("wait_timeout=15s")
        try:
            ffmpeg.terminate()
            ffmpeg.wait(timeout=5)
        except Exception as e:
            error_messages.append(f"terminate={repr(e)}")
            try:
                ffmpeg.kill()
                ffmpeg.wait(timeout=5)
            except Exception as kill_error:
                error_messages.append(f"kill={repr(kill_error)}")
    except Exception as e:
        error_messages.append(f"wait={repr(e)}")
    try:
        delattr(task_env, "eval_video_ffmpeg")
    except Exception:
        pass
    event_payload = {
        "episode_id": episode_id,
        "seed": seed,
        "reason": reason,
        "returncode": getattr(ffmpeg, "returncode", None),
    }
    if error_messages:
        _write_agent_trace_event(
            trace_file,
            "eval_video_finalize_error",
            **event_payload,
            errors=error_messages,
        )
    else:
        _write_agent_trace_event(trace_file, "eval_video_finalize", **event_payload)


def _finalize_rollout_visualization(
    model,
    rollout_dir: Path | None,
    trace_file: Path | None,
    fps: int = 10,
    *,
    continuous_video_ready: bool = False,
) -> None:
    if rollout_dir is None:
        return
    try:
        if not continuous_video_ready:
            if hasattr(model, "session") and hasattr(model.session, "agent"):
                agent = model.session.agent
                agent.finalize_rollout_videos(fps=fps)
            else:
                for camera_name in ("head", "left", "right"):
                    _make_rollout_video(rollout_dir, camera_name, fps=fps)
        _write_agent_trace_event(
            trace_file, "rollout_video_finalize", rollout_dir=str(rollout_dir)
        )
    except Exception as e:
        _write_agent_trace_event(
            trace_file,
            "rollout_video_finalize_error",
            rollout_dir=str(rollout_dir),
            error=repr(e),
        )
    try:
        report_script = (
            BENCHMARK_ROOT
            / "policy"
            / "roboharn_evo"
            / "scripts"
            / "visualize_rollout_report.py"
        )
        run_dir = rollout_dir.parent
        subprocess.run(
            [
                sys.executable,
                str(report_script),
                "--input",
                str(run_dir),
                "--output",
                str(run_dir / "rollout_report.html"),
            ],
            check=False,
        )
        _write_agent_trace_event(
            trace_file,
            "rollout_report_finalize",
            report=str(run_dir / "rollout_report.html"),
        )
    except Exception as e:
        _write_agent_trace_event(
            trace_file,
            "rollout_report_finalize_error",
            rollout_dir=str(rollout_dir),
            error=repr(e),
        )
    try:
        contact_sheet_script = (
            BENCHMARK_ROOT
            / "policy"
            / "roboharn_evo"
            / "scripts"
            / "visualize_rollout_contact_sheet.py"
        )
        run_dir = rollout_dir.parent
        subprocess.run(
            [
                sys.executable,
                str(contact_sheet_script),
                "--input",
                str(run_dir),
                "--output",
                str(run_dir / "rollout_contact_sheet.png"),
            ],
            check=False,
        )
        _write_agent_trace_event(
            trace_file,
            "rollout_contact_sheet_finalize",
            report=str(run_dir / "rollout_contact_sheet.png"),
        )
    except Exception as e:
        _write_agent_trace_event(
            trace_file,
            "rollout_contact_sheet_finalize_error",
            rollout_dir=str(rollout_dir),
            error=repr(e),
        )


def _evolving_hpk_enabled(model) -> bool:
    runtime = getattr(model, "hpk_runtime", None)
    return runtime is not None and getattr(runtime, "evolving_enabled", False) is True


def _hierarchical_hpk_enabled(model) -> bool:
    runtime = getattr(model, "hpk_runtime", None)
    return (
        runtime is not None
        and getattr(runtime, "hierarchical_full_enabled", False) is True
        and getattr(runtime, "online_store_enabled", False) is True
    )


def _hierarchical_hpk_read_only_utility_gate(args: dict, model=None) -> bool:
    from roboharn_evo.agent.hpk.compatibility import normalize_agent_knowledge_config

    if getattr(model, "rmbench_formal_method", None) in {"task", "action", "full"}:
        return True
    formal = args.get("rmbench_formal")
    if isinstance(formal, dict) and set(formal) == {"config_path"}:
        return True
    agent = args.get("agent", {})
    raw_v3 = normalize_agent_knowledge_config(agent).get("hpk_v3", {}) if isinstance(agent, dict) else {}
    value = (
        raw_v3.get("rmbench_read_only_utility_gate", False)
        if isinstance(raw_v3, dict)
        else False
    )
    if not isinstance(value, bool):
        raise TypeError("agent.hpk_v3.rmbench_read_only_utility_gate must be a boolean")
    return value


def _resolve_failure_boundary_replay_config(args: dict) -> dict:
    agent = args.get("agent", {})
    raw = agent.get("failure_boundary_replay", {}) if isinstance(agent, dict) else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("agent.failure_boundary_replay must be a mapping")
    mode = str(raw.get("mode", "off") or "off").strip().lower()
    if mode not in {"off", "capture", "capture_bank", "restore"}:
        raise ValueError(
            "agent.failure_boundary_replay.mode must be off, capture, "
            "capture_bank, or restore"
        )
    if mode == "off":
        return {"mode": "off"}
    if mode == "capture_bank":
        from roboharn_evo.benchmark_adapters.rmbench.v31_boundary_bank import (
            RMBenchV31BoundaryBankConfig,
        )

        return RMBenchV31BoundaryBankConfig.from_mapping(raw).to_runtime_dict()
    path = str(raw.get("path", "") or "").strip()
    if not path:
        raise ValueError("agent.failure_boundary_replay.path must be explicit")
    result = {
        "mode": mode,
        "path": Path(path).expanduser().resolve(),
    }
    if mode == "capture":
        step = raw.get("capture_at_or_after_env_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(
                "capture_at_or_after_env_step must be a non-negative integer"
            )
        stop_after_capture = raw.get("stop_after_capture", True)
        if not isinstance(stop_after_capture, bool):
            raise TypeError("stop_after_capture must be a boolean")
        result.update(
            {
                "capture_at_or_after_env_step": step,
                "stop_after_capture": stop_after_capture,
                "label": str(
                    raw.get(
                        "label",
                        f"first quiescent boundary at or after environment step {step}",
                    )
                    or ""
                ).strip(),
            }
        )
        if not result["label"]:
            raise ValueError("failure boundary label must be non-empty")
    return result


def _restore_failure_boundary_before_episode(
    task_env,
    model,
    config: dict,
    *,
    task_name: str,
    seed: int,
    instruction: str,
) -> dict | None:
    if config.get("mode") != "restore":
        return None
    from roboharn_evo.agent.failure_boundary_replay import (
        load_failure_boundary,
        restore_failure_boundary,
    )

    boundary = load_failure_boundary(config["path"])
    report = restore_failure_boundary(
        task_env,
        model,
        boundary,
        expected_task=task_name,
        expected_seed=seed,
        expected_instruction=instruction,
    )
    return {
        "path": str(config["path"]),
        "label": boundary.label,
        "reason": boundary.reason,
        "boundary_env_step": boundary.environment["task_state"].get(
            "take_action_cnt", 0
        ),
        "match": report,
    }


def _capture_failure_boundary_after_turn(
    task_env,
    model,
    config: dict,
    *,
    task_name: str,
    seed: int,
    instruction: str,
    agent_status: dict,
    action_prefix,
    observation_calls_after_last_action: int,
):
    if config.get("mode") != "capture" or _task_env_step(task_env) < config.get(
        "capture_at_or_after_env_step", 0
    ):
        return None
    from roboharn_evo.agent.failure_boundary_replay import (
        capture_failure_boundary,
        save_failure_boundary,
    )

    reason = str(
        agent_status.get("terminal_failure_reason")
        or agent_status.get("failure_reason")
        or agent_status.get("monitor_status")
        or "failure boundary requested by the experiment"
    ).strip()
    boundary = capture_failure_boundary(
        task_env,
        model,
        task=task_name,
        seed=seed,
        instruction=instruction,
        label=config["label"],
        reason=reason,
        action_prefix=action_prefix,
        observation_calls_after_last_action=(observation_calls_after_last_action),
    )
    path = save_failure_boundary(config["path"], boundary)
    return {
        "path": str(path),
        "label": boundary.label,
        "reason": boundary.reason,
        "env_step": _task_env_step(task_env),
        "actions_recorded": len(boundary.action_prefix),
        "observation_calls_recorded": (
            sum(
                int(item["observation_calls_before"]) for item in boundary.action_prefix
            )
            + boundary.observation_calls_after_last_action
        ),
    }


def _start_failure_boundary_action_recorder(task_env, config: dict):
    if config.get("mode") not in {"capture", "capture_bank"}:
        return None
    from roboharn_evo.agent.failure_boundary_replay import ActionPrefixRecorder

    recorder = ActionPrefixRecorder(task_env)
    recorder.start()
    return recorder


def _task_env_step(task_env) -> int:
    return int(getattr(task_env, "take_action_cnt", 0))


def _failure_boundary_episode_summary(
    *,
    config: dict,
    restored_boundary: dict | None,
    captured_boundary: dict | None,
    task_name: str,
    seed: int,
    instruction: str,
    result: str,
    success: bool,
    failure_reason: str,
    final_env_step: int,
    max_reward: float,
    captured_boundaries: list[dict] | None = None,
) -> dict | None:
    mode = config.get("mode", "off")
    if mode == "restore" and restored_boundary is not None:
        boundary_step = int(restored_boundary.get("boundary_env_step", 0))
        return {
            "schema": "roboharn_evo/rmbench_failure_boundary_result/v1",
            "mode": "restore",
            "task": task_name,
            "seed": seed,
            "instruction": instruction,
            "boundary": {
                "path": restored_boundary["path"],
                "label": restored_boundary["label"],
                "reason": restored_boundary["reason"],
                "env_step": boundary_step,
                "state_match": restored_boundary["match"],
            },
            "outcome": {
                "result": result,
                "success": bool(success),
                "failure_reason": failure_reason,
                "final_env_step": final_env_step,
                "additional_environment_actions": max(
                    0, final_env_step - boundary_step
                ),
                "max_reward": float(max_reward),
            },
        }
    if mode == "capture" and captured_boundary is not None:
        return {
            "schema": "roboharn_evo/rmbench_failure_boundary_result/v1",
            "mode": "capture",
            "task": task_name,
            "seed": seed,
            "instruction": instruction,
            "boundary": dict(captured_boundary),
            "outcome": {
                "result": result,
                "success": bool(success),
                "failure_reason": failure_reason,
                "final_env_step": final_env_step,
                "max_reward": float(max_reward),
            },
        }
    if mode == "capture_bank":
        boundaries = [dict(item) for item in (captured_boundaries or [])]
        return {
            "schema": "roboharn_evo/rmbench_failure_boundary_bank_result/v1",
            "mode": "capture_bank",
            "task": task_name,
            "seed": seed,
            "instruction": instruction,
            "boundaries": boundaries,
            "capture_count": len(boundaries),
            "requested": {
                "min_boundaries": int(config["min_boundaries"]),
                "max_boundaries": int(config["max_boundaries"]),
            },
            "outcome": {
                "result": result,
                "success": bool(success),
                "failure_reason": failure_reason,
                "final_env_step": final_env_step,
                "max_reward": float(max_reward),
            },
        }
    return None


def _write_failure_boundary_episode_summary(
    rollout_dir: Path | None,
    summary: dict | None,
) -> Path | None:
    if rollout_dir is None or summary is None:
        return None
    destination = Path(rollout_dir) / "failure_boundary_result.json"
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def _finalize_hierarchical_hpk_episode(
    model,
    *,
    rollout_dir: Path | None,
    trace_file: Path | None,
    episode_id: int,
    seed: int,
    result: str,
):
    """Write one episode's v3 action evidence to the two-file Store."""

    if not _hierarchical_hpk_enabled(model):
        return None
    finalize = getattr(model, "finalize_hpk_v3_episode", None)
    if not callable(finalize):
        raise RuntimeError("HPK v3 model is missing its episode finalizer")
    if rollout_dir is None:
        raise RuntimeError("HPK v3 reflection requires a rollout directory")
    summary = finalize(rollout_dir=str(rollout_dir), result=result)
    _write_agent_trace_event(
        trace_file,
        "hpk_v3_episode_update",
        episode_id=episode_id,
        seed=seed,
        **summary,
    )
    return summary


def _frozen_hpk_created_at(
    public_trace: Path,
    *,
    expected_episode_id: int,
) -> str:
    """Derive the updater timestamp from the hash-pinned public episode end.

    The child snapshot must be a pure function of its parent, evidence, and
    frozen policies.  Using a second wall-clock read after the trace is sealed
    would make an otherwise identical finalization produce different bytes.
    """

    episode_end_timestamps: list[float] = []
    with public_trace.open("r", encoding="utf-8") as stream:
        for ordinal, line in enumerate(stream, start=1):
            if not line.strip():
                raise RuntimeError(
                    f"HPK public trace contains a blank line at {ordinal}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"HPK public trace contains invalid JSON at {ordinal}: {exc}"
                ) from exc
            if not isinstance(record, dict) or record.get("event") != "episode_end":
                continue
            if (
                type(record.get("episode_id")) is not type(expected_episode_id)
                or record.get("episode_id") != expected_episode_id
            ):
                raise RuntimeError("HPK public episode_end identity mismatch")
            timestamp = record.get("timestamp")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(float(timestamp))
            ):
                raise RuntimeError("HPK public episode_end timestamp must be finite")
            episode_end_timestamps.append(float(timestamp))
    if len(episode_end_timestamps) != 1:
        raise RuntimeError(
            "HPK public trace must contain exactly one episode_end before finalization"
        )
    try:
        frozen = datetime.fromtimestamp(
            episode_end_timestamps[0],
            tz=timezone.utc,
        )
    except (OverflowError, OSError, ValueError) as exc:
        raise RuntimeError("HPK public episode_end timestamp is out of range") from exc
    return frozen.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _finalize_evolving_hpk_episode(
    model,
    *,
    rollout_dir: Path | None,
    trace_file: Path | None,
    episode_id: int,
    result: str,
    natural_episode_end: bool,
    episode_validity: dict,
):
    """Finalize one evolving HPK episode."""

    if not _evolving_hpk_enabled(model):
        return None
    if rollout_dir is None:
        raise RuntimeError("evolving HPK requires an explicit rollout directory")
    required = (
        "write_hpk_public_episode_event",
        "publish_hpk_rollout_import_manifest",
        "finalize_hpk_episode",
    )
    if any(not callable(getattr(model, name, None)) for name in required):
        raise RuntimeError("evolving HPK model is missing its finalization API")

    model.write_hpk_public_episode_event(
        "episode_end",
        {
            "episode_id": episode_id,
            "result": result,
            "natural_episode_end": natural_episode_end,
            "episode_validity": dict(episode_validity),
        },
    )
    manifest_path = (rollout_dir / "hpk_rollout_import_manifest.json").resolve()
    receipt_path = (rollout_dir / "hpk_finalization_receipt.json").resolve()
    published_manifest = model.publish_hpk_rollout_import_manifest(str(manifest_path))
    trace_paths = getattr(model, "hpk_trace_paths", None)
    if not isinstance(trace_paths, dict):
        raise RuntimeError("evolving HPK did not expose its trace paths")
    public_trace_value = trace_paths.get("public_trace")
    if not isinstance(public_trace_value, str) or not public_trace_value:
        raise RuntimeError("evolving HPK public trace path is unavailable")
    public_trace = Path(public_trace_value).resolve(strict=True)
    frozen_created_at = _frozen_hpk_created_at(
        public_trace,
        expected_episode_id=episode_id,
    )
    durable = model.finalize_hpk_episode(
        manifest_path=str(published_manifest.path),
        trace_path=str(public_trace),
        expected_manifest_sha256=published_manifest.sha256,
        expected_trace_sha256=published_manifest.payload["public_trace"]["sha256"],
        created_at=frozen_created_at,
        receipt_path=str(receipt_path),
    )
    coordinated = durable.coordinated_result
    receipt = durable.receipt
    summary = {
        "schema": "roboharn_evo/hpk/eval_episode_finalization/v1",
        "rollout_import_manifest": {
            "path": str(published_manifest.path),
            "sha256": published_manifest.sha256,
        },
        "finalization_receipt": {
            "path": str(receipt.path),
            "sha256": receipt.sha256,
            "receipt_id": receipt.payload["receipt_id"],
        },
        "finalization": coordinated.finalization.to_dict(),
        "next_snapshot": coordinated.next_snapshot_ref.identity(),
        "advanced": coordinated.advanced,
    }
    if durable.semantic_result is not None:
        summary["semantic_knowledge"] = durable.semantic_result.to_dict()
    _write_agent_trace_event(
        trace_file,
        "hpk_episode_finalization",
        episode_id=episode_id,
        **summary,
    )
    return summary


def main(usr_args, *, sequential_controller=None):
    if sequential_controller is not None:
        sequential_controller.validate_config(usr_args)
    failure_boundary_replay = _resolve_failure_boundary_replay_config(usr_args)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    instruction_set = str(
        usr_args.get("instruction_set", ORIGINAL_INSTRUCTION_SET)
        or ORIGINAL_INSTRUCTION_SET
    ).strip()
    instruction_source = describe_instruction_source(
        task_name,
        instruction_set=instruction_set,
    )
    if instruction_type not in instruction_source["available_instruction_types"]:
        raise ValueError(
            f"instruction type {instruction_type!r} is unavailable for task "
            f"{task_name!r} in instruction set {instruction_set!r}; available: "
            + ", ".join(instruction_source["available_instruction_types"])
        )
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    task_config_path = (
        Path(
            usr_args.get("task_config_path")
            or task_config_root() / f"{task_config}.yml"
        )
        .expanduser()
        .resolve()
    )
    with task_config_path.open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    applied_env_overrides = _merge_task_config_overrides(args, usr_args)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["task_config_path"] = str(task_config_path)
    args["ckpt_setting"] = ckpt_setting
    args["instruction_set"] = instruction_set
    args["instruction_source"] = instruction_source
    st_seed = _resolve_start_seed(usr_args)
    test_num = _resolve_test_num(args, usr_args)
    args["eval_step_limit"] = _resolve_eval_step_limit(usr_args)
    args["eval_step_limit_mode"] = (usr_args.get("eval") or {}).get("step_limit_mode", "cap")
    args["eval_exact_seed_fail_closed"] = _resolve_exact_seed_fail_closed(usr_args)
    if sequential_controller is not None:
        sequential_controller.validate_effective_task_args(args)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_item):
        robot_file = _embodiment_types[embodiment_item]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return str(resolve_asset_reference(robot_file))

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    embodiment_name = (
        str(embodiment_type[0])
        if len(embodiment_type) == 1
        else str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    )

    configured_run_output_root = configured_output_root(usr_args.get("output_root"))
    save_dir = (
        sequential_controller.output_root
        if sequential_controller is not None
        else (
            configured_run_output_root
            / task_name
            / policy_name
            / task_config
            / ckpt_setting
            / current_time
        )
    )
    if str(os.environ.get(_FORMAL_PROTOCOL_ENV, "0") or "0").strip() == "1":
        provenance_module = _load_content_provenance_module()
        try:
            provenance_module.validate_output_path(
                BENCHMARK_ROOT.parents[1].resolve(),
                save_dir / "runtime_provenance.json",
            )
        except ValueError as exc:
            raise RuntimeError(
                f"formal output path is outside RoboHarn-Evo/eval_result/rmbench: {exc}"
            ) from exc
    if sequential_controller is not None:
        sequential_controller.reserve_output_capacity()
        if not save_dir.is_dir() or save_dir.is_symlink():
            raise RuntimeError(
                "preregistered sequential output_root claim was not created safely"
            )
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
    # Task YAML and deploy overrides are read-only inputs.  Base_Task uses
    # save_path for cache/data writes and may recursively remove its cache, so
    # pin it only after the output-owned run directory has been created.
    args["save_path"] = str((save_dir / "episode_data").resolve())
    runtime_provenance = _archive_runtime_provenance(save_dir)
    args["runtime_provenance"] = runtime_provenance
    from roboharn_evo.agent.hpk.compatibility import normalize_agent_knowledge_config

    agent_config = usr_args.get("agent", {})
    hpk_config = normalize_agent_knowledge_config(agent_config).get("hpk", {}) if isinstance(agent_config, dict) else {}
    if (
        isinstance(hpk_config, dict)
        and str(hpk_config.get("mode", "off") or "off").strip().lower() == "evolving"
    ):
        # The Agent runtime is constructed only after provenance has been
        # archived.  Bind the exact current config/model/source identities to
        # EvidenceV2 and the child manifest instead of inheriting K0's builder
        # identity or relying on a neighbouring diagnostic trace.
        usr_args["runtime_provenance"] = dict(runtime_provenance)
    instruction_provenance = {
        **instruction_source,
        "instruction_type": instruction_type,
    }
    (save_dir / "instruction_provenance.json").write_text(
        json.dumps(
            instruction_provenance,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    log_file = save_dir / "eval_log.txt"
    args["log_file"] = str(log_file)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            f"Eval log for {task_name} | {policy_name} | {task_config} | {ckpt_setting}\n"
        )
        f.write(f"Timestamp: {current_time}\n\n")
        f.write(
            f"Applied task-config overrides: {json.dumps(applied_env_overrides, ensure_ascii=False, default=str)}\n"
        )
        f.write(
            f"Effective data_type: {json.dumps(args.get('data_type', {}), ensure_ascii=False, default=str)}\n\n"
        )
        f.write(f"Instruction set: {instruction_set}\n")
        f.write(f"Instruction type: {instruction_type}\n")
        f.write(
            "Instruction source: "
            f"{json.dumps(instruction_source, ensure_ascii=False, sort_keys=True)}\n\n"
        )
        f.write(f"Evaluation start seed: {st_seed}\n")
        f.write(f"Evaluation episode count: {test_num}\n\n")
        f.write(f"Requested environment step limit: {args.get('eval_step_limit')}\n\n")
        if args["eval_exact_seed_fail_closed"]:
            f.write("Exact-seed expert validation: fail_closed\n\n")
        if _is_debug_migration_smoke(args):
            f.write(
                "Debug migration smoke: expert/demo check only; skipping remote planner/executor service loop.\n\n"
            )

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    print("============= Config =============\n")
    print(
        "\033[95mMessy Table:\033[0m "
        + str(args["domain_randomization"]["cluttered_table"])
    )
    print(
        "\033[95mRandom Background:\033[0m "
        + str(args["domain_randomization"]["random_background"])
    )
    if args["domain_randomization"]["random_background"]:
        print(
            " - Clean Background Rate: "
            + str(args["domain_randomization"]["clean_background_rate"])
        )
    print(
        "\033[95mRandom Light:\033[0m "
        + str(args["domain_randomization"]["random_light"])
    )
    if args["domain_randomization"]["random_light"]:
        print(
            " - Crazy Random Light Rate: "
            + str(args["domain_randomization"]["crazy_random_light_rate"])
        )
    print(
        "\033[95mRandom Table Height:\033[0m "
        + str(args["domain_randomization"]["random_table_height"])
    )
    print(
        "\033[95mRandom Head Camera Distance:\033[0m "
        + str(args["domain_randomization"]["random_head_camera_dis"])
    )
    print(
        "\033[94mHead Camera Config:\033[0m "
        + str(args["camera"]["head_camera_type"])
        + f", "
        + str(args["camera"]["collect_head_camera"])
    )
    print(
        "\033[94mWrist Camera Config:\033[0m "
        + str(args["camera"]["wrist_camera_type"])
        + f", "
        + str(args["camera"]["collect_wrist_camera"])
    )
    print(
        "\033[94mData Type Config:\033[0m "
        + json.dumps(args.get("data_type", {}), ensure_ascii=False, default=str)
    )
    if applied_env_overrides:
        print(
            "\033[94mTask Config Overrides:\033[0m "
            + ", ".join(sorted(applied_env_overrides.keys()))
        )
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(
        args["right_embodiment_config"]["arm_joints_name"][1]
    )

    suc_nums = []

    transport_context = (
        contextlib.nullcontext()
        if sequential_controller is None
        else sequential_controller.guarded_transport()
    )
    with transport_context:
        if sequential_controller is not None:
            sequential_controller.preflight_services()
        model = get_model(usr_args)
        if sequential_controller is not None:
            sequential_controller.bind_model_before_episode(model)
        st_seed, suc_num, task_total_reward = eval_policy(
            task_name,
            TASK_ENV,
            args,
            model,
            st_seed,
            test_num=test_num,
            video_size=video_size,
            instruction_type=instruction_type,
            save_dir=save_dir,
            sequential_controller=sequential_controller,
            failure_boundary_replay=failure_boundary_replay,
        )
    suc_nums.append(suc_num)

    file_path = save_dir / "_result.txt"
    with file_path.open("w", encoding="utf-8") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Set: {instruction_set}\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(
            f"Instruction Source SHA256: {instruction_source['task_file_sha256']}\n\n"
        )
        success_rates = (np.asarray(suc_nums, dtype=float) / float(test_num)).reshape(
            -1
        )
        for sr in success_rates:
            file.write(f"Success Rate: {sr}\n")
        file.write("\n")
        rewards = task_total_reward / test_num
        if np.isscalar(rewards):
            file.write(f"Reward: {float(rewards)}\n")
        else:
            rewards = np.asarray(rewards, dtype=float).reshape(-1)
            for r in rewards:
                file.write(f"Reward: {r}\n")

    print(f"Data has been saved to {file_path}")
    if sequential_controller is not None:
        return sequential_controller.finalize_run(save_dir)


def eval_policy(
    task_name,
    TASK_ENV,
    args,
    model,
    st_seed,
    test_num=100,
    video_size=None,
    instruction_type=None,
    save_dir=None,
    sequential_controller=None,
    failure_boundary_replay=None,
):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    debug_migration_smoke = _is_debug_migration_smoke(args)
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True
    runtime_provenance = args.get("runtime_provenance", {})
    if not isinstance(runtime_provenance, dict):
        runtime_provenance = {}
    formal_protocol = bool(runtime_provenance.get("formal_protocol", False))
    formal_protocol_version = int(
        runtime_provenance.get("formal_protocol_version", 0) or 0
    )
    exact_seed_fail_closed = args.get("eval_exact_seed_fail_closed", False)
    if not isinstance(exact_seed_fail_closed, bool):
        raise ValueError("eval_exact_seed_fail_closed must be a boolean")
    if failure_boundary_replay is None:
        failure_boundary_replay = _resolve_failure_boundary_replay_config(args)
    if failure_boundary_replay["mode"] != "off" and test_num != 1:
        raise ValueError(
            "failure-boundary capture/restore requires one episode per run"
        )

    while succ_seed < test_num:
        sequential_request = None
        if sequential_controller is not None:
            if sequential_controller.should_stop:
                print(
                    "HPK update finished without accepted knowledge; "
                    "the probe episode was skipped."
                )
                break
            sequential_request = sequential_controller.begin_episode()
            if sequential_request.ordinal != now_id:
                raise RuntimeError(
                    "sequential controller episode order disagrees with RMBench"
                )
            now_seed = sequential_request.seed
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(
                    now_ep_num=now_id, seed=now_seed, is_test=True, **args
                )
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as exc:
                TASK_ENV.close_env()
                if debug_migration_smoke:
                    raise
                if exact_seed_fail_closed:
                    args["render_freq"] = render_freq
                    raise ExactSeedValidationError(
                        "exact-seed expert validation failed closed: "
                        f"seed={now_seed}, reason=unstable_expert_demo"
                    ) from exc
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except RendererDeviceContractError:
                # A renderer/device mismatch is an infrastructure failure, not
                # an invalid benchmark seed.  Never hide it by incrementing the
                # seed and retrying forever.
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass
                raise
            except Exception as exc:
                TASK_ENV.close_env()
                if debug_migration_smoke:
                    raise
                if exact_seed_fail_closed:
                    args["render_freq"] = render_freq
                    raise ExactSeedValidationError(
                        "exact-seed expert validation failed closed: "
                        f"seed={now_seed}, reason=expert_demo_exception, "
                        f"error_type={type(exc).__name__}"
                    ) from exc
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        expert_success = (not expert_check) or (
            TASK_ENV.plan_success and TASK_ENV.check_success()
        )
        if debug_migration_smoke:
            args["render_freq"] = render_freq
            smoke_trace_file = (
                None
                if save_dir is None
                else Path(save_dir) / "migration_smoke_trace.jsonl"
            )
            if not expert_success:
                _write_agent_trace_event(
                    smoke_trace_file,
                    "migration_smoke_fail",
                    seed=now_seed,
                    task_name=task_name,
                    policy_name=args["policy_name"],
                    plan_success=bool(getattr(TASK_ENV, "plan_success", False)),
                    eval_success=bool(getattr(TASK_ENV, "eval_success", False)),
                    max_reward=float(getattr(TASK_ENV, "max_reward", 0.0)),
                )
                raise RuntimeError(
                    "debug_migration_smoke expert/demo check failed: "
                    f"plan_success={getattr(TASK_ENV, 'plan_success', None)}, "
                    f"eval_success={getattr(TASK_ENV, 'eval_success', None)}, "
                    f"max_reward={getattr(TASK_ENV, 'max_reward', None)}"
                )
            task_total_reward += TASK_ENV.max_reward
            TASK_ENV.suc = 1
            TASK_ENV.test_num = 1
            _write_agent_trace_event(
                smoke_trace_file,
                "migration_smoke_pass",
                seed=now_seed,
                task_name=task_name,
                policy_name=args["policy_name"],
                plan_success=bool(getattr(TASK_ENV, "plan_success", False)),
                eval_success=bool(getattr(TASK_ENV, "eval_success", False)),
                max_reward=float(getattr(TASK_ENV, "max_reward", 0.0)),
            )
            log_file = args.get("log_file", None)
            if log_file is not None:
                try:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"migration_smoke=pass, seed={now_seed}, "
                            f"plan_success={getattr(TASK_ENV, 'plan_success', None)}, "
                            f"eval_success={getattr(TASK_ENV, 'eval_success', None)}, "
                            f"reward={getattr(TASK_ENV, 'max_reward', None)}\n"
                        )
                except Exception as e:
                    print(f"[Log Warning] Failed to write log: {e}")
            print(
                "\033[92mMigration smoke passed!\033[0m",
                " | max reward:",
                TASK_ENV.max_reward,
            )
            return now_seed + 1, 1, task_total_reward

        if expert_success:
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            if exact_seed_fail_closed:
                args["render_freq"] = render_freq
                raise ExactSeedValidationError(
                    "exact-seed expert validation failed closed: "
                    f"seed={now_seed}, reason=expert_demo_unsuccessful"
                )
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        step_limit_provenance = _apply_eval_step_limit(
            TASK_ENV,
            args.get("eval_step_limit"),
            mode=args.get("eval_step_limit_mode", "cap"),
        )
        print(
            "[eval-step-limit] "
            f"requested={step_limit_provenance['requested']} "
            f"task_default={step_limit_provenance['task_default']} "
            f"effective={step_limit_provenance['effective']} "
            f"mode={step_limit_provenance['mode']}"
        )
        episode_info_list = [episode_info["info"]]
        instruction_set = str(
            args.get("instruction_set", ORIGINAL_INSTRUCTION_SET)
            or ORIGINAL_INSTRUCTION_SET
        )
        instruction_source = args.get("instruction_source", {})
        if not isinstance(instruction_source, dict):
            instruction_source = {}
        results = generate_episode_descriptions(
            args["task_name"],
            episode_info_list,
            test_num,
            instruction_set=instruction_set,
        )
        instruction_choices = results[0].get(instruction_type, [])
        if not instruction_choices:
            raise ValueError(
                f"no {instruction_type!r} instructions generated for task "
                f"{args['task_name']!r} from instruction set {instruction_set!r}"
            )
        if sequential_controller is None:
            instruction = str(np.random.choice(instruction_choices))
        else:
            instruction = sequential_controller.select_instruction(
                request=sequential_request,
                choices=tuple(str(value) for value in instruction_choices),
                instruction_source_sha256=str(
                    instruction_source.get("task_file_sha256", "") or ""
                ),
            )
        selected_instruction_provenance = {
            **instruction_source,
            "instruction_type": instruction_type,
            "instruction_sha256": hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest(),
        }
        TASK_ENV.set_instruction(instruction=instruction)

        trace_file = (
            None
            if save_dir is None
            else Path(save_dir) / f"episode_{now_id:04d}_agent_trace.jsonl"
        )
        rollout_dir = (
            None
            if save_dir is None
            else Path(save_dir) / f"episode_{now_id:04d}_rollout"
        )
        if rollout_dir is not None:
            for name in ("head", "left", "right", "video"):
                (rollout_dir / name).mkdir(parents=True, exist_ok=True)

        def configure_agent_rollout_dump():
            if not (hasattr(model, "session") and hasattr(model.session, "agent")):
                if _evolving_hpk_enabled(model):
                    raise RuntimeError(
                        "evolving HPK requires the RoboHarn-Evo Agent session before rollout"
                    )
                return
            try:
                agent = model.session.agent
                agent.set_trace_file(str(trace_file), episode_id=now_id, seed=now_seed)
                agent.set_rollout_dump_dir(
                    str(rollout_dir) if rollout_dir is not None else None
                )
                agent.write_rollout_meta(
                    {
                        "episode_id": now_id,
                        "seed": now_seed,
                        "task_name": task_name,
                        "task_config": args.get("task_config", ""),
                        "policy_name": args["policy_name"],
                        "ckpt_setting": args.get("ckpt_setting", ""),
                        "instruction": instruction,
                        "instruction_set": instruction_set,
                        "instruction_type": instruction_type,
                        "instruction_provenance": selected_instruction_provenance,
                        "data_type": args.get("data_type", {}),
                        "camera": args.get("camera", {}),
                        "formal_protocol": formal_protocol,
                        "formal_protocol_version": formal_protocol_version,
                        "step_limit": int(TASK_ENV.step_lim),
                        "step_limit_provenance": step_limit_provenance,
                        "runtime_provenance": runtime_provenance,
                        **(
                            {"exact_seed_fail_closed": True}
                            if exact_seed_fail_closed
                            else {}
                        ),
                    }
                )
            except Exception:
                if _evolving_hpk_enabled(model):
                    raise

        succ = False
        failure_reason = "step_limit_exhausted"
        interrupted = False
        episode_error = None
        episode_error_traceback = None
        result_str = "Fail"
        continuous_video_recorder = None
        continuous_video_summary = None
        failure_boundary_captured = False
        failure_boundary_last_skip_step = None
        failure_boundary_action_recorder = None
        restored_boundary = None
        captured_boundary = None
        captured_boundaries = []
        failure_boundary_bank_signatures = set()
        try:
            _write_agent_trace_event(
                trace_file,
                "episode_start",
                episode_id=now_id,
                seed=now_seed,
                task_name=task_name,
                task_config=args.get("task_config", ""),
                policy_name=args["policy_name"],
                ckpt_setting=args.get("ckpt_setting", ""),
                instruction=instruction,
                instruction_set=instruction_set,
                instruction_type=instruction_type,
                instruction_provenance=selected_instruction_provenance,
                data_type=args.get("data_type", {}),
                camera=args.get("camera", {}),
                formal_protocol=formal_protocol,
                formal_protocol_version=formal_protocol_version,
                step_limit=int(TASK_ENV.step_lim),
                step_limit_provenance=step_limit_provenance,
                runtime_provenance=runtime_provenance,
                **({"exact_seed_fail_closed": True} if exact_seed_fail_closed else {}),
            )
            if TASK_ENV.eval_video_path is not None:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        video_size,
                        "-framerate",
                        "10",
                        "-i",
                        "-",
                        "-pix_fmt",
                        "yuv420p",
                        "-vcodec",
                        "libx264",
                        "-threads",
                        "2",
                        "-crf",
                        "23",
                        f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                    bufsize=0,
                    start_new_session=True,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

            reset_func(model)
            configure_agent_rollout_dump()
            failure_boundary_action_recorder = _start_failure_boundary_action_recorder(
                TASK_ENV,
                failure_boundary_replay,
            )
            _write_agent_trace_event(
                trace_file, "episode_reset_model", episode_id=now_id, seed=now_seed
            )
            restored_boundary = _restore_failure_boundary_before_episode(
                TASK_ENV,
                model,
                failure_boundary_replay,
                task_name=task_name,
                seed=now_seed,
                instruction=instruction,
            )
            if restored_boundary is not None:
                _write_agent_trace_event(
                    trace_file,
                    "failure_boundary_restored",
                    episode_id=now_id,
                    seed=now_seed,
                    **restored_boundary,
                )
            try:
                continuous_video_recorder = _start_continuous_rollout_video(
                    TASK_ENV,
                    rollout_dir,
                    trace_file,
                    episode_id=now_id,
                    seed=now_seed,
                    fps=30,
                )
            except Exception as video_error:
                _write_agent_trace_event(
                    trace_file,
                    "continuous_rollout_video_start_error",
                    episode_id=now_id,
                    seed=now_seed,
                    error=repr(video_error),
                )
            if sequential_controller is not None:
                sequential_controller.validate_model_pin(
                    model,
                    sequential_request,
                )

            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = _acquire_eval_observation(
                    TASK_ENV,
                    model,
                    trace_file,
                    episode_id=now_id,
                    seed=now_seed,
                )
                eval_func(TASK_ENV, model, observation)
                agent_status = _extract_agent_status(model)
                if (
                    failure_boundary_replay["mode"] == "capture"
                    and not failure_boundary_captured
                    and _task_env_step(TASK_ENV)
                    >= failure_boundary_replay["capture_at_or_after_env_step"]
                ):
                    try:
                        captured_boundary = _capture_failure_boundary_after_turn(
                            TASK_ENV,
                            model,
                            failure_boundary_replay,
                            task_name=task_name,
                            seed=now_seed,
                            instruction=instruction,
                            agent_status=agent_status,
                            action_prefix=(
                                failure_boundary_action_recorder.records
                                if failure_boundary_action_recorder is not None
                                else []
                            ),
                            observation_calls_after_last_action=(
                                failure_boundary_action_recorder.observation_calls_after_last_action
                                if failure_boundary_action_recorder is not None
                                else 0
                            ),
                        )
                    except Exception as boundary_error:
                        if failure_boundary_last_skip_step != _task_env_step(TASK_ENV):
                            failure_boundary_last_skip_step = _task_env_step(TASK_ENV)
                            _write_agent_trace_event(
                                trace_file,
                                "failure_boundary_capture_deferred",
                                episode_id=now_id,
                                seed=now_seed,
                                step=_task_env_step(TASK_ENV),
                                error_type=type(boundary_error).__name__,
                                reason=str(boundary_error),
                            )
                    else:
                        failure_boundary_captured = True
                        if failure_boundary_action_recorder is not None:
                            failure_boundary_action_recorder.stop()
                        _write_agent_trace_event(
                            trace_file,
                            "failure_boundary_captured",
                            episode_id=now_id,
                            seed=now_seed,
                            **captured_boundary,
                        )
                        if failure_boundary_replay["stop_after_capture"]:
                            failure_reason = "failure_boundary_captured"
                            break
                if (
                    failure_boundary_replay["mode"] == "capture_bank"
                    and len(captured_boundaries)
                    < failure_boundary_replay["max_boundaries"]
                    and _task_env_step(TASK_ENV)
                    >= failure_boundary_replay["capture_at_or_after_env_step"]
                ):
                    from roboharn_evo.benchmark_adapters.rmbench.v31_boundary_bank import (
                        capture_bank_boundary_path,
                        recoverable_failure_boundary_v31,
                    )

                    failure_boundary = recoverable_failure_boundary_v31(model)
                    holding_boundary = (
                        None if failure_boundary is None else failure_boundary[0]
                    )
                    if (
                        holding_boundary is not None
                        and holding_boundary.signature
                        not in failure_boundary_bank_signatures
                    ):
                        ordinal = len(captured_boundaries)
                        env_step = _task_env_step(TASK_ENV)
                        bank_path = capture_bank_boundary_path(
                            failure_boundary_replay["typed_config"],
                            seed=now_seed,
                            ordinal=ordinal,
                            env_step=env_step,
                        )
                        capture_config = {
                            "mode": "capture",
                            "path": bank_path,
                            "label": (
                                f"{failure_boundary_replay['label_prefix']}; "
                                f"seed={now_seed}; ordinal={ordinal}; "
                                f"env_step={env_step}"
                            ),
                        }
                        failure_evidence = failure_boundary[1]
                        capture_agent_status = dict(agent_status)
                        capture_agent_status["failure_reason"] = (
                            "recoverable strict-hold Runtime failure: "
                            + ", ".join(
                                f"{key}={value}"
                                for key, value in failure_evidence.items()
                                if key != "monitor_status" and value
                            )
                        )
                        captured = _capture_failure_boundary_after_turn(
                            TASK_ENV,
                            model,
                            capture_config,
                            task_name=task_name,
                            seed=now_seed,
                            instruction=instruction,
                            agent_status=capture_agent_status,
                            action_prefix=(
                                failure_boundary_action_recorder.records
                                if failure_boundary_action_recorder is not None
                                else []
                            ),
                            observation_calls_after_last_action=(
                                failure_boundary_action_recorder.observation_calls_after_last_action
                                if failure_boundary_action_recorder is not None
                                else 0
                            ),
                        )
                        captured["holding_boundary"] = (
                            holding_boundary.to_private_dict()
                        )
                        captured["failure_evidence"] = failure_evidence
                        captured_boundaries.append(captured)
                        failure_boundary_bank_signatures.add(holding_boundary.signature)
                        _write_agent_trace_event(
                            trace_file,
                            "failure_boundary_bank_captured",
                            episode_id=now_id,
                            seed=now_seed,
                            **captured,
                        )
                        if (
                            len(captured_boundaries)
                            >= failure_boundary_replay["max_boundaries"]
                        ):
                            if failure_boundary_action_recorder is not None:
                                failure_boundary_action_recorder.stop()
                            if failure_boundary_replay["stop_after_bank_full"]:
                                failure_reason = "failure_boundary_bank_full"
                                break
                if TASK_ENV.take_action_cnt % 10 == 0 or TASK_ENV.eval_success:
                    _write_agent_trace_event(
                        trace_file,
                        "episode_progress",
                        episode_id=now_id,
                        seed=now_seed,
                        step=TASK_ENV.take_action_cnt,
                        step_limit=TASK_ENV.step_lim,
                        eval_success=bool(TASK_ENV.eval_success),
                        max_reward=float(TASK_ENV.max_reward),
                        **agent_status,
                    )
                if TASK_ENV.eval_success:
                    succ = True
                    failure_reason = ""
                    break
                if bool(agent_status.get("terminal_failure", False)):
                    failure_reason = (
                        str(
                            agent_status.get(
                                "terminal_failure_reason", "agent_terminal_failure"
                            )
                        )
                        or "agent_terminal_failure"
                    )
                    _write_agent_trace_event(
                        trace_file,
                        "episode_agent_terminal_failure",
                        episode_id=now_id,
                        seed=now_seed,
                        step=TASK_ENV.take_action_cnt,
                        reason=failure_reason,
                        **agent_status,
                    )
                    break
                if bool(agent_status.get("task_finished", False)):
                    if bool(agent_status.get("pure_tool_control", False)) and not bool(
                        agent_status.get("task_finish_validated", False)
                    ):
                        _write_agent_trace_event(
                            trace_file,
                            "episode_agent_finish_rejected",
                            episode_id=now_id,
                            seed=now_seed,
                            step=TASK_ENV.take_action_cnt,
                            reason="pure tool-control finish was not validated by environment success",
                            **agent_status,
                        )
                        continue
                    failure_reason = (
                        str(agent_status.get("monitor_status", "agent_task_finished"))
                        or "agent_task_finished"
                    )
                    _write_agent_trace_event(
                        trace_file,
                        "episode_agent_finished",
                        episode_id=now_id,
                        seed=now_seed,
                        step=TASK_ENV.take_action_cnt,
                        reason=failure_reason,
                        **agent_status,
                    )
                    break
            if (
                failure_boundary_replay["mode"] == "capture"
                and not failure_boundary_captured
            ):
                raise RuntimeError(
                    "the requested failure boundary was not captured before "
                    "the episode ended"
                )
            if (
                failure_boundary_replay["mode"] == "capture_bank"
                and len(captured_boundaries) < failure_boundary_replay["min_boundaries"]
            ):
                raise RuntimeError(
                    "the episode produced fewer confirmed holding boundaries "
                    "than requested: "
                    f"captured={len(captured_boundaries)}, "
                    f"minimum={failure_boundary_replay['min_boundaries']}"
                )
        except KeyboardInterrupt:
            interrupted = True
            interrupt_signal = _TERMINATION_SIGNAL_NAME
            failure_reason = (
                f"external_{interrupt_signal}"
                if interrupt_signal
                else "keyboard_interrupt"
            )
            _write_agent_trace_event(
                trace_file,
                "episode_interrupt",
                episode_id=now_id,
                seed=now_seed,
                step=TASK_ENV.take_action_cnt,
                reason=failure_reason,
                interrupt_signal=interrupt_signal or "sigint",
                **_extract_agent_status(model),
            )
        except Exception as e:
            episode_error = e
            episode_error_traceback = e.__traceback__
            failure_reason = repr(e)
            _write_agent_trace_event(
                trace_file,
                "episode_exception",
                episode_id=now_id,
                seed=now_seed,
                step=TASK_ENV.take_action_cnt,
                error=repr(e),
                traceback=traceback.format_exc(),
                **_extract_agent_status(model),
            )
        finally:
            if failure_boundary_action_recorder is not None:
                failure_boundary_action_recorder.stop()
            continuous_video_summary = _close_continuous_rollout_video(
                continuous_video_recorder,
                trace_file,
                episode_id=now_id,
                seed=now_seed,
            )
            _close_eval_video_writer(
                TASK_ENV,
                trace_file,
                episode_id=now_id,
                seed=now_seed,
                reason=failure_reason or "episode_end",
            )

        task_total_reward += TASK_ENV.max_reward

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m", " | max reward:", TASK_ENV.max_reward)
            result_str = "Success"
        else:
            print("\033[91mFail!\033[0m", " | max reward:", TASK_ENV.max_reward)
            result_str = "Fail"

        final_agent_status = _extract_agent_status(model)
        episode_validity = _classify_episode_validity(
            success=succ,
            interrupted=interrupted,
            episode_error=episode_error,
            agent_status=final_agent_status,
        )
        _write_agent_trace_event(
            trace_file,
            "episode_end",
            episode_id=now_id,
            seed=now_seed,
            task_name=task_name,
            task_config=args.get("task_config", ""),
            policy_name=args.get("policy_name", ""),
            ckpt_setting=args.get("ckpt_setting", ""),
            result=result_str,
            success=succ,
            total_steps=TASK_ENV.take_action_cnt,
            step_limit=int(TASK_ENV.step_lim),
            step_limit_provenance=step_limit_provenance,
            max_reward=float(TASK_ENV.max_reward),
            failure_reason=failure_reason,
            natural_episode_end=not interrupted and episode_error is None,
            episode_validity=episode_validity,
            formal_protocol=formal_protocol,
            formal_protocol_version=formal_protocol_version,
            runtime_provenance=runtime_provenance,
            **({"exact_seed_fail_closed": True} if exact_seed_fail_closed else {}),
            **final_agent_status,
        )
        failure_boundary_summary = _failure_boundary_episode_summary(
            config=failure_boundary_replay,
            restored_boundary=restored_boundary,
            captured_boundary=captured_boundary,
            task_name=task_name,
            seed=now_seed,
            instruction=instruction,
            result=result_str,
            success=succ,
            failure_reason=failure_reason,
            final_env_step=TASK_ENV.take_action_cnt,
            max_reward=TASK_ENV.max_reward,
            captured_boundaries=captured_boundaries,
        )
        failure_boundary_summary_path = _write_failure_boundary_episode_summary(
            rollout_dir,
            failure_boundary_summary,
        )
        if failure_boundary_summary_path is not None:
            _write_agent_trace_event(
                trace_file,
                "failure_boundary_result",
                episode_id=now_id,
                seed=now_seed,
                path=str(failure_boundary_summary_path),
                summary=failure_boundary_summary,
            )

        hpk_finalization_error = None
        hpk_finalization_error_traceback = None
        hpk_finalization_summary = None
        try:
            if _hierarchical_hpk_enabled(model):
                if _hierarchical_hpk_read_only_utility_gate(args, model=model):
                    hpk_finalization_summary = {
                        "schema": "roboharn_evo/rmbench/v31/read_only_store_retained",
                        "store_write_performed": False,
                    }
                    _write_agent_trace_event(
                        trace_file,
                        "hpk_v31_read_only_store_retained",
                        episode_id=now_id,
                        seed=now_seed,
                        **hpk_finalization_summary,
                    )
                else:
                    hpk_finalization_summary = _finalize_hierarchical_hpk_episode(
                        model,
                        rollout_dir=rollout_dir,
                        trace_file=trace_file,
                        episode_id=now_id,
                        seed=now_seed,
                        result=result_str,
                    )
            else:
                hpk_finalization_summary = _finalize_evolving_hpk_episode(
                    model,
                    rollout_dir=rollout_dir,
                    trace_file=trace_file,
                    episode_id=now_id,
                    result=result_str,
                    natural_episode_end=(not interrupted and episode_error is None),
                    episode_validity=episode_validity,
                )
            if sequential_controller is not None:
                if not isinstance(hpk_finalization_summary, dict):
                    raise RuntimeError(
                        "sequential controller requires an evolving HPK finalization summary"
                    )
                sequential_controller.complete_episode(
                    model=model,
                    hpk_finalization_summary=hpk_finalization_summary,
                    environment_actions=TASK_ENV.take_action_cnt,
                    agent_status=final_agent_status,
                )
        except Exception as error:
            hpk_finalization_error = error
            hpk_finalization_error_traceback = error.__traceback__
            _write_agent_trace_event(
                trace_file,
                "hpk_episode_finalization_error",
                episode_id=now_id,
                seed=now_seed,
                error_type=type(error).__name__,
                error=repr(error),
            )

        _finalize_rollout_visualization(
            model,
            rollout_dir,
            trace_file,
            fps=10,
            continuous_video_ready=(
                isinstance(continuous_video_summary, dict)
                and continuous_video_summary.get("complete") is True
            ),
        )

        log_file = args.get("log_file", None)
        if log_file is not None:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"episode_id={now_id}, seed={now_seed}, instruction={instruction}, result={result_str}, steps={TASK_ENV.take_action_cnt}, reward={TASK_ENV.max_reward}, failure_reason={failure_reason}\n"
                    )
            except Exception as e:
                print(f"[Log Warning] Failed to write log: {e}")

        if (
            interrupted
            or episode_error is not None
            or hpk_finalization_error is not None
        ):
            try:
                TASK_ENV.close_env(clear_cache=True)
            except Exception:
                pass
            if TASK_ENV.render_freq:
                try:
                    TASK_ENV.viewer.close()
                except Exception:
                    pass
            if interrupted:
                raise KeyboardInterrupt
            if episode_error is not None:
                raise episode_error.with_traceback(episode_error_traceback)
            raise hpk_finalization_error.with_traceback(
                hpk_finalization_error_traceback
            )

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()
        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc, task_total_reward


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--hpk-sequential-preregistration", type=str)
    parser.add_argument("--hpk-sequential-preregistration-sha256", type=str)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def parse_override_pairs(pairs):
        override_dict = {}

        def assign_nested(target, dotted_key, value):
            keys = dotted_key.split(".")
            current = target
            for key in keys[:-1]:
                if not isinstance(current.get(key), dict):
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value

        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except Exception:
                pass
            if "." in key:
                assign_nested(override_dict, key, value)
            else:
                override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        _deep_update_config(config, overrides)

    if (args.hpk_sequential_preregistration is None) != (
        args.hpk_sequential_preregistration_sha256 is None
    ):
        raise ValueError(
            "HPK sequential preregistration path and SHA-256 must be supplied together"
        )
    if args.hpk_sequential_preregistration is not None:
        config["_hpk_sequential_preregistration"] = {
            "path": args.hpk_sequential_preregistration,
            "sha256": args.hpk_sequential_preregistration_sha256,
        }

    return config


if __name__ == "__main__":
    from benchmarks.rmbench.script.test_render import Sapien_TEST

    _install_termination_signal_handlers()
    Sapien_TEST()
    usr_args = parse_args_and_config()
    sequential_controller = None
    sequential_config = usr_args.pop("_hpk_sequential_preregistration", None)
    if sequential_config is not None:
        from roboharn_evo.agent.hpk.rmbench_sequential_controller import (
            RMBenchSequentialController,
        )
        from roboharn_evo.agent.hpk.sequential_experiment import load_preregistration

        published = load_preregistration(
            sequential_config["path"],
            sequential_config["sha256"],
        )
        sequential_controller = RMBenchSequentialController(published)
    main(usr_args, sequential_controller=sequential_controller)
