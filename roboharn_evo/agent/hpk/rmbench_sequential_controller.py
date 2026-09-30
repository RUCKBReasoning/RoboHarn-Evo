"""Real RMBench controller for the preregistered HPK two-episode gate.

The controller is intentionally small and evaluator-facing.  RMBench keeps one
model/runtime instance and asks this object for the exact next episode.  The
controller does not issue a probe request until the update receipt and child
snapshot pass :class:`IncrementalSequentialPairGate`.

All supported real backends use ``urllib.request.urlopen``.  While the
controller is active, that transport is wrapped by the invocation-wide quota
gate.  Every attempted HTTP call is reserved before socket I/O; image payload
occurrences and decoded image bytes are measured from the exact JSON request
body.  The preregistered backend set is checked before model construction, so a
different transport cannot silently bypass this adapter.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import json
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.parse import urlsplit

from roboharn_evo.agent.hpk.compatibility import normalize_agent_knowledge_config
from roboharn_evo.agent.hpk.policy_config import load_policy_config
from roboharn_evo.agent.hpk.schemas import (
    HPKValidationError,
    canonical_json_bytes,
    stable_content_id,
    validate_content_id,
)
from roboharn_evo.agent.hpk.sequential_experiment import (
    ArtifactRef,
    ControlUsage,
    EpisodeExecutionRequest,
    EpisodeExecutionResult,
    IncrementalSequentialPairGate,
    PublishedPreregistration,
    ResourceUsage,
    SequentialExperimentError,
    SequentialQuotaExceeded,
    SequentialRunResult,
    SnapshotState,
)

_SUPPORTED_REMOTE_BACKEND = "agent_api"
_IMAGE_B64_SUFFIXES = ("_image_b64", "_rgb_b64", "_png_b64", "_jpeg_b64")
_RUNTIME_PROVENANCE_PATH_ENV = "ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH"
_SEGMENTATION_ARTIFACT_DIR_ENV = "ROBOHARN_EVO_SEGMENTATION_ARTIFACT_DIR"
_MAX_RUNTIME_PROVENANCE_BYTES = 64 * 1024 * 1024


def _fail(message: str) -> None:
    raise SequentialExperimentError(message)


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object")
    return copy.deepcopy(dict(value))


def _finalization_receipt_artifact_ref(value: Any) -> ArtifactRef:
    path = "hpk_finalization_summary.finalization_receipt"
    payload = _mapping(value, path=path)
    expected = {"path", "sha256", "receipt_id"}
    if set(payload) != expected:
        _fail(
            f"{path} fields mismatch: "
            f"missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )
    try:
        validate_content_id(
            payload["receipt_id"],
            prefix="afkfinal",
            path=f"{path}.receipt_id",
        )
    except HPKValidationError as exc:
        raise SequentialExperimentError(str(exc)) from exc
    return ArtifactRef.from_mapping(
        {"path": payload["path"], "sha256": payload["sha256"]},
        path=path,
    )


def _nested(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key, {})
    return _mapping(value, path=key)


def _decode_image_string(value: str, *, key: str) -> bytes | None:
    encoded = value
    if value.startswith("data:image/") and ";base64," in value:
        encoded = value.split(";base64,", 1)[1]
    elif not key.lower().endswith(_IMAGE_B64_SUFFIXES):
        return None
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise SequentialExperimentError(
            f"outbound image payload {key!r} is invalid base64"
        ) from exc


def _outbound_images(value: Any, *, key: str = "") -> tuple[bytes, ...]:
    result: list[bytes] = []
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            result.extend(_outbound_images(child, key=str(child_key)))
    elif isinstance(value, list):
        for child in value:
            result.extend(_outbound_images(child, key=key))
    elif isinstance(value, str):
        decoded = _decode_image_string(value, key=key)
        if decoded is not None:
            result.append(decoded)
    return tuple(result)


def _request_image_payloads(target: Any) -> tuple[bytes, ...]:
    raw = getattr(target, "data", None)
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not isinstance(raw, bytes):
        _fail("outbound HTTP request body must be bytes")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # A non-JSON request is still counted as an external call.  The frozen
        # real controller supports images only in explicit JSON base64 fields.
        return ()
    return _outbound_images(payload)


def _read_regular_bytes(path: Path, *, label: str, maximum: int) -> bytes:
    """Read one explicit non-symlink file and reject concurrent replacement."""

    absolute = path.expanduser().absolute()
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise SequentialExperimentError(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a non-symlink regular file")
    if metadata.st_size <= 0 or metadata.st_size > maximum:
        _fail(f"{label} is empty or exceeds its byte limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) > maximum or len(raw) != after.st_size:
        _fail(f"{label} changed size while being read")
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        _fail(f"{label} changed while being read")
    return raw


class RMBenchSequentialController:
    """Incremental controller consumed by ``benchmarks.rmbench.eval_policy``."""

    def __init__(self, published: PublishedPreregistration) -> None:
        self.published = published
        self.preregistration = published.preregistration
        self.gate = IncrementalSequentialPairGate(
            self.preregistration,
            planned_usage=self.preregistration.planned_usage,
        )
        self._active_request: EpisodeExecutionRequest | None = None
        self._original_urlopen: Any = None
        self._artifact_capacity_reserved = False
        self._child_manifest_dirs: set[Path] = set()
        self._network_call_ordinal = 0
        self._effective_runtime_config: dict[str, Any] | None = None
        self._service_preflight_complete = False
        self._service_call_counts = {"agent_api": 0, "sam3": 0}

    @property
    def output_root(self) -> Path:
        return Path(self.preregistration.output_root)

    @property
    def should_stop(self) -> bool:
        return self.gate.should_stop

    def validate_config(self, usr_args: Mapping[str, Any]) -> None:
        """Fail before model/service/simulator/output construction on drift."""

        effective = self._validate_runtime_provenance()
        task = self.preregistration.task
        if (
            usr_args.get("task_name") != task["task_name"]
            or usr_args.get("task_config") != task["task_config"]
        ):
            _fail("RMBench task/config differs from the preregistration")
        task_path = usr_args.get("task_config_path")
        if not isinstance(task_path, str) or not task_path.strip():
            _fail("sequential experiment requires an explicit task_config_path")
        if Path(task_path).expanduser().absolute() != Path(
            task["task_definition"]["path"]
        ):
            _fail("RMBench task_config_path differs from the pinned task definition")
        task_bytes = _read_regular_bytes(
            Path(task_path),
            label="preregistered task definition",
            maximum=64 * 1024 * 1024,
        )
        if hashlib.sha256(task_bytes).hexdigest() != task["task_definition"]["sha256"]:
            _fail("RMBench task definition SHA-256 mismatch")
        if (
            effective.get("task_name") != task["task_name"]
            or effective.get("task_config") != task["task_config"]
            or effective.get("policy_name") != usr_args.get("policy_name")
            or effective.get("instruction_set") != usr_args.get("instruction_set")
            or effective.get("eval_start_seed") != self.preregistration.episodes[0].seed
            or effective.get("eval_start_seeds")
            != [self.preregistration.episodes[0].seed]
            or effective.get("n_per_worker") != 2
            or effective.get("num_workers") != 1
        ):
            _fail("runtime provenance task/schedule differs from preregistration")
        eval_config = _nested(usr_args, "eval")
        if (
            eval_config.get("test_num") != 2
            or eval_config.get("exact_seed_fail_closed") is not True
            or eval_config.get("step_limit")
            != self.preregistration.budgets.max_environment_actions_per_episode
        ):
            _fail("RMBench eval config must freeze two exact episodes and action cap")
        agent = normalize_agent_knowledge_config(_nested(usr_args, "agent"))
        if agent.get("enabled") is not True:
            _fail("sequential experiment requires the RoboHarn-Evo Agent runtime")
        hpk = _nested(agent, "hpk")
        if hpk.get("mode") != "evolving":
            _fail("sequential experiment requires agent.hpk.mode=evolving")
        if (
            hpk.get("snapshot_manifest")
            != self.preregistration.initial_snapshot.manifest_path
            or hpk.get("expected_manifest_sha256")
            != self.preregistration.initial_snapshot.manifest_sha256
        ):
            _fail("agent.hpk does not pin preregistered K0")
        expected_scope = (
            "integration"
            if self.preregistration.payload.get("hpk", self.preregistration.payload.get("afk"))["scope"]["name"]
            == "integration_development"
            else "formal_no_prior"
        )
        expected_snapshot_root = Path(
            self.preregistration.snapshot_output_root
        ).absolute()
        if (
            hpk.get("run_scope") != expected_scope
            or hpk.get("allow_expert_prior", False) is not False
            or hpk.get("allow_human_integration_prior", False) is not False
            or hpk.get("allow_oracle_evidence", False) is not False
            or hpk.get("all_hard_mismatch_behavior", "fail_closed") != "fail_closed"
            or hpk.get("max_prompt_chars", 4000) != 4000
            or Path(str(hpk.get("snapshot_output_root", ""))).absolute()
            != expected_snapshot_root
        ):
            _fail("agent.hpk evolving safety/configuration fields drifted")
        self._validate_hpk_policy_config(hpk)
        procedure = _nested(agent, "procedure_experience")
        if str(procedure.get("mode", "off") or "off").strip().lower() != "off":
            _fail("procedure experience must be off")
        recovery = _nested(agent, "recovery")
        if (
            recovery.get("enable_reobserve") is not True
            or recovery.get("enable_release_guard") is not True
            or recovery.get("grasp_transport_policy") != "strict"
            or recovery.get("action_geometry_repair_pending_policy") != "strict"
        ):
            _fail("RMBench recovery config differs from strict preregistered policy")
        pure = _nested(agent, "pure_tool_control")
        profile = self.preregistration.budgets
        if (
            pure.get("enabled") is not True
            or pure.get("retry_budget") != 0
            or pure.get("backend_error_budget") != 1
            or pure.get("max_rounds") != profile.max_semantic_rounds_per_episode
            or pure.get("max_control_turns") != profile.max_control_turns_per_episode
            or pure.get("max_no_progress_control_turns")
            != profile.max_no_progress_control_turns_per_episode
            or pure.get("skip_vla_rollout") is not True
        ):
            _fail("pure-tool-control config violates the no-retry acceptance profile")
        for backend_name in ("planner", "ood", "recovery"):
            backend = _nested(usr_args, backend_name)
            if backend.get("backend") != _SUPPORTED_REMOTE_BACKEND:
                _fail(f"{backend_name} backend is not covered by the quota transport")
            endpoint = _nested(backend, "agent_api").get("server_url")
            base = str(effective.get("agent_api_base_url", "") or "").rstrip("/")
            endpoint_path = {
                "planner": "plan",
                "ood": "ood",
                "recovery": "recover",
            }[backend_name]
            expected_endpoint = f"{base}/{endpoint_path}"
            if endpoint != expected_endpoint:
                _fail(f"{backend_name} Agent API endpoint differs from provenance")
        observation = _nested(agent, "observation_preprocess")
        segmentation = _nested(observation, "segmentation")
        configured_segmentation_root = str(
            os.environ.get(_SEGMENTATION_ARTIFACT_DIR_ENV, "") or ""
        ).strip()
        if (
            observation.get("enabled") is not True
            or observation.get("auto_objects") is not True
            or segmentation.get("backend") != "sam3"
            or segmentation.get("service_url") != effective.get("sam3_service_url")
            or _nested(observation, "oracle_objects").get("enabled") is not False
            or effective.get("perception_condition") != "no_oracle"
            or effective.get("oracle_objects_enabled") is not False
            or Path(configured_segmentation_root).expanduser().absolute()
            != Path(self.preregistration.segmentation_artifact_root)
        ):
            _fail("perception/SAM/no-oracle config differs from provenance")
        if (
            effective.get("expected_agent_model") != "gpt-5.5"
            or effective.get("expected_agent_api_mode") != "responses_compat"
            or effective.get("expected_reasoning_effort") != "xhigh"
            or effective.get("retry_budget") != 0
            or effective.get("backend_error_budget") != 1
            or effective.get("max_rounds") != profile.max_semantic_rounds_per_episode
            or effective.get("max_control_turns")
            != profile.max_control_turns_per_episode
            or effective.get("max_no_progress_control_turns")
            != profile.max_no_progress_control_turns_per_episode
            or effective.get("eval_step_limit")
            != profile.max_environment_actions_per_episode
        ):
            _fail("model identity or bounded-control provenance drifted")
        if usr_args.get("eval_video_log") is True:
            _fail("eval video must be disabled under the bounded artifact profile")
        self._validate_launch_config(usr_args)
        self._effective_runtime_config = effective

    def _validate_launch_config(self, usr_args: Mapping[str, Any]) -> None:
        reference = self.preregistration.launch_config
        raw = _read_regular_bytes(
            Path(reference.path),
            label="preregistered launch config",
            maximum=4 * 1024 * 1024,
        )
        if hashlib.sha256(raw).hexdigest() != reference.sha256:
            _fail("launch config SHA-256 mismatch")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SequentialExperimentError(
                f"launch config is invalid JSON: {exc}"
            ) from exc
        launch = _mapping(payload, path="launch_config")
        if raw != canonical_json_bytes(launch) + b"\n":
            _fail("launch config must be canonical JSON plus one LF")
        actual = _mapping(usr_args, path="effective_usr_args")
        if actual != launch:
            _fail("effective RMBench launch config differs from preregistration")

    def _validate_runtime_provenance(self) -> dict[str, Any]:
        source_value = str(
            os.environ.get(_RUNTIME_PROVENANCE_PATH_ENV, "") or ""
        ).strip()
        if not source_value:
            _fail("sequential experiment requires ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH")
        raw = _read_regular_bytes(
            Path(source_value),
            label="preregistered runtime provenance",
            maximum=_MAX_RUNTIME_PROVENANCE_BYTES,
        )
        provenance = self.preregistration.runtime_binding["provenance"]
        if hashlib.sha256(raw).hexdigest() != provenance["manifest_sha256"]:
            _fail("runtime provenance manifest SHA-256 mismatch")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SequentialExperimentError(
                f"runtime provenance is invalid JSON: {exc}"
            ) from exc
        manifest = _mapping(payload, path="runtime_provenance")
        effective = _mapping(
            manifest.get("effective_config"),
            path="runtime_provenance.effective_config",
        )
        if (
            manifest.get("schema") != provenance["schema"]
            or manifest.get("effective_config_sha256") != provenance["config_sha256"]
            or hashlib.sha256(canonical_json_bytes(effective)).hexdigest()
            != provenance["config_sha256"]
            or manifest.get("runtime_tree_sha256")
            != provenance["runtime_source_sha256"]
            or manifest.get("agent_service_identity_sha256")
            != provenance["model_identity_sha256"]
            or manifest.get("secrets_recorded") is not False
        ):
            _fail("runtime provenance identities disagree with preregistration")
        return effective

    def _validate_hpk_policy_config(self, hpk: Mapping[str, Any]) -> None:
        policy_refs = self.preregistration.runtime_binding["policy_refs"]
        for kind, path_key, hash_key in (
            (
                "geometry",
                "geometry_policy_path",
                "expected_geometry_policy_sha256",
            ),
            (
                "promotion",
                "promotion_policy_path",
                "expected_promotion_policy_sha256",
            ),
        ):
            path = hpk.get(path_key)
            digest = hpk.get(hash_key)
            if not isinstance(path, str) or not Path(path).is_absolute():
                _fail(f"agent.hpk.{path_key} must be an explicit absolute path")
            if not isinstance(digest, str):
                _fail(f"agent.hpk.{hash_key} is missing")
            try:
                loaded = load_policy_config(path, digest, kind)
            except Exception as exc:
                raise SequentialExperimentError(
                    f"agent.hpk {kind} policy failed exact load: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            expected = policy_refs[kind]
            if kind == "promotion":
                actual = {**loaded.identity(), "payload": loaded.payload}
            else:
                actual = loaded.identity()
            if actual != expected:
                _fail(f"agent.hpk {kind} policy differs from preregistration")

    def validate_effective_task_args(self, args: Mapping[str, Any]) -> None:
        if args.get("eval_video_log") is not False:
            _fail("effective RMBench eval_video_log must be false")
        if args.get("eval_step_limit") != (
            self.preregistration.budgets.max_environment_actions_per_episode
        ):
            _fail("effective RMBench action limit differs from preregistration")

    def bind_model_before_episode(self, model: Any) -> None:
        if not self._service_preflight_complete:
            _fail("service health preflight must complete before model construction")
        binder = getattr(model, "bind_hpk_sequential_run", None)
        if not callable(binder):
            _fail("model does not expose the sequential HPK run-binding API")
        bound = binder(self.preregistration.runtime_binding["run_binding"])
        if bound != self.preregistration.runtime_binding["run_binding"]:
            _fail("model changed the preregistered sequential run binding")

    def reserve_output_capacity(self) -> None:
        if self._artifact_capacity_reserved:
            _fail("output artifact capacity was already reserved")
        self.gate.reserve_planned_artifact_capacity()
        self.gate.claim_run_once()
        self._artifact_capacity_reserved = True

    def preflight_services(self) -> None:
        """Count and validate both local service health calls before model use."""

        if self._original_urlopen is None:
            _fail("service preflight requires the guarded quota transport")
        if self._service_preflight_complete:
            _fail("service health preflight may run exactly once")
        effective = self._effective_runtime_config
        if effective is None:
            _fail("service preflight requires validated runtime provenance")

        def fetch(url: str, *, label: str) -> dict[str, Any]:
            request = urllib_request.Request(url, method="GET")
            try:
                with urllib_request.urlopen(request, timeout=30) as response:
                    raw = response.read(1024 * 1024 + 1)
            except Exception as exc:
                raise SequentialExperimentError(
                    f"{label} health preflight failed: {type(exc).__name__}: {exc}"
                ) from exc
            if len(raw) > 1024 * 1024:
                _fail(f"{label} health response exceeds its byte limit")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SequentialExperimentError(
                    f"{label} health response is invalid JSON: {exc}"
                ) from exc
            return _mapping(payload, path=f"{label}_health")

        agent_base = str(effective["agent_api_base_url"]).rstrip("/")
        agent = fetch(f"{agent_base}/health", label="agent_api")
        if (
            str(agent.get("status", "")).lower() not in {"ok", "healthy"}
            or agent.get("model") != effective["expected_agent_model"]
            or agent.get("api_mode") != effective["expected_agent_api_mode"]
            or agent.get("reasoning_effort") != effective["expected_reasoning_effort"]
        ):
            _fail("Agent API health identity differs from runtime provenance")
        sam_base = str(effective["sam3_service_url"]).rstrip("/")
        sam = fetch(f"{sam_base}/health", label="sam3")
        segmentation_root = Path(
            self.preregistration.segmentation_artifact_root
        ).absolute()
        if (
            str(sam.get("status", "")).lower() not in {"ok", "healthy"}
            or sam.get("backend") != "sam3_image_text_prompt"
            or sam.get("output_root") != str(segmentation_root / "masks")
            or sam.get("allowed_output_root") != str(segmentation_root / "masks")
            or sam.get("allowed_input_roots") != [str(segmentation_root / "inputs")]
        ):
            _fail("SAM3 health preflight did not report ready")
        self._service_preflight_complete = True

    @contextlib.contextmanager
    def guarded_transport(self) -> Iterator[None]:
        """Install the only allowed outbound transport for the real pair."""

        if self._original_urlopen is not None:
            _fail("quota transport is already active")
        original = urllib_request.urlopen
        self._original_urlopen = original

        def guarded(target, *args, **kwargs):
            images = _request_image_payloads(target)
            service_class = self._service_class(target)
            self._check_service_call_capacity(service_class)

            def invoke(remaining: float):
                self._append_network_attempt(
                    target,
                    images=images,
                    service_class=service_class,
                )
                requested = kwargs.get("timeout")
                if requested is None and len(args) >= 2:
                    requested = args[1]
                timeout = remaining
                if isinstance(requested, (int, float)) and not isinstance(
                    requested, bool
                ):
                    timeout = min(timeout, float(requested))
                call_kwargs = dict(kwargs)
                bounded_timeout = max(0.001, timeout)
                if len(args) >= 2:
                    call_args = (*args[:1], bounded_timeout, *args[2:])
                    call_kwargs.pop("timeout", None)
                else:
                    call_args = args
                    call_kwargs["timeout"] = bounded_timeout
                return original(target, *call_args, **call_kwargs)

            return self.gate.metered_io.external_call(
                invoke,
                image_payloads=images,
            )

        urllib_request.urlopen = guarded
        try:
            yield
        finally:
            urllib_request.urlopen = original
            self._original_urlopen = None

    def _service_class(self, target: Any) -> str:
        effective = self._effective_runtime_config
        if effective is None:
            _fail("outbound transport requires validated runtime provenance")
        raw_url = getattr(target, "full_url", target)
        if not isinstance(raw_url, str):
            _fail("outbound HTTP request lacks its URL")
        parsed = urlsplit(raw_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        expected = {
            "agent_api": str(effective["agent_api_base_url"]).rstrip("/"),
            "sam3": str(effective["sam3_service_url"]).rstrip("/"),
        }
        matches = [name for name, value in expected.items() if origin == value]
        if len(matches) != 1:
            _fail("outbound HTTP origin is not covered by the preregistration")
        return matches[0]

    def _check_service_call_capacity(self, service_class: str) -> None:
        budget_field = {
            "agent_api": "max_agent_api_calls",
            "sam3": "max_sam3_calls",
        }[service_class]
        configured_cap = getattr(self.preregistration.budgets, budget_field)
        if service_class == "agent_api" and configured_cap == 0:
            return
        if self._service_call_counts[service_class] + 1 > configured_cap:
            raise SequentialQuotaExceeded(
                f"{service_class} call quota would be exceeded before I/O"
            )

    def _append_network_attempt(
        self,
        target: Any,
        *,
        images: Sequence[bytes],
        service_class: str,
    ) -> None:
        raw_url = getattr(target, "full_url", target)
        if not isinstance(raw_url, str) or not raw_url:
            _fail("outbound HTTP request lacks its URL")
        parsed = urlsplit(raw_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            _fail("outbound HTTP request URL is unsupported")
        raw_body = getattr(target, "data", b"") or b""
        if isinstance(raw_body, str):
            raw_body = raw_body.encode("utf-8")
        if not isinstance(raw_body, bytes):
            _fail("outbound HTTP request body must be bytes")
        request = self._active_request
        payload = {
            "schema": "roboharn_evo/hpk/sequential_network_attempt/v1",
            "preregistration_id": self.preregistration.preregistration_id,
            "run_id": self.preregistration.run_id,
            "run_binding_id": self.preregistration.runtime_binding["run_binding"][
                "run_binding_id"
            ],
            "call_ordinal": self._network_call_ordinal,
            "episode_ordinal": None if request is None else request.ordinal,
            "episode_role": None if request is None else request.role,
            "service_class": service_class,
            "endpoint_origin_sha256": hashlib.sha256(
                f"{parsed.scheme}://{parsed.netloc}".encode()
            ).hexdigest(),
            "endpoint_path": parsed.path or "/",
            "request_sha256": hashlib.sha256(raw_body).hexdigest(),
            "image_count": len(images),
            "image_bytes": sum(len(value) for value in images),
            "delivery_status": "sent_or_attempted",
        }
        payload["network_attempt_id"] = stable_content_id("afknetcall", payload)
        encoded = canonical_json_bytes(payload) + b"\n"
        ledger = self.output_root / "network_usage.jsonl"

        def append(raw: bytes) -> None:
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(ledger, flags, 0o600)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    _fail("network usage ledger must be a regular file")
                os.fchmod(descriptor, 0o600)
                offset = 0
                while offset < len(raw):
                    offset += os.write(descriptor, raw[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        self.gate.metered_io.artifact_write(encoded, append)
        self._service_call_counts[service_class] += 1
        self._network_call_ordinal += 1

    def begin_episode(self) -> EpisodeExecutionRequest:
        if not self._artifact_capacity_reserved:
            _fail("artifact capacity must be reserved before evaluator output")
        request = self.gate.begin_episode()
        self._active_request = request
        return request

    def validate_instruction(
        self,
        *,
        request: EpisodeExecutionRequest,
        instruction: str,
        instruction_source_sha256: str,
    ) -> None:
        if request is not self._active_request:
            _fail("instruction validation does not match the active episode")
        task = self.preregistration.task
        if instruction != task["instruction"]:
            _fail("generated instruction differs from preregistered exact text")
        import hashlib

        if (
            hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            != task["instruction_sha256"]
        ):
            _fail("generated instruction hash differs from preregistration")
        if instruction_source_sha256 != task["instruction_source"]["sha256"]:
            _fail("instruction catalog hash differs from preregistration")

    def select_instruction(
        self,
        *,
        request: EpisodeExecutionRequest,
        choices: Sequence[str],
        instruction_source_sha256: str,
    ) -> str:
        exact = self.preregistration.task["instruction"]
        if exact not in choices:
            _fail("preregistered instruction is absent from the generated catalog")
        self.validate_instruction(
            request=request,
            instruction=exact,
            instruction_source_sha256=instruction_source_sha256,
        )
        return exact

    def validate_model_pin(self, model: Any, request: EpisodeExecutionRequest) -> None:
        runtime = getattr(model, "hpk_runtime", None)
        if runtime is None or getattr(runtime, "evolving_enabled", False) is not True:
            _fail("model does not expose the evolving HPK runtime")
        lease = getattr(runtime, "episode_lease", None)
        if lease is None:
            _fail("evolving runtime did not pin an episode lease")
        if lease.expected_episode_id != request.ordinal:
            _fail("evolving runtime lease episode differs from the ordered schedule")
        if lease.snapshot_ref.identity() != request.parent_snapshot.identity():
            _fail("evolving runtime lease did not pin the requested snapshot")
        if runtime.episode_runtime_binding() != request.runtime_binding:
            _fail("evolving runtime binding differs from preregistered request")

    def complete_episode(
        self,
        *,
        model: Any,
        hpk_finalization_summary: Mapping[str, Any],
        environment_actions: int,
        agent_status: Mapping[str, Any],
    ) -> EpisodeExecutionResult:
        request = self._active_request
        if request is None:
            _fail("there is no active preregistered episode")
        summary = _mapping(hpk_finalization_summary, path="hpk_finalization_summary")
        runtime = getattr(model, "hpk_runtime", None)
        coordinated = getattr(runtime, "finalization_result", None)
        if coordinated is None:
            _fail("evolving runtime did not retain the coordinated finalization")
        semantic = getattr(runtime, "semantic_finalization_result", None)
        next_ref = coordinated.next_snapshot_ref
        child = None
        if coordinated.finalization.child is not None:
            child = SnapshotState.from_loaded(next_ref.snapshot)
            self._child_manifest_dirs.add(Path(child.manifest_path).parent)
        control_errors = int(agent_status.get("control_backend_error_count", 0) or 0)
        recovery_errors = int(agent_status.get("recovery_backend_error_count", 0) or 0)
        if control_errors > 1 or recovery_errors > 1:
            _fail("backend error counters imply an automatic retry")
        outcome = EpisodeExecutionResult(
            preregistration_id=request.preregistration_id,
            ordinal=request.ordinal,
            role=request.role,
            seed=request.seed,
            loaded_snapshot=request.parent_snapshot,
            runtime_binding_id=request.runtime_binding["binding_id"],
            control_usage=ControlUsage(
                environment_actions=int(environment_actions),
                control_turns=int(agent_status.get("control_turn_count", 0) or 0),
                no_progress_control_turns=int(
                    agent_status.get("no_progress_control_turn_count", 0) or 0
                ),
                semantic_rounds=int(agent_status.get("semantic_round_index", 0) or 0),
                backend_retries=0,
                seed_retries=0,
            ),
            provider_usage=self.gate.active_usage_delta(),
            rollout_manifest=ArtifactRef.from_mapping(
                summary["rollout_import_manifest"],
                path="hpk_finalization_summary.rollout_import_manifest",
            ),
            finalization_receipt=_finalization_receipt_artifact_ref(
                summary["finalization_receipt"]
            ),
            finalization_status=str(summary["finalization"]["status"]),
            child_snapshot=child,
            semantic_knowledge=(
                () if semantic is None else tuple(semantic.knowledge)
            ),
            semantic_newly_accepted=bool(
                semantic is not None and semantic.newly_accepted
            ),
            semantic_knowledge_used=bool(
                getattr(runtime, "semantic_knowledge_used", False)
            ),
        )
        self.gate.complete_episode(outcome)
        self._active_request = None
        return outcome

    def finalize_run(self, output_root: str | os.PathLike[str]) -> SequentialRunResult:
        result = self.gate.result()
        self._validate_network_ledger(result.final_usage)
        actual_artifact_bytes = self._artifact_tree_bytes(
            {
                Path(output_root).resolve(),
                Path(self.preregistration.snapshot_output_root).resolve(),
                Path(self.preregistration.segmentation_artifact_root).resolve(),
                *self._child_manifest_dirs,
            }
        )
        if actual_artifact_bytes > self.preregistration.planned_usage.artifact_bytes:
            _fail("actual artifact bytes exceeded the statically reserved capacity")
        return SequentialRunResult(
            preregistration_id=result.preregistration_id,
            status=result.status,
            update=result.update,
            probe=result.probe,
            learned_entry_ids=result.learned_entry_ids,
            final_usage=ResourceUsage(
                external_model_calls=result.final_usage.external_model_calls,
                images=result.final_usage.images,
                image_bytes=result.final_usage.image_bytes,
                artifact_bytes=actual_artifact_bytes,
            ),
            learned_knowledge=result.learned_knowledge,
        )

    def _validate_network_ledger(self, usage: ResourceUsage) -> None:
        ledger = self.output_root / "network_usage.jsonl"
        try:
            raw = ledger.read_bytes()
        except OSError as exc:
            raise SequentialExperimentError(
                f"network usage ledger is unavailable: {exc}"
            ) from exc
        records: list[dict[str, Any]] = []
        for ordinal, line in enumerate(raw.splitlines()):
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SequentialExperimentError(
                    f"network usage ledger line {ordinal} is invalid: {exc}"
                ) from exc
            if (
                not isinstance(record, dict)
                or record.get("call_ordinal") != ordinal
                or record.get("run_id") != self.preregistration.run_id
                or record.get("run_binding_id")
                != self.preregistration.runtime_binding["run_binding"]["run_binding_id"]
            ):
                _fail("network usage ledger binding or order mismatch")
            identity = dict(record)
            network_attempt_id = identity.pop("network_attempt_id", None)
            if network_attempt_id != stable_content_id("afknetcall", identity):
                _fail("network usage ledger content identity mismatch")
            records.append(record)
        if (
            len(records) != usage.external_model_calls
            or sum(int(item["image_count"]) for item in records) != usage.images
            or sum(int(item["image_bytes"]) for item in records) != usage.image_bytes
        ):
            _fail("network usage ledger does not reconcile with quota totals")
        observed_classes = {
            name: sum(item.get("service_class") == name for item in records)
            for name in self._service_call_counts
        }
        if observed_classes != self._service_call_counts:
            _fail("network usage ledger service-class totals mismatch")
        agent_api_cap = self.preregistration.budgets.max_agent_api_calls
        if (
            agent_api_cap != 0 and observed_classes["agent_api"] > agent_api_cap
        ) or observed_classes["sam3"] > self.preregistration.budgets.max_sam3_calls:
            _fail("network usage ledger exceeds a service-specific call cap")

    @staticmethod
    def _artifact_tree_bytes(roots: set[Path]) -> int:
        total = 0
        seen: set[tuple[int, int]] = set()
        for root in sorted(roots):
            if not root.exists():
                continue
            for path in (root, *root.rglob("*")):
                metadata = path.lstat()
                if path.is_symlink():
                    _fail("experiment artifact tree must not contain symlinks")
                if not path.is_file():
                    continue
                identity = (metadata.st_dev, metadata.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                total += metadata.st_size
        return total


__all__ = ["RMBenchSequentialController"]
