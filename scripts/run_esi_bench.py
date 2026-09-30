from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    AgentApiESIEvaluatedModel,
    AgentApiESIMultimodalRetrievalBackend,
    ESIKnowledgeError,
    ESIPublicHistoryStep,
    ESIPublicStepContext,
    ESITaskOnlyRetrievalRuntime,
    VLMCurrentImageSubtaskKnowledgeRetriever,
    VLMFlatLessonRetriever,
    capture_store_contents,
    load_frozen_task_store,
    load_shuffle_manifest,
    load_split_manifest,
    require_store_unchanged,
    validate_store_split_provenance,
)

_UPSTREAM_COMMIT = "3c1756396f32b1a90c1f72356a7fde45f418e179"
_NVIDIA_GRAPHICS_ENVIRONMENT = {
    "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
    "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
}
_LOOPBACK_NO_PROXY = ("127.0.0.1", "localhost", "::1")
_QUERY_RESULT_KEYS = (
    "handled",
    "operation",
    "action",
    "success",
    "error",
    "reason",
)
_AUDIT_FILENAMES = frozenset(
    {
        "hpk_action_trace.jsonl",
        "hpk_prompt_projection.txt",
        "hpk_query.jsonl",
        "hpk_retrieval.jsonl",
        "frozen_store_before_after_check.json",
        "frozen_state.json",
        "metrics.json",
        "question_public.json",
        "run_config.json",
        "upstream_answer.json",
    }
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="RoboHarn-Evo runner for the ESI Task Knowledge diagnostic.",
        add_help=True,
    )
    parser.add_argument(
        "--esi-root",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks" / "esi_bench" / "upstream",
    )
    parser.add_argument(
        "--hpk-mode",
        choices=("off", "flat", "task", "shuffled"),
        required=True,
    )
    parser.add_argument("--hpk-store-root", type=Path)
    parser.add_argument("--hpk-shuffle-manifest", type=Path)
    parser.add_argument("--hpk-candidate-cap", type=int, default=8)
    parser.add_argument("--hpk-context-token-cap", type=int, default=256)
    parser.add_argument("--hpk-audit-root", type=Path, required=True)
    parser.add_argument("--retrieval-planner-url")
    parser.add_argument("--retrieval-provider")
    parser.add_argument("--retrieval-model")
    parser.add_argument("--evaluated-model-agent-api-url")
    parser.add_argument(
        "--evaluated-model-reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--canonical-render-audit", type=Path)
    parser.add_argument(
        "--split-part",
        choices=("source", "development", "heldout"),
        default="heldout",
    )
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--overwrite-audit", action="store_true")
    parser.add_argument("--capture-frozen-state-only", action="store_true")
    args, upstream_args = parser.parse_known_args()
    if not upstream_args:
        parser.error(
            "official ESI arguments such as --task and --metadata are required"
        )
    if args.hpk_mode != "off":
        if (
            args.hpk_store_root is None
            or not args.retrieval_planner_url
            or not args.retrieval_provider
            or not args.retrieval_model
        ):
            parser.error(
                "active HPK modes require store, retrieval URL, provider, and model"
            )
        if args.split_manifest is None:
            parser.error("active HPK modes require --split-manifest")
    if args.hpk_mode == "shuffled" and args.hpk_shuffle_manifest is None:
        parser.error("shuffled mode requires --hpk-shuffle-manifest")
    if args.formal and args.split_manifest is None:
        parser.error("formal evaluation requires --split-manifest")
    if args.formal and args.split_part != "heldout":
        parser.error("formal evaluation is restricted to the heldout split")
    if args.formal and args.canonical_render_audit is None:
        parser.error("formal evaluation requires --canonical-render-audit")
    if args.capture_frozen_state_only:
        if args.hpk_mode != "off":
            parser.error("frozen-state capture requires HPK off")
        if args.split_manifest is None or args.split_part != "development":
            parser.error("frozen-state capture requires the development split")
        if args.formal or args.evaluated_model_agent_api_url:
            parser.error("frozen-state capture cannot run a formal/evaluated model")
    return args, upstream_args


def _nvidia_graphics_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(_NVIDIA_GRAPHICS_ENVIRONMENT)
    for key in ("NO_PROXY", "no_proxy"):
        values = [item.strip() for item in environment.get(key, "").split(",")]
        values = [item for item in values if item]
        for item in _LOOPBACK_NO_PROXY:
            if item not in values:
                values.append(item)
        environment[key] = ",".join(values)
    return environment


def _nvidia_vulkan_devices(vulkaninfo_output: str) -> list[str]:
    devices = []
    for line in vulkaninfo_output.splitlines():
        key, separator, raw_value = line.partition("=")
        if separator and key.strip() == "deviceName":
            value = raw_value.strip()
            if value.casefold().startswith("nvidia "):
                devices.append(value)
    return devices


def renderer_preflight(
    *,
    formal: bool,
    canonical_render_validated: bool = False,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    runtime_environment = dict(environment or _nvidia_graphics_environment())
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    omnigibson_available = importlib.util.find_spec("omnigibson") is not None
    vulkan_devices: list[str] = []
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=runtime_environment,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result = {
            "available": False,
            "compatible_for_formal": False,
            "warning": f"nvidia-smi unavailable: {type(exc).__name__}: {exc}",
            "gpus": [],
            "incompatible_gpus": [],
            "hardware_warning": False,
            "canonical_render_validated": bool(canonical_render_validated),
            "omnigibson_available": omnigibson_available,
        }
    else:
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        danger = (
            "blackwell",
            "rtx 50",
            "5090",
            "5080",
            "5070",
            "b100",
            "b200",
            "gb200",
            "gb300",
        )
        incompatible = [
            row for row in rows if any(token in row.casefold() for token in danger)
        ]
        available = completed.returncode == 0 and bool(rows)
        result = {
            "available": available,
            "compatible_for_formal": (
                available
                and omnigibson_available
                and (not incompatible or canonical_render_validated)
            ),
            "warning": (
                "Official ESI-Bench warns that 50-series/Blackwell rendering "
                "may be invalid; canonical render inspection is required."
            ),
            "gpus": rows,
            "incompatible_gpus": incompatible,
            "hardware_warning": bool(incompatible),
            "canonical_render_validated": bool(canonical_render_validated),
            "omnigibson_available": omnigibson_available,
        }
    try:
        vulkan = subprocess.run(
            ["vulkaninfo", "--summary"],
            check=False,
            capture_output=True,
            env=runtime_environment,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        vulkan_available = False
        vulkan_diagnostic = f"{type(exc).__name__}: {exc}"
    else:
        vulkan_output = "\n".join((vulkan.stdout or "", vulkan.stderr or ""))
        vulkan_devices = _nvidia_vulkan_devices(vulkan_output)
        vulkan_available = vulkan.returncode == 0 and bool(vulkan_devices)
        if vulkan_available:
            vulkan_diagnostic = "NVIDIA devices: " + ", ".join(vulkan_devices)
        elif vulkan.returncode == 0:
            vulkan_diagnostic = (
                "vulkaninfo succeeded but enumerated no NVIDIA device; "
                "software Vulkan is not accepted"
            )
        else:
            vulkan_diagnostic = " ".join((vulkan.stderr or vulkan.stdout).split())[:500]
    runtime_ready = bool(
        result["available"] and omnigibson_available and vulkan_available
    )
    result["vulkan_available"] = vulkan_available
    result["vulkan_diagnostic"] = vulkan_diagnostic
    result["vulkan_nvidia_devices"] = vulkan_devices
    result["nvidia_graphics_environment"] = dict(_NVIDIA_GRAPHICS_ENVIRONMENT)
    result["runtime_ready"] = runtime_ready
    result["compatible_for_formal"] = bool(
        runtime_ready
        and (not result.get("incompatible_gpus") or canonical_render_validated)
    )
    print("[ESI renderer preflight] " + result["warning"], file=sys.stderr)
    if not runtime_ready:
        missing = []
        if not result["available"]:
            missing.append("NVIDIA GPU is not visible through nvidia-smi")
        if not omnigibson_available:
            missing.append("OmniGibson is not installed")
        if not vulkan_available:
            missing.append(
                "NVIDIA Vulkan is unavailable or did not enumerate an NVIDIA GPU"
            )
        raise RuntimeError("ESI run refused before model call: " + "; ".join(missing))
    if formal and not result["compatible_for_formal"]:
        raise RuntimeError(
            "formal ESI run refused: compatible renderer hardware was not established"
        )
    return result


def load_canonical_render_audit(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot read canonical render audit") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("canonical render audit must be one JSON object")
    required = {
        "schema",
        "upstream_commit",
        "human_inspected",
        "renderer_valid",
        "artifact_paths",
        "notes",
    }
    if set(value) != required:
        raise RuntimeError("canonical render audit fields mismatch")
    if value["schema"] not in {"roboharn_evo/esi_bench/canonical_render_audit/v1", "tcm/esi_bench/canonical_render_audit/v1"}:
        raise RuntimeError("canonical render audit schema mismatch")
    if value["upstream_commit"] != _UPSTREAM_COMMIT:
        raise RuntimeError("canonical render audit upstream commit mismatch")
    if value["human_inspected"] is not True or value["renderer_valid"] is not True:
        raise RuntimeError("canonical renderer was not human-validated")
    artifacts = value["artifact_paths"]
    if (
        isinstance(artifacts, (str, bytes))
        or not isinstance(artifacts, list)
        or not artifacts
    ):
        raise RuntimeError("canonical render audit needs artifact paths")
    resolved = []
    for item in artifacts:
        candidate = Path(str(item)).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise RuntimeError("canonical render artifact is missing")
        resolved.append(str(candidate))
    return {**dict(value), "artifact_paths": resolved, "audit_path": str(source)}


def _run_with_deferred_shutdown(
    pipeline: Any,
    upstream_config: Any,
    hook: Any,
) -> tuple[
    Any,
    tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]] | None,
]:
    og_module = getattr(pipeline, "og", None)
    original_shutdown = getattr(og_module, "shutdown", None)
    if not callable(original_shutdown):
        try:
            return pipeline.run_one(upstream_config, prompt_hook=hook), None
        except _FrozenStateCaptured as captured:
            return captured.payload, None

    requested: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def defer_shutdown(*args: Any, **kwargs: Any) -> None:
        requested.append((args, kwargs))

    og_module.shutdown = defer_shutdown
    try:
        try:
            result = pipeline.run_one(upstream_config, prompt_hook=hook)
        except _FrozenStateCaptured as captured:
            result = captured.payload
    finally:
        og_module.shutdown = original_shutdown
    pending = (
        (original_shutdown, requested[-1][0], requested[-1][1]) if requested else None
    )
    return result, pending


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def _prepare_audit_root(root: Path, *, overwrite: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    existing = tuple(root.iterdir())
    if not existing:
        return
    if not overwrite:
        raise FileExistsError(f"ESI audit root is not empty: {root}")
    unknown = [path.name for path in existing if path.name not in _AUDIT_FILENAMES]
    if unknown or any(not path.is_file() for path in existing):
        raise FileExistsError(
            "refusing to overwrite an ESI audit root with unknown artifacts: "
            + ", ".join(sorted(unknown))
        )
    for path in existing:
        path.unlink()


def _safe_history_item(value: Mapping[str, Any]) -> ESIPublicHistoryStep:
    result = value.get("action_result_public")
    if not isinstance(result, Mapping):
        result = {}
    return ESIPublicHistoryStep(
        step=int(value["step"]),
        action=str(value.get("action") or "no action"),
        answer=str(value.get("answer") or "not sure"),
        confidence=float(value.get("confidence", 0.0)),
        reasoning=str(value.get("reasoning") or "no reasoning provided"),
        action_result_public={
            key: copy.deepcopy(result[key])
            for key in _QUERY_RESULT_KEYS
            if key in result
        },
    )


class _FrozenStateCaptured(RuntimeError):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__("initial public state captured before model execution")
        self.payload = dict(payload)


class _CaptureOnlyModel:
    def generate_json(self, **_kwargs: Any) -> Any:
        raise RuntimeError("capture-only mode reached the evaluated model")


class _EpisodePromptHook:
    def __init__(
        self,
        *,
        runtime: ESITaskOnlyRetrievalRuntime,
        small_task: str | None,
        big_task: str,
        audit_root: Path,
        capture_only: bool = False,
    ) -> None:
        self.runtime = runtime
        self.small_task = small_task
        self.big_task = big_task
        self.audit_root = audit_root
        self.capture_only = capture_only
        self.first_public_question: dict[str, Any] | None = None
        self.audits: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        official_prompt: str,
        task_name: str,
        step: int,
        current_image_path: Path,
        public_history: tuple[dict[str, Any], ...],
    ) -> str:
        history = tuple(_safe_history_item(item) for item in public_history)
        if history:
            latest = history[-1]
            evidence_status = (
                f"{len(history)} public action-observation steps are available; "
                f"the latest public action was {latest.action} with result "
                + json.dumps(
                    dict(latest.action_result_public),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        else:
            evidence_status = "no public action-observation evidence has been collected"
        context = ESIPublicStepContext(
            small_task=self.small_task or task_name,
            big_task=self.big_task,
            question_or_goal=official_prompt,
            step=step,
            history=history,
            public_object_roles=(),
            evidence_status=evidence_status,
        )
        if self.capture_only:
            if step != 1 or history:
                raise ESIKnowledgeError(
                    "frozen-state capture must occur at the empty-history first step"
                )
            self.first_public_question = context.to_dict()
            raise _FrozenStateCaptured(
                {
                    "schema": "roboharn_evo/esi_bench/initial_state_capture/v1",
                    "step": step,
                    "question_or_goal": official_prompt,
                    "current_image_path": str(current_image_path.resolve()),
                    "public_history": [],
                }
            )
        projection = self.runtime.project(
            official_prompt=official_prompt,
            model_arguments={},
            context=context,
            current_image_path=current_image_path,
        )
        audit_payload = projection.audit.to_dict()
        query_payload = {
            "step": step,
            "mode": self.runtime.mode,
            "query": audit_payload["query"],
        }
        _append_jsonl(self.audit_root / "hpk_query.jsonl", query_payload)
        _append_jsonl(self.audit_root / "hpk_retrieval.jsonl", audit_payload)
        with (self.audit_root / "hpk_prompt_projection.txt").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(f"\n=== step {step} / {self.runtime.mode} ===\n")
            handle.write(projection.prompt)
            handle.write("\n")
        if self.first_public_question is None:
            self.first_public_question = context.to_dict()
        self.audits.append(audit_payload)
        return projection.prompt


def _task_label(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").split())


def _find_split_entry(
    manifest,
    *,
    split_part: str,
    task: str,
    metadata: Path,
    question_index: int,
):
    target = metadata.expanduser().resolve()
    split = getattr(manifest, split_part)
    matches = [
        item
        for item in split
        if Path(item.metadata_path).expanduser().resolve() == target
        and item.question_index == question_index
    ]
    if len(matches) != 1:
        raise ESIKnowledgeError(
            "requested task/metadata is not exactly one instance in the frozen "
            f"{split_part} split"
        )
    try:
        public_metadata = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ESIKnowledgeError("cannot validate split metadata labels") from exc
    if not isinstance(public_metadata, Mapping):
        raise ESIKnowledgeError("split metadata must be one public object")
    runner_task = public_metadata.get("runner_task") or public_metadata.get("task")
    small_task = public_metadata.get("small_task")
    if _task_label(runner_task) != _task_label(task):
        raise ESIKnowledgeError("requested runner task differs from frozen metadata")
    if _task_label(small_task) != _task_label(matches[0].small_task):
        raise ESIKnowledgeError("frozen small-task label differs from metadata")
    return matches[0]


def main() -> int:
    args, upstream_args = _parse_args()
    graphics_environment = _nvidia_graphics_environment()
    os.environ.update(
        {
            key: graphics_environment[key]
            for key in (*_NVIDIA_GRAPHICS_ENVIRONMENT, "NO_PROXY", "no_proxy")
        }
    )
    audit_root = args.hpk_audit_root.expanduser().resolve()
    _prepare_audit_root(audit_root, overwrite=args.overwrite_audit)
    render_audit = (
        load_canonical_render_audit(args.canonical_render_audit)
        if args.canonical_render_audit is not None
        else None
    )
    preflight = renderer_preflight(
        formal=args.formal,
        environment=graphics_environment,
        canonical_render_validated=(
            render_audit is not None and render_audit["renderer_valid"] is True
        ),
    )

    upstream_root = args.esi_root.expanduser().resolve()
    active_root = upstream_root / "src" / "active_explore"
    if not (active_root / "pipeline.py").is_file():
        raise FileNotFoundError(f"ESI application is missing: {active_root}")
    sys.path.insert(0, str(active_root))
    import pipeline  # noqa: PLC0415

    upstream_config = pipeline.parse_args(upstream_args)
    if args.capture_frozen_state_only:
        if upstream_config.max_steps != 1:
            raise ValueError("frozen-state capture requires upstream --max-steps 1")

        def _capture_only_model_factory(
            _provider: str, _api_key: str | None, _model: str
        ) -> _CaptureOnlyModel:
            return _CaptureOnlyModel()

        pipeline.build_model_client = _capture_only_model_factory
    evaluated_model = None
    evaluated_model_health = None
    if args.evaluated_model_agent_api_url:
        if upstream_config.provider != "gpt":
            raise ValueError(
                "--evaluated-model-agent-api-url requires upstream --provider gpt"
            )
        evaluated_model = AgentApiESIEvaluatedModel(
            args.evaluated_model_agent_api_url,
            model=upstream_config.model or "gpt-5",
            reasoning_effort=args.evaluated_model_reasoning_effort,
        )
        evaluated_model_health = evaluated_model.preflight()

        def _agent_api_model_factory(
            provider: str, api_key: str | None, model: str
        ) -> AgentApiESIEvaluatedModel:
            if provider != "gpt" or model != evaluated_model.model:
                raise ValueError(
                    "upstream evaluated-model request differs from Agent API preflight"
                )
            return evaluated_model

        pipeline.build_model_client = _agent_api_model_factory
    manifest = (
        None
        if args.split_manifest is None
        else load_split_manifest(args.split_manifest)
    )
    split_entry = None
    if manifest is not None:
        split_entry = _find_split_entry(
            manifest,
            split_part=args.split_part,
            task=upstream_config.task,
            metadata=upstream_config.metadata,
            question_index=upstream_config.question_index,
        )

    store = None
    store_before: dict[str, bytes] = {}
    shuffle_before: bytes | None = None
    retrieval_health: dict[str, Any] | None = None
    if args.hpk_mode != "off":
        store = load_frozen_task_store(args.hpk_store_root)
        validate_store_split_provenance(store, manifest)
        store_before = capture_store_contents(args.hpk_store_root)
        backend = AgentApiESIMultimodalRetrievalBackend(
            args.retrieval_planner_url,
            model=args.retrieval_model,
        )
        retrieval_health = backend.preflight()
        task_retriever = VLMCurrentImageSubtaskKnowledgeRetriever(backend)
        flat_retriever = VLMFlatLessonRetriever(backend)
        shuffle = (
            load_shuffle_manifest(args.hpk_shuffle_manifest)
            if args.hpk_mode == "shuffled"
            else None
        )
        if args.hpk_mode == "shuffled":
            shuffle_before = (
                args.hpk_shuffle_manifest.expanduser().resolve().read_bytes()
            )
    else:
        task_retriever = None
        flat_retriever = None
        shuffle = None

    runtime = ESITaskOnlyRetrievalRuntime(
        mode=args.hpk_mode,
        candidate_cap=args.hpk_candidate_cap,
        context_token_cap=args.hpk_context_token_cap,
        store=store,
        task_retriever=task_retriever,
        flat_retriever=flat_retriever,
        shuffle_manifest=shuffle,
    )
    big_task = (
        split_entry.big_task
        if split_entry is not None
        else "public task family unavailable"
    )
    hook = _EpisodePromptHook(
        runtime=runtime,
        small_task=(split_entry.small_task if split_entry is not None else None),
        big_task=big_task,
        audit_root=audit_root,
        capture_only=args.capture_frozen_state_only,
    )
    run_config = {
        "scientific_scope": "ESI RQ2 Task Knowledge only",
        "upstream_commit": _UPSTREAM_COMMIT,
        "hpk_mode": args.hpk_mode,
        "formal": args.formal,
        "split_part": args.split_part if manifest is not None else None,
        "upstream_arguments": upstream_args,
        "candidate_cap": args.hpk_candidate_cap,
        "context_token_cap": args.hpk_context_token_cap,
        "split_manifest_path": (
            str(args.split_manifest.expanduser().resolve())
            if args.split_manifest is not None
            else None
        ),
        "split_entry": split_entry.to_dict() if split_entry is not None else None,
        "frozen_store_root": (
            str(args.hpk_store_root.expanduser().resolve())
            if args.hpk_store_root is not None
            else None
        ),
        "frozen_store_contents": (
            {
                name: content.decode("utf-8")
                for name, content in sorted(store_before.items())
            }
            if store_before
            else None
        ),
        "shuffle_manifest_path": (
            str(args.hpk_shuffle_manifest.expanduser().resolve())
            if args.hpk_shuffle_manifest is not None
            else None
        ),
        "shuffle_manifest_content": (
            shuffle_before.decode("utf-8") if shuffle_before is not None else None
        ),
        "retrieval_provider": args.retrieval_provider,
        "retrieval_model": args.retrieval_model,
        "retrieval_current_image_grounded": args.hpk_mode != "off",
        "retrieval_health": retrieval_health,
        "evaluated_model_transport": (
            "capture_only_no_model"
            if args.capture_frozen_state_only
            else (
                "existing_agent_api_responses"
                if evaluated_model is not None
                else "upstream_provider_client"
            )
        ),
        "evaluated_model_agent_api_url": (
            args.evaluated_model_agent_api_url if evaluated_model is not None else None
        ),
        "evaluated_model_reasoning_effort": args.evaluated_model_reasoning_effort,
        "evaluated_model_health": evaluated_model_health,
        "heldout_store_update_enabled": False,
        "capture_frozen_state_only": args.capture_frozen_state_only,
        "renderer_preflight": preflight,
        "canonical_render_audit": render_audit,
    }
    (audit_root / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    result, pending_shutdown = _run_with_deferred_shutdown(
        pipeline, upstream_config, hook
    )
    elapsed = time.monotonic() - started
    if args.capture_frozen_state_only:
        if (
            not isinstance(result, Mapping)
            or result.get("schema") not in {"roboharn_evo/esi_bench/initial_state_capture/v1", "tcm/esi_bench/initial_state_capture/v1"}
            or split_entry is None
        ):
            raise ESIKnowledgeError("frozen-state capture result is invalid")
        packet = {
            "schema": "roboharn_evo/esi_bench/frozen_public_state/v1",
            "split_part": "development",
            "instance_ref": split_entry.instance_ref,
            "small_task": split_entry.small_task,
            "big_task": split_entry.big_task,
            "question_text": split_entry.question_text,
            "question_index": split_entry.question_index,
            "question_or_goal": result["question_or_goal"],
            "current_image_path": result["current_image_path"],
            "public_history": [],
            "evaluated_model_calls": 0,
            "retrieval_model_calls": 0,
            "executed_actions": 0,
            "answer_read": False,
            "score_read": False,
            "hidden_state_exported": False,
        }
        (audit_root / "frozen_state.json").write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (audit_root / "question_public.json").write_text(
            json.dumps(hook.first_public_question or {}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        (audit_root / "frozen_store_before_after_check.json").write_text(
            json.dumps(
                {
                    "checked": False,
                    "unchanged": True,
                    "heldout_update_performed": False,
                    "files": [],
                    "shuffle_manifest_checked": False,
                    "shuffle_manifest_unchanged": True,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        metrics = {
            "correct": None,
            "steps": 0,
            "evaluated_model_calls": 0,
            "retrieval_model_calls": 0,
            "input_tokens": None,
            "output_tokens": None,
            "simulator_steps": None,
            "api_cost_usd": None,
            "knowledge_adoption_count": 0,
            "wall_time_sec": elapsed,
            "heldout_store_update_performed": False,
            "capture_only": True,
        }
        (audit_root / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        if pending_shutdown is not None:
            shutdown, shutdown_args, shutdown_kwargs = pending_shutdown
            shutdown(*shutdown_args, **shutdown_kwargs)
        return 0
    if (
        isinstance(result, Mapping)
        and not result.get("skipped", False)
        and not hook.audits
    ):
        raise ESIKnowledgeError(
            "upstream returned a cached result; a fresh prompt-audited run is required"
        )

    if args.hpk_mode != "off":
        require_store_unchanged(args.hpk_store_root, store_before)
    shuffle_unchanged = (
        shuffle_before is None
        or args.hpk_shuffle_manifest.expanduser().resolve().read_bytes()
        == shuffle_before
    )
    if not shuffle_unchanged:
        raise ESIKnowledgeError("held-out evaluation modified the shuffle manifest")
    frozen_check = {
        "checked": args.hpk_mode != "off",
        "unchanged": True,
        "heldout_update_performed": False,
        "files": sorted(store_before),
        "shuffle_manifest_checked": shuffle_before is not None,
        "shuffle_manifest_unchanged": shuffle_unchanged,
    }
    (audit_root / "frozen_store_before_after_check.json").write_text(
        json.dumps(frozen_check, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (audit_root / "question_public.json").write_text(
        json.dumps(hook.first_public_question or {}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (audit_root / "upstream_answer.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    history = result.get("history", []) if isinstance(result, Mapping) else []
    for item in history if isinstance(history, list) else []:
        _append_jsonl(
            audit_root / "hpk_action_trace.jsonl",
            {
                "step": item.get("step"),
                "official_action": item.get("action"),
                "official_answer": item.get("answer"),
                "official_action_result": item.get("action_result", {}),
                "knowledge_adopted": next(
                    (
                        audit["knowledge_adopted"]
                        for audit in hook.audits
                        if audit["step"] == item.get("step")
                    ),
                    False,
                ),
            },
        )
    summary = {
        "correct": result.get("correct") if isinstance(result, Mapping) else None,
        "steps": len(history) if isinstance(history, list) else 0,
        "evaluated_model_calls": (
            len(history)
            + sum(
                isinstance(item, Mapping) and "raw_output_post_action" in item
                for item in history
            )
            if isinstance(history, list)
            else 0
        ),
        "retrieval_model_calls": sum(
            int(item["retrieval_calls"]) for item in hook.audits
        ),
        "input_tokens": None,
        "output_tokens": None,
        "simulator_steps": None,
        "api_cost_usd": None,
        "knowledge_adoption_count": sum(
            bool(item["knowledge_adopted"]) for item in hook.audits
        ),
        "wall_time_sec": elapsed,
        "heldout_store_update_performed": False,
    }
    (audit_root / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if pending_shutdown is not None:
        shutdown, shutdown_args, shutdown_kwargs = pending_shutdown
        shutdown(*shutdown_args, **shutdown_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
