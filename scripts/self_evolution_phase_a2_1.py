#!/usr/bin/env python3
"""Offline Phase A2.1 hierarchical trajectory reflection runner.

Image egress is denied by default.  The runner never starts a simulator,
rollout, SAM3, promotion, or runtime retrieval.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import self_evolution_phase_a2 as phase_a2  # noqa: E402
from roboharn_evo.agent.reflector.hierarchical import (  # noqa: E402
    HierarchicalTrajectoryReflectionResult,
    HierarchicalTrajectoryReflector,
)
from roboharn_evo.agent.reflector.hierarchical_inputs import (  # noqa: E402
    HierarchicalPreparedInputs,
    build_hierarchical_inputs,
)
from roboharn_evo.agent.reflector.hierarchical_review import (  # noqa: E402
    render_hierarchical_review_report,
)
from roboharn_evo.agent.reflector.hierarchical_transport import (  # noqa: E402
    HierarchicalTransportLimits,
    OpenAICompatibleHierarchicalBackend,
)
from roboharn_evo.agent.reflector.multimodal_transport import (  # noqa: E402
    CapabilityPreflightError,
    ImageEgressAuthorization,
    MultimodalTransportError,
    OpenAICompatibleMultimodalTransport,
    contains_secret_or_image_payload,
)
from roboharn_evo.agent.reflector.visual_presentations import (  # noqa: E402
    VisualPresentationError,
    VisualPresentationLimits,
    VisualPresentationResult,
    build_visual_presentation_plan,
)
from roboharn_evo.agent.reflector.hierarchical_transport import (  # noqa: E402
    VisualPresentationImagePlanProvider,
)
from roboharn_evo.benchmark_adapters.rmbench.action_chunk_evidence import (  # noqa: E402
    ActionChunkDetectionConfig,
    ActionChunkEvidenceError,
    RMBenchActionChunkEvidenceEntry,
    extract_action_chunk_evidence_manifest,
)
from roboharn_evo.benchmark_adapters.rmbench.trajectory_evidence import (  # noqa: E402
    RMBenchTrajectoryEvidenceEntry,
    TrajectoryEvidenceAdapterConfig,
    TrajectoryEvidenceError,
    extract_rmbench_visual_evidence_manifest,
    rmbench_dual_arm_trajectory_evidence_config,
)


_PHASE_ROOT = _REPO_ROOT / "eval_result" / "self_evolution" / "phase_a2_1"
_SAFE_ENV_NAME = phase_a2._SAFE_ENV_NAME_RE
_RUNTIME_SOURCE_PATHS = (
    "scripts/self_evolution_phase_a2_1.py",
    "roboharn_evo/agent/reflector/hierarchical.py",
    "roboharn_evo/agent/reflector/hierarchical_inputs.py",
    "roboharn_evo/agent/reflector/hierarchical_review.py",
    "roboharn_evo/agent/reflector/hierarchical_transport.py",
    "roboharn_evo/agent/reflector/multimodal_transport.py",
    "roboharn_evo/agent/reflector/visual_presentations.py",
    "roboharn_evo/benchmark_adapters/rmbench/action_chunk_evidence.py",
    "roboharn_evo/benchmark_adapters/rmbench/trajectory_evidence.py",
    "roboharn_evo/services/agent_api/openai_planner.py",
    "roboharn_evo/services/agent_api/openai_responses_compat.py",
)
_JSONL_SCHEMAS = {
    "action_chunks.jsonl": "roboharn_evo/action_chunk/v1",
    "visual_evidence.jsonl": "roboharn_evo/visual_evidence_bundle/v1",
    "derived_visual_presentations.jsonl": "roboharn_evo/derived_visual_presentation/v1",
    "presentation_image_index.jsonl": "roboharn_evo/presentation_image_index/v1",
    "presentation_request_plans.jsonl": "roboharn_evo/visual_request_plan/v1",
    "subtask_summaries.jsonl": "roboharn_evo/subtask_summary_record/v1",
    "episode_specific_facts.jsonl": "roboharn_evo/episode_specific_fact_record/v1",
    "candidate_experiences.jsonl": "roboharn_evo/experience/procedure/v1",
    "claim_attributions.jsonl": "roboharn_evo/reflection_attribution_record/v1",
    "abstentions.jsonl": "roboharn_evo/hierarchical_layer_abstention/v1",
    "source_abstentions.jsonl": "roboharn_evo/bootstrap_abstention/v1",
    "model_call_audit.jsonl": "roboharn_evo/phase_a2_model_call_audit/v1",
    "model_outputs.jsonl": "roboharn_evo/hierarchical_model_output/v1",
}


class PhaseA21RuntimeError(RuntimeError):
    """Fail-closed Phase A2.1 orchestration or publication error."""


@dataclass(frozen=True, slots=True)
class PhaseA21AuthorizationLimits:
    """Operator-authorized external-call envelope for one invocation."""

    max_total_presentation_payloads: int
    max_total_unique_source_timepoints: int
    max_total_unique_source_images: int
    max_total_expert_image_bytes: int
    max_total_external_model_calls: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name)) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class _PreparedTrajectory:
    action_entry: RMBenchActionChunkEvidenceEntry
    visual_entry: RMBenchTrajectoryEvidenceEntry
    presentation: VisualPresentationResult
    hierarchical_inputs: HierarchicalPreparedInputs | None


@dataclass(frozen=True, slots=True)
class _PlannedExternalFootprint:
    presentation_payloads: int
    image_bytes: int
    hierarchical_model_calls: int
    evidence_refs: frozenset[str]
    source_timepoints: frozenset[tuple[str, int]]
    source_images: frozenset[tuple[str, str, int, str]]


@dataclass(slots=True)
class _Artifacts:
    action_chunks: list[dict[str, Any]] = field(default_factory=list)
    visual_evidence: list[dict[str, Any]] = field(default_factory=list)
    derived_presentations: list[dict[str, Any]] = field(default_factory=list)
    presentation_request_plans: list[dict[str, Any]] = field(default_factory=list)
    subtask_summaries: list[dict[str, Any]] = field(default_factory=list)
    episode_facts: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    attributions: list[dict[str, Any]] = field(default_factory=list)
    abstentions: list[dict[str, Any]] = field(default_factory=list)
    source_abstentions: list[dict[str, Any]] = field(default_factory=list)
    model_call_audits: list[dict[str, Any]] = field(default_factory=list)
    model_outputs: list[dict[str, Any]] = field(default_factory=list)
    original_review_images: list[dict[str, Any]] = field(default_factory=list)
    presentation_images: list[dict[str, Any]] = field(default_factory=list)
    coverage_audits: list[dict[str, Any]] = field(default_factory=list)
    leakage_audit: dict[str, Any] = field(
        default_factory=lambda: {
            "schema": "roboharn_evo/semantic_leakage_audit/v1",
            "schema_version": 1,
            "verdict": "not_run_no_candidate",
        }
    )


@dataclass(frozen=True, slots=True)
class PhaseA21RunResult:
    output_root: Path
    candidate_count: int
    abstention_count: int
    action_chunk_count: int
    subtask_summary_count: int
    model_call_count: int
    presentation_payloads_sent: int
    presentation_bytes_sent: int
    file_sha256: Mapping[str, str]

    @property
    def expert_images_sent(self) -> int:
        """Backward-compatible alias; the count is outbound presentations."""

        return self.presentation_payloads_sent

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "candidate_count": self.candidate_count,
            "abstention_count": self.abstention_count,
            "action_chunk_count": self.action_chunk_count,
            "subtask_summary_count": self.subtask_summary_count,
            "model_call_count": self.model_call_count,
            "presentation_payloads_sent": self.presentation_payloads_sent,
            "presentation_bytes_sent": self.presentation_bytes_sent,
            "integrity_file_count": len(self.file_sha256),
            "run_manifest_sha256": self.file_sha256.get("run_manifest.json"),
            "sha256sums_sha256": self.file_sha256.get("sha256sums.json"),
        }


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_source_identity() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative_path in _RUNTIME_SOURCE_PATHS:
        path = _REPO_ROOT / relative_path
        if path.is_symlink() or not path.is_file():
            raise PhaseA21RuntimeError(
                f"runtime source is missing or not a regular file: {relative_path}"
            )
        files[relative_path] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    identity = {
        "schema": "roboharn_evo/runtime_source_identity/v1",
        "schema_version": 1,
        "algorithm": "sha256",
        "files": files,
    }
    return {**identity, "identity_sha256": _json_sha256(identity)}


def _resolve_output_root(value: str | os.PathLike[str]) -> Path:
    raw = Path(value)
    if not raw.is_absolute() or ".." in raw.parts:
        raise PhaseA21RuntimeError(
            "--output-root must be an absolute path without '..'"
        )
    if raw.exists() or raw.is_symlink():
        raise PhaseA21RuntimeError(f"output root already exists: {raw}")
    phase_a2._assert_no_symlink_components(raw)
    resolved = raw.resolve(strict=False)
    phase_root = _PHASE_ROOT.resolve(strict=False)
    tmp_root = Path("/tmp").resolve(strict=True)
    under_phase = resolved != phase_root and resolved.is_relative_to(phase_root)
    under_tmp = resolved != tmp_root and resolved.is_relative_to(tmp_root)
    if not (under_phase or under_tmp):
        raise PhaseA21RuntimeError(
            f"output root must be a fresh descendant of /tmp or {phase_root}"
        )
    if resolved.exists() or resolved.is_symlink():
        raise PhaseA21RuntimeError(f"output root already exists: {resolved}")
    if under_phase:
        _PHASE_ROOT.mkdir(parents=True, exist_ok=True)
        phase_a2._assert_no_symlink_components(_PHASE_ROOT)
    if not resolved.parent.is_dir():
        raise PhaseA21RuntimeError("output parent does not exist")
    return resolved


def _presentation_limits_dict(limits: VisualPresentationLimits) -> dict[str, Any]:
    return {name: getattr(limits, name) for name in limits.__dataclass_fields__}


def _prepare(
    *,
    manifest_path: str | os.PathLike[str],
    visual_adapter: TrajectoryEvidenceAdapterConfig,
    detection_config: ActionChunkDetectionConfig,
    presentation_limits: VisualPresentationLimits,
    context_radius: int,
) -> tuple[list[_PreparedTrajectory], dict[str, Any], list[dict[str, Any]]]:
    manifest = phase_a2._resolve_phase_a_manifest_link(manifest_path)
    action_batch = extract_action_chunk_evidence_manifest(
        manifest,
        detection_config=detection_config,
    )
    bundle_by_trajectory = {
        entry.trajectory_id: entry.action_chunks.to_dict()
        for entry in action_batch.entries
    }
    visual_batch = extract_rmbench_visual_evidence_manifest(
        manifest,
        adapter_config=visual_adapter,
        context_radius=context_radius,
        action_chunk_bundles_by_trajectory=bundle_by_trajectory,
    )
    action_by_id = {entry.trajectory_id: entry for entry in action_batch.entries}
    visual_by_id = {entry.trajectory_id: entry for entry in visual_batch.entries}
    if set(action_by_id) != set(visual_by_id):
        raise PhaseA21RuntimeError(
            "action and visual extraction produced different trajectory scopes"
        )
    prepared: list[_PreparedTrajectory] = []
    for trajectory_id in sorted(action_by_id):
        action_entry = action_by_id[trajectory_id]
        visual_entry = visual_by_id[trajectory_id]
        presentation = build_visual_presentation_plan(
            action_chunk_bundle=action_entry.action_chunks,
            visual_evidence_bundle=visual_entry.extracted.bundle,
            payload_resolver=lambda image_ref, entry=visual_entry: (
                entry.extracted.image_payload(image_ref).data
            ),
            limits=presentation_limits,
        )
        hierarchical_inputs = (
            build_hierarchical_inputs(
                whole_request=phase_a2._whole_request(visual_entry),
                action_chunk_bundle=action_entry.action_chunks,
                visual_evidence_bundle=visual_entry.extracted.bundle,
                presentation_result=presentation,
            )
            if presentation.admissible
            else None
        )
        prepared.append(
            _PreparedTrajectory(
                action_entry=action_entry,
                visual_entry=visual_entry,
                presentation=presentation,
                hierarchical_inputs=hierarchical_inputs,
            )
        )
    source_abstentions = [
        *[deepcopy_dict(value) for value in action_batch.abstentions],
        *[deepcopy_dict(value) for value in visual_batch.abstentions],
    ]
    return prepared, visual_batch.input_manifest, source_abstentions


def deepcopy_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _original_review_images(
    entry: RMBenchTrajectoryEvidenceEntry,
) -> list[dict[str, Any]]:
    evidence_by_image: dict[str, list[str]] = {}
    for item in entry.extracted.bundle.items:
        evidence_by_image.setdefault(item.image_ref_id, []).append(item.evidence_id)
    result: list[dict[str, Any]] = []
    for payload in entry.extracted.image_payloads:
        extension = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
        }[payload.media_type]
        result.append(
            {
                "schema": "roboharn_evo/review_image_index/v1",
                "schema_version": 1,
                "trajectory_id": entry.trajectory_id,
                "image_ref_id": payload.image_ref_id,
                "evidence_ids": evidence_by_image[payload.image_ref_id],
                "relative_path": (
                    f"review_keyframes/{entry.trajectory_id}/"
                    f"{payload.image_ref_id}.{extension}"
                ),
                "sha256": payload.sha256,
                "byte_length": len(payload.data),
                "media_type": payload.media_type,
                "_payload": payload.data,
            }
        )
    return result


def _presentation_images(
    presentation: VisualPresentationResult,
) -> list[dict[str, Any]]:
    records = {
        str(value["presentation_id"]): value
        for value in presentation.presentation_records
    }
    result: list[dict[str, Any]] = []
    for payload in presentation.outbound_payloads:
        record = records[payload.presentation_id]
        extension = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
        }[payload.media_type]
        result.append(
            {
                "schema": "roboharn_evo/presentation_image_index/v1",
                "schema_version": 1,
                "presentation_id": payload.presentation_id,
                "underlying_evidence_refs": record[
                    "underlying_evidence_refs"
                ],
                "relative_path": (
                    f"review_presentations/{payload.presentation_id}.{extension}"
                ),
                "sha256": payload.sha256,
                "byte_length": len(payload.data),
                "media_type": payload.media_type,
                "_payload": payload.data,
            }
        )
    return result


def _base_artifacts(
    prepared: Sequence[_PreparedTrajectory],
    source_abstentions: Sequence[Mapping[str, Any]],
) -> _Artifacts:
    artifacts = _Artifacts(
        source_abstentions=[deepcopy_dict(value) for value in source_abstentions]
    )
    for value in prepared:
        trajectory_id = value.action_entry.trajectory_id
        artifacts.action_chunks.extend(
            value.action_entry.action_chunks.action_chunk_records()
        )
        artifacts.visual_evidence.append(
            value.visual_entry.extracted.bundle.to_dict()
        )
        artifacts.derived_presentations.extend(
            deepcopy_dict(record)
            for record in value.presentation.derived_visual_presentations
        )
        artifacts.original_review_images.extend(
            _original_review_images(value.visual_entry)
        )
        artifacts.presentation_images.extend(
            _presentation_images(value.presentation)
        )
        artifacts.coverage_audits.append(
            {
                "trajectory_id": trajectory_id,
                **deepcopy_dict(value.presentation.coverage_audit),
            }
        )
        artifacts.presentation_request_plans.extend(
            {
                "schema": "roboharn_evo/visual_request_plan/v1",
                "schema_version": 1,
                "trajectory_id": trajectory_id,
                "layer": "action_chunk_batch",
                **deepcopy_dict(plan),
            }
            for plan in value.presentation.layer1_request_plans
        )
        artifacts.presentation_request_plans.extend(
            {
                "schema": "roboharn_evo/visual_request_plan/v1",
                "schema_version": 1,
                "trajectory_id": trajectory_id,
                "layer": "procedure_draft",
                **deepcopy_dict(plan),
            }
            for plan in value.presentation.cross_segment_feedback_plan.get(
                "requests", []
            )
        )
    return artifacts


def _record_no_egress_abstentions(
    artifacts: _Artifacts,
    prepared: Sequence[_PreparedTrajectory],
    *,
    created_at: str,
) -> None:
    for value in prepared:
        artifacts.abstentions.append(
            {
                "schema": "roboharn_evo/hierarchical_layer_abstention/v1",
                "schema_version": 1,
                "layer": "external_model",
                "scope_id": value.action_entry.trajectory_id,
                "status": "abstained",
                "reason": "image_egress_not_authorized",
                "issues": [
                    {
                        "code": "image_egress_not_authorized",
                        "offending_field": "permissions.image_egress_authorized",
                        "detail": (
                            "operator did not authorize the hierarchical expert-image "
                            "scope; no capability probe or model request was sent"
                        ),
                        "evidence_refs": [],
                    }
                ],
                "backend": "not_invoked",
                "prompt_template_hash": "",
                "output_sha256": "",
                "created_at": created_at,
            }
        )


def _record_visual_coverage_abstentions(
    artifacts: _Artifacts,
    prepared: Sequence[_PreparedTrajectory],
    *,
    created_at: str,
) -> None:
    for value in prepared:
        if value.hierarchical_inputs is not None:
            continue
        gaps = value.presentation.coverage_audit.get("coverage_gaps", [])
        blocking = [
            deepcopy_dict(gap)
            for gap in gaps
            if isinstance(gap, Mapping) and gap.get("blocking") is True
        ]
        artifacts.abstentions.append(
            {
                "schema": "roboharn_evo/hierarchical_layer_abstention/v1",
                "schema_version": 1,
                "layer": "visual_presentation",
                "scope_id": value.action_entry.trajectory_id,
                "status": "abstained",
                "reason": "visual_presentation_coverage_incomplete",
                "issues": [
                    {
                        "code": str(
                            gap.get("code", "visual_presentation_coverage_gap")
                        ),
                        "offending_field": "visual_coverage_audit.coverage_gaps",
                        "detail": json.dumps(
                            gap,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "evidence_refs": [],
                    }
                    for gap in blocking
                ],
                "backend": "not_invoked",
                "prompt_template_hash": "",
                "output_sha256": "",
                "created_at": created_at,
            }
        )


def _fact_record(
    *,
    trajectory_id: str,
    layer: str,
    scope_id: str,
    fact: Mapping[str, Any],
    admitted: bool,
    issues: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "schema": "roboharn_evo/episode_specific_fact_record/v1",
        "schema_version": 1,
        "trajectory_id": trajectory_id,
        "layer": layer,
        "scope_id": scope_id,
        "admission_status": (
            "admitted_episode_fact"
            if admitted
            else "rejected_with_local_abstention"
        ),
        "fact": deepcopy_dict(fact),
        "issues": [deepcopy_dict(value) for value in issues],
        "planner_visible": False,
    }


def _apply_hierarchical_result(
    artifacts: _Artifacts,
    *,
    trajectory_id: str,
    result: HierarchicalTrajectoryReflectionResult,
) -> None:
    for value in result.action_chunk_results:
        artifacts.episode_facts.extend(
            _fact_record(
                trajectory_id=trajectory_id,
                layer="action_chunk",
                scope_id=value.action_chunk_id,
                fact=fact,
                admitted=True,
            )
            for fact in value.admitted_facts
        )
        for rejected in value.rejected_facts:
            fact = rejected.get("fact") if isinstance(rejected, Mapping) else None
            issues = (
                rejected.get("issues", [])
                if isinstance(rejected, Mapping)
                else []
            )
            if isinstance(fact, Mapping):
                artifacts.episode_facts.append(
                    _fact_record(
                        trajectory_id=trajectory_id,
                        layer="action_chunk",
                        scope_id=value.action_chunk_id,
                        fact=fact,
                        admitted=False,
                        issues=(
                            issues
                            if isinstance(issues, Sequence)
                            and not isinstance(issues, (str, bytes))
                            else ()
                        ),
                    )
                )

    for value in result.subtask_summary_results:
        artifacts.subtask_summaries.append(
            {
                "schema": "roboharn_evo/subtask_summary_record/v1",
                "schema_version": 1,
                "trajectory_id": trajectory_id,
                "segment_id": value.segment_id,
                "admission_status": (
                    "admitted" if value.summary is not None else "abstained"
                ),
                "summary": (
                    None
                    if value.summary is None
                    else deepcopy_dict(value.summary)
                ),
                "abstention": (
                    None
                    if value.abstention is None
                    else deepcopy_dict(value.abstention)
                ),
                "output_sha256": value.output_sha256,
            }
        )
        artifacts.episode_facts.extend(
            _fact_record(
                trajectory_id=trajectory_id,
                layer="subtask_summary",
                scope_id=value.segment_id,
                fact=fact,
                admitted=True,
            )
            for fact in value.admitted_episode_facts
        )
        for rejected in value.rejected_facts:
            fact = rejected.get("fact") if isinstance(rejected, Mapping) else None
            issues = (
                rejected.get("issues", [])
                if isinstance(rejected, Mapping)
                else []
            )
            if isinstance(fact, Mapping):
                artifacts.episode_facts.append(
                    _fact_record(
                        trajectory_id=trajectory_id,
                        layer="subtask_summary",
                        scope_id=value.segment_id,
                        fact=fact,
                        admitted=False,
                        issues=(
                            issues
                            if isinstance(issues, Sequence)
                            and not isinstance(issues, (str, bytes))
                            else ()
                        ),
                    )
                )

    draft = result.procedure_draft_result
    if draft is not None:
        artifacts.episode_facts.extend(
            _fact_record(
                trajectory_id=trajectory_id,
                layer="procedure_draft",
                scope_id=trajectory_id,
                fact=fact,
                admitted=True,
            )
            for fact in draft.admitted_episode_facts
        )
    if result.attribution_result is not None:
        artifacts.leakage_audit = deepcopy_dict(
            result.attribution_result.leakage_audit
        )
        artifacts.attributions.extend(
            {
                "schema": "roboharn_evo/reflection_attribution_record/v1",
                "schema_version": 1,
                "trajectory_id": trajectory_id,
                "admission_status": (
                    "admitted_candidate" if result.accepted else "private_draft"
                ),
                "attribution": deepcopy_dict(value),
            }
            for value in result.attribution_result.attributions
        )
    if result.candidate_experience is not None:
        artifacts.candidates.append(deepcopy_dict(result.candidate_experience))
    artifacts.abstentions.extend(
        deepcopy_dict(value) for value in result.abstentions
    )
    artifacts.model_outputs.extend(
        deepcopy_dict(value) for value in result.model_outputs
    )


def _render_report(
    *,
    output_root: Path,
    prepared: Sequence[_PreparedTrajectory],
    artifacts: _Artifacts,
    permissions: Mapping[str, Any],
) -> str:
    segments = [
        deepcopy_dict(segment)
        for value in prepared
        for segment in value.action_entry.segments
    ]
    review_images = [
        *artifacts.original_review_images,
        *artifacts.presentation_images,
    ]
    return render_hierarchical_review_report(
        run_id=output_root.name,
        segments=segments,
        action_chunks=artifacts.action_chunks,
        visual_evidence=artifacts.visual_evidence,
        derived_presentations=artifacts.derived_presentations,
        review_images=review_images,
        subtask_summaries=artifacts.subtask_summaries,
        episode_facts=artifacts.episode_facts,
        candidates=artifacts.candidates,
        attributions=artifacts.attributions,
        abstentions=artifacts.abstentions,
        model_call_audits=artifacts.model_call_audits,
        model_outputs=artifacts.model_outputs,
        permissions=permissions,
        leakage_audit=artifacts.leakage_audit,
    )


def _write_payload_records(
    staging: Path,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for value in records:
        record = {
            key: item for key, item in value.items() if key != "_payload"
        }
        payload = value.get("_payload")
        relative_path = value.get("relative_path")
        if not isinstance(payload, bytes) or not isinstance(relative_path, str):
            raise PhaseA21RuntimeError("review image record is incomplete")
        phase_a2._write_bytes(staging / relative_path, payload)
        serialized.append(deepcopy_dict(record))
    return serialized


def _publish(
    *,
    output_root: Path,
    input_manifest: Mapping[str, Any],
    prepared: Sequence[_PreparedTrajectory],
    artifacts: _Artifacts,
    visual_adapter: TrajectoryEvidenceAdapterConfig,
    detection_config: ActionChunkDetectionConfig,
    presentation_limits: VisualPresentationLimits,
    transport: OpenAICompatibleMultimodalTransport | None,
    runtime_source_identity: Mapping[str, Any],
    permissions: Mapping[str, Any],
    created_at: str,
    api_key: str | None,
) -> dict[str, str]:
    parent = output_root.parent
    staging_name = (
        f".{output_root.name}.phase-a2-1-object-{secrets.token_hex(12)}"
    )
    staging = parent / staging_name
    try:
        staging.mkdir(mode=0o700)
    except OSError as exc:
        raise PhaseA21RuntimeError("cannot create Phase A2.1 staging") from exc
    published = False
    try:
        original_index = _write_payload_records(
            staging, artifacts.original_review_images
        )
        presentation_index = _write_payload_records(
            staging, artifacts.presentation_images
        )
        jsonl = {
            "action_chunks.jsonl": artifacts.action_chunks,
            "visual_evidence.jsonl": artifacts.visual_evidence,
            "derived_visual_presentations.jsonl": (
                artifacts.derived_presentations
            ),
            "presentation_image_index.jsonl": presentation_index,
            "presentation_request_plans.jsonl": (
                artifacts.presentation_request_plans
            ),
            "subtask_summaries.jsonl": artifacts.subtask_summaries,
            "episode_specific_facts.jsonl": artifacts.episode_facts,
            "candidate_experiences.jsonl": artifacts.candidates,
            "claim_attributions.jsonl": artifacts.attributions,
            "abstentions.jsonl": artifacts.abstentions,
            "source_abstentions.jsonl": artifacts.source_abstentions,
            "model_call_audit.jsonl": artifacts.model_call_audits,
            "model_outputs.jsonl": artifacts.model_outputs,
            "review_image_index.jsonl": original_index,
        }
        for name, values in jsonl.items():
            phase_a2._write_jsonl(staging / name, values)
        phase_a2._write_json(staging / "input_manifest.json", input_manifest)
        phase_a2._write_json(
            staging / "visual_coverage_audit.json",
            {
                "schema": "roboharn_evo/visual_presentation_coverage_audit/v1",
                "schema_version": 1,
                "trajectories": artifacts.coverage_audits,
            },
        )
        leakage_audit = {
            "schema": "roboharn_evo/semantic_leakage_audit/v1",
            "schema_version": 1,
            **artifacts.leakage_audit,
        }
        phase_a2._write_json(
            staging / "fixed_answer_leakage_audit.json",
            leakage_audit,
        )
        phase_a2._write_json(
            staging / "snapshot_draft" / "metadata.json",
            phase_a2._snapshot_metadata(
                output_root=output_root,
                input_manifest=input_manifest,
                candidates=artifacts.candidates,
            ),
        )
        report = _render_report(
            output_root=output_root,
            prepared=prepared,
            artifacts=artifacts,
            permissions=permissions,
        )
        phase_a2._write_bytes(
            staging / "review_report.md", report.encode("utf-8")
        )

        serialized_artifacts = json.dumps(
            {
                "jsonl": jsonl,
                "coverage": artifacts.coverage_audits,
                "leakage": leakage_audit,
                "permissions": dict(permissions),
                "report": report,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if (
            "data:image/" in serialized_artifacts
            or ";base64," in serialized_artifacts
            or contains_secret_or_image_payload(serialized_artifacts)
            or (api_key is not None and api_key in serialized_artifacts)
        ):
            raise PhaseA21RuntimeError(
                "refusing to publish image data, a secret, or unsafe text"
            )

        pre_manifest_files = sorted(
            str(path.relative_to(staging))
            for path in staging.rglob("*")
            if path.is_file()
        )
        configuration = {
            "visual_adapter": visual_adapter.to_dict(),
            "visual_adapter_sha256": visual_adapter.configuration_sha256,
            "action_chunk_detection": detection_config.to_dict(),
            "action_chunk_detection_sha256": (
                detection_config.configuration_sha256
            ),
            "visual_presentation_limits": _presentation_limits_dict(
                presentation_limits
            ),
            "transport": (
                None if transport is None else transport.configuration_identity()
            ),
            "runtime_source_identity": deepcopy_dict(runtime_source_identity),
            "hierarchy": [
                "action_chunk_batch",
                "subtask_summary",
                "procedure_draft",
                "evidence_attribution_and_leakage_audit",
            ],
            "candidate_only": True,
            "automatic_promotion": False,
            "runtime_retrieval": False,
            "rollout_integration": False,
            "atomic_publication": True,
        }
        counts = {
            "input_entries": len(input_manifest.get("entries", [])),
            "subtasks": sum(
                len(value.action_entry.segments) for value in prepared
            ),
            "action_chunks": len(artifacts.action_chunks),
            "visual_evidence_bundles": len(artifacts.visual_evidence),
            "original_review_images": len(original_index),
            "derived_visual_presentations": len(
                artifacts.derived_presentations
            ),
            "subtask_summaries": len(artifacts.subtask_summaries),
            "episode_specific_facts": len(artifacts.episode_facts),
            "candidate_experiences": len(artifacts.candidates),
            "claim_attributions": len(artifacts.attributions),
            "abstentions": len(artifacts.abstentions),
            "source_abstentions": len(artifacts.source_abstentions),
            "model_calls": len(artifacts.model_call_audits),
            "model_outputs": len(artifacts.model_outputs),
        }
        manifest = {
            "schema": "roboharn_evo/self_evolution_phase_a2_1_run/v1",
            "schema_version": 1,
            "run_id": output_root.name,
            "created_at": created_at,
            "phase": "self_evolution_phase_a2_1",
            "mode": "offline_candidate_only",
            "producer": {
                "kind": "reflector",
                "name": "roboharn_evo_hierarchical_trajectory_reflection",
                "version": "1",
            },
            "input": {
                "manifest_sha256": _json_sha256(input_manifest),
                "manifest_schema": input_manifest.get("schema"),
                "entry_count": len(input_manifest.get("entries", [])),
            },
            "configuration": configuration,
            "configuration_sha256": _json_sha256(configuration),
            "permissions": dict(permissions),
            "counts": counts,
            "output_schemas": {
                **_JSONL_SCHEMAS,
                "review_image_index.jsonl": "roboharn_evo/review_image_index/v1",
                "visual_coverage_audit.json": (
                    "roboharn_evo/visual_presentation_coverage_audit/v1"
                ),
                "fixed_answer_leakage_audit.json": (
                    "roboharn_evo/semantic_leakage_audit/v1"
                ),
                "review_report.md": (
                    "roboharn_evo/self_evolution_review_report/phase_a2_1"
                ),
                "snapshot_draft/metadata.json": (
                    "roboharn_evo/experience_snapshot_draft/v1"
                ),
            },
            "file_integrity": {
                "algorithm": "sha256",
                "files": {
                    name: {
                        "sha256": _sha256_file(staging / name),
                        "size_bytes": (staging / name).stat().st_size,
                    }
                    for name in pre_manifest_files
                },
                "run_manifest_self_hash": "listed_in_sha256sums_json",
                "sha256sums_self_hash": "excluded_recursive_digest",
            },
            "retrieval_boundary": {
                "candidate_only": True,
                "development_only": True,
                "runtime_retrieval": False,
                "retrieval_eligible": False,
                "human_review_required": True,
            },
            "publication": {
                "strategy": "same_parent_relative_symlink_no_clobber"
            },
        }
        phase_a2._write_json(staging / "run_manifest.json", manifest)
        checksum_names = sorted(
            str(path.relative_to(staging))
            for path in staging.rglob("*")
            if path.is_file()
        )
        phase_a2._write_json(
            staging / "sha256sums.json",
            {
                "schema": "roboharn_evo/sha256_manifest/v1",
                "schema_version": 1,
                "algorithm": "sha256",
                "files": {
                    name: _sha256_file(staging / name)
                    for name in checksum_names
                },
                "self_hash": "excluded_recursive_digest",
            },
        )
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        try:
            os.symlink(staging_name, output_root)
        except FileExistsError as exc:
            raise PhaseA21RuntimeError(
                f"output root already exists: {output_root}"
            ) from exc
        except OSError as exc:
            raise PhaseA21RuntimeError("atomic publication failed") from exc
        published = True
        if output_root.resolve(strict=True) != staging.resolve(strict=True):
            raise PhaseA21RuntimeError("published output identity check failed")
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return {
            str(path.relative_to(staging)): _sha256_file(path)
            for path in staging.rglob("*")
            if path.is_file()
        }
    except BaseException:
        if published:
            try:
                if output_root.is_symlink() and os.readlink(output_root) == staging_name:
                    output_root.unlink()
                    published = False
            except OSError:
                pass
        if not published and staging.exists():
            shutil.rmtree(staging)
        raise


def _planned_external_footprint(
    value: _PreparedTrajectory,
) -> _PlannedExternalFootprint:
    if value.hierarchical_inputs is None:
        return _PlannedExternalFootprint(
            presentation_payloads=0,
            image_bytes=0,
            hierarchical_model_calls=0,
            evidence_refs=frozenset(),
            source_timepoints=frozenset(),
            source_images=frozenset(),
        )
    plans = [
        *value.presentation.layer1_request_plans,
        *value.presentation.cross_segment_feedback_plan.get("requests", []),
    ]
    presentation_payloads = sum(
        int(plan.get("image_count", 0)) for plan in plans
    )
    image_bytes = sum(int(plan.get("total_image_bytes", 0)) for plan in plans)
    hierarchical_model_calls = (
        value.hierarchical_inputs.planned_hierarchical_model_calls
    )
    presentation_ids = [
        str(presentation_id)
        for plan in plans
        for presentation_id in plan.get("presentation_ids", [])
    ]
    if len(presentation_ids) != presentation_payloads:
        raise PhaseA21RuntimeError(
            "presentation request counts do not match their payload IDs"
        )
    record_by_id = {
        str(record.get("presentation_id")): record
        for record in value.presentation.presentation_records
    }
    evidence_refs = {
        str(evidence_ref)
        for presentation_id in presentation_ids
        for evidence_ref in record_by_id.get(presentation_id, {}).get(
            "underlying_evidence_refs", []
        )
    }
    if any(presentation_id not in record_by_id for presentation_id in presentation_ids):
        raise PhaseA21RuntimeError(
            "presentation request references an unknown local payload"
        )
    bundle = value.visual_entry.extracted.bundle.to_dict()
    item_by_id = {
        str(item.get("evidence_id")): item
        for item in bundle.get("items", [])
        if isinstance(item, Mapping)
    }
    if not evidence_refs.issubset(item_by_id):
        raise PhaseA21RuntimeError(
            "presentation request references unknown source evidence"
        )
    timepoints: set[tuple[str, int]] = set()
    source_images: set[tuple[str, str, int, str]] = set()
    for evidence_ref in evidence_refs:
        item = item_by_id[evidence_ref]
        frame = item.get("frame")
        image = item.get("image")
        if not isinstance(frame, Mapping) or not isinstance(image, Mapping):
            raise PhaseA21RuntimeError(
                "source evidence lacks frame or image metadata"
            )
        frame_index = frame.get("index")
        camera = item.get("camera")
        content_sha256 = image.get("sha256")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not isinstance(camera, str)
            or not isinstance(content_sha256, str)
        ):
            raise PhaseA21RuntimeError(
                "source evidence has invalid frame identity"
            )
        timepoints.add((value.action_entry.trajectory_id, frame_index))
        source_images.add(
            (
                value.action_entry.trajectory_id,
                camera,
                frame_index,
                content_sha256,
            )
        )
    return _PlannedExternalFootprint(
        presentation_payloads=presentation_payloads,
        image_bytes=image_bytes,
        hierarchical_model_calls=hierarchical_model_calls,
        evidence_refs=frozenset(evidence_refs),
        source_timepoints=frozenset(timepoints),
        source_images=frozenset(source_images),
    )


def _validate_total_authorization(
    *,
    presentation_payloads: int,
    unique_source_timepoints: int,
    unique_source_images: int,
    image_bytes: int,
    external_model_calls: int,
    limits: PhaseA21AuthorizationLimits,
) -> None:
    """Enforce one invocation-wide operator envelope before preflight I/O."""

    if (
        presentation_payloads > limits.max_total_presentation_payloads
        or unique_source_timepoints
        > limits.max_total_unique_source_timepoints
        or unique_source_images > limits.max_total_unique_source_images
        or image_bytes > limits.max_total_expert_image_bytes
        or external_model_calls > limits.max_total_external_model_calls
    ):
        raise PhaseA21RuntimeError(
            "planned invocation exceeds the explicit total authorization"
        )


def _license_policy(
    manifest: Mapping[str, Any],
    *,
    acknowledged_unverified: bool,
) -> tuple[str, bool, str | None]:
    status = str(manifest.get("license_status", "unverified"))
    if status == "verified":
        return status, True, None
    if status == "unverified":
        return (
            status,
            acknowledged_unverified,
            None
            if acknowledged_unverified
            else "unverified_license_not_acknowledged",
        )
    if status == "restricted":
        return status, False, "restricted_license_forbids_egress"
    return status, False, "unsupported_license_status"


def run_phase_a2_1(
    *,
    manifest_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    visual_adapter: TrajectoryEvidenceAdapterConfig,
    detection_config: ActionChunkDetectionConfig | None = None,
    presentation_limits: VisualPresentationLimits | None = None,
    authorization: ImageEgressAuthorization | None = None,
    authorization_limits: PhaseA21AuthorizationLimits | None = None,
    acknowledge_unverified_license: bool = False,
    context_radius: int = 1,
    service_url: str = "http://127.0.0.1:9104",
    model: str = "gpt-5.5",
    reasoning_effort: str = "xhigh",
    timeout_sec: float = 600.0,
    api_key: str | None = None,
    created_at: str | None = None,
    transport: OpenAICompatibleMultimodalTransport | None = None,
) -> PhaseA21RunResult:
    """Extract, hierarchically reflect or abstain, then publish atomically."""

    output = _resolve_output_root(output_root)
    timestamp = created_at or _utc_now()
    runtime_source_identity = _runtime_source_identity()
    detection = detection_config or ActionChunkDetectionConfig()
    visual_limits = presentation_limits or VisualPresentationLimits()
    decision = authorization or ImageEgressAuthorization(
        scope="phase_a2_1_hierarchical_evidence"
    )
    prepared, manifest, source_abstentions = _prepare(
        manifest_path=manifest_path,
        visual_adapter=visual_adapter,
        detection_config=detection,
        presentation_limits=visual_limits,
        context_radius=context_radius,
    )
    if not prepared and not source_abstentions:
        raise PhaseA21RuntimeError("extraction returned no trajectory or abstention")
    artifacts = _base_artifacts(prepared, source_abstentions)
    reflectable = [
        value for value in prepared if value.hierarchical_inputs is not None
    ]
    _record_visual_coverage_abstentions(
        artifacts,
        prepared,
        created_at=timestamp,
    )
    license_status, license_allows, license_reason = _license_policy(
        manifest,
        acknowledged_unverified=acknowledge_unverified_license,
    )
    planned_footprints = [
        _planned_external_footprint(value) for value in prepared
    ]
    planned_presentation_payloads = sum(
        value.presentation_payloads for value in planned_footprints
    )
    planned_image_bytes = sum(value.image_bytes for value in planned_footprints)
    planned_hierarchical_model_calls = sum(
        value.hierarchical_model_calls for value in planned_footprints
    )
    planned_evidence_refs = frozenset(
        evidence_ref
        for value in planned_footprints
        for evidence_ref in value.evidence_refs
    )
    planned_source_timepoints = frozenset(
        timepoint
        for value in planned_footprints
        for timepoint in value.source_timepoints
    )
    planned_source_images = frozenset(
        source_image
        for value in planned_footprints
        for source_image in value.source_images
    )
    planned_external_model_calls = planned_hierarchical_model_calls + (
        1 if reflectable else 0
    )
    active_transport = transport
    preflight_passed = False

    if not decision.granted:
        _record_no_egress_abstentions(
            artifacts, reflectable, created_at=timestamp
        )
    elif not license_allows:
        for value in reflectable:
            artifacts.abstentions.append(
                {
                    "schema": "roboharn_evo/hierarchical_layer_abstention/v1",
                    "schema_version": 1,
                    "layer": "external_model",
                    "scope_id": value.action_entry.trajectory_id,
                    "status": "abstained",
                    "reason": license_reason,
                    "issues": [
                        {
                            "code": license_reason,
                            "offending_field": "permissions.license_status",
                            "detail": "declared license policy forbids this egress",
                            "evidence_refs": [],
                        }
                    ],
                    "backend": "not_invoked",
                    "prompt_template_hash": "",
                    "output_sha256": "",
                    "created_at": timestamp,
                }
            )
    elif reflectable:
        if authorization_limits is None:
            raise PhaseA21RuntimeError(
                "authorized hierarchical egress requires explicit image/byte/call limits"
            )
        _validate_total_authorization(
            presentation_payloads=planned_presentation_payloads,
            unique_source_timepoints=len(planned_source_timepoints),
            unique_source_images=len(planned_source_images),
            image_bytes=planned_image_bytes,
            external_model_calls=planned_external_model_calls,
            limits=authorization_limits,
        )
        for value in reflectable:
            model_calls = _planned_external_footprint(
                value
            ).hierarchical_model_calls
            if model_calls > visual_limits.max_model_calls_per_trajectory:
                raise PhaseA21RuntimeError(
                    "planned hierarchy exceeds the configured per-trajectory call budget"
                )
        active_transport = active_transport or OpenAICompatibleMultimodalTransport(
            service_url=service_url,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_sec=timeout_sec,
            api_key=api_key,
            max_images=visual_limits.max_images_per_request,
            max_total_image_bytes=(
                visual_limits.max_total_image_bytes_per_request
            ),
        )
        try:
            report = active_transport.capability_preflight(
                authorization=decision
            )
        except (MultimodalTransportError, ValueError) as exc:
            failed = active_transport.last_preflight_audit
            artifacts.model_call_audits.append(
                {
                    "schema": "roboharn_evo/phase_a2_model_call_audit/v1",
                    "schema_version": 1,
                    "trajectory_id": None,
                    "call_kind": "synthetic_capability_preflight",
                    **({} if failed is None else dict(failed)),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            )
            for value in reflectable:
                artifacts.abstentions.append(
                    {
                        "schema": "roboharn_evo/hierarchical_layer_abstention/v1",
                        "schema_version": 1,
                        "layer": "capability_preflight",
                        "scope_id": value.action_entry.trajectory_id,
                        "status": "abstained",
                        "reason": "capability_preflight_failed",
                        "issues": [
                            {
                                "code": "capability_preflight_failed",
                                "offending_field": "backend.capabilities",
                                "detail": type(exc).__name__,
                                "evidence_refs": [],
                            }
                        ],
                        "backend": active_transport.backend_name,
                        "prompt_template_hash": "",
                        "output_sha256": "",
                        "created_at": timestamp,
                    }
                )
        else:
            preflight_passed = True
            preflight_audit = active_transport.last_preflight_audit
            artifacts.model_call_audits.append(
                {
                    "schema": "roboharn_evo/phase_a2_model_call_audit/v1",
                    "schema_version": 1,
                    "trajectory_id": None,
                    "call_kind": "synthetic_capability_preflight",
                    **report.to_dict(),
                    **(
                        {}
                        if preflight_audit is None
                        else dict(preflight_audit)
                    ),
                    "status": "passed",
                }
            )
            for value in reflectable:
                backend = OpenAICompatibleHierarchicalBackend(
                    active_transport,
                    authorization=decision,
                    image_plan_provider=VisualPresentationImagePlanProvider(
                        value.presentation
                    ),
                    limits=HierarchicalTransportLimits(
                        max_model_calls_per_trajectory=(
                            visual_limits.max_model_calls_per_trajectory
                        ),
                        max_images_per_request=(
                            visual_limits.max_images_per_request
                        ),
                        max_images_per_trajectory=(
                            authorization_limits.max_total_presentation_payloads
                        ),
                        max_image_bytes=visual_limits.max_total_image_bytes_per_request,
                        max_total_image_bytes_per_request=(
                            visual_limits.max_total_image_bytes_per_request
                        ),
                        max_total_image_bytes_per_trajectory=(
                            authorization_limits.max_total_expert_image_bytes
                        ),
                        layer2_verification_images=0,
                    ),
                )
                reflector = HierarchicalTrajectoryReflector(
                    backend,
                    max_model_calls=(
                        visual_limits.max_model_calls_per_trajectory
                    ),
                )
                result = reflector.reflect(
                    phase_a2._whole_request(value.visual_entry),
                    action_chunk_requests=(
                        value.hierarchical_inputs.action_chunk_requests
                    ),
                    segment_visual_evidence_by_segment=(
                        value.hierarchical_inputs.segment_visual_evidence_by_segment
                    ),
                    cross_segment_visual_evidence=(
                        value.hierarchical_inputs.cross_segment_visual_evidence
                    ),
                )
                artifacts.model_call_audits.extend(
                    {
                        "schema": "roboharn_evo/phase_a2_model_call_audit/v1",
                        "schema_version": 1,
                        "trajectory_id": value.action_entry.trajectory_id,
                        "call_kind": str(audit.get("layer", "hierarchical_call")),
                        **dict(audit),
                    }
                    for audit in backend.call_audits
                )
                _apply_hierarchical_result(
                    artifacts,
                    trajectory_id=value.action_entry.trajectory_id,
                    result=result,
                )

    presentation_payloads_sent = sum(
        int(value.get("images_sent", 0))
        for value in artifacts.model_call_audits
        if value.get("call_kind") != "synthetic_capability_preflight"
    )
    presentation_bytes_sent = sum(
        int(value.get("image_bytes_sent", 0))
        for value in artifacts.model_call_audits
        if value.get("call_kind") != "synthetic_capability_preflight"
    )
    external_model_calls_attempted = sum(
        1
        for value in artifacts.model_call_audits
        if value.get("delivery_status") in {"sent", "sent_or_attempted"}
    )
    sent_evidence_refs = {
        str(evidence_ref)
        for audit in artifacts.model_call_audits
        if audit.get("delivery_status") in {"sent", "sent_or_attempted"}
        for binding in audit.get("presentation_bindings", [])
        if isinstance(binding, Mapping)
        for evidence_ref in binding.get("underlying_evidence_refs", [])
    }
    source_identity_by_ref: dict[
        str, tuple[tuple[str, int], tuple[str, str, int, str]]
    ] = {}
    for value in prepared:
        for item in value.visual_entry.extracted.bundle.to_dict().get("items", []):
            if not isinstance(item, Mapping):
                continue
            evidence_ref = item.get("evidence_id")
            frame = item.get("frame")
            image = item.get("image")
            camera = item.get("camera")
            if (
                not isinstance(evidence_ref, str)
                or not isinstance(frame, Mapping)
                or not isinstance(image, Mapping)
                or not isinstance(camera, str)
            ):
                continue
            frame_index = frame.get("index")
            content_sha256 = image.get("sha256")
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or not isinstance(content_sha256, str)
            ):
                continue
            source_identity_by_ref[evidence_ref] = (
                (value.action_entry.trajectory_id, frame_index),
                (
                    value.action_entry.trajectory_id,
                    camera,
                    frame_index,
                    content_sha256,
                ),
            )
    if not sent_evidence_refs.issubset(source_identity_by_ref):
        raise PhaseA21RuntimeError(
            "sent-presentation audit references unknown source evidence"
        )
    sent_source_timepoints = {
        source_identity_by_ref[evidence_ref][0]
        for evidence_ref in sent_evidence_refs
    }
    sent_source_images = {
        source_identity_by_ref[evidence_ref][1]
        for evidence_ref in sent_evidence_refs
    }
    permissions = {
        "manifest_authorized_read_only_input": True,
        "license_status": license_status,
        "license_allows_image_egress": license_allows,
        "license_denial_reason": license_reason,
        "unverified_license_acknowledged": acknowledge_unverified_license,
        "image_egress_authorized": decision.granted,
        "authorization_scope": decision.scope,
        "authorization_sha256": decision.audit_sha256(),
        "authorization_limits": (
            None if authorization_limits is None else authorization_limits.to_dict()
        ),
        "planned_presentation_payloads": planned_presentation_payloads,
        "planned_unique_source_timepoints": len(planned_source_timepoints),
        "planned_unique_source_images": len(planned_source_images),
        "planned_logical_evidence_refs": len(planned_evidence_refs),
        "planned_expert_image_bytes": planned_image_bytes,
        "planned_hierarchical_model_calls": (
            planned_hierarchical_model_calls
        ),
        "planned_external_model_calls": planned_external_model_calls,
        "capability_preflight_passed": preflight_passed,
        "presentation_payloads_sent": presentation_payloads_sent,
        "presentation_bytes_sent": presentation_bytes_sent,
        "underlying_logical_evidence_refs_sent": len(sent_evidence_refs),
        "underlying_unique_source_timepoints_sent": len(sent_source_timepoints),
        "underlying_unique_source_images_sent": len(sent_source_images),
        "external_model_calls_attempted": external_model_calls_attempted,
        "external_model_invoked": external_model_calls_attempted > 0,
        "api_secret_logged": False,
        "image_base64_logged": False,
        "runtime_retrieval_activated": False,
        "rollout_started": False,
        "sam3_started": False,
        "automatic_promotion": False,
    }
    if _runtime_source_identity() != runtime_source_identity:
        raise PhaseA21RuntimeError("runtime source changed during the invocation")
    hashes = _publish(
        output_root=output,
        input_manifest=manifest,
        prepared=prepared,
        artifacts=artifacts,
        visual_adapter=visual_adapter,
        detection_config=detection,
        presentation_limits=visual_limits,
        transport=active_transport,
        runtime_source_identity=runtime_source_identity,
        permissions=permissions,
        created_at=timestamp,
        api_key=api_key,
    )
    return PhaseA21RunResult(
        output_root=output,
        candidate_count=len(artifacts.candidates),
        abstention_count=len(artifacts.abstentions),
        action_chunk_count=len(artifacts.action_chunks),
        subtask_summary_count=len(artifacts.subtask_summaries),
        model_call_count=len(artifacts.model_call_audits),
        presentation_payloads_sent=presentation_payloads_sent,
        presentation_bytes_sent=presentation_bytes_sent,
        file_sha256=hashes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline Subtask→Action Chunk→Visual Evidence hierarchical "
            "reflection. Image egress is denied by default."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--camera", action="append", dest="cameras")
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--service-url", default="http://127.0.0.1:9104")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument("--api-key-env")
    parser.add_argument("--max-images-per-request", type=int, default=16)
    parser.add_argument("--max-unique-source-frames", type=int, default=40)
    parser.add_argument(
        "--max-total-image-bytes-per-request",
        type=int,
        default=8 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-total-image-bytes-per-trajectory",
        type=int,
        default=8 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-total-pixels-per-trajectory", type=int, default=32_000_000
    )
    parser.add_argument(
        "--max-derived-presentations-per-source-frame", type=int, default=4
    )
    parser.add_argument("--max-model-calls-per-trajectory", type=int, default=8)
    parser.add_argument("--authorize-image-egress", action="store_true")
    parser.add_argument("--authorization-assertion")
    parser.add_argument("--acknowledge-unverified-license", action="store_true")
    parser.add_argument(
        "--authorization-max-presentation-payloads",
        "--authorization-max-total-images",
        dest="authorization_max_presentation_payloads",
        type=int,
        help="total outbound image payload occurrences across the invocation",
    )
    parser.add_argument(
        "--authorization-max-unique-source-timepoints",
        type=int,
        help="unique original trajectory timepoints represented in payloads",
    )
    parser.add_argument(
        "--authorization-max-unique-source-images",
        type=int,
        help="unique original camera/frame images represented in payloads",
    )
    parser.add_argument("--authorization-max-total-image-bytes", type=int)
    parser.add_argument(
        "--authorization-max-total-external-calls",
        "--authorization-max-model-calls",
        dest="authorization_max_total_external_calls",
        type=int,
        help="all external calls, including the synthetic capability preflight",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        visual_adapter = (
            rmbench_dual_arm_trajectory_evidence_config(
                camera_ids=tuple(args.cameras)
            )
            if args.cameras
            else rmbench_dual_arm_trajectory_evidence_config()
        )
        api_key: str | None = None
        if args.api_key_env:
            if _SAFE_ENV_NAME.fullmatch(args.api_key_env) is None:
                raise PhaseA21RuntimeError("--api-key-env is unsafe")
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                raise PhaseA21RuntimeError("configured API key is empty")
        authorization_limits: PhaseA21AuthorizationLimits | None = None
        if args.authorize_image_egress:
            if not args.authorization_assertion:
                raise PhaseA21RuntimeError(
                    "--authorize-image-egress requires --authorization-assertion"
                )
            if not args.acknowledge_unverified_license:
                raise PhaseA21RuntimeError(
                    "authorized development egress requires the unverified-license acknowledgement"
                )
            if any(
                value is None
                for value in (
                    args.authorization_max_presentation_payloads,
                    args.authorization_max_unique_source_timepoints,
                    args.authorization_max_unique_source_images,
                    args.authorization_max_total_image_bytes,
                    args.authorization_max_total_external_calls,
                )
            ):
                raise PhaseA21RuntimeError(
                    "authorized egress requires explicit payload/source/byte/call limits"
                )
            decision = ImageEgressAuthorization.operator_granted(
                assertion=args.authorization_assertion,
                scope="phase_a2_1_hierarchical_evidence",
            )
            authorization_limits = PhaseA21AuthorizationLimits(
                max_total_presentation_payloads=(
                    args.authorization_max_presentation_payloads
                ),
                max_total_unique_source_timepoints=(
                    args.authorization_max_unique_source_timepoints
                ),
                max_total_unique_source_images=(
                    args.authorization_max_unique_source_images
                ),
                max_total_expert_image_bytes=(
                    args.authorization_max_total_image_bytes
                ),
                max_total_external_model_calls=(
                    args.authorization_max_total_external_calls
                ),
            )
        else:
            if args.authorization_assertion:
                raise PhaseA21RuntimeError(
                    "--authorization-assertion requires --authorize-image-egress"
                )
            decision = ImageEgressAuthorization(
                scope="phase_a2_1_hierarchical_evidence"
            )
        result = run_phase_a2_1(
            manifest_path=args.manifest,
            output_root=args.output_root,
            visual_adapter=visual_adapter,
            detection_config=ActionChunkDetectionConfig(),
            presentation_limits=VisualPresentationLimits(
                max_images_per_request=args.max_images_per_request,
                max_unique_source_frames_per_trajectory=(
                    args.max_unique_source_frames
                ),
                max_total_image_bytes_per_request=(
                    args.max_total_image_bytes_per_request
                ),
                max_total_image_bytes_per_trajectory=(
                    args.max_total_image_bytes_per_trajectory
                ),
                max_total_pixels_per_trajectory=(
                    args.max_total_pixels_per_trajectory
                ),
                max_derived_presentations_per_source_frame=(
                    args.max_derived_presentations_per_source_frame
                ),
                max_model_calls_per_trajectory=(
                    args.max_model_calls_per_trajectory
                ),
            ),
            authorization=decision,
            authorization_limits=authorization_limits,
            acknowledge_unverified_license=(
                args.acknowledge_unverified_license
            ),
            context_radius=args.context_radius,
            service_url=args.service_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout_sec=args.timeout_sec,
            api_key=api_key,
        )
    except (
        ActionChunkEvidenceError,
        CapabilityPreflightError,
        MultimodalTransportError,
        phase_a2.PhaseA2RuntimeError,
        PhaseA21RuntimeError,
        TrajectoryEvidenceError,
        VisualPresentationError,
        ValueError,
    ) as exc:
        print(f"Phase A2.1 failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
