#!/usr/bin/env python3
"""Run offline, candidate-only Self-Evolution Phase A2.

This command reads only an explicit content-addressed expert manifest, extracts
sparse visual evidence, optionally invokes an explicitly authorized multimodal
backend, and atomically publishes audit artifacts.  It never starts a rollout,
simulator, SAM service, or runtime retrieval; it never promotes a candidate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import sys
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from roboharn_evo.agent.reflector.multimodal_transport import (  # noqa: E402
    CapabilityPreflightError,
    ImageEgressAuthorization,
    MultimodalImage,
    MultimodalTransportError,
    OpenAICompatibleLeakageCritic,
    OpenAICompatibleMultimodalBackend,
    OpenAICompatibleMultimodalTransport,
    contains_secret_or_image_payload,
    generic_reflection_system_prompt,
)
from roboharn_evo.agent.reflector.quality import ReflectionQualityGate  # noqa: E402
from roboharn_evo.agent.reflector.whole_trajectory import (  # noqa: E402
    WholeTrajectoryReflectionRequest,
    WholeTrajectoryReflector,
)
from roboharn_evo.benchmark_adapters.rmbench.trajectory_evidence import (  # noqa: E402
    ActionChannelConfig,
    CameraChannelConfig,
    GripperChannelConfig,
    RMBenchTrajectoryEvidenceBatch,
    RMBenchTrajectoryEvidenceEntry,
    TrajectoryEvidenceAdapterConfig,
    TrajectoryEvidenceError,
    TrajectoryEvidenceLimits,
    extract_rmbench_visual_evidence_manifest,
    rmbench_dual_arm_trajectory_evidence_config,
)


_REPO_ROOT = _PROJECT_ROOT
_PHASE_A_ROOT = _REPO_ROOT / "eval_result" / "self_evolution" / "phase_a"
_PHASE_A2_ROOT = _REPO_ROOT / "eval_result" / "self_evolution" / "phase_a2"
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_BASELINE_BYTES = 64 * 1024 * 1024
_DATA_IMAGE_MARKER = b"data:image/"
_BASE64_MARKER = b";base64,"
_SAFE_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")

_JSONL_SCHEMAS = {
    "visual_evidence.jsonl": "roboharn_evo/visual_evidence_bundle/v1",
    "review_image_index.jsonl": "roboharn_evo/review_image_index/v1",
    "episode_specific_facts.jsonl": "roboharn_evo/episode_specific_fact_record/v1",
    "claim_attributions.jsonl": "roboharn_evo/reflection_attribution_record/v1",
    "candidate_experiences.jsonl": "roboharn_evo/experience/procedure/v1",
    "abstentions.jsonl": "roboharn_evo/reflection_abstention/v1",
    "source_abstentions.jsonl": "roboharn_evo/bootstrap_abstention/v1",
    "model_call_audit.jsonl": "roboharn_evo/phase_a2_model_call_audit/v1",
}


class PhaseA2RuntimeError(RuntimeError):
    """Fail-closed Phase A2 orchestration or publication error."""


@dataclass(frozen=True, slots=True)
class PhaseA2RunResult:
    output_root: Path
    candidate_count: int
    abstention_count: int
    source_abstention_count: int
    image_egress_authorized: bool
    expert_images_sent: int
    file_sha256: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "candidate_count": self.candidate_count,
            "abstention_count": self.abstention_count,
            "source_abstention_count": self.source_abstention_count,
            "image_egress_authorized": self.image_egress_authorized,
            "expert_images_sent": self.expert_images_sent,
            "file_sha256": dict(self.file_sha256),
        }


@dataclass(slots=True)
class _RunArtifacts:
    visual_evidence: list[dict[str, Any]]
    review_images: list[dict[str, Any]]
    episode_facts: list[dict[str, Any]]
    attributions: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    abstentions: list[dict[str, Any]]
    source_abstentions: list[dict[str, Any]]
    model_call_audits: list[dict[str, Any]]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PhaseA2RuntimeError("artifact is not strict JSON") from exc
    return (text + "\n").encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value).rstrip(b"\n")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PhaseA2RuntimeError(f"cannot stat {label}") from exc
    if size <= 0 or size > max_bytes:
        raise PhaseA2RuntimeError(f"{label} exceeds its byte budget or is empty")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseA2RuntimeError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PhaseA2RuntimeError(f"{label} must contain one JSON object")
    return value


def _adapter_config_from_dict(payload: Mapping[str, Any]) -> TrajectoryEvidenceAdapterConfig:
    allowed = {
        "adapter_id",
        "adapter_version",
        "embodiment_id",
        "cameras",
        "action",
        "grippers",
    }
    if set(payload) != allowed:
        raise PhaseA2RuntimeError(
            "adapter config fields must be exactly: " + ", ".join(sorted(allowed))
        )
    cameras_raw = payload["cameras"]
    grippers_raw = payload["grippers"]
    action_raw = payload["action"]
    if (
        isinstance(cameras_raw, (str, bytes))
        or not isinstance(cameras_raw, Sequence)
        or isinstance(grippers_raw, (str, bytes))
        or not isinstance(grippers_raw, Sequence)
        or not isinstance(action_raw, Mapping)
    ):
        raise PhaseA2RuntimeError("adapter channels have invalid shapes")
    if any(not isinstance(value, Mapping) for value in cameras_raw) or any(
        not isinstance(value, Mapping) for value in grippers_raw
    ):
        raise PhaseA2RuntimeError("adapter camera/gripper entries must be objects")
    try:
        return TrajectoryEvidenceAdapterConfig(
            adapter_id=str(payload["adapter_id"]),
            adapter_version=str(payload["adapter_version"]),
            embodiment_id=str(payload["embodiment_id"]),
            cameras=tuple(
                CameraChannelConfig(**dict(value))
                for value in cameras_raw
            ),
            action=ActionChannelConfig(**dict(action_raw)),
            grippers=tuple(
                GripperChannelConfig(**dict(value))
                for value in grippers_raw
            ),
        )
    except (TypeError, ValueError) as exc:
        raise PhaseA2RuntimeError("adapter config failed validation") from exc


def _assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        except OSError as exc:
            raise PhaseA2RuntimeError("cannot inspect output path") from exc
        if stat.S_ISLNK(mode):
            raise PhaseA2RuntimeError(
                f"output path must not contain a symlink component: {current}"
            )
        if current != path and not stat.S_ISDIR(mode):
            raise PhaseA2RuntimeError(
                f"output parent component is not a directory: {current}"
            )


def _resolve_output_root(value: str | os.PathLike[str]) -> Path:
    raw = Path(value)
    if not raw.is_absolute() or ".." in raw.parts:
        raise PhaseA2RuntimeError("--output-root must be an absolute path without '..'")
    _assert_no_symlink_components(raw)
    resolved = raw.resolve(strict=False)
    phase_root = _PHASE_A2_ROOT.resolve(strict=False)
    tmp_root = Path("/tmp").resolve(strict=True)
    under_phase = resolved != phase_root and resolved.is_relative_to(phase_root)
    under_tmp = resolved != tmp_root and resolved.is_relative_to(tmp_root)
    if not (under_phase or under_tmp):
        raise PhaseA2RuntimeError(
            f"output root must be a fresh descendant of /tmp or {phase_root}"
        )
    if resolved.exists() or resolved.is_symlink():
        raise PhaseA2RuntimeError(f"output root already exists: {resolved}")
    if under_phase:
        _PHASE_A2_ROOT.mkdir(parents=True, exist_ok=True)
        _assert_no_symlink_components(_PHASE_A2_ROOT)
    if not resolved.parent.is_dir():
        raise PhaseA2RuntimeError("output parent does not exist")
    return resolved


def _resolve_phase_a_manifest_link(
    value: str | os.PathLike[str],
) -> Path:
    """Resolve only this repository's audited Phase A publication symlink.

    The secure importer correctly rejects arbitrary symlinks.  Phase A's own
    atomic publication format is a narrowly recognizable same-parent link to a
    hidden no-clobber backing directory, so the CLI resolves that one trusted
    wrapper before handing the regular manifest file to the importer.
    """

    path = Path(value)
    if not path.is_absolute():
        path = (Path.cwd() / path).absolute()
    phase_root = _PHASE_A_ROOT.resolve(strict=False)
    run_link = path.parent
    if path.name != "input_manifest.json" or run_link.parent != phase_root:
        return path
    try:
        mode = os.lstat(run_link).st_mode
    except FileNotFoundError:
        return path
    except OSError as exc:
        raise PhaseA2RuntimeError("cannot inspect Phase A run link") from exc
    if not stat.S_ISLNK(mode):
        return path
    try:
        target = os.readlink(run_link)
    except OSError as exc:
        raise PhaseA2RuntimeError("cannot read Phase A run link") from exc
    expected_prefix = f".{run_link.name}.phase-a-object-"
    if (
        not target.startswith(expected_prefix)
        or target in {".", ".."}
        or "/" in target
        or "\\" in target
    ):
        raise PhaseA2RuntimeError("Phase A run link has an unexpected target")
    backing = phase_root / target
    try:
        backing_mode = os.lstat(backing).st_mode
        manifest_mode = os.lstat(backing / "input_manifest.json").st_mode
    except OSError as exc:
        raise PhaseA2RuntimeError("Phase A backing manifest is unavailable") from exc
    if not stat.S_ISDIR(backing_mode) or not stat.S_ISREG(manifest_mode):
        raise PhaseA2RuntimeError("Phase A backing manifest has an unsafe type")
    return backing / "input_manifest.json"


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PhaseA2RuntimeError(f"failed to create artifact {path.name}") from exc


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, _canonical_json_bytes(value, pretty=True))


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            for value in values:
                handle.write(_canonical_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PhaseA2RuntimeError(f"failed to create artifact {path.name}") from exc


def _markdown_text(value: Any, *, limit: int = 1200) -> str:
    text = str(value).replace("\x00", "").replace("\r", " ").replace("\n", " ")
    text = text.replace("`", "\\`").replace("|", "\\|")
    return text[:limit] + ("…" if len(text) > limit else "")


def _record_abstention(
    *,
    trajectory_id: str,
    reason: str,
    offending_field: str,
    backend: str,
    created_at: str,
    detail: str,
    prompt_template_hash: str | None = None,
    output_sha256: str | None = None,
) -> dict[str, Any]:
    prompt_hash = prompt_template_hash or hashlib.sha256(b"not-invoked").hexdigest()
    output_hash = output_sha256 or hashlib.sha256(b"").hexdigest()
    issue = {
        "code": reason,
        "offending_field": offending_field,
        "detail": detail,
        "evidence_refs": [],
    }
    return {
        "schema": "roboharn_evo/reflection_abstention/v1",
        "schema_version": 1,
        "trajectory_id": trajectory_id,
        "status": "abstained",
        "reason": reason,
        "offending_field": offending_field,
        "evidence_refs": [],
        "issues": [issue],
        "backend": backend,
        "prompt_template_hash": prompt_hash,
        "output_sha256": output_hash,
        "created_at": created_at,
    }


def _uncertainties(bundle: Mapping[str, Any]) -> tuple[str, ...]:
    capabilities = bundle.get("data_capabilities", {})
    if not isinstance(capabilities, Mapping):
        return ("data capabilities are unavailable",)
    result: list[str] = []
    for name, value in sorted(capabilities.items()):
        if isinstance(value, Mapping) and value.get("available") is False:
            reason = value.get("reason", "not available")
            result.append(f"{name}: {reason}")
    return tuple(result)


def _whole_request(entry: RMBenchTrajectoryEvidenceEntry) -> WholeTrajectoryReflectionRequest:
    normalized = entry.normalized_trajectory
    trajectory = normalized.get("trajectory")
    provenance = normalized.get("provenance")
    if not isinstance(trajectory, Mapping) or not isinstance(provenance, Mapping):
        raise PhaseA2RuntimeError("normalized trajectory is missing trajectory/provenance")
    bundle = entry.extracted.bundle.to_dict()
    return WholeTrajectoryReflectionRequest(
        trajectory_id=entry.trajectory_id,
        instruction=str(trajectory.get("instruction", "")),
        trajectory_outcome=dict(trajectory.get("outcome", {})),
        segments=tuple(entry.segments),
        visual_evidence=bundle,
        provenance=dict(provenance),
        uncertainties=_uncertainties(bundle),
    )


def _image_resolver(entry: RMBenchTrajectoryEvidenceEntry) -> Any:
    def resolve(
        request: WholeTrajectoryReflectionRequest,
        evidence_ids: Sequence[str],
    ) -> tuple[MultimodalImage, ...]:
        if request.trajectory_id != entry.trajectory_id:
            raise PhaseA2RuntimeError("image resolver received a cross-trajectory request")
        images: list[MultimodalImage] = []
        for evidence_id in evidence_ids:
            item = entry.extracted.bundle.resolve(evidence_id)
            payload = entry.extracted.image_payload(item.image_ref_id)
            images.append(
                MultimodalImage(
                    evidence_id=evidence_id,
                    mime_type=payload.media_type,
                    content=payload.data,
                    detail="high",
                )
            )
        return tuple(images)

    return resolve


def _sensitive_payload(value: Any, *, api_key: str | None) -> bool:
    encoded = _canonical_json_bytes(value)
    text = encoded.decode("utf-8")
    if (
        _DATA_IMAGE_MARKER in encoded
        or _BASE64_MARKER in encoded
        or contains_secret_or_image_payload(text)
    ):
        return True
    return bool(api_key and api_key.encode("utf-8") in encoded)


def _review_image_records(
    batch: RMBenchTrajectoryEvidenceBatch,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in batch.entries:
        by_image: dict[str, list[str]] = {}
        for item in entry.extracted.bundle.items:
            by_image.setdefault(item.image_ref_id, []).append(item.evidence_id)
        for payload in entry.extracted.image_payloads:
            key = (entry.trajectory_id, payload.image_ref_id)
            if key in seen:
                continue
            seen.add(key)
            extension = {
                "image/jpeg": "jpg",
                "image/png": "png",
                "image/webp": "webp",
            }[payload.media_type]
            records.append(
                {
                    "schema": "roboharn_evo/review_image_index/v1",
                    "schema_version": 1,
                    "trajectory_id": entry.trajectory_id,
                    "image_ref_id": payload.image_ref_id,
                    "evidence_ids": by_image.get(payload.image_ref_id, []),
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
    return records


def _baseline_comparison(
    baseline_root: Path | None,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    old: list[dict[str, Any]] = []
    source: dict[str, Any] | None = None
    if baseline_root is not None:
        candidate_path = baseline_root / "candidate_experiences.jsonl"
        try:
            if candidate_path.stat().st_size > _MAX_BASELINE_BYTES:
                raise PhaseA2RuntimeError("baseline candidate file exceeds byte budget")
            for line in candidate_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        old.append(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PhaseA2RuntimeError("cannot read baseline Phase A candidates") from exc
        source = {
            "path": str(baseline_root),
            "candidate_file_sha256": _sha256_file(candidate_path),
        }

    def metrics(values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "candidate_count": len(values),
            "nonempty_predicted_effects": sum(
                bool(value.get("predicted_effects")) for value in values
            ),
            "feedback_policy_count": sum(
                isinstance(value.get("guidance"), Mapping)
                and isinstance(value["guidance"].get("feedback_policy"), Mapping)
                for value in values
            ),
            "multi_segment_support_count": sum(
                isinstance(value.get("evidence"), Mapping)
                and len(value["evidence"].get("supporting_segment_refs", [])) > 1
                for value in values
            ),
        }

    return {
        "schema": "roboharn_evo/phase_a2_old_new_comparison/v1",
        "schema_version": 1,
        "baseline": {"source": source, "metrics": metrics(old)},
        "phase_a2": {"metrics": metrics(candidates)},
        "comparison_is_structural_not_effectiveness_claim": True,
    }


def _fact_lines(
    records: Sequence[Mapping[str, Any]],
    source_type: str,
) -> list[str]:
    result: list[str] = []
    for record in records:
        if record.get("admission_status") != "admitted_candidate":
            continue
        fact = record.get("fact")
        if not isinstance(fact, Mapping) or fact.get("source_type") != source_type:
            continue
        refs = fact.get("supporting_evidence_refs", [])
        segments = fact.get("supporting_segment_refs", [])
        result.append(
            "- "
            + _markdown_text(fact.get("claim", ""))
            + f" — evidence={_markdown_text(refs)}, segments={_markdown_text(segments)}, "
            + f"confidence={_markdown_text(fact.get('confidence', ''))}, "
            + f"uncertainty={_markdown_text(fact.get('uncertainty', ''))}"
        )
    return result or ["- None recorded."]


def _transferable_attribution_lines(artifacts: _RunArtifacts) -> list[str]:
    visual_by_ref: dict[str, Mapping[str, Any]] = {}
    for bundle in artifacts.visual_evidence:
        items = bundle.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, Mapping) and isinstance(item.get("evidence_id"), str):
                visual_by_ref[item["evidence_id"]] = item
    review_path_by_ref: dict[str, str] = {}
    for record in artifacts.review_images:
        path = record.get("relative_path")
        refs = record.get("evidence_ids", [])
        if not isinstance(path, str) or not isinstance(refs, list):
            continue
        for ref in refs:
            if isinstance(ref, str):
                review_path_by_ref[ref] = path

    result: list[str] = []
    for record in artifacts.attributions:
        if record.get("admission_status") != "admitted_candidate":
            continue
        attribution = record.get("attribution")
        if not isinstance(attribution, Mapping):
            continue
        refs = attribution.get("supporting_evidence_refs", [])
        evidence_details: list[str] = []
        if isinstance(refs, list):
            for ref in refs:
                if not isinstance(ref, str):
                    continue
                item = visual_by_ref.get(ref)
                if item is None:
                    evidence_details.append(ref)
                    continue
                frame = item.get("frame", {})
                grippers = item.get("gripper_states", [])
                gripper_states = [
                    f"{value.get('channel_id')}={value.get('state')}"
                    for value in grippers
                    if isinstance(value, Mapping)
                ] if isinstance(grippers, list) else []
                evidence_details.append(
                    f"{ref}(frame={frame.get('index') if isinstance(frame, Mapping) else None},"
                    f"camera={item.get('camera')},role={item.get('temporal_role')},"
                    f"grippers={gripper_states},review={review_path_by_ref.get(ref)})"
                )
        result.append(
            "- target="
            + _markdown_text(attribution.get("target_path", ""))
            + "; claim="
            + _markdown_text(attribution.get("claim", ""))
            + "; source_type="
            + _markdown_text(attribution.get("source_type", ""))
            + "; evidence="
            + _markdown_text(evidence_details)
            + "; segments="
            + _markdown_text(attribution.get("supporting_segment_refs", []))
            + "; confidence="
            + _markdown_text(attribution.get("confidence", ""))
            + "; uncertainty="
            + _markdown_text(attribution.get("uncertainty", ""))
            + "; prediction_status="
            + _markdown_text(attribution.get("prediction_status", ""))
        )
    return result or ["- No admitted transferable conclusion attributions."]


def _render_review_report(
    *,
    run_id: str,
    artifacts: _RunArtifacts,
    comparison: Mapping[str, Any],
    permissions: Mapping[str, Any],
    counts: Mapping[str, int],
) -> str:
    sections: list[str] = [
        "# Self-Evolution Phase A2 Review Report",
        "",
        f"Run: `{_markdown_text(run_id)}`",
        "",
        "This is an offline, candidate-only audit. It is not a rollout, an "
        "accepted experience, a promotion, or evidence of policy improvement.",
        "",
        "## Counts",
        "",
        *[f"- {key}: {value}" for key, value in sorted(counts.items())],
        "",
    ]
    headings = (
        ("Direct visual facts", "observed_visual"),
        ("Robot-state facts", "observed_robot_state"),
        ("Benchmark annotations", "annotation"),
        ("Cross-segment inferences", "cross_segment_inference"),
    )
    for heading, source_type in headings:
        sections.extend(
            [f"## {heading}", "", *_fact_lines(artifacts.episode_facts, source_type), ""]
        )

    sections.extend(["## Episode-specific facts", ""])
    if artifacts.episode_facts:
        admitted_facts = [
            record
            for record in artifacts.episode_facts
            if record.get("admission_status") == "admitted_candidate"
        ]
        for record in admitted_facts:
            fact = record.get("fact", {})
            sections.append(
                "- "
                + _markdown_text(fact.get("claim", ""))
                + f" — transferable_allowed={fact.get('transferable_allowed')}, "
                + f"trajectory={_markdown_text(record.get('trajectory_id', ''))}"
            )
        if not admitted_facts:
            sections.append("- None recorded for an admitted candidate.")
    else:
        sections.append("- None recorded.")

    sections.extend(["", "## Untrusted rejected claims", ""])
    rejected_facts = [
        record
        for record in artifacts.episode_facts
        if record.get("admission_status") == "rejected_with_abstention"
    ]
    if rejected_facts:
        for record in rejected_facts:
            fact = record.get("fact", {})
            linkage = record.get("abstention_linkage", {})
            sections.append(
                "- UNTRUSTED: "
                + _markdown_text(fact.get("claim", ""))
                + f" — abstention={_markdown_text(linkage.get('reason', 'unknown'))}, "
                + f"output_sha256={_markdown_text(linkage.get('output_sha256', ''))}"
            )
    else:
        sections.append("- None.")

    sections.extend(["", "## Transferable guidance", ""])
    if artifacts.candidates:
        for candidate in artifacts.candidates:
            sections.append(
                f"- `{_markdown_text(candidate.get('experience_id', ''))}`: "
                + _markdown_text(candidate.get("guidance", {}))
            )
    else:
        sections.append("- No candidate passed all gates.")

    sections.extend(["", "## Transferable conclusion attributions", ""])
    sections.extend(_transferable_attribution_lines(artifacts))

    sections.extend(["", "## Unverified predicted effects", ""])
    pending = 0
    for candidate in artifacts.candidates:
        for effect in candidate.get("predicted_effects", []):
            if not isinstance(effect, Mapping) or effect.get(
                "validation_status"
            ) != "pending_validation":
                continue
            pending += 1
            sections.append(f"- {_markdown_text(effect)}")
    if not pending:
        sections.append("- None published.")

    sections.extend(["", "## Abstentions", ""])
    if artifacts.abstentions:
        for abstention in artifacts.abstentions:
            sections.append(
                f"- {_markdown_text(abstention.get('trajectory_id', ''))}: "
                f"`{_markdown_text(abstention.get('reason', 'unknown'))}` — "
                f"field={_markdown_text(abstention.get('offending_field', ''))}, "
                f"evidence={_markdown_text(abstention.get('evidence_refs', []))}"
            )
    else:
        sections.append("- None.")
    if artifacts.source_abstentions:
        sections.extend(["", "### Import/evidence-source abstentions", ""])
        for abstention in artifacts.source_abstentions:
            sections.append(
                f"- {_markdown_text(abstention.get('trajectory_id', 'unknown'))}: "
                f"`{_markdown_text(abstention.get('reason', 'unknown'))}` — "
                f"stage={_markdown_text(abstention.get('stage', 'unknown'))}, "
                f"schema={_markdown_text(abstention.get('schema', 'unknown'))}"
            )

    sections.extend(
        [
            "",
            "## Data permission and model calls",
            "",
            *[f"- {key}: `{_markdown_text(value)}`" for key, value in sorted(permissions.items())],
            "",
            "Only sparse review keyframes are copied locally. No HDF5, PKL, "
            "complete image sequence, video, API secret, or image base64 is stored "
            "in the JSON/Markdown artifacts.",
            "",
            "## Old Phase A versus Phase A2",
            "",
            f"- Baseline structural metrics: `{_markdown_text(comparison['baseline']['metrics'])}`",
            f"- Phase A2 structural metrics: `{_markdown_text(comparison['phase_a2']['metrics'])}`",
            "- This comparison is structural and does not claim runtime effectiveness.",
            "",
            "## Retrieval boundary",
            "",
            "The snapshot draft is development-only, candidate-only, requires "
            "human review, and has runtime retrieval disabled.",
            "",
        ]
    )
    return "\n".join(sections)


def _snapshot_metadata(
    *,
    output_root: Path,
    input_manifest: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seeds: list[dict[str, Any]] = []
    entries = input_manifest.get("entries", [])
    if isinstance(entries, list):
        for index, value in enumerate(entries):
            if not isinstance(value, Mapping):
                continue
            seeds.append(
                {
                    "entry_index": index,
                    "trajectory_seed": value.get("seed"),
                    "episode_id": value.get("episode_id"),
                    "task": value.get("task"),
                    "task_config": value.get("task_config"),
                }
            )
    return {
        "schema": "roboharn_evo/experience_snapshot_draft/v1",
        "schema_version": 1,
        "draft_id": f"{output_root.name}_snapshot_draft",
        "development_only": True,
        "candidate_only": True,
        "runtime_retrieval": False,
        "retrieval_eligible": False,
        "human_review_required": True,
        "automatic_promotion": False,
        "expert_prior_used": True,
        "bootstrap_seed": seeds,
        "candidate_ids": [value.get("experience_id") for value in candidates],
        "candidate_file": "../candidate_experiences.jsonl",
        "candidate_content_sha256": _json_sha256(list(candidates)),
        "activation_status": "not_activated",
    }


def _publish(
    *,
    output_root: Path,
    input_manifest: Mapping[str, Any],
    artifacts: _RunArtifacts,
    batch: RMBenchTrajectoryEvidenceBatch,
    adapter_config: TrajectoryEvidenceAdapterConfig,
    transport: OpenAICompatibleMultimodalTransport | None,
    comparison: Mapping[str, Any],
    created_at: str,
    permissions: Mapping[str, Any],
    api_key: str | None,
) -> dict[str, str]:
    parent = output_root.parent
    staging_name = f".{output_root.name}.phase-a2-object-{secrets.token_hex(12)}"
    staging = parent / staging_name
    try:
        staging.mkdir(mode=0o700)
    except OSError as exc:
        raise PhaseA2RuntimeError("cannot create Phase A2 staging directory") from exc
    published = False
    try:
        review_records: list[dict[str, Any]] = []
        for record in artifacts.review_images:
            serializable = {key: value for key, value in record.items() if key != "_payload"}
            review_records.append(serializable)
            _write_bytes(staging / str(record["relative_path"]), record["_payload"])

        jsonl_values = {
            "visual_evidence.jsonl": artifacts.visual_evidence,
            "review_image_index.jsonl": review_records,
            "episode_specific_facts.jsonl": artifacts.episode_facts,
            "claim_attributions.jsonl": artifacts.attributions,
            "candidate_experiences.jsonl": artifacts.candidates,
            "abstentions.jsonl": artifacts.abstentions,
            "source_abstentions.jsonl": artifacts.source_abstentions,
            "model_call_audit.jsonl": artifacts.model_call_audits,
        }
        for name, records in jsonl_values.items():
            _write_jsonl(staging / name, records)
        _write_json(staging / "input_manifest.json", input_manifest)
        _write_json(staging / "old_new_comparison.json", comparison)
        snapshot = _snapshot_metadata(
            output_root=output_root,
            input_manifest=input_manifest,
            candidates=artifacts.candidates,
        )
        _write_json(staging / "snapshot_draft" / "metadata.json", snapshot)

        counts = {
            "input_entries": len(batch.input_manifest.get("entries", [])),
            "visual_evidence_bundles": len(artifacts.visual_evidence),
            "review_keyframes": len(review_records),
            "episode_specific_facts": len(artifacts.episode_facts),
            "admitted_episode_specific_facts": sum(
                value.get("admission_status") == "admitted_candidate"
                for value in artifacts.episode_facts
            ),
            "rejected_untrusted_fact_claims": sum(
                value.get("admission_status") == "rejected_with_abstention"
                for value in artifacts.episode_facts
            ),
            "claim_attributions": len(artifacts.attributions),
            "candidate_experiences": len(artifacts.candidates),
            "abstentions": len(artifacts.abstentions),
            "source_abstentions": len(artifacts.source_abstentions),
            "model_calls": len(artifacts.model_call_audits),
        }
        report = _render_review_report(
            run_id=output_root.name,
            artifacts=artifacts,
            comparison=comparison,
            permissions=permissions,
            counts=counts,
        )
        _write_bytes(staging / "review_report.md", report.encode("utf-8"))

        if _sensitive_payload(
            {
                "visual": artifacts.visual_evidence,
                "facts": artifacts.episode_facts,
                "attributions": artifacts.attributions,
                "candidates": artifacts.candidates,
                "abstentions": artifacts.abstentions,
                "source_abstentions": artifacts.source_abstentions,
                "audits": artifacts.model_call_audits,
                "comparison": comparison,
                "snapshot": snapshot,
                "report": report,
            },
            api_key=api_key,
        ):
            raise PhaseA2RuntimeError(
                "refusing to publish a secret or image-base64 text artifact"
            )

        pre_manifest_files = sorted(
            str(path.relative_to(staging))
            for path in staging.rglob("*")
            if path.is_file()
        )
        integrity = {
            name: {
                "sha256": _sha256_file(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in pre_manifest_files
        }
        configuration = {
            "adapter": adapter_config.to_dict(),
            "adapter_configuration_sha256": adapter_config.configuration_sha256,
            "transport": None if transport is None else transport.configuration_identity(),
            "candidate_only": True,
            "automatic_promotion": False,
            "runtime_retrieval": False,
            "rollout_integration": False,
            "raw_artifact_copy": False,
            "sparse_review_keyframes": True,
            "atomic_publication": True,
            "publication_strategy": "same_parent_relative_symlink_no_clobber",
        }
        manifest = {
            "schema": "roboharn_evo/self_evolution_phase_a2_run/v1",
            "schema_version": 1,
            "run_id": output_root.name,
            "created_at": created_at,
            "phase": "self_evolution_phase_a2",
            "mode": "offline_candidate_only",
            "producer": {
                "kind": "reflector",
                "name": "roboharn_evo_visual_whole_trajectory_reflection",
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
                "snapshot_draft/metadata.json": "roboharn_evo/experience_snapshot_draft/v1",
                "old_new_comparison.json": "roboharn_evo/phase_a2_old_new_comparison/v1",
                "review_report.md": "roboharn_evo/self_evolution_review_report/phase_a2",
            },
            "file_integrity": {
                "algorithm": "sha256",
                "files": integrity,
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
                "strategy": "same_parent_relative_symlink_no_clobber",
                "archive_guidance": {
                    "dereference_output_symlink": True,
                    "archive_hidden_backing_with_link": True,
                },
            },
        }
        _write_json(staging / "run_manifest.json", manifest)

        checksum_names = sorted(
            str(path.relative_to(staging))
            for path in staging.rglob("*")
            if path.is_file()
        )
        checksums = {
            "schema": "roboharn_evo/sha256_manifest/v1",
            "schema_version": 1,
            "algorithm": "sha256",
            "files": {
                name: _sha256_file(staging / name) for name in checksum_names
            },
            "self_hash": "excluded_recursive_digest",
        }
        _write_json(staging / "sha256sums.json", checksums)

        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        try:
            os.symlink(staging_name, output_root)
        except FileExistsError as exc:
            raise PhaseA2RuntimeError(f"output root already exists: {output_root}") from exc
        except OSError as exc:
            raise PhaseA2RuntimeError("atomic no-clobber publication failed") from exc
        published = True
        if output_root.resolve(strict=True) != staging.resolve(strict=True):
            raise PhaseA2RuntimeError("published output failed identity verification")
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return {
            name: _sha256_file(staging / name)
            for name in sorted(
                str(path.relative_to(staging))
                for path in staging.rglob("*")
                if path.is_file()
            )
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


def run_phase_a2(
    *,
    manifest_path: str | os.PathLike[str],
    adapter_config: TrajectoryEvidenceAdapterConfig,
    output_root: str | os.PathLike[str],
    authorization: ImageEgressAuthorization | None = None,
    context_radius: int = 1,
    evidence_limits: TrajectoryEvidenceLimits | None = None,
    service_url: str = "http://127.0.0.1:9104",
    model: str = "gpt-5.5",
    reasoning_effort: str = "xhigh",
    timeout_sec: float = 600.0,
    api_key: str | None = None,
    max_model_images: int = 8,
    unverified_license_acknowledged: bool = False,
    baseline_phase_a_root: str | os.PathLike[str] | None = None,
    created_at: str | None = None,
    transport: OpenAICompatibleMultimodalTransport | None = None,
    evidence_batch: RMBenchTrajectoryEvidenceBatch | None = None,
) -> PhaseA2RunResult:
    """Extract, reflect or abstain, and atomically publish one Phase A2 run."""

    resolved_output = _resolve_output_root(output_root)
    timestamp = created_at or _utc_now()
    decision = authorization or ImageEgressAuthorization()
    manifest_import_verified = evidence_batch is None
    if evidence_batch is None:
        authorized_manifest = _resolve_phase_a_manifest_link(manifest_path)
        evidence_batch = extract_rmbench_visual_evidence_manifest(
            authorized_manifest,
            adapter_config=adapter_config,
            context_radius=context_radius,
            extraction_limits=evidence_limits,
        )
    if not evidence_batch.entries and not evidence_batch.abstentions:
        raise PhaseA2RuntimeError("evidence extraction returned no entries or abstentions")
    manifest = evidence_batch.input_manifest
    license_status = str(manifest.get("license_status", "unverified"))
    if license_status == "verified":
        license_allows_egress = True
        license_denial_reason: str | None = None
    elif license_status == "unverified":
        license_allows_egress = unverified_license_acknowledged
        license_denial_reason = (
            None
            if license_allows_egress
            else "unverified_license_not_acknowledged"
        )
    elif license_status == "restricted":
        license_allows_egress = False
        license_denial_reason = "restricted_license_forbids_egress"
    else:
        license_allows_egress = False
        license_denial_reason = "unsupported_license_status"
    artifacts = _RunArtifacts(
        visual_evidence=[entry.extracted.bundle.to_dict() for entry in evidence_batch.entries],
        review_images=_review_image_records(evidence_batch),
        episode_facts=[],
        attributions=[],
        candidates=[],
        abstentions=[],
        source_abstentions=[dict(value) for value in evidence_batch.abstentions],
        model_call_audits=[],
    )

    active_transport = transport
    capability_failure: str | None = None
    if decision.granted and license_allows_egress:
        active_transport = active_transport or OpenAICompatibleMultimodalTransport(
            service_url=service_url,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_sec=timeout_sec,
            api_key=api_key,
        )
        try:
            report = active_transport.capability_preflight(authorization=decision)
            artifacts.model_call_audits.append(
                {
                    "schema": "roboharn_evo/phase_a2_model_call_audit/v1",
                    "schema_version": 1,
                    "trajectory_id": None,
                    "call_kind": "synthetic_capability_preflight",
                    "status": "passed",
                    **report.to_dict(),
                }
            )
        except (MultimodalTransportError, ValueError) as exc:
            capability_failure = type(exc).__name__
            failed_audit = active_transport.last_preflight_audit
            artifacts.model_call_audits.append(
                {
                    "schema": "roboharn_evo/phase_a2_model_call_audit/v1",
                    "schema_version": 1,
                    "trajectory_id": None,
                    "call_kind": "synthetic_capability_preflight",
                    **({} if failed_audit is None else dict(failed_audit)),
                    "status": "failed",
                    "error_type": capability_failure,
                }
            )

    prompt_hash = hashlib.sha256(
        generic_reflection_system_prompt().encode("utf-8")
    ).hexdigest()
    for entry in evidence_batch.entries:
        if not decision.granted:
            artifacts.abstentions.append(
                _record_abstention(
                    trajectory_id=entry.trajectory_id,
                    reason="image_egress_not_authorized",
                    offending_field="backend.authorization",
                    backend="not_invoked",
                    created_at=timestamp,
                    detail=(
                        "operator did not authorize expert-image egress; no "
                        "capability probe or expert image was sent"
                    ),
                    prompt_template_hash=prompt_hash,
                )
            )
            continue
        if not license_allows_egress:
            reason = license_denial_reason or "license_forbids_egress"
            artifacts.abstentions.append(
                _record_abstention(
                    trajectory_id=entry.trajectory_id,
                    reason=reason,
                    offending_field="permissions.license_status",
                    backend="not_invoked",
                    created_at=timestamp,
                    detail=(
                        "expert image egress is denied by the declared license "
                        f"policy ({license_status})"
                        if reason != "unverified_license_not_acknowledged"
                        else (
                            "expert image egress requires an explicit "
                            "acknowledgement of license_status=unverified"
                        )
                    ),
                    prompt_template_hash=prompt_hash,
                )
            )
            continue
        if capability_failure is not None or active_transport is None:
            artifacts.abstentions.append(
                _record_abstention(
                    trajectory_id=entry.trajectory_id,
                    reason="capability_preflight_failed",
                    offending_field="backend.capabilities",
                    backend=(
                        "unavailable"
                        if active_transport is None
                        else active_transport.backend_name
                    ),
                    created_at=timestamp,
                    detail=(
                        "active multimodal structured-output capability was not "
                        f"proven ({capability_failure or 'unavailable'})"
                    ),
                    prompt_template_hash=prompt_hash,
                )
            )
            continue

        critic = OpenAICompatibleLeakageCritic(
            active_transport,
            authorization=decision,
        )
        backend = OpenAICompatibleMultimodalBackend(
            active_transport,
            authorization=decision,
            image_resolver=_image_resolver(entry),
            max_images=max_model_images,
        )
        reflector = WholeTrajectoryReflector(
            backend,
            quality_gate=ReflectionQualityGate(leakage_critic=critic),
        )
        request = _whole_request(entry)
        result = reflector.reflect(request)
        if backend.last_call_audit is not None:
            artifacts.model_call_audits.append(
                {
                    "schema": "roboharn_evo/phase_a2_model_call_audit/v1",
                    "schema_version": 1,
                    "trajectory_id": entry.trajectory_id,
                    "call_kind": "whole_trajectory_reflection",
                    **backend.last_call_audit,
                }
            )
        if critic.last_call_audit is not None:
            artifacts.model_call_audits.append(
                {
                    "schema": "roboharn_evo/phase_a2_model_call_audit/v1",
                    "schema_version": 1,
                    "trajectory_id": entry.trajectory_id,
                    "call_kind": "episode_leakage_critic",
                    **critic.last_call_audit,
                }
            )

        result_dict = result.to_dict()
        if _sensitive_payload(result_dict, api_key=api_key):
            artifacts.abstentions.append(
                _record_abstention(
                    trajectory_id=entry.trajectory_id,
                    reason="sensitive_payload_rejected",
                    offending_field="backend_output",
                    backend=backend.backend_name,
                    created_at=timestamp,
                    detail="model output contained a secret or image-data marker",
                    prompt_template_hash=prompt_hash,
                    output_sha256=result.output_sha256,
                )
            )
            continue
        admitted = result.candidate_experience is not None
        abstention_linkage = (
            None
            if result.abstention is None
            else {
                "reason": result.abstention.get("reason"),
                "output_sha256": result.abstention.get(
                    "output_sha256", result.output_sha256
                ),
            }
        )
        admission_status = (
            "admitted_candidate" if admitted else "rejected_with_abstention"
        )
        artifacts.episode_facts.extend(
            {
                "schema": "roboharn_evo/episode_specific_fact_record/v1",
                "schema_version": 1,
                "trajectory_id": entry.trajectory_id,
                "admission_status": admission_status,
                "abstention_linkage": abstention_linkage,
                "fact": dict(value),
            }
            for value in result.episode_specific_facts
        )
        artifacts.attributions.extend(
            {
                "schema": "roboharn_evo/reflection_attribution_record/v1",
                "schema_version": 1,
                "trajectory_id": entry.trajectory_id,
                "admission_status": admission_status,
                "abstention_linkage": abstention_linkage,
                "attribution": dict(value),
            }
            for value in result.attributions
        )
        if result.candidate_experience is not None:
            artifacts.candidates.append(dict(result.candidate_experience))
        elif result.abstention is not None:
            artifacts.abstentions.append(dict(result.abstention))

    baseline = None
    if baseline_phase_a_root is not None:
        baseline = Path(baseline_phase_a_root).resolve(strict=True)
        if not baseline.is_dir():
            raise PhaseA2RuntimeError("baseline Phase A root must be a directory")
    comparison = _baseline_comparison(baseline, artifacts.candidates)
    expert_images_sent = sum(
        int(value.get("image_count", 0))
        for value in artifacts.model_call_audits
        if value.get("call_kind") == "whole_trajectory_reflection"
    )
    permissions = {
        "manifest_authorized_read_only_input": manifest_import_verified,
        "input_assurance": (
            "verified_manifest_import"
            if manifest_import_verified
            else "injected_test_seam_unverified"
        ),
        "license_status": license_status,
        "license_allows_image_egress": license_allows_egress,
        "license_denial_reason": license_denial_reason,
        "unverified_license_acknowledged": unverified_license_acknowledged,
        "image_egress_authorized": decision.granted,
        "authorization_scope": decision.scope,
        "authorization_sha256": decision.audit_sha256(),
        "capability_preflight_passed": bool(
            active_transport is not None
            and active_transport.capability_report is not None
        ),
        "expert_images_sent": expert_images_sent,
        "external_model_invoked": bool(artifacts.model_call_audits),
        "api_secret_logged": False,
        "image_base64_logged": False,
        "runtime_retrieval_activated": False,
    }
    file_hashes = _publish(
        output_root=resolved_output,
        input_manifest=manifest,
        artifacts=artifacts,
        batch=evidence_batch,
        adapter_config=adapter_config,
        transport=active_transport,
        comparison=comparison,
        created_at=timestamp,
        permissions=permissions,
        api_key=api_key,
    )
    return PhaseA2RunResult(
        output_root=resolved_output,
        candidate_count=len(artifacts.candidates),
        abstention_count=len(artifacts.abstentions),
        source_abstention_count=len(artifacts.source_abstentions),
        image_egress_authorized=decision.granted,
        expert_images_sent=expert_images_sent,
        file_sha256=file_hashes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract sparse expert evidence and run offline whole-trajectory "
            "candidate reflection. Image egress is denied by default."
        )
    )
    parser.add_argument("--manifest", required=True, help="explicit Phase A manifest JSON")
    parser.add_argument(
        "--adapter-config",
        help=(
            "optional validated benchmark/embodiment evidence adapter JSON; "
            "otherwise use the copied RMBench dual-arm adapter contract"
        ),
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help=(
            "RMBench RGB camera for the built-in adapter; repeat to set order "
            "(default: adapter-defined head_camera then third_view)"
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="fresh descendant of RoboHarn-Evo/eval_result/self_evolution/phase_a2",
    )
    parser.add_argument(
        "--baseline-phase-a-root",
        help="optional previous Phase A run used only for structural comparison",
    )
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--max-model-images", type=int, default=8)
    parser.add_argument("--service-url", default="http://127.0.0.1:9104")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument(
        "--api-key-env",
        help="optional environment-variable name; the value is never serialized",
    )
    parser.add_argument(
        "--authorize-image-egress",
        action="store_true",
        help=(
            "explicitly authorize the synthetic image capability probe and sparse "
            "expert keyframes to the configured model service"
        ),
    )
    parser.add_argument(
        "--authorization-assertion",
        help=(
            "required audit statement when --authorize-image-egress is used; "
            "must describe this operator decision without credentials"
        ),
    )
    parser.add_argument(
        "--acknowledge-unverified-license",
        action="store_true",
        help=(
            "explicitly acknowledge that the expert manifest currently has "
            "license_status=unverified before image egress"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.adapter_config:
            adapter_payload = _read_json_object(
                Path(args.adapter_config),
                max_bytes=_MAX_CONFIG_BYTES,
                label="adapter config",
            )
            adapter = _adapter_config_from_dict(adapter_payload)
        else:
            adapter = (
                rmbench_dual_arm_trajectory_evidence_config(
                    camera_ids=tuple(args.cameras)
                )
                if args.cameras
                else rmbench_dual_arm_trajectory_evidence_config()
            )
        api_key: str | None = None
        if args.api_key_env:
            if _SAFE_ENV_NAME_RE.fullmatch(args.api_key_env) is None:
                raise PhaseA2RuntimeError("--api-key-env is not a safe environment name")
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                raise PhaseA2RuntimeError("configured API key environment variable is empty")
        if args.authorize_image_egress:
            if not args.authorization_assertion:
                raise PhaseA2RuntimeError(
                    "--authorize-image-egress requires --authorization-assertion"
                )
            if not args.acknowledge_unverified_license:
                raise PhaseA2RuntimeError(
                    "--authorize-image-egress requires "
                    "--acknowledge-unverified-license for this development protocol"
                )
            authorization = ImageEgressAuthorization.operator_granted(
                assertion=args.authorization_assertion
            )
        else:
            if args.authorization_assertion:
                raise PhaseA2RuntimeError(
                    "--authorization-assertion requires --authorize-image-egress"
                )
            authorization = ImageEgressAuthorization()
        result = run_phase_a2(
            manifest_path=args.manifest,
            adapter_config=adapter,
            output_root=args.output_root,
            authorization=authorization,
            context_radius=args.context_radius,
            service_url=args.service_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout_sec=args.timeout_sec,
            api_key=api_key,
            max_model_images=args.max_model_images,
            unverified_license_acknowledged=args.acknowledge_unverified_license,
            baseline_phase_a_root=args.baseline_phase_a_root,
        )
    except (
        CapabilityPreflightError,
        MultimodalTransportError,
        PhaseA2RuntimeError,
        TrajectoryEvidenceError,
        ValueError,
    ) as exc:
        print(f"Phase A2 failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
