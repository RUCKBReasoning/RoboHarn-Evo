"""Human-review rendering for Phase A2.1 hierarchical reflection.

The renderer is deliberately provider- and benchmark-neutral.  It only formats
already admitted/rejected records and audited image bindings; it never infers a
fact or changes a model disposition.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def _text(value: Any, *, limit: int = 1600) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            text = repr(value)
    return text.replace("\n", " ").replace("|", "\\|")[:limit]


def _link(label: Any, path: Any) -> str:
    label_text = _text(label, limit=200).replace("[", "\\[").replace("]", "\\]")
    if not isinstance(path, str) or not path or any(
        marker in path for marker in ("\n", "\r", ")", "data:")
    ):
        return label_text
    return f"[{label_text}]({path})"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend("| " + " | ".join(_text(value) for value in row) + " |" for row in rows)
    if not rows:
        result.append("| " + " | ".join("None" for _ in headers) + " |")
    return result


def _segment_range(segment: Mapping[str, Any]) -> tuple[Any, Any]:
    derivation = segment.get("derivation")
    frame_range = (
        derivation.get("frame_range") if isinstance(derivation, Mapping) else None
    )
    if not isinstance(frame_range, Mapping):
        return None, None
    return frame_range.get("start_inclusive"), frame_range.get("end_inclusive")


def _chunk_range(chunk: Mapping[str, Any]) -> tuple[Any, Any]:
    frame_range = chunk.get("frame_range")
    if isinstance(frame_range, Mapping):
        return frame_range.get("start_inclusive"), frame_range.get("end_inclusive")
    return chunk.get("start_frame"), chunk.get("end_frame")


def render_hierarchical_review_report(
    *,
    run_id: str,
    segments: Sequence[Mapping[str, Any]],
    action_chunks: Sequence[Mapping[str, Any]],
    visual_evidence: Sequence[Mapping[str, Any]],
    derived_presentations: Sequence[Mapping[str, Any]],
    review_images: Sequence[Mapping[str, Any]],
    subtask_summaries: Sequence[Mapping[str, Any]],
    episode_facts: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    attributions: Sequence[Mapping[str, Any]],
    abstentions: Sequence[Mapping[str, Any]],
    model_call_audits: Sequence[Mapping[str, Any]],
    model_outputs: Sequence[Mapping[str, Any]],
    permissions: Mapping[str, Any],
    leakage_audit: Mapping[str, Any],
) -> str:
    """Render a direct evidence/call map without reinterpreting its contents."""

    image_path_by_ref: dict[str, str] = {}
    for record in review_images:
        path = record.get("relative_path")
        refs = record.get("evidence_ids", [])
        if not isinstance(path, str) or not isinstance(refs, list):
            continue
        for ref in refs:
            if isinstance(ref, str):
                image_path_by_ref[ref] = path
        presentation_id = record.get("presentation_id")
        if isinstance(presentation_id, str):
            image_path_by_ref[presentation_id] = path

    segment_rows: list[list[Any]] = []
    for segment in sorted(segments, key=lambda value: value.get("segment_index", 0)):
        start, end = _segment_range(segment)
        context = segment.get("context")
        annotation = (
            context.get("subtask_instruction")
            if isinstance(context, Mapping)
            else None
        )
        segment_rows.append(
            [
                segment.get("segment_index"),
                segment.get("segment_id"),
                f"{start}..{end}",
                annotation,
            ]
        )

    chunk_rows: list[list[Any]] = []
    for chunk in action_chunks:
        start, end = _chunk_range(chunk)
        transitions = chunk.get(
            "gripper_transition_refs", chunk.get("gripper_transitions", [])
        )
        chunk_rows.append(
            [
                chunk.get("segment_index"),
                chunk.get("action_chunk_id"),
                f"{start}..{end}",
                chunk.get("active_arms", []),
                chunk.get("observed_motion_state"),
                chunk.get("inferred_phase"),
                transitions,
            ]
        )

    presentation_rows: list[list[Any]] = []
    for record in derived_presentations:
        presentation_id = record.get("presentation_id")
        provenance = record.get("presentation_provenance")
        if not isinstance(provenance, Mapping):
            provenance = {}
        source_refs = record.get(
            "underlying_evidence_refs",
            provenance.get("source_evidence_refs", []),
        )
        linked_source_refs = (
            " ".join(
                _link(reference, image_path_by_ref.get(str(reference)))
                for reference in source_refs
            )
            if isinstance(source_refs, Sequence)
            and not isinstance(source_refs, (str, bytes))
            else _text(source_refs)
        )
        presentation_rows.append(
            [
                presentation_id,
                record.get("presentation_kind", record.get("kind")),
                linked_source_refs,
                record.get(
                    "crop_box",
                    provenance.get("crop_box_xyxy_exclusive"),
                ),
                record.get(
                    "resize_transform",
                    provenance.get("resize_transform"),
                ),
                _link(presentation_id, image_path_by_ref.get(str(presentation_id))),
            ]
        )

    call_rows: list[list[Any]] = []
    call_image_rows: list[list[Any]] = []
    provider_output_rows: list[list[Any]] = []
    for audit in model_call_audits:
        bindings = audit.get(
            "presentation_bindings",
            audit.get("image_bindings", audit.get("model_image_bindings", [])),
        )
        binding_count = 0
        if isinstance(bindings, list):
            for binding in bindings:
                if not isinstance(binding, Mapping):
                    continue
                binding_count += 1
                presentation_id = binding.get(
                    "presentation_id", binding.get("model_image_id")
                )
                source_refs = binding.get(
                    "source_evidence_refs", binding.get("underlying_evidence_refs", [])
                )
                path = image_path_by_ref.get(str(presentation_id))
                linked_refs = (
                    " ".join(
                        _link(reference, image_path_by_ref.get(str(reference)))
                        for reference in source_refs
                    )
                    if isinstance(source_refs, Sequence)
                    and not isinstance(source_refs, (str, bytes))
                    else _text(source_refs)
                )
                call_image_rows.append(
                    [
                        audit.get("model_call_index"),
                        audit.get("request_sha256"),
                        audit.get("layer"),
                        audit.get("scope_id", audit.get("trajectory_id")),
                        binding.get("model_image_alias"),
                        _link(presentation_id, path),
                        binding.get("presentation_kind"),
                        linked_refs,
                    ]
                )
        call_rows.append(
            [
                audit.get("call_kind", audit.get("purpose")),
                audit.get("layer"),
                audit.get("scope_id", audit.get("trajectory_id")),
                audit.get("status"),
                audit.get("image_count", 0),
                audit.get("image_total_bytes", 0),
                binding_count,
            ]
        )
        if "provider_parsed_json" in audit:
            provider_output_rows.append(
                [
                    audit.get("model_call_index"),
                    audit.get("layer"),
                    audit.get("scope_id", audit.get("trajectory_id")),
                    audit.get("model_output_sha256"),
                    audit.get("provider_canonical_json_sha256"),
                    audit.get("restored_output_sha256"),
                    audit.get("reference_restoration_applied"),
                    audit.get("provider_parsed_json"),
                ]
            )

    output_rows: list[list[Any]] = []
    for record in model_outputs:
        payload = record.get("parsed_json")
        if payload is None:
            payload = record.get("raw_text")
        if payload is None and record.get("payload_suppressed") is True:
            payload = "suppressed_unsafe_output"
        output_rows.append(
            [
                record.get("layer"),
                record.get("scope_id"),
                record.get("payload_kind"),
                record.get("output_sha256"),
                record.get("parse_error"),
                payload,
            ]
        )

    admitted_rows: list[list[Any]] = []
    rejected_rows: list[list[Any]] = []
    for record in episode_facts:
        fact = record.get("fact") if isinstance(record.get("fact"), Mapping) else record
        row = [
            record.get("layer"),
            record.get("scope_id", record.get("trajectory_id")),
            fact.get("source_type"),
            fact.get("claim"),
            fact.get("supporting_evidence_refs", []),
            fact.get("confidence"),
            fact.get("uncertainty"),
        ]
        if str(record.get("admission_status", "")).startswith("admitted"):
            admitted_rows.append(row)
        else:
            rejected_rows.append([*row, record.get("issues", record.get("abstention_linkage"))])

    lines = [
        "# Self-Evolution Phase A2.1 Review Report",
        "",
        f"Run: `{_text(run_id)}`",
        "",
        "This is an offline candidate-only audit. It is not rollout evidence, "
        "promotion, evaluation, or runtime retrieval activation.",
        "",
        "## Subtasks",
        "",
        *_table(["index", "segment", "frames", "annotation intent"], segment_rows),
        "",
        "## Action chunks",
        "",
        *_table(
            [
                "segment",
                "chunk",
                "frames",
                "active arms",
                "observed motion",
                "inferred phase",
                "gripper transitions",
            ],
            chunk_rows,
        ),
        "",
        "## Derived visual presentations",
        "",
        *_table(
            ["presentation", "kind", "source refs", "crop", "resize", "review image"],
            presentation_rows,
        ),
        "",
        "## Model calls and images actually sent",
        "",
        *_table(
            ["call", "layer", "scope", "status", "images", "bytes", "bindings"],
            call_rows,
        ),
        "",
        "## Every model image and its request binding",
        "",
        *_table(
            [
                "call index",
                "request SHA-256",
                "layer",
                "scope",
                "image alias",
                "presentation",
                "kind",
                "original evidence",
            ],
            call_image_rows,
        ),
        "",
        "## Provider output and deterministic reference restoration",
        "",
        *_table(
            [
                "call index",
                "layer",
                "scope",
                "provider text SHA-256",
                "provider JSON SHA-256",
                "restored JSON SHA-256",
                "refs restored",
                "provider alias JSON (preview)",
            ],
            provider_output_rows,
        ),
        "",
        "## Reference-restored local model outputs",
        "",
        *_table(
            ["layer", "scope", "kind", "SHA-256", "parse error", "payload"],
            output_rows,
        ),
        "",
        "## Admitted episode-specific facts",
        "",
        *_table(
            ["layer", "scope", "source", "claim", "evidence", "confidence", "uncertainty"],
            admitted_rows,
        ),
        "",
        "## Rejected or locally abstained facts",
        "",
        *_table(
            [
                "layer",
                "scope",
                "source",
                "claim",
                "evidence",
                "confidence",
                "uncertainty",
                "reason",
            ],
            rejected_rows,
        ),
        "",
        "## Subtask summaries",
        "",
    ]
    lines.extend(f"- `{_text(value.get('segment_id'))}`: {_text(value)}" for value in subtask_summaries)
    if not subtask_summaries:
        lines.append("- None.")

    lines.extend(["", "## Final candidate or abstention", ""])
    if candidates:
        lines.extend(f"- Candidate `{_text(value.get('experience_id'))}`: {_text(value)}" for value in candidates)
    else:
        lines.append("- No candidate passed all gates.")
    lines.extend(f"- Abstention: {_text(value)}" for value in abstentions)
    if not abstentions and not candidates:
        lines.append("- No terminal disposition was recorded.")

    lines.extend(["", "## Claim attributions", ""])
    lines.extend(f"- {_text(value)}" for value in attributions)
    if not attributions:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Episode-specific versus transferable boundary",
            "",
            f"- Episode-specific facts: `{len(episode_facts)}`",
            f"- Transferable candidates: `{len(candidates)}`",
            "- Episode facts remain private evidence and are not planner input.",
            "",
            "## Fixed-answer leakage audit",
            "",
            f"- `{_text(leakage_audit)}`",
            "",
            "## Permissions and lifecycle",
            "",
        ]
    )
    lines.extend(f"- {key}: `{_text(value)}`" for key, value in sorted(permissions.items()))
    lines.extend(
        [
            "",
            "Candidate records, when present, remain candidate-only, not evaluated, "
            "not promoted, and not retrievable.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["render_hierarchical_review_report"]
