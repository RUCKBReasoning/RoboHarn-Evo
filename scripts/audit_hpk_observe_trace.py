from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboharn_evo.agent.hpk import (  # noqa: E402
    ABSTRACT_EFFECT_SCHEMA,
    HPKConditionBuilder,
    HPKRuntimePolicy,
    HPKUnresolved,
    AbstractEffectExtractor,
    AbstractEffectV1,
    ConditionV1,
    TaskStrategyNormalizer,
    TaskStrategyV1,
    build_observe_audit,
    candidate_semantic_features,
    determine_evidence_verdict,
    reject_private_transferable,
)


MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_TRACE_EVENTS = 100_000
REPORT_PROFILE = "hpk_p0_ab_observe_trace/2026-08-20"
PRIVATE_PROFILE = "hpk_p0_ab_private_observe_sidecar/2026-08-20"

_PRIVATE_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "instance_id",
        "track_id",
        "target_id",
        "object_contact_pose",
        "tcp_pose",
        "ee_target_pose",
        "approach_pose",
    }
)

_PLACE_EXPECTED = (
    "object_supported_by_target",
    "target_relation_satisfied",
    "object_released",
    "placement_stable",
    "gripper_empty",
)


class ObserveTraceError(RuntimeError):
    """Raised for a malformed or unsafe offline observe request."""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_explicit_trace(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    before = resolved.stat()
    if not resolved.is_file() or before.st_size > MAX_TRACE_BYTES:
        raise ObserveTraceError(
            f"trace must be one regular file no larger than {MAX_TRACE_BYTES} bytes"
        )
    raw = resolved.read_bytes()
    after = resolved.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise ObserveTraceError("trace identity changed while it was read")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ObserveTraceError(f"trace is not UTF-8: {exc}") from exc

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(events) >= MAX_TRACE_EVENTS:
            raise ObserveTraceError(
                f"trace exceeds the {MAX_TRACE_EVENTS} event budget"
            )
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ObserveTraceError(
                f"invalid JSON on trace line {line_number}: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise ObserveTraceError(
                f"trace line {line_number} must contain a JSON object"
            )
        events.append({**event, "_hpk_source_line": line_number})
    return events, {
        "path": str(resolved),
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
        "event_count": len(events),
        "read_only_observation": True,
    }


def _event_ref(event: Mapping[str, Any] | None) -> str | None:
    if not event:
        return None
    line = event.get("_hpk_source_line")
    kind = _text(event.get("event"))
    return f"trace_line_{line}:{kind}" if line is not None and kind else None


def _nearest_before(
    events: Sequence[dict[str, Any]], index: int, kind: str
) -> dict[str, Any]:
    for candidate in reversed(events[:index]):
        if candidate.get("event") == kind:
            return candidate
    return {}


def _first_after(
    events: Sequence[dict[str, Any]], index: int, kind: str
) -> tuple[int, dict[str, Any]]:
    for offset, candidate in enumerate(events[index + 1 :], start=index + 1):
        if candidate.get("event") == kind:
            return offset, candidate
    return -1, {}


def _selected_event(
    events: Sequence[dict[str, Any]], *, operation: str
) -> tuple[int, dict[str, Any]]:
    for index, event in enumerate(events):
        if (
            event.get("event") == "operation_candidate_selected"
            and _text(event.get("action_mode")).lower() == operation
        ):
            return index, event
    raise ObserveTraceError(
        f"trace has no operation_candidate_selected event for {operation!r}"
    )


def _effect_after(
    events: Sequence[dict[str, Any]], selected_index: int, operation: str
) -> tuple[int, dict[str, Any]]:
    for index, event in enumerate(events[selected_index + 1 :], selected_index + 1):
        if event.get("event") != "action_effect_verification":
            continue
        result = _mapping(event.get("result"))
        if _text(result.get("effect_type")).lower() == operation:
            return index, event
    return -1, {}


def _instance_by_private_ref(
    scene: Mapping[str, Any], private_ref: str
) -> dict[str, Any]:
    raw = scene.get("instances")
    if not isinstance(raw, list):
        return {}
    matches = [
        dict(item)
        for item in raw
        if isinstance(item, Mapping)
        and private_ref
        in {
            _text(item.get("instance_id")),
            _text(item.get("track_id")),
        }
    ]
    return matches[0] if len(matches) == 1 else {}


def _target_by_private_ref(
    scene: Mapping[str, Any], private_ref: str
) -> dict[str, Any]:
    raw = scene.get("operation_targets")
    if not isinstance(raw, list):
        return {}
    matches = [
        dict(item)
        for item in raw
        if isinstance(item, Mapping) and _text(item.get("target_id")) == private_ref
    ]
    return matches[0] if len(matches) == 1 else {}


def _target_role(target: Mapping[str, Any]) -> str:
    explicit = _text(target.get("target_role") or target.get("reference_role"))
    if explicit:
        return explicit
    return {
        "reference_region": "current_reference_region",
        "relational_reference_region": "current_reference_region",
        "object_top": "current_support_object",
        "free_support": "current_free_support_region",
        "vacated_pose": "current_vacated_support_region",
    }.get(_text(target.get("target_kind")).lower(), "")


def _phase_from_scene(scene: Mapping[str, Any], arm: str) -> str:
    manipulation = _mapping(scene.get("manipulation_state"))
    return _text(_mapping(manipulation.get(arm)).get("phase"))


def _semantic_descriptor(instance: Mapping[str, Any]) -> dict[str, Any]:
    semantic_class = _text(
        instance.get("semantic_class")
        or instance.get("object_class")
        or instance.get("class_name")
        or instance.get("class")
        or instance.get("category")
    )
    geometry_class = _text(
        instance.get("geometry_class") or instance.get("shape_class")
    )
    return {
        **({"semantic_class": semantic_class} if semantic_class else {}),
        **({"geometry_class": geometry_class} if geometry_class else {}),
    }


def _build_runtime_projection(
    episode: Mapping[str, Any],
    selected: Mapping[str, Any],
    scene: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    operation = _text(selected.get("action_mode")).lower()
    arm = _text(selected.get("arm")).lower()
    instance_ref = _text(selected.get("instance_id") or selected.get("track_id"))
    target_ref = _text(selected.get("target_id"))
    manipulated_instance = _instance_by_private_ref(scene, instance_ref)
    target = _target_by_private_ref(scene, target_ref)
    target_role = _target_role(target)
    relation = _text(
        target.get("placement_relation") or target.get("target_relation")
    ).lower()
    if relation != "center_of":
        relation = ""
    manipulated_role = "currently_held_object"
    phase = _phase_from_scene(scene, arm)
    task_family = _text(episode.get("task_name") or scene.get("task_family"))

    bound = {
        "operation": operation,
        "action_mode": operation,
        "manipulated_role": manipulated_role,
        "target_role": target_role,
        "target_relation": relation,
        "manipulation_phase": phase,
        "arm": arm,
        "preferred_arm": arm,
        "manipulated_instance_id": instance_ref,
        "target_instance_id": _text(target.get("support_instance_id")),
        "manipulated_object": _semantic_descriptor(manipulated_instance),
        "target": _semantic_descriptor(
            _instance_by_private_ref(
                scene,
                _text(target.get("support_instance_id")),
            )
        ),
        "support_valid": target.get("support_valid"),
        "free": target.get("free"),
    }
    runtime = {
        "task_family": task_family,
        "operation": operation,
        "manipulation_phase": phase,
        "manipulated_role": manipulated_role,
        "target_role": target_role,
        "target_relation": relation,
        "manipulation_state": _mapping(scene.get("manipulation_state")),
        "manipulated_instance_id": instance_ref,
        "manipulated_object": _semantic_descriptor(manipulated_instance),
        "target": bound["target"],
        "preconditions": [],
    }
    return runtime, bound, task_family


def _record_or_gap(value: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if isinstance(value, HPKUnresolved):
        return value.to_dict(), value.to_dict()
    return value.to_dict(), None


def _expected_effect(operation: str) -> AbstractEffectV1:
    expected = {
        "grasp": ["object_attached"],
        "place": list(_PLACE_EXPECTED),
        "contact": [],
    }[operation]
    return AbstractEffectV1.from_dict(
        {
            "schema": ABSTRACT_EFFECT_SCHEMA,
            "effect_type": operation,
            "expected_predicates": expected,
            "verifiability": "unverified",
        }
    )


def _effect_runtime_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("runtime_grasp_validation", "runtime_place_validation"):
        value = _mapping(result.get(key))
        if value:
            payload[key] = value
    return payload


def build_report(
    events: Sequence[dict[str, Any]],
    source: Mapping[str, Any],
    *,
    operation: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode = next(
        (event for event in events if event.get("event") == "episode_start"),
        {},
    )
    selected_index, selected = _selected_event(events, operation=operation)
    control = _nearest_before(events, selected_index, "control_decision")
    pre_scene_event = _nearest_before(events, selected_index, "scene_memory_update")
    effect_index, effect_event = _effect_after(events, selected_index, operation)
    post_scene_event = {}
    if effect_index >= 0:
        _, post_scene_event = _first_after(
            events[: effect_index + 1], selected_index, "scene_memory_update"
        )
    pre_scene = _mapping(pre_scene_event.get("scene_memory"))
    runtime, bound, task_family = _build_runtime_projection(
        episode,
        selected,
        pre_scene,
    )

    task_strategy = TaskStrategyNormalizer().normalize(
        control,
        pre_scene,
        bound,
        {"selected_skill": control.get("selected_skill", "")},
    )
    condition = HPKConditionBuilder().build(
        {**pre_scene, "task_family": task_family},
        task_strategy if isinstance(task_strategy, TaskStrategyV1) else None,
        runtime,
        bound,
        task_family=task_family,
    )
    features = candidate_semantic_features(pre_scene, selected)
    expected = _expected_effect(operation)
    effect_result = _mapping(effect_event.get("result"))
    runtime_validation = _effect_runtime_payload(effect_result)
    motion_status = "unknown"
    realization_status = "unknown"
    observed = AbstractEffectExtractor().extract(
        expected,
        pre_effect_state={},
        post_effect_state={},
        runtime_validation=runtime_validation,
        verifier_result=effect_result,
        motion_status=motion_status,
    )
    verdict = determine_evidence_verdict(
        expected,
        observed,
        realization_status=realization_status,
        motion_status=motion_status,
        runtime_validation=runtime_validation,
        verifier_result=effect_result,
    )

    condition_payload, condition_gap = _record_or_gap(condition)
    strategy_payload, strategy_gap = _record_or_gap(task_strategy)
    features_payload, features_gap = _record_or_gap(features)
    gaps = [
        value
        for value in (condition_gap, strategy_gap, features_gap)
        if value is not None
    ]
    gaps.extend(
        [
            {
                "status": "unresolved",
                "component": "geometric_strategy",
                "reason": "historical_trace_has_no_active_typed_hpk_strategy",
                "missing_fields": ["geometric_strategy_id"],
            },
            {
                "status": "unresolved",
                "component": "execution_realization",
                "reason": "historical_trace_does_not_bind_motion_and_strategy_realization_status",
                "missing_fields": ["motion_status", "realization_status"],
            },
            {
                "status": "unresolved",
                "component": "evidence",
                "reason": "complete_evidence_requires_resolved_condition_task_strategy_and_geometric_strategy",
                "missing_fields": ["geometric_strategy_id"],
            },
        ]
    )

    public_values = {
        "condition": condition_payload,
        "task_strategy": strategy_payload,
        "candidate_semantic_features": features_payload,
        "expected_effect": expected.to_dict(),
        "observed_effect": observed.to_dict(),
    }
    safety: dict[str, Any] = {}
    for name, value in public_values.items():
        reject_private_transferable(value, path=name)
        safety[name] = "passed_recursive_private_field_check"

    policy = HPKRuntimePolicy.from_mapping({"mode": "observe"})
    observe_audit = build_observe_audit(
        policy,
        condition=condition if isinstance(condition, ConditionV1) else None,
        task_strategy=(
            task_strategy if isinstance(task_strategy, TaskStrategyV1) else None
        ),
        candidates=(selected,),
        selected_candidate_id=_text(selected.get("candidate_id")),
        motion_status=motion_status,
        effect_verdict=verdict,
        evidence=None,
    )
    assert observe_audit is not None

    refs = [
        ref
        for ref in (
            _event_ref(episode),
            _event_ref(control),
            _event_ref(pre_scene_event),
            _event_ref(selected),
            _event_ref(post_scene_event),
            _event_ref(effect_event),
        )
        if ref is not None
    ]
    report = {
        "report_profile": REPORT_PROFILE,
        "mode": "observe",
        "task_family": task_family,
        "selected_operation": operation,
        **public_values,
        "geometric_strategy": None,
        "evidence": None,
        "verdict": verdict,
        "private_field_audit": {
            "status": "passed",
            "checks": safety,
            "candidate_private_fields_excluded": sorted(
                set(selected) & _PRIVATE_CANDIDATE_FIELDS
            ),
        },
        "coverage_gaps": gaps,
        "provenance": {
            "source_trace_sha256": source["sha256"],
            "source_trace_size_bytes": source["size_bytes"],
            "source_trace_event_count": source["event_count"],
            "trace_event_refs": refs,
            "source_read_only": True,
            "expert_data_scanned": False,
            "external_services_called": False,
            "persistent_knowledge_written": False,
        },
    }
    private = {
        "profile": PRIVATE_PROFILE,
        "source_trace_path": source["path"],
        "source_trace_sha256": source["sha256"],
        "source_episode_ref": {
            "episode_id": episode.get("episode_id"),
            "seed": episode.get("seed"),
        },
        "selected_candidate_private_ref": selected.get("candidate_id"),
        "selected_instance_private_ref": selected.get("instance_id"),
        "selected_target_private_ref": selected.get("target_id"),
        "observe_audit": observe_audit.to_dict(),
        "trace_event_refs": refs,
    }
    return report, private


def _markdown(report: Mapping[str, Any]) -> str:
    gaps = report.get("coverage_gaps", [])
    gap_lines = [
        f"- `{item.get('component')}`: `{item.get('reason')}`"
        for item in gaps
        if isinstance(item, Mapping)
    ]
    return "\n".join(
        [
            "# HPK P0-A Observe-Only Real-Trace Report",
            "",
            "This report was produced offline from one explicitly selected existing Agent trace.",
            "No model, simulator, SAM, GPU, rollout, expert-directory scan, retrieval, or knowledge write occurred.",
            "",
            f"- report profile: `{report['report_profile']}`",
            f"- task family: `{report['task_family']}`",
            f"- selected operation: `{report['selected_operation']}`",
            f"- effect verdict: `{report['verdict']}`",
            "- complete EvidenceV1 emitted: `no`",
            "",
            "## Coverage gaps",
            "",
            *(gap_lines or ["- none"]),
            "",
            "The missing active typed geometric strategy is not replaced by a baseline sentinel.",
            "Private candidate/episode references are confined to `private_audit.json`.",
            "",
        ]
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def publish_report(
    output_root: Path,
    report: Mapping[str, Any],
    private: Mapping[str, Any],
) -> dict[str, Any]:
    eval_root = (PROJECT_ROOT / "eval_result").resolve()
    output = output_root.expanduser().resolve(strict=False)
    if output == eval_root or eval_root not in output.parents:
        raise ObserveTraceError(f"output must be a child of {eval_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as exc:
        raise ObserveTraceError(
            f"refusing to overwrite existing output: {output}"
        ) from exc
    try:
        _write_json(output / "observe_report.json", report)
        _write_json(output / "private_audit.json", private)
        (output / "report.md").write_text(_markdown(report), encoding="utf-8")
        files: dict[str, Any] = {}
        for name in ("observe_report.json", "private_audit.json", "report.md"):
            raw = (output / name).read_bytes()
            files[name] = {"sha256": _sha256_bytes(raw), "size_bytes": len(raw)}
        manifest = {
            "profile": REPORT_PROFILE,
            "files": files,
            "source_trace_sha256": report["provenance"]["source_trace_sha256"],
            "external_services_called": False,
            "persistent_knowledge_written": False,
        }
        _write_json(output / "manifest.json", manifest)
        return {
            "output_root": str(output),
            "manifest_sha256": _sha256_bytes((output / "manifest.json").read_bytes()),
            "coverage_gap_count": len(report.get("coverage_gaps", [])),
            "verdict": report.get("verdict"),
        }
    except BaseException:
        shutil.rmtree(output)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an offline HPK observe report from one explicit Agent trace.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--operation",
        choices=("contact", "grasp", "place"),
        default="place",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        events, source = _load_explicit_trace(args.trace)
        report, private = build_report(
            events,
            source,
            operation=args.operation,
        )
        result = publish_report(args.output_root, report, private)
    except (OSError, ObserveTraceError, ValueError, TypeError) as exc:
        print(f"HPK observe trace failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
