from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboharn_evo.agent.hpk.retriever import (  # noqa: E402
    build_hierarchical_physical_knowledge_context,
)
from roboharn_evo.agent.hpk.runtime_policy import HPKRuntimePolicy  # noqa: E402
from roboharn_evo.agent.hpk.schemas import (  # noqa: E402
    ABSTRACT_EFFECT_SCHEMA,
    CONDITION_ABSTRACTION_VERSION,
    CONDITION_SCHEMA,
    TASK_STRATEGY_SCHEMA,
    GEOMETRIC_STRATEGY_SCHEMA,
    ENTRY_SCHEMA,
    SNAPSHOT_SCHEMA,
    EntryV1,
    SnapshotManifestV1,
    canonical_json_bytes,
    entry_id_for,
    snapshot_id_for_manifest,
)
from roboharn_evo.agent.hpk.static_runtime import HPKStaticRuntime  # noqa: E402
from roboharn_evo.agent.operation_candidates import (  # noqa: E402
    operation_pose_candidates,
    select_operation_pose_candidate,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "eval_result" / "hpk" / "p0_c" / "cpu_static_integration"
)
EXPECTED_ARTIFACT_NAMES = frozenset(
    {
        "snapshot_manifest.json",
        "entries.jsonl",
        "off_result.json",
        "static_result.json",
        "public_usage_audit.json",
        "private_ranking_audit.json",
        "report.md",
    }
)

_CREATED_AT = "2026-08-21T00:00:00Z"
_TASK_FAMILY = "hpk_cpu_static_geometry_integration"
_DOMAIN_ID = "hpk_cpu_integration"
_MANIPULATED_ROLE = "held_object"
_TARGET_ROLE = "empty_support_region"
_PRIVATE_INSTANCE_REF = "cpu-held-object-private-ref"
_PRIVATE_TARGET_REF = "cpu-support-private-ref"
_BASELINE_CANDIDATE_REF = "cpu-baseline-private-ref"
_STATIC_CANDIDATE_REF = "cpu-static-private-ref"
_SELECTED_SKILL = "place-held-object-on-free-support"
_CURRENT_SUBGOAL = "place held object on current empty support region"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_line(payload: Any) -> bytes:
    return canonical_json_bytes(payload) + b"\n"


def _write_bytes(path: Path, payload: bytes, *, private: bool = False) -> None:
    path.write_bytes(payload)
    path.chmod(0o600 if private else 0o644)


def _fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target = {
        "target_id": _PRIVATE_TARGET_REF,
        "target_kind": "relational_reference_region",
        "target_role": _TARGET_ROLE,
        "target_reference_role": _TARGET_ROLE,
        "placement_relation": "center_of",
        "semantic_class": "support_surface",
        "geometry_class": "bounded_support_region",
        "support_valid": True,
        "free": True,
        "target_region_free": True,
        "geometry_source": "runtime_relational",
    }

    def candidate(
        *, private_ref: str, priority: int, source_index: int, reach: float
    ) -> dict[str, Any]:
        target_x = 0.46 if private_ref == _BASELINE_CANDIDATE_REF else 0.18
        return {
            "candidate_id": private_ref,
            "target_id": _PRIVATE_TARGET_REF,
            "target_kind": "relational_reference_region",
            "action_mode": "place",
            "arm": "right",
            "ee_target_pose": [target_x, 0.0, 0.16, 1.0, 0.0, 0.0, 0.0],
            "approach_pose": [target_x, 0.0, 0.26, 1.0, 0.0, 0.0, 0.0],
            "object_contact_pose": [
                target_x,
                0.0,
                0.13,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "tcp_pose": [target_x, 0.0, 0.16, 1.0, 0.0, 0.0, 0.0],
            "placement_relation": "center_of",
            "target_reference_role": _TARGET_ROLE,
            "reference_frame": "current_attachment",
            "approach_family": "clearance_first",
            "approach_direction_bucket": "above",
            "orientation_policy": "preserve_current_rigid_attachment",
            "orientation_relation": "preserve_current_attachment",
            "attachment_transform_source": (
                "verifier_confirmed_tcp_local_attachment"
            ),
            "holding_status": "verified",
            "grasp_transport_policy": "strict",
            "approach_clearance_m": 0.10,
            "reach_distance_m": reach,
            "geometry_source": "runtime_relational",
            "place_target_revalidated": True,
            "valid": True,
            "reachable_estimate": True,
            "support_valid": True,
            "free": True,
            "target_region_free": True,
            "occupied_by": [],
            "repeat_equivalent_failed_candidate": False,
            "priority": priority,
            "source_candidate_index": source_index,
        }

    candidates = [
        candidate(
            private_ref=_BASELINE_CANDIDATE_REF,
            priority=0,
            source_index=0,
            reach=0.40,
        ),
        candidate(
            private_ref=_STATIC_CANDIDATE_REF,
            priority=5,
            source_index=1,
            reach=0.10,
        ),
    ]
    instance = {
        "instance_id": _PRIVATE_INSTANCE_REF,
        "semantic_role": _MANIPULATED_ROLE,
        "semantic_class": "block",
        "geometry_class": "compact_rigid_object",
        "operation_pose_candidates": candidates,
    }
    scene = {
        "operation_targets": [target],
        "manipulation_state": {
            "right": {
                "phase": "temporary_buffer",
                "held_instance_id": _PRIVATE_INSTANCE_REF,
            }
        },
    }
    active_skill = {
        "skill_name": _SELECTED_SKILL,
        "instruction": _CURRENT_SUBGOAL,
        "preferred_arm": "right",
    }
    return scene, instance, active_skill


def _entry() -> EntryV1:
    payload: dict[str, Any] = {
        "schema": ENTRY_SCHEMA,
        "entry_id": "",
        "status": "accepted",
        "condition": {
            "schema": CONDITION_SCHEMA,
            "task_family": _TASK_FAMILY,
            "manipulation_phase": "temporary_buffer",
            "operation": "place",
            "manipulated_object": {
                "semantic_class": "block",
                "geometry_class": "compact_rigid_object",
                "role": _MANIPULATED_ROLE,
                "held_state": "held",
            },
            "target": {
                "semantic_class": "support_surface",
                "geometry_class": "bounded_support_region",
                "role": _TARGET_ROLE,
                "relation": "center_of",
            },
            "scene_predicates": ["support_valid", "target_region_free"],
            "preconditions": ["exactly_one_object_held"],
            "abstraction_version": CONDITION_ABSTRACTION_VERSION,
        },
        "task_strategy": {
            "schema": TASK_STRATEGY_SCHEMA,
            "operation": "place",
            "manipulated_role": _MANIPULATED_ROLE,
            "target_role": _TARGET_ROLE,
            "target_relation": "center_of",
            "manipulation_phase": "temporary_buffer",
            "subgoal_purpose": _CURRENT_SUBGOAL,
            "preferred_arm": "right",
            "source": {
                "planner_subtask_text": _CURRENT_SUBGOAL,
                "selected_skill": _SELECTED_SKILL,
                "action_mode": "place",
                "normalization_version": "task_strategy_normalizer/v1",
            },
        },
        "geometric_strategy": {
            "schema": GEOMETRIC_STRATEGY_SCHEMA,
            "strategy_family": "placement_relation",
            "reference_frame": "current_attachment",
            "target_relation": {
                "relation": "center_of",
                "reference_role": _TARGET_ROLE,
            },
            "approach": {
                "family": "clearance_first",
                "direction_bucket": "above",
            },
            "orientation": {"relation": "preserve_current_attachment"},
            "grasp": {"region": "unknown", "semantic_part": None},
            "hard_constraints": ["support_valid", "target_region_free"],
            "soft_preferences": ["lower_reach_distance"],
            "avoid": ["repeat_equivalent_failed_candidate"],
            "capability_evidence": {
                "semantic_part_observed": False,
                "geometry_source_class": "runtime_relational",
            },
        },
        "expected_effect": {
            "schema": ABSTRACT_EFFECT_SCHEMA,
            "effect_type": "place",
            "expected_predicates": [
                "gripper_empty",
                "object_released",
                "object_supported_by_target",
                "placement_stable",
                "target_relation_satisfied",
            ],
            "verifiability": "unverified",
        },
        "effect_statistics": {
            "support_count": 0,
            "oppose_count": 0,
            "unverified_count": 0,
            "posterior_alpha": 1.0,
            "posterior_beta": 1.0,
            "estimated_success_probability": 0.5,
            "lower_confidence_bound": 0.0,
            "last_updated_at": _CREATED_AT,
        },
        "evidence_refs": {"supporting": [], "opposing": [], "unverified": []},
        "promotion_policy_id": "hpk_static_integration_fixture/v1",
        "evaluation_status": "passed",
        "evaluation_ref": "afkeval_"
        + _sha256(
            b"HPK P0-C human-authored CPU integration fixture evaluation passed.\n"
        ),
        "provenance": {
            "source_kind": "human_authored",
            "expert_derived": False,
            "oracle_derived": False,
            "human_prior_used": True,
            "learned_hpk": False,
            "formal_evaluation_eligible": False,
            "domain_ids": [_DOMAIN_ID],
        },
        "acceptance_scope": "integration_only",
    }
    payload["entry_id"] = entry_id_for(payload)
    return EntryV1.from_dict(payload)


def _snapshot(entry: EntryV1) -> tuple[SnapshotManifestV1, bytes, bytes, str]:
    entry_bytes = _canonical_line(entry.to_dict())
    capabilities = {
        "task_families": [_TASK_FAMILY],
        "domain_ids": [_DOMAIN_ID],
        "operations": ["place"],
        "strategy_families": ["placement_relation"],
        "target_relations": ["center_of"],
        "hard_constraints": ["support_valid", "target_region_free"],
        "geometry_source_classes": ["runtime_relational"],
        "semantic_part_observed": False,
    }
    payload: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "schema_version": 1,
        "snapshot_id": "",
        "parent_snapshot_id": None,
        "created_at": _CREATED_AT,
        "purpose": "static_integration",
        "member": {
            "path": "entries.jsonl",
            "media_type": "application/x-ndjson",
            "sha256": _sha256(entry_bytes),
            "size_bytes": len(entry_bytes),
            "record_count": 1,
        },
        "accepted_entry_ids": [entry["entry_id"]],
        "capabilities": capabilities,
        "information_access_flags": {
            "expert_prior_present": False,
            "human_integration_prior_present": True,
            "oracle_derived_present": False,
            "learned_hpk_present": False,
            "all_entries_formal_evaluation_eligible": False,
        },
        "runtime_status": "static_ready",
        "immutable": True,
    }
    payload["snapshot_id"] = snapshot_id_for_manifest(payload)
    manifest = SnapshotManifestV1.from_dict(payload)
    manifest_bytes = _canonical_line(manifest.to_dict())
    return manifest, manifest_bytes, entry_bytes, _sha256(manifest_bytes)


def _eligible_candidates(instance: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in operation_pose_candidates(instance)
        if candidate["arm"] == "right"
        and candidate["action_mode"] == "place"
        and candidate["target_id"] == _PRIVATE_TARGET_REF
    ]


def _selection_is_legal(
    selection: dict[str, Any] | None,
    eligible: list[dict[str, Any]],
) -> bool:
    if selection is None or selection not in eligible:
        return False
    return bool(
        selection.get("target_id") == _PRIVATE_TARGET_REF
        and selection.get("place_target_revalidated") is True
        and selection.get("valid") is True
        and selection.get("reachable_estimate") is True
        and selection.get("support_valid") is True
        and selection.get("free") is True
        and not selection.get("occupied_by")
        and selection.get("attachment_transform_source")
        == "verifier_confirmed_tcp_local_attachment"
        and selection.get("holding_status") == "verified"
        and selection.get("grasp_transport_policy") == "strict"
    )


def _artifact_hashes(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: _sha256((root / name).read_bytes()) for name in names}


def _report(
    *,
    manifest: SnapshotManifestV1,
    manifest_sha256: str,
    member_sha256: str,
    entry: EntryV1,
    audit_id: str,
    scene_sha256: str,
    candidates_sha256: str,
    artifact_hashes: dict[str, str],
) -> bytes:
    hashes = "\n".join(
        f"- `{name}`: `{digest}`" for name, digest in sorted(artifact_hashes.items())
    )
    text = f"""# HPK P0-C CPU static integration report

Status: P0-C CPU static integration passed; real rollout gate remains pending.

This bundle is a deterministic, CPU-only integration fixture. It made no model
or service call, ran no simulator or robot motion, and ran no rollout. It proves
implementation wiring only; it is not evidence of task performance,
self-evolution, or learned-HPK effectiveness.

## Provenance and scope

- `source_kind=human_authored`
- `acceptance_scope=integration_only`
- `human_prior_used=true`
- `learned_hpk=false`
- `formal_evaluation_eligible=false`
- expert/oracle derivation: false
- evidence counts and evidence-reference arrays: zero/empty
- planner-context confidence: `0.0`

The accepted entry is `{entry['entry_id']}`. Its passed evaluation reference is
an opaque digest of a fixed fixture acceptance statement, not rollout evidence.

## Before/after proof

The off and static runs used the same immutable in-memory scene
(`{scene_sha256}`) and eligible candidate set (`{candidates_sha256}`). Off used
the existing selector with `z=None` and performed no snapshot load, audit, or
HPK behavior change. Static explicitly loaded the one accepted snapshot once,
retrieved one exact entry after target binding, passed its typed geometric
strategy to the existing selector, and selected a different candidate. Both
selections remained legal, reachable, target-bound place candidates.

The public audit `{audit_id}` records only the boolean geometry behavior change
and contains no candidate identity. Candidate before/after ranks and the
selected private reference are confined to the validated private ranking audit.

## Snapshot integrity

- snapshot ID: `{manifest['snapshot_id']}`
- exact manifest SHA-256 before/after: `{manifest_sha256}` / `{manifest_sha256}`
- exact member SHA-256 before/after: `{member_sha256}` / `{member_sha256}`
- manifest immutable: true
- runtime load count: 1

The exact snapshot bytes were unchanged across runtime use.

## Artifact SHA-256

{hashes}
"""
    return text.encode("utf-8")


def build_cpu_static_integration(output_dir: Path | str) -> dict[str, Any]:
    """Build and atomically publish one deterministic no-clobber proof bundle."""

    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to clobber existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    temporary.chmod(0o700)

    try:
        scene, instance, active_skill = _fixture()
        eligible = _eligible_candidates(instance)
        if len(eligible) != 2:
            raise RuntimeError("CPU fixture must expose exactly two eligible candidates")
        scene_bytes_before = canonical_json_bytes(scene)
        candidates_bytes_before = canonical_json_bytes(eligible)
        scene_sha256 = _sha256(scene_bytes_before)
        candidates_sha256 = _sha256(candidates_bytes_before)

        # The inactive path deliberately carries an invalid stale path.  The
        # off policy ignores it, and z=None keeps geometry state/auditing inert.
        off_policy = HPKRuntimePolicy.from_mapping(
            {
                "mode": "off",
                "snapshot_manifest": str(temporary / "must-not-be-read.json"),
                "expected_manifest_sha256": "not-a-digest",
            }
        )
        if off_policy.retrieves or off_policy.changes_candidate_selection:
            raise RuntimeError("off policy unexpectedly activated HPK")
        off_selected = select_operation_pose_candidate(
            instance,
            arm="right",
            action_mode="place",
            requested_target_id=_PRIVATE_TARGET_REF,
            geometric_strategy=None,
            ranking_audit=None,
            geometry_scene_state=object(),
        )
        if off_selected is None or off_selected.get("candidate_id") != (
            _BASELINE_CANDIDATE_REF
        ):
            raise RuntimeError("off selector did not preserve baseline selection")
        if not _selection_is_legal(off_selected, eligible):
            raise RuntimeError("off selector produced an illegal candidate")

        entry = _entry()
        manifest, manifest_bytes, member_bytes, manifest_sha256 = _snapshot(entry)
        _write_bytes(temporary / "entries.jsonl", member_bytes)
        _write_bytes(temporary / "snapshot_manifest.json", manifest_bytes)
        member_sha256 = _sha256(member_bytes)
        integrity_before = {
            "manifest_sha256": _sha256(
                (temporary / "snapshot_manifest.json").read_bytes()
            ),
            "member_sha256": _sha256((temporary / "entries.jsonl").read_bytes()),
        }

        policy = HPKRuntimePolicy.from_mapping(
            {
                "mode": "static",
                "snapshot_manifest": str(
                    (temporary / "snapshot_manifest.json").resolve()
                ),
                "expected_manifest_sha256": manifest_sha256,
                "run_scope": "integration",
                "max_prompt_chars": 4000,
                "allow_expert_prior": False,
                "allow_human_integration_prior": True,
                "allow_oracle_evidence": False,
                "all_hard_mismatch_behavior": "fail_closed",
            }
        )
        runtime = HPKStaticRuntime.from_policy(
            policy,
            task_family=_TASK_FAMILY,
            domain_id=_DOMAIN_ID,
            current_capabilities=manifest["capabilities"],
        )
        if runtime.load_count != 1:
            raise RuntimeError("static runtime did not load the snapshot exactly once")
        query = runtime.build_bound_geometry_query(
            scene_memory=scene,
            instance=instance,
            action_mode="place",
            arm="right",
            requested_target_id=_PRIVATE_TARGET_REF,
            active_skill=active_skill,
        )
        decision = runtime.retrieve_geometry(query)
        if decision.retrieval.selected_entry_id != entry["entry_id"]:
            raise RuntimeError("static runtime did not retrieve the exact fixture entry")
        if decision.retrieval.geometric_strategy is None:
            raise RuntimeError("exact match did not yield a typed geometric strategy")

        sink = runtime.new_private_ranking_sink(decision)
        static_selected = select_operation_pose_candidate(
            instance,
            arm="right",
            action_mode="place",
            requested_target_id=_PRIVATE_TARGET_REF,
            geometric_strategy=decision.retrieval.geometric_strategy,
            ranking_audit=sink,
            geometry_scene_state=scene,
        )
        finalized = runtime.finalize_geometry_ranking(decision, sink)
        if finalized.audit is None or not sink.finalized:
            raise RuntimeError("static selection was not fully audited")
        if static_selected is None or static_selected.get("candidate_id") != (
            _STATIC_CANDIDATE_REF
        ):
            raise RuntimeError("static strategy did not change the expected selection")
        if static_selected.get("candidate_id") == off_selected.get("candidate_id"):
            raise RuntimeError("static selection did not differ from the off baseline")
        if not _selection_is_legal(static_selected, eligible):
            raise RuntimeError("static strategy selected an illegal candidate")
        if sink.geometric_compliance is not True or not sink.behavior_changed:
            raise RuntimeError("private audit did not prove a compliant behavior change")

        context = build_hierarchical_physical_knowledge_context(
            decision.retrieval.selected_entry
        )
        if context["confidence"] != 0.0:
            raise RuntimeError("human-authored fixture context confidence must be zero")
        trace_binding = runtime.geometry_trace_binding(finalized, sink)
        integrity_after = {
            "manifest_sha256": _sha256(
                (temporary / "snapshot_manifest.json").read_bytes()
            ),
            "member_sha256": _sha256((temporary / "entries.jsonl").read_bytes()),
        }
        if integrity_before != integrity_after:
            raise RuntimeError("snapshot bytes changed during static runtime use")
        if canonical_json_bytes(scene) != scene_bytes_before:
            raise RuntimeError("runtime mutated the fixture scene")
        if canonical_json_bytes(_eligible_candidates(instance)) != (
            candidates_bytes_before
        ):
            raise RuntimeError("runtime mutated the eligible candidate set")

        integrity = {
            "manifest_sha256_before": integrity_before["manifest_sha256"],
            "manifest_sha256_after": integrity_after["manifest_sha256"],
            "member_sha256_before": integrity_before["member_sha256"],
            "member_sha256_after": integrity_after["member_sha256"],
            "unchanged": integrity_before == integrity_after,
        }
        shared_fixture = {
            "scene_sha256": scene_sha256,
            "eligible_candidate_set_sha256": candidates_sha256,
            "eligible_candidate_count": len(eligible),
            "requested_target_private_ref": _PRIVATE_TARGET_REF,
        }
        off_result = {
            "artifact": "off_result",
            "mode": "off",
            "privacy": "private_cpu_integration_result",
            "fixture": shared_fixture,
            "selected_candidate_private_ref": off_selected["candidate_id"],
            "selected_candidate_legal": True,
            "target_binding_preserved": True,
            "hpk_static_runtime_constructed": False,
            "hpk_runtime_snapshot_io": False,
            "public_usage_audit_emitted": False,
            "private_ranking_audit_emitted": False,
            "harness_snapshot_integrity": integrity,
        }
        static_result = {
            "artifact": "static_result",
            "mode": "static",
            "privacy": "private_cpu_integration_result",
            "fixture": shared_fixture,
            "snapshot_id": manifest["snapshot_id"],
            "snapshot_manifest_sha256": manifest_sha256,
            "snapshot_member_sha256": member_sha256,
            "snapshot_load_count": runtime.load_count,
            "retrieved_entry_id": entry["entry_id"],
            "selected_geometric_strategy_id": (
                decision.retrieval.geometric_strategy.stable_id
            ),
            "selected_candidate_private_ref": static_selected["candidate_id"],
            "baseline_selected_candidate_private_ref": off_selected["candidate_id"],
            "selection_changed": True,
            "selected_candidate_legal": True,
            "target_binding_preserved": True,
            "geometric_compliance": sink.geometric_compliance,
            "scoring_profile": finalized.audit["scoring_profile"],
            "planner_context_injected": False,
            "planner_context_confidence": context["confidence"],
            "source_disclosure": context["source_disclosure"],
            "evidence_counts": {
                "supporting": 0,
                "opposing": 0,
                "unverified": 0,
            },
            "snapshot_integrity": integrity,
            "trace_binding": trace_binding,
        }

        _write_bytes(temporary / "off_result.json", _canonical_line(off_result))
        _write_bytes(
            temporary / "static_result.json", _canonical_line(static_result)
        )
        _write_bytes(
            temporary / "public_usage_audit.json",
            _canonical_line(finalized.audit.to_dict()),
        )
        _write_bytes(
            temporary / "private_ranking_audit.json",
            sink.to_canonical_json_line(),
            private=True,
        )

        hashes_before_report = _artifact_hashes(
            temporary,
            (
                "entries.jsonl",
                "off_result.json",
                "private_ranking_audit.json",
                "public_usage_audit.json",
                "snapshot_manifest.json",
                "static_result.json",
            ),
        )
        _write_bytes(
            temporary / "report.md",
            _report(
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                member_sha256=member_sha256,
                entry=entry,
                audit_id=finalized.audit.stable_id,
                scene_sha256=scene_sha256,
                candidates_sha256=candidates_sha256,
                artifact_hashes=hashes_before_report,
            ),
        )
        names = frozenset(path.name for path in temporary.iterdir())
        if names != EXPECTED_ARTIFACT_NAMES:
            raise RuntimeError(
                "unexpected CPU integration artifact set: "
                + ", ".join(sorted(names))
            )
        temporary.chmod(0o755)
        temporary.rename(output)
        final_hashes = _artifact_hashes(output, tuple(sorted(EXPECTED_ARTIFACT_NAMES)))
        return {
            "output_dir": str(output),
            "snapshot_id": manifest["snapshot_id"],
            "entry_id": entry["entry_id"],
            "manifest_sha256": manifest_sha256,
            "member_sha256": member_sha256,
            "artifact_sha256": final_hashes,
        }
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic HPK P0-C CPU integration fixture.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"new no-clobber output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    result = build_cpu_static_integration(_parse_args().output_dir)
    print(canonical_json_bytes(result).decode("utf-8"))


if __name__ == "__main__":
    main()
