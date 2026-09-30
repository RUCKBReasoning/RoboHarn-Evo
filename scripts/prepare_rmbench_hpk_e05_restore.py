#!/usr/bin/env python3
"""Prepare an isolated empty HPK store for one RMBench E0.5 restore smoke.

This utility does not run a model or simulator.  It binds a new empty K0 to an
already recorded RMBench runtime provenance file and writes the exact evolving
HPK configuration fragment needed by the subsequent restore episode.
"""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from roboharn_evo.agent.hpk.evolving_runtime import EvolvingRuntimeProvenance
from roboharn_evo.resources import config_path
from roboharn_evo.agent.hpk.policy_config import (
    SAFE_EXPLORATION_GEOMETRY_POLICY_RAW_SHA256,
    load_policy_config,
    safe_exploration_geometry_policy_path,
)
from roboharn_evo.agent.hpk.promotion import PromotionPolicyV1
from roboharn_evo.agent.hpk.schemas import canonical_json_bytes
from roboharn_evo.agent.hpk.snapshot_publisher import publish_child_snapshot
from roboharn_evo.agent.hpk.updater import updater_policy_identity


DEV_PROMOTION_POLICY_RAW_SHA256 = (
    "972449f06850a8dac415ac9ffa9c53ae5e4a91010082ed53fcda5c3acd7ee756"
)


class E05PreparationError(RuntimeError):
    """The restore artifact could not be prepared without ambiguity."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_mapping(path: Path) -> tuple[bytes, Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise E05PreparationError(f"input must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise E05PreparationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E05PreparationError(f"JSON input must be an object: {path}")
    return raw, payload


def _runtime_provenance(path: Path) -> EvolvingRuntimeProvenance:
    raw, payload = _read_mapping(path)
    return EvolvingRuntimeProvenance.from_mapping(
        {
            "recorded": True,
            "schema": payload.get("schema"),
            "manifest_sha256": _sha256(raw),
            "effective_config_sha256": payload.get("effective_config_sha256"),
            "agent_service_identity_sha256": payload.get(
                "agent_service_identity_sha256"
            ),
            "runtime_tree_sha256": payload.get("runtime_tree_sha256"),
        }
    )


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(dict(payload)) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare(
    *,
    runtime_provenance_path: Path,
    snapshot_output_root: Path,
    config_output_path: Path,
    summary_output_path: Path,
    created_at: str,
) -> dict[str, Any]:
    for path in (snapshot_output_root, config_output_path, summary_output_path):
        if not path.is_absolute():
            raise E05PreparationError("all output paths must be absolute")
        if path.exists() or path.is_symlink():
            raise E05PreparationError(f"refusing to overwrite output: {path}")
    if (
        not config_output_path.parent.is_dir()
        or not summary_output_path.parent.is_dir()
    ):
        raise E05PreparationError("output parents must already exist")

    provenance = _runtime_provenance(runtime_provenance_path)
    geometry_path = safe_exploration_geometry_policy_path()
    promotion_path = config_path("hpk_promotion_policy_integration_dev_v1.yaml")
    geometry = load_policy_config(
        geometry_path,
        SAFE_EXPLORATION_GEOMETRY_POLICY_RAW_SHA256,
        "geometry",
    )
    loaded_promotion = load_policy_config(
        promotion_path,
        DEV_PROMOTION_POLICY_RAW_SHA256,
        "promotion",
    )
    promotion = PromotionPolicyV1.from_mapping(loaded_promotion.payload)
    if (
        promotion["development_only"] is not True
        or promotion["formal_evaluation_eligible"] is not False
        or promotion["allow_oracle_evidence"] is not False
        or promotion["allow_expert_prior"] is not False
    ):
        raise E05PreparationError("E0.5 requires the no-prior development policy")

    policy_refs = {
        "geometry": geometry.identity(),
        "promotion": {
            **loaded_promotion.identity(),
            "payload": loaded_promotion.payload,
        },
        "updater": updater_policy_identity(promotion),
    }
    root = publish_child_snapshot(
        parent=None,
        expected_parent_snapshot_id=None,
        expected_parent_manifest_sha256=None,
        entries=(),
        policy_refs=policy_refs,
        evidence_batch=None,
        source_episode_ids=(),
        created_at=created_at,
        destination_root=snapshot_output_root,
        runtime_source_identity=provenance.runtime_source_identity(),
        purpose="evolving_integration",
    )
    config = {
        "mode": "evolving",
        "snapshot_manifest": str(root.manifest_path),
        "expected_manifest_sha256": root.manifest_sha256,
        "run_scope": "integration",
        "max_prompt_chars": 4000,
        "allow_expert_prior": False,
        "allow_human_integration_prior": False,
        "allow_oracle_evidence": False,
        "all_hard_mismatch_behavior": "fail_closed",
        "geometry_policy_path": str(geometry_path),
        "expected_geometry_policy_sha256": (
            SAFE_EXPLORATION_GEOMETRY_POLICY_RAW_SHA256
        ),
        "promotion_policy_path": str(promotion_path),
        "expected_promotion_policy_sha256": DEV_PROMOTION_POLICY_RAW_SHA256,
        "snapshot_output_root": str(snapshot_output_root),
        "proposer": {
            "enabled": True,
            "max_proposals": 1,
            "trigger": "verified_oppose",
            "scope": "geometry",
        },
    }
    summary = {
        "schema": "roboharn_evo/hpk/rmbench_e05_restore_preparation/v1",
        "runtime_provenance_path": str(runtime_provenance_path),
        "runtime_provenance_sha256": provenance.manifest_sha256,
        "runtime_tree_sha256": provenance.runtime_tree_sha256,
        "k0_snapshot_id": root.snapshot_id,
        "k0_manifest_path": str(root.manifest_path),
        "k0_manifest_sha256": root.manifest_sha256,
        "hpk_config_path": str(config_output_path),
        "hpk_config_sha256": _sha256(canonical_json_bytes(config) + b"\n"),
        "snapshot_output_root": str(snapshot_output_root),
        "created_at": created_at,
        "information_access": {
            "oracle": False,
            "expert_prior": False,
            "human_integration_prior": False,
            "procedure_experience": "off",
        },
    }
    _write_once(config_output_path, config)
    _write_once(summary_output_path, summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    parser.add_argument("--snapshot-output-root", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare(
        runtime_provenance_path=args.runtime_provenance.resolve(strict=True),
        snapshot_output_root=args.snapshot_output_root.absolute(),
        config_output_path=args.config_output.absolute(),
        summary_output_path=args.summary_output.absolute(),
        created_at=args.created_at,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
