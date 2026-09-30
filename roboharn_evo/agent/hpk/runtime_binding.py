"""Content-bound runtime identity carried by evolving rollout inputs."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from roboharn_evo.agent.hpk.promotion import PromotionPolicyV1
from roboharn_evo.agent.hpk.policy_config import EVOLVING_GEOMETRY_POLICY_IDENTITIES
from roboharn_evo.agent.hpk.schemas import (
    canonical_json_bytes,
    stable_content_id,
    validate_content_id,
)
from roboharn_evo.agent.hpk.updater import updater_policy_identity


HPK_RUNTIME_BINDING_SCHEMA = "roboharn_evo/hpk/runtime_binding/v1"
HPK_SEQUENTIAL_RUNTIME_BINDING_SCHEMA = "roboharn_evo/hpk/runtime_binding/v2"
HPK_SEQUENTIAL_RUN_BINDING_SCHEMA = "roboharn_evo/hpk/sequential_run_binding/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HPKRuntimeBindingError(ValueError):
    pass


def _fail(message: str) -> None:
    raise HPKRuntimeBindingError(message)


def _sha(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{path} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object")
    return copy.deepcopy(dict(value))


def _exact(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    if set(value) != expected:
        _fail(
            f"{path} fields mismatch: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def validate_sequential_run_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    run_binding = _mapping(value, path="runtime_binding.run_binding")
    _exact(
        run_binding,
        {
            "schema",
            "run_binding_id",
            "run_id",
            "experiment_contract_sha256",
            "output_root_sha256",
            "claim_path_sha256",
        },
        path="runtime_binding.run_binding",
    )
    if run_binding["schema"] not in {
        HPK_SEQUENTIAL_RUN_BINDING_SCHEMA, "tcm/afk/sequential_run_binding/v1"
    }:
        _fail("runtime_binding.run_binding.schema is unsupported")
    validate_content_id(
        run_binding["run_binding_id"],
        prefix="afkrunbinding",
        path="runtime_binding.run_binding.run_binding_id",
    )
    validate_content_id(
        run_binding["run_id"],
        prefix="afkrun",
        path="runtime_binding.run_binding.run_id",
    )
    for key in (
        "experiment_contract_sha256",
        "output_root_sha256",
        "claim_path_sha256",
    ):
        _sha(run_binding[key], path=f"runtime_binding.run_binding.{key}")
    run_identity = copy.deepcopy(run_binding)
    run_binding_id = run_identity.pop("run_binding_id")
    if run_binding_id != stable_content_id("afkrunbinding", run_identity):
        _fail("runtime_binding.run_binding.run_binding_id content mismatch")
    return copy.deepcopy(run_binding)


def validate_runtime_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    binding = _mapping(value, path="runtime_binding")
    schema = binding.get("schema")
    expected_fields = {
        "schema",
        "binding_id",
        "snapshot_ref",
        "policy_refs",
        "provenance",
    }
    if schema in {HPK_SEQUENTIAL_RUNTIME_BINDING_SCHEMA, "tcm/afk/runtime_binding/v2"}:
        expected_fields.add("run_binding")
    _exact(
        binding,
        expected_fields,
        path="runtime_binding",
    )
    if schema not in {
        HPK_RUNTIME_BINDING_SCHEMA,
        HPK_SEQUENTIAL_RUNTIME_BINDING_SCHEMA,
        "tcm/afk/runtime_binding/v1",
        "tcm/afk/runtime_binding/v2",
    }:
        _fail("runtime_binding.schema is unsupported")
    validate_content_id(
        binding["binding_id"], prefix="afkruntime", path="runtime_binding.binding_id"
    )
    snapshot = _mapping(binding["snapshot_ref"], path="runtime_binding.snapshot_ref")
    _exact(snapshot, {"snapshot_id", "manifest_sha256"}, path="snapshot_ref")
    validate_content_id(
        snapshot["snapshot_id"], prefix="afksnap", path="snapshot_ref.snapshot_id"
    )
    _sha(snapshot["manifest_sha256"], path="snapshot_ref.manifest_sha256")

    policies = _mapping(binding["policy_refs"], path="runtime_binding.policy_refs")
    _exact(policies, {"geometry", "promotion", "updater"}, path="policy_refs")
    geometry = _mapping(policies["geometry"], path="policy_refs.geometry")
    _exact(geometry, {"policy_id", "config_sha256"}, path="policy_refs.geometry")
    if not isinstance(geometry["policy_id"], str) or not geometry["policy_id"].strip():
        _fail("policy_refs.geometry.policy_id is invalid")
    geometry_sha = _sha(
        geometry["config_sha256"], path="policy_refs.geometry.config_sha256"
    )
    if (geometry["policy_id"], geometry_sha) not in (
        EVOLVING_GEOMETRY_POLICY_IDENTITIES
    ):
        _fail("evolving runtime binding requires exact geometry policy v2 or v3")
    promotion_ref = _mapping(policies["promotion"], path="policy_refs.promotion")
    _exact(
        promotion_ref,
        {"policy_id", "config_sha256", "payload"},
        path="policy_refs.promotion",
    )
    try:
        promotion = PromotionPolicyV1.from_mapping(
            _mapping(promotion_ref["payload"], path="policy_refs.promotion.payload")
        )
    except Exception as exc:
        raise HPKRuntimeBindingError(
            f"policy_refs.promotion.payload is invalid: {exc}"
        ) from exc
    if (
        promotion_ref["policy_id"] != promotion.policy_id
        or promotion_ref["config_sha256"] != promotion.config_sha256
    ):
        _fail("policy_refs.promotion identity disagrees with its payload")
    if policies["updater"] != updater_policy_identity(
        promotion, legacy=policies["updater"]["policy_id"] == "afk_updater/v1"
    ):
        _fail("policy_refs.updater disagrees with the embedded promotion policy")

    provenance = _mapping(binding["provenance"], path="runtime_binding.provenance")
    _exact(
        provenance,
        {
            "schema",
            "manifest_sha256",
            "config_sha256",
            "model_identity_sha256",
            "runtime_source_sha256",
            "evidence_runtime_schema",
        },
        path="runtime_binding.provenance",
    )
    for key in (
        "schema",
        "evidence_runtime_schema",
    ):
        if not isinstance(provenance[key], str) or not provenance[key].strip():
            _fail(f"runtime_binding.provenance.{key} must be non-empty")
    for key in (
        "manifest_sha256",
        "config_sha256",
        "model_identity_sha256",
        "runtime_source_sha256",
    ):
        _sha(provenance[key], path=f"runtime_binding.provenance.{key}")
    if schema in {HPK_SEQUENTIAL_RUNTIME_BINDING_SCHEMA, "tcm/afk/runtime_binding/v2"}:
        validate_sequential_run_binding(binding["run_binding"])
    identity = copy.deepcopy(binding)
    binding_id = identity.pop("binding_id")
    if binding_id != stable_content_id("afkruntime", identity):
        _fail("runtime_binding.binding_id content mismatch")
    return copy.deepcopy(binding)


def build_runtime_binding(
    *,
    snapshot_id: str,
    snapshot_manifest_sha256: str,
    policy_refs: Mapping[str, Any],
    provenance_schema: str,
    provenance_manifest_sha256: str,
    config_sha256: str,
    model_identity_sha256: str,
    runtime_source_sha256: str,
    evidence_runtime_schema: str,
    run_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": (
            HPK_RUNTIME_BINDING_SCHEMA
            if run_binding is None
            else HPK_SEQUENTIAL_RUNTIME_BINDING_SCHEMA
        ),
        "snapshot_ref": {
            "snapshot_id": snapshot_id,
            "manifest_sha256": snapshot_manifest_sha256,
        },
        "policy_refs": copy.deepcopy(dict(policy_refs)),
        "provenance": {
            "schema": provenance_schema,
            "manifest_sha256": provenance_manifest_sha256,
            "config_sha256": config_sha256,
            "model_identity_sha256": model_identity_sha256,
            "runtime_source_sha256": runtime_source_sha256,
            "evidence_runtime_schema": evidence_runtime_schema,
        },
    }
    if run_binding is not None:
        payload["run_binding"] = copy.deepcopy(dict(run_binding))
    payload["binding_id"] = stable_content_id("afkruntime", payload)
    validated = validate_runtime_binding(payload)
    if canonical_json_bytes(validated) != canonical_json_bytes(payload):
        _fail("runtime binding normalization changed its canonical bytes")
    return validated


def build_sequential_run_binding(
    *,
    run_id: str,
    experiment_contract_sha256: str,
    output_root_sha256: str,
    claim_path_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": HPK_SEQUENTIAL_RUN_BINDING_SCHEMA,
        "run_id": run_id,
        "experiment_contract_sha256": experiment_contract_sha256,
        "output_root_sha256": output_root_sha256,
        "claim_path_sha256": claim_path_sha256,
    }
    payload["run_binding_id"] = stable_content_id("afkrunbinding", payload)
    # Reuse the full validator with a minimal structurally valid wrapper is
    # intentionally avoided; each field is validated here and the wrapper is
    # content-validated when build_runtime_binding is called.
    validate_content_id(run_id, prefix="afkrun", path="run_id")
    for key in (
        "experiment_contract_sha256",
        "output_root_sha256",
        "claim_path_sha256",
    ):
        _sha(payload[key], path=key)
    return copy.deepcopy(payload)


__all__ = [
    "HPK_RUNTIME_BINDING_SCHEMA",
    "HPK_SEQUENTIAL_RUN_BINDING_SCHEMA",
    "HPK_SEQUENTIAL_RUNTIME_BINDING_SCHEMA",
    "HPKRuntimeBindingError",
    "build_runtime_binding",
    "build_sequential_run_binding",
    "validate_runtime_binding",
    "validate_sequential_run_binding",
]
