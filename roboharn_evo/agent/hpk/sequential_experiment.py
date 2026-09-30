"""Preregistered two-episode HPK self-evolution experiment harness.

The harness is deliberately independent from RMBench's legacy multi-episode
loop.  It owns one ordered ``update -> probe`` schedule, one worker, one local
quota ledger, and one immutable runtime-binding template.  The caller supplies
an executor, but every external call and artifact write must pass through the
metered adapter and the executor's provider audit must reconcile byte-for-byte
with the local ledger.

This module performs no service, simulator, GPU, tmux, or rollout work on
import.  It also contains no task-, seed-, object-, candidate-, coordinate-,
or arm-specific branch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from roboharn_evo.agent.hpk.evolving_store import load_evolving_snapshot
from roboharn_evo.agent.hpk.policy_config import (
    SAFE_EXPLORATION_GEOMETRY_POLICY_IDENTITIES,
)
from roboharn_evo.agent.hpk.rollout_importer import AgentRolloutImporter
from roboharn_evo.agent.hpk.runtime_binding import (
    HPK_SEQUENTIAL_RUNTIME_BINDING_SCHEMA,
    build_runtime_binding,
    build_sequential_run_binding,
    validate_runtime_binding,
)
from roboharn_evo.agent.hpk.schemas import (
    AuditV1,
    canonical_json_bytes,
    stable_content_id,
    validate_content_id,
)
from roboharn_evo.agent.hpk.semantic_knowledge import validate_semantic_knowledge
from roboharn_evo.resources import config_path

PREREGISTRATION_SCHEMA = "roboharn_evo/hpk/p0de_sequential_preregistration/v1"
RUN_RESULT_SCHEMA = "roboharn_evo/hpk/p0de_sequential_run_result/v1"
EPISODE_ROLES = ("update", "probe")
ACCEPTANCE_SCOPES = frozenset({"integration_development", "formal_no_prior"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_PREREGISTRATION_BYTES = 1024 * 1024
_ACCEPTANCE_PROFILE_SCHEMA = "roboharn_evo/hpk/p0de_sequential_acceptance_profile/v1"
_ACCEPTANCE_PROFILE_ID = "hpk_p0de_sequential_acceptance/v2"
_ACCEPTANCE_PROFILE_RAW_SHA256 = (
    "0ddd81e6a92c970186377e9fc51176c28579dd96a4ff6538b49f4df12c984ef2"
)
_ACCEPTANCE_PROFILE_CONFIG_SHA256 = (
    "d725fca81c7b461d323edbca079976ae4c8fd1aaa4769312b641819aac55a00e"
)
_EXPERIMENT_CONTRACT_SCHEMA = "roboharn_evo/hpk/sequential_experiment_contract/v1"


class SequentialExperimentError(RuntimeError):
    """A preregistration, schedule, quota, or child barrier failed closed."""


class SequentialQuotaExceeded(SequentialExperimentError):
    """A local resource reservation would exceed the preregistered cap."""


def _fail(message: str) -> None:
    raise SequentialExperimentError(message)


def _exact(value: Mapping[str, Any], fields: set[str], *, path: str) -> None:
    actual = set(value)
    if actual != fields:
        _fail(
            f"{path} fields mismatch: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object")
    try:
        return json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    except Exception as exc:
        raise SequentialExperimentError(
            f"{path} must be finite canonical JSON: {exc}"
        ) from exc


def _string(value: Any, *, path: str, max_chars: int = 8192) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_chars:
        _fail(f"{path} must be a non-empty string of at most {max_chars} characters")
    return value


def _sha(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{path} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{path} must be a non-negative integer")
    return value


def _positive_int(value: Any, *, path: str) -> int:
    result = _nonnegative_int(value, path=path)
    if result == 0:
        _fail(f"{path} must be a positive integer")
    return result


def _path_identity_sha256(value: str) -> str:
    return hashlib.sha256(str(Path(value).absolute()).encode("utf-8")).hexdigest()


def _experiment_contract_sha256(
    payload: Mapping[str, Any],
    *,
    runtime_binding: Mapping[str, Any],
) -> str:
    knowledge_field = "hpk" if "hpk" in payload else "afk"
    hpk = _mapping(payload[knowledge_field], path="hpk")
    binding = validate_runtime_binding(runtime_binding)
    contract = {
        "schema": (
            "tcm/afk/sequential_experiment_contract/v1"
            if payload.get("schema") == "tcm/afk/p0de_sequential_preregistration/v1"
            else _EXPERIMENT_CONTRACT_SCHEMA
        ),
        "acceptance_profile": copy.deepcopy(payload["acceptance_profile"]),
        "task": copy.deepcopy(payload["task"]),
        "launch_config": copy.deepcopy(payload["launch_config"]),
        "episodes": copy.deepcopy(payload["episodes"]),
        "execution": copy.deepcopy(payload["execution"]),
        knowledge_field: {
            "mode": hpk["mode"],
            "scope": copy.deepcopy(hpk["scope"]),
            "initial_snapshot": copy.deepcopy(hpk["initial_snapshot"]),
            "procedure_experience_mode": hpk["procedure_experience_mode"],
            "allow_oracle_evidence": hpk["allow_oracle_evidence"],
            "allow_expert_prior": hpk["allow_expert_prior"],
            "snapshot_ref": copy.deepcopy(binding["snapshot_ref"]),
            "policy_refs": copy.deepcopy(binding["policy_refs"]),
            "provenance": copy.deepcopy(binding["provenance"]),
        },
        "manipulation_policy": copy.deepcopy(payload["manipulation_policy"]),
        "budgets": copy.deepcopy(payload["budgets"]),
        "planned_usage": copy.deepcopy(payload["planned_usage"]),
        "run_id": payload["run_id"],
        "output_root_sha256": _path_identity_sha256(str(payload["output_root"])),
        "snapshot_output_root_sha256": _path_identity_sha256(
            str(payload["snapshot_output_root"])
        ),
        "segmentation_artifact_root_sha256": _path_identity_sha256(
            str(payload["segmentation_artifact_root"])
        ),
        "claim_path_sha256": _path_identity_sha256(str(payload["run_claim_path"])),
    }
    return hashlib.sha256(canonical_json_bytes(contract)).hexdigest()


def _absolute_path(value: Any, *, path: str) -> str:
    raw = _string(value, path=path)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        _fail(f"{path} must be absolute")
    return str(candidate.absolute())


def _snapshot_identity(value: Mapping[str, Any], *, path: str) -> dict[str, str]:
    snapshot = _mapping(value, path=path)
    _exact(snapshot, {"snapshot_id", "manifest_sha256"}, path=path)
    try:
        validate_content_id(
            snapshot["snapshot_id"], prefix="afksnap", path=f"{path}.snapshot_id"
        )
    except Exception as exc:
        raise SequentialExperimentError(str(exc)) from exc
    _sha(snapshot["manifest_sha256"], path=f"{path}.manifest_sha256")
    return snapshot


def _entry_ids(value: Any, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(f"{path} must be a sorted unique list")
    result: list[str] = []
    for index, item in enumerate(value):
        try:
            validate_content_id(
                item,
                prefix="afkentry",
                path=f"{path}[{index}]",
            )
        except Exception as exc:
            raise SequentialExperimentError(str(exc)) from exc
        result.append(str(item))
    if result != sorted(set(result)):
        _fail(f"{path} must be sorted and unique")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SnapshotState:
    manifest_path: str
    snapshot_id: str
    manifest_sha256: str
    accepted_entry_ids: tuple[str, ...]

    def identity(self) -> dict[str, str]:
        return {
            "snapshot_id": self.snapshot_id,
            "manifest_sha256": self.manifest_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            **self.identity(),
            "accepted_entry_ids": list(self.accepted_entry_ids),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, path: str) -> SnapshotState:
        payload = _mapping(value, path=path)
        _exact(
            payload,
            {
                "manifest_path",
                "snapshot_id",
                "manifest_sha256",
                "accepted_entry_ids",
            },
            path=path,
        )
        identity = _snapshot_identity(
            {
                "snapshot_id": payload["snapshot_id"],
                "manifest_sha256": payload["manifest_sha256"],
            },
            path=path,
        )
        return cls(
            manifest_path=_absolute_path(
                payload["manifest_path"], path=f"{path}.manifest_path"
            ),
            snapshot_id=identity["snapshot_id"],
            manifest_sha256=identity["manifest_sha256"],
            accepted_entry_ids=_entry_ids(
                payload["accepted_entry_ids"], path=f"{path}.accepted_entry_ids"
            ),
        )

    @classmethod
    def from_loaded(cls, value: Any) -> SnapshotState:
        return cls(
            manifest_path=str(value.manifest_path),
            snapshot_id=str(value.snapshot_id),
            manifest_sha256=str(value.manifest_sha256),
            accepted_entry_ids=tuple(
                sorted(str(entry["entry_id"]) for entry in value.accepted_entries)
            ),
        )


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: str
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, path: str) -> ArtifactRef:
        payload = _mapping(value, path=path)
        _exact(payload, {"path", "sha256"}, path=path)
        return cls(
            path=_absolute_path(payload["path"], path=f"{path}.path"),
            sha256=_sha(payload["sha256"], path=f"{path}.sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    role: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    external_model_calls: int = 0
    images: int = 0
    image_bytes: int = 0
    artifact_bytes: int = 0

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            _nonnegative_int(getattr(self, field_name), path=field_name)

    def to_dict(self) -> dict[str, int]:
        return {
            field_name: int(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
        }

    def minus(self, other: ResourceUsage) -> ResourceUsage:
        values = {
            field_name: getattr(self, field_name) - getattr(other, field_name)
            for field_name in self.__dataclass_fields__
        }
        if any(value < 0 for value in values.values()):
            _fail("quota usage regressed")
        return ResourceUsage(**values)


@dataclass(frozen=True, slots=True)
class ControlUsage:
    environment_actions: int
    control_turns: int
    no_progress_control_turns: int
    semantic_rounds: int
    backend_retries: int = 0
    seed_retries: int = 0

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            _nonnegative_int(getattr(self, field_name), path=field_name)

    def to_dict(self) -> dict[str, int]:
        return {
            field_name: int(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class AcceptanceProfile:
    """Repository-frozen ceilings, distinct from a run's tighter authorization.

    ``external_model_calls`` conservatively counts every outbound model/service
    invocation, including both local service health preflights. ``images``
    counts outbound image payload occurrences, not unique source frames.
    ``image_bytes`` sums the
    corresponding encoded outbound payload bytes. ``artifact_bytes`` sums all
    bytes the executor creates for the experiment after preregistration input
    validation. Wall time is monotonic elapsed time under the active quota gate.
    Every actual preregistration must choose tighter explicit caps beneath these
    ceilings and its pure static plan must fit those tighter caps.
    """

    payload: dict[str, Any]
    raw_sha256: str
    config_sha256: str

    @property
    def schedule(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload["schedule"])

    @property
    def per_episode(self) -> dict[str, int]:
        return copy.deepcopy(self.payload["per_episode"])

    @property
    def hard_resource_ceilings(self) -> dict[str, int]:
        return copy.deepcopy(self.payload["hard_resource_ceilings"])

    def identity(self) -> dict[str, str]:
        return {
            "profile_id": self.payload["profile_id"],
            "raw_sha256": self.raw_sha256,
            "config_sha256": self.config_sha256,
        }


def load_acceptance_profile() -> AcceptanceProfile:
    profile_path = config_path("hpk_p0de_sequential_acceptance_profile_v2.json")
    raw = _read_regular_file(
        profile_path,
        label="sequential acceptance profile",
        max_bytes=_MAX_PREREGISTRATION_BYTES,
    )
    raw_sha = hashlib.sha256(raw).hexdigest()
    if raw_sha != _ACCEPTANCE_PROFILE_RAW_SHA256:
        _fail("sequential acceptance profile raw SHA-256 mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SequentialExperimentError(
            f"sequential acceptance profile is invalid JSON: {exc}"
        ) from exc
    profile = _mapping(payload, path="acceptance_profile.payload")
    _exact(
        profile,
        {
            "schema",
            "profile_id",
            "schedule",
            "per_episode",
            "hard_resource_ceilings",
        },
        path="acceptance_profile.payload",
    )
    if (
        profile["schema"] != _ACCEPTANCE_PROFILE_SCHEMA
        or profile["profile_id"] != _ACCEPTANCE_PROFILE_ID
    ):
        _fail("sequential acceptance profile identity is unsupported")
    if canonical_json_bytes(profile) + b"\n" != raw:
        _fail("sequential acceptance profile is not canonical JSON plus one LF")
    config_sha = hashlib.sha256(canonical_json_bytes(profile)).hexdigest()
    if config_sha != _ACCEPTANCE_PROFILE_CONFIG_SHA256:
        _fail("sequential acceptance profile config SHA-256 mismatch")
    schedule = _mapping(profile["schedule"], path="acceptance_profile.schedule")
    _exact(
        schedule,
        {
            "worker_count",
            "episode_count",
            "seed_replacement",
            "seed_retry_count",
            "backend_retry_count",
        },
        path="acceptance_profile.schedule",
    )
    if schedule != {
        "worker_count": 1,
        "episode_count": 2,
        "seed_replacement": False,
        "seed_retry_count": 0,
        "backend_retry_count": 0,
    }:
        _fail("sequential acceptance profile schedule is invalid")
    per_episode = _mapping(
        profile["per_episode"], path="acceptance_profile.per_episode"
    )
    _exact(
        per_episode,
        {
            "max_environment_actions",
            "max_control_turns",
            "max_no_progress_control_turns",
            "max_semantic_rounds",
        },
        path="acceptance_profile.per_episode",
    )
    for key, value in per_episode.items():
        _positive_int(value, path=f"acceptance_profile.per_episode.{key}")
    ceilings = _mapping(
        profile["hard_resource_ceilings"],
        path="acceptance_profile.hard_resource_ceilings",
    )
    _exact(
        ceilings,
        {
            "max_external_model_calls",
            "max_agent_api_calls",
            "max_sam3_calls",
            "max_images",
            "max_images_per_request",
            "max_image_bytes",
            "max_image_bytes_per_request",
            "max_wall_time_seconds",
            "max_artifact_bytes",
        },
        path="acceptance_profile.hard_resource_ceilings",
    )
    for key, value in ceilings.items():
        _positive_int(value, path=f"acceptance_profile.hard_resource_ceilings.{key}")
    return AcceptanceProfile(profile, raw_sha, config_sha)


@dataclass(frozen=True, slots=True)
class ExperimentBudgets:
    max_episodes: int
    max_workers: int
    max_environment_actions_per_episode: int
    max_control_turns_per_episode: int
    max_no_progress_control_turns_per_episode: int
    max_semantic_rounds_per_episode: int
    max_external_model_calls: int
    max_agent_api_calls: int
    max_sam3_calls: int
    max_images: int
    max_images_per_request: int
    max_image_bytes: int
    max_image_bytes_per_request: int
    max_wall_time_seconds: int
    max_artifact_bytes: int

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        profile: AcceptanceProfile,
        path: str = "budgets",
    ) -> ExperimentBudgets:
        payload = _mapping(value, path=path)
        fields = set(cls.__dataclass_fields__)
        _exact(payload, fields, path=path)
        values = {
            field_name: (
                _nonnegative_int(payload[field_name], path=f"{path}.{field_name}")
                if field_name == "max_agent_api_calls"
                else _positive_int(payload[field_name], path=f"{path}.{field_name}")
            )
            for field_name in fields
        }
        expected_fixed = {
            "max_episodes": profile.schedule["episode_count"],
            "max_workers": profile.schedule["worker_count"],
            "max_environment_actions_per_episode": profile.per_episode[
                "max_environment_actions"
            ],
            "max_control_turns_per_episode": profile.per_episode["max_control_turns"],
            "max_no_progress_control_turns_per_episode": profile.per_episode[
                "max_no_progress_control_turns"
            ],
            "max_semantic_rounds_per_episode": profile.per_episode[
                "max_semantic_rounds"
            ],
        }
        for field_name, expected in expected_fixed.items():
            if values[field_name] != expected:
                _fail(
                    f"{path}.{field_name} must equal the frozen acceptance "
                    f"profile value {expected}"
                )
        for field_name, ceiling in profile.hard_resource_ceilings.items():
            if values[field_name] > ceiling:
                _fail(f"{path}.{field_name} exceeds the frozen profile ceiling")
        if values["max_images_per_request"] > values["max_images"]:
            _fail("budgets.max_images_per_request exceeds the invocation cap")
        if values["max_image_bytes_per_request"] > values["max_image_bytes"]:
            _fail("budgets.max_image_bytes_per_request exceeds the invocation cap")
        if (
            values["max_agent_api_calls"] != 0
            and values["max_agent_api_calls"] > values["max_external_model_calls"]
        ):
            _fail("budgets.max_agent_api_calls exceeds the total external-call cap")
        if values["max_sam3_calls"] > values["max_external_model_calls"]:
            _fail("budgets.max_sam3_calls exceeds the total external-call cap")
        return cls(**values)

    def to_dict(self) -> dict[str, int]:
        return {
            field_name: int(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class SequentialPreregistration:
    payload: dict[str, Any]
    preregistration_id: str
    task: dict[str, Any]
    launch_config: ArtifactRef
    episodes: tuple[EpisodeSpec, EpisodeSpec]
    initial_snapshot: SnapshotState
    runtime_binding: dict[str, Any]
    budgets: ExperimentBudgets
    planned_usage: ResourceUsage
    run_id: str
    run_claim_path: str
    output_root: str
    snapshot_output_root: str
    segmentation_artifact_root: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SequentialPreregistration:
        payload = _mapping(value, path="preregistration")
        knowledge_field = "hpk" if "hpk" in payload else "afk"
        _exact(
            payload,
            {
                "schema",
                "preregistration_id",
                "acceptance_profile",
                "task",
                "launch_config",
                "episodes",
                "execution",
                knowledge_field,
                "manipulation_policy",
                "budgets",
                "planned_usage",
                "run_id",
                "run_claim_path",
                "output_root",
                "snapshot_output_root",
                "segmentation_artifact_root",
            },
            path="preregistration",
        )
        if payload["schema"] not in {PREREGISTRATION_SCHEMA, "tcm/afk/p0de_sequential_preregistration/v1"}:
            _fail("preregistration.schema is unsupported")

        profile = load_acceptance_profile()
        declared_profile = _mapping(
            payload["acceptance_profile"], path="acceptance_profile"
        )
        _exact(
            declared_profile,
            {"profile_id", "raw_sha256", "config_sha256"},
            path="acceptance_profile",
        )
        if declared_profile not in (
            profile.identity(),
            {
                "profile_id": "afk_p0de_sequential_acceptance/v2",
                "raw_sha256": "f1ee9caf5ff2e717b9c0721809014d8b7d73de8744f99c882c131ea2fa3f8e47",
                "config_sha256": "9df04ff42226b575bff2b2baed9c47ee1b45ab9d28ad6347de112027b2ec0e66",
            },
        ):
            _fail("preregistration does not pin the exact acceptance profile")

        task = _mapping(payload["task"], path="task")
        _exact(
            task,
            {
                "task_name",
                "task_config",
                "task_definition",
                "instruction",
                "instruction_sha256",
                "instruction_source",
            },
            path="task",
        )
        _string(task["task_name"], path="task.task_name", max_chars=256)
        _string(task["task_config"], path="task.task_config", max_chars=256)
        ArtifactRef.from_mapping(task["task_definition"], path="task.task_definition")
        instruction = _string(task["instruction"], path="task.instruction")
        expected_instruction_sha = hashlib.sha256(
            instruction.encode("utf-8")
        ).hexdigest()
        if task["instruction_sha256"] != expected_instruction_sha:
            _fail("task.instruction_sha256 does not match instruction UTF-8 bytes")
        ArtifactRef.from_mapping(
            task["instruction_source"], path="task.instruction_source"
        )
        launch_config = ArtifactRef.from_mapping(
            payload["launch_config"], path="launch_config"
        )

        raw_episodes = payload["episodes"]
        if not isinstance(raw_episodes, list) or len(raw_episodes) != 2:
            _fail("episodes must contain exactly [update, probe]")
        episodes: list[EpisodeSpec] = []
        for index, expected_role in enumerate(EPISODE_ROLES):
            item = _mapping(raw_episodes[index], path=f"episodes[{index}]")
            _exact(item, {"role", "seed"}, path=f"episodes[{index}]")
            if item["role"] != expected_role:
                _fail("episodes must be ordered exactly as update then probe")
            episodes.append(
                EpisodeSpec(
                    role=expected_role,
                    seed=_nonnegative_int(item["seed"], path=f"episodes[{index}].seed"),
                )
            )
        if episodes[0].seed == episodes[1].seed:
            _fail("update and held-out probe seeds must be distinct")

        execution = _mapping(payload["execution"], path="execution")
        _exact(
            execution,
            {
                "worker_count",
                "episode_count",
                "seed_replacement",
                "seed_retry_count",
                "backend_retry_count",
            },
            path="execution",
        )
        if execution != profile.schedule:
            _fail("execution must freeze one worker, two episodes, and zero retries")

        hpk = _mapping(payload[knowledge_field], path="hpk")
        _exact(
            hpk,
            {
                "mode",
                "scope",
                "initial_snapshot",
                "runtime_binding",
                "procedure_experience_mode",
                "allow_oracle_evidence",
                "allow_expert_prior",
            },
            path="hpk",
        )
        if hpk["mode"] != "evolving":
            _fail("hpk.mode must be evolving for the sequential experiment")
        scope = _mapping(hpk["scope"], path="hpk.scope")
        _exact(
            scope,
            {"name", "development_only", "formal_evaluation"},
            path="hpk.scope",
        )
        if scope["name"] not in ACCEPTANCE_SCOPES:
            _fail("hpk.scope.name is unsupported")
        if scope["name"] == "integration_development":
            expected_scope_flags = (True, False)
        else:
            expected_scope_flags = (False, True)
        if (
            scope["development_only"],
            scope["formal_evaluation"],
        ) != expected_scope_flags:
            _fail("hpk.scope eligibility flags disagree with its name")
        if (
            hpk["procedure_experience_mode"] != "off"
            or hpk["allow_oracle_evidence"] is not False
            or hpk["allow_expert_prior"] is not False
        ):
            _fail(
                "the no-oracle sequential experiment requires procedure experience "
                "off and no expert/oracle prior"
            )
        initial_snapshot = SnapshotState.from_mapping(
            hpk["initial_snapshot"], path="hpk.initial_snapshot"
        )
        try:
            runtime_binding = validate_runtime_binding(hpk["runtime_binding"])
        except Exception as exc:
            raise SequentialExperimentError(
                f"hpk.runtime_binding is invalid: {exc}"
            ) from exc
        if runtime_binding["schema"] not in {
            HPK_SEQUENTIAL_RUNTIME_BINDING_SCHEMA, "tcm/afk/runtime_binding/v2"
        }:
            _fail("sequential experiment requires a run-bound runtime binding v2")
        if runtime_binding["snapshot_ref"] != initial_snapshot.identity():
            _fail("hpk.runtime_binding does not bind the exact K0 snapshot")
        geometry_ref = runtime_binding["policy_refs"]["geometry"]
        if (geometry_ref["policy_id"], geometry_ref["config_sha256"]) not in SAFE_EXPLORATION_GEOMETRY_POLICY_IDENTITIES:
            _fail(
                "real sequential experiment requires exact safe-exploration geometry v3"
            )
        promotion_payload = runtime_binding["policy_refs"]["promotion"]["payload"]
        if bool(promotion_payload["development_only"]) != bool(
            scope["development_only"]
        ) or bool(promotion_payload["formal_evaluation_eligible"]) != bool(
            scope["formal_evaluation"]
        ):
            _fail("hpk.scope disagrees with the bound promotion policy")

        policy = _mapping(payload["manipulation_policy"], path="manipulation_policy")
        _exact(
            policy,
            {
                "grasp_transport_policy",
                "release_guard_enabled",
                "reobserve_enabled",
                "action_geometry_repair_pending_policy",
            },
            path="manipulation_policy",
        )
        if policy != {
            "grasp_transport_policy": "strict",
            "release_guard_enabled": True,
            "reobserve_enabled": True,
            "action_geometry_repair_pending_policy": "strict",
        }:
            _fail("manipulation_policy must freeze strict grasp/release/reobserve")

        budgets = ExperimentBudgets.from_mapping(payload["budgets"], profile=profile)
        planned_payload = _mapping(payload["planned_usage"], path="planned_usage")
        _exact(
            planned_payload,
            set(ResourceUsage.__dataclass_fields__),
            path="planned_usage",
        )
        planned_usage = ResourceUsage(
            **{
                key: _nonnegative_int(value, path=f"planned_usage.{key}")
                for key, value in planned_payload.items()
            }
        )
        planned_caps = {
            "external_model_calls": budgets.max_external_model_calls,
            "images": budgets.max_images,
            "image_bytes": budgets.max_image_bytes,
            "artifact_bytes": budgets.max_artifact_bytes,
        }
        for field_name, cap in planned_caps.items():
            if getattr(planned_usage, field_name) > cap:
                _fail(f"planned_usage.{field_name} exceeds the preregistered cap")
        run_claim_path = _absolute_path(
            payload["run_claim_path"], path="run_claim_path"
        )
        output_root = _absolute_path(payload["output_root"], path="output_root")
        snapshot_output_root = _absolute_path(
            payload["snapshot_output_root"], path="snapshot_output_root"
        )
        segmentation_artifact_root = _absolute_path(
            payload["segmentation_artifact_root"],
            path="segmentation_artifact_root",
        )
        if (
            len(
                {
                    Path(output_root),
                    Path(snapshot_output_root),
                    Path(segmentation_artifact_root),
                }
            )
            != 3
        ):
            _fail("run, snapshot, and segmentation roots must be distinct")
        expected_initial_manifest = (
            Path(snapshot_output_root)
            / initial_snapshot.snapshot_id
            / "snapshot_manifest.json"
        )
        if Path(initial_snapshot.manifest_path) != expected_initial_manifest:
            _fail(
                "hpk.initial_snapshot manifest must be the exact root member of "
                "snapshot_output_root"
            )
        try:
            validate_content_id(payload["run_id"], prefix="afkrun", path="run_id")
        except Exception as exc:
            raise SequentialExperimentError(str(exc)) from exc
        if Path(run_claim_path) not in {
            Path(output_root) / "hpk_run_claim.json",
            Path(output_root) / "afk_run_claim.json",
        }:
            _fail("run_claim_path must be the frozen claim member inside output_root")
        expected_run_binding = build_sequential_run_binding(
            run_id=str(payload["run_id"]),
            experiment_contract_sha256=_experiment_contract_sha256(
                payload,
                runtime_binding=runtime_binding,
            ),
            output_root_sha256=_path_identity_sha256(output_root),
            claim_path_sha256=_path_identity_sha256(run_claim_path),
        )
        if runtime_binding["run_binding"]["schema"] == "tcm/afk/sequential_run_binding/v1":
            expected_run_binding["schema"] = "tcm/afk/sequential_run_binding/v1"
            expected_run_binding["run_binding_id"] = stable_content_id(
                "afkrunbinding",
                {key: value for key, value in expected_run_binding.items() if key != "run_binding_id"},
            )
        if runtime_binding["run_binding"] != expected_run_binding:
            _fail("runtime binding does not match the preregistered run contract")
        claimed_id = payload["preregistration_id"]
        try:
            validate_content_id(
                claimed_id,
                prefix="afkprereg",
                path="preregistration.preregistration_id",
            )
        except Exception as exc:
            raise SequentialExperimentError(str(exc)) from exc
        identity_payload = copy.deepcopy(payload)
        identity_payload.pop("preregistration_id")
        if claimed_id != stable_content_id("afkprereg", identity_payload):
            _fail("preregistration_id content mismatch")
        return cls(
            payload=payload,
            preregistration_id=claimed_id,
            task=task,
            launch_config=launch_config,
            episodes=(episodes[0], episodes[1]),
            initial_snapshot=initial_snapshot,
            runtime_binding=runtime_binding,
            budgets=budgets,
            planned_usage=planned_usage,
            run_id=str(payload["run_id"]),
            run_claim_path=run_claim_path,
            output_root=output_root,
            snapshot_output_root=snapshot_output_root,
            segmentation_artifact_root=segmentation_artifact_root,
        )

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)

    def runtime_binding_for_snapshot(self, snapshot: SnapshotState) -> dict[str, Any]:
        provenance = self.runtime_binding["provenance"]
        return build_runtime_binding(
            snapshot_id=snapshot.snapshot_id,
            snapshot_manifest_sha256=snapshot.manifest_sha256,
            policy_refs=self.runtime_binding["policy_refs"],
            provenance_schema=provenance["schema"],
            provenance_manifest_sha256=provenance["manifest_sha256"],
            config_sha256=provenance["config_sha256"],
            model_identity_sha256=provenance["model_identity_sha256"],
            runtime_source_sha256=provenance["runtime_source_sha256"],
            evidence_runtime_schema=provenance["evidence_runtime_schema"],
            run_binding=self.runtime_binding["run_binding"],
        )


def build_preregistration(
    payload_without_id: Mapping[str, Any],
) -> SequentialPreregistration:
    payload = _mapping(payload_without_id, path="preregistration")
    if "preregistration_id" in payload:
        _fail("build_preregistration input must omit preregistration_id")
    knowledge_field = "hpk" if "hpk" in payload else "afk"
    hpk = _mapping(payload.get(knowledge_field), path="hpk")
    base_binding = validate_runtime_binding(hpk.get("runtime_binding"))
    run_binding = build_sequential_run_binding(
        run_id=str(payload.get("run_id", "")),
        experiment_contract_sha256=_experiment_contract_sha256(
            payload,
            runtime_binding=base_binding,
        ),
        output_root_sha256=_path_identity_sha256(str(payload.get("output_root", ""))),
        claim_path_sha256=_path_identity_sha256(str(payload.get("run_claim_path", ""))),
    )
    provenance = base_binding["provenance"]
    hpk["runtime_binding"] = build_runtime_binding(
        snapshot_id=base_binding["snapshot_ref"]["snapshot_id"],
        snapshot_manifest_sha256=base_binding["snapshot_ref"]["manifest_sha256"],
        policy_refs=base_binding["policy_refs"],
        provenance_schema=provenance["schema"],
        provenance_manifest_sha256=provenance["manifest_sha256"],
        config_sha256=provenance["config_sha256"],
        model_identity_sha256=provenance["model_identity_sha256"],
        runtime_source_sha256=provenance["runtime_source_sha256"],
        evidence_runtime_schema=provenance["evidence_runtime_schema"],
        run_binding=run_binding,
    )
    payload[knowledge_field] = hpk
    payload["preregistration_id"] = stable_content_id("afkprereg", payload)
    return SequentialPreregistration.from_mapping(payload)


@dataclass(frozen=True, slots=True)
class PublishedPreregistration:
    path: Path
    sha256: str
    preregistration: SequentialPreregistration


def _read_regular_file(path: Path, *, label: str, max_bytes: int) -> bytes:
    absolute = path.expanduser().absolute()
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise SequentialExperimentError(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a non-symlink regular file")
    if metadata.st_size <= 0 or metadata.st_size > max_bytes:
        _fail(f"{label} exceeds its byte budget or is empty")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        raw = os.read(descriptor, max_bytes + 1)
        after = os.fstat(descriptor)
        if len(raw) > max_bytes or len(raw) != after.st_size:
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
    finally:
        os.close(descriptor)


def _read_pinned_artifact(
    reference: ArtifactRef,
    *,
    label: str,
    max_bytes: int = _MAX_PREREGISTRATION_BYTES,
) -> bytes:
    raw = _read_regular_file(Path(reference.path), label=label, max_bytes=max_bytes)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != reference.sha256:
        _fail(f"{label} SHA-256 mismatch: expected={reference.sha256}, actual={actual}")
    return raw


def _canonical_object_artifact(
    reference: ArtifactRef,
    *,
    label: str,
    max_bytes: int = _MAX_PREREGISTRATION_BYTES,
) -> dict[str, Any]:
    raw = _read_pinned_artifact(reference, label=label, max_bytes=max_bytes)
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw:
        _fail(f"{label} must end in exactly one LF")
    try:
        value = json.loads(raw[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SequentialExperimentError(f"{label} is invalid JSON: {exc}") from exc
    payload = _mapping(value, path=label)
    if canonical_json_bytes(payload) + b"\n" != raw:
        _fail(f"{label} is not canonical JSON plus one LF")
    return payload


def _load_snapshot_state(
    expected: SnapshotState, *, label: str
) -> tuple[SnapshotState, Any]:
    try:
        loaded = load_evolving_snapshot(
            expected.manifest_path,
            expected.manifest_sha256,
        )
    except Exception as exc:
        raise SequentialExperimentError(
            f"{label} failed explicit evolving-snapshot validation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    actual = SnapshotState.from_loaded(loaded)
    if actual != expected:
        _fail(f"{label} descriptor disagrees with the loaded snapshot")
    return actual, loaded


def _validate_finalization_receipt(
    reference: ArtifactRef,
    *,
    rollout_manifest: ArtifactRef,
    request: EpisodeExecutionRequest,
    expected_status: str,
    child: SnapshotState | None,
) -> dict[str, Any]:
    receipt = _canonical_object_artifact(
        reference,
        label="finalization receipt",
    )
    _exact(
        receipt,
        {
            "schema",
            "receipt_id",
            "rollout_import_manifest",
            "finalization",
            "next_snapshot",
            "advanced",
        },
        path="finalization receipt",
    )
    if receipt["schema"] not in {"roboharn_evo/hpk/finalization_receipt/v1", "tcm/afk/finalization_receipt/v1"}:
        _fail("finalization receipt schema is unsupported")
    identity = copy.deepcopy(receipt)
    receipt_id = identity.pop("receipt_id")
    if receipt_id != stable_content_id("afkfinal", identity):
        _fail("finalization receipt content identity mismatch")
    if receipt["rollout_import_manifest"] != rollout_manifest.to_dict():
        _fail("finalization receipt references another rollout manifest")
    finalization = _mapping(receipt["finalization"], path="finalization")
    _exact(
        finalization,
        {
            "episode_id",
            "status",
            "parent",
            "active",
            "child",
            "evidence_ids",
            "updated_entry_ids",
            "decision_ids",
            "abstentions",
        },
        path="finalization",
    )
    if finalization["status"] != expected_status:
        _fail("finalization receipt status disagrees with the episode result")
    if finalization["parent"] != request.parent_snapshot.identity():
        _fail("finalization receipt parent disagrees with the episode lease")
    if child is None:
        if (
            finalization["child"] is not None
            or finalization["active"] != request.parent_snapshot.identity()
            or receipt["next_snapshot"] != request.parent_snapshot.identity()
            or receipt["advanced"] is not False
        ):
            _fail("finalization receipt unexpectedly published a child")
    else:
        if (
            finalization["child"] != child.identity()
            or finalization["active"] != child.identity()
            or receipt["next_snapshot"] != child.identity()
            or receipt["advanced"] is not True
        ):
            _fail("finalization receipt does not activate the explicit child")
    return receipt


def _validate_rollout_manifest(
    reference: ArtifactRef,
    *,
    request: EpisodeExecutionRequest,
) -> tuple[Any, dict[str, Any]]:
    manifest = _canonical_object_artifact(
        reference,
        label="rollout import manifest",
    )
    if manifest.get("runtime_binding") != request.runtime_binding:
        _fail("rollout import manifest runtime binding mismatch")
    if (
        type(manifest.get("episode_id")) is not int
        or manifest.get("episode_id") != request.ordinal
    ):
        _fail("rollout import manifest episode order mismatch")
    public = ArtifactRef.from_mapping(
        manifest.get("public_trace"), path="rollout_manifest.public_trace"
    )
    try:
        imported = AgentRolloutImporter().import_files(
            manifest_path=reference.path,
            trace_path=public.path,
            expected_manifest_sha256=reference.sha256,
            expected_trace_sha256=public.sha256,
        )
    except Exception as exc:
        raise SequentialExperimentError(
            f"rollout import manifest failed strict import: {type(exc).__name__}: {exc}"
        ) from exc
    # Importer validates all V2 public/private members.  Pin again after import
    # so a cooperating executor cannot swap the manifest between the harness'
    # first read and the importer's own reads.
    if _read_pinned_artifact(reference, label="rollout import manifest") != (
        canonical_json_bytes(manifest) + b"\n"
    ):
        _fail("rollout import manifest changed during harness validation")
    return imported, manifest


def _probe_usage_proof(
    imported: Any,
    *,
    request: EpisodeExecutionRequest,
    learned_entry_ids: tuple[str, ...],
) -> AuditV1:
    raw = _read_regular_file(
        Path(imported.trace_path),
        label="probe public trace",
        max_bytes=64 * 1024 * 1024,
    )
    if hashlib.sha256(raw).hexdigest() != imported.trace_sha256:
        _fail("probe public trace changed after strict import")
    audits_by_id: dict[str, tuple[bytes, AuditV1]] = {}
    for ordinal, line in enumerate(raw.splitlines(), 1):
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SequentialExperimentError(
                f"probe public trace line {ordinal} is invalid: {exc}"
            ) from exc
        if not isinstance(event, Mapping):
            _fail(f"probe public trace line {ordinal} is not an object")
        if event.get("seed") != request.seed:
            _fail("probe public trace seed differs from preregistration")
        if event.get("event") != "hpk_public_usage_audit":
            continue
        payload = {
            key: copy.deepcopy(value)
            for key, value in event.items()
            if key not in {"event", "timestamp", "episode_id", "seed", "env_step"}
        }
        audit = AuditV1.from_dict(payload)
        encoded_audit = canonical_json_bytes(audit.to_dict())
        previous = audits_by_id.get(audit.stable_id)
        if previous is not None and previous[0] != encoded_audit:
            _fail("probe public trace reuses one audit ID with different bytes")
        audits_by_id[audit.stable_id] = (encoded_audit, audit)

    matching = [
        audit
        for _encoded, audit in audits_by_id.values()
        if (
            audit["snapshot_id"] == request.parent_snapshot.snapshot_id
            and audit["snapshot_manifest_sha256"]
            == request.parent_snapshot.manifest_sha256
            and audit["selected_entry_id"] in learned_entry_ids
            and audit["behavior_changed"] is True
            and audit["geometric_compliance"] is True
        )
    ]
    if not matching:
        _fail(
            "probe requires a typed public audit proving retrieval/use, "
            "geometric compliance, and behavior change for a learned entry"
        )
    for audit in sorted(matching, key=lambda value: value.stable_id):
        for transition in imported.transitions:
            transition_id = getattr(transition, "stable_id", None)
            if transition_id is None and isinstance(transition, Mapping):
                transition_id = transition.get("transition_id")
            if (
                (audit.stable_id, transition_id)
                in imported.ranked_usage_transition_links
                and transition["selected_hpk_entry_id"] == audit["selected_entry_id"]
                and transition["condition_id"] == audit["condition_id"]
                and transition["task_strategy_id"] == audit["task_strategy_id"]
                and transition["geometric_strategy_id"]
                == audit["selected_geometric_strategy_id"]
                and transition["selected_hpk_entry_id"] in learned_entry_ids
                and transition["geometric_compliance"] is True
                and transition["physical_action_executed"] is True
                and transition["motion_status"] == "completed"
                and transition["realization_status"] == "satisfied"
                and transition["effect_observation_scope"] == "independent"
                and transition["pre_effect_state"] is not None
                and transition["post_effect_state"] is not None
                and isinstance(transition["env_step_before"], int)
                and isinstance(transition["env_step_after"], int)
                and transition["env_step_after"] > transition["env_step_before"]
                and transition["infrastructure_valid"] is True
                and transition["oracle_derived"] is False
                and transition["expert_derived"] is False
            ):
                return audit
    _fail(
        "probe usage audit did not survive guard/dispatch into a strict imported "
        "physical transition with the same learned c/u/z binding"
    )


def publish_preregistration(
    path: str | os.PathLike[str], preregistration: SequentialPreregistration
) -> PublishedPreregistration:
    destination = Path(path).expanduser().absolute()
    if not destination.parent.is_dir():
        _fail("preregistration destination parent must already exist")
    raw = canonical_json_bytes(preregistration.to_dict()) + b"\n"
    if len(raw) > _MAX_PREREGISTRATION_BYTES:
        _fail("preregistration exceeds its byte budget")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as exc:
        raise SequentialExperimentError(
            "preregistration destination already exists; no-clobber publication refused"
        ) from exc
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        # On failure the exclusively created destination remains visible for
        # forensic recovery; it is never replaced or reused silently.
        os.close(descriptor)
    parent_descriptor = os.open(
        destination.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return PublishedPreregistration(
        path=destination,
        sha256=hashlib.sha256(raw).hexdigest(),
        preregistration=preregistration,
    )


def load_preregistration(
    path: str | os.PathLike[str], expected_sha256: str
) -> PublishedPreregistration:
    expected = _sha(expected_sha256, path="expected_preregistration_sha256")
    source = Path(path).expanduser().absolute()
    raw = _read_regular_file(
        source,
        label="preregistration",
        max_bytes=_MAX_PREREGISTRATION_BYTES,
    )
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        _fail(f"preregistration SHA-256 mismatch: expected={expected}, actual={actual}")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SequentialExperimentError(f"invalid preregistration JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        _fail("preregistration must contain one JSON object")
    preregistration = SequentialPreregistration.from_mapping(decoded)
    canonical = canonical_json_bytes(preregistration.to_dict()) + b"\n"
    if raw != canonical:
        _fail("preregistration bytes are not canonical JSON plus one LF")
    return PublishedPreregistration(source, actual, preregistration)


class QuotaGate:
    """Invocation-wide reservation ledger checked before every admitted I/O."""

    def __init__(
        self,
        budgets: ExperimentBudgets,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._budgets = budgets
        self._clock = clock
        self._started_at = float(clock())
        if not math.isfinite(self._started_at):
            _fail("monotonic clock returned a non-finite start value")
        self._usage = ResourceUsage()
        self._artifact_reserved_capacity: int | None = None

    @property
    def usage(self) -> ResourceUsage:
        return self._usage

    def elapsed_seconds(self) -> float:
        current = float(self._clock())
        if not math.isfinite(current) or current < self._started_at:
            raise SequentialQuotaExceeded("wall clock is non-finite or regressed")
        return current - self._started_at

    def remaining_wall_seconds(self) -> float:
        remaining = self._budgets.max_wall_time_seconds - self.elapsed_seconds()
        if remaining <= 0:
            raise SequentialQuotaExceeded("wall-time quota exhausted")
        return remaining

    def checkpoint(self) -> None:
        self.remaining_wall_seconds()

    def reserve_external_call(self, *, images: int, image_bytes: int) -> None:
        self.checkpoint()
        images = _nonnegative_int(images, path="external_call.images")
        image_bytes = _nonnegative_int(image_bytes, path="external_call.image_bytes")
        if images > self._budgets.max_images_per_request:
            raise SequentialQuotaExceeded(
                "per-request image quota would be exceeded before I/O"
            )
        if image_bytes > self._budgets.max_image_bytes_per_request:
            raise SequentialQuotaExceeded(
                "per-request image-byte quota would be exceeded before I/O"
            )
        proposed = ResourceUsage(
            external_model_calls=self._usage.external_model_calls + 1,
            images=self._usage.images + images,
            image_bytes=self._usage.image_bytes + image_bytes,
            artifact_bytes=self._usage.artifact_bytes,
        )
        self._require_within_caps(proposed)
        self._usage = proposed

    def reserve_artifact_bytes(self, byte_count: int) -> None:
        self.checkpoint()
        byte_count = _nonnegative_int(byte_count, path="artifact.byte_count")
        proposed = ResourceUsage(
            external_model_calls=self._usage.external_model_calls,
            images=self._usage.images,
            image_bytes=self._usage.image_bytes,
            artifact_bytes=self._usage.artifact_bytes + byte_count,
        )
        if (
            self._artifact_reserved_capacity is not None
            and proposed.artifact_bytes > self._artifact_reserved_capacity
        ):
            raise SequentialQuotaExceeded(
                "statically reserved artifact capacity would be exceeded before I/O"
            )
        self._require_within_caps(proposed)
        self._usage = proposed

    def reserve_artifact_capacity(self, byte_count: int) -> None:
        """Freeze an upper bound without charging bytes that are not written."""

        self.checkpoint()
        capacity = _nonnegative_int(byte_count, path="artifact.capacity")
        if self._artifact_reserved_capacity is not None:
            _fail("artifact capacity was already reserved")
        if capacity > self._budgets.max_artifact_bytes:
            raise SequentialQuotaExceeded(
                "artifact capacity exceeds the preregistered total cap"
            )
        if self._usage.artifact_bytes > capacity:
            raise SequentialQuotaExceeded(
                "artifact capacity is below bytes already written"
            )
        self._artifact_reserved_capacity = capacity

    def _require_within_caps(self, usage: ResourceUsage) -> None:
        caps = {
            "external_model_calls": self._budgets.max_external_model_calls,
            "images": self._budgets.max_images,
            "image_bytes": self._budgets.max_image_bytes,
            "artifact_bytes": self._budgets.max_artifact_bytes,
        }
        for field_name, cap in caps.items():
            if getattr(usage, field_name) > cap:
                raise SequentialQuotaExceeded(
                    f"{field_name} quota would be exceeded before I/O"
                )


T = TypeVar("T")


class MeteredIO(Generic[T]):
    """Only admitted I/O seam exposed to a sequential episode executor."""

    def __init__(self, gate: QuotaGate) -> None:
        self._gate = gate

    @property
    def usage(self) -> ResourceUsage:
        return self._gate.usage

    def external_call(
        self,
        operation: Callable[[float], T],
        *,
        image_payloads: Sequence[bytes] = (),
    ) -> T:
        payloads = tuple(image_payloads)
        if any(not isinstance(value, bytes) for value in payloads):
            _fail("external-call image payloads must be bytes")
        self._gate.reserve_external_call(
            images=len(payloads),
            image_bytes=sum(len(value) for value in payloads),
        )
        return operation(self._gate.remaining_wall_seconds())

    def artifact_write(self, payload: bytes, operation: Callable[[bytes], T]) -> T:
        if not isinstance(payload, bytes):
            _fail("artifact payload must be bytes")
        self._gate.reserve_artifact_bytes(len(payload))
        return operation(payload)

    def checkpoint(self) -> None:
        self._gate.checkpoint()


def _claim_run_once(
    preregistration: SequentialPreregistration,
    *,
    metered_io: MeteredIO[Any],
) -> ArtifactRef:
    preregistration_sha256 = hashlib.sha256(
        canonical_json_bytes(preregistration.to_dict()) + b"\n"
    ).hexdigest()
    payload = {
        "schema": "roboharn_evo/hpk/p0de_sequential_run_claim/v1",
        "preregistration_id": preregistration.preregistration_id,
        "preregistration_sha256": preregistration_sha256,
        "run_id": preregistration.run_id,
        "output_root": preregistration.output_root,
        "runtime_binding_id": preregistration.runtime_binding["binding_id"],
        "run_binding_id": preregistration.runtime_binding["run_binding"][
            "run_binding_id"
        ],
        "effective_config_sha256": preregistration.runtime_binding["provenance"][
            "config_sha256"
        ],
        "ordered_episodes": [value.to_dict() for value in preregistration.episodes],
    }
    payload["claim_id"] = stable_content_id("afkrunclaim", payload)
    encoded = canonical_json_bytes(payload) + b"\n"
    destination = Path(preregistration.run_claim_path)
    output_root = Path(preregistration.output_root)
    try:
        os.mkdir(output_root, mode=0o700)
    except FileExistsError as exc:
        raise SequentialExperimentError(
            "dedicated output_root already exists; one-shot replay refused"
        ) from exc
    parent_directory = os.open(
        output_root.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(parent_directory)
    finally:
        os.close(parent_directory)

    def publish(raw: bytes) -> None:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(destination, flags, 0o600)
        except FileExistsError as exc:
            raise SequentialExperimentError(
                "preregistered sequential run was already claimed; replay refused"
            ) from exc
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    metered_io.artifact_write(encoded, publish)
    return ArtifactRef(str(destination), hashlib.sha256(encoded).hexdigest())


def _validate_pristine_output_root(preregistration: SequentialPreregistration) -> None:
    output = Path(preregistration.output_root)
    if not output.parent.is_dir() or output.parent.is_symlink():
        _fail("output_root parent must be an existing non-symlink directory")
    try:
        output.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SequentialExperimentError(f"output_root is unavailable: {exc}") from exc
    _fail("dedicated output_root must not exist before the one-shot run")


def _require_artifact_within_root(
    root_path: str,
    path: str,
    *,
    label: str,
) -> None:
    try:
        output = Path(root_path).resolve(strict=True)
    except OSError as exc:
        raise SequentialExperimentError(
            f"{label} must resolve strictly inside its preregistered root; "
            "the dedicated root is absent"
        ) from exc
    if output.is_symlink() or not output.is_dir():
        _fail("dedicated output_root must be a non-symlink directory")
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(output)
    except (OSError, ValueError) as exc:
        raise SequentialExperimentError(
            f"{label} must resolve strictly inside the preregistered output_root"
        ) from exc
    current = candidate.absolute()
    while current != output:
        if current.is_symlink():
            _fail(f"{label} path contains a symlink")
        if current.parent == current:
            _fail(f"{label} path escaped output_root")
        current = current.parent


def _require_artifact_within_output(
    preregistration: SequentialPreregistration,
    path: str,
    *,
    label: str,
) -> None:
    _require_artifact_within_root(
        preregistration.output_root,
        path,
        label=label,
    )


@dataclass(frozen=True, slots=True)
class EpisodeExecutionRequest:
    preregistration_id: str
    ordinal: int
    role: str
    seed: int
    task: dict[str, Any]
    parent_snapshot: SnapshotState
    runtime_binding: dict[str, Any]
    required_learned_entry_ids: tuple[str, ...]
    control_limits: dict[str, int]
    required_semantic_knowledge: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class EpisodeExecutionResult:
    preregistration_id: str
    ordinal: int
    role: str
    seed: int
    loaded_snapshot: SnapshotState
    runtime_binding_id: str
    control_usage: ControlUsage
    provider_usage: ResourceUsage
    rollout_manifest: ArtifactRef
    finalization_receipt: ArtifactRef
    finalization_status: str
    child_snapshot: SnapshotState | None = None
    semantic_knowledge: tuple[dict[str, Any], ...] = ()
    semantic_newly_accepted: bool = False
    semantic_knowledge_used: bool = False


class SequentialEpisodeExecutor(Protocol):
    """Executor contract used by the real adapter and deterministic tests.

    ``static_plan`` must be pure and perform no file/network/simulator I/O.
    ``execute`` must route every external call and artifact write through
    ``metered_io`` and return the independent provider/artifact audit totals for
    that episode.  The harness rejects any discrepancy.
    """

    def static_plan(
        self, preregistration: SequentialPreregistration
    ) -> ResourceUsage: ...

    def execute(
        self,
        request: EpisodeExecutionRequest,
        *,
        metered_io: MeteredIO[Any],
    ) -> EpisodeExecutionResult: ...


@dataclass(frozen=True, slots=True)
class SequentialRunResult:
    preregistration_id: str
    status: str
    update: EpisodeExecutionResult
    probe: EpisodeExecutionResult | None
    learned_entry_ids: tuple[str, ...]
    final_usage: ResourceUsage
    learned_knowledge: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        def episode(value: EpisodeExecutionResult) -> dict[str, Any]:
            return {
                "ordinal": value.ordinal,
                "role": value.role,
                "seed": value.seed,
                "loaded_snapshot": value.loaded_snapshot.to_dict(),
                "runtime_binding_id": value.runtime_binding_id,
                "control_usage": value.control_usage.to_dict(),
                "provider_usage": value.provider_usage.to_dict(),
                "rollout_manifest": value.rollout_manifest.to_dict(),
                "finalization_receipt": value.finalization_receipt.to_dict(),
                "finalization_status": value.finalization_status,
                "child_snapshot": (
                    None
                    if value.child_snapshot is None
                    else value.child_snapshot.to_dict()
                ),
                "semantic_knowledge": [
                    copy.deepcopy(item) for item in value.semantic_knowledge
                ],
                "semantic_newly_accepted": value.semantic_newly_accepted,
                "semantic_knowledge_used": value.semantic_knowledge_used,
            }

        return {
            "schema": RUN_RESULT_SCHEMA,
            "preregistration_id": self.preregistration_id,
            "status": self.status,
            "update": episode(self.update),
            "probe": None if self.probe is None else episode(self.probe),
            "learned_entry_ids": list(self.learned_entry_ids),
            "learned_knowledge": [
                copy.deepcopy(item) for item in self.learned_knowledge
            ],
            "final_usage": self.final_usage.to_dict(),
        }


class SequentialPairHarness:
    """Execute exactly one update episode and one held-out probe in order."""

    def __init__(
        self,
        preregistration: SequentialPreregistration,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._preregistration = preregistration
        self._clock = clock

    def run(self, executor: SequentialEpisodeExecutor) -> SequentialRunResult:
        # This is the mandatory static preflight.  The executor contract makes
        # static_plan pure; no experimental I/O is exposed until it returns and
        # its complete upper bound fits within the frozen authorization.
        planned = executor.static_plan(self._preregistration)
        if not isinstance(planned, ResourceUsage):
            _fail("executor.static_plan must return ResourceUsage")
        if planned != self._preregistration.planned_usage:
            _fail("executor static plan differs from the hash-pinned preregistration")
        self._validate_planned_usage(planned)

        self._validate_bound_task_inputs()
        parent, loaded_parent = _load_snapshot_state(
            self._preregistration.initial_snapshot,
            label="initial K0",
        )
        if (
            loaded_parent.manifest["policy_refs"]
            != self._preregistration.runtime_binding["policy_refs"]
        ):
            _fail("initial K0 policy refs disagree with preregistered runtime binding")

        gate = QuotaGate(self._preregistration.budgets, clock=self._clock)
        metered = MeteredIO[Any](gate)
        _validate_pristine_output_root(self._preregistration)
        _claim_run_once(self._preregistration, metered_io=metered)
        learned: tuple[str, ...] = ()
        learned_knowledge: tuple[dict[str, Any], ...] = ()
        outcomes: list[EpisodeExecutionResult] = []

        for ordinal, spec in enumerate(self._preregistration.episodes):
            if ordinal == 1 and not learned and not learned_knowledge:
                break
            binding = self._preregistration.runtime_binding_for_snapshot(parent)
            request = EpisodeExecutionRequest(
                preregistration_id=self._preregistration.preregistration_id,
                ordinal=ordinal,
                role=spec.role,
                seed=spec.seed,
                task=copy.deepcopy(self._preregistration.task),
                parent_snapshot=parent,
                runtime_binding=binding,
                required_learned_entry_ids=learned,
                control_limits={
                    "environment_actions": self._preregistration.budgets.max_environment_actions_per_episode,
                    "control_turns": self._preregistration.budgets.max_control_turns_per_episode,
                    "no_progress_control_turns": self._preregistration.budgets.max_no_progress_control_turns_per_episode,
                    "semantic_rounds": self._preregistration.budgets.max_semantic_rounds_per_episode,
                },
                required_semantic_knowledge=learned_knowledge,
            )
            before = gate.usage
            outcome = executor.execute(request, metered_io=metered)
            gate.checkpoint()
            observed_delta = gate.usage.minus(before)
            self._validate_common_result(request, outcome, observed_delta)
            outcomes.append(outcome)

            if ordinal == 0:
                parent, learned, learned_knowledge = self._validate_update_barrier(
                    request, outcome
                )
            else:
                self._validate_probe(request, outcome, learned)

        if len(outcomes) not in {1, 2}:
            _fail("sequential harness did not complete its update episode")
        return SequentialRunResult(
            preregistration_id=self._preregistration.preregistration_id,
            status=(
                "completed"
                if learned or learned_knowledge
                else "update completed without accepted knowledge; probe skipped"
            ),
            update=outcomes[0],
            probe=outcomes[1] if len(outcomes) == 2 else None,
            learned_entry_ids=learned,
            final_usage=gate.usage,
            learned_knowledge=learned_knowledge,
        )

    def _validate_planned_usage(self, planned: ResourceUsage) -> None:
        budgets = self._preregistration.budgets
        caps = {
            "external_model_calls": budgets.max_external_model_calls,
            "images": budgets.max_images,
            "image_bytes": budgets.max_image_bytes,
            "artifact_bytes": budgets.max_artifact_bytes,
        }
        for field_name, cap in caps.items():
            if getattr(planned, field_name) > cap:
                _fail(f"static plan exceeds preregistered {field_name} quota")

    def _validate_bound_task_inputs(self) -> None:
        for field_name in ("task_definition", "instruction_source"):
            reference = ArtifactRef.from_mapping(
                self._preregistration.task[field_name],
                path=f"task.{field_name}",
            )
            _read_pinned_artifact(
                reference,
                label=f"task {field_name}",
                max_bytes=64 * 1024 * 1024,
            )
        _canonical_object_artifact(
            self._preregistration.launch_config,
            label="launch config",
            max_bytes=4 * 1024 * 1024,
        )

    def _validate_common_result(
        self,
        request: EpisodeExecutionRequest,
        outcome: EpisodeExecutionResult,
        observed_delta: ResourceUsage,
    ) -> None:
        if not isinstance(outcome, EpisodeExecutionResult):
            _fail("executor.execute must return EpisodeExecutionResult")
        if (
            outcome.preregistration_id != request.preregistration_id
            or outcome.ordinal != request.ordinal
            or outcome.role != request.role
            or outcome.seed != request.seed
        ):
            _fail("executor replaced the preregistered episode role, order, or seed")
        if outcome.loaded_snapshot != request.parent_snapshot:
            _fail("episode loaded a snapshot other than its exact requested parent")
        if outcome.runtime_binding_id != request.runtime_binding["binding_id"]:
            _fail(
                "episode runtime binding does not match the requested parent/provenance"
            )
        if not isinstance(outcome.semantic_newly_accepted, bool) or not isinstance(
            outcome.semantic_knowledge_used, bool
        ):
            _fail("semantic HPK result flags must be booleans")
        for value in outcome.semantic_knowledge:
            validate_semantic_knowledge(value)
        if outcome.provider_usage != observed_delta:
            _fail(
                "executor/provider usage audit does not reconcile with the local quota gate"
            )
        if observed_delta.external_model_calls < 1:
            _fail("each accepted sequential episode requires a metered external call")
        _require_artifact_within_output(
            self._preregistration,
            outcome.rollout_manifest.path,
            label="rollout manifest",
        )
        _require_artifact_within_output(
            self._preregistration,
            outcome.finalization_receipt.path,
            label="finalization receipt",
        )
        if outcome.child_snapshot is not None:
            _require_artifact_within_root(
                self._preregistration.snapshot_output_root,
                outcome.child_snapshot.manifest_path,
                label="child snapshot manifest",
            )
        usage = outcome.control_usage
        limits = request.control_limits
        for field_name in (
            "environment_actions",
            "control_turns",
            "no_progress_control_turns",
            "semantic_rounds",
        ):
            if getattr(usage, field_name) > limits[field_name]:
                _fail(f"episode exceeded {field_name} before the barrier")
        if usage.backend_retries != 0 or usage.seed_retries != 0:
            _fail("backend retry or seed retry is forbidden")
        imported, _manifest = _validate_rollout_manifest(
            outcome.rollout_manifest,
            request=request,
        )
        _require_artifact_within_output(
            self._preregistration,
            str(imported.trace_path),
            label="public rollout trace",
        )
        if imported.private_transition_trace_path is not None:
            _require_artifact_within_output(
                self._preregistration,
                str(imported.private_transition_trace_path),
                label="private transition trace",
            )
        if imported.runtime_binding != request.runtime_binding:
            _fail("strict importer returned a different runtime binding")

    @staticmethod
    def _validate_update_barrier(
        request: EpisodeExecutionRequest,
        outcome: EpisodeExecutionResult,
    ) -> tuple[
        SnapshotState,
        tuple[str, ...],
        tuple[dict[str, Any], ...],
    ]:
        claimed_child = outcome.child_snapshot
        if outcome.semantic_newly_accepted:
            semantic_records = tuple(
                validate_semantic_knowledge(value)
                for value in outcome.semantic_knowledge
            )
            accepted = tuple(
                value for value in semantic_records if value["status"] == "accepted"
            )
            if not accepted:
                _fail(
                    "semantic finalizer claimed acceptance without accepted knowledge"
                )
            _validate_finalization_receipt(
                outcome.finalization_receipt,
                rollout_manifest=outcome.rollout_manifest,
                request=request,
                expected_status=outcome.finalization_status,
                child=None,
            )
            return request.parent_snapshot, (), accepted
        if outcome.finalization_status != "published" or claimed_child is None:
            _validate_finalization_receipt(
                outcome.finalization_receipt,
                rollout_manifest=outcome.rollout_manifest,
                request=request,
                expected_status=outcome.finalization_status,
                child=None,
            )
            return request.parent_snapshot, (), ()
        child, loaded_child = _load_snapshot_state(
            claimed_child,
            label="update child K1",
        )
        if loaded_child.manifest["parent"] != request.parent_snapshot.identity():
            _fail("published child parent does not match the update K0 lease")
        if child.identity() == request.parent_snapshot.identity():
            _fail("update finalizer returned the unchanged parent as its child")
        if not set(request.parent_snapshot.accepted_entry_ids).issubset(
            child.accepted_entry_ids
        ):
            _fail("child removed a previously accepted entry")
        new_accepted = tuple(
            sorted(
                set(child.accepted_entry_ids)
                - set(request.parent_snapshot.accepted_entry_ids)
            )
        )
        receipt = _validate_finalization_receipt(
            outcome.finalization_receipt,
            rollout_manifest=outcome.rollout_manifest,
            request=request,
            expected_status="published",
            child=child,
        )
        updated_ids = set(receipt["finalization"]["updated_entry_ids"])
        learned_candidates = set(new_accepted) & updated_ids
        learned = tuple(
            sorted(
                str(entry["entry_id"])
                for entry in loaded_child.accepted_entries
                if str(entry["entry_id"]) in learned_candidates
                and entry["provenance"]["source_kind"] == "agent_generated"
                and entry["provenance"].get("learned_hpk", entry["provenance"].get("learned_afk")) is True
                and entry["provenance"]["oracle_derived"] is False
                and entry["provenance"]["expert_derived"] is False
            )
        )
        return child, learned, ()

    @staticmethod
    def _validate_probe(
        request: EpisodeExecutionRequest,
        outcome: EpisodeExecutionResult,
        learned: tuple[str, ...],
    ) -> None:
        if request.required_learned_entry_ids != learned:
            _fail("probe request lost the update episode's learned-entry barrier")
        imported, _manifest = _validate_rollout_manifest(
            outcome.rollout_manifest,
            request=request,
        )
        if request.required_semantic_knowledge:
            required = tuple(
                validate_semantic_knowledge(value)
                for value in request.required_semantic_knowledge
            )
            observed = tuple(
                validate_semantic_knowledge(value)
                for value in outcome.semantic_knowledge
            )
            if not all(value in observed for value in required):
                _fail("probe lost the accepted semantic knowledge")
            if outcome.semantic_knowledge_used is not True:
                _fail("probe did not use accepted semantic knowledge")
            if not any(
                transition["physical_action_executed"] is True
                and transition["motion_status"] == "completed"
                for transition in imported.transitions
            ):
                _fail("probe semantic knowledge did not reach a physical action")
            claimed_child = outcome.child_snapshot
            validated_child = None
            if claimed_child is not None:
                validated_child, _loaded = _load_snapshot_state(
                    claimed_child,
                    label="probe finalization child",
                )
            _validate_finalization_receipt(
                outcome.finalization_receipt,
                rollout_manifest=outcome.rollout_manifest,
                request=request,
                expected_status=outcome.finalization_status,
                child=validated_child,
            )
            return
        _probe_usage_proof(
            imported,
            request=request,
            learned_entry_ids=learned,
        )
        claimed_child = outcome.child_snapshot
        validated_child = None
        if claimed_child is not None:
            validated_child, _loaded = _load_snapshot_state(
                claimed_child,
                label="probe finalization child",
            )
        _validate_finalization_receipt(
            outcome.finalization_receipt,
            rollout_manifest=outcome.rollout_manifest,
            request=request,
            expected_status=outcome.finalization_status,
            child=validated_child,
        )


class IncrementalSequentialPairGate:
    """Incremental form used by a real evaluator that owns one model instance.

    The evaluator calls :meth:`begin_episode`, performs exactly that episode
    through the exposed :class:`MeteredIO`, and then calls
    :meth:`complete_episode`.  The second request does not exist until the
    update barrier has loaded and validated K1.
    """

    def __init__(
        self,
        preregistration: SequentialPreregistration,
        *,
        planned_usage: ResourceUsage,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._validator = SequentialPairHarness(preregistration, clock=clock)
        self._preregistration = preregistration
        self._validator._validate_planned_usage(planned_usage)
        if planned_usage != preregistration.planned_usage:
            _fail("controller static plan differs from the hash-pinned preregistration")
        self._validator._validate_bound_task_inputs()
        parent, loaded_parent = _load_snapshot_state(
            preregistration.initial_snapshot,
            label="initial K0",
        )
        if (
            loaded_parent.manifest["policy_refs"]
            != preregistration.runtime_binding["policy_refs"]
        ):
            _fail("initial K0 policy refs disagree with preregistered runtime binding")
        _validate_pristine_output_root(preregistration)
        self._parent = parent
        self._learned: tuple[str, ...] = ()
        self._learned_knowledge: tuple[dict[str, Any], ...] = ()
        self._outcomes: list[EpisodeExecutionResult] = []
        self._gate = QuotaGate(preregistration.budgets, clock=clock)
        self._metered_io = MeteredIO[Any](self._gate)
        self._active_request: EpisodeExecutionRequest | None = None
        self._usage_before_episode: ResourceUsage | None = None
        self._artifact_capacity_reserved = False
        self._run_claim: ArtifactRef | None = None

    @property
    def metered_io(self) -> MeteredIO[Any]:
        return self._metered_io

    @property
    def completed_episode_count(self) -> int:
        return len(self._outcomes)

    @property
    def should_stop(self) -> bool:
        """True after update when no accepted knowledge exists for a probe."""

        return bool(
            len(self._outcomes) == 1
            and not self._learned
            and not self._learned_knowledge
        )

    def reserve_planned_artifact_capacity(self) -> None:
        """Reserve the static artifact upper bound before evaluator output I/O."""

        if self._active_request is not None or self._outcomes:
            _fail("artifact capacity must be reserved before the first episode")
        if self._artifact_capacity_reserved:
            _fail("artifact capacity was already reserved")
        self._gate.reserve_artifact_capacity(
            self._preregistration.planned_usage.artifact_bytes
        )
        self._artifact_capacity_reserved = True

    def claim_run_once(self) -> ArtifactRef:
        if not self._artifact_capacity_reserved:
            _fail("artifact capacity must be reserved before the run claim")
        if self._active_request is not None or self._outcomes:
            _fail("run claim must be published before the first episode")
        if self._run_claim is not None:
            _fail("run claim was already published")
        self._run_claim = _claim_run_once(
            self._preregistration,
            metered_io=self._metered_io,
        )
        return self._run_claim

    def active_usage_delta(self) -> ResourceUsage:
        if self._active_request is None or self._usage_before_episode is None:
            _fail("usage delta requires one active episode")
        return self._gate.usage.minus(self._usage_before_episode)

    def begin_episode(self) -> EpisodeExecutionRequest:
        if self._run_claim is None:
            _fail("run claim must be published before the first episode")
        if self._active_request is not None:
            _fail("one-worker gate already has an active episode")
        ordinal = len(self._outcomes)
        if self.should_stop:
            _fail("update produced no accepted knowledge; probe was skipped")
        if ordinal >= len(self._preregistration.episodes):
            _fail("the exact two-episode schedule is already complete")
        if ordinal == 1 and not self._learned and not self._learned_knowledge:
            _fail("probe cannot start before update accepts new HPK knowledge")
        spec = self._preregistration.episodes[ordinal]
        binding = self._preregistration.runtime_binding_for_snapshot(self._parent)
        request = EpisodeExecutionRequest(
            preregistration_id=self._preregistration.preregistration_id,
            ordinal=ordinal,
            role=spec.role,
            seed=spec.seed,
            task=copy.deepcopy(self._preregistration.task),
            parent_snapshot=self._parent,
            runtime_binding=binding,
            required_learned_entry_ids=self._learned,
            control_limits={
                "environment_actions": self._preregistration.budgets.max_environment_actions_per_episode,
                "control_turns": self._preregistration.budgets.max_control_turns_per_episode,
                "no_progress_control_turns": self._preregistration.budgets.max_no_progress_control_turns_per_episode,
                "semantic_rounds": self._preregistration.budgets.max_semantic_rounds_per_episode,
            },
            required_semantic_knowledge=self._learned_knowledge,
        )
        self._active_request = request
        self._usage_before_episode = self._gate.usage
        return request

    def complete_episode(
        self, outcome: EpisodeExecutionResult
    ) -> EpisodeExecutionResult:
        request = self._active_request
        usage_before = self._usage_before_episode
        if request is None or usage_before is None:
            _fail("complete_episode requires the one active episode request")
        self._gate.checkpoint()
        delta = self._gate.usage.minus(usage_before)
        self._validator._validate_common_result(request, outcome, delta)
        if request.ordinal == 0:
            (
                self._parent,
                self._learned,
                self._learned_knowledge,
            ) = self._validator._validate_update_barrier(request, outcome)
        else:
            self._validator._validate_probe(request, outcome, self._learned)
        self._outcomes.append(outcome)
        self._active_request = None
        self._usage_before_episode = None
        return outcome

    def result(self) -> SequentialRunResult:
        if self._active_request is not None or (
            len(self._outcomes) != 2 and not self.should_stop
        ):
            _fail("sequential result is unavailable before the run is complete")
        return SequentialRunResult(
            preregistration_id=self._preregistration.preregistration_id,
            status=(
                "completed"
                if self._learned or self._learned_knowledge
                else "update completed without accepted knowledge; probe skipped"
            ),
            update=self._outcomes[0],
            probe=self._outcomes[1] if len(self._outcomes) == 2 else None,
            learned_entry_ids=self._learned,
            final_usage=self._gate.usage,
            learned_knowledge=self._learned_knowledge,
        )


__all__ = [
    "PREREGISTRATION_SCHEMA",
    "RUN_RESULT_SCHEMA",
    "AcceptanceProfile",
    "ArtifactRef",
    "ControlUsage",
    "EpisodeExecutionRequest",
    "EpisodeExecutionResult",
    "EpisodeSpec",
    "ExperimentBudgets",
    "IncrementalSequentialPairGate",
    "MeteredIO",
    "PublishedPreregistration",
    "QuotaGate",
    "ResourceUsage",
    "SequentialEpisodeExecutor",
    "SequentialExperimentError",
    "SequentialPairHarness",
    "SequentialPreregistration",
    "SequentialQuotaExceeded",
    "SequentialRunResult",
    "SnapshotState",
    "build_preregistration",
    "load_acceptance_profile",
    "load_preregistration",
    "publish_preregistration",
]
