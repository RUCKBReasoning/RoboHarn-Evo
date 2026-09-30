from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roboharn_evo.agent.hpk.schemas import HPKValidationError
from roboharn_evo.agent.hpk.compatibility import normalize_agent_knowledge_config

if TYPE_CHECKING:
    from roboharn_evo.agent.hpk.store import LoadedHPKSnapshot


HPK_OFF_MODE = "off"
HPK_OBSERVE_MODE = "observe"
HPK_STATIC_MODE = "static"
HPK_EVOLVING_MODE = "evolving"
P0A_MODES = frozenset({HPK_OFF_MODE, HPK_OBSERVE_MODE})
P0C_MODES = frozenset({HPK_OFF_MODE, HPK_OBSERVE_MODE, HPK_STATIC_MODE})
P0DE_MODES = frozenset(
    {HPK_OFF_MODE, HPK_OBSERVE_MODE, HPK_STATIC_MODE, HPK_EVOLVING_MODE}
)

DEFAULT_MAX_PROMPT_CHARS = 4000
DEFAULT_RUN_SCOPE = "formal_no_prior"
RUN_SCOPES = frozenset(
    {"formal_no_prior", "integration", "expert_prior", "oracle_diagnostic"}
)

_OBSERVE_CONFIG_KEYS = frozenset(
    {
        "mode",
        "allow_oracle_evidence",
    }
)

_STATIC_CONFIG_KEYS = frozenset(
    {
        "mode",
        "snapshot_manifest",
        "expected_manifest_sha256",
        "run_scope",
        "max_prompt_chars",
        "allow_expert_prior",
        "allow_human_integration_prior",
        "allow_oracle_evidence",
        "all_hard_mismatch_behavior",
        "geometry_policy_path",
        "expected_geometry_policy_sha256",
    }
)

_EVOLVING_CONFIG_KEYS = frozenset(
    {
        "mode",
        "snapshot_manifest",
        "expected_manifest_sha256",
        "run_scope",
        "max_prompt_chars",
        "allow_expert_prior",
        "allow_human_integration_prior",
        "allow_oracle_evidence",
        "all_hard_mismatch_behavior",
        "geometry_policy_path",
        "expected_geometry_policy_sha256",
        "promotion_policy_path",
        "expected_promotion_policy_sha256",
        "snapshot_output_root",
        "proposer",
    }
)


def _hpk_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise HPKValidationError("agent.hpk must be an object")
    payload = dict(value)
    if "agent" in payload and isinstance(payload["agent"], Mapping):
        payload = dict(payload["agent"])
    payload = normalize_agent_knowledge_config(payload)
    if "hpk" in payload and isinstance(payload["hpk"], Mapping):
        payload = dict(payload["hpk"])
    return payload


@dataclass(frozen=True, slots=True)
class HPKRuntimePolicy:
    """HPK 配置与运行模式的能力约定。"""

    mode: str = HPK_OFF_MODE
    snapshot_manifest: str = ""
    expected_manifest_sha256: str = ""
    run_scope: str = DEFAULT_RUN_SCOPE
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS
    allow_expert_prior: bool = False
    allow_human_integration_prior: bool = False
    allow_oracle_evidence: bool = False
    all_hard_mismatch_behavior: str = "fail_closed"
    geometry_policy_path: str = ""
    expected_geometry_policy_sha256: str = ""
    promotion_policy_path: str = ""
    expected_promotion_policy_sha256: str = ""
    snapshot_output_root: str = ""
    proposer_enabled: bool = False
    proposer_max_proposals: int = 1
    proposer_trigger: str = "verified_oppose"
    proposer_scope: str = "geometry"

    def __post_init__(self) -> None:
        if self.mode not in P0DE_MODES:
            raise HPKValidationError(
                f"HPK mode {self.mode!r} is unavailable; expected "
                "'off', 'observe', 'static', or 'evolving'"
            )
        for name in (
            "allow_expert_prior",
            "allow_human_integration_prior",
            "allow_oracle_evidence",
        ):
            if not isinstance(getattr(self, name), bool):
                raise HPKValidationError(f"{name} must be a boolean")
        if not isinstance(self.proposer_enabled, bool):
            raise HPKValidationError("proposer_enabled must be a boolean")
        if (
            isinstance(self.proposer_max_proposals, bool)
            or not isinstance(self.proposer_max_proposals, int)
            or self.proposer_max_proposals not in {1, 2}
        ):
            raise HPKValidationError("proposer_max_proposals must be 1 or 2")
        if self.proposer_trigger != "verified_oppose":
            raise HPKValidationError(
                "P0 HPK v2 proposer trigger must be verified_oppose"
            )
        if self.proposer_scope != "geometry":
            raise HPKValidationError("P0 HPK v2 proposer scope must be geometry")
        if isinstance(self.max_prompt_chars, bool) or not isinstance(
            self.max_prompt_chars, int
        ):
            raise HPKValidationError("max_prompt_chars must be an integer")
        if self.max_prompt_chars < 1:
            raise HPKValidationError("max_prompt_chars must be >= 1")
        if self.mode == HPK_OFF_MODE and (
            self.snapshot_manifest
            or self.expected_manifest_sha256
            or self.allow_expert_prior
            or self.allow_human_integration_prior
            or self.allow_oracle_evidence
            or self.geometry_policy_path
            or self.expected_geometry_policy_sha256
            or self.promotion_policy_path
            or self.expected_promotion_policy_sha256
            or self.snapshot_output_root
        ):
            raise HPKValidationError("off mode cannot activate HPK configuration")
        if self.mode == HPK_OBSERVE_MODE and (
            self.snapshot_manifest
            or self.expected_manifest_sha256
            or self.allow_expert_prior
            or self.allow_human_integration_prior
            or self.geometry_policy_path
            or self.expected_geometry_policy_sha256
            or self.promotion_policy_path
            or self.expected_promotion_policy_sha256
            or self.snapshot_output_root
        ):
            raise HPKValidationError("observe mode cannot activate snapshot retrieval")
        if self.mode in {HPK_STATIC_MODE, HPK_EVOLVING_MODE}:
            if not self.snapshot_manifest:
                raise HPKValidationError(
                    "active HPK requires an explicit snapshot_manifest"
                )
            if not Path(self.snapshot_manifest).is_absolute():
                raise HPKValidationError("snapshot_manifest must be an absolute path")
            if not _is_sha256(self.expected_manifest_sha256):
                raise HPKValidationError(
                    "expected_manifest_sha256 must be a lowercase SHA-256 digest"
                )
            if self.run_scope not in RUN_SCOPES:
                raise HPKValidationError(
                    f"run_scope must be one of {sorted(RUN_SCOPES)}"
                )
            if self.all_hard_mismatch_behavior != "fail_closed":
                raise HPKValidationError(
                    "P0-C static all_hard_mismatch_behavior must be 'fail_closed'"
                )
        if bool(self.geometry_policy_path) != bool(
            self.expected_geometry_policy_sha256
        ):
            raise HPKValidationError(
                "geometry_policy_path and expected_geometry_policy_sha256 "
                "must be supplied together"
            )
        if self.geometry_policy_path:
            if not Path(self.geometry_policy_path).is_absolute():
                raise HPKValidationError("geometry_policy_path must be absolute")
            if not _is_sha256(self.expected_geometry_policy_sha256):
                raise HPKValidationError(
                    "expected_geometry_policy_sha256 must be lowercase SHA-256"
                )
        if self.mode == HPK_EVOLVING_MODE:
            if self.run_scope not in {"formal_no_prior", "integration"}:
                raise HPKValidationError(
                    "evolving run_scope must be formal_no_prior or integration"
                )
            if (
                self.allow_expert_prior
                or self.allow_human_integration_prior
                or self.allow_oracle_evidence
            ):
                raise HPKValidationError("P0-D/E evolving is no-prior and no-oracle")
            for name, value in (
                ("promotion_policy_path", self.promotion_policy_path),
                ("snapshot_output_root", self.snapshot_output_root),
            ):
                if not value or not Path(value).is_absolute():
                    raise HPKValidationError(f"{name} must be an absolute path")
            if not _is_sha256(self.expected_promotion_policy_sha256):
                raise HPKValidationError(
                    "expected_promotion_policy_sha256 must be lowercase SHA-256"
                )
        elif self.proposer_enabled:
            raise HPKValidationError("HPK v2 proposer requires evolving mode")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "HPKRuntimePolicy":
        payload = _hpk_mapping(value)
        mode = str(payload.get("mode", HPK_OFF_MODE) or HPK_OFF_MODE).strip().lower()
        if mode not in P0DE_MODES:
            raise HPKValidationError(
                f"HPK mode {mode!r} is unavailable; legacy modes are unsupported"
            )
        # Off 模式不会读取其他 HPK 配置。
        # or validated, so stale snapshot paths cannot cause I/O or behavior
        # changes merely by being present in a disabled configuration.
        if mode == HPK_OFF_MODE:
            return cls()

        if mode == HPK_OBSERVE_MODE:
            unknown = sorted(set(payload) - _OBSERVE_CONFIG_KEYS)
            if unknown:
                raise HPKValidationError(
                    "agent.hpk contains unknown field(s): " + ", ".join(unknown)
                )
            allow_oracle = payload.get("allow_oracle_evidence", False)
            if not isinstance(allow_oracle, bool):
                raise HPKValidationError("allow_oracle_evidence must be a boolean")
            return cls(mode=HPK_OBSERVE_MODE, allow_oracle_evidence=allow_oracle)

        allowed_keys = (
            _EVOLVING_CONFIG_KEYS if mode == HPK_EVOLVING_MODE else _STATIC_CONFIG_KEYS
        )
        unknown = sorted(set(payload) - allowed_keys)
        if unknown:
            raise HPKValidationError(
                "agent.hpk contains unknown field(s): " + ", ".join(unknown)
            )
        if (
            not isinstance(payload.get("snapshot_manifest"), str)
            or not str(payload.get("snapshot_manifest", "")).strip()
        ):
            raise HPKValidationError(
                "active HPK requires an explicit snapshot_manifest"
            )
        proposer = _proposer_config(payload.get("proposer"))
        return cls(
            mode=mode,
            snapshot_manifest=_required_string(
                payload.get("snapshot_manifest"), label="snapshot_manifest"
            ),
            expected_manifest_sha256=_required_string(
                payload.get("expected_manifest_sha256"),
                label="expected_manifest_sha256",
            ),
            run_scope=_optional_string(
                payload.get("run_scope", DEFAULT_RUN_SCOPE), label="run_scope"
            ),
            max_prompt_chars=_positive_integer(
                payload.get("max_prompt_chars", DEFAULT_MAX_PROMPT_CHARS),
                label="max_prompt_chars",
            ),
            allow_expert_prior=_strict_bool(
                payload.get("allow_expert_prior", False),
                label="allow_expert_prior",
            ),
            allow_human_integration_prior=_strict_bool(
                payload.get("allow_human_integration_prior", False),
                label="allow_human_integration_prior",
            ),
            allow_oracle_evidence=_strict_bool(
                payload.get("allow_oracle_evidence", False),
                label="allow_oracle_evidence",
            ),
            all_hard_mismatch_behavior=_optional_string(
                payload.get("all_hard_mismatch_behavior", "fail_closed"),
                label="all_hard_mismatch_behavior",
            ),
            geometry_policy_path=_optional_path_string(
                payload.get("geometry_policy_path", ""),
                label="geometry_policy_path",
            ),
            expected_geometry_policy_sha256=_optional_digest_string(
                payload.get("expected_geometry_policy_sha256", ""),
                label="expected_geometry_policy_sha256",
            ),
            promotion_policy_path=_optional_path_string(
                payload.get("promotion_policy_path", ""),
                label="promotion_policy_path",
            ),
            expected_promotion_policy_sha256=_optional_digest_string(
                payload.get("expected_promotion_policy_sha256", ""),
                label="expected_promotion_policy_sha256",
            ),
            snapshot_output_root=_optional_path_string(
                payload.get("snapshot_output_root", ""),
                label="snapshot_output_root",
            ),
            proposer_enabled=proposer["enabled"],
            proposer_max_proposals=proposer["max_proposals"],
            proposer_trigger=proposer["trigger"],
            proposer_scope=proposer["scope"],
        )

    @property
    def enabled(self) -> bool:
        """Compatibility flag for the P0-A observe sidecar only."""

        return self.mode == HPK_OBSERVE_MODE

    @property
    def static_enabled(self) -> bool:
        return self.mode == HPK_STATIC_MODE

    @property
    def evolving_enabled(self) -> bool:
        return self.mode == HPK_EVOLVING_MODE

    @property
    def constructs_records(self) -> bool:
        return self.mode in {HPK_OBSERVE_MODE, HPK_STATIC_MODE, HPK_EVOLVING_MODE}

    @property
    def emits_trace_audit(self) -> bool:
        return self.constructs_records

    @property
    def retrieves(self) -> bool:
        return self.static_enabled or self.evolving_enabled

    @property
    def changes_planner_input(self) -> bool:
        return self.static_enabled or self.evolving_enabled

    @property
    def changes_candidate_selection(self) -> bool:
        return self.static_enabled or self.evolving_enabled

    @property
    def writes_persistent_store(self) -> bool:
        return self.evolving_enabled

    def permits_geometry_source(self, geometry_source_class: str) -> bool:
        source = str(geometry_source_class or "").strip()
        if source == "oracle":
            return self.constructs_records and self.allow_oracle_evidence
        return self.constructs_records and source in {
            "rgbd_observed",
            "runtime_relational",
            "unknown",
        }

    def to_dict(self) -> dict[str, Any]:
        if self.mode == HPK_OFF_MODE:
            return {"mode": HPK_OFF_MODE}
        if self.mode == HPK_OBSERVE_MODE:
            return {
                "mode": HPK_OBSERVE_MODE,
                "allow_oracle_evidence": self.allow_oracle_evidence,
                "retrieval_enabled": False,
                "planner_change_enabled": False,
                "candidate_change_enabled": False,
                "persistent_store_enabled": False,
            }
        payload = {
            "mode": self.mode,
            "snapshot_manifest": self.snapshot_manifest,
            "expected_manifest_sha256": self.expected_manifest_sha256,
            "run_scope": self.run_scope,
            "max_prompt_chars": self.max_prompt_chars,
            "allow_expert_prior": self.allow_expert_prior,
            "allow_human_integration_prior": self.allow_human_integration_prior,
            "allow_oracle_evidence": self.allow_oracle_evidence,
            "all_hard_mismatch_behavior": self.all_hard_mismatch_behavior,
            "retrieval_enabled": True,
            "planner_change_enabled": True,
            "candidate_change_enabled": True,
            "persistent_store_enabled": self.evolving_enabled,
        }
        if self.geometry_policy_path:
            payload.update(
                {
                    "geometry_policy_path": self.geometry_policy_path,
                    "expected_geometry_policy_sha256": (
                        self.expected_geometry_policy_sha256
                    ),
                }
            )
        if self.evolving_enabled:
            payload.update(
                {
                    "promotion_policy_path": self.promotion_policy_path,
                    "expected_promotion_policy_sha256": (
                        self.expected_promotion_policy_sha256
                    ),
                    "snapshot_output_root": self.snapshot_output_root,
                }
            )
            if self.proposer_enabled:
                payload["proposer"] = {
                    "enabled": True,
                    "max_proposals": self.proposer_max_proposals,
                    "trigger": self.proposer_trigger,
                    "scope": self.proposer_scope,
                }
        return payload

    def load_static_snapshot(self) -> LoadedHPKSnapshot:
        """Perform the one explicit load requested by a higher-level runtime factory."""

        if not self.static_enabled:
            raise HPKValidationError("only static mode can load an HPK snapshot")
        from roboharn_evo.agent.hpk.store import load_hpk_snapshot

        return load_hpk_snapshot(
            self.snapshot_manifest,
            self.expected_manifest_sha256,
            run_scope=self.run_scope,
            allow_expert_prior=self.allow_expert_prior,
            allow_human_integration_prior=self.allow_human_integration_prior,
            allow_oracle_evidence=self.allow_oracle_evidence,
        )

    def load_geometry_policy(self):
        from roboharn_evo.agent.hpk.policy_config import (
            load_default_geometry_policy,
            load_evolving_geometry_policy,
            load_policy_config,
        )

        if not self.retrieves:
            raise HPKValidationError("only static/evolving may load geometry policy")
        if not self.geometry_policy_path:
            return (
                load_evolving_geometry_policy()
                if self.evolving_enabled
                else load_default_geometry_policy()
            )
        return load_policy_config(
            self.geometry_policy_path,
            self.expected_geometry_policy_sha256,
            "geometry",
        )

    def load_promotion_policy(self):
        if not self.evolving_enabled:
            raise HPKValidationError("only evolving may load promotion policy")
        from roboharn_evo.agent.hpk.policy_config import load_policy_config

        return load_policy_config(
            self.promotion_policy_path,
            self.expected_promotion_policy_sha256,
            "promotion",
        )

    def load_evolving_snapshot(self):
        if not self.evolving_enabled:
            raise HPKValidationError("only evolving may load SnapshotV2")
        from roboharn_evo.agent.hpk.evolving_store import load_evolving_snapshot

        return load_evolving_snapshot(
            self.snapshot_manifest,
            self.expected_manifest_sha256,
        )


def _strict_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise HPKValidationError(f"{label} must be a boolean")
    return value


def _proposer_config(value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "enabled": False,
            "max_proposals": 1,
            "trigger": "verified_oppose",
            "scope": "geometry",
        }
    if not isinstance(value, Mapping):
        raise HPKValidationError("agent.hpk.proposer must be an object")
    expected = {"enabled", "max_proposals", "trigger", "scope"}
    if set(value) != expected:
        raise HPKValidationError(
            "agent.hpk.proposer fields mismatch: "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    enabled = _strict_bool(value["enabled"], label="proposer.enabled")
    max_proposals = value["max_proposals"]
    if (
        isinstance(max_proposals, bool)
        or not isinstance(max_proposals, int)
        or max_proposals not in {1, 2}
    ):
        raise HPKValidationError("proposer.max_proposals must be 1 or 2")
    trigger = _required_string(value["trigger"], label="proposer.trigger")
    scope = _required_string(value["scope"], label="proposer.scope")
    return {
        "enabled": enabled,
        "max_proposals": max_proposals,
        "trigger": trigger,
        "scope": scope,
    }


def _required_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HPKValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, *, label: str) -> str:
    return _required_string(value, label=label)


def _optional_path_string(value: Any, *, label: str) -> str:
    if value is None or value == "":
        return ""
    return _required_string(value, label=label)


def _optional_digest_string(value: Any, *, label: str) -> str:
    if value is None or value == "":
        return ""
    return _required_string(value, label=label)


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HPKValidationError(f"{label} must be an integer >= 1")
    return value


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def parse_hpk_runtime_policy(
    value: Mapping[str, Any] | None,
) -> HPKRuntimePolicy:
    return HPKRuntimePolicy.from_mapping(value)


__all__ = [
    "HPK_OBSERVE_MODE",
    "HPK_OFF_MODE",
    "HPK_EVOLVING_MODE",
    "HPK_STATIC_MODE",
    "HPKRuntimePolicy",
    "DEFAULT_MAX_PROMPT_CHARS",
    "DEFAULT_RUN_SCOPE",
    "P0A_MODES",
    "P0C_MODES",
    "P0DE_MODES",
    "RUN_SCOPES",
    "parse_hpk_runtime_policy",
]
