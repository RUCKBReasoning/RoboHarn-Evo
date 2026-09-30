from __future__ import annotations

from typing import Any, Callable

from roboharn_evo.agent.components.agent_tools.local_skill_registry import LocalSkillRegistry
from roboharn_evo.models.backend_factory import build_recovery_backend
from roboharn_evo.models.backend_interfaces import RecoveryBackend
from roboharn_evo.models.agent_api_recovery_adapter import AgentApiRecoveryAdapter, AgentApiRecoveryConfig


_VALID_EFFECT_VALUES = {"true", "false", "unverified"}
_VALID_SUBTASK_STATUS_VALUES = {"in_progress", "completed", "failed", "uncertain"}
_VALID_RECOMMENDED_CONTROL_VALUES = {"continue", "retry", "replan"}


class ActionEffectVerifierBackendError(RuntimeError):
    """Infrastructure failure while requesting an action-effect judgment."""


class ActionEffectVerifier:
    """VLM-backed verifier for recovery action effects.

    The verifier receives a temporary before/after evidence payload and returns a
    compact judgment. It does not own persistent memory fields.
    """

    def __init__(
        self,
        *,
        skill_registry: LocalSkillRegistry,
        backend_config: dict[str, Any] | None = None,
        backend: RecoveryBackend | None = None,
        request_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._skill_registry = skill_registry
        self._backend_config = dict(backend_config or {})
        self._backend = backend
        self._request_observer = request_observer

    def verify(self, evidence_payload: dict[str, Any]) -> dict[str, Any]:
        skill = self._skill_registry.get_skill("action-effect-verification", refresh=True)
        if skill is None:
            return self._fallback("verifier_skill_missing")
        payload = {
            "mode": "action_effect_verification",
            "evidence_payload": dict(evidence_payload),
            "verification_skill": {
                "name": skill.name,
                "description": skill.description,
                "body": skill.body,
                "path": str(skill.skill_md_path),
            },
        }
        try:
            backend = self._backend_for_skill(skill.body)
            self._notify_request_observer(payload)
            raw = backend.plan_recovery(recovery_payload=payload)
        except Exception as exc:
            raise ActionEffectVerifierBackendError(
                f"action-effect verifier unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        return self._normalize(raw)

    def _notify_request_observer(self, payload: dict[str, Any]) -> None:
        observer = self._request_observer
        if observer is None:
            return
        try:
            observer(payload)
        except Exception:
            # The audit sidecar is fail-open by contract.
            pass

    def _normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return self._fallback("invalid_verifier_response")
        effect = str(raw.get("effect_verified", raw.get("effect", "unverified"))).strip().lower()
        if effect not in _VALID_EFFECT_VALUES:
            effect = "unverified"
        subtask_status = self._enum_value(
            raw.get("subtask_status"),
            allowed=_VALID_SUBTASK_STATUS_VALUES,
            default="uncertain",
        )
        recommended_control = self._enum_value(
            raw.get("recommended_control"),
            allowed=_VALID_RECOMMENDED_CONTROL_VALUES,
            default="retry",
        )
        return {
            "effect_verified": effect,
            "effect_type": self._short_text(raw.get("effect_type", "unknown")) or "unknown",
            "confidence": self._confidence(raw.get("confidence")),
            "evidence_summary": self._short_text(raw.get("evidence_summary", "")),
            "failure_reason": self._short_text(raw.get("failure_reason", "")),
            "next_constraint": self._short_text(raw.get("next_constraint", "")),
            "memory_update": self._short_text(raw.get("memory_update", "")),
            "subtask_status": subtask_status,
            "recommended_control": recommended_control,
        }

    def _backend_for_skill(self, skill_prompt: str) -> RecoveryBackend:
        if self._backend is not None:
            return self._backend
        return self._build_backend(self._backend_config, skill_prompt=skill_prompt)

    def _build_backend(self, backend_config: dict[str, Any], *, skill_prompt: str) -> RecoveryBackend:
        backend_name = str(backend_config.get("backend", "agent_api"))
        if backend_name == "agent_api":
            agent_cfg = dict(backend_config.get("agent_api", backend_config))
            return AgentApiRecoveryAdapter(
                AgentApiRecoveryConfig(
                    server_url=str(agent_cfg.get("server_url", "http://127.0.0.1:9101/recover")),
                    timeout_sec=int(agent_cfg.get("timeout_sec", 120)),
                    prompt_template=skill_prompt,
                    auth_token=str(agent_cfg.get("auth_token", "")),
                    auth_header=str(agent_cfg.get("auth_header", "Authorization")),
                    extra_headers=dict(agent_cfg.get("extra_headers", {})),
                    extra_body=dict(agent_cfg.get("extra_body", {})),
                )
            )
        return build_recovery_backend(backend_config)

    def _fallback(self, reason: str) -> dict[str, Any]:
        return {
            "effect_verified": "unverified",
            "effect_type": "unknown",
            "confidence": 0.0,
            "evidence_summary": "",
            "failure_reason": self._short_text(reason),
            "next_constraint": "do not assume the recovery action achieved its intended effect",
            "memory_update": f"action effect unverified: {self._short_text(reason)}",
            "subtask_status": "uncertain",
            "recommended_control": "retry",
        }

    @staticmethod
    def _enum_value(value: Any, *, allowed: set[str], default: str) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        return normalized if normalized in allowed else default

    def _confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        if confidence < 0.0:
            return 0.0
        if confidence > 1.0:
            return 1.0
        return confidence

    def _short_text(self, value: Any, *, limit: int = 240) -> str:
        text = " ".join(str(value or "").strip().split())
        return text[:limit]
