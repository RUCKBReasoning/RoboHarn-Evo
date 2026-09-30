"""Minimal VLM-driven GeometryStrategy proposal over compact verified evidence."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, parse, request

from roboharn_evo.agent.hpk.schemas import canonical_json_bytes
from roboharn_evo.agent.hpk.strategy_proposal import (
    STRATEGY_PROPOSAL_SCHEMA,
    VLM_GEOMETRY_PROPOSER_VERSION,
    ProposalEvidencePacketV1,
    StrategyDeltaV1,
    StrategyProposalV1,
    StrategyProposalValidationError,
    proposal_id_for,
    strategy_proposal_output_json_schema,
    validate_delta_capability_and_non_equivalence,
)

_PROMPT = """You are proposing the next manipulation geometry strategy to TEST, not declaring a fact.

Use only the compact verified evidence packet supplied by the user. Return at most {max_proposals} non-equivalent geometry deltas. Change only fields explicitly listed in available_capabilities.geometry_fields. Do not change the task strategy or expected physical effect.

Do not output coordinates, poses, candidate IDs, joint commands, source paths, new tools, or absolute geometry. A proposal is only a hypothesis and will not become knowledge without later physical verification. Return strict JSON matching the supplied output schema. Return {{\"proposals\":[]}} when no grounded alternative is justified."""


class StrategyProposalBackend(Protocol):
    """Existing structured GPT backend seam used by the offline proposer."""

    def complete(
        self,
        *,
        instructions: str,
        input_text: str,
        output_schema: Mapping[str, Any],
        schema_name: str,
    ) -> ProposalBackendCompletion: ...


@dataclass(frozen=True, slots=True)
class ProposalBackendCompletion:
    output: Mapping[str, Any] | str
    audit: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ProposalCallResult:
    packet_id: str
    proposer_version: str
    backend_called: bool
    raw_response: dict[str, Any] | str | None
    proposals: tuple[StrategyProposalV1, ...]
    validation_errors: tuple[str, ...]
    backend_audit: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "proposer_version": self.proposer_version,
            "backend_called": self.backend_called,
            "raw_response": copy.deepcopy(self.raw_response),
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "validation_errors": list(self.validation_errors),
            "backend_audit": copy.deepcopy(self.backend_audit),
        }


class ExistingGPTTransportProposalBackend:
    """Adapter over the repository's existing structured GPT transport.

    The wrapped object is the existing
    ``OpenAICompatibleMultimodalTransport``.  Its already-configured
    authorization and capability preflight remain outside this module; this
    adapter only reuses ``complete_text`` and adds no service or HTTP stack.
    """

    def __init__(self, transport: Any, *, authorization: Any) -> None:
        if not callable(getattr(transport, "complete_text", None)):
            raise TypeError("transport must expose complete_text")
        self._transport = transport
        self._authorization = authorization

    def complete(
        self,
        *,
        instructions: str,
        input_text: str,
        output_schema: Mapping[str, Any],
        schema_name: str,
    ) -> ProposalBackendCompletion:
        completion = self._transport.complete_text(
            instructions=instructions,
            input_text=input_text,
            output_schema=output_schema,
            schema_name=schema_name,
            authorization=self._authorization,
            purpose="hpk_geometry_strategy_proposal",
        )
        return ProposalBackendCompletion(
            output=completion.output,
            audit=completion.audit,
        )


class AgentApiStrategyProposalBackend:
    """Reuse the rollout's existing Agent API service for one text proposal.

    The endpoint is derived from the already configured planner URL.  This
    adapter adds no provider, retry loop, image upload, or second service.  The
    response remains untrusted and is parsed by :class:`VLMStrategyProposer`.
    """

    def __init__(
        self,
        planner_url: str,
        *,
        timeout_sec: int = 600,
        headers: Mapping[str, str] | None = None,
        max_response_bytes: int = 1024 * 1024,
    ) -> None:
        parsed = parse.urlsplit(str(planner_url or "").strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("planner_url must be an absolute HTTP(S) URL")
        if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, int):
            raise TypeError("timeout_sec must be an integer")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        self._endpoint = parse.urlunsplit(
            (parsed.scheme, parsed.netloc, "/hpk_strategy_proposal", "", "")
        )
        self._timeout_sec = timeout_sec
        self._headers = {str(key): str(value) for key, value in (headers or {}).items()}
        self._max_response_bytes = max_response_bytes

    def complete(
        self,
        *,
        instructions: str,
        input_text: str,
        output_schema: Mapping[str, Any],
        schema_name: str,
    ) -> ProposalBackendCompletion:
        rendered_instructions = (
            instructions
            + "\n\nAuthoritative output JSON schema:\n"
            + canonical_json_bytes(dict(output_schema)).decode("utf-8")
        )
        payload = json.dumps(
            {
                "instructions": rendered_instructions,
                "input_text": input_text,
                "schema_name": schema_name,
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        http_request = request.Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json", **self._headers},
            method="POST",
        )
        provider_usage: dict[str, int] = {}
        try:
            with request.urlopen(http_request, timeout=self._timeout_sec) as response:
                raw = response.read(self._max_response_bytes + 1)
                response_headers = getattr(response, "headers", None)
                if response_headers is not None:
                    for header, key in (
                        ("X-HPK-Input-Tokens", "input_tokens"),
                        ("X-HPK-Output-Tokens", "output_tokens"),
                        ("X-HPK-Total-Tokens", "total_tokens"),
                    ):
                        value = response_headers.get(header)
                        if isinstance(value, str) and value.isdecimal():
                            provider_usage[key] = int(value)
        except error.HTTPError as exc:
            try:
                detail = exc.read(4097).decode("utf-8", errors="replace").strip()
            except Exception:  # noqa: BLE001 - error body decoding is best effort
                detail = ""
            if len(detail) > 4096:
                detail = detail[:4096] + "…"
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"HPK proposal Agent API call failed (HTTP {exc.code}){suffix}"
            ) from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"HPK proposal Agent API call failed ({type(exc).__name__})"
            ) from exc
        if len(raw) > self._max_response_bytes:
            raise RuntimeError("HPK proposal Agent API response exceeds byte limit")
        try:
            output = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("HPK proposal Agent API returned invalid JSON") from exc
        if not isinstance(output, Mapping):
            raise RuntimeError(  # noqa: TRY004 - remote protocol failure
                "HPK proposal Agent API response must be an object"
            )
        audit: dict[str, Any] = {
            "backend": "existing_agent_api",
            "endpoint_path": "/hpk_strategy_proposal",
            "external_model_call": True,
            "retry_count": 0,
        }
        if provider_usage:
            audit["provider_usage"] = provider_usage
        return ProposalBackendCompletion(output=dict(output), audit=audit)


def build_geometry_proposal_prompt(
    packet: ProposalEvidencePacketV1,
    *,
    max_proposals: int = 2,
) -> tuple[str, str]:
    if isinstance(max_proposals, bool) or not isinstance(max_proposals, int):
        raise TypeError("max_proposals must be an integer")
    if max_proposals < 1 or max_proposals > 2:
        raise ValueError("max_proposals must be 1 or 2")
    return (
        _PROMPT.format(max_proposals=max_proposals),
        canonical_json_bytes(packet.to_dict()).decode("utf-8"),
    )


def _strict_response(value: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        try:
            return json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
        except Exception as exc:
            raise StrategyProposalValidationError(
                f"model response is not finite strict JSON: {type(exc).__name__}"
            ) from exc
    if not isinstance(value, str):
        raise StrategyProposalValidationError(
            "model response must be JSON text or object"
        )

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise StrategyProposalValidationError(
                    f"model response contains duplicate key {key!r}"
                )
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                StrategyProposalValidationError(
                    f"model response contains non-standard constant {constant}"
                )
            ),
        )
    except StrategyProposalValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise StrategyProposalValidationError(
            "model response is not strict JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise StrategyProposalValidationError("model response must be a JSON object")
    canonical_json_bytes(parsed)
    return parsed


class VLMStrategyProposer:
    """One-call, no-retry geometry proposer returning zero to two hypotheses."""

    def __init__(
        self,
        backend: StrategyProposalBackend,
        *,
        max_proposals: int = 2,
        proposer_version: str = VLM_GEOMETRY_PROPOSER_VERSION,
    ) -> None:
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must expose complete")
        if isinstance(max_proposals, bool) or not isinstance(max_proposals, int):
            raise TypeError("max_proposals must be an integer")
        if max_proposals < 1 or max_proposals > 2:
            raise ValueError("max_proposals must be 1 or 2")
        if proposer_version != VLM_GEOMETRY_PROPOSER_VERSION:
            raise ValueError("unsupported proposer version")
        self._backend = backend
        self._max_proposals = max_proposals
        self._proposer_version = proposer_version
        self._last_result: ProposalCallResult | None = None

    @property
    def last_result(self) -> ProposalCallResult | None:
        return self._last_result

    def propose(
        self,
        packet: ProposalEvidencePacketV1 | Mapping[str, Any],
    ) -> ProposalCallResult:
        typed = (
            packet
            if isinstance(packet, ProposalEvidencePacketV1)
            else ProposalEvidencePacketV1(packet)
        )
        if typed["verdict"] != "oppose":
            result = ProposalCallResult(
                packet_id=typed.stable_id,
                proposer_version=self._proposer_version,
                backend_called=False,
                raw_response=None,
                proposals=(),
                validation_errors=("trigger_requires_verified_oppose",),
                backend_audit=None,
            )
            self._last_result = result
            return result

        instructions, input_text = build_geometry_proposal_prompt(
            typed, max_proposals=self._max_proposals
        )
        try:
            completion = self._backend.complete(
                instructions=instructions,
                input_text=input_text,
                output_schema=strategy_proposal_output_json_schema(),
                schema_name="hpk_geometry_strategy_proposal_v1",
            )
        except Exception as exc:  # noqa: BLE001 - injected GPT backend boundary
            result = ProposalCallResult(
                packet_id=typed.stable_id,
                proposer_version=self._proposer_version,
                backend_called=True,
                raw_response=None,
                proposals=(),
                validation_errors=(f"backend_error:{type(exc).__name__}",),
                backend_audit=None,
            )
            self._last_result = result
            return result

        raw_response: dict[str, Any] | str
        if isinstance(completion.output, Mapping):
            raw_response = copy.deepcopy(dict(completion.output))
        else:
            raw_response = completion.output
        audit = (
            None if completion.audit is None else copy.deepcopy(dict(completion.audit))
        )
        try:
            parsed = _strict_response(completion.output)
            if set(parsed) != {"proposals"}:
                raise StrategyProposalValidationError(
                    "model response must contain only the proposals field"
                )
            values = parsed["proposals"]
            if not isinstance(values, list) or len(values) > self._max_proposals:
                raise StrategyProposalValidationError(
                    f"model response must contain at most {self._max_proposals} proposals"
                )
            proposals: list[StrategyProposalV1] = []
            packet_refs = set(typed.evidence_refs)
            for index, item in enumerate(values):
                if not isinstance(item, Mapping):
                    raise StrategyProposalValidationError(
                        f"proposals[{index}] must be an object"
                    )
                value = dict(item)
                expected_fields = {
                    "delta",
                    "expected_effect",
                    "evidence_refs",
                    "rationale",
                }
                if set(value) != expected_fields:
                    raise StrategyProposalValidationError(
                        f"proposals[{index}] fields are not exact"
                    )
                delta = StrategyDeltaV1(value["delta"])
                validate_delta_capability_and_non_equivalence(typed, delta)
                expected_effect = value["expected_effect"]
                if canonical_json_bytes(expected_effect) != canonical_json_bytes(
                    typed["expected_effect"]
                ):
                    raise StrategyProposalValidationError(
                        f"proposals[{index}] changed the expected effect"
                    )
                evidence_refs = value["evidence_refs"]
                if (
                    not isinstance(evidence_refs, list)
                    or not evidence_refs
                    or any(ref not in packet_refs for ref in evidence_refs)
                ):
                    raise StrategyProposalValidationError(
                        f"proposals[{index}] cites evidence absent from the packet"
                    )
                proposal_id = proposal_id_for(
                    condition=typed["condition"],
                    base_task_strategy=typed["current_task_strategy"],
                    base_geometric_strategy=typed["current_geometric_strategy"],
                    delta=delta,
                    expected_effect=expected_effect,
                    proposer_version=self._proposer_version,
                )
                proposal = StrategyProposalV1(
                    {
                        "schema": STRATEGY_PROPOSAL_SCHEMA,
                        "proposal_id": proposal_id,
                        "condition": typed["condition"],
                        "base_task_strategy": typed["current_task_strategy"],
                        "base_geometric_strategy": typed["current_geometric_strategy"],
                        "delta": delta.to_dict(),
                        "expected_effect": expected_effect,
                        "evidence_refs": evidence_refs,
                        "rationale": value["rationale"],
                        "status": "proposed",
                        "resolution": {
                            "verdict": None,
                            "evidence_ref": None,
                            "realized_task_strategy_id": None,
                            "realized_geometric_strategy_id": None,
                        },
                    }
                )
                proposals.append(proposal)
            proposal_ids = [proposal["proposal_id"] for proposal in proposals]
            if len(proposal_ids) != len(set(proposal_ids)):
                raise StrategyProposalValidationError(
                    "model response contains equivalent duplicate proposals"
                )
        except (
            StrategyProposalValidationError,
            TypeError,
            ValueError,
            KeyError,
        ) as exc:
            error = (
                str(exc)
                if isinstance(exc, StrategyProposalValidationError)
                else f"{type(exc).__name__}:{exc}"
            )
            result = ProposalCallResult(
                packet_id=typed.stable_id,
                proposer_version=self._proposer_version,
                backend_called=True,
                raw_response=raw_response,
                proposals=(),
                validation_errors=(error,),
                backend_audit=audit,
            )
            self._last_result = result
            return result

        result = ProposalCallResult(
            packet_id=typed.stable_id,
            proposer_version=self._proposer_version,
            backend_called=True,
            raw_response=raw_response,
            proposals=tuple(proposals),
            validation_errors=(),
            backend_audit=audit,
        )
        self._last_result = result
        return result


__all__ = [
    "AgentApiStrategyProposalBackend",
    "ExistingGPTTransportProposalBackend",
    "ProposalBackendCompletion",
    "ProposalCallResult",
    "StrategyProposalBackend",
    "VLMStrategyProposer",
    "build_geometry_proposal_prompt",
]
