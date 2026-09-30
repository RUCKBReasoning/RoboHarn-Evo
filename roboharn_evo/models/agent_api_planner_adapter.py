from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any
from urllib import request

import numpy as np
from PIL import Image

from roboharn_evo.models.prompt_rendering import render_known_prompt_fields
from roboharn_evo.agent.hpk.rgb_retrieval import current_rgb_image

if TYPE_CHECKING:
    from roboharn_evo.agent.experience.procedure_retriever import (
        ProcedureExperienceRetriever,
        ProcedureExperienceRuntimeConfig,
    )


def _encode_image_b64(image: np.ndarray) -> str:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _post_json(*, url: str, payload: dict[str, Any], timeout_sec: int, headers: dict[str, str] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with request.urlopen(http_request, timeout=timeout_sec) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(response_payload, dict):
        raise ValueError(f"Agent API planner response must be a JSON object, got {response_payload!r}")
    return response_payload


@dataclass(frozen=True)
class AgentApiPlannerConfig:
    server_url: str
    timeout_sec: int
    prompt_template: str
    auth_token: str
    auth_header: str
    extra_headers: dict[str, str]
    extra_body: dict[str, Any]
    procedure_experience: ProcedureExperienceRuntimeConfig | None = None
    hpk_runtime: Any = None


class AgentApiPlannerAdapter:
    """Configurable shell for planner backends served behind an HTTP JSON API.

    Expected response schema:
    {
      "commit_label": "no_update | subtask_complete | state_change",
      "memory_text": "...",
      "subtask_text": "...",
      "preferred_arm": "left | right | either"
    }
    """

    def __init__(self, config: AgentApiPlannerConfig) -> None:
        self.config = config
        reserved = {
            "procedure_experience_context",
            "hierarchical_physical_knowledge_context",
            "hpk_task_usage_audit",
            "hpk_v3_subtask_usage",
            "hpk_goal_consistency_enabled",
        }
        overridden = sorted(reserved & set(config.extra_body))
        if overridden:
            raise ValueError(
                "planner.agent_api.extra_body may not override reserved field(s): "
                + ", ".join(overridden)
            )
        self._procedure_experience_retriever: ProcedureExperienceRetriever | None = None
        self._last_procedure_experience_audit: dict[str, Any] | None = None
        self._hpk_runtime = config.hpk_runtime
        self._last_hpk_task_usage_audit: dict[str, Any] | None = None
        procedure_config = config.procedure_experience
        if (
            procedure_config is not None
            and procedure_config.mode == "candidate_dev"
        ):
            from roboharn_evo.agent.experience.procedure_retriever import (
                ProcedureExperienceRetriever,
            )

            self._procedure_experience_retriever = ProcedureExperienceRetriever(
                procedure_config
            )

    def reset(self) -> None:
        self._last_procedure_experience_audit = None
        self._last_hpk_task_usage_audit = None
        runtime = self._hpk_runtime
        if runtime is not None:
            runtime.set_preplanner_query(None)
        return

    @property
    def last_procedure_experience_audit(self) -> dict[str, Any] | None:
        if self._last_procedure_experience_audit is None:
            return None
        return json.loads(
            json.dumps(self._last_procedure_experience_audit, ensure_ascii=False)
        )

    @property
    def hpk_runtime(self) -> Any:
        return self._hpk_runtime

    @property
    def last_hpk_task_usage_audit(self) -> dict[str, Any] | None:
        if self._last_hpk_task_usage_audit is None:
            return None
        return json.loads(
            json.dumps(self._last_hpk_task_usage_audit, ensure_ascii=False)
        )

    def set_hpk_preplanner_query(self, query: Any) -> None:
        runtime = self._hpk_runtime
        if runtime is None:
            if query is not None:
                raise ValueError("HPK pre-planner query requires a static HPK runtime")
            return
        runtime.set_preplanner_query(query)

    def predict_planner_step(
        self,
        *,
        task: str,
        previous_memory_text: str,
        planner_start_image: np.ndarray,
        planner_end_image: np.ndarray,
        planner_state: np.ndarray,
    ) -> dict[str, Any]:
        self._last_hpk_task_usage_audit = None
        prompt = render_known_prompt_fields(
            self.config.prompt_template,
            task=task.strip(),
            previous_memory_text=previous_memory_text.strip(),
            state_summary=json.dumps(np.asarray(planner_state, dtype=np.float32).reshape(-1).tolist(), ensure_ascii=False),
        )
        headers = dict(self.config.extra_headers)
        if self.config.auth_token:
            headers[self.config.auth_header] = self.config.auth_token
        payload = {
            "task": task,
            "previous_memory_text": previous_memory_text,
            "planner_state": np.asarray(planner_state, dtype=np.float32).reshape(-1).tolist(),
            "planner_start_image_b64": _encode_image_b64(planner_start_image),
            "planner_end_image_b64": _encode_image_b64(planner_end_image),
            "prompt": prompt,
            **self.config.extra_body,
        }
        retriever = self._procedure_experience_retriever
        if retriever is not None:
            from roboharn_evo.agent.experience.procedure_retriever import (
                ProcedureExperienceQuery,
            )

            procedure_config = self.config.procedure_experience
            assert procedure_config is not None
            retrieval = retriever.retrieve(
                ProcedureExperienceQuery.from_planner_task(
                    task,
                    task_family=procedure_config.task_family,
                )
            )
            self._last_procedure_experience_audit = retrieval.audit
            if retrieval.context is not None:
                payload["procedure_experience_context"] = retrieval.context
        hpk_decision = None
        semantic_hpk_context = None
        hpk_runtime = self._hpk_runtime
        hierarchical_v3 = bool(
            hpk_runtime is not None
            and getattr(hpk_runtime, "hierarchical_v3_runtime", False) is True
        )
        hierarchical_full = bool(
            hierarchical_v3
            and getattr(hpk_runtime, "hierarchical_full_enabled", False) is True
        )
        goal_consistency_enabled = bool(
            hierarchical_full
            and getattr(hpk_runtime, "hpk_goal_consistency_enabled", False) is True
        )
        if goal_consistency_enabled:
            payload["hpk_goal_consistency_enabled"] = True
            hpk_runtime.set_current_task_strategy(None)
        if hpk_runtime is not None and not hierarchical_v3:
            semantic_context = getattr(hpk_runtime, "semantic_planner_context", None)
            semantic_hpk_context = (
                semantic_context() if callable(semantic_context) else None
            )
            if semantic_hpk_context is not None:
                payload["hierarchical_physical_knowledge_context"] = semantic_hpk_context
            elif getattr(hpk_runtime, "semantic_runtime_enabled", False) is not True:
                hpk_decision = hpk_runtime.retrieve_preplanner()
                self._last_hpk_task_usage_audit = hpk_decision.audit.to_dict()
                if hpk_decision.retrieval.context is not None:
                    payload["hierarchical_physical_knowledge_context"] = (
                        hpk_decision.retrieval.context.to_dict()
                    )
        prediction = _post_json(
            url=self.config.server_url,
            payload=payload,
            timeout_sec=self.config.timeout_sec,
            headers=headers,
        )
        raw_hpk_receipt = prediction.pop("hpk_task_usage_audit", None)
        if goal_consistency_enabled and "subtask_goal" not in prediction:
            raise ValueError(
                "HPK v3.1 planner response omitted subtask_goal; update the Agent API service"
            )
        if hierarchical_v3 and not hierarchical_full:
            if raw_hpk_receipt is not None:
                raise ValueError(
                    "Agent API returned an unexpected HPK audit in v3 off/rule mode"
                )
            return prediction
        if hierarchical_full:
            if getattr(hpk_runtime, "retrieval_rgb_enabled", False):
                hpk_runtime.set_current_rgb([current_rgb_image(planner_end_image, camera="planner scene")])
            if raw_hpk_receipt is not None:
                raise ValueError(
                    "Agent API returned an unexpected legacy HPK audit in v3 full mode"
                )
            apply_v3 = getattr(hpk_runtime, "apply_v3_planner_prediction", None)
            if not callable(apply_v3):
                raise TypeError("v3 full runtime must expose apply_v3_planner_prediction")
            prediction, public_audit = apply_v3(task=task, prediction=prediction)
            self._last_hpk_task_usage_audit = public_audit
            prediction["hpk_v3_subtask_usage"] = public_audit
            return prediction
        if semantic_hpk_context is not None:
            if not isinstance(raw_hpk_receipt, dict):
                raise ValueError("Agent API omitted the semantic HPK rendering receipt")
            from roboharn_evo.agent.hpk.semantic_knowledge import semantic_usage_receipt

            public_audit = semantic_usage_receipt(
                raw_hpk_receipt,
                expected_context=semantic_hpk_context,
            )
            self._last_hpk_task_usage_audit = public_audit
            prediction["hpk_task_usage_audit"] = public_audit
            return prediction
        if hpk_decision is None or hpk_decision.retrieval.context is None:
            if raw_hpk_receipt is not None:
                raise ValueError(
                    "Agent API returned an unexpected HPK task audit without context"
                )
            return prediction
        if not isinstance(raw_hpk_receipt, dict):
            raise ValueError("Agent API omitted the HPK task rendering receipt")
        finalized = hpk_runtime.accept_task_transport_receipt(
            hpk_decision,
            raw_hpk_receipt,
        )
        public_audit = finalized.audit.to_dict()
        self._last_hpk_task_usage_audit = public_audit
        prediction["hpk_task_usage_audit"] = public_audit
        return prediction
