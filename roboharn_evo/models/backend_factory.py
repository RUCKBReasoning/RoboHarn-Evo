from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roboharn_evo.models.backend_interfaces import ExecutorBackend, OODBackend, PlannerBackend, RecoveryBackend
from roboharn_evo.models.paths import agentic_model_path
from roboharn_evo.models.prompt_skills import resolve_prompt_template
from roboharn_evo.agent.execution_feedback import DIRECT_FEEDBACK_PLANNER_PROMPT

if TYPE_CHECKING:
    from roboharn_evo.agent.experience.procedure_retriever import (
        ProcedureExperienceRuntimeConfig,
    )


def build_planner_backend(
    config: dict[str, Any],
    *,
    procedure_experience_config: ProcedureExperienceRuntimeConfig | None = None,
    hpk_runtime: Any = None,
    execution_evidence_enabled: bool = True,
) -> PlannerBackend:
    backend = str(config.get("backend", "robobrain_adapter"))
    if hpk_runtime is not None and backend != "agent_api":
        raise ValueError("static HPK planner integration requires planner.backend=agent_api")
    if (
        procedure_experience_config is not None
        and procedure_experience_config.mode == "candidate_dev"
        and backend != "agent_api"
    ):
        from roboharn_evo.agent.experience.procedure_retriever import (
            ProcedureExperienceConfigurationError,
        )

        raise ProcedureExperienceConfigurationError(
            "candidate_dev procedure experience requires planner.backend=agent_api"
        )
    if backend in {"offline_replay", "replay"}:
        from roboharn_evo.models.offline_replay_planner import OfflineReplayPlanner, OfflineReplayPlannerConfig

        replay_cfg = config.get("offline_replay", config)
        return OfflineReplayPlanner(
            OfflineReplayPlannerConfig(
                subtask_text=str(replay_cfg.get("subtask_text", "")),
                memory_text=str(replay_cfg.get("memory_text", "Offline replay is observing a recorded trajectory.")),
                commit_label=str(replay_cfg.get("commit_label", "state_change")),
            )
        )
    if backend == "robobrain_adapter":
        from roboharn_evo.models.robobrain_adapter import (
            DEFAULT_ROBOBRAIN_PLANNER_PROMPT,
            RoboBrainPlannerAdapter,
            RoboBrainPlannerConfig,
        )
        robobrain_cfg = config.get("robobrain", config)
        return RoboBrainPlannerAdapter(
            RoboBrainPlannerConfig(
                mode=str(robobrain_cfg.get("mode", "remote")),
                model_id=str(robobrain_cfg.get("model_id", "BAAI/RoboBrain2.5-8B-NV")),
                device_map=str(robobrain_cfg.get("device_map", "auto")),
                server_url=str(robobrain_cfg.get("server_url", "http://127.0.0.1:9001/plan")),
                timeout_sec=int(robobrain_cfg.get("timeout_sec", 120)),
                do_sample=bool(robobrain_cfg.get("do_sample", False)),
                temperature=float(robobrain_cfg.get("temperature", 0.0)),
                prompt_template=resolve_prompt_template(
                    robobrain_cfg,
                    DEFAULT_ROBOBRAIN_PLANNER_PROMPT,
                    default_skill_key="control-turn-planner",
                ) if execution_evidence_enabled else DIRECT_FEEDBACK_PLANNER_PROMPT,
            )
        )
    if backend == "agent_api":
        from roboharn_evo.models.agent_api_planner_adapter import AgentApiPlannerAdapter, AgentApiPlannerConfig
        from roboharn_evo.models.robobrain_adapter import DEFAULT_ROBOBRAIN_PLANNER_PROMPT
        agent_cfg = config.get("agent_api", config)
        return AgentApiPlannerAdapter(
            AgentApiPlannerConfig(
                server_url=str(agent_cfg["server_url"]),
                timeout_sec=int(agent_cfg.get("timeout_sec", 120)),
                prompt_template=resolve_prompt_template(
                    agent_cfg,
                    DEFAULT_ROBOBRAIN_PLANNER_PROMPT,
                    default_skill_key="control-turn-planner",
                ) if execution_evidence_enabled else DIRECT_FEEDBACK_PLANNER_PROMPT,
                auth_token=str(agent_cfg.get("auth_token", "")),
                auth_header=str(agent_cfg.get("auth_header", "Authorization")),
                extra_headers=dict(agent_cfg.get("extra_headers", {})),
                extra_body=dict(agent_cfg.get("extra_body", {})),
                procedure_experience=procedure_experience_config,
                hpk_runtime=hpk_runtime,
            )
        )
    if backend == "agentic":
        from roboharn_evo.agent import AgenticPlanner, AgenticPlannerConfig

        ag_cfg = config.get("agentic", config)
        return AgenticPlanner(
            AgenticPlannerConfig(
                base_url=str(ag_cfg.get("base_url", "http://127.0.0.1:8000/v1")),
                api_key=str(ag_cfg.get("api_key", "EMPTY")),
                model=str(ag_cfg.get("model", agentic_model_path())),
                timeout_sec=int(ag_cfg.get("timeout_sec", 3600)),
                max_tokens=int(ag_cfg.get("max_tokens", 2048)),
                temperature=float(ag_cfg.get("temperature", 0.0)),
                top_p=float(ag_cfg.get("top_p", 1.0)),
                max_rounds=int(ag_cfg.get("max_rounds", 6)),
                initial_memory_text=str(ag_cfg.get("initial_memory_text", "The task has started.")),
            )
        )
    if backend in {"qwen_vl", "qwen_vl_8b"}:
        from roboharn_evo.models.qwen_vl_planner_adapter import QwenVlPlannerAdapter, QwenVlPlannerConfig
        from roboharn_evo.models.robobrain_adapter import DEFAULT_ROBOBRAIN_PLANNER_PROMPT
        qwen_cfg = config.get("qwen_vl_8b", config.get("qwen_vl", config))
        return QwenVlPlannerAdapter(
            QwenVlPlannerConfig(
                base_url=str(qwen_cfg["base_url"]),
                api_key=str(qwen_cfg.get("api_key", "EMPTY")),
                model_path=str(qwen_cfg["model_path"]),
                timeout_sec=int(qwen_cfg.get("timeout_sec", 3600)),
                max_tokens=int(qwen_cfg.get("max_tokens", 2048)),
                temperature=float(qwen_cfg.get("temperature", 0.0)),
                top_p=float(qwen_cfg.get("top_p", 1.0)),
                prompt_template=resolve_prompt_template(
                    qwen_cfg,
                    DEFAULT_ROBOBRAIN_PLANNER_PROMPT,
                    default_skill_key="control-turn-planner",
                ),
            )
        )
    raise ValueError(f"Unsupported planner backend: {backend}")


def build_executor_backend(config: dict[str, Any]) -> ExecutorBackend:
    backend = str(config.get("backend", "pi05_adapter"))
    if backend == "tool_only":
        from roboharn_evo.models.tool_only_executor import ToolOnlyExecutor
        return ToolOnlyExecutor()
    if backend == "pi05_adapter":
        from roboharn_evo.models.pi05_adapter import Pi05ExecutorAdapter, Pi05ExecutorConfig
        pi05_cfg = config.get("pi05", config)
        return Pi05ExecutorAdapter(
            Pi05ExecutorConfig(
                train_config_name=str(pi05_cfg.get("train_config_name", "pi05_aloha_robotwin_full")),
                model_name=str(pi05_cfg["model_name"]),
                checkpoint_id=int(pi05_cfg["checkpoint_id"]),
                pi0_step=int(pi05_cfg.get("pi0_step", 50)),
                prompt_template=str(
                    pi05_cfg.get(
                        "prompt_template",
                        "{subtask}",
                    )
                ),
            )
        )
    if backend == "pi05_api":
        from roboharn_evo.models.pi05_api_executor_adapter import Pi05ApiExecutorAdapter, Pi05ApiExecutorConfig
        pi05_api_cfg = config.get("pi05_api", config)
        return Pi05ApiExecutorAdapter(
            Pi05ApiExecutorConfig(
                server_url=str(pi05_api_cfg["server_url"]),
                timeout_sec=int(pi05_api_cfg.get("timeout_sec", 120)),
                auth_token=str(pi05_api_cfg.get("auth_token", "")),
                auth_header=str(pi05_api_cfg.get("auth_header", "Authorization")),
                extra_headers=dict(pi05_api_cfg.get("extra_headers", {})),
                extra_body=dict(pi05_api_cfg.get("extra_body", {})),
                max_chunk_steps=int(pi05_api_cfg.get("max_chunk_steps", 50)),
                action_dim=int(pi05_api_cfg.get("action_dim", 14)),
            )
        )
    if backend in {"world_action_model", "wam"}:
        from roboharn_evo.models.world_action_model_adapter import (
            WorldActionModelExecutorAdapter,
            WorldActionModelExecutorConfig,
        )
        wam_cfg = config.get("world_action_model", config)
        return WorldActionModelExecutorAdapter(
            WorldActionModelExecutorConfig(
                server_url=str(wam_cfg["server_url"]),
                timeout_sec=int(wam_cfg.get("timeout_sec", 120)),
                auth_token=str(wam_cfg.get("auth_token", "")),
                auth_header=str(wam_cfg.get("auth_header", "Authorization")),
                extra_headers=dict(wam_cfg.get("extra_headers", {})),
                extra_body=dict(wam_cfg.get("extra_body", {})),
                max_chunk_steps=int(wam_cfg.get("max_chunk_steps", 50)),
                action_dim=int(wam_cfg.get("action_dim", 14)),
            )
        )
    if backend in {"offline_replay", "replay"}:
        from roboharn_evo.models.offline_replay_executor import OfflineReplayExecutor, OfflineReplayExecutorConfig

        replay_cfg = config.get("offline_replay", config)
        return OfflineReplayExecutor(
            OfflineReplayExecutorConfig(
                action_dim=int(replay_cfg.get("action_dim", 14)),
                chunk_steps=int(replay_cfg.get("chunk_steps", 1)),
            )
        )
    raise ValueError(f"Unsupported executor backend: {backend}")


def build_ood_backend(config: dict[str, Any]) -> OODBackend:
    backend = str(config.get("backend", "agent_api"))
    if backend in {"offline_replay", "replay", "none"}:
        from roboharn_evo.models.offline_replay_ood import OfflineReplayOODAdapter, OfflineReplayOODConfig

        replay_cfg = config.get("offline_replay", config)
        return OfflineReplayOODAdapter(
            OfflineReplayOODConfig(
                OOD_scenario=str(replay_cfg.get("OOD_scenario", "none")),
                reason=str(replay_cfg.get("reason", "offline replay local OOD fallback")),
                confidence=float(replay_cfg.get("confidence", 0.0)),
            )
        )
    if backend == "agent_api":
        from roboharn_evo.models.agent_api_ood_adapter import AgentApiOODAdapter, AgentApiOODConfig
        agent_cfg = config.get("agent_api", config)
        return AgentApiOODAdapter(
            AgentApiOODConfig(
                server_url=str(agent_cfg["server_url"]),
                timeout_sec=int(agent_cfg.get("timeout_sec", 120)),
                prompt_template=str(agent_cfg["prompt_template"]),
                auth_token=str(agent_cfg.get("auth_token", "")),
                auth_header=str(agent_cfg.get("auth_header", "Authorization")),
                extra_headers=dict(agent_cfg.get("extra_headers", {})),
                extra_body=dict(agent_cfg.get("extra_body", {})),
            )
        )
    raise ValueError(f"Unsupported OOD backend: {backend}")


def build_recovery_backend(config: dict[str, Any]) -> RecoveryBackend:
    backend = str(config.get("backend", "agent_api"))
    if backend == "agent_api":
        from roboharn_evo.models.agent_api_recovery_adapter import AgentApiRecoveryAdapter, AgentApiRecoveryConfig, DEFAULT_RECOVERY_PROMPT
        agent_cfg = config.get("agent_api", config)
        return AgentApiRecoveryAdapter(
            AgentApiRecoveryConfig(
                server_url=str(agent_cfg.get("server_url", "http://127.0.0.1:9101/recover")),
                timeout_sec=int(agent_cfg.get("timeout_sec", 120)),
                prompt_template=str(agent_cfg.get("prompt_template", DEFAULT_RECOVERY_PROMPT)),
                auth_token=str(agent_cfg.get("auth_token", "")),
                auth_header=str(agent_cfg.get("auth_header", "Authorization")),
                extra_headers=dict(agent_cfg.get("extra_headers", {})),
                extra_body=dict(agent_cfg.get("extra_body", {})),
            )
        )
    raise ValueError(f"Unsupported recovery backend: {backend}")
