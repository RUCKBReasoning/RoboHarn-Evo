from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillRecoveryPolicy:
    retry_budget: int = 0
    reset_budget: int = 0
    replan_budget: int = 1
    abort_on_exhausted: bool = True
    signal_actions: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "step_budget_exhausted": ("retry", "replan", "abort"),
            "stall_detected": ("retry", "reset", "replan", "abort"),
        }
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "retry_budget": self.retry_budget,
            "reset_budget": self.reset_budget,
            "replan_budget": self.replan_budget,
            "abort_on_exhausted": self.abort_on_exhausted,
            "signal_actions": {key: list(value) for key, value in self.signal_actions.items()},
        }


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    policy_binding: str
    instruction_template: str
    max_steps: int = 80
    max_retries: int = 2
    success_checks: list[str] = field(default_factory=list)
    failure_checks: list[str] = field(default_factory=list)
    recovery_skills: list[str] = field(default_factory=list)
    recovery_policy: SkillRecoveryPolicy = field(default_factory=SkillRecoveryPolicy)

    def render_instruction(self, subtask_text: str) -> str:
        return self.instruction_template.format(subtask=subtask_text.strip())
