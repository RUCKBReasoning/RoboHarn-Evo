from __future__ import annotations

from pathlib import Path

from ..components.agent_tools.local_skill_registry import LocalSkillRegistry as _LocalSkillRegistry


class SkillRegistry(_LocalSkillRegistry):
    def resolve(self, subtask_text: str):
        skill = self.get_skill(subtask_text, refresh=True)
        if skill is not None:
            return skill.to_skill_spec(subtask_text)
        return None


def build_default_skill_registry() -> SkillRegistry:
    workspace_root = Path(__file__).resolve().parents[4]
    return SkillRegistry(workspace_root=str(workspace_root))
