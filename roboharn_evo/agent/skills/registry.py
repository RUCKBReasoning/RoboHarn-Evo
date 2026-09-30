from __future__ import annotations

from roboharn_evo.agent.components.agent_tools.local_skill_registry import LocalSkillRegistry as _LocalSkillRegistry
from roboharn_evo.agent.paths import project_root, roboharn_skills_dir


class SkillRegistry(_LocalSkillRegistry):
    def resolve(self, subtask_text: str):
        skill = self.get_skill(subtask_text, refresh=True)
        if skill is not None:
            return skill.to_skill_spec(subtask_text)
        return None


def build_default_skill_registry() -> SkillRegistry:
    return SkillRegistry(
        configured_paths=[str(roboharn_skills_dir())],
        workspace_root=str(project_root()),
    )
