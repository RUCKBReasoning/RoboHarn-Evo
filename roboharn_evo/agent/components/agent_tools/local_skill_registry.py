from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re
from typing import Any

import yaml


_SKILL_FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<frontmatter>.*?)\n---\s*\n?(?P<body>.*)\Z", re.DOTALL)
_INLINE_SKILL_TOKEN_RE = re.compile(r"(?<![\w-])\$([A-Za-z0-9][A-Za-z0-9-]*)")


@dataclass(slots=True)
class LocalSkill:
    name: str
    description: str
    skill_dir: Path
    skill_md_path: Path
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)
    display_name: str = ""
    short_description: str = ""
    default_prompt: str = ""
    scripts: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    linked_skills: list[str] = field(default_factory=list)
    policy_binding: str = "default_manipulation_policy"
    max_steps: int = 80
    max_retries: int = 2
    recovery_skills: list[str] = field(default_factory=list)
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

    def summary_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "skill_dir": str(self.skill_dir),
            "skill_md_path": str(self.skill_md_path),
            "display_name": self.display_name,
            "short_description": self.short_description,
            "resource_counts": {
                "scripts": len(self.scripts),
                "references": len(self.references),
                "assets": len(self.assets),
            },
            "policy_binding": self.policy_binding,
            "recovery_skills": list(self.recovery_skills),
            "recovery_policy": {
                "retry_budget": self.retry_budget,
                "reset_budget": self.reset_budget,
                "replan_budget": self.replan_budget,
                "abort_on_exhausted": self.abort_on_exhausted,
                "signal_actions": {key: list(value) for key, value in self.signal_actions.items()},
            },
        }

    def detail_dict(self) -> dict[str, Any]:
        return {
            **self.summary_dict(),
            "default_prompt": self.default_prompt,
            "body": self.body,
            "metadata": self.metadata,
            "scripts": self.scripts,
            "references": self.references,
            "assets": self.assets,
            "linked_skills": self.linked_skills,
        }

    def to_skill_spec(self, subtask_text: str):
        from roboharn_evo.agent.skills.base import SkillRecoveryPolicy, SkillSpec
        return SkillSpec(
            name=self.name,
            description=self.description,
            policy_binding=self.policy_binding,
            instruction_template="{subtask}",
            max_steps=self.max_steps,
            max_retries=self.max_retries,
            recovery_skills=list(self.recovery_skills),
            recovery_policy=SkillRecoveryPolicy(
                retry_budget=self.retry_budget,
                reset_budget=self.reset_budget,
                replan_budget=self.replan_budget,
                abort_on_exhausted=self.abort_on_exhausted,
                signal_actions={key: tuple(value) for key, value in self.signal_actions.items()},
            ),
        )


@dataclass(slots=True)
class ExpandedSkillMessage:
    message: str
    requested_skills: tuple[str, ...]
    included_skills: tuple[str, ...]
    missing_requested_skills: tuple[str, ...]
    missing_referenced_skills: tuple[str, ...]


class LocalSkillRegistry:
    _REFERENCE_MAX_FILES = 5
    _REFERENCE_MAX_CHARS_PER_FILE = 4000
    _REFERENCE_MAX_TOTAL_CHARS = 12000

    def __init__(self, configured_paths: list[str] | None = None, workspace_root: str | None = None):
        self._configured_paths: list[str] = configured_paths or []
        self._workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
        self._skills_by_name: dict[str, LocalSkill] = {}

    @property
    def search_roots(self) -> list[Path]:
        roots: list[Path] = []
        env_paths = os.environ.get("ROBOHARN_EVO_SKILLS_PATHS", "")
        if env_paths:
            for raw_path in env_paths.split(os.pathsep):
                if raw_path.strip():
                    roots.append(Path(raw_path).expanduser())
        for raw_path in self._configured_paths:
            if raw_path.strip():
                roots.append(Path(raw_path).expanduser())
        roots.extend([
            self._workspace_root / "skills",
        ])
        unique_roots: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen:
                unique_roots.append(resolved)
                seen.add(resolved)
        return unique_roots

    def refresh(self) -> dict[str, LocalSkill]:
        skills: dict[str, LocalSkill] = {}
        seen_dirs: set[Path] = set()
        for root in self.search_roots:
            if not root.exists() or not root.is_dir():
                continue
            for skill_md_path in sorted(root.rglob("SKILL.md")):
                skill_dir = skill_md_path.parent.resolve()
                if skill_dir in seen_dirs:
                    continue
                seen_dirs.add(skill_dir)
                try:
                    skill = self._load_skill(skill_md_path.resolve())
                except Exception as exc:
                    print(f"[local_skill_registry] failed to load {skill_md_path}: {exc!r}", flush=True)
                    continue
                if skill.name in skills:
                    continue
                skills[skill.name] = skill
        self._skills_by_name = skills
        return self._skills_by_name

    def list_skills(self, refresh: bool = True) -> list[LocalSkill]:
        if refresh or not self._skills_by_name:
            self.refresh()
        return list(self._skills_by_name.values())

    def get_skill(self, skill_name: str, refresh: bool = True) -> LocalSkill | None:
        if refresh or not self._skills_by_name:
            self.refresh()
        return self._skills_by_name.get(skill_name)

    def list_skill_summaries(self, refresh: bool = True) -> list[str]:
        skills = self.list_skills(refresh=refresh)
        return [
            f"{skill.name}: {skill.description} | policy={skill.policy_binding} | recovery={','.join(skill.recovery_skills) if skill.recovery_skills else 'none'} | retry={skill.retry_budget} reset={skill.reset_budget} replan={skill.replan_budget}"
            for skill in skills
        ]

    def extract_skill_names(self, text: str) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for match in _INLINE_SKILL_TOKEN_RE.finditer(text):
            name = match.group(1)
            if name not in seen:
                names.append(name)
                seen.add(name)
        return names

    def expand_inline_request(
        self,
        message: str,
        refresh: bool = True,
        execution_context: str | None = None,
    ) -> ExpandedSkillMessage:
        if refresh or not self._skills_by_name:
            self.refresh()
        requested_skills = self.extract_skill_names(message)
        if not requested_skills:
            return ExpandedSkillMessage(message=message, requested_skills=(), included_skills=(), missing_requested_skills=(), missing_referenced_skills=())
        available_requested = [name for name in requested_skills if name in self._skills_by_name]
        missing_requested = [name for name in requested_skills if name not in self._skills_by_name]
        ordered_skills, missing_references = self._resolve_requested_skills(available_requested)
        return ExpandedSkillMessage(
            message=self._build_inline_skill_message(original_message=message, requested_skills=requested_skills, ordered_skills=ordered_skills, missing_references=missing_references, execution_context=execution_context),
            requested_skills=tuple(requested_skills),
            included_skills=tuple(skill.name for skill in ordered_skills),
            missing_requested_skills=tuple(missing_requested),
            missing_referenced_skills=tuple(missing_references),
        )

    def build_service_description(self, max_items: int = 6) -> str:
        skills = self.list_skills(refresh=True)
        base = "Local RoboHarn-Evo skill service discovering SKILL.md workflow packages."
        if not skills:
            return base + " No skills found."
        skill_fragments = [f"{skill.name}: {skill.description}" for skill in skills[:max_items]]
        if len(skills) > max_items:
            skill_fragments.append(f"and {len(skills) - max_items} more")
        return base + " Available skills: " + "; ".join(skill_fragments)

    def _resolve_requested_skills(self, requested_skills: list[str]) -> tuple[list[LocalSkill], list[str]]:
        ordered_skills: list[LocalSkill] = []
        missing_references: list[str] = []
        visited: set[str] = set()
        missing_seen: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            skill = self._skills_by_name.get(name)
            if skill is None:
                if name not in missing_seen:
                    missing_references.append(name)
                    missing_seen.add(name)
                return
            ordered_skills.append(skill)
            for linked_skill_name in skill.linked_skills:
                visit(linked_skill_name)

        for name in requested_skills:
            visit(name)
        return ordered_skills, missing_references

    def _build_inline_skill_message(self, original_message: str, requested_skills: list[str], ordered_skills: list[LocalSkill], missing_references: list[str], execution_context: str | None = None) -> str:
        requested_display = ", ".join(f"${name}" for name in requested_skills)
        included_display = ", ".join(f"${skill.name}" for skill in ordered_skills) or "(none)"
        sections = [
            "The user explicitly requested local skill guidance from the local skill library.",
            "Apply the following skill instructions as task-specific guidance for this turn.",
            f"Requested skills: {requested_display}",
            f"Included skill packages: {included_display}",
            "",
        ]
        if execution_context:
            sections.extend(["## Skill Execution Environment", execution_context, ""])
        sections.extend(["## User Request", self._strip_skill_tokens(original_message), "", "## Local Skill Package"])
        for skill in ordered_skills:
            sections.extend([f"### Skill `{skill.name}`", f"Source: {skill.skill_md_path}", f"Description: {skill.description}", "Instructions:", skill.body, ""])
        if missing_references:
            sections.extend(["## Missing Referenced Skills", ", ".join(f"${name}" for name in missing_references), ""])
        return "\n".join(sections).strip()

    def _strip_skill_tokens(self, text: str) -> str:
        return _INLINE_SKILL_TOKEN_RE.sub("", text).strip()

    def _collect_relative_files(self, folder: Path) -> list[str]:
        if not folder.exists() or not folder.is_dir():
            return []
        return [str(path.relative_to(folder.parent)) for path in sorted(folder.rglob("*")) if path.is_file()]

    def _load_skill(self, skill_md_path: Path) -> LocalSkill:
        raw_text = skill_md_path.read_text(encoding="utf-8")
        match = _SKILL_FRONTMATTER_RE.match(raw_text)
        if not match:
            raise ValueError("SKILL.md must start with YAML frontmatter")
        metadata = yaml.safe_load(match.group("frontmatter")) or {}
        body = match.group("body").strip()
        name = str(metadata.get("name", "")).strip()
        description = str(metadata.get("description", "")).strip()
        if not name or not description:
            raise ValueError("SKILL.md frontmatter must define name and description")
        skill_dir = skill_md_path.parent
        linked_skills = self.extract_skill_names(body)
        recovery_cfg = metadata.get("recovery_policy", {}) or {}
        return LocalSkill(
            name=name,
            description=description,
            skill_dir=skill_dir,
            skill_md_path=skill_md_path,
            body=body,
            metadata=dict(metadata),
            scripts=self._collect_relative_files(skill_dir / "scripts"),
            references=self._collect_relative_files(skill_dir / "references"),
            assets=self._collect_relative_files(skill_dir / "assets"),
            linked_skills=linked_skills,
            recovery_skills=list(linked_skills),
            retry_budget=max(0, int(recovery_cfg.get("retry_budget", metadata.get("retry_budget", 0)))),
            reset_budget=max(0, int(recovery_cfg.get("reset_budget", metadata.get("reset_budget", 0)))),
            replan_budget=max(0, int(recovery_cfg.get("replan_budget", metadata.get("replan_budget", 1)))),
            signal_actions=dict(
                (str(key), tuple(value))
                for key, value in recovery_cfg.get(
                    "signal_actions",
                    metadata.get(
                        "signal_actions",
                        {
                            "step_budget_exhausted": ["retry", "replan", "abort"],
                            "stall_detected": ["retry", "reset", "replan", "abort"],
                        },
                    ),
                ).items()
            ),
            abort_on_exhausted=bool(recovery_cfg.get("abort_on_exhausted", metadata.get("abort_on_exhausted", True))),
        )
