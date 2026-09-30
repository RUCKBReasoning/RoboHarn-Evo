from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re

import yaml


_SKILL_FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<frontmatter>.*?)\n---\s*\n?(?P<body>.*)\Z", re.DOTALL)


def _extract_prompt_body(skill_body: str) -> str:
    marker = "## Prompt Body"
    if marker in skill_body:
        return skill_body.split(marker, 1)[1].strip()
    return skill_body.strip()


@lru_cache(maxsize=32)
def load_prompt_skill(skill_name: str) -> str:
    for root in _skill_roots():
        if not root.is_dir():
            continue
        for skill_md_path in sorted(root.rglob("SKILL.md")):
            match = _SKILL_FRONTMATTER_RE.match(skill_md_path.read_text(encoding="utf-8"))
            if not match:
                continue
            metadata = yaml.safe_load(match.group("frontmatter")) or {}
            if str(metadata.get("name", "")).strip() == skill_name:
                return _extract_prompt_body(match.group("body").strip())
    raise ValueError(f"Prompt skill not found: {skill_name}")


def _skill_roots() -> list[Path]:
    roots: list[Path] = []
    env_paths = os.environ.get("ROBOHARN_EVO_SKILLS_PATHS", "")
    for raw_path in env_paths.split(os.pathsep):
        if raw_path.strip():
            roots.append(Path(raw_path).expanduser())
    repo_root = Path(__file__).resolve().parents[3]
    roots.append(repo_root / "policy" / "roboharn_evo" / "skills")
    roots.append(repo_root / "skills")
    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            unique_roots.append(resolved)
            seen.add(resolved)
    return unique_roots


def resolve_prompt_template(config: dict, default_prompt: str, *, default_skill_key: str = "") -> str:
    skill_name = str(config.get("prompt_skill", "")).strip()
    if skill_name:
        return load_prompt_skill(skill_name)
    prompt_template = str(config.get("prompt_template", "")).strip()
    if prompt_template:
        return prompt_template
    if default_skill_key:
        return load_prompt_skill(default_skill_key)
    return default_prompt
