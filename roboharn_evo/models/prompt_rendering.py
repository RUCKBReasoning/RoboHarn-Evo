from __future__ import annotations


def render_known_prompt_fields(template: str, **fields: str) -> str:
    rendered = str(template)
    for key, value in fields.items():
        rendered = rendered.replace("{" + str(key) + "}", str(value))
    return rendered
