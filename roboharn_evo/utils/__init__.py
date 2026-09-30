"""Legacy utility exports with optional training dependencies loaded lazily."""

from __future__ import annotations

from typing import Any

_EXPORT_MODULES = {
    "apply_cli_overrides": "roboharn_evo.utils.config",
    "load_config": "roboharn_evo.utils.config",
    "set_seed": "roboharn_evo.utils.config",
    "load_jsonl": "roboharn_evo.utils.io",
    "load_yaml": "roboharn_evo.utils.io",
    "VocabularySet": "roboharn_evo.utils.text",
    "build_vocabulary": "roboharn_evo.utils.text",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))


__all__ = sorted(_EXPORT_MODULES)
