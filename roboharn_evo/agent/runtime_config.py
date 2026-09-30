"""Load an Agent profile with an optional shared base, independent of benchmark."""
from copy import deepcopy
from pathlib import Path

from roboharn_evo.agent.paths import project_root
from roboharn_evo.utils.io import load_yaml


def merge_runtime_config(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_runtime_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_agent_runtime_config(path):
    profile = load_yaml(str(path))
    base = profile.pop("base_config", None)
    if base is None:
        return profile
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = project_root() / base_path
    return merge_runtime_config(load_yaml(str(base_path)), profile)
