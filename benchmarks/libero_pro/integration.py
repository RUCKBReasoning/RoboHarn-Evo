"""Simulator-lazy integration helpers for the copied LIBERO-PRO release."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import importlib
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from benchmarks._provenance import (
    REPOSITORY_ROOT,
    assert_path_within,
    dependency_available,
    module_was_imported,
    sha256_file,
)


BENCHMARK_ROOT = Path(__file__).resolve().parent
COPIED_DISTRIBUTION_ROOT = BENCHMARK_ROOT
COPIED_PACKAGE_ROOT = BENCHMARK_ROOT / "liberopro" / "liberopro"
BDDL_ROOT = COPIED_PACKAGE_ROOT / "bddl_files"
INIT_STATES_ROOT = COPIED_PACKAGE_ROOT / "init_files"
SOURCE_WHEEL_SHA256 = "882d12f9b245dcc7dde687a7de0cd1969d3e4d0422e7e3e2a22c3abd92505b3d"
SOURCE_VERSION = "0.1.1"

PRO_SUITES = (
    "libero_spatial_swap",
    "libero_spatial_task",
    "libero_spatial_lan",
    "libero_spatial_object",
    "libero_object_swap",
    "libero_object_task",
    "libero_object_lan",
    "libero_object_object",
    "libero_goal_swap",
    "libero_goal_task",
    "libero_goal_lan",
    "libero_goal_object",
    "libero_10_swap",
    "libero_10_task",
    "libero_10_lan",
    "libero_10_object",
)

EMPTY_INIT_STATE_TASKS = frozenset(
    {
        (
            "libero_spatial_task",
            "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        ),
        (
            "libero_spatial_task",
            "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        ),
        (
            "libero_10_task",
            "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        ),
        (
            "libero_10_object",
            "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        ),
    }
)

_LANGUAGE_RE = re.compile(r"\(:language\s+([^)]+)\)")


@dataclass(frozen=True, slots=True)
class LiberoProTask:
    suite: str
    task_id: int
    name: str
    instruction: str
    bddl_path: Path
    init_states_path: Path
    init_states_available: bool

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bddl_path"] = str(self.bddl_path)
        payload["init_states_path"] = str(self.init_states_path)
        return payload


def _literal_task_map() -> Mapping[str, list[str]]:
    source = COPIED_PACKAGE_ROOT / "benchmark" / "libero_suite_task_map.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                break
            # The copied registry appends these aliases in a small loop after
            # the literal. Reproduce exactly that data-only expansion without
            # importing torch, LIBERO, robosuite, or a renderer.
            for base in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
                for perturbation in ("swap", "task", "lan", "object"):
                    value[f"{base}_{perturbation}"] = list(value[base])
            return value
    raise RuntimeError(f"libero_task_map literal not found in copied source: {source}")


def discover_suites(*, pro_only: bool = True) -> tuple[str, ...]:
    available = tuple(sorted(_literal_task_map()))
    if not pro_only:
        return available
    missing = sorted(set(PRO_SUITES) - set(available))
    if missing:
        raise RuntimeError(f"copied LIBERO-PRO source lacks registered suites: {missing}")
    return PRO_SUITES


def _instruction_from_bddl(path: Path) -> str:
    match = _LANGUAGE_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"BDDL file lacks :language instruction: {path}")
    return match.group(1).strip()


def discover_tasks(suite: str) -> tuple[LiberoProTask, ...]:
    task_map = _literal_task_map()
    if suite not in task_map:
        raise KeyError(f"unknown LIBERO-PRO suite {suite!r}; choose from {sorted(task_map)}")
    tasks: list[LiberoProTask] = []
    for task_id, name in enumerate(task_map[suite]):
        bddl_path = BDDL_ROOT / suite / f"{name}.bddl"
        init_path = INIT_STATES_ROOT / suite / f"{name}.pruned_init"
        if not bddl_path.is_file():
            raise FileNotFoundError(f"copied BDDL is missing: {bddl_path}")
        if not init_path.is_file():
            raise FileNotFoundError(f"copied initial-state file is missing: {init_path}")
        tasks.append(
            LiberoProTask(
                suite=suite,
                task_id=task_id,
                name=name,
                instruction=_instruction_from_bddl(bddl_path),
                bddl_path=bddl_path.resolve(),
                init_states_path=init_path.resolve(),
                init_states_available=(suite, name) not in EMPTY_INIT_STATE_TASKS,
            )
        )
    return tuple(tasks)


def task_by_id(suite: str, task_id: int) -> LiberoProTask:
    tasks = discover_tasks(suite)
    if task_id < 0 or task_id >= len(tasks):
        raise IndexError(f"task id {task_id} outside [0, {len(tasks) - 1}] for {suite}")
    return tasks[task_id]


def dependency_status() -> dict[str, bool]:
    return {
        name: dependency_available(name)
        for name in ("bddl", "mujoco", "robosuite", "torch")
    }


def assets_status(assets_root: Path | None) -> dict[str, Any]:
    candidate = assets_root
    if candidate is None:
        raw = os.environ.get("LIBERO_PRO_ASSETS_ROOT", "").strip()
        candidate = Path(raw) if raw else COPIED_PACKAGE_ROOT / "assets"
    candidate = candidate.expanduser().resolve()
    markers = ("scenes", "stable_hope_objects", "stable_scanned_objects")
    missing = [name for name in markers if not (candidate / name).is_dir()]
    return {"root": str(candidate), "available": not missing, "missing_markers": missing}


def dry_run_provenance(
    *,
    suite: str,
    task_id: int,
    seed: int,
    config: Path | None,
    output_dir: Path,
    assets_root: Path | None,
) -> dict[str, Any]:
    import roboharn_evo

    package_path = assert_path_within(Path(roboharn_evo.__file__), REPOSITORY_ROOT, label="roboharn_evo package")
    task = task_by_id(suite, task_id)
    config_path = config.expanduser().resolve() if config else None
    return {
        "benchmark": {
            "name": "LIBERO-PRO",
            "distribution": "rpent-liberopro",
            "version": SOURCE_VERSION,
            "copy_root": str(BENCHMARK_ROOT.resolve()),
            "source_package_root": str(COPIED_PACKAGE_ROOT.resolve()),
            "source_wheel_sha256": SOURCE_WHEEL_SHA256,
        },
        "agent": {
            "name": "roboharn-evo",
            "version": roboharn_evo.__version__,
            "module": str(package_path),
            "module_sha256": sha256_file(package_path),
            "end_to_end_supported": False,
            "capability_gate": "current RoboHarnAgentRuntime requires RMBench dual-arm/qpos schema",
        },
        "task": {
            **task.as_json(),
            "bddl_sha256": sha256_file(task.bddl_path),
            "init_states_sha256": sha256_file(task.init_states_path),
        },
        "run": {
            "seed": seed,
            "config": str(config_path) if config_path else None,
            "config_sha256": sha256_file(config_path) if config_path and config_path.is_file() else None,
            "output_dir": str(output_dir.expanduser().resolve()),
            "assets": assets_status(assets_root),
        },
        "dependencies": dependency_status(),
        "simulator_imported": any(
            module_was_imported(name) for name in ("mujoco", "robosuite", "liberopro")
        ),
        "initial_states": {
            "pro_tasks": 160,
            "available": 156,
            "empty": 4,
            "empty_tasks": [f"{suite_name}/{name}" for suite_name, name in sorted(EMPTY_INIT_STATE_TASKS)],
        },
    }


def _activate_copied_distribution(assets_root: Path, runtime_config_dir: Path) -> None:
    existing = sys.modules.get("liberopro")
    if existing is not None:
        origin = Path(getattr(existing, "__file__", "")).resolve()
        try:
            origin.relative_to(COPIED_DISTRIBUTION_ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"refusing preloaded external liberopro module: {origin}") from exc
    source = str(COPIED_DISTRIBUTION_ROOT.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    os.environ["LIBERO_PRO_ASSETS_ROOT"] = str(assets_root.resolve())
    os.environ["LIBERO_CONFIG_PATH"] = str(runtime_config_dir.resolve())


def build_environment(
    task: LiberoProTask,
    *,
    seed: int,
    assets_root: Path,
    runtime_config_dir: Path,
    camera_height: int = 128,
    camera_width: int = 128,
    camera_depths: bool = False,
    horizon: int = 1000,
) -> Any:
    """Create the canonical copied OffScreenRenderEnv, importing it lazily."""
    status = assets_status(assets_root)
    if not status["available"]:
        raise FileNotFoundError(
            "LIBERO-PRO assets are not bundled because their redistribution license "
            f"was not available; pass --assets-root. Missing {status['missing_markers']} "
            f"under {status['root']}"
        )
    if not task.init_states_available:
        raise RuntimeError(
            f"canonical rpent-liberopro 0.1.1 contains an empty initial-state array for "
            f"{task.suite}/{task.name}; provide a corrected canonical distribution"
        )
    runtime_config_dir.mkdir(parents=True, exist_ok=True)
    _activate_copied_distribution(assets_root, runtime_config_dir)
    wrapper = importlib.import_module("liberopro.liberopro.envs.env_wrapper")
    origin = Path(wrapper.__file__).resolve()
    try:
        origin.relative_to(COPIED_DISTRIBUTION_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"LIBERO environment loaded outside RoboHarn-Evo tree: {origin}") from exc
    env = wrapper.OffScreenRenderEnv(
        bddl_file_name=str(task.bddl_path),
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_depths=camera_depths,
        horizon=horizon,
    )
    env.seed(seed)
    return env


def reset_to_initial_state(env: Any, task: LiberoProTask, *, init_state_id: int = 0) -> Any:
    """Reset once, then apply one canonical packaged simulator state."""
    if not task.init_states_available:
        raise RuntimeError(f"no usable initial states for {task.suite}/{task.name}")
    torch = importlib.import_module("torch")
    states = torch.load(task.init_states_path, map_location="cpu", weights_only=False)
    if len(states) == 0:
        raise RuntimeError(f"initial-state file decoded to an empty array: {task.init_states_path}")
    if init_state_id < 0 or init_state_id >= len(states):
        raise IndexError(f"init-state id {init_state_id} outside [0, {len(states) - 1}]")
    env.reset()
    return env.set_init_state(states[init_state_id])


__all__ = [
    "BDDL_ROOT",
    "BENCHMARK_ROOT",
    "COPIED_DISTRIBUTION_ROOT",
    "EMPTY_INIT_STATE_TASKS",
    "INIT_STATES_ROOT",
    "LiberoProTask",
    "PRO_SUITES",
    "assets_status",
    "build_environment",
    "dependency_status",
    "discover_suites",
    "discover_tasks",
    "dry_run_provenance",
    "reset_to_initial_state",
    "task_by_id",
]
