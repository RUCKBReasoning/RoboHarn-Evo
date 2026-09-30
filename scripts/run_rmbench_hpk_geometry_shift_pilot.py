from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.candidate_features import candidate_semantic_features
from roboharn_evo.agent.hpk.geometry_policy import rank_operation_pose_candidates
from roboharn_evo.agent.hpk.policy_config import (
    LoadedHPKPolicy,
    load_safe_exploration_geometry_policy,
)
from roboharn_evo.agent.hpk.schemas import HPKUnresolved, GeometricStrategyV1


INPUT_SCHEMA = "roboharn_evo/hpk/rmbench_geometry_shift_pilot_input/v1"
REPORT_SCHEMA = "roboharn_evo/hpk/rmbench_geometry_shift_pilot_report/v1"
_POSE_FIELDS = ("approach_pose", "ee_target_pose")


class GeometryShiftPilotError(ValueError):
    """Raised when a geometry-shift case is ambiguous or unsafe to score."""


def _exact(value: Mapping[str, Any], keys: set[str], path: str) -> None:
    if set(value) != keys:
        raise GeometryShiftPilotError(f"{path} fields mismatch")


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeometryShiftPilotError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise GeometryShiftPilotError(f"{path} must be a finite number")
    return result


def _vector3(value: Any, path: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GeometryShiftPilotError(f"{path} must be a length-3 array")
    if len(value) != 3:
        raise GeometryShiftPilotError(f"{path} must be a length-3 array")
    return tuple(_finite(item, f"{path}[{index}]") for index, item in enumerate(value))


def _pose(value: Any, path: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GeometryShiftPilotError(f"{path} must be a length-7 pose")
    if len(value) != 7:
        raise GeometryShiftPilotError(f"{path} must be a length-7 pose")
    return tuple(_finite(item, f"{path}[{index}]") for index, item in enumerate(value))


def _quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _transform_pose(
    pose: tuple[float, ...],
    *,
    translation: tuple[float, float, float],
    yaw_degrees: float,
) -> list[float]:
    yaw = math.radians(yaw_degrees)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    x, y, z = pose[:3]
    rotated = (
        cosine * x - sine * y + translation[0],
        sine * x + cosine * y + translation[1],
        z + translation[2],
    )
    half = yaw / 2.0
    yaw_quaternion = (math.cos(half), 0.0, 0.0, math.sin(half))
    orientation = _quat_multiply(yaw_quaternion, tuple(pose[3:7]))
    return [*rotated, *orientation]


def _features(
    scene_state: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    geometry_policy: LoadedHPKPolicy,
) -> dict[str, Any]:
    projected = candidate_semantic_features(
        scene_state,
        candidate,
        geometry_policy=geometry_policy,
    )
    if isinstance(projected, HPKUnresolved):
        raise GeometryShiftPilotError(
            f"candidate feature projection unresolved: {projected.reason}"
        )
    return projected.to_dict()


def _transformed_candidate(
    candidate: Mapping[str, Any],
    *,
    translation: tuple[float, float, float],
    yaw_degrees: float,
    path: str,
) -> dict[str, Any]:
    result = dict(candidate)
    found_pose = False
    for field in _POSE_FIELDS:
        if field not in result:
            continue
        found_pose = True
        result[field] = _transform_pose(
            _pose(result[field], f"{path}.{field}"),
            translation=translation,
            yaw_degrees=yaw_degrees,
        )
    if not found_pose:
        raise GeometryShiftPilotError(f"{path} has no transformable pose")
    return result


def _ranked_semantics(
    candidates: Sequence[Mapping[str, Any]],
    *,
    strategy: GeometricStrategyV1,
    scene_state: Mapping[str, Any],
    geometry_policy: LoadedHPKPolicy,
) -> list[dict[str, Any]]:
    result = rank_operation_pose_candidates(
        candidates,
        geometric_strategy=strategy,
        scene_state=scene_state,
        geometry_policy=geometry_policy,
    )
    return [
        _features(
            scene_state,
            candidate,
            geometry_policy=geometry_policy,
        )
        for candidate in result.ranked_candidates
    ]


def run_pilot(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact(
        raw,
        {"schema", "scene_state", "candidates", "geometric_strategy", "shifts"},
        "input",
    )
    if raw["schema"] not in {INPUT_SCHEMA, "tcm/afk/rmbench_geometry_shift_pilot_input/v1"}:
        raise GeometryShiftPilotError("input.schema is unsupported")
    geometry_policy = load_safe_exploration_geometry_policy()
    if not isinstance(raw["scene_state"], Mapping):
        raise GeometryShiftPilotError("input.scene_state must be an object")
    scene_state = dict(raw["scene_state"])
    candidates_raw = raw["candidates"]
    if isinstance(candidates_raw, (str, bytes)) or not isinstance(
        candidates_raw, Sequence
    ):
        raise GeometryShiftPilotError("input.candidates must be an array")
    if not candidates_raw:
        raise GeometryShiftPilotError("input.candidates must not be empty")
    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates_raw):
        if not isinstance(candidate, Mapping):
            raise GeometryShiftPilotError(
                f"input.candidates[{index}] must be an object"
            )
        candidates.append(dict(candidate))
    if not isinstance(raw["geometric_strategy"], Mapping):
        raise GeometryShiftPilotError("input.geometric_strategy must be an object")
    strategy = GeometricStrategyV1.from_dict(raw["geometric_strategy"])
    source_features = [
        _features(
            scene_state,
            candidate,
            geometry_policy=geometry_policy,
        )
        for candidate in candidates
    ]
    source_ranking = _ranked_semantics(
        candidates,
        strategy=strategy,
        scene_state=scene_state,
        geometry_policy=geometry_policy,
    )

    shifts_raw = raw["shifts"]
    if isinstance(shifts_raw, (str, bytes)) or not isinstance(shifts_raw, Sequence):
        raise GeometryShiftPilotError("input.shifts must be an array")
    if not shifts_raw:
        raise GeometryShiftPilotError("input.shifts must not be empty")
    shifts: list[dict[str, Any]] = []
    names: set[str] = set()
    for shift_index, shift in enumerate(shifts_raw):
        path = f"input.shifts[{shift_index}]"
        if not isinstance(shift, Mapping):
            raise GeometryShiftPilotError(f"{path} must be an object")
        _exact(shift, {"name", "translation_m", "yaw_degrees"}, path)
        name = " ".join(str(shift["name"]).split())
        if not name or name in names:
            raise GeometryShiftPilotError("shift names must be non-empty and unique")
        names.add(name)
        translation = _vector3(shift["translation_m"], f"{path}.translation_m")
        yaw_degrees = _finite(shift["yaw_degrees"], f"{path}.yaw_degrees")
        if not any(abs(value) > 0.0 for value in translation) and yaw_degrees == 0.0:
            raise GeometryShiftPilotError(f"{path} must change the private pose")
        transformed = [
            _transformed_candidate(
                candidate,
                translation=translation,
                yaw_degrees=yaw_degrees,
                path=f"input.candidates[{candidate_index}]",
            )
            for candidate_index, candidate in enumerate(candidates)
        ]
        target_features = [
            _features(
                scene_state,
                candidate,
                geometry_policy=geometry_policy,
            )
            for candidate in transformed
        ]
        target_ranking = _ranked_semantics(
            transformed,
            strategy=strategy,
            scene_state=scene_state,
            geometry_policy=geometry_policy,
        )
        semantic_projection_preserved = target_features == source_features
        ranking_preserved = target_ranking == source_ranking
        shifts.append(
            {
                "name": name,
                "translation_norm_m": round(
                    math.sqrt(sum(value * value for value in translation)), 9
                ),
                "absolute_yaw_degrees": abs(yaw_degrees),
                "candidate_count": len(candidates),
                "semantic_projection_preserved": semantic_projection_preserved,
                "ranking_preserved": ranking_preserved,
                "compliant_candidate_count_before": len(source_ranking),
                "compliant_candidate_count_after": len(target_ranking),
                "mechanism_pass": bool(
                    semantic_projection_preserved
                    and ranking_preserved
                    and source_ranking
                ),
            }
        )
    passed = sum(bool(shift["mechanism_pass"]) for shift in shifts)
    return {
        "schema": REPORT_SCHEMA,
        "experiment_scope": "simulator-free relative-geometry mechanism check",
        "geometry_policy": geometry_policy.policy_id,
        "shift_count": len(shifts),
        "passed_shift_count": passed,
        "all_shifts_passed": passed == len(shifts),
        "motion_executed": False,
        "simulator_used": False,
        "physical_success_claimed": False,
        "public_output_is_semantic_only": True,
        "shifts": shifts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise GeometryShiftPilotError("input must be an object")
    report = run_pilot(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
