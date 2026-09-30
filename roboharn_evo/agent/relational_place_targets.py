from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from roboharn_evo.agent.perception.query_normalization import (
    normalize_entity_scope,
    normalize_expected_count,
    normalize_perception_object_id,
    normalize_placement_relation,
    sanitize_hint_text,
)


REFERENCE_REGIONS_KEY = "reference_regions"
RELATIONAL_PLACE_TARGET_KIND = "reference_region"

_CURRENT_VERIFIED = "current_verified"
_MEMORY_VALID = "memory_valid"
_MOTION_UNCERTAIN = "motion_uncertain"
_MAX_REGION_EXTENT_M = 0.60
_MIN_REGION_EXTENT_M = 0.01
_PLANAR_ABSOLUTE_Z_EXTENT_M = 0.04
_MULTIVIEW_DUPLICATE_DISTANCE_M = 0.035
_MIN_DOMINANT_PLANE_INLIER_RATIO = 0.25
_MIN_DOMINANT_PLANE_AXIS_COVERAGE = 0.60


class ReferenceRegionTracker:
    """Track geometry for semantic reference sets independently of objects.

    A reference region is not a manipulable scene instance.  It is a bounded
    spatial fact requested explicitly by the perception planner (for example,
    ``center_of`` a set).  Keeping it separate prevents a group mask from
    becoming a fake physical occupant or a low-level grasp target.
    """

    def __init__(self, *, max_missing_steps: int = 8) -> None:
        self.max_missing_steps = max(1, int(max_missing_steps))
        self._regions: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        self._regions.clear()

    def bind_instances(
        self,
        scene_memory: dict[str, Any],
        queries: list[dict[str, Any]],
        *,
        current_subtask: str,
        goal_contract: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        regions = bind_instance_placement_regions(
            scene_memory, queries, current_subtask=current_subtask,
            goal_contract=goal_contract,
        )
        self._regions = {
            key: region for key, region in self._regions.items()
            if region.get("binding_source") != "single_instance_query"
        }
        for region in regions:
            if region.get("binding_source") == "single_instance_query":
                self._regions[region["reference_region_id"]] = dict(region)
        return regions

    def update(
        self,
        *,
        segmentation: list[dict[str, Any]],
        instances: list[dict[str, Any]],
        env_step: int,
        current_subtask: str,
        position_events: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        step = int(env_step)
        subtask_key = " ".join(str(current_subtask or "").split())
        # Runtime motion facts invalidate prior geometry first; a grounded
        # observation from this update may then establish a newer fact.
        self.apply_runtime_events(position_events or [], env_step=step)
        grouped: dict[str, dict[str, Any]] = {}
        for item in segmentation:
            request = _reference_request(item)
            if request is None:
                continue
            key = str(request["reference_region_id"])
            group = grouped.setdefault(
                key,
                {
                    **request,
                    "observations": [],
                },
            )
            group["observations"].extend(
                _grounded_observations(item)
            )
            expected = normalize_expected_count(
                request.get("expected_count")
            )
            prior_expected = normalize_expected_count(
                group.get("expected_count")
            )
            if expected is not None:
                group["expected_count"] = max(
                    expected,
                    prior_expected or expected,
                )

        for key, request in grouped.items():
            observations = _deduplicate_observations(
                request.get("observations", [])
            )
            fresh = _build_region(
                request=request,
                observations=observations,
                instances=instances,
                env_step=step,
                current_subtask=subtask_key,
            )
            previous = self._regions.get(key)
            if fresh.get("support_valid") is True:
                continuity_rejection = (
                    _reference_region_continuity_rejection(
                        previous,
                        fresh,
                    )
                )
                if continuity_rejection:
                    retained = self._retained_previous_region(
                        previous,
                        env_step=step,
                        current_subtask=subtask_key,
                        evidence_status=(
                            "retained_after_lower_quality_current_observation"
                        ),
                    )
                    if retained is not None:
                        retained["current_observation_rejection_reason"] = (
                            continuity_rejection
                        )
                        retained["last_rejected_observation_step"] = step
                        self._regions[key] = retained
                        continue
                    unresolved = dict(previous or fresh)
                    unresolved["active_subtask"] = subtask_key
                    unresolved["last_requested_step"] = step
                    unresolved_last_observed = _finite_int(
                        unresolved.get("last_observed_step")
                    )
                    unresolved["missing_observation_steps"] = max(
                        0,
                        step
                        - (
                            step
                            if unresolved_last_observed is None
                            else unresolved_last_observed
                        ),
                    )
                    unresolved["support_valid"] = False
                    unresolved["position_state"] = _MOTION_UNCERTAIN
                    unresolved["evidence_status"] = (
                        "reference_region_continuity_unresolved"
                    )
                    unresolved["current_observation_rejection_reason"] = (
                        continuity_rejection
                    )
                    self._regions[key] = unresolved
                    continue
                self._regions[key] = fresh
                continue
            retained = self._retained_previous_region(
                previous,
                env_step=step,
                current_subtask=subtask_key,
                evidence_status=(
                    "retained_after_missing_current_observation"
                ),
            )
            if retained is not None:
                self._regions[key] = retained
                continue
            self._regions[key] = fresh

        active: list[dict[str, Any]] = []
        for key, region in list(self._regions.items()):
            if region.get("active_subtask") != subtask_key:
                continue
            last_requested = _finite_int(
                region.get("last_requested_step")
            )
            if last_requested is None:
                last_requested = -10**9
            if key not in grouped and step - last_requested > self.max_missing_steps:
                continue
            last_observed = _finite_int(
                region.get("last_observed_step")
            )
            if last_observed is None:
                last_observed = -10**9
            missing = max(0, step - last_observed)
            compact = dict(region)
            compact["missing_observation_steps"] = missing
            if (
                missing > 0
                and compact.get("position_state") == _CURRENT_VERIFIED
            ):
                compact["position_state"] = _MEMORY_VALID
            if missing > self.max_missing_steps:
                compact["support_valid"] = False
                compact["position_state"] = _MOTION_UNCERTAIN
                if compact.get("evidence_status") != (
                    "reference_region_continuity_unresolved"
                ):
                    compact["evidence_status"] = "reference_region_expired"
            active.append(compact)
        active.sort(key=lambda item: str(item.get("reference_region_id", "")))
        return active

    def _retained_previous_region(
        self,
        previous: Any,
        *,
        env_step: int,
        current_subtask: str,
        evidence_status: str,
    ) -> dict[str, Any] | None:
        if not isinstance(previous, dict):
            return None
        retained = dict(previous)
        # Re-emitting the same structured relation is the stable lifecycle
        # signal even if the planner paraphrases the current subtask text.
        retained["active_subtask"] = current_subtask
        retained["last_requested_step"] = int(env_step)
        retained_last_observed = _finite_int(
            retained.get("last_observed_step")
        )
        retained["missing_observation_steps"] = max(
            0,
            int(env_step)
            - (
                int(env_step)
                if retained_last_observed is None
                else retained_last_observed
            ),
        )
        if (
            retained["missing_observation_steps"] > self.max_missing_steps
            or retained.get("position_state") == _MOTION_UNCERTAIN
        ):
            return None
        retained["retained_evidence_status"] = str(
            retained.get(
                "retained_evidence_status",
                retained.get("evidence_status", ""),
            )
            or ""
        )
        retained["position_state"] = _MEMORY_VALID
        retained["evidence_status"] = evidence_status
        return retained

    def apply_runtime_events(
        self,
        events: list[dict[str, Any]],
        *,
        env_step: int,
    ) -> None:
        uncertain_refs = {
            str(event.get("instance_ref", "") or "").strip()
            for event in events
            if isinstance(event, dict)
            and str(event.get("position_state", "") or "").strip().lower()
            == _MOTION_UNCERTAIN
            and str(event.get("instance_ref", "") or "").strip()
        }
        if not uncertain_refs:
            return
        for key, region in list(self._regions.items()):
            reference_ids = {
                str(item or "").strip()
                for item in region.get("reference_instance_ids", []) or []
                if str(item or "").strip()
            }
            if not reference_ids.intersection(uncertain_refs):
                continue
            invalid = dict(region)
            invalid["support_valid"] = False
            invalid["position_state"] = _MOTION_UNCERTAIN
            invalid["evidence_status"] = (
                "reference_member_motion_uncertain"
            )
            invalid["invalidated_step"] = int(env_step)
            self._regions[key] = invalid


def bind_instance_placement_regions(
    scene_memory: dict[str, Any],
    queries: list[dict[str, Any]],
    *,
    current_subtask: str,
    goal_contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    regions = [
        dict(item) for item in _reference_regions(scene_memory)
        if item.get("binding_source") != "single_instance_query"
    ]
    for query in queries:
        relation = normalize_placement_relation(query.get("placement_relation"))
        if relation not in {"around", "on_top"}:
            continue
        reference = str(query.get("instance_ref", "") or "").strip()
        if not reference:
            continue
        matches = [
            item for item in scene_memory.get("instances", [])
            if reference in {item.get("instance_id"), item.get("track_id")}
        ]
        if len(matches) != 1:
            raise ValueError("placement reference must resolve to one observed scene instance")
        instance = matches[0]
        world = _xyz(instance.get("latest_world_m", instance.get("world_m")))
        extent = _xyz((instance.get("quality") or {}).get("world_extent_m"))
        if (
            world is None or extent is None
            or instance.get("position_state") not in {_CURRENT_VERIFIED, _MEMORY_VALID}
        ):
            continue
        top_surface = _xyz(instance.get("top_surface_world_m"))
        upper_z = (
            top_surface[2] if top_surface is not None
            else _finite_float((instance.get("quality") or {}).get("world_z_max_m"))
        )
        reference = str(instance["instance_id"])
        region = {
            "reference_region_id": f"reference-instance:{reference}:{relation}",
            "binding_source": "single_instance_query",
            "reference_object_id": str(query["object_id"]),
            "reference_instance_ids": [reference],
            "occupancy_exempt_instance_ids": [reference],
            "placement_relation": relation,
            "center_world_xy": world[:2],
            "footprint_world_min": [world[index] - abs(extent[index]) / 2.0 for index in range(3)],
            "footprint_world_max": [world[index] + abs(extent[index]) / 2.0 for index in range(3)],
            "footprint_extent_m": extent,
            "top_surface_world_z": upper_z,
            "position_state": instance["position_state"],
            "support_valid": True,
            "evidence_status": "bound_observed_reference_instance",
            "expected_count": 1,
            "observed_member_count": 1,
            "active_subtask": " ".join(current_subtask.split()),
            "last_requested_step": int(scene_memory["env_step"]),
            "last_observed_step": int(instance["last_verified_step"]),
        }
        if goal_contract is not None and goal_contract.get("operation") == "place":
            region["goal_target_role"] = goal_contract["required_target_role"]
            region["goal_target_relation"] = goal_contract["required_target_relation"]
        regions.append(region)
    return regions


def _region_object_target(
    region: dict[str, Any],
    extent: list[float],
    support_plane_z: float,
    source_supported_world: Any = None,
) -> list[float] | None:
    center = _xy(region.get("center_world_xy"))
    relation = normalize_placement_relation(region.get("placement_relation"))
    if center is None:
        return None
    if relation == "center_of":
        height = float(support_plane_z) + abs(extent[2]) / 2.0
    elif relation in {"around", "on_top"}:
        reference_extent = _xyz(region.get("footprint_extent_m"))
        if reference_extent is None:
            return None
        if relation == "around":
            source = _xyz(source_supported_world)
            if source is None or any(abs(extent[axis]) < abs(reference_extent[axis]) for axis in (0, 1)):
                return None
            # 保持被操作物体已观察到的支撑高度，并移动到参考物体周围。
            height = source[2]
        else:
            if any(abs(reference_extent[axis]) < abs(extent[axis]) for axis in (0, 1)):
                return None
            upper_z = _finite_float(region.get("top_surface_world_z"))
            if upper_z is None:
                return None
            height = upper_z + abs(extent[2]) / 2.0
    else:
        return None
    return [center[0], center[1], height]


def build_relational_place_target_specs(
    scene_memory: dict[str, Any],
    *,
    held_instance_id: str,
    held_extent_m: Any,
    support_plane_z: float,
    source_supported_world: Any = None,
) -> list[dict[str, Any]]:
    extent = _xyz(held_extent_m)
    held_id = str(held_instance_id or "").strip()
    if (
        not held_id
        or extent is None
        or not math.isfinite(float(support_plane_z))
    ):
        return []
    specs: list[dict[str, Any]] = []
    for region in _reference_regions(scene_memory):
        relation = normalize_placement_relation(
            region.get("placement_relation")
        )
        center = _xy(region.get("center_world_xy"))
        if (
            not relation
            or region.get("support_valid") is not True
            or region.get("position_state")
            not in {_CURRENT_VERIFIED, _MEMORY_VALID}
            or center is None
        ):
            continue
        region_id = str(
            region.get("reference_region_id", "") or ""
        ).strip()
        if not region_id:
            continue
        target = _region_object_target(region, extent, support_plane_z, source_supported_world)
        if target is None:
            continue
        specs.append(
            {
                "target_id": (
                    f"place:held:{held_id}:{region_id}"
                ),
                "target_kind": RELATIONAL_PLACE_TARGET_KIND,
                "held_object_target_world_m": _round_list(target),
                "placement_relation": relation,
                "goal_target_role": region.get("goal_target_role"),
                "goal_target_relation": region.get("goal_target_relation"),
                "source_supported_world_m": _xyz(source_supported_world),
                "reference_region_id": region_id,
                "reference_object_id": region.get("reference_object_id"),
                "reference_instance_ids": list(
                    region.get("reference_instance_ids", []) or []
                ),
                "occupancy_exempt_instance_ids": list(
                    region.get("occupancy_exempt_instance_ids", []) or []
                ),
                "expected_count": region.get("expected_count"),
                "observed_member_count": region.get(
                    "observed_member_count"
                ),
                "reference_evidence_status": region.get(
                    "evidence_status"
                ),
                "support_instance_id": None,
                "vacated_by_instance_id": None,
                "support_evidence": (
                    "grounded_reference_region_center" if relation == "center_of"
                    else "grounded_reference_region_" + relation
                ),
                # The relation was explicitly requested for this subtask, so
                # it wins metric de-duplication against generic buffer poses.
                "priority": -1.0,
            }
        )
    return specs


def revalidate_relational_place_target(
    scene_memory: dict[str, Any],
    *,
    candidate: dict[str, Any],
    held_extent_m: Any,
    support_plane_z: float | None,
) -> dict[str, Any]:
    region_id = str(
        candidate.get("reference_region_id", "") or ""
    ).strip()
    relation = normalize_placement_relation(
        candidate.get("placement_relation")
    )
    target = _xyz(candidate.get("held_object_target_world_m"))
    extent = _xyz(held_extent_m)
    region = next(
        (
            item
            for item in _reference_regions(scene_memory)
            if str(item.get("reference_region_id", "") or "").strip()
            == region_id
        ),
        None,
    )
    if (
        region is None
        or not relation
        or normalize_placement_relation(
            region.get("placement_relation")
        )
        != relation
        or region.get("support_valid") is not True
        or region.get("position_state")
        not in {_CURRENT_VERIFIED, _MEMORY_VALID}
        or target is None
        or extent is None
        or support_plane_z is None
        or not math.isfinite(float(support_plane_z))
    ):
        return {
            "support_valid": False,
            "target_geometry_drift_m": None,
            "occupancy_exempt_instance_ids": [],
        }
    expected = _region_object_target(
        region, extent, float(support_plane_z),
        candidate.get("source_supported_world_m"),
    )
    if expected is None:
        return {
            "support_valid": False,
            "target_geometry_drift_m": None,
            "occupancy_exempt_instance_ids": [],
        }
    drift = math.sqrt(
        sum((expected[index] - target[index]) ** 2 for index in range(3))
    )
    return {
        "support_valid": True,
        "target_geometry_drift_m": drift,
        "occupancy_exempt_instance_ids": list(
            region.get("occupancy_exempt_instance_ids", []) or []
        ),
    }


def compact_reference_regions(
    scene_memory: Any,
    *,
    max_regions: int = 4,
) -> list[dict[str, Any]]:
    public_keys = (
        "reference_region_id",
        "reference_object_id",
        "placement_relation",
        "goal_target_role",
        "goal_target_relation",
        "center_world_xy",
        "position_state",
        "support_valid",
        "evidence_status",
        "expected_count",
        "observed_member_count",
        "reference_instance_ids",
    )
    return [
        {
            key: item.get(key)
            for key in public_keys
            if key in item
        }
        for item in _reference_regions(scene_memory)[: max(0, int(max_regions))]
    ]


def _reference_request(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    relation = normalize_placement_relation(
        item.get("placement_relation")
    )
    scope = normalize_entity_scope(item.get("entity_scope"))
    if scope != "reference_set" or relation != "center_of":
        return None
    object_id, _ = normalize_perception_object_id(
        item.get("object_id", "")
    )
    if not object_id:
        return None
    hint = sanitize_hint_text(
        item.get("query_instance_hint", item.get("instance_hint", ""))
    )
    identity = {
        "object_id": object_id,
        "placement_relation": relation,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    request: dict[str, Any] = {
        "reference_region_id": f"reference-region:{digest}",
        "reference_object_id": object_id,
        "reference_instance_hint": hint,
        "placement_relation": relation,
        "query_reason": str(
            item.get("query_reason", item.get("reason", "")) or ""
        ).strip(),
    }
    expected = normalize_expected_count(item.get("expected_count"))
    if expected is not None:
        request["expected_count"] = expected
    return request


def _grounded_observations(item: dict[str, Any]) -> list[dict[str, Any]]:
    detections = item.get("detections")
    raw = (
        [entry for entry in detections if isinstance(entry, dict)]
        if isinstance(detections, list) and detections
        else [item]
    )
    observations: list[dict[str, Any]] = []
    for entry in raw:
        quality = entry.get("quality")
        if isinstance(quality, dict) and quality.get("actionable") is False:
            continue
        grounding = entry.get("grounding_3d")
        if not isinstance(grounding, dict) or grounding.get("success") is not True:
            continue
        lower = _xyz(grounding.get("bbox_world_min"))
        upper = _xyz(grounding.get("bbox_world_max"))
        center = _xyz(grounding.get("centroid_world"))
        if lower is None or upper is None:
            continue
        minimum = [min(lower[index], upper[index]) for index in range(3)]
        maximum = [max(lower[index], upper[index]) for index in range(3)]
        raw_minimum = list(minimum)
        raw_maximum = list(maximum)
        geometry_source = "raw_grounded_mask_bounds"
        plane_metadata: dict[str, Any] = {}
        dominant_bounds = _usable_dominant_plane_bounds(
            grounding,
            raw_minimum=raw_minimum,
            raw_maximum=raw_maximum,
        )
        if dominant_bounds is not None:
            minimum, maximum, plane_metadata = dominant_bounds
            center = [
                (minimum[index] + maximum[index]) / 2.0
                for index in range(3)
            ]
            geometry_source = "dominant_horizontal_plane_footprint"
        extent = [maximum[index] - minimum[index] for index in range(3)]
        if (
            any(not math.isfinite(value) for value in extent)
            or extent[0] <= 0.0
            or extent[1] <= 0.0
            or extent[2] < 0.0
        ):
            continue
        if center is None:
            center = [
                (minimum[index] + maximum[index]) / 2.0
                for index in range(3)
            ]
        observations.append(
            {
                "center_world_m": center,
                "bbox_world_min": minimum,
                "bbox_world_max": maximum,
                "extent_m": extent,
                "geometry_source": geometry_source,
                **(
                    {
                        "raw_bbox_world_min": raw_minimum,
                        "raw_bbox_world_max": raw_maximum,
                        **plane_metadata,
                    }
                    if dominant_bounds is not None
                    else {}
                ),
                "score": _finite_float(entry.get("score", item.get("score"))),
                "camera": str(
                    entry.get("camera", item.get("camera", "")) or ""
                ).strip(),
                "rank": _finite_int(entry.get("rank")),
            }
        )
    return observations


def _usable_dominant_plane_bounds(
    grounding: dict[str, Any],
    *,
    raw_minimum: list[float],
    raw_maximum: list[float],
) -> tuple[list[float], list[float], dict[str, Any]] | None:
    plane = grounding.get("dominant_plane_footprint")
    if not isinstance(plane, dict) or plane.get("valid") is not True:
        return None
    lower = _xyz(plane.get("bbox_world_min"))
    upper = _xyz(plane.get("bbox_world_max"))
    inlier_ratio = _finite_float(plane.get("inlier_ratio"))
    if (
        lower is None
        or upper is None
        or inlier_ratio is None
        or inlier_ratio < _MIN_DOMINANT_PLANE_INLIER_RATIO
    ):
        return None
    minimum = [min(lower[index], upper[index]) for index in range(3)]
    maximum = [max(lower[index], upper[index]) for index in range(3)]
    raw_extent = [
        raw_maximum[index] - raw_minimum[index]
        for index in range(3)
    ]
    plane_extent = [
        maximum[index] - minimum[index]
        for index in range(3)
    ]
    if any(
        raw_extent[index] <= 0.0 or plane_extent[index] <= 0.0
        for index in (0, 1)
    ):
        return None
    axis_coverage = [
        plane_extent[index] / raw_extent[index]
        for index in (0, 1)
    ]
    if min(axis_coverage) < _MIN_DOMINANT_PLANE_AXIS_COVERAGE:
        return None
    return (
        minimum,
        maximum,
        {
            "dominant_plane_inlier_ratio": inlier_ratio,
            "dominant_plane_axis_coverage": axis_coverage,
            "dominant_plane_source": str(
                plane.get("source", "dominant_horizontal_z_band")
                or "dominant_horizontal_z_band"
            ),
        },
    )


def _deduplicate_observations(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for observation in sorted(
        observations,
        key=lambda item: (
            -_footprint_area(item),
            -(_finite_float(item.get("score")) or 0.0),
            str(item.get("camera", "")),
        ),
    ):
        duplicate = next(
            (
                existing
                for existing in unique
                if _same_physical_footprint(existing, observation)
            ),
            None,
        )
        if duplicate is None:
            unique.append(dict(observation))
            continue
        cameras = sorted(
            {
                str(duplicate.get("camera", "") or "").strip(),
                str(observation.get("camera", "") or "").strip(),
            }
            - {""}
        )
        duplicate["supporting_cameras"] = cameras
        if (_finite_float(observation.get("score")) or 0.0) > (
            _finite_float(duplicate.get("score")) or 0.0
        ):
            cameras = duplicate.get("supporting_cameras", cameras)
            duplicate.update(dict(observation))
            duplicate["supporting_cameras"] = cameras
    return unique


def _build_region(
    *,
    request: dict[str, Any],
    observations: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    env_step: int,
    current_subtask: str,
) -> dict[str, Any]:
    base = {
        key: value
        for key, value in request.items()
        if key != "observations"
    }
    base.update(
        {
            "active_subtask": current_subtask,
            "last_requested_step": int(env_step),
            "last_observed_step": int(env_step),
            "missing_observation_steps": 0,
            "position_state": _MOTION_UNCERTAIN,
            "support_valid": False,
            "evidence_status": "insufficient_grounded_reference_evidence",
            "reference_instance_ids": [],
            "occupancy_exempt_instance_ids": [],
            "observed_member_count": 0,
        }
    )
    if not observations:
        return base

    lower = [
        min(float(item["bbox_world_min"][axis]) for item in observations)
        for axis in range(3)
    ]
    upper = [
        max(float(item["bbox_world_max"][axis]) for item in observations)
        for axis in range(3)
    ]
    region_extent = [upper[index] - lower[index] for index in range(3)]
    if (
        min(region_extent[:2]) < _MIN_REGION_EXTENT_M
        or max(region_extent[:2]) > _MAX_REGION_EXTENT_M
    ):
        base["evidence_status"] = "reference_region_extent_invalid"
        return base

    aggregate = _aggregate_observation(observations)
    if aggregate is not None:
        # A group mask's 3-D centroid is biased by visible pixel density.  Its
        # world AABB midpoint is the symmetric reference-set center.
        center_xy = [
            (
                float(aggregate["bbox_world_min"][axis])
                + float(aggregate["bbox_world_max"][axis])
            )
            / 2.0
            for axis in (0, 1)
        ]
        footprint_min = list(aggregate["bbox_world_min"])
        footprint_max = list(aggregate["bbox_world_max"])
    else:
        center_xy = [
            (lower[axis] + upper[axis]) / 2.0
            for axis in (0, 1)
        ]
        footprint_min = lower
        footprint_max = upper

    planar_aggregate = bool(
        aggregate is not None and _is_planar_footprint(aggregate)
    )
    component_observations = [
        item for item in observations if item is not aggregate
    ]
    expected_count = normalize_expected_count(
        request.get("expected_count")
    )
    count_satisfied = bool(
        aggregate is not None
        or expected_count is None
        or len(observations) >= expected_count
    )
    sufficient = bool(
        count_satisfied
        and (
            len(observations) >= 2
            or planar_aggregate
            or expected_count == 1
        )
    )
    if not sufficient:
        base["evidence_status"] = (
            "reference_set_member_count_incomplete"
            if expected_count is not None
            and len(observations) < expected_count
            and aggregate is None
            else "reference_set_requires_multiple_or_planar_evidence"
        )
        base["observation_count"] = len(observations)
        base["observed_member_count"] = len(observations)
        return base

    matched_ids = [
        match
        for observation in observations
        if (
            match := _matching_instance_id(
                observation,
                instances=instances,
                object_id=str(request.get("reference_object_id", "")),
            )
        )
    ]
    matched_ids = sorted(set(matched_ids))
    aggregate_id = (
        _matching_instance_id(
            aggregate,
            instances=instances,
            object_id=str(request.get("reference_object_id", "")),
        )
        if aggregate is not None
        else ""
    )
    occupancy_exempt = []
    if planar_aggregate and aggregate_id:
        occupancy_exempt.append(aggregate_id)
    elif (
        len(observations) == 1
        and _is_planar_footprint(observations[0])
        and normalize_expected_count(request.get("expected_count")) == 1
        and matched_ids
    ):
        occupancy_exempt.append(matched_ids[0])

    base.update(
        {
            "center_world_xy": _round_list(center_xy),
            "footprint_world_min": _round_list(footprint_min),
            "footprint_world_max": _round_list(footprint_max),
            "footprint_extent_m": _round_list(
                [
                    footprint_max[index] - footprint_min[index]
                    for index in range(3)
                ]
            ),
            "position_state": _CURRENT_VERIFIED,
            "support_valid": True,
            "evidence_status": (
                "grounded_aggregate_reference_footprint"
                if aggregate is not None
                else "grounded_reference_member_envelope"
            ),
            "observation_count": len(observations),
            "expected_count_satisfied": count_satisfied,
            "observed_member_count": (
                len(component_observations)
                if aggregate is not None
                else len(observations)
            ),
            "reference_instance_ids": matched_ids,
            "aggregate_instance_id": aggregate_id or None,
            "occupancy_exempt_instance_ids": occupancy_exempt,
            "supporting_cameras": sorted(
                {
                    camera
                    for observation in observations
                    for camera in (
                        list(observation.get("supporting_cameras", []) or [])
                        or [str(observation.get("camera", "") or "")]
                    )
                    if camera
                }
            ),
            "confidence": max(
                (
                    _finite_float(item.get("score")) or 0.0
                    for item in observations
                ),
                default=0.0,
            ),
        }
    )
    return base


def _aggregate_observation(
    observations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(observations) == 1:
        return observations[0]
    ranked = sorted(observations, key=_footprint_area, reverse=True)
    largest = ranked[0]
    second_area = _footprint_area(ranked[1])
    contained = sum(
        1
        for item in ranked[1:]
        if _contains_xy(largest, item.get("center_world_m"))
    )
    if (
        contained >= min(2, len(ranked) - 1)
        and _footprint_area(largest) >= max(1e-9, second_area) * 2.0
    ):
        return largest
    return None


def _matching_instance_id(
    observation: dict[str, Any] | None,
    *,
    instances: list[dict[str, Any]],
    object_id: str,
) -> str:
    if not isinstance(observation, dict):
        return ""
    center = _xyz(observation.get("center_world_m"))
    extent = _xyz(observation.get("extent_m"))
    if center is None or extent is None:
        return ""
    candidates: list[tuple[float, str]] = []
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        tokens = {
            normalize_perception_object_id(instance.get(key, ""))[0]
            for key in ("class", "source_object_id")
        }
        if object_id and object_id not in tokens:
            continue
        world = _xyz(
            instance.get("latest_world_m", instance.get("world_m"))
        )
        quality = instance.get("quality")
        instance_extent = _xyz(
            quality.get("world_extent_m")
            if isinstance(quality, dict)
            else None
        )
        instance_id = str(instance.get("instance_id", "") or "").strip()
        if world is None or instance_extent is None or not instance_id:
            continue
        distance = math.sqrt(
            sum((world[index] - center[index]) ** 2 for index in range(3))
        )
        tolerance = max(
            0.02,
            0.20 * math.hypot(extent[0], extent[1]),
        )
        if distance > tolerance:
            continue
        extent_error = sum(
            abs(instance_extent[index] - extent[index])
            / max(0.01, extent[index])
            for index in range(3)
        )
        candidates.append((distance + 0.005 * extent_error, instance_id))
    if not candidates:
        return ""
    candidates.sort()
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.005:
        return ""
    return candidates[0][1]


def _same_physical_footprint(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    a = _xyz(first.get("center_world_m"))
    b = _xyz(second.get("center_world_m"))
    if a is None or b is None:
        return False
    distance = math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))
    if distance > _MULTIVIEW_DUPLICATE_DISTANCE_M:
        return False
    return _xy_iou(first, second) >= 0.35


def _xy_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    a_min = _xyz(first.get("bbox_world_min"))
    a_max = _xyz(first.get("bbox_world_max"))
    b_min = _xyz(second.get("bbox_world_min"))
    b_max = _xyz(second.get("bbox_world_max"))
    if None in (a_min, a_max, b_min, b_max):
        return 0.0
    assert a_min is not None and a_max is not None
    assert b_min is not None and b_max is not None
    intersection = 1.0
    for axis in (0, 1):
        intersection *= max(
            0.0,
            min(a_max[axis], b_max[axis])
            - max(a_min[axis], b_min[axis]),
        )
    first_area = _footprint_area(first)
    second_area = _footprint_area(second)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _contains_xy(container: dict[str, Any], point: Any) -> bool:
    lower = _xyz(container.get("bbox_world_min"))
    upper = _xyz(container.get("bbox_world_max"))
    xyz = _xyz(point)
    if lower is None or upper is None or xyz is None:
        return False
    margin = 0.005
    return all(
        lower[axis] - margin <= xyz[axis] <= upper[axis] + margin
        for axis in (0, 1)
    )


def _is_planar_footprint(observation: dict[str, Any]) -> bool:
    extent = _xyz(observation.get("extent_m"))
    return _is_planar_extent(extent)


def _is_planar_extent(extent: Any) -> bool:
    extent = _xyz(extent)
    if extent is None or min(extent[:2]) <= 0.0:
        return False
    return extent[2] <= min(
        _PLANAR_ABSOLUTE_Z_EXTENT_M,
        0.25 * min(extent[0], extent[1]),
    )


def _reference_region_continuity_rejection(
    previous: Any,
    fresh: dict[str, Any],
) -> str:
    """Reject a lower-quality view without creating a new lifecycle.

    Runtime motion events invalidate a region before this comparison.  While a
    complete prior reference set remains physically valid, a robot-contaminated
    volume or a lower-evidence contained crop must not replace its world
    geometry.  A complete new observation with equal or better evidence remains
    free to refine the region.
    """

    if not isinstance(previous, dict):
        return ""
    trusted_or_unresolved = bool(
        (
            previous.get("support_valid") is True
            and previous.get("position_state")
            in {_CURRENT_VERIFIED, _MEMORY_VALID}
        )
        or previous.get("evidence_status")
        == "reference_region_continuity_unresolved"
    )
    if (
        not trusted_or_unresolved
        or not _has_complete_reference_member_evidence(previous)
    ):
        return ""
    previous_extent = _xyz(previous.get("footprint_extent_m"))
    fresh_extent = _xyz(fresh.get("footprint_extent_m"))
    if (
        _is_planar_extent(previous_extent)
        and not _is_planar_extent(fresh_extent)
    ):
        return "complete_planar_reference_became_volumetric"
    if (
        _reference_region_evidence_priority(fresh)
        < _reference_region_evidence_priority(previous)
        and _is_materially_contained_reference_footprint(
            previous,
            fresh,
        )
    ):
        return "lower_evidence_partial_reference_footprint"
    return ""


def _has_complete_reference_member_evidence(region: dict[str, Any]) -> bool:
    evidence_status = str(
        region.get(
            "retained_evidence_status",
            region.get("evidence_status", ""),
        )
        or ""
    )
    if evidence_status != "grounded_reference_member_envelope":
        return False
    observed = _finite_int(region.get("observed_member_count")) or 0
    expected = normalize_expected_count(region.get("expected_count"))
    return observed >= (expected if expected is not None else 2)


def _reference_region_evidence_priority(
    region: dict[str, Any],
) -> tuple[int, int, int]:
    observed = _finite_int(region.get("observed_member_count")) or 0
    expected = normalize_expected_count(region.get("expected_count"))
    complete = int(expected is None or observed >= expected)
    evidence_status = str(
        region.get(
            "retained_evidence_status",
            region.get("evidence_status", ""),
        )
        or ""
    )
    evidence_rank = {
        "grounded_aggregate_reference_footprint": 1,
        "grounded_reference_member_envelope": 2,
    }.get(evidence_status, 0)
    return complete, evidence_rank, observed


def _is_materially_contained_reference_footprint(
    previous: dict[str, Any],
    fresh: dict[str, Any],
) -> bool:
    previous_min = _xyz(previous.get("footprint_world_min"))
    previous_max = _xyz(previous.get("footprint_world_max"))
    fresh_min = _xyz(fresh.get("footprint_world_min"))
    fresh_max = _xyz(fresh.get("footprint_world_max"))
    if None in (previous_min, previous_max, fresh_min, fresh_max):
        return False
    assert previous_min is not None and previous_max is not None
    assert fresh_min is not None and fresh_max is not None
    tolerance = _MIN_REGION_EXTENT_M
    if not all(
        fresh_min[axis] >= previous_min[axis] - tolerance
        and fresh_max[axis] <= previous_max[axis] + tolerance
        for axis in (0, 1)
    ):
        return False
    lost_boundary = max(
        max(0.0, fresh_min[axis] - previous_min[axis])
        for axis in (0, 1)
    )
    lost_boundary = max(
        lost_boundary,
        max(
            max(0.0, previous_max[axis] - fresh_max[axis])
            for axis in (0, 1)
        ),
    )
    return lost_boundary > _MIN_REGION_EXTENT_M


def _footprint_area(observation: dict[str, Any]) -> float:
    extent = _xyz(observation.get("extent_m"))
    return 0.0 if extent is None else max(0.0, extent[0] * extent[1])


def _reference_regions(scene_memory: Any) -> list[dict[str, Any]]:
    if not isinstance(scene_memory, dict):
        return []
    return [
        dict(item)
        for item in scene_memory.get(REFERENCE_REGIONS_KEY, []) or []
        if isinstance(item, dict)
    ]


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _xyz(value: Any) -> list[float] | None:
    try:
        xyz = [float(item) for item in list(value)]
    except (TypeError, ValueError):
        return None
    if len(xyz) != 3 or not all(math.isfinite(item) for item in xyz):
        return None
    return xyz


def _xy(value: Any) -> list[float] | None:
    try:
        xy = [float(item) for item in list(value)]
    except (TypeError, ValueError):
        return None
    if len(xy) != 2 or not all(math.isfinite(item) for item in xy):
        return None
    return xy


def _round_list(value: list[float], *, digits: int = 6) -> list[float]:
    return [round(float(item), digits) for item in value]
