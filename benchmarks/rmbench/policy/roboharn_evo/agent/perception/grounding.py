from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..operation_candidates import (
    action_to_tcp_transform,
    matrix_to_pose7,
    pose7_to_matrix,
)


_ROBUST_GEOMETRY_LOWER_PERCENTILE = 5.0
_DOMINANT_HORIZONTAL_BAND_WIDTH_M = 0.008
_DOMINANT_HORIZONTAL_MIN_INLIER_RATIO = 0.25
_DOMINANT_HORIZONTAL_MIN_INLIERS = 6
_DOMINANT_HORIZONTAL_XY_PERCENTILE = 0.5


def ground_segmentation_result(
    *,
    segmentation: dict[str, Any],
    depth_mm: Any,
    intrinsic_cv: Any,
    cam2world_gl: Any,
    camera: str = "head",
    min_valid_ratio: float = 0.05,
    max_points: int = 5000,
    approach_height_m: float = 0.08,
    ee_to_contact_m: float = 0.0,
    tcp_calibration_by_arm: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Lift a 2D segmentation result into a compact RGB-D grounding summary."""
    object_id = str(segmentation.get("object_id", "object"))
    try:
        depth = np.asarray(depth_mm, dtype=np.float64)
    except Exception:
        return _failure(object_id=object_id, camera=camera, error="invalid depth array")
    if depth.ndim != 2 or depth.size == 0:
        return _failure(object_id=object_id, camera=camera, error="missing depth image")

    try:
        intrinsic = np.asarray(intrinsic_cv, dtype=np.float64)
        transform = np.asarray(cam2world_gl, dtype=np.float64)
    except Exception:
        return _failure(object_id=object_id, camera=camera, error="invalid camera calibration")
    if intrinsic.shape != (3, 3):
        return _failure(object_id=object_id, camera=camera, error="missing intrinsic_cv")
    if transform.shape != (4, 4):
        return _failure(object_id=object_id, camera=camera, error="missing cam2world_gl")

    mask = _mask_from_segmentation(segmentation, depth.shape)
    if mask is None:
        return _failure(
            object_id=object_id,
            camera=camera,
            error="missing mask_path and bbox_xyxy",
        )
    mask_pixel_count = int(np.count_nonzero(mask))
    if mask_pixel_count <= 0:
        return _failure(object_id=object_id, camera=camera, error="empty segmentation mask")

    valid = mask & np.isfinite(depth) & (depth > 0.0)
    valid_pixel_count = int(np.count_nonzero(valid))
    valid_ratio = valid_pixel_count / max(1, mask_pixel_count)
    if valid_pixel_count <= 0:
        return _failure(
            object_id=object_id,
            camera=camera,
            error="no valid depth inside segmentation",
            mask_pixel_count=mask_pixel_count,
            valid_pixel_count=valid_pixel_count,
            valid_ratio=round(valid_ratio, 6),
        )
    if valid_ratio < max(0.0, float(min_valid_ratio)):
        return _failure(
            object_id=object_id,
            camera=camera,
            error=(
                f"valid depth ratio {valid_ratio:.4f} below threshold "
                f"{float(min_valid_ratio):.4f}"
            ),
            mask_pixel_count=mask_pixel_count,
            valid_pixel_count=valid_pixel_count,
            valid_ratio=round(valid_ratio, 6),
        )

    rows, cols = np.nonzero(valid)
    if rows.size > max(1, int(max_points)):
        indexes = np.linspace(
            0,
            rows.size - 1,
            num=max(1, int(max_points)),
            dtype=np.int64,
        )
        rows = rows[indexes]
        cols = cols[indexes]

    world_points = _pixels_to_world(
        u=cols.astype(np.float64),
        v=rows.astype(np.float64),
        depth_mm=depth[rows, cols],
        intrinsic_cv=intrinsic,
        cam2world_gl=transform,
    )
    if world_points.size == 0:
        return _failure(
            object_id=object_id,
            camera=camera,
            error="failed to project depth to world",
        )

    centroid_world = np.mean(world_points, axis=0)
    bbox_world_min = np.min(world_points, axis=0)
    bbox_world_max = np.max(world_points, axis=0)
    dominant_plane_footprint = _dominant_horizontal_plane_footprint(
        world_points
    )
    top_surface_world, surface_normal_world, surface_sample_count = _robust_top_surface(
        world_points
    )
    principal_axes_world = _surface_principal_axes(
        world_points,
        surface_normal_world=surface_normal_world,
    )
    grasp_geometry = _observed_grasp_geometry(
        world_points,
        surface_world=top_surface_world,
        surface_normal_world=surface_normal_world,
    )
    contact_geometry = _surface_contact_geometry(
        surface_world=top_surface_world,
        surface_normal_world=surface_normal_world,
        approach_height_m=approach_height_m,
        ee_to_contact_m=ee_to_contact_m,
        grasp_object_contact_world=(
            grasp_geometry["object_contact_world"]
            if grasp_geometry is not None
            else None
        ),
        grasp_tcp_world=(
            grasp_geometry["tcp_target_world"]
            if grasp_geometry is not None
            else None
        ),
    )
    operation_candidates = _surface_operation_candidates(
        world_points=world_points,
        surface_world=top_surface_world,
        surface_normal_world=surface_normal_world,
        principal_axes_world=principal_axes_world,
        grasp_geometry=grasp_geometry,
        approach_height_m=approach_height_m,
        ee_to_contact_m=ee_to_contact_m,
        tcp_calibration_by_arm=tcp_calibration_by_arm,
    )

    return {
        "success": True,
        "object_id": object_id,
        "camera": str(camera),
        "depth_units": "mm",
        "mask_pixel_count": mask_pixel_count,
        "valid_pixel_count": valid_pixel_count,
        "valid_ratio": _round_float(valid_ratio),
        "sampled_point_count": int(world_points.shape[0]),
        "surface_sample_count": int(surface_sample_count),
        "centroid_world": _round_list(centroid_world),
        "bbox_world_min": _round_list(bbox_world_min),
        "bbox_world_max": _round_list(bbox_world_max),
        "dominant_plane_footprint": dominant_plane_footprint,
        "top_surface_world": _round_list(top_surface_world),
        "surface_normal_world": _round_list(surface_normal_world),
        "principal_axes_world": [_round_list(axis) for axis in principal_axes_world],
        "grasp_geometry": (
            _public_grasp_geometry(grasp_geometry)
            if grasp_geometry is not None
            else {
                "valid": False,
                "source": "insufficient_observed_normal_extent",
            }
        ),
        "operation_pose_candidates": operation_candidates,
        **contact_geometry,
    }


def _mask_from_segmentation(
    segmentation: dict[str, Any],
    shape: tuple[int, int],
) -> np.ndarray | None:
    mask = _load_mask(segmentation.get("mask_path"), shape)
    if mask is not None and np.count_nonzero(mask) > 0:
        return mask
    bbox = _parse_bbox(segmentation.get("bbox_xyxy"))
    if bbox is None:
        detections = segmentation.get("detections")
        if isinstance(detections, list):
            for detection in detections:
                if isinstance(detection, dict):
                    bbox = _parse_bbox(detection.get("bbox_xyxy"))
                    if bbox is not None:
                        break
    if bbox is None:
        return None
    return _bbox_mask(bbox, shape)


def _load_mask(mask_path: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if not mask_path:
        return None
    try:
        path = Path(str(mask_path)).expanduser()
        if not path.exists():
            return None
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"))
    except Exception:
        return None
    if mask.shape != shape:
        try:
            with Image.open(path) as image:
                resized = image.convert("L").resize(
                    (shape[1], shape[0]),
                    Image.Resampling.NEAREST,
                )
                mask = np.asarray(resized)
        except Exception:
            return None
    return mask > 0


def _parse_bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        import json

        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except Exception:
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    return bbox


def _bbox_mask(bbox: list[float], shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = bbox
    left = max(0, min(width, int(math.floor(min(x1, x2)))))
    right = max(0, min(width, int(math.ceil(max(x1, x2)))))
    top = max(0, min(height, int(math.floor(min(y1, y2)))))
    bottom = max(0, min(height, int(math.ceil(max(y1, y2)))))
    mask = np.zeros((height, width), dtype=bool)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = True
    return mask


def _pixels_to_world(
    *,
    u: np.ndarray,
    v: np.ndarray,
    depth_mm: np.ndarray,
    intrinsic_cv: np.ndarray,
    cam2world_gl: np.ndarray,
) -> np.ndarray:
    fx = float(intrinsic_cv[0, 0])
    fy = float(intrinsic_cv[1, 1])
    cx = float(intrinsic_cv[0, 2])
    cy = float(intrinsic_cv[1, 2])
    if abs(fx) <= 1e-9 or abs(fy) <= 1e-9:
        return np.zeros((0, 3), dtype=np.float64)

    z_m = np.asarray(depth_mm, dtype=np.float64) / 1000.0
    x_gl = (u - cx) * z_m / fx
    y_gl = -(v - cy) * z_m / fy
    z_gl = -z_m
    ones = np.ones_like(z_m)
    camera_points = np.stack([x_gl, y_gl, z_gl, ones], axis=0)
    world_h = cam2world_gl @ camera_points
    world = world_h[:3, :].T
    finite = np.all(np.isfinite(world), axis=1)
    return world[finite]


def _robust_top_surface(
    world_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    z_values = np.asarray(world_points[:, 2], dtype=np.float64)
    if z_values.size <= 1:
        return (
            world_points[0].copy(),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            1,
        )
    threshold = float(np.percentile(z_values, 90.0))
    top_band = world_points[z_values >= threshold]
    if top_band.size == 0:
        top_band = world_points[[int(np.argmax(z_values))]]
    # Preserve the full-mask lateral center. On rounded or perspective-viewed
    # objects, a highest-depth band is often concentrated at one edge.
    surface = np.mean(world_points, axis=0)
    surface[2] = float(np.percentile(top_band[:, 2], 95.0))
    return surface, _outward_top_surface_normal(top_band), int(top_band.shape[0])


def _dominant_horizontal_plane_footprint(
    world_points: np.ndarray,
) -> dict[str, Any]:
    """Extract the largest thin horizontal layer from a grounded mask.

    Aggregate segmentation masks can contain a broad support surface together
    with a much smaller raised object.  Their raw 3-D AABB is then thick even
    though the relation-defining footprint is planar.  This routine reports a
    data-derived horizontal layer while retaining the unmodified raw bounds in
    the parent grounding result.  Consumers must still check the layer's
    support ratio and XY coverage before using it.
    """

    points = np.asarray(world_points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] == 0
    ):
        return {
            "valid": False,
            "source": "dominant_horizontal_z_band",
            "reason": "missing_world_points",
        }
    points = points[np.all(np.isfinite(points), axis=1)]
    point_count = int(points.shape[0])
    if point_count == 0:
        return {
            "valid": False,
            "source": "dominant_horizontal_z_band",
            "reason": "missing_finite_world_points",
        }

    order = np.argsort(points[:, 2], kind="stable")
    sorted_z = points[order, 2]
    left = 0
    best_left = 0
    best_right = 0
    best_count = 0
    for right in range(point_count):
        while (
            left < right
            and sorted_z[right] - sorted_z[left]
            > _DOMINANT_HORIZONTAL_BAND_WIDTH_M
        ):
            left += 1
        count = right - left + 1
        # On an exact tie retain the lower world-Z layer encountered first.
        # For a support mask plus a raised object this prefers the support,
        # without using any task, class, or fixed-coordinate knowledge.
        if count > best_count:
            best_left = left
            best_right = right
            best_count = count

    inlier_ratio = best_count / max(1, point_count)
    required_inliers = min(
        point_count,
        max(
            _DOMINANT_HORIZONTAL_MIN_INLIERS,
            int(
                math.ceil(
                    _DOMINANT_HORIZONTAL_MIN_INLIER_RATIO
                    * point_count
                )
            ),
        ),
    )
    if best_count < required_inliers:
        return {
            "valid": False,
            "source": "dominant_horizontal_z_band",
            "reason": "insufficient_dominant_plane_support",
            "point_count": point_count,
            "inlier_count": best_count,
            "inlier_ratio": _round_float(inlier_ratio),
            "band_width_m": _DOMINANT_HORIZONTAL_BAND_WIDTH_M,
        }

    inlier_indices = order[best_left : best_right + 1]
    inliers = points[inlier_indices]
    lower_percentile = _DOMINANT_HORIZONTAL_XY_PERCENTILE
    upper_percentile = 100.0 - lower_percentile
    lower = np.asarray(
        [
            np.percentile(inliers[:, 0], lower_percentile),
            np.percentile(inliers[:, 1], lower_percentile),
            np.min(inliers[:, 2]),
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            np.percentile(inliers[:, 0], upper_percentile),
            np.percentile(inliers[:, 1], upper_percentile),
            np.max(inliers[:, 2]),
        ],
        dtype=np.float64,
    )
    extent = upper - lower
    if (
        not np.all(np.isfinite(extent))
        or extent[0] <= 0.0
        or extent[1] <= 0.0
    ):
        return {
            "valid": False,
            "source": "dominant_horizontal_z_band",
            "reason": "degenerate_dominant_plane_footprint",
            "point_count": point_count,
            "inlier_count": best_count,
            "inlier_ratio": _round_float(inlier_ratio),
            "band_width_m": _DOMINANT_HORIZONTAL_BAND_WIDTH_M,
        }
    return {
        "valid": True,
        "source": "dominant_horizontal_z_band",
        "plane_z_world_m": _round_float(
            float(np.median(inliers[:, 2]))
        ),
        "point_count": point_count,
        "inlier_count": best_count,
        "inlier_ratio": _round_float(inlier_ratio),
        "band_width_m": _DOMINANT_HORIZONTAL_BAND_WIDTH_M,
        "bbox_world_min": _round_list(lower),
        "bbox_world_max": _round_list(upper),
        "extent_m": _round_list(extent),
    }


def _outward_top_surface_normal(surface_points: np.ndarray) -> np.ndarray:
    if surface_points.shape[0] < 3:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    centered = surface_points - np.mean(surface_points, axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm <= 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal
    if normal[2] < 0.5:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return normal


def _surface_principal_axes(
    world_points: np.ndarray,
    *,
    surface_normal_world: np.ndarray,
) -> list[np.ndarray]:
    normal = np.asarray(surface_normal_world, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    normal = (
        normal / normal_norm
        if math.isfinite(normal_norm) and normal_norm > 1e-8
        else np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    )
    centered = np.asarray(world_points, dtype=np.float64) - np.mean(world_points, axis=0)
    tangent_points = centered - np.outer(centered @ normal, normal)
    try:
        _, _, vh = np.linalg.svd(tangent_points, full_matrices=False)
        first = np.asarray(vh[0], dtype=np.float64)
    except np.linalg.LinAlgError:
        first = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    first = first - float(np.dot(first, normal)) * normal
    first_norm = float(np.linalg.norm(first))
    if not math.isfinite(first_norm) or first_norm <= 1e-8:
        seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(seed, normal))) > 0.9:
            seed = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        first = seed - float(np.dot(seed, normal)) * normal
        first_norm = float(np.linalg.norm(first))
    first = first / max(first_norm, 1e-8)
    dominant = int(np.argmax(np.abs(first)))
    if first[dominant] < 0.0:
        first = -first
    second = np.cross(normal, first)
    second_norm = float(np.linalg.norm(second))
    if not math.isfinite(second_norm) or second_norm <= 1e-8:
        second = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        second = second / second_norm
    dominant = int(np.argmax(np.abs(second)))
    if second[dominant] < 0.0:
        second = -second
    return [first, second]


def _robust_projected_interval(
    world_points: np.ndarray,
    *,
    axis_world: np.ndarray,
) -> tuple[float, float] | None:
    points = np.asarray(world_points, dtype=np.float64)
    axis = np.asarray(axis_world, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] == 0
        or not math.isfinite(axis_norm)
        or axis_norm <= 1e-8
    ):
        return None
    axis = axis / axis_norm
    projections = points @ axis
    projections = projections[np.isfinite(projections)]
    if projections.size == 0:
        return None
    lower = float(np.percentile(projections, _ROBUST_GEOMETRY_LOWER_PERCENTILE))
    upper = float(
        np.percentile(projections, 100.0 - _ROBUST_GEOMETRY_LOWER_PERCENTILE)
    )
    if not math.isfinite(lower) or not math.isfinite(upper):
        return None
    return min(lower, upper), max(lower, upper)


def _observed_grasp_geometry(
    world_points: np.ndarray,
    *,
    surface_world: np.ndarray,
    surface_normal_world: np.ndarray,
) -> dict[str, Any] | None:
    """Derive a parallel-jaw grasp target from the observed object volume.

    Contact remains on the upper surface.  A grasp uses the midpoint of the
    robust observed interval as its object-contact reference and the upper
    interval boundary as its executable TCP target.  This mirrors the Oracle
    contact/clearance contract without reading object metadata or assuming an
    object class, task, fixed height, or world coordinate.
    """

    surface = np.asarray(surface_world, dtype=np.float64).reshape(3)
    normal = np.asarray(surface_normal_world, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= 1e-8:
        return None
    normal = normal / normal_norm
    interval = _robust_projected_interval(world_points, axis_world=normal)
    if interval is None:
        return None
    observed_lower_scalar, _ = interval
    surface_scalar = float(np.dot(surface, normal))
    lower_scalar = min(observed_lower_scalar, surface_scalar)
    normal_extent = surface_scalar - lower_scalar
    if not math.isfinite(normal_extent) or normal_extent <= 1e-6:
        return None
    target_scalar = (lower_scalar + surface_scalar) / 2.0
    object_contact_world = surface + normal * (target_scalar - surface_scalar)
    lower_world = surface + normal * (lower_scalar - surface_scalar)
    boundary_clearance = min(
        target_scalar - lower_scalar,
        surface_scalar - target_scalar,
    )
    return {
        "object_contact_world": object_contact_world,
        "tcp_target_world": surface.copy(),
        "lower_boundary_world": lower_world,
        "upper_boundary_world": surface.copy(),
        "normal_extent_m": float(normal_extent),
        "depth_from_surface_m": float(surface_scalar - target_scalar),
        "boundary_clearance_m": float(boundary_clearance),
        "interval_scalar_m": [float(lower_scalar), float(surface_scalar)],
        "source": "rgbd_observed_volume_normal_interval_midpoint",
    }


def _public_grasp_geometry(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": True,
        "source": str(value.get("source", "")),
        "object_contact_world": _round_list(value["object_contact_world"]),
        "tcp_target_world": _round_list(value["tcp_target_world"]),
        "lower_boundary_world": _round_list(value["lower_boundary_world"]),
        "upper_boundary_world": _round_list(value["upper_boundary_world"]),
        "normal_extent_m": _round_float(value["normal_extent_m"]),
        "depth_from_surface_m": _round_float(value["depth_from_surface_m"]),
        "boundary_clearance_m": _round_float(value["boundary_clearance_m"]),
        "interval_scalar_m": [
            _round_float(item) for item in value["interval_scalar_m"]
        ],
    }


def _surface_operation_candidates(
    *,
    world_points: np.ndarray,
    surface_world: np.ndarray,
    surface_normal_world: np.ndarray,
    principal_axes_world: list[np.ndarray],
    grasp_geometry: dict[str, Any] | None,
    approach_height_m: float,
    ee_to_contact_m: float,
    tcp_calibration_by_arm: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    surface = np.asarray(surface_world, dtype=np.float64).reshape(3)
    normal = np.asarray(surface_normal_world, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    normal = (
        normal / normal_norm
        if math.isfinite(normal_norm) and normal_norm > 1e-8
        else np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    )
    clearance = max(0.0, float(approach_height_m))
    fallback_offset = max(0.0, float(ee_to_contact_m))
    calibrations = dict(tcp_calibration_by_arm or {})
    candidates: list[dict[str, Any]] = []
    for source_index, tangent in enumerate(principal_axes_world[:2]):
        ee_x = -normal
        ee_y = np.asarray(tangent, dtype=np.float64).reshape(3)
        ee_y = ee_y - float(np.dot(ee_y, ee_x)) * ee_x
        ee_y_norm = float(np.linalg.norm(ee_y))
        if not math.isfinite(ee_y_norm) or ee_y_norm <= 1e-8:
            continue
        ee_y = ee_y / ee_y_norm
        ee_z = np.cross(ee_x, ee_y)
        ee_z = ee_z / max(float(np.linalg.norm(ee_z)), 1e-8)
        action_rotation = np.column_stack([ee_x, ee_y, ee_z])
        tangent_interval = _robust_projected_interval(
            world_points,
            axis_world=ee_y,
        )
        grasp_width = (
            max(0.0, tangent_interval[1] - tangent_interval[0])
            if tangent_interval is not None
            else float("inf")
        )
        for arm in ("left", "right"):
            calibration = calibrations.get(arm)
            action_to_tcp = action_to_tcp_transform(
                calibration,
                fallback_offset_m=fallback_offset,
            )
            current_action_pose = (
                pose7_to_matrix(calibration.get("action_pose_world"))
                if isinstance(calibration, dict)
                else None
            )
            try:
                tcp_to_action = np.linalg.inv(action_to_tcp)
            except np.linalg.LinAlgError:
                continue
            mode_targets: list[
                tuple[str, np.ndarray, np.ndarray, dict[str, str]]
            ] = [
                (
                    "contact",
                    surface,
                    surface,
                    {
                        "geometry_source": "rgbd_surface_normal_principal_axes",
                        "priority_source": "reach_distance_m",
                    },
                )
            ]
            if grasp_geometry is not None:
                mode_targets.insert(
                    0,
                    (
                        "grasp",
                        np.asarray(
                            grasp_geometry["object_contact_world"],
                            dtype=np.float64,
                        ).reshape(3),
                        np.asarray(
                            grasp_geometry["tcp_target_world"],
                            dtype=np.float64,
                        ).reshape(3),
                        {
                            "geometry_source": "rgbd_observed_volume_principal_axes",
                            "priority_source": "observed_grasp_width_m",
                        },
                    ),
                )
            for (
                action_mode,
                object_contact_world,
                target_tcp_world,
                mode_metadata,
            ) in mode_targets:
                object_contact_transform = np.eye(4, dtype=np.float64)
                object_contact_transform[:3, :3] = action_rotation
                object_contact_transform[:3, 3] = object_contact_world
                object_contact_pose = matrix_to_pose7(object_contact_transform)
                tcp_transform = np.eye(4, dtype=np.float64)
                tcp_transform[:3, :3] = action_rotation @ action_to_tcp[:3, :3]
                tcp_transform[:3, 3] = target_tcp_world
                ee_target_transform = tcp_transform @ tcp_to_action
                approach_tcp_transform = tcp_transform.copy()
                # Both modes stage above the observed upper surface. Only a
                # grasp subsequently enters the volume-derived interval.
                approach_tcp_transform[:3, 3] = surface + normal * clearance
                approach_transform = approach_tcp_transform @ tcp_to_action
                tcp_pose = matrix_to_pose7(tcp_transform)
                ee_target_pose = matrix_to_pose7(ee_target_transform)
                approach_pose = matrix_to_pose7(approach_transform)
                if (
                    object_contact_pose is None
                    or tcp_pose is None
                    or ee_target_pose is None
                    or approach_pose is None
                ):
                    continue
                reach_distance = (
                    float(
                        np.linalg.norm(
                            ee_target_transform[:3, 3]
                            - current_action_pose[:3, 3]
                        )
                    )
                    if current_action_pose is not None
                    else 0.0
                )
                priority = (
                    grasp_width
                    if action_mode == "grasp" and math.isfinite(grasp_width)
                    else reach_distance
                )
                candidates.append(
                    {
                        "candidate_id": (
                            "rgbd_"
                            f"{'volume' if action_mode == 'grasp' else 'surface'}:"
                            f"{action_mode}:{arm}:{source_index:03d}"
                        ),
                        "source_candidate_index": source_index,
                        "action_mode": action_mode,
                        "arm": arm,
                        "object_contact_pose": object_contact_pose,
                        "tcp_pose": tcp_pose,
                        "ee_target_pose": ee_target_pose,
                        "approach_pose": approach_pose,
                        "approach_direction": _round_list(-normal),
                        **mode_metadata,
                        "calibration_source": (
                            str(calibration.get("source", "robot_kinematics"))
                            if isinstance(calibration, dict)
                            else "scalar_offset_fallback"
                        ),
                        "observed_grasp_width_m": (
                            _round_float(grasp_width)
                            if math.isfinite(grasp_width)
                            else None
                        ),
                        "grasp_depth_m": (
                            _round_float(grasp_geometry["depth_from_surface_m"])
                            if action_mode == "grasp" and grasp_geometry is not None
                            else 0.0
                        ),
                        "grasp_clearance_m": (
                            _round_float(grasp_geometry["boundary_clearance_m"])
                            if action_mode == "grasp" and grasp_geometry is not None
                            else 0.0
                        ),
                        "observed_normal_extent_m": (
                            _round_float(grasp_geometry["normal_extent_m"])
                            if action_mode == "grasp" and grasp_geometry is not None
                            else 0.0
                        ),
                        "grasp_interval_scalar_m": (
                            [
                                _round_float(item)
                                for item in grasp_geometry["interval_scalar_m"]
                            ]
                            if action_mode == "grasp" and grasp_geometry is not None
                            else []
                        ),
                        "reach_distance_m": round(reach_distance, 6),
                        "priority": round(float(priority), 6),
                    }
                )
    return candidates


def _surface_contact_geometry(
    *,
    surface_world: np.ndarray,
    surface_normal_world: np.ndarray,
    approach_height_m: float,
    ee_to_contact_m: float,
    grasp_object_contact_world: np.ndarray | None = None,
    grasp_tcp_world: np.ndarray | None = None,
) -> dict[str, Any]:
    normal = np.asarray(surface_normal_world, dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= 1e-8:
        normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / normal_norm

    surface = np.asarray(surface_world, dtype=np.float64).reshape(3)
    tcp_offset = max(0.0, float(ee_to_contact_m))
    clearance = max(0.0, float(approach_height_m))
    contact_world = surface + normal * tcp_offset
    grasp_world = (
        np.asarray(grasp_tcp_world, dtype=np.float64).reshape(3)
        + normal * tcp_offset
        if grasp_tcp_world is not None
        else None
    )
    approach_world = contact_world + normal * clearance
    surface_approach_world = surface + normal * clearance

    # RMBench's action-frame +X points from the commanded EE origin toward TCP.
    ee_x = -normal
    tangent_seed = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ee_x, tangent_seed))) > 0.95:
        tangent_seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    ee_y = tangent_seed - float(np.dot(tangent_seed, ee_x)) * ee_x
    ee_y = ee_y / max(float(np.linalg.norm(ee_y)), 1e-8)
    ee_z = np.cross(ee_x, ee_y)
    ee_z = ee_z / max(float(np.linalg.norm(ee_z)), 1e-8)
    rotation = np.column_stack([ee_x, ee_y, ee_z])
    quat = _matrix_to_quat_wxyz(rotation)

    object_contact_pose = np.concatenate([surface, quat])
    contact_pose = np.concatenate([contact_world, quat])
    approach_pose = np.concatenate([approach_world, quat])
    result = {
        "surface_approach_point_world": _round_list(surface_approach_world),
        "object_contact_point_world": _round_list(surface),
        "contact_point_world": _round_list(contact_world),
        "approach_point_world": _round_list(approach_world),
        "object_contact_pose_world": _round_list(object_contact_pose),
        "contact_pose_world": _round_list(contact_pose),
        "approach_pose_world": _round_list(approach_pose),
        "ee_to_contact_m": _round_float(tcp_offset),
        "contact_geometry_source": "rgbd_surface_normal_and_robot_tcp",
    }
    if grasp_world is not None:
        grasp_pose = np.concatenate([grasp_world, quat])
        grasp_object_pose = (
            np.concatenate(
                [
                    np.asarray(
                        grasp_object_contact_world,
                        dtype=np.float64,
                    ).reshape(3),
                    quat,
                ]
            )
            if grasp_object_contact_world is not None
            else None
        )
        result.update(
            {
                "grasp_tcp_point_world": _round_list(grasp_tcp_world),
                "grasp_point_world": _round_list(grasp_world),
                "grasp_pose_world": _round_list(grasp_pose),
            }
        )
        if grasp_object_pose is not None:
            result["grasp_object_pose_world"] = _round_list(
                grasp_object_pose
            )
    return result


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(max(0.0, trace + 1.0)) * 2.0
        quat = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(
                max(
                    0.0,
                    1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
                )
            ) * 2.0
            quat = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(
                max(
                    0.0,
                    1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2],
                )
            ) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(
                max(
                    0.0,
                    1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1],
                )
            ) * 2.0
            quat = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat = quat / norm
    return -quat if quat[0] < 0.0 else quat


def _failure(
    *,
    object_id: str,
    camera: str,
    error: str,
    mask_pixel_count: int | None = None,
    valid_pixel_count: int | None = None,
    valid_ratio: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": False,
        "object_id": object_id,
        "camera": str(camera),
        "error": error,
    }
    if mask_pixel_count is not None:
        result["mask_pixel_count"] = int(mask_pixel_count)
    if valid_pixel_count is not None:
        result["valid_pixel_count"] = int(valid_pixel_count)
    if valid_ratio is not None:
        result["valid_ratio"] = _round_float(valid_ratio)
    return result


def _round_list(values: Any) -> list[float]:
    return [_round_float(float(item)) for item in np.asarray(values).reshape(-1)]


def _round_float(value: float) -> float:
    return round(float(value), 6)
