from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

import numpy as np
from roboharn_evo.agent.gripper_state import normalized_gripper_target

from roboharn_evo.agent.operation_candidates import (
    action_to_tcp_transform,
    matrix_to_pose7,
    pose7_to_matrix,
)


_DEFAULT_EE_TO_CONTACT_M = 0.12
_DEFAULT_APPROACH_CLEARANCE_M = 0.08
_CONTACT_TO_EE_FRAME = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass
class EnvSnapshot:
    """Shared sensor view: depth arrays in millimeters; world poses in meters."""

    raw: dict[str, Any]
    head_rgb: np.ndarray
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    joint_vector: np.ndarray
    left_endpose: np.ndarray
    right_endpose: np.ndarray
    step_count: int
    step_limit: int
    eval_success: bool
    check_success: bool
    max_reward: float
    instruction: str
    third_rgb: np.ndarray | None = None
    head_depth: np.ndarray | None = None
    left_depth: np.ndarray | None = None
    right_depth: np.ndarray | None = None
    third_depth: np.ndarray | None = None
    head_intrinsic_cv: np.ndarray | None = None
    left_intrinsic_cv: np.ndarray | None = None
    right_intrinsic_cv: np.ndarray | None = None
    third_intrinsic_cv: np.ndarray | None = None
    head_cam2world_gl: np.ndarray | None = None
    left_cam2world_gl: np.ndarray | None = None
    right_cam2world_gl: np.ndarray | None = None
    third_cam2world_gl: np.ndarray | None = None
    oracle_objects: list[dict[str, Any]] | None = None
    ee_to_tcp_m: float = _DEFAULT_EE_TO_CONTACT_M
    tcp_calibration_by_arm: dict[str, dict[str, Any]] = field(default_factory=dict)
    gripper_command_by_arm: dict[str, float] = field(default_factory=dict)


def capture_env_snapshot(
    task_env: Any,
    observation: dict[str, Any],
    *,
    oracle_objects_enabled: bool = False,
) -> EnvSnapshot:
    """Capture through the benchmark's sensor projection, sharing one runtime.

    Non-RMBench bridges supply ``capture_snapshot``. It translates sensors and
    robot calibration only, never planning, scene memory, or action lifecycle.
    Existing RMBench callers retain the exact previous conversion by default.
    """
    capture = getattr(task_env, "capture_snapshot", None)
    if callable(capture):
        snapshot = capture(observation, oracle_objects_enabled=oracle_objects_enabled)
        if not isinstance(snapshot, EnvSnapshot):
            raise TypeError("capture_snapshot must return EnvSnapshot")
        return snapshot
    return RMBenchEnvAdapter.from_env(
        task_env, observation, oracle_objects_enabled=oracle_objects_enabled,
    )


class RMBenchEnvAdapter:
    @staticmethod
    def from_env(
        task_env: Any,
        observation: dict[str, Any],
        *,
        oracle_objects_enabled: bool = False,
        evaluate_success_predicate: bool = True,
    ) -> EnvSnapshot:
        check_success = False
        if evaluate_success_predicate:
            try:
                check_success = bool(task_env.check_success())
            except Exception:
                check_success = False

        # Simulator object introspection is an explicit Oracle-only capability.
        # In particular, the no-Oracle path must not even walk task_env fields:
        # keeping the resulting catalog out of downstream payloads is too late.
        tcp_calibration_by_arm = RMBenchEnvAdapter.robot_tcp_calibration_by_arm(task_env)
        oracle_objects = (
            RMBenchEnvAdapter.oracle_objects(
                task_env,
                tcp_calibration_by_arm=tcp_calibration_by_arm,
            )
            if bool(oracle_objects_enabled)
            else None
        )
        ee_to_tcp_m = RMBenchEnvAdapter.robot_ee_to_tcp_m(
            task_env,
            tcp_calibration_by_arm=tcp_calibration_by_arm,
        )

        raw_observation = dict(observation)
        # Do not preserve the historical side channel even if a caller supplies
        # it. Oracle data lives only in the explicitly enabled typed field.
        raw_observation.pop("_tcm_oracle_objects", None)

        def camera_array(camera: str, key: str, dtype: Any | None = None) -> np.ndarray | None:
            camera_payload = observation.get("observation", {}).get(f"{camera}_camera", {})
            if not isinstance(camera_payload, dict) or key not in camera_payload:
                return None
            try:
                array = np.asarray(camera_payload[key], dtype=dtype)
            except Exception:
                return None
            return array

        third_rgb = camera_array(
            "third",
            "rgb",
            np.uint8,
        )
        if third_rgb is None and observation.get(
            "third_view_rgb"
        ) is not None:
            try:
                third_rgb = np.asarray(
                    observation["third_view_rgb"],
                    dtype=np.uint8,
                )
            except Exception:
                third_rgb = None

        return EnvSnapshot(
            raw=raw_observation,
            head_rgb=np.asarray(observation["observation"]["head_camera"]["rgb"], dtype=np.uint8),
            left_rgb=np.asarray(observation["observation"]["left_camera"]["rgb"], dtype=np.uint8),
            right_rgb=np.asarray(observation["observation"]["right_camera"]["rgb"], dtype=np.uint8),
            third_rgb=third_rgb,
            head_depth=camera_array("head", "depth", np.float64),
            left_depth=camera_array("left", "depth", np.float64),
            right_depth=camera_array("right", "depth", np.float64),
            third_depth=camera_array("third", "depth", np.float64),
            head_intrinsic_cv=camera_array("head", "intrinsic_cv", np.float64),
            left_intrinsic_cv=camera_array("left", "intrinsic_cv", np.float64),
            right_intrinsic_cv=camera_array("right", "intrinsic_cv", np.float64),
            third_intrinsic_cv=camera_array("third", "intrinsic_cv", np.float64),
            head_cam2world_gl=camera_array("head", "cam2world_gl", np.float64),
            left_cam2world_gl=camera_array("left", "cam2world_gl", np.float64),
            right_cam2world_gl=camera_array("right", "cam2world_gl", np.float64),
            third_cam2world_gl=camera_array("third", "cam2world_gl", np.float64),
            oracle_objects=oracle_objects,
            ee_to_tcp_m=ee_to_tcp_m,
            tcp_calibration_by_arm=tcp_calibration_by_arm,
            gripper_command_by_arm={
                arm: normalized_gripper_target(value)
                for arm, value in zip(
                    ("left", "right"), task_env.robot.get_normal_real_gripper_val(),
                    strict=False,
                )
            },
            joint_vector=np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
            left_endpose=np.asarray(observation["endpose"]["left_endpose"], dtype=np.float32),
            right_endpose=np.asarray(observation["endpose"]["right_endpose"], dtype=np.float32),
            step_count=int(getattr(task_env, "take_action_cnt", 0)),
            step_limit=int(getattr(task_env, "step_lim", 0) or 0),
            eval_success=bool(getattr(task_env, "eval_success", False)),
            check_success=check_success,
            max_reward=float(getattr(task_env, "max_reward", 0.0)),
            instruction=str(getattr(task_env, "instruction", "") or ""),
        )

    @staticmethod
    def robot_tcp_calibration_by_arm(task_env: Any) -> dict[str, dict[str, Any]]:
        """Measure each arm's full action-frame-to-TCP transform from robot kinematics."""
        robot = getattr(task_env, "robot", None)
        calibrations: dict[str, dict[str, Any]] = {}
        for arm in ("left", "right"):
            ee_getter = getattr(robot, f"get_{arm}_ee_pose", None)
            tcp_getter = getattr(robot, f"get_{arm}_tcp_pose", None)
            if not callable(ee_getter) or not callable(tcp_getter):
                continue
            try:
                ee_pose = np.asarray(ee_getter(), dtype=np.float64).reshape(-1)
                tcp_pose = np.asarray(tcp_getter(), dtype=np.float64).reshape(-1)
            except Exception:
                continue
            world_action = pose7_to_matrix(ee_pose)
            world_tcp = pose7_to_matrix(tcp_pose)
            if world_action is None or world_tcp is None:
                continue
            try:
                action_to_tcp = np.linalg.inv(world_action) @ world_tcp
            except np.linalg.LinAlgError:
                continue
            translation = action_to_tcp[:3, 3]
            offset = float(np.linalg.norm(translation))
            action_to_tcp_pose = matrix_to_pose7(action_to_tcp)
            if action_to_tcp_pose is None or not math.isfinite(offset) or not 0.0 <= offset <= 0.30:
                continue
            calibrations[arm] = {
                "arm": arm,
                "action_to_tcp_matrix": [
                    [round(float(item), 9) for item in row]
                    for row in action_to_tcp
                ],
                "translation_m": action_to_tcp_pose[:3],
                "quat_wxyz": action_to_tcp_pose[3:7],
                "offset_norm_m": round(offset, 6),
                "action_pose_world": matrix_to_pose7(world_action),
                "tcp_pose_world": matrix_to_pose7(world_tcp),
                "source": "robot_kinematics",
            }
        return calibrations

    @staticmethod
    def robot_ee_to_tcp_m(
        task_env: Any,
        *,
        tcp_calibration_by_arm: dict[str, dict[str, Any]] | None = None,
    ) -> float:
        """Return the compatibility scalar derived from per-arm full calibration."""
        calibrations = (
            tcp_calibration_by_arm
            if isinstance(tcp_calibration_by_arm, dict)
            else RMBenchEnvAdapter.robot_tcp_calibration_by_arm(task_env)
        )
        offsets: list[float] = []
        for calibration in calibrations.values():
            try:
                offset = float(calibration.get("offset_norm_m"))
            except (AttributeError, TypeError, ValueError):
                continue
            if math.isfinite(offset) and 0.0 <= offset <= 0.30:
                offsets.append(offset)
        if not offsets:
            return _DEFAULT_EE_TO_CONTACT_M
        return float(np.median(np.asarray(offsets, dtype=np.float64)))

    @staticmethod
    def oracle_objects(
        task_env: Any,
        *,
        tcp_calibration_by_arm: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        records: dict[tuple[str, int], dict[str, Any]] = {}
        if task_env is None:
            return []
        ee_to_contact_m = RMBenchEnvAdapter._oracle_ee_to_contact_m(task_env)
        calibrations = (
            tcp_calibration_by_arm
            if isinstance(tcp_calibration_by_arm, dict)
            else RMBenchEnvAdapter.robot_tcp_calibration_by_arm(task_env)
        )
        try:
            top_level_items = vars(task_env).items()
        except TypeError:
            return []
        current_actor_keys = RMBenchEnvAdapter._oracle_scene_actor_keys(
            getattr(task_env, "scene", None)
        )

        for attr_name, value in top_level_items:
            if RMBenchEnvAdapter._skip_oracle_attr(attr_name):
                continue
            RMBenchEnvAdapter._collect_oracle_actor(
                value,
                source_path=str(attr_name),
                records=records,
                depth=0,
                ee_to_contact_m=ee_to_contact_m,
                tcp_calibration_by_arm=calibrations,
                current_actor_keys=current_actor_keys,
            )

        objects = list(records.values())
        objects.sort(key=lambda item: (str(item.get("class_name", "")), float(item.get("position_world", [0.0, 0.0, 0.0])[0]), str(item.get("source_path", ""))))
        for index, item in enumerate(objects):
            item["rank"] = index
        return objects

    @staticmethod
    def _skip_oracle_attr(name: str) -> bool:
        if not name or name.startswith("_"):
            return True
        structural_names = {
            "engine",
            "renderer",
            "scene",
            "viewer",
            "robot",
            "cameras",
            "table",
            "wall",
            "ground",
            "direction_light_lst",
            "point_light_lst",
            "static_camera_list",
            "static_camera_name",
            "now_obs",
            "info",
            "file_path",
            "language_annotation",
            "language_annotation_cache",
            "raw_head_pcl",
            "real_head_pcl",
            "real_head_pcl_color",
            "world_pcd",
        }
        return name in structural_names

    @staticmethod
    def _collect_oracle_actor(
        value: Any,
        *,
        source_path: str,
        records: dict[tuple[str, int], dict[str, Any]],
        depth: int,
        ee_to_contact_m: float,
        tcp_calibration_by_arm: dict[str, dict[str, Any]],
        current_actor_keys: set[tuple[str, int]] | None,
    ) -> None:
        if value is None or depth > 4:
            return
        if RMBenchEnvAdapter._is_oracle_actor(value):
            key = RMBenchEnvAdapter._oracle_actor_key(value)
            if current_actor_keys is not None and key not in current_actor_keys:
                return
            existing = records.get(key)
            if existing is None:
                record = RMBenchEnvAdapter._oracle_record_for_actor(
                    value,
                    source_path=source_path,
                    ee_to_contact_m=ee_to_contact_m,
                    tcp_calibration_by_arm=tcp_calibration_by_arm,
                )
                if record:
                    records[key] = record
                return
            paths = existing.setdefault("source_paths", [])
            if source_path not in paths:
                paths.append(source_path)
            aliases = set(existing.get("aliases", []))
            aliases.update(RMBenchEnvAdapter._oracle_aliases(source_path, existing.get("actor_name", "")))
            existing["aliases"] = sorted(aliases)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = RMBenchEnvAdapter._safe_path_token(key)
                if not key_text:
                    continue
                RMBenchEnvAdapter._collect_oracle_actor(
                    item,
                    source_path=f"{source_path}.{key_text}",
                    records=records,
                    depth=depth + 1,
                    ee_to_contact_m=ee_to_contact_m,
                    tcp_calibration_by_arm=tcp_calibration_by_arm,
                    current_actor_keys=current_actor_keys,
                )
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                RMBenchEnvAdapter._collect_oracle_actor(
                    item,
                    source_path=f"{source_path}[{index}]",
                    records=records,
                    depth=depth + 1,
                    ee_to_contact_m=ee_to_contact_m,
                    tcp_calibration_by_arm=tcp_calibration_by_arm,
                    current_actor_keys=current_actor_keys,
                )

    @staticmethod
    def _is_oracle_actor(value: Any) -> bool:
        get_pose = getattr(value, "get_pose", None)
        return callable(get_pose)

    @staticmethod
    def _oracle_actor_identity_holders(value: Any) -> list[Any]:
        holders: list[Any] = []

        def append_holder(holder: Any) -> None:
            if holder is None:
                return
            if any(holder is existing for existing in holders):
                return
            holders.append(holder)

        append_holder(value)
        try:
            append_holder(getattr(value, "actor", None))
        except Exception:
            pass

        for holder in list(holders):
            get_links = getattr(holder, "get_links", None)
            if not callable(get_links):
                continue
            try:
                links = list(get_links())
            except Exception:
                continue
            for link in links:
                entity = None
                get_entity = getattr(link, "get_entity", None)
                if callable(get_entity):
                    try:
                        entity = get_entity()
                    except Exception:
                        entity = None
                if entity is None:
                    try:
                        entity = getattr(link, "entity", None)
                    except Exception:
                        entity = None
                append_holder(entity)
        return holders

    @staticmethod
    def _oracle_scene_actor_keys(
        scene: Any,
    ) -> set[tuple[str, int]] | None:
        """Return identities owned by the currently active simulator scene.

        RMBench reuses one Python task object for the expert check and policy
        rollout. Task-specific actor aliases created by the expert can survive
        the second setup while still pointing into the closed expert scene.
        Scene ownership is therefore the liveness authority; field names and
        geometric proximity are not.
        """

        if scene is None:
            return None
        actor_keys: set[tuple[str, int]] = set()
        successful_enumeration = False
        for getter_name in ("get_all_actors", "get_all_articulations"):
            getter = getattr(scene, getter_name, None)
            if not callable(getter):
                continue
            try:
                actors = list(getter())
            except Exception:
                continue
            successful_enumeration = True
            for actor in actors:
                actor_keys.add(RMBenchEnvAdapter._oracle_actor_key(actor))
        return actor_keys if successful_enumeration else None

    @staticmethod
    def _oracle_actor_key(value: Any) -> tuple[str, int]:
        """Return one stable key for every wrapper of the same simulator actor.

        SAPIEN may materialize a fresh Python proxy when ``wrapper.actor`` is
        accessed.  ``id(wrapper.actor)`` can therefore differ for two task
        attributes that reference the same physical entity (for example both
        ``block1`` and the dynamic alias ``left_block``).  Inspect both the
        stable outer wrapper and the nested actor proxy for simulator identity.
        If neither exposes one, fall back to the outer wrapper identity rather
        than the ephemeral proxy identity.  This preserves exact task aliases
        without geometrically merging nearby but distinct actors.
        """

        holders = RMBenchEnvAdapter._oracle_actor_identity_holders(value)
        for identity_kind, names in (
            ("simulator_global_id", ("get_global_id", "global_id")),
            (
                "simulator_per_scene_id",
                ("get_per_scene_id", "per_scene_id"),
            ),
        ):
            for holder in holders:
                for name in names:
                    member = getattr(holder, name, None)
                    if member is None:
                        continue
                    try:
                        raw_value = member() if callable(member) else member
                        stable_id = int(raw_value)
                    except (TypeError, ValueError, RuntimeError):
                        continue
                    if stable_id >= 0:
                        return identity_kind, stable_id
        return "python_object", id(value)

    @staticmethod
    def _oracle_record_for_actor(
        value: Any,
        *,
        source_path: str,
        ee_to_contact_m: float,
        tcp_calibration_by_arm: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        pose = RMBenchEnvAdapter._actor_pose(value)
        if pose is None:
            return {}
        position, quat = pose
        actor_name = RMBenchEnvAdapter._actor_name(value)
        class_name = RMBenchEnvAdapter._class_name_from_source_path(source_path, actor_name)
        aliases = RMBenchEnvAdapter._oracle_aliases(source_path, actor_name, class_name)
        config = getattr(value, "config", None)
        if not isinstance(config, dict):
            config = {}
        grounding = RMBenchEnvAdapter._oracle_grounding(
            object_id=class_name,
            position=position,
            quat=quat,
            config=config,
            contact_matrices=RMBenchEnvAdapter._oracle_contact_matrices(value),
            ee_to_contact_m=ee_to_contact_m,
            tcp_calibration_by_arm=tcp_calibration_by_arm,
        )
        record = {
            "oracle_id": RMBenchEnvAdapter._safe_path_token(source_path) or class_name,
            "class_name": class_name,
            "source_path": source_path,
            "source_paths": [source_path],
            "actor_name": actor_name,
            "aliases": aliases,
            "position_world": RMBenchEnvAdapter._rounded_list(position),
            "quat_wxyz": RMBenchEnvAdapter._rounded_list(quat),
            "grounding_3d": grounding,
        }
        extents = grounding.get("world_extent_m") if isinstance(grounding, dict) else None
        if extents is not None:
            record["world_extent_m"] = extents
        return record

    @staticmethod
    def _actor_pose(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            pose = value.get_pose()
        except Exception:
            return None
        position = getattr(pose, "p", None)
        quat = getattr(pose, "q", None)
        try:
            position_array = np.asarray(position, dtype=np.float64).reshape(3)
            quat_array = np.asarray(quat, dtype=np.float64).reshape(4)
        except Exception:
            return None
        if not np.all(np.isfinite(position_array)) or not np.all(np.isfinite(quat_array)):
            return None
        norm = float(np.linalg.norm(quat_array))
        if norm <= 1e-8:
            return None
        return position_array, quat_array / norm

    @staticmethod
    def _actor_name(value: Any) -> str:
        for holder in (value, getattr(value, "actor", None)):
            if holder is None:
                continue
            get_name = getattr(holder, "get_name", None)
            if callable(get_name):
                try:
                    name = str(get_name() or "").strip()
                except Exception:
                    name = ""
                if name:
                    return name
        return ""

    @staticmethod
    def _oracle_grounding(
        *,
        object_id: str,
        position: np.ndarray,
        quat: np.ndarray,
        config: dict[str, Any],
        contact_matrices: list[np.ndarray] | None = None,
        ee_to_contact_m: float = _DEFAULT_EE_TO_CONTACT_M,
        tcp_calibration_by_arm: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        rotation = RMBenchEnvAdapter._quat_wxyz_to_matrix(quat)
        center = RMBenchEnvAdapter._numeric_vector(config.get("center"), length=3)
        if center is None:
            center = np.zeros(3, dtype=np.float64)
        center = center * RMBenchEnvAdapter._oracle_geometry_scale(config)
        centroid = position + rotation @ center
        half_size = RMBenchEnvAdapter._oracle_bbox_half_size(config)
        bbox_min = None
        bbox_max = None
        top_surface = centroid.copy()
        if half_size is not None:
            corners = []
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        local = center + np.asarray([sx * half_size[0], sy * half_size[1], sz * half_size[2]], dtype=np.float64)
                        corners.append(position + rotation @ local)
            corner_array = np.asarray(corners, dtype=np.float64)
            bbox_min = np.min(corner_array, axis=0)
            bbox_max = np.max(corner_array, axis=0)
            top_surface = np.asarray([centroid[0], centroid[1], bbox_max[2]], dtype=np.float64)
        approach = top_surface.copy()
        approach[2] += 0.08
        result = {
            "success": True,
            "object_id": object_id,
            "camera": "oracle",
            "depth_units": "simulator_pose",
            "mask_pixel_count": 0,
            "valid_pixel_count": 0,
            "valid_ratio": 1.0,
            "sampled_point_count": 0,
            "surface_sample_count": 0,
            "centroid_world": RMBenchEnvAdapter._rounded_list(centroid),
            "top_surface_world": RMBenchEnvAdapter._rounded_list(top_surface),
            "approach_point_world": RMBenchEnvAdapter._rounded_list(approach),
        }
        if bbox_min is not None and bbox_max is not None:
            result["bbox_world_min"] = RMBenchEnvAdapter._rounded_list(bbox_min)
            result["bbox_world_max"] = RMBenchEnvAdapter._rounded_list(bbox_max)
            result["world_extent_m"] = RMBenchEnvAdapter._rounded_list(np.abs(bbox_max - bbox_min))
        contact_grounding = RMBenchEnvAdapter._oracle_contact_grounding(
            contact_matrices or [],
            ee_to_contact_m=ee_to_contact_m,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            object_bbox_center=centroid if half_size is not None else None,
            object_bbox_rotation=rotation if half_size is not None else None,
            object_bbox_half_size=half_size,
            tcp_calibration_by_arm=tcp_calibration_by_arm,
        )
        if contact_grounding:
            result["surface_approach_point_world"] = result["approach_point_world"]
            result.update(contact_grounding)
        return result

    @staticmethod
    def _oracle_bbox_half_size(config: dict[str, Any]) -> np.ndarray | None:
        extents = RMBenchEnvAdapter._numeric_vector(config.get("extents"), length=3)
        if extents is None:
            return None
        scale = RMBenchEnvAdapter._numeric_vector(config.get("scale"), length=3)
        if scale is not None and np.allclose(extents, scale, rtol=1e-5, atol=1e-8):
            return np.abs(extents)
        return np.abs(extents * RMBenchEnvAdapter._oracle_geometry_scale(config)) / 2.0

    @staticmethod
    def _oracle_geometry_scale(config: dict[str, Any]) -> np.ndarray:
        scale = RMBenchEnvAdapter._numeric_vector(config.get("scale"), length=3)
        if scale is None:
            return np.ones(3, dtype=np.float64)
        extents = RMBenchEnvAdapter._numeric_vector(config.get("extents"), length=3)
        # Procedural primitives may store their half-size in both fields; their
        # actor geometry is already scaled when constructed.
        if extents is not None and np.allclose(extents, scale, rtol=1e-5, atol=1e-8):
            return np.ones(3, dtype=np.float64)
        return np.abs(scale)

    @staticmethod
    def _oracle_ee_to_contact_m(task_env: Any) -> float:
        robot = getattr(task_env, "robot", None)
        offsets: list[float] = []
        for name in ("left_gripper_bias", "right_gripper_bias"):
            try:
                value = float(getattr(robot, name))
            except Exception:
                continue
            if math.isfinite(value) and 0.0 < value < 1.0:
                offsets.append(value)
        if not offsets:
            return _DEFAULT_EE_TO_CONTACT_M
        return float(np.median(np.asarray(offsets, dtype=np.float64)))

    @staticmethod
    def _oracle_contact_matrices(value: Any) -> list[np.ndarray]:
        iterator = getattr(value, "iter_contact_points", None)
        if not callable(iterator):
            return []
        try:
            raw_points = list(iterator("matrix"))
        except Exception:
            return []
        matrices: list[np.ndarray] = []
        for item in raw_points:
            raw_matrix = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            try:
                matrix = np.asarray(raw_matrix, dtype=np.float64).reshape(4, 4)
            except Exception:
                continue
            if np.all(np.isfinite(matrix)):
                matrices.append(matrix)
        return matrices

    @staticmethod
    def _oracle_contact_grounding(
        contact_matrices: list[np.ndarray],
        *,
        ee_to_contact_m: float,
        bbox_min: np.ndarray | None = None,
        bbox_max: np.ndarray | None = None,
        object_bbox_center: np.ndarray | None = None,
        object_bbox_rotation: np.ndarray | None = None,
        object_bbox_half_size: np.ndarray | None = None,
        tcp_calibration_by_arm: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        object_contact_poses: list[list[float]] = []
        grasp_poses: list[list[float]] = []
        approach_poses: list[list[float]] = []
        operation_candidates: list[dict[str, Any]] = []
        offset = max(0.0, float(ee_to_contact_m))
        calibrations = dict(tcp_calibration_by_arm or {})
        for source_index, contact_matrix in enumerate(contact_matrices):
            try:
                matrix = np.asarray(contact_matrix, dtype=np.float64).reshape(4, 4)
            except Exception:
                continue
            ee_frame = matrix @ _CONTACT_TO_EE_FRAME
            quat = RMBenchEnvAdapter._matrix_to_quat_wxyz(ee_frame[:3, :3])
            if quat is None:
                continue
            contact_xyz = matrix[:3, 3]
            grasp_xyz = contact_xyz + ee_frame[:3, :3] @ np.asarray([-offset, 0.0, 0.0], dtype=np.float64)
            approach_xyz = contact_xyz + ee_frame[:3, :3] @ np.asarray(
                [-offset - _DEFAULT_APPROACH_CLEARANCE_M, 0.0, 0.0],
                dtype=np.float64,
            )
            object_contact_poses.append(RMBenchEnvAdapter._rounded_list(np.concatenate([contact_xyz, quat])))
            grasp_poses.append(RMBenchEnvAdapter._rounded_list(np.concatenate([grasp_xyz, quat])))
            approach_poses.append(RMBenchEnvAdapter._rounded_list(np.concatenate([approach_xyz, quat])))
            outward = -ee_frame[:3, 0]
            outward_norm = float(np.linalg.norm(outward))
            if not math.isfinite(outward_norm) or outward_norm <= 1e-8:
                continue
            outward = outward / outward_norm
            grasp_clearance = RMBenchEnvAdapter._ray_box_exit_distance(
                contact_xyz,
                outward,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                object_bbox_center=object_bbox_center,
                object_bbox_rotation=object_bbox_rotation,
                object_bbox_half_size=object_bbox_half_size,
            )
            object_contact_pose = matrix_to_pose7(matrix)
            if object_contact_pose is None:
                continue
            for arm in ("left", "right"):
                calibration = calibrations.get(arm)
                action_to_tcp = action_to_tcp_transform(
                    calibration,
                    fallback_offset_m=offset,
                )
                current_action_pose = (
                    pose7_to_matrix(calibration.get("action_pose_world"))
                    if isinstance(calibration, dict)
                    else None
                )
                for action_mode in ("grasp", "contact"):
                    mode_clearance = grasp_clearance if action_mode == "grasp" else 0.0
                    tcp_transform = np.eye(4, dtype=np.float64)
                    tcp_transform[:3, :3] = ee_frame[:3, :3] @ action_to_tcp[:3, :3]
                    tcp_transform[:3, 3] = contact_xyz + outward * mode_clearance
                    try:
                        ee_target_transform = tcp_transform @ np.linalg.inv(action_to_tcp)
                    except np.linalg.LinAlgError:
                        continue
                    approach_tcp_transform = tcp_transform.copy()
                    approach_tcp_transform[:3, 3] += outward * _DEFAULT_APPROACH_CLEARANCE_M
                    approach_transform = approach_tcp_transform @ np.linalg.inv(action_to_tcp)
                    tcp_pose = matrix_to_pose7(tcp_transform)
                    ee_target_pose = matrix_to_pose7(ee_target_transform)
                    candidate_approach_pose = matrix_to_pose7(approach_transform)
                    if tcp_pose is None or ee_target_pose is None or candidate_approach_pose is None:
                        continue
                    reach_distance = (
                        float(np.linalg.norm(ee_target_transform[:3, 3] - current_action_pose[:3, 3]))
                        if current_action_pose is not None
                        else 0.0
                    )
                    operation_candidates.append(
                        {
                            "candidate_id": (
                                f"oracle_contact:{action_mode}:{arm}:{source_index:03d}"
                            ),
                            "source_candidate_index": source_index,
                            "action_mode": action_mode,
                            "arm": arm,
                            "object_contact_pose": object_contact_pose,
                            "tcp_pose": tcp_pose,
                            "ee_target_pose": ee_target_pose,
                            "approach_pose": candidate_approach_pose,
                            "approach_direction": RMBenchEnvAdapter._rounded_list(-outward),
                            "geometry_source": "oracle_actor_contact_matrix",
                            "calibration_source": (
                                str(calibration.get("source", "robot_kinematics"))
                                if isinstance(calibration, dict)
                                else "scalar_offset_fallback"
                            ),
                            "grasp_clearance_m": round(float(mode_clearance), 6),
                            "reach_distance_m": round(reach_distance, 6),
                            "priority": round(reach_distance, 6),
                        }
                    )
        if not grasp_poses:
            return {}
        return {
            "object_contact_pose_world": object_contact_poses[0],
            "contact_pose_world": grasp_poses[0],
            "grasp_pose_world": grasp_poses[0],
            "approach_pose_world": approach_poses[0],
            "object_contact_pose_world_candidates": object_contact_poses,
            "contact_pose_world_candidates": grasp_poses,
            "grasp_pose_world_candidates": grasp_poses,
            "approach_pose_world_candidates": approach_poses,
            "operation_pose_candidates": operation_candidates,
            "object_contact_point_world": object_contact_poses[0][:3],
            "contact_point_world": grasp_poses[0][:3],
            "grasp_point_world": grasp_poses[0][:3],
            "approach_point_world": approach_poses[0][:3],
        }

    @staticmethod
    def _ray_box_exit_distance(
        origin: np.ndarray,
        direction: np.ndarray,
        *,
        bbox_min: np.ndarray | None,
        bbox_max: np.ndarray | None,
        object_bbox_center: np.ndarray | None = None,
        object_bbox_rotation: np.ndarray | None = None,
        object_bbox_half_size: np.ndarray | None = None,
    ) -> float:
        try:
            point = np.asarray(origin, dtype=np.float64).reshape(3)
            ray = np.asarray(direction, dtype=np.float64).reshape(3)
        except Exception:
            return 0.0
        if (
            object_bbox_center is not None
            and object_bbox_rotation is not None
            and object_bbox_half_size is not None
        ):
            try:
                center = np.asarray(object_bbox_center, dtype=np.float64).reshape(3)
                rotation = np.asarray(object_bbox_rotation, dtype=np.float64).reshape(3, 3)
                half_size = np.abs(
                    np.asarray(object_bbox_half_size, dtype=np.float64).reshape(3)
                )
                point = rotation.T @ (point - center)
                ray = rotation.T @ ray
                lower = -half_size
                upper = half_size
            except Exception:
                return 0.0
        elif bbox_min is not None and bbox_max is not None:
            try:
                lower = np.asarray(bbox_min, dtype=np.float64).reshape(3)
                upper = np.asarray(bbox_max, dtype=np.float64).reshape(3)
            except Exception:
                return 0.0
        else:
            return 0.0
        if not np.all(np.isfinite(np.concatenate([point, ray, lower, upper]))):
            return 0.0
        # Contact-matrix origins that are already outside the reconstructed
        # actor bounds represent surface contacts and need no grasp clearance.
        if np.any(point < lower - 1e-4) or np.any(point > upper + 1e-4):
            return 0.0
        distances: list[float] = []
        for axis in range(3):
            if ray[axis] > 1e-8:
                distance = (upper[axis] - point[axis]) / ray[axis]
            elif ray[axis] < -1e-8:
                distance = (lower[axis] - point[axis]) / ray[axis]
            else:
                continue
            if math.isfinite(float(distance)) and distance >= -1e-6:
                distances.append(max(0.0, float(distance)))
        if not distances:
            return 0.0
        clearance = min(distances)
        diagonal = float(np.linalg.norm(upper - lower))
        if not math.isfinite(clearance) or clearance > max(0.30, diagonal + 1e-6):
            return 0.0
        return clearance

    @staticmethod
    def _matrix_to_quat_wxyz(rotation: Any) -> np.ndarray | None:
        try:
            matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        except Exception:
            return None
        if not np.all(np.isfinite(matrix)):
            return None
        u, _, vh = np.linalg.svd(matrix)
        matrix = u @ vh
        if np.linalg.det(matrix) < 0.0:
            u[:, -1] *= -1.0
            matrix = u @ vh
        trace = float(np.trace(matrix))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            quat = np.asarray(
                [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s],
                dtype=np.float64,
            )
        else:
            index = int(np.argmax(np.diag(matrix)))
            if index == 0:
                s = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
                quat = np.asarray([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s])
            elif index == 1:
                s = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
                quat = np.asarray([(matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s])
            else:
                s = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
                quat = np.asarray([(matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s])
        norm = float(np.linalg.norm(quat))
        if not math.isfinite(norm) or norm <= 1e-8:
            return None
        quat = quat / norm
        if quat[0] < 0.0:
            quat = -quat
        return quat

    @staticmethod
    def _numeric_vector(value: Any, *, length: int) -> np.ndarray | None:
        try:
            array = np.asarray(value, dtype=np.float64).reshape(length)
        except Exception:
            return None
        if not np.all(np.isfinite(array)):
            return None
        return array

    @staticmethod
    def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(item) for item in quat]
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _oracle_aliases(source_path: str, actor_name: Any, class_name: str = "") -> list[str]:
        raw_values = [source_path, str(actor_name or ""), str(class_name or "")]
        aliases: list[str] = []
        for value in raw_values:
            token = RMBenchEnvAdapter._safe_path_token(value)
            if not token:
                continue
            aliases.append(token)
            aliases.extend(part for part in token.split("_") if part and not part.isdigit())
        expanded: list[str] = []
        for alias in aliases:
            expanded.append(alias)
            stem = re.sub(r"\d+$", "", alias).strip("_")
            if stem and stem != alias:
                expanded.append(stem)
            singular = RMBenchEnvAdapter._singularize_token(stem or alias)
            if singular and singular != alias:
                expanded.append(singular)
        return sorted({item for item in expanded if item})

    @staticmethod
    def _class_name_from_source_path(source_path: str, actor_name: str) -> str:
        root = source_path.split(".", 1)[0].split("[", 1)[0]
        root_token = RMBenchEnvAdapter._safe_path_token(root)
        root_stem = re.sub(r"\d+$", "", root_token).strip("_")
        class_name = RMBenchEnvAdapter._singularize_token(root_stem or root_token)
        if class_name:
            return class_name
        actor_token = RMBenchEnvAdapter._safe_path_token(actor_name)
        actor_stem = re.sub(r"^\d+_", "", re.sub(r"\d+$", "", actor_token)).strip("_")
        return RMBenchEnvAdapter._singularize_token(actor_stem or actor_token) or "object"

    @staticmethod
    def _safe_path_token(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\[[0-9]+\]", lambda match: "_" + match.group(0).strip("[]"), text)
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")

    @staticmethod
    def _singularize_token(value: str) -> str:
        token = str(value or "").strip("_")
        if len(token) > 3 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 2 and token.endswith("ses"):
            return token[:-2]
        if len(token) > 1 and token.endswith("s"):
            return token[:-1]
        return token

    @staticmethod
    def build_reasoner_payload(start: EnvSnapshot, end: EnvSnapshot) -> dict[str, Any]:
        planner_state = np.concatenate(
            [
                start.joint_vector,
                end.joint_vector,
                end.left_endpose,
                end.right_endpose,
            ],
            axis=0,
        ).astype(np.float32)
        return {
            "planner_start_image": start.head_rgb,
            "planner_end_image": end.head_rgb,
            "planner_state": planner_state,
        }

    @staticmethod
    def camera_data(snapshot: EnvSnapshot, camera: str) -> dict[str, np.ndarray | None]:
        normalized = str(camera or "head").strip().lower().replace("_camera", "")
        if normalized not in {"head", "left", "right", "third"}:
            normalized = "head"
        return {
            "rgb": getattr(snapshot, f"{normalized}_rgb", None),
            "depth": getattr(snapshot, f"{normalized}_depth", None),
            "intrinsic_cv": getattr(snapshot, f"{normalized}_intrinsic_cv", None),
            "cam2world_gl": getattr(snapshot, f"{normalized}_cam2world_gl", None),
        }

    @staticmethod
    def summarize(snapshot: EnvSnapshot) -> str:
        state = RMBenchEnvAdapter.robot_state(snapshot)
        left_xyz = state.get("left", {}).get("xyz", [])
        right_xyz = state.get("right", {}).get("xyz", [])
        return (
            f"step={snapshot.step_count}/{snapshot.step_limit}, eval_success={snapshot.eval_success}, "
            f"check_success={snapshot.check_success}, reward={snapshot.max_reward:.3f}, "
            f"joint_norm={state.get('joint_norm', 0.0):.3f}, left_xyz={left_xyz}, right_xyz={right_xyz}"
        )

    @staticmethod
    def robot_state(snapshot: EnvSnapshot) -> dict[str, Any]:
        return {
            "step": int(snapshot.step_count),
            "step_limit": int(snapshot.step_limit),
            "joint_dim": int(snapshot.joint_vector.size),
            "joint_norm": RMBenchEnvAdapter._round_float(float(np.linalg.norm(snapshot.joint_vector))),
            "left": RMBenchEnvAdapter._arm_state(snapshot, "left"),
            "right": RMBenchEnvAdapter._arm_state(snapshot, "right"),
        }

    @staticmethod
    def _arm_state(snapshot: EnvSnapshot, arm: str) -> dict[str, Any]:
        pose = np.asarray(getattr(snapshot, f"{arm}_endpose"), dtype=np.float32)
        xyz = RMBenchEnvAdapter._rounded_list(pose[:3]) if pose.size >= 3 else []
        quat = RMBenchEnvAdapter._normalize_quat_wxyz(pose[3:7]) if pose.size >= 7 else []
        return {
            "xyz": xyz,
            "quat_wxyz": RMBenchEnvAdapter._rounded_list(quat),
            "rpy": RMBenchEnvAdapter._quat_wxyz_to_rpy(quat),
            "gripper": RMBenchEnvAdapter._snapshot_gripper(snapshot, arm),
            "gripper_command": snapshot.gripper_command_by_arm.get(arm),
        }

    @staticmethod
    def _snapshot_gripper(snapshot: EnvSnapshot, arm: str) -> float | None:
        raw = snapshot.raw if isinstance(snapshot.raw, dict) else {}
        endpose = raw.get("endpose", {}) if isinstance(raw.get("endpose", {}), dict) else {}
        joint_action = raw.get("joint_action", {}) if isinstance(raw.get("joint_action", {}), dict) else {}
        candidates: list[Any] = []
        if arm == "left":
            candidates.extend([endpose.get("left_gripper"), endpose.get("gripper"), joint_action.get("left_gripper"), joint_action.get("gripper")])
        else:
            candidates.extend([endpose.get("right_gripper"), joint_action.get("right_gripper")])
        for value in candidates:
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return RMBenchEnvAdapter._round_float(number)
        return None

    @staticmethod
    def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            return np.asarray([], dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-8:
            return np.asarray([], dtype=np.float64)
        return quat / norm

    @staticmethod
    def _quat_wxyz_to_rpy(quat: np.ndarray) -> list[float]:
        quat = np.asarray(quat, dtype=np.float64)
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            return []
        w, x, y, z = [float(item) for item in quat]
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(sin_pitch)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return RMBenchEnvAdapter._rounded_list([roll, pitch, yaw])

    @staticmethod
    def _rounded_list(values: Any) -> list[float]:
        return [RMBenchEnvAdapter._round_float(float(item)) for item in np.asarray(values).reshape(-1) if math.isfinite(float(item))]

    @staticmethod
    def _round_float(value: float) -> float:
        return round(float(value), 5)

    @staticmethod
    def measure_progress(previous: EnvSnapshot | None, current: EnvSnapshot) -> dict[str, float | bool]:
        if previous is None:
            return {
                "image_delta": 0.0,
                "joint_delta": 0.0,
                "made_progress": True,
            }

        image_delta = float(
            np.mean(
                np.abs(
                    current.head_rgb.astype(np.float32) - previous.head_rgb.astype(np.float32)
                )
            )
        )
        joint_delta = float(np.linalg.norm(current.joint_vector - previous.joint_vector))
        made_progress = bool(image_delta >= 1.0 or joint_delta >= 1e-3)
        return {
            "image_delta": image_delta,
            "joint_delta": joint_delta,
            "made_progress": made_progress,
        }
