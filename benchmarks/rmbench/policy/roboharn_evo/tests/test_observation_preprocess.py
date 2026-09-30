from __future__ import annotations

from dataclasses import dataclass
import json
import unittest
from unittest import mock

import numpy as np

from policy.roboharn_evo.agent.environment import EnvSnapshot
from policy.roboharn_evo.agent.environment import RMBenchEnvAdapter
from policy.roboharn_evo.agent.core.img_agent import ImgAgent
from policy.roboharn_evo.agent.operation_candidates import pose7_to_matrix


@dataclass(frozen=True)
class DummyConfig:
    initial_memory_text: str = "The task has started."
    observation_preprocess_enabled: bool = True
    observation_preprocess_every_n_steps: int = 1
    observation_preprocess_auto_objects: bool = True
    observation_preprocess_query_url: str = ""
    observation_preprocess_normalization_url: str = ""
    observation_preprocess_query_timeout_sec: int = 120
    observation_preprocess_max_objects: int = 3
    observation_preprocess_objects: tuple[dict[str, str], ...] = ()
    observation_preprocess_backend: str = "sam3"
    observation_preprocess_camera: str = "head"
    observation_preprocess_cameras: tuple[str, ...] = ()
    observation_verification_cameras: tuple[str, ...] = ()
    observation_preprocess_service_url: str = "http://127.0.0.1:9301"
    oracle_objects_enabled: bool = False
    oracle_objects_include_all: bool = True
    oracle_objects_max_objects: int = 12
    observation_grounding_enabled: bool = False
    observation_grounding_camera: str = "head"
    observation_grounding_cameras: tuple[str, ...] = ()
    observation_grounding_min_valid_ratio: float = 0.05
    observation_grounding_max_points: int = 5000
    observation_grounding_approach_height_m: float = 0.08
    scene_memory_enabled: bool = True
    scene_memory_max_missing_steps: int = 20
    scene_memory_stable_distance_m: float = 0.08
    scene_memory_temporal_window_size: int = 8
    scene_memory_min_candidate_score: float = 0.12
    scene_memory_max_object_z_extent_m: float = 0.18
    scene_memory_max_object_xy_extent_m: float = 0.30
    scene_memory_max_object_world_z_m: float = 0.95
    scene_memory_robot_self_filter_radius_m: float = 0.08
    scene_memory_robot_self_filter_z_margin_m: float = 0.08
    decision_interval: int = 1
    max_retries_per_skill: int = 2
    max_steps_per_skill: int = 80
    stall_patience: int = 12
    mode: str = "deployment"
    interrupt_on_skill_change: bool = True


class DummyAgentCard:
    config = DummyConfig()
    control_runtime = None
    executor_runtime = None
    control_model_name = "test"
    executor_name = "test"
    ood_backend_config = None
    recovery_backend_config = None


def make_card(config: DummyConfig) -> object:
    class Card:
        control_runtime = None
        executor_runtime = None
        control_model_name = "test"
        executor_name = "test"
        ood_backend_config = None
        recovery_backend_config = None

    card = Card()
    card.config = config
    return card


def make_snapshot(*, include_depth: bool = False) -> EnvSnapshot:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    joint_vector = np.zeros(14, dtype=np.float32)
    endpose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
    kwargs = {}
    if include_depth:
        left_cam2world = np.eye(4, dtype=np.float64)
        left_cam2world[0, 3] = 1.0
        right_cam2world = np.eye(4, dtype=np.float64)
        right_cam2world[0, 3] = 2.0
        intrinsic = np.array(
            [
                [100.0, 0.0, 3.5],
                [0.0, 100.0, 3.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        kwargs.update(
            {
                "head_depth": np.full((8, 8), 1000.0, dtype=np.float64),
                "left_depth": np.full((8, 8), 1000.0, dtype=np.float64),
                "right_depth": np.full((8, 8), 1000.0, dtype=np.float64),
                "head_intrinsic_cv": intrinsic,
                "left_intrinsic_cv": intrinsic,
                "right_intrinsic_cv": intrinsic,
                "head_cam2world_gl": np.eye(4, dtype=np.float64),
                "left_cam2world_gl": left_cam2world,
                "right_cam2world_gl": right_cam2world,
            }
        )
    return EnvSnapshot(
        raw={"endpose": {}, "joint_action": {"vector": joint_vector.tolist()}},
        head_rgb=image,
        left_rgb=image,
        right_rgb=image,
        joint_vector=joint_vector,
        left_endpose=endpose,
        right_endpose=endpose,
        step_count=0,
        step_limit=100,
        eval_success=False,
        check_success=False,
        max_reward=0.0,
        instruction="cover the block",
        **kwargs,
    )


class FakePose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)) -> None:
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)


class FakeActor:
    def __init__(self, name: str, position, config: dict | None = None, contact_matrices: list[np.ndarray] | None = None) -> None:
        self._name = name
        self.position = list(position)
        self.config = config or {"center": [0, 0, 0], "extents": [0.02, 0.02, 0.02], "scale": [0.02, 0.02, 0.02]}
        self.contact_matrices = list(contact_matrices or [])

    def get_pose(self) -> FakePose:
        return FakePose(self._position)

    @property
    def _position(self):
        return getattr(self, "position", [-0.2, -0.18, 0.76])

    def get_name(self) -> str:
        return self._name

    def iter_contact_points(self, ret: str = "matrix"):
        if ret != "matrix":
            raise ValueError(ret)
        for index, matrix in enumerate(self.contact_matrices):
            yield index, matrix


class FakeTaskEnv:
    def __init__(self) -> None:
        self.blocks = [FakeActor("box", [-0.2, -0.18, 0.76])]
        self.covers = [FakeActor("003_cover", [-0.2, -0.05, 0.82], {"center": [0, 0, 0], "extents": [0.12, 0.16, 0.04], "scale": [1, 1, 1]})]
        self.table = FakeActor("table", [0.0, 0.0, 0.74])


class OracleObjectExtractionTest(unittest.TestCase):
    def test_robot_ee_to_tcp_offset_uses_robot_kinematics_only(self) -> None:
        class FakeRobot:
            @staticmethod
            def get_left_ee_pose():
                return [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]

            @staticmethod
            def get_left_tcp_pose():
                return [0.1, 0.2, 0.42, 1.0, 0.0, 0.0, 0.0]

        class FakeRobotEnv:
            robot = FakeRobot()

        self.assertAlmostEqual(RMBenchEnvAdapter.robot_ee_to_tcp_m(FakeRobotEnv()), 0.12)

    def test_robot_tcp_calibration_preserves_full_transform_for_each_arm(self) -> None:
        half_sqrt = 2**-0.5

        class FakeRobot:
            @staticmethod
            def get_left_ee_pose():
                return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

            @staticmethod
            def get_left_tcp_pose():
                return [0.12, 0.0, 0.0, half_sqrt, 0.0, 0.0, half_sqrt]

            @staticmethod
            def get_right_ee_pose():
                return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

            @staticmethod
            def get_right_tcp_pose():
                return [0.0, 0.13, 0.0, 1.0, 0.0, 0.0, 0.0]

        class FakeRobotEnv:
            robot = FakeRobot()

        calibrations = RMBenchEnvAdapter.robot_tcp_calibration_by_arm(FakeRobotEnv())

        self.assertEqual(set(calibrations), {"left", "right"})
        np.testing.assert_allclose(calibrations["left"]["translation_m"], [0.12, 0.0, 0.0])
        np.testing.assert_allclose(
            calibrations["left"]["quat_wxyz"],
            [half_sqrt, 0.0, 0.0, half_sqrt],
            atol=1e-6,
        )
        np.testing.assert_allclose(calibrations["right"]["translation_m"], [0.0, 0.13, 0.0])
        self.assertNotEqual(
            calibrations["left"]["action_to_tcp_matrix"],
            calibrations["right"]["action_to_tcp_matrix"],
        )

    def test_oracle_objects_extracts_task_actors_and_skips_scene_fixtures(self) -> None:
        objects = RMBenchEnvAdapter.oracle_objects(FakeTaskEnv())

        source_paths = {item["source_path"] for item in objects}
        self.assertIn("blocks[0]", source_paths)
        self.assertIn("covers[0]", source_paths)
        self.assertNotIn("table", source_paths)

        by_source = {item["source_path"]: item for item in objects}
        self.assertEqual(by_source["blocks[0]"]["class_name"], "block")
        self.assertIn("block", by_source["blocks[0]"]["aliases"])
        self.assertEqual(by_source["blocks[0]"]["grounding_3d"]["centroid_world"], [-0.2, -0.18, 0.76])
        self.assertEqual(by_source["covers[0]"]["class_name"], "cover")
        self.assertNotIn("lid", by_source["covers[0]"]["aliases"])

    def test_oracle_objects_deduplicate_dynamic_aliases_by_simulator_id(self) -> None:
        class EntityProxy:
            def __init__(self, global_id: int) -> None:
                self._global_id = global_id

            def get_global_id(self) -> int:
                return self._global_id

        class ProxyBackedActor(FakeActor):
            def __init__(self, name: str, position, global_id: int) -> None:
                super().__init__(name, position)
                self._global_id = global_id

            @property
            def actor(self):
                # Model a pybind property that materializes a new Python proxy
                # for the same simulator entity on every access.
                return EntityProxy(self._global_id)

        class AliasTaskEnv:
            def __init__(self) -> None:
                self.block1 = ProxyBackedActor(
                    "box",
                    [0.04, -0.1, 0.76],
                    global_id=17,
                )
                self.left_block = ProxyBackedActor(
                    "box",
                    [0.04, -0.1, 0.76],
                    global_id=17,
                )

        objects = RMBenchEnvAdapter.oracle_objects(AliasTaskEnv())

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["source_path"], "block1")
        self.assertEqual(
            set(objects[0]["source_paths"]),
            {"block1", "left_block"},
        )
        self.assertIn("left_block", objects[0]["aliases"])

    def test_oracle_objects_deduplicate_same_wrapper_with_ephemeral_actor_proxy(
        self,
    ) -> None:
        class EphemeralEntityProxy:
            pass

        class ProxyBackedActor(FakeActor):
            @property
            def actor(self):
                # Match a runtime wrapper whose nested pybind proxy has no
                # exposed simulator ID and is rematerialized on every access.
                return EphemeralEntityProxy()

        class AliasTaskEnv:
            def __init__(self) -> None:
                self.block1 = ProxyBackedActor(
                    "box",
                    [0.16, -0.1, 0.76],
                )
                self.block2 = ProxyBackedActor(
                    "box",
                    [0.28, -0.1, 0.76],
                )
                self.block3 = ProxyBackedActor(
                    "box",
                    [0.04, -0.1, 0.76],
                )
                self.button = ProxyBackedActor(
                    "button",
                    [-0.2, -0.1, 0.74],
                )
                self.left_block = self.block2
                self.middle_block = self.block1
                self.right_block = self.block3

        objects = RMBenchEnvAdapter.oracle_objects(AliasTaskEnv())

        self.assertEqual(len(objects), 4)
        by_source = {item["source_path"]: item for item in objects}
        self.assertEqual(
            set(by_source["block1"]["source_paths"]),
            {"block1", "middle_block"},
        )
        self.assertEqual(
            set(by_source["block2"]["source_paths"]),
            {"block2", "left_block"},
        )
        self.assertEqual(
            set(by_source["block3"]["source_paths"]),
            {"block3", "right_block"},
        )
        self.assertIn("right_block", by_source["block3"]["aliases"])

    def test_oracle_objects_excludes_actor_handles_from_previous_scene(self) -> None:
        class EntityProxy:
            def __init__(self, global_id: int) -> None:
                self._global_id = global_id

            def get_global_id(self) -> int:
                return self._global_id

        class SceneBoundActor(FakeActor):
            def __init__(
                self,
                name: str,
                position,
                *,
                entity: EntityProxy,
            ) -> None:
                super().__init__(name, position)
                self.actor = entity

        current_entity = EntityProxy(101)
        stale_expert_entity = EntityProxy(17)

        class CurrentScene:
            @staticmethod
            def get_all_actors():
                return [current_entity]

            @staticmethod
            def get_all_articulations():
                return []

        class ReusedTaskEnv:
            def __init__(self) -> None:
                self.scene = CurrentScene()
                self.block1 = SceneBoundActor(
                    "box",
                    [0.04, -0.1, 0.76],
                    entity=current_entity,
                )
                # Models a task-specific alias retained after expert close and
                # a second setup on the same Python task object.
                self.left_block = SceneBoundActor(
                    "box",
                    [0.047, -0.098, 0.76],
                    entity=stale_expert_entity,
                )

        objects = RMBenchEnvAdapter.oracle_objects(ReusedTaskEnv())

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["source_path"], "block1")
        self.assertEqual(objects[0]["source_paths"], ["block1"])
        self.assertNotIn("left_block", objects[0]["aliases"])

    def test_oracle_aliases_do_not_invent_task_vocabulary(self) -> None:
        cover_aliases = RMBenchEnvAdapter._oracle_aliases("covers[0]", "", "cover")
        box_aliases = RMBenchEnvAdapter._oracle_aliases("containers[0]", "", "box")
        lid_aliases = RMBenchEnvAdapter._oracle_aliases("lids[0]", "", "lid")

        self.assertNotIn("lid", cover_aliases)
        self.assertNotIn("block", box_aliases)
        self.assertNotIn("cover", lid_aliases)

    def test_scaled_mesh_geometry_and_contact_pose_are_converted_to_ee_targets(self) -> None:
        position = np.asarray([-0.2, -0.050005, 0.740576], dtype=np.float64)
        quat = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        rotation = RMBenchEnvAdapter._quat_wxyz_to_matrix(quat)
        contact_matrix = np.eye(4, dtype=np.float64)
        contact_matrix[:3, :3] = rotation
        contact_matrix[:3, 3] = position + rotation @ np.asarray([0.0, 0.165 * 0.65, 0.0])
        grounding = RMBenchEnvAdapter._oracle_grounding(
            object_id="generic_object",
            position=position,
            quat=quat,
            config={
                "center": [-5.464e-6, 0.0799938, -1.222e-6],
                "extents": [0.12181696, 0.1617982, 0.12175755],
                "scale": [1.0, 0.65, 1.0],
            },
            contact_matrices=[contact_matrix],
            ee_to_contact_m=0.12,
        )

        self.assertAlmostEqual(grounding["centroid_world"][2], 0.792572, places=5)
        self.assertAlmostEqual(grounding["bbox_world_max"][2], 0.84516, places=5)
        self.assertAlmostEqual(grounding["grasp_pose_world"][2], 0.967826, places=5)
        self.assertAlmostEqual(grounding["approach_pose_world"][2], 1.047826, places=5)
        self.assertAlmostEqual(grounding["object_contact_pose_world"][2], 0.847826, places=5)
        self.assertEqual(grounding["contact_pose_world"], grounding["grasp_pose_world"])
        np.testing.assert_allclose(grounding["grasp_pose_world"][3:7], [0.5, -0.5, 0.5, 0.5], atol=1e-6)

    def test_oracle_contact_matrices_generate_per_arm_operation_candidates(self) -> None:
        contact_matrix = np.eye(4, dtype=np.float64)
        calibration = {
            arm: {
                "action_to_tcp_matrix": [
                    [1.0, 0.0, 0.0, 0.12],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "action_pose_world": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "source": "robot_kinematics",
            }
            for arm in ("left", "right")
        }
        calibration["left"]["action_to_tcp_matrix"] = [
            [0.0, -1.0, 0.0, 0.12],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        grounding = RMBenchEnvAdapter._oracle_grounding(
            object_id="generic_object",
            position=np.zeros(3, dtype=np.float64),
            quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            config={
                "center": [0.0, 0.0, 0.0],
                "extents": [0.02, 0.02, 0.02],
                "scale": [0.02, 0.02, 0.02],
            },
            contact_matrices=[contact_matrix],
            tcp_calibration_by_arm=calibration,
        )

        candidates = grounding["operation_pose_candidates"]
        self.assertEqual(len(candidates), 4)
        self.assertEqual({item["arm"] for item in candidates}, {"left", "right"})
        self.assertEqual({item["action_mode"] for item in candidates}, {"grasp", "contact"})
        for candidate in candidates:
            self.assertTrue(
                {
                    "candidate_id",
                    "action_mode",
                    "arm",
                    "object_contact_pose",
                    "tcp_pose",
                    "approach_pose",
                    "approach_direction",
                    "geometry_source",
                }.issubset(candidate)
            )
        left_grasp = next(
            item
            for item in candidates
            if item["arm"] == "left" and item["action_mode"] == "grasp"
        )
        left_contact = next(
            item
            for item in candidates
            if item["arm"] == "left" and item["action_mode"] == "contact"
        )
        self.assertAlmostEqual(left_grasp["grasp_clearance_m"], 0.02)
        self.assertAlmostEqual(left_contact["grasp_clearance_m"], 0.0)
        np.testing.assert_allclose(left_grasp["tcp_pose"][:3], [0.0, 0.02, 0.0])
        np.testing.assert_allclose(left_contact["tcp_pose"][:3], [0.0, 0.0, 0.0])
        left_action_transform = pose7_to_matrix(left_contact["ee_target_pose"])
        left_tcp_transform = pose7_to_matrix(left_contact["tcp_pose"])
        assert left_action_transform is not None
        assert left_tcp_transform is not None
        np.testing.assert_allclose(
            np.linalg.inv(left_action_transform) @ left_tcp_transform,
            calibration["left"]["action_to_tcp_matrix"],
            atol=1e-6,
        )
        # Legacy single fields remain a compatibility view; runtime selection
        # prefers the structured candidate above.
        np.testing.assert_allclose(grounding["grasp_pose_world"][:3], [0.0, 0.12, 0.0])

    def test_from_env_no_oracle_does_not_read_or_preserve_oracle_catalog(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        observation = {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            },
            "joint_action": {"vector": [0.0] * 14},
            "endpose": {
                "left_endpose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "right_endpose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            },
            "_tcm_oracle_objects": [{"oracle_id": "must_not_leak"}],
        }

        with mock.patch.object(
            RMBenchEnvAdapter,
            "oracle_objects",
            side_effect=AssertionError("no-oracle snapshot must not inspect task actors"),
        ):
            snapshot = RMBenchEnvAdapter.from_env(FakeTaskEnv(), observation)

        self.assertNotIn("_tcm_oracle_objects", snapshot.raw)
        self.assertIsNone(snapshot.oracle_objects)

    def test_from_env_explicit_oracle_exposes_typed_catalog(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        observation = {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            },
            "joint_action": {"vector": [0.0] * 14},
            "endpose": {
                "left_endpose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "right_endpose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            },
        }

        snapshot = RMBenchEnvAdapter.from_env(
            FakeTaskEnv(),
            observation,
            oracle_objects_enabled=True,
        )

        self.assertIsInstance(snapshot.oracle_objects, list)
        self.assertEqual({item["source_path"] for item in snapshot.oracle_objects or []}, {"blocks[0]", "covers[0]"})

    def test_from_env_exposes_calibrated_third_view_camera(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.full((8, 8), 900.0, dtype=np.float64)
        intrinsic = np.array(
            [
                [100.0, 0.0, 3.5],
                [0.0, 100.0, 3.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cam2world = np.eye(4, dtype=np.float64)
        cam2world[1, 3] = 0.23
        observation = {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
                "third_camera": {
                    "rgb": image + 7,
                    "depth": depth,
                    "intrinsic_cv": intrinsic,
                    "cam2world_gl": cam2world,
                },
            },
            "third_view_rgb": image + 7,
            "joint_action": {"vector": [0.0] * 14},
            "endpose": {
                "left_endpose": [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "right_endpose": [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            },
        }

        snapshot = RMBenchEnvAdapter.from_env(
            FakeTaskEnv(),
            observation,
        )
        third = RMBenchEnvAdapter.camera_data(
            snapshot,
            "third",
        )

        np.testing.assert_array_equal(third["rgb"], image + 7)
        np.testing.assert_array_equal(third["depth"], depth)
        np.testing.assert_array_equal(
            third["intrinsic_cv"],
            intrinsic,
        )
        np.testing.assert_array_equal(
            third["cam2world_gl"],
            cam2world,
        )


class ObservationPreprocessTest(unittest.TestCase):
    def test_reference_set_query_stays_unbound_and_keeps_relation_metadata(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        normalized = agent._normalize_perception_queries(
            [
                {
                    "object_id": "mats",
                    "text_prompt": "the complete set of four mats",
                    "role": "context",
                    "entity_scope": "reference_set",
                    "placement_relation": "center_of",
                    "expected_count": 4,
                    "instance_ref": "track_0001",
                }
            ],
            max_objects=3,
        )
        marked = agent._mark_agent_identity_binding_required(
            normalized,
            allow_context_binding=True,
        )

        self.assertEqual(marked[0]["entity_scope"], "reference_set")
        self.assertEqual(marked[0]["placement_relation"], "center_of")
        self.assertEqual(marked[0]["expected_count"], 4)
        self.assertIs(marked[0]["identity_binding_required"], False)
        self.assertNotIn("instance_ref", marked[0])

    def test_img_agent_threads_oracle_capability_to_runtime_and_recovery_refreshes(self) -> None:
        task_env = object()
        observation: dict = {}
        snapshot = make_snapshot()

        for enabled in (False, True):
            with self.subTest(oracle_objects_enabled=enabled):
                agent = ImgAgent(make_card(DummyConfig(oracle_objects_enabled=enabled)))
                agent.current_instruction = "opaque instruction"
                recovery_adapter = agent._recovery_dispatcher._executor.build_adapter(task_env)
                self.assertEqual(recovery_adapter.oracle_objects_enabled, enabled)
                with (
                    mock.patch(
                        "policy.roboharn_evo.agent.core.img_agent.RMBenchEnvAdapter.from_env",
                        return_value=snapshot,
                    ) as from_env,
                    mock.patch.object(agent, "update_snapshot"),
                    mock.patch.object(
                        agent,
                        "_commit_pure_tool_control_environment_success",
                        return_value=True,
                    ),
                ):
                    agent.run_step(task_env, observation)

                from_env.assert_called_once_with(
                    task_env,
                    observation,
                    oracle_objects_enabled=enabled,
                )

    def test_oracle_matching_uses_object_identity_before_relational_description(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        catalog = [
            {
                "oracle_id": "blocks_0",
                "source_path": "blocks[0]",
                "class_name": "block",
                "aliases": ["block", "green_block"],
                "position_world": [-0.2, -0.18, 0.76],
            },
            {
                "oracle_id": "covers_0",
                "source_path": "covers[0]",
                "class_name": "cover",
                "aliases": ["cover", "lid"],
                "position_world": [-0.2, -0.05, 0.82],
            },
        ]

        matches = agent._match_oracle_catalog(
            query={
                "object_id": "lid",
                "text_prompt": "left lid above the green block",
                "instance_hint": "tool used to cover the green block",
            },
            catalog=catalog,
        )

        self.assertEqual([item["oracle_id"] for item in matches], ["covers_0"])

    def test_oracle_matching_obeys_exact_agent_identity_without_semantic_fallback(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        catalog = [
            {"oracle_id": "parts_a", "source_path": "parts[0]", "class_name": "component"},
            {"oracle_id": "parts_b", "source_path": "parts[1]", "class_name": "component"},
        ]

        matches = agent._match_oracle_catalog(
            query={
                "object_id": "component",
                "instance_hint": "the other candidate",
                "oracle_id": "parts_b",
                "identity_binding_required": True,
            },
            catalog=catalog,
        )
        unresolved = agent._match_oracle_catalog(
            query={
                "object_id": "component",
                "identity_binding_required": True,
            },
            catalog=catalog,
        )

        self.assertEqual([item["oracle_id"] for item in matches], ["parts_b"])
        self.assertEqual(unresolved, [])

    def test_runtime_canonicalizes_valid_agent_track_reference_and_rejects_unknown_reference(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        candidates = [
            {
                "instance_id": "component_01",
                "track_id": "track_alpha",
                "oracle_id": None,
                "status": "visible",
                "world_m": [0.11, -0.03, 0.82],
            }
        ]

        valid = agent._prepare_agent_identity_bindings(
            [{"object_id": "component", "role": "target", "instance_ref": "component_01"}],
            scene_instances=candidates,
        )
        invalid = agent._prepare_agent_identity_bindings(
            [{"object_id": "component", "role": "target", "instance_ref": "track_missing"}],
            scene_instances=candidates,
        )

        self.assertEqual(valid[0]["instance_ref"], "track_alpha")
        self.assertNotIn("identity_binding_error", valid[0])
        self.assertEqual(invalid[0]["identity_binding_error"], "selected_instance_not_visible_or_grounded")

    def test_release_verification_marks_exact_context_binding_as_required(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        scene_memory = {
            "env_step": 25,
            "instances": [
                {
                    "instance_id": "track_0007",
                    "track_id": "track_0007",
                    "class": "cube",
                    "class_aliases": ["cube"],
                    "status": "visible",
                    "actionable": True,
                    "world_m": [0.12, -0.16, 0.76],
                }
            ],
            "task_focus": {},
            "uncertainty": [],
            "temporal_memory": {},
        }

        rebound, queries = agent._bind_scene_memory_focus_with_agent(
            scene_memory=scene_memory,
            perception_queries=[
                {
                    "object_id": "cube",
                    "text_prompt": "cube",
                    "role": "context",
                    "instance_ref": "track_0007",
                }
            ],
            observation_summary="synthetic observation",
            current_subtask="verify the released cube without touching it",
            identity_binding_required=True,
            allow_context_binding=True,
        )

        self.assertTrue(queries[0]["identity_binding_required"])
        self.assertEqual(
            rebound["task_focus"]["context_instances"],
            ["track_0007"],
        )
        self.assertEqual(
            rebound["task_focus"]["identity_binding_roles"],
            ["context"],
        )
        self.assertEqual(
            rebound["task_focus"]["identity_binding_errors"],
            [],
        )

    def test_tracked_identity_is_advertised_for_exact_recovery_binding(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        scene_memory = {
            "env_step": 13,
            "instances": [
                {
                    "instance_id": "component_left",
                    "track_id": "track_0001",
                    "class": "component",
                    "class_aliases": ["component"],
                    "status": "visible",
                    "stability": "stable",
                    "actionable": True,
                    "world_m": [-0.12, -0.04, 0.81],
                },
                {
                    "instance_id": "component_middle",
                    "track_id": "track_0002",
                    "class": "component",
                    "class_aliases": ["component"],
                    "status": "tracked",
                    "stability": "geometry_inconsistent_current_frame",
                    "missing_steps": 6,
                    "actionable": True,
                    "world_m": [0.0, -0.04, 0.81],
                    "approach_world_m": [0.0, -0.04, 0.99],
                    "contact_world_m": [0.0, -0.04, 0.91],
                    "quality_warnings": [
                        "temporal_action_geometry_inconsistent:approach_relative_jump=0.168m>0.040m"
                    ],
                },
            ],
            "task_focus": {},
            "uncertainty": [],
            "temporal_memory": {},
        }
        agent.memory_store.record_scene_memory(scene_memory)

        catalog = agent._scene_instance_catalog_for_query_payload()
        tracked = next(item for item in catalog if item["track_id"] == "track_0002")

        self.assertTrue(tracked["recovery_binding_only"])
        self.assertFalse(tracked["grounded_execution_allowed"])
        self.assertEqual(tracked["stability"], "geometry_inconsistent_current_frame")
        self.assertEqual(tracked["missing_steps"], 6)
        rebound, rebound_queries = agent._bind_scene_memory_focus_with_agent(
            scene_memory=scene_memory,
            perception_queries=[
                {
                    "object_id": "component",
                    "text_prompt": "component",
                    "role": "target",
                    "instance_ref": "track_0002",
                    "identity_binding_required": True,
                }
            ],
            observation_summary="synthetic observation",
            current_subtask="operate on the explicitly selected retained component",
        )

        self.assertEqual(rebound_queries[0]["instance_ref"], "track_0002")
        self.assertEqual(rebound["task_focus"]["target_instances"], ["component_middle"])
        self.assertEqual(rebound["task_focus"]["identity_binding_errors"], [])
        agent.memory_store.record_scene_memory(rebound)
        self.assertTrue(agent._debug_recovery_scene_ready())

    def test_non_oracle_candidates_are_rebound_by_agent_selected_track(self) -> None:
        agent = ImgAgent(make_card(DummyConfig(observation_preprocess_normalization_url="http://127.0.0.1:9/normalize_perception_queries")))
        scene_memory = {
            "env_step": 0,
            "instances": [
                {
                    "instance_id": "component_01",
                    "track_id": "track_alpha",
                    "class": "component",
                    "class_aliases": ["component"],
                    "status": "visible",
                    "actionable": True,
                    "world_m": [0.11, -0.03, 0.82],
                },
                {
                    "instance_id": "component_02",
                    "track_id": "track_beta",
                    "class": "component",
                    "class_aliases": ["component"],
                    "status": "visible",
                    "actionable": True,
                    "world_m": [-0.07, 0.09, 0.84],
                },
            ],
            "task_focus": {},
            "uncertainty": [],
            "temporal_memory": {},
        }
        queries = [
            {
                "object_id": "component",
                "text_prompt": "component",
                "role": "target",
                "identity_binding_required": True,
            }
        ]

        with mock.patch.object(
            agent,
            "_normalize_perception_queries_with_api",
            return_value=[
                {
                    "object_id": "component",
                    "text_prompt": "component",
                    "role": "target",
                    "instance_ref": "track_beta",
                }
            ],
        ):
            rebound, rebound_queries = agent._bind_scene_memory_focus_with_agent(
                scene_memory=scene_memory,
                perception_queries=queries,
                observation_summary="synthetic observation",
                current_subtask="operate on the selected component",
            )

        self.assertEqual(rebound_queries[0]["instance_ref"], "track_beta")
        self.assertEqual(rebound["task_focus"]["target_instances"], ["component_02"])
        self.assertEqual(rebound["task_focus"]["identity_binding_errors"], [])

    def test_control_waits_while_required_identity_binding_is_unresolved(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        agent.memory_store.record_scene_memory(
            {
                "instances": [{"instance_id": "component_01"}],
                "task_focus": {
                    "target_instances": [],
                    "tool_instances": [],
                    "identity_binding_required": True,
                    "identity_binding_errors": ["component:missing_instance_reference"],
                },
            }
        )

        self.assertFalse(agent._debug_recovery_scene_ready())

    def test_control_accepts_bound_context_for_release_verification(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        agent.memory_store.record_scene_memory(
            {
                "instances": [
                    {
                        "instance_id": "track_0007",
                        "world_m": [0.12, -0.16, 0.76],
                    }
                ],
                "task_focus": {
                    "target_instances": [],
                    "tool_instances": [],
                    "context_instances": ["track_0007"],
                    "identity_binding_required": True,
                    "identity_binding_roles": ["context"],
                    "identity_binding_errors": [],
                },
            }
        )

        self.assertTrue(agent._debug_recovery_scene_ready())

    def test_empty_agent_perception_response_clears_focus_instead_of_reusing_candidate(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=True,
            observation_preprocess_query_url="http://127.0.0.1:9/perception_queries",
        )
        agent = ImgAgent(make_card(config))

        class EmptyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"queries": []}'

        with mock.patch("policy.roboharn_evo.agent.core.img_agent.request.urlopen", return_value=EmptyResponse()):
            agent.update_snapshot(make_snapshot())

        focus = agent.memory_store.state.working.scene_memory["task_focus"]
        self.assertEqual(focus["target_instances"], [])
        self.assertEqual(focus["tool_instances"], [])
        self.assertEqual(focus["identity_binding_errors"], ["agent_returned_no_target_or_tool_binding"])

    def test_normalization_rejects_context_only_action_binding(self) -> None:
        config = DummyConfig(
            observation_preprocess_normalization_url=(
                "http://127.0.0.1:9104/normalize_perception_queries"
            ),
        )
        agent = ImgAgent(make_card(config))
        captured_payloads: list[dict] = []

        class ContextOnlyResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "queries": [
                            {
                                "object_id": "reference_object",
                                "text_prompt": "reference object",
                                "role": "context",
                            }
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(request_obj, timeout=0):
            captured_payloads.append(
                json.loads(request_obj.data.decode("utf-8"))
            )
            return ContextOnlyResponse()

        requirement = {
            "required_any_roles": ["target", "tool"],
            "failure_reason": "agent_returned_no_target_or_tool_binding",
            "retry_attempt": 1,
            "force_refresh": True,
        }
        with mock.patch(
            "policy.roboharn_evo.agent.core.img_agent.request.urlopen",
            side_effect=fake_urlopen,
        ):
            queries = agent._normalize_perception_queries_with_api(
                [
                    {
                        "object_id": "reference_object",
                        "text_prompt": "reference object",
                        "role": "context",
                    }
                ],
                max_objects=3,
                observation_summary="synthetic observation",
                binding_requirement=requirement,
            )

        self.assertEqual(queries, [])
        self.assertEqual(
            captured_payloads[0]["binding_requirement"],
            requirement,
        )

    def test_normalization_can_repair_context_only_query_to_actionable_role(self) -> None:
        config = DummyConfig(
            observation_preprocess_normalization_url=(
                "http://127.0.0.1:9104/normalize_perception_queries"
            ),
        )
        agent = ImgAgent(make_card(config))

        class ActionableResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "queries": [
                            {
                                "object_id": "movable_object",
                                "text_prompt": "movable object",
                                "role": "target",
                                "instance_ref": "track_0002",
                            }
                        ]
                    }
                ).encode("utf-8")

        with mock.patch(
            "policy.roboharn_evo.agent.core.img_agent.request.urlopen",
            return_value=ActionableResponse(),
        ):
            queries = agent._normalize_perception_queries_with_api(
                [
                    {
                        "object_id": "reference_object",
                        "text_prompt": "reference object",
                        "role": "context",
                    }
                ],
                max_objects=3,
                observation_summary="synthetic observation",
                binding_requirement={
                    "required_any_roles": ["target", "tool"],
                },
            )

        self.assertEqual(queries[0]["role"], "target")
        self.assertEqual(queries[0]["instance_ref"], "track_0002")

    def test_update_snapshot_records_segmentation_preprocess(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=False,
            observation_preprocess_objects=({"object_id": "block", "text_prompt": "block", "role": "target", "reason": "test query"},),
        )
        agent = ImgAgent(make_card(config))
        tool_result = {
            "success": True,
            "object_id": "block",
            "text_prompt": "block",
            "bbox_xyxy": [1, 2, 3, 4],
            "centroid_px": [2.0, 3.0],
            "score": 0.9,
            "mask_path": "/tmp/block.png",
            "backend": "sam3",
            "camera": "head",
            "env_step": 0,
        }

        async def fake_segment_object(**kwargs):
            return json.dumps(tool_result)

        with mock.patch.object(agent.agent_tools, "segment_object", fake_segment_object):
            agent.update_snapshot(make_snapshot())

        preprocess = agent.memory_store.state.working.observation_preprocess
        self.assertEqual(preprocess["stage"], "observation_preprocess")
        self.assertEqual(preprocess["segmentation"][0]["object_id"], "block")
        self.assertEqual(preprocess["segmentation"][0]["bbox_xyxy"], [1, 2, 3, 4])
        self.assertEqual(preprocess["segmentation"][0]["query_role"], "target")
        summary = agent.memory_store.state.working.recent_observation_summary
        self.assertIn("scene_memory=task_focus", summary)
        self.assertIn("target=[(track_0001", summary)
        self.assertIn("sam3_candidates=block@head:count=1,role=target", summary)
        self.assertNotIn("perception=block:bbox=", summary)

    def test_update_snapshot_trace_events_share_capture_identity(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        recorded: list[tuple[str, dict]] = []

        with (
            mock.patch.object(
                agent,
                "preprocess_observation",
                return_value={
                    "stage": "observation_preprocess",
                    "segmentation": [],
                },
            ),
            mock.patch.object(
                agent,
                "update_scene_memory",
                return_value={"instances": []},
            ),
            mock.patch.object(
                agent,
                "_with_runtime_manipulation_state",
                side_effect=lambda scene: dict(scene),
            ),
            mock.patch.object(
                agent,
                "_resolve_pending_release_from_runtime",
            ),
            mock.patch.object(
                agent,
                "_record_trace_and_rollout_event",
                side_effect=lambda event, payload: recorded.append(
                    (event, dict(payload))
                ),
            ),
        ):
            agent.update_snapshot(make_snapshot())

        self.assertEqual(
            [event for event, _ in recorded],
            [
                "observation_preprocess",
                "scene_memory_update",
                "observation_preprocess_finalized",
            ],
        )
        identities = [
            (
                payload.get("observation_generation"),
                payload.get("observation_capture_id"),
            )
            for _, payload in recorded
        ]
        self.assertEqual(identities, [(1, 1), (1, 1), (1, 1)])

    def test_empty_descriptive_prompt_retries_normalized_object_id(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=False,
            observation_preprocess_objects=(
                {
                    "object_id": "lid",
                    "text_prompt": "wooden lid or box cover",
                    "role": "tool",
                },
            ),
        )
        agent = ImgAgent(make_card(config))
        prompts: list[str] = []

        async def fake_segment_object(**kwargs):
            prompt = str(kwargs["text_prompt"])
            prompts.append(prompt)
            if prompt != "lid":
                return json.dumps(
                    {
                        "success": True,
                        "object_id": "lid",
                        "text_prompt": prompt,
                        "num_detections": 0,
                        "detections": [],
                        "backend": "sam3",
                        "camera": "head",
                    }
                )
            return json.dumps(
                {
                    "success": True,
                    "object_id": "lid",
                    "text_prompt": prompt,
                    "num_detections": 1,
                    "detections": [
                        {
                            "rank": 0,
                            "bbox_xyxy": [1, 2, 3, 4],
                            "centroid_px": [2.0, 3.0],
                            "score": 0.9,
                        }
                    ],
                    "backend": "sam3",
                    "camera": "head",
                }
            )

        with mock.patch.object(agent.agent_tools, "segment_object", fake_segment_object):
            agent.update_snapshot(make_snapshot())

        segment = agent.memory_store.state.working.observation_preprocess["segmentation"][0]
        self.assertEqual(prompts, ["wooden lid or box cover", "lid"])
        self.assertEqual(segment["num_detections"], 1)
        self.assertTrue(segment["text_prompt_fallback_attempted"])
        self.assertTrue(segment["text_prompt_fallback_used"])
        self.assertEqual(segment["requested_text_prompt"], "wooden lid or box cover")
        self.assertEqual(segment["fallback_text_prompt"], "lid")

    def test_grounding_enabled_adds_3d_summary(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=False,
            observation_preprocess_objects=({"object_id": "block", "text_prompt": "block", "role": "target"},),
            observation_grounding_enabled=True,
        )
        agent = ImgAgent(make_card(config))
        tool_result = {
            "success": True,
            "object_id": "block",
            "text_prompt": "block",
            "bbox_xyxy": [2, 2, 6, 6],
            "centroid_px": [4.0, 4.0],
            "score": 0.9,
            "backend": "sam3",
            "camera": "head",
            "env_step": 0,
        }

        async def fake_segment_object(**kwargs):
            return json.dumps(tool_result)

        with mock.patch.object(agent.agent_tools, "segment_object", fake_segment_object):
            agent.update_snapshot(make_snapshot(include_depth=True))

        preprocess = agent.memory_store.state.working.observation_preprocess
        grounding = preprocess["segmentation"][0]["grounding_3d"]
        self.assertTrue(grounding["success"])
        self.assertIn("centroid_world", grounding)
        self.assertIn("world_cm=", agent.memory_store.state.working.recent_observation_summary)
        self.assertIn("scene_memory", agent.memory_store.state.working.to_dict())
        self.assertIn("scene_memory=task_focus", agent.memory_store.state.working.recent_observation_summary)

    def test_detection_candidates_receive_own_grounding(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=False,
            observation_preprocess_objects=({"object_id": "lid", "text_prompt": "lid", "role": "tool"},),
            observation_grounding_enabled=True,
        )
        agent = ImgAgent(make_card(config))
        tool_result = {
            "success": True,
            "object_id": "lid",
            "text_prompt": "lid",
            "num_detections": 2,
            "backend": "sam3",
            "camera": "head",
            "env_step": 0,
            "detections": [
                {"rank": 0, "bbox_xyxy": [1, 1, 3, 3], "centroid_px": [2.0, 2.0], "score": 0.8},
                {"rank": 1, "bbox_xyxy": [5, 5, 7, 7], "centroid_px": [6.0, 6.0], "score": 0.7},
            ],
        }

        async def fake_segment_object(**kwargs):
            return json.dumps(tool_result)

        with mock.patch.object(agent.agent_tools, "segment_object", fake_segment_object):
            agent.update_snapshot(make_snapshot(include_depth=True))

        preprocess = agent.memory_store.state.working.observation_preprocess
        detections = preprocess["segmentation"][0]["detections"]
        self.assertEqual(len(detections), 2)
        self.assertTrue(detections[0]["grounding_3d"]["success"])
        self.assertTrue(detections[1]["grounding_3d"]["success"])
        self.assertNotEqual(
            detections[0]["grounding_3d"]["centroid_world"],
            detections[1]["grounding_3d"]["centroid_world"],
        )
        scene_instances = agent.memory_store.state.working.scene_memory["instances"]
        self.assertEqual(len(scene_instances), 2)
        self.assertTrue(all(instance["world_m"] is not None for instance in scene_instances))

    def test_grounded_detection_filter_drops_high_workspace_false_positive(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=False,
            observation_preprocess_objects=({"object_id": "lid", "text_prompt": "lid", "role": "tool"},),
            observation_grounding_enabled=True,
        )
        agent = ImgAgent(make_card(config))
        tool_result = {
            "success": True,
            "object_id": "lid",
            "text_prompt": "lid",
            "num_detections": 2,
            "backend": "sam3",
            "camera": "head",
            "env_step": 0,
            "detections": [
                {"rank": 0, "bbox_xyxy": [1, 1, 4, 4], "centroid_px": [2.5, 2.5], "score": 0.8},
                {"rank": 1, "bbox_xyxy": [2, 2, 6, 6], "centroid_px": [4.0, 4.0], "score": 0.7},
            ],
        }

        snapshot = make_snapshot(include_depth=False)

        async def fake_segment_object(**kwargs):
            return json.dumps(tool_result)

        def fake_grounding(*, snapshot, result, camera):
            bbox = result.get("bbox_xyxy")
            if bbox == [1, 1, 4, 4]:
                return {
                    "success": True,
                    "object_id": result.get("object_id"),
                    "camera": camera,
                    "centroid_world": [-0.049, -0.018, 1.003],
                    "bbox_world_min": [-0.06, -0.03, 0.99],
                    "bbox_world_max": [-0.04, -0.01, 1.014],
                    "top_surface_world": [-0.049, -0.018, 1.014],
                    "approach_point_world": [-0.049, -0.018, 1.094],
                }
            return {
                "success": True,
                "object_id": result.get("object_id"),
                "camera": camera,
                "centroid_world": [-0.2, -0.18, 0.77],
                "bbox_world_min": [-0.22, -0.2, 0.74],
                "bbox_world_max": [-0.18, -0.16, 0.78],
                "top_surface_world": [-0.2, -0.18, 0.78],
                "approach_point_world": [-0.2, -0.18, 0.86],
            }

        with mock.patch.object(agent.agent_tools, "segment_object", fake_segment_object), mock.patch.object(
            agent, "_ground_segmentation_result", side_effect=fake_grounding
        ):
            agent.update_snapshot(snapshot)

        segment = agent.memory_store.state.working.observation_preprocess["segmentation"][0]
        self.assertEqual(segment["bbox_xyxy"], [2, 2, 6, 6])
        self.assertEqual(segment["num_detections"], 1)
        self.assertEqual(segment["detections"][0]["rank"], 1)
        self.assertEqual(segment["dropped_detections"][0]["bbox_xyxy"], [1, 1, 4, 4])
        warnings = segment["dropped_detections"][0]["quality"]["warnings"]
        self.assertTrue(any("world_z_above_workspace" in item for item in warnings))
        scene_instances = agent.memory_store.state.working.scene_memory["instances"]
        self.assertEqual(len(scene_instances), 1)
        self.assertLess(scene_instances[0]["top_surface_world_m"][2], 0.95)

    def test_grounding_enabled_without_depth_does_not_crash(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=False,
            observation_preprocess_objects=({"object_id": "block", "text_prompt": "block", "role": "target"},),
            observation_grounding_enabled=True,
        )
        agent = ImgAgent(make_card(config))
        tool_result = {
            "success": True,
            "object_id": "block",
            "text_prompt": "block",
            "bbox_xyxy": [2, 2, 6, 6],
            "centroid_px": [4.0, 4.0],
            "score": 0.9,
            "backend": "sam3",
            "camera": "head",
            "env_step": 0,
        }

        async def fake_segment_object(**kwargs):
            return json.dumps(tool_result)

        with mock.patch.object(agent.agent_tools, "segment_object", fake_segment_object):
            agent.update_snapshot(make_snapshot(include_depth=False))

        grounding = agent.memory_store.state.working.observation_preprocess["segmentation"][0]["grounding_3d"]
        self.assertFalse(grounding["success"])
        self.assertIn("depth", grounding["error"])

    def test_multi_camera_preprocess_runs_wrist_views_and_records_camera(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=False,
            observation_preprocess_objects=({"object_id": "block", "text_prompt": "block", "role": "target"},),
            observation_preprocess_cameras=("head", "left", "right"),
        )
        agent = ImgAgent(make_card(config))
        called_cameras: list[str] = []

        async def fake_segment_object(**kwargs):
            camera = kwargs["camera"]
            called_cameras.append(camera)
            return json.dumps(
                {
                    "success": True,
                    "object_id": "block",
                    "text_prompt": "block",
                    "bbox_xyxy": [1, 2, 3, 4],
                    "centroid_px": [2.0, 3.0],
                    "score": 0.9,
                    "backend": "sam3",
                    "camera": camera,
                    "env_step": 0,
                }
            )

        with mock.patch.object(agent.agent_tools, "segment_object", fake_segment_object):
            agent.update_snapshot(make_snapshot())

        self.assertEqual(called_cameras, ["head", "left", "right"])
        segments = agent.memory_store.state.working.observation_preprocess["segmentation"]
        self.assertEqual([item["camera"] for item in segments], ["head", "left", "right"])
        summary = agent.memory_store.state.working.recent_observation_summary
        self.assertIn("block@head:count=1", summary)
        self.assertIn("block@left:count=1", summary)
        self.assertIn("block@right:count=1", summary)

    def test_multi_camera_grounding_uses_each_result_camera(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=False,
            observation_preprocess_objects=({"object_id": "block", "text_prompt": "block", "role": "target"},),
            observation_preprocess_cameras=("left", "right"),
            observation_grounding_enabled=True,
            # The launcher's legacy fallback is head-only.  Camera provenance
            # on each segmentation result must take precedence, otherwise a
            # wrist/third-view mask would be projected with head-camera depth.
            observation_grounding_cameras=("head",),
        )
        agent = ImgAgent(make_card(config))

        async def fake_segment_object(**kwargs):
            camera = kwargs["camera"]
            return json.dumps(
                {
                    "success": True,
                    "object_id": "block",
                    "text_prompt": "block",
                    "bbox_xyxy": [2, 2, 6, 6],
                    "centroid_px": [4.0, 4.0],
                    "score": 0.9,
                    "backend": "sam3",
                    "camera": camera,
                    "env_step": 0,
                }
            )

        with mock.patch.object(agent.agent_tools, "segment_object", fake_segment_object):
            agent.update_snapshot(make_snapshot(include_depth=True))

        segments = agent.memory_store.state.working.observation_preprocess["segmentation"]
        self.assertEqual([item["grounding_3d"]["camera"] for item in segments], ["left", "right"])
        self.assertGreater(segments[0]["grounding_3d"]["centroid_world"][0], 0.9)
        self.assertGreater(segments[1]["grounding_3d"]["centroid_world"][0], 1.9)

    def test_calibrated_third_view_always_participates(self) -> None:
        config = DummyConfig(
            observation_preprocess_cameras=("head",),
            observation_verification_cameras=("third",),
        )
        agent = ImgAgent(make_card(config))
        agent.set_instruction("move the selected object")
        snapshot = make_snapshot(include_depth=True)
        snapshot.third_rgb = np.zeros(
            (8, 8, 3),
            dtype=np.uint8,
        )
        snapshot.third_depth = np.full(
            (8, 8),
            1000.0,
            dtype=np.float64,
        )
        snapshot.third_intrinsic_cv = (
            snapshot.head_intrinsic_cv
        )
        snapshot.third_cam2world_gl = np.eye(
            4,
            dtype=np.float64,
        )
        agent.latest_snapshot = snapshot

        self.assertEqual(
            agent._observation_preprocess_cameras(),
            ("head", "third"),
        )

        agent.memory_store.state.working.manipulation_state = {
            "right": {
                "phase": "grasp_candidate",
                "held_instance_id": "track_0001",
            }
        }
        self.assertEqual(
            agent._observation_preprocess_cameras(),
            ("head", "third"),
        )

        snapshot.third_depth = None
        self.assertEqual(
            agent._observation_preprocess_cameras(),
            ("head",),
        )

    def test_robot_gripper_query_uses_robot_state_not_scene_binding(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        prepared = agent._prepare_agent_identity_bindings(
            [
                {
                    "object_id": "right_gripper",
                    "text_prompt": "right robot gripper",
                    "role": "tool",
                    "instance_hint": "clear of the released cube",
                },
                {
                    "object_id": "button",
                    "text_prompt": "red button",
                    "role": "target",
                },
            ],
            scene_instances=[],
        )

        self.assertEqual(
            prepared[0]["entity_scope"],
            "robot_state",
        )
        self.assertFalse(
            prepared[0]["identity_binding_required"]
        )
        self.assertNotIn(
            "identity_binding_error",
            prepared[0],
        )
        self.assertTrue(
            prepared[1]["identity_binding_required"]
        )
        self.assertEqual(
            prepared[1]["identity_binding_error"],
            "agent_did_not_return_instance_reference",
        )

    def test_no_fallback_object_queries_when_vlm_query_unavailable(self) -> None:
        agent = ImgAgent(DummyAgentCard())

        async def fake_segment_object(**kwargs):
            raise AssertionError("segmentation should not run without explicit or VLM perception queries")

        with mock.patch.object(agent.agent_tools, "segment_object", fake_segment_object):
            agent.update_snapshot(make_snapshot())

        self.assertEqual(agent.memory_store.state.working.observation_preprocess, {})

    def test_perception_query_normalization_can_be_delegated_to_api(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=False,
            observation_preprocess_normalization_url="http://127.0.0.1:9101/normalize_perception_queries",
            observation_preprocess_objects=(
                {"object_id": "left brown lid", "text_prompt": "brown lid", "role": "tool", "reason": "raw query"},
            ),
        )
        agent = ImgAgent(make_card(config))
        tool_result = {
            "success": True,
            "object_id": "lid",
            "text_prompt": "brown lid",
            "bbox_xyxy": [1, 2, 3, 4],
            "centroid_px": [2.0, 3.0],
            "score": 0.9,
            "backend": "sam3",
            "camera": "head",
            "env_step": 0,
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "queries": [
                            {
                                "object_id": "lid",
                                "text_prompt": "brown lid",
                                "role": "tool",
                                "instance_hint": "left brown",
                                "reason": "normalized by api",
                            }
                        ]
                    }
                ).encode("utf-8")

        async def fake_segment_object(**kwargs):
            self.assertEqual(kwargs["object_id"], "lid")
            self.assertEqual(kwargs["text_prompt"], "brown lid")
            return json.dumps(tool_result)

        with mock.patch("policy.roboharn_evo.agent.core.img_agent.request.urlopen", return_value=FakeResponse()), mock.patch.object(
            agent.agent_tools, "segment_object", fake_segment_object
        ):
            agent.update_snapshot(make_snapshot())

        segment = agent.memory_store.state.working.observation_preprocess["segmentation"][0]
        self.assertEqual(segment["object_id"], "lid")
        self.assertEqual(segment["query_instance_hint"], "left brown")
        self.assertEqual(agent.memory_store.state.working.scene_memory["instances"][0]["class"], "lid")

    def test_oracle_preprocess_uses_simulator_catalog_without_segmentation_tool(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=False,
            observation_preprocess_objects=({"object_id": "block", "text_prompt": "block", "role": "target"},),
            oracle_objects_enabled=True,
        )
        agent = ImgAgent(make_card(config))
        snapshot = make_snapshot()
        snapshot.oracle_objects = [
            {
                "oracle_id": "blocks_0",
                "class_name": "block",
                "source_path": "blocks[0]",
                "actor_name": "box",
                "aliases": ["blocks", "block", "box"],
                "position_world": [-0.2, -0.18, 0.76],
                "grounding_3d": {
                    "success": True,
                    "object_id": "block",
                    "camera": "oracle",
                    "centroid_world": [-0.2, -0.18, 0.76],
                    "bbox_world_min": [-0.22, -0.2, 0.74],
                    "bbox_world_max": [-0.18, -0.16, 0.78],
                    "top_surface_world": [-0.2, -0.18, 0.78],
                    "approach_point_world": [-0.2, -0.18, 0.86],
                },
            }
        ]

        async def fake_segment_object(**kwargs):
            raise AssertionError("oracle preprocess should not call SAM segmentation")

        with mock.patch.object(agent.agent_tools, "segment_object", fake_segment_object):
            agent.update_snapshot(snapshot)

        preprocess = agent.memory_store.state.working.observation_preprocess
        self.assertEqual(preprocess["source"], "oracle_simulator")
        self.assertEqual(preprocess["segmentation"][0]["backend"], "oracle_simulator")
        self.assertEqual(preprocess["segmentation"][0]["query_role"], "target")
        scene_memory = agent.memory_store.state.working.scene_memory
        self.assertEqual(scene_memory["task_focus"]["target_instances"], ["track_0001"])
        self.assertEqual(scene_memory["instances"][0]["world_m"], [-0.2, -0.18, 0.76])

    def test_no_oracle_perception_query_payload_omits_simulator_catalog(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=True,
            observation_preprocess_query_url="http://127.0.0.1:9/perception_queries",
            oracle_objects_enabled=False,
        )
        agent = ImgAgent(make_card(config))
        snapshot = make_snapshot()
        snapshot.oracle_objects = [
            {
                "oracle_id": "blocks_0",
                "class_name": "block",
                "source_path": "blocks[0]",
                "aliases": ["block"],
                "position_world": [-0.2, -0.18, 0.76],
            }
        ]
        agent.latest_snapshot = snapshot
        agent.memory_store.record_scene_memory(
            {
                "instances": [
                    {
                        "instance_id": "component_01",
                        "track_id": "track_alpha",
                        "class": "component",
                        "class_aliases": ["component"],
                        "status": "visible",
                        "actionable": True,
                        "camera": "head",
                        "bbox_xyxy": [1, 1, 4, 4],
                        "world_m": [0.11, -0.03, 0.82],
                    }
                ]
            }
        )

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "queries": [
                            {
                                "object_id": "block",
                                "text_prompt": "block",
                                "role": "target",
                            }
                        ]
                    }
                ).encode("utf-8")

        captured_payloads = []

        def fake_urlopen(request_obj, timeout=0):
            captured_payloads.append(json.loads(request_obj.data.decode("utf-8")))
            return FakeResponse()

        with mock.patch("policy.roboharn_evo.agent.core.img_agent.request.urlopen", side_effect=fake_urlopen):
            queries = agent._build_perception_queries("scene summary")

        self.assertEqual(queries[0]["object_id"], "block")
        self.assertEqual(captured_payloads[0]["oracle_objects"], [])
        self.assertEqual(captured_payloads[0]["scene_instances"][0]["track_id"], "track_alpha")
        self.assertFalse(captured_payloads[0]["require_instance_binding"])
        self.assertEqual(
            captured_payloads[0]["instance_binding_phase"],
            "candidate_discovery",
        )
        self.assertEqual(
            captured_payloads[0]["binding_requirement"][
                "required_any_roles"
            ],
            ["target", "tool"],
        )

    def test_no_oracle_new_subtask_object_survives_unrelated_track_catalog(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=True,
            observation_preprocess_query_url=(
                "http://127.0.0.1:9104/perception_queries"
            ),
            observation_preprocess_normalization_url=(
                "http://127.0.0.1:9104/normalize_perception_queries"
            ),
            oracle_objects_enabled=False,
        )
        agent = ImgAgent(make_card(config))
        agent.latest_snapshot = make_snapshot()
        agent.current_instruction = (
            "Manipulate several objects in stages."
        )
        skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
        agent.memory_store.start_or_replace_skill(
            subtask_text=(
                "Rearrange two movable objects that have not been observed yet."
            ),
            skill_spec=skill_spec,
        )
        agent.memory_store.record_scene_memory(
            {
                "instances": [
                    {
                        "instance_id": "track_0001",
                        "track_id": "track_0001",
                        "class": "button",
                        "class_aliases": ["button"],
                        "status": "visible",
                        "stability": "stable",
                        "actionable": True,
                        "camera": "head",
                        "bbox_xyxy": [1, 1, 4, 4],
                        "world_m": [-0.2, -0.1, 0.78],
                    }
                ]
            }
        )
        captured_payloads: list[dict] = []
        discovered_queries = [
            {
                "object_id": "movable_object",
                "text_prompt": "blue movable object",
                "role": "target",
                "instance_hint": "blue",
                "reason": "Required by the current rearrangement subtask.",
            },
            {
                "object_id": "movable_object",
                "text_prompt": "red movable object",
                "role": "tool",
                "instance_hint": "red",
                "reason": "Required by the current rearrangement subtask.",
            },
        ]

        class DiscoveryResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {"queries": discovered_queries}
                ).encode("utf-8")

        def fake_urlopen(request_obj, timeout=0):
            payload = json.loads(request_obj.data.decode("utf-8"))
            captured_payloads.append(payload)
            return DiscoveryResponse()

        with mock.patch(
            "policy.roboharn_evo.agent.core.img_agent.request.urlopen",
            side_effect=fake_urlopen,
        ):
            queries = agent._build_perception_queries("synthetic scene")

        self.assertEqual(len(queries), 2)
        self.assertEqual(
            {query["role"] for query in queries},
            {"target", "tool"},
        )
        self.assertTrue(
            all(not query.get("instance_ref") for query in queries)
        )
        self.assertTrue(
            all(query["identity_binding_required"] for query in queries)
        )
        self.assertEqual(len(captured_payloads), 2)
        for payload in captured_payloads:
            self.assertFalse(payload["require_instance_binding"])
            self.assertEqual(
                payload["instance_binding_phase"],
                "candidate_discovery",
            )

    def test_pending_release_replaces_stale_relational_bound_query(
        self,
    ) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        agent.memory_store.reset(
            task="move a selected object",
            control_model_name="test",
            available_policies=[],
            available_tools=[],
            mode="deployment",
        )
        agent.memory_store.record_scene_memory(
            {
                "instances": [
                    {
                        "instance_id": "track_0002",
                        "track_id": "track_0002",
                        "class": "cube",
                        "class_aliases": ["cube"],
                        "source_object_id": "cube",
                        "source_text_prompt": "red cube",
                        "query_instance_hint": "originally rightmost",
                        "status": "tracked",
                        "stability": "missing_current_frame",
                        "last_seen_step": 4,
                        "world_m": [0.27, -0.10, 0.77],
                    }
                ]
            }
        )
        agent.memory_store.state.working.manipulation_state = {
            "right": {
                "phase": "release_pending_verification",
                "held_instance_id": "track_0002",
                "held_object_target_world_m": [0.10, 0.01, 0.77],
                "held_extent_m": [0.06, 0.06, 0.04],
                "updated_step": 8,
                "held_object_perception_descriptor": {
                    "object_id": "cube",
                    "text_prompt": "red cube",
                    "instance_hint": "originally rightmost",
                },
            }
        }
        queries = agent._with_runtime_manipulation_perception_queries(
            [
                {
                    "object_id": "cube",
                    "text_prompt": "rightmost cube",
                    "role": "target",
                    "instance_ref": "track_0002",
                },
                {
                    "object_id": "cube",
                    "text_prompt": "blue cube",
                    "role": "target",
                    "instance_ref": "track_0003",
                },
            ],
            max_objects=3,
        )

        self.assertEqual(queries[0]["instance_ref"], "track_0002")
        self.assertEqual(queries[0]["text_prompt"], "red cube")
        self.assertEqual(queries[0]["role"], "context")
        self.assertTrue(queries[0]["identity_binding_required"])
        self.assertFalse(
            any(
                query.get("text_prompt") == "rightmost cube"
                for query in queries
            )
        )
        self.assertEqual(queries[1]["instance_ref"], "track_0003")
        leases = agent._pending_release_identity_relocation_leases()
        self.assertEqual(leases[0]["instance_ref"], "track_0002")
        self.assertEqual(
            leases[0]["target_world_m"],
            [0.10, 0.01, 0.77],
        )
        self.assertLess(leases[0]["tolerance_m"], 0.08)
        self.assertTrue(
            leases[0][
                "allow_position_only_action_geometry_quarantine"
            ]
        )

    def test_holding_identity_replaces_semantic_query_with_wrong_ref(
        self,
    ) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        agent.memory_store.reset(
            task="move a selected object",
            control_model_name="test",
            available_policies=[],
            available_tools=[],
            mode="deployment",
        )
        agent.memory_store.state.working.manipulation_state = {
            "right": {
                "phase": "holding_provisional",
                "held_instance_id": "track_0002",
                "held_object_perception_descriptor": {
                    "object_id": "red_cube",
                    "text_prompt": "red cube",
                    "instance_hint": "originally rightmost",
                },
            }
        }

        queries = agent._with_runtime_manipulation_perception_queries(
            [
                {
                    "object_id": "red_cube",
                    "text_prompt": "red cube",
                    "role": "target",
                    "instance_ref": "track_0001",
                },
                {
                    "object_id": "green_cube",
                    "text_prompt": "green cube",
                    "role": "target",
                    "instance_ref": "track_0003",
                },
                {
                    "object_id": "blue_cube",
                    "text_prompt": "blue cube",
                    "role": "target",
                    "instance_ref": "track_0004",
                },
            ],
            max_objects=3,
        )

        self.assertEqual(
            [query["instance_ref"] for query in queries],
            ["track_0002", "track_0003", "track_0004"],
        )
        self.assertEqual(queries[0]["role"], "context")
        self.assertTrue(queries[0]["identity_binding_required"])
        self.assertNotIn(
            "track_0001",
            {query.get("instance_ref") for query in queries},
        )
        self.assertTrue(agent._active_manipulation_identity_states())
        self.assertFalse(agent._pending_release_verification_states())

    def test_holding_identity_does_not_collapse_repeated_generic_objects(
        self,
    ) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        agent.memory_store.reset(
            task="move a selected object",
            control_model_name="test",
            available_policies=[],
            available_tools=[],
            mode="deployment",
        )
        agent.memory_store.state.working.manipulation_state = {
            "right": {
                "phase": "holding",
                "held_instance_id": "track_0002",
                "held_object_perception_descriptor": {
                    "object_id": "cube",
                    "text_prompt": "red cube",
                },
            }
        }

        queries = agent._with_runtime_manipulation_perception_queries(
            [
                {
                    "object_id": "cube",
                    "text_prompt": "red cube",
                    "role": "target",
                    "instance_ref": "track_0001",
                },
                {
                    "object_id": "cube",
                    "text_prompt": "green cube",
                    "role": "target",
                    "instance_ref": "track_0003",
                },
                {
                    "object_id": "cube",
                    "text_prompt": "blue cube",
                    "role": "target",
                    "instance_ref": "track_0004",
                },
            ],
            max_objects=3,
        )

        self.assertEqual(
            [query["instance_ref"] for query in queries],
            ["track_0002", "track_0003", "track_0004"],
        )

    def test_post_detection_binding_remains_strict_after_discovery(self) -> None:
        agent = ImgAgent(
            make_card(
                DummyConfig(
                    observation_preprocess_normalization_url=(
                        "http://127.0.0.1:9104/normalize_perception_queries"
                    )
                )
            )
        )
        scene_memory = {
            "env_step": 7,
            "instances": [
                {
                    "instance_id": "track_0001",
                    "track_id": "track_0001",
                    "class": "button",
                    "class_aliases": ["button"],
                    "status": "visible",
                    "stability": "stable",
                    "actionable": True,
                    "world_m": [-0.2, -0.1, 0.78],
                },
                {
                    "instance_id": "track_0002",
                    "track_id": "track_0002",
                    "class": "movable_object",
                    "class_aliases": ["movable_object"],
                    "status": "visible",
                    "stability": "new",
                    "actionable": True,
                    "world_m": [0.0, -0.1, 0.78],
                },
                {
                    "instance_id": "track_0003",
                    "track_id": "track_0003",
                    "class": "movable_object",
                    "class_aliases": ["movable_object"],
                    "status": "visible",
                    "stability": "new",
                    "actionable": True,
                    "world_m": [0.15, -0.1, 0.78],
                },
            ],
            "task_focus": {},
            "uncertainty": [],
            "temporal_memory": {},
        }
        captured_payloads: list[dict] = []

        class BoundResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "queries": [
                            {
                                "object_id": "movable_object",
                                "text_prompt": "blue movable object",
                                "role": "target",
                                "instance_hint": "blue",
                                "instance_ref": "track_0002",
                            },
                            {
                                "object_id": "movable_object",
                                "text_prompt": "red movable object",
                                "role": "tool",
                                "instance_hint": "red",
                                "instance_ref": "track_0003",
                            },
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(request_obj, timeout=0):
            captured_payloads.append(
                json.loads(request_obj.data.decode("utf-8"))
            )
            return BoundResponse()

        with mock.patch(
            "policy.roboharn_evo.agent.core.img_agent.request.urlopen",
            side_effect=fake_urlopen,
        ):
            rebound, queries = agent._bind_scene_memory_focus_with_agent(
                scene_memory=scene_memory,
                perception_queries=[
                    {
                        "object_id": "movable_object",
                        "text_prompt": "blue movable object",
                        "role": "target",
                        "instance_hint": "blue",
                        "identity_binding_required": True,
                    },
                    {
                        "object_id": "movable_object",
                        "text_prompt": "red movable object",
                        "role": "tool",
                        "instance_hint": "red",
                        "identity_binding_required": True,
                    },
                ],
                observation_summary="synthetic scene",
                current_subtask="rearrange the two movable objects",
            )

        self.assertEqual(
            [query["instance_ref"] for query in queries],
            ["track_0002", "track_0003"],
        )
        self.assertEqual(
            rebound["task_focus"]["target_instances"],
            ["track_0002"],
        )
        self.assertEqual(
            rebound["task_focus"]["tool_instances"],
            ["track_0003"],
        )
        self.assertEqual(
            captured_payloads[0]["instance_binding_phase"],
            "post_detection_selection",
        )
        self.assertTrue(
            captured_payloads[0]["require_instance_binding"]
        )

    def test_fresh_query_receives_rejected_roles_and_retry_feedback(self) -> None:
        config = DummyConfig(
            observation_preprocess_auto_objects=True,
            observation_preprocess_query_url=(
                "http://127.0.0.1:9104/perception_queries"
            ),
            observation_preprocess_normalization_url=(
                "http://127.0.0.1:9104/normalize_perception_queries"
            ),
        )
        agent = ImgAgent(make_card(config))
        agent.latest_snapshot = make_snapshot()
        captured_payloads: list[dict] = []

        class ActionableResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "queries": [
                            {
                                "object_id": "movable_object",
                                "text_prompt": "movable object",
                                "role": "target",
                            }
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(request_obj, timeout=0):
            captured_payloads.append(
                json.loads(request_obj.data.decode("utf-8"))
            )
            return ActionableResponse()

        requirement = agent._perception_binding_requirement(
            allow_context_binding=False,
            retry_context={
                "failure_reason": (
                    "agent_returned_no_target_or_tool_binding"
                ),
                "retry_attempt": 1,
                "force_refresh": True,
                "previous_queries": [
                    {
                        "object_id": "reference_object",
                        "text_prompt": "reference object",
                        "role": "context",
                    }
                ],
            },
        )
        with mock.patch(
            "policy.roboharn_evo.agent.core.img_agent.request.urlopen",
            side_effect=fake_urlopen,
        ):
            queries = agent._build_perception_queries(
                "scene summary",
                binding_requirement=requirement,
            )

        self.assertEqual(queries[0]["role"], "target")
        self.assertGreaterEqual(len(captured_payloads), 2)
        for payload in captured_payloads[:2]:
            retry = payload["binding_requirement"]
            self.assertTrue(retry["force_refresh"])
            self.assertEqual(retry["retry_attempt"], 1)
            self.assertEqual(
                retry["previous_queries"][0]["role"],
                "context",
            )

    def test_live_snapshot_preprocess_cache_is_scoped_to_active_skill(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        snapshot = make_snapshot()

        with mock.patch.object(agent, "preprocess_observation", return_value={}) as preprocess:
            agent.update_snapshot(snapshot)
            agent.update_snapshot(snapshot)
            self.assertEqual(preprocess.call_count, 1)

            skill_spec = agent._resolve_skill_spec("monitored-subtask-execution")
            agent.memory_store.start_or_replace_skill(
                subtask_text="perform one grounded manipulation step",
                skill_spec=skill_spec,
            )
            agent.update_snapshot(snapshot)
            agent.update_snapshot(snapshot)

        self.assertEqual(preprocess.call_count, 2)

    def test_explicit_reobserve_can_force_fresh_preprocess_sample(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        snapshot = make_snapshot()

        with mock.patch.object(
            agent,
            "preprocess_observation",
            return_value={},
        ) as preprocess:
            agent.update_snapshot(snapshot)
            agent.update_snapshot(snapshot)
            agent.update_snapshot(
                snapshot,
                force_preprocess=True,
            )

        self.assertEqual(preprocess.call_count, 2)

    def test_observation_capture_id_changes_only_for_a_new_snapshot(self) -> None:
        agent = ImgAgent(make_card(DummyConfig()))
        snapshot = make_snapshot()

        with mock.patch.object(
            agent,
            "preprocess_observation",
            return_value={"segmentation": []},
        ):
            agent.update_snapshot(snapshot)
            self.assertEqual(agent._observation_capture_generation, 1)
            self.assertEqual(agent._observation_preprocess_generation, 1)

            agent.update_snapshot(snapshot, force_preprocess=True)
            self.assertEqual(agent._observation_capture_generation, 1)
            self.assertEqual(agent._observation_preprocess_generation, 2)

            agent.update_snapshot(
                make_snapshot(),
                force_preprocess=True,
            )
            self.assertEqual(agent._observation_capture_generation, 2)
            self.assertEqual(agent._observation_preprocess_generation, 3)


if __name__ == "__main__":
    unittest.main()
