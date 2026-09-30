from __future__ import annotations

import unittest

import numpy as np

from policy.roboharn_evo.agent.perception.grounding import ground_segmentation_result


class RGBDGroundingTest(unittest.TestCase):
    def test_bbox_depth_projects_to_world(self) -> None:
        depth = np.full((4, 4), 1000.0, dtype=np.float64)
        intrinsic = np.array(
            [
                [100.0, 0.0, 1.5],
                [0.0, 100.0, 1.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cam2world = np.eye(4, dtype=np.float64)
        result = ground_segmentation_result(
            segmentation={"success": True, "object_id": "block", "bbox_xyxy": [1, 1, 3, 3]},
            depth_mm=depth,
            intrinsic_cv=intrinsic,
            cam2world_gl=cam2world,
            camera="head",
            min_valid_ratio=0.1,
            approach_height_m=0.08,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["mask_pixel_count"], 4)
        self.assertEqual(result["valid_pixel_count"], 4)
        self.assertEqual(result["centroid_world"], [0.0, 0.0, -1.0])
        self.assertEqual(result["top_surface_world"], [0.0, 0.0, -1.0])
        self.assertEqual(result["surface_normal_world"], [0.0, 0.0, 1.0])
        self.assertEqual(result["approach_point_world"], [0.0, 0.0, -0.92])

    def test_surface_contact_pose_converts_tcp_contact_to_ee_target(self) -> None:
        depth = np.full((4, 4), 1000.0, dtype=np.float64)
        intrinsic = np.array(
            [
                [100.0, 0.0, 1.5],
                [0.0, 100.0, 1.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        result = ground_segmentation_result(
            segmentation={"success": True, "object_id": "surface", "bbox_xyxy": [1, 1, 3, 3]},
            depth_mm=depth,
            intrinsic_cv=intrinsic,
            cam2world_gl=np.eye(4, dtype=np.float64),
            camera="head",
            approach_height_m=0.08,
            ee_to_contact_m=0.12,
        )

        self.assertEqual(result["object_contact_point_world"], [0.0, 0.0, -1.0])
        self.assertEqual(result["contact_point_world"], [0.0, 0.0, -0.88])
        self.assertEqual(result["approach_point_world"], [0.0, 0.0, -0.8])
        np.testing.assert_allclose(
            result["contact_pose_world"][3:7],
            [2**-0.5, 0.0, 2**-0.5, 0.0],
            atol=1e-6,
        )
        self.assertEqual(result["contact_geometry_source"], "rgbd_surface_normal_and_robot_tcp")

    def test_surface_contact_uses_full_mask_lateral_center(self) -> None:
        depth = np.full((6, 6), 1000.0, dtype=np.float64)
        depth[1, 1:5] = 900.0
        intrinsic = np.array(
            [
                [100.0, 0.0, 2.5],
                [0.0, 100.0, 2.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        result = ground_segmentation_result(
            segmentation={"success": True, "object_id": "surface", "bbox_xyxy": [1, 1, 5, 5]},
            depth_mm=depth,
            intrinsic_cv=intrinsic,
            cam2world_gl=np.eye(4, dtype=np.float64),
            camera="head",
        )

        self.assertEqual(result["top_surface_world"][:2], result["centroid_world"][:2])
        self.assertGreater(result["top_surface_world"][2], result["centroid_world"][2])

    def test_dominant_plane_footprint_ignores_small_raised_patch(self) -> None:
        depth = np.full((20, 20), 1000.0, dtype=np.float64)
        depth[7:13, 7:13] = 960.0
        intrinsic = np.array(
            [
                [100.0, 0.0, 9.5],
                [0.0, 100.0, 9.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        result = ground_segmentation_result(
            segmentation={
                "success": True,
                "object_id": "reference surface",
                "bbox_xyxy": [0, 0, 20, 20],
            },
            depth_mm=depth,
            intrinsic_cv=intrinsic,
            cam2world_gl=np.eye(4, dtype=np.float64),
            camera="third",
        )

        self.assertAlmostEqual(
            result["bbox_world_max"][2]
            - result["bbox_world_min"][2],
            0.04,
            places=6,
        )
        plane = result["dominant_plane_footprint"]
        self.assertTrue(plane["valid"])
        self.assertEqual(plane["source"], "dominant_horizontal_z_band")
        self.assertEqual(plane["inlier_count"], 364)
        self.assertAlmostEqual(plane["inlier_ratio"], 0.91, places=6)
        self.assertEqual(plane["extent_m"][2], 0.0)
        self.assertGreater(plane["extent_m"][0], 0.18)
        self.assertGreater(plane["extent_m"][1], 0.18)

    def test_rgbd_surface_generates_principal_axis_candidates_for_both_arms(self) -> None:
        depth = np.full((6, 8), 1000.0, dtype=np.float64)
        # A closer upper face plus the farther visible side/bottom band gives
        # the RGB-D mask a measured extent along the outward surface normal.
        depth[2:4, 2:6] = 900.0
        intrinsic = np.array(
            [
                [100.0, 0.0, 3.5],
                [0.0, 100.0, 2.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        calibrations = {
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
        result = ground_segmentation_result(
            segmentation={
                "success": True,
                "object_id": "object",
                "bbox_xyxy": [1, 1, 7, 5],
            },
            depth_mm=depth,
            intrinsic_cv=intrinsic,
            cam2world_gl=np.eye(4, dtype=np.float64),
            camera="head",
            tcp_calibration_by_arm=calibrations,
        )

        candidates = result["operation_pose_candidates"]
        self.assertEqual(len(candidates), 8)
        self.assertEqual({item["arm"] for item in candidates}, {"left", "right"})
        self.assertEqual({item["action_mode"] for item in candidates}, {"grasp", "contact"})
        self.assertEqual({item["source_candidate_index"] for item in candidates}, {0, 1})
        self.assertEqual(
            {item["geometry_source"] for item in candidates},
            {
                "rgbd_surface_normal_principal_axes",
                "rgbd_observed_volume_principal_axes",
            },
        )
        top_z = result["top_surface_world"][2]
        lower_z = result["grasp_geometry"]["lower_boundary_world"][2]
        self.assertTrue(result["grasp_geometry"]["valid"])
        object_contact_z = result["grasp_geometry"]["object_contact_world"][2]
        self.assertLess(lower_z, object_contact_z)
        self.assertLess(object_contact_z, top_z)
        self.assertEqual(
            result["grasp_geometry"]["tcp_target_world"],
            result["top_surface_world"],
        )
        for candidate in candidates:
            self.assertEqual(candidate["approach_direction"], [0.0, 0.0, -1.0])
            np.testing.assert_allclose(
                candidate["approach_pose"][:3],
                [
                    candidate["ee_target_pose"][0],
                    candidate["ee_target_pose"][1],
                    top_z + 0.08 + 0.12,
                ],
                atol=1e-6,
            )
            if candidate["action_mode"] == "contact":
                np.testing.assert_allclose(
                    candidate["tcp_pose"][:3],
                    result["top_surface_world"],
                    atol=1e-6,
                )
                self.assertEqual(candidate["grasp_clearance_m"], 0.0)
            else:
                np.testing.assert_allclose(
                    candidate["tcp_pose"][:3],
                    result["grasp_geometry"]["tcp_target_world"],
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    candidate["object_contact_pose"][:3],
                    result["grasp_geometry"]["object_contact_world"],
                    atol=1e-6,
                )
                self.assertGreater(candidate["grasp_depth_m"], 0.0)
                self.assertGreater(candidate["grasp_clearance_m"], 0.0)

    def test_degenerate_single_surface_emits_contact_but_no_unsafe_grasp(self) -> None:
        depth = np.full((6, 8), 1000.0, dtype=np.float64)
        intrinsic = np.array(
            [
                [100.0, 0.0, 3.5],
                [0.0, 100.0, 2.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        result = ground_segmentation_result(
            segmentation={
                "success": True,
                "object_id": "flat_surface",
                "bbox_xyxy": [1, 1, 7, 5],
            },
            depth_mm=depth,
            intrinsic_cv=intrinsic,
            cam2world_gl=np.eye(4, dtype=np.float64),
            camera="head",
        )

        self.assertFalse(result["grasp_geometry"]["valid"])
        candidates = result["operation_pose_candidates"]
        self.assertEqual(len(candidates), 4)
        self.assertEqual({item["action_mode"] for item in candidates}, {"contact"})

    def test_missing_depth_returns_failure(self) -> None:
        intrinsic = np.eye(3, dtype=np.float64)
        cam2world = np.eye(4, dtype=np.float64)
        result = ground_segmentation_result(
            segmentation={"success": True, "object_id": "block", "bbox_xyxy": [1, 1, 3, 3]},
            depth_mm=None,
            intrinsic_cv=intrinsic,
            cam2world_gl=cam2world,
            camera="head",
        )

        self.assertFalse(result["success"])
        self.assertIn("depth", result["error"])


if __name__ == "__main__":
    unittest.main()
