from __future__ import annotations

import unittest

from lowvram3d.pose_policy import (
    APoseSpecification,
    PoseMode,
    resolve_pose_contract,
    validate_pose_measurements,
)


class PosePolicyTests(unittest.TestCase):
    def test_humanoid_defaults_to_a_pose_and_keeps_source(self):
        contract = resolve_pose_contract("character", generate_rig=True)
        self.assertEqual(contract.source_pose, PoseMode.PRESERVE_SOURCE)
        self.assertEqual(contract.bind_pose, PoseMode.A_POSE)
        self.assertEqual(contract.alternate_pose, PoseMode.T_POSE)
        self.assertEqual(contract.retarget_pose, "ue5_mannequin_a_pose")
        self.assertTrue(contract.compare_before_normalization)
        self.assertEqual(contract.source_output_name, "geometry/source_pose.glb")
        self.assertEqual(contract.bind_output_name, "rigged/bind_a_pose.glb")

    def test_master_comparison_never_normalizes_pose(self):
        contract = resolve_pose_contract(
            "character",
            generate_rig=True,
            benchmark_compare=True,
        )
        self.assertEqual(contract.bind_pose, PoseMode.PRESERVE_SOURCE)

    def test_quadruped_and_static_defaults(self):
        self.assertEqual(
            resolve_pose_contract("quadruped", generate_rig=True).bind_pose,
            PoseMode.NEUTRAL_QUADRUPED,
        )
        self.assertEqual(
            resolve_pose_contract("building", generate_rig=False).bind_pose,
            PoseMode.PRESERVE_SOURCE,
        )

    def test_explicit_t_pose_is_supported_but_not_default(self):
        contract = resolve_pose_contract("character", generate_rig=True, requested="t_pose")
        self.assertEqual(contract.bind_pose, PoseMode.T_POSE)
        self.assertIsNone(contract.alternate_pose)

    def test_valid_a_pose_measurements_pass(self):
        contract = resolve_pose_contract("character", generate_rig=True)
        measurements = {
            "expected_limbs_present": True,
            "left_right_bone_pairs_complete": True,
            "skeleton_hierarchy_valid": True,
            "skin_weight_coverage_complete": True,
            "no_visible_unweighted_vertices": True,
            "no_new_self_intersections": True,
            "equipment_assignment_valid": True,
            "source_pose_artifact_present": True,
            "arm_from_torso_degrees": [40.0, 40.0],
            "minimum_hand_torso_clearance_fraction": 0.05,
            "maximum_foot_floor_error_fraction": 0.005,
            "maximum_vertex_deformation_fraction": 0.15,
        }
        self.assertTrue(validate_pose_measurements(measurements, contract).valid)

    def test_missing_limb_or_intersection_fails_closed(self):
        contract = resolve_pose_contract("character", generate_rig=True)
        result = validate_pose_measurements(
            {
                "expected_limbs_present": False,
                "left_right_bone_pairs_complete": True,
                "skeleton_hierarchy_valid": True,
                "skin_weight_coverage_complete": True,
                "no_visible_unweighted_vertices": True,
                "no_new_self_intersections": False,
                "equipment_assignment_valid": True,
                "source_pose_artifact_present": True,
                "arm_from_torso_degrees": [40.0, 40.0],
                "minimum_hand_torso_clearance_fraction": 0.05,
                "maximum_foot_floor_error_fraction": 0.0,
                "maximum_vertex_deformation_fraction": 0.1,
            },
            contract,
        )
        self.assertFalse(result.valid)
        self.assertIn("expected_limbs_present", result.errors)
        self.assertIn("no_new_self_intersections", result.errors)

    def test_a_pose_specification_is_stable(self):
        spec = APoseSpecification()
        self.assertEqual(spec.arm_from_torso_degrees, 40.0)
        self.assertEqual(spec.elbow_bend_degrees, 8.0)


if __name__ == "__main__":
    unittest.main()
