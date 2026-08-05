from __future__ import annotations

import unittest

from workers.profile_discovery import discover_profile


class ProfileDiscoveryTests(unittest.TestCase):
    def test_boat_avoids_character_routing(self):
        result = discover_profile("lucky_drown_casino_boat", observed_tokens=["paddlewheel", "railings", "windows"])
        self.assertEqual(result["base_profile"], "vehicle")
        self.assertEqual(result["selected_safe_strategy"]["rig"], "none")
        self.assertFalse(result["geometry_modified"])

    def test_barn_is_static_building(self):
        result = discover_profile("barn_with_trees_static_scene", observed_tokens=["barn", "trees", "outdoor", "static"])
        self.assertEqual(result["base_profile"], "building")
        self.assertIn("natural_vegetation", result["traits"])
        self.assertEqual(result["selected_safe_strategy"]["rig"], "none")

    def test_turtle_is_quadruped_without_a_pose(self):
        result = discover_profile("steampunk_snapping_turtle", observed_tokens=["shell", "tail", "four_limbs", "jaw"])
        self.assertEqual(result["base_profile"], "quadruped")
        self.assertIn("shell", result["traits"])

    def test_equipped_frog_is_humanoid_with_equipment_trait(self):
        result = discover_profile("frog_salvage_diver", observed_tokens=["backpack", "tank", "hoses", "lantern"])
        self.assertEqual(result["base_profile"], "humanoid")
        self.assertIn("attached_equipment", result["traits"])

    def test_weak_evidence_falls_back_and_requests_review(self):
        result = discover_profile("unnamed_asset")
        self.assertEqual(result["base_profile"], "unknown")
        self.assertTrue(result["user_input_required"])
        self.assertFalse(result["geometry_modified"])


if __name__ == "__main__":
    unittest.main()
