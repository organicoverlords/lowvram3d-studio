from __future__ import annotations

import unittest

from lowvram3d.presets import AssetType, get_profile, infer_asset_type


class PresetTests(unittest.TestCase):

    def test_avatar_is_identity_preserving_animated_humanoid(self):
        profile = get_profile("avatar", "hero")
        self.assertTrue(profile.preserve_continuous_body)
        self.assertTrue(profile.generate_rig)
        self.assertFalse(profile.rigid_rig)
        self.assertEqual(profile.export_strategy, "animated_human_avatar")
        self.assertEqual(profile.texture_strategy, "identity_preserving_bake")

    def test_character_keeps_body_continuous(self):
        profile = get_profile("character", "gameplay")
        self.assertTrue(profile.preserve_continuous_body)
        self.assertFalse(profile.rigid_rig)
        self.assertTrue(profile.generate_rig)

    def test_vehicle_uses_rigid_parts(self):
        profile = get_profile("vehicle", "hero")
        self.assertTrue(profile.detect_round_parts)
        self.assertTrue(profile.rigid_rig)
        self.assertEqual(profile.target_triangles, 70_000)

    def test_world_profiles_chunk(self):
        self.assertTrue(get_profile("scene").spatial_chunking)
        self.assertTrue(get_profile("level").spatial_chunking)
        self.assertEqual(get_profile("level").atlas_mode, "preserve_or_per_object")

    def test_scene_uses_per_object_budget_and_preserves_materials(self):
        profile = get_profile("scene", "gameplay")
        self.assertEqual(profile.budget_mode, "per_object")
        self.assertEqual(profile.per_object_target, 15_000)
        self.assertEqual(profile.texture_strategy, "preserve_existing")
        self.assertEqual(profile.cell_divisions, 4)

    def test_prop_studio_settings_are_cpu_bounded(self):
        profile = get_profile("prop", "gameplay")
        self.assertEqual(profile.studio_retopo_mode, "single_object")
        self.assertEqual(profile.retopo_options["device"], "cpu")
        self.assertLessEqual(profile.retopo_options["max_memory_gb"], 3.0)
        self.assertEqual(profile.uv_options["resolution"], 2048)
        self.assertEqual(profile.uv_options["padding_texels"], 8)

    def test_auto_inference(self):
        self.assertEqual(infer_asset_type("armoured forest beast"), AssetType.CREATURE)
        self.assertEqual(infer_asset_type("open-plan hotel interior"), AssetType.ROOM)
        self.assertEqual(infer_asset_type("make a dancing version of me"), AssetType.AVATAR)


if __name__ == "__main__":
    unittest.main()
