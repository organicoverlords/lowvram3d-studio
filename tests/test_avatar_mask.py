from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from lowvram3d.avatar_mask import (
    decontaminate_edges,
    framing_report,
    keep_largest_component,
    normalize_subject,
    refine_alpha,
)
from workers.make_fallback_views import foreground_rgba


class AvatarMaskTests(unittest.TestCase):
    def test_largest_subject_removes_secondary_person_blob(self):
        alpha = np.zeros((160, 200), dtype=np.float32)
        alpha[20:145, 20:95] = 1.0
        alpha[55:115, 145:185] = 1.0
        kept, report = keep_largest_component(alpha)
        self.assertGreater(float(kept[60, 50]), 0.9)
        self.assertEqual(float(kept[75, 165]), 0.0)
        self.assertEqual(report["significant_component_count"], 2)
        self.assertGreater(report["second_to_first_ratio"], 0.1)

    def test_refinement_preserves_soft_edge_and_fills_pose_hole(self):
        rgb = np.full((128, 128, 3), 245, dtype=np.uint8)
        rgb[16:116, 36:92] = (55, 90, 130)
        alpha = np.zeros((128, 128), dtype=np.float32)
        alpha[18:114, 38:90] = 0.96
        alpha[60:76, 54:72] = 0.0
        pose = np.zeros_like(alpha)
        pose[16:116, 36:92] = 1.0
        refined, _ = refine_alpha(alpha, rgb, pose)
        self.assertGreater(float(refined[68, 63]), 0.5)
        self.assertGreater(float(refined[20, 40]), 0.4)
        self.assertLess(float(refined[5, 5]), 0.05)

    def test_normalization_transforms_landmarks_to_square_canvas(self):
        rgb = np.zeros((200, 100, 3), dtype=np.uint8)
        rgb[20:190, 25:75] = (120, 80, 60)
        alpha = np.zeros((200, 100), dtype=np.float32)
        alpha[20:190, 25:75] = 1.0
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]
        pose = {"detected": True, "landmarks": landmarks, "world_landmarks": []}
        image, normalized_alpha, transformed, transform = normalize_subject(rgb, alpha, pose, canvas_size=512)
        self.assertEqual(image.size, (512, 512))
        self.assertEqual(normalized_alpha.shape, (512, 512))
        self.assertAlmostEqual(transformed["landmarks"][0]["x"], 0.5, delta=0.03)
        self.assertAlmostEqual(transformed["landmarks"][0]["y"], 0.5, delta=0.06)
        self.assertIn("source_landmarks", transformed)
        self.assertEqual(transform["canvas_size"], 512)

    def test_framing_reports_rig_ready_only_with_visible_full_body(self):
        alpha = np.zeros((200, 120), dtype=np.float32)
        alpha[10:190, 30:90] = 1.0
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]
        report = framing_report(alpha, {"detected": True, "landmarks": landmarks})
        self.assertTrue(report["rig_ready"])
        landmarks[28]["visibility"] = 0.1
        report = framing_report(alpha, {"detected": True, "landmarks": landmarks})
        self.assertFalse(report["rig_ready"])
        self.assertTrue(any("ankles" in warning for warning in report["warnings"]))


    def test_existing_transparent_avatar_is_not_background_removed_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "subject.png"
            image = Image.new("RGBA", (96, 128), (0, 0, 0, 0))
            patch = Image.new("RGBA", (36, 100), (120, 70, 45, 255))
            image.alpha_composite(patch, (30, 14))
            image.save(path)
            prepared, report = foreground_rgba(path, 256)
            self.assertEqual(report["backend"], "preserved_source_alpha")
            self.assertEqual(prepared.size, (256, 256))
            self.assertGreater(float(np.mean(np.asarray(prepared.getchannel("A")) > 0)), 0.05)

    def test_edge_decontamination_moves_fringe_toward_foreground(self):
        rgb = np.full((64, 64, 3), (20, 220, 20), dtype=np.uint8)
        rgb[18:46, 18:46] = (160, 80, 55)
        alpha = np.zeros((64, 64), dtype=np.float32)
        alpha[18:46, 18:46] = 1.0
        alpha[16:48, 16:48] = np.maximum(alpha[16:48, 16:48], 0.35)
        before_green = int(rgb[16, 32, 1])
        cleaned = decontaminate_edges(rgb, alpha)
        self.assertLess(int(cleaned[16, 32, 1]), before_green)


if __name__ == "__main__":
    unittest.main()
