from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from workers.benchmark_inventory import build_inventory


class BenchmarkInventoryTests(unittest.TestCase):
    def test_missing_local_pack_is_reported_without_executing_commands(self):
        with tempfile.TemporaryDirectory() as td:
            payload = build_inventory(Path(td), "", set())
            self.assertGreaterEqual(payload["fixture_count"], 16)
            self.assertEqual(payload["available_images"], 0)
            self.assertEqual(payload["available_masters"], 0)
            self.assertEqual(payload["commands_executed"], 0)
            self.assertIn("turbo_bird_high_detail", payload["missing_anchor_masters"])
            self.assertIn("red_panda_character", payload["missing_anchor_masters"])

    def test_available_anchor_gets_a_deterministic_geometry_command(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            masters = root / "masters"
            images = root / "images"
            masters.mkdir()
            images.mkdir()
            (masters / "turbo_bird_master.glb").write_bytes(b"fixture")
            (images / "turbo_bird.png").write_bytes(b"fixture")
            payload = build_inventory(
                root,
                r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
                {"turbo_bird_high_detail"},
            )
            self.assertEqual(payload["fixture_count"], 1)
            item = payload["items"][0]
            self.assertTrue(item["master_present"])
            self.assertTrue(item["image_present"])
            self.assertIn("--benchmark-fixture", item["command"])
            self.assertIn("turbo_bird_high_detail", item["command"])
            self.assertIn("--source-image", item["command"])


if __name__ == "__main__":
    unittest.main()
