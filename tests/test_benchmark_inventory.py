from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lowvram3d.benchmark_suite import PRIMARY_ANCHOR_ID
from workers.benchmark_inventory import build_inventory, primary_status_path


class BenchmarkInventoryTests(unittest.TestCase):
    def test_primary_anchor_is_first_and_missing_pack_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            payload = build_inventory(Path(td), "", set())
            self.assertGreaterEqual(payload["fixture_count"], 17)
            self.assertEqual(payload["items"][0]["fixture_id"], PRIMARY_ANCHOR_ID)
            self.assertEqual(payload["overall_gate"], "BLOCKED_PRIMARY_ANCHOR_SOURCE_NOT_VERIFIED")
            self.assertIn(PRIMARY_ANCHOR_ID, payload["missing_anchor_images"])
            self.assertIn(PRIMARY_ANCHOR_ID, payload["missing_anchor_masters"])
            self.assertEqual(payload["commands_executed"], 0)

    def test_primary_anchor_gets_the_only_command_before_it_is_proven(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images").mkdir()
            (root / "masters").mkdir()
            (root / "images" / "antlered_bird_shaman_anchor.png").write_bytes(b"image")
            (root / "masters" / "antlered_bird_shaman_anchor.glb").write_bytes(b"mesh")
            (root / "images" / "turbo_bird.png").write_bytes(b"image")
            (root / "masters" / "turbo_bird_master.glb").write_bytes(b"mesh")
            payload = build_inventory(root, "blender", set())
            primary = payload["items"][0]
            bird = next(item for item in payload["items"] if item["fixture_id"] == "turbo_bird_high_detail")
            self.assertIsNotNone(primary["command"])
            self.assertIsNone(bird["command"])
            self.assertTrue(bird["blocked_by_primary_anchor"])

    def test_proven_receipt_unlocks_later_fixtures(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images").mkdir()
            (root / "masters").mkdir()
            (root / "images" / "turbo_bird.png").write_bytes(b"image")
            (root / "masters" / "turbo_bird_master.glb").write_bytes(b"mesh")
            status = primary_status_path(root)
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps({"classification": "PROVEN"}), encoding="utf-8")
            payload = build_inventory(root, "blender", {"turbo_bird_high_detail"})
            self.assertEqual(payload["primary_anchor_status"], "PROVEN")
            self.assertIsNotNone(payload["items"][0]["command"])
            self.assertTrue(payload["overall_pass_permitted"])

    def test_bypass_never_permits_overall_pass(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images").mkdir()
            (root / "masters").mkdir()
            (root / "images" / "turbo_bird.png").write_bytes(b"image")
            (root / "masters" / "turbo_bird_master.glb").write_bytes(b"mesh")
            payload = build_inventory(
                root,
                "blender",
                {"turbo_bird_high_detail"},
                allow_primary_bypass=True,
            )
            self.assertIsNotNone(payload["items"][0]["command"])
            self.assertFalse(payload["overall_pass_permitted"])
            self.assertTrue(payload["primary_bypass_used"])


if __name__ == "__main__":
    unittest.main()
