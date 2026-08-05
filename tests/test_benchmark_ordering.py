from __future__ import annotations

import unittest

from lowvram3d.benchmark_suite import (
    ANCHOR_IDS,
    FIXTURES,
    PRIMARY_ANCHOR_ID,
    fixture_by_id,
    manifest,
    ordered_fixtures,
)


class BenchmarkOrderingTests(unittest.TestCase):
    def test_priority_not_tuple_position_controls_order(self):
        shuffled = tuple(reversed(FIXTURES))
        ordered = tuple(sorted(shuffled, key=lambda item: (item.priority, item.fixture_id)))
        self.assertEqual(ordered[0].fixture_id, PRIMARY_ANCHOR_ID)
        self.assertTrue(ordered[0].required_first)

    def test_three_mandatory_anchors_are_explicit(self):
        self.assertEqual(
            ANCHOR_IDS,
            (PRIMARY_ANCHOR_ID, "turbo_bird_high_detail", "red_panda_character"),
        )
        self.assertTrue(all(fixture_by_id(item).anchor for item in ANCHOR_IDS))

    def test_manifest_declares_primary_gate(self):
        payload = manifest()
        self.assertEqual(payload["policy"]["primary_anchor"], PRIMARY_ANCHOR_ID)
        self.assertTrue(payload["policy"]["primary_anchor_must_be_proven_before_later_fixtures"])
        self.assertEqual(payload["fixtures"][0]["fixture_id"], PRIMARY_ANCHOR_ID)
        self.assertEqual(ordered_fixtures()[1].fixture_id, "turbo_bird_high_detail")
        self.assertEqual(ordered_fixtures()[2].fixture_id, "red_panda_character")


if __name__ == "__main__":
    unittest.main()
