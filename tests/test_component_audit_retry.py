from __future__ import annotations

import unittest
from unittest.mock import patch

from lowvram3d.component_audit import AuditConfig
from workers.component_audit_cleanup import (
    _micro_retry_candidates,
    _strict_micro_retry_config,
    run_cleanup,
)


def unresolved_result(*, source_support: float = 0.0, area_fraction: float = 0.0012) -> dict:
    return {
        "success": False,
        "errors": [
            "cleanup did not converge within max_passes",
            "1 visible components remain audit-required",
        ],
        "final_audit": {
            "decisions": [
                {
                    "component_id": 32,
                    "signature": "micro-fragment",
                    "action": "AUDIT_REQUIRED",
                    "area_fraction": area_fraction,
                    "nearest_distance_diag": 0.045,
                    "elongation": 1.4,
                    "projection": {
                        "source_support_percent": source_support,
                        "island_views": 8,
                        "gap_views": 8,
                        "aggregate_outside_percent": 75.0,
                        "depth_separated_views": 4,
                        "overlap_views": 4,
                        "median_depth_gap_diag": 0.17,
                    },
                }
            ]
        },
    }


class MicroDebrisEligibilityTests(unittest.TestCase):
    def test_tiny_unsupported_character_fragment_is_eligible(self):
        candidates = _micro_retry_candidates(unresolved_result(), "character")
        self.assertEqual([item["signature"] for item in candidates], ["micro-fragment"])

    def test_source_supported_fragment_is_not_eligible(self):
        self.assertEqual(
            _micro_retry_candidates(unresolved_result(source_support=12.0), "character"),
            [],
        )

    def test_larger_fragment_is_not_eligible(self):
        self.assertEqual(
            _micro_retry_candidates(unresolved_result(area_fraction=0.01), "creature"),
            [],
        )

    def test_non_organic_asset_never_uses_retry(self):
        self.assertEqual(_micro_retry_candidates(unresolved_result(), "prop"), [])


class MicroDebrisRetryTests(unittest.TestCase):
    def test_retry_raises_sampling_without_widening_area_cap(self):
        baseline = AuditConfig(min_component_samples=128, max_component_samples=8192)
        strict = _strict_micro_retry_config(baseline)
        self.assertGreaterEqual(strict.min_component_samples, 1024)
        self.assertGreaterEqual(strict.max_component_samples, 16_384)
        self.assertGreaterEqual(strict.max_passes, 6)
        self.assertEqual(strict.outboard_views, 6)
        self.assertEqual(strict.gap_views, 6)
        self.assertLessEqual(strict.outboard_max_area_fraction, 0.003)

    def test_run_cleanup_retries_once_from_original_input(self):
        baseline = unresolved_result()
        passed = {
            "success": True,
            "errors": [],
            "topology_before": {"faces": 100, "boundary_edges": 0},
            "topology_after": {"faces": 96, "boundary_edges": 0},
            "faces_removed_percent": 4.0,
            "passes": [{}],
        }
        config = AuditConfig()
        with patch(
            "workers.component_audit_cleanup.audit_and_cleanup",
            side_effect=[baseline, passed],
        ) as mocked:
            result = run_cleanup(
                "source.glb",
                "clean.glb",
                asset_type="character",
                source_image="source.png",
                config=config,
                seed=0,
            )
        self.assertTrue(result["success"])
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(
            result["audit_policy"]["selected"],
            "strict_micro_debris_retry",
        )
        second_config = mocked.call_args_list[1].kwargs["config"]
        self.assertGreaterEqual(second_config.min_component_samples, 1024)


if __name__ == "__main__":
    unittest.main()
