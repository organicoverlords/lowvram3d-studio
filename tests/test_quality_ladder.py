from __future__ import annotations

import unittest

from lowvram3d.benchmark_suite import FIXTURES, fixture_by_id, manifest
from lowvram3d.quality_ladder import (
    AssetFamily,
    CandidateEvaluation,
    candidate_ladder,
    evaluate_candidate,
    select_lowest_passing,
    should_stop_descending,
    thresholds_for,
)


def candidate(name: str, faces: int, *, valid_shape: bool = True) -> CandidateEvaluation:
    return CandidateEvaluation(
        name=name,
        face_count=faces,
        silhouette_iou_min=0.998 if valid_shape else 0.90,
        surface_distance_p95_diag=0.001,
        surface_distance_p99_diag=0.003,
        reverse_distance_p95_diag=0.0015,
        normal_deviation_p95_deg=8.0,
        thin_feature_recall=0.99,
        meaningful_component_recall=1.0,
        boundary_edges_before=100,
        boundary_edges_after=100,
        non_manifold_before=1,
        non_manifold_after=1,
    )


class QualityLadderTests(unittest.TestCase):
    def test_turbo_scale_hero_ladder_starts_high_and_descends(self):
        plans = candidate_ladder(1_800_000, "hero", AssetFamily.ORGANIC)
        budgets = [item.target_faces for item in plans]
        self.assertEqual(budgets[0], 1_440_000)
        self.assertIn(900_000, budgets)
        self.assertIn(450_000, budgets)
        self.assertEqual(budgets[-1], 180_000)
        self.assertEqual(budgets, sorted(budgets, reverse=True))

    def test_small_source_is_not_upscaled_by_face_budget(self):
        plans = candidate_ladder(42_000, "hero", AssetFamily.MIXED)
        self.assertTrue(all(item.target_faces < 42_000 for item in plans))
        self.assertGreaterEqual(min(item.target_faces for item in plans), 30_000)

    def test_lowest_passing_candidate_wins(self):
        thresholds = thresholds_for(AssetFamily.ORGANIC, "hero")
        high = evaluate_candidate(candidate("high", 900_000), thresholds)
        medium = evaluate_candidate(candidate("medium", 450_000), thresholds)
        low = evaluate_candidate(candidate("low", 180_000, valid_shape=False), thresholds)
        selected = select_lowest_passing([high, medium, low])
        self.assertIsNotNone(selected)
        self.assertEqual(selected.name, "medium")

    def test_component_loss_always_fails(self):
        thresholds = thresholds_for(AssetFamily.MIXED, "hero")
        item = candidate("missing-part", 500_000)
        item.meaningful_component_recall = 0.99
        evaluated = evaluate_candidate(item, thresholds)
        self.assertFalse(evaluated.valid)
        self.assertTrue(any("components" in error for error in evaluated.errors))

    def test_boundary_regression_fails(self):
        thresholds = thresholds_for(AssetFamily.ORGANIC, "hero")
        item = candidate("holes", 500_000)
        item.boundary_edges_after = 200
        evaluated = evaluate_candidate(item, thresholds)
        self.assertFalse(evaluated.valid)
        self.assertTrue(any("boundary" in error for error in evaluated.errors))

    def test_two_failures_after_a_pass_stop_the_ladder(self):
        thresholds = thresholds_for(AssetFamily.ORGANIC, "hero")
        evaluations = [
            evaluate_candidate(candidate("pass", 900_000), thresholds),
            evaluate_candidate(candidate("fail-1", 450_000, valid_shape=False), thresholds),
            evaluate_candidate(candidate("fail-2", 288_000, valid_shape=False), thresholds),
        ]
        self.assertTrue(should_stop_descending(evaluations))

    def test_no_pass_does_not_stop_before_master_fallback(self):
        thresholds = thresholds_for(AssetFamily.ORGANIC, "hero")
        evaluations = [
            evaluate_candidate(candidate("fail-1", 900_000, valid_shape=False), thresholds),
            evaluate_candidate(candidate("fail-2", 450_000, valid_shape=False), thresholds),
        ]
        self.assertFalse(should_stop_descending(evaluations))


class BenchmarkSuiteTests(unittest.TestCase):
    def test_suite_covers_generated_cross_category_examples(self):
        self.assertGreaterEqual(len(FIXTURES), 16)
        self.assertEqual(len({fixture.fixture_id for fixture in FIXTURES}), len(FIXTURES))
        self.assertEqual(len({fixture.source_image_name for fixture in FIXTURES}), len(FIXTURES))
        families = {fixture.family for fixture in FIXTURES}
        self.assertIn(AssetFamily.ORGANIC, families)
        self.assertIn(AssetFamily.HARD_SURFACE, families)
        self.assertIn(AssetFamily.ARCHITECTURAL, families)
        self.assertIn(AssetFamily.MIXED, families)
        groups = {fixture.group for fixture in FIXTURES}
        self.assertIn("generated_character_examples", groups)
        self.assertIn("generated_creature_examples", groups)
        self.assertIn("generated_building_examples", groups)
        self.assertIn("generated_natural_examples", groups)

    def test_turbo_bird_is_the_high_resolution_anchor(self):
        fixture = fixture_by_id("turbo_bird_high_detail")
        self.assertEqual(fixture.master_reference_name, "turbo_bird_master.glb")
        self.assertIn("master_to_lod_detail_comparison", fixture.required_checks)

    def test_generated_building_and_creature_examples_are_registered(self):
        for fixture_id in (
            "open_plan_building",
            "lighthouse_archipelago_fortress",
            "river_casino_vessel",
            "mountain_demigod",
            "eternal_great_tree",
            "steampunk_snapping_turtle",
        ):
            self.assertEqual(fixture_by_id(fixture_id).fixture_id, fixture_id)

    def test_manifest_forbids_fixture_specific_production_rules(self):
        payload = manifest()
        self.assertTrue(payload["policy"]["production_rules_may_not_reference_fixture_ids"])
        self.assertTrue(payload["policy"]["clean_high_resolution_master_required"])
        self.assertFalse(payload["policy"]["large_binary_fixtures_committed"])
        self.assertEqual(payload["fixture_count"], len(FIXTURES))


if __name__ == "__main__":
    unittest.main()
