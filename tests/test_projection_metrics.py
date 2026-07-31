"""Equivalence and scaling tests for the sparse ID-buffer projection metrics.

``_projection_metrics`` was rewritten to stop materialising one HxW mask and one HxW float64
depth image per component. These tests pin the rewrite to the retained dense reference
implementation, metric by metric, and assert the memory behaviour that motivated it.
"""
from __future__ import annotations

import unittest

import numpy as np

from lowvram3d.component_audit import AuditConfig, _projection_metrics

from reference_projection import _projection_metrics_dense

INTEGRAL_KEYS = (
    "visible_views",
    "island_views",
    "gap_views",
    "overlap_views",
    "depth_separated_views",
    "visible_pixels",
)
FLOAT_KEYS = ("aggregate_outside_percent", "median_depth_gap_diag", "source_support_percent")

CENTER = np.zeros(3)
HALF_EXTENT = 1.6
MODEL_DIAGONAL = 3.2


def blob(centre, radius=0.25, count=400, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    points /= np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
    return np.asarray(centre, np.float64) + points * radius


class ProjectionMetricsEquivalenceTests(unittest.TestCase):
    config = AuditConfig(render_size=96, total_samples=5_000, min_component_samples=64)

    def assert_equivalent(self, samples, main_id=0, source_mask=None, msg=""):
        expected = _projection_metrics_dense(
            samples, main_id, CENTER, HALF_EXTENT, MODEL_DIAGONAL, source_mask, self.config
        )
        actual = _projection_metrics(
            samples, main_id, CENTER, HALF_EXTENT, MODEL_DIAGONAL, source_mask, self.config
        )
        self.assertEqual(set(expected), set(actual), f"component id sets differ {msg}")
        for component_id, reference in expected.items():
            produced = actual[component_id]
            self.assertEqual(set(reference), set(produced), f"metric keys differ {msg}")
            for key in INTEGRAL_KEYS:
                self.assertEqual(
                    reference[key],
                    produced[key],
                    f"{key} mismatch for component {component_id} {msg}",
                )
            for key in FLOAT_KEYS:
                self.assertAlmostEqual(
                    reference[key],
                    produced[key],
                    delta=1e-9,
                    msg=f"{key} mismatch for component {component_id} {msg}",
                )
        return expected, actual

    def test_single_component_besides_main(self):
        self.assert_equivalent({0: blob((0.0, 0.0, 0.0), 0.6, 800, 1), 1: blob((0.9, 0.0, 0.0), 0.08, 200, 2)})

    def test_two_widely_separated_components(self):
        self.assert_equivalent(
            {
                0: blob((0.0, 0.0, 0.0), 0.5, 800, 3),
                1: blob((1.1, 0.0, 0.0), 0.07, 200, 4),
                2: blob((-1.1, 0.0, 0.4), 0.07, 200, 5),
            }
        )

    def test_partially_overlapping_component(self):
        self.assert_equivalent(
            {0: blob((0.0, 0.0, 0.0), 0.5, 900, 6), 1: blob((0.45, 0.0, 0.0), 0.2, 300, 7)}
        )

    def test_depth_separated_overlapping_component(self):
        """Same screen footprint as the body, offset along depth -- exercises the median gap."""
        self.assert_equivalent(
            {0: blob((0.0, 0.0, 0.0), 0.5, 900, 8), 1: blob((0.0, -1.0, 0.0), 0.18, 300, 9)}
        )

    def test_fully_occluded_small_component(self):
        self.assert_equivalent(
            {0: blob((0.0, 0.0, 0.0), 0.7, 1200, 10), 1: blob((0.0, 0.0, 0.0), 0.05, 200, 11)}
        )

    def test_isolated_debris_component(self):
        self.assert_equivalent(
            {0: blob((0.0, 0.0, 0.0), 0.4, 800, 12), 1: blob((1.25, 1.0, 0.9), 0.04, 150, 13)}
        )

    def test_components_projecting_into_the_same_pixel(self):
        point = np.tile(np.array([[0.5, 0.5, 0.5]]), (120, 1))
        self.assert_equivalent(
            {
                0: blob((0.0, 0.0, 0.0), 0.5, 700, 14),
                1: point.copy(),
                2: point + 1e-9,
            }
        )

    def test_out_of_frame_and_degenerate_components(self):
        """A component projecting entirely outside every view must be skipped identically."""
        self.assert_equivalent(
            {
                0: blob((0.0, 0.0, 0.0), 0.5, 700, 15),
                1: blob((40.0, 40.0, 40.0), 0.05, 100, 16),
                2: np.tile(np.array([[0.3, 0.3, 0.3]]), (64, 1)),
            }
        )

    def test_source_support_path_matches_reference(self):
        rng = np.random.default_rng(21)
        source_mask = np.zeros((120, 90), dtype=bool)
        source_mask[20:100, 15:75] = True
        source_mask &= rng.random((120, 90)) > 0.15
        self.assert_equivalent(
            {
                0: blob((0.0, 0.0, 0.0), 0.5, 900, 17),
                1: blob((0.6, 0.1, 0.1), 0.12, 250, 18),
                2: blob((-1.0, 0.5, 0.3), 0.05, 150, 19),
            },
            source_mask=source_mask,
        )

    def test_randomised_small_cases_with_fixed_seeds(self):
        for seed in range(8):
            rng = np.random.default_rng(1000 + seed)
            samples = {0: blob((0.0, 0.0, 0.0), 0.5, 700, seed)}
            for index in range(1, rng.integers(2, 6) + 1):
                centre = rng.uniform(-1.2, 1.2, size=3)
                samples[index] = blob(centre, float(rng.uniform(0.03, 0.25)), 160, seed * 50 + index)
            with self.subTest(seed=seed):
                self.assert_equivalent(samples, msg=f"(seed {seed})")

    def test_component_ordering_does_not_change_results(self):
        samples = {
            0: blob((0.0, 0.0, 0.0), 0.5, 800, 30),
            1: blob((0.8, 0.2, 0.0), 0.1, 200, 31),
            2: blob((-0.9, 0.3, 0.2), 0.08, 200, 32),
            3: blob((0.1, 1.0, 0.4), 0.06, 200, 33),
        }
        forward = _projection_metrics(
            samples, 0, CENTER, HALF_EXTENT, MODEL_DIAGONAL, None, self.config
        )
        reordered = {0: samples[0], 3: samples[3], 1: samples[1], 2: samples[2]}
        backward = _projection_metrics(
            reordered, 0, CENTER, HALF_EXTENT, MODEL_DIAGONAL, None, self.config
        )
        for component_id, reference in forward.items():
            self.assertEqual(reference, backward[component_id], f"component {component_id}")


class ProjectionMetricsScalingTests(unittest.TestCase):
    """The rewrite must not allocate per-component rasters as component count grows."""

    config = AuditConfig(render_size=96, total_samples=5_000, min_component_samples=64)

    def _fragmented(self, count: int) -> dict[int, np.ndarray]:
        rng = np.random.default_rng(4)
        samples = {0: blob((0.0, 0.0, 0.0), 0.5, 900, 99)}
        for index in range(1, count + 1):
            centre = rng.uniform(-1.2, 1.2, size=3)
            samples[index] = blob(centre, 0.03, 64, index)
        return samples

    def test_matches_reference_on_a_fragmented_mesh(self):
        samples = self._fragmented(60)
        expected = _projection_metrics_dense(
            samples, 0, CENTER, HALF_EXTENT, MODEL_DIAGONAL, None, self.config
        )
        actual = _projection_metrics(
            samples, 0, CENTER, HALF_EXTENT, MODEL_DIAGONAL, None, self.config
        )
        self.assertEqual(expected, actual)

    def test_no_dense_per_component_arrays_are_allocated(self):
        """Fails against the old implementation, which allocated one HxW array per component."""
        size = self.config.render_size
        plane = size * size
        samples = self._fragmented(80)
        big_allocations = []
        real_full, real_zeros = np.full, np.zeros

        def record(shape):
            count = int(np.prod(shape)) if not np.isscalar(shape) else int(shape)
            if count >= plane:
                big_allocations.append(count)

        def traced_full(shape, *args, **kwargs):
            record(shape)
            return real_full(shape, *args, **kwargs)

        def traced_zeros(shape, *args, **kwargs):
            record(shape)
            return real_zeros(shape, *args, **kwargs)

        np.full, np.zeros = traced_full, traced_zeros
        try:
            _projection_metrics(samples, 0, CENTER, HALF_EXTENT, MODEL_DIAGONAL, None, self.config)
        finally:
            np.full, np.zeros = real_full, real_zeros

        views = 14
        # The main body needs a handful of view-sized buffers per view; anything approaching
        # one per component would be far above this.
        self.assertLess(
            len(big_allocations),
            views * 6,
            f"{len(big_allocations)} view-sized allocations for {len(samples) - 1} components",
        )

    def test_work_tracks_projected_samples_not_components_times_resolution(self):
        small = self._fragmented(40)
        large = self._fragmented(160)
        base = _projection_metrics(small, 0, CENTER, HALF_EXTENT, MODEL_DIAGONAL, None, self.config)
        grown = _projection_metrics(large, 0, CENTER, HALF_EXTENT, MODEL_DIAGONAL, None, self.config)
        self.assertEqual(len(base), 40)
        self.assertEqual(len(grown), 160)
        # Shared components must be unaffected by the presence of additional ones.
        for component_id in range(1, 41):
            self.assertEqual(base[component_id], grown[component_id], f"component {component_id}")


if __name__ == "__main__":
    unittest.main()
