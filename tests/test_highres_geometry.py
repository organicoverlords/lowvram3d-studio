from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from lowvram3d.component_audit import AuditConfig, audit_and_cleanup
from lowvram3d.geometry_compare import (
    sample_face_subset,
    sample_surface,
    silhouette_metrics,
    topology_counts,
)
from workers.component_audit_cleanup import config_for_asset_type


class GeometryComparisonPrimitiveTests(unittest.TestCase):
    def test_identical_point_sets_have_perfect_silhouette(self):
        rng = np.random.default_rng(7)
        points = rng.normal(size=(40_000, 3))
        points /= np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
        report = silhouette_metrics(
            points,
            points.copy(),
            center=np.zeros(3),
            half_extent=1.2,
            size=192,
        )
        self.assertEqual(report["iou_min"], 1.0)
        self.assertEqual(report["thin_feature_recall_min"], 1.0)

    def test_topology_counts_closed_box(self):
        box = trimesh.creation.box()
        counts = topology_counts(box)
        self.assertEqual(counts["boundary_edges"], 0)
        self.assertEqual(counts["non_manifold_edges"], 0)
        self.assertEqual(counts["faces"], 12)


class SubsetSamplingTests(unittest.TestCase):
    """Per-component sampling must not scale with the size of the whole mesh.

    Regression for the component-audit OOM: ``_component_points`` used
    ``mesh.submesh(..., append=True)``, and ``trimesh.util.submesh`` allocates an index array
    the size of the entire source mesh on every call. A 750k-vertex master split into 223,679
    components therefore churned >1 TiB of transient memory and exhausted the heap.
    """

    def _two_component_mesh(self) -> tuple[trimesh.Trimesh, np.ndarray]:
        main = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        debris = trimesh.creation.icosphere(subdivisions=1, radius=0.1)
        debris.apply_translation((2.0, 0.0, 0.0))
        combined = trimesh.util.concatenate((main, debris))
        debris_faces = np.arange(len(main.faces), len(combined.faces), dtype=np.int64)
        return combined, debris_faces

    def test_subset_sampling_matches_submesh_sampling_exactly(self):
        mesh, debris_faces = self._two_component_mesh()
        legacy = sample_surface(
            mesh.submesh([debris_faces], append=True, repair=False), 512, 11
        )
        subset = sample_face_subset(mesh, debris_faces, 512, 11)
        np.testing.assert_allclose(subset.points, legacy.points, rtol=0, atol=1e-12)
        np.testing.assert_allclose(subset.normals, legacy.normals, rtol=0, atol=1e-12)

    def test_subset_sampling_returns_global_face_ids_within_the_subset(self):
        mesh, debris_faces = self._two_component_mesh()
        subset = sample_face_subset(mesh, debris_faces, 256, 3)
        self.assertTrue(np.isin(subset.face_ids, debris_faces).all())
        self.assertEqual(len(subset.points), 256)

    def test_sampled_points_lie_on_the_requested_faces_only(self):
        mesh, debris_faces = self._two_component_mesh()
        points = sample_face_subset(mesh, debris_faces, 400, 5).points
        # Every sample must sit inside the debris sphere's bounds, never on the main body.
        debris_bounds = mesh.submesh([debris_faces], append=True, repair=False).bounds
        self.assertTrue((points >= debris_bounds[0] - 1e-9).all())
        self.assertTrue((points <= debris_bounds[1] + 1e-9).all())

    def test_subset_sampling_is_deterministic_for_a_fixed_seed(self):
        mesh, debris_faces = self._two_component_mesh()
        first = sample_face_subset(mesh, debris_faces, 300, 42).points
        second = sample_face_subset(mesh, debris_faces, 300, 42).points
        np.testing.assert_array_equal(first, second)

    def test_subset_sampling_rejects_empty_and_non_positive_counts(self):
        mesh, debris_faces = self._two_component_mesh()
        with self.assertRaises(ValueError):
            sample_face_subset(mesh, np.array([], dtype=np.int64), 16, 0)
        with self.assertRaises(ValueError):
            sample_face_subset(mesh, debris_faces, 0, 0)

    def test_audit_submesh_calls_do_not_scale_with_component_count(self):
        """``Trimesh.submesh`` is O(whole mesh) per call.

        One call per removal pass is fine; one call per *component* is the defect. This builds a
        mesh with many components and asserts the call count stays bounded by the pass budget.
        """
        main = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        parts = [main]
        for index in range(24):
            debris = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
            debris.apply_translation((2.0 + 0.3 * index, 0.2 * index, 0.0))
            parts.append(debris)
        mesh = trimesh.util.concatenate(parts)

        max_passes = 2
        calls = []
        original = trimesh.Trimesh.submesh

        def counting_submesh(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.glb"
            mesh.export(source)
            trimesh.Trimesh.submesh = counting_submesh
            try:
                audit_and_cleanup(
                    source,
                    root / "clean.glb",
                    asset_type="creature",
                    config=AuditConfig(
                        render_size=128,
                        total_samples=20_000,
                        min_component_samples=64,
                        max_component_samples=512,
                        max_passes=max_passes,
                    ),
                )
            finally:
                trimesh.Trimesh.submesh = original

        self.assertLessEqual(
            len(calls),
            max_passes,
            f"submesh called {len(calls)} times for {len(parts)} components -- "
            "per-component submeshing has regressed",
        )


class ComponentAuditPolicyTests(unittest.TestCase):
    def test_architecture_uses_tighter_auto_removal_caps_than_organic_assets(self):
        building = config_for_asset_type("building", render_size=384, samples=220_000, max_passes=4)
        creature = config_for_asset_type("creature", render_size=384, samples=220_000, max_passes=4)
        self.assertLess(building.outboard_max_area_fraction, creature.outboard_max_area_fraction)
        self.assertLess(building.hover_max_area_fraction, creature.hover_max_area_fraction)
        self.assertEqual(building.internal_max_area_fraction, 0.0)

    def test_scene_policy_preserves_disconnected_structural_parts(self):
        scene = config_for_asset_type("scene", render_size=100, samples=10_000, max_passes=20)
        self.assertGreaterEqual(scene.render_size, 192)
        self.assertGreaterEqual(scene.total_samples, 50_000)
        self.assertLessEqual(scene.max_passes, 6)
        self.assertLessEqual(scene.outboard_max_area_fraction, 0.0015)


class ComponentAuditTests(unittest.TestCase):
    def _run(self, mesh: trimesh.Trimesh, asset_type: str = "creature") -> dict:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.glb"
            output = root / "clean.glb"
            mesh.export(source)
            result = audit_and_cleanup(
                source,
                output,
                asset_type=asset_type,
                config=AuditConfig(
                    render_size=160,
                    total_samples=50_000,
                    min_component_samples=256,
                    max_component_samples=4096,
                    max_passes=3,
                ),
            )
            if result["success"]:
                self.assertTrue(output.is_file())
            return result

    def test_outboard_tiny_component_is_removed_without_touching_main(self):
        main = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        debris = trimesh.creation.icosphere(subdivisions=1, radius=0.08)
        debris.apply_translation((1.8, 0.0, 0.0))
        combined = trimesh.util.concatenate((main, debris))
        result = self._run(combined)
        self.assertTrue(result["success"], result["errors"])
        self.assertGreater(result["faces_removed"], 0, result["final_audit"])
        self.assertEqual(
            result["main_component_faces_before"],
            result["main_component_faces_after"],
        )

    def test_large_detached_component_is_not_deleted_from_size_or_detachment_alone(self):
        main = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        second = trimesh.creation.box(extents=(1.4, 1.4, 1.4))
        second.apply_translation((3.0, 0.0, 0.0))
        combined = trimesh.util.concatenate((main, second))
        result = self._run(combined, asset_type="vehicle")
        self.assertEqual(result["faces_removed"], 0)
        decisions = result["final_audit"]["decisions"]
        self.assertTrue(any(item["action"] == "KEEP_CONFIRMED" for item in decisions[1:]))

    def test_cleanup_never_increases_boundary_edges(self):
        main = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        debris = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
        debris.apply_translation((2.0, 0.0, 0.0))
        result = self._run(trimesh.util.concatenate((main, debris)))
        self.assertLessEqual(
            result["topology_after"]["boundary_edges"],
            result["topology_before"]["boundary_edges"],
        )


if __name__ == "__main__":
    unittest.main()
