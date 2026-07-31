from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from lowvram3d.component_audit import AuditConfig, audit_and_cleanup
from lowvram3d.geometry_compare import silhouette_metrics, topology_counts


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
