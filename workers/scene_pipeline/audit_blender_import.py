"""Import selected and fixture GLBs in Blender and measure world transforms."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"


def import_one(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [obj for obj in bpy.context.scene.objects if obj not in before and obj.type == "MESH"]


def affine_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    design = np.column_stack([source.astype(np.float64), np.ones(len(source))])
    coefficients, _, _, _ = np.linalg.lstsq(design, target.astype(np.float64), rcond=None)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :] = coefficients.T
    residual = design @ coefficients - target
    return matrix, float(np.linalg.norm(residual, axis=1).max()), float(np.linalg.norm(residual, axis=1).mean())


def world_vertices(obj: bpy.types.Object) -> np.ndarray:
    return np.asarray([(obj.matrix_world @ vertex.co)[:] for vertex in obj.data.vertices], dtype=np.float64)


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    selected_objects = import_one(ROOT / "balanced_010.glb")
    fixture_objects = import_one(ROOT / "axis_fixture.glb")
    if not selected_objects or not fixture_objects:
        raise RuntimeError("BLENDER_GLTF_IMPORT_AUDIT_MESH_MISSING")
    selected = selected_objects[0]
    fixture = fixture_objects[0]
    raw = np.load(ROOT / "points.npy").reshape(-1, 3)
    mask = np.load(ROOT / "mask.npy").astype(bool).ravel()
    raw_vertices = raw[mask]
    used_indices = np.load(ROOT / "balanced_010_used_vertex_indices.npy")
    raw_used_vertices = raw_vertices[used_indices]
    selected_world = world_vertices(selected)
    print("AUDIT_COUNTS selected", len(selected_world), "raw_used", len(raw_used_vertices), "fixture", len(fixture.data.vertices))
    if len(selected_world) != len(raw_used_vertices):
        raise RuntimeError("BLENDER_SELECTED_VERTEX_COUNT_MISMATCH:%d:%d" % (len(selected_world), len(raw_used_vertices)))
    selected_matrix, selected_max, selected_mean = affine_fit(raw_used_vertices, selected_world)
    fixture_input = np.asarray(json.loads((ROOT / "axis_fixture_input.json").read_text(encoding="utf-8"))["vertices"], dtype=np.float64)
    fixture_world = world_vertices(fixture)
    fixture_matrix, fixture_max, fixture_mean = affine_fit(fixture_input, fixture_world)
    matrix = np.asarray(fixture_matrix, dtype=np.float64)
    determinant = float(np.linalg.det(matrix[:3, :3]))
    report = {
        "schema": "moge_glb_blender_transform_audit_v1",
        "classification": "MOGE_GLB_BLENDER_TRANSFORM_PROVEN" if fixture_max < 1e-5 and selected_max < 1e-3 else "MOGE_GLB_BLENDER_TRANSFORM_REJECTED",
        "selected": {"object": selected.name, "matrix_world": [list(row) for row in selected.matrix_world], "accessor_vertex_count": int(len(raw_vertices)), "used_vertex_count": int(len(selected_world)), "M_raw_moge_to_blender_fit": selected_matrix.tolist(), "max_residual": selected_max, "mean_residual": selected_mean, "world_bounds_min": selected_world.min(axis=0).tolist(), "world_bounds_max": selected_world.max(axis=0).tolist()},
        "axis_fixture": {"object": fixture.name, "input_vertices": fixture_input.tolist(), "world_vertices": fixture_world.tolist(), "M_glb_to_blender": matrix.tolist(), "max_residual": fixture_max, "mean_residual": fixture_mean, "determinant": determinant, "handedness": "preserved" if determinant > 0 else "flipped"},
        "composition_check": "M_raw_moge_to_blender = M_glb_to_blender x M_raw_moge_to_glb; raw-to-GLB is audited separately",
        "uv_orientation": "GLB UVs retained; source Y-down is represented by v=1-y/(height-1)",
        "normal_transform": "inverse-transpose of linear component required for transformed normals; identity at raw-to-GLB boundary",
    }
    (ROOT / "blender_transform_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (PROOF / "blender_transform_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
