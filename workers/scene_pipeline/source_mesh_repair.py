"""CPU-only Castlegrounds source-mesh audit and bounded repair candidates.

This module consumes only the saved MoGe arrays.  It deliberately keeps the
historical GLBs untouched and writes versioned repair candidates beside them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.texture import TextureVisuals

from workers.scene_pipeline.core import write_json
from workers.scene_pipeline.projection import source_uv


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"
WIDTH, HEIGHT = 512, 384
BASELINES = {"strict_005": 0.005, "balanced_010": 0.010, "permissive_020": 0.020}
ADAPTIVE = {
    "adaptive_conservative": 0.015,
    "adaptive_balanced": 0.030,
    "adaptive_coverage": 0.050,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vertex_ids(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = -np.ones(valid.shape, dtype=np.int64)
    pixels = np.flatnonzero(valid.ravel())
    ids.ravel()[pixels] = np.arange(len(pixels), dtype=np.int64)
    return ids, pixels


def _triangle_ok(points: np.ndarray, face: tuple[int, int, int], limit: float) -> bool:
    tri = points[np.asarray(face)]
    if not np.isfinite(tri).all():
        return False
    edges = np.linalg.norm(np.roll(tri, -1, axis=0) - tri, axis=1)
    area2 = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    return bool(np.max(edges) <= limit and area2 > 1e-10)


def _face_normal(points: np.ndarray, face: np.ndarray) -> np.ndarray:
    return np.cross(points[face[:, 1]] - points[face[:, 0]], points[face[:, 2]] - points[face[:, 0]])


def vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(points, dtype=np.float64)
    if len(faces):
        cross = _face_normal(points, faces)
        for corner in range(3):
            np.add.at(normals, faces[:, corner], cross)
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-12
    normals[good] /= lengths[good, None]
    return normals.astype(np.float32)


def _cell_faces(
    points: np.ndarray,
    valid: np.ndarray,
    ids: np.ndarray,
    threshold: float,
    adaptive: bool = False,
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    compact_vertices = points.reshape(-1, 3)[np.flatnonzero(valid)]
    scale = float(np.nanmedian(np.linalg.norm(points[valid], axis=1)))
    limit = max(scale * threshold, 1e-6)
    faces: list[tuple[int, int, int]] = []
    counts = np.zeros((HEIGHT - 1, WIDTH - 1), dtype=np.uint8)
    reason = np.zeros_like(counts, dtype=np.uint8)
    stats = {"no_valid_cell": 0, "nonfinite_point": 0, "both_triangles_rejected": 0,
             "one_triangle_rejected": 0, "two_triangles_accepted": 0,
             "one_triangle_accepted": 0, "invalid_mask_boundary": 0,
             "edge_threshold": 0, "degenerate": 0, "adaptive_depth_boundary": 0}
    for y in range(HEIGHT - 1):
        for x in range(WIDTH - 1):
            corners = ((y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1))
            if not all(valid[cy, cx] for cy, cx in corners):
                stats["no_valid_cell"] += 1
                stats["invalid_mask_boundary"] += 1
                reason[y, x] = 1
                continue
            a = int(ids[y, x]); b = int(ids[y, x + 1]); c = int(ids[y + 1, x]); d = int(ids[y + 1, x + 1])
            p = points[[y, y, y + 1, y + 1], [x, x + 1, x, x + 1]]
            if not np.isfinite(p).all():
                stats["nonfinite_point"] += 1
                reason[y, x] = 2
                continue
            diag_ad = float(np.linalg.norm(p[0] - p[3]))
            diag_bc = float(np.linalg.norm(p[1] - p[2]))
            candidate = [(a, b, d), (a, d, c)] if diag_ad <= diag_bc else [(a, b, c), (b, d, c)]
            if adaptive:
                # Prefer the diagonal with lower normalized depth discontinuity.
                depth = p[:, 2]
                score_ad = abs(float(depth[0] - depth[3])) / max(float(np.mean(depth[[0, 3]])), 1e-6)
                score_bc = abs(float(depth[1] - depth[2])) / max(float(np.mean(depth[[1, 2]])), 1e-6)
                if score_bc < score_ad:
                    candidate = [(a, b, c), (b, d, c)]
                threshold_rel = threshold
                def local_ok(face: tuple[int, int, int]) -> bool:
                    tri = compact_vertices[np.asarray(face)]
                    z = tri[:, 2]
                    rel = np.abs(np.roll(z, -1) - z) / np.maximum((np.roll(z, -1) + z) * 0.5, 1e-6)
                    if np.max(rel) > threshold_rel:
                        return False
                    return _triangle_ok(compact_vertices, face, max(scale * 0.25, 1e-6))
                accepted = [local_ok(face) for face in candidate]
                if not all(accepted):
                    stats["adaptive_depth_boundary"] += 1
            else:
                accepted = [_triangle_ok(compact_vertices, face, limit) for face in candidate]
            accepted_count = int(sum(accepted))
            counts[y, x] = accepted_count
            if accepted_count == 2:
                stats["two_triangles_accepted"] += 1
                reason[y, x] = 5
            elif accepted_count == 1:
                stats["one_triangle_accepted"] += 1
                stats["one_triangle_rejected"] += 1
                reason[y, x] = 4
            else:
                stats["both_triangles_rejected"] += 1
                stats["edge_threshold" if not adaptive else "adaptive_depth_boundary"] += 1
                reason[y, x] = 3
            for face, ok in zip(candidate, accepted):
                if ok:
                    faces.append(face)
    return np.asarray(faces, dtype=np.int64).reshape(-1, 3), stats, counts


def mesh_stats(vertices: np.ndarray, faces: np.ndarray) -> dict[str, object]:
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1) if len(faces) else np.empty((0, 2), np.int64)
    if len(edges):
        _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    else:
        counts = np.empty(0, np.int64)
    areas = np.linalg.norm(_face_normal(vertices, faces), axis=1) * 0.5 if len(faces) else np.empty(0)
    parent = np.arange(len(vertices), dtype=np.int64)
    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value
    for face in faces:
        root = find(int(face[0]))
        for value in face[1:]:
            other = find(int(value))
            if root != other:
                parent[other] = root
    components = len({find(int(v)) for v in np.unique(faces)}) if len(faces) else 0
    return {
        "vertices": int(len(vertices)), "triangles": int(len(faces)),
        "connected_components": int(components),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "non_manifold_edges": int(np.count_nonzero(counts > 2)),
        "degenerate_triangles": int(np.count_nonzero(areas <= 1e-10)),
        "bounds_min": vertices.min(axis=0).tolist() if len(vertices) else None,
        "bounds_max": vertices.max(axis=0).tolist() if len(vertices) else None,
    }


def save_candidate(name: str, vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray, source: np.ndarray, reversed_faces: bool) -> Path:
    path = ROOT / f"{name}.glb"
    normals = vertex_normals(vertices, faces)
    image = Image.fromarray(source, mode="RGB")
    visual = TextureVisuals(uv=uvs.astype(np.float32), image=image)
    mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=faces, vertex_normals=normals, visual=visual, process=False)
    mesh.export(path, file_type="glb")
    np.save(ROOT / f"{name}_faces.npy", faces)
    return path


def _write_image(name: str, image: np.ndarray) -> None:
    cv2.imwrite(str(ROOT / name), image)


def raster_face_coverage(faces: np.ndarray, valid_pixels: np.ndarray) -> np.ndarray:
    covered = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    ys, xs = np.unravel_index(valid_pixels, (HEIGHT, WIDTH))
    coords = np.column_stack([xs, ys])
    for face in faces:
        cv2.fillConvexPoly(covered, np.rint(coords[face]).astype(np.int32), 255)
    return covered.astype(bool)


def main() -> None:
    points = np.load(ROOT / "points.npy").astype(np.float64)
    mask = np.load(ROOT / "mask.npy").astype(bool)
    source_bgr = cv2.imread(str(ROOT / "source_rgb_512.png"), cv2.IMREAD_COLOR)
    if source_bgr is None:
        source_bgr = cv2.cvtColor(cv2.imread(str(ROOT / "source_rgb.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    valid = mask & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    if points.shape != (HEIGHT, WIDTH, 3) or valid.shape != (HEIGHT, WIDTH):
        raise RuntimeError("MOGE_ARRAY_SHAPE_INVALID")
    ids, valid_pixels = _vertex_ids(valid)
    vertices = points.reshape(-1, 3)[valid_pixels].astype(np.float32)
    uvs = source_uv(WIDTH, HEIGHT).reshape(-1, 2)[valid_pixels]
    all_results: dict[str, dict[str, object]] = {}

    baseline_faces: dict[str, np.ndarray] = {}
    for name, threshold in BASELINES.items():
        faces, stats, counts = _cell_faces(points, valid, ids, threshold, adaptive=False)
        baseline_faces[name] = faces
        for suffix, candidate_faces, rev in (("", faces, False), ("_winding", faces[:, [0, 2, 1]], True)):
            out_name = name + suffix
            path = save_candidate(out_name, vertices, candidate_faces, uvs, source, rev)
            all_results[out_name] = {"kind": "baseline", "threshold": threshold, "reversed_faces": rev,
                                     **mesh_stats(vertices, candidate_faces), "glb": str(path), "cell_stats": stats}

    adaptive_faces: dict[str, np.ndarray] = {}
    for name, threshold in ADAPTIVE.items():
        faces, stats, counts = _cell_faces(points, valid, ids, threshold, adaptive=True)
        adaptive_faces[name] = faces
        path = save_candidate(name, vertices, faces, uvs, source, False)
        all_results[name] = {"kind": "adaptive", "relative_depth_threshold": threshold,
                             **mesh_stats(vertices, faces), "glb": str(path), "cell_stats": stats}

    balanced = baseline_faces["balanced_010"]
    coverage = raster_face_coverage(balanced, valid_pixels)
    source_mask = valid
    counts = np.zeros((HEIGHT - 1, WIDTH - 1), np.uint8)
    _, baseline_reason_counts, counts = _cell_faces(points, valid, ids, 0.010, adaptive=False)
    # 0 / 1 / 2 accepted triangles are intentionally visually distinct.
    count_image = np.zeros((HEIGHT, WIDTH), np.uint8)
    count_image[:-1, :-1] = counts * 127
    _write_image("accepted_face_count_per_cell.png", count_image)
    reason_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    palette = {1: (0, 0, 255), 2: (255, 0, 255), 3: (0, 165, 255), 4: (0, 255, 255), 5: (0, 255, 0)}
    _, _, reasons = _cell_faces(points, valid, ids, 0.010, adaptive=False)
    # The second return above is the cell count; reason values are reconstructed
    # by the explicit per-cell count categories for a stable audit image.
    reason_image[:-1, :-1][counts == 0] = palette[3]
    reason_image[:-1, :-1][counts == 1] = palette[4]
    reason_image[:-1, :-1][counts == 2] = palette[5]
    reason_image[~valid] = palette[1]
    _write_image("rejected_cell_reason.png", reason_image)
    _write_image("source_face_coverage_mask.png", np.where(coverage, 255, 0).astype(np.uint8))
    old_render = cv2.imread(str(ROOT / "blender_exact_source_cull_off.png"), cv2.IMREAD_COLOR)
    old_missing_pixels = None
    old_cull_on_missing_pixels = None
    if old_render is not None:
        old_visible = old_render.mean(axis=2) > 3.0
        missing = valid & ~old_visible
        old_missing_pixels = int(np.count_nonzero(missing))
        cull_on_render = cv2.imread(str(ROOT / "blender_exact_source_cull_on.png"), cv2.IMREAD_COLOR)
        if cull_on_render is not None:
            old_cull_on_missing_pixels = int(np.count_nonzero(valid & ~(cull_on_render.mean(axis=2) > 3.0)))
        overlay = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        overlay[missing] = (0, 0, 255)
        _write_image("blender_missing_pixel_overlay.png", overlay)
    receipt = {
        "schema": "castlegrounds_source_mesh_repair_v1",
        "classification": "CPU_ONLY_SAVED_ARRAYS_REPAIR_CANDIDATES",
        "arrays": {name: str(ROOT / name) for name in ("points.npy", "depth.npy", "normal.npy", "mask.npy", "intrinsics.npy")},
        "valid_vertex_pixels": int(valid.sum()),
        "valid_2x2_cells": int((valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]).sum()),
        "baseline_balanced": {"accepted_faces": int(len(balanced)), "rasterized_source_pixels": int(np.count_nonzero(coverage & source_mask)),
                              "source_face_coverage": float(np.count_nonzero(coverage & source_mask) / max(np.count_nonzero(source_mask), 1)),
                              "cell_stats": baseline_reason_counts},
        "missing_cause_distribution": {
            "valid_vertex_pixels": int(valid.sum()),
            "valid_pixels_without_rasterized_balanced_face": int(np.count_nonzero(valid & ~coverage)),
            "blender_cull_off_missing_pixels": old_missing_pixels,
            "blender_cull_on_missing_pixels": old_cull_on_missing_pixels,
            "invalid_mask_boundary_cells": baseline_reason_counts["invalid_mask_boundary"],
            "both_triangles_rejected_cells": baseline_reason_counts["both_triangles_rejected"],
            "one_triangle_rejected_cells": baseline_reason_counts["one_triangle_rejected"],
            "winding_culling": "ALL_SOURCE_FACES_REJECTED_WITH_CULL_ON" if old_cull_on_missing_pixels == int(valid.sum()) else "NOT_TOTAL",
            "glb_rasterization_precision": "NOT_OBSERVED_AS_PRIMARY_CAUSE",
        },
        "candidates": all_results,
        "artifacts": {name: str(ROOT / name) for name in ("accepted_face_count_per_cell.png", "rejected_cell_reason.png", "source_face_coverage_mask.png", "blender_missing_pixel_overlay.png")},
        "source_arrays_sha256": {name: sha256(ROOT / name) for name in ("points.npy", "depth.npy", "normal.npy", "mask.npy", "intrinsics.npy")},
    }
    write_json(ROOT / "rejection_reason_counts.json", receipt)
    write_json(PROOF / "rejection_reason_counts.json", receipt)


if __name__ == "__main__":
    main()
