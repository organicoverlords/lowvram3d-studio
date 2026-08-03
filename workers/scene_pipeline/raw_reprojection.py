"""CPU-only source-frame reprojection using saved MoGe arrays."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from workers.scene_pipeline.core import write_json
from workers.scene_pipeline.projection import project_points, reprojection_metrics, source_uv


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"
WIDTH, HEIGHT = 512, 384


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_candidate(points: np.ndarray, mask: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    horizontal = np.linalg.norm(points[:, 1:] - points[:, :-1], axis=-1)
    vertical = np.linalg.norm(points[1:, :] - points[:-1, :], axis=-1)
    vertex_id = -np.ones((height, width), np.int64)
    valid_pixels = np.flatnonzero(mask.ravel())
    vertex_id.ravel()[valid_pixels] = np.arange(len(valid_pixels), dtype=np.int64)
    vertices = points.reshape(-1, 3)[valid_pixels].astype(np.float64)
    scale = float(np.nanmedian(np.linalg.norm(points[mask], axis=1)))
    edge_limit = max(scale * threshold, 1e-6)
    faces: list[list[int]] = []
    for y in range(height - 1):
        for x in range(width - 1):
            if not (mask[y, x] and mask[y, x + 1] and mask[y + 1, x] and mask[y + 1, x + 1]):
                continue
            a, b, c, d = (int(vertex_id[y, x]), int(vertex_id[y, x + 1]), int(vertex_id[y + 1, x]), int(vertex_id[y + 1, x + 1]))
            diag_ad = float(np.linalg.norm(points[y, x] - points[y + 1, x + 1]))
            diag_bc = float(np.linalg.norm(points[y, x + 1] - points[y + 1, x]))
            candidate_faces = ([a, b, d], [a, d, c]) if diag_ad <= diag_bc else ([a, b, c], [b, d, c])
            for face in candidate_faces:
                tri = vertices[np.asarray(face)]
                edges = [np.linalg.norm(tri[1] - tri[0]), np.linalg.norm(tri[2] - tri[1]), np.linalg.norm(tri[0] - tri[2])]
                if max(edges) <= edge_limit and np.isfinite(tri).all() and np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 1e-10:
                    faces.append(face)
    face_array = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    uvs = source_uv(width, height).reshape(-1, 2)[valid_pixels]
    return vertices, face_array, uvs, valid_pixels


def colorize_ids(ids: np.ndarray) -> np.ndarray:
    normalized = np.zeros(ids.shape, np.float32)
    valid = ids >= 0
    if np.any(valid):
        normalized[valid] = (ids[valid] % 4096) / 4095.0
    return cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)


def main() -> None:
    points = np.load(ROOT / "points.npy").astype(np.float64)
    mask = np.load(ROOT / "mask.npy").astype(bool)
    intrinsics = np.load(ROOT / "intrinsics.npy").astype(np.float64)
    source = cv2.cvtColor(cv2.imread(str(ROOT / "source_rgb.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    source = cv2.resize(source, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(ROOT / "source_rgb_512.png"), cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
    if points.shape != (HEIGHT, WIDTH, 3) or mask.shape != (HEIGHT, WIDTH) or intrinsics.shape != (3, 3):
        raise RuntimeError("MOGE_ARRAY_SHAPE_INVALID")
    valid = mask & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    vertices, faces, uvs, valid_pixels = build_candidate(points, valid, 0.010)
    projected = project_points(vertices, intrinsics, WIDTH, HEIGHT)
    y, x = np.unravel_index(valid_pixels, (HEIGHT, WIDTH))
    expected = np.column_stack([x.astype(np.float64), y.astype(np.float64)])
    metrics = reprojection_metrics(projected, expected)
    rendered = np.zeros_like(source)
    rendered_mask = np.zeros((HEIGHT, WIDTH), bool)
    rounded = np.rint(projected).astype(np.int64)
    inside = (rounded[:, 0] >= 0) & (rounded[:, 0] < WIDTH) & (rounded[:, 1] >= 0) & (rounded[:, 1] < HEIGHT)
    rendered[rounded[inside, 1], rounded[inside, 0]] = source[y[inside], x[inside]]
    rendered_mask[rounded[inside, 1], rounded[inside, 0]] = True
    face_for_vertex = np.full(len(vertices), -1, np.int64)
    for face_id, face in enumerate(faces):
        for vertex in face:
            if face_for_vertex[vertex] < 0:
                face_for_vertex[vertex] = face_id
    triangle_ids = np.full((HEIGHT, WIDTH), -1, np.int64)
    triangle_ids[y, x] = face_for_vertex
    depth = np.zeros((HEIGHT, WIDTH), np.float32)
    depth[y, x] = vertices[:, 2].astype(np.float32)
    source_mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    source_mask[mask] = 255
    diff = np.abs(rendered.astype(np.int16) - source.astype(np.int16)).astype(np.uint8)
    diff[~rendered_mask] = 0
    coverage = float(np.count_nonzero(rendered_mask & mask) / max(np.count_nonzero(mask), 1))
    valid_source = np.count_nonzero(mask)
    classification = "MOGE_RAW_SOURCE_REPROJECTION_PROVEN" if metrics["median_px"] < 0.25 and metrics["p99_px"] < 1.0 and coverage >= 0.90 else "MOGE_RAW_SOURCE_REPROJECTION_REJECTED"
    cv2.imwrite(str(ROOT / "raw_reprojection_color.png"), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(ROOT / "raw_reprojection_triangle_id.png"), colorize_ids(triangle_ids))
    depth_vis = np.zeros_like(depth, np.uint8)
    if np.any(depth > 0):
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(ROOT / "raw_reprojection_depth.png"), cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO))
    cv2.imwrite(str(ROOT / "raw_reprojection_valid_mask.png"), source_mask)
    cv2.imwrite(str(ROOT / "raw_reprojection_diff.png"), cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
    receipt = {
        "schema": "moge_raw_source_reprojection_v1",
        "classification": classification,
        "arrays": {"points": str(ROOT / "points.npy"), "mask": str(ROOT / "mask.npy"), "intrinsics": str(ROOT / "intrinsics.npy"), "source_rgb_512": str(ROOT / "source_rgb_512.png")},
        "array_sha256": {name: sha256(ROOT / name) for name in ("points.npy", "mask.npy", "intrinsics.npy")},
        "camera": {"origin": [0.0, 0.0, 0.0], "forward": [0.0, 0.0, 1.0], "resolution": [WIDTH, HEIGHT], "intrinsics": intrinsics.tolist()},
        "candidate": {"name": "balanced_010", "threshold": 0.010, "vertices": int(len(vertices)), "triangles": int(len(faces)), "uv_policy": "u=x/(width-1),v=1-y/(height-1)"},
        "source_valid_pixels": int(valid_source),
        "projected_valid_pixels": int(np.count_nonzero(inside)),
        "mesh_incident_coverage": coverage,
        "reprojection_error_px": metrics,
        "color_error": {"mean_abs_rgb": float(diff[rendered_mask].mean()) if np.any(rendered_mask) else None, "max_abs_rgb": int(diff.max())},
        "artifacts": {name: str(ROOT / name) for name in ("raw_reprojection_color.png", "raw_reprojection_triangle_id.png", "raw_reprojection_depth.png", "raw_reprojection_valid_mask.png", "raw_reprojection_diff.png")},
    }
    write_json(ROOT / "raw_reprojection_receipt.json", receipt)
    write_json(PROOF / "raw_reprojection_receipt.json", receipt)


if __name__ == "__main__":
    main()
