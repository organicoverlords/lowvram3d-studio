"""Deterministic comparison of a reduced candidate against a clean high-resolution master."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .quality_ladder import (
    AssetFamily,
    CandidateEvaluation,
    evaluate_candidate,
    thresholds_for,
)


@dataclass(frozen=True, slots=True)
class SurfaceSamples:
    points: np.ndarray
    normals: np.ndarray
    face_ids: np.ndarray


VIEW_DIRECTIONS = np.asarray(
    [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.55),
        (-1.0, 1.0, 0.55),
        (1.0, -1.0, 0.55),
        (-1.0, -1.0, 0.55),
        (1.0, 1.0, -0.55),
        (-1.0, 1.0, -0.55),
        (1.0, -1.0, -0.55),
        (-1.0, -1.0, -0.55),
    ],
    dtype=np.float64,
)
VIEW_DIRECTIONS /= np.linalg.norm(VIEW_DIRECTIONS, axis=1, keepdims=True)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        try:
            mesh = loaded.to_geometry()
        except Exception:
            mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"mesh contains no triangles: {path}")
    if mesh.faces.shape[1] != 3:
        raise ValueError("comparison requires triangular meshes")
    return mesh


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> SurfaceSamples:
    if count <= 0:
        raise ValueError("sample count must be positive")
    areas = np.asarray(mesh.area_faces, np.float64)
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("mesh has no finite positive surface area")
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(mesh.faces), size=count, replace=True, p=areas / total)
    triangles = np.asarray(mesh.triangles, np.float64)[face_ids]
    random_uv = rng.random((count, 2))
    root = np.sqrt(random_uv[:, 0])
    barycentric = np.column_stack((1.0 - root, root * (1.0 - random_uv[:, 1]), root * random_uv[:, 1]))
    points = np.einsum("ni,nij->nj", barycentric, triangles)
    normals = np.asarray(mesh.face_normals, np.float64)[face_ids]
    length = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(length, 1e-12)[:, None]
    return SurfaceSamples(points=points, normals=normals, face_ids=face_ids)


def sample_face_subset(
    mesh: trimesh.Trimesh,
    face_subset: np.ndarray,
    count: int,
    seed: int,
) -> SurfaceSamples:
    """Area-weighted surface sampling restricted to a subset of a mesh's faces.

    Numerically identical to sampling the equivalent ``mesh.submesh([face_subset], append=True)``,
    but it never materialises a submesh. ``trimesh.util.submesh`` allocates an index array the
    size of the *whole* source mesh on every call, so per-component sampling on a high-resolution
    master costs O(components x total_vertices) memory churn -- 223,679 components against a
    750k-vertex master is over a terabyte of transient allocation, which exhausts the heap long
    before the audit finishes. Cost here scales with the subset instead.

    Returned ``face_ids`` are global indices into ``mesh.faces``.
    """
    if count <= 0:
        raise ValueError("sample count must be positive")
    subset = np.asarray(face_subset, np.int64).reshape(-1)
    if subset.size == 0:
        raise ValueError("face subset is empty")
    areas = np.asarray(mesh.area_faces, np.float64)[subset]
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("face subset has no finite positive surface area")
    rng = np.random.default_rng(seed)
    local_ids = rng.choice(subset.size, size=count, replace=True, p=areas / total)
    global_ids = subset[local_ids]
    triangles = np.asarray(mesh.triangles, np.float64)[global_ids]
    random_uv = rng.random((count, 2))
    root = np.sqrt(random_uv[:, 0])
    barycentric = np.column_stack((1.0 - root, root * (1.0 - random_uv[:, 1]), root * random_uv[:, 1]))
    points = np.einsum("ni,nij->nj", barycentric, triangles)
    normals = np.asarray(mesh.face_normals, np.float64)[global_ids]
    length = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(length, 1e-12)[:, None]
    return SurfaceSamples(points=points, normals=normals, face_ids=global_ids)


def topology_counts(mesh: trimesh.Trimesh) -> dict[str, int]:
    faces = np.asarray(mesh.faces, np.int64)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    maximum_vertex = int(edges.max(initial=0))
    if maximum_vertex < 2**32:
        keys = (edges[:, 0].astype(np.uint64) << np.uint64(32)) | edges[:, 1].astype(np.uint64)
        _, counts = np.unique(keys, return_counts=True)
    else:
        _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "faces": int(len(faces)),
        "boundary_edges": int((counts == 1).sum()),
        "non_manifold_edges": int((counts > 2).sum()),
    }


def _camera_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = np.asarray(direction, np.float64)
    forward /= max(np.linalg.norm(forward), 1e-12)
    up_hint = np.asarray((0.0, 0.0, 1.0), np.float64)
    if abs(float(np.dot(forward, up_hint))) > 0.92:
        up_hint = np.asarray((0.0, 1.0, 0.0), np.float64)
    right = np.cross(up_hint, forward)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(forward, right)
    return right, up


def silhouette_mask(
    points: np.ndarray,
    *,
    center: np.ndarray,
    half_extent: float,
    direction: np.ndarray,
    size: int,
) -> np.ndarray:
    right, up = _camera_basis(direction)
    relative = points - center
    x = relative @ right
    y = relative @ up
    px = np.rint((x / half_extent * 0.5 + 0.5) * (size - 1)).astype(np.int32)
    py = np.rint((1.0 - (y / half_extent * 0.5 + 0.5)) * (size - 1)).astype(np.int32)
    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    mask = np.zeros((size, size), np.uint8)
    mask[py[valid], px[valid]] = 255
    # Dense deterministic point splats are substantially cheaper than rasterising millions of
    # triangles for every candidate.  Closing fills sampling pinholes without merging distant parts.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def silhouette_metrics(
    master_points: np.ndarray,
    candidate_points: np.ndarray,
    *,
    center: np.ndarray,
    half_extent: float,
    size: int,
) -> dict:
    per_view = []
    for index, direction in enumerate(VIEW_DIRECTIONS):
        master = silhouette_mask(
            master_points, center=center, half_extent=half_extent, direction=direction, size=size
        )
        candidate = silhouette_mask(
            candidate_points, center=center, half_extent=half_extent, direction=direction, size=size
        )
        master_bool = master > 0
        candidate_bool = candidate > 0
        union = int((master_bool | candidate_bool).sum())
        intersection = int((master_bool & candidate_bool).sum())
        iou = intersection / max(union, 1)

        boundary = master_bool & ~(cv2.erode(master, np.ones((3, 3), np.uint8), iterations=1) > 0)
        distance_to_candidate = cv2.distanceTransform((~candidate_bool).astype(np.uint8), cv2.DIST_L2, 3)
        boundary_count = int(boundary.sum())
        boundary_recall = (
            float((distance_to_candidate[boundary] <= 2.0).sum() / boundary_count)
            if boundary_count
            else 1.0
        )
        per_view.append(
            {
                "view": index,
                "direction": [float(value) for value in direction],
                "iou": float(iou),
                "thin_boundary_recall": boundary_recall,
                "master_pixels": int(master_bool.sum()),
                "candidate_pixels": int(candidate_bool.sum()),
            }
        )
    return {
        "per_view": per_view,
        "iou_min": min(item["iou"] for item in per_view),
        "iou_mean": float(np.mean([item["iou"] for item in per_view])),
        "thin_feature_recall_min": min(item["thin_boundary_recall"] for item in per_view),
        "thin_feature_recall_mean": float(np.mean([item["thin_boundary_recall"] for item in per_view])),
    }


def meaningful_component_recall(
    master: trimesh.Trimesh,
    candidate_tree: cKDTree,
    *,
    model_diagonal: float,
    distance_limit_diag: float,
    seed: int,
) -> tuple[float, list[dict]]:
    try:
        components = master.split(only_watertight=False)
    except Exception as exc:
        return 0.0, [{"error": f"component split failed: {exc}"}]
    total_area = max(float(master.area), 1e-12)
    meaningful = [
        component
        for component in components
        if len(component.faces) >= 32 and float(component.area) / total_area >= 0.0005
    ]
    if not meaningful:
        return 1.0, []
    records = []
    retained = 0
    for index, component in enumerate(meaningful):
        area_fraction = float(component.area) / total_area
        samples = min(4096, max(256, round(50_000 * area_fraction)))
        points = sample_surface(component, samples, seed + 10_000 + index).points
        distances, _ = candidate_tree.query(points, k=1, workers=-1)
        p95 = float(np.percentile(distances, 95)) / model_diagonal
        ok = p95 <= distance_limit_diag
        retained += int(ok)
        records.append(
            {
                "component": index,
                "faces": int(len(component.faces)),
                "area_fraction": area_fraction,
                "distance_p95_diag": p95,
                "retained": ok,
            }
        )
    return retained / len(meaningful), records


def compare_meshes(
    master_path: str | Path,
    candidate_path: str | Path,
    *,
    asset_family: AssetFamily,
    quality: str,
    sample_count: int = 200_000,
    silhouette_size: int = 384,
    seed: int = 0,
    candidate_name: str = "candidate",
) -> dict:
    master = load_mesh(master_path)
    candidate = load_mesh(candidate_path)
    master_bounds = np.asarray(master.bounds, np.float64)
    center = master_bounds.mean(axis=0)
    extent = master_bounds[1] - master_bounds[0]
    model_diagonal = max(float(np.linalg.norm(extent)), 1e-12)
    half_extent = max(float(extent.max()) * 0.58, 1e-9)

    master_samples = sample_surface(master, sample_count, seed)
    candidate_samples = sample_surface(candidate, sample_count, seed + 1)
    candidate_tree = cKDTree(candidate_samples.points)
    master_tree = cKDTree(master_samples.points)

    forward_distance, nearest_candidate = candidate_tree.query(master_samples.points, k=1, workers=-1)
    reverse_distance, _ = master_tree.query(candidate_samples.points, k=1, workers=-1)
    nearest_normals = candidate_samples.normals[np.asarray(nearest_candidate, np.int64)]
    dot = np.einsum("ij,ij->i", master_samples.normals, nearest_normals)
    normal_degrees = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    silhouettes = silhouette_metrics(
        master_samples.points,
        candidate_samples.points,
        center=center,
        half_extent=half_extent,
        size=silhouette_size,
    )
    thresholds = thresholds_for(asset_family, quality)
    component_recall, component_records = meaningful_component_recall(
        master,
        candidate_tree,
        model_diagonal=model_diagonal,
        distance_limit_diag=thresholds.surface_distance_p99_diag,
        seed=seed,
    )
    before = topology_counts(master)
    after = topology_counts(candidate)

    evaluation = CandidateEvaluation(
        name=candidate_name,
        face_count=int(len(candidate.faces)),
        silhouette_iou_min=float(silhouettes["iou_min"]),
        surface_distance_p95_diag=float(np.percentile(forward_distance, 95)) / model_diagonal,
        surface_distance_p99_diag=float(np.percentile(forward_distance, 99)) / model_diagonal,
        reverse_distance_p95_diag=float(np.percentile(reverse_distance, 95)) / model_diagonal,
        normal_deviation_p95_deg=float(np.percentile(normal_degrees, 95)),
        thin_feature_recall=float(silhouettes["thin_feature_recall_min"]),
        meaningful_component_recall=float(component_recall),
        boundary_edges_before=before["boundary_edges"],
        boundary_edges_after=after["boundary_edges"],
        non_manifold_before=before["non_manifold_edges"],
        non_manifold_after=after["non_manifold_edges"],
    )
    evaluate_candidate(evaluation, thresholds)
    return {
        "success": evaluation.valid,
        "master": str(master_path),
        "candidate": str(candidate_path),
        "asset_family": asset_family.value,
        "quality": quality,
        "sample_count_each_direction": sample_count,
        "silhouette_size": silhouette_size,
        "view_count": len(VIEW_DIRECTIONS),
        "model_diagonal": model_diagonal,
        "thresholds": {
            field: getattr(thresholds, field)
            for field in thresholds.__dataclass_fields__
        },
        "evaluation": evaluation.as_dict(),
        "surface_distance": {
            "source_to_candidate_median_diag": float(np.median(forward_distance)) / model_diagonal,
            "source_to_candidate_p95_diag": float(np.percentile(forward_distance, 95)) / model_diagonal,
            "source_to_candidate_p99_diag": float(np.percentile(forward_distance, 99)) / model_diagonal,
            "source_to_candidate_max_diag": float(np.max(forward_distance)) / model_diagonal,
            "candidate_to_source_median_diag": float(np.median(reverse_distance)) / model_diagonal,
            "candidate_to_source_p95_diag": float(np.percentile(reverse_distance, 95)) / model_diagonal,
            "candidate_to_source_p99_diag": float(np.percentile(reverse_distance, 99)) / model_diagonal,
            "candidate_to_source_max_diag": float(np.max(reverse_distance)) / model_diagonal,
        },
        "normal_deviation": {
            "median_deg": float(np.median(normal_degrees)),
            "p95_deg": float(np.percentile(normal_degrees, 95)),
            "p99_deg": float(np.percentile(normal_degrees, 99)),
        },
        "silhouette": silhouettes,
        "meaningful_components": {
            "recall": component_recall,
            "components": component_records,
        },
        "topology": {"master": before, "candidate": after},
    }
