"""CPU-only albedo preparation and surface-aware texture completion.

This worker is deliberately downstream of the proven projection/UV ownership
repair.  It never generates geometry, changes UVs, or relaxes source-view
visibility gates.

Two bounded operations are exposed:

``preprocess``
    Remove only low-frequency illumination from a registered source image.
    The alpha/foreground mask is immutable.  This is a lightweight intrinsic-
    image approximation intended to provide a cleaner albedo source to the
    existing strict face-ID projector.

``complete``
    Preserve every projected texel owned by an observed triangle and colour
    only unobserved triangles.  Colours propagate through a welded triangle
    graph whose edges are pruned at sharp normal/geometry discontinuities.
    This prevents Euclidean jumps between nearby but unrelated surfaces such
    as fur, rifle, backpack and clothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

WELD_TOLERANCE = 4e-4
DEFAULT_MIN_NORMAL_DOT = 0.45
DEFAULT_MAX_CENTROID_FRACTION = 0.035
DEFAULT_MAX_ITERATIONS = 128


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1.0 / 2.4) - 0.055)


def foreground_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("source image must be RGB/RGBA")
    if image.shape[2] == 4:
        return image[:, :, 3] > 0
    rgb = image[:, :, :3].astype(np.float32)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    return np.linalg.norm(rgb - background, axis=2) > 12.0


def low_frequency_delight(
    rgb_u8: np.ndarray,
    mask: np.ndarray,
    *,
    sigma_fraction: float = 0.075,
    minimum_gain: float = 0.60,
    maximum_gain: float = 1.85,
) -> tuple[np.ndarray, dict]:
    """Estimate albedo by removing only smooth luminance variation.

    A normalized Gaussian filter estimates log shading inside the foreground.
    Fine texture and chroma are retained; the correction is clamped so it
    cannot turn deep holes or invalid source pixels into bright texture.
    """
    rgb = np.asarray(rgb_u8, dtype=np.float32) / 255.0
    mask = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("RGB and mask dimensions do not match")
    if not mask.any():
        raise ValueError("foreground mask is empty")

    linear = srgb_to_linear(rgb)
    luminance = np.maximum(
        0.2126 * linear[:, :, 0] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 2],
        1e-4,
    )
    log_luminance = np.log(luminance)
    sigma = max(2.0, float(max(mask.shape)) * float(sigma_fraction))
    weights = mask.astype(np.float32)
    numerator = cv2.GaussianBlur(log_luminance * weights, (0, 0), sigma)
    denominator = cv2.GaussianBlur(weights, (0, 0), sigma)
    smooth_log = numerator / np.maximum(denominator, 1e-5)
    reference = float(np.median(smooth_log[mask]))
    gain = np.exp(reference - smooth_log)
    gain = np.clip(gain, float(minimum_gain), float(maximum_gain))

    corrected_linear = np.clip(linear * gain[..., None], 0.0, 1.0)
    corrected = np.clip(linear_to_srgb(corrected_linear), 0.0, 1.0)
    corrected[~mask] = rgb[~mask]

    before_luma = luminance[mask]
    after_linear = srgb_to_linear(corrected)
    after_luma = (
        0.2126 * after_linear[:, :, 0]
        + 0.7152 * after_linear[:, :, 1]
        + 0.0722 * after_linear[:, :, 2]
    )[mask]
    report = {
        "algorithm": "masked_low_frequency_log_luminance_delight_v1",
        "sigma_pixels": round(sigma, 3),
        "foreground_pixels": int(mask.sum()),
        "gain_min": round(float(gain[mask].min()), 6),
        "gain_median": round(float(np.median(gain[mask])), 6),
        "gain_max": round(float(gain[mask].max()), 6),
        "luminance_std_before": round(float(before_luma.std()), 6),
        "luminance_std_after": round(float(after_luma.std()), 6),
        "mask_preserved": True,
    }
    return np.round(corrected * 255.0).astype(np.uint8), report


def preprocess_source(input_path: Path, output_path: Path, report_path: Path) -> dict:
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"SOURCE_IMAGE_UNREADABLE:{input_path}")
    if image.shape[2] == 4:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = image[:, :, 3].copy()
    else:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = None
    mask = foreground_mask(image)
    corrected, details = low_frequency_delight(rgb, mask)
    bgr = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
    encoded = np.dstack([bgr, alpha]) if alpha is not None else bgr
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), encoded):
        raise RuntimeError(f"SOURCE_ALBEDO_WRITE_FAILED:{output_path}")
    decoded = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.shape != encoded.shape:
        raise RuntimeError("SOURCE_ALBEDO_ROUNDTRIP_FAILED")
    if alpha is not None and not np.array_equal(decoded[:, :, 3], alpha):
        raise RuntimeError("SOURCE_ALPHA_CHANGED")
    report = {
        "schema": "cpu_source_albedo_report_v1",
        "success": True,
        "input": str(input_path),
        "output": str(output_path),
        "input_sha256": _sha256(input_path),
        "output_sha256": _sha256(output_path),
        **details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def raster_triangle_ids(uvs: np.ndarray, atlas_size: int) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uvs, dtype=np.float64)
    if uv.ndim != 3 or uv.shape[1:] != (3, 2):
        raise ValueError(f"expected per-face UVs [T,3,2], got {uv.shape}")
    if not np.isfinite(uv).all() or np.any(uv < -1e-6) or np.any(uv > 1.0 + 1e-6):
        raise ValueError("UV coordinates are non-finite or outside [0,1]")
    pixel = uv.copy()
    pixel[:, :, 0] *= atlas_size - 1
    pixel[:, :, 1] = (1.0 - pixel[:, :, 1]) * (atlas_size - 1)
    triangle_id = np.full((atlas_size, atlas_size), -1, np.int32)
    occupied = np.zeros((atlas_size, atlas_size), np.uint8)
    for index, triangle in enumerate(pixel):
        polygon = np.round(triangle).astype(np.int32)
        cv2.fillConvexPoly(triangle_id, polygon, int(index), lineType=cv2.LINE_8)
        cv2.fillConvexPoly(occupied, polygon, 255, lineType=cv2.LINE_8)
    return triangle_id, occupied > 0


def triangle_mean_colours(
    atlas_rgb: np.ndarray, triangle_id: np.ndarray, triangle_count: int
) -> tuple[np.ndarray, np.ndarray]:
    ids = triangle_id.reshape(-1)
    valid = ids >= 0
    colours = atlas_rgb.reshape(-1, 3).astype(np.float64)
    count = np.bincount(ids[valid], minlength=triangle_count).astype(np.int64)
    result = np.zeros((triangle_count, 3), np.float32)
    for channel in range(3):
        sums = np.bincount(ids[valid], weights=colours[valid, channel], minlength=triangle_count)
        nonzero = count > 0
        result[nonzero, channel] = (sums[nonzero] / count[nonzero]).astype(np.float32)
    return result, count


def build_surface_graph(
    verts: np.ndarray,
    tris: np.ndarray,
    normals: np.ndarray,
    *,
    min_normal_dot: float = DEFAULT_MIN_NORMAL_DOT,
    max_centroid_fraction: float = DEFAULT_MAX_CENTROID_FRACTION,
) -> tuple[csr_matrix, dict]:
    vertices = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(tris, dtype=np.int64)
    face_normals = np.asarray(normals, dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(face_normals) != len(faces):
        raise ValueError("invalid triangle geometry arrays")

    quantised = np.round(vertices / WELD_TOLERANCE).astype(np.int64)
    _, welded = np.unique(quantised, axis=0, return_inverse=True)
    welded_faces = welded[faces]
    rows = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
    cols = welded_faces.reshape(-1)
    incidence = coo_matrix(
        (np.ones(rows.size, np.float32), (rows, cols)),
        shape=(len(faces), int(welded_faces.max()) + 1),
    ).tocsr()
    adjacency = (incidence @ incidence.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    coo = adjacency.tocoo()

    centroids = vertices[faces].mean(axis=1)
    scene_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    maximum_distance = max(scene_diag * float(max_centroid_fraction), 1e-8)
    dot = np.einsum("ij,ij->i", face_normals[coo.row], face_normals[coo.col])
    distance = np.linalg.norm(centroids[coo.row] - centroids[coo.col], axis=1)
    keep = np.isfinite(dot) & (dot >= float(min_normal_dot)) & (distance <= maximum_distance)
    scaled_dot = np.clip((dot[keep] - min_normal_dot) / max(1.0 - min_normal_dot, 1e-6), 0.0, 1.0)
    weight = np.maximum(scaled_dot, 1e-3) ** 2 / (1.0 + distance[keep] / maximum_distance)
    graph = coo_matrix(
        (weight.astype(np.float32), (coo.row[keep], coo.col[keep])),
        shape=adjacency.shape,
    ).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    graph.eliminate_zeros()
    region_count, _ = connected_components(graph, directed=False)
    return graph, {
        "triangle_count": int(len(faces)),
        "welded_vertex_count": int(welded.max()) + 1,
        "candidate_adjacency_edges": int(adjacency.nnz),
        "accepted_adjacency_edges": int(graph.nnz),
        "surface_region_count": int(region_count),
        "minimum_normal_dot": float(min_normal_dot),
        "maximum_centroid_distance": float(maximum_distance),
    }


def propagate_surface_colours(
    graph: csr_matrix,
    triangle_colours: np.ndarray,
    observed: np.ndarray,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    colours = np.asarray(triangle_colours, dtype=np.float32).copy()
    observed = np.asarray(observed, dtype=bool)
    if colours.shape != (graph.shape[0], 3) or observed.shape != (graph.shape[0],):
        raise ValueError("colour/observation arrays do not match graph")
    if not observed.any():
        raise ValueError("no observed triangle colours")

    known = observed.copy()
    fill_iteration = np.full(graph.shape[0], -1, np.int16)
    fill_iteration[observed] = 0
    iterations = 0
    for iteration in range(1, max(int(max_iterations), 1) + 1):
        neighbour_weight = graph @ known.astype(np.float32)
        new = (~known) & (neighbour_weight > 1e-8)
        if not new.any():
            break
        weighted = graph @ colours
        colours[new] = weighted[new] / neighbour_weight[new, None]
        known[new] = True
        fill_iteration[new] = iteration
        iterations = iteration

    region_count, regions = connected_components(graph, directed=False)
    global_prior = np.median(colours[observed], axis=0)
    unresolved = ~known
    for region in range(region_count):
        targets = unresolved & (regions == region)
        if not targets.any():
            continue
        donors = observed & (regions == region)
        colours[targets] = np.median(colours[donors], axis=0) if donors.any() else global_prior
        fill_iteration[targets] = max(iterations + 1, 1)
        known[targets] = True

    return colours, fill_iteration, {
        "iterations": int(iterations),
        "observed_triangles": int(observed.sum()),
        "propagated_triangles": int(np.count_nonzero(fill_iteration > 0)),
        "remaining_unresolved": int(np.count_nonzero(~known)),
        "maximum_fill_iteration": int(fill_iteration.max()),
    }


def _ownership_is_zero(report: dict) -> bool:
    preferred = (
        "after_positive_overlap_pair_count",
        "remaining_positive_overlap_pair_count",
        "positive_overlap_pair_count_after",
        "positive_overlap_pair_count",
    )
    for key in preferred:
        if key in report:
            return int(report[key]) == 0
    after = report.get("after")
    if isinstance(after, dict):
        for key in ("positive_overlap_pair_count", "pair_count", "conflict_count"):
            if key in after:
                return int(after[key]) == 0
    return False


def complete_atlas(
    npz_path: Path,
    basecolor_path: Path,
    observed_path: Path,
    ownership_report_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    min_normal_dot: float = DEFAULT_MIN_NORMAL_DOT,
) -> dict:
    ownership = json.loads(ownership_report_path.read_text(encoding="utf-8"))
    if not _ownership_is_zero(ownership):
        raise RuntimeError("UV_OWNERSHIP_ZERO_CONFLICT_NOT_PROVEN")

    data = np.load(npz_path)
    verts = np.asarray(data["verts"], dtype=np.float32)
    tris = np.asarray(data["tris"], dtype=np.int32)
    uvs = np.asarray(data["uvs"], dtype=np.float32)
    normals = np.asarray(data["normals"], dtype=np.float32)
    observed = np.load(observed_path).astype(bool)
    if len(tris) != len(uvs) or observed.shape != (len(tris),):
        raise RuntimeError("TRIANGLE_ARRAY_IDENTITY_MISMATCH")

    atlas_bgr = cv2.imread(str(basecolor_path), cv2.IMREAD_COLOR)
    if atlas_bgr is None or atlas_bgr.shape[0] != atlas_bgr.shape[1]:
        raise RuntimeError("BASECOLOR_ATLAS_INVALID")
    atlas = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGB)
    triangle_id, occupied = raster_triangle_ids(uvs, atlas.shape[0])
    triangle_colours, pixel_count = triangle_mean_colours(atlas, triangle_id, len(tris))
    usable_observed = observed & (pixel_count > 0)

    graph, graph_report = build_surface_graph(
        verts, tris, normals, min_normal_dot=min_normal_dot
    )
    completed_colours, fill_iteration, propagation_report = propagate_surface_colours(
        graph, triangle_colours, usable_observed, max_iterations=max_iterations
    )

    result = atlas.copy()
    owner = triangle_id >= 0
    owner_ids = triangle_id[owner]
    synthesized_pixels = owner & (~usable_observed[np.clip(triangle_id, 0, len(tris) - 1)])
    result[synthesized_pixels] = np.clip(
        completed_colours[triangle_id[synthesized_pixels]], 0.0, 255.0
    ).astype(np.uint8)
    # Pixels belonging to observed triangles are never changed.
    observed_pixels = owner & usable_observed[np.clip(triangle_id, 0, len(tris) - 1)]
    if not np.array_equal(result[observed_pixels], atlas[observed_pixels]):
        raise RuntimeError("OBSERVED_TEXELS_CHANGED")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("COMPLETED_ATLAS_WRITE_FAILED")
    report = {
        "schema": "surface_albedo_completion_report_v1",
        "success": True,
        "npz": str(npz_path),
        "basecolor": str(basecolor_path),
        "observed_triangles": str(observed_path),
        "ownership_report": str(ownership_report_path),
        "output": str(output_path),
        "input_basecolor_sha256": _sha256(basecolor_path),
        "output_sha256": _sha256(output_path),
        "uv_ownership_zero_conflict": True,
        "observed_texels_preserved": True,
        "atlas_resolution": int(atlas.shape[0]),
        "occupied_pixels": int(occupied.sum()),
        "observed_pixels": int(observed_pixels.sum()),
        "synthesized_pixels": int(synthesized_pixels.sum()),
        "triangle_pixel_owners": int(np.count_nonzero(pixel_count)),
        "fill_iteration_histogram": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(fill_iteration, return_counts=True))
        },
        "graph": graph_report,
        "propagation": propagation_report,
        "policy": (
            "preserve_observed_texels; welded_surface_graph; curvature_and_distance_gates; "
            "no_uv_change; no_geometry_change; no_front_fallback"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--output", required=True)
    preprocess.add_argument("--report", required=True)

    complete = subparsers.add_parser("complete")
    complete.add_argument("--npz", required=True)
    complete.add_argument("--basecolor", required=True)
    complete.add_argument("--observed-triangles", required=True)
    complete.add_argument("--ownership-report", required=True)
    complete.add_argument("--output", required=True)
    complete.add_argument("--report", required=True)
    complete.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    complete.add_argument("--min-normal-dot", type=float, default=DEFAULT_MIN_NORMAL_DOT)

    args = parser.parse_args()
    if args.command == "preprocess":
        result = preprocess_source(Path(args.input), Path(args.output), Path(args.report))
    else:
        result = complete_atlas(
            Path(args.npz),
            Path(args.basecolor),
            Path(args.observed_triangles),
            Path(args.ownership_report),
            Path(args.output),
            Path(args.report),
            max_iterations=args.max_iterations,
            min_normal_dot=args.min_normal_dot,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
