"""Second bounded panda face-surface candidate pass.

The generic raycaster and atlas patcher remain character-agnostic. This worker
accepts an external source-image fixture, uses border-connected background
removal instead of whole-object centroid heuristics, welds duplicated mesh
vertices for surface connectivity, records patch scores before selection, and
builds at most one candidate from the immutable 2048 baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .conservative_atlas import derive_uv_chart_ids
    from .face_patch_texture import build_face_patch_atlas
    from .face_surface_candidate import anchors_from_layers, camera_from_projection_fit, camera_to_dict
    from .face_surface_ownership import trace_mask_layers
    from .fast_texture_projection import bind_texture, fit_camera, immutable_buffer_hashes
    from .mesh_io import read_glb
except ImportError:  # pragma: no cover
    from conservative_atlas import derive_uv_chart_ids
    from face_patch_texture import build_face_patch_atlas
    from face_surface_candidate import anchors_from_layers, camera_from_projection_fit, camera_to_dict
    from face_surface_ownership import trace_mask_layers
    from fast_texture_projection import bind_texture, fit_camera, immutable_buffer_hashes
    from mesh_io import read_glb

EPS = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def border_connected_foreground(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Separate a near-white canvas without deleting enclosed white fur.

    Only near-background pixels connected to an image border become background.
    Bright object regions enclosed by darker fur remain foreground. The largest
    foreground component is retained and hole-filled for camera fitting.
    """

    image = np.asarray(image_rgb, np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("SOURCE_IMAGE_MUST_BE_RGB")
    height, width = image.shape[:2]
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0).astype(np.float32)
    reference = np.median(border, axis=0)
    values = image.astype(np.float32)
    distance = np.linalg.norm(values - reference[None, None, :], axis=2)
    chroma = values.max(axis=2) - values.min(axis=2)
    near_white = (values.min(axis=2) >= 210.0) & (chroma <= 34.0) & (distance <= 42.0)

    count, labels = cv2.connectedComponents(near_white.astype(np.uint8), connectivity=8)
    touching = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    touching = touching[touching != 0]
    background = np.isin(labels, touching)
    foreground = ~background

    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        raise RuntimeError("SOURCE_FOREGROUND_NOT_FOUND")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    foreground = component_labels == largest
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    ) > 0

    inverse = (~foreground).astype(np.uint8)
    flood = inverse.copy()
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 2)
    holes = flood == 1
    foreground |= holes

    soft = cv2.GaussianBlur(foreground.astype(np.float32), (0, 0), 1.15)
    alpha = np.clip(soft, 0.0, 1.0)
    alpha[foreground] = np.maximum(alpha[foreground], 0.92)
    report = {
        "background_reference_rgb": reference.tolist(),
        "near_white_candidate_pixels": int(near_white.sum()),
        "border_background_pixels": int(background.sum()),
        "foreground_pixels": int(foreground.sum()),
        "largest_component_area": int(stats[largest, cv2.CC_STAT_AREA]),
    }
    return foreground, alpha, report


def load_source_fixture(path: Path, source_path: Path, image_shape: tuple[int, int]) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = str(fixture.get("source_image_sha256", "")).lower()
    actual_hash = sha256(source_path)
    if expected_hash != actual_hash:
        raise RuntimeError(f"SOURCE_FIXTURE_HASH_MISMATCH:{actual_hash}")
    expected_size = tuple(int(value) for value in fixture.get("image_size", []))
    actual_size = (int(image_shape[1]), int(image_shape[0]))
    if expected_size != actual_size:
        raise RuntimeError(f"SOURCE_FIXTURE_SIZE_MISMATCH:{actual_size}")
    polygon = np.asarray(fixture.get("face_polygon"), np.int32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("SOURCE_FIXTURE_FACE_POLYGON_INVALID")
    landmarks = fixture.get("landmarks")
    if not isinstance(landmarks, list) or len(landmarks) < 3:
        raise ValueError("SOURCE_FIXTURE_LANDMARKS_INVALID")
    for record in landmarks:
        point = np.asarray(record.get("source_xy"), np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("SOURCE_FIXTURE_LANDMARK_INVALID")
    return fixture


def fixture_face_mask(shape: tuple[int, int], fixture: dict[str, Any]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    polygon = np.asarray(fixture["face_polygon"], np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    if int(mask.sum()) < 512:
        raise RuntimeError("SOURCE_FIXTURE_FACE_MASK_TOO_SMALL")
    return mask > 0


def welded_surface_patches(positions: np.ndarray, triangles: np.ndarray,
                           candidate_ids: np.ndarray, weld_tolerance: float | None = None) -> list[np.ndarray]:
    """Connect triangles through complete edges after deterministic position welding."""

    positions = np.asarray(positions, np.float64)
    triangles = np.asarray(triangles, np.int64)
    candidate_ids = np.unique(np.asarray(candidate_ids, np.int64))
    if candidate_ids.size == 0:
        return []
    diagonal = float(np.linalg.norm(np.ptp(positions, axis=0)))
    tolerance = float(weld_tolerance if weld_tolerance is not None else max(diagonal * 2e-4, 1e-6))
    referenced = np.unique(triangles[candidate_ids].reshape(-1))
    quantized: dict[int, tuple[int, int, int]] = {}
    values = np.rint(positions[referenced] / tolerance).astype(np.int64)
    for vertex_id, key in zip(referenced.tolist(), values.tolist(), strict=True):
        quantized[int(vertex_id)] = (int(key[0]), int(key[1]), int(key[2]))

    adjacency: dict[int, set[int]] = {int(triangle_id): set() for triangle_id in candidate_ids}
    edge_owner: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = {}
    for triangle_id in candidate_ids.tolist():
        vertices = triangles[triangle_id]
        keys = [quantized[int(vertex)] for vertex in vertices]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            a, b = keys[first], keys[second]
            edge = (a, b) if a <= b else (b, a)
            other = edge_owner.get(edge)
            if other is None:
                edge_owner[edge] = int(triangle_id)
            elif other != triangle_id:
                adjacency[int(triangle_id)].add(other)
                adjacency[other].add(int(triangle_id))

    remaining = set(adjacency)
    patches: list[np.ndarray] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        patch: list[int] = []
        while stack:
            current = stack.pop()
            patch.append(current)
            for neighbour in adjacency[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        patches.append(np.asarray(sorted(patch), np.int64))
    return patches


def hit_ranks(offsets: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets, np.int64)
    result = np.empty(int(offsets[-1]), np.int16)
    for ray in range(len(offsets) - 1):
        result[offsets[ray]:offsets[ray + 1]] = np.arange(offsets[ray + 1] - offsets[ray])
    return result


def score_welded_patches(layers: dict[str, np.ndarray], positions: np.ndarray,
                         triangles: np.ndarray, chart_ids: np.ndarray,
                         landmarks: list[dict[str, Any]], max_rank: int = 10,
                         minimum_facing: float = -0.25) -> tuple[list[dict[str, Any]], np.ndarray]:
    triangle_ids = np.asarray(layers["triangle_ids"], np.int64)
    offsets = np.asarray(layers["offsets"], np.int64)
    pixels = np.asarray(layers["pixels_xy"], np.float64)
    depths = np.asarray(layers["depth"], np.float64)
    facing = np.asarray(layers["normal_facing"], np.float64)
    ranks = hit_ranks(offsets)
    eligible = (ranks < max_rank) & (facing >= minimum_facing)
    candidates = np.unique(triangle_ids[eligible])
    patches = welded_surface_patches(positions, triangles, candidates)
    triangle_to_patch = np.full(len(triangles), -1, np.int32)
    for patch_id, patch in enumerate(patches):
        triangle_to_patch[patch] = patch_id
    ray_for_hit = np.repeat(np.arange(len(pixels), dtype=np.int64), np.diff(offsets))
    hit_patch = triangle_to_patch[triangle_ids]
    landmark_xy = np.asarray([record["source_xy"] for record in landmarks], np.float64)
    image_diagonal = float(np.hypot(np.ptp(pixels[:, 0]) + 1.0, np.ptp(pixels[:, 1]) + 1.0))
    landmark_limit = max(24.0, image_diagonal * 0.085)
    records: list[dict[str, Any]] = []

    for patch_id, patch in enumerate(patches):
        selected_hits = (hit_patch == patch_id) & eligible
        if not np.any(selected_hits):
            continue
        ray_ids = np.unique(ray_for_hit[selected_hits])
        patch_pixels = pixels[ray_ids]
        distances = np.linalg.norm(patch_pixels[:, None, :] - landmark_xy[None, :, :], axis=2)
        landmark_min = np.min(distances, axis=0)
        landmark_support_count = int(np.count_nonzero(landmark_min <= landmark_limit))
        coverage = float(len(ray_ids) / max(len(pixels), 1))
        patch_depth = depths[selected_hits]
        patch_rank = ranks[selected_hits]
        patch_facing = facing[selected_hits]
        depth_median = float(np.median(patch_depth))
        depth_mad = float(np.median(np.abs(patch_depth - depth_median)))
        rank_mean = float(np.mean(patch_rank))
        facing_mean = float(np.mean(patch_facing))
        score = (
            6.0 * (landmark_support_count / max(len(landmarks), 1))
            + 4.0 * coverage
            + 0.7 * np.clip(facing_mean, -1.0, 1.0)
            + 1.0 / (1.0 + rank_mean)
            - 2.5 * depth_mad
        )
        records.append({
            "patch_id": int(patch_id),
            "triangle_ids": patch,
            "triangle_count": int(len(patch)),
            "chart_count": int(len(np.unique(chart_ids[patch]))),
            "ray_count": int(len(ray_ids)),
            "ray_coverage": coverage,
            "landmark_support_count": landmark_support_count,
            "landmark_min_distances": landmark_min.tolist(),
            "landmark_limit_px": float(landmark_limit),
            "depth_median": depth_median,
            "depth_mad": depth_mad,
            "rank_mean": rank_mean,
            "facing_mean": facing_mean,
            "score": float(score),
        })
    records.sort(key=lambda row: (-row["score"], -row["landmark_support_count"],
                                  -row["ray_coverage"], row["patch_id"]))
    return records, triangle_to_patch


def select_patch_union(records: list[dict[str, Any]], landmark_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    if not records:
        raise RuntimeError("FACE_SURFACE_NO_WELDED_PATCHES")
    primary = records[0]
    required_landmarks = max(4, int(np.ceil(landmark_count * 0.57)))
    selected_records = [primary]
    depth_window = max(0.025, 4.0 * float(primary["depth_mad"]) + 0.015)
    for record in records[1:]:
        if record["landmark_support_count"] <= 0:
            continue
        if abs(float(record["depth_median"]) - float(primary["depth_median"])) > depth_window:
            continue
        if float(record["ray_coverage"]) < 0.002:
            continue
        selected_records.append(record)
    selected_ids = np.unique(np.concatenate([record["triangle_ids"] for record in selected_records]))
    combined_landmarks = max(int(record["landmark_support_count"]) for record in selected_records)
    combined_coverage = min(1.0, sum(float(record["ray_coverage"]) for record in selected_records))
    if combined_landmarks < required_landmarks or combined_coverage < 0.025:
        raise RuntimeError(
            f"FACE_SURFACE_PATCH_NOT_PROVEN:landmarks={combined_landmarks}/{landmark_count};coverage={combined_coverage:.6f}"
        )
    decision = {
        "primary_patch_id": int(primary["patch_id"]),
        "selected_patch_ids": [int(record["patch_id"]) for record in selected_records],
        "selected_triangle_count": int(len(selected_ids)),
        "landmark_support_count": combined_landmarks,
        "required_landmark_support": required_landmarks,
        "combined_ray_coverage_upper_bound": combined_coverage,
        "depth_window": float(depth_window),
    }
    return selected_ids, decision


def serializable_records(records: list[dict[str, Any]], limit: int = 128) -> list[dict[str, Any]]:
    output = []
    for record in records[:limit]:
        row = dict(record)
        row["triangle_ids"] = np.asarray(record["triangle_ids"], np.int64).tolist()
        output.append(row)
    return output


def build_candidate(baseline_glb: Path, baseline_atlas: Path, source_image: Path,
                    source_fixture: Path, output_dir: Path, ray_stride: int = 3,
                    minimum_alpha: float = 0.30) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    positions, normals, uv, triangles = read_glb(baseline_glb)
    if uv is None:
        raise RuntimeError("FACE_CANDIDATE_REQUIRES_UV")
    source_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    atlas_bgr = cv2.imread(str(baseline_atlas), cv2.IMREAD_COLOR)
    if source_bgr is None or atlas_bgr is None:
        raise RuntimeError("FACE_CANDIDATE_IMAGE_MISSING")
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    baseline_rgb = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGB)
    fixture = load_source_fixture(source_fixture, source_image, source_rgb.shape[:2])
    foreground, source_alpha, foreground_report = border_connected_foreground(source_rgb)
    face_mask = fixture_face_mask(source_rgb.shape[:2], fixture)
    source_alpha[face_mask] = 1.0
    landmarks = fixture["landmarks"]

    cv2.imwrite(str(output_dir / "foreground_mask.png"), foreground.astype(np.uint8) * 255)
    cv2.imwrite(str(output_dir / "face_mask.png"), face_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output_dir / "source_alpha.png"), np.rint(source_alpha * 255.0).astype(np.uint8))
    np.save(output_dir / "face_mask.npy", face_mask)
    (output_dir / "source_fixture_used.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    (output_dir / "foreground_report.json").write_text(json.dumps(foreground_report, indent=2), encoding="utf-8")

    camera_fit = fit_camera(np.asarray(positions, np.float64), np.asarray(triangles, np.int64), foreground)
    camera = camera_from_projection_fit(camera_fit, positions, source_rgb.shape[1], source_rgb.shape[0])
    camera_payload = camera_to_dict(camera, camera_fit)
    (output_dir / "camera_contract.json").write_text(json.dumps(camera_payload, indent=2), encoding="utf-8")

    chart_ids, chart_inventory = derive_uv_chart_ids(uv, triangles)
    layers = trace_mask_layers(
        positions, normals, triangles, camera, face_mask,
        stride=ray_stride, max_hits=24, leaf_size=16,
    )
    np.savez_compressed(output_dir / "all_ray_hits.npz", **layers)
    records, _triangle_to_patch = score_welded_patches(
        layers, positions, triangles, chart_ids, landmarks,
    )
    (output_dir / "welded_patch_scores.json").write_text(
        json.dumps(serializable_records(records), indent=2), encoding="utf-8"
    )
    try:
        selected_ids, selection = select_patch_union(records, len(landmarks))
    except Exception as error:
        failure = {
            "schema": "face_surface_selection_failure_v1",
            "error_type": type(error).__name__,
            "error": str(error),
            "ray_count": int(len(layers["pixels_xy"])),
            "hit_count": int(len(layers["triangle_ids"])),
            "patch_count": int(len(records)),
            "top_patch": serializable_records(records, limit=1),
        }
        (output_dir / "selection_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise

    np.save(output_dir / "selected_face_triangles.npy", selected_ids)
    (output_dir / "selection_decision.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    anchors = anchors_from_layers(layers, selected_ids, landmarks, neighbour_rays=96)
    (output_dir / "auto_target_anchors.json").write_text(json.dumps(anchors, indent=2), encoding="utf-8")

    target_points = []
    source_points = []
    for anchor in anchors:
        bary = np.asarray(anchor["barycentric"], np.float64)
        point_3d = bary @ positions[triangles[int(anchor["triangle_id"])]]
        target_xy, _ = camera.project(point_3d[None, :])
        target_points.append(target_xy[0])
        source_points.append(anchor["source_xy"])

    candidate_atlas, texture_report, writable = build_face_patch_atlas(
        baseline_rgb, source_rgb, source_alpha, positions, uv, triangles,
        selected_ids, camera, np.asarray(target_points), np.asarray(source_points),
        minimum_alpha=minimum_alpha, tps_regularization=1e-3,
    )
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(candidate_atlas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("FACE_CANDIDATE_ATLAS_ENCODE_FAILED")
    atlas_path = output_dir / "atlas_face_surface_owned_2048.png"
    glb_path = output_dir / "panda_face_surface_owned_2048.glb"
    atlas_path.write_bytes(encoded.tobytes())
    before_hashes = immutable_buffer_hashes(baseline_glb)
    bind_texture(baseline_glb, glb_path, encoded.tobytes(), textured_triangles=np.ones(len(triangles), bool))
    after_hashes = immutable_buffer_hashes(glb_path)
    if before_hashes != after_hashes:
        raise RuntimeError("FACE_CANDIDATE_GEOMETRY_UV_INDEX_CHANGED")
    non_face_changed = int(np.any(candidate_atlas[~writable] != baseline_rgb[~writable], axis=1).sum())
    if non_face_changed:
        raise RuntimeError("FACE_CANDIDATE_NON_FACE_CHANGED")

    report = {
        "schema": "face_surface_candidate_v2",
        "classification": "CANDIDATE_REQUIRES_VISUAL_REVIEW",
        "baseline_glb": str(baseline_glb),
        "baseline_glb_sha256": sha256(baseline_glb),
        "baseline_atlas": str(baseline_atlas),
        "baseline_atlas_sha256": sha256(baseline_atlas),
        "source_image": str(source_image),
        "source_image_sha256": sha256(source_image),
        "source_fixture": str(source_fixture),
        "source_fixture_sha256": sha256(source_fixture),
        "foreground": foreground_report,
        "face_mask_pixels": int(face_mask.sum()),
        "ray_stride": int(ray_stride),
        "ray_count": int(len(layers["pixels_xy"])),
        "hit_count": int(len(layers["triangle_ids"])),
        "maximum_depth_layers": int(np.diff(layers["offsets"]).max()) if len(layers["offsets"]) > 1 else 0,
        "chart_inventory": chart_inventory,
        "selection": selection,
        "anchors": anchors,
        "texture": texture_report,
        "non_face_atlas_pixels_changed": non_face_changed,
        "immutable_hashes_before": before_hashes,
        "immutable_hashes_after": after_hashes,
        "output_atlas": str(atlas_path),
        "output_atlas_sha256": sha256(atlas_path),
        "output_glb": str(glb_path),
        "output_glb_sha256": sha256(glb_path),
        "promotion_authorized": False,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-glb", type=Path, required=True)
    parser.add_argument("--baseline-atlas", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--source-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ray-stride", type=int, default=3)
    parser.add_argument("--minimum-alpha", type=float, default=0.30)
    args = parser.parse_args()
    report = build_candidate(
        args.baseline_glb, args.baseline_atlas, args.source_image,
        args.source_fixture, args.output_dir,
        ray_stride=args.ray_stride, minimum_alpha=args.minimum_alpha,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
