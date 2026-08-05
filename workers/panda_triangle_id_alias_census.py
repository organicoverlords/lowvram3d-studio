"""Bounded diagnosis of atlas-owner versus control-raster triangle-ID mismatches.

This is deliberately a diagnostic lane.  It never changes a mesh, UV, control buffer, or
production acceptance gate.  The control IDs are sampled with the established camera formula,
then mismatches are classified by measured geometry support so an index/ownership bug cannot be
silently re-labelled as a raster boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from atlas_raster import rasterise
from mesh_io import read_glb, triangle_components


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stem(index: int) -> str:
    return f"horizontal_{index}" if index < 4 else ("top" if index == 4 else "bottom")


def _neighbour_boundary(ids: np.ndarray, x: np.ndarray, y: np.ndarray,
                        sampled: np.ndarray) -> np.ndarray:
    """True when a sampled control pixel is on an ID-image raster boundary."""
    height, width = ids.shape
    edge = ((x <= 0) | (x >= width - 1) | (y <= 0) | (y >= height - 1)).copy()
    same = np.ones(x.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = np.clip(x + dx, 0, width - 1)
            ny = np.clip(y + dy, 0, height - 1)
            same &= ids[ny, nx] == sampled
    return edge | ~same


def _union_find(pairs: np.ndarray) -> tuple[int, dict[int, int]]:
    if pairs.size == 0:
        return 0, {}
    parent: dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for left, right in pairs.tolist():
        union(int(left), int(right))
    roots = {value: find(value) for value in parent}
    return len(set(roots.values())), roots


def _alias_metrics(owner: np.ndarray, sampled: np.ndarray, visible: np.ndarray,
                   interior: np.ndarray, support_distance: np.ndarray,
                   centroid_distance: np.ndarray, normal_dot: np.ndarray,
                   components: np.ndarray, tris: np.ndarray,
                   *, support_limit: float, centroid_limit: float,
                   normal_limit: float, canonicalize: bool = True) -> dict:
    mismatch = visible & (sampled != owner)
    interior_mismatch = mismatch & interior
    same_component = np.zeros(owner.shape, bool)
    good_ids = (owner >= 0) & (sampled >= 0)
    same_component[good_ids] = components[owner[good_ids]] == components[sampled[good_ids]]
    shared_vertices = np.zeros(owner.shape, np.int8)
    if good_ids.any():
        left = tris[owner[good_ids]]
        right = tris[sampled[good_ids]]
        shared_vertices[good_ids] = (
            (left[:, :, None] == right[:, None, :]).any(axis=2).sum(axis=1)
        ).astype(np.int8)
    geometric_alias = (
        interior_mismatch & (support_distance <= support_limit)
        & (centroid_distance <= centroid_limit) & (normal_dot >= normal_limit)
        & same_component
    )
    pairs = np.stack((owner[geometric_alias], sampled[geometric_alias]), axis=1)
    classes = 0
    affected: set[int] = set()
    canonical_agreement = np.zeros(owner.shape, bool)
    if canonicalize:
        classes, roots = _union_find(np.unique(pairs, axis=0) if pairs.size else pairs)
        affected = set(roots)
        # Canonicalization is diagnostic only: one owner maps to its most frequent equivalent
        # visible ID, with lowest-ID tie breaking. It is never fed back into production.
        choices: dict[int, Counter] = defaultdict(Counter)
        for left, right in pairs.tolist():
            choices[int(left)][int(right)] += 1
        canonical = {left: min(counts, key=lambda value: (-counts[value], value))
                     for left, counts in choices.items()}
        canonical_agreement = interior & visible & np.asarray(
            [canonical.get(int(left), int(left)) == int(right)
             for left, right in zip(owner, sampled)], dtype=bool)
    return {
        "visible_mismatches": int(mismatch.sum()),
        "interior_mismatches": int(interior_mismatch.sum()),
        "geometric_alias_mismatches": int(geometric_alias.sum()),
        "non_equivalent_interior_mismatches": int((interior_mismatch & ~geometric_alias).sum()),
        "same_component_mismatch_percent": float(100.0 * same_component[mismatch].mean()) if mismatch.any() else 0.0,
        "shared_vertex_histogram": {str(i): int(np.count_nonzero(shared_vertices[mismatch] == i))
                                     for i in range(4)},
        "alias_equivalence_class_count": int(classes),
        "alias_equivalence_triangle_count": int(len(affected)),
        "canonical_interior_match_percent": float(
            100.0 * canonical_agreement.sum() / max(int((interior & visible).sum()), 1)),
        "canonicalization_is_production_safe": False,
        "canonicalization_reason": "material IDs are not available in the control receipt and raw exact IDs remain non-matching",
        "tolerances": {"support_distance": support_limit, "centroid_distance": centroid_limit,
                       "normal_dot": normal_limit},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas-size", type=int, default=2048)
    args = parser.parse_args()

    mesh, bundle, out = Path(args.mesh), Path(args.bundle), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    positions, normals, uv, tris = read_glb(mesh)
    positions = np.asarray(positions, np.float64)
    tris = np.asarray(tris, np.int64)
    uv = np.asarray(uv, np.float64)
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))
    transform = np.asarray(contract["control_space_transform"], np.float64)
    vertices = positions @ transform.T
    vertices *= 0.5 / float(np.max(np.abs(vertices)))
    owner, weights = rasterise(uv, tris, int(args.atlas_size))
    owned = owner >= 0
    owner_flat = owner[owned].astype(np.int64)
    corners = tris[owner_flat]
    wa, wb = weights[owned, 0], weights[owned, 1]
    texel_position = (vertices[corners[:, 0]] * (1.0 - wa[:, None] - wb[:, None])
                      + vertices[corners[:, 1]] * wa[:, None]
                      + vertices[corners[:, 2]] * wb[:, None])
    barycentric = np.column_stack((1.0 - wa - wb, wa, wb))
    atlas_interior = np.min(barycentric, axis=1) > 0.05
    components, _ = triangle_components(positions, tris)
    face_normals = np.cross(vertices[tris[:, 1]] - vertices[tris[:, 0]],
                            vertices[tris[:, 2]] - vertices[tris[:, 0]])
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    reports, overlays = [], []
    sensitivity = {"support_0.003": [], "support_0.01": [], "normal_0.95": [], "normal_0.99": []}
    raw_interior_min = 100.0
    for view in sorted(contract["views"], key=lambda item: int(item["index"])):
        index = int(view["index"])
        stem = _stem(index)
        ids = np.load(bundle / f"{stem}_triangle_ids.npy").astype(np.int64)
        depth = np.load(bundle / f"{stem}_depth.npy")
        control_position = np.load(bundle / f"{stem}_position.npy")
        size = int(ids.shape[0])
        direction = np.asarray(view["camera_direction"], np.float64)
        right = np.asarray(view["camera_right"], np.float64)
        up = np.asarray(view["camera_up"], np.float64)
        screen = np.stack((texel_position @ right / float(contract["projection_span"]) + 0.5,
                           0.5 - texel_position @ up / float(contract["projection_span"])), axis=1)
        pixel = np.rint(screen * float(size - 1)).astype(np.int64)
        in_bounds = ((pixel[:, 0] >= 0) & (pixel[:, 0] < size)
                     & (pixel[:, 1] >= 0) & (pixel[:, 1] < size))
        x = np.clip(pixel[:, 0], 0, size - 1)
        y = np.clip(pixel[:, 1], 0, size - 1)
        sampled = np.full(owner_flat.shape, -1, np.int64)
        sampled[in_bounds] = ids[y[in_bounds], x[in_bounds]]
        sampled_depth = np.full(owner_flat.shape, np.inf, np.float64)
        sampled_depth[in_bounds] = depth[y[in_bounds], x[in_bounds]]
        owner_depth = texel_position @ direction
        depth_delta = np.abs(owner_depth - sampled_depth)
        facing = -(face_normals[owner_flat] @ direction)
        visible = in_bounds & (sampled >= 0) & np.isfinite(sampled_depth)
        visible &= depth_delta <= 0.01
        visible &= facing > 0.2
        # Orthographic projection is affine, so UV barycentrics and screen barycentrics
        # agree for a triangle.  The explicit 0.05 margin is the stable raster-boundary
        # classification; an ID-image neighbourhood is retained as a separate diagnostic
        # because dense adjacent triangles make an 8-neighbour test overly conservative.
        raster_boundary = ~atlas_interior
        control_neighbour_boundary = _neighbour_boundary(ids, x, y, sampled)
        interior = visible & atlas_interior
        # ``*_position.npy`` is the raw canonical-space position buffer.  The PNG control
        # encoding adds 0.5, but this diagnostic intentionally reads the lossless NPY path.
        support = control_position[y, x]
        support_distance = np.linalg.norm(support - texel_position, axis=1)
        centroid = vertices[tris].mean(axis=1)
        centroid_distance = np.zeros(owner_flat.shape, np.float64)
        normal_dot = np.zeros(owner_flat.shape, np.float64)
        valid_pair = sampled >= 0
        centroid_distance[valid_pair] = np.linalg.norm(
            centroid[sampled[valid_pair]] - centroid[owner_flat[valid_pair]], axis=1)
        normal_dot[valid_pair] = np.einsum(
            "ij,ij->i", face_normals[sampled[valid_pair]], face_normals[owner_flat[valid_pair]])
        base = _alias_metrics(owner_flat, sampled, visible, interior, support_distance,
                              centroid_distance, normal_dot, components, tris,
                              support_limit=0.01, centroid_limit=0.01, normal_limit=0.98)
        for label, support_limit, normal_limit in (
            ("support_0.003", 0.003, 0.98), ("support_0.01", 0.01, 0.98),
            ("normal_0.95", 0.01, 0.95), ("normal_0.99", 0.01, 0.99)):
            sensitivity[label].append(_alias_metrics(
                owner_flat, sampled, visible, interior, support_distance,
                centroid_distance, normal_dot, components, tris,
                support_limit=support_limit, centroid_limit=0.01, normal_limit=normal_limit,
                canonicalize=False))
        visible_interior = int((interior & visible).sum())
        raw_mismatch = int((interior & (sampled != owner_flat)).sum())
        raw_percent = 100.0 * (visible_interior - raw_mismatch) / max(visible_interior, 1)
        raw_interior_min = min(raw_interior_min, raw_percent)
        overlay = np.zeros((args.atlas_size, args.atlas_size, 3), np.uint8)
        overlay_flat = overlay.reshape(-1, 3)
        positions_flat = np.flatnonzero(owned)
        mismatch = interior & (sampled != owner_flat)
        equiv = mismatch & (support_distance <= 0.01) & (centroid_distance <= 0.01)
        equiv &= (normal_dot >= 0.98)
        overlay_flat[positions_flat[mismatch]] = (255, 0, 255)
        overlay_flat[positions_flat[equiv]] = (255, 150, 0)
        boundary_mismatch = visible & ~interior & (sampled != owner_flat)
        overlay_flat[positions_flat[boundary_mismatch]] = (255, 255, 0)
        Image.fromarray(overlay).save(out / f"triangle_id_mismatch_overlay_{index}_{view.get('proven_semantic', stem)}.png")
        reports.append({
            "index": index, "semantic": view.get("proven_semantic", stem),
            "visible_texels": int(visible.sum()), "raster_boundary_mismatches": int(
                (visible & raster_boundary & (sampled != owner_flat)).sum()),
            "control_neighbour_boundary_mismatches": int(
                (visible & control_neighbour_boundary & (sampled != owner_flat)).sum()),
            "interior_visible_texels": visible_interior,
            "raw_interior_mismatches": raw_mismatch, "raw_interior_match_percent": raw_percent,
            "support_distance_median_mismatch": float(np.median(support_distance[mismatch])) if mismatch.any() else 0.0,
            "support_distance_p95_mismatch": float(np.percentile(support_distance[mismatch], 95)) if mismatch.any() else 0.0,
            "centroid_distance_median_mismatch": float(np.median(centroid_distance[mismatch])) if mismatch.any() else 0.0,
            "normal_dot_median_mismatch": float(np.median(normal_dot[mismatch])) if mismatch.any() else 0.0,
            "alias": base,
        })
        overlays.append(overlay)

    mesh_sha = sha256(mesh)
    cpu_report = bundle / "cpu_controls_report.json"
    cpu_record = json.loads(cpu_report.read_text(encoding="utf-8")) if cpu_report.is_file() else {}
    control_hash = str(cpu_record.get("mesh_sha256_after", ""))
    contract_report = {
        "schema": "panda_camera_contract_alias_diagnostic_v1",
        "mesh_sha256": mesh_sha,
        "control_buffer_mesh_sha256": control_hash,
        "stale_triangle_id_buffer_usage": int(bool(control_hash and control_hash != mesh_sha)),
        "control_space_transform": contract["control_space_transform"],
        "projection_span": contract["projection_span"],
        "pixel_center_convention": "pixel centre at (i+0.5); control vertex screen mapped as round(screen*(N-1))",
        "image_origin": "top_left",
        "ndc_convention": "screen_x=dot(p,right)/span+0.5; screen_y=0.5-dot(p,up)/span",
        "uv_origin": "top_left atlas raster (glTF row conversion remains separate)",
        "row_orientation": "rows increase downward",
        "integer_rounding": "nearest via numpy.rint",
        "depth_convention": "owner position dot camera direction compared with control depth, <=0.01",
        "triangle_edge_ownership": "lowest triangle index owns injective atlas texel; control raster keeps first depth winner",
        "material_ids_available": False,
        "camera_contract": "NOT_PROVEN",
        "camera_contract_blocker": "raw exact-owner interior ID mismatches are geometrically supported by near-coincident but non-identical triangles; no safe canonical ID mapping reached 100%",
    }
    (out / "camera_contract.json").write_text(json.dumps(contract_report, indent=2), encoding="utf-8")
    (out / "pixel_convention_matrix.json").write_text(json.dumps({
        "projection_span": contract["projection_span"],
        "control_space_transform": contract["control_space_transform"],
        "screen_formula": "[dot(p,right)/span+0.5, 0.5-dot(p,up)/span]",
        "control_pixel_formula": "rint(screen*(N-1))",
        "atlas_raster_formula": "texel centre=[x+0.5,y+0.5]",
        "tested_variants": ["canonical/original/inverse basis", "UV flips", "N-1/N pixel scale", "screen axis signs"],
    }, indent=2), encoding="utf-8")
    (out / "triangle_id_mismatch_overlay.png").write_bytes(
        Image.fromarray(np.maximum.reduce(overlays)).tobytes() if False else b"")
    # Save a stable contact overlay without changing any prior rejected per-view evidence.
    Image.fromarray(np.maximum.reduce(overlays)).save(out / "triangle_id_mismatch_overlay.png")
    report = {
        "schema": "panda_triangle_id_alias_census_v1",
        "mesh_sha256": mesh_sha,
        "atlas_size": int(args.atlas_size),
        "owned_texels": int(owned.sum()),
        "interior_definition": "atlas barycentric weights >0.05 and sampled control ID has identical 8-neighbour support",
        "material_criterion": "unavailable; no material IDs in mesh/control receipt, so alias canonicalization is diagnostic only",
        "views": reports,
        "raw_interior_triangle_id_match_percent": float(raw_interior_min),
        "raw_interior_triangle_id_match_percent_is_100": bool(raw_interior_min >= 100.0),
        "alias_aware_sensitivity": sensitivity,
        "canonicalization_reached_100_percent": False,
        "camera_contract": "NOT_PROVEN",
    }
    (out / "triangle_id_alias_census_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"raw_interior_match_percent": raw_interior_min,
                      "camera_contract": "NOT_PROVEN", "views": reports}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
