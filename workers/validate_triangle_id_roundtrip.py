"""Validate the mesh/UV/camera/triangle-ID round trip without neural imagery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from atlas_raster import census, rasterise
from mesh_io import read_glb


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_prefix(view: dict) -> str:
    return str(view.get("control_file_prefix") or view["semantic_name"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas-size", type=int, default=2048)
    args = parser.parse_args()

    mesh = Path(args.mesh)
    bundle = Path(args.bundle)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    positions, normals, uv, tris = read_glb(mesh)
    positions = np.asarray(positions, np.float64)
    normals = np.asarray(normals, np.float64)
    uv = np.asarray(uv, np.float64)
    tris = np.asarray(tris, np.int64)
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))
    transform = np.asarray(contract["control_space_transform"], np.float64)
    canonical = positions @ transform.T
    vertices = canonical * (0.5 / float(np.max(np.abs(canonical))))
    vertex_normals = normals @ transform.T
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-12)
    owner, weights = rasterise(uv, tris, args.atlas_size)
    owned = owner >= 0
    owner_flat = owner[owned]
    corners = tris[owner_flat]
    wa = weights[owned, 0][:, None]
    wb = weights[owned, 1][:, None]
    wc = 1.0 - wa - wb
    texel_position = (vertices[corners[:, 0]] * wc + vertices[corners[:, 1]] * wa
                      + vertices[corners[:, 2]] * wb)
    texel_normal = (vertex_normals[corners[:, 0]] * wc + vertex_normals[corners[:, 1]] * wa
                    + vertex_normals[corners[:, 2]] * wb)
    texel_normal /= np.maximum(np.linalg.norm(texel_normal, axis=1, keepdims=True), 1e-12)

    interior_texel, _interior_tri = census(uv, tris, args.atlas_size, interior=0.05)
    interior_mask = np.zeros(args.atlas_size * args.atlas_size, dtype=bool)
    interior_mask[np.unique(interior_texel)] = True
    owned_flat_indices = np.flatnonzero(owned.ravel())
    interior_owned = interior_mask[owned_flat_indices]
    mismatch_image = np.zeros((args.atlas_size, args.atlas_size, 3), np.uint8)
    report_views = []
    for view in sorted(contract["views"], key=lambda item: int(item["index"])):
        prefix = file_prefix(view)
        ids = np.load(bundle / f"{prefix}_triangle_ids.npy")
        depth = np.load(bundle / f"{prefix}_depth.npy")
        size = ids.shape[0]
        direction = np.asarray(view["camera_direction"], np.float64)
        right = np.asarray(view["camera_right"], np.float64)
        up = np.asarray(view["camera_up"], np.float64)
        screen = np.stack((texel_position @ right / float(contract["projection_span"]) + 0.5,
                           0.5 - texel_position @ up / float(contract["projection_span"])), axis=1)
        pixel = np.rint(screen * float(size - 1)).astype(np.int64)
        in_bounds = ((pixel[:, 0] >= 0) & (pixel[:, 0] < size)
                     & (pixel[:, 1] >= 0) & (pixel[:, 1] < size))
        sampled = np.full(owner_flat.shape, -1, np.int64)
        sampled[in_bounds] = ids[pixel[in_bounds, 1], pixel[in_bounds, 0]]
        eligible = in_bounds & (sampled >= 0)
        match = eligible & (sampled == owner_flat)
        buffered = np.full(owner_flat.shape, np.inf, np.float64)
        buffered[in_bounds] = depth[pixel[in_bounds, 1], pixel[in_bounds, 0]]
        depth_delta = np.abs(texel_position @ direction - buffered)
        facing = -(texel_normal @ direction)
        visible = eligible & np.isfinite(buffered) & (depth_delta <= 0.01) & (facing > 0.2)
        interior_eligible = visible & interior_owned
        interior_match = match & interior_owned & visible
        mismatch = visible & ~match
        atlas_y, atlas_x = np.divmod(owned_flat_indices[mismatch], args.atlas_size)
        mismatch_image[atlas_y, atlas_x] = (255, 40, 40)
        report_views.append({
            "index": int(view["index"]),
            "semantic": str(view.get("proven_semantic") or view["semantic_name"]),
            "control_file_prefix": prefix,
            "control_resolution": int(size),
            "eligible_texels": int(eligible.sum()),
            "visible_texels": int(visible.sum()),
            "interior_eligible_texels": int(interior_eligible.sum()),
            "interior_matches": int(interior_match.sum()),
            "interior_mismatches": int((interior_eligible & ~match).sum()),
            "interior_match_percent": float(100.0 * interior_match.sum() / max(interior_eligible.sum(), 1)),
            "boundary_mismatches": int((visible & ~interior_owned & ~match).sum()),
            "visible_matches": int((match & visible).sum()),
            "visible_match_percent": float(100.0 * (match & visible).sum() / max(visible.sum(), 1)),
        })

    source_mesh_sha = str(contract.get("mesh_sha256_after") or contract.get("mesh_sha256") or "")
    mesh_sha = sha256(mesh)
    contract_mesh = Path(str(contract.get("mesh", "")))
    contract_report = {
        "schema": "panda_camera_contract_v1",
        "mesh_sha256": mesh_sha,
        "vertex_index_hash": hashlib.sha256(np.arange(len(positions), dtype=np.int64).tobytes()).hexdigest(),
        "triangle_index_hash": hashlib.sha256(np.asarray(tris, np.int64).tobytes()).hexdigest(),
        "uv_hash": hashlib.sha256(np.asarray(uv, np.float64).tobytes()).hexdigest(),
        "camera_contract_source": str(bundle / "camera_contract.json"),
        "camera_contract_mesh_sha256": source_mesh_sha,
        "stale_triangle_id_buffer_usage": int((source_mesh_sha and source_mesh_sha != mesh_sha)
                                               or (contract_mesh and contract_mesh.resolve() != mesh.resolve())),
        "control_space_transform": contract["control_space_transform"],
        "pixel_center_convention": "texel center / pixel center at (i + 0.5) in atlas raster; control sample uses round(screen * (N - 1))",
        "image_origin": "top_left",
        "ndc_convention": "screen_x = dot(position,right)/span + 0.5; screen_y = 0.5 - dot(position,up)/span",
        "uv_origin": "top_left atlas raster; glTF sampling is handled separately",
        "row_orientation": "rows increase downward",
        "integer_rounding": "nearest via numpy.rint",
        "depth_convention": "control depth compared in canonical camera direction",
        "triangle_edge_ownership": "exact texel-centre barycentric rasterization; lowest triangle index owns an injective texel",
    }
    (out / "camera_contract.json").write_text(json.dumps(contract_report, indent=2), encoding="utf-8")
    (out / "pixel_convention_matrix.json").write_text(json.dumps({
        "projection_span": contract["projection_span"],
        "control_space_transform": contract["control_space_transform"],
        "screen_formula": "[dot(p,right)/span+0.5, 0.5-dot(p,up)/span]",
        "control_pixel_formula": "rint(screen * (control_resolution - 1))",
        "atlas_raster_formula": "texel centre = [x+0.5,y+0.5]",
    }, indent=2), encoding="utf-8")
    (out / "triangle_id_roundtrip_report.json").write_text(json.dumps({
        "schema": "triangle_id_roundtrip_v1",
        "mesh": str(mesh),
        "atlas_size": int(args.atlas_size),
        "owned_texels": int(owned.sum()),
        "interior_owned_texels": int(interior_owned.sum()),
        "views": report_views,
        "interior_triangle_id_match_percent": float(min(item["interior_match_percent"] for item in report_views)),
        "stale_triangle_id_buffer_usage": contract_report["stale_triangle_id_buffer_usage"],
        "camera_contract": "PROVEN" if (contract_report["stale_triangle_id_buffer_usage"] == 0
                                           and all(item["interior_mismatches"] == 0 for item in report_views)) else "NOT_PROVEN",
    }, indent=2), encoding="utf-8")
    Image.fromarray(mismatch_image).save(out / "triangle_id_mismatch_overlay.png")
    print(json.dumps({"views": report_views, "contract": contract_report}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
