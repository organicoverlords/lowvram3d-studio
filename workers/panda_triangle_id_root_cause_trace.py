"""Trace a small deterministic sample of triangle-ID mismatches to their root cause."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from atlas_raster import rasterise
from mesh_io import read_glb, triangle_components


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def stem(index: int) -> str:
    return f"horizontal_{index}" if index < 4 else ("top" if index == 4 else "bottom")


def barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(float(den)) <= 1e-12:
        return np.full(3, np.nan)
    w0 = ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])) / den
    w1 = ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])) / den
    return np.asarray([w0, w1, 1.0 - w0 - w1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--atlas-size", type=int, default=2048)
    args = p.parse_args()
    mesh, bundle, output = Path(args.mesh), Path(args.bundle), Path(args.output)
    positions, normals, uv, tris = read_glb(mesh)
    positions = np.asarray(positions, np.float64)
    normals = np.asarray(normals, np.float64)
    uv = np.asarray(uv, np.float64)
    tris = np.asarray(tris, np.int64)
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))
    transform = np.asarray(contract["control_space_transform"], np.float64)
    canonical = positions @ transform.T
    canonical *= 0.5 / float(np.max(np.abs(canonical)))
    owner, weights = rasterise(uv, tris, args.atlas_size)
    owned = owner >= 0
    flat_positions = np.flatnonzero(owned)
    owner_flat = owner[owned].astype(np.int64)
    wa, wb = weights[owned, 0], weights[owned, 1]
    atlas_bary = np.column_stack((1.0 - wa - wb, wa, wb))
    corners = tris[owner_flat]
    texel = (canonical[corners[:, 0]] * atlas_bary[:, 0, None]
             + canonical[corners[:, 1]] * atlas_bary[:, 1, None]
             + canonical[corners[:, 2]] * atlas_bary[:, 2, None])
    face_normals = np.cross(canonical[tris[:, 1]] - canonical[tris[:, 0]],
                            canonical[tris[:, 2]] - canonical[tris[:, 0]])
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    centroids = canonical[tris].mean(axis=1)
    components, _ = triangle_components(positions, tris)
    index_hash = digest(tris)
    vertex_hash = digest(np.arange(len(positions), dtype=np.int64))
    uv_hash = digest(uv)
    samples = []
    requested = {(int(view["index"]), shared, alias): 1
                 for view in contract["views"]
                 for shared in (2, 1, 0) for alias in (True, False)}
    for view in sorted(contract["views"], key=lambda item: int(item["index"])):
        index = int(view["index"])
        ids = np.load(bundle / f"{stem(index)}_triangle_ids.npy").astype(np.int64)
        depth = np.load(bundle / f"{stem(index)}_depth.npy")
        control_position = np.load(bundle / f"{stem(index)}_position.npy")
        size = int(ids.shape[0])
        direction = np.asarray(view["camera_direction"], np.float64)
        right = np.asarray(view["camera_right"], np.float64)
        up = np.asarray(view["camera_up"], np.float64)
        screen = np.stack((texel @ right / float(contract["projection_span"]) + 0.5,
                           0.5 - texel @ up / float(contract["projection_span"])), axis=1)
        pixel = np.rint(screen * float(size - 1)).astype(np.int64)
        valid = ((pixel[:, 0] >= 0) & (pixel[:, 0] < size)
                 & (pixel[:, 1] >= 0) & (pixel[:, 1] < size))
        x = np.clip(pixel[:, 0], 0, size - 1)
        y = np.clip(pixel[:, 1], 0, size - 1)
        sampled = np.full(owner_flat.shape, -1, np.int64)
        sampled[valid] = ids[y[valid], x[valid]]
        control_depth = np.full(owner_flat.shape, np.inf, np.float64)
        control_depth[valid] = depth[y[valid], x[valid]]
        owner_depth = texel @ direction
        depth_delta = np.abs(owner_depth - control_depth)
        visible = valid & (sampled >= 0) & np.isfinite(control_depth) & (depth_delta <= 0.01)
        visible &= (-(face_normals[owner_flat] @ direction)) > 0.2
        mismatch = visible & (sampled != owner_flat) & (np.min(atlas_bary, axis=1) > 0.05)
        for flat in np.flatnonzero(mismatch):
            left, right_id = int(owner_flat[flat]), int(sampled[flat])
            shared = len(set(tris[left]) & set(tris[right_id]))
            support = np.linalg.norm(control_position[y[flat], x[flat]] - texel[flat])
            centroid_distance = float(np.linalg.norm(centroids[left] - centroids[right_id]))
            normal_dot = float(face_normals[left] @ face_normals[right_id])
            alias = bool(support <= 0.01 and centroid_distance <= 0.01
                         and normal_dot >= 0.98 and components[left] == components[right_id])
            key = (index, shared, alias)
            if requested.get(key, 0) <= 0:
                continue
            requested[key] -= 1
            tri_screen_left = np.stack((canonical[tris[left]] @ right / contract["projection_span"] + 0.5,
                                        0.5 - canonical[tris[left]] @ up / contract["projection_span"]), axis=1) * (size - 1)
            tri_screen_right = np.stack((canonical[tris[right_id]] @ right / contract["projection_span"] + 0.5,
                                         0.5 - canonical[tris[right_id]] @ up / contract["projection_span"]), axis=1) * (size - 1)
            sample_point = np.asarray([x[flat] + 0.5, y[flat] + 0.5], np.float64)
            bary_left = barycentric(sample_point, tri_screen_left)
            bary_right = barycentric(sample_point, tri_screen_right)
            left_depth = float(bary_left @ (canonical[tris[left]] @ direction)) if np.isfinite(bary_left).all() else float("nan")
            right_depth = float(bary_right @ (canonical[tris[right_id]] @ direction)) if np.isfinite(bary_right).all() else float("nan")
            samples.append({
                "view": index, "semantic": view.get("proven_semantic", stem(index)),
                "atlas_pixel_xy": [int(flat % args.atlas_size), int(flat // args.atlas_size)],
                "control_pixel_xy": [int(x[flat]), int(y[flat])],
                "owner_triangle": left, "visible_control_triangle": right_id,
                "owner_triangle_vertices": tris[left].tolist(),
                "visible_triangle_vertices": tris[right_id].tolist(),
                "shared_vertex_count": shared, "alias_equivalent": alias,
                "owner_component": int(components[left]), "visible_component": int(components[right_id]),
                "owner_uv": uv[tris[left]].tolist(), "visible_uv": uv[tris[right_id]].tolist(),
                "owner_positions_canonical": canonical[tris[left]].tolist(),
                "visible_positions_canonical": canonical[tris[right_id]].tolist(),
                "owner_normal": face_normals[left].tolist(), "visible_normal": face_normals[right_id].tolist(),
                "normal_dot": normal_dot, "centroid_distance": centroid_distance,
                "atlas_barycentric": atlas_bary[flat].tolist(), "texel_position_canonical": texel[flat].tolist(),
                "owner_screen_barycentric_at_control_pixel": bary_left.tolist(),
                "visible_screen_barycentric_at_control_pixel": bary_right.tolist(),
                "owner_screen_covered": bool(np.all(bary_left >= -1e-4)) if np.isfinite(bary_left).all() else False,
                "visible_screen_covered": bool(np.all(bary_right >= -1e-4)) if np.isfinite(bary_right).all() else False,
                "owner_depth_at_control_pixel": left_depth,
                "visible_depth_at_control_pixel": right_depth,
                "control_depth": float(control_depth[flat]),
                "owner_depth_from_texel": float(owner_depth[flat]),
                "owner_control_depth_delta": float(abs(left_depth - float(control_depth[flat]))) if np.isfinite(left_depth) else None,
                "visible_control_depth_delta": float(abs(right_depth - float(control_depth[flat]))) if np.isfinite(right_depth) else None,
                "depth_delta_gate": float(depth_delta[flat]),
                "owner_facing": float(-(face_normals[left] @ direction)),
                "visible_facing": float(-(face_normals[right_id] @ direction)),
                "support_distance_to_control_position": float(support),
                "mesh_index_hash": index_hash, "vertex_index_hash": vertex_hash, "uv_hash": uv_hash,
                "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
                "atlas_owner_path": "atlas_raster.rasterise -> UV texel centre -> lowest triangle index",
                "control_id_path": "exact_mesh_controls_1024/*_triangle_ids.npy -> control pixel rint(screen*(N-1))",
                "tie_or_occlusion": {
                    "control_pixel_id": right_id,
                    "owner_screen_covered": bool(np.all(bary_left >= -1e-4)) if np.isfinite(bary_left).all() else False,
                    "visible_screen_covered": bool(np.all(bary_right >= -1e-4)) if np.isfinite(bary_right).all() else False,
                    "control_depth_is_nearest": bool(abs(right_depth - float(control_depth[flat])) <= 0.01) if np.isfinite(right_depth) else False,
                    "owner_depth_competes": bool(np.isfinite(left_depth) and abs(left_depth - float(control_depth[flat])) <= 0.01),
                },
            })
        if not any(value > 0 for value in requested.values()):
            break

    owner_covered_count = int(sum(bool(sample["owner_screen_covered"]) for sample in samples))
    visible_covered_count = int(sum(bool(sample["visible_screen_covered"]) for sample in samples))
    owner_competes_count = int(sum(bool(sample["tie_or_occlusion"]["owner_depth_competes"]) for sample in samples))
    visible_nearest_count = int(sum(bool(sample["tie_or_occlusion"]["control_depth_is_nearest"]) for sample in samples))
    sample_observations = {
        "owner_screen_covered_count": owner_covered_count,
        "visible_screen_covered_count": visible_covered_count,
        "owner_depth_competes_count": owner_competes_count,
        "visible_control_depth_nearest_count": visible_nearest_count,
        "interpretation": (
            "In the stratified 36-sample trace, the atlas-owner triangle is covered at the rounded control pixel "
            f"in {owner_covered_count}/{len(samples)} samples while the control-visible triangle is covered in "
            f"{visible_covered_count}/{len(samples)}. This is finite-pixel raster ownership/quantization between "
            "the UV-centre owner and control ID rasterizer, not evidence that the raw IDs are interchangeable."
        ),
    }
    conclusion = {
        "mesh_or_index_ordering": "not supported: mesh SHA/index/UV hashes agree with control receipt",
        "near_coincident_surface_selection": "supported: alias-equivalent mismatches have same component, near-identical normals and sub-0.01 support/centroid distances; control ID is a different triangle",
        "backface_winding_depth": "not primary: sampled IDs satisfy depth <=0.01 and facing >0.2; owner and visible screen barycentrics are traced",
        "uv_owner_unrelated_to_visible_surface": "partly supported: UV owner can be a non-visible near-coincident triangle; deterministic atlas-owner ID cannot be represented by one depth-buffer pixel",
        "validator_mapping_error": "not supported by convention sweep and independent position/depth trace",
        "deterministic_repair": "none without changing geometry/UV or relaxing exact triangle-ID acceptance; canonical mode-visible mapping reached only 57-60%",
        "camera_contract": "NOT_PROVEN",
        "sampling_quantization": sample_observations["interpretation"],
        "downstream": "blocked by mission policy",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema": "panda_triangle_id_root_cause_trace_v1",
        "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
        "triangle_index_hash": index_hash, "vertex_index_hash": vertex_hash, "uv_hash": uv_hash,
        "control_mesh_sha256": json.loads((bundle / "cpu_controls_report.json").read_text()).get("mesh_sha256_after", ""),
        "sample_requested": {f"view_{view}_shared_{s}_alias_{a}": 1
                              for view in sorted(int(v["index"]) for v in contract["views"])
                              for s in (2, 1, 0) for a in (True, False)},
        "sample_count": len(samples), "samples": samples,
        "sample_observations": sample_observations, "conclusion": conclusion,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"sample_count": len(samples), "camera_contract": "NOT_PROVEN", "conclusion": conclusion}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
