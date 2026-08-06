"""Generic multi-layer raycasting and coherent face-surface ownership.

A caller supplies a GLB, one camera contract, a source-space mask and optional
landmarks. Every ray/triangle depth layer is retained. The selected result is an
exact triangle mask; this module never mutates textures or promotes a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .conservative_atlas import derive_uv_chart_ids
    from .mesh_io import read_glb
except ImportError:  # pragma: no cover
    from conservative_atlas import derive_uv_chart_ids
    from mesh_io import read_glb

EPS = 1e-9


@dataclass(frozen=True)
class Camera:
    origin: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    width: int
    height: int
    projection: str = "orthographic"
    ortho_width: float = 2.0
    ortho_height: float = 2.0
    fov_y_degrees: float = 50.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Camera":
        def vec(name: str, default: Iterable[float] | None = None) -> np.ndarray:
            value = np.asarray(payload.get(name, default), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"CAMERA_{name.upper()}_INVALID")
            return value

        origin = vec("origin", (0.0, 0.0, 0.0))
        right = vec("right")
        up = vec("up")
        forward = vec("forward", np.cross(right, up))
        right /= max(float(np.linalg.norm(right)), EPS)
        up -= right * float(np.dot(up, right))
        up /= max(float(np.linalg.norm(up)), EPS)
        forward -= right * float(np.dot(forward, right)) + up * float(np.dot(forward, up))
        forward /= max(float(np.linalg.norm(forward)), EPS)
        if float(np.dot(np.cross(right, up), forward)) < 0:
            forward = -forward
        width = int(payload["width"])
        height = int(payload["height"])
        if width <= 0 or height <= 0:
            raise ValueError("CAMERA_DIMENSIONS_INVALID")
        projection = str(payload.get("projection", "orthographic")).lower()
        if projection not in {"orthographic", "perspective"}:
            raise ValueError("CAMERA_PROJECTION_UNSUPPORTED")
        return cls(
            origin=origin,
            right=right,
            up=up,
            forward=forward,
            width=width,
            height=height,
            projection=projection,
            ortho_width=float(payload.get("ortho_width", payload.get("view_width", 2.0))),
            ortho_height=float(payload.get("ortho_height", payload.get("view_height", 2.0))),
            fov_y_degrees=float(payload.get("fov_y_degrees", 50.0)),
        )

    def pixel_rays(self, pixels_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pixels = np.asarray(pixels_xy, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("PIXELS_MUST_BE_N_BY_2")
        nx = (pixels[:, 0] + 0.5) / self.width
        ny = (pixels[:, 1] + 0.5) / self.height
        if self.projection == "orthographic":
            x = (nx - 0.5) * self.ortho_width
            y = (0.5 - ny) * self.ortho_height
            origins = self.origin + x[:, None] * self.right + y[:, None] * self.up
            directions = np.repeat(self.forward[None, :], len(pixels), axis=0)
        else:
            aspect = self.width / self.height
            tan_y = np.tan(np.deg2rad(self.fov_y_degrees) * 0.5)
            x = (2.0 * nx - 1.0) * tan_y * aspect
            y = (1.0 - 2.0 * ny) * tan_y
            directions = self.forward + x[:, None] * self.right + y[:, None] * self.up
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), EPS)
            origins = np.repeat(self.origin[None, :], len(pixels), axis=0)
        return origins, directions

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = np.asarray(points, dtype=np.float64) - self.origin
        x = delta @ self.right
        y = delta @ self.up
        depth = delta @ self.forward
        if self.projection == "orthographic":
            nx = x / self.ortho_width + 0.5
            ny = 0.5 - y / self.ortho_height
        else:
            tan_y = np.tan(np.deg2rad(self.fov_y_degrees) * 0.5)
            aspect = self.width / self.height
            safe = np.where(np.abs(depth) < EPS, np.nan, depth)
            nx = 0.5 + 0.5 * x / (safe * tan_y * aspect)
            ny = 0.5 - 0.5 * y / (safe * tan_y)
        return np.column_stack((nx * self.width - 0.5, ny * self.height - 0.5)), depth


@dataclass
class BVH:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    left: np.ndarray
    right: np.ndarray
    start: np.ndarray
    count: np.ndarray
    triangle_order: np.ndarray
    root: int


def build_bvh(positions: np.ndarray, triangles: np.ndarray, leaf_size: int = 16) -> BVH:
    positions = np.asarray(positions, np.float64)
    triangles = np.asarray(triangles, np.int64)
    if leaf_size <= 0:
        raise ValueError("BVH_LEAF_SIZE_INVALID")
    if len(triangles) == 0:
        empty = np.empty((0, 3), np.float64)
        return BVH(empty, empty, np.empty(0, np.int32), np.empty(0, np.int32),
                   np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, np.int64), -1)
    corners = positions[triangles]
    tri_min = corners.min(axis=1)
    tri_max = corners.max(axis=1)
    centers = (tri_min + tri_max) * 0.5
    order = np.arange(len(triangles), dtype=np.int64)
    nodes: list[tuple[np.ndarray, np.ndarray, int, int, int, int]] = []

    def split(lo: int, hi: int) -> int:
        ids = order[lo:hi]
        node = len(nodes)
        bmin = tri_min[ids].min(axis=0)
        bmax = tri_max[ids].max(axis=0)
        nodes.append((bmin, bmax, -1, -1, lo, hi - lo))
        if hi - lo <= leaf_size:
            return node
        axis = int(np.argmax(np.ptp(centers[ids], axis=0)))
        middle = (lo + hi) // 2
        order[lo:hi] = ids[np.argsort(centers[ids, axis], kind="mergesort")]
        left = split(lo, middle)
        right = split(middle, hi)
        nodes[node] = (bmin, bmax, left, right, lo, 0)
        return node

    root = split(0, len(triangles))
    return BVH(
        np.stack([row[0] for row in nodes]), np.stack([row[1] for row in nodes]),
        np.asarray([row[2] for row in nodes], np.int32),
        np.asarray([row[3] for row in nodes], np.int32),
        np.asarray([row[4] for row in nodes], np.int64),
        np.asarray([row[5] for row in nodes], np.int32), order, root,
    )


def _ray_aabb(origin: np.ndarray, direction: np.ndarray, bmin: np.ndarray, bmax: np.ndarray,
              t_min: float, t_max: float) -> bool:
    parallel = np.abs(direction) < EPS
    if np.any(parallel & ((origin < bmin) | (origin > bmax))):
        return False
    active = ~parallel
    if not np.any(active):
        return True
    inv = 1.0 / direction[active]
    a = (bmin[active] - origin[active]) * inv
    b = (bmax[active] - origin[active]) * inv
    entry = max(float(np.max(np.minimum(a, b))), t_min)
    exit_ = min(float(np.min(np.maximum(a, b))), t_max)
    return exit_ >= entry


def _ray_triangles(origin: np.ndarray, direction: np.ndarray, positions: np.ndarray,
                   triangles: np.ndarray, triangle_ids: np.ndarray, t_min: float,
                   t_max: float, backface_cull: bool) -> tuple[np.ndarray, np.ndarray]:
    tri = triangles[triangle_ids]
    v0 = positions[tri[:, 0]]
    e1 = positions[tri[:, 1]] - v0
    e2 = positions[tri[:, 2]] - v0
    pvec = np.cross(np.repeat(direction[None, :], len(tri), axis=0), e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    valid = det > EPS if backface_cull else np.abs(det) > EPS
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    valid &= (u >= -EPS) & (u <= 1.0 + EPS)
    qvec = np.cross(tvec, e1)
    v = np.einsum("j,ij->i", direction, qvec) * inv_det
    valid &= (v >= -EPS) & ((u + v) <= 1.0 + EPS)
    t = np.einsum("ij,ij->i", e2, qvec) * inv_det
    valid &= (t >= t_min) & (t <= t_max)
    ids = triangle_ids[valid]
    bary = np.column_stack((1.0 - u[valid] - v[valid], u[valid], v[valid]))
    return ids, np.column_stack((t[valid], bary))


def trace_ray(origin: np.ndarray, direction: np.ndarray, positions: np.ndarray,
              triangles: np.ndarray, bvh: BVH, max_hits: int = 32,
              t_min: float = 1e-6, t_max: float = np.inf,
              backface_cull: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if bvh.root < 0:
        return np.empty(0, np.int64), np.empty((0, 4), np.float64)
    stack = [bvh.root]
    ids_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    while stack:
        node = int(stack.pop())
        if not _ray_aabb(origin, direction, bvh.bounds_min[node], bvh.bounds_max[node], t_min, t_max):
            continue
        if bvh.count[node] > 0:
            ids = bvh.triangle_order[bvh.start[node]:bvh.start[node] + bvh.count[node]]
            hit_ids, hit_data = _ray_triangles(origin, direction, positions, triangles, ids,
                                               t_min, t_max, backface_cull)
            if len(hit_ids):
                ids_parts.append(hit_ids)
                data_parts.append(hit_data)
        else:
            stack.extend((int(bvh.left[node]), int(bvh.right[node])))
    if not ids_parts:
        return np.empty(0, np.int64), np.empty((0, 4), np.float64)
    ids = np.concatenate(ids_parts)
    data = np.concatenate(data_parts)
    order = np.lexsort((ids, data[:, 0]))
    ids = ids[order]
    data = data[order]
    duplicate = np.zeros(len(ids), bool)
    if len(ids) > 1:
        duplicate[1:] = (ids[1:] == ids[:-1]) & (np.abs(data[1:, 0] - data[:-1, 0]) < 1e-8)
    return ids[~duplicate][:max_hits], data[~duplicate][:max_hits]


def trace_mask_layers(positions: np.ndarray, normals: np.ndarray, triangles: np.ndarray,
                      camera: Camera, mask: np.ndarray, stride: int = 2,
                      max_hits: int = 16, leaf_size: int = 16) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, bool)
    if mask.shape != (camera.height, camera.width):
        raise ValueError("FACE_MASK_CAMERA_SHAPE_MISMATCH")
    yy, xx = np.nonzero(mask)
    keep = (xx % stride == 0) & (yy % stride == 0)
    pixels = np.column_stack((xx[keep], yy[keep])).astype(np.int32)
    origins, directions = camera.pixel_rays(pixels)
    bvh = build_bvh(positions, triangles, leaf_size)
    offsets = np.zeros(len(pixels) + 1, np.int64)
    id_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    bary_parts: list[np.ndarray] = []
    facing_parts: list[np.ndarray] = []
    for ray_index, (origin, direction) in enumerate(zip(origins, directions, strict=True)):
        ids, data = trace_ray(origin, direction, positions, triangles, bvh, max_hits=max_hits)
        offsets[ray_index + 1] = offsets[ray_index] + len(ids)
        id_parts.append(ids)
        depth_parts.append(data[:, 0])
        bary_parts.append(data[:, 1:])
        if len(ids):
            hit_normals = np.einsum("ij,ijk->ik", data[:, 1:], normals[triangles[ids]])
            hit_normals /= np.maximum(np.linalg.norm(hit_normals, axis=1, keepdims=True), EPS)
            facing_parts.append(hit_normals @ (-direction))
        else:
            facing_parts.append(np.empty(0))
    return {
        "pixels_xy": pixels,
        "ray_origins": origins,
        "ray_directions": directions,
        "offsets": offsets,
        "triangle_ids": np.concatenate(id_parts) if id_parts else np.empty(0, np.int64),
        "depth": np.concatenate(depth_parts) if depth_parts else np.empty(0),
        "barycentric": np.concatenate(bary_parts) if bary_parts else np.empty((0, 3)),
        "normal_facing": np.concatenate(facing_parts) if facing_parts else np.empty(0),
    }


def candidate_triangle_adjacency(triangles: np.ndarray, candidate_ids: np.ndarray) -> dict[int, set[int]]:
    candidate_ids = np.unique(np.asarray(candidate_ids, np.int64))
    adjacency = {int(value): set() for value in candidate_ids}
    edge_owner: dict[tuple[int, int], int] = {}
    for triangle_id in candidate_ids.tolist():
        vertices = triangles[triangle_id]
        for a, b in ((vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            other = edge_owner.get(edge)
            if other is None:
                edge_owner[edge] = triangle_id
            else:
                adjacency[triangle_id].add(other)
                adjacency[other].add(triangle_id)
    return adjacency


def connected_patches(triangles: np.ndarray, candidate_ids: np.ndarray) -> list[np.ndarray]:
    adjacency = candidate_triangle_adjacency(triangles, candidate_ids)
    remaining = set(adjacency)
    patches: list[np.ndarray] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        patches.append(np.asarray(sorted(component), np.int64))
    return patches


def hit_ranks(offsets: np.ndarray) -> np.ndarray:
    result = np.empty(int(offsets[-1]), np.int16)
    for ray in range(len(offsets) - 1):
        result[offsets[ray]:offsets[ray + 1]] = np.arange(offsets[ray + 1] - offsets[ray])
    return result


def score_surface_patches(layers: dict[str, np.ndarray], triangles: np.ndarray,
                          positions: np.ndarray, normals: np.ndarray, chart_ids: np.ndarray,
                          camera: Camera, landmarks_xy: np.ndarray | None = None,
                          max_rank: int = 4, minimum_facing: float = 0.05) -> list[dict[str, Any]]:
    hit_ids = layers["triangle_ids"]
    if not len(hit_ids):
        return []
    ranks = hit_ranks(layers["offsets"])
    eligible = (ranks < max_rank) & (layers["normal_facing"] >= minimum_facing)
    candidates = np.unique(hit_ids[eligible])
    patches = connected_patches(triangles, candidates)
    pixels = layers["pixels_xy"]
    ray_for_hit = np.repeat(np.arange(len(pixels)), np.diff(layers["offsets"]))
    landmarks = None if landmarks_xy is None else np.asarray(landmarks_xy, np.float64)
    tri_centers = positions[triangles].mean(axis=1)
    tri_normals = normals[triangles].mean(axis=1)
    tri_normals /= np.maximum(np.linalg.norm(tri_normals, axis=1, keepdims=True), EPS)
    image_center = np.asarray((camera.width * 0.5, camera.height * 0.5))
    diagonal = float(np.hypot(camera.width, camera.height))
    records: list[dict[str, Any]] = []
    for patch_index, patch in enumerate(patches):
        selected = np.isin(hit_ids, patch)
        patch_rays = np.unique(ray_for_hit[selected])
        patch_pixels = pixels[patch_rays]
        coverage = len(patch_rays) / max(len(pixels), 1)
        centrality = 1.0 - min(float(np.mean(np.linalg.norm(patch_pixels - image_center, axis=1))) /
                               max(diagonal * 0.5, EPS), 1.0)
        rank_score = float(np.mean(np.exp(-0.75 * ranks[selected])))
        facing_score = float(np.clip(np.mean(layers["normal_facing"][selected]), 0.0, 1.0))
        depth = layers["depth"][selected]
        depth_cv = float(np.std(depth) / max(abs(float(np.mean(depth))), EPS))
        depth_score = float(np.exp(-4.0 * depth_cv))
        projected, _ = camera.project(tri_centers[patch])
        side_wrap = float(np.mean(np.linalg.norm(projected - image_center, axis=1) > diagonal * 0.42))
        front_score = float(np.clip(np.mean(tri_normals[patch] @ (-camera.forward)), 0.0, 1.0))
        landmark_support = 0.0
        if landmarks is not None and len(projected):
            distance = np.linalg.norm(projected[:, None, :] - landmarks[None, :, :], axis=2)
            landmark_support = float(np.mean(np.min(distance, axis=0) <= diagonal * 0.04))
        score = (2.8 * coverage + 1.6 * centrality + 1.4 * rank_score + facing_score +
                 front_score + 1.2 * depth_score + 2.5 * landmark_support - 2.0 * side_wrap)
        records.append({
            "patch_index": patch_index,
            "triangle_ids": patch,
            "triangle_count": int(len(patch)),
            "chart_count": int(len(np.unique(chart_ids[patch]))),
            "ray_coverage": float(coverage),
            "centrality": float(centrality),
            "rank_score": rank_score,
            "facing_score": facing_score,
            "front_normal_score": front_score,
            "depth_cv": depth_cv,
            "depth_score": depth_score,
            "landmark_support": landmark_support,
            "side_wrap_fraction": side_wrap,
            "score": float(score),
        })
    return sorted(records, key=lambda row: (-row["score"], -row["triangle_count"], row["patch_index"]))


def select_face_patch(records: list[dict[str, Any]], minimum_ray_coverage: float = 0.03,
                      maximum_side_wrap: float = 0.35,
                      minimum_landmark_support: float = 0.0) -> dict[str, Any]:
    for record in records:
        if record["ray_coverage"] < minimum_ray_coverage:
            continue
        if record["side_wrap_fraction"] > maximum_side_wrap:
            continue
        if record["landmark_support"] < minimum_landmark_support:
            continue
        return record
    raise RuntimeError("FACE_SURFACE_PATCH_NOT_FOUND")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(mesh_path: Path, camera_path: Path, face_mask_path: Path, output_dir: Path,
        landmarks_path: Path | None = None, stride: int = 2, max_hits: int = 16,
        leaf_size: int = 16) -> dict[str, Any]:
    positions, normals, uv, triangles = read_glb(mesh_path)
    if uv is None:
        raise RuntimeError("FACE_SURFACE_OWNERSHIP_REQUIRES_UV")
    payload = json.loads(camera_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("OWNERSHIP_CAMERA_FILE_MUST_CONTAIN_ONE_CAMERA")
        payload = payload[0]
    camera = Camera.from_dict(payload)
    face_mask = np.asarray(np.load(face_mask_path), bool)
    landmarks = None
    if landmarks_path is not None:
        landmarks = np.asarray(json.loads(landmarks_path.read_text(encoding="utf-8")), np.float64)
    chart_ids, chart_inventory = derive_uv_chart_ids(uv, triangles)
    layers = trace_mask_layers(positions, normals, triangles, camera, face_mask,
                               stride=stride, max_hits=max_hits, leaf_size=leaf_size)
    records = score_surface_patches(layers, triangles, positions, normals, chart_ids, camera, landmarks)
    selected = select_face_patch(records, minimum_landmark_support=0.25 if landmarks is not None else 0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "all_ray_hits.npz", **layers)
    selected_ids = np.asarray(selected["triangle_ids"], np.int64)
    np.save(output_dir / "selected_face_triangles.npy", selected_ids)
    selected_mask = np.zeros(len(triangles), bool)
    selected_mask[selected_ids] = True
    np.save(output_dir / "selected_face_triangle_mask.npy", selected_mask)
    candidates = []
    for record in records[:64]:
        row = dict(record)
        row["triangle_ids"] = record["triangle_ids"].astype(int).tolist()
        candidates.append(row)
    selected_json = dict(selected)
    selected_json["triangle_ids"] = selected_ids.astype(int).tolist()
    report = {
        "schema": "face_surface_ownership_v1",
        "classification": "DIAGNOSTIC_UNTIL_RENDERED",
        "mesh": str(mesh_path),
        "mesh_sha256": sha256(mesh_path),
        "camera": payload,
        "mask_sample_count": int(len(layers["pixels_xy"])),
        "hit_count": int(len(layers["triangle_ids"])),
        "maximum_depth_layers": int(np.diff(layers["offsets"]).max()) if len(layers["offsets"]) > 1 else 0,
        "candidate_patch_count": int(len(records)),
        "selected_patch": selected_json,
        "chart_inventory": chart_inventory,
        "promotion_authorized": False,
    }
    (output_dir / "ownership_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "triangle_patch_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--face-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-hits", type=int, default=16)
    parser.add_argument("--leaf-size", type=int, default=16)
    args = parser.parse_args()
    report = run(args.mesh, args.camera, args.face_mask, args.output_dir,
                 args.landmarks, args.stride, args.max_hits, args.leaf_size)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
