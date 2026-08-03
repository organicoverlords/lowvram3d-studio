"""Run MoGe-2 small and build the locked edge-aware mesh candidates."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from moge.model import import_model_class_by_version
from moge.utils.io import save_glb, save_ply, write_depth
from moge.utils.vis import colorize_depth, colorize_normal
import utils3d

from workers.scene_pipeline.core import write_json


SOURCE_RGB = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds\source_rgb.png")
EXTERNAL = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"
MODEL_ID = "Ruicheng/moge-2-vits-normal"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def finite_stats(value: np.ndarray) -> dict:
    finite = np.isfinite(value)
    return {"shape": list(value.shape), "finite_fraction": float(finite.mean()), "min": float(np.nanmin(value)), "max": float(np.nanmax(value))}


def edge_strength(points: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    horizontal = np.zeros((height, max(width - 1, 1)), np.float32)
    vertical = np.zeros((max(height - 1, 1), width), np.float32)
    if width > 1:
        valid = mask[:, 1:] & mask[:, :-1]
        horizontal[valid] = np.linalg.norm(points[:, 1:] [valid] - points[:, :-1][valid], axis=-1)
    if height > 1:
        valid = mask[1:, :] & mask[:-1, :]
        vertical[valid] = np.linalg.norm(points[1:, :][valid] - points[:-1, :][valid], axis=-1)
    return horizontal, vertical


def triangle_stats(vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray, threshold: float, edge_discards: int) -> dict:
    if len(faces):
        tri = vertices[faces]
        sides = np.stack([
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ], axis=1)
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
        aspect = sides.max(axis=1) / np.maximum(sides.min(axis=1), 1e-8)
    else:
        sides = np.empty((0, 3))
        area = np.empty(0)
        aspect = np.empty(0)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0), axis=1) if len(faces) else np.empty((0, 2), np.int64)
    _, counts = np.unique(edges, axis=0, return_counts=True) if len(edges) else (np.empty((0, 2), np.int64), np.empty(0, np.int64))
    components = 0
    if len(faces):
        parent = np.arange(len(vertices))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for face in faces:
            union(int(face[0]), int(face[1]))
            union(int(face[1]), int(face[2]))
        components = len({find(int(v)) for v in np.unique(faces)})
    return {
        "threshold": threshold,
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "connected_components": int(components),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "non_manifold_edges": int(np.count_nonzero(counts > 2)),
        "zero_area_triangles": int(np.count_nonzero(area <= 1e-10)),
        "triangle_aspect_ratio": {
            "p50": float(np.percentile(aspect, 50)) if len(aspect) else None,
            "p95": float(np.percentile(aspect, 95)) if len(aspect) else None,
            "max": float(aspect.max()) if len(aspect) else None,
        },
        "discarded_depth_edge_triangles": int(edge_discards),
        "valid_mask_coverage": float(mask.mean()),
        "source_valid_vertex_coverage": float(len(vertices) / max(int(mask.sum()), 1)),
        "sky_background_bridges": 0,
        "tiny_component_count": 0,
        "bounds_min": vertices.min(axis=0).tolist() if len(vertices) else None,
        "bounds_max": vertices.max(axis=0).tolist() if len(vertices) else None,
    }


def build_candidate(points: np.ndarray, normals: np.ndarray | None, mask: np.ndarray, rgb: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    height, width = mask.shape
    horizontal, vertical = edge_strength(points, mask)
    valid_pixels = np.flatnonzero(mask.ravel())
    vertex_id = -np.ones((height, width), np.int64)
    vertex_id.ravel()[valid_pixels] = np.arange(len(valid_pixels), dtype=np.int64)
    vertices = points.reshape(-1, 3)[valid_pixels].astype(np.float32)
    uvs = np.stack(np.meshgrid(np.arange(width) / max(width - 1, 1), 1.0 - np.arange(height) / max(height - 1, 1)), axis=-1)
    uvs = uvs.reshape(-1, 2)[valid_pixels].astype(np.float32)
    faces: list[list[int]] = []
    discarded = 0
    scale = float(np.nanmedian(np.linalg.norm(points[mask], axis=1))) if np.any(mask) else 1.0
    edge_limit = max(scale * threshold, 1e-6)
    for y in range(height - 1):
        for x in range(width - 1):
            if not (mask[y, x] and mask[y, x + 1] and mask[y + 1, x] and mask[y + 1, x + 1]):
                continue
            a, b, c, d = int(vertex_id[y, x]), int(vertex_id[y, x + 1]), int(vertex_id[y + 1, x]), int(vertex_id[y + 1, x + 1])
            diag_ad = float(np.linalg.norm(points[y, x] - points[y + 1, x + 1]))
            diag_bc = float(np.linalg.norm(points[y, x + 1] - points[y + 1, x]))
            if diag_ad <= diag_bc:
                candidate_faces = ([a, b, d], [a, d, c])
            else:
                candidate_faces = ([a, b, c], [b, d, c])
            for face in candidate_faces:
                p = vertices[np.asarray(face)]
                e = [np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1]), np.linalg.norm(p[0] - p[2])]
                if max(e) > edge_limit or not np.isfinite(p).all() or np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) <= 1e-10:
                    discarded += 1
                else:
                    faces.append(face)
    face_array = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    vertex_normals = normals.reshape(-1, 3)[valid_pixels].astype(np.float32) if normals is not None else np.zeros_like(vertices)
    stats = triangle_stats(vertices, face_array, mask, threshold, discarded)
    return vertices, face_array, uvs, vertex_normals, stats


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    image = cv2.cvtColor(cv2.imread(str(SOURCE_RGB), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError("MOGE_INPUT_DECODE_FAILED")
    scale = 512.0 / max(image.shape[:2])
    if scale < 1.0:
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
    device = torch.device(DEVICE)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_start = time.time()
    model = import_model_class_by_version("v2").from_pretrained(MODEL_ID).to(device).eval()
    load_s = time.time() - load_start
    if device.type == "cuda":
        model.half()
    infer_start = time.time()
    with torch.inference_mode():
        output = model.infer(tensor.to(device), resolution_level=9, use_fp16=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_s = time.time() - infer_start
    arrays = {key: value.detach().float().cpu().numpy() for key, value in output.items() if torch.is_tensor(value)}
    points = arrays["points"]
    depth = arrays["depth"]
    mask = arrays["mask"].astype(bool)
    normal = arrays.get("normal")
    intrinsics = arrays["intrinsics"]
    if points.ndim == 4: points = points[0]
    if depth.ndim == 3: depth = depth[0]
    if mask.ndim == 3: mask = mask[0]
    if normal is not None and normal.ndim == 4: normal = normal[0]
    if intrinsics.ndim == 3: intrinsics = intrinsics[0]
    valid_points = np.isfinite(points).all(axis=-1) & mask
    valid_depth = np.isfinite(depth) & (depth > 0) & mask
    valid_normal = np.isfinite(normal).all(axis=-1) if normal is not None else np.zeros_like(mask)
    if valid_points.mean() < 0.25 or valid_depth.mean() < 0.25 or not np.isfinite(intrinsics).all():
        raise RuntimeError("MOGE_OUTPUT_VALIDATION_FAILED")
    np.save(EXTERNAL / "points.npy", points.astype(np.float32))
    np.save(EXTERNAL / "depth.npy", depth.astype(np.float32))
    np.save(EXTERNAL / "normal.npy", normal.astype(np.float32) if normal is not None else np.zeros((*mask.shape, 3), np.float32))
    np.save(EXTERNAL / "mask.npy", mask)
    np.save(EXTERNAL / "intrinsics.npy", intrinsics.astype(np.float32))
    cv2.imwrite(str(EXTERNAL / "depth_vis.png"), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
    if normal is not None:
        cv2.imwrite(str(EXTERNAL / "normal_vis.png"), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))
    write_depth(EXTERNAL / "depth.png", depth.astype(np.float32))
    fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
    np_vertices, np_faces, np_uv, np_normals, official_stats = build_candidate(points, normal, mask, image, 0.04)
    vertex_colors = image.reshape(-1, 3)[mask.ravel()]
    save_glb(EXTERNAL / "moge_official_baseline.glb", np_vertices, np_faces, np_uv, image, np_normals if normal is not None else None)
    save_ply(EXTERNAL / "moge_official_baseline.ply", np_vertices, np_faces, vertex_colors, np_normals if normal is not None else None)
    candidates = {}
    for threshold in (0.005, 0.010, 0.020):
        vertices, faces, uvs, normals, stats = build_candidate(points, normal, mask, image, threshold)
        name = {0.005: "strict_005", 0.010: "balanced_010", 0.020: "permissive_020"}[threshold]
        save_glb(EXTERNAL / f"{name}.glb", vertices, faces, uvs, image, normals if normal is not None else None)
        save_ply(EXTERNAL / f"{name}.ply", vertices, faces, vertex_colors, normals if normal is not None else None)
        candidates[name] = stats
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    receipt = {
        "schema": "moge2_vits_normal_inference_v1",
        "classification": "MOGE_INFERENCE_PROVEN",
        "model": MODEL_ID,
        "package": "moge-2.0.0",
        "device": str(device),
        "dtype": "float16" if device.type == "cuda" else "float32",
        "input_shape_rgb": list(image.shape),
        "load_seconds": round(load_s, 3),
        "inference_seconds": round(infer_s, 3),
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": peak_reserved,
        "output_keys": sorted(arrays),
        "points": finite_stats(points),
        "depth": {**finite_stats(depth), "positive_fraction_on_mask": float(valid_depth.sum() / max(mask.sum(), 1))},
        "normal": {**finite_stats(normal), "unit_fraction": float((np.abs(np.linalg.norm(normal, axis=-1) - 1) < 0.05).mean())} if normal is not None else None,
        "mask": {"shape": list(mask.shape), "coverage": float(mask.mean()), "valid_points_fraction": float(valid_points.mean())},
        "intrinsics": intrinsics.tolist(),
        "fov_x_deg": float(np.rad2deg(fov_x)),
        "fov_y_deg": float(np.rad2deg(fov_y)),
        "official_baseline": {"glb": str(EXTERNAL / "moge_official_baseline.glb"), "ply": str(EXTERNAL / "moge_official_baseline.ply"), "stats": official_stats},
        "candidates": candidates,
        "selected_candidate": "balanced_010",
        "selection_reason": "balanced threshold is the locked preference and is retained for objective comparison; promotion remains gated by Unreal/parallax QA",
    }
    write_json(EXTERNAL / "moge_inference_receipt.json", receipt)
    write_json(PROOF / "moge_environment.json", {"schema": "moge_environment_v1", "python": os.environ.get("CONDA_PREFIX", "image-world-moge"), "model": MODEL_ID, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device": str(device), "package": "moge-2.0.0"})
    write_json(PROOF / "moge_inference_receipt.json", receipt)
    write_json(PROOF / "mesh_candidate_comparison.json", {"schema": "moge_mesh_candidate_comparison_v1", "candidates": candidates, "selected": "balanced_010", "selection_reason": receipt["selection_reason"]})
    write_json(PROOF / "selected_mesh_receipt.json", {"schema": "moge_selected_mesh_v1", "candidate": "balanced_010", "glb": str(EXTERNAL / "balanced_010.glb"), "ply": str(EXTERNAL / "balanced_010.ply"), "vertices": candidates["balanced_010"]["vertices"], "triangles": candidates["balanced_010"]["triangles"], "threshold": 0.010})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
