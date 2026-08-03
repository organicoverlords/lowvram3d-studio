"""Run one bounded fresh MoGe comparison inference without touching baseline arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from moge.model import import_model_class_by_version
from moge.utils.io import save_glb
from workers.scene_pipeline.run_moge_scene import build_candidate


SOURCE = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds\source_rgb.png")
EXTERNAL = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds\moge_comparison")
MODEL_DEFAULT = "Ruicheng/moge-2-vits-normal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_stats(value: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(value)
    return {"shape": list(value.shape), "finite_fraction": float(finite.mean()), "min": float(np.nanmin(value)), "max": float(np.nanmax(value))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    args = parser.parse_args()
    out = EXTERNAL / args.name
    out.mkdir(parents=True, exist_ok=True)
    image = cv2.cvtColor(cv2.imread(str(SOURCE), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    scale = float(args.resolution) / max(image.shape[:2])
    image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    source_path = out / "source_rgb.png"
    cv2.imwrite(str(source_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("MOGE_COMPARISON_REQUIRES_CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_start = time.time()
    model = import_model_class_by_version("v2").from_pretrained(args.model).to(device).eval()
    load_s = time.time() - load_start
    model.half()
    infer_start = time.time()
    with torch.inference_mode():
        output = model.infer(tensor.to(device), resolution_level=9, use_fp16=True)
    torch.cuda.synchronize(device)
    infer_s = time.time() - infer_start
    arrays = {key: value.detach().float().cpu().numpy() for key, value in output.items() if torch.is_tensor(value)}
    points = arrays["points"]; depth = arrays["depth"]; mask = arrays["mask"].astype(bool); normal = arrays.get("normal"); intrinsics = arrays["intrinsics"]
    if points.ndim == 4: points = points[0]
    if depth.ndim == 3: depth = depth[0]
    if mask.ndim == 3: mask = mask[0]
    if normal is not None and normal.ndim == 4: normal = normal[0]
    if intrinsics.ndim == 3: intrinsics = intrinsics[0]
    np.save(out / "points.npy", points.astype(np.float32)); np.save(out / "depth.npy", depth.astype(np.float32)); np.save(out / "mask.npy", mask); np.save(out / "normal.npy", normal.astype(np.float32)); np.save(out / "intrinsics.npy", intrinsics.astype(np.float32))
    valid_points = mask & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    vertices, faces, uvs, normals, mesh_stats = build_candidate(points, normal, valid_points, image, 0.04)
    glb = out / "official.glb"
    save_glb(glb, vertices, faces, uvs, image, normals)
    fy = float(intrinsics[1, 1]); fx = float(intrinsics[0, 0])
    fov_x = float(np.degrees(2.0 * np.arctan(0.5 / fx))); fov_y = float(np.degrees(2.0 * np.arctan(0.5 / fy)))
    receipt = {
        "schema": "moge_bounded_comparison_v1", "classification": "MOGE_COMPARISON_PROVEN",
        "name": args.name, "model": args.model, "requested_max_dimension": args.resolution,
        "input_shape_rgb": list(image.shape), "source_sha256": sha256(source_path),
        "device": str(device), "dtype": "float16", "load_seconds": load_s, "inference_seconds": infer_s,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "arrays": {"points": finite_stats(points), "depth": finite_stats(depth), "normal": finite_stats(normal), "mask": {"shape": list(mask.shape), "coverage": float(mask.mean())}, "intrinsics": intrinsics.tolist()},
        "camera": {"fov_x_deg": fov_x, "fov_y_deg": fov_y, "resolution": [int(points.shape[1]), int(points.shape[0])]},
        "mesh": {**mesh_stats, "glb": str(glb)},
        "artifacts": {name: str(out / name) for name in ("points.npy", "depth.npy", "normal.npy", "mask.npy", "intrinsics.npy", "source_rgb.png")},
    }
    (out / "comparison_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
