"""Record (uv, colour) ground-truth samples so atlas orientation can be tested, not assumed.

For a set of triangles that were genuinely observed from a view, this writes the UV coordinate of
each triangle's centroid together with the source colour that was projected onto it. A correctly
oriented atlas reproduces that colour when sampled at row = v*(size-1); a vertically mirrored one
reproduces it at row = (1-v)*(size-1).

That distinction cannot be made by looking at the atlas: both conventions produce a full, finite,
plausible-looking image, and the wrong one only reveals itself on the model as a patchwork.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def project(vertices: np.ndarray, direction: np.ndarray, ortho: float) -> np.ndarray:
    """Mirror of raster_project.py's orthographic mapping, in normalised screen space."""
    axis = int(np.argmax(np.abs(direction)))
    ua, va = (0, 2) if axis == 1 else ((1, 2) if axis == 0 else (0, 1))
    flip_u = -1.0 if direction[axis] > 0 else 1.0
    u = (vertices[:, ua] * flip_u) / ortho + 0.5
    v = 0.5 - vertices[:, va] / ortho
    return np.stack([u, v], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--view-name", default="front")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--min-facing", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = np.load(args.npz)
    verts = data["verts"].astype(np.float64)
    tris = data["tris"].astype(np.int64)
    uvs = data["uvs"].astype(np.float64)
    normals = data["normals"].astype(np.float64)
    ortho = float(data["ortho_scale"])
    visible = data[f"vis_{args.view_name}"]
    camera = data["view_locs"][list(data["view_names"]).index(args.view_name)].astype(np.float64)
    direction = camera / (np.linalg.norm(camera) + 1e-9)

    view = cv2.imread(args.view, cv2.IMREAD_UNCHANGED)
    height, width = view.shape[:2]
    alpha = view[:, :, 3] if view.shape[2] == 4 else np.full((height, width), 255, np.uint8)

    facing = normals @ direction
    candidates = np.flatnonzero(visible & (facing > args.min_facing))
    if candidates.size == 0:
        raise RuntimeError("no visible, front-facing triangles to sample")
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(candidates, min(args.samples, candidates.size), replace=False)

    centroids = verts[tris[chosen]].mean(axis=1)
    screen = project(centroids, direction, ortho)
    sx = np.clip((screen[:, 0] * (width - 1)).astype(int), 0, width - 1)
    sy = np.clip((screen[:, 1] * (height - 1)).astype(int), 0, height - 1)
    keep = alpha[sy, sx] > 200
    chosen, sx, sy = chosen[keep], sx[keep], sy[keep]

    # cv2 hands back BGR; the atlas is compared in RGB.
    colours = view[sy, sx, :3][:, ::-1].astype(float)
    uv = uvs[chosen].mean(axis=1)

    payload = {"view": args.view_name, "count": int(len(uv)),
               "uv": uv.tolist(), "rgb": colours.tolist()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload), encoding="utf-8")
    print(f"ORIENTATION_TRUTH {len(uv)} samples -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
