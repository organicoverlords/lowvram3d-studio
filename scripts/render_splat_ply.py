"""Render a 3DGS PLY with a plain z-buffer so the export can be checked anywhere.

This is a validator, not a renderer: no alpha blending, no anisotropic
projection, no sorting beyond nearest-wins. It exists to answer whether the PLY
holds the geometry and colour it claims, without needing a GPU viewer or an
engine plugin. If this produces the source image from the source viewpoint, the
positions, colours and encodings are right.

    py -3.12 scripts/render_splat_ply.py --ply scene.ply --fov 92.9 --out check.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


def read_ply(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        header = b""
        while b"end_header" not in header:
            chunk = handle.readline()
            if not chunk:
                raise ValueError("no end_header found")
            header += chunk
        text = header.decode("ascii")
        names = [line.split()[-1] for line in text.splitlines()
                 if line.startswith("property float")]
        count = next(int(line.split()[-1]) for line in text.splitlines()
                     if line.startswith("element vertex"))
        data = np.frombuffer(handle.read(count * len(names) * 4),
                             dtype=np.float32).reshape(count, len(names))
    return {name: data[:, index] for index, name in enumerate(names)}


def render(ply: Path, out: Path, fov_x: float, width: int, height: int,
           yaw: float = 0.0, pitch: float = 0.0, distance: float = 0.0) -> dict:
    from PIL import Image

    fields = read_ply(ply)
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=-1).astype(np.float64)
    dc = np.stack([fields["f_dc_0"], fields["f_dc_1"], fields["f_dc_2"]], axis=-1)
    rgb = np.clip(dc.astype(np.float64) * SH_C0 + 0.5, 0.0, 1.0)

    # Export convention is Y up, Z back, so the camera looks down -Z.
    eye = np.array([math.sin(math.radians(yaw)) * distance,
                    math.sin(math.radians(pitch)) * distance,
                    math.cos(math.radians(yaw)) * distance])
    points = xyz - eye

    cy, sy = math.cos(math.radians(-yaw)), math.sin(math.radians(-yaw))
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    cp, sp = math.cos(math.radians(-pitch)), math.sin(math.radians(-pitch))
    rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    points = points @ rot_y.T @ rot_x.T

    depth = -points[:, 2]
    visible = depth > 1e-6
    points, rgb, depth = points[visible], rgb[visible], depth[visible]

    focal = 0.5 * width / math.tan(math.radians(fov_x) * 0.5)
    us = (points[:, 0] / depth) * focal + width * 0.5
    vs = (-points[:, 1] / depth) * focal + height * 0.5

    xi, yi = np.floor(us).astype(np.int64), np.floor(vs).astype(np.int64)
    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    xi, yi, depth, rgb = xi[inside], yi[inside], depth[inside], rgb[inside]

    # Nearest splat wins: sort far-to-near and let later writes overwrite.
    order = np.argsort(-depth)
    xi, yi, rgb = xi[order], yi[order], rgb[order]

    canvas = np.zeros((height, width, 3), dtype=np.float64)
    canvas[yi, xi] = rgb
    image = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)

    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out)

    covered = float((canvas.sum(axis=-1) > 0).mean())
    return {"splats_total": int(xyz.shape[0]), "splats_in_frame": int(xi.shape[0]),
            "pixel_coverage": covered, "output": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fov", type=float, required=True)
    parser.add_argument("--width", type=int, default=1448)
    parser.add_argument("--height", type=int, default=1086)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--distance", type=float, default=0.0)
    args = parser.parse_args()

    report = render(Path(args.ply), Path(args.out), args.fov, args.width,
                    args.height, args.yaw, args.pitch, args.distance)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
