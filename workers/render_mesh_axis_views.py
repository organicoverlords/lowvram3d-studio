"""Textured orthographic renders along the mesh's own axes, for a chosen up vector.

The six-view control rig bakes in an assumption about which mesh axis points up. If that
assumption is wrong the character is rendered lying on its side and every semantic label
downstream is meaningless, so the assumption has to be testable on its own. This renders
the asset along +/-X, +/-Y and +/-Z for a candidate up axis and samples the base-colour
atlas, which is enough to see where the head is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from build_mvadapter_cpu_controls import _rasterise
from mesh_io import read_glb
from render_control_bundle_texture import base_colour_image, contact_sheet, sample

AXES = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0),
        "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


def unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    return array / max(float(np.linalg.norm(array)), 1e-12)


def basis(direction: np.ndarray, up_axis: np.ndarray):
    """Camera right/up for a view direction, falling back when the view looks along up."""
    reference = up_axis
    if abs(float(direction @ reference)) > 0.99:
        reference = np.array([1.0, 0.0, 0.0]) if abs(reference[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    right = unit(np.cross(direction, reference))
    return right, unit(np.cross(right, direction))


def caption(image: Image.Image, text: str) -> Image.Image:
    framed = Image.new("RGB", (image.width, image.height + 18), (18, 18, 20))
    framed.paste(image, (0, 18))
    ImageDraw.Draw(framed).text((4, 4), text, fill=(240, 240, 240))
    return framed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--up-axis", default="+Y", choices=sorted(AXES))
    parser.add_argument("--size", type=int, default=320)
    args = parser.parse_args()

    mesh = Path(args.mesh)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions, normals, uv, tris = read_glb(mesh)
    if uv is None:
        raise RuntimeError("AXIS_VIEW_UV_MISSING")
    texture = np.asarray(base_colour_image(mesh))

    centred = positions.astype(np.float64) - positions.astype(np.float64).mean(axis=0)
    vertices = centred * (0.5 / float(np.max(np.abs(centred))))
    shaded = np.asarray(normals, np.float64)
    up_axis = unit(AXES[args.up_axis])

    tiles, records = [], []
    for name, axis in AXES.items():
        direction = -unit(axis)          # camera sits on the axis and looks inward
        right, up = basis(direction, up_axis)
        screen = np.stack([vertices @ right / 1.1 + 0.5,
                           0.5 - (vertices @ up) / 1.1], axis=1)
        depth = vertices @ direction
        face_id, bary, _position, _normal, _z = _rasterise(
            screen, depth, vertices, shaded, tris, args.size)
        visible = face_id >= 0
        canvas = np.full(face_id.shape + (3,), 24, np.uint8)
        if visible.any():
            pixel_uv = np.einsum("nc,ncd->nd", bary[visible], uv[tris[face_id[visible]]])
            canvas[visible] = sample(texture, pixel_uv)
        image = Image.fromarray(canvas)
        path = output_dir / f"axis_{name.replace('+', 'p').replace('-', 'm')}.png"
        image.save(path)
        tiles.append(caption(image, f"camera on {name}, up={args.up_axis}"))
        records.append({"axis": name, "camera_direction_mesh_local": direction.tolist(),
                        "camera_up_mesh_local": up.tolist(), "image": str(path),
                        "foreground_pixels": int(visible.sum())})
        print(f"AXIS_VIEW {name} {int(visible.sum())}", flush=True)

    sheet_path = output_dir / f"axis_views_up_{args.up_axis.replace('+', 'p').replace('-', 'm')}.png"
    contact_sheet(tiles).save(sheet_path)
    Path(args.report).write_text(json.dumps({
        "schema": "mesh_axis_views_v1", "mesh": str(mesh), "up_axis": args.up_axis,
        "contact_sheet": str(sheet_path), "views": records}, indent=2), encoding="utf-8")
    print(f"AXIS_VIEWS_DONE sheet={sheet_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
