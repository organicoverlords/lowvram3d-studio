"""Render a textured mesh from several angles, on the CPU.

`preview_generated_mesh` shades by surface normal and ignores the texture
entirely, which answers "is this the right shape" and says nothing at all about
appearance. Every judgement of texture quality in this project has so far been
made from either the flat-shaded preview or a full Unreal scene -- one that
cannot show the texture, and one where lighting, materials and placement are
all confounded with it.

This is the missing middle: the asset's own colour, unlit apart from a faint
normal-based shade so the form stays readable, from angles the conditioning
camera did not have. Faces that were never observed appear in their flat fill,
so the boundary between what was seen and what was invented is visible directly
rather than only as a number in a receipt.

    py -3.12 workers/render_textured_views.py --glb textured.glb --out views.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Views chosen to straddle the conditioning camera: the front is what was seen,
# the rear is entirely synthesis, and the three-quarters show the transition.
VIEWS = (
    ("front", (0.0, 0.0, -1.0)),
    ("three_quarter", (-0.7, -0.2, -0.7)),
    ("side", (-1.0, 0.0, 0.0)),
    ("rear", (0.0, 0.0, 1.0)),
)
UP = (0.0, 1.0, 0.0)
LIGHT = (-0.3, 0.6, -0.75)
# How much normal shading to mix over the flat colour. Enough to read the form,
# little enough that the texture is still what dominates.
SHADE_STRENGTH = 0.45


def render(vertices, faces, uv, texture, forward, up, size):
    """Orthographic z-buffered render with per-pixel texture lookup."""
    import numpy as np

    forward = np.asarray(forward, float)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right = right / max(np.linalg.norm(right), 1e-9)
    true_up = np.cross(right, forward)
    basis = np.stack([right, true_up, forward])

    camera = vertices @ basis.T
    low, high = camera[:, :2].min(axis=0), camera[:, :2].max(axis=0)
    span = float(max(high - low)) or 1.0
    margin = size * 0.06
    scale = (size - 2 * margin) / span
    centre = (low + high) * 0.5
    screen = (camera[:, :2] - centre) * scale + size * 0.5
    screen[:, 1] = size - screen[:, 1]

    triangles = screen[faces]
    depth = camera[faces][:, :, 2].mean(axis=1)

    edge1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    normals = np.cross(edge1, edge2)
    normals = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True),
                                1e-9, None)
    light = np.asarray(LIGHT, float)
    light = light / np.linalg.norm(light)
    shade = np.clip(np.abs(normals @ light), 0.0, 1.0)
    shade = 1.0 - SHADE_STRENGTH + SHADE_STRENGTH * shade

    height_px, width_px = texture.shape[:2]
    image = np.full((size, size, 3), 255.0)
    zbuffer = np.full((size, size), np.inf)

    for index in np.argsort(-depth):
        tri = triangles[index]
        x0, y0 = np.floor(tri.min(axis=0)).astype(int)
        x1, y1 = np.ceil(tri.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, size), min(y1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        px, py = xs + 0.5, ys + 0.5
        (ax, ay), (bx, by), (cx, cy) = tri
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-9:
            continue
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = depth[index]
        write = inside & (z < zbuffer[y0:y1, x0:x1])
        if not write.any():
            continue
        zbuffer[y0:y1, x0:x1][write] = z

        corners = uv[faces[index]]
        u = w0 * corners[0, 0] + w1 * corners[1, 0] + w2 * corners[2, 0]
        v = w0 * corners[0, 1] + w1 * corners[1, 1] + w2 * corners[2, 1]
        # glTF's v origin is the top of the image.
        tx = np.clip((u * width_px).astype(int), 0, width_px - 1)
        ty = np.clip((v * height_px).astype(int), 0, height_px - 1)
        colour = texture[ty, tx] * shade[index]
        image[y0:y1, x0:x1][write] = colour[write]
    return np.clip(image, 0, 255).astype("uint8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=420)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    from PIL import Image, ImageDraw

    source, out = Path(args.glb), Path(args.out)
    scene = trimesh.load(str(source), process=False)
    meshes = (list(scene.geometry.values())
              if hasattr(scene, "geometry") else [scene])
    mesh = max(meshes, key=lambda m: len(m.faces))

    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    image_source = getattr(getattr(visual, "material", None),
                           "baseColorTexture", None)
    if uv is None or image_source is None:
        receipt = {"schema_version": "textured_views_v1",
                   "classification": "NOT_APPLICABLE",
                   "reason": "mesh carries no UVs or no base colour texture",
                   "glb": str(source)}
        Path(args.receipt or out.with_suffix(".json")).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    texture = np.asarray(image_source.convert("RGB"), dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    uv = np.asarray(uv, dtype=np.float64)

    panels = []
    for _, forward in VIEWS:
        panels.append(render(vertices, faces, uv, texture, forward,
                             UP, args.size))

    sheet = Image.new("RGB", (args.size * len(panels), args.size + 22), "white")
    draw = ImageDraw.Draw(sheet)
    for position, ((name, _), panel) in enumerate(zip(VIEWS, panels)):
        sheet.paste(Image.fromarray(panel), (position * args.size, 22))
        draw.text((position * args.size + 6, 6), name, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    receipt = {
        "schema_version": "textured_views_v1",
        "classification": "PROVEN",
        "glb": str(source),
        "out": str(out),
        "views": [name for name, _ in VIEWS],
        "texture_size": [int(texture.shape[1]), int(texture.shape[0])],
        "triangles": int(len(faces)),
    }
    Path(args.receipt or out.with_suffix(".json")).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
