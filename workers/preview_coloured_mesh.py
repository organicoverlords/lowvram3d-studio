"""Six-view colour preview of a vertex-coloured mesh, on the CPU.

`preview_generated_mesh.py` shades by surface normal and answers "is the shape
right". This answers "is the colour right", which is a different question and
needs the six canonical view directions the texture pipeline itself uses --
including top and bottom, where the generated views rather than the reference
drawings supply the colour, and so where the bake is most likely to be wrong.

Rasterises near-to-far behind an occupancy mask: once a triangle's bounding box
is fully covered it is skipped, which on a 900k-triangle mesh is most of them
after the first depth layer.

    py preview_coloured_mesh.py --glb coloured.glb --out sheet.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: The six texture-pipeline views, in slot order, plus a three-quarter for
#: reading the object as a whole. (forward, up) in the mesh's Y-up frame.
VIEWS = {
    "front":         ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "right":         ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "back":          ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "left":          ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "top":           ((0.0, -1.0, -0.001), (0.0, 0.0, -1.0)),
    "bottom":        ((0.0, 1.0, -0.001), (0.0, 0.0, 1.0)),
    "three_quarter": ((-0.7, -0.2, -0.7), (0.0, 1.0, 0.0)),
}

#: Shading is deliberately shallow. A strong key light makes a bad bake look
#: dramatic and a good one look muddy; the question here is what colour is on
#: the surface, so the normal term only prevents the object reading as flat.
AMBIENT = 0.55
DIFFUSE = 0.55


def render(vertices, faces, face_colours, forward, up, size):
    import numpy as np

    forward = np.asarray(forward, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right /= max(np.linalg.norm(right), 1e-9)
    true_up = np.cross(right, forward)

    camera = vertices @ np.stack([right, true_up, forward]).T
    low, high = camera[:, :2].min(axis=0), camera[:, :2].max(axis=0)
    span = float(max(high - low)) or 1.0
    margin = size * 0.06
    scale = (size - 2 * margin) / span
    centre = (low + high) * 0.5
    screen = (camera[:, :2] - centre) * scale + size * 0.5
    screen[:, 1] = size - screen[:, 1]

    edge1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    normals = np.cross(edge1, edge2)
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9, None)
    shade = np.abs(normals @ forward) * DIFFUSE + AMBIENT

    triangles = screen[faces]
    depth = camera[faces][:, :, 2].mean(axis=1)

    image = np.ones((size, size, 3))
    covered = np.zeros((size, size), dtype=bool)
    for index in np.argsort(depth):                    # near to far
        tri = triangles[index]
        x0, y0 = np.floor(tri.min(axis=0)).astype(int)
        x1, y1 = np.ceil(tri.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, size), min(y1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        window = covered[y0:y1, x0:x1]
        if window.all():
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        px, py = xs + 0.5, ys + 0.5
        ax, ay = tri[0]
        bx, by = tri[1]
        cx, cy = tri[2]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-12:
            continue
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1) & ~window
        if not inside.any():
            continue
        image[y0:y1, x0:x1][inside] = np.clip(face_colours[index] * shade[index], 0, 1)
        window |= inside
    return (image * 255).astype(np.uint8), covered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=430)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    from PIL import Image, ImageDraw

    scene = trimesh.load(args.glb, process=False)
    if hasattr(scene, "geometry"):
        mesh = (scene.to_geometry() if hasattr(scene, "to_geometry")
                else scene.dump(concatenate=True))
    else:
        mesh = scene
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    colours = getattr(mesh.visual, "vertex_colors", None)
    if colours is None or not len(colours):
        raise SystemExit("NO_VERTEX_COLOURS: this preview needs a coloured mesh")
    # Flat per-face colour: interpolating across a triangle would smooth over
    # exactly the per-vertex sampling limit this preview is meant to expose.
    face_colours = (np.asarray(colours, dtype=np.float64)[:, :3] / 255.0)[faces].mean(axis=1)

    low, high = vertices.min(axis=0), vertices.max(axis=0)
    vertices = (vertices - (low + high) * 0.5) / max(float((high - low).max()), 1e-9)

    size, bar = args.size, 24
    columns = min(args.columns, len(VIEWS))
    rows = (len(VIEWS) + columns - 1) // columns
    sheet = Image.new("RGB", (size * columns, (size + bar) * rows), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    coverage = {}
    for index, (name, (forward, up)) in enumerate(VIEWS.items()):
        pixels, covered = render(vertices, faces, face_colours, forward, up, size)
        coverage[name] = round(float(covered.mean()), 4)
        column, row = index % columns, index // columns
        x, y = column * size, row * (size + bar)
        draw.rectangle([x, y, x + size, y + bar], fill=(28, 28, 32))
        draw.text((x + 8, y + 7), f"{name}   {coverage[name] * 100:.1f}%",
                  fill=(255, 255, 255))
        sheet.paste(Image.fromarray(pixels), (x, y + bar))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    receipt = {
        "schema_version": "coloured_mesh_preview_v1",
        "glb": str(Path(args.glb).resolve()),
        "preview_png": str(out.resolve()),
        "triangles": int(len(faces)),
        "vertices": int(len(vertices)),
        "view_coverage": coverage,
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
