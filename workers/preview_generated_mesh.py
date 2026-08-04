"""Shaded orthographic previews of a generated mesh, on the CPU.

"Is this actually a barn?" should be answerable without building a scene. Inside
a scene the asset competes with fifteen primitives, the lighting, and the
camera, so a bad generation and a bad placement look identical -- and this
project has already spent sessions on exactly that confusion.

Mini Turbo returns geometry with no texture, so shading is by surface normal
against a fixed light. That is enough to read a silhouette and see whether the
result is a closed object or the ragged shell a broken matte produces.

    py -3.12 workers/preview_generated_mesh.py --glb asset.glb --out preview.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VIEWS = {
    # name: (forward, up) in the mesh's own glTF frame, which is Y up.
    "front": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "three_quarter": ((-0.7, -0.2, -0.7), (0.0, 1.0, 0.0)),
    "side": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "top": ((0.0, -1.0, -0.001), (0.0, 0.0, -1.0)),
}
LIGHT = (0.4, 0.8, 0.45)


def render(vertices, faces, forward, up, size):
    """Orthographic z-buffer render, shaded by face normal."""
    import numpy as np

    forward = np.asarray(forward, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right /= max(np.linalg.norm(right), 1e-9)
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
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(lengths, 1e-9, None)
    light = np.asarray(LIGHT, float)
    light /= np.linalg.norm(light)
    shade = np.clip(np.abs(normals @ light), 0.0, 1.0) * 0.75 + 0.2

    image = np.full((size, size), 1.0)
    zbuffer = np.full((size, size), np.inf)
    # Painter's algorithm with a z-test: far faces first, so the z-test only has
    # to resolve the interpenetrating ones.
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
        ax, ay = tri[0]
        bx, by = tri[1]
        cx, cy = tri[2]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-9:
            continue
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        if not inside.any():
            continue
        z = depth[index]
        target = zbuffer[y0:y1, x0:x1]
        write = inside & (z < target)
        target[write] = z
        image[y0:y1, x0:x1][write] = shade[index]
    return (image * 255).astype(np.uint8)


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

    source = Path(args.glb).resolve()
    scene = trimesh.load(str(source), process=False)
    geometries = (list(scene.geometry.values())
                  if hasattr(scene, "geometry") else [scene])
    mesh = trimesh.util.concatenate(geometries) if len(geometries) > 1 else geometries[0]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    sheet = Image.new("RGB", (args.size * len(VIEWS), args.size + 24), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    coverage = {}
    for index, (name, (forward, up)) in enumerate(VIEWS.items()):
        pixels = render(vertices, faces, forward, up, args.size)
        coverage[name] = round(float((pixels < 250).mean()), 4)
        sheet.paste(Image.fromarray(pixels, mode="L").convert("RGB"),
                    (index * args.size, 24))
        draw.text((index * args.size + 8, 6),
                  f"{name}  {coverage[name] * 100:.1f}%", fill=(20, 20, 20))

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    # glTF stores a vertex per face corner wherever normals differ, so a
    # flat-shaded export loads as one disconnected body per triangle. Merge
    # before asking anything about topology, or every mesh looks like soup.
    topology = mesh.copy()
    topology.merge_vertices()
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    receipt = {
        "schema_version": "generated_mesh_preview_v1",
        "classification": "PROVEN",
        "glb": str(source),
        "preview_png": str(out),
        "triangles": int(len(faces)),
        "vertices": int(len(vertices)),
        "extent": [round(float(v), 4) for v in extent],
        "view_coverage": coverage,
        "watertight": bool(topology.is_watertight),
        "euler_number": int(topology.euler_number),
        "body_count": int(len(topology.split(only_watertight=False))),
        "merged_vertices": int(len(topology.vertices)),
    }
    receipt_path = Path(args.receipt) if args.receipt else out.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
