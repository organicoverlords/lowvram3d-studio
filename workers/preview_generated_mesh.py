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
    parser.add_argument(
        "--up-axis", choices=("y", "z", "auto"), default="auto",
        help="Up axis of the source mesh. glTF is Y-up by specification and "
             "Hunyuan3D emits Y-up, but the online service's exports are Z-up, "
             "which renders every character flat on its back. 'auto' guesses "
             "from which axis the mesh rests on.")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    from PIL import Image, ImageDraw

    source = Path(args.glb).resolve()
    scene = trimesh.load(str(source), process=False)
    if hasattr(scene, "geometry"):
        # Flatten through the scene graph, not around it. Concatenating
        # `scene.geometry.values()` directly discards every node transform, so a
        # multi-node glTF renders with its parts stacked at the origin at
        # whatever scale they were authored. Hunyuan3D emits a single node and
        # never exposed this; the online service's exports do not, and they
        # rendered as an unrecognisable close-up of one component.
        mesh = (scene.to_geometry() if hasattr(scene, "to_geometry")
                else scene.dump(concatenate=True))
    else:
        mesh = scene
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Which way is up. The cameras below assume Y-up, which glTF mandates and
    # Hunyuan3D honours -- but the online service's exports are Z-up, and every
    # one of them renders flat on its back, which reads as a broken mesh rather
    # than a broken camera.
    #
    # The auto guess uses the one thing a generator almost always does: it rests
    # the subject on the ground, so the up axis is the one whose minimum sits at
    # zero while the others straddle it. Ambiguity falls back to Y, the
    # specification default, rather than to a coin toss.
    up_axis = args.up_axis
    if up_axis == "auto":
        low = vertices.min(axis=0)
        high = vertices.max(axis=0)
        span = np.maximum(high - low, 1e-9)
        grounded = np.abs(low) / span          # 0 means the mesh sits on it
        centred = np.abs(low + high) / span    # 0 means it straddles it
        candidates = [i for i in (1, 2) if grounded[i] < 0.02 and centred[i] > 0.5]
        up_axis = {1: "y", 2: "z"}.get(candidates[0], "y") if len(candidates) == 1 else "y"

    if up_axis == "z":
        # Z-up to Y-up: (x, y, z) -> (x, z, -y). A rotation, so handedness and
        # therefore winding order are preserved and the shading stays correct.
        vertices = np.column_stack(
            [vertices[:, 0], vertices[:, 2], -vertices[:, 1]])

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
    #
    # Merging makes the *statistics* right and leaves the *file* wrong, and for
    # a long time this receipt reported a tidy 9 bodies for a mesh that was
    # 149,960 disconnected triangles on disk. Unreal does not merge on import:
    # it took six times the vertex data it needed, which is the likeliest cause
    # of the import handler timeouts. The ratio is now reported rather than
    # quietly normalised away, because a known quirk that is always corrected
    # for is indistinguishable from one that has been fixed.
    #
    # Weld by *position* only. merge_vertices() refuses to merge vertices whose
    # UVs differ, and a textured mesh must split vertices at every UV seam --
    # so a correctly seamed mesh reports as thousands of "bodies" while the
    # unwelded soup it replaced reported nine. Body count is a question about
    # geometry, and the answer must not depend on how the thing is textured.
    topology = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
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
        "vertex_split_ratio": round(
            len(vertices) / max(len(topology.vertices), 1), 3),
        # 3.0 is exactly soup: every triangle carrying its own three vertices.
        # A correctly textured mesh sits well below it -- the barn lands at 1.70
        # once its unobserved faces are split off at UV seams, which is real
        # seaming rather than a broken export. The gate is set above that.
        "unwelded": bool(len(vertices) > len(topology.vertices) * 2.5),
    }
    receipt_path = Path(args.receipt) if args.receipt else out.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
