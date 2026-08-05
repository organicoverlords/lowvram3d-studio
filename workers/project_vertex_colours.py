"""Colour a mesh from orthographic elevations, without a UV atlas.

The full texture route is unwrap -> control maps -> multiview bake, and on a
900k-triangle mesh with thousands of open shells the unwrap alone runs for a
long time. This is the shortcut that answers "what does it look like coloured"
in a couple of minutes: sample the reference elevations directly onto vertices
and store the result as glTF vertex colours.

It is genuinely lower fidelity than the atlas bake and is not a replacement for
it. Colour resolution is bounded by vertex density rather than texel density,
so flat well-tessellated areas look fine and a sparsely tessellated plank does
not. There is no UV seam handling, no albedo normalisation, and no inpainting
of unseen regions. What it does give is a truthful preview of how well the
drawings register to this geometry, which is the thing actually in question.

Registration is by silhouette box. Each drawing is matted, so the bounding box
of its alpha is the subject's extent in that view; the mesh's projected bounding
box is mapped onto it. That assumes the drawing is a real orthographic elevation
of the same subject -- true for a reference sheet, false for a perspective
photograph, which is why the source photo is not used here.

Contributions are weighted by how squarely a surface faces each camera, so a
wall takes its colour from the elevation that saw it face-on rather than from
one that saw it edge-on and smeared.

    py project_vertex_colours.py --mesh in.glb --out coloured.glb \
        --view front=front.png --view right=side.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: (forward, up) per named view, in the mesh's Y-up frame. 'forward' points from
#: the camera toward the subject, matching workers/preview_generated_mesh.py so
#: a coloured mesh and its preview agree about which side is the front.
VIEW_AXES = {
    "front": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "back":  ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "right": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "left":  ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "top":   ((0.0, -1.0, -0.001), (0.0, 0.0, -1.0)),
}

#: Below this facing a surface is edge-on and its samples are smeared along it.
#: Same threshold and reasoning as the atlas baker, so the two routes do not
#: disagree about which view owns a surface.
MIN_FACING = 0.15

#: Higher makes the most square-on view dominate rather than averaging views
#: together, which keeps plank edges from blurring across the corners.
FACING_POWER = 3.0

#: Neutral for vertices no elevation saw -- the underside, and anything in
#: shadow of every camera. Deliberately mid-grey rather than black so unseen
#: area is visible as unseen instead of reading as painted dark.
UNSEEN = (0.45, 0.44, 0.42)


def load_mesh(path):
    import numpy as np
    import trimesh

    scene = trimesh.load(str(path), process=False)
    if hasattr(scene, "geometry"):
        mesh = (scene.to_geometry() if hasattr(scene, "to_geometry")
                else scene.dump(concatenate=True))
    else:
        mesh = scene
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=np.float64),
                           faces=np.asarray(mesh.faces, dtype=np.int64),
                           process=False)


def alpha_bbox(image):
    """Bounding box of the matted subject, or the whole frame if opaque."""
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 4:
        mask = array[..., 3] > 8
        if mask.any():
            ys, xs = np.nonzero(mask)
            return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return 0, 0, image.width, image.height


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--view", action="append", default=[],
                        help="NAME=PATH, repeatable. NAME in "
                             "front/back/left/right/top.")
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    from PIL import Image

    mesh = load_mesh(args.mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    low, high = vertices.min(axis=0), vertices.max(axis=0)
    centre = (low + high) * 0.5
    scale = max(float((high - low).max()), 1e-9)
    unit = (vertices - centre) / scale

    total = np.zeros((len(vertices), 3))
    weight = np.zeros(len(vertices))
    used = {}

    for spec in args.view:
        name, _, path = spec.partition("=")
        name = name.strip().lower()
        if name not in VIEW_AXES:
            raise SystemExit(f"UNKNOWN_VIEW:{name}; expected {sorted(VIEW_AXES)}")
        image = Image.open(path).convert("RGBA")
        x0, y0, x1, y1 = alpha_bbox(image)
        pixels = np.asarray(image, dtype=np.float32) / 255.0

        forward = np.asarray(VIEW_AXES[name][0], float)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(VIEW_AXES[name][1], float))
        right /= max(np.linalg.norm(right), 1e-9)
        true_up = np.cross(right, forward)

        # Project into the camera's image plane, then map the mesh's own
        # silhouette box onto the drawing's alpha box.
        u = unit @ right
        v = unit @ true_up
        span_u = max(u.max() - u.min(), 1e-9)
        span_v = max(v.max() - v.min(), 1e-9)
        px = x0 + (u - u.min()) / span_u * (x1 - x0 - 1)
        py = y1 - 1 - (v - v.min()) / span_v * (y1 - y0 - 1)
        xi = np.clip(np.round(px).astype(int), 0, image.width - 1)
        yi = np.clip(np.round(py).astype(int), 0, image.height - 1)

        sample = pixels[yi, xi]
        facing = -(normals @ forward)              # 1 when square-on
        visible = (facing > MIN_FACING) & (sample[:, 3] > 0.5)
        contribution = np.where(visible, facing, 0.0) ** FACING_POWER

        total += sample[:, :3] * contribution[:, None]
        weight += contribution
        used[name] = {"image": str(Path(path).name),
                      "alpha_bbox": [x0, y0, x1, y1],
                      "vertices_coloured": int(visible.sum())}

    covered = weight > 1e-6
    colours = np.tile(np.asarray(UNSEEN, float), (len(vertices), 1))
    colours[covered] = total[covered] / weight[covered, None]

    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.clip(
            np.concatenate([colours, np.ones((len(vertices), 1))], axis=1),
            0.0, 1.0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))

    receipt = {
        "schema_version": "vertex_colour_projection_v1",
        "mesh": str(Path(args.mesh).resolve()),
        "output": str(out.resolve()),
        "vertices": int(len(vertices)),
        "triangles": int(len(mesh.faces)),
        "views": used,
        "coverage": round(float(covered.mean()), 4),
        "min_facing": MIN_FACING,
        "facing_power": FACING_POWER,
        "note": ("vertex colours, not a UV atlas; colour resolution is bounded "
                 "by vertex density. Preview quality, not production."),
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
