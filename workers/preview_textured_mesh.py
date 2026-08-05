"""Six-view preview of a UV-textured mesh, sampling the atlas per pixel.

`preview_coloured_mesh.py` samples one colour per vertex. That was adequate
while the pipeline produced vertex colours, and it became actively misleading
the moment there was a UV atlas: on a 145k-face mesh a 1024 atlas carries far
more detail than 127k vertices can represent, so the preview showed muddy,
smudged surfaces that were an artefact of the sampling rather than a property of
the asset. Several review verdicts in this project were formed on those images.

A decal makes the failure total rather than merely lossy: a 956x476 sign mapped
onto 267 triangles is invisible at vertex resolution, so the check for whether
the decal worked returned "no" when the answer was "yes".

This interpolates UVs per pixel with a depth test, and honours the alpha channel
so keyed-out decal background is not painted. It also renders a whole Scene, not
one mesh, because a decal is a second geometry sharing the boat's framing.

    py preview_textured_mesh.py --glb textured.glb --out sheet.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VIEWS = {
    "front":         ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "right":         ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "back":          ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "left":          ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "top":           ((0.0, -1.0, -0.001), (0.0, 0.0, -1.0)),
    "bottom":        ((0.0, 1.0, -0.001), (0.0, 0.0, 1.0)),
    "three_quarter": ((-0.7, -0.2, -0.7), (0.0, 1.0, 0.0)),
}

#: Shallow shading: the question is what colour is on the surface, so the normal
#: term exists only to stop the object reading as a flat sticker.
AMBIENT = 0.60
DIFFUSE = 0.50

#: Alpha below this is treated as absent, so a keyed decal background does not
#: paint over the boat or claim the depth buffer.
ALPHA_CUT = 0.5

#: How strongly an emissive texture is added into the previewed colour. This is
#: a verification aid, not a render: it exists so that "are the emitters in the
#: right places" is answerable from the sheet. A real engine would light the
#: scene from them.
EMISSIVE_PREVIEW_GAIN = 0.85


def collect(scene):
    """Every geometry as (vertices, faces, uv, texture RGBA float)."""
    import numpy as np
    import trimesh

    geometries = (scene.geometry.items() if hasattr(scene, "geometry")
                  else [("mesh", scene)])
    parts = []
    for name, geometry in geometries:
        visual = geometry.visual
        uv = getattr(visual, "uv", None)
        texture = None
        if uv is not None and getattr(visual, "material", None) is not None:
            image = getattr(visual.material, "baseColorTexture", None)
            if image is not None:
                texture = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
            # Emission, added into the sampled colour. This preview has no
            # lighting model, so there is nothing to add emission *to* -- but
            # without it an emissive texture is invisible here, and a check for
            # "did the lights land in the right places" would silently answer
            # no. Additive is the crudest correct answer: an emitter reads
            # brighter than its surroundings, which is the only property being
            # verified.
            glow = getattr(visual.material, "emissiveTexture", None)
            if glow is not None and texture is not None:
                emissive = np.asarray(glow.convert("RGB"),
                                      dtype=np.float64) / 255.0
                if emissive.shape[:2] == texture.shape[:2]:
                    texture = texture.copy()
                    texture[..., :3] = np.clip(
                        texture[..., :3] + emissive * EMISSIVE_PREVIEW_GAIN,
                        0.0, 1.0)
        if texture is None:
            continue
        parts.append((name,
                      np.asarray(geometry.vertices, dtype=np.float64),
                      np.asarray(geometry.faces, dtype=np.int64),
                      np.asarray(uv, dtype=np.float64),
                      texture))
    if not parts:
        raise SystemExit("NO_TEXTURED_GEOMETRY: this preview needs UVs and a "
                         "base colour texture")
    return parts


def render(parts, forward, up, size):
    import numpy as np

    forward = np.asarray(forward, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right /= max(np.linalg.norm(right), 1e-9)
    true_up = np.cross(right, forward)
    basis = np.stack([right, true_up, forward])

    # One framing across every geometry, so a decal cannot rescale the boat.
    everything = np.vstack([p[1] for p in parts]) @ basis.T
    low, high = everything[:, :2].min(axis=0), everything[:, :2].max(axis=0)
    span = float(max(high - low)) or 1.0
    margin = size * 0.06
    scale = (size - 2 * margin) / span
    centre = (low + high) * 0.5

    image = np.ones((size, size, 3))
    zbuffer = np.full((size, size), np.inf)
    covered = np.zeros((size, size), dtype=bool)

    for _, vertices, faces, uv, texture in parts:
        height, width = texture.shape[:2]
        camera = vertices @ basis.T
        screen = (camera[:, :2] - centre) * scale + size * 0.5
        screen[:, 1] = size - screen[:, 1]

        edge1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        edge2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        normals = np.cross(edge1, edge2)
        normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9, None)
        shade = np.abs(normals @ forward) * DIFFUSE + AMBIENT

        triangles = screen[faces]
        depth = camera[faces][:, :, 2]

        for index in np.argsort(depth.mean(axis=1)):
            tri = triangles[index]
            x0, y0 = np.floor(tri.min(axis=0)).astype(int)
            x1, y1 = np.ceil(tri.max(axis=0)).astype(int) + 1
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
            if abs(area) < 1e-12:
                continue
            w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
            w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            z = w0 * depth[index, 0] + w1 * depth[index, 1] + w2 * depth[index, 2]
            window = zbuffer[y0:y1, x0:x1]
            write = inside & (z < window)
            if not write.any():
                continue
            corners = faces[index]
            u = w0 * uv[corners[0], 0] + w1 * uv[corners[1], 0] + w2 * uv[corners[2], 0]
            v = w0 * uv[corners[0], 1] + w1 * uv[corners[1], 1] + w2 * uv[corners[2], 1]
            tx = np.clip((u % 1.0) * (width - 1), 0, width - 1).astype(int)
            ty = np.clip((1.0 - (v % 1.0)) * (height - 1), 0, height - 1).astype(int)
            texel = texture[ty, tx]
            opaque = write & (texel[..., 3] > ALPHA_CUT)
            if not opaque.any():
                continue
            window[opaque] = z[opaque]
            image[y0:y1, x0:x1][opaque] = np.clip(
                texel[..., :3][opaque] * shade[index], 0.0, 1.0)
            covered[y0:y1, x0:x1] |= opaque
    return (image * 255).astype(np.uint8), covered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=430)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import trimesh
    from PIL import Image, ImageDraw

    parts = collect(trimesh.load(args.glb, process=False))
    size, bar = args.size, 24
    columns = min(args.columns, len(VIEWS))
    rows = (len(VIEWS) + columns - 1) // columns
    sheet = Image.new("RGB", (size * columns, (size + bar) * rows), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    coverage = {}
    for index, (name, (forward, up)) in enumerate(VIEWS.items()):
        pixels, covered = render(parts, forward, up, size)
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
        "schema_version": "textured_mesh_preview_v1",
        "glb": str(Path(args.glb).resolve()),
        "preview_png": str(out.resolve()),
        "geometries": [{"name": p[0], "triangles": int(len(p[2])),
                        "texture": list(p[4].shape[:2])} for p in parts],
        "view_coverage": coverage,
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
