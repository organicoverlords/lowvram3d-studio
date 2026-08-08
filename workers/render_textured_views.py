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
#
# Kept low deliberately. At 0.45 this view rendered the barn at mean luminance
# 28 against the crop's own 38 -- darkening a genuinely dark subject by a
# further quarter and making the asset look worse than it is. The point of this
# view is to judge a texture, so the texture must survive it.
SHADE_STRENGTH = 0.22
# Mid grey, not white. A near-black subject on white reads as a silhouette
# whatever its texture, which is exactly the misjudgement this view exists to
# prevent.
BACKGROUND = 190
# Neutral mid-grey, bright enough to read form against the 190 backdrop without
# competing with it.
CLAY_COLOUR = (158.0, 158.0, 158.0)


def render(vertices, faces, forward, up, size,
           uv=None, texture=None, corner_colours=None, flat_colour=None):
    """Orthographic z-buffered render.

    Colour comes from exactly one of three sources, because the two generators
    hand back different things and the clay pass needs neither:

      texture + uv    per-pixel lookup -- TRELLIS bakes a real baseColorTexture
      corner_colours  (F, 3, 3) per-face-corner RGB, interpolated -- Mini Turbo
                      returns per-vertex COLOR_0 and no UVs at all
      flat_colour     a single RGB, for clay

    Before this, only the first existed, so a Mini Turbo mesh produced no file
    and its native appearance went unseen.
    """
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

    if texture is not None:
        height_px, width_px = texture.shape[:2]
    image = np.full((size, size, 3), float(BACKGROUND))
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

        if texture is not None:
            corners = uv[faces[index]]
            u = w0 * corners[0, 0] + w1 * corners[1, 0] + w2 * corners[2, 0]
            v = w0 * corners[0, 1] + w1 * corners[1, 1] + w2 * corners[2, 1]

            # Bilinear, not nearest. This mattered enormously: with a nearest
            # lookup, a face covering only one or two texels takes a single
            # texel's colour across its whole area, and adjacent faces landing
            # in different texels meet at a hard edge. The result is a surface
            # covered in hard-edged colour plates -- produced entirely by the
            # sampler, from a texture that is perfectly fine.
            #
            # That artifact cost this project most of an evening. The fennec
            # paints were declared unusable by me and by two vision models, a
            # multiview-hallucination diagnosis was written up, an atlas was
            # rebaked at 4x density to chase it, and a UV-fragmentation theory
            # was documented -- all explaining plates that Blender, which
            # filters properly, does not show at all. Rendering the same GLBs
            # through Blender put mean absolute difference between the 1024 and
            # 2048 bakes at 0.45/255.
            #
            # glTF's v origin is the top of the image.
            # Names deliberately distinct from the tile bounds x0/y0/x1/y1
            # above -- reusing them silently shadowed the loop's own bbox.
            fx = u * width_px - 0.5
            fy = v * height_px - 0.5
            tx0 = np.floor(fx).astype(int)
            ty0 = np.floor(fy).astype(int)
            ax = (fx - tx0)[..., None]
            ay = (fy - ty0)[..., None]
            x0c = np.clip(tx0, 0, width_px - 1)
            x1c = np.clip(tx0 + 1, 0, width_px - 1)
            y0c = np.clip(ty0, 0, height_px - 1)
            y1c = np.clip(ty0 + 1, 0, height_px - 1)
            top = texture[y0c, x0c] * (1 - ax) + texture[y0c, x1c] * ax
            bottom = texture[y1c, x0c] * (1 - ax) + texture[y1c, x1c] * ax
            colour = (top * (1 - ay) + bottom * ay) * shade[index]
        elif corner_colours is not None:
            c = corner_colours[index]
            colour = (w0[..., None] * c[0] + w1[..., None] * c[1]
                      + w2[..., None] * c[2]) * shade[index]
        else:
            colour = np.broadcast_to(
                np.asarray(flat_colour, float) * shade[index],
                w0.shape + (3,))
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

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)

    # Whichever appearance the mesh already carries. A mesh with neither still
    # gets the clay row rather than the old empty NOT_APPLICABLE receipt, so
    # "the renderer wrote nothing" stops being a silent outcome.
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    image_source = getattr(getattr(visual, "material", None),
                           "baseColorTexture", None)
    texture = corner_colours = None
    native_kind = "none"
    texture_size = None
    if uv is not None and image_source is not None:
        texture = np.asarray(image_source.convert("RGB"), dtype=np.float64)
        uv = np.asarray(uv, dtype=np.float64)
        native_kind = "texture"
        texture_size = [int(texture.shape[1]), int(texture.shape[0])]
    else:
        colours = getattr(visual, "vertex_colors", None)
        if colours is not None and len(colours) == len(vertices):
            rgb = np.asarray(colours, dtype=np.float64)[:, :3]
            corner_colours = rgb[faces]
            native_kind = "vertex_colour"

    rows = []
    if native_kind != "none":
        rows.append(("native " + native_kind,
                     dict(uv=uv, texture=texture,
                          corner_colours=corner_colours)))
    rows.append(("clay", dict(flat_colour=CLAY_COLOUR)))

    band = 22
    sheet = Image.new(
        "RGB",
        (args.size * len(VIEWS), (args.size + band) * len(rows)),
        (BACKGROUND, BACKGROUND, BACKGROUND))
    draw = ImageDraw.Draw(sheet)
    for row, (label, kwargs) in enumerate(rows):
        top = row * (args.size + band)
        for position, (name, forward) in enumerate(VIEWS):
            panel = render(vertices, faces, forward, UP, args.size, **kwargs)
            sheet.paste(Image.fromarray(panel),
                        (position * args.size, top + band))
            draw.text((position * args.size + 6, top + 6),
                      "%s  %s" % (label, name), fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    receipt = {
        "schema_version": "textured_views_v2",
        "classification": "PROVEN",
        "glb": str(source),
        "out": str(out),
        "views": [name for name, _ in VIEWS],
        "rows": [label for label, _ in rows],
        "native_kind": native_kind,
        "texture_size": texture_size,
        "triangles": int(len(faces)),
    }
    Path(args.receipt or out.with_suffix(".json")).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
