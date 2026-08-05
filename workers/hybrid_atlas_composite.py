"""Composite reference elevations into a generated PBR atlas.

Two texture sources, each strong where the other is weak.

TRELLIS.2's native texturing covers the whole surface -- roof, underside,
occluded interiors -- with consistent UVs and PBR channels, and it is what makes
the asset shippable. But it regresses to a muted grey-brown and smudges authored
detail, which is the documented weakness of native 3D texturing on flat
geometry, and it is obvious next to the concept art.

The reference sheet's orthographic elevations are the actual artwork: correct
hue, full contrast, real lettering and ornament. They only cover four sides, and
projecting them alone leaves the roof and underside blank.

So: keep the generated atlas as the base, and paint the artwork over it wherever
an elevation genuinely sees a surface. Everything else is untouched.

The compositing happens in **atlas space**, not on vertices. Each texel is
mapped back to its 3D position and normal by rasterising the mesh through its
own UVs, then projected into each elevation camera. Colour resolution is then
bounded by the atlas, not by tessellation -- which is the whole reason the
earlier vertex-colour attempt looked faceted.

Deliberate choices, each with a reason:

- **Chroma from the art, luma blended.** The elevations are painted with their
  own light and shade; taking their luminance wholesale would fight the base
  colour and double-darken where painted shadow lands on already-dark texture.
  Hue and saturation are what the generated atlas gets wrong, so that is what is
  replaced. The `--luma` weight exposes the trade rather than hiding it.
- **Linear light.** Blending sRGB values directly darkens midtones; the mix is
  done after the EOTF and re-encoded on write.
- **Facing-weighted, with a floor.** A texel takes colour from the elevation
  that saw it square-on. Below `MIN_FACING` the surface is edge-on and its
  samples are smeared along it, so it keeps the generated colour instead.
- **Occlusion-tested.** An orthographic elevation cannot see through the boat,
  so a texel on the far side of the hull must not receive the near side's paint.
  A depth buffer per view rejects those.
- **Feathered.** A hard cut between painted and generated regions is a visible
  seam. Coverage is smoothed so the transition falls off over a few texels.

    py hybrid_atlas_composite.py --mesh textured.glb --out hybrid.glb \\
        --view front=front.png --view right=side.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VIEW_AXES = {
    "front": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "back":  ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "right": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "left":  ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "top":   ((0.0, -1.0, -0.001), (0.0, 0.0, -1.0)),
}

MIN_FACING = 0.30          # below this a surface is too edge-on to sample
FACING_POWER = 3.0         # how strongly the most square-on view dominates
OCCLUSION_TOLERANCE = 0.01  # depth slack, in normalised model units

#: Seam softening. The first pass used 3 and the palette break between painted
#: and generated regions was still visible; a wider ramp trades a little
#: sharpness at the boundary for a transition the eye does not catch.
FEATHER_TEXELS = 12

#: Radius, in texels, that painted colour is pushed outward past the edge of
#: each UV island. Without it, mipmapping and bilinear filtering pull the
#: unwritten background in at island borders and every chart edge shows as a
#: dark seam at distance.
DILATION_TEXELS = 8

#: De-lighting. The elevations are paintings: they carry their own broad shadow
#: gradients, and multiplying those into an albedo that already contains baked
#: shading darkens everything twice. Dividing by a heavily blurred copy of the
#: art's own luminance removes the low-frequency lighting while leaving the
#: high-frequency detail -- lettering, mullions, planking -- untouched. The
#: radius is a fraction of the image's larger side so it scales with the panel.
#:
#: 0.10 with a [0.5, 2.0] clamp was too aggressive: it allowed a 2x swing on
#: any region, which exaggerates dark artefacts rather than flattening them. A
#: tighter radius removes only the broad gradient, and a tighter clamp keeps
#: this a correction instead of a re-light.
DELIGHT_RADIUS_FRACTION = 0.06
DELIGHT_GAIN_MIN = 0.65
DELIGHT_GAIN_MAX = 1.60


def srgb_to_linear(x):
    import numpy as np
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    import numpy as np
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def delight(linear_rgb, alpha):
    """Divide out the artwork's own low-frequency lighting.

    A painted elevation is not an albedo: the artist put shadow under every
    balcony and light on every upper surface. Projected raw, that shading is
    multiplied again by the renderer, and by whatever shading the generated
    base colour already baked in. Dividing by a blurred copy of the art's own
    luminance flattens the broad gradients and keeps the detail.

    Blurring is masked so background pixels outside the matte cannot bleed in
    and darken the subject's edges.
    """
    import numpy as np
    from scipy import ndimage

    weights = np.array([0.2126, 0.7152, 0.0722])
    luma = linear_rgb @ weights
    mask = (alpha > 0.5).astype(np.float64)
    radius = max(3.0, DELIGHT_RADIUS_FRACTION * max(linear_rgb.shape[:2]))

    blurred = ndimage.gaussian_filter(luma * mask, radius)
    norm = ndimage.gaussian_filter(mask, radius)
    low = np.where(norm > 1e-6, blurred / np.maximum(norm, 1e-6), 0.0)

    # Normalise around the subject's own mean so de-lighting changes the
    # distribution of light, not the overall exposure.
    inside = mask > 0.5
    if not inside.any():
        return linear_rgb
    mean = float(luma[inside].mean())
    gain = np.where(low > 1e-4, mean / np.maximum(low, 1e-4), 1.0)
    # Bounded: this is a correction, not a re-light.
    gain = np.clip(gain, DELIGHT_GAIN_MIN, DELIGHT_GAIN_MAX)
    return np.clip(linear_rgb * gain[..., None], 0.0, 1.0)


def tone_curve(linear_rgb, contrast, shadows, highlights, saturation):
    """S-curve contrast, shadow/highlight trim and saturation, in linear light.

    Luna's measured critique of the generated atlas was: too many midtones,
    shadows lifted and grey-brown, highlights neither bright nor selective,
    saturation short in the gold and the lit windows. These are the four knobs
    that address exactly that, in that order.
    """
    import numpy as np

    x = np.clip(linear_rgb, 0.0, 1.0)
    weights = np.array([0.2126, 0.7152, 0.0722])
    luma = np.clip(x @ weights, 1e-6, 1.0)

    # Smoothstep-based S-curve around mid grey, strength `contrast`.
    pivot = 0.18
    curved = np.where(
        luma < pivot,
        pivot * (luma / pivot) ** (1.0 + contrast),
        1.0 - (1.0 - pivot) * ((1.0 - luma) / (1.0 - pivot)) ** (1.0 + contrast))
    # Shadow pull-down and highlight lift, weighted to their own ends.
    dark = np.clip(1.0 - luma / pivot, 0.0, 1.0)
    bright = np.clip((luma - pivot) / (1.0 - pivot), 0.0, 1.0)
    curved = curved * (1.0 - shadows * dark) * (1.0 + highlights * bright)

    scaled = x * (curved / luma)[..., None]
    if saturation != 1.0:
        grey = (scaled @ weights)[..., None]
        scaled = grey + (scaled - grey) * saturation
    return np.clip(scaled, 0.0, 1.0)


def dilate_atlas(pixels, written, radius):
    """Push written colour outward past island edges, killing mipmap seams."""
    import numpy as np
    from scipy import ndimage

    _, indices = ndimage.distance_transform_edt(~written, return_indices=True)
    near = pixels[indices[0], indices[1]]
    distance = ndimage.distance_transform_edt(~written)
    grow = (~written) & (distance <= radius)
    out = pixels.copy()
    out[grow] = near[grow]
    return out, grow


def rasterise_uv(uv, faces, positions, normals, size):
    """Per-texel 3D position and normal, by rasterising the mesh in UV space."""
    import numpy as np

    pos_map = np.zeros((size, size, 3))
    nrm_map = np.zeros((size, size, 3))
    filled = np.zeros((size, size), dtype=bool)

    tri_uv = uv[faces] * (size - 1)
    tri_uv[:, :, 1] = (size - 1) - tri_uv[:, :, 1]     # glTF v is bottom-up
    tri_pos = positions[faces]
    tri_nrm = normals[faces]

    for index in range(len(faces)):
        tri = tri_uv[index]
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
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        bary = np.stack([w0, w1, w2], axis=-1)[inside]
        pos_map[y0:y1, x0:x1][inside] = bary @ tri_pos[index]
        nrm_map[y0:y1, x0:x1][inside] = bary @ tri_nrm[index]
        filled[y0:y1, x0:x1] |= inside

    lengths = np.linalg.norm(nrm_map, axis=2, keepdims=True)
    nrm_map = nrm_map / np.clip(lengths, 1e-9, None)
    return pos_map, nrm_map, filled


def depth_buffer(points, forward, right, true_up, resolution=1024):
    """Nearest-surface depth per view cell, for rejecting occluded texels."""
    import numpy as np

    u = points @ right
    v = points @ true_up
    d = points @ forward
    ui = np.clip(((u - u.min()) / max(np.ptp(u), 1e-9) * (resolution - 1)),
                 0, resolution - 1).astype(int)
    vi = np.clip(((v - v.min()) / max(np.ptp(v), 1e-9) * (resolution - 1)),
                 0, resolution - 1).astype(int)
    buffer = np.full((resolution, resolution), np.inf)
    np.minimum.at(buffer, (vi, ui), d)
    return buffer, (u.min(), np.ptp(u), v.min(), np.ptp(v))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Textured GLB with UVs.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--view", action="append", default=[])
    parser.add_argument("--atlas-out", default="")
    # Default 1.0: take the artwork's luminance as well as its chroma. The
    # first version defaulted to 0.35, keeping most of the generated luminance,
    # and that was backwards -- the generated luminance is itself one of the
    # main defects (lifted grey-brown shadows, no local contrast). Full
    # replacement is safe only because the art is de-lit first; without
    # --delight this double-darkens.
    parser.add_argument("--luma", type=float, default=1.0,
                        help="How much of the (de-lit) artwork luminance to "
                             "take in fully-painted regions.")
    parser.add_argument("--delight", action="store_true", default=True,
                        help="Divide the artwork's own low-frequency lighting "
                             "out before use. On by default.")
    parser.add_argument("--no-delight", dest="delight", action="store_false")
    parser.add_argument("--saturation", type=float, default=1.2,
                        help="Saturation applied to the composited atlas.")
    parser.add_argument("--contrast", type=float, default=0.28,
                        help="S-curve strength, ~0.20-0.35 per the review.")
    parser.add_argument("--shadows", type=float, default=0.15,
                        help="Pull shadows down; counteracts the lifted, "
                             "grey-brown blacks in the generated atlas.")
    parser.add_argument("--highlights", type=float, default=0.15,
                        help="Lift highlights, so gold and lit windows read.")
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    from PIL import Image
    from scipy import ndimage

    scene = trimesh.load(args.mesh, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    texture = mesh.visual.material.baseColorTexture.convert("RGB")
    size = texture.size[0]
    base_srgb = np.asarray(texture, dtype=np.float64) / 255.0
    base = srgb_to_linear(base_srgb)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    unit = (vertices - (low + high) * 0.5) / max(float((high - low).max()), 1e-9)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    pos_map, nrm_map, filled = rasterise_uv(uv, faces, unit, normals, size)

    accumulated = np.zeros((size, size, 3))
    weight = np.zeros((size, size))
    per_view = {}

    for spec in args.view:
        name, _, path = spec.partition("=")
        name = name.strip().lower()
        if name not in VIEW_AXES:
            raise SystemExit(f"UNKNOWN_VIEW:{name}")
        art = Image.open(path).convert("RGBA")
        art_pixels = np.asarray(art, dtype=np.float64) / 255.0
        alpha = art_pixels[..., 3]
        art_linear = srgb_to_linear(art_pixels[..., :3])
        if args.delight:
            art_linear = delight(art_linear, alpha)
        art_pixels = np.concatenate([art_linear, alpha[..., None]], axis=-1)
        if not (alpha > 0.5).any():
            raise SystemExit(f"VIEW_FULLY_TRANSPARENT:{name}")
        ys, xs = np.nonzero(alpha > 0.5)
        ax0, ax1, ay0, ay1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1

        forward = np.asarray(VIEW_AXES[name][0], float)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(VIEW_AXES[name][1], float))
        right /= max(np.linalg.norm(right), 1e-9)
        true_up = np.cross(right, forward)

        buffer, (umin, uptp, vmin, vptp) = depth_buffer(unit, forward, right, true_up)

        flat_pos = pos_map[filled]
        u = flat_pos @ right
        v = flat_pos @ true_up
        d = flat_pos @ forward
        su = (u - umin) / max(uptp, 1e-9)
        sv = (v - vmin) / max(vptp, 1e-9)

        # Occlusion: compare against the nearest surface along this ray.
        bi = np.clip((sv * (buffer.shape[0] - 1)).astype(int), 0, buffer.shape[0] - 1)
        bj = np.clip((su * (buffer.shape[1] - 1)).astype(int), 0, buffer.shape[1] - 1)
        visible = d <= buffer[bi, bj] + OCCLUSION_TOLERANCE

        px = np.clip(ax0 + su * (ax1 - ax0 - 1), 0, art.width - 1).astype(int)
        py = np.clip(ay1 - 1 - sv * (ay1 - ay0 - 1), 0, art.height - 1).astype(int)
        sample = art_pixels[py, px]

        facing = -(nrm_map[filled] @ forward)
        usable = visible & (facing > MIN_FACING) & (sample[:, 3] > 0.5)
        contribution = np.where(usable, np.clip(facing, 0, 1), 0.0) ** FACING_POWER

        view_colour = np.zeros((size, size, 3))
        view_weight = np.zeros((size, size))
        # `sample` is already linear: the art was decoded and de-lit above.
        view_colour[filled] = sample[:, :3] * contribution[:, None]
        view_weight[filled] = contribution
        accumulated += view_colour
        weight += view_weight
        per_view[name] = {"art": Path(path).name,
                          "texels_painted": int((contribution > 0).sum())}

    painted = weight > 1e-6
    art_linear = np.zeros_like(base)
    art_linear[painted] = accumulated[painted] / weight[painted, None]

    # Feather the coverage so the painted/generated boundary is not a hard edge.
    coverage = ndimage.gaussian_filter(painted.astype(np.float64), FEATHER_TEXELS)
    coverage = np.clip(coverage / max(coverage.max(), 1e-9), 0.0, 1.0)
    coverage[painted] = 1.0
    coverage[~filled] = 0.0

    # Chroma from the art, luminance mostly from the generated atlas. The art
    # carries its own painted light; taking it wholesale double-darkens shadow
    # that lands on already-dark texture.
    luma_weights = np.array([0.2126, 0.7152, 0.0722])
    base_luma = base @ luma_weights
    art_luma = np.clip(art_linear @ luma_weights, 1e-6, None)
    chroma = art_linear / art_luma[..., None]
    if args.saturation != 1.0:
        chroma = 1.0 + (chroma - 1.0) * args.saturation
    target_luma = base_luma * (1.0 - args.luma) + (art_linear @ luma_weights) * args.luma
    transferred = np.clip(chroma * target_luma[..., None], 0.0, 1.0)

    blended = base * (1.0 - coverage[..., None]) + transferred * coverage[..., None]

    # Grade the whole atlas, not just the painted part: the generated regions
    # are the ones with lifted shadows and no local contrast, and grading only
    # the painted half would make the seam worse rather than better.
    blended = tone_curve(blended, args.contrast, args.shadows,
                         args.highlights, args.saturation)

    out_pixels = (linear_to_srgb(blended) * 255.0).round().astype(np.uint8)
    # Bleed colour past island edges so filtering cannot sample the background.
    out_pixels, grown = dilate_atlas(out_pixels, filled, DILATION_TEXELS)

    atlas_path = Path(args.atlas_out) if args.atlas_out else \
        Path(args.out).with_name(Path(args.out).stem + "_base.png")
    Image.fromarray(out_pixels).save(atlas_path)

    material = mesh.visual.material
    material.baseColorTexture = Image.fromarray(out_pixels)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))

    receipt = {
        "schema_version": "hybrid_atlas_composite_v1",
        "mesh": str(Path(args.mesh).resolve()),
        "output": str(out.resolve()),
        "atlas": str(atlas_path.resolve()),
        "atlas_size": size,
        "views": per_view,
        "texels_in_atlas": int(filled.sum()),
        "texels_painted": int(painted.sum()),
        "painted_fraction_of_surface": round(float(painted.sum() / max(filled.sum(), 1)), 4),
        "luma_transfer": args.luma,
        "delight": bool(args.delight),
        "grade": {"contrast": args.contrast, "shadows": args.shadows,
                  "highlights": args.highlights, "saturation": args.saturation},
        "min_facing": MIN_FACING,
        "feather_texels": FEATHER_TEXELS,
        "dilated_texels": int(grown.sum()),
        # Native metallic/roughness are left untouched on purpose. Deriving
        # them from painted highlights would read lighting as material and
        # scatter false metal across wood and shadow.
        "material_channels": "native, unmodified",
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
