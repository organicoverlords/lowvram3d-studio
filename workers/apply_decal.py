"""Attach a high-resolution decal to generated geometry, without modelling it.

The single biggest thing missing from the textured boat is its identity. Review
put it bluntly: it reads as a generic dark paddle-steamer rather than
specifically "LUCKY DROWN", and the largest perceptual gain per hour is making
the marquee legible again.

That cannot come from the atlas. A 1024 atlas spread over a whole boat gives the
sign a few dozen texels, and the 2048 path crashes on this GPU. Lettering needs
its own texture at its own resolution, which is what a decal is.

The decal is not a floating quad. It reuses the mesh's **own triangles** in the
region of interest, offset fractionally along their average normal, so the sign
follows the curvature of the facade it sits on and cannot float or clip when the
camera moves. A quad would be fine for a distant asset and wrong for a
mid-ground one.

Placement needs no hand-modelling: the caller gives a rectangle in the front
view's normalised image coordinates, and the worker selects the front-facing
triangles whose projection lands inside it, fits a plane through them by PCA,
and builds UVs in that plane's basis.

    py apply_decal.py --mesh textured.glb --decal marquee.png \\
        --out with_decal.glb --roi 0.28,0.06,0.72,0.26
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Front view basis, matching the rest of the pipeline's canonical cameras.
FRONT_FORWARD = (0.0, 0.0, -1.0)
FRONT_UP = (0.0, 1.0, 0.0)

#: A triangle must face the camera at least this squarely to carry the decal.
#: Prevents the sign wrapping onto surfaces that merely happen to project into
#: the rectangle -- the roof behind the arch, or the deck below it.
MIN_FACING = 0.35

#: Offset along the patch normal, as a fraction of the model's diagonal. Large
#: enough to beat depth-buffer precision, small enough not to visibly float.
OFFSET_FRACTION = 0.0012

#: The reference sheet is a flat RGB image, so a transparent background arrives
#: as a light checkerboard. Anything this bright and this desaturated is keyed
#: out rather than painted onto the boat.
KEY_LUMA = 0.72
KEY_SATURATION = 0.18


def key_background(image):
    """Alpha from a light, desaturated checkerboard background."""
    import numpy as np

    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    background = (luma > KEY_LUMA) & (saturation < KEY_SATURATION)

    from scipy import ndimage
    # Keep only background connected to the border: a bright gold highlight in
    # the middle of the sign is not background, however desaturated it reads.
    labels, count = ndimage.label(background)
    if count:
        border = set(labels[0, :]) | set(labels[-1, :]) | \
                 set(labels[:, 0]) | set(labels[:, -1])
        border.discard(0)
        background = np.isin(labels, list(border)) if border else np.zeros_like(background)

    alpha = (~background).astype(np.float64)
    alpha = ndimage.gaussian_filter(alpha, 1.0)      # soften the cut
    out = np.concatenate([rgb, alpha[..., None]], axis=2)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--decal", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--roi", required=True,
                        help="x0,y0,x1,y1 in the front view's normalised image "
                             "coordinates, y down from the top.")
    parser.add_argument("--decal-out", default="")
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    from PIL import Image

    scene = trimesh.load(args.mesh, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    forward = np.asarray(FRONT_FORWARD, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(FRONT_UP, float))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    # Project every vertex into the front view's normalised frame.
    u = vertices @ right
    v = vertices @ true_up
    su = (u - u.min()) / max(np.ptp(u), 1e-9)
    sv = 1.0 - (v - v.min()) / max(np.ptp(v), 1e-9)      # y down

    x0, y0, x1, y1 = (float(t) for t in args.roi.split(","))
    inside_vertex = (su >= x0) & (su <= x1) & (sv >= y0) & (sv <= y1)

    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    facing = -(normals @ forward)
    # A triangle carries the decal when all three corners are in the rectangle
    # and it faces the camera. Requiring all three avoids a fringe of
    # half-covered triangles at the boundary.
    inside_face = inside_vertex[faces].all(axis=1) & (facing > MIN_FACING)
    selected = np.flatnonzero(inside_face)
    if not len(selected):
        raise SystemExit("DECAL_ROI_EMPTY: no front-facing triangles in the "
                         "rectangle; widen --roi or lower MIN_FACING")

    patch_faces = faces[selected]
    used = np.unique(patch_faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    patch_vertices = vertices[used].copy()

    # PCA plane through the patch, for a stable UV basis that follows the
    # facade rather than the world axes.
    centre = patch_vertices.mean(axis=0)
    _, _, basis = np.linalg.svd(patch_vertices - centre, full_matrices=False)
    plane_u, plane_v = basis[0], basis[1]
    patch_normal = normals[selected].mean(axis=0)
    patch_normal /= max(np.linalg.norm(patch_normal), 1e-9)

    # Offset so the decal wins the depth test against the surface it copies.
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    patch_vertices += patch_normal * (OFFSET_FRACTION * diagonal)

    pu = (patch_vertices - centre) @ plane_u
    pv = (patch_vertices - centre) @ plane_v
    uv = np.stack([(pu - pu.min()) / max(np.ptp(pu), 1e-9),
                   (pv - pv.min()) / max(np.ptp(pv), 1e-9)], axis=1)

    decal_image = Image.open(args.decal)
    keyed = Image.fromarray(key_background(decal_image), mode="RGBA")
    if args.decal_out:
        keyed.save(args.decal_out)

    patch = trimesh.Trimesh(vertices=patch_vertices,
                            faces=remap[patch_faces], process=False)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=keyed, alphaMode="BLEND",
        metallicFactor=0.0, roughnessFactor=0.6)
    patch.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    combined = trimesh.Scene()
    combined.add_geometry(mesh, node_name="boat")
    combined.add_geometry(patch, node_name="decal")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(out))

    receipt = {
        "schema_version": "apply_decal_v1",
        "mesh": str(Path(args.mesh).resolve()),
        "decal": str(Path(args.decal).resolve()),
        "output": str(out.resolve()),
        "roi": [x0, y0, x1, y1],
        "decal_pixels": list(keyed.size),
        "patch_triangles": int(len(selected)),
        "patch_vertices": int(len(used)),
        "patch_fraction_of_mesh": round(float(len(selected) / len(faces)), 5),
        "offset_units": round(OFFSET_FRACTION * diagonal, 6),
        "min_facing": MIN_FACING,
        "note": ("decal reuses the mesh's own triangles, so it follows facade "
                 "curvature instead of floating as a quad"),
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
