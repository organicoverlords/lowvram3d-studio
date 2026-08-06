"""Render a textured mesh through a control bundle's own camera.

MV-Adapter is conditioned on one image, and how well that image's silhouette
matches the mesh decides how much of the generated result is usable. Feeding it
the source photograph looks obviously right and is not: the mesh is a
reconstruction, so its outline differs from the photo's. On the shaman the front
control matched the photo at IoU 0.677, the structural QA gate failed on the
front and rear views specifically, and the generated front was the weakest of
the six.

The fix is to condition on a render of the *already-textured* mesh instead. Its
silhouette is the mesh's silhouette by construction, so the disagreement is
zero, and the texture it carries is real photographic colour projected in the
high-resolution single-view pass. MV-Adapter is then asked only to continue an
existing surface around the object rather than to reconcile two different
shapes.

The render has to sit in the control camera's frame exactly, or the whole point
is lost. So this does not rasterise anything. `build_mvadapter_cpu_controls`
already wrote, per pixel of each view, which triangle is visible and the
barycentric coordinates within it:

    <view>_triangle_ids.npy    int32, -1 where nothing is visible
    <view>_barycentric.npy     float, two weights per pixel
    <view>_mask.png            the same coverage as a mask

Interpolating the mesh's UVs with those weights and sampling the atlas gives a
pixel-exact render in the control's frame, with no camera maths repeated here
and no opportunity for the two to drift apart.

    py render_view_from_controls.py --mesh uv.glb --atlas basecolor.png \\
       --controls controls_384_audited --view front --out front_render.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Background for pixels no triangle covers. Mid grey rather than black or
#: white: MV-Adapter's own outputs sit on grey, and a black surround would be
#: read as part of the subject by anything that mattes on luminance.
BACKGROUND = (128, 128, 128)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True,
                        help="The UV'd GLB the controls were built from.")
    parser.add_argument("--atlas", required=True,
                        help="Base colour atlas to sample.")
    parser.add_argument("--controls", required=True,
                        help="Control bundle directory.")
    parser.add_argument("--view", default="front")
    parser.add_argument("--out", required=True)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    from PIL import Image

    bundle = Path(args.controls)
    triangle_ids = np.load(bundle / f"{args.view}_triangle_ids.npy")
    barycentric = np.load(bundle / f"{args.view}_barycentric.npy")
    height, width = triangle_ids.shape

    scene = trimesh.load(args.mesh, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    uv = np.asarray(mesh.visual.uv, np.float64)
    faces = np.asarray(mesh.faces)
    if triangle_ids.max() >= len(faces):
        raise SystemExit(
            f"CONTROL_MESH_MISMATCH: control references triangle "
            f"{int(triangle_ids.max())} but the mesh has {len(faces)}. The "
            f"controls were built from different geometry.")

    atlas = np.asarray(Image.open(args.atlas).convert("RGB"), np.float32)
    atlas_h, atlas_w = atlas.shape[:2]

    visible = triangle_ids >= 0
    corners = faces[triangle_ids[visible]]
    wa = barycentric[visible][:, 0][:, None]
    wb = barycentric[visible][:, 1][:, None]
    wc = 1.0 - wa - wb
    texel_uv = (uv[corners[:, 0]] * wc + uv[corners[:, 1]] * wa
                + uv[corners[:, 2]] * wb)

    # glTF UV origin is top-left; atlas row 0 is the top, so v is flipped once.
    xs = np.clip((texel_uv[:, 0] * (atlas_w - 1)).astype(np.int32), 0, atlas_w - 1)
    ys = np.clip(((1.0 - texel_uv[:, 1]) * (atlas_h - 1)).astype(np.int32),
                 0, atlas_h - 1)

    out = np.zeros((height, width, 3), np.uint8)
    out[:] = BACKGROUND
    out[visible] = np.clip(atlas[ys, xs], 0, 255).astype(np.uint8)
    Image.fromarray(out).save(args.out)

    receipt = {
        "schema_version": "render_view_from_controls_v1",
        "mesh": str(Path(args.mesh).resolve()),
        "atlas": str(Path(args.atlas).resolve()),
        "controls": str(bundle.resolve()),
        "view": args.view,
        "output": str(Path(args.out).resolve()),
        "size": [int(width), int(height)],
        "covered_pixels": int(visible.sum()),
        "coverage": round(float(visible.mean()), 6),
        "note": ("silhouette is the control's own coverage, so conditioning "
                 "IoU against this view is 1.0 by construction"),
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
