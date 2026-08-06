"""Composite a high-resolution single-view atlas over a multiview one.

The two texture passes fail in opposite ways and the same asset needs both.

`multiview_texture_projection` gives coverage: six cameras, so the sides and back
carry real texels instead of dilated invention. But every view is 384 px --
MV-Adapter's working resolution -- and the photographed front is squeezed into
that same 384 px along with everything else. On the shaman that puts the face in
roughly 40 pixels, and a 40-pixel face is a pale blob where the source has eye
sockets and a beak.

`fast_texture_projection` gives resolution: it samples the source at whatever
size you hand it, 2048 here, so the face is ~300 px and reads correctly. But it
sees one camera, so 83.5% of the atlas is dilated invention.

Neither is a better tool. The single-view pass is authoritative exactly where it
observed real pixels and worthless elsewhere, so this takes its `observed_mask`
as the authority and keeps multiview everywhere else.

The two atlases must come from the same unwrap. A rotation does not touch UVs,
so a Y-up mesh and its Z-up rotation share a layout and composite directly --
but a re-unwrap does not, and there is no way to detect that from the images, so
the atlas sizes are checked and the rest is the caller's responsibility.

    py atlas_prefer_observed.py --multiview mv/basecolor.png \\
       --single mt_texture2/basecolor.png --observed mt_texture2/observed_mask.png \\
       --mesh uv.glb --out textured.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Texels to feather across the seam between the two sources. The single-view
#: pass has baked lighting the multiview pass does not, so an abrupt boundary
#: reads as a hard edge halfway across a cheek. Wide enough to hide it, narrow
#: enough not to smear the detail this exists to preserve.
FEATHER_TEXELS = 6

#: Erode the observed mask before feathering. Its boundary texels are the ones
#: that grazed the silhouette at a shallow angle, so they carry the source's
#: background as much as its subject -- the same reason `auto_matte` measures the
#: interior of what it removes rather than the whole region.
ERODE_TEXELS = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiview", required=True,
                        help="Base colour atlas from multiview_texture_projection.")
    parser.add_argument("--single", required=True,
                        help="Base colour atlas from fast_texture_projection.")
    parser.add_argument("--observed", required=True,
                        help="observed_mask.png from the single-view pass. Only "
                             "these texels are authoritative.")
    parser.add_argument("--mesh", required=True,
                        help="UV'd GLB to bind the composited atlas to.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--atlas-out", default="")
    parser.add_argument("--feather", type=int, default=FEATHER_TEXELS)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    from PIL import Image
    from scipy import ndimage

    multiview = np.asarray(Image.open(args.multiview).convert("RGB"), np.float32)
    single = np.asarray(Image.open(args.single).convert("RGB"), np.float32)
    observed = np.asarray(Image.open(args.observed).convert("L")) > 127

    if multiview.shape != single.shape or observed.shape != multiview.shape[:2]:
        raise SystemExit(
            f"ATLAS_SIZE_MISMATCH: multiview {multiview.shape[:2]}, "
            f"single {single.shape[:2]}, observed {observed.shape}")

    core = ndimage.binary_erosion(observed, iterations=ERODE_TEXELS)
    if not core.any():
        raise SystemExit("OBSERVED_MASK_EMPTY_AFTER_EROSION")

    # Feather by distance from the core, so the weight reaches 1 well inside the
    # observed region rather than only at its centre.
    distance = ndimage.distance_transform_edt(core)
    weight = np.clip(distance / max(args.feather, 1), 0.0, 1.0)[..., None]
    composited = single * weight + multiview * (1.0 - weight)

    atlas_path = Path(args.atlas_out or
                      Path(args.out).with_suffix("").as_posix() + "_base.png")
    Image.fromarray(np.clip(composited, 0, 255).astype(np.uint8)).save(atlas_path)

    scene = trimesh.load(args.mesh, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=mesh.visual.uv,
        image=Image.open(atlas_path).convert("RGB"))
    mesh.export(args.out)

    receipt = {
        "schema_version": "atlas_prefer_observed_v1",
        "multiview": str(Path(args.multiview).resolve()),
        "single": str(Path(args.single).resolve()),
        "observed_mask": str(Path(args.observed).resolve()),
        "atlas": str(atlas_path),
        "output": str(Path(args.out).resolve()),
        "atlas_size": list(multiview.shape[:2]),
        "observed_texels": int(observed.sum()),
        "observed_fraction": round(float(observed.mean()), 6),
        "core_texels_after_erosion": int(core.sum()),
        "fully_single_sourced_texels": int((weight[..., 0] >= 0.999).sum()),
        "feather_texels": args.feather,
        "erode_texels": ERODE_TEXELS,
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
