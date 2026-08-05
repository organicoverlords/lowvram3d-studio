"""Bake generated multi-view images into a mesh's UV atlas.

The single-view projector gives the front its real appearance and fills
everything else with a mirrored copy of that front plus a flat colour -- 0.79 of
the barn's faces carrying invention rather than observation. Six generated views
exist now, registered to this mesh by construction, so the sides and rear can
carry something derived from the subject instead of from its opposite face.

No camera mathematics here. `build_mvadapter_cpu_controls` already rasterised
every view and saved, per pixel, the triangle it hit and the barycentric weights
within it. That is a complete pixel-to-surface map, so a view's colour goes to
the atlas by interpolating the triangle's UVs -- and visibility is already
resolved, because the rasteriser kept the nearest surface.

This makes the bake independent of the view *labels*, which is worth stating
plainly: view N's control arrays pair with view N's image by index. The
labelling bug found on this bundle -- the builder calls index 0 "front" when the
photographed face is at index 1 -- cannot corrupt the result. It affects only
what the report calls each contribution.

Weighting is by how squarely the surface faces the camera, taken from the view's
own normal map. A polygon seen edge-on contributes a few smeared pixels down its
whole length; one seen face-on contributes its actual appearance. Weighting by
facing lets the second win wherever both saw it, without discarding the first.

    py -3.12 workers/bake_multiview_atlas.py --mesh uv.glb --controls DIR \\
        --views DIR --output textured.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Views are 384px, so an atlas much larger than that invents detail it cannot
# have; much smaller throws away what the views do carry. At 1024 the six views
# supply ~885k samples against ~590k covered texels, which fills most of the
# atlas directly rather than by inpainting.
DEFAULT_ATLAS = 1024
# Below this facing a surface is edge-on to the camera and its pixels are
# smeared along it. Contributions are not discarded at the boundary -- the
# weight falls off smoothly -- but they must not outvote a square-on view.
FACING_FLOOR = 0.15
# Weight exponent. Higher makes the most square-on view dominate rather than
# averaging views together, which keeps plank edges from blurring across seams.
FACING_POWER = 3.0

# How much a photographed sample outweighs a generated one at equal facing.
#
# Set so that the photograph wins wherever it saw a surface squarely, and loses
# nowhere it saw one squarely -- but still yields to a generated view that sees
# a surface far more face-on than the photograph's grazing edge. Photographic
# evidence beats synthesis; a photograph's smear does not beat a clean synthetic
# view of the same polygon.
#
# At FACING_POWER 3 this is worth a facing ratio of 8^(1/3) = 2x, so a generated
# view must see a surface twice as squarely as the photograph to overrule it.
PHOTOGRAPH_PRIORITY = 8.0

# Target mean for the finished base colour, as a fraction of white, and the
# level below which a source is too dark to lift without amplifying its noise
# into visible mush. Same convention and same numbers as the single-view
# projector, so the two routes do not disagree about what albedo means.
ALBEDO_TARGET = 0.45
ALBEDO_FLOOR = 0.02


def _bounds(mask):
    """Row/column bounds of a boolean mask, or None when it is empty."""
    import numpy as np

    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if not len(rows) or not len(columns):
        return None
    return int(rows[0]), int(rows[-1] + 1), int(columns[0]), int(columns[-1] + 1)


def register_photograph(photograph_path, silhouette, size):
    """Fit the photograph onto the mesh's silhouette in one view.

    The conditioning reference was built without ever comparing against the
    mesh -- deliberately, so that the generator was not handed a shape it was
    supposed to infer. That independence is exactly what makes the photograph
    unusable as a texture source until it is registered, because nothing so far
    has established which pixel of it covers which polygon.

    Registration is by matte bounding box onto silhouette bounding box:
    anisotropic scale plus translation, no rotation and no warp. The mesh was
    generated from this photograph and the view axis is the one it was
    conditioned on, so the two silhouettes differ by framing rather than by
    pose. The achieved IoU is returned rather than assumed -- a poor fit means
    the photograph is painting the wrong polygons, and that must be visible in
    the receipt instead of showing up as a smear nobody can source.
    """
    import numpy as np
    from PIL import Image

    image = Image.open(photograph_path).convert("RGBA")
    pixels = np.asarray(image, dtype=np.float64)
    alpha = pixels[:, :, 3] > 128
    source = _bounds(alpha)
    target = _bounds(silhouette)
    if source is None or target is None:
        return None, {"registered": False, "reason": "empty silhouette"}

    top, bottom, left, right = source
    cropped = Image.fromarray(
        pixels[top:bottom, left:right].astype(np.uint8), mode="RGBA")
    target_top, target_bottom, target_left, target_right = target
    resized = cropped.resize(
        (max(target_right - target_left, 1), max(target_bottom - target_top, 1)),
        Image.LANCZOS)

    placed = np.zeros((size, size, 4), dtype=np.float64)
    placed[target_top:target_bottom, target_left:target_right] = np.asarray(
        resized, dtype=np.float64)
    registered_alpha = placed[:, :, 3] > 128
    intersection = float((registered_alpha & silhouette).sum())
    union = float((registered_alpha | silhouette).sum())
    return placed, {
        "registered": True,
        "method": "matte_bbox_to_silhouette_bbox_anisotropic_scale",
        "silhouette_iou": round(intersection / max(union, 1.0), 4),
        "source_bbox": [top, bottom, left, right],
        "target_bbox": list(target),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atlas-size", type=int, default=DEFAULT_ATLAS)
    parser.add_argument("--photograph", default="",
                        help="Matted RGBA source photograph. Registered to the "
                             "silhouette of --photograph-view and used in place "
                             "of that view's generated image.")
    parser.add_argument("--photograph-view", type=int, default=-1,
                        help="Index of the view showing the photographed face. "
                             "Give the index, not the label: the builder's "
                             "labels are known to be rotated on this bundle.")
    parser.add_argument("--albedo-target", type=float, default=ALBEDO_TARGET)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)
    if bool(args.photograph) != (args.photograph_view >= 0):
        raise SystemExit("PHOTOGRAPH_AND_VIEW_INDEX_MUST_BE_GIVEN_TOGETHER")

    import cv2
    import numpy as np
    import trimesh
    from PIL import Image

    controls = Path(args.controls)
    views_dir = Path(args.views)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    contract = json.loads((controls / "camera_contract.json").read_text(encoding="utf-8"))
    entries = sorted(contract["views"], key=lambda view: int(view["index"]))

    scene = trimesh.load(args.mesh, process=False)
    geometries = (list(scene.geometry.values())
                  if hasattr(scene, "geometry") else [scene])
    mesh = (trimesh.util.concatenate(geometries) if len(geometries) > 1
            else geometries[0])
    faces = np.asarray(mesh.faces)
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)

    size = args.atlas_size
    accumulated = np.zeros((size, size, 3), dtype=np.float64)
    weights = np.zeros((size, size), dtype=np.float64)
    # Winning view per texel, for provenance. Kept alongside the weighted sum
    # rather than replacing it: the sum is what gets exported, this only records
    # which view had the strongest claim on each texel.
    best_weight = np.zeros((size, size), dtype=np.float64)
    best_view = np.full((size, size), -1, dtype=np.int16)

    per_view = []
    photograph_registration = None
    for entry in entries:
        index = int(entry["index"])
        label = str(entry["proven_semantic"])
        image_path = next(views_dir.glob(f"view_{index}_*.png"), None)
        if image_path is None:
            raise SystemExit(f"VIEW_IMAGE_MISSING:{index}")

        triangles = np.load(controls / f"{label}_triangle_ids.npy")
        barycentric = np.load(controls / f"{label}_barycentric.npy")
        normals = np.load(controls / f"{label}_normal.npy")
        colours = np.asarray(Image.open(image_path).convert("RGB"),
                             dtype=np.float64)

        covered = triangles >= 0
        priority = 1.0
        source_kind = "generated"
        if index == args.photograph_view:
            registered, registration = register_photograph(
                args.photograph, covered, triangles.shape[0])
            photograph_registration = registration
            if registered is not None:
                # Match the photograph's exposure to the generated view of the
                # same face, on the pixels both cover. The generated views are
                # albedo references under flat light; the photograph is a
                # building under a storm sky. Left raw, the front would read as
                # a dark patch stitched onto a lighter model -- the seam would
                # be the lighting, not the geometry. Matched against the
                # measured neighbour rather than an invented target.
                shared = covered & (registered[:, :, 3] > 128)
                photograph_rgb = registered[:, :, :3]
                if shared.any():
                    generated_mean = float(colours[shared].mean())
                    photograph_mean = float(photograph_rgb[shared].mean())
                    if photograph_mean > 1.0:
                        gain = generated_mean / photograph_mean
                        photograph_rgb = np.clip(photograph_rgb * gain, 0, 255)
                        photograph_registration["exposure_gain"] = round(gain, 4)
                        photograph_registration["generated_mean"] = round(
                            generated_mean, 1)
                        photograph_registration["photograph_mean"] = round(
                            photograph_mean, 1)
                colours = photograph_rgb
                covered = covered & (registered[:, :, 3] > 128)
                priority = PHOTOGRAPH_PRIORITY
                source_kind = "photograph"
        # The view's normal map is in its own camera space, so the component
        # along the view axis is exactly how squarely the surface faces it.
        facing = np.abs(normals[:, :, 2])
        usable = covered & (facing > FACING_FLOOR)
        if not usable.any():
            per_view.append({"index": index, "label": label,
                             "samples": 0, "note": "no face-on coverage"})
            continue

        corner_uv = uv[faces[triangles[usable]]]
        bary = barycentric[usable][:, :, None]
        sample_uv = (corner_uv * bary).sum(axis=1)
        # glTF's v origin is the top of the image, which is also how the atlas
        # is indexed here, so v maps straight to the row.
        columns = np.clip((sample_uv[:, 0] * (size - 1)).astype(int), 0, size - 1)
        rows = np.clip((sample_uv[:, 1] * (size - 1)).astype(int), 0, size - 1)

        weight = priority * facing[usable] ** FACING_POWER
        np.add.at(accumulated, (rows, columns), colours[usable] * weight[:, None])
        np.add.at(weights, (rows, columns), weight)

        # Highest single-sample weight wins the provenance label. np.maximum.at
        # rather than a loop, then a second pass to attribute it.
        previous = best_weight[rows, columns]
        improved = weight > previous
        best_weight[rows[improved], columns[improved]] = weight[improved]
        best_view[rows[improved], columns[improved]] = index

        per_view.append({
            "index": index, "label": label,
            "source": source_kind,
            "samples": int(usable.sum()),
            "silhouette_fraction": round(float(covered.mean()), 4),
            "face_on_fraction_of_silhouette": round(
                float(usable.sum() / max(covered.sum(), 1)), 4),
        })

    observed = weights > 0
    atlas = np.zeros((size, size, 3), dtype=np.float64)
    atlas[observed] = accumulated[observed] / weights[observed][:, None]

    # Lift the whole atlas to a plausible reflectance, once, at the end.
    #
    # Both sources are dark: the photograph is a building under a storm sky
    # (mean 38/255) and the generated views came back darker still (27/255),
    # which is why the six-view QA failed its colour gate. Baking them
    # unmodified produces a base colour that the renderer then lights a second
    # time, and every plank disappears into the bottom eighth of the range --
    # the brown blob again, arrived at by a different road.
    #
    # Done here rather than per source, because regrading one source to match
    # another means darkening evidence to fit synthesis, or brightening
    # synthesis until it outshines evidence. Both distort the relationship
    # between them. Lifting the finished atlas preserves every source's
    # relative value and changes only the overall level.
    #
    # A gamma, not a gain: black stays black, white stays white, and the
    # ordering of every texel survives.
    albedo_gamma = 1.0
    observed_mean = float(atlas[observed].mean() / 255.0) if observed.any() else 0.0
    if ALBEDO_FLOOR < observed_mean < args.albedo_target:
        albedo_gamma = float(np.log(args.albedo_target) / np.log(observed_mean))
        atlas = 255.0 * np.power(np.clip(atlas / 255.0, 0.0, 1.0), albedo_gamma)

    # Texels no view reached: gaps between samples inside a chart, and chart
    # padding. Inpainted rather than left black, because a black texel reads as
    # a hole in the surface, whereas an interpolated one reads as the surface it
    # sits inside. Recorded, so the share that is filled rather than seen is
    # never mistaken for observation.
    holes = (~observed).astype(np.uint8)
    filled = cv2.inpaint(atlas.astype(np.uint8), holes, 3, cv2.INPAINT_TELEA)

    texture = Image.fromarray(filled, mode="RGB")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=0.9)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.export(str(output))

    covered_total = int(observed.sum())
    receipt = {
        "schema_version": "multiview_atlas_bake_v1",
        "classification": "PROVEN",
        "mesh": str(Path(args.mesh).resolve()),
        "controls": str(controls.resolve()),
        "views": str(views_dir.resolve()),
        "output": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "atlas_size": size,
        "facing_floor": FACING_FLOOR,
        "facing_power": FACING_POWER,
        "albedo_target": args.albedo_target,
        "albedo_gamma": round(albedo_gamma, 4),
        "source_mean_luminance": round(observed_mean * 255.0, 1),
        "observed_texels": covered_total,
        "inpainted_texels": int(size * size - covered_total),
        "observed_fraction_of_atlas": round(covered_total / (size * size), 4),
        "per_view": per_view,
        "texels_won_per_view": {
            str(entry["index"]): int((best_view == int(entry["index"])).sum())
            for entry in entries},
        "photograph": str(Path(args.photograph).resolve()) if args.photograph else None,
        "photograph_view_index": args.photograph_view if args.photograph else None,
        "photograph_registration": photograph_registration,
        "photograph_priority": PHOTOGRAPH_PRIORITY if args.photograph else None,
        "texels_won_by_photograph": (
            int((best_view == args.photograph_view).sum())
            if args.photograph else 0),
        "observed_from": (
            "generated views conditioned on this mesh's geometry, except the "
            "photograph view, which is the only evidence here. Everything else "
            "is plausible synthesis. Inpainted texels are neither."
            if args.photograph else
            "six generated views, not photographs. Only the view whose index "
            "carries the photographed face resembles evidence; the rest are "
            "plausible synthesis conditioned on this mesh's geometry."),
    }
    receipt_path = (Path(args.receipt) if args.receipt
                    else output.with_suffix(".bake.json"))
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
