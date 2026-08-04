"""Turn semantic regions into world-space actor specs the builders can place.

The structural builders previously received regions with no geometry, so every
actor was placed at an arbitrary transform and scaled to an arbitrary size --
which is what a field of identical cubes actually is. Segmentation now carries
a measured median depth and a pixel footprint per region, and those two facts
are enough to recover a real position and size by unprojecting the footprint
through the recovered camera.

Each class gets the treatment its geometry deserves rather than one generic
box:

``terrain``      a ground plane at the region's depth, sized to its footprint
``architecture`` a volume standing on the ground, height from its pixel extent
``vegetation``   scatter points across the footprint, not one merged blob
``water``        a flat surface, no vertical extent
``sky``          a backdrop, excluded from placement entirely
``crossing``     a strip following the footprint's long axis

Positions are in Unreal's frame and centimetres, with +X forward from the
camera, so a builder can spawn them directly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CM_PER_M = 100.0

# Vegetation is scattered rather than merged: one actor covering a whole tree
# line reads as a wall, several read as trees.
SCATTER_TARGET = {"vegetation": 12, "prop": 6}

# Classes that describe appearance rather than occupancy.
NON_PLACED = {"sky"}

# Below this the model was not confident enough for the label to be treated as
# an object. Set from measurement: on this scene the real regions score
# 0.89-0.98 and the two spurious ones 0.62.
MIN_PLACEMENT_CONFIDENCE = 0.7
# Classes where an uncertain label means the thing may not exist at all. A
# surface is exempt: unsure between "field" and "grass" is still ground.
CONFIDENCE_GATED_LAYERS = {"architecture", "prop", "object", "clutter"}

# A canopy clump sitting this far above the ground needs something holding it
# up, and a trunk is that thin a fraction of the crown it carries.
MIN_TRUNK_HEIGHT = 0.5
TRUNK_GIRTH_FRACTION = 0.12
MIN_TRUNK_GIRTH = 0.15


def unproject(u: float, v: float, depth: float, fov_x_deg: float,
              aspect: float) -> tuple[float, float, float]:
    """Normalised image coords (0..1) plus depth -> Unreal metres, +X forward.

    The camera looks down +X with +Y right and +Z up, so image x maps to Y and
    image y maps to inverted Z.
    """
    half_x = math.tan(math.radians(fov_x_deg) * 0.5)
    half_y = half_x / aspect
    ndc_x = (u - 0.5) * 2.0
    ndc_y = (v - 0.5) * 2.0
    return (depth, ndc_x * half_x * depth, -ndc_y * half_y * depth)


def _receding_surface(region: dict[str, Any], fov_x: float, aspect: float
                      ) -> tuple[tuple[float, float, float], float, float]:
    """Box a ground-like surface that recedes from the camera.

    A surface is not a volume seen at one distance, and treating it as one gets
    both its size and its position wrong. Ground here spans 1.6 m to 10.3 m, so
    sizing its lateral extent at the *median* depth gave a 4.9 m strip when the
    frustum is 21.7 m wide at the far edge, and centring it on the median put
    part of it behind the camera.

    So: extend along +X across the whole observed depth band, and take the
    lateral width at the band's far edge, where a receding plane is widest.
    Overshooting the near end is harmless -- ground should be underfoot -- while
    undershooting the far end leaves the scene standing on a floating strip.

    Height is the plane's own height below the camera, taken at the footprint's
    vertical centre; a flat ground gives the same answer at any depth.
    """
    x0, y0, x1, y1 = region["bbox_norm_xyxy"]
    band = region.get("depth_m", {})
    median = float(band.get("median") or 10.0)
    near = float(band.get("near") or median)
    far = max(float(band.get("far") or median), near + 0.5)

    left = unproject(x0, (y0 + y1) * 0.5, far, fov_x, aspect)
    right = unproject(x1, (y0 + y1) * 0.5, far, fov_x, aspect)
    lateral = abs(right[1] - left[1])

    # Prefer the measured height of the surface's own points. Unprojecting the
    # bbox centre answers a different question -- how far below the camera the
    # middle of the box is -- and for a plane spanning an order of magnitude in
    # depth that is neither its near height nor its far one.
    # Negated: the measurement is a drop below the camera in MoGe's Y-down
    # frame, and this function returns a position in Unreal's Z-up one.
    drop = (region.get("surface") or {}).get("drop_below_camera_m")
    centre_height = (-float(drop) if drop is not None
                     else unproject((x0 + x1) * 0.5, (y0 + y1) * 0.5, median,
                                    fov_x, aspect)[2])
    centre = ((near + far) * 0.5, (left[1] + right[1]) * 0.5, centre_height)
    return centre, lateral, far - near


def _footprint(region: dict[str, Any], fov_x: float, aspect: float
               ) -> tuple[tuple[float, float, float], float, float]:
    x0, y0, x1, y1 = region["bbox_norm_xyxy"]
    depth = float(region.get("depth_m", {}).get("median") or 10.0)

    centre = unproject((x0 + x1) * 0.5, (y0 + y1) * 0.5, depth, fov_x, aspect)
    left = unproject(x0, (y0 + y1) * 0.5, depth, fov_x, aspect)
    right = unproject(x1, (y0 + y1) * 0.5, depth, fov_x, aspect)
    top = unproject((x0 + x1) * 0.5, y0, depth, fov_x, aspect)
    bottom = unproject((x0 + x1) * 0.5, y1, depth, fov_x, aspect)

    width = abs(right[1] - left[1])
    height = abs(top[2] - bottom[2])
    return centre, width, height


def scatter_offsets(mask_path: Path | None, bbox: list[float], count: int
                    ) -> list[float] | None:
    """Where along a scatter region's width its instances should actually stand.

    Spreading instances evenly across the bounding box puts them wherever the
    box reaches, including columns the region does not occupy. On this barn
    scene the tree line's box spans the barn, so an even spread planted a tree
    inside the building -- the trees are *behind* it, and the pixels say so.

    Sampling the mask's column distribution at even quantiles instead places
    instances only where the region was actually observed, and concentrates
    them where it is densest. Returns offsets in 0..1 across the bbox width, or
    None when there is no mask to consult.
    """
    if mask_path is None or not Path(mask_path).is_file():
        return None
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None

    mask = np.asarray(Image.open(mask_path).convert("L")) >= 128
    height, width = mask.shape
    x0 = int(round(bbox[0] * width))
    x1 = max(x0 + 1, int(round(bbox[2] * width)))
    columns = mask[:, x0:x1].sum(axis=0).astype(float)
    total = columns.sum()
    if total <= 0:
        return None

    # Even quantiles of the column mass: dense parts of the line get more
    # instances, and empty columns get none.
    cumulative = np.cumsum(columns) / total
    targets = (np.arange(count) + 0.5) / count
    indices = np.searchsorted(cumulative, targets)
    span = max(1, x1 - x0 - 1)
    return [float(min(index, span)) / span for index in indices]


def _intersects_any(actor: dict[str, Any], others: list[dict[str, Any]]) -> bool:
    """Do this actor's world bounds meet any of the others'?"""
    def bounds(spec):
        centre = [c / CM_PER_M for c in spec["location_cm"]]
        half = [s * 0.5 for s in spec["size_m"]]
        return [(centre[i] - half[i], centre[i] + half[i]) for i in range(3)]

    mine = bounds(actor)
    for other in others:
        theirs = bounds(other)
        if all(mine[axis][0] < theirs[axis][1] and theirs[axis][0] < mine[axis][1]
               for axis in range(3)):
            return True
    return False


def ground_height_at(plane: dict[str, Any] | None, forward: float, right: float
                     ) -> float | None:
    """Height of the fitted ground plane under a point, or None if unfitted."""
    if not plane:
        return None
    return (float(plane["slope_forward"]) * forward
            + float(plane["slope_right"]) * right
            + float(plane["height_at_camera_m"]))


def place(segmentation: dict[str, Any], fov_x_deg: float | None = None,
          mask_dir: Path | str | None = None) -> dict[str, Any]:
    aspect = segmentation["image_dimensions"][0] / segmentation["image_dimensions"][1]
    fov_x = float(fov_x_deg or segmentation.get("camera", {}).get("fov_x_deg") or 90.0)
    ground_plane = segmentation.get("ground_plane_unreal")

    actors: list[dict[str, Any]] = []
    skipped: list[str] = []
    excluded: list[dict[str, Any]] = []

    for region in segmentation["regions"]:
        layer = region["layer_type"]
        if layer in NON_PLACED:
            skipped.append(region["id"])
            excluded.append({"id": region["id"], "reason": "describes appearance, "
                                                           "not occupancy"})
            continue

        # A label the model was unsure of is not an object. ADE20K's "hovel"
        # landed on a dark gap between two tree trunks here, and it was built as
        # a second building in a picture containing one barn -- and generated a
        # mesh for it, at ten GPU-minutes a run. The old confidence could not
        # have caught that: it was `0.5 + pixel_fraction`, a function of size.
        #
        # Objects only. For a surface, low confidence means the model could not
        # decide between "field" and "grass", and either way the ground is
        # there; dropping it took the floor out from under the scene.
        confidence = region.get("confidence")
        if (layer in CONFIDENCE_GATED_LAYERS and confidence is not None
                and confidence < MIN_PLACEMENT_CONFIDENCE):
            skipped.append(region["id"])
            excluded.append({
                "id": region["id"], "semantic_label": region.get("semantic_label"),
                "confidence": confidence,
                "reason": f"model confidence {confidence:.3f} is below "
                          f"{MIN_PLACEMENT_CONFIDENCE}"})
            continue

        measured = region.get("measured_unreal_m")
        if measured:
            # Where the region's points actually are, already in this frame.
            # Preferred over unprojecting a bounding box at one depth, which
            # answers a different question for anything with depth extent and
            # disagrees with the measurement by metres.
            centre = tuple(float(v) for v in measured["centroid"])
            thickness = max(0.5, float(measured["size"][0]))
            width = max(float(measured["size"][1]), 0.5)
            height = max(float(measured["size"][2]), 0.5)
        else:
            centre, width, height = _footprint(region, fov_x, aspect)
            depth_band = region.get("depth_m", {})
            thickness = max(0.5, float(depth_band.get("far", 0.0))
                            - float(depth_band.get("near", 0.0)))

        bbox = [float(v) for v in region["bbox_norm_xyxy"]]
        common = {
            "region_id": region["id"],
            "semantic_label": region["semantic_label"],
            "layer_type": layer,
            "confidence": region.get("confidence"),
            "observed_fraction": region.get("observed_fraction"),
            # Carried through so a generator can crop the source image back to
            # the pixels this actor was measured from. Without it the actor
            # knows its size and position but not what it looks like.
            # `source_bbox` is this actor's own window and `region_bbox` the
            # whole region's; they differ only for scattered instances, and a
            # generator needs the second to know how far it may widen a crop.
            "source_bbox_norm_xyxy": bbox,
            "region_bbox_norm_xyxy": bbox,
            # So a region that keeps its primitive is at least the right colour.
            "mean_colour_srgb": region.get("mean_colour_srgb"),
        }

        if layer in ("terrain", "water"):
            if measured:
                # A surface is flat: keep its measured footprint and drop the
                # vertical spread, which for ground is slope and noise rather
                # than thickness.
                surface, lateral, depth_span = centre, width, thickness
                sized_at = "measured_points"
            else:
                surface, lateral, depth_span = _receding_surface(
                    region, fov_x, aspect)
                sized_at = "far_depth_edge"
            actors.append({
                **common,
                "kind": "ground_plane" if layer == "terrain" else "water_surface",
                "location_cm": [c * CM_PER_M for c in surface],
                "size_m": [max(depth_span, 1.0), max(lateral, 1.0),
                           0.2 if layer == "terrain" else 0.05],
                "sized_at": sized_at,
            })

        elif layer == "architecture":
            actors.append({**common, "kind": "structure",
                           "location_cm": [c * CM_PER_M for c in centre],
                           # Depth is unobservable from one view; assume a
                           # footprint roughly as deep as it is wide.
                           "size_m": [max(width * 0.6, 1.0), max(width, 1.0),
                                      max(height, 1.0)],
                           "assumption": "building depth inferred from width"})

        elif layer in SCATTER_TARGET and region.get("clusters"):
            # Segmentation split this region into spatially coherent clumps.
            # Place one instance per clump, each at its *own* depth and size: a
            # semantic class is not an object, and this region's pixels span
            # 2.4 m to 21 m with a median identical to the building in front of
            # it, which is how a barn ended up inside a tree.
            for index, cluster in enumerate(region["clusters"]):
                cluster_bbox = [float(v) for v in cluster["bbox_norm_xyxy"]]
                cluster_depth = cluster["depth_m"]
                cluster_measured = cluster.get("measured_unreal_m")
                if cluster_measured:
                    centre_c = tuple(float(v) for v in cluster_measured["centroid"])
                    thickness_c = max(0.5, float(cluster_measured["size"][0]))
                    width_c = max(float(cluster_measured["size"][1]), 0.5)
                    height_c = max(float(cluster_measured["size"][2]), 0.5)
                else:
                    cluster_region = {**region, "bbox_norm_xyxy": cluster_bbox,
                                      "depth_m": cluster_depth}
                    centre_c, width_c, height_c = _footprint(
                        cluster_region, fov_x, aspect)
                    thickness_c = max(0.5, float(cluster_depth.get("far", 0.0))
                                      - float(cluster_depth.get("near", 0.0)))
                # The clump keeps its own measured extent. Stretching it down to
                # the ground instead looks like grounding and is not: the mesh
                # is a foliage blob, roughly as deep as it is tall, so scaling
                # it uniformly to a ground-to-canopy height makes it that wide
                # too -- overlapping pairs went 1 to 31 when it was tried.
                base_z = centre_c[2] - height_c * 0.5
                actors.append({
                    **common,
                    "kind": "scatter_instance",
                    "instance_index": index,
                    "source_bbox_norm_xyxy": cluster_bbox,
                    "cluster_pixel_count": cluster["pixel_count"],
                    "cluster_depth_m": cluster_depth.get("median"),
                    # Whether this is an object or a slice of a larger mass.
                    # Segmentation knows; without carrying it here, generation
                    # has to rediscover it from the crop's borders and the
                    # overlap audit from pairwise volumes.
                    "separable": bool(cluster.get("separable", True)),
                    "component_id": cluster.get("component_id"),
                    "location_cm": [centre_c[0] * CM_PER_M,
                                    centre_c[1] * CM_PER_M,
                                    base_z * CM_PER_M],
                    "size_m": [max(min(width_c, thickness_c), 0.5),
                               max(width_c, 0.5), max(height_c, 0.5)],
                })

                # What holds it up. Clustering finds canopy because canopy is
                # what the camera saw -- trunks are thin, dark and mostly
                # occluded -- so the clump alone hangs in the air. A support
                # from the fitted ground plane to the canopy's base states the
                # unobserved part explicitly, as a primitive, instead of
                # deforming the observed part to hide the gap.
                ground_z = ground_height_at(ground_plane, centre_c[0], centre_c[1])
                if ground_z is not None and base_z - ground_z > MIN_TRUNK_HEIGHT:
                    girth = max(min(width_c, thickness_c) * TRUNK_GIRTH_FRACTION,
                                MIN_TRUNK_GIRTH)
                    actors.append({
                        **common,
                        "kind": "trunk_support",
                        "instance_index": index,
                        # Centred: a primitive's origin is its middle, so a
                        # trunk placed at ground level would be half buried.
                        "location_cm": [centre_c[0] * CM_PER_M,
                                        centre_c[1] * CM_PER_M,
                                        (ground_z + base_z) * 0.5 * CM_PER_M],
                        "size_m": [girth, girth, base_z - ground_z],
                        "inferred": "not observed; spans the fitted ground plane "
                                    "to the canopy this instance was measured at",
                    })

        elif layer in SCATTER_TARGET:
            count = SCATTER_TARGET[layer]
            observed = scatter_offsets(
                Path(mask_dir) / f"{region['id']}.png" if mask_dir else None,
                bbox, count)
            for index in range(count):
                # Deterministic spread across the footprint; no RNG so a rerun
                # reproduces the same scene. Where the region's mask is
                # available the spread follows the pixels it actually occupies
                # instead of its bounding box.
                t = observed[index] if observed else (index + 0.5) / count
                offset = (t - 0.5) * width
                jitter = ((index * 37) % 11 - 5) / 10.0
                # A scatter region's bbox covers the whole tree line, so the
                # region crop is a hedge rather than a tree. Give each instance
                # the slice of the image it actually stands in, which is a crop
                # of one subject and the right input for a generator.
                slice_width = (bbox[2] - bbox[0]) / count
                slice_centre = bbox[0] + t * (bbox[2] - bbox[0])
                instance_bbox = [
                    max(bbox[0], slice_centre - slice_width * 0.5), bbox[1],
                    min(bbox[2], slice_centre + slice_width * 0.5), bbox[3]]
                actors.append({
                    **common,
                    "kind": "scatter_instance",
                    "instance_index": index,
                    "source_bbox_norm_xyxy": instance_bbox,
                    "location_cm": [(centre[0] + jitter * thickness * 0.3) * CM_PER_M,
                                    (centre[1] + offset) * CM_PER_M,
                                    (centre[2] - height * 0.5) * CM_PER_M],
                    "size_m": [max(width / count, 0.5)] * 2 + [max(height, 1.0)],
                })

        elif layer == "crossing":
            actors.append({**common, "kind": "path_strip",
                           "location_cm": [c * CM_PER_M for c in centre],
                           "size_m": [thickness, max(width, 1.0), 0.1]})

        else:
            actors.append({**common, "kind": "clutter_volume",
                           "location_cm": [c * CM_PER_M for c in centre],
                           "size_m": [max(width * 0.5, 0.5), max(width, 0.5),
                                      max(height, 0.5)]})

    # An inferred trunk that would pass through an observed building is an
    # inference the evidence does not support: the canopy centroid is not the
    # trunk's position, crowns overhang, and a tree behind a barn has its trunk
    # behind the barn. Where the guess collides with something that *was*
    # observed, drop it rather than intersect it.
    solids = [a for a in actors if a["kind"] in ("structure", "clutter_volume")]
    kept, withdrawn = [], []
    for actor in actors:
        if actor["kind"] == "trunk_support" and _intersects_any(actor, solids):
            withdrawn.append(actor["region_id"])
            continue
        kept.append(actor)
    actors = kept

    return {
        "schema_version": "region_placement_v1",
        "classification": "PROVEN" if actors else "EMPTY",
        "withdrawn_trunk_supports": len(withdrawn),
        "camera_fov_x_deg": fov_x,
        "aspect_ratio": aspect,
        "actor_count": len(actors),
        "kinds": sorted({a["kind"] for a in actors}),
        "skipped_regions": skipped,
        # Why each one was skipped, so a missing object is explicable rather
        # than simply absent.
        "excluded_regions": excluded,
        "actors": actors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--fov-x", type=float, default=None)
    args = parser.parse_args(argv)

    segmentation = json.loads(Path(args.segmentation).read_text(encoding="utf-8"))
    result = place(segmentation, args.fov_x)

    receipt = Path(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")

    summary = {k: v for k, v in result.items() if k != "actors"}
    summary["actors"] = [
        {"id": a["region_id"], "kind": a["kind"],
         "at_m": [round(c / CM_PER_M, 1) for c in a["location_cm"]],
         "size_m": [round(s, 1) for s in a["size_m"]]}
        for a in result["actors"]
    ]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
