"""Real semantic regions for the structural pipeline.

This replaces the stub that collapsed every image to one `visual_shell` region.
That stub is the reason the structural builders emit scaled cubes: with no
semantic classes to plan against, asset strategy has nothing to choose between,
so every region falls back to a primitive. No amount of builder work fixes that
downstream.

Segmentation alone is not enough, because the builders need to know *where in
the world* each region sits, not just which pixels it covers. So the ADE20K
labels are fused with the MoGe point map: each region carries real depth
extent, a ground-plane footprint, and a surface orientation. That is what lets
a planner decide a region is walkable ground rather than a vertical facade.

ADE20K's 150 classes map onto the pipeline's layers directly -- sky, tree,
grass, building, water, mountain, fence -- which is why it is used here rather
than unsupervised clustering.

    .../envs/image-world-moge/Scripts/python.exe -m lowvram3d.scene_segmentation \\
        --image in.png --receipt regions.json --overlay regions.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SEG_MODEL = "nvidia/segformer-b4-finetuned-ade-512-512"

# ADE20K label -> the structural pipeline's layer vocabulary. Anything absent
# is carried through as scene clutter rather than being silently dropped, so an
# unmapped class shows up in the receipt instead of disappearing.
LAYER_BY_LABEL = {
    "sky": "sky",
    "tree": "vegetation", "plant": "vegetation", "flower": "vegetation",
    "palm": "vegetation", "bush": "vegetation",
    "grass": "terrain", "earth": "terrain", "field": "terrain", "dirt track": "terrain",
    "sand": "terrain", "hill": "terrain", "land": "terrain", "path": "terrain",
    "mountain": "terrain", "rock": "terrain",
    "water": "water", "sea": "water", "river": "water", "lake": "water",
    "waterfall": "water", "pool": "water",
    "building": "architecture", "house": "architecture", "hovel": "architecture",
    "skyscraper": "architecture", "hut": "architecture", "tower": "architecture",
    "wall": "architecture", "roof": "architecture", "door": "architecture",
    "windowpane": "architecture", "stairs": "architecture", "column": "architecture",
    "fence": "prop", "railing": "prop", "pole": "prop", "signboard": "prop",
    "bridge": "crossing", "road": "crossing", "sidewalk": "crossing",
}

# How each layer should be represented downstream. Sky is a backdrop rather
# than geometry; terrain is walkable; vegetation is scattered instances.
REPRESENTATION_BY_LAYER = {
    "sky": "backdrop",
    "terrain": "heightfield",
    "water": "water_surface",
    "architecture": "structure_mesh",
    "vegetation": "scatter_instances",
    "crossing": "spline_surface",
    "prop": "instanced_prop",
    "clutter": "visual_shell",
}

MIN_PIXEL_FRACTION = 0.002


def _labels(model) -> dict[int, str]:
    return {int(k): str(v).split(",")[0].strip()
            for k, v in model.config.id2label.items()}


# Semantic classes are not objects. One "tree" region here is a 28 m hedge line
# whose pixels run from 2.4 m to 21 m deep, so a single placement for it is
# wrong at both ends -- and its median depth (13.46 m) is identical to the
# barn's, which is how a building ended up entirely inside a tree. Classes that
# get scattered are split into spatially coherent clumps first, using the point
# map already computed here. This is instance separation by geometry: cheaper
# than a panoptic model and grounded in measurements this stage already has.
CLUSTERED_LAYERS = {"vegetation", "prop", "clutter"}
MAX_CLUSTERS = 12
MIN_CLUSTER_POINTS = 64
CLUSTER_ITERATIONS = 12


def fit_ground_plane(samples):
    """Least-squares z = a*x + b*y + c over every horizontal surface's points.

    A flat height would not do: MoGe's ground here drops 1.3 m below the camera
    at 2.3 m out and 4.9 m at 13.5 m, so the scene slopes away and a constant
    would be wrong at one end or the other by metres.

    This exists so things that stand on the ground can be *put* on it. Clustering
    a tree line by geometry finds canopy, because canopy is what the camera saw;
    the trunks are dark, thin and mostly occluded. Placed at their own measured
    extent, those clumps hang in mid-air, which is an artefact of what was
    observed rather than a fact about the scene.
    """
    import numpy as np

    if len(samples) < 32:
        return None
    points = np.asarray(samples, dtype=np.float64)
    design = np.column_stack([points[:, 0], points[:, 1],
                              np.ones(len(points))])
    coefficients, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    residual = points[:, 2] - design @ coefficients
    return {
        "slope_forward": round(float(coefficients[0]), 5),
        "slope_right": round(float(coefficients[1]), 5),
        "height_at_camera_m": round(float(coefficients[2]), 3),
        "residual_p95_m": round(float(np.percentile(np.abs(residual), 95)), 3),
        "sample_count": int(len(points)),
    }


def _measured_extent(selected):
    """Where a set of MoGe points actually sits, in Unreal's frame and metres.

    MoGe is X right, Y down, Z forward; Unreal is X forward, Y right, Z up.
    """
    import numpy as np

    forward, right, up = selected[:, 2], selected[:, 0], -selected[:, 1]
    low = [float(np.percentile(axis, 5)) for axis in (forward, right, up)]
    high = [float(np.percentile(axis, 95)) for axis in (forward, right, up)]
    centre = [float(np.median(axis)) for axis in (forward, right, up)]
    return {
        "centroid": [round(v, 3) for v in centre],
        "min": [round(v, 3) for v in low],
        "max": [round(v, 3) for v in high],
        "size": [round(high[axis] - low[axis], 3) for axis in range(3)],
    }


def connected_components(observed, min_pixels=MIN_CLUSTER_POINTS):
    """Split a region mask into its disconnected pieces, largest first.

    This is the one instance signal available without an instance model: two
    trees with sky between them are two components, and no amount of clustering
    3D points is needed to know it. Two trees whose canopies touch are one
    component, and nothing here can tell otherwise -- which is the honest
    boundary of what a semantic mask can support, and why the design notes rank
    a real instance model first.
    """
    import numpy as np
    from scipy import ndimage

    labelled, count = ndimage.label(observed)
    pieces = []
    for index in range(1, count + 1):
        piece = labelled == index
        if piece.sum() >= min_pixels:
            pieces.append(piece)
    pieces.sort(key=lambda p: -p.sum())
    return pieces


def _cluster_one_mass(points, selection, observed, width, height,
                      max_clusters=MAX_CLUSTERS):
    """Split one connected mass into spatially coherent clumps.

    Returns image-space descriptions -- normalised centroid, depth, pixel bbox
    -- rather than world points, so placement keeps a single unprojection and a
    single convention. Deterministic: seeded from evenly spaced quantiles of
    the dominant axis, no RNG, so a rerun reproduces the same scene.
    """
    import numpy as np

    ys, xs = np.nonzero(observed)
    if len(xs) < MIN_CLUSTER_POINTS:
        return []
    coords = points[observed]
    finite = np.isfinite(coords).all(axis=1)
    coords, xs, ys = coords[finite], xs[finite], ys[finite]
    if len(coords) < MIN_CLUSTER_POINTS:
        return []

    count = int(min(max_clusters, max(1, len(coords) // MIN_CLUSTER_POINTS)))
    if count == 1:
        labels = np.zeros(len(coords), dtype=int)
        centres = coords.mean(axis=0, keepdims=True)
    else:
        # Scale each axis to unit variance so depth, which spans an order of
        # magnitude more than the lateral extent, does not dominate the metric.
        spread = coords.std(axis=0)
        spread[spread < 1e-6] = 1.0
        scaled = coords / spread
        dominant = int(np.argmax(coords.std(axis=0)))
        order = np.argsort(coords[:, dominant])
        seeds = order[((np.arange(count) + 0.5) / count * len(order)).astype(int)]
        centres = scaled[seeds].copy()
        labels = np.zeros(len(coords), dtype=int)
        for _ in range(CLUSTER_ITERATIONS):
            distances = ((scaled[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for index in range(count):
                member = labels == index
                if member.any():
                    centres[index] = scaled[member].mean(axis=0)
        centres = centres * spread

    clusters = []
    for index in range(len(centres)):
        member = labels == index
        if member.sum() < MIN_CLUSTER_POINTS:
            continue
        member_x, member_y = xs[member], ys[member]
        depths = coords[member][:, 2]
        clusters.append({
            "pixel_count": int(member.sum()),
            "measured_unreal_m": _measured_extent(coords[member]),
            "centroid_norm_xy": [round(float(member_x.mean()) / width, 5),
                                 round(float(member_y.mean()) / height, 5)],
            "bbox_norm_xyxy": [round(float(member_x.min()) / width, 5),
                               round(float(member_y.min()) / height, 5),
                               round(float(member_x.max() + 1) / width, 5),
                               round(float(member_y.max() + 1) / height, 5)],
            "depth_m": {
                "near": round(float(np.percentile(depths, 5)), 3),
                "median": round(float(np.median(depths)), 3),
                "far": round(float(np.percentile(depths, 95)), 3),
            },
        })
    clusters.sort(key=lambda c: -c["pixel_count"])
    return clusters


def _slices_for_shape(points, piece):
    """How many objects a connected mass plausibly contains, from its shape.

    The count used to come from pixel count alone, which meant any region with
    enough pixels was diced -- a single clean tree became six clumps because it
    was photographed close up. Pixel count measures how much of the image a
    thing occupies, not how many things it is.

    Shape does carry a signal: a tree is roughly as wide as it is tall, and a
    hedge line is several times wider. Slicing by that ratio at least ties the
    number of pieces to a measurement instead of to resolution. It is still a
    decomposition rather than instance detection, which is why everything it
    produces is marked `separable: false`.
    """
    import numpy as np

    coords = points[piece]
    coords = coords[np.isfinite(coords).all(axis=1)]
    if len(coords) < MIN_CLUSTER_POINTS:
        return 1
    extent = _measured_extent(coords)["size"]
    lateral = max(extent[0], extent[1])
    upright = max(extent[2], 1e-3)
    return max(1, int(round(lateral / upright)))


def cluster_region_points(points, selection, observed, width, height,
                          max_clusters=MAX_CLUSTERS):
    """Split a region into instances, saying which ones are actually instances.

    Connected components first, because that is measured: a tree with sky either
    side of it is its own object, and the pipeline can say so without a model.
    Only a component too large to be one thing is subdivided further, and those
    slices are marked `separable: false` -- they are a decomposition of one mass
    for placement, not objects in their own right.

    Downstream this is what lets the generator refuse a crop with no subject and
    the overlap audit stop counting a hedge against itself, both of which were
    previously rediscovering the same fact from their own side.
    """
    pieces = connected_components(observed)
    if not pieces:
        return _cluster_one_mass(points, selection, observed, width, height,
                                 max_clusters)

    total = sum(int(piece.sum()) for piece in pieces)
    clusters = []
    for component_id, piece in enumerate(pieces):
        pixels = int(piece.sum())
        # Share the cluster budget by size, so one large mass does not consume
        # every slot and leave real separate objects unrepresented.
        budget = max(1, int(round(max_clusters * pixels / max(total, 1))))
        budget = min(budget, _slices_for_shape(points, piece))
        found = _cluster_one_mass(points, selection, piece, width, height, budget)
        for cluster in found:
            cluster["component_id"] = component_id
            cluster["component_pixel_count"] = pixels
            cluster["separable"] = bool(len(found) == 1)
        clusters.extend(found)
    clusters.sort(key=lambda c: -c["pixel_count"])
    return clusters[:max_clusters]


def segment(image_path: Path, model_name: str = SEG_MODEL,
            with_geometry: bool = True,
            mask_dir: Path | None = None) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device).eval()

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        # Logits come back at a reduced stride; resample to full resolution so
        # region masks line up with the point map pixel for pixel.
        upsampled = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False)
        labels = upsampled.argmax(dim=1)[0].cpu().numpy()
        # What the model actually believed, per pixel. Softmax on the *logits*
        # rather than the upsampled tensor: the latter is 150 channels at full
        # resolution, roughly a gigabyte, on a card this pipeline is already
        # sharing.
        certainty = torch.nn.functional.interpolate(
            torch.softmax(logits, dim=1).amax(dim=1, keepdim=True),
            size=(height, width), mode="bilinear", align_corners=False)
        certainty = certainty[0, 0].cpu().numpy()

    names = _labels(model)

    points = mask = normals = None
    if with_geometry:
        from moge.model.v2 import MoGeModel

        moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitb-normal").to(device).eval()
        array = np.asarray(image, dtype=np.uint8)
        tensor = torch.tensor(array / 255.0, dtype=torch.float32,
                              device=device).permute(2, 0, 1)
        with torch.no_grad():
            prediction = moge.infer(tensor)
        points = prediction["points"].cpu().numpy().astype(np.float64)
        mask = prediction["mask"].cpu().numpy().astype(bool)
        if "normal" in prediction:
            normals = prediction["normal"].cpu().numpy().astype(np.float64)

    total = float(width * height)
    regions: list[dict[str, Any]] = []
    unmapped: set[str] = set()
    ground_samples: list[list[float]] = []

    for class_id in np.unique(labels):
        selection = labels == class_id
        fraction = float(selection.sum()) / total
        if fraction < MIN_PIXEL_FRACTION:
            continue

        label = names.get(int(class_id), f"class_{int(class_id)}")
        layer = LAYER_BY_LABEL.get(label)
        if layer is None:
            layer = "clutter"
            unmapped.add(label)

        ys, xs = np.nonzero(selection)
        region: dict[str, Any] = {
            "id": f"{layer}_{label.replace(' ', '_')}_{int(class_id):03d}",
            "semantic_label": label,
            "layer_type": layer,
            "representation": REPRESENTATION_BY_LAYER.get(layer, "visual_shell"),
            "pixel_fraction": round(fraction, 5),
            "bbox_norm_xyxy": [round(float(xs.min()) / width, 4),
                               round(float(ys.min()) / height, 4),
                               round(float(xs.max() + 1) / width, 4),
                               round(float(ys.max() + 1) / height, 4)],
            # The model's own mean probability over this region's pixels.
            # This used to be `0.5 + pixel_fraction`, which is a function of
            # size and nothing else -- so a dark gap between two tree trunks,
            # labelled "hovel", scored 0.52 and was built as a second building
            # in a picture containing one barn.
            "confidence": round(float(certainty[selection].mean()), 4),
            "confidence_p10": round(float(np.percentile(certainty[selection], 10)), 4),
            # Regions that keep a primitive render default white, which makes
            # ground, water and every unbuilt volume read as the same blank
            # slab and hides whatever is standing on them. One mean colour per
            # region is nearly free here, where the image already is.
            "mean_colour_srgb": [int(c) for c in
                                 np.asarray(image, dtype=np.uint8)[selection]
                                 .reshape(-1, 3).mean(axis=0).round()],
        }

        # Keep the mask, not just its bounding box. A downstream generator needs
        # the subject cut out of its background, and re-deriving that with a
        # generic saliency matte is both slower and worse than the segmentation
        # already computed here: on a dark, low-contrast source rembg erased
        # half a barn that these labels had cleanly separated.
        if mask_dir is not None:
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{region['id']}.png"
            Image.fromarray((selection * 255).astype(np.uint8), mode="L").save(mask_path)
            region["mask_png"] = str(mask_path)

        if points is not None:
            observed = selection & mask & np.isfinite(points).all(axis=-1)
            if observed.sum() > 32:
                selected = points[observed]
                depth = selected[:, 2]
                region["depth_m"] = {
                    "near": round(float(np.percentile(depth, 5)), 3),
                    "far": round(float(np.percentile(depth, 95)), 3),
                    "median": round(float(np.median(depth)), 3),
                }
                extent = selected.max(axis=0) - selected.min(axis=0)
                region["world_extent_m"] = [round(float(v), 3) for v in extent]
                region["observed_fraction"] = round(
                    float(observed.sum()) / max(1.0, float(selection.sum())), 4)

                if normals is not None:
                    mean_normal = normals[observed].mean(axis=0)
                    norm = float(np.linalg.norm(mean_normal))
                    if norm > 1e-6:
                        mean_normal = mean_normal / norm
                        # MoGe normals are OpenCV-style with +Y down, so a
                        # ground plane faces -Y. Verticality separates walkable
                        # surfaces from facades.
                        verticality = float(abs(mean_normal[1]))
                        region["surface"] = {
                            "mean_normal": [round(float(v), 4) for v in mean_normal],
                            "orientation": "horizontal" if verticality > 0.6 else "vertical",
                            "walkable_candidate": bool(
                                verticality > 0.6 and layer in ("terrain", "crossing")),
                        }
                # A surface's height is a measurement, not something to infer
                # from a bounding box. Unprojecting a ground region's bbox
                # centre at its median depth answers "how far below the camera
                # is the middle of this box", which for a plane spanning 1.6 m
                # to 10.3 m is neither its near height nor its far one -- it put
                # the ground 3.7 m above the barn's base and sliced the building
                # in half. MoGe's Y is up, so the median is the answer directly.
                # The region's actual occupancy, converted once, here, into the
                # frame the builders use: Unreal's X forward, Y right, Z up,
                # from MoGe's X right, Y down, Z forward.
                #
                # Placement used to unproject a bounding box at a single depth
                # instead. That answers a different question for anything with
                # depth extent, and mixing the two methods is what put the
                # ground plane 3.6 m above the barn's base -- the ground was
                # measured and the barn was unprojected, and MoGe's ground is
                # not flat (1.3 m below the camera at 2.3 m out, 4.9 m below at
                # 13.5 m). Percentiles rather than min/max, so one stray point
                # does not size the actor.
                region["measured_unreal_m"] = _measured_extent(selected)

                if (region.get("surface") or {}).get("orientation") == "horizontal":
                    # Subsample: the fit needs coverage, not every pixel.
                    ground_samples.extend(np.column_stack([
                        selected[::37, 2], selected[::37, 0],
                        -selected[::37, 1]]).tolist())
                    # MoGe's Y is down, so a positive median is a surface below
                    # the camera. Named for what it measures rather than "height",
                    # because the sign is the whole question and a consumer in a
                    # Z-up frame has to negate it.
                    region["surface"]["drop_below_camera_m"] = round(
                        float(np.median(selected[:, 1])), 3)
                    region["surface"]["drop_spread_m"] = round(
                        float(np.percentile(selected[:, 1], 95)
                              - np.percentile(selected[:, 1], 5)), 3)

                if layer in CLUSTERED_LAYERS:
                    region["clusters"] = cluster_region_points(
                        points, selection, observed, width, height)
            else:
                region["observed_fraction"] = 0.0
                region["uncertainty"] = "region is masked out of the depth estimate"

        regions.append(region)

    regions.sort(key=lambda r: -r["pixel_fraction"])
    layers = sorted({r["layer_type"] for r in regions})

    return {
        "schema_version": "scene_segmentation_v1",
        "classification": "PROVEN" if len(regions) > 1 else "DEGENERATE",
        "model": model_name,
        "device": device,
        "image": str(image_path),
        "image_dimensions": [width, height],
        "region_count": len(regions),
        "layers_present": layers,
        "unmapped_labels": sorted(unmapped),
        # A sloped plane through every horizontal surface, so anything that
        # stands on the ground can be put on it.
        "ground_plane_unreal": fit_ground_plane(ground_samples),
        "regions": regions,
    }


def write_overlay(image_path: Path, result: dict[str, Any], out: Path) -> None:
    """Draw region boxes and labels so the segmentation can be eyeballed."""
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    palette = [(255, 90, 90), (90, 200, 255), (140, 255, 120), (255, 210, 90),
               (210, 130, 255), (255, 150, 200), (150, 255, 220), (255, 255, 255)]

    for index, region in enumerate(result["regions"]):
        x0, y0, x1, y1 = region["bbox_norm_xyxy"]
        box = (x0 * width, y0 * height, x1 * width, y1 * height)
        colour = palette[index % len(palette)]
        draw.rectangle(box, outline=colour, width=3)
        caption = f"{region['semantic_label']} [{region['layer_type']}]"
        if "depth_m" in region:
            caption += f" {region['depth_m']['median']}m"
        draw.text((box[0] + 6, box[1] + 6), caption, fill=colour)

    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--overlay", default=None)
    parser.add_argument("--model", default=SEG_MODEL)
    parser.add_argument("--no-geometry", action="store_true")
    parser.add_argument("--mask-dir", default=None,
                        help="write a per-region mask PNG here")
    args = parser.parse_args(argv)

    image = Path(args.image)
    result = segment(image, args.model, not args.no_geometry,
                     Path(args.mask_dir) if args.mask_dir else None)

    receipt = Path(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    if args.overlay:
        write_overlay(image, result, Path(args.overlay))

    summary = {k: v for k, v in result.items() if k != "regions"}
    summary["regions"] = [
        {"id": r["id"], "layer": r["layer_type"], "pct": round(r["pixel_fraction"] * 100, 1),
         "depth_median_m": r.get("depth_m", {}).get("median")}
        for r in result["regions"]
    ]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
