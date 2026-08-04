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


def segment(image_path: Path, model_name: str = SEG_MODEL,
            with_geometry: bool = True) -> dict[str, Any]:
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
            "confidence": round(min(0.95, 0.5 + fraction), 3),
        }

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
    args = parser.parse_args(argv)

    image = Path(args.image)
    result = segment(image, args.model, not args.no_geometry)

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
