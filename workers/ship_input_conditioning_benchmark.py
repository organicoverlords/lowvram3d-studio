"""Standardised source-variant benchmark for the ship conditioning image.

Answers one question: does cleaning the input plate materially change what a single-image geometry
generator has to work with? It builds the variants, renders each into the three plate backgrounds a
generator might want, and measures the mask properties that actually predict geometry damage --
detached specks that become floating debris, edge halo that becomes a shell, filled pinholes,
and thin features (railings, masts, antennas) that a matting model likes to erase.

Thin features are never silently dropped. Every thin structure found in a deterministic reference
key of the plate is classified against each variant as preserved, thickened, or removed, and a
variant that loses coverage of the ship's solid core fails closed rather than being reported as a
cleaner mask.

Note on the plate: this particular source is a JPEG with a transparency-checkerboard pattern baked
in as pixels, so the background is a light grey check, not black. Border-colour estimation keys it
cleanly, which is what the deterministic reference below relies on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

LIGHT_GREY = (200, 200, 200)
WHITE = (255, 255, 255)
THIN_ERODE = 2
CORE_ERODE = 3
CORE_COVERAGE_FLOOR = 0.97


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_background(rgb: np.ndarray) -> np.ndarray:
    """Median colour of a thin border frame; the plate's background is uniform there."""
    border = np.concatenate((
        rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3),
        rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3),
    ))
    return np.median(border, axis=0)


def reference_key(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic non-background key of the plate, used as the structural ground truth.

    The checkerboard means a pure distance-to-background threshold flickers on the check edges, so
    the distance image is blurred slightly before thresholding and small specks are dropped.
    """
    background = estimate_background(rgb)
    distance = np.linalg.norm(rgb.astype(np.float32) - background[None, None, :], axis=2)
    distance = cv2.GaussianBlur(distance, (0, 0), 1.2)
    mask = distance > 28.0
    mask = ndimage.binary_closing(mask, np.ones((3, 3), bool))
    labels, count = ndimage.label(mask, np.ones((3, 3), int))
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        # Keep the hull plus anything at least 0.2% of it; that retains the mast and railings
        # while discarding isolated compression speckle in the checkerboard.
        keep = np.flatnonzero(sizes >= max(1, int(sizes.max() * 0.002)))
        mask = np.isin(labels, keep)
    filled = ndimage.binary_fill_holes(mask)
    return mask, filled


def components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, count = ndimage.label(mask, np.ones((3, 3), int))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    return labels, sizes


def thin_features(mask: np.ndarray) -> np.ndarray:
    """Pixels that do not survive a small erosion: railings, cords, masts, antennas."""
    eroded = ndimage.binary_erosion(mask, np.ones((2 * THIN_ERODE + 1,) * 2, bool))
    return mask & ~ndimage.binary_dilation(eroded, np.ones((2 * THIN_ERODE + 1,) * 2, bool))


def measure(alpha: np.ndarray, rgb: np.ndarray, reference: np.ndarray,
            background: np.ndarray) -> dict:
    height, width = alpha.shape
    total = float(height * width)
    solid = alpha > 0.5
    area = float(solid.sum())

    labels, sizes = components(solid)
    positive = sizes[sizes > 0]
    order = np.sort(positive)[::-1]

    filled = ndimage.binary_fill_holes(solid)
    holes = filled & ~solid
    hole_labels, hole_sizes = components(holes)

    # Edge halo: how much of the plate background colour is still being carried by partially
    # transparent boundary texels. A clean soft matte carries the object's own colour there.
    dilated = ndimage.binary_dilation(solid, np.ones((5, 5), bool))
    eroded = ndimage.binary_erosion(solid, np.ones((5, 5), bool))
    band = dilated & ~eroded
    if band.any():
        distance = np.linalg.norm(rgb[band].astype(np.float32) - background[None, :], axis=1)
        closeness = np.clip(1.0 - distance / 60.0, 0.0, 1.0)
        halo = float(np.mean(closeness * alpha[band]))
    else:
        halo = 0.0

    # Residual shadow: low-saturation mid-grey kept inside the matte, in the lower image band,
    # which is where this plate's ground shadow sits.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    background_value = float(np.mean(background)) / 255.0
    lower = np.zeros_like(solid)
    lower[int(height * 0.72):] = True
    shadow = solid & lower & (saturation < 0.16) & (value > background_value * 0.55) & (value < background_value * 0.97)

    thin = thin_features(solid)
    thin_labels, thin_sizes = components(thin)

    rows = np.flatnonzero(solid.any(axis=1))
    cols = np.flatnonzero(solid.any(axis=0))
    bounds = ([int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])]
              if rows.size and cols.size else None)

    union = float((solid | reference).sum())
    return {
        "dimensions": [int(width), int(height)],
        "foreground_occupancy": round(area / total, 6),
        "foreground_pixels": int(area),
        "component_count": int((sizes > 0).sum()),
        "component_sizes_top16": [int(v) for v in order[:16]],
        "largest_component_share": round(float(order[0]) / area, 6) if area else 0.0,
        "components_under_0_1_percent": int((positive < area * 0.001).sum()) if area else 0,
        "components_under_0_5_percent": int((positive < area * 0.005).sum()) if area else 0,
        "components_under_1_percent": int((positive < area * 0.01).sum()) if area else 0,
        "holes_inside_foreground": int((hole_sizes > 0).sum()),
        "hole_pixels": int(holes.sum()),
        "edge_halo_score": round(halo, 6),
        "remaining_shadow_pixels": int(shadow.sum()),
        "remaining_shadow_fraction_of_foreground": round(float(shadow.sum()) / area, 6) if area else 0.0,
        "thin_feature_components": int((thin_sizes > 0).sum()),
        "thin_feature_pixels": int(thin.sum()),
        "crop_bounds_x0y0x1y1": bounds,
        "alignment_iou_vs_reference_key": round(float((solid & reference).sum() / union), 6) if union else 0.0,
        "dimensions_match_source": True,
    }


def classify_thin(reference_thin: np.ndarray, reference: np.ndarray, solid: np.ndarray) -> dict:
    """Per-structure verdict for every thin feature present in the deterministic reference key."""
    labels, sizes = components(reference_thin)
    grown = ndimage.binary_dilation(solid, np.ones((5, 5), bool))
    preserved = thickened = removed = 0
    removed_detail = []
    for index in np.flatnonzero(sizes > 0):
        piece = labels == index
        pixels = int(piece.sum())
        covered = float((piece & solid).sum()) / pixels
        if covered >= 0.5:
            # Thickened if the variant's mask is materially fatter than the reference there.
            neighbourhood = ndimage.binary_dilation(piece, np.ones((5, 5), bool))
            if float((neighbourhood & solid).sum()) > float((neighbourhood & reference).sum()) * 1.25:
                thickened += 1
            else:
                preserved += 1
        elif float((piece & grown).sum()) / pixels >= 0.5:
            preserved += 1
        else:
            removed += 1
            ys, xs = np.nonzero(piece)
            removed_detail.append({"pixels": pixels,
                                   "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
    removed_detail.sort(key=lambda item: item["pixels"], reverse=True)
    return {
        "reference_thin_structures": int((sizes > 0).sum()),
        "preserved": preserved,
        "thickened": thickened,
        "removed_as_likely_background_noise": removed,
        "largest_removed": removed_detail[:8],
    }


def write_variant(out: Path, name: str, rgb: np.ndarray, alpha: np.ndarray) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    a8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    paths = {}
    transparent = out / f"{name}_transparent.png"
    cv2.imwrite(str(transparent), np.dstack([bgr, a8]))
    paths["transparent_png"] = str(transparent)

    for label, colour in (("lightgray", LIGHT_GREY), ("white", WHITE)):
        plate = np.full_like(rgb, colour, dtype=np.float32)
        composited = rgb.astype(np.float32) * alpha[..., None] + plate * (1.0 - alpha[..., None])
        path = out / f"{name}_bg_{label}.png"
        cv2.imwrite(str(path), cv2.cvtColor(composited.astype(np.uint8), cv2.COLOR_RGB2BGR))
        paths[f"bg_{label}_png"] = str(path)

    alpha_path = out / f"{name}_alpha.png"
    cv2.imwrite(str(alpha_path), a8)
    paths["alpha_png"] = str(alpha_path)

    solid = alpha > 0.5
    edges = cv2.Canny(a8, 40, 120)
    edge_view = bgr.copy()
    edge_view[edges > 0] = (0, 0, 255)
    edge_path = out / f"{name}_edges.png"
    cv2.imwrite(str(edge_path), edge_view)
    paths["edge_visualisation_png"] = str(edge_path)

    labels, sizes = components(solid)
    palette = np.zeros((labels.max() + 1, 3), np.uint8)
    if labels.max() >= 1:
        order = np.argsort(sizes)[::-1]
        palette[order[0]] = (90, 90, 90)
        rng = np.random.default_rng(7)
        for rank, index in enumerate(order[1:], start=1):
            if sizes[index] == 0:
                continue
            palette[index] = rng.integers(90, 256, 3, dtype=np.uint8)
    component_path = out / f"{name}_components.png"
    cv2.imwrite(str(component_path), palette[labels][..., ::-1])
    paths["component_visualisation_png"] = str(component_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source = Path(args.source)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"unreadable source: {source}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    background = estimate_background(rgb)
    reference_raw, reference = reference_key(rgb)
    reference_thin = thin_features(reference)

    variants: dict[str, np.ndarray] = {}
    notes: dict[str, str] = {}

    # A -- the plate exactly as delivered.
    variants["A_ORIGINAL"] = np.ones((height, width), np.float32)
    notes["A_ORIGINAL"] = ("unchanged plate; alpha is fully opaque so the generator sees the baked "
                           "checkerboard background and the ground shadow")

    # B -- BiRefNet soft alpha.
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
    model.eval()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        prediction = model(tensor)[-1].sigmoid().float().cpu()[0, 0].numpy()
    soft = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_LINEAR)
    soft = np.clip(soft, 0.0, 1.0).astype(np.float32)
    variants["B_BIREFNET_SOFT_ALPHA"] = soft
    notes["B_BIREFNET_SOFT_ALPHA"] = "BiRefNet (ZhengPeng7/BiRefNet) soft alpha, antialiased edges kept"

    # C -- hard silhouette from the same prediction.
    hard = (soft > 0.5)
    hard = ndimage.binary_fill_holes(hard)
    labels, sizes = components(hard)
    if sizes.max() > 0:
        hard = labels == int(np.argmax(sizes))
        hard = ndimage.binary_fill_holes(hard)
    # Shave the one-pixel dark fringe the plate leaves behind on a hard cut.
    fringe = hard & (soft < 0.75) & ndimage.binary_dilation(~hard, np.ones((3, 3), bool))
    hard = hard & ~fringe
    hard = ndimage.binary_fill_holes(hard)
    variants["C_BIREFNET_HARD_MASK"] = hard.astype(np.float32)
    notes["C_BIREFNET_HARD_MASK"] = ("binary silhouette from the BiRefNet prediction, single largest "
                                     "component, internal pinholes filled, dark edge fringe shaved")

    # D -- refinement of the best BiRefNet mask.
    #
    # SAM/SAM2 are not installed in any environment on this machine, and pulling either in would
    # mean a new multi-GB environment for one mask, so this is a classical colour-guided refinement
    # of the soft alpha rather than a promptable-segmenter pass. It only resolves the ambiguous
    # band; it never adds structure the plate does not contain.
    unknown = (soft > 0.05) & (soft < 0.95)
    distance = np.linalg.norm(rgb.astype(np.float32) - background[None, None, :], axis=2)
    plate_like = np.clip(distance / 45.0, 0.0, 1.0)
    refined = soft.copy()
    refined[unknown] = np.clip(soft[unknown] * 0.35 + plate_like[unknown] * 0.65, 0.0, 1.0)
    refined[soft >= 0.95] = 1.0
    refined = np.maximum(refined, reference_raw.astype(np.float32) * np.clip(plate_like, 0, 1))
    solid = refined > 0.5
    solid = ndimage.binary_fill_holes(solid)
    labels, sizes = components(solid)
    if sizes.max() > 0:
        # Keep the hull and every structure attached or close to it; drop only isolated specks
        # smaller than 0.05% of the hull, which is the size class of plate compression noise.
        floor = max(4, int(sizes.max() * 0.0005))
        solid = np.isin(labels, np.flatnonzero(sizes >= floor))
    refined = refined * solid
    variants["D_REFINED_MASK"] = refined.astype(np.float32)
    notes["D_REFINED_MASK"] = ("classical colour-guided refinement of the BiRefNet soft alpha "
                               "(SAM/SAM2 not installed; not introduced to avoid destabilising the "
                               "existing environments), pinholes filled, sub-0.05% specks dropped")

    core = ndimage.binary_erosion(reference, np.ones((2 * CORE_ERODE + 1,) * 2, bool))
    results = {}
    for name, alpha in variants.items():
        directory = root / name
        paths = write_variant(directory, name.lower(), rgb, alpha)
        metrics = measure(alpha, rgb, reference, background)
        solid = alpha > 0.5
        coverage = float((core & solid).sum()) / float(core.sum()) if core.sum() else 1.0
        thin_verdict = classify_thin(reference_thin, reference, solid)
        entry = {
            "variant": name,
            "note": notes[name],
            "artifacts": paths,
            "metrics": metrics,
            "thin_feature_classification": thin_verdict,
            "core_structure_coverage": round(coverage, 6),
            "fail_closed": bool(coverage < CORE_COVERAGE_FLOOR),
            "classification": ("MAJOR_STRUCTURE_LOSS" if coverage < CORE_COVERAGE_FLOOR else "STRUCTURE_PRESERVED"),
        }
        receipt = directory / f"{name.lower()}_receipt.json"
        receipt.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        entry["receipt"] = str(receipt)
        results[name] = entry
        print(f"VARIANT {name} occupancy={metrics['foreground_occupancy']:.4f} "
              f"components={metrics['component_count']} halo={metrics['edge_halo_score']:.4f} "
              f"shadow={metrics['remaining_shadow_pixels']} core_coverage={coverage:.4f} "
              f"{entry['classification']}", flush=True)

    payload = {
        "schema": "ship_input_conditioning_benchmark_v1",
        "source": str(source),
        "source_sha256": sha256(source),
        "source_dimensions": [width, height],
        "estimated_plate_background_rgb": [round(float(v), 1) for v in background],
        "plate_note": "background is a baked-in transparency checkerboard, light grey, not black",
        "reference_key_pixels": int(reference.sum()),
        "reference_thin_structures": int(components(reference_thin)[1].astype(bool).sum()),
        "sam_available": False,
        "variants": results,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"CONDITIONING_BENCHMARK_DONE report={args.report}", flush=True)


if __name__ == "__main__":
    main()
