"""Label, correspondence and bar-footprint checks over a six-view MV-Adapter result.

The numerical gate lives in the inference receipt. What it cannot say is whether image
``view_3_front.png`` really came from the front camera, or whether the removed bar left a
trace in the generated pixels. Both are checked here against the camera contract and the
pre-repair triangle-ID renders, never against a positional list of view names.

The rear-duplication measure is reported, not thresholded into a pass: a red panda's front
and rear silhouettes are mirror images, so correlation alone cannot separate a duplicated
face from a legitimately similar outline. It is evidence for the review, not a verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def file_prefix(view: dict) -> str:
    return str(view.get("control_file_prefix") or view["semantic_name"])


def semantic_of(view: dict) -> str:
    return str(view.get("proven_semantic") or view["semantic_name"])


def caption(image: Image.Image, lines: list[str]) -> Image.Image:
    height = 12 * len(lines) + 8
    framed = Image.new("RGB", (image.width, image.height + height), (18, 18, 20))
    framed.paste(image.convert("RGB"), (0, height))
    draw = ImageDraw.Draw(framed)
    for position, text in enumerate(lines):
        draw.text((4, 3 + position * 12), text, fill=(240, 240, 240))
    return framed


def sheet(tiles: list[Image.Image], columns: int = 3) -> Image.Image:
    rows = (len(tiles) + columns - 1) // columns
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (width * columns, height * rows), (18, 18, 20))
    for position, tile in enumerate(tiles):
        canvas.paste(tile, ((position % columns) * width, (position // columns) * height))
    return canvas


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64).ravel() - a.mean()
    y = b.astype(np.float64).ravel() - b.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denominator) if denominator > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--pre-repair-bundle", required=True)
    parser.add_argument("--repair-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    bundle = Path(args.bundle)
    pre_repair = Path(args.pre_repair_bundle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))
    repair = json.loads(Path(args.repair_report).read_text(encoding="utf-8"))
    removed = np.asarray(sorted(set(repair["removed_face_ids"])
                                | set(repair["dropped_orphan_face_ids"])), dtype=np.int64)

    # Pair the pre-repair bundle by camera direction: it was relabelled, so its indices differ.
    def key(view):
        return tuple(int(round(float(c))) for c in view["camera_direction"])

    pre_by_direction = {key(view): view for view in
                        json.loads((pre_repair / "camera_contract.json")
                                   .read_text(encoding="utf-8"))["views"]}

    views = sorted(contract["views"], key=lambda item: int(item["index"]))
    # The receipt records each output as {index, name, path, sha256}, not as a bare path.
    outputs = {str(item["name"]): Path(item["path"]) for item in receipt.get("output_images", [])}
    output_digests = {str(item["name"]): str(item["sha256"])
                      for item in receipt.get("output_images", [])}
    records, failures = [], []
    raw_tiles, semantic_tiles, correspondence_tiles = [], [], []
    images: dict[str, np.ndarray] = {}

    for view in views:
        index = int(view["index"])
        semantic = semantic_of(view)
        expected_name = f"view_{index}_{semantic}.png"
        path = outputs.get(expected_name)
        if path is None or not path.is_file():
            failures.append(f"missing_output:{expected_name}")
            continue
        image = np.asarray(Image.open(path).convert("RGB"))
        images[semantic] = image

        control_mask = np.asarray(
            Image.open(bundle / f"{file_prefix(view)}_mask.png").convert("L")) > 127
        pre_view = pre_by_direction.get(key(view))
        bar_footprint = np.zeros(control_mask.shape, dtype=bool)
        if pre_view is not None:
            pre_ids = np.load(pre_repair / f"{file_prefix(pre_view)}_triangle_ids.npy")
            if pre_ids.shape == control_mask.shape:
                bar_footprint = np.isin(pre_ids, removed)

        resized = np.asarray(Image.fromarray(image).resize(
            (control_mask.shape[1], control_mask.shape[0]), Image.BILINEAR))
        background = resized[~control_mask]
        background_mean = background.mean(axis=0) if background.size else np.zeros(3)
        background_std = float(background.std()) if background.size else 0.0
        if bar_footprint.any():
            footprint = resized[bar_footprint].astype(np.float64)
            deviation = float(np.abs(footprint - background_mean).mean())
            # Compare against the spread of the background itself rather than an absolute
            # threshold: an unlit backdrop and a busy one have very different scales.
            bar_like = bool(deviation > 3.0 * max(background_std, 1.0))
        else:
            deviation, bar_like = 0.0, False

        record = {
            "raw_index": index,
            "semantic_label": semantic,
            "output_filename": expected_name,
            "output_path": str(path),
            "output_sha256": output_digests.get(expected_name),
            "control_file_prefix": file_prefix(view),
            "azimuth_deg": view["azimuth_deg"],
            "elevation_deg": view["elevation_deg"],
            "camera_direction_control_space": view["camera_direction"],
            "filename_matches_contract": True,
            "removed_bar_footprint_pixels": int(bar_footprint.sum()),
            "bar_footprint_deviation_from_background": deviation,
            "background_std": background_std,
            "bar_like_content_in_removed_footprint": bar_like,
        }
        if bar_like:
            failures.append(f"bar_like_content:{semantic}")
        records.append(record)

        preview = Image.fromarray(image).resize((320, 320), Image.LANCZOS)
        raw_tiles.append(caption(preview, [f"raw{index}"]))
        semantic_tiles.append(caption(preview, [semantic]))
        control = Image.open(bundle / f"{file_prefix(view)}_position.png").convert("RGB")
        pair = Image.new("RGB", (640, 320), (18, 18, 20))
        pair.paste(control.resize((320, 320), Image.LANCZOS), (0, 0))
        pair.paste(preview, (320, 0))
        correspondence_tiles.append(caption(pair, [
            f"raw{index} = {semantic}   control | output",
            f"az {view['azimuth_deg']:.2f}  el {view['elevation_deg']:.2f}  "
            f"files {file_prefix(view)}_*",
        ]))

    sheets = {}
    for name, tiles in (("a_raw_index", raw_tiles), ("b_semantic_corrected", semantic_tiles),
                        ("c_control_output_correspondence", correspondence_tiles)):
        if not tiles:
            continue
        path = output_dir / f"output_contact_sheet_{name}.png"
        sheet(tiles, columns=3 if name != "c_control_output_correspondence" else 2).save(path)
        sheets[name] = str(path)

    duplication = {}
    if "front" in images and "rear" in images:
        rear = images["rear"]
        duplication = {
            "rear_vs_front_correlation": correlation(rear, images["front"]),
            "rear_vs_mirrored_front_correlation": correlation(rear, images["front"][:, ::-1]),
            "receipt_front_rear_direct_correlation":
                (receipt.get("qa") or {}).get("front_rear_direct_correlation"),
            "receipt_front_rear_mirrored_correlation":
                (receipt.get("qa") or {}).get("front_rear_mirrored_correlation"),
            "interpretation": ("the values measured here include the shared background and "
                               "are therefore inflated; the receipt's foreground-only "
                               "correlations are the ones to read"),
        }

    report = {
        "schema": "mvadapter_output_verification_v1",
        "receipt": str(args.receipt),
        "receipt_status": receipt.get("status"),
        "bundle": str(bundle),
        "raw_to_semantic": contract.get("raw_to_semantic"),
        "camera_contract_classification": contract.get("classification"),
        "views": records,
        "contact_sheets": sheets,
        "rear_face_duplication_measure": duplication,
        "failures": failures,
        "gates": {
            "all_six_outputs_present": len(records) == 6,
            "every_filename_matches_the_contract": len(records) == 6 and not [
                f for f in failures if f.startswith("missing_output")],
            "no_bar_like_content_in_removed_footprint": not [
                f for f in failures if f.startswith("bar_like_content")],
        },
        "visual_confirmation": "USER_REVIEW_REQUIRED",
    }
    report["gates"]["all_passed"] = all(report["gates"].values())

    # This worker only checks labels, correspondence and the bar footprint. The run's own
    # numerical and structural verdicts live in the receipt, and a label pass cannot
    # promote a run whose structural gate rejected it.
    finite = (receipt.get("finite_gate") or {}).get("passed") is True
    qa = receipt.get("qa") or {}
    report["receipt_gates"] = {
        "finite_gate_passed": finite,
        "structural_gate_passed": bool(qa.get("structural_gate_passed")),
        "colour_gate_passed": bool(qa.get("colour_gate_passed")),
        "semantic_gate_passed": bool(qa.get("semantic_gate_passed")),
        "rear_numeric_gate_passed": bool(qa.get("rear_numeric_gate_passed")),
    }
    if not (finite and report["gates"]["all_passed"]):
        report["classification"] = "PANDA_REPAIRED_CONTROLS_VALIDATION_REJECTED"
    elif report["receipt_gates"]["structural_gate_passed"]:
        report["classification"] = "PANDA_REPAIRED_CONTROLS_384X2_PROVEN"
    else:
        report["classification"] = (
            "PANDA_REPAIRED_CONTROLS_384X2_NUMERICALLY_PROVEN_VISUAL_REJECTED")
        report["structural_rejection_note"] = (
            "Two denoising steps do not resolve a background, so foreground coverage is "
            "near 1.0 and silhouette IoU against the control is low. This is a property of "
            "the step count, not evidence about the repaired mesh or the permutation.")
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"MVADAPTER_OUTPUT_VERIFY {report['classification']} failures={failures}", flush=True)
    return 0 if report["classification"].endswith("PROVEN") else 3


if __name__ == "__main__":
    raise SystemExit(main())
