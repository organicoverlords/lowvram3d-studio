"""Pipeline V2 visual evaluator: turn the shaman's manual rejections into automatic failure codes.

Every check here exists because a specific defect shipped past a green build during the shaman run.
The rule the whole module is written against: an image that is finite, fully covered and correctly
sized can still be completely wrong, so no check may rest on aggregate statistics alone when a
structural test is available.

What the notable ones caught:

* UV_ROW_ORIENTATION_MISMATCH - the projector writes row = (1-v)*(size-1) while glTF reads v=0 as
  the first row. Unconverted, the atlas is mirrored against its own UVs, and it does not look
  mirrored - it looks like a plausible patchwork. Detected by sampling the atlas both ways at known
  projected positions: the wrong convention was off by 61.9/255 against 3.2/255 for the right one.
* FLAT_NEUTRAL_ATLAS_REGIONS - the component-local prior collapses to a single colour when the mesh
  is one large component, painting a third of the model flat grey while every global metric stays
  healthy. Detected per UV island.
* MATERIAL_ID_NOISE - a per-triangle-noise material map passes coverage, finiteness and blackness
  checks perfectly. Detected by resolved component count.
* CAMERA_LABEL_MISMATCH - glTF -Z imports as Blender +Y, so a view labelled "front" can be the back.
* REAR_MIRRORS_FRONT - mirrored front imagery on the rear, the classic single-view failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# --- failure codes -----------------------------------------------------------------------------
MISSING_THIN_FEATURES = "MISSING_THIN_FEATURES"
FLOATING_DEBRIS = "FLOATING_DEBRIS"
BAD_ORIENTATION = "BAD_ORIENTATION"
UV_OVERLAP = "UV_OVERLAP"
UV_DEGENERATE = "UV_DEGENERATE"
MATERIAL_ID_NOISE = "MATERIAL_ID_NOISE"
FLAT_NEUTRAL_ATLAS_REGIONS = "FLAT_NEUTRAL_ATLAS_REGIONS"
FACE_UNREADABLE = "FACE_UNREADABLE"
CAMERA_LABEL_MISMATCH = "CAMERA_LABEL_MISMATCH"
UV_ROW_ORIENTATION_MISMATCH = "UV_ROW_ORIENTATION_MISMATCH"
REAR_MIRRORS_FRONT = "REAR_MIRRORS_FRONT"
CROSS_COMPONENT_PROJECTION = "CROSS_COMPONENT_PROJECTION"
PLASTIC_ROUGHNESS = "PLASTIC_ROUGHNESS"
UNFINISHED_SYNTHESIS = "UNFINISHED_SYNTHESIS"
BACKGROUND_CONTAMINATION = "BACKGROUND_CONTAMINATION"

# Blocking codes stop a stage. Advisory codes are recorded for the human but do not fail the run,
# because their detectors are weaker than the defect they describe.
BLOCKING = {
    UV_ROW_ORIENTATION_MISMATCH, FLAT_NEUTRAL_ATLAS_REGIONS, MATERIAL_ID_NOISE,
    CAMERA_LABEL_MISMATCH, REAR_MIRRORS_FRONT, PLASTIC_ROUGHNESS, BACKGROUND_CONTAMINATION,
    FLOATING_DEBRIS, BAD_ORIENTATION, UV_OVERLAP, UV_DEGENERATE, UNFINISHED_SYNTHESIS,
}
ADVISORY = {MISSING_THIN_FEATURES, FACE_UNREADABLE, CROSS_COMPONENT_PROJECTION}


def finding(code: str, detail: str, measured: dict) -> dict:
    return {"code": code, "severity": "blocking" if code in BLOCKING else "advisory",
            "detail": detail, "measured": measured}


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = 65535.0 if image.dtype == np.uint16 else 255.0
    rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32) / scale
    alpha = image[:, :, 3].astype(np.float32) / scale if image.shape[2] == 4 else None
    return rgb, alpha


def silhouette(path: Path) -> np.ndarray:
    rgb, alpha = load_rgb(path)
    if alpha is not None:
        return alpha > 0.5
    return rgb.max(axis=2) < 0.96


def normalised_silhouette(mask: np.ndarray, size: int = 256) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros((size, size), np.float32)
    crop = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1].astype(np.float32)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def check_atlas_orientation(atlas_path, truth_path, report) -> list[dict]:
    """Compare the atlas against recorded (uv, colour) truth samples in both row conventions."""
    if not truth_path or not Path(truth_path).exists():
        return []
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    uv = np.array(truth["uv"], np.float64)
    expected = np.array(truth["rgb"], np.float64)
    if uv.size == 0:
        return []
    atlas, _ = load_rgb(Path(atlas_path))
    size = atlas.shape[0]
    x = np.clip((uv[:, 0] * (size - 1)).astype(int), 0, size - 1)
    standard = np.clip((uv[:, 1] * (size - 1)).astype(int), 0, size - 1)
    inverted = np.clip(((1.0 - uv[:, 1]) * (size - 1)).astype(int), 0, size - 1)
    error_standard = float(np.abs(atlas[standard, x] * 255.0 - expected).mean())
    error_inverted = float(np.abs(atlas[inverted, x] * 255.0 - expected).mean())
    report["atlas_orientation"] = {"error_gltf_standard": round(error_standard, 3),
                                   "error_inverted": round(error_inverted, 3),
                                   "samples": int(len(uv))}
    if error_inverted < error_standard * 0.5:
        return [finding(UV_ROW_ORIENTATION_MISMATCH,
                        "atlas matches the inverted row convention; it is vertically mirrored "
                        "against its own UVs and every chart samples an unrelated chart",
                        {"error_gltf_standard": round(error_standard, 3),
                         "error_inverted": round(error_inverted, 3)})]
    return []


def check_flat_regions(region_report_path, report, max_flat_triangle_percent: float) -> list[dict]:
    if not region_report_path or not Path(region_report_path).exists():
        return []
    data = json.loads(Path(region_report_path).read_text(encoding="utf-8"))
    examined = max(int(data.get("triangles_examined", 0)), 1)
    flat = int(data.get("flat_island_triangles", 0))
    percent = flat / examined * 100.0
    report["flat_regions"] = {"flat_islands": data.get("flat_islands"),
                              "flat_triangle_percent": round(percent, 3)}
    if percent > max_flat_triangle_percent:
        return [finding(FLAT_NEUTRAL_ATLAS_REGIONS,
                        f"{percent:.2f}% of examined triangles sit in UV islands with effectively "
                        "constant colour, which renders as untextured flat fill",
                        {"flat_triangle_percent": round(percent, 3),
                         "limit": max_flat_triangle_percent})]
    return []


def check_material_id(material_id_path, component_count, report) -> list[dict]:
    findings = []
    if component_count is not None:
        report["material_id_components"] = int(component_count)
        if int(component_count) > 500:
            findings.append(finding(MATERIAL_ID_NOISE,
                                    f"material-ID resolved {component_count} components; the map is "
                                    "per-triangle noise rather than per-part identity",
                                    {"components": int(component_count), "limit": 500}))
    if material_id_path and Path(material_id_path).exists():
        rgb, _ = load_rgb(Path(material_id_path))
        quantised = (rgb * 255).astype(np.int32) // 8
        key = quantised[..., 0] * 4096 + quantised[..., 1] * 64 + quantised[..., 2]
        distinct = int(np.unique(key).size)
        report["material_id_distinct_colours"] = distinct
        if distinct > 20000:
            findings.append(finding(MATERIAL_ID_NOISE,
                                    f"material-ID carries {distinct} distinct colours",
                                    {"distinct_colours": distinct, "limit": 20000}))
    return findings


def check_orm(orm_path, report) -> list[dict]:
    if not orm_path or not Path(orm_path).exists():
        return []
    rgb, _ = load_rgb(Path(orm_path))
    roughness, metallic = rgb[..., 1], rgb[..., 2]
    covered = rgb.max(axis=2) > 0.02
    if not covered.any():
        covered = np.ones(roughness.shape, bool)
    stats = {"roughness_mean": round(float(roughness[covered].mean()), 4),
             "roughness_p05": round(float(np.percentile(roughness[covered], 5)), 4),
             "roughness_std": round(float(roughness[covered].std()), 4),
             "metallic_percent_above_0_5": round(float((metallic[covered] > 0.5).mean() * 100), 4)}
    report["orm"] = stats
    findings = []
    # Two distinct plastic failures: uniformly glossy, or uniformly constant.
    if stats["roughness_p05"] < 0.30:
        findings.append(finding(PLASTIC_ROUGHNESS,
                                f"5th-percentile roughness {stats['roughness_p05']} produces broad "
                                "specular sheen that washes out surface detail",
                                stats))
    if stats["roughness_std"] < 0.02:
        findings.append(finding(PLASTIC_ROUGHNESS,
                                f"roughness standard deviation {stats['roughness_std']} is flat; "
                                "no material variation is present",
                                stats))
    if stats["metallic_percent_above_0_5"] > 12.0:
        findings.append(finding(PLASTIC_ROUGHNESS,
                                f"{stats['metallic_percent_above_0_5']}% of the atlas is metallic",
                                stats))
    return findings


def _normalised_patch(rgb: np.ndarray, mask: np.ndarray, size: int = 192):
    """Crop to the subject, resize to a fixed box, and normalise over the subject only.

    Masking matters more than it looks: the source matte carries a white background and the renders
    carry a transparent-black one, so correlating the raw crops mostly compares backgrounds and
    comes out strongly negative for every view alike.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    crop = rgb[box].mean(axis=2)
    inside = mask[box].astype(np.float32)
    small = cv2.resize(crop * inside, (size, size), interpolation=cv2.INTER_AREA)
    weight = cv2.resize(inside, (size, size), interpolation=cv2.INTER_AREA)
    valid = weight > 0.35
    if valid.sum() < size:
        return None
    small = np.where(valid, small / np.maximum(weight, 1e-6), 0.0)
    values = small[valid]
    normalised = np.zeros_like(small)
    normalised[valid] = (values - values.mean()) / max(values.std(), 1e-6)
    return normalised


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).mean())


def check_views(render_dir: Path, source_mask, source_rgb, report) -> list[dict]:
    findings = []
    front_path, back_path = render_dir / "front.png", render_dir / "back.png"
    if not (front_path.exists() and back_path.exists()):
        return findings

    front_mask, back_mask = silhouette(front_path), silhouette(back_path)
    front_rgb, _ = load_rgb(front_path)
    back_rgb, _ = load_rgb(back_path)

    if source_rgb is not None:
        # Silhouette cannot decide this. On a roughly symmetric subject front and back outlines are
        # nearly identical, and when the generated mesh is mirrored relative to the source - which
        # happens - the BACK silhouette actually matches the source better, so a silhouette test
        # reports a swapped camera on a correct pipeline.
        #
        # Colour content can decide it: the front is projected from the source, so it must resemble
        # the source far more than the synthesized rear does. Scored mirror-invariantly, because a
        # mirrored mesh is a generator property and not a camera fault.
        def resemblance(rgb, mask):
            patch = _normalised_patch(rgb, mask)
            if patch is None:
                return 0.0
            return max(_correlate(patch, source_patch), _correlate(patch, source_patch[:, ::-1]))

        source_patch = _normalised_patch(source_rgb, source_mask)
        if source_patch is not None:
            front_score = resemblance(front_rgb, front_mask)
            back_score = resemblance(back_rgb, back_mask)
            report["view_labels"] = {"front_source_resemblance": round(front_score, 4),
                                     "back_source_resemblance": round(back_score, 4),
                                     "basis": "colour correlation, mirror-invariant"}
            # Margin measured on the shaman: front 0.162 vs back 0.093 when labelled correctly, and
            # exactly reversed when the front/back cameras are swapped. 0.05 sits inside that gap
            # without firing on the noise of a symmetric subject.
            if back_score > front_score + 0.05:
                findings.append(finding(CAMERA_LABEL_MISMATCH,
                                        "the view labelled back resembles the source more than the "
                                        "one labelled front; the camera convention is inverted",
                                        report["view_labels"]))

    # Rear-mirrors-front: compare the front render against the horizontally flipped back render.
    a, b = _normalised_patch(front_rgb, front_mask, 256), _normalised_patch(back_rgb, back_mask, 256)
    if a is not None and b is not None:
        mirrored = _correlate(a, b[:, ::-1])
        direct = _correlate(a, b)
        report["rear_front_correlation"] = {"mirrored": round(mirrored, 4), "direct": round(direct, 4)}
        if mirrored > 0.82 and mirrored > direct + 0.10:
            findings.append(finding(REAR_MIRRORS_FRONT,
                                    f"rear render correlates {mirrored:.3f} with the mirrored front; "
                                    "front imagery has been stamped onto the back",
                                    report["rear_front_correlation"]))
    return findings


def check_synthesis(coverage_path, basecolor_path, report, min_contrast_ratio: float) -> list[dict]:
    if not (coverage_path and Path(coverage_path).exists() and Path(basecolor_path).exists()):
        return []
    coverage = cv2.imread(str(coverage_path), cv2.IMREAD_GRAYSCALE)
    rgb, _ = load_rgb(Path(basecolor_path))
    if coverage.shape[0] != rgb.shape[0]:
        return []
    island, observed = coverage >= 40, coverage >= 255
    synthesized = island & ~observed
    if not (observed.any() and synthesized.any()):
        return []
    grey = rgb.mean(axis=2)
    detail = np.abs(grey - cv2.GaussianBlur(grey, (0, 0), 4.0))
    observed_contrast = float(detail[observed].std())
    synthesized_contrast = float(detail[synthesized].std())
    ratio = synthesized_contrast / max(observed_contrast, 1e-6)
    report["synthesis"] = {"observed_contrast": round(observed_contrast, 5),
                           "synthesized_contrast": round(synthesized_contrast, 5),
                           "ratio": round(ratio, 4),
                           "synthesized_percent": round(float(synthesized.sum() / island.sum() * 100), 2)}
    if ratio < min_contrast_ratio:
        return [finding(UNFINISHED_SYNTHESIS,
                        f"synthesized regions carry {ratio:.2f} of the observed surface detail; "
                        "they read as unfinished clay next to the projected areas",
                        report["synthesis"])]
    return []


def check_background(basecolor_path, coverage_path, background_rgb, report) -> list[dict]:
    if background_rgb is None or not Path(basecolor_path).exists():
        return []
    rgb, _ = load_rgb(Path(basecolor_path))
    target = np.array(background_rgb, np.float32) / 255.0
    distance = np.linalg.norm(rgb - target, axis=2)
    if coverage_path and Path(coverage_path).exists():
        coverage = cv2.imread(str(coverage_path), cv2.IMREAD_GRAYSCALE)
        region = coverage >= 255 if coverage.shape[0] == rgb.shape[0] else np.ones(distance.shape, bool)
    else:
        region = np.ones(distance.shape, bool)
    percent = float((distance[region] < 0.06).mean() * 100)
    report["background_contamination_percent"] = round(percent, 4)
    if percent > 2.0:
        return [finding(BACKGROUND_CONTAMINATION,
                        f"{percent:.2f}% of observed texels sit within 6% of the source background "
                        "colour; matting leaked into the projection",
                        {"percent": round(percent, 4), "limit": 2.0})]
    return []


def check_geometry(geometry_report_path, profile, report) -> list[dict]:
    if not geometry_report_path or not Path(geometry_report_path).exists():
        return []
    data = json.loads(Path(geometry_report_path).read_text(encoding="utf-8"))
    findings = []
    extent = data.get("extent") or {}
    if extent:
        values = sorted(float(v) for v in extent.values())
        ratio = values[-1] / max(values[0], 1e-9)
        report["axis_ratio"] = round(ratio, 3)
        limit = float(profile.get("max_axis_ratio", 8.0))
        if ratio > limit:
            findings.append(finding(BAD_ORIENTATION,
                                    f"longest-to-shortest axis ratio {ratio:.2f} exceeds {limit} for "
                                    "this profile; the mesh is collapsed or lying down",
                                    {"axis_ratio": round(ratio, 3), "limit": limit}))
    debris = data.get("debris") or {}
    if debris:
        removed = int(debris.get("triangles_removed", 0))
        remaining = int(debris.get("unsupported_components_remaining", 0))
        report["debris"] = {"triangles_removed": removed, "components_remaining": remaining}
        if remaining > 0:
            findings.append(finding(FLOATING_DEBRIS,
                                    f"{remaining} unsupported detached components remain above the "
                                    "debris height threshold",
                                    report["debris"]))
    return findings


def check_uv(uv_report_path, report) -> list[dict]:
    if not uv_report_path or not Path(uv_report_path).exists():
        return []
    data = json.loads(Path(uv_report_path).read_text(encoding="utf-8"))
    exact = data.get("exact_overlap", data)
    findings = []
    overlap = float(data.get("positive_overlap_total_texels_equivalent",
                             exact.get("positive_overlap_total_texels_equivalent", 0.0)))
    degenerate = int(exact.get("degenerate_uv_triangle_count", data.get("degenerate_uv_triangles", 0)))
    tested = int(exact.get("tested_pair_count", 0))
    timed_out = bool(exact.get("timed_out", False))
    report["uv"] = {"overlap_texels": overlap, "degenerate": degenerate,
                    "tested_pairs": tested, "timed_out": timed_out}
    # A timed-out detector reports zero overlap because it never tested a pair. Those zeroes are
    # not evidence of a clean atlas and must not be read as one.
    if timed_out or tested <= 0:
        findings.append(finding(UV_OVERLAP,
                                "overlap detector did not complete; reported zeroes are not evidence",
                                report["uv"]))
    elif overlap > 1.0:
        findings.append(finding(UV_OVERLAP, f"{overlap:.3f} texel-equivalents of positive-area "
                                            "UV overlap", report["uv"]))
    if degenerate > 0:
        findings.append(finding(UV_DEGENERATE, f"{degenerate} degenerate UV triangles", report["uv"]))
    return findings


def check_face(render_dir: Path, source_face_contrast, report, min_ratio: float) -> list[dict]:
    face_path = render_dir / "close_face.png"
    if not face_path.exists():
        return []
    rgb, alpha = load_rgb(face_path)
    mask = alpha > 0.5 if alpha is not None else np.ones(rgb.shape[:2], bool)
    if not mask.any():
        return []
    grey = rgb.mean(axis=2)
    edges = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
    contrast = float(np.abs(edges)[mask].std())
    report["face"] = {"edge_energy": round(contrast, 5)}
    if source_face_contrast:
        ratio = contrast / max(float(source_face_contrast), 1e-6)
        report["face"]["ratio_to_source"] = round(ratio, 4)
        if ratio < min_ratio:
            return [finding(FACE_UNREADABLE,
                            f"face close-up carries {ratio:.2f} of the source's edge energy; "
                            "the beak and eye region are smeared",
                            report["face"])]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--source-image", default="")
    parser.add_argument("--basecolor", default="")
    parser.add_argument("--orm", default="")
    parser.add_argument("--material-id", default="")
    parser.add_argument("--coverage", default="")
    parser.add_argument("--orientation-truth", default="")
    parser.add_argument("--region-report", default="")
    parser.add_argument("--uv-report", default="")
    parser.add_argument("--geometry-report", default="")
    parser.add_argument("--material-id-components", type=int, default=None)
    parser.add_argument("--profile-json", default="")
    parser.add_argument("--background-rgb", default="")
    parser.add_argument("--source-face-contrast", type=float, default=0.0)
    parser.add_argument("--max-flat-triangle-percent", type=float, default=5.0)
    parser.add_argument("--min-synthesis-contrast-ratio", type=float, default=0.45)
    parser.add_argument("--min-face-contrast-ratio", type=float, default=0.35)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    measured: dict = {}
    profile = json.loads(Path(args.profile_json).read_text(encoding="utf-8")) if args.profile_json else {}
    source_mask = source_rgb = None
    if args.source_image and Path(args.source_image).exists():
        source_mask = silhouette(Path(args.source_image))
        source_rgb = load_rgb(Path(args.source_image))[0]
    background = [float(v) for v in args.background_rgb.split(",")] if args.background_rgb else None

    findings: list[dict] = []
    if args.basecolor:
        findings += check_atlas_orientation(args.basecolor, args.orientation_truth, measured)
        findings += check_synthesis(args.coverage, args.basecolor, measured, args.min_synthesis_contrast_ratio)
        findings += check_background(args.basecolor, args.coverage, background, measured)
    findings += check_flat_regions(args.region_report, measured, args.max_flat_triangle_percent)
    findings += check_material_id(args.material_id, args.material_id_components, measured)
    findings += check_orm(args.orm, measured)
    findings += check_views(render_dir, source_mask, source_rgb, measured)
    findings += check_geometry(args.geometry_report, profile, measured)
    findings += check_uv(args.uv_report, measured)
    findings += check_face(render_dir, args.source_face_contrast, measured, args.min_face_contrast_ratio)

    blocking = [f for f in findings if f["severity"] == "blocking"]
    result = {
        "render_dir": str(render_dir),
        "passed": not blocking,
        "blocking_codes": sorted({f["code"] for f in blocking}),
        "advisory_codes": sorted({f["code"] for f in findings if f["severity"] == "advisory"}),
        "findings": findings,
        "measured": measured,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"VISUAL_EVAL passed={result['passed']} blocking={result['blocking_codes']} "
          f"advisory={result['advisory_codes']}", flush=True)
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
