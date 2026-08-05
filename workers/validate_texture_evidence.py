"""Fail-closed per-texel evidence QA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lowvram3d.texture_provenance import EvidenceState, FrequencyAuthority, SourceClass, load_npz, validate_evidence_invariants


REQUIRED_FIELDS = (
    "triangle_id", "evidence_state", "source_class", "source_view", "source_pixel",
    "barycentric", "visibility", "facing", "face_id_match", "source_mask_valid",
    "confidence", "frequency_authority",
)


def validate(provenance: dict[str, np.ndarray], *, uv_conflicts: int = 0,
             protected_changed: int = 0) -> dict:
    """Compatibility API used by focused unit tests; strict CLI validation is below."""
    result = validate_evidence_invariants(provenance)
    state = np.asarray(provenance["evidence_state"])
    authority = np.asarray(provenance["frequency_authority"])
    gates = {
        **result,
        "direct_texels_without_visibility": int(provenance.get("direct_visibility_missing", 0)),
        "direct_texels_without_face_id_match": int(provenance.get("direct_face_id_missing", 0)),
        "cross_component_raw_rgb_transfer": int(provenance.get("cross_component_raw_rgb", 0)),
        "positive_area_uv_conflicts": int(uv_conflicts),
        "protected_source_texels_changed": int(protected_changed),
        "unobserved_full_frequency_texels": int(np.count_nonzero(
            ~np.isin(state, (EvidenceState.DIRECT_OBSERVED, EvidenceState.GENERATED_OBSERVED,
                              EvidenceState.VISIBLE_SOURCE_GAP))
            & (authority == FrequencyAuthority.FULL))),
    }
    gates["passed"] = all(int(gates[key]) == 0 for key in (
        "unobserved_raw_image_rgb_texels", "unobserved_full_frequency_texels",
        "direct_texels_without_visibility", "direct_texels_without_face_id_match",
        "cross_component_raw_rgb_transfer", "positive_area_uv_conflicts",
        "protected_source_texels_changed"))
    return gates


def _load_required(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"EVIDENCE_MEASUREMENT_MISSING:{label}")
    return np.load(path, allow_pickle=False)


def strict_validate(final_provenance: Path, owner_path: Path, occupied_path: Path,
                    direct_path: Path, gap_path: Path, unobserved_path: Path,
                    visibility_path: Path, face_match_path: Path, source_view_path: Path,
                    source_pixel_path: Path, source_mask_path: Path, triangle_path: Path,
                    basecolor_path: Path, uv_report_path: Path,
                    protected_report_path: Path, stage_receipts_path: Path) -> dict:
    for path, label in ((uv_report_path, "uv-audit-report"),
                        (protected_report_path, "protected-hash-report"),
                        (stage_receipts_path, "stage-receipts")):
        if not path.is_file():
            raise RuntimeError(f"EVIDENCE_MEASUREMENT_MISSING:{label}")
    prov = load_npz(final_provenance)
    missing = [key for key in REQUIRED_FIELDS if key not in prov]
    if missing:
        raise RuntimeError("EVIDENCE_MEASUREMENT_MISSING:final-provenance:" + ",".join(missing))
    owner = _load_required(owner_path, "atlas-owner")
    occupied = _load_required(occupied_path, "atlas-occupied-mask").astype(bool)
    direct = _load_required(direct_path, "direct-observed-mask").astype(bool)
    gap = _load_required(gap_path, "visible-gap-mask").astype(bool)
    unobserved = _load_required(unobserved_path, "unobserved-mask").astype(bool)
    visibility = _load_required(visibility_path, "direct-visibility").astype(bool)
    face_match = _load_required(face_match_path, "direct-face-id-match").astype(bool)
    source_view = _load_required(source_view_path, "direct-source-view")
    source_pixel = _load_required(source_pixel_path, "direct-source-pixel")
    source_mask = _load_required(source_mask_path, "direct-source-mask-valid").astype(bool)
    triangle = _load_required(triangle_path, "direct-triangle-id")
    image = cv2.imread(str(basecolor_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("EVIDENCE_MEASUREMENT_MISSING:basecolor")
    shape = occupied.shape
    arrays = (direct, gap, unobserved, visibility, face_match, source_view, source_pixel,
              source_mask, triangle)
    if any(np.asarray(item).shape[:2] != shape for item in arrays):
        raise RuntimeError("EVIDENCE_MEASUREMENT_MISSING:array-shape-mismatch")
    state = np.asarray(prov["evidence_state"])
    authority = np.asarray(prov["frequency_authority"])
    observed = direct | gap | (state == EvidenceState.GENERATED_OBSERVED)
    direct_without_sample = direct & ((triangle < 0) | (source_view < 0) |
                                      (source_pixel[..., 0] < 0) | (source_pixel[..., 1] < 0) |
                                      ~source_mask)
    # A true black source pixel is valid direct evidence. The failure is an occupied
    # texel that remains black without direct, gap, or completion authority.
    occupied_black = occupied & ~observed & np.all(image == 0, axis=2)
    uv_report = json.loads(uv_report_path.read_text(encoding="utf-8"))
    protected = json.loads(protected_report_path.read_text(encoding="utf-8"))
    stages = json.loads(stage_receipts_path.read_text(encoding="utf-8"))
    positive_conflicts = int(
        (uv_report.get("exact_overlap") or {}).get("positive_overlap_pair_count",
         uv_report.get("interior_texels_claimed_twice", 0) or 0))
    protected_changed = int(protected.get("protected_source_texels_changed", 0) or 0)
    gates = {
        "direct_texels_without_source_sample": int(direct_without_sample.sum()),
        "direct_texels_without_visibility": int((direct & ~visibility).sum()),
        "direct_texels_without_face_id_match": int((direct & ~face_match).sum()),
        "direct_texels_without_source_view": int((direct & (source_view < 0)).sum()),
        "direct_texels_without_source_pixel": int((direct & (source_pixel[..., 0] < 0)).sum()),
        "unobserved_raw_image_rgb_texels": int(validate_evidence_invariants(prov)["unobserved_raw_image_rgb_texels"]),
        "unobserved_full_frequency_texels": int((~observed & (authority == FrequencyAuthority.FULL)).sum()),
        "visible_gap_cross_view_transfer": int(prov.get("visible_gap_cross_view_transfer", 0)),
        "cross_component_raw_rgb_transfer": int(prov.get("cross_component_raw_rgb", 0)),
        "positive_area_uv_conflicts": positive_conflicts,
        "occupied_black_texels": int(occupied_black.sum()),
        "protected_source_texels_changed": protected_changed,
        "unknown_provenance_on_occupied_texels": int((occupied &
            (state == EvidenceState.UNKNOWN)).sum()),
        "uv_occupied_texels": int(occupied.sum()),
        "direct_observed_texels": int(direct.sum()),
        "visible_gap_repaired_texels": int(gap.sum()),
        "unobserved_surface_texels": int(unobserved.sum()),
        "stage_receipts_present": bool(stages.get("stages")),
        "protected_hash_report_present": bool(protected),
    }
    zero_keys = tuple(key for key in gates if key not in {
        "uv_occupied_texels", "direct_observed_texels", "visible_gap_repaired_texels",
        "unobserved_surface_texels", "stage_receipts_present", "protected_hash_report_present",
    })
    gates["passed"] = bool(gates["stage_receipts_present"] and gates["protected_hash_report_present"]
                            and all(int(gates[key]) == 0 for key in zero_keys))
    return gates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-provenance", "--provenance", dest="final_provenance", required=True)
    parser.add_argument("--atlas-owner", required=True)
    parser.add_argument("--atlas-occupied-mask", required=True)
    parser.add_argument("--direct-observed-mask", required=True)
    parser.add_argument("--visible-gap-mask", required=True)
    parser.add_argument("--unobserved-mask", required=True)
    parser.add_argument("--direct-visibility", required=True)
    parser.add_argument("--direct-face-id-match", required=True)
    parser.add_argument("--direct-source-view", required=True)
    parser.add_argument("--direct-source-pixel", required=True)
    parser.add_argument("--direct-source-mask-valid", required=True)
    parser.add_argument("--direct-triangle-id", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--uv-audit-report", required=True)
    parser.add_argument("--protected-hash-report", required=True)
    parser.add_argument("--stage-receipts", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        report = strict_validate(
            Path(args.final_provenance), Path(args.atlas_owner), Path(args.atlas_occupied_mask),
            Path(args.direct_observed_mask), Path(args.visible_gap_mask), Path(args.unobserved_mask),
            Path(args.direct_visibility), Path(args.direct_face_id_match), Path(args.direct_source_view),
            Path(args.direct_source_pixel), Path(args.direct_source_mask_valid),
            Path(args.direct_triangle_id), Path(args.basecolor), Path(args.uv_audit_report),
            Path(args.protected_hash_report), Path(args.stage_receipts))
    except RuntimeError as exc:
        report = {"passed": False, "failure_code": str(exc)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
