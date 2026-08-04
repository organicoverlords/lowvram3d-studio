"""Fail-closed generic evidence and provenance QA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lowvram3d.texture_provenance import EvidenceState, FrequencyAuthority, validate_evidence_invariants


def validate(provenance: dict[str, np.ndarray], *, uv_conflicts: int = 0,
             protected_changed: int = 0) -> dict:
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
            (state != EvidenceState.DIRECT_OBSERVED) & (authority == FrequencyAuthority.FULL))),
    }
    gates["passed"] = all(int(gates[key]) == 0 for key in (
        "unobserved_raw_image_rgb_texels", "unobserved_full_frequency_texels",
        "direct_texels_without_visibility", "direct_texels_without_face_id_match",
        "cross_component_raw_rgb_transfer", "positive_area_uv_conflicts",
        "protected_source_texels_changed"))
    return gates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--uv-conflicts", type=int, default=0)
    parser.add_argument("--protected-changed", type=int, default=0)
    args = parser.parse_args()
    with np.load(args.provenance, allow_pickle=False) as data:
        provenance = {key: data[key] for key in data.files}
    report = validate(provenance, uv_conflicts=args.uv_conflicts,
                      protected_changed=args.protected_changed)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
