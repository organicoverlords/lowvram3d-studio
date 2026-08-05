"""Complete unobserved triangles with aggregate low-frequency material values only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lowvram3d.texture_provenance import EvidenceState, FrequencyAuthority


def complete(triangle_rgb: np.ndarray, evidence_state: np.ndarray, regions: np.ndarray,
             *, global_prior=(96, 96, 96)) -> tuple[np.ndarray, dict]:
    rgb = np.asarray(triangle_rgb, np.float32).copy()
    state = np.asarray(evidence_state, np.uint8).copy()
    regions = np.asarray(regions, np.int32)
    direct = np.isin(state, (EvidenceState.DIRECT_OBSERVED, EvidenceState.GENERATED_OBSERVED))
    if rgb.shape != (len(state), 3) or regions.shape != state.shape:
        raise ValueError("triangle RGB, evidence state, and regions must align")
    global_value = np.median(rgb[direct], axis=0) if direct.any() else np.asarray(global_prior, np.float32)
    methods = np.full(len(state), "unresolved", dtype="U32")
    authority = np.full(len(state), FrequencyAuthority.NONE, np.uint8)
    authority[direct] = FrequencyAuthority.FULL
    for region in np.unique(regions):
        targets = (regions == region) & ~direct
        donors = (regions == region) & direct
        if not targets.any():
            continue
        rgb[targets] = np.median(rgb[donors], axis=0) if donors.any() else global_value
        state[targets] = np.uint8(EvidenceState.PROCEDURAL_COMPLETION if donors.any()
                                  else EvidenceState.MATERIAL_PRIOR)
        authority[targets] = np.uint8(FrequencyAuthority.LOW_ONLY)
        methods[targets] = "region_palette" if donors.any() else "global_palette"
    report = {
        "schema": "unobserved_surface_completion_v1",
        "procedural_triangles": int(np.count_nonzero(state == EvidenceState.PROCEDURAL_COMPLETION)),
        "material_prior_triangles": int(np.count_nonzero(state == EvidenceState.MATERIAL_PRIOR)),
        "unresolved_triangles": int(np.count_nonzero(state == EvidenceState.UNRESOLVED)),
        "raw_rgb_donors": 0, "high_frequency_donors": 0,
        "authority": "LOW_ONLY_FOR_UNOBSERVED", "methods": methods.tolist(),
    }
    return {"rgb": rgb.astype(np.uint8), "evidence_state": state,
            "frequency_authority": authority, "completion_method": methods}, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triangle-rgb", required=True)
    parser.add_argument("--evidence-state", required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result, report = complete(np.load(args.triangle_rgb), np.load(args.evidence_state),
                              np.load(args.regions))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
