"""Frequency-authority-aware fusion; never imports direct RGB into low-only regions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lowvram3d.texture_provenance import EvidenceState, FrequencyAuthority


def fuse(direct_rgb: np.ndarray, completion_rgb: np.ndarray, evidence_state: np.ndarray) -> tuple[np.ndarray, dict]:
    direct = np.asarray(direct_rgb, np.uint8)
    completion = np.asarray(completion_rgb, np.uint8)
    state = np.asarray(evidence_state, np.uint8)
    if direct.shape != completion.shape or state.shape != direct.shape[:2]:
        raise ValueError("fusion arrays do not align")
    out = completion.copy()
    observed = np.isin(state, (EvidenceState.DIRECT_OBSERVED, EvidenceState.GENERATED_OBSERVED))
    out[observed] = direct[observed]
    authority = np.where(observed, FrequencyAuthority.FULL, FrequencyAuthority.LOW_ONLY).astype(np.uint8)
    return out, {"schema": "texture_frequency_fusion_v1", "direct_texels": int(observed.sum()),
                 "low_only_texels": int((~observed).sum()), "unobserved_full_frequency_texels": 0,
                 "global_high_frequency_blur": False, "authority": authority.tolist()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--evidence-state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    direct = cv2.cvtColor(cv2.imread(args.direct, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    completion = cv2.cvtColor(cv2.imread(args.completion, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    state = np.load(args.evidence_state)
    output, report = fuse(direct, completion, state)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
