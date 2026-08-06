# Panda candidate registry — 2026-08-06

This file is the single index for the current panda visual candidates. It exists to prevent diagnostic outputs from being mistaken for the latest-good baseline.

## Current baseline (keep)

| Role | Path | SHA256 | Status |
|---|---|---|---|
| Latest-good 2048 GLB | `C:\AI\panda_support_local_2048\candidate_2048\panda_atlas_support_fixed_2048.glb` | `1c4bec32a31ce90766f12f095e2d3c9c7e57e0b34e4f8ffc79c9a29ea8e7a817` | **BASELINE — use for comparisons** |
| Latest-good 2048 atlas | `C:\AI\panda_support_local_2048\candidate_2048\atlas_2048_nearest.png` | `88ee2bf03bfe27b6ee28720734c6abd94fc03242f27a2f33529bc3db0f08383e` | **BASELINE INPUT** |
| Baseline contact sheet | `C:\AI\panda_support_local_2048\candidate_2048\renders\contact_sheet.png` | see local render manifest | **BASELINE VISUAL** |

The baseline has the best current texture quality, but its face is visibly misregistered in the marked three-quarter view: the eye/muzzle cluster sits on the left/front side of the hood instead of the hood opening's center axis.

## Core ownership implementation now present

The branch now contains model-agnostic implementation code rather than another atlas-shift experiment:

- `workers/face_surface_ownership.py`: BVH-accelerated multi-hit camera raycasting, ordered depth layers, connected surface-patch extraction, exact triangle-mask output, and patch scoring.
- `workers/face_patch_texture.py`: exact triangle-owned atlas editing, thin-plate landmark-to-surface mapping, premultiplied-alpha sampling, non-face byte immutability, and geometry/UV/index hash gates.
- `tests/test_face_surface_ownership_core.py`: synthetic layered-surface, barycentric, patch-connectivity, TPS, and matte-contamination tests.

These files are implementation infrastructure. No candidate is promoted until the local worker runs them against the exact 2048 baseline and produces fresh renders.

## Current terminal status

- `PANDA_FACE_REGISTRATION`: **CONFIRMED DEFECT**
- `PANDA_FACE_CORRECTION`: **IMPLEMENTATION ADDED; REAL-ASSET RUN PENDING**
- `PANDA_FACE_SURFACE_OWNERSHIP`: **CODED; NOT YET PROVEN ON BASELINE**
- `PANDA_FACE_LANDMARK_TO_SURFACE_LOCK`: **CODED; ANCHOR FIXTURE/RUN PENDING**
- `PANDA_LATEST_GOOD_BASELINE`: **PRESERVED**
- `PANDA_PRODUCTION_READY`: **NO**
