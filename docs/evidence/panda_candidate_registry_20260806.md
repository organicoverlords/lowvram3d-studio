# Panda candidate registry — 2026-08-06

This file is the single index for the current panda visual candidates. It
exists to prevent diagnostic outputs from being mistaken for the latest-good
baseline.

## Current baseline (keep)

| Role | Path | SHA256 | Status |
|---|---|---|---|
| Latest-good 2048 GLB | `C:\AI\panda_support_local_2048\candidate_2048\panda_atlas_support_fixed_2048.glb` | `1c4bec32a31ce90766f12f095e2d3c9c7e57e0b34e4f8ffc79c9a29ea8e7a817` | **BASELINE — use for comparisons** |
| Baseline contact sheet | `C:\AI\panda_support_local_2048\candidate_2048\renders\contact_sheet.png` | see local render manifest | **BASELINE VISUAL** |

The baseline has the best current texture quality, but its face is visibly
misregistered in the marked three-quarter view: the eye/muzzle cluster sits on
the left/front side of the hood instead of the hood opening's center axis.

## Face-correction diagnostics (do not mix into baseline)

| Path | Input | Status | Reason |
|---|---|---|---|
| `C:\AI\panda_support_local_2048\face_recenter_diagnostic_only\face_recentered_2048_diagnostic_only.glb` | 1024 chart-harmonization GLB (`3d8747...`) | **SUPERSEDED / REJECTED** | Reintroduced white feet and blur; wrong source baseline. |
| `C:\AI\panda_support_local_2048\candidate_2048\panda_face_axis_corrected_2048.glb` | mixed/older atlas route | **SUPERSEDED / REJECTED** | Visual quality regressed; do not use. |
| `C:\AI\panda_support_local_2048\face_recenter_latest_diagnostic_only\face_recentered_latest_2048.glb` | exact 2048 baseline (`1c4bec...`) | **DIAGNOSTIC ONLY / UNACCEPTED** | Face remap preserves non-face texels, but three-quarter alignment still requires camera-aware verification. |

The latest diagnostic receipt records the measured front-view offset as
`(-20,-7)` render pixels (feature centroid `(111,66)`, hood/body axis
`(131,73)`) and an attempted `(+20,+7)` atlas remap. This is evidence for the
registration problem, not a production acceptance claim.

## Rules for future work

1. Start comparisons from the **Current baseline** row only.
2. Never overwrite or relabel the baseline when testing a correction.
3. A face-only correction must preserve the baseline's non-face atlas bytes,
   geometry, UVs, and provenance, and must be rendered at the marked
   three-quarter angle before acceptance.
4. Any candidate that changes feet, rear, bottom, or overall sharpness is
   rejected as a face-fix candidate, even if the front face moves.
5. `DIAGNOSTIC_ONLY` means review-only; it is not production-ready.

## Current terminal status

- `PANDA_FACE_REGISTRATION`: **CONFIRMED DEFECT**
- `PANDA_FACE_CORRECTION`: **NOT_ACCEPTED**
- `PANDA_LATEST_GOOD_BASELINE`: **PRESERVED**
- `PANDA_PRODUCTION_READY`: **NO**
