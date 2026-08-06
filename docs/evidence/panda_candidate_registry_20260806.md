# Panda candidate registry — 2026-08-06

This file is the single index for the current panda visual candidates. It exists to prevent diagnostic outputs from being mistaken for the latest-good baseline.

## Current baseline (keep)

| Role | Path | SHA256 | Status |
|---|---|---|---|
| Latest-good 2048 GLB | `C:\AI\panda_support_local_2048\candidate_2048\panda_atlas_support_fixed_2048.glb` | `1c4bec32a31ce90766f12f095e2d3c9c7e57e0b34e4f8ffc79c9a29ea8e7a817` | **BASELINE — use for comparisons** |
| Latest-good 2048 atlas | `C:\AI\panda_support_local_2048\candidate_2048\atlas_2048_nearest.png` | `88ee2bf03bfe27b6ee28720734c6abd94fc03242f27a2f33529bc3db0f08383e` | **BASELINE INPUT** |

Core ownership implementation is present in `workers/face_surface_ownership.py`, `workers/face_patch_texture.py`, and `tests/test_face_surface_ownership_core.py`. No candidate is promoted until the local worker renders it.

- `PANDA_FACE_REGISTRATION`: **CONFIRMED DEFECT**
- `PANDA_FACE_CORRECTION`: **IMPLEMENTATION ADDED; REAL-ASSET RUN PENDING**
- `PANDA_PRODUCTION_READY`: **NO**
