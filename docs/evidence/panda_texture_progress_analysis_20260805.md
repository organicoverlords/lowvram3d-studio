# Panda texture progress analysis — 2026-08-05

## Executive conclusion

The work has produced useful diagnostics and one modest visual improvement,
but not a production-ready textured panda. The protected original front is
stable and recognizable. The remaining quality failure is concentrated in
mutable side/rear/bottom atlas content: low-frequency source disagreement,
pixelation, and direct white/gray underside evidence.

## What was proven

- The clean projected material is not inheriting normal, occlusion, emissive,
  or other non-basecolor slots. `clean_pbr_material()` constructs a fresh
  material and `bind_texture()` uses it for atlas/neutral materials.
- The sampler and UV orientation are deterministic; the synthetic atlas tests
  show the dominant issue is sparse/subpixel atlas support, not stale PBR
  shading.
- The canonical mesh, UVs, triangle order, and proof inputs remained unchanged
  in the accepted diagnostic passes.
- The downloaded online models were identified. `besgu.glb` is the red-panda
  appearance target by mission role; `20260730004905_ab3519e2.glb` is explicitly
  a fox-like scout and is not a panda target.

## What materially improved

The surface-space completion diagnostic removed most neutral fallback strips
from the rasterized surface and made front/left/right/rear views cleaner than
cleanup-v3. The chart-local material harmonization pass then reduced broad
side/rear tonal variation modestly while preserving the protected face and
direct evidence.

Latest diagnostic:

- GLB: `C:\AI\panda_diag_neutral_v1\chart_local_harmonization_diagnostic_only\chart_local_harmonization_canonical_mesh_diagnostic_only.glb`
- Contact sheet: `C:\AI\panda_diag_neutral_v1\chart_local_harmonization_diagnostic_only\renders\contact_sheet_chart_local_harmonization_9view.png`
- GLB SHA256: `3d87476140967d1adfd86b05d4b8a838dc9b997c42f07ee051e3aa3ecfb654a4`

## What did not improve

- Closeups remain visibly mottled and pixelated.
- The underside still contains bright white/gray structures outside the narrow
  invalid-white correction class.
- The Besgu nearest-surface palette reference made side/rear appearance worse
  through broad gray/brown overlays and was rejected.
- Canonical-mesh transfer from the alternate surface-completion result did not
  change the visible defect pattern.

## Root-cause assessment

The dominant defect is not a single stale-material bug. It is the combination
of inadequate canonical UV support for many small triangles, sparse ownership
and gutter behavior, and conflicting low-frequency evidence on mutable charts.
The evidence supports surface-space/material-aware harmonization; repeated
global denoise, global palette transfer, and unrestricted atlas binding do not
solve it.

## Current decision

`PANDA_VISUAL_ACCEPTANCE = REJECTED`

The latest GLB remains diagnostic-only. Do not promote it, overwrite the
golden panda, or claim production readiness. The branch contains the focused
material-contract fixes, coverage-mask guard, diagnostic worker, and evidence
documentation; unrelated OpenCode/script changes remain uncommitted and
preserved.

## Agent allocation

- Luna: visual inspection, bounded CPU candidates, and render/contact-sheet QA.
- NVIDIA Nemotron: requested for non-vision code analysis; the large model hit
  its local request cap before returning a recommendation.
- DeepSeek Zen: OpenCode transport is healthy, but the free model is currently
  rate-limited; no visual work was delegated to it.

