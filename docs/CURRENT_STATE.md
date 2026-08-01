# Current production state

**Updated:** 2026-08-01

This file is the short source of truth for active high-resolution pipeline work. Older session notes remain useful history, but this document takes precedence when they conflict.

## Reference success: TurboBird

TurboBird is the user's first real successful result and the current qualitative reference. The user considers it a good result, and it proves that the practical route can produce a useful asset on the target Windows 10 / GTX 1660 SUPER 6 GB machine.

Do not redesign the pipeline so aggressively that it can no longer preserve a TurboBird-class result. Validation must catch demonstrated damage, not reject usable geometry because a classifier or proxy metric is uncertain.

The exact canonical TurboBird receipt, source identity and comparison statistics still need to be consolidated. Until then, TurboBird is a **user-validated qualitative reference**, not a fully reproduced quantitative benchmark.

## Canonical shaman source

- path: `C:\Users\Lauri\Downloads\ChatGPT Image 29.7.2026 klo 20.00.45.png`
- dimensions: `1122x1402`
- SHA-256: `4d23adc758c5b700dd29939e37c043ce61919792b566bdcf13f58b1409d6cf6f`

The prior recorded hash `eccef854f816f446ce2bf2e08559df519adff223b425c1dccc1c0a9b299f13f6` is rejected because it did not match that exact file.

## Earlier full-route lesson

GitHub Actions run `30657995651` proved that the self-hosted worker, PNG entrypoint, geometry generation and ingest worked. It produced a valid 55,684-face mesh with boundary edges `0 -> 0`. The run stopped because component audit treated two tiny `AUDIT_REQUIRED` fragments as fatal.

`workers/component_audit_cleanup.py` now follows a preserve-first policy: ambiguity-only failures with safe topology preserve the original generated GLB byte-for-byte, record `manual_review_required=true`, and continue. Real process errors, changed main geometry and increased boundary/non-manifold edges still fail closed.

## Current high-detail final-pipeline state

A later geometry-first run produced and preserved:

- high master: `final-pipeline/high/shaman_high_master.glb`
- SHA-256: `db1818a3b804c64e38767fde3235dc388621f08a8e8dd3da9ffc3b9692c654d4`
- bytes: `24,237,656`
- clean master: `1,093,124` triangles and `59` connected components
- confirmed cleanup removed `28` of `90` welded components and `0.196%` of faces

The measured run reports that cords, bowl charm, leaf pendants, hollow pod, lantern, far-right pendant, staff ring/charm, robe fringe and strips survive cleanup. Visual review remains required.

### LOD chain

| LOD | triangles | components | boundary edges |
|---:|---:|---:|---:|
| 0 | 219,967 | 59 | 100 |
| 1 | 89,965 | 59 | 109 |
| 2 | 39,961 | 59 | 115 |
| 3 | 14,951 | 59 | 151 |

Constant component count is evidence that the LOD route did not fragment cords or fuse ornaments into the main body.

## Current UV correction

The provisional 4096 UV run reported finite, in-bounds UVs, zero degenerate triangles and approximately `47.99%` raster-estimated utilisation. It then stopped because an analytic-area / raster-covered ratio of `1.1016` was interpreted as roughly 10% real overlap.

That interpretation is invalid. The repository already proved that:

- raster collision counts include ordinary shared chart edges and sub-texel collisions;
- analytic triangle area divided by rasterised coverage conflates boundary rounding with genuine overlap.

`src/lowvram3d/uv_overlap.py` is the canonical detector. It uses exact convex clipping and counts only positive-area triangle intersections.

Commit `18b9b29fca28a99cae127e2e88021aec1472d660` changes `blender/final_pipeline_uv.py` to:

- remove the slow 4096x4096 Python raster loop over approximately 220,000 triangles;
- compute utilisation analytically;
- use exact positive-area intersection as the only bake-safety overlap gate;
- permit at most `1.0` texel-equivalent of exact overlap;
- treat low utilisation as a packing warning, not proof of corrupt UVs.

The exact next action is a **Stage-4-only rerun** against the existing `final-pipeline/game/shaman_lod0.glb`. Do not regenerate, clean or decimate geometry. If exact overlap is within budget, promote the UV GLB and start Stage 5 baking. Only if exact overlap really exceeds budget should the unwrap or atlas strategy change.

See `evidence/SHAMAN_FINAL_PIPELINE_GEOMETRY_UV_20260801.md` for the measured handoff.

## Staff ring through-hole: repaired and proven

**Updated:** 2026-08-02

Visual review established that the circular feature on the staff was a **closed recess**, not a
through-hole. That is now fixed and proven, by a **localized** repair run on the workstation.

The earlier approach was rejected. `nearest_object()` does not localize anything: the fused main
surface carries about `98.97%` of faces, so cutting "one object" is still a Boolean against roughly
`1.09M` triangles. The current route isolates the ring region first.

Route, as executed:

1. detect the ring from the front silhouette and size a patch box around it;
2. select only faces intersecting that box, grown by `2` adjacency rings;
3. `bpy.ops.mesh.separate` that patch out, preserving world coordinates and object transforms;
4. apply the cylindrical EXACT Boolean **to the patch alone**;
5. restore corner splits, because the EXACT solver welds its own output;
6. join the patch back; no vertex welding is performed (see below);
7. prove nothing outside the repair box moved.

### Measured result

| item | value |
|---|---|
| input (unchanged) | `pipeline-v2-validation\shaman_v2_validation\state\CLEAN\proven\shaman_v2_validation_stance_clean.glb` |
| input SHA-256 | `2f712b49a88a39cb10fb08e6bfb08becef2025b153ce87103e86fde97dfb8c80` |
| output | `C:\AI\LowVRAM3D-benchmarks\shaman-staff-hole-repair\local-20260802\shaman_v2_staff_hole_repaired.glb` |
| output SHA-256 | `6b085579488fb9b3b91da592bd030a512f754a6c7c21691ab5715b69ce0f7a13` |
| triangles | `1,086,965` -> `1,083,588` (delta `-3,377`) |
| patch size | `29,608` faces = `2.724%` of `1,086,965` (limit `5%`) |
| ring thickness / hole radius | `0.14198` / `0.06019` world units |
| stage times | import `7.17s`, preflight detect `4.09s`, localized Boolean `6.73s` |

### Validation gates, all passed

| gate | threshold | measured |
|---|---|---|
| centre foreground ratio | `<= 0.08` | `0.0000` |
| annulus foreground ratio | `>= 0.42` | `1.0000` |
| axial open-ray fraction | `>= 0.95` | `1.0000` |
| oblique open-ray fraction | `>= 0.80` | `1.0000` (all 8 azimuths) |
| backlit centre ratio | `>= 0.60` | `1.0000` |
| direct ray blocked | must be false | `false` |
| geometry outside repair box | unchanged | `539,998` unique positions identical, no collapse |
| fresh-process import | `> 1M` triangles | `1,083,588` over `1` object |

### Two measurement traps that produced false failures

Both were defects in the *test*, not the geometry, and both are recorded so they are not
reintroduced:

1. **Whole-model silhouette cannot gate an oblique view.** The shaman's body sits behind the
   foreshortened ring, so the centre reads as filled even through an open hole. Openness is now
   measured with rays that only count hits **inside the repair box**.
2. **The ray bundle must start at the ring, not the model centre.** The ring sits at `y ~ -0.174`
   while the model centre is `y ~ 0.0033`. A bundle launched from the model centre drifts about
   `0.043` sideways before reaching the ring - most of the `0.0602` hole radius - and reports a
   false blockage. The oblique tilt is also derived from measured geometry: a centre ray only
   clears while `tan(theta) < hole_radius / thickness`, here `22.97deg`, so the probe uses `13.78deg`.

### Why no vertex welding

The source is a **triangle soup**: `3,258,048` vertices for `1,086,965` triangles, three per
triangle, so patch and remainder share no vertex and there is no seam to stitch. An earlier weld
pass collapsed `56,446` coincident vertices and would have destroyed corner splits, and with them
UV and normal seams, across the patch. Welding now runs only when a source genuinely shares
vertices (`corner_split_ratio < 0.99`).

Evidence: `C:\AI\LowVRAM3D-benchmarks\shaman-staff-hole-repair\local-20260802\`
- `staff_hole_before_after_sheet.png` - four-row before/after sheet
- `evidence\{before,after}_staff_front.png` - front close-up
- `evidence\{before,after}_staff_oblique.png` - oblique depth view; the before frame shows the
  recess floor directly, the after frame shows the bore wall
- `evidence\{before,after}_staff_backlit.png` - backlit silhouette
- `evidence\{before,after}_full_character.png` - full-character comparison
- `staff_hole_report.json`, `repair.log`

The head, mask, antlers, cords, pendants, robe, legs and other staff ornaments are unchanged, by
coordinate proof and by full-character render.

## Semantic part separation

Stage 2 remains incomplete. Loose-part splitting cannot isolate the staff because the main connected surface contains about `98.97%` of faces and spans the body, staff, antlers, cords and ornaments. Movable-part extraction requires semantic segmentation or authored cuts. It is not required before validating the current fused-character LOD and UV route.

## Remaining proof boundary

- The repaired-baseline chain now continues from
  `shaman-staff-hole-repair\local-20260802\shaman_v2_staff_hole_repaired.glb`, **not** from the
  inferior `1.47M` comparison candidate.
- UV, texture, rigging, animation and Unreal export have not started on the staff-hole-repaired
  baseline.
- The repaired GLB is `116,900,380` bytes against a `91,237,596` byte input. The triangle count
  fell, so this is an export-encoding difference, not added geometry. It has not been traced.
- The staff-hole workflow is dispatch-only and has never completed in CI; the proven result is the
  local run.
- Exact UV overlap has not yet been rerun on the current LOD0.
- Baking, texturing, A-pose, rigging, animation and Unreal export have not started for this high-detail chain.
- The model is genuinely shallow in depth (`Y` extent about `0.595` versus `Z` extent about `1.981`), a single-view reconstruction limitation that texturing and rigging cannot repair.
- The A-pose normalizer is not yet connected to the main route or visually validated in Blender/Unreal.
- Final GLBs, reports and renders must be inspected before any production claim.

## Current workflow branches

- `fix/highres-production-pipeline-20260731`: high-resolution geometry, benchmark ordering, component audit, measured LOD selection and pose-policy work.
- `infra/windows-self-hosted-runner-20260731`: trusted Windows worker and current shaman execution, stacked through PR #3.
- `improve/stage-provenance-20260731`: stage artifact provenance, stacked separately through PR #2.

Do not merge into `main` until required local evidence is complete.

## Production rules

1. Preserve source and generated masters byte-for-byte.
2. Hard-fail on demonstrated corruption, topology regression, missing inputs, invalid exports or subprocess failure.
3. Do not hard-fail solely because a classifier or proxy metric is uncertain.
4. Exact geometry/UV measurements outrank raster proxies.
5. Promote final deliverables only after fresh-process re-import, renders and receipt validation.
6. Rigging or pose normalization must never overwrite a valid unrigged result when they fail.
