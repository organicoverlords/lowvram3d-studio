# Panda high-resolution texture handoff — 2026-08-05

## Current classification

```text
GOLDEN_PANDA_PRESERVED = PROVEN
PANDA_HIGHRES_IS_AUTHORITATIVE_ORIGINAL = PROVEN
PANDA_HIGHRES_TO_CONDITIONING_TRANSFORM = PROVEN
CURRENT_2048_TEXTURE = REJECTED_MANUAL_VISUAL_REVIEW
FRONT_FACE_SOURCE_AUTHORITY = ORIGINAL_SOURCE_REQUIRED
WHITE_BACKGROUND_LEAK = PARTIALLY_DIAGNOSED
CAMERA_CONTRACT = NOT_PROVEN
FULL_360_PRODUCTION = NOT_PROVEN
CANONICAL_STAGE_INTEGRATION = BLOCKED
```

## Visual prototype update — 2026-08-05

An explicitly experimental visual-only candidate was generated without changing the
geometry, UV master, camera contract, or provenance receipts. The candidate uses the
existing six generated views plus the authoritative high-resolution source front.

- 512 prototype: `C:\AI\LowVRAM3D-benchmarks\production\panda_visual_prototype_experimental_20260805\smooth\panda_visual_prototype_512_smooth_textured.glb`
- Current 1024 fidelity candidate: `C:\AI\LowVRAM3D-benchmarks\production\panda_visual_prototype_experimental_20260805\fidelity_pass_1024\panda_fidelity_pass_1024_textured.glb`
- Current localized face-alignment candidate: `C:\AI\LowVRAM3D-benchmarks\production\panda_visual_prototype_experimental_20260805\fidelity_pass_1024_face_landmark_right\panda_face_landmark_right_1024_textured.glb`
- Native Blender wrapper for inspection: same directory, `panda_face_landmark_right_1024_textured.blend`
- Turntable sheet: `...\fidelity_pass_1024\renders_unlit\turntable_360_sheet.png`

Visual result: the front identity is recognizable and the rear remains face-free, but the
body/fur is still noisy and the face-to-muzzle registration needs further landmark work.
These candidates are visual references only and are not production evidence or promotion
artifacts. The source online model targets remain appearance references only.

Current contract status is unchanged: 256 has two hard opposing-normal pixel ownership
errors; 384 is replay-clean; no production 512 proof or downstream promotion is claimed.

## Fidelity pass update — generated residual attenuation

`workers/injective_atlas_texture.py` now accepts `--generated-detail-scale` (default `0.45`).
It attenuates only generated-view high-frequency residuals before fusion; the authoritative
original-front detail path is not attenuated. This directly reduces diffusion speckle without
blurring protected source detail.

The bounded visual candidate used `--generated-detail-scale 0.35` at atlas size 1024:

`C:\AI\LowVRAM3D-benchmarks\production\panda_visual_prototype_experimental_20260805\fidelity_pass_1024_residual_attenuated`

Fresh renders show reduced body speckle, a recognizable centered front face, and no face on
the rear. This remains an experimental visual candidate, not a production acceptance claim.

## Repository state

- Repository: `C:\Users\Lauri\Desktop\lowvram3d-two-character-production-20260804`
- Branch: `production/two-character-models-20260804`
- HEAD: `aea780c12ef669361ae36f604c7920ee3886c181`
- The worktree was already intentionally dirty. Do not reset, clean, stash, or overwrite unrelated changes.
- Recovery diff for the bounded run: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\recovery\pre-panda-highres-registration-fix.patch`
- Recovery diff SHA256: `1904809E5FDED261EC70F0577DE07869BA43CE428D3433E50A08B1C8700FCB49`

## Immutable visual baseline

- Golden GLB SHA256: `8F24B7D1E3245CD96B5CDC40A350DF1A33CD2103502CC0D9E4903F865426BC17`
- Golden atlas SHA256: `8B834F3338C5569F639162D0B697CD2CFABDDA7BDD9483F7463A916F8EAFCF9F`
- Golden controls/renders: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\forensics\latest_panda_control_views`
- The golden front is coherent and the rear is clean. Its known remaining defect is the white bottom view; it must not be overwritten.

## Authoritative source proof

- Source: `C:\AI\LowVRAM3D-benchmarks\images\red_panda_character.png`
- Dimensions: `1117x1409`
- SHA256: `FBDA3719B42366477F77E94D87CB1D32A5BF6ADC4B6F29782194CC154471159B`
- It is the same panda design as the historical low-resolution source, not an alternate character.
- High-resolution → low-resolution partial-affine SIFT fit: 538/637 inliers, median error 0.443 px, foreground IoU 0.9714.
- High-resolution → conditioning transform: 633/656 inliers, median error 0.096 px, subject-mask IoU 0.9710.
- Camera registration solution: scale `1.02`, dx `-7`, dy `-1`, foreground IoU `0.8812`.
- Forensics: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\panda_original_highres_authority_20260805\forensics`

## Buffers and mesh

- Final injective UV mesh: `C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\pipeline_v2_injective_20260804\state\UV\proven\tactical_red_panda_scout_rewrapped.glb`
- Final mesh SHA256: `950343DD7FF76877CE6ADB83D6D4A80A8D123E7AAFB397B566A762D454A9A5F1`
- Exact final-mesh controls at 1024: `...\panda_original_highres_authority_20260805\buffers\exact_mesh_controls_1024`
- Semantic-remapped 1024 bundle used by projection: `...\buffers\exact_mesh_controls_semantic_1024`
- Geometry and UV were not regenerated or changed.

## Latest candidate

Output directory:

`C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\panda_original_highres_authority_20260805\projection\diagnostic_2048_exact1024_original_boundary_background_gate`

- GLB SHA256: `75D8B3DF4DF28B9CEF6C4C849CE0A92185F4C911D0E7992308826DBEDE46C477`
- Atlas SHA256: `55B2A9F8C7C5A28D4F565552A4725D1FC2AC82616B707B802B762018D789CFE4`
- Contact sheet: `...\renders\contact_sheet_b_semantic_corrected_textured.png`
- Observed texels: `940,179`
- Completion texels: `893,743`
- Unresolved: `0` (not a visual acceptance claim)
- Manual result: face identity is the best current high-resolution result and is sourced from the original, but the front remains mottled/glitchy and the bottom remains visibly white/invalid. Do not promote.

## Changes made in this bounded pass

1. Added deterministic source-transform forensics and bounded original-camera registration.
2. Added exact final-mesh 1024 control rebuild and semantic remapping without changing camera matrices.
3. Kept generated-view projection strict on exact triangle IDs.
4. Added a one-pixel, depth-compatible raster-boundary allowance only for the original-front projection; it is not used for generated side/rear evidence.
5. Added source foreground and near-white evidence rejection for generated images.
6. Removed white-source-canvas interpolation bleed by neutralising the source before warping.
7. Corrected the shared foreground mask so saturation noise alone cannot classify white canvas or ground shadow as subject.
8. Added `workers/validate_triangle_id_roundtrip.py`.

## What the diagnostics proved

- Strict 1024 ID matching rejects roughly 230k–257k candidate texels per view.
- A one-pixel neighbouring owner exists for about 90% of visible mismatches, confirming finite-raster boundary effects; broad neighbouring acceptance caused visible speckles and was rejected.
- Generated-view boundary acceptance is disabled and must remain disabled until a stronger contract is proven.
- The current face improvement comes from protected original-front texels, not from a successful full-texture solve.
- White pixels in the latest atlas are predominantly `ORIGINAL_SOURCE`/front provenance or generated bottom evidence; do not assume all are background after the first matte fix.

## Do not do next

- Do not run 4096.
- Do not regenerate geometry, UVs, controls, or neural views.
- Do not change camera semantics.
- Do not broaden boundary ID acceptance to generated views.
- Do not blur, median-fill, or paint over the face.
- Do not commit or push this rejected candidate as a production asset.

## Decisive next work

1. Inspect the registered original source matte and remove the connected ground-shadow/white-canvas component without removing white facial fur.
2. Render an original-only front diagnostic and verify that white texels map to face/hood source coordinates, never feet or weapon surfaces.
3. Keep the original face/front texels immutable; use existing generated views only for complementary side/rear/top/bottom surfaces.
4. Re-render front, rear, sides, and bottom from fresh import. Stop if any face-like rear artifact or front speckle returns.
5. Only after manual visual acceptance, package and commit the causal mask/provenance fix; preserve all rejected diagnostics as evidence.

## Atlas artifact-root repair update (2026-08-05)

The latest black peppering was isolated from neural/source quality using a
synthetic unique-triangle atlas. Clean Base Color-only and unlit probes still
sampled the magenta unowned sentinel, proving sparse UV coverage and continuous
renderer sampling—not diffusion noise—are the dominant cause. The coverage
candidate also carried `alphaMode=MASK`; that amplified transparent holes.

The focused code repair is committed and pushed on
`agent/panda-texture-artifact-root-fix-20260805` at `f1452c5`:

- projected atlas materials are rebuilt from scratch as opaque Base Color-only
  materials;
- atlas sampling uses a dedicated linear, clamp-to-edge sampler;
- owner-aware projection routes bind only triangles represented in the support
  mask;
- 18 focused tests pass.

Synthetic evidence remains blocked under the fixed UV master. A common-lattice
pass reduced sentinel hits from 7,902 to 1,581; area-weighted support did not
improve that; fallback masking removed zero-support triangles from atlas binding
but left 2,868 atlas-bound sentinel samples; the final strict safety mask left
203. These are preserved at:

- `C:\AI\panda_sampler_diag_v1\synthetic_sampler_isolation_report.json`
- `C:\AI\panda_lattice_diag_v1\synthetic_conservative_lattice_report.json`
- `C:\AI\panda_area_diag_v1\synthetic_area_weighted_sampler_report.json`
- `C:\AI\panda_bound_diag_v1\synthetic_bound_support_gate_report.json`
- `C:\AI\panda_safety_diag_v1\synthetic_safety_mask_gate_report.json`

The user approved a minimum-support UV/atlas redesign and one explicitly
diagnostic-only 1024 panda with neutral fallback materials. Neither may be
promoted until the redesigned UV candidate passes the synthetic gate and fresh
visual review. Existing golden and rejected candidates remain untouched.

### Approved UV redesign result (diagnostic only)

The one approved CPU xatlas rewrap completed with padding 8 at nominal 2048:

- Candidate: `C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\panda_uv_2048_padding8_candidate_20260805\tactical_red_panda_scout_uv2048_p8_candidate.glb`
- Candidate SHA256: `86955b43a47499d0d5d111e3195a741d80853c7bc21321270b34d88045851493`
- Report: `...\panda_uv_2048_padding8_candidate_20260805\uv_rewrap_report.json`
- xatlas: 0.0.11; 16,470 charts; packed 4664x4668; 644,348 triangles; 665,304 output vertices; 80,542 seam vertices added; strict interior overlap 0; injective true.
- Coverage audit: 190,179 exact-center support-zero triangles; 451,140 triangles with 1–4 center texels; 3,020 with 5–16; 9 above 16; 1,424 degenerate UV triangles.

This candidate preserves semantic triangle geometry/topology but does **not**
preserve raw position, normal, or index accessor hashes because xatlas seam
duplication/remapping changes the vertex/index buffers. It is therefore not an
authoritative mesh replacement. Luna is running one synthetic unique-triangle
gate against it as an isolated diagnostic; no panda texture or production
promotion is authorized from this candidate yet.

The synthetic gate then ran once against that candidate:

- Report: `C:\AI\panda_xatlas_gate_v1\synthetic_xatlas_gate_report.json`
- Contact sheet: `C:\AI\panda_xatlas_gate_v1\renders\basecolor_only_mip_linear_contact_sheet.png`
- Mip-linear magenta interior samples: left 2,280; front 2,764; right 2,408; rear 2,653; top 2,115; bottom 1,696 (13,916 total); black interiors 0.
- No-mip linear produced the same sentinel counts; nearest sampling produced tens of thousands of magenta hits per view.
- Direct V orientation remained correct; flip-Y was worse.

Result: `SYNTHETIC_GATE_BLOCKED`. The UV rewrap improves strict injectivity but
does not establish a reliable GPU sampling contract, and its raw mesh hashes
are different from the canonical asset. No real panda bake was run from it.

The follow-up native-packed diagnostic used the actual xatlas dimensions
(4664x4668) rather than shrinking to 2048:

- Report: `C:\AI\panda_xatlas_native_v1\native_synthetic_gate_report.json`
- Report SHA256: `33722690a0479105e4734a91c9088a906a0a40b7a5f9f6aade5e7ddf95fc6dca`
- Mip-linear magenta interiors: 10,052; no-mip/unlit 16,298; nearest 128,800; mip-nearest 150,943; black interiors 0.
- Direct V orientation remained correct.

Native resolution reduces but does not eliminate unowned sampling. The UV
redesign therefore remains `SYNTHETIC_GATE_BLOCKED`; no panda appearance or
proof promotion was performed.

### Sampler binding correction (diagnostic-only visual result)

The neutral-fallback candidate's vertical stretching was traced to one
deterministic difference: its atlas sampler used `minFilter=9729` (linear,
no mipmaps), while the source v4 GLB used `minFilter=9987`
(`LINEAR_MIPMAP_LINEAR`). TEXCOORD_0, atlas bytes, direct-V orientation, and
geometry/UV values were identical.

Restoring only that sampler field produced:

- GLB: `C:\AI\panda_diag_neutral_v1\panda_diag_neutral_sampler_v2.glb`
- GLB SHA256: `cab2357b6d8c46eb058de7589ac967efe247cb3347fe7a9dd12f1bac409c9f44`
- Contact sheet: `C:\AI\panda_diag_neutral_v1\renders_sampler_v2\contact_sheet_sampler_v2.png`
- Contact sheet SHA256: `a64a1f911e1e9aa01835feb88b8b3ed62099e35a5ab38def060a538b30ed5230`
- Receipt SHA256: `c548ca2ce7621c25a42ec335cc8ad7ce85d6cb3d48f571bc78900e8c168ceab0`

The stripes/stretching are materially reduced and the face is readable in all
six views. Residual speckle, unsupported neutral areas, and bottom white
patches remain, so this is still `DIAGNOSTIC_ONLY_NOT_PRODUCTION_READY`.

An independent binding audit found no further deterministic fix: TEXCOORD_0,
the embedded 2048 atlas bytes, texture/image indices, UV orientation, and
triangle partition all match the v4 source. The remaining bands correspond to
the 86,489 zero-support triangles in the 2048 owner mask. Painting them or
binding the atlas to every triangle would bypass the coverage gate, so no
additional cleanup was promoted.

### Final diagnostic cleanup variant

One bounded CPU cleanup was applied to the sampler-corrected candidate: a 5x5
median plus 0.35 low-frequency blend on 1,019,393 non-direct procedural or
material-prior texels. The protected 396,216 texels and all 940,179 direct
owners were untouched byte-for-byte; 14 provenance-invalid procedural white
texels were replaced.

- GLB: `C:\AI\panda_diag_neutral_v1\cleanup_v3\panda_diag_neutral_cleanup_v3.glb`
- GLB SHA256: `59ff5418146730e2741be70a6849b3d2c41df5c82c6b4718ec5b9a9e4d9caeb2`
- Atlas SHA256: `8cc8d96335e89a44017f7d9842bb757d3d8323b162c839cb774cc010eb93218b`
- Contact sheet: `C:\AI\panda_diag_neutral_v1\cleanup_v3\renders\contact_sheet_cleanup_v3.png`
- Contact sheet SHA256: `2c45e1f8983fdbb4b5c7ce0ae7dfcf87282d93bd83da2fa4a1d6b942690b9de4`
- Receipt SHA256: `2c16b3f6f5e949874402a410b0573f736632c05ca7ac69be13be3845a7a40dfa`

Speckle is modestly reduced and the sampler stripes remain absent, but
unsupported neutral areas and bottom gaps remain. Status stays
`DIAGNOSTIC_ONLY_NOT_PRODUCTION_READY`; no further cleanup loop is authorized.

### Coverage-bypass visual comparison (not production evidence)

One comparison bound the existing atlas material to both mesh primitives,
changing only material binding `[1,2]` to `[1,1]`. Geometry, UVs, sampler,
atlas bytes, provenance, and proof receipts were untouched.

- GLB: `C:\AI\panda_diag_neutral_v1\coverage_bypass_diagnostic_only\panda_coverage_bypass_diagnostic_only.glb`
- GLB SHA256: `2241f1c4205159e4b1dba04bc22180cd8ba308f21fe3cccc1c6575238a2f7269`
- Contact sheet: `C:\AI\panda_diag_neutral_v1\coverage_bypass_diagnostic_only\renders\contact_sheet_coverage_bypass_9view_diagnostic_only.png`
- Contact sheet SHA256: `3c524643e69bf4bf71de041f91a4a1d206ea068ff4c43eb13645ca57b6b059a9`
- Receipt SHA256: `46c73fbb8ec5c84a014c28cc077ca47ad67f8c6d052b4b94487eaede069654d1`

This removes the explicit neutral strips and gives the clearest front/side/
rear diagnostic, but deliberately bypasses the 86,489-triangle support mask.
Unsupported atlas samples, speckle, and bottom white patches remain. Status:
`COVERAGE_BYPASS_DIAGNOSTIC_ONLY`, never production evidence.

### Chart-local denoise diagnostic

One final CPU-only pass smoothed mutable atlas texels using a same-triangle
proxy for chart locality (no serialized chart-ID array was available). It used
one 3x3 median plus 0.35 blend, changed 191,674 mutable pixels, and preserved
the protected 396,216 and direct 940,179 texels byte-exactly.

- GLB: `C:\AI\panda_diag_neutral_v1\chart_local_denoise_diagnostic_only\panda_chart_local_denoise_diagnostic_only.glb`
- GLB SHA256: `cbf1f77e5094bd55d05c49c0cf8c1bc870b97184d1fdf1976331efb259ebcc98`
- Contact sheet: `C:\AI\panda_diag_neutral_v1\chart_local_denoise_diagnostic_only\renders\contact_sheet_chart_local_denoise_9view_diagnostic_only.png`
- Contact sheet SHA256: `8f487f88b3b857dc17af248ea9aaa7d6d20dde5c18528c55f31e285783989bb9`
- Receipt SHA256: `53e02a8acb9288aa3bd786e945a33a8bbc169dbb3d84467d63499716d82e98ae`

The result modestly reduces mutable speckle while preserving the face, but
the bottom white/support defects remain. Status:
`CHART_LOCAL_DENOISE_DIAGNOSTIC_ONLY`; no further appearance loop is planned.
