# Panda front-projection report — 2026-08-05

## Decision

The new direct-original-front projections are rejected by manual visual review. The sharper PNG source has more pixels, but it is a different framing/source variant from the mesh-registration reference, so the projected front is worse than the preserved golden multiview result. It is diagnostic only and is not promoted.

```text
PANDA_GOLDEN_MULTIVIEW_BASELINE = PRESERVED
PANDA_FRONT_QUALITY_NOT_WORSE_THAN_GOLDEN = REJECTED
PANDA_REAR_DUPLICATED_FACE = 0 in the inspected high-resolution diagnostic
PANDA_SIDE_WRAPPED_FACE = 0 in the inspected high-resolution diagnostic
PANDA_BOTTOM_WHITE_INVALID_TEXELS = PRESENT
PANDA_FULL_360_PRODUCTION = NOT_PROVEN
HIGHRES_ORIGINAL_FRONT_PROJECTION = REJECTED
```

## Preserved visual baseline

The historical golden control-bundle render remains the comparison reference:

- front: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\forensics\latest_panda_control_views\raw1_front_textured.png`
- rear: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\forensics\latest_panda_control_views\raw3_rear_textured.png`
- bottom: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\forensics\latest_panda_control_views\raw5_bottom_textured.png`
- golden GLB SHA256: `8F24B7D1E3245CD96B5CDC40A350DF1A33CD2103502CC0D9E4903F865426BC17`
- golden atlas SHA256: `8B834F3338C5569F639162D0B697CD2CFABDDA7BDD9483F7463A916F8EAFCF9F`

The golden result has the coherent front identity, clean rear, and valid side profiles. Its bottom still contains white invalid/missing regions and therefore remains a visual baseline, not a full-360 acceptance.

## Rejected high-resolution diagnostic

Source used:

- `C:\AI\LowVRAM3D-benchmarks\images\red_panda_character.png`
- dimensions: `1117 x 1409`
- SHA256: `FBDA3719B42366477F77E94D87CB1D32A5BF6ADC4B6F29782194CC154471159B`

Command:

```text
py -3 workers/injective_atlas_texture.py --mesh C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\pipeline_v2_injective_20260804\state\UV\proven\tactical_red_panda_scout_rewrapped.glb --bundle C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\sd21_cpu_controls_384_v9_upright_raw --views-receipt C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\sd21_upright_384x20_20260803\inference_receipt.json --original-front C:\AI\LowVRAM3D-benchmarks\images\red_panda_character.png --output-dir C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\panda\original_front_highres_4096 --atlas-size 4096 --output-basename panda_original_front_highres --direct-only
```

Result:

- GLB: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\panda\original_front_highres_4096\panda_original_front_highres_textured.glb`
- GLB SHA256: `C1B2C5C9910E85E8AC0C96D2A4379075D96C0B92EF8D0EB4D41143AC46054240`
- atlas: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\panda\original_front_highres_4096\panda_original_front_highres_basecolor.png`
- atlas SHA256: `4CA001515D6AA6EBB31397B969132D7C062986658AF80320D45D87139B4F0A8C`
- projected front: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\panda\original_front_highres_4096\views\raw1_front_textured.png`
- rear: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\panda\original_front_highres_4096\views\raw3_rear_textured.png`
- left: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\panda\original_front_highres_4096\views\raw0_left_textured.png`
- right: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\panda\original_front_highres_4096\views\raw2_right_textured.png`
- bottom: `C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\panda\original_front_highres_4096\views\raw5_bottom_textured.png`
- observed texels: `5,737,690`
- donated texels: `1,562,150`
- unresolved texels: `0` (material-prior fallback was used; this is not a visual acceptance)

Manual result: front quality is worse than the golden baseline; rear and sides show no duplicated facial mask in the inspected views; bottom white invalid region remains.

## Code/test note

The narrow compatibility change keeps cached triangle-ID comparison in per-texel provenance but does not discard otherwise valid legacy generated-view samples solely on that diagnostic mismatch. The existing focused suite passed: `41 passed`.

No generated neural views, camera mappings, geometry, or UV master were changed for this diagnostic.
