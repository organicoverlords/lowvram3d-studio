# Tactical red panda sanitized 4/5-step textured comparison

This was a CPU-only comparison. No GPU generation ran, and the original
`raw_4step_retry2.glb` and `raw_5step.glb` files were never passed through the production Blender
path.

The complete external report is:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\panda_texture_comparison\panda_4_5_comparison_report.json`

The labelled contact sheet is:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\panda_texture_comparison\panda_4_5_v7_contact_sheet.png`

## Inputs and outputs

| Candidate | Sanitized input SHA-256 | Textured output SHA-256 | Fresh-import polygons | Texture coverage |
|---|---|---|---:|---:|
| 5-step | `1170cff8c1e29b6ab210cac1f8100ca160575e692063c7de0f478db0d7597ff2` | `3ce51b096e28ad3527421703d6f0cb953ef86ed8cc7a5e66aacd96ba7bd30f57` | 644,412 | 83.6% observed / 16.4% synthesized |
| 4-step | `2ac94a09384679dca5720a6b87cecf236643cd9f9bebf69ef57c4b39e735514a` | `737595b202d9cfada0a6aa57b35fab3b199a90c95bea31c236292eb98ce2dbd4` | 639,000 | 84.4% observed / 15.6% synthesized |

Both candidates passed fresh import, UV presence, material binding, packed readable base-colour,
finite/non-black/non-constant texture, and matched front/three-quarter/side/rear rendering.

## Visual decision

The rear render exposes the same failure in both new candidates and in the repaired v7 reference:
the source-facing red-panda face is visibly projected onto the rear surface. The clay geometry does
not have a rear face, so this is a texture/UV projection failure rather than a new geometry defect.

- `PANDA_5STEP_FRONT_IDENTITY=PROVEN`
- `PANDA_4STEP_FRONT_IDENTITY=PROVEN`
- `PANDA_5STEP_REAR_FACE_PROJECTION=REJECTED`
- `PANDA_4STEP_REAR_FACE_PROJECTION=REJECTED`
- `PANDA_5STEP_VS_V7=NOT_PROVEN_BETTER`
- `PANDA_4STEP_VS_V7=NOT_PROVEN_BETTER`
- `PANDA_COMPARISON_WINNER=NOT_PROVEN`
- `REPAIRED_V7_RETAINED=PROVEN_REFERENCE_ONLY`

Neither candidate replaces repaired v7. The post-decoder sanitizer remains mandatory for future
Mini Turbo generation. This comparison does not change the general step-count policy.
