# Tactical red panda sanitized 4/5-step projection-gate comparison

This comparison was CPU/Blender-only. No Mini Turbo generation ran, and the
original `raw_4step_retry2.glb` and `raw_5step.glb` files were not modified or
passed through the production Blender path.

The preserved sanitized inputs were re-textured in separate runs using the
same cleaned geometry, UVs, atlas resolution, source image, and export
settings. The external report is:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\panda_texture_comparison\panda_4_5_projection_fixed_comparison_report.json`

The visual proof contact sheet is:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\panda_texture_comparison\panda_4_5_v7_projection_fixed_contact_sheet.png`

## Shared projection correction

The four-view bundle contains `front`, `right`, `back`, and `left`, but its
metadata marks only `front` as a real semantic source. The other three views
are mirrored fallback views with zero semantic confidence. They remain
available for fill policy bookkeeping but are barred from source-pixel
projection; otherwise a mirrored face can be copied onto the rear.

The projector now requires both the depth/occlusion visibility mask and a
finite front-facing normal (`normal dot view > 0.15`) before sampling or
blending a source pixel. It also records a per-polygon observation mask that
is independent of overlapping planar UV pixels. Blender assigns every
unobserved polygon to `RasterNeutralSynthesis`, so unseen rear geometry cannot
sample or wrap front-face pixels from the shared atlas.

## Fixed candidates

| Candidate | Sanitized input SHA-256 | Fixed textured GLB SHA-256 | Polygons | Observed polygons | Neutral polygons | Fresh import | Rear face |
|---|---|---|---:|---:|---:|---|---|
| 5-step | `1170cff8c1e29b6ab210cac1f8100ca160575e692063c7de0f478db0d7597ff2` | `296e274db799efb86a0e6c1984ce9fca46f32ece6ab28b061eda00658ac94e0e` | 644,412 | 92,388 | 552,024 | PROVEN | PROVEN_ABSENT |
| 4-step | `2ac94a09384679dca5720a6b87cecf236643cd9f9bebf69ef57c4b39e735514a` | `78d2fecf8f6243ac34e98ed2854720021f33b6d1ba5e2cb03043d531cf9a81d1` | 639,000 | 92,970 | 546,030 | PROVEN | PROVEN_ABSENT |

Both fixed outputs have packed 512x512 base-colour textures, valid UVs, two
materials, no armature/actions, readable front face/gear/tail colour, and
fresh front, three-quarter, side, and rear renders. The rear renders show
neutral dark synthesis and no face. Rear regions are intentionally not claimed
as source-observed.

## Historical invalid outputs

The previous textured GLBs remain preserved as rejected evidence:

- `panda_texture_comparison_5step\tactical_red_panda_scout_5step_textured.glb` — `REJECTED_REAR_FACE_PROJECTION`
- `panda_texture_comparison_4step\tactical_red_panda_scout_4step_textured.glb` — `REJECTED_REAR_FACE_PROJECTION`

## Classifications

- `PANDA_5STEP_PROJECTION_GATE=PROVEN`
- `PANDA_4STEP_PROJECTION_GATE=PROVEN`
- `PANDA_5STEP_REAR_FACE_PROJECTION=PROVEN_ABSENT`
- `PANDA_4STEP_REAR_FACE_PROJECTION=PROVEN_ABSENT`
- `PANDA_5STEP_FIXED_TEXTURED_BASELINE=PROVEN_WITH_NEUTRAL_UNSEEN_REAR`
- `PANDA_4STEP_FIXED_TEXTURED_BASELINE=PROVEN_WITH_NEUTRAL_UNSEEN_REAR`
- `PANDA_FIXED_VISUAL_WINNER=NOT_PROVEN`
- `REPAIRED_V7=NOT_REPLACED`

The repaired v7 comparison remains reference-only; this task did not alter it
or regenerate geometry.
