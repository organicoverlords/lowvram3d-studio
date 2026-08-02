# Unified Pipeline V2 integration

Branch: `integration/unified-pipeline-v2-20260802`

Base: `3971a1ba9b17d46d98f2e92eba9e97efdbe61d36`

The older `infra/windows-self-hosted-runner-20260731` head was reviewed as an
ancestor diff only. No merge commit and no wholesale old-worker replacement
were performed.

## Integrated contract

`workers/run_unified_pipeline_v2.py` exposes the canonical static stage range:

`INGEST -> GENERATE -> GEOMETRY_QA -> CLEAN -> LOD -> UV -> BAKE -> TEXTURE -> TEXTURE_QA -> EXPORT_QA`

It composes the current `Pipeline` state machine, current production workers,
current repair policy, current Mini Turbo generator/sanitizer, and a fresh
Blender-process `EXPORT_QA` boundary. Rigging and animation are absent from the
static baseline range. Existing geometry can be adopted at GENERATE, CLEAN, or
UV boundaries without regeneration; an already-proven textured GLB can be
adopted for export QA only.

## Tests and preserved-asset checks

- Integration and focused texture tests: `24 passed`.
- Panda Test A: the existing fixed 5-step textured GLB was adopted as a TEXTURE
  receipt and passed fresh Blender `EXPORT_QA`; no geometry or GPU work ran.
- Replaying the same Panda Test A command skipped `EXPORT_QA` because the input
  hash and passing receipt were unchanged.
- Shaman Test B: the existing clean shaman geometry was adopted at GENERATE;
  output hash remained `2f712b49a88a39cb10fb08e6bfb08becef2025b153ce87103e86fde97dfb8c80`.
- Test C resume/hash invalidation is covered by
  `tests/test_unified_pipeline_v2.py`: unchanged input skips, changed input
  reruns only the dependent fixture stage.
- CPU-only focused tests after the bounded LOD/xatlas hardening: `12 passed`.
- `blender/final_pipeline_lods.py` now accepts an asset prefix, uses a direct
  bmesh weld in background mode, records welded topology before/after
  decimation, and the production adapter rejects boundary or non-manifold
  regressions.
- `workers/uv_xatlas_isolated.py` runs preset A/B/C in separate child
  processes, with a 600-second default timeout, heartbeat reports, raw
  candidate checkpoints, array hashes, and promotion only after exact UV
  gates. The parent stops at the first valid preset.

## First real preserved-panda LOD attempt

Input was the immutable sanitized 5-step panda:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\panda_4_5_degenerate_boundary\5step_sanitized.glb`

The 220k candidate was written at:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\texture_v2_unified\lod_candidate\tactical_red_panda_scout_lod0.glb`

The independent fresh-import report and neutral four-view renders are in:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\texture_v2_unified\lod_candidate\fresh_import_geometry_validation.json`

The 220k candidate achieved 219,866 triangles and passed finite-bounds,
round-trip, and four-view structural checks. It did not pass the topology gate:
the source had 0 boundary edges and 18 non-manifold edges; the candidate had
406 boundary edges and 6 non-manifold edges. A separate 250k candidate also
failed the same gate with 397 boundary edges and 6 non-manifold edges. Neither
candidate entered UV or texturing.

Panda export-QA summary:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\unified_pipeline_v2_integration\panda_export_qa\tactical_red_panda_unified\pipeline_summary.json`

Panda fresh-import report:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\unified_pipeline_v2_integration\panda_export_qa\tactical_red_panda_unified\state\EXPORT_QA\proven\fresh_import_validation.json`

Shaman reuse receipt:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\shaman\diagnostics\unified_pipeline_v2_integration\shaman_reuse\shaman_existing_reuse\state\GENERATE\receipt.json`

## Preserve-source LOD and existing-UV continuation

The current manifest uses `lod.mode=preserve_source`. The LOD receipt records
`LOD_STAGE=BYPASSED_SOURCE_GEOMETRY`, `LOD_REQUIRED=false`, and source hash
`1170cff8c1e29b6ab210cac1f8100ca160575e692063c7de0f478db0d7597ff2`.
The rejected 220k and 250k LOD candidates remain diagnostics only.

The exact projection-fixed 5-step UV source was copied immutably to:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\texture_completion_v2\uv_source\tactical_red_panda_scout_5step_uv_source_immutable.glb`

Its verified SHA-256 is
`296e274db799efb86a0e6c1984ce9fca46f32ece6ab28b061eda00658ac94e0e`.
The existing-UV validation report proves a fresh Blender import, finite and
in-bounds UVs, material resolution, and packed texture resolution. A byte/face
identity claim against the separate sanitized raw source is not made: the
historical textured artifact has 644,412 imported triangles while the raw
sanitized source parses to 693,576 triangles.

## Texture-completion evidence

The preserved original proof root is:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\texture_completion_v2\unified_pipeline`

The TEXTURE receipt is proven after the all-primitive diagnostic-reader fix.
The native 1024x1024 base-colour atlas, material manifest, textured GLB, and
Blender review renders are in `state\TEXTURE\proven`. The review set contains
front, three-quarter, side, rear, and close-up renders. The projection report
records 91.52% observed and 8.48% synthesized UV coverage. The rear render is
retained as direct visual evidence; it shows the existing artifact’s visible
black/debris defects, so those are not hidden by the structural receipt.

TEXTURE_QA was run against the preserved artifact and rejected with
`FLOATING_DEBRIS` and `UV_OVERLAP`. Its proposed repair included xatlas; that
repair was not authorized and was not run. EXPORT_QA for this new completion
artifact therefore remains unproven.

## Classifications

- `UNIFIED_STAGE_INTERFACE=PROVEN`
- `RESUME_WITH_HASH_INVALIDATION=PROVEN`
- `CURRENT_GENERATOR_PRESERVED=PROVEN`
- `DECODER_SANITIZER_PRESERVED=PROVEN`
- `PROJECTION_VISIBILITY_GATE_PRESERVED=PROVEN`
- `PANDA_REAR_FACE_ABSENT=PROVEN` (existing fixed artifact and prior rear render)
- `SHAMAN_EXISTING_GEOMETRY_REUSED=PROVEN`
- `FRESH_IMPORT_EXPORT_QA=PROVEN`
- `LOD_STAGE=BYPASSED_SOURCE_GEOMETRY`
- `LOD_REQUIRED=false`
- `PANDA_EXISTING_UV_REUSE=PROVEN`
- `PANDA_NEW_UV_UNWRAP=NOT_REQUIRED`
- `XATLAS_DEFAULT_ROUTE=REJECTED_FOR_RUNTIME`
- `PANDA_GEOMETRY_IDENTITY=NOT_PROVEN`
- `GRAPH_MATERIAL_PROPAGATION=NOT_PROVEN`
- `TEXTURE_COMPLETION_STAGE=PROVEN` (with visible limitations documented above)
- `TEXTURE_QA=REJECTED`
- `EXPORT_QA_TEXTURE_COMPLETION=NOT_PROVEN`
- `WHOLE_OLD_BRANCH_MERGE=REJECTED`
- `LOD_220K_STRUCTURAL_IMPORT=PROVEN`
- `LOD_TOPOLOGY_GATE=REJECTED`
- `LOD_BASELINE=NOT_PROVEN`
- `UV_ATLAS_V2=NOT_PROVEN` (blocked by the LOD topology gate; no real xatlas
  process was started in this proof)
- `FULL_AROUND_TEXTURE_BASELINE=REJECTED`

No new dropped-image GPU run was started. The production branch remained
untouched; this worktree is the only location containing the integration
changes.
