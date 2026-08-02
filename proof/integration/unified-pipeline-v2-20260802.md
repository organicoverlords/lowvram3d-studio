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

Panda export-QA summary:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\unified_pipeline_v2_integration\panda_export_qa\tactical_red_panda_unified\pipeline_summary.json`

Panda fresh-import report:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\unified_pipeline_v2_integration\panda_export_qa\tactical_red_panda_unified\state\EXPORT_QA\proven\fresh_import_validation.json`

Shaman reuse receipt:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\shaman\diagnostics\unified_pipeline_v2_integration\shaman_reuse\shaman_existing_reuse\state\GENERATE\receipt.json`

## Classifications

- `UNIFIED_STAGE_INTERFACE=PROVEN`
- `RESUME_WITH_HASH_INVALIDATION=PROVEN`
- `CURRENT_GENERATOR_PRESERVED=PROVEN`
- `DECODER_SANITIZER_PRESERVED=PROVEN`
- `PROJECTION_VISIBILITY_GATE_PRESERVED=PROVEN`
- `PANDA_REAR_FACE_ABSENT=PROVEN` (existing fixed artifact and prior rear render)
- `SHAMAN_EXISTING_GEOMETRY_REUSED=PROVEN`
- `FRESH_IMPORT_EXPORT_QA=PROVEN`
- `WHOLE_OLD_BRANCH_MERGE=REJECTED_NOT_PERFORMED`
- `UV_ATLAS_V2=NOT_PROVEN` (the long-running 1024 xatlas attempt was explicitly cancelled)
- `FULL_AROUND_TEXTURE_BASELINE=NOT_PROVEN`

No new dropped-image GPU run was started. The production branch remained
untouched; this worktree is the only location containing the integration
changes.
