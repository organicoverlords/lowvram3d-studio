# Panda 4/5-step degenerate-face boundary proof

The experiment inputs were corrected to both be `tactical_red_panda_scout` candidates:

- 4-step: `C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\generate_steps4_5\raw_4step_retry2.glb`
- 5-step: `C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\generate_steps4_5\raw_5step.glb`

The complete boundary report is preserved at:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260802\tactical_red_panda_scout\diagnostics\panda_4_5_degenerate_boundary\panda_4_5_degenerate_boundary_report.json`

## Boundary result

| Boundary | 4-step | 5-step | Classification |
|---|---:|---:|---|
| Decoder output counts from the generation reports | 341,513 vertices / 1,157,010 faces | 346,913 vertices / 1,172,394 faces | PROVEN |
| Trimesh conversion duplicate-index faces | 474,175 (40.982792%) | 478,818 (40.841048%) | PROVEN |
| Direct GLB accessor arrays | Same counts, hashes, and duplicate faces as trimesh | Same counts, hashes, and duplicate faces as trimesh | PROVEN |
| Blender fresh import | 341,513 vertices / 682,835 faces | 346,913 vertices / 693,576 faces | PROVEN |

Both candidates contain exact duplicate-index, zero-area faces before GLB serialization. Direct
accessor arrays and trimesh arrays are byte-equivalent at the measured boundary, so GLB
serialization is not the corruption source. Blender is dropping the invalid faces during import,
which previously masked the decoder defect.

The original candidates remain preserved. CPU-only sanitized copies are:

- `...\\panda_4_5_degenerate_boundary\\4step_sanitized.glb`
- `...\\panda_4_5_degenerate_boundary\\5step_sanitized.glb`

They remove only the exact duplicate-index faces and fresh-import with the reduced face counts.

## Correction

`workers/mini_turbo_generate.py` now records array hashes/counts immediately after decoder return,
removes exact duplicate-index faces before downstream stages, records the sanitized boundary, and
records the post-export in-memory boundary. The correction is covered by the synthetic decoder
fixture in `tests/test_mini_turbo_telemetry.py`.

- `PANDA_DECODER_DEGENERATE_FACE_SANITIZATION=PROVEN_CPU_FIX`
- `PANDA_GLB_SERIALIZATION_CORRUPTION=REJECTED_NOT_OBSERVED`
- `PANDA_DECODER_ARRAY_HASHES_FOR_EXISTING_RUNS=NOT_PROVEN_NOT_PERSISTED`
- `PANDA_4STEP_TEXTURING=NOT_PROVEN_STOPPED_BEFORE_EXPORT`
- `PANDA_5STEP_TEXTURING=NOT_PROVEN_DEFERRED_UNTIL_CORRECTED_GENERATION`
- `FROG_3_4_5_EXPERIMENT=NOT_PROVEN_SEPARATE_UNREPRESENTED_FILES`

Validation: `tests/test_mini_turbo_telemetry.py` — 14 passed; Python compile check passed; no GPU
rerun was performed after the correction.
