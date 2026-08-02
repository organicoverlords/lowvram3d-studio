# Mini Turbo production-quality pipeline correction

## Why this change is required

The pipeline promoted one-step and two-step Mini Turbo outputs into textured benchmark baselines. Those settings were introduced as fast execution/decoder diagnostics. They are not the production Turbo setting.

Tencent's official Hunyuan3D-2 Gradio app sets Turbo mode to **5 inference steps**, octree resolution 256, and guidance scale 5.0. The frog's fused debris halo was generated at two steps and cannot be repaired safely by deleting connected surface patches. The pipeline must stop promoting under-denoised smoke geometry.

## Authoritative classification

- `tactical_red_panda_scout` repaired v7 remains the accepted basic baseline.
- `frog_salvage_diver` v7 and v8 remain rejected/blocked evidence.
- `BLOCKED_FROG_COMPONENTS_FUSED_REQUIRES_DIFFERENT_GEOMETRY` means the current geometry is not repairable by detached-component cleanup.
- No more per-asset frog deletion work is authorized.

## Required pipeline change

Implement this as a shared generation-quality policy, not a frog-specific script.

### 1. Separate smoke and baseline intents

Add an explicit run intent accepted by the generation orchestrator and worker:

- `smoke`
- `baseline`

`smoke`:

- may use `steps=1`;
- proves only model load, diffusion execution, decoder execution, and GLB serialization;
- writes only into a diagnostic/canary directory;
- can never be promoted to a canonical asset;
- can never enter UV, texture, material, or final-render stages;
- must report `NON_PROMOTABLE_SMOKE_OUTPUT=true`.

`baseline`:

- Mini Turbo `num_inference_steps=5`;
- `guidance_scale=5.0`;
- `octree_resolution=256`;
- `num_chunks=1500` for this 6 GB machine unless an already-proven lower-memory value is required;
- one GPU-heavy process at a time;
- no automatic downgrade to steps 1 or 2;
- only a baseline-intent candidate may proceed toward texturing.

Remove the previous policy that treated a successful two-step retry as production geometry.

### 2. Add a pre-texture geometry quality gate

Run this gate after raw GLB generation and clean fresh import, before UV generation, projection, or texture work.

Inputs:

- canonical source image;
- exact conditioning alpha/matte used by generation;
- raw generated GLB;
- source-facing camera transform used for the benchmark.

Required evidence:

- source-aligned mesh silhouette PNG;
- conditioning/source alpha PNG;
- silhouette overlay PNG;
- triangle/component ID render;
- JSON metrics and decision.

Required metrics:

- source-mask/mesh-mask IoU;
- mesh silhouette area;
- source silhouette area;
- unsupported mesh-silhouette ratio outside a bounded dilation of the source mask;
- missing-source-silhouette ratio;
- number and area of unsupported screen-space islands;
- welded connected-component count;
- detached face/area fractions;
- boundary and non-manifold edges;
- single-use indexed-vertex percentage;
- source-facing depth-separated outboard region count.

Do not invent fixed thresholds from theory. Calibrate deterministic thresholds using the existing anchors:

- `tactical_red_panda_scout` repaired v7 must pass;
- frog v7 and frog v8 must fail because of their visible debris halo and fragmented silhouette.

Record the selected threshold values and both anchor metric sets in tests/fixtures. Thresholds must remain asset-agnostic.

The gate returns only:

- `PASS_GEOMETRY_FOR_TEXTURE`
- `REJECT_GEOMETRY_DEBRIS_HALO`
- `REJECT_GEOMETRY_FRAGMENTED_SILHOUETTE`
- `REJECT_GEOMETRY_TOPOLOGY`
- `NOT_PROVEN_GATE_ERROR`

A rejected geometry candidate must not be textured.

### 3. Bounded automatic candidate policy

For an asset without reusable accepted geometry:

Candidate A:

- baseline intent;
- Mini Turbo steps 5;
- guidance 5.0;
- octree 256;
- chunks 1500;
- seed 12345.

If Candidate A fails the pre-texture geometry gate, permit exactly one Candidate B:

- same model, conditioning, steps, guidance, octree, and chunks;
- seed 1234;
- no other parameter changes.

Select the passing candidate. If both pass, select the higher deterministic geometry-gate score. If neither passes, block that asset and continue to the next benchmark.

Do not use detached-component cleanup to force a rejected fused candidate through the gate.

### 4. Cleanup remains secondary

The existing welded connected-component cleanup remains valid for genuinely detached rods, blobs, and tiny islands after a candidate passes the pre-texture quality gate.

It must not:

- rescue a geometry candidate that fails the fused-halo gate;
- delete connected surface patches merely because they are unsupported;
- use frog-specific coordinates or component IDs;
- promote structural import success over visible geometry rejection.

### 5. Texture only accepted geometry

Only after `PASS_GEOMETRY_FOR_TEXTURE`:

1. run conservative detached-component cleanup;
2. validate topology;
3. unwrap/preserve UVs;
4. run the repaired raster projection route;
5. export textured GLB;
6. fresh-import in a clean Blender process;
7. render front, three-quarter, side, and rear views;
8. classify the basic textured baseline.

## Frog regression execution

After code and CPU fixture tests pass:

1. Prove the new gate rejects preserved frog v7 and v8 without modifying them.
2. Prove the gate accepts panda repaired v7.
3. Run exactly one new frog Candidate A at the 5-step baseline settings.
4. Run Candidate B only if Candidate A fails the gate.
5. Texture only a passing candidate.
6. Do not perform another manual or component-deletion repair of v7/v8.

Possible final outcomes:

- `FROG_5STEP_GEOMETRY_GATE=PROVEN`
- `FROG_BASIC_TEXTURED_BASELINE=PROVEN_WITH_LIMITATIONS`
- `BLOCKED_MINITURBO_FROG_5STEP_CANDIDATES_FAILED_GEOMETRY_GATE`

## Scope limits

Do not add a new generator, telemetry framework, asset database, profile framework, or manual review step.

Do not rerun the barn.

Do not change panda v7.

Commit the shared policy/code/tests first, then run the bounded frog regression. Preserve all old rejected artifacts and proof receipts.