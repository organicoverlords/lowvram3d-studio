# Mini Turbo production-quality pipeline correction

## Why this change is required

The pipeline promoted one-step and two-step Mini Turbo outputs into textured benchmark baselines without first proving geometry quality. The real pipeline defect is **missing pre-texture geometry acceptance**, not simply the use of low step counts.

A blanket five-step policy would waste production time on easy assets. The corrected policy uses a fast candidate first and escalates only when the geometry gate rejects it.

## Authoritative classification

- `tactical_red_panda_scout` repaired v7 remains the accepted basic baseline.
- `frog_salvage_diver` v7 and v8 remain rejected/blocked evidence.
- `BLOCKED_FROG_COMPONENTS_FUSED_REQUIRES_DIFFERENT_GEOMETRY` means those preserved frog candidates cannot be repaired safely by detached-component cleanup.
- No more per-asset deletion work on frog v7/v8 is authorized.

## Required pipeline change

Implement this as a shared generation-quality policy, not a frog-specific script.

### 1. Separate smoke and baseline intents

Add explicit run intent:

- `smoke`
- `baseline`

`smoke`:

- may use `steps=1`;
- is used only after model/runtime/pipeline changes or for explicit diagnostics;
- proves model load, diffusion execution, decoder execution, and GLB serialization only;
- writes to a diagnostic/canary directory;
- can never be promoted to a canonical asset;
- can never enter UV, texture, material, or final-render stages;
- reports `NON_PROMOTABLE_SMOKE_OUTPUT=true`.

Do **not** run a smoke attempt before every normal asset. That would add avoidable production time.

`baseline` Candidate A:

- Mini Turbo `num_inference_steps=2`;
- `guidance_scale=5.0`;
- `octree_resolution=256`;
- `num_chunks=1500` on this 6 GB machine;
- seed `12345`;
- one GPU-heavy process at a time;
- may proceed only to the pre-texture geometry gate.

A two-step candidate is not automatically accepted. It becomes promotable only when the geometry gate passes.

### 2. Add a pre-texture geometry quality gate

Run after raw GLB generation and clean fresh import, before UV generation, projection, or texture work.

Inputs:

- canonical source image;
- exact conditioning alpha/matte;
- raw generated GLB;
- source-facing camera transform.

Required evidence:

- source-aligned mesh silhouette PNG;
- conditioning/source alpha PNG;
- silhouette overlay PNG;
- triangle/component ID render;
- JSON metrics and decision.

Required metrics:

- source-mask/mesh-mask IoU;
- mesh and source silhouette areas;
- unsupported mesh-silhouette ratio outside a bounded source-mask dilation;
- missing-source-silhouette ratio;
- unsupported screen-space island count and area;
- welded connected-component count;
- detached face/area fractions;
- boundary and non-manifold edges;
- single-use indexed-vertex percentage;
- source-facing depth-separated outboard region count.

Calibrate deterministic asset-agnostic thresholds using existing anchors:

- `tactical_red_panda_scout` repaired v7 must pass;
- frog v7 and frog v8 must fail.

The gate returns only:

- `PASS_GEOMETRY_FOR_TEXTURE`
- `REJECT_GEOMETRY_DEBRIS_HALO`
- `REJECT_GEOMETRY_FRAGMENTED_SILHOUETTE`
- `REJECT_GEOMETRY_TOPOLOGY`
- `NOT_PROVEN_GATE_ERROR`

Rejected geometry must not be textured.

### 3. Quality-gated escalation policy

For an asset without reusable accepted geometry:

#### Candidate A — fast baseline

- steps 2;
- guidance 5.0;
- octree 256;
- chunks 1500;
- seed 12345.

If Candidate A passes, texture it immediately. Do not run five steps.

#### Candidate B — quality escalation

Run only when Candidate A fails specifically because of fused debris, fragmented silhouette, or another geometry-quality rejection—not for unrelated I/O or implementation errors.

- steps 5;
- same guidance, octree, chunks, conditioning, and seed;
- no other parameter changes.

Candidate B is the only authorized quality escalation. Do not run an intermediate three-step candidate and do not change seed at the same time, because that would obscure which correction helped.

If Candidate B passes, texture it. If it fails, block that asset and continue to the next benchmark.

This means normal assets cost one two-step generation. Only rejected geometry pays for a five-step rerun.

### 4. Cleanup remains secondary

The welded connected-component cleanup remains valid for genuinely detached rods, blobs, and tiny islands after a candidate passes the pre-texture gate.

It must not:

- rescue geometry that fails the fused-halo gate;
- delete connected surface patches merely because they are unsupported;
- use asset-specific coordinates or component IDs;
- promote structural import success over visible geometry rejection.

### 5. Texture only accepted geometry

Only after `PASS_GEOMETRY_FOR_TEXTURE`:

1. conservative detached-component cleanup;
2. topology validation;
3. UV preservation/unwrap;
4. repaired raster projection;
5. textured GLB export;
6. clean fresh import;
7. front, three-quarter, side, and rear renders;
8. baseline classification.

## Frog regression execution

After code and CPU fixture tests pass:

1. Prove the gate rejects preserved frog v7 and v8.
2. Prove it accepts panda repaired v7.
3. Run one new frog Candidate A at two steps.
4. If Candidate A fails the geometry gate, run one Candidate B at five steps with every other input unchanged.
5. Texture only a passing candidate.
6. Do not perform another deletion repair of v7/v8.

Possible outcomes:

- `FROG_2STEP_GEOMETRY_GATE=PROVEN`
- `FROG_5STEP_ESCALATION_GEOMETRY_GATE=PROVEN`
- `FROG_BASIC_TEXTURED_BASELINE=PROVEN_WITH_LIMITATIONS`
- `BLOCKED_MINITURBO_FROG_2_AND_5_STEP_CANDIDATES_FAILED_GEOMETRY_GATE`

## Scope limits

Do not add a new generator, telemetry framework, asset database, profile framework, or manual review step.

Do not rerun the barn.

Do not change panda v7.

Commit shared policy/code/tests first, then run the bounded frog regression. Preserve all old rejected artifacts and proof receipts.