# Mini Turbo three-step comparison: panda and frog

## Purpose

Run a controlled comparison to determine whether `num_inference_steps=3` improves geometry quality over the existing two-step candidates without paying the full five-step cost.

This is an execution experiment, not a new framework or permanent production rule.

## Assets

1. `tactical_red_panda_scout`
2. `frog_salvage_diver`

Use the canonical repository source images and the exact conditioning images/normalization route used by the successful two-step runs. Preserve every existing artifact unchanged.

## Controlled variables

Change exactly one generation variable:

- `num_inference_steps`: `2 -> 3`

Keep identical:

- Mini Turbo model and weights;
- conditioning image bytes and hash;
- guidance scale `5.0`;
- octree resolution `256`;
- chunks `1500`;
- seed `12345`;
- dtype/device/FlashVDM settings;
- source-facing camera and render settings;
- cleanup, UV, projection, export, and validation code.

Run one GPU-heavy process at a time.

Do not run five steps during this experiment.
Do not change seed.
Do not perform manual edits.
Do not rerun the barn or any other benchmark.

## Panda experiment

Generate a new raw three-step panda candidate into a separate immutable experiment directory. Do not overwrite repaired panda v7 or make the new candidate canonical.

Required stages:

1. generate raw GLB at three steps;
2. clean fresh import;
3. record welded topology and component metrics;
4. render untextured/clay front, three-quarter, side, and rear views;
5. run only the existing generic detached-component cleanup, if required;
6. apply the same repaired raster texture route;
7. export and fresh-import textured GLB;
8. render a matched textured contact sheet.

Compare directly against repaired panda v7 using the same cameras and image sizes.

Panda comparison criteria:

- source-facing identity and facial readability;
- rifle, backpack, ears, hands, feet, and tail preservation;
- detached rod/blob count;
- fused surface noise and debris halo;
- side/rear silhouette coherence;
- welded components, boundary/non-manifold edges, and single-use vertices;
- observed versus synthesized texture coverage;
- runtime and peak VRAM.

The three-step panda replaces repaired v7 only when it is materially better or equal on every safety gate and better on at least one visible geometry criterion. Otherwise repaired v7 remains canonical.

## Frog experiment

Generate a new raw three-step frog candidate into a separate immutable experiment directory. Never mutate or clean frog v7/v8 in place.

Required stages:

1. generate raw GLB at three steps;
2. clean fresh import;
3. record welded topology and component metrics;
4. render untextured/clay front, three-quarter, side, and rear views;
5. compare the raw three-step candidate against preserved frog v7/v8 before destructive cleanup;
6. proceed through conservative detached-component cleanup and texturing only when the three-step raw geometry is visibly cleaner than frog v7/v8 and does not lose required equipment;
7. fresh-import and render the textured candidate.

Required frog equipment preservation:

- frog head and torso;
- both arms and feet;
- backpack/tank;
- lantern;
- hanging bag;
- major canisters and connected hoses where generated.

Frog comparison criteria:

- debris halo area/count around shoulders and backpack;
- floating shard count;
- fused noise on the main surface;
- fragmented side/rear silhouette;
- equipment retention;
- source-facing identity;
- topology metrics;
- texture coherence;
- runtime and peak VRAM.

Do not attempt another deletion repair when the three-step debris is fused into the main surface. In that case preserve and reject the candidate.

## Required outputs

For each asset:

- source and conditioning SHA-256;
- exact command/settings;
- raw three-step GLB;
- raw fresh-import report;
- clay contact sheet;
- cleanup report if cleanup runs;
- textured GLB if texture stage runs;
- textured fresh-import report;
- textured contact sheet;
- metrics JSON;
- comparison report against the existing two-step reference;
- runtime and peak VRAM.

Create one side-by-side comparison sheet per asset with labels:

- `2 STEP REFERENCE`
- `3 STEP CANDIDATE`

## Required classifications

### Panda

- `PANDA_3STEP_GEOMETRY`
- `PANDA_3STEP_TEXTURED_BASELINE`
- `PANDA_3STEP_VS_V7`
- `PANDA_CANONICAL_DECISION`

### Frog

- `FROG_3STEP_GEOMETRY`
- `FROG_3STEP_DEBRIS_HALO`
- `FROG_3STEP_TEXTURED_BASELINE`
- `FROG_3STEP_VS_2STEP`

Use `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED` with exact reasons.

## Continuation

Commit and push the experiment evidence after both runs. Verify remote equals local and the worktree is clean. Stop for user review before changing the general production step policy or starting another asset.