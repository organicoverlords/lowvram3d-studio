# Codex: continue here

## Latest visual checkpoint — 2026-08-05

Do not spend the next turn only on contract debugging. A separate visual-only prototype
lane now exists at:

`C:\AI\LowVRAM3D-benchmarks\production\panda_visual_prototype_experimental_20260805`

The latest candidate is:

`fidelity_pass_1024_face_landmark_right\panda_face_landmark_right_1024_textured.glb`

Its fresh front render is under the same folder at `renders_unlit\front.png`. The face is
recognizable and the rear is face-free, but the muzzle landmark is still slightly offset
and the body texture remains noisy. Continue with one bounded visual alignment/fidelity
pass at a time. Do not open Blender automatically; write a native `.blend` only when the
user asks to inspect it. Rendered PNGs may be shown directly and opened in Paint when
requested.

The visual candidates do not alter the strict proof lane. Current proof status remains:
256 = two ordinary CONTRACT_ERROR rows; 384 = replay-clean; 512 production proof and
downstream texture promotion remain blocked.

Latest visual pipeline improvement: `workers/injective_atlas_texture.py` now supports
`--generated-detail-scale`. The 1024 visual candidate used `0.35` to suppress generated-view
diffusion speckle while keeping the authoritative original front unattenuated. Do not use
this as proof evidence; compare its fresh renders visually before selecting a later pass.

Work only on branch `production/two-character-models-20260804`.

Before editing, verify repository origin, branch, local HEAD, remote HEAD, and a clean worktree. Never work on `main`, merge, force-push, or overwrite preserved artifacts.

## Immediate task: hybrid 6 GB GPU texture repair

The current chart-separated panda proves the face-on-back defect can be removed, but its texture is still pale and fragmented. Stop CPU-only color boosting. Use the GTX 1660 SUPER for the image-generation work while retaining CPU geometry/UV/projection correctness.

New shared workers on this branch:

- `workers/comfyui_gpu_texture_job.py`
- `workers/gpu_texture_repair_sequence.py`
- `workers/surface_albedo_completion.py` — deterministic fallback only
- `workers/material_aware_color_recovery.py` — deterministic fallback only
- `tests/test_comfyui_gpu_texture_job.py`
- `tests/test_gpu_texture_repair_sequence.py`

The GPU workers do not mutate geometry, UVs, atlases, or GLBs. They generate candidate 2-D texture references only. The existing CPU face-ID/depth projector remains the only route allowed to write those references into the atlas.

### Immutable inputs

Preserve all existing 4-step, 5-step, sanitized, projection-repaired, chart-separated, and rejected postprocess artifacts.

Use the exact chart-separated panda from the latest bounded repair as the mesh/UV input. Verify its source and output hashes from the existing receipt rather than guessing a path.

Do not run:

- Mini Turbo geometry generation;
- LOD generation;
- xatlas;
- general UV unwrap;
- orientation postprocessing;
- global saturation recovery;
- parallel GPU jobs.

### Research-backed execution model

The local 6 GB texture-projection workflow studied from PixelArtistry uses generated front/side/back views, an albedo-conditioning route, 1024 output for low-VRAM operation, and retexturing of existing GLBs. Its weak points are projection defects, top/bottom gaps, custom-rasterizer failures, OOMs, and front-looking rear generations. Therefore:

1. use its exported ComfyUI API workflow only as a **2-D reference generator**;
2. do not use its rasterizer, UV, mesh, or export nodes;
3. reject a generated rear before projection when it resembles the front;
4. project accepted pixels through our proven depth/normal/face-ID route;
5. keep accepted source texels immutable;
6. fill remaining unseen triangles by bounded material-local surface propagation.

### ComfyUI compatibility and VRAM policy

Use the existing installed ComfyUI environment. Do not install another Trellis stack or replace the current environment.

Prefer an already installed AeroX/PixelArtistry-style texture-projection or equivalent geometry-conditioned image workflow. Export it with `File -> Export (API)` and create a small binding JSON for `workers/comfyui_gpu_texture_job.py`.

Required binding keys:

- `source`
- `width`
- `height`
- `seed`

Bind these when the workflow exposes them:

- `depth`
- `normal`
- `mask`
- `prompt`
- `negative_prompt`
- `view` or `view_name`
- `resolution`
- `output_prefix`

The config also lists exactly one final output node under `output_nodes`; intermediate previews must not be mistaken for the accepted image.

Launch ComfyUI with the already proven Turing-compatible path:

```text
DTYPE=FP16
BF16=DISABLED
FLASH_ATTENTION_2=DISABLED
ATTENTION=PYTORCH_SDPA_OR_CURRENT_PROVEN_DEFAULT
BATCH_SIZE=1
PARALLEL_GPU_JOBS=1
```

Do not use a launcher that requires Ampere-only FlashAttention. Do not keep geometry-generation models loaded during texturing.

GPU job defaults:

```text
initial_resolution=512
fallback_resolution=384
timeout_seconds=300
minimum_free_mb=1200
oom_retries=1
unload_models_after_attempt=true
```

`workers/comfyui_gpu_texture_job.py` already:

- acquires an exclusive GPU lock;
- verifies ComfyUI and free VRAM;
- uploads source/normal/depth/mask inputs;
- queues one API workflow;
- interrupts and removes it on timeout;
- retries once at lower resolution only after an OOM;
- calls `/free` to unload models;
- preserves attempt receipts and hashes.

### Required condition renders

From the immutable chart-separated GLB, render matched square condition images for:

1. front;
2. left side;
3. right side;
4. true rear.

For every view save:

- normal;
- depth;
- alpha/silhouette mask;
- camera transform and normalized direction;
- visible triangle IDs.

Front and rear camera directions must have dot product `<= -0.999`.

Do not generate side/rear views from prompt alone. The workflow must receive geometry conditions where supported.

### GPU sequence

Create a repo-local manifest for `workers/gpu_texture_repair_sequence.py` with these serial jobs:

1. `front_albedo`
   - source: registered original panda image;
   - mask: registered source alpha;
   - purpose: remove baked lighting and recover true unlit color;
   - do not change silhouette or identity.

2. `left_reference`
   - source/reference: accepted front albedo;
   - left normal/depth/mask;
   - explicit prompt for the tactical red panda scout from the left side.

3. `right_reference`
   - source/reference: accepted front albedo;
   - right normal/depth/mask;
   - explicit right-side prompt.

4. `rear_reference`
   - source/reference: accepted front albedo;
   - true rear normal/depth/mask;
   - explicit true-back prompt: backpack, rear ghillie suit, rear tail; no face, eyes, muzzle, front chest, or mirrored front view.

Every job runs separately. Models must be unloaded between jobs.

### Pre-projection image QA

`workers/gpu_texture_repair_sequence.py` must pass each required output before any atlas write:

- non-empty foreground;
- output saturation above the configured floor;
- silhouette-mask IoU at least `0.55`;
- one and only one final output image;
- true rear direct and mirrored correlation with the front both below `0.82`.

A rejected rear is not projected and is not repaired by color grading. Preserve it as diagnostic evidence and stop the sequence.

### CPU projection and ownership

After all GPU references pass:

1. generate/verify depth visibility and face-ID buffers for each accepted view;
2. project a sample only when all are true:
   - depth-visible triangle;
   - front-facing normal;
   - valid foreground pixel;
   - rendered face-ID match;
   - confidence at or above threshold;
3. record per-triangle winning view, confidence, visibility, face-ID status, and fallback mode;
4. reject any rear-dominant triangle receiving front facial provenance;
5. never mirror, wrap, or copy front facial content to an unseen triangle;
6. preserve observed source texels exactly after projection.

The existing targeted chart separation remains authoritative. Do not change UV ownership unless a new exact conflict is proven.

### Remaining unseen areas

Use `workers/surface_albedo_completion.py` or the material-aware recovery worker only after strict multiview projection.

Rules:

- operate on unobserved triangles only;
- use welded surface adjacency/geodesic propagation;
- prune propagation across sharp normal/geometry boundaries;
- keep fur, clothing, rifle, backpack, and tail color families separate where evidence permits;
- never allow face-region donors to fill rear-head triangles;
- do not apply global HSV recovery.

### Final QA and promotion

Fresh-import the new GLB in Blender and render:

- front;
- three-quarter;
- side;
- true rear;
- rear provenance overlay.

Promote only when all are true:

```text
GEOMETRY_REUSED=PROVEN
UV_CHART_SEPARATION_PRESERVED=PROVEN
GPU_REFERENCE_GENERATION=PROVEN
GPU_JOBS_SERIAL=PROVEN
PROJECTION_GATING=PROVEN
TRIANGLE_PROVENANCE=PROVEN
REAR_FACE_PROJECTION=PROVEN_ABSENT
TEXTURE_COLOR_QA=PROVEN
FRESH_IMPORT=PROVEN
PANDA_BASIC_TEXTURED_MODEL=PROVEN
```

Visual requirements:

- no face on the back;
- substantially less gray/pale than the current chart-separated candidate;
- recognizable red-panda face, ghillie/tactical clothing, rifle, backpack, and tail;
- no large black holes or transparent gaps;
- rear may remain less detailed than front but must have plausible material color.

Maximum budget:

- one complete four-job GPU sequence;
- one lower-resolution retry per job only on OOM;
- no seed sweep;
- no unchanged retry;
- no full geometry/UV rerun.

Stop with `USER_REVIEW_REQUIRED` once the new contact sheet is ready. Do not continue to unrelated refinements before visual review.

## Repository validation before local GPU run

Run first:

```text
python -m compileall -q workers tests
python -m pytest -q \
  tests/test_comfyui_gpu_texture_job.py \
  tests/test_gpu_texture_repair_sequence.py \
  tests/test_projection_repair.py
```

Create the exported API workflow and binding config only after these focused tests pass. Commit shared code/config/tests before the expensive run. Preserve generated images and GLBs outside Git; commit only compact receipts, hashes, configuration, and contact-sheet proof.

## Six-model milestone remains authoritative

After the panda is accepted, continue with usable static textured models in this order:

1. antlered bird shaman — reuse canonical clean geometry; no sleeve/rig loop;
2. wind-bent barn and trees — reuse best verified geometry; no long regeneration;
3. Lucky Drown casino boat — one bounded sanitized generation if needed;
4. mossy mountain titan — one bounded sanitized generation if needed;
5. frog salvage diver — one higher-step sanitized candidate at most; no destructive deletion loop.

For every asset, reuse existing geometry and UVs first, keep LOD optional, keep xatlas opt-in only, cap texture attempts at two, and package the first visually usable fresh-importable GLB.

Commit and push validated integration-branch changes. Verify local equals remote and leave the worktree clean. Do not merge branches yet.
