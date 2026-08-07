# Rigging Pipeline V3 — vendor-first low-VRAM integration lane

Status: **VENDOR-FIRST TEST LANE / GPU RUNTIME NOT YET PROVEN**

Working foundation: `agent/scene-pipeline-smoke-20260803` at commit `9533cf8eef4309cdcacc213b9f1bbfbb920e044e`.

This lane keeps the proven image-to-3D foundation and studies what comes after it. It does **not** replace TRELLIS/Hunyuan generation or invent a new rigging stack before existing solutions are measured.

## Non-negotiable rule

Use known upstream implementations unchanged first. Only after a stock backend is actually run, visually graded, and rejected on the target hardware may this branch patch or replace it.

That means:

1. stock 3D Gen Studio / ComfyUI tooling first;
2. stock Make-It-Animatable / UniRig nodes first;
3. existing Studio animation-retarget path first;
4. existing Unreal retarget/LOD tooling first;
5. only then custom adapters, patched attention, custom segmentation, or new rig algorithms if a measured failure requires them.

Unreferenced experimental adapter ideas are not part of the branch until that proof boundary is crossed.

## Foundation inherited from smoke

The branch deliberately follows the current smoke work rather than rebuilding earlier stages.

Current production geometry/paint direction remains:

```text
source image
  -> TRELLIS.2 high-quality geometry
  -> native Stage 6 finalization
  -> Hunyuan3D-Paint
  -> truthful Blender raster QA
```

The latest smoke commit also corrects two assumptions that downstream work must inherit:

- Hunyuan Paint `--render-size` / `--texture-size` had previously been set too late; old runs silently used vendor 2048/2048. Future texture-size A/Bs must use the fixed renderer and record the baked atlas rather than echo the request.
- Deliverables now carry real-world size metadata instead of a flat scale multiplier. Rig/export tests must preserve or explicitly restore that scale before Unreal validation.

## Required post-texture ordering

For an organic asset:

```text
preserve textured LOD0
  -> stock rigger
  -> structural rig QA
  -> deformation QA
  -> stock animation retarget
  -> engine export
  -> skeletal LOD generation
```

Do not return to the old `LOD0/LOD1/LOD2 -> rig LOD0` ordering for skeletal characters unless weight transfer across independent LODs is proven.

Semantic segmentation is **not** a mandatory organic pre-rig stage. Use it only when deformation evidence proves weight bleed or a rigid accessory must be isolated.

## Test order — known solutions first

### A. Humanoid

First runtime candidate: **Make-It-Animatable through the existing ComfyUI-UniRig nodes**, unchanged.

Use the upstream graph components already provided by `PozzettiAndrea/ComfyUI-UniRig`:

```text
UniRigLoadMesh
  -> MIALoadModel (FP16 first)
  -> MIAAutoRig
```

Do not write a separate MIA inference implementation before this path is tested.

Acceptance evidence:

- runtime on the GTX 1660 SUPER;
- measured peak VRAM/RAM;
- original texture/material preservation;
- armature and skin weights present;
- rest, elbow, knee, crouch and shoulder-raise deformation renders;
- Mixamo/Studio animation retarget test;
- Unreal import after deformation proof.

If stock MIA fails for a reason that stock **UniRig** can address, test stock UniRig next before writing custom rig code.

### B. General creature

First runtime candidate: **stock UniRig through ComfyUI-UniRig**, FP16/SDPA where the existing node exposes it.

Puppeteer is a later challenger, not the default. Its upstream skeleton path assumes FlashAttention-2, so changing that to SDPA/eager would already be a code modification. That modification is justified only after the known stock general-rig route has been measured and rejected.

### C. 3D Gen Studio TokenRig

Keep it in the comparison set because Studio already integrates it well, including rig transfer, naming templates and animation workflow. But do not make it the first 6 GB runtime target: the stock implementation loads the model directly on CUDA and upstream SkinTokens documents a much larger VRAM requirement.

A bounded compatibility/load test is useful; repeated OOM tuning is not, unless upstream adds an actual low-VRAM mode.

### D. Animation

Use **3D Gen Studio's existing retarget system** before authoring new animation code. Validate at least idle, walk and one high-motion clip.

For the game deliverable, use Unreal's existing IK Retargeter/Auto Retarget tools rather than duplicating retarget mathematics in Python.

### E. Segmentation

Do not segment organic characters by default.

Known tools can be benchmarked only when required by a visible failure:

- Studio/ComfyUI SAM3 for 2D source masks;
- existing 3D segmentation tools for accessory isolation if deformation QA proves the need.

No new semantic-mesh segmenter is to be built before that point.

## CPU-only planning already implemented

`src/lowvram3d/rigging_policy.py` and `workers/rig_backend_preflight.py` remain deliberately lightweight. They do not implement rigging; they choose/probe known backends without loading torch or CUDA.

Current default policy:

- humanoid -> stock MIA first, stock UniRig fallback;
- general organic creature -> stock UniRig first;
- Puppeteer -> explicit later experiment only;
- mechanical -> existing rigid hierarchy;
- static -> no rig.

## Promotion gates

A written FBX/GLB is not success. Promotion requires:

- armature present;
- skin weights present;
- materials preserved;
- measured peak memory within the configured ceiling when available;
- required deformation poses visually pass;
- animation retarget works;
- final Blender raster review passes.

Visual rejection overrides automatic metrics.

## Current CI note

The first PR-wide hosted test run failed during collection because both `tests/test_core.py` and `tests/scene_pipeline/test_core.py` import as `test_core`. That failure is outside this lane's four files and is not evidence against the rigging policy. The parallel session owns the pytest/conftest area, so this branch does not fix it by collision.

## Next action

1. Sync this branch with the newest smoke commit.
2. Keep the new CPU-only policy/preflight.
3. Do **not** merge a custom MIA runtime adapter yet.
4. On the target machine, run the stock ComfyUI-UniRig MIA graph on one proven humanoid while respecting the shared GPU lock.
5. Produce the five-pose Blender contact sheet and memory receipt.
6. Only if stock MIA is rejected, test stock UniRig.
7. Only after stock known routes are rejected should we patch/build a better backend.
