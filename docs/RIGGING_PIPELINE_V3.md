# Rigging Pipeline V3 — low-VRAM integration lane

Status: **IMPLEMENTATION STARTED / GPU RUNTIME NOT PROVEN**

Branch baseline: `agent/scene-pipeline-smoke-20260803` at `a82793de3c859369e5dedfad9546d7a605c703ea`.

This lane replaces the assumption that every organic mesh should be split and then passed to the hand-authored `rig_animate.py` route.  It keeps 3D Gen Studio as the control layer, but treats neural riggers as isolated backends with one shared receipt and promotion contract.

## Required ordering

For an organic asset after texture completion:

```text
textured LOD0 (preserve byte-for-byte source copy)
  -> select rig backend without loading a model
  -> rig + skin
  -> static rig QA
  -> five-pose deformation QA
  -> animation retarget
  -> engine export
  -> skeletal LOD generation
```

The old ordering `LOD0/LOD1/LOD2 -> rig LOD0` is not the target architecture for skeletal characters.  Until weight transfer between independently generated LOD meshes is proven, lower skeletal LODs are generated only after a valid rigged LOD0 exists (preferably in Unreal for the game deliverable).

Semantic segmentation is **not** a mandatory organic pre-rig stage.  It is invoked only when deformation evidence proves weight bleed or when a rigid accessory needs isolation.

## Backend policy

### Humanoid / avatar

Primary: **Make-It-Animatable (MIA)**

- FP16 first.
- Preserve original mesh/materials.
- Reset to rest pose only in a derived output; never overwrite the textured source.
- Mixamo-compatible skeleton for the first benchmark.
- Fallback: UniRig, explicitly recorded.  No silent backend substitution.

### General organic creature

Primary experimental backend: **Puppeteer**

- Run skeleton and skinning sequentially.
- The GTX 1660 SUPER lane requires an SDPA/eager-compatible attention path; do not assume FlashAttention-2.
- Published upstream memory numbers justify a bounded 6 GB experiment, but local peak VRAM is **NOT PROVEN** until measured on the target card.
- Fallback: UniRig, explicitly recorded.

### Mechanical / rigid multipart

Keep the existing rigid hierarchy route.  Do not spend neural skinning memory on a machine or vehicle that can be animated as rigid parts.

### Static asset

No rig.

## Promotion gates

A neural rig is not promoted merely because an FBX/GLB was written.  Machine-readable evidence must prove:

- armature present;
- skin weights present;
- materials preserved;
- peak VRAM does not exceed the configured ceiling when peak measurement is available;
- rest pose passes;
- elbow bend passes;
- knee bend passes;
- hip crouch passes;
- shoulder raise passes.

A human reviewer may still reject a machine-pass result.  Visual rejection overrides automatic metrics.

## CPU-only preflight

`workers/rig_backend_preflight.py` chooses a backend and checks configured source/weight roots without importing torch, initializing CUDA, downloading weights or starting inference.

Example shape:

```text
python workers/rig_backend_preflight.py \
  --asset-type avatar \
  --rig-kind humanoid \
  --mia-root <path> \
  --puppeteer-root <path> \
  --unirig-root <path> \
  --output <job>/reports/rig_backend_preflight.json
```

The report is `READY` only when the selected backend is locally present.  It never silently chooses a different backend when the selected one is unavailable.

## Implemented in the first slice

- `src/lowvram3d/rigging_policy.py`
  - deterministic backend selection;
  - SDPA requirement for Puppeteer;
  - segmentation-as-recovery policy;
  - post-rig skeletal LOD ordering;
  - common fail-closed promotion evaluator.
- `workers/rig_backend_preflight.py`
  - CPU-only availability check;
  - no model imports/downloads/CUDA work.
- `tests/test_rigging_policy.py`
  - policy and promotion-gate coverage.

## Deliberately not claimed yet

- MIA inference on the target GTX 1660 SUPER: **NOT PROVEN**.
- Puppeteer with SDPA on sm75: **NOT PROVEN**.
- UniRig fallback on 6 GB: **NOT PROVEN**.
- five-pose Blender deformation contact sheet: **NOT IMPLEMENTED in this first slice**.
- 3D Gen Studio graph/MCP wiring for these isolated rig services: **NOT IMPLEMENTED in this first slice**.
- Unreal skeletal LOD proof: **NOT STARTED**.

## Next bounded implementation slice

1. Add an isolated MIA runtime adapter that consumes a textured GLB and emits a rigged GLB/FBX plus a receipt, without mutating the input.
2. Add Blender structural QA and deterministic five-pose render generation.
3. Run one humanoid FP16 benchmark under the 5,600 MB ceiling and record peak VRAM.
4. Only after MIA is graded, add Puppeteer with an explicit SDPA/eager attention switch and run one creature benchmark.
5. Wire the proven backend into the main postprocess route; leave the old `rig_animate.py` path as mechanical/fallback behavior rather than the organic default.
