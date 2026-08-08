# Vendor-first rig runtime ladder

This lane does not replace the working TRELLIS/Hunyuan/Blender foundation and does not reimplement a rigging model before upstream code is measured.

## Phase 0 — environment proof (no CUDA)

Run `tools/rig_vendor_preflight.ps1`. It only locates ComfyUI, its Python environment, and an existing `ComfyUI-UniRig` install. It does **not** import torch, initialize CUDA, install packages, download weights, or mutate the managed 3D Gen Studio ComfyUI.

The expected upstream baseline is pinned in `integrations/vendor/ComfyUI-UniRig/VENDOR_LOCK.json`:

- repo: `PozzettiAndrea/ComfyUI-UniRig`
- commit: `69ee59dc459d2da7cb0291930c1f944886c31d7c`
- first humanoid workflow: upstream `workflows/mia_humanoid.json`
- first animation workflow: upstream `workflows/apply_animation.json`

Exact copies of those two upstream workflow files are stored beside the lock so later changes can be distinguished from the stock baseline.

## Phase 1 — stock vendor asset

Only after preflight says the stock pack is available:

1. run upstream `mia_humanoid.json` unchanged on upstream `assets/realistic_male_character.glb`;
2. use FP16 on the target card, changing only the workflow precision widget, not model code;
3. measure wall time, peak dedicated VRAM, peak shared GPU/system spill, output armature, skin weights, and material preservation;
4. render deformation evidence before calling the backend usable;
5. run upstream `apply_animation.json` only after the rig passes deformation QA.

The vendor asset is intentional: it separates installation/model-runtime failure from a failure caused by our generated topology.

## Phase 2 — our production humanoid

If and only if Phase 1 passes, rerun the same stock MIA path on a preserved production humanoid from the current image-to-3D chain:

`TRELLIS -> native Stage 6 -> Hunyuan Paint -> Blender QA -> stock MIA`

Do not segment the body first. Segmentation is a recovery operation only if deformation QA proves weight bleed or a rigid accessory must be isolated.

## Failure ladder

- Stock MIA fails on vendor asset: diagnose installation/hardware/backend before touching our mesh.
- Stock MIA passes vendor asset but fails our humanoid: diagnose pose/topology/material handoff; then test stock UniRig.
- Stock UniRig fails on the same asset: only then justify patched/alternative riggers such as Puppeteer+SDPA.
- Stock animation retarget fails after a valid rig: test 3D Gen Studio/Unreal retargeting before custom animation math.

All GPU runs must serialize through `%LOCALAPPDATA%/LowVRAM3DStudio/locks/gpu.lock` so rig testing cannot collide with TRELLIS, Hunyuan Paint, or Blender GPU work.
