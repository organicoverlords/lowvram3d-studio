# Codex: continue here

Work only on branch `magicmusic/parts-pose-materials-20260802`.

Before editing, verify repository origin, branch, local HEAD, remote HEAD, and a clean worktree. Never work on `main`, merge, or force-push.

## Immediate authoritative task

Read and execute:

`docs/TEXTURE_COMPLETION_PIPELINE_V2_20260802.md`

The rear-face projection defect is fixed. The next task is to improve side/rear texture quality without weakening the depth/normal visibility gate or overwriting observed front texels.

## Benchmark input

Use the existing **CPU-sanitized 5-step tactical red panda geometry** first.

Do not:

- regenerate geometry;
- modify the immutable raw 4-step/5-step GLBs;
- modify repaired v7;
- run frog;
- restart MV-Adapter debugging;
- install Hunyuan3D-Paint, Paint3D, or MVPaint;
- add another framework.

## Required implementation order

1. Enforce a real 1024 atlas-resolution contract. The current route must not request 1024 and silently emit 512.
2. Restore the production xatlas/UV-quality route and require the exact overlap gate plus >=55% utilization on the panda benchmark.
3. Replace direct Euclidean unseen-surface donor fill with welded surface-graph low-frequency material propagation.
4. Add deterministic region-aware material priors and per-texel provenance.
5. Produce a CPU-only improved panda comparison before adding diffusion.
6. Add KEEP / REFINE / GENERATE trimaps and coverage-driven next-best-view selection.
7. Use only an already-installed low-VRAM inpainting backend, one 512 view at a time, with geometry control and source-image reference conditioning. If unavailable, report the optional backend blocker after completing the CPU route. Do not download a large model automatically.
8. Back-project only accepted GENERATE pixels through depth, normal, occlusion, triangle-ID, and trimap ownership gates.
9. Add robust colour harmonization, UV seam leveling, and a final native-1024 merge that restores observed source detail.
10. Fresh-import and render front / three-quarter / side / rear. Report observed versus generated coverage, runtime, VRAM, and all gates.

## Acceptance

The sanitized 5-step result must:

- keep geometry unchanged;
- preserve observed front texels within tolerance;
- keep the rear face absent;
- improve side/rear material separation for tail, ghillie, backpack, cloth, and rifle over the neutral-fill baseline;
- have no black holes or cross-component colour bleeding;
- pass fresh Blender import;
- produce provenance/debug atlases and a labelled contact sheet.

Commit shared code/tests first, then run the sanitized 5-step benchmark. Only after it passes may the same route be applied to sanitized 4-step.

Push the validated commit, verify local equals remote, and leave a clean worktree.