# Codex: continue here

Work only on branch `magicmusic/parts-pose-materials-20260802`.

Before editing, verify repository origin, branch, local HEAD, remote HEAD, and a clean worktree. Never work on `main`, merge, or force-push.

## Immediate authoritative task

Read and execute first:

`docs/MINITURBO_PRODUCTION_QUALITY_FIX_20260802.md`

This supersedes every older instruction that allows Mini Turbo `steps=1` or `steps=2` output to become a textured or canonical baseline.

The pipeline-level fault is now identified:

- one-step and two-step runs were diagnostic shortcuts;
- they were incorrectly promoted into production geometry;
- the frog's debris halo is fused into the under-denoised main surface;
- detached-component cleanup cannot safely repair that geometry.

## Required implementation

1. Add explicit `smoke` versus `baseline` run intent.
2. Make smoke output non-promotable and block it from UV/texture/final stages.
3. Make Mini Turbo baseline intent use 5 inference steps, guidance 5.0, octree 256, and chunks 1500 on this machine.
4. Add the shared pre-texture geometry quality gate defined in the authoritative document.
5. Calibrate the gate so panda repaired v7 passes while frog v7 and v8 fail.
6. Do not texture geometry that fails the gate.
7. After CPU tests, run one new five-step frog candidate; permit one alternate fixed seed only if the first candidate fails.
8. Texture and fresh-import only a passing candidate.

Do not perform another frog-specific deletion repair.
Do not rerun the barn.
Do not change panda repaired v7.
Do not add another generator or framework.

The user-defined milestone remains one basic textured, non-animated, fresh-importable model for each of the six benchmark pictures.