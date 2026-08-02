# Codex: continue here

Work only on branch `magicmusic/parts-pose-materials-20260802`.

Before editing, verify repository origin, branch, local HEAD, remote HEAD, and a clean worktree. Never work on `main`, merge, or force-push.

## Current authoritative state

The Shaman automatic sleeve lane has reached its final bounded result:

`BLOCKED_SHAMAN_SLEEVE_REQUIRES_MANUAL_RETOPOLOGY`

Commit containing the garment-separation proof:

`a5bf67f463557eea259c50fb1c8fe3bf1b25e25b`

Do not run another automatic sleeve extraction, mask-radius adjustment, or weighting retry. Preserve the Shaman as a mandatory regression anchor and resume that sleeve only in a separately approved manual-retopology lane.

The user's broader models folder may still be moving. Do not inventory or reorganize the whole library. Use only bounded read-only discovery for the required benchmark inputs, copy selected inputs into a content-hashed immutable staging directory, and work from those staged copies.

Read these files in order:

1. `docs/PIPELINE_EXECUTION_NOW_20260802.md`
2. `configs/benchmarks/multi_asset_validation_20260802.json`
3. `docs/BENCHMARK_ADDENDUM_BARN_AND_TREES_20260802.md`
4. `docs/CODEX_MULTI_ASSET_CONTINUATION_20260802.md`
5. `proof/shaman-rig/latest/progress.json`

## Required benchmark matrix

1. Shaman regression anchor — sleeve blocked pending manual retopology.
2. Lucky Drown casino boat — vehicle/building route.
3. Steampunk snapping turtle — quadruped creature route.
4. Frog salvage diver — equipped nonhuman humanoid route.
5. Barn with trees — static building/environment route; nothing is walking or mechanical.

## Immediate execution order

1. Verify and preserve the Shaman blocker proof.
2. Implement or finish automatic profile discovery.
3. Perform bounded discovery and immutable staging for the casino boat, barn with trees, turtle, and frog diver.
4. Run the casino-boat smoke test.
5. Run the barn-and-trees static-scene smoke test with no rig or animation.
6. Run the turtle smoke test.
7. Run the frog-diver smoke test.
8. After shared pipeline changes, rerun only the proven Shaman regression gates; do not retry its blocked sleeve.

Run one GPU-heavy process at a time.

Do not claim a stage complete from code/tests alone. Execute Blender workers and produce visible artifacts. Classify every claim as `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED`.
