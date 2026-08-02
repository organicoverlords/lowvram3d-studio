# Pipeline execution now — organization deferred

## User override

Continue the 3D pipeline now. Do not spend the current execution window organizing the entire pictures/models library.

Therefore:

- do not inventory or render the entire collection;
- do not move, rename, deduplicate, or delete user files;
- do only bounded read-only discovery for required benchmark inputs;
- stage selected benchmark inputs by copying them to an immutable benchmark run directory;
- work from staged copies while originals may still move;
- defer broad unnamed-asset pairing.

## Required five-asset benchmark

1. `antlered_bird_shaman_anchor`
2. `lucky_drown_casino_boat`
3. `steampunk_snapping_turtle`
4. `frog_salvage_diver`
5. `barn_with_trees_static_scene`

The fifth case is exactly a **static barn with trees**. Nothing is walking or mechanical. Read `docs/BENCHMARK_ADDENDUM_BARN_AND_TREES_20260802.md`.

## Current Shaman status

The automatic sleeve lane is complete and blocked:

`BLOCKED_SHAMAN_SLEEVE_REQUIRES_MANUAL_RETOPOLOGY`

Proven:

- fresh Blender import;
- wrist and elbow separation from the sleeve drape;
- normalized weights and no fallback vertices from the preceding V3 work;
- preserved blocker evidence and artifacts.

Rejected:

- body closure: 1,834 boundary edges, 13 non-manifold edges, 0 closure faces;
- garment motion: 104 flipped triangles and 260 extreme stretch/shear triangles.

Do not run another automatic sleeve extraction, mask-radius adjustment, or weighting retry. Preserve the Shaman as a regression anchor. Resume its sleeve only in a separately approved manual-retopology lane.

## Safe bounded discovery despite a moving folder

For each non-shaman benchmark:

1. Search only likely source roots and candidate-name/visual descriptors.
2. Do not crawl every unrelated file on the machine.
3. When a likely source image or online model is found, compute SHA-256 and copy it to:

   `C:\AI\LowVRAM3D-benchmarks\multi-asset-validation\run-<timestamp>\staged_inputs\<asset-id>\`

4. Record original path, staged path, bytes, hash, modification time, and selection evidence.
5. Re-hash the staged copy and require exact equality.
6. Never modify the original.
7. Continue from the verified staged copy even if the original later moves.
8. If several candidates remain plausible, create a small candidate sheet and classify only that asset `USER_REVIEW_REQUIRED`; continue unambiguous assets.

A missing online reference blocks only the comparison lane, not local generation or audit.

## Execution order

### Phase A — cheap preflight

- verify repository, branch, local/remote head, and worktree;
- preserve the existing Shaman proof and regression inputs;
- bounded discovery and staging for casino boat, turtle, frog diver, and barn-with-trees scene;
- implement or finish automatic profile discovery;
- run structural source audits and source contact sheets;
- create the five-row benchmark matrix.

### Phase B — shared pipeline proof

Run one GPU-heavy job at a time:

1. Casino boat smoke test — noncharacter vehicle/building routing.
2. Barn-and-trees smoke test — static building/environment routing with no rig or animation.
3. Turtle smoke test — quadruped routing.
4. Frog-diver smoke test — equipped nonhuman humanoid routing.
5. Rerun only the proven Shaman regression gates after shared pipeline changes; do not retry the blocked sleeve.

### Phase C — comparison

For every asset with an online reference, compare the staged online model against the local candidate. Missing references do not block other work.

## Required behavior

- no manual profile choice in normal operation;
- no concurrent GPU-heavy tasks on the 6 GB card;
- preserve completed stages;
- maximum two corrected attempts per failing stage;
- do not rerun unchanged failures;
- classify all claims as `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED`;
- Blender-visible artifacts are required; code/tests alone are not completion;
- after every bounded validated milestone: test, update proof, commit, push, verify local equals remote, verify clean worktree.

Stop only for:

- `USER_REVIEW_REQUIRED`
- `WAIT_FOR_USER_DECISION`
- `HARD_BLOCKER` affecting all routes
- `PROJECT_DONE`
- `TASK_DONE_NO_NEXT_ACTION`
