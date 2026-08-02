# Pipeline execution now — organization deferred

## User override

The user has explicitly asked to continue the 3D pipeline now instead of spending more time organizing the entire pictures/models library.

Therefore:

- do not design or execute broad library sorting now;
- do not wait for the whole models folder to become stable;
- do not inventory or render the entire collection;
- do not move, rename, deduplicate, or delete user files;
- do only bounded read-only discovery for required benchmark inputs;
- stage selected benchmark inputs by copying them to an immutable benchmark run directory;
- work from staged copies while the user continues reorganizing originals;
- defer the broad unnamed-asset pairing workflow until it is needed again.

## Required five-asset benchmark

1. `antlered_bird_shaman_anchor`
2. `lucky_drown_casino_boat`
3. `steampunk_snapping_turtle`
4. `frog_salvage_diver`
5. `giant_tripod_vending_machine_walker`

The fifth case is the user-provided image of a giant green vending-machine/kiosk body on three long mechanical legs with industrial feet, rusted attachments, signs, cables/hoses, and a camera/light/sensor cluster above it. Read `docs/BENCHMARK_ADDENDUM_TRIPOD_VENDING_WALKER_20260802.md`.

## Safe bounded discovery despite a moving folder

For each non-shaman benchmark:

1. Search only likely source roots and candidate-name/visual descriptors.
2. Do not crawl every unrelated file on the machine.
3. When a likely source image or online model is found, compute SHA-256 and copy it to:

   `C:\AI\LowVRAM3D-benchmarks\multi-asset-validation\run-<timestamp>\staged_inputs\<asset-id>\`

4. Record original path, staged path, bytes, hash, modification time, and selection evidence.
5. Re-hash the staged copy and require exact equality.
6. Never modify the original.
7. If the original moves later, continue using the verified staged copy.
8. If several candidates remain plausible, create a small candidate sheet and classify only that asset `USER_REVIEW_REQUIRED`; continue unambiguous assets.

Do not let uncertain model pairing block source-image pipeline smoke tests. A missing online reference blocks only the comparison lane, not local generation/audit.

## Execution order

### Phase A — cheap preflight

- verify repository, branch, local/remote head, and worktree;
- stage the shaman's existing canonical inputs;
- bounded discovery and staging for casino boat, turtle, frog diver, and tripod vending walker;
- implement/finish automatic profile discovery;
- run structural source audits and source contact sheets;
- create the five-row benchmark matrix.

### Phase B — shared pipeline proof

Run one GPU-heavy job at a time:

1. Casino boat smoke test — noncharacter vehicle/building routing.
2. Tripod vending walker smoke test — articulated hard-surface routing and three-leg detection.
3. Turtle smoke test — quadruped routing.
4. Frog-diver smoke test — equipped nonhuman humanoid routing.
5. Shaman V3 regression — one bounded sleeve core/drape repair and rerender.

This order tests broad routing early while retaining the shaman as a mandatory regression anchor.

### Phase C — comparison

For every asset with an online reference, compare staged online model against local candidate. Missing references do not block other work.

## Immediate shaman task

The old torso/rear/side-cape bleed was fixed, but V2 still has a rigid triangular sleeve panel.

Perform exactly one V3 attempt:

- separate anatomical arm/hand core from sleeve anchor/drape;
- add triangle-boundary and deformation diagnostics;
- add a small damped sleeve-drape chain;
- prevent broad drape from direct hand weighting;
- rerender isolated tests, Milestone B, and wave;
- same rigid panel after corrected attempt means `BLOCKED_FUSED_SLEEVE_TOPOLOGY_REQUIRES_GARMENT_SEPARATION`.

Do not repeatedly tune radii or reduce motion to hide the defect.

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
- `HARD_BLOCKER`
- `PROJECT_DONE`
- `TASK_DONE_NO_NEXT_ACTION`
