# Sol architects; Luna executes

## Authoritative model-role split

This project must not delegate architecture to GPT-5.6 Luna Medium.

### GPT-5.6 Sol XHigh owns

- pipeline architecture;
- stage definitions and ordering;
- benchmark design;
- asset-class routing rules;
- failure policy and retry limits;
- acceptance gates;
- topology, rigging, texturing, animation, export, and comparison strategy;
- design of any new engine, worker family, profile system, data model, or orchestration layer;
- review of evidence and promotion decisions;
- deciding when a local defect requires a new architecture rather than a bounded patch.

### GPT-5.6 Luna Medium may do only bounded execution

- verify repository identity, branch, local/remote SHA, and clean worktree;
- read and follow an already-committed specification;
- implement a narrowly specified function, worker, test, or report;
- run existing scripts, Blender workers, tests, renders, and audits;
- collect metrics and visible artifacts;
- make one exact local correction when the failure cause is already identified and the correction does not change architecture;
- commit, push, verify local equals remote, and update receipts;
- report `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED`.

Luna must not:

- invent or redesign an engine;
- design a new pipeline stage family;
- redesign profile discovery;
- reinterpret benchmark images or rename benchmark identities;
- introduce broad abstractions, plugin systems, or orchestration frameworks;
- refactor unrelated systems;
- choose a new model stack or dependency strategy;
- continue after a specification becomes ambiguous;
- hide a failure by weakening gates or reducing motion/quality.

When execution requires architecture, Luna must stop that lane with:

`WAIT_FOR_SOL_ARCHITECTURE`

and return:

- exact current stage;
- exact blocker;
- relevant files and lines;
- evidence already collected;
- the smallest architectural question Sol must answer.

## Missing benchmark pictures

The benchmark pictures originally uploaded in ChatGPT are not automatically visible to Codex/Luna on the Windows filesystem. Luna must not respond by designing a discovery or pairing engine.

Use this canonical Windows staging directory:

`C:\AI\LowVRAM3D-benchmarks\benchmark-inputs\`

Expected staged filenames:

- `lucky_drown_casino_boat.png`
- `steampunk_snapping_turtle.png`
- `frog_salvage_diver.png`
- `barn_with_trees.png`

Execution rule:

1. Look only for these exact staged files first.
2. When present, compute SHA-256, record bytes and dimensions, and copy them into the active immutable run directory.
3. Do not rename or move the staged originals.
4. When one is absent, classify only that asset:

   `BLOCKED_SOURCE_FILE_NOT_STAGED`

5. Continue any other asset whose staged source exists.
6. Do not crawl the whole machine, build a pairing system, or infer a different source image.

## Current work order

1. Preserve the Shaman manual-retopology blocker; do not retry its automatic sleeve lane.
2. Run the Lucky Drown boat smoke test when its staged picture exists.
3. Run the barn-with-trees static-scene smoke test when its staged picture exists.
4. Run the turtle smoke test when its staged picture exists.
5. Run the frog-diver smoke test when its staged picture exists.
6. Rerun only proven Shaman regression gates after shared code changes.

One GPU-heavy task at a time. No architecture work by Luna.
