# Codex: continue here

Work only on branch `magicmusic/parts-pose-materials-20260802`.

Before editing, verify repository origin, branch, local HEAD, remote HEAD, and a clean worktree. Never work on `main`, merge, or force-push.

## Important local-state change

The user reorganized the local models folder after the earlier handoff was written.

Treat every previously recorded non-canonical local model path as stale until rediscovered. Do not report an asset missing merely because an old path no longer exists, and do not recreate the previous folder arrangement.

At the start of discovery:

1. enumerate the current directory structure under the user’s active models folders and the configured search roots;
2. search by filename stem, extension, file hash, embedded object/material names, and preview similarity rather than by old absolute path;
3. produce `inputs/current_models_tree.txt` and `inputs/path_reconciliation.json` mapping any old known paths to current candidates;
4. preserve the new organization exactly—read only, no moving, renaming, deduplicating, or deleting;
5. use the discovered current paths in all new manifests and commands;
6. keep canonical shaman paths and hashes authoritative unless their files themselves fail verification.

Read these files in order:

1. `docs/CODEX_MULTI_ASSET_CONTINUATION_20260802.md`
2. `configs/benchmarks/multi_asset_validation_20260802.json`
3. `proof/shaman-rig/latest/progress.json`
4. `proof/shaman-rig/latest/CONTINUE_HERE.md` only as historical context; its final next-action section predates the independent V2 visual review.

Current priorities:

1. Keep the shaman as a mandatory regression anchor and perform one bounded V3 sleeve core/drape repair.
2. Discover the user's existing source concepts and online-generated models for the Lucky Drown casino boat, steampunk snapping turtle, and frog salvage diver from the current reordered folder structure.
3. Build immutable input/reference manifests and standardized audits before expensive generation.
4. Implement automatic base-profile plus composable-trait discovery; do not ask the user to choose a profile.
5. Run GPU-heavy jobs one at a time on the 6 GB card.
6. Compare local candidates against online reference models using metrics and contact sheets, not a single aggregate score.

The chosen boat benchmark is the existing **Lucky Drown casino boat**, not the ragged rowboat.

Do not claim a stage complete from code/tests alone. Execute Blender workers and produce visible artifacts. Classify claims as `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED`.
