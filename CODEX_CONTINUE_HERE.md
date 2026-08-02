# Codex: continue here

Work only on branch `magicmusic/parts-pose-materials-20260802`.

Before editing, verify repository origin, branch, local HEAD, remote HEAD, and a clean worktree. Never work on `main`, merge, or force-push.

## Important local-state change

The user reorganized the local models folder after the earlier handoff was written and is **still moving files now**.

Treat every previously recorded non-canonical local model path as stale until rediscovered. Do not report an asset missing merely because an old path no longer exists, and do not recreate the previous folder arrangement.

The live models folder is currently an unstable dataset:

`MODELS_FOLDER_STABILITY=NOT_PROVEN_USER_STILL_MOVING_FILES`

Until the user explicitly says the move is complete, or a `.lowvram3d-models-ready` marker exists at the selected root:

- do not run a full production inventory;
- do not render the entire model collection;
- do not make final image↔model pair claims;
- do not rename, move, deduplicate, delete, or reorganize user files;
- only implement and test pairing tooling against repository fixtures or a bounded read-only sample outside the active move path.

When production discovery becomes allowed:

1. enumerate the current directory structure under the user’s active models folders and configured search roots;
2. search by filename stem, extension, SHA-256, embedded object/material names, generated preview similarity, and package provenance—not by old absolute path;
3. use content hashes as stable identity so moving the same file updates path history instead of creating a new asset;
4. produce `inputs/current_models_tree.txt` and `inputs/path_reconciliation.json` mapping old paths to current candidates;
5. take two read-only snapshots at least five minutes apart and require stable file count, aggregate bytes, and identities before final pairing;
6. preserve the new organization exactly—read only, no moving, renaming, deduplicating, or deleting;
7. use discovered current paths in all new manifests and commands;
8. keep canonical shaman paths and hashes authoritative unless the files themselves fail verification.

There are many unnamed images and unnamed 3D models. They must be paired by evidence, not filename. Read and implement the dedicated many-to-many, fail-closed workflow in `docs/UNNAMED_ASSET_PAIRING_WORKFLOW_20260802.md`.

Read these files in order:

1. `docs/UNNAMED_ASSET_PAIRING_WORKFLOW_20260802.md`
2. `docs/CODEX_MULTI_ASSET_CONTINUATION_20260802.md`
3. `configs/benchmarks/multi_asset_validation_20260802.json`
4. `proof/shaman-rig/latest/progress.json`
5. `proof/shaman-rig/latest/CONTINUE_HERE.md` only as historical context; its final next-action section predates the independent V2 visual review.

Current priorities while the user is still moving files:

1. Keep the shaman as a mandatory regression anchor and perform one bounded V3 sleeve core/drape repair.
2. Implement/test the content-addressed inventory and image↔model pairing workers against fixtures only.
3. Do not begin final live-folder pairing until stability is proven.
4. Prepare automatic base-profile plus composable-trait discovery; do not ask the user to choose a profile.
5. Run GPU-heavy jobs one at a time on the 6 GB card.

Current priorities after the folder is stable:

1. Inventory and pair the user's unnamed source images and online-generated models.
2. Identify the Lucky Drown casino boat, steampunk snapping turtle, frog salvage diver, and shaman variants from evidence.
3. Build immutable input/reference manifests and standardized audits before expensive generation.
4. Compare local candidates against online reference models using metrics and contact sheets, not a single aggregate score.

The chosen boat benchmark is the existing **Lucky Drown casino boat**, not the ragged rowboat.

Do not claim a stage complete from code/tests alone. Execute Blender workers and produce visible artifacts. Classify claims as `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED`.
