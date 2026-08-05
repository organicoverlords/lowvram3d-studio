# Unnamed image ↔ 3D model pairing workflow

## Current operating state

The user is still reorganizing the local models folder. The live folder is an **unstable dataset**.

Until the user explicitly says the move is complete, or creates a readiness marker, classify:

`MODELS_FOLDER_STABILITY=NOT_PROVEN_USER_STILL_MOVING_FILES`

During this period:

- do not begin a full inventory;
- do not render every model;
- do not rename, move, deduplicate, delete, or reorganize anything;
- do not report files missing from previously known paths;
- only implement/test the pairing tooling against fixtures or a read-only small sample outside the active move path.

The pairing run may start when either:

1. the user explicitly says the models move is finished; or
2. a file named `.lowvram3d-models-ready` exists at the selected models root.

Before a production pairing run, take two read-only snapshots at least five minutes apart. Require equal file count, aggregate bytes, and content identities for all files already completely readable. If they differ, classify the snapshot provisional and retry later rather than pairing against a moving target.

## Core rule: identity is content, not path

Every image and model gets a stable asset identity derived from content:

- SHA-256 of the complete file;
- byte count;
- file type;
- discovered current paths;
- prior paths when known;
- first-seen and last-seen timestamps within the inventory process.

A path move must update `current_paths` and `path_history`; it must not create a new logical asset when the SHA-256 is unchanged.

For packaged formats and ZIPs, record both package SHA-256 and hashes of safe extracted members. Do not modify the original package.

## Inventory outputs

Create under the run root:

- `pairing/inventory/images.jsonl`
- `pairing/inventory/models.jsonl`
- `pairing/inventory/packages.jsonl`
- `pairing/inventory/path_history.json`
- `pairing/inventory/duplicates.json`
- `pairing/inventory/near_duplicate_images.json`
- `pairing/inventory/current_tree.txt`
- `pairing/inventory/snapshot_a.json`
- `pairing/inventory/snapshot_b.json`
- `pairing/inventory/stability_report.json`

Do not include secret-bearing files or unrelated personal documents. Restrict scanning to explicitly configured asset roots and supported extensions.

## Supported image evidence

For each source image, record:

- SHA-256 and perceptual hash;
- dimensions and channels;
- alpha/background estimate;
- aspect ratio;
- dominant colour histogram;
- silhouette descriptor;
- edge-density descriptor;
- detected contact-sheet/panel layout;
- likely single subject versus multi-view sheet;
- optional local visual embedding when a configured model is available;
- filename/path tokens as weak evidence only.

Group exact and near-duplicate images. One concept may have several crops, backgrounds, upscales, or alternate views. Preserve these relationships rather than forcing a one-image/one-model assumption.

## Supported model evidence

For each 3D candidate, use Blender in background mode to create a non-destructive audit and standardized renders:

- import status and importer used;
- object, mesh, material, texture, armature, and action counts;
- dimensions and principal-axis orientation;
- vertices, triangles, connected components, boundaries, and non-manifold counts;
- embedded object/material/image names;
- animation/rig presence;
- texture paths and package relationships;
- source metadata when embedded;
- front, rear, left, right, top, bottom, and two three-quarter renders;
- silhouette masks for each rendered view;
- one contact sheet.

Normalize camera framing only. Do not alter model geometry, origin, materials, or textures in the source file.

Record render failures without excluding the model from other metadata matching.

## Pairing is ranked, many-to-many, and fail-closed

Do not pair solely by filename, folder adjacency, or modification time.

For every concept-image group and model, calculate independent evidence channels:

1. **Exact provenance evidence**
   - shared archive/folder manifest;
   - explicit source path in metadata/logs;
   - exact embedded name or job ID;
   - existing generation receipt.

2. **Temporal evidence**
   - image and model created close together;
   - nearby export/download package timestamps.
   - Treat as weak evidence because files may have been copied.

3. **Semantic/category evidence**
   - boat versus character versus quadruped versus prop/building;
   - major traits such as paddlewheel, shell, antlers, staff, backpack, hoses.

4. **Visual evidence**
   - source silhouette versus best matching model view;
   - colour/material distribution;
   - structural landmarks;
   - thin-feature correspondence;
   - multi-view consistency when the source is a concept sheet.

5. **Geometry evidence**
   - expected proportions and component layout;
   - articulated/static/rigged consistency;
   - known output characteristics of generation jobs.

6. **Advisory visual-model evidence**
   - optional and never authoritative by itself.

Return top candidates rather than forcing a match.

## Confidence policy

Each proposed pair must include:

- `image_group_id`
- `model_asset_id`
- `confidence`
- `evidence_by_channel`
- `contradictions`
- `runner_up_candidates`
- `review_required`
- `classification`

Use:

- `>= 0.92`: `PROVEN_HIGH_CONFIDENCE_PAIR` only when at least two independent strong evidence channels agree and no material contradiction exists;
- `0.80–0.919`: `PROVISIONAL_PAIR_REVIEW_REQUIRED`;
- `0.55–0.799`: `AMBIGUOUS_TOP_CANDIDATES`;
- `< 0.55`: `UNPAIRED`.

A visual embedding score alone cannot produce `PROVEN_HIGH_CONFIDENCE_PAIR`.

Allow:

- one concept group → several generated models;
- several concept images → one model;
- model without a source image;
- image without a model;
- duplicate model exports in different formats.

## Pair-review artifacts

Create:

- `pairing/proposals/pairs.json`
- `pairing/proposals/unpaired_images.json`
- `pairing/proposals/unpaired_models.json`
- `pairing/proposals/ambiguous.json`
- `pairing/proposals/duplicate_exports.json`
- `pairing/review/index.html` or equivalent local static gallery;
- per-proposal contact sheets showing source image(s), candidate model views, confidence, evidence, contradictions, and runner-up candidates.

The user should review only ambiguous or consequential matches. High-confidence provenance-backed matches can be accepted automatically.

## Naming and organization policy

Pairing does not rename or move files.

After review, generate a separate proposal:

- `pairing/organization/proposed_names.json`
- `pairing/organization/proposed_moves.json`
- `pairing/organization/collision_report.json`
- `pairing/organization/undo_plan.json`

Every proposed name must be deterministic, filesystem-safe, collision-free, and retain the original extension.

Do not execute any rename or move without explicit user approval of that separate proposal.

## Priority assets

Before broad pairing, ensure the system can identify the benchmark targets among the unnamed pool:

- Lucky Drown casino boat;
- steampunk snapping turtle;
- frog salvage diver;
- shaman and its online/local variants.

These priority matches still require evidence and must not be guessed from recency.

## Implementation constraints

- Keep workers bounded and resumable.
- Use incremental manifests so interrupted scans do not restart hashing completed files.
- Cache model render results by file SHA-256 plus renderer version.
- Limit Blender processes and run them sequentially unless memory-safe proof exists.
- Never load every high-poly model into one Blender process.
- Enforce per-file timeout and continue other files after a failure.
- Log skipped, unreadable, corrupt, and unsupported files.
- The 6 GB GPU must not be used for pairing unless an optional visual model is explicitly configured; structural audits and Blender previews should default to CPU/Eevee-safe settings.

## Required tests

Include fixtures proving:

- same file at a new path retains identity;
- duplicate formats are linked, not forced into separate subjects;
- near-duplicate concept crops group correctly;
- one concept can rank multiple generated models;
- ambiguous pairs remain unpaired;
- filename-only matches cannot pass high confidence;
- interrupted inventory resumes without rehashing unchanged files;
- moving files during a scan causes a provisional snapshot rather than false missing-file claims;
- no source file is modified.
