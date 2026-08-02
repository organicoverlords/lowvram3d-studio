# Codex execution handoff — six basic textured benchmark models

## User-defined milestone

The first production milestone is now:

> Produce one **basic textured, non-animated, fresh-importable model for each of the six source pictures** before rigging, animation, advanced retopology, LOD work, material refinement, or further infrastructure work.

A geometry-only GLB does not satisfy this milestone.

## Role boundary

Codex/Luna is the bounded executor. Do not design a new engine, telemetry system, asset database, folder hierarchy, profile framework, or orchestration layer. Use the existing pipeline and the already-proven telemetry implementation.

## Authoritative source folder

Use only:

`C:\Users\Lauri\Downloads\benchmarkpics`

Enumerate only immediate image files in this exact folder. Expected count: exactly six usable images.

Do not search the whole computer. Do not scan the moving models collection. Do not move, rename, delete, deduplicate, recompress, or edit the originals.

If there are not exactly six usable images, record the filenames and stop source-pack promotion with:

`BLOCKED_SIX_SOURCE_COUNT_MISMATCH`

## Stable benchmark identities

Map by visual content, not filename:

1. `frog_salvage_diver`
   - anthropomorphic frog in mud-stained salvage/diving equipment;
   - backpack/tank, hoses, hanging canisters, bag and lantern.

2. `tactical_red_panda_scout`
   - red panda in ghillie/tactical clothing;
   - rifle, backpack and large tail.

3. `lucky_drown_casino_boat`
   - side-view dark multi-deck casino riverboat;
   - `LUCKY DROWN` sign, paddlewheel, decks, railings and stairs.

4. `antlered_bird_shaman_anchor`
   - tall robed bird-mask shaman;
   - wide antler-like structure, hanging charms, staff and layered robes.

5. `wind_bent_barn_and_trees`
   - static weathered barn beside large wind-shaped trees;
   - grass and stormy rural context;
   - nothing walking or mechanical.

6. `mossy_mountain_titan`
   - enormous quadruped made of rock, bark, roots and moss;
   - mountain back, rib-like opening and four massive legs.

If identity is ambiguous, generate one labelled six-thumbnail contact sheet and stop with:

`USER_REVIEW_REQUIRED_SOURCE_IDENTITY`

Do not invent extra identities.

## Preserve the exact six sources in GitHub first

Create exactly one folder, not six folders:

`benchmarks/source-images/20260802-six-variation-pack/`

Required files:

- `01_frog_salvage_diver.<original-extension>`
- `02_tactical_red_panda_scout.<original-extension>`
- `03_lucky_drown_casino_boat.<original-extension>`
- `04_antlered_bird_shaman_anchor.<original-extension>`
- `05_wind_bent_barn_and_trees.<original-extension>`
- `06_mossy_mountain_titan.<original-extension>`
- `manifest.json`
- `README.md`
- `contact_sheet.png`

Copy exact image bytes. Do not recompress canonical repository copies.

For each file, `manifest.json` records:

- stable ID;
- repository filename;
- original filename and path;
- MIME type and extension;
- bytes;
- width, height and channels;
- alpha presence;
- original SHA-256;
- copied SHA-256;
- exact equality;
- intended profile;
- forbidden routes.

Commit and push the source pack before GPU work. Verify remote equality and a clean worktree.

## Definition: basic textured non-animated baseline

Each of the six assets must end with:

1. a real non-empty GLB;
2. geometry that fresh-imports in a clean Blender process;
3. at least one UV map or another valid texture-coordinate route;
4. at least one image texture bound to a material used by visible mesh faces;
5. visible base colour in Blender, not an unbound source image;
6. no armature required;
7. no animation actions required;
8. source-match render;
9. front, rear, left, right and upper-three-quarter renders;
10. one contact sheet;
11. a report with geometry, texture, import, runtime and VRAM facts.

The baseline may use simple projected colour and simple synthesized unseen-side fill. It is not yet required to be game-ready, fully PBR, rigged, animated, perfectly retopologized, or production-final.

Do not classify texture as proven when it is:

- entirely white, black or transparent;
- not connected to the imported material;
- only a viewport background;
- badly dominated by the source background;
- missing after fresh import;
- stored only in an external temporary path without being packaged or copied beside the deliverable.

Required classifications per asset:

- `SOURCE_STAGING`
- `GEOMETRY_BASELINE`
- `UV_OR_COORDINATE_ROUTE`
- `BASE_COLOR_TEXTURE`
- `MATERIAL_BINDING`
- `GLB_WRITE`
- `FRESH_BLENDER_IMPORT`
- `TEXTURED_CONTACT_SHEET`
- `BASIC_TEXTURED_BASELINE`

Use `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED` with exact reason codes.

## Minimal baseline route

For a source without reusable geometry:

1. immutable source staging by SHA-256;
2. background/alpha preparation only as already implemented;
3. one fast geometry generation attempt using the existing 6 GB-compatible Mini Turbo route;
4. default initial envelope: `steps=1`, `octree=256`, existing compatible chunk setting and seed;
5. fresh Blender import of geometry;
6. preserve existing UVs when valid, otherwise one conservative Blender auto-unwrap;
7. create one basic source-projected base-colour texture;
8. fill unseen areas with a simple local colour/material prior rather than mirrored source pixels labelled as observed;
9. bind the texture to the visible material;
10. export textured GLB;
11. fresh-import in a second clean Blender process;
12. render and classify.

Do not begin rigging or animation.

## Reuse policy to avoid wasting GPU time

Reuse a prior model only when its source identity and artifact hash are proven and it fresh-imports.

### Shaman

Do not regenerate Shaman geometry. Use the best existing canonical clean geometry and the best existing basic texture candidate as the starting point. Produce a fresh-imported non-animated textured baseline and classify its visible limitations. Do not retry sleeve rigging or garment separation.

The sleeve lane remains:

`BLOCKED_SHAMAN_SLEEVE_REQUIRES_MANUAL_RETOPOLOGY`

That blocker does not prevent a non-animated textured baseline.

### Barn

Do not rerun the expensive five-step barn workload.

Prefer the best proven real one-step barn GLB already written by the reduced canaries, especially octree 256 or 384 when source identity is verified. Apply the basic texture stage to that existing geometry.

Retained diagnostic state:

- one-step reduced canaries: `PROVEN_REDUCED_CANARIES`;
- steps=2 diffusion: `PROVEN_BY_TRACE`;
- steps=2 mesh decode: `PROVEN_IN_MEMORY`;
- steps=2 GLB write: `NOT_PROVEN_LOGGER_ABORT`;
- original steps=5 cause: `NOT_PROVEN_DEFERRED`;
- another expensive barn generation: `NOT_AUTHORIZED`.

A short texture/import pass on an existing proven GLB is authorized. Another long geometry run is not.

## Production order

Run one GPU-heavy process at a time.

1. `tactical_red_panda_scout`
2. `frog_salvage_diver`
3. `mossy_mountain_titan`
4. `lucky_drown_casino_boat`
5. `antlered_bird_shaman_anchor` using existing geometry
6. `wind_bent_barn_and_trees` using the best existing proven one-step geometry

This order should yield visible textured character results early while avoiding another long barn generation.

## Asset-specific constraints

### Tactical red panda

- route as equipped nonhuman humanoid/character shell;
- preserve rifle and backpack as rigid visual parts;
- preserve large tail silhouette;
- no rig or animation in this milestone.

### Frog diver

- route as equipped nonhuman humanoid;
- preserve tank, lantern and canisters;
- preserve hoses and hanging bag where geometry supports them;
- no rig or animation.

### Mountain titan

- route as massive quadruped/creature shell;
- preserve four legs, head, mountain back and rib opening;
- no humanoid pose normalization;
- no rig or animation.

### Casino boat

- route as vehicle/building static asset;
- preserve hull, superstructure, paddlewheel and sign silhouette;
- no humanoid rigging;
- no animation.

### Shaman

- use existing geometry;
- basic texture and material binding only;
- no sleeve/rig retry;
- no animation.

### Barn and trees

- use existing proven one-step geometry when available;
- static scene only;
- no rig or animation;
- reject sky becoming solid hero geometry or permanent barn/tree material contamination.

## Time and retry policy

The goal is six viewable baselines, not perfection on the first asset.

- one initial geometry attempt for each new asset;
- one texture attempt for each available geometry candidate;
- a failure blocks only that asset and stage;
- continue to the next asset unless a shared defect affects all routes;
- no repeated unchanged attempts;
- one corrected retry only after a concrete local correction;
- do not spend more than approximately 15 minutes of GPU time on one initial baseline without an explicit progress checkpoint and evidence that artifact completion is near.

## Stop infrastructure churn

Do not perform additional work on:

- telemetry architecture;
- logger redesign;
- generic profile engines;
- whole-library pairing;
- broad refactors;
- unrelated documentation;
- manual-retopology automation;
- animation systems.

Use telemetry from commit:

`91c454f0b9ad5f282940149234f4852913817582`

## Per-asset output layout

Use one run root with one subfolder per asset; do not create hundreds of folders:

```text
<run-root>/
  source-pack/
  tactical_red_panda_scout/
  frog_salvage_diver/
  mossy_mountain_titan/
  lucky_drown_casino_boat/
  antlered_bird_shaman_anchor/
  wind_bent_barn_and_trees/
  overall_matrix.json
  overall_contact_sheet.png
```

Inside each asset folder keep only:

```text
source.json
geometry.glb
textured.glb
texture/
report.json
renders/
contact_sheet.png
logs/
```

Temporary worker directories may exist during execution but must not become the permanent benchmark structure.

## Source control and continuation

After source preservation and after each textured baseline:

1. preserve artifacts and logs;
2. commit and push proof/metadata and repository-safe artifacts;
3. verify local equals remote;
4. verify clean worktree;
5. continue automatically to the next asset.

Do not stop after geometry-only success. Continue that asset through basic texture, material binding, textured export, fresh import and contact sheet unless a concrete blocker occurs.

## Required progress matrix

Report after every asset:

| Asset | Source | Geometry | Texture bound | Textured GLB | Fresh import | Contact sheet | Runtime | Next |
|---|---|---|---|---|---|---|---|---|

The six-asset first milestone is complete only when every row has a basic textured non-animated baseline or an exact isolated blocker.