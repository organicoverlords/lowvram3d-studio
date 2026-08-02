# Codex execution handoff — six staged benchmark sources

## Role boundary

Codex/Luna is the bounded executor. Do not design a new engine, telemetry system, asset database, folder hierarchy, profile framework, or orchestration layer. The architecture is already decided. Execute the steps below and produce actual benchmark artifacts.

## Authoritative source folder

Use only this folder for the six new benchmark pictures:

`C:\Users\Lauri\Downloads\benchmarkpics`

Do not search the whole computer. Do not scan the user's moving models library. Do not move, rename, delete, deduplicate, or edit the originals.

Enumerate only immediate image files in this exact folder. Expected count: exactly six usable images. If the count is not six, record the actual filenames and stop source-pack promotion with `BLOCKED_SIX_SOURCE_COUNT_MISMATCH`; do not guess which extras to delete.

## Stable benchmark identities

Map the six images by visual evidence, not by filename, to these stable IDs:

1. `frog_salvage_diver`
   - anthropomorphic frog in mud-stained salvage/diving equipment;
   - large backpack/tank, hoses, hanging canisters and lantern;
   - full-body frontal character.

2. `tactical_red_panda_scout`
   - red panda in ghillie/tactical clothing;
   - rifle and backpack;
   - full-body frontal character.

3. `lucky_drown_casino_boat`
   - side-view dark multi-deck casino riverboat;
   - large `LUCKY DROWN` sign;
   - paddlewheel, decks, railings and stairs.

4. `antlered_bird_shaman_anchor`
   - tall robed bird-mask shaman;
   - wide antler-like head structure with hanging charms;
   - staff, layered robes and bird feet.

5. `wind_bent_barn_and_trees`
   - static weathered barn beneath or beside large wind-shaped trees;
   - grass, storm clouds and rural landscape context;
   - nothing walking, mechanical or industrial.

6. `mossy_mountain_titan`
   - enormous quadruped made of rock, bark, roots and moss;
   - mountain-like back, rib-like side opening and massive column legs;
   - isolated on transparent or dark background.

If any image cannot be mapped unambiguously, make one six-thumbnail contact sheet labelled only A–F and stop with `USER_REVIEW_REQUIRED_SOURCE_IDENTITY`. Do not invent a seventh identity.

## Preserve the exact source pack in GitHub

Create exactly one repository folder:

`benchmarks/source-images/20260802-six-variation-pack/`

Do not create one directory per image.

Inside it create exactly:

- `01_frog_salvage_diver.<original-extension>`
- `02_tactical_red_panda_scout.<original-extension>`
- `03_lucky_drown_casino_boat.<original-extension>`
- `04_antlered_bird_shaman_anchor.<original-extension>`
- `05_wind_bent_barn_and_trees.<original-extension>`
- `06_mossy_mountain_titan.<original-extension>`
- `manifest.json`
- `README.md`
- `contact_sheet.png`

Copy exact bytes for the six image files; do not recompress or resize them for the canonical repository copies. The contact sheet may be generated separately.

`manifest.json` must record for every image:

- stable ID;
- repository filename;
- original local filename and path;
- extension and MIME type;
- byte size;
- width, height and channels;
- SHA-256 of original;
- SHA-256 of repository copy;
- equality result;
- alpha presence;
- concise visual identity;
- intended pipeline profile;
- forbidden routes.

Require source and copied SHA-256 equality before commit. Git LFS may be used only if already configured in this repository; do not introduce LFS architecture during this task. Normal Git storage is acceptable for these six files.

Commit and push this preservation milestone before any GPU work. Verify local SHA equals remote SHA and the worktree is clean.

## Stop infrastructure churn

Do not perform more work on:

- telemetry architecture;
- logger redesign;
- generic profile engines;
- whole-library pairing;
- broad refactors;
- README expansion unrelated to the six sources;
- barn CUDA reproduction;
- automatic Shaman sleeve repair.

Use the telemetry code already proven at commit `91c454f0b9ad5f282940149234f4852913817582`. Do not modify it unless it blocks an actual asset run.

## Actual model-production order

Run one GPU-heavy process at a time.

### 1. Tactical red panda scout

This is the first new geometry target because it provides a bounded character baseline.

- stage immutable source copy by SHA-256;
- route as equipped nonhuman humanoid/character shell;
- preserve rifle and backpack as rigid equipment candidates;
- run exactly one fast Mini Turbo geometry smoke attempt using the existing proven 6 GB-compatible route;
- default smoke envelope: steps=1, octree=256, existing compatible chunk setting and seed;
- write a real non-empty GLB;
- fresh-import in Blender;
- render source view plus front/rear/left/right/top-three-quarter contact sheet;
- report geometry, components, runtime and peak VRAM;
- do not rig, texture or animate before geometry/import review.

### 2. Frog salvage diver

After the red panda milestone is committed and clean:

- same one-attempt fast geometry smoke envelope;
- route as equipped nonhuman humanoid;
- preserve backpack/tank, lantern and canisters as rigid equipment candidates;
- hoses and hanging bag remain separate semantic candidates;
- write GLB, fresh-import and contact sheet before further work.

### 3. Mossy mountain titan

- route as massive quadruped/creature shell;
- no humanoid A-pose;
- preserve four main legs, head silhouette, mountain back and rib opening;
- one fast geometry smoke attempt;
- GLB, fresh import and contact sheet required.

### 4. Lucky Drown casino boat

- route as vehicle/building static hard-surface asset;
- no humanoid rigging or character skinning;
- preserve hull, deck silhouette, paddlewheel, railings, sign and stairs;
- one fast geometry smoke attempt;
- GLB, fresh import and contact sheet required.

### 5. Barn and trees

Do not run another expensive barn generation now. Preserve this source in the six-image pack and link it to existing barn evidence.

Current retained classifications:

- reduced one-step canaries: `PROVEN_REDUCED_CANARIES`;
- steps=2 diffusion: `PROVEN_BY_TRACE`;
- steps=2 mesh decode: `PROVEN_IN_MEMORY`;
- steps=2 GLB write: `NOT_PROVEN_LOGGER_ABORT`;
- original steps=5 root cause: `NOT_PROVEN_DEFERRED`;
- next barn GPU rerun: `NOT_AUTHORIZED`.

A later barn rerun requires an actual shared-pipeline improvement already proven on another asset, not telemetry validation.

### 6. Shaman anchor

Preserve and link the source image. Do not regenerate geometry and do not retry sleeve extraction, weighting or automatic garment separation.

Current status:

`BLOCKED_SHAMAN_SLEEVE_REQUIRES_MANUAL_RETOPOLOGY`

Use only its proven regression gates after shared character-pipeline changes.

## Failure and continuation policy

For the four new GPU targets, one initial smoke attempt each. A failure blocks only that asset and stage. Continue to the next staged asset unless the failure proves a shared defect affecting every route.

Do not run repeated unchanged attempts. One narrowly corrected retry is allowed only when the first failure has a concrete, local correction and the correction is implemented first.

After every actual asset milestone:

1. preserve artifacts and logs;
2. fresh-import in Blender when a model exists;
3. generate visible contact sheet;
4. update per-asset proof receipt;
5. commit and push;
6. verify local equals remote;
7. verify clean worktree;
8. continue automatically to the next asset.

Use only:

- `PROVEN`
- `REJECTED`
- `NOT_PROVEN`
- `BLOCKED`
- `USER_REVIEW_REQUIRED`
- `WAIT_FOR_SOL_ARCHITECTURE`

## Required progress report

Report this compact matrix after each milestone:

| Asset | Source preserved | Geometry | GLB | Fresh import | Contact sheet | Next action |
|---|---|---|---|---|---|---|

The task is not complete after source preservation. The immediate production target is a real `tactical_red_panda_scout` GLB, followed by the frog, titan and boat.