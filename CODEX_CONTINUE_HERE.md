# Codex: continue here

Work only on branch `integration/unified-pipeline-v2-20260802`.

Before editing, verify repository origin, branch, local HEAD, remote HEAD, and a clean worktree. Never work on `main`, merge, or force-push.

## Immediate authoritative milestone

Stop using the panda as an open-ended research specimen.

Deliver **finished, usable, static textured GLB models** from the six canonical benchmark images. Pipeline improvements are valuable only when they move an asset to an accepted deliverable.

The milestone is six basic textured non-animated models:

1. tactical red panda scout
2. antlered bird shaman
3. frog salvage diver
4. Lucky Drown casino boat
5. wind-bent barn and trees
6. mossy mountain titan

A basic usable model is not required to be production-retopologized, rigged, animated, perfectly reconstructed on unseen surfaces, or optimized into a complete LOD chain.

## Definition of done for each asset

An asset is accepted when all are true:

- recognizable as the source subject;
- valid fresh-importable GLB;
- visible base-colour texture embedded or resolvable after clean import;
- front, three-quarter, side, and rear renders exist;
- no face or other front-only semantic feature projected onto the rear;
- no large black holes or transparent gaps inside the subject;
- no catastrophic floating debris halo;
- finite bounds and non-empty geometry;
- source, geometry, texture, and GLB SHA-256 values recorded;
- accepted artifact copied to a stable `deliverables/<asset_id>/` location;
- receipt states `BASIC_TEXTURED_MODEL=PROVEN`.

Minor synthesized rear material, imperfect seams, dense geometry, fused static accessories, and limited unseen detail are acceptable when clearly reported.

## Production policy: finish assets, do not loop forever

Every expensive stage has a hard budget and a fallback.

### Geometry

1. Reuse an already accepted or best preserved geometry whenever available.
2. All new Mini Turbo output must use the post-decoder duplicate-index sanitizer.
3. Default generation is the fastest already-proven setting.
4. One higher-step retry is allowed only when the first geometry is visibly unusable.
5. No repeated step-count sweep.
6. No manual component deletion when debris is fused into the subject.

### LOD

LOD is optional for this milestone.

Default manifest policy:

```yaml
lod:
  mode: preserve_source
```

A rejected or skipped LOD must not block UV, texture, export, or delivery.

### UV

Use this order:

1. preserve and validate existing UVs;
2. bounded Blender Smart UV with a 180-second timeout;
3. bounded planar/triplanar-to-atlas fallback where appropriate for static buildings/props;
4. xatlas only as an explicit later quality-mode opt-in.

Do not run xatlas automatically. Do not block a usable baseline on atlas utilization targets.

### Texture

Use the known safe raster route that enforces depth visibility and front-facing normals.

Rules:

- real source pixels may project only to triangles visible from that source camera;
- mirrored fallback views may never contribute semantic pixels;
- unobserved surfaces receive material-aware neutral/local synthesis, never copied facial content;
- observed front pixels remain protected;
- 1024 base colour is the baseline target; 2048 is optional when already cheap and proven;
- optional diffusion or multiview completion may improve an accepted baseline but must never block it.

The earlier projection-gate-fixed panda output with a neutral rear is preferable to a richer output that puts a face on the back.

### QA and retries

For each asset:

- maximum two texture attempts;
- maximum one UV fallback;
- no unchanged retry after timeout or rejection;
- visual renders outrank proxy metrics;
- an optional quality-stage failure cannot invalidate a previously accepted basic model.

## First task: finish and package the panda

Do not regenerate panda geometry, rerun LOD, or run xatlas.

1. Locate the exact earlier projection-gate-fixed sanitized 5-step panda artifact whose matched renders proved the rear face absent. Resolve it from existing reports/hashes; do not guess a path.
2. Fresh-import that GLB in clean Blender.
3. Render front, three-quarter, side, and true rear views.
4. Confirm no facial projection on the back and no large black holes.
5. If it passes, promote it immediately as the panda basic deliverable, even if its side/rear material is comparatively neutral.
6. Package:
   - textured GLB;
   - base colour;
   - four-view contact sheet;
   - compact JSON receipt;
   - source and artifact hashes.
7. Do not replace it with the rejected Texture Completion V2 artifact.

Only if the known-safe panda artifact cannot be located or fails fresh import may one bounded retexture be run using the same known-safe projection route. No experimental completion route is allowed in that fallback.

## Then finish the remaining five assets in this order

### 2. Antlered bird shaman

- reuse the canonical clean geometry;
- do not retry sleeve separation, garment extraction, rigging, or manual retopology;
- bypass LOD when it blocks delivery;
- apply the safe texture route and package the static textured baseline.

### 3. Wind-bent barn and trees

- reuse the best verified one-step geometry;
- do not rerun the long barn generator;
- preserve the static rural barn/tree identity;
- use bounded UV and safe texture/export stages.

### 4. Lucky Drown casino boat

- run one bounded sanitized geometry generation when no accepted geometry exists;
- prioritize readable side silhouette, decks, paddlewheel, stairs, rails, and `LUCKY DROWN` identity;
- accept static fused geometry when recognizable and structurally valid.

### 5. Mossy mountain titan

- run one bounded sanitized geometry generation when needed;
- prioritize quadruped silhouette, mountain back, root/rock body, and rib opening;
- accept synthesized unseen materials when no front semantic leakage exists.

### 6. Frog salvage diver

- preserve rejected v7/v8 as diagnostics;
- run at most one higher-step sanitized geometry candidate;
- require recognizable frog, suit/tank, lantern, bag, arms, and feet with materially reduced halo;
- if Mini Turbo still produces fused debris, use one already-available alternative geometry route or produce the best honest static baseline; do not perform another deletion-repair loop.

## Shared pipeline changes allowed

Implement only changes that directly support repeated delivery:

- optional LOD bypass;
- existing-UV reuse;
- bounded Smart UV fallback;
- safe projection route selection;
- candidate-to-deliverable promotion;
- compact delivery receipt;
- per-stage timeout and fallback recording;
- batch summary for six assets.

Do not build another experimental framework, global semantic segmentation system, large-model installer, or benchmark-only subsystem before the first accepted deliverable is packaged.

## Required progress reporting

Maintain one compact table:

| Asset | Geometry | UV | Texture | Fresh import | Visual QA | Deliverable |
|---|---|---|---|---|---|---|

Use only `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED`.

After each accepted asset, commit and push the receipt and lightweight proof. Continue automatically to the next asset. Stop only for:

- `USER_REVIEW_REQUIRED` after a contact sheet is ready;
- a real missing input;
- a destructive choice between two accepted artifacts;
- hardware/environment blocker with no bounded fallback.

## First completion receipt required

The next report must be either:

```text
PANDA_BASIC_TEXTURED_MODEL=PROVEN
```

with a stable deliverable path and four-view contact sheet, or:

```text
PANDA_BASIC_TEXTURED_MODEL=BLOCKED
```

with one exact blocker after exhausting only the bounded safe fallback.

Commit and push validated integration-branch changes. Verify local equals remote and leave the worktree clean. Do not merge branches yet.
