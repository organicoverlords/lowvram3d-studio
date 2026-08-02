# Codex: continue here

Work only on branch `magicmusic/parts-pose-materials-20260802`.

Before editing, verify repository origin, branch, local HEAD, remote HEAD, and a clean worktree. Never work on `main`, merge, or force-push.

## Immediate authoritative task

Perform a CPU-only textured comparison of the existing sanitized panda candidates.

Do **not** run new GPU generation.
Do **not** run the frog.
Do **not** overwrite any original or accepted artifact.

## Immutable originals

Preserve these exact original GLBs byte-for-byte:

- `raw_4step_retry2.glb`
- `raw_5step.glb`

They are diagnostic source artifacts and must never be textured or promoted directly. Blender silently drops their duplicate-index faces during import, which would hide the decoder defect.

## Inputs for this comparison

Use only the CPU-sanitized copies created by the boundary investigation:

- `4step_sanitized.glb`
- `5step_sanitized.glb`

Verify each input hash before processing and record it in the comparison report.

## Required order

1. Texture the sanitized 5-step candidate first.
2. Fresh-import it in a clean Blender process.
3. Render matched front, three-quarter, side, and rear views.
4. Texture the sanitized 4-step candidate using the exact same UV, projection, fill, material, export, camera, lighting, and render settings.
5. Fresh-import it in a clean Blender process.
6. Render the same matched views.
7. Create one labelled side-by-side comparison sheet.
8. Compare both against repaired panda v7 without modifying v7.

## Fair-comparison rules

The only geometry difference may be the sanitized 4-step versus sanitized 5-step mesh.

Keep identical:

- canonical panda source image;
- conditioning/matte and source-facing camera;
- cleanup policy;
- UV method and parameters;
- texture resolution;
- raster projection settings;
- unseen-region synthesis policy;
- material settings;
- export settings;
- Blender version;
- camera transforms;
- lighting and render settings.

Do not run another component deletion pass unless the exact same deterministic cleanup policy is applied independently to both candidates and all removals are reported.

## Required evidence

For each candidate record:

- sanitized input SHA-256 and byte size;
- fresh-import vertex, face, component, boundary, and non-manifold counts;
- UV and material validation;
- observed versus synthesized texture coverage;
- packed texture resolution and SHA-256;
- textured GLB SHA-256 and byte size;
- fresh-import material/texture resolution result;
- front, three-quarter, side, and rear renders;
- visible source-identity and artifact assessment.

## Selection rule

The 5-step candidate has slightly more valid geometry, but that is not proof that it is visually better.

Choose a winner only from the matched fresh-import renders and structural gates.

A candidate may replace repaired panda v7 only when it is:

- structurally safe;
- at least as recognizable from the front;
- no worse in side/rear silhouette;
- no worse for rifle, backpack, tail, ears, and face retention;
- free of materially worse rods, fused padding, debris, black holes, or texture smearing.

When neither candidate clearly beats repaired panda v7, retain v7 and classify both comparison candidates honestly.

## Future generation policy

The corrected post-decode sanitizer from commit `0bffd3955ee2525edbb4527c2c50d84b4c49f727` remains mandatory for all future Mini Turbo generations.

Do not change the general step-count policy from this comparison alone.

Commit and push the comparison report and lightweight proof artifacts. Verify local equals remote and the worktree is clean.