# Codex task: repair and prove the full multiview texture pipeline

You are running non-interactively on the repository's trusted Windows self-hosted worker. Own the diagnosis, implementation, testing, visual proof, and repository hygiene. Do not stop at another plausible hypothesis or another prettier single render.

## User goal

Fix the full reusable image-to-3D texture pipeline so the tactical red panda has a coherent detailed face at the front, no duplicate face on the rear or sides, valid provenance, and a freshly importable textured GLB. The repair must be generic pipeline code, not a panda-only paint-over or coordinate hack.

## Terminal states

Continue autonomously until exactly one is justified:

- `FRONT_REAR_TEXTURE_REPAIR_PROVEN`
- `HARD_BLOCKER`

Do not use `USER_REVIEW_REQUIRED` merely because visual inspection is needed: generate deterministic contact sheets and machine-readable evidence so the result can be reviewed after the run. Manual visual evidence remains the decisive acceptance criterion.

## Canonical repository and branch policy

The GitHub Actions checkout is the writable source of truth for this run.

Expected repository:

- `organicoverlords/lowvram3d-studio`
- working branch: `fix/panda-full-texture-pipeline-20260804`
- base branch: `integration/unified-pipeline-v2-20260802`

Never edit `main` or the older production branch. Do not merge wholesale from old branches.

A separate Claude-era local worktree may exist at:

`C:\Users\Lauri\Desktop\lowvram3d-unified-pipeline-v2-20260802`

The workflow captures its status and uncommitted patches into `.agent/local-claude-state/`. Treat those files as evidence and possible implementation material. Do not modify, reset, clean, stash, checkout, or commit inside that desktop worktree. Inspect each change and selectively port only changes supported by tests and visual evidence.

## Mandatory preflight

Before editing source:

1. Record checkout root, branch, HEAD, origin, status, Python executables, Blender version, Codex version, available RAM, GPU name/VRAM, and running Blender/Python/CUDA processes.
2. Verify the checkout is the expected repository and branch.
3. Inspect `.agent/local-claude-state/` and compare any captured patch against the GitHub checkout.
4. Inventory existing panda artifacts, modification times, hashes, reports, and generating code where reconstructable.
5. Do not start a new neural-generation/GPU inference pass. Existing six-view outputs and controls are the immutable inputs for this repair. The required work is projection, visibility, atlas fusion, QA, and pipeline integration. CPU projection and Blender QA are authorized.
6. Fail closed on repository identity mismatch or missing canonical assets.

## Hardware constraints

The worker machine is approximately:

- Windows 10
- NVIDIA GTX 1660 SUPER, 6 GB VRAM
- Ryzen 5 5600G
- 16 GB RAM with a large page file
- Blender 5.2 at `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`

Avoid unbounded full-atlas per-owner arrays, unbounded pair enumeration, and GPU workloads that approach known 6 GB failure modes. Use bounded memory, chunking, sparse representations, deterministic fallbacks, and early proof probes before expensive final runs.

## Canonical asset inputs

Mesh:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\bar_local_closure_v1\tactical_red_panda_scout_bar_repaired.glb`

Existing generated view images and inference receipt:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\sd21_upright_384x20_20260803`

Control bundle:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\sd21_cpu_controls_384_v9_upright_raw`

Face-region configuration:

`configs/texture/panda_face_priority_region.json`

Historical audit root:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\historical_front_audit_v1`

Historical front-face leader identified by the audit:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\panda_semantic_isolation_diagnostics_v1\front_only.glb`

Expected hashes from the audit, which must be independently verified before use:

- GLB SHA-256: `21ecb923428ce22145bccf67ddacf410f01f7d739cb9558566ecf89bc5ff4c8c`
- atlas SHA-256: `160052314021ccce13629864bf1a123d2e73288e8e1cbae3b1fd619fb8f2103d`
- fresh front render SHA-256: `1c8ddbaa35dd0b2d2cb37da0f847aa4b76399fe298264191d1c90d5e219ddb8c`
- fresh rear render SHA-256: `0f6b9674d9bb1551b3d3b58934323d63a54cbf66cd1a3ddbd19947e3aacec6b9`

This artifact is only the best historical **front-face evidence**. It is not a valid final output because its front pattern appears at the rear.

## Current evidence state

### Proven or strongly supported

- A diagnostic once resolved the active base colour incorrectly through `images[0]`; active texture lookup must follow primitive/material/baseColorTexture/texture/source/image references.
- The originally generated atlas was genuinely speckled and faceless after the active atlas was read correctly.
- Winner-take-all did not repair that original defect; approximately 98.2% of texels were already winner-take-all.
- The old UV atlas ownership rasterization used scatter/last-writer behaviour and created long streaks and broken fragments.
- Exact texel-centre barycentric ownership restored a coherent face in a front-only atlas while changing occupied coverage by only about 0.15%.
- `push_pull_fill` previously used `np.roll`, allowing opposite atlas edges to communicate.
- Completion must be domain-limited during every pass; the second pass must not silently refill the rectangular canvas.
- Front and rear UV charts were reported separated; direct UV overlap is not the leading explanation.
- Rear-only source projection is clean from both QA directions.
- Front-only projection produces the facial pattern visible from both QA directions.
- Existing per-pixel triangle-ID buffers showed only about 12–18% naive direct agreement. They must not be applied as a destructive hard gate until coordinate/index correspondence is proven.
- Strict face-ID gating previously removed rear-face leakage but left the asset pale/neutral because valid coverage collapsed.

### Rejected or unsupported approaches

Do not stack these again without new decisive evidence:

- blaming lighting
- blaming colour averaging
- broad smoothing, blur, random fill, inpainting, or manual painting
- assuming the first image or material is active
- simple rear-priority ownership rules
- semantic front/rear slot swaps based only on filenames
- guessed mesh-local camera vectors
- surface-normal owner labels without visibility proof
- regenerating all controls merely because a semantic label appears wrong
- hiding contamination by enabling backface culling
- loosening QA thresholds to promote a visibly bad result

### Current honest classifications

- `FACE_PRESENT_BASELINE_ONLY=PROVEN`
- `GOOD_FACE_FINAL_CHECKPOINT=NOT_PROVEN`
- `FULL_FRONT_REAR_RESULT=REJECTED`
- `REAR_REPAIR=NOT_STARTED` at the start of this task

## Required architecture repair

The pipeline needs an explicit, testable coordinate-and-visibility contract instead of implicit semantic conventions.

Implement or consolidate a reusable projection contract with these properties:

1. Mesh coordinates, normalization transform, camera basis, view direction convention, screen mapping, depth convention, image origin, and control-buffer dimensions are explicit and serialized.
2. Each source view carries an immutable ID separate from a human semantic label.
3. Projection uses the exact same contract that created depth/mask/triangle-ID controls.
4. Camera axes are validated geometrically with asymmetric cardinal markers or equivalent round-trip probes.
5. Visibility uses foremost-surface evidence. At minimum, depth agreement is required; triangle-ID evidence should be used when its index domain is proven, with boundary tolerance rather than blind equality.
6. Every atlas texel records triangle ownership, barycentrics, selected source view, source pixel, confidence, facing, depth error, and rejection reason counts.
7. Filling is confined to the same UV island or an explicitly approved local domain. No wrapping and no unrelated-chart propagation.
8. Asset-specific protected regions may influence view selection only after a view is valid and visible; they cannot override visibility.
9. The implementation scales to approximately 644k triangles and a 2048 atlas without allocating full-atlas vote arrays once per triangle/owner.

## Investigation sequence

Do not jump directly to a final bake. Find the first broken stage and preserve evidence at every boundary.

### Phase A: recover and reproduce

1. Verify the historical audit hashes and candidate metadata.
2. Reproduce the historical `front_only` atlas using current or reconstructed code.
3. Render that GLB with deterministic front, rear, left, right, and three-quarter cameras, both unlit and with ordinary lighting.
4. Record camera transforms and object transforms in JSON. Do not trust view filenames as proof of direction.
5. Prove that exact texel-centre rasterization is actually used in the reproduction.

### Phase B: map suspicious rear pixels to geometry

For visible rear-face pixels that look like the front face, resolve:

- Blender object and primitive
- material
- triangle ID in each relevant index domain
- world-space centroid and normal
- UV and atlas texel
- selected source view and source pixel
- projected depth and depth-buffer value
- front/back-facing status relative to the QA camera
- whether the triangle is exterior rear surface, exterior front surface, internal shell, duplicate component, intersecting geometry, or a two-sided/backface artefact

Produce deterministic diagnostic renders:

- triangle ID
- material ID
- geometric normal
- frontface/backface mask
- UV island
- source-view provenance
- confidence
- depth error
- observed versus synthesized

This phase must distinguish projection leakage from topology/culling/camera-label errors.

### Phase C: derive the camera transform

Trace the exact chain:

`original GLB mesh -> normalization -> control-space transform -> camera basis -> screen/depth raster -> generated source image registration -> atlas sample`

Use the control bundle's actual `camera_contract.json`, masks, depth buffers, and triangle-ID buffers. Do not replace saved matrices with assumptions.

Build an automatic registration/calibration test using an asymmetric mesh or six cardinal markers. Score candidate axis/sign mappings by silhouette, depth, and landmark registration. Select the mapping by measured error and serialize the decision. Add a fail-closed ambiguity threshold.

### Phase D: repair foremost-surface visibility

Once transforms are proven, ensure a source observation can colour only the surface that generated or matches its control pixel.

Use a layered synthetic fixture with two triangles on one camera ray. Prove that the farther triangle never receives the nearer surface's colour.

If triangle IDs differ because of primitive splitting, vertex duplication, or local/global index domains, implement an explicit mapping. If exact ID is not reliable at raster boundaries, use bounded neighbourhood matching plus depth and primitive evidence. Record match rates before and after correction.

### Phase E: fuse and complete without regression

Retain the sharp front observation from the best valid frontal source. Prevent side or rear views from diluting high-frequency face detail. But a protected face region may only choose among valid visible candidates.

For unobserved texels:

- operate per UV island or proven connected surface domain
- use bounded local propagation or donors with normal/component constraints
- never allow front facial colours to synthesize rear-facing surfaces
- label all synthesized texels and donor provenance
- report observed and synthesized coverage separately

Avoid the previous expensive owner-vote implementation that allocated full atlas arrays per owner. Use sparse/chunked or fixed-view-count operations.

### Phase F: integrate the reusable stage

Wire the repair into the canonical unified pipeline texture stage. The final path must not depend on one-off notebook snippets or a manually rebound GLB.

Provide one supported command that takes:

- UV-bearing GLB
- multiview image receipt
- camera/control bundle
- output root
- optional protected-region config

and produces:

- atlas
- provenance buffers
- diagnostics
- textured GLB
- fresh-import QA
- front/rear/side contact sheet
- machine-readable acceptance receipt

Update pipeline documentation and stage receipts. Do not replace unrelated generation, geometry, LOD, rigging, or export systems.

## Required tests

Add deterministic focused tests for at least:

1. Active base-colour resolution when the active image is not `images[0]`.
2. Exact texel-centre barycentric reconstruction, including winding and shared seams.
3. No opposite-edge wrap during fill or regularisation.
4. Domain-limited completion across every pass.
5. Camera-contract round trip.
6. Asymmetric front/rear projection.
7. Two surfaces on one camera ray: only the foremost surface is valid.
8. Primitive-local/global triangle-ID mapping or its explicit fallback.
9. Backface versus frontface QA classification.
10. Protected regions cannot override an invalid/occluded source.
11. Source-view provenance remains aligned after registration.
12. A bounded memory regression for the 2048/644k-triangle class.

Run focused tests first, then the complete relevant texture/pipeline test set. Use the repository's intended Python environment and set `PYTHONPATH` explicitly. A successful process exit is not visual proof.

## Canonical output root for this run

Write final and diagnostic artifacts under:

`C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\panda_full_pipeline_repair_20260804`

Never overwrite historical artifacts. Use immutable subdirectories for probes and one `final_candidate` directory only after all automated gates pass.

Required final files include:

- `acceptance.json`
- `pipeline_receipt.json`
- `root_cause.md`
- `final_candidate\tactical_red_panda_scout_textured.glb`
- active base-colour atlas
- provenance/source-view atlas
- triangle-ID atlas
- UV-island atlas
- confidence atlas
- depth-error atlas
- observed/synthesized mask
- deterministic unlit and lit renders for front, rear, left, right, and two three-quarter views
- enlarged front-face and rear-head crops
- labelled contact sheet
- fresh Blender import report
- exact commands and environment manifest

## Acceptance gate

`FRONT_REAR_TEXTURE_REPAIR_PROVEN` requires all of the following:

1. The front face is visibly coherent, recognizable, and at least as sharp as the historical `#11_semantic_front_only` candidate at the same render settings.
2. The unlit rear render contains no duplicated eyes, muzzle, nose, or front facial mask.
3. Left and right views contain no face wrapped around the head.
4. Representative front and rear texels have valid, traceable provenance.
5. No occluded surface wins a source observation.
6. The active material resolves to the newly produced atlas through glTF references.
7. Geometry and UV hashes are unchanged unless a topology defect is proven and separately repaired; any change must be explicitly justified and compared.
8. The GLB imports in a fresh Blender process.
9. Focused and relevant existing tests pass.
10. Observed/synthesized coverage is reported honestly.
11. No legacy atlas is silently used in any QA render.
12. The canonical unified pipeline can reproduce the result through its supported command.

Do not promote based on hashes, filenames, exit code, or structural validity alone.

## Repository deliverables

Keep generic source changes, tests, workflow-safe scripts, and concise proof metadata in Git. Do not commit large generated GLBs or full atlases unless they are already within repository policy. Store large artifacts in the canonical local output root and expose them through the workflow artifact upload.

Create or update:

- a concise proof document under `proof/integration/`
- a compact machine-readable receipt under `evidence/latest-panda-full-texture-repair/`
- documentation for the supported pipeline command

Before finishing:

1. Remove code from rejected experiments.
2. Inspect the complete diff.
3. Run the final test and canonical reproduction commands from a clean working tree except for intended changes.
4. Do not commit from inside Codex; leave the intended changes and evidence in the Actions checkout for the workflow's independent gate and commit step.
5. Write your final response to the path supplied by `--output-last-message` with root cause, first broken stage, retained/reverted changes, commands, test results, output paths, hashes, coverage, and classification.

## Final machine-readable contract

`acceptance.json` must contain at least:

```json
{
  "schema": "panda_full_texture_repair_acceptance_v1",
  "classification": "FRONT_REAR_TEXTURE_REPAIR_PROVEN",
  "full_pipeline_texture_qa": "PROVEN",
  "front_face_quality": "PROVEN",
  "rear_face_projection": "PROVEN_ABSENT",
  "side_face_wrap": "PROVEN_ABSENT",
  "fresh_import": "PROVEN",
  "active_texture_resolution": "PROVEN",
  "geometry_hash_preserved": true,
  "uv_hash_preserved": true,
  "tests_passed": true,
  "canonical_reproduction_passed": true,
  "final_glb": "absolute path",
  "contact_sheet": "absolute path",
  "proof_document": "repository-relative path"
}
```

If a genuine external blocker prevents completion, write the same file with `classification: HARD_BLOCKER`, exact evidence, and the smallest next action. Do not misclassify an unresolved software bug as an external blocker.
