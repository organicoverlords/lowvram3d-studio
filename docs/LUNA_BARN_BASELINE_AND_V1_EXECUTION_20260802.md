# Luna execution plan — finish baseline, then implement static-scene v1

## Role

Luna Medium is the bounded executor for this plan.

Architecture is fixed by:

- `docs/STATIC_SCENE_PIPELINE_ARCHITECTURE_20260802.md`
- `configs/profiles/static_scene_barn_trees_v1.json`

Do not redesign either file. Do not create a new engine, framework, profile system, or orchestrator.

## Part 1 — finish the active barn baseline

Do not restart the active run merely because this plan exists.

When the run completes or fails:

1. preserve its entire output directory;
2. write `baseline_run_v0_manifest.json` with:
   - source path and SHA-256;
   - exact command;
   - repository HEAD;
   - configuration;
   - stage start/end times;
   - exit codes;
   - peak VRAM/RAM when available;
   - every produced artifact;
3. fresh-import every produced model in a new Blender process;
4. render source-match, front, rear, left, right, and top-three-quarter views;
5. produce a contact sheet;
6. audit objects, meshes, vertices, triangles, connected components, boundary edges, non-manifold edges, materials, textures, UVs, armatures, actions, and dimensions;
7. classify the baseline without changing generation code.

Required classifications:

- `BASELINE_EXECUTION`
- `SOURCE_MATCH`
- `BARN_GEOMETRY`
- `TREE_GEOMETRY`
- `BARN_TREE_SEPARATION`
- `BACKGROUND_CONTAMINATION`
- `MATERIAL_SEPARATION`
- `TEXTURE_QUALITY`
- `EXPORT_QA`

Use `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED` with exact reason codes.

## Part 2 — return baseline evidence before implementation

Return:

- baseline output directory;
- contact sheet path;
- audit/report paths;
- the five most severe visible or structural defects;
- which defects are shared-pipeline defects versus barn-profile defects;
- exact files likely requiring modification for Milestone B.

Do not begin Milestone B when the baseline run itself has not been sealed.

## Part 3 — Milestone B only: semantic front end

After architecture review authorizes the baseline result, implement only:

- `SOURCE_AUDIT`;
- `SEMANTIC_LAYERING`;
- `OCCLUSION_GRAPH`;
- component manifest generation.

Required outputs for the same staged source:

```text
static_scene_v1/
  source_audit.json
  source_preview.png
  source_edge_map.png
  source_depth_advisory.png
  source_uncertainty.json
  semantic_masks/
    barn_structure.png
    tree_trunks_and_major_branches.png
    tree_foliage_mass.png
    terrain_context.png
    grass_context.png
    fence_post_context.png
    sky_and_distant_background.png
    unknown_or_ambiguous.png
  semantic_layers_overlay.png
  semantic_layer_report.json
  occlusion_graph.json
  occlusion_overlay.png
  component_manifest.json
```

Milestone B must not:

- alter the geometry backend;
- alter texturing;
- add LOD/collision;
- create a new visual model dependency without architecture approval;
- scan the entire models library;
- rename or move source files.

Run focused tests proving:

- every pixel is accounted for or explicitly unknown;
- mask overlap is reported;
- barn roof edges are not swallowed by the tree mask;
- trees do not absorb barn walls;
- sky does not leak into hero masks;
- moving the staged source path does not alter content identity;
- original source bytes remain unchanged.

Maximum one bounded correction.

If masks remain unusable, stop with:

`BLOCKED_SEMANTIC_LAYERING_UNUSABLE`

Do not generate a fused fallback scene.

## Part 4 — source control

For the active baseline:

- commit only evidence and code already produced by the active run;
- do not mix the architecture branch into the middle of the run;
- push and verify clean worktree.

For the v1 implementation after authorization:

- apply the architecture files to the execution branch only after the baseline commit is remote and clean;
- commit Milestone B independently;
- push;
- verify local SHA equals remote SHA;
- verify clean worktree.

## Stop codes

- `USER_REVIEW_REQUIRED`
- `WAIT_FOR_SOL_ARCHITECTURE`
- `BLOCKED_SEMANTIC_LAYERING_UNUSABLE`
- `HARD_BLOCKER`
- `TASK_DONE_NO_NEXT_ACTION`

Do not continue into component reconstruction without explicit architecture review of Milestone B evidence.
