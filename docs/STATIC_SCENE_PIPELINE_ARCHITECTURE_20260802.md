# Static-scene pipeline architecture v1 — barn and trees

## Ownership and execution model

This document is the architecture specification.

- High-reasoning architecture/review owns stage design, routing rules, acceptance gates, and failure policy.
- Luna Medium is a bounded executor. It may implement exactly the milestones defined here, run workers, collect evidence, and make one narrowly specified correction.
- Luna must not replace this design with a new engine, framework, profile system, or unrelated refactor.
- When a new architectural decision is required, Luna stops only that lane with `WAIT_FOR_SOL_ARCHITECTURE` and returns evidence plus the smallest unresolved design question.

## Current barn run

Do not interrupt or invalidate the barn run already in progress.

Treat that run as `BASELINE_RUN_V0` regardless of visual quality. Preserve:

- exact source SHA-256;
- exact command and configuration;
- all intermediate meshes and images;
- stdout/stderr logs;
- stage timings;
- peak VRAM and RAM when available;
- final exports;
- standardized renders.

The current run is evidence. It is not automatically the production candidate.

Do not retrofit this architecture into the middle of an active generation process. Apply it only after the baseline artifacts are sealed and the worktree is clean.

## Source truth

The benchmark image contains a static rural scene dominated by:

- one weathered barn or connected barn structure;
- large wind-shaped trees directly behind and beside the barn;
- grass/ground and a few foreground posts as scene context;
- sky and distant land as background.

Nothing is walking or mechanical. No rig or animation is required.

The required benchmark asset is the **barn plus trees**. Terrain, long grass, fence posts, sky, and distant landscape are optional scene context and must never be silently fused into the required asset.

## Required production outputs

Produce two explicit variants from the same immutable staged source:

### `HERO_PACK`

Required benchmark deliverable:

- `Barn_Main`
- one or more `Tree_*` objects or a defensible `TreeCluster_*`
- separate materials by physical role
- no sky
- no distant landscape
- no rig
- no animation
- no foreground fence posts unless explicitly promoted as an attached prop

### `SCENE_PACK`

Optional secondary deliverable:

- everything in `HERO_PACK`
- a shallow terrain patch
- optional grass cards/instances
- optional fence-post objects
- scene dressing must remain separate from barn and trees

The pipeline must never claim `SCENE_PACK` is proven merely because `HERO_PACK` passed.

## Core architectural change

Static scenes must not use the same monolithic route as a single character or prop.

The route becomes:

```text
SOURCE_AUDIT
→ SEMANTIC_LAYERING
→ OCCLUSION_GRAPH
→ COMPONENT_RECONSTRUCTION
→ COMPONENT_QA
→ MATERIAL_RECONSTRUCTION
→ LOD_AND_COLLISION
→ STATIC_SCENE_ASSEMBLY
→ EXPORT_QA
```

Each component owns its own geometry, materials, topology gates, LOD policy, and uncertainty record.

## Stage 1 — `SOURCE_AUDIT`

Inputs:

- immutable staged image;
- source hash and dimensions;
- no inferred labels from filenames.

Outputs:

- `source_audit.json`
- `source_preview.png`
- `source_edge_map.png`
- `source_depth_advisory.png`
- `source_uncertainty.json`

Record:

- image dimensions and channels;
- alpha/background status;
- camera-perspective estimate;
- visible barn bounds;
- visible tree/canopy bounds;
- ground/context bounds;
- occluded regions;
- single-view limitations;
- likely contact regions between barn, trees, and ground.

Do not claim rear structure, hidden roof surfaces, tree-root layout, or internal rooms as observed.

## Stage 2 — `SEMANTIC_LAYERING`

Create independent masks for:

- `barn_structure`
- `tree_trunks_and_major_branches`
- `tree_foliage_mass`
- `terrain_context`
- `grass_context`
- `fence_post_context`
- `sky_and_distant_background`
- `unknown_or_ambiguous`

Use a two-pass process:

1. coarse semantic segmentation;
2. edge- and connectivity-aware refinement against the original resolution.

Required properties:

- masks are mutually auditable;
- overlap is explicit rather than silently resolved;
- thin branches are not erased merely because confidence is low;
- grass and sky do not leak into barn or tree masks;
- barn roof edges remain separated from dark tree canopy;
- occlusion boundaries are preserved.

Required outputs:

- one lossless mask per class;
- `semantic_layers_overlay.png`;
- `semantic_layer_report.json`;
- mask coverage, overlap, holes, and boundary length metrics.

Fail closed with `BLOCKED_SEMANTIC_LAYERING_UNUSABLE` when barn and tree masks cannot be separated well enough for component routes. Do not compensate by generating one fused scene blob.

## Stage 3 — `OCCLUSION_GRAPH`

Build a simple ordered relationship graph, not a full inferred scene model.

Required nodes:

- barn;
- tree trunks/branches;
- foliage mass;
- terrain/context.

Required relations:

- tree behind barn where visibly supported;
- tree overlapping barn silhouette where visibly supported;
- barn grounded on terrain;
- uncertain relations remain `UNKNOWN`.

Outputs:

- `occlusion_graph.json`;
- `occlusion_overlay.png`;
- per-relation confidence and evidence.

This stage prevents common failures:

- tree trunks fused into barn walls;
- foliage treated as a roof extension;
- barn wall texture projected onto trees;
- grass extruded into structural geometry.

## Stage 4 — `COMPONENT_RECONSTRUCTION`

### Barn route

The barn uses a building-specific route:

- identify major wall planes;
- identify roof planes and roof breaks;
- infer wall and roof thickness conservatively;
- preserve visible door/window/opening silhouettes;
- preserve visible lean, collapse, missing boards, and roof damage when structurally readable;
- do not convert every texture crack into geometry;
- do not enforce perfect symmetry or perfect orthogonality over clearly weathered shapes;
- do not fabricate an interior.

Preferred representation:

- planar or low-curvature structural surfaces;
- separate roof and wall groups;
- explicit openings where visually supported;
- optional small debris/board pieces only when silhouette-significant.

Barn-specific failure codes:

- `REJECTED_BARN_BLOB_GEOMETRY`
- `REJECTED_ROOF_WALL_FUSION`
- `REJECTED_OPENINGS_FILLED`
- `REJECTED_VISIBLE_DAMAGE_ERASED`
- `NOT_PROVEN_UNSEEN_REAR_STRUCTURE`

### Tree route

Trees use a vegetation-specific route:

- reconstruct visible trunks and major branches as volume geometry;
- preserve the dominant wind-swept silhouette;
- separate trunk/branch structure from foliage;
- use foliage clusters, cards, or bounded low-poly canopy geometry rather than a fully solid leaf blob;
- allow alpha-tested leaf materials where export target supports them;
- retain branch/canopy gaps that materially affect silhouette;
- do not generate individual leaves at hero-film density for the 6 GB target.

Tree-specific failure codes:

- `REJECTED_TREE_SOLID_BLOB`
- `REJECTED_TRUNK_CANOPY_FUSION`
- `REJECTED_WIND_SILHOUETTE_LOST`
- `REJECTED_BRANCH_STRUCTURE_ERASED`
- `NOT_PROVEN_HIDDEN_BRANCH_LAYOUT`

### Context route

Terrain, grass, and posts are optional and separate:

- terrain is a shallow patch, never a deep image-extrusion slab;
- grass uses cards or instances, not dense reconstructed blades by default;
- fence posts remain separate props;
- sky and distant landscape are excluded from geometry.

## Stage 5 — `COMPONENT_QA`

Do not use one universal topology gate for every component.

### Barn gates

Required:

- fresh import succeeds;
- finite vertices/transforms;
- no unintended floating structural pieces;
- no catastrophic non-manifold regions;
- exterior wall/roof solids have defensible closure;
- openings remain openings where expected;
- no tree/grass geometry embedded in barn surfaces;
- source-view silhouette remains within the configured error threshold.

Boundary edges are expected only for explicitly open architectural or damage regions. Report them by named component.

### Tree gates

Required:

- fresh import succeeds;
- finite vertices/transforms;
- trunk/major branch geometry remains connected where expected;
- foliage cards may have boundary edges and must not fail merely for being open surfaces;
- no barn boards or roof planes inside tree objects;
- alpha materials remain bound;
- wind-swept silhouette survives standardized renders.

### Context gates

Required:

- context does not merge into required hero objects;
- no sky/distant-land geometry;
- no deep terrain wall around the scene patch;
- optional context failures do not invalidate a passing `HERO_PACK`.

## Stage 6 — `MATERIAL_RECONSTRUCTION`

Create material classes before projection:

- weathered timber;
- rusted/corrugated roof metal;
- dark interior/opening material;
- bark/wood;
- foliage with alpha where used;
- optional terrain/grass materials.

Improve the existing observed-versus-synthesized texture policy:

- calculate observation coverage per component and per material class;
- front-observed barn pixels may not become tree donors;
- foliage may not donate colour into timber or roof metal;
- visible front imagery may not be mirrored onto unseen rear surfaces and called observed;
- synthesized rear/hidden areas use constrained material priors and tileable detail;
- retain a per-texel or per-chart provenance map: `OBSERVED`, `SYNTHESIZED_LOCAL`, `SYNTHESIZED_MATERIAL_PRIOR`, `UNRESOLVED`.

Required PBR outputs where defensible:

- base colour;
- normal;
- roughness;
- metallic for roof metal only where appropriate;
- alpha for foliage cards;
- AO as a separate optional map, not baked into base colour as the only shading.

Reject:

- sky baked into foliage or roof;
- grass baked into barn walls;
- source shadow treated as permanent albedo;
- front facade duplicated onto rear walls;
- one material covering barn and trees.

## Stage 7 — `LOD_AND_COLLISION`

Use component budgets rather than one scene-wide decimation target.

Suggested default budget envelope for the first production candidate:

- barn LOD0: 40k–100k triangles depending on structural detail;
- combined tree trunks/major branches LOD0: 30k–100k triangles;
- foliage LOD0: bounded by cluster/card count and overdraw, not only triangles;
- context LOD0: separate and optional.

These are soft envelopes, not pass conditions. Preserve silhouette and named details before hitting a numeric target.

Create:

- `LOD0` source-quality game candidate;
- `LOD1` approximately 50–65% of LOD0 while preserving silhouette;
- `LOD2` approximately 20–35% of LOD0;
- billboard/impostor option for distant tree clusters when useful.

Collision:

- barn: simple convex/box decomposition, doors/openings respected when gameplay requires access;
- trees: trunk/major-branch collision only;
- foliage: no collision by default;
- terrain: optional simple terrain collision;
- never use full high-poly scene collision as the default.

## Stage 8 — `STATIC_SCENE_ASSEMBLY`

Required object hierarchy:

```text
BarnTrees_Root
├── Barn_Main
├── Tree_*
├── Foliage_*
└── Context_Optional
    ├── Terrain
    ├── Grass_*
    └── FencePost_*
```

Required pivots:

- root at defensible ground centre;
- barn pivot on ground plane;
- tree pivots at trunk-base estimates;
- context objects retain their own pivots.

Scale:

- estimate only when an anchor is defensible;
- record confidence and assumed anchor;
- do not present inferred real-world dimensions as observed fact.

## Stage 9 — `EXPORT_QA`

Produce:

- `barn_trees_hero_lod0.glb`
- optional `barn_trees_scene_lod0.glb`
- split GLBs for barn and trees when requested by downstream tools;
- optional Blender master;
- LODs and collision objects in a documented package.

Fresh-import gates:

- object hierarchy preserved;
- object count and names preserved;
- material-slot count preserved;
- texture bindings preserved;
- alpha mode preserved for foliage;
- triangle counts within exporter-expected tolerances;
- no unexpected second meshes or helper objects;
- transforms finite;
- no animation or armature introduced.

## Standardized visual proof

Render every candidate in neutral lighting from:

- source-matching camera;
- front;
- rear;
- left;
- right;
- top three-quarter;
- close barn detail;
- close tree silhouette detail.

Produce comparison sheets:

1. source image versus source-matching render;
2. semantic masks and component IDs;
3. barn-only views;
4. tree-only views;
5. hero pack views;
6. optional scene pack views;
7. texture provenance visualization;
8. LOD comparison.

## Required metrics

Per component and total:

- objects and meshes;
- vertices and triangles;
- connected components;
- boundary and non-manifold edges interpreted by component policy;
- dimensions and orientation;
- material slots and textures;
- UV charts, utilization, true overlap, and out-of-bounds area;
- source-view silhouette IoU or equivalent;
- observed versus synthesized texture coverage;
- thin-branch survival;
- barn opening preservation;
- tree/barn cross-contamination count;
- export/import parity;
- runtime, peak VRAM, peak RAM, and file size.

## Promotion policy

A candidate is not promoted from a pretty source-angle render alone.

`HERO_PACK=PROVEN` requires:

- barn and tree components are independently identifiable;
- source-view resemblance is acceptable;
- side/rear views are structurally plausible and clearly marked as inferred where unseen;
- no sky/ground smear on hero objects;
- barn and tree materials are separated;
- no rig/animation;
- fresh import passes;
- contact sheets and reports exist.

`SCENE_PACK=PROVEN` additionally requires context geometry and materials to pass their own gates.

Use only:

- `PROVEN`
- `REJECTED`
- `NOT_PROVEN`
- `BLOCKED`
- `WAIT_FOR_SOL_ARCHITECTURE`

## Bounded implementation milestones for Luna

After `BASELINE_RUN_V0` is sealed:

### Milestone A — evidence only

- audit the baseline output;
- produce standardized renders and metrics;
- do not modify generation code;
- classify exact failure modes.

### Milestone B — semantic front end

Implement only:

- source audit;
- semantic masks;
- occlusion graph;
- component manifest.

No geometry generation changes yet.

### Milestone C — component split

Implement routing so barn, trunks/branches, foliage, and optional context are separate outputs.

Do not redesign model backends.

### Milestone D — component-specific QA and materials

Add the profile-specific topology gates, material classes, and texture-provenance reporting described above.

### Milestone E — LOD, collision, assembly, export QA

Implement only after C and D produce usable components.

After each milestone:

- focused tests;
- one bounded Blender execution;
- visible proof;
- proof/state update;
- commit and push on the execution branch;
- local/remote equality and clean worktree.

Maximum one correction per milestone before returning evidence for architecture review.
