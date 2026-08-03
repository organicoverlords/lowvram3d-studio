# Scene Pipeline v2 — SceneSpec, PCG patterns, and proof

Date: 2026-08-03  
Branch: `agent/scene-pipeline-smoke-20260803`  
Status: design contract added; implementation remains incremental and proof-gated.

## 1. Why this exists

The current Castlegrounds lane proves:

- MoGe source reprojection;
- a repaired source-visible 2.5D mesh;
- fresh Blender import and exact source-camera coverage;
- Unreal Interchange GLB import;
- Unreal map save and fresh-process reload.

It does **not** yet prove a complete game-ready scene. Deterministic Unreal rendering and rendered parallax remain blocked in the current null-RHI path. The existing `scene_interpretation.json` and `geometry_spec.json` are useful evidence, but they are one-off Castlegrounds formats. They do not define a reusable contract for:

- separating visual, editable, and gameplay representations;
- spline-driven roads, rivers, walls, and bridges;
- CPU versus GPU PCG policy;
- collision, navigation, LOD, and material ownership;
- deterministic budgets and seeds;
- required proof gates.

`schemas/scene_spec_v1.schema.json` closes that contract gap.

## 2. Core architecture

A generated scene must not rely on one geometry representation for every purpose.

### 2.1 Visual shell

Purpose:

- source-camera fidelity;
- difficult distant scenery;
- sky, clouds, and non-interactive background;
- low-cost parallax inside a bounded camera envelope.

Allowed forms:

- source-projected 2.5D mesh;
- layered image cards;
- Gaussian splats where supported;
- distant HLOD or baked proxies.

Default policy:

- no collision;
- ignored by navigation;
- source-locked material or unlit projection;
- never silently promoted to gameplay geometry.

### 2.2 Editable semantic mesh

Purpose:

- architecture;
- terrain;
- hero props;
- modular roads, bridge pieces, walls, and buildings;
- assets requiring material, LOD, destruction, or animation control.

Default policy:

- Blender-normalized transform and naming;
- explicit triangle and texture budget;
- Interchange import;
- replaceable independently of the visual shell.

### 2.3 Gameplay proxy

Purpose:

- collision;
- navigation;
- interaction volumes;
- traversal and occlusion;
- runtime testing.

Default policy:

- simple, deterministic geometry;
- independent proof from appearance;
- source-visible shell is not accepted as collision merely because it looks correct.

## 3. SceneSpec lifecycle

```text
source image / brief
→ scene interpretation
→ SceneSpec generation
→ JSON Schema validation
→ deterministic adapters
   ├── source-visible shell
   ├── editable asset manifest
   ├── gameplay proxies
   ├── spline and exclusion data
   └── PCG layer manifest
→ Blender preparation
→ Unreal Interchange import
→ reference-driven PCG assembly
→ save/reload
→ rendered and gameplay proof
```

The SceneSpec is the source of truth. Generated Blender files, GLBs, Unreal assets, PCG graphs, screenshots, and receipts are derived outputs.

## 4. PCG policy

Unreal Engine 5.8 guidance recommends:

- loading the PCG toolset and PCG graph generation skill;
- inspecting working example graphs before planning;
- reusing or duplicating an existing graph when possible;
- giving the model explicit selected assets and actors;
- planning before execution;
- executing incrementally;
- inspecting attributes through Data View.

The pipeline therefore uses `reference_driven_incremental` as its only v1 PCG policy.

### 4.1 Required reference patterns

The first implementation should establish reviewed reference graphs for:

1. **Exclusion Mask**
   - water, roads, hero structures, cliffs, and no-build zones;
   - outputs deterministic masks or tagged point sets.

2. **Large Objects**
   - buildings, collision-bearing trees, boulders, bridge structures;
   - CPU execution;
   - larger hierarchical grid;
   - explicit collision and navigation.

3. **Small Scatter**
   - grass, flowers, pebbles, leaf litter;
   - GPU execution only when no collision, navigation, ray tracing, or distance-field dependency exists;
   - connected GPU nodes grouped to avoid CPU/GPU transfer overhead.

4. **Spline Modules**
   - fences, walls, bridge tiles, road furniture;
   - Shape Grammar or an equivalent proven module chain;
   - deterministic seed and incomplete-subdivision handling.

5. **Intersection Modules**
   - road intersections;
   - road/river crossings;
   - bridge placement;
   - exclusion of buildings and foliage from crossing zones.

### 4.2 CPU/GPU split

Use CPU PCG for:

- gameplay geometry;
- collision-bearing objects;
- navigation-affecting objects;
- intersections and bridge logic;
- attribute logic that must be inspected or downloaded;
- low-point-count graphs where GPU dispatch overhead dominates.

Use GPU PCG for:

- large decorative point populations;
- grass and small ground cover;
- connected compute-graph sections with minimal transfers;
- objects that require no collision, physics, navigation, ray tracing, or distance-field behavior.

On the GTX 1660 SUPER:

- never run a neural model concurrently with Unreal GPU population work;
- run one bounded GPU process at a time;
- preserve a CPU fallback for scene planning and graph validation;
- treat GPU PCG as an optimization, not a correctness dependency.

## 5. Hierarchical world policy

Use grid scale according to object scale:

- 128 m or larger: cliffs, large buildings, tree clusters;
- 64 m: bridges, roads, structures, large rocks;
- 16–32 m: shrubs, small rocks, repeated props;
- 8–16 m: grass, flowers, litter, pebbles.

Large-grid results may feed smaller grids as cached data. Runtime generation should use increasing generation radii as grid size increases. World Partition and HLOD are later production gates, not required for the first Castlegrounds smoke.

## 6. Asset import policy

GLB remains the preferred scene and static-asset interchange format when the source contains embedded textures and scene structure.

Every import must record:

- source SHA-256;
- import pipeline stack;
- destination path;
- mesh, material, and texture counts;
- transform and unit conversion;
- collision policy;
- LOD policy;
- imported asset paths;
- warnings;
- save and fresh-process reload result.

Use an explicit Interchange pipeline implemented through Python, Blueprint, or C++. UI drag-and-drop is not accepted as the only automation path.

## 7. Required implementation phases

### Phase A — contract and validation

Deliver:

- `schemas/scene_spec_v1.schema.json`;
- one valid Castlegrounds example;
- a validator that emits `SCENE_SPEC_VALID`;
- uniqueness and cross-reference checks not expressible cleanly in JSON Schema.

Additional semantic validation must reject:

- duplicate IDs;
- asset references to missing regions;
- population references to missing assets;
- GPU populations requiring collision or navigation;
- gameplay proxies with `collision=none`;
- source cameras with `far_m <= near_m`;
- PCG GPU layers containing gameplay outputs;
- unsupported concurrent GPU policy.

### Phase B — legacy evidence adapter

Convert:

- `scene_interpretation.json`;
- `geometry_spec.json`;
- existing transform and camera receipts;
- selected GLB metadata;

into a SceneSpec without rerunning MoGe.

The adapter must preserve the existing source-camera and transform contracts exactly.

### Phase C — Blender preparation adapter

Create deterministic collections:

```text
SCENE_VISUAL_SHELL
SCENE_EDITABLE
SCENE_GAMEPLAY_PROXY
SCENE_PROCEDURAL_MODULES
SCENE_REFERENCE_ONLY
```

For each asset:

- apply or validate transform;
- validate finite geometry;
- classify material policy;
- generate or attach collision proxy only when requested;
- preserve instances until concrete geometry is required;
- export an asset manifest and one GLB per independently replaceable asset family.

### Phase D — Unreal import adapter

Implement an Interchange import preset for:

- GLB/glTF;
- source name preservation;
- MikkTSpace;
- degenerates removal;
- LOD import when provided;
- collision import only when requested;
- destination paths derived from SceneSpec IDs.

Attach SceneSpec IDs and source hashes as Unreal asset metadata.

### Phase E — PCG pattern library

Do not let an agent invent the first production graphs from scratch.

Create and manually prove these project references:

```text
/Game/ScenePipeline/PCG/Reference/PCG_ExclusionMask
/Game/ScenePipeline/PCG/Reference/PCG_LargeObjects
/Game/ScenePipeline/PCG/Reference/PCG_SmallScatter
/Game/ScenePipeline/PCG/Reference/PCG_SplineModules
/Game/ScenePipeline/PCG/Reference/PCG_Intersections
```

Each reference graph needs:

- a known input fixture;
- expected node and pin manifest;
- expected attributes;
- deterministic seed;
- expected generated counts or bounds;
- screenshot or Data View proof;
- save/reload receipt.

### Phase F — Castlegrounds hybrid scene

Use the already-proven `castlegrounds_source_mesh_v2.glb` as the visual shell.

Add only bounded new layers:

1. simple castle gameplay proxy;
2. one river exclusion spline;
3. one bridge axis and modular bridge grammar;
4. one walkable surface;
5. decorative GPU grass outside exclusions;
6. collision and navigation proof;
7. live-editor source and offset captures.

Do not rerun MoGe unless later evidence proves the selected mesh itself is inadequate.

### Phase G — automated proof runner

Required outputs:

```text
scene_spec_validation.json
asset_import_receipt.json
pcg_graph_manifest.json
pcg_data_view_receipt.json
collision_probe.json
navigation_probe.json
map_save_reload_receipt.json
source_camera_render_receipt.json
offset_left_render_receipt.json
offset_right_render_receipt.json
parallax_comparison.json
FINAL_REPORT.md
```

Promotion requires all mandatory gates to be independently satisfied.

## 8. Acceptance matrix

| Gate | Smoke requirement |
|---|---|
| `SCENE_SPEC_VALID` | Schema and semantic checks pass |
| `SOURCE_CAMERA_COVERAGE_PROVEN` | Existing exact source-camera threshold remains satisfied |
| `UNREAL_INTERCHANGE_IMPORT_PROVEN` | Imported assets match expected paths and counts |
| `UNREAL_SAVE_RELOAD_PROVEN` | Fresh process reload preserves references |
| `PCG_REFERENCE_GRAPH_PROVEN` | Graph copied or adapted from a proved reference |
| `COLLISION_NAV_PROVEN` | Gameplay route traverses expected surface and is blocked by expected obstacles |
| `UNREAL_SOURCE_RENDER_PROVEN` | Render-capable Unreal path writes a deterministic source view |
| `UNREAL_PARALLAX_PROVEN` | Offset renders show correct bounded displacement and occlusion ordering |
| `GPU_BUDGET_PROVEN` | Peak VRAM is recorded and below the configured limit |
| `NO_COLLATERAL_CHANGE_PROVEN` | Source shell and unrelated scene layers remain unchanged |

Final classifications:

- `PROVEN`: all required gates pass;
- `PARTIAL`: useful intermediate proof exists but at least one required gate is incomplete;
- `BLOCKED`: a named external or environment limitation prevents a required gate;
- `REJECTED`: a result contradicts the target requirement.

## 9. Research incorporated

Primary references:

- Epic Games, **Working with PCG and LLMs Using Unreal MCP in Unreal Engine 5.8**  
  https://dev.epicgames.com/documentation/unreal-engine/working-with-pcg-and-llms-using-unreal-mcp-in-unreal-engine
- Epic Games, **Using PCG with GPU Processing**  
  https://dev.epicgames.com/documentation/unreal-engine/using-pcg-with-gpu-processing-in-unreal-engine
- Epic Games, **Using PCG Generation Modes**  
  https://dev.epicgames.com/documentation/unreal-engine/using-pcg-generation-modes-in-unreal-engine
- Epic Games, **Using Shape Grammar With PCG**  
  https://dev.epicgames.com/documentation/unreal-engine/using-shape-grammar-with-pcg-in-unreal-engine
- Epic Games, **Importing Assets Using Interchange**  
  https://dev.epicgames.com/documentation/unreal-engine/importing-assets-using-interchange-in-unreal-engine
- Yin et al., **AutoUE: Automated Generation of 3D Games in Unreal Engine via Multi-Agent Systems**  
  https://arxiv.org/abs/2603.07106

AutoUE's most relevant result for this project is not “use more agents.” It is that predefined PCG patterns, retrieved node definitions, parameter semantics, pin semantics, engine constraints, and automated play-testing materially improve correctness. The LowVRAM3D implementation should keep ChatGPT as planner and verifier, while deterministic adapters and reference patterns perform execution.
