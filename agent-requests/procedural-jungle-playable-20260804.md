# Codex task: create and prove a playable procedural jungle in Unreal Engine 5.8

You are running non-interactively on the repository's trusted Windows self-hosted worker. Own the scene design, implementation, diagnosis, testing, visual proof, performance tuning, and repository hygiene. Continue beyond a grey blockout. The required result is a realistic, dense, explorable jungle with a river and waterfall, generated without downloaded or marketplace art assets and playable in Unreal Engine.

## User goal

Create a complete procedural jungle scene with:

- a readable river that begins upstream, bends through the jungle, drops as a waterfall, and forms a lower pool;
- realistic terrain composition: elevated canopy ridges, eroded banks, wet rock faces, cliffs, muddy shelves, exposed roots, and walkable clearings;
- multiple visibly distinct tree, palm, sapling, fern, broadleaf, vine, log, rock, and ground-cover forms;
- animated foliage/wind, moving water, waterfall spray/mist/foam, atmospheric depth, lighting, collision, and a controllable player;
- no downloaded meshes, textures, scans, Megascans, marketplace packs, or third-party art assets;
- final persistence as a standalone Unreal project that opens and plays.

Generated assets are allowed and expected. Engine plugins and code are tools, not external art assets.

## Terminal states

Continue autonomously until exactly one is justified:

- `JUNGLE_PLAYABLE_PROVEN`
- `HARD_BLOCKER`

Do not stop at `USER_REVIEW_REQUIRED` merely because visual review is useful. Produce deterministic screenshots, a contact sheet, measurable scene inventory, audit receipts, a standalone launch smoke test, and a performance capture. Manual review can refine art later, but the first complete playable scene must be delivered now.

## Canonical repository and branch policy

The GitHub Actions checkout is the writable source of truth for generic scripts, tests, task documentation, compact receipts, and proof metadata.

Expected repository:

- `organicoverlords/lowvram3d-studio`
- branch: `feature/procedural-jungle-playable-20260804`
- base: `agent/scene-pipeline-smoke-20260803`

Do not edit `main`. Do not merge unrelated image-to-3D work. Do not commit from inside Codex; leave intended repository changes for the workflow's independent gate and commit step.

The generated Unreal project is a separate deliverable at:

`C:\Users\Lauri\Desktop\ProceduralJungle58\ProceduralJungle58.uproject`

The canonical large-output root is:

`C:\AI\ProceduralJungle\20260804`

The target project directory may be created if absent. If it already exists, verify it contains a marker proving it belongs to this task before changing it. Never delete or overwrite an unrelated project.

## Hardware and software constraints

Target machine:

- Windows 10
- NVIDIA GTX 1660 SUPER, 6 GB VRAM
- Ryzen 5 5600G
- 16 GB RAM plus page file
- Unreal Engine 5.8: `C:\Program Files\Epic Games\UE_5.8`
- Blender 5.2: `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`

Design for a sustained minimum of 30 FPS at 1920x1080 on an evidence camera route under an appropriate medium-quality profile. Prefer 45+ FPS, but do not destroy density or composition merely to inflate a benchmark. Use bounded generation, instancing, cull distances, LODs or Nanite where measured appropriate, and avoid 3D Niagara fluid grids on this 6 GB GPU.

One heavy GPU process at a time. Do not interfere with unrelated interactive Unreal, Blender, Python, or CUDA work. Record relevant running processes. If another process makes rendering unsafe, continue all CPU/editor-generation work possible and classify only the blocked proof narrowly; do not kill user processes.

## Current official Unreal techniques to inspect and use where practical

The installed UE 5.8 tree is authoritative for exact APIs. Inspect plugin descriptors and local headers/content instead of guessing.

Prioritize:

1. Procedural Content Generation Framework (PCG) for biome distribution, density masks, slope/height/water-distance filtering, and hierarchical or partitioned generation.
2. Procedural Vegetation Editor (PVE) plus Dynamic Wind for native, botanically varied, Nanite-ready vegetation if its installed API can be automated reliably.
3. Geometry Script / Dynamic Mesh for terrain-adjacent procedural meshes, waterfall ribbon geometry, cliff shelves, roots, rocks, and baked static meshes.
4. Water, Landmass, PCG Water Interop, Niagara, and Single Layer Water for the river and pools.
5. Niagara sprite/ribbon particles for waterfall spray, foam, mist, droplets, and insects. Do not use expensive 3D liquid simulation for the playable version.
6. World Partition or PCG partitioning only if it materially helps this roughly one-kilometre scene. Do not add operational complexity without a measured benefit.

PVE, PCG Biome Core, Geometry Script, and Water contain experimental/beta surfaces. Use them behind a generated-content boundary and retain a deterministic fallback. An experimental plugin is not proof by itself.

## No-external-art rule

Fail the acceptance gate if any project asset outside these sources is referenced:

- assets generated by scripts in this task;
- Unreal engine primitives/default shaders/plugin code needed as implementation scaffolding;
- project code, Blueprints, materials, Niagara systems, PCG graphs, and textures generated by this task.

Forbidden:

- Marketplace/Fab/Megascans content;
- downloaded texture images, HDRIs, meshes, sounds, scans, or foliage packs;
- copying sample art from PCG Biome Sample, Water examples, Starter Content, or templates as final scene art;
- using Engine content meshes as disguised final vegetation or rocks.

A primitive may be used temporarily during generation or as hidden collision, but visible final art must be generated. Generate any required procedural noise/masks internally. Record every visible static mesh, material, texture, Niagara system, and sound asset with origin `GENERATED_THIS_TASK`.

## Required architecture

Implement a reproducible generator, not a one-off hand-edited level.

Repository deliverables should include a supported entrypoint such as:

`powershell -File scripts/procedural_jungle/build-procedural-jungle.ps1`

and source under locations such as:

- `scripts/procedural_jungle/`
- `blender/procedural_jungle/`
- `unreal/procedural_jungle/`
- `tests/procedural_jungle/`

The entrypoint must:

1. verify repository, branch, engine, Blender, target project identity, and process safety;
2. create or incrementally repair the dedicated Unreal project;
3. generate source meshes/material data deterministically from recorded seeds;
4. import/bake assets and build the map;
5. compile project code if used;
6. save, close, reload, and audit the map;
7. run gameplay and visual/performance proof;
8. write `acceptance.json` and compact evidence.

It must be resumable by stage and must not erase the last proven project when a later stage fails.

## Scene design specification

### World scale and composition

Create an explorable region approximately 800-1200 metres across. The player route should form a 3-6 minute loop rather than a flat showcase circle.

Use a coherent vertical composition:

- upper river source at a shaded plateau;
- river bends toward a cliff lip;
- waterfall drop approximately 25-60 metres;
- lower pool with mist and wet rock amphitheatre;
- downstream river or overflow channel;
- at least two secondary paths, one elevated overlook, one close waterfall approach, and one calmer riverbank clearing;
- terrain boundaries concealed by ridges, dense canopy, fog, and composition rather than an obvious square edge.

Generate terrain from deterministic multi-octave/domain-warped noise plus authored analytic masks. Carve the river corridor before scattering vegetation. Ensure slopes and cliff faces are geologically readable and the player cannot trivially see under or through terrain.

### Terrain and surfaces

Build a Landscape or another collision-capable terrain representation with:

- large-form ridges and valleys;
- erosion-inspired channels and talus;
- flattened but natural walkable shelves;
- riverbed and pool depressions;
- waterfall cliff and undercut recess;
- material layers based on slope, elevation, curvature, wetness, river distance, and procedural noise.

The terrain material must visually distinguish at least:

- leaf litter / dark soil;
- mud / wet bank;
- mossy rock;
- exposed lighter rock;
- underwater riverbed.

Do not rely on imported texture maps. Use generated textures, Material Noise/Voronoi, world-aligned/triplanar logic, vertex/weight masks, or generated render targets. Minimize expensive per-pixel procedural noise by baking reusable generated textures where it improves performance.

### Vegetation generation

Create multiple families with seed variation and clear silhouettes:

- 2-3 canopy tree species, including at least one buttress-root tree;
- one emergent giant tree form;
- one palm family;
- 2 understory tree/sapling forms;
- ferns;
- broadleaf ground plants;
- vines/lianas;
- fallen logs, root clusters, and deadwood;
- sparse flowers or fungi as accents, generated procedurally.

Each family needs multiple variants through seed, branching, lean, taper, trunk radius, crown shape, leaf scale, and hue variation. Avoid identical repeated trees.

Preferred route:

- generate hero/canopy species with PVE when its installed API is stable and automatable;
- otherwise generate them through deterministic Blender Python/Geometry Nodes or Unreal Geometry Script and import/bake them;
- use HISM/ISM or PCG static-mesh spawning for population;
- use LODs, Nanite, or both only after actual triangle/instance/performance evidence.

Leaves must use generated geometry and a two-sided foliage material. They may use simple low-poly leaf meshes or clustered cards, but no external alpha texture. If cards are used, generate the alpha/normal mask procedurally. Add wind through PVE Dynamic Wind, World Position Offset, Pivot Painter-like generated metadata, or a bounded custom vertex scheme.

Place vegetation ecologically:

- exclude deep water and primary walking paths;
- prefer ferns and broadleaf plants in wet/shaded regions;
- place buttress trees on flatter deep soil;
- expose roots near banks and slopes;
- increase moss and wetness near the waterfall;
- vary canopy density to create readable light shafts and navigation landmarks.

### Rocks, cliffs, debris, and detail

Generate several rock families with noise-displaced geometry and varied scale/orientation. Include cliff plates, boulders, river stones, mossy wet rocks, and small ground debris. Avoid sphere-like silhouettes.

Use spline/curve generation for roots and vines. Ensure collision is limited to gameplay-relevant trunks, cliffs, large roots, rocks, and logs; tiny foliage must not trap the player.

### River, waterfall, and pool

Use the UE Water system for spline river and horizontal pool surfaces where reliable. The vertical waterfall must use a dedicated generated ribbon/sheet or layered mesh rather than forcing a horizontal Water Body actor to behave vertically.

Required visual layers:

- moving river surface with direction following the spline;
- shallow/deep colour variation;
- bank foam and turbulence at bends/rocks;
- waterfall main sheet with breakup and panning flow;
- secondary strands;
- impact foam;
- mist/spray/droplets using efficient Niagara sprites/ribbons;
- wetness and moss response on nearby rocks;
- lower pool surface and downstream continuation.

Use Single Layer Water or another physically sensible material. Refraction, depth fade, normal motion, foam, and colour must be generated internally. Water collision/overlap must be sufficient for gameplay logic and audit, even if swimming is not implemented.

### Lighting and atmosphere

Build a realistic humid-jungle daylight setup:

- directional sun with warm broken shafts;
- skylight;
- exponential height fog / volumetric fog tuned for depth without whitening the scene;
- local mist around waterfall;
- restrained post-process colour grading;
- exposure locked or tightly bounded for proof renders;
- optional cloud layer only if performance permits.

Benchmark at least two rendering configurations if Lumen, Virtual Shadow Maps, or dense Nanite vegetation threaten the target GPU. Select the best measured visual/performance tradeoff and record rejected configurations.

### Playability

Provide a controllable walking character or generated capsule character with:

- WASD movement;
- mouse look;
- jump;
- gravity and floor collision;
- a third-person or first-person camera;
- spawn in a readable clearing facing the river route;
- no dependence on downloaded template art.

A generated primitive body is acceptable. A flying DefaultPawn is not sufficient.

Build navigation/collision so the player can:

1. walk from spawn to the river;
2. reach an overlook of the waterfall;
3. descend or follow a path to the lower pool;
4. approach the waterfall without falling through terrain;
5. complete a route back toward a clearing or downstream area.

Add invisible blocking only where necessary for terrain boundaries or lethal drops; do not wall off the entire scene visibly.

## Implementation phases

### Phase A: preflight and research the installed engine

- Record repo root, branch, HEAD, remote, status, engine version, Blender version, Codex version, GPU/VRAM, RAM, and relevant processes.
- Inspect installed `.uplugin` descriptors and exposed Python/C++ APIs for PCG, PVE, Dynamic Wind, Geometry Script, Water, Landmass, Niagara, Niagara Fluids, and PCG interop plugins.
- Write `engine_capabilities.json` with exact availability and chosen/fallback route.
- Create the target project marker before any destructive project action.

### Phase B: deterministic source generation

- Implement seed-controlled terrain masks and height data.
- Generate vegetation, rocks, roots, vines, and waterfall geometry.
- Generate all material/noise/mask textures internally.
- Add pure-Python tests for deterministic seeds, finite geometry, bounds, river continuity, no deep-water vegetation, path clearance, and asset provenance.
- Produce low-resolution source previews before full import.

### Phase C: Unreal project and asset build

- Create/repair the dedicated UE 5.8 project.
- Enable only required plugins in the project file.
- Add minimal project C++ or Blueprint code only where it improves reliability, particularly for player controls, deterministic generation, audit, or gameplay smoke tests.
- Import or bake generated meshes and create material instances/graphs.
- Build PCG graphs or deterministic HISM population.
- Save the canonical level:

`/Game/ProceduralJungle/Maps/L_ProceduralJungle`

### Phase D: water, waterfall, lighting, and atmosphere

- Build river spline, pool, waterfall meshes/materials, Niagara systems, fog, lighting, wind, and post process.
- Generate fixed proof cameras: spawn, river bend, waterfall overlook, waterfall base, lower pool, and canopy/path view.
- Save and reload before claiming success.

### Phase E: gameplay and collision proof

- Build and run an automated path/collision audit.
- Use line traces and/or an automation pawn to prove ground continuity and obstacle collision at route checkpoints.
- Launch standalone game or `-game` mode with the canonical map using a non-null rendering path.
- Prove the configured player class, game mode, input mappings, movement, jump, and camera setup exist and initialize.
- Capture a short deterministic traversal or at minimum multiple rendered frames from in-game cameras.

### Phase F: performance and visual quality

- Collect primitive/instance/triangle/material counts, VRAM if available, frame times, and scalability settings.
- Benchmark the fixed camera route at 1920x1080.
- Optimize ground cover density, cull distances, shadows, translucency, Niagara counts, LODs/Nanite, and water cost based on measurements.
- Do not hide failure by lowering resolution below 1920x1080 for the final benchmark.

### Phase G: independent audit and reproducibility

- Close the editor.
- Reopen with `UnrealEditor-Cmd.exe` and run a read-only audit script.
- Verify map loads, expected assets resolve, generated provenance is complete, no forbidden external content references exist, and all critical actors/components are present.
- Run the supported build command from the intended clean repository state and prove it can resume without replacing accepted outputs.

## Required tests and gates

Add deterministic tests for at least:

1. same seed -> same terrain/river/asset manifest hashes;
2. river centreline is continuous and descends toward the waterfall;
3. waterfall lip and lower pool elevations are ordered correctly;
4. vegetation exclusion from deep water and clear path corridors;
5. minimum species and variant counts;
6. finite/non-degenerate generated meshes and valid normals/UVs or explicit triplanar material contract;
7. collision assignment policy;
8. every visible asset has generated provenance;
9. no forbidden `/Game/Megascans`, `/Game/StarterContent`, Fab, sample-content, or external source references;
10. map save/reload audit;
11. player/game mode/input setup;
12. water/river/waterfall actor/material presence;
13. screenshot non-black/non-empty checks;
14. performance receipt schema and minimum target gate;
15. rerun/resume safety and project identity marker.

## Required outputs

Large outputs under:

`C:\AI\ProceduralJungle\20260804`

Required:

- `acceptance.json`
- `pipeline_receipt.json`
- `engine_capabilities.json`
- `scene_inventory.json`
- `asset_provenance.json`
- `performance.json`
- `gameplay_smoke.json`
- `map_reload_audit.json`
- `root_cause_and_decisions.md`
- fixed-camera screenshots at 1920x1080 or higher;
- labelled contact sheet;
- standalone/game launch log;
- Unreal commandlet audit log;
- exact commands and environment manifest.

Project deliverable:

`C:\Users\Lauri\Desktop\ProceduralJungle58\ProceduralJungle58.uproject`

Repository proof:

- `proof/scene/20260804-procedural-jungle/FINAL_REPORT.md`
- compact JSON receipt under `evidence/latest-procedural-jungle/`

Do not commit large `.uasset`, `.umap`, generated meshes, full screenshots, or the Unreal project to this Git repository. Upload compact proof and use workflow artifacts for selected renders/logs.

## Acceptance contract

`JUNGLE_PLAYABLE_PROVEN` requires all of the following:

1. Dedicated Unreal project exists and opens under UE 5.8.
2. Canonical map saves and reloads in a fresh Unreal process.
3. No forbidden external/downloaded art assets are referenced.
4. Terrain, river, waterfall, lower pool, vegetation, rocks, lighting, fog, wind, and waterfall effects are present.
5. At least 8 vegetation/ground-detail families and 20 total deterministic mesh variants are generated, unless PVE assets provide equivalent measured variety.
6. River continuity, waterfall elevation ordering, and pool placement pass deterministic geometry checks.
7. Player game mode, pawn/character, movement, camera, gravity, jump, spawn, and collision initialize in standalone/game mode.
8. Route checkpoint traces prove a traversable path from spawn to river, overlook, and lower pool.
9. At least six fixed-camera images are non-empty, correctly exposed, and show distinct scene regions; the waterfall must be clearly visible in at least two.
10. The final 1920x1080 evidence route sustains at least 30 FPS average and no fixed camera falls below 20 FPS for more than a transient sample. Record exact settings.
11. Generated asset provenance is complete and forbidden references are zero.
12. A supported one-command generator/rebuilder exists in the repository and its focused tests pass.
13. Final project and proof outputs are not placeholders, greybox-only, or simple primitive scatter.

Machine-readable `acceptance.json` must include at least:

```json
{
  "schema": "procedural_jungle_acceptance_v1",
  "classification": "JUNGLE_PLAYABLE_PROVEN",
  "project_exists": true,
  "map_save_reload": "PROVEN",
  "no_external_art_assets": "PROVEN",
  "terrain": "PROVEN",
  "river": "PROVEN",
  "waterfall": "PROVEN",
  "lower_pool": "PROVEN",
  "vegetation": "PROVEN",
  "wind": "PROVEN",
  "lighting_atmosphere": "PROVEN",
  "player_controls": "PROVEN",
  "collision_route": "PROVEN",
  "standalone_launch": "PROVEN",
  "visual_capture": "PROVEN",
  "performance_1080p": "PROVEN",
  "tests_passed": true,
  "canonical_map": "/Game/ProceduralJungle/Maps/L_ProceduralJungle",
  "uproject": "C:\\Users\\Lauri\\Desktop\\ProceduralJungle58\\ProceduralJungle58.uproject",
  "contact_sheet": "C:\\AI\\ProceduralJungle\\20260804\\proof\\contact_sheet.png",
  "average_fps": 30.0,
  "minimum_fps": 20.0,
  "forbidden_asset_reference_count": 0
}
```

A successful command exit, existing `.uproject`, or structural map load alone is not enough. Do not mark proven until the complete acceptance contract is supported by fresh evidence.

## Final response

Write the Codex final response to the path supplied by `--output-last-message`. Include:

- exact project and map paths;
- architecture selected and fallbacks used;
- generated species/variant/instance counts;
- player and collision proof;
- render and performance results;
- tests and audit results;
- rejected approaches/configurations;
- output paths and hashes;
- final classification.
