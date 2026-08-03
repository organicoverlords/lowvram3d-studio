# Castlegrounds image-to-scene smoke

Principal classification: `IMAGE_TO_SCENE_PARTIAL`

Full-pipeline classification: `FULL_IMAGE_TO_SCENE_PIPELINE_NOT_PROVEN`

The primary MoGe route completed and produced a real source-visible, edge-aware
2.5D mesh. The mesh was exported as GLB, imported through Unreal 5.8
Interchange, saved into a dedicated map, and reloaded in a fresh commandlet.
The visual Unreal viewport gate is not proven: the headless commandlet produced
no PNGs, the scripted AutomationLibrary capture crashed in this host context,
and the project MCP listener was not active. CPU Blender renders are retained
as supplemental visual evidence, not substituted as Unreal proof.

## Locked inputs and route

- Base SHA: `ca9f1cdcb6713ba19acc054dc98aa399a5d08042`
- Branch: `agent/scene-pipeline-smoke-20260803`
- Source: `C:\Users\Lauri\Downloads\benchmarkpics\castlegrounds.png`
- Source SHA-256: `e8ea9e307327169d998df9fd6757db718e5647ac46fc8235f971416e132df6ba`
- Model: `Ruicheng/moge-2-vits-normal`
- Inference input: `512x384` RGB (`source_rgb.png`)
- Inference runtime: `3.293 s`; load-only cached runtime: `2.728 s`
- Peak VRAM: allocated `772,156,416` bytes; reserved `1,075,838,976` bytes
- Camera: `fov_x=66.5083847°`, `fov_y=52.3759155°`, normalized intrinsics
  `[[0.762505,0,0.5],[0,1.016673,0.5],[0,0,1]]`
- Camera direction: MoGe point-map `+Z` away from the source camera

## Mesh candidates

All three candidates were generated from the same validated point map:

| Candidate | Triangles | Components | Boundary edges | Aspect p95 / max |
|---|---:|---:|---:|---:|
| strict 0.005 | 182,496 | 308 | 17,760 | 2.164 / 4.221 |
| balanced 0.010 | 250,574 | 345 | 15,592 | 2.707 / 7.865 |
| permissive 0.020 | 305,800 | 253 | 13,154 | 3.379 / 13.643 |

Selected candidate: `balanced_010`, retained as the locked preference. It is
not promoted as a complete scene because side/rear visual quality and Unreal
render capture remain unproven.

## Proven gates

- `SOURCE_NORMALIZATION=PROVEN`
- `MOGE_LOAD_ONLY=PROVEN`
- `MOGE_INFERENCE=PROVEN`
- `EDGE_AWARE_MESH_EXPORT=PROVEN`
- `GLB_VALIDATION=PROVEN`
- `UNREAL_INTERCHANGE_IMPORT=PROVEN`
- `UNREAL_MAP_SAVE=PROVEN`
- `UNREAL_MAP_FRESH_RELOAD=PROVEN`
- `SOURCE_PLANE_PRESENT=false`
- `WALKABLE_GROUND=NOT_PROVEN`
- `UNREAL_VISUAL_RENDER=BLOCKED`
- `PARALLAX=NOT_PROVEN_AS_UNREAL_RENDER_GATE`

The existing CPU depth displacement receipt measured near/far displacement
ratio `9.7729x`, but its newly revealed geometry field explicitly requires
render occlusion comparison. It is retained as diagnostic evidence and is not
promoted to the Unreal parallax gate.

## Evidence paths

External proof root:
`C:\AI\ScenePipelineSmoke\20260803\castlegrounds\`

Key artifacts:

- `source_rgb.png`
- `points.npy`, `depth.npy`, `normal.npy`, `mask.npy`, `intrinsics.npy`
- `balanced_010.glb`, `balanced_010.ply`
- `moge_official_baseline.glb`, `moge_official_baseline.ply`
- `contact_sheet.png`
- `blender_source.png`, `blender_left.png`, `blender_right.png`,
  `blender_forward.png`, `blender_elevated.png`, `blender_rear.png`
- `unreal_import_receipt.json`
- `scene_build_receipt.json`
- `map_reload_receipt.json`
- `parallax_receipt.json`
- `render_manifest.json`

Repository proof root:
`proof/scene/20260803-image-to-scene-smoke/`

The next concrete action is to run the patched UnrealMCP viewport capture
against this saved map with a live editor/bridge, then update only the visual
render and parallax receipts. Do not regenerate the MoGe mesh for that step.
