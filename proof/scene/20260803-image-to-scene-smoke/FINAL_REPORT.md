# Castlegrounds image-to-scene smoke

Principal classification: `IMAGE_TO_SCENE_PARTIAL`

Full-pipeline classification: `FULL_IMAGE_TO_SCENE_PIPELINE_NOT_PROVEN`

This isolated lane produced a real source-visible edge-aware 2.5D mesh from
the Castlegrounds image and proved the raw reprojection, GLB transform, Blender
import transform, Unreal Interchange transform, map save, and fresh map reload.
It did not prove a complete source-camera visual match or a live Unreal render.
The prior arbitrary-camera Blender contact sheet is retained as diagnostic only.

## Locked inputs and route

- Starting commit: `201586f6f5bd43f9492dd56bb7b033d039e1e95e`
- Branch: `agent/scene-pipeline-smoke-20260803`
- Source: `C:\Users\Lauri\Downloads\benchmarkpics\castlegrounds.png`
- Source SHA-256: `e8ea9e307327169d998df9fd6757db718e5647ac46fc8235f971416e132df6ba`
- Model: `Ruicheng/moge-2-vits-normal`
- Inference input: `512x384` RGB
- Mesh: `balanced_010`, 191,847 vertices and 250,574 triangles
- Geometry regeneration after the starting commit: `NOT_RUN`
- Source plane: `ABSENT`

## Proven CPU and transform gates

`MOGE_RAW_SOURCE_REPROJECTION=PROVEN`

- normalized K: `[[0.7625050545,0,0.5],[0,1.0166734457,0.5],[0,0,1]]`
- median reprojection error: `0.0000131 px`
- p99 reprojection error: `0.0000436 px`
- max reprojection error: `0.0000775 px`
- mesh incident coverage: `100%`
- mean/max RGB error: `0 / 0`

`MOGE_RAW_GLB_TRANSFORM=PROVEN`

- `M_raw_moge_to_glb = I` within max residual `1.40e-11`
- GLB has one mesh node with no explicit non-identity node transform.

`MOGE_GLB_BLENDER_TRANSFORM=PROVEN`

- `M_glb_to_blender = [[1,0,0],[0,0,-1],[0,1,0]]`
- determinant `+1`; handedness preserved
- fit max residual `3.89e-12`
- embedded normals require inverse-transpose under any later non-identity transform.

`MOGE_GLB_UNREAL_TRANSFORM=PROVEN`

- axis fixture measured `M_glb_to_unreal = [[100,0,0],[0,0,100],[0,100,0]]`
- raw X maps to Unreal X, raw Y to Unreal Z, raw Z to Unreal Y
- scale `100`; determinant is negative because the measured axis conversion flips handedness.

`SCENE_SAVE_RELOAD_PROVEN`

- map loads in a fresh commandlet;
- all required actor labels are present;
- imported mesh is present;
- source camera FOV is `66.5083847°`, aspect ratio `4:3`;
- source camera location is `[0,0,0]`, rotation is `[pitch=0,yaw=90,roll=0]`;
- no source image plane is present.

## Exact source-camera gate

`BLENDER_EXACT_SOURCE_CAMERA=REJECTED_MESH_COVERAGE`

The exact Blender camera was derived from the audited transform, with origin
`[0,0,0]`, forward `[0,-1,0]`, up `[0,0,-1]`, right `[1,0,0]`, and the measured
MoGe FOV. The cull-off render is directionally aligned, but the mesh has large
holes/edge rejections:

- valid source-mask coverage: `97.5784%`
- rendered non-black coverage inside mask: `71.2755%`
- silhouette IoU inside mask: `0.712755`
- edge correlation: `0.327606`
- mean absolute colour error: `44.3416`
- cull-on render: all visible faces culled, `REJECTED_ALL_VISIBLE_FACES_CULLED`

Therefore the exact camera and transform are proven, but the source-facing
visual gate is rejected for the current balanced mesh. This localizes the next
repair to mesh coverage/winding/edge handling, not to MoGe reprojection or
global transform guessing.

The older `blender_source.png` and contact sheet were produced with an
externalized CPU visual camera. Their classification is
`BLENDER_EXTERNAL_VIEW_DIAGNOSTIC_ONLY`; they are not source-camera proof.

## Unreal visual and parallax gates

`UNREAL_VISUAL_RENDER=BLOCKED`

The map and camera contract are proven, but the headless commandlet produced no
PNG captures and no live UnrealMCP viewport was available. Known project-level
GameFeatureData warnings remain in the commandlet log; they do not invalidate
the successful Python reload receipt.

`PARALLAX_UNREAL=NOT_PROVEN`

The saved analytical depth displacement is retained only as diagnostic data:
near/far lateral displacement ratio `9.7729x`. Its receipt explicitly requires
rendered occlusion comparison, which is unavailable while the exact source
camera gate and Unreal PNG capture gate are unresolved.

## Evidence

External proof root:
`C:\AI\ScenePipelineSmoke\20260803\castlegrounds\`

Repository proof root:
`proof/scene/20260803-image-to-scene-smoke/`

Key evidence includes `raw_reprojection_receipt.json`,
`glb_transform_audit.json`, `blender_transform_audit.json`,
`blender_exact_source_receipt.json`,
`blender_exact_source_comparison.json`,
`unreal_axis_transform_audit.json`, `scene_build_receipt.json`, and
`map_reload_receipt.json`.

## Next bounded action

Do not rerun MoGe or regenerate candidates. Repair the current balanced mesh
using the saved arrays only, targeting the proven source-camera mesh holes and
edge rejection. Re-run the exact Blender source-camera gate, then attempt live
Unreal capture if an editor/bridge is available. Keep the principal
classification `IMAGE_TO_SCENE_PARTIAL` until those gates pass.
