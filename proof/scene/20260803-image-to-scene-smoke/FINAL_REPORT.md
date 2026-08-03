# Castlegrounds image-to-scene lane

Principal classification: `IMAGE_TO_SCENE_PARTIAL`

Full-pipeline classification: `FULL_IMAGE_TO_SCENE_PIPELINE_NOT_PROVEN`

Starting commit: `cdb0aeeed71920161d99177fcdd4c1d0ebdda383`

This continuation preserved the original MoGe arrays and historical GLBs,
audited the custom mesher, ran one bounded stronger-model comparison, selected
and versioned a source-visible mesh, and proved fresh Blender import and exact
source-camera coverage. Unreal v2 import/save/reload is proven. Deterministic
Unreal PNG capture remains blocked in the current headless/null-rhi host, so
rendered Unreal parallax is not promoted.

## Scope and immutable evidence

- Branch: `agent/scene-pipeline-smoke-20260803`
- Source: `C:\Users\Lauri\Downloads\benchmarkpics\castlegrounds.png`
- Source SHA-256: `e8ea9e307327169d998df9fd6757db718e5647ac46fc8235f971416e132df6ba`
- MoGe arrays used for CPU repair: `points.npy`, `depth.npy`, `normal.npy`,
  `mask.npy`, `intrinsics.npy`
- MoGe reruns were added only as separate comparison directories; the original
  arrays and `balanced_010.glb`, `strict_005.glb`, `permissive_020.glb`, and
  official baseline were not overwritten.

## Phase 1 — face coverage versus vertex coverage

`MOGE_RAW_SOURCE_REPROJECTION=PROVEN` remains unchanged: median error
`0.0000131 px`, p99 `0.0000436 px`, max `0.0000775 px`, and 100% incident
vertex coverage for 191,847 valid source pixels.

The old balanced custom mesher was not equivalent to that vertex coverage:

| Measure | Result |
|---|---:|
| valid vertex pixels | 191,847 |
| valid 2×2 cells | 190,840 |
| invalid-mask boundary cells | 4,873 |
| cells with two accepted triangles | 123,149 |
| cells with one accepted triangle | 4,276 |
| cells with zero accepted triangles | 63,415 |
| balanced accepted triangles | 250,574 |
| balanced rasterized face coverage | 69.3949% |
| old Blender cull-off missing valid pixels | 55,107 / 191,847 = 28.7245% |
| old Blender cull-on missing valid pixels | 191,847 / 191,847 = 100% |

The exact cause split is therefore:

- `WINDING_CULLING`: all source-facing pixels disappear with culling enabled;
- `EDGE_REJECTION_OR_MISSING_FACE`: the cull-off render still misses 55,107
  valid source pixels, corresponding to the rejected-cell population above;
- `INVALID_MASK_BOUNDARY`: 4,873 cells are rejected because they touch invalid
  mask regions;
- `GLB_RASTERIZATION_PRECISION`: not observed as the primary cause;
- other cause: not observed.

Evidence: `rejection_reason_counts.json`,
`accepted_face_count_per_cell.png`, `rejected_cell_reason.png`,
`source_face_coverage_mask.png`, and `blender_missing_pixel_overlay.png`.

## Phase 2 — winding repair

`CASTLEGROUNDS_WINDING_REPAIR_PROVEN`

The repair reverses every accepted face `[a,b,c] → [a,c,b]` and regenerates
consistent vertex normals. Vertex positions, triangle membership, UV ownership,
camera transforms, and reflection/axis conventions are unchanged. For the
selected candidate, cull-off and cull-on exact-source coverage are both
`99.6205%` with zero delta. The old cull-on blank result is not used as final
proof; double-sided materials are not the repair.

## Phase 3/4 — candidate comparison

Existing candidates, rendered with the exact audited Blender camera, were
rejected for source coverage until adaptive construction:

| Candidate | Exact coverage / IoU | Cull delta | Components |
|---|---:|---:|---:|
| strict 0.005 winding | 52.16% | 0 | 308 |
| balanced 0.010 winding | 69.79% | 0 | 345 |
| permissive 0.020 winding | 83.75% | 0 | 253 |
| adaptive conservative + winding | 96.48% | 0 | 130 |
| adaptive balanced + winding | 99.11% | 0 | 18 |
| adaptive coverage + winding | 99.62% | 0 | 9 |

The selected adaptive policy evaluates local relative depth per source cell,
chooses the safer diagonal, accepts one triangle when only one half is locally
continuous, and refuses invalid-mask bridges. It does not globally fill holes.

## Bounded stronger-model comparison

The official existing MoGe GLB first reached `92.74%` exact source coverage,
proving the custom global-edge mesher was a major problem. Separate fresh
comparisons then ran one GPU process at a time:

| Run | Inference | Exact coverage / IoU | PSNR | Mean colour error | Peak reserved |
|---|---:|---:|---:|---:|---:|
| VITS 768 | 4.02 s | 95.51% | 16.14 dB | 15.22 | 1.08 GB |
| VITB 512 | 5.28 s | 95.01% | 15.85 dB | 16.67 | 1.35 GB |
| VITB 640 | 4.32 s | 96.79% | 17.14 dB | 13.59 | 1.35 GB |

VITB 640 is the strongest fresh-model diagnostic, but the CPU adaptive
coverage candidate remains the selected geometry because it reaches the
highest exact source coverage (`99.62%`) while retaining local depth boundaries.
VITB 640 artifacts remain preserved under `moge_comparison/vitb_640/`.

## Phase 6 — versioned v2 GLB

`CASTLEGROUNDS_SOURCE_VISIBLE_MESH_REPAIR_PROVEN`

- GLB: `C:\AI\ScenePipelineSmoke\20260803\castlegrounds\castlegrounds_source_mesh_v2.glb`
- SHA-256: `2ee23fd3816f44931a25628392fa5962b23a55dcc17e207540dd3d937e940011`
- vertices: `191,847`
- triangles: `374,959`
- connected components: `9`
- boundary edges: `6,071`
- non-manifold edges: `0`
- degenerate triangles: `0`
- geometry hash: `05de7363a9a7a69510c8845eed3bb40aa92f31c4d3fda6d3bc82533f358bde69`
- UV hash: `1ddf05330776071b518cc0a0c2546a968256f688705f31f0f9ff728f1917003c`

Fresh Blender import and exact source-camera render are proven:

- coverage / silhouette IoU: `99.6205%`
- cull-off/cull-on delta: `0`
- source-camera transform contract unchanged
- positions, triangle order, and UV ownership hash-equivalent to the selected
  adaptive candidate

## Phase 7 — Unreal v2

`UNREAL_INTERCHANGE_IMPORT=PROVEN`

The GLB imported at:

`/Game/AgentProof/ImageToSceneSmoke_20260803/Geometry/CastlegroundsSourceMeshV2/`

The importer bound one material, saved the existing dedicated map, and the
fresh-process reload receipt proves the map actor still references the v2
static mesh. Historical balanced assets remain separate.

## Phase 8/9 — Unreal capture and parallax

`UNREAL_VISUAL_RENDER=BLOCKED`

The preferred `SceneCaptureComponent2D → TextureRenderTarget2D → export` probe
was attempted. The current headless Unreal Python environment lacks the
required transient render-target factory/API, and null-rhi produced no PNGs.
The receipt records the valid v2 mesh reference and the exact API blockers.

`PARALLAX_UNREAL=NOT_PROVEN`

Blender offset renders exist as CPU diagnostics, but they are not a substitute
for the required Unreal rendered occlusion comparison. The prior analytical
near/far displacement ratio remains diagnostic only.

## Final state and next blocker

`IMAGE_TO_SCENE_SOURCE_VISIBLE_2P5D_SMOKE_PROVEN` is not claimed because the
Unreal deterministic source render and rendered parallax gates are incomplete.
The honest final result is `IMAGE_TO_SCENE_PARTIAL`.

Next action: run the same saved Unreal map through a live editor/bridge or a
non-null-rhi render-capable commandlet, capture the v2 source/offset/depth
outputs, and only then evaluate rendered parallax. Do not rerun MoGe or replace
the selected v2 mesh unless that evidence requires it.
