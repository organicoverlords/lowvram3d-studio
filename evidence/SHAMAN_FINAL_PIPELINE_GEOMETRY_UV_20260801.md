# Shaman final-pipeline geometry, LOD and UV state

**Recorded:** 2026-08-01  
**Branch:** `infra/windows-self-hosted-runner-20260731`  
**Classification:** `GEOMETRY_AND_LODS_PROVEN_UV_GATE_REQUIRES_EXACT_RECHECK`

## Preserved high master

- path: `final-pipeline/high/shaman_high_master.glb`
- SHA-256: `db1818a3b804c64e38767fde3235dc388621f08a8e8dd3da9ffc3b9692c654d4`
- bytes: `24,237,656`
- preserved byte-for-byte and made read-only during the measured run
- comparison-only `shaman.fbx` was not modified or used as pipeline input

## Stage 1: conservative cleanup

- source components after welding: `90`
- confirmed removed components: `28`
- removed faces: `0.196%`
- clean master: `final-pipeline/high/shaman_high_clean.glb`
- clean triangles: `1,093,124`
- surviving connected components: `59`

The measured run reports that cords, bowl charm, leaf pendants, hollow pod, lantern, far-right pendant, staff ring/charm, robe fringe and strips survive cleanup. This remains subject to visual review of the committed/local preview set.

## Stage 2: semantic part separation

Not completed. Loose-part splitting is insufficient because the main component contains approximately `98.97%` of faces and spans the body, staff, antlers, cords and ornaments. Future movable-part extraction requires semantic segmentation or authored cuts. It is not a prerequisite for validating the existing fused-character LOD and UV route.

## Stage 3: LOD chain

| LOD | triangles | components | boundary edges |
|---:|---:|---:|---:|
| 0 | 219,967 | 59 | 100 |
| 1 | 89,965 | 59 | 109 |
| 2 | 39,961 | 59 | 115 |
| 3 | 14,951 | 59 | 151 |

All four LODs were reported as fresh-process validated. Constant component count is evidence that the LOD route did not fragment cords or fuse ornaments into the main body.

## Stage 4: provisional UV result

The first measured UV output reported:

- atlas resolution: `4096`
- raster-estimated utilisation: `47.99%`
- finite and in bounds: yes
- degenerate UV triangles: `0`
- analytic-area / raster-covered ratio: `1.1016`

The run stopped because that ratio was interpreted as approximately 10% genuine overlap.

## Corrected diagnosis

That interpretation is invalid. The repository had already demonstrated and documented that both of these metrics are unsuitable as an overlap acceptance gate:

1. counting raster texels covered by multiple triangles counts ordinary shared chart edges and sub-texel collisions;
2. comparing summed analytic triangle area with rasterised coverage conflates boundary rounding and sub-texel sampling with genuine positive-area overlap.

`src/lowvram3d/uv_overlap.py` is the canonical detector. It uses exact convex clipping and counts an overlap only when two UV triangles have a positive-area intersection. Shared vertices and edges are zero-area and do not fail the gate.

Commit `18b9b29fca28a99cae127e2e88021aec1472d660` updates `blender/final_pipeline_uv.py` to:

- remove the expensive 4096x4096 per-triangle Python raster loop;
- compute utilisation analytically;
- use exact positive-area intersection as the only overlap safety gate;
- retain a maximum overlap budget of `1.0` texel-equivalent;
- report low utilisation as a warning rather than confusing it with bake corruption.

## Exact next action

Rerun **Stage 4 only** against the existing `final-pipeline/game/shaman_lod0.glb`. Do not regenerate, clean or decimate geometry.

- If exact positive-area overlap is at or below `1.0` texel-equivalent, promote the UV GLB and start Stage 5 baking.
- If exact overlap exceeds the budget, only then try another unwrap/pack or semantic-region atlases.
- If exact validation times out, preserve the existing LOD0 and treat the UV stage as unproven rather than declaring overlap from a raster proxy.

## Remaining visual limitation

The reconstructed model is shallow in depth (`Y` extent approximately `0.595` versus `Z` extent approximately `1.981`). This is a single-view geometry limitation. UV, texture, A-pose and rigging cannot restore missing side/back volume.
