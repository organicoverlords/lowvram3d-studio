# Geometry repair and texture run — measured results

Working asset: a red panda sniper in a ghillie suit, generated from a single front photograph.
Source image 1409x1117 (an upscale of an original 525x416).

Everything below is measured, not estimated. Raw reports are in `reports/`, the renders that
decisions were actually made from are in `renders/`, and the resulting asset is in `deliverable/`.

## Geometry

| | before | after |
|---|---|---|
| faces | 44,951 | 40,317 |
| connected components | 65 | 52 |
| boundary edges | 136 | 54 |
| non-manifold edges | 1 | 1 |
| main component faces | 36,705 | 36,705 |
| detached screen-space islands (6 views) | many | 0 |

The main component was never touched. All removal was individually proven debris: 4,634 faces
across 11 components, each confirmed as a separate screen-space island, outboard of the dilated
silhouette, with a >=2px gap, in multiple views. Boundary-edge count fell rather than rose, so no
removal opened a hole.

## UV

| | before | after |
|---|---|---|
| charts | "thousands" (fragmented) | 1,139 |
| native xatlas utilization | 27.89% | 81.03% |
| island occupancy at 2048 | 27.89% | ~58% |

Preset B (`maxCost=4.0`, `maxIterations=2`). B and C were exactly tied; B wins on the
less-aggressive-configuration tie-breaker.

## Texture

Single 2048 pass, ~22s.

| metric | value |
|---|---|
| observed_semantic_coverage_percent | 23.36 |
| synthesized_surface_coverage_percent | 76.64 |
| final_filled_uv_percent | 100.0 |

Fill tiers (triangles): observed 10,845 · constrained donor 17,585 · component-local prior 10,034 ·
global-prior emergency 2,855.

Observed and synthesized coverage are always reported separately. Synthesized pixels are never
counted as observed.

## Known limitations, stated plainly

* **The rear has no real material detail.** 77% of the surface has no source observation, so the
  fill is deliberately low-frequency. A single front image cannot produce accurate rear texture;
  that requires generated rear views, not a repair pass.
* **Rear geometry is inflated and rounded.** Equipment reads as smooth lobes away from the front.
  This is inherent to single-image reconstruction and is not a defect cleanup can address.
* **2 positive-area UV overlap pairs remain**, totalling 0.81 texel-equivalents across a
  4.19M-texel atlas. Under the <=1.0 texel budget, but not zero.
* **10 degenerate-UV triangles remain.** They are *not* degenerate in 3D (areas 1.2e-7 to 4.0e-5
  against a median of 1.8e-4); they are real but tiny triangles that the unwrap compressed to
  near-zero UV area. ~0.001% of surface area.
* **xatlas reports a 2656x2656 internal packing grid.** The returned UVs are normalized to
  [0.000188, 0.999812] with `atlas_count == 1`, so the 2048 deliverable is unaffected.
* **Rigging is untested.** The run stops at a validated unrigged export.

## Measurement bugs found and fixed

These invalidated earlier conclusions and are worth recording:

1. **Two UV overlap metrics were unsound.** Counting atlas pixels covered by more than one
   triangle counts the shared edge between every pair of adjacent triangles, which is ordinary
   connectivity; comparing summed analytic UV area against rasterised coverage conflates boundary
   rounding with overlap. Both reported ~10-25% "overlap" on a layout whose true overlap is
   0.81 texels. Replaced with exact Sutherland-Hodgman convex clipping
   (`src/lowvram3d/uv_overlap.py`, 12 unit tests including the folded-neighbour case). The old
   numbers survive only as `raster_collision_diagnostic_percent` and
   `analytic_area_to_raster_coverage_ratio`.

2. **Axis convention mismatch.** trimesh preserves the glTF Y-up convention while the pipeline's
   cameras are Blender Z-up, so the "front" camera was looking down the model's up-axis and the
   front-view source-support test was meaningless. Fixed as `blender = (x, -z, y)`.

3. **Front-view source support was a false positive.** Overlap with the source foreground scores
   100% for anything floating *in front of* the subject, which is how a detached plate above the
   head survived the first cleanup pass. That evidence is now inadmissible when a fragment is a
   separate screen-space island in >=3 views and >=80% outboard.

4. **Donor scoping collapsed to UV charts.** A UV unwrap duplicates vertices along every seam, so
   computing 3D connectivity from the unwrapped indices split one continuous surface into
   per-chart islands. Donors were therefore confined to a single chart and most triangles fell
   through to the global prior. Welding by position first moved 15,426 triangles out of the
   emergency tier (global prior 18,281 -> 2,855).

5. **Smooth shading was never applied on export.** `blender/common.py` provided `shade_smooth()`
   but the raster export path did not call it, so flat marching-cubes normals reached every GLB
   and read as polygonal banding once textured. It is a normals-only change: no vertex moves, and
   face count and UVs are unchanged.

6. **A stale canonical GLB was published.** An earlier `game_ready_unrigged.glb` was byte-identical
   to a previously rejected artefact, because stages were validated in scratch directories without
   re-running the export chain. The route now writes a candidate and promotes it only after
   fresh-process validation.
