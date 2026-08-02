# Semantic terrain projection

The projection stage accepts two modes.

## Diagnostic mode

No semantic package is supplied. MoGe validity and normal alignment are used only to build an unclassified surface diagnostic.

Classification:

`UNCLASSIFIED_SURFACE_BASELINE_NOT_TERRAIN_PROOF`

This mode must never be promoted to Blender or Unreal production terrain.

## Semantically filtered mode

A validated semantic package is supplied. Only pixels accepted as terrain candidates are allowed into world-up estimation, XY bounds, point rasterization, completion, and hydrology.

Classification:

`SEMANTICALLY_FILTERED_TERRAIN_BASELINE_NOT_SOURCE_QUALITY_PROOF`

This proves that the code path excludes pixels marked as water, sky, vegetation, structures, residual objects, or unresolved. It does not prove that an upstream segmentation model classified the source image correctly.

## Promotion remains blocked until

- the segmentation backend and weights are recorded;
- source-view overlays show no significant structure or vegetation leakage;
- water and coastline masks are reviewed;
- observed and generated regions are reported separately;
- Blender proof rendering succeeds;
- Unreal Landscape import and scale validation succeed.
