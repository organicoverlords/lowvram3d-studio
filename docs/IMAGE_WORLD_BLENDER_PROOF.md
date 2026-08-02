# Image World Blender Proof

This stage converts a surface-projection package into a diagnostic Blender scene.
It is not a production terrain export and cannot be promoted automatically.

## Required inputs

- `arrays/completed-height.npy`
- `arrays/observed-mask.npy`
- `arrays/generated-mask.npy`
- `arrays/confidence.npy`

## Required outputs

- `.blend` diagnostic scene
- fixed proof render
- diagnostic GLB containing only the terrain mesh
- JSON report with mesh counts and proof classification

## Scene contract

The scene contains four top-level collections:

- `IMAGE_WORLD_SOURCE`
- `IMAGE_WORLD_TERRAIN`
- `IMAGE_WORLD_RESIDUALS`
- `IMAGE_WORLD_DEBUG`

Observed, generated and low-confidence surface regions use distinct materials.
The mesh also stores `source_confidence`, `source_observed` and
`source_generated` point attributes.

## Promotion boundary

`promotion_allowed` must remain `false` until all of these are proven:

1. semantic terrain/water/vegetation/structure separation;
2. recovered source camera and source-view comparison;
3. residual cliff and landmark extraction;
4. valid Unreal Landscape height and scale conversion;
5. visual review of observed versus generated areas.

The current headless workflow uses a deterministic synthetic fixture to prove
that Blender can build, save, render and export the diagnostic scene. It does
not prove the real lighthouse reconstruction.
