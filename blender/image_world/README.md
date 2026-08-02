# Image World Blender Handoff

The Blender stage converts validated observations into a proof scene.

## Input

- point/depth/normal observations from MoGe
- surface projection report
- observed/generated masks
- confidence maps

## Collections

```
IMAGE_WORLD_SOURCE
    camera
    reference geometry

IMAGE_WORLD_TERRAIN
    observed terrain
    generated completion

IMAGE_WORLD_RESIDUALS
    cliffs
    structures
    landmarks

IMAGE_WORLD_DEBUG
    masks
    confidence
    validation renders
```

## Rules

- Never merge observed and generated geometry silently.
- Never bake buildings into the landscape heightfield.
- Never export to Unreal without a proof render and validation receipt.
- Hero assets are separate reconstruction jobs.

The lighthouse tower is a landmark asset. The terrain system only provides its supporting world context.
