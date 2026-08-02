# Image-to-World implementation foundation

**Branch base:** `fix/highres-production-pipeline-20260731` at
`6f5a34d04d31e74fb7dd66677f42aa0a86fc230a`.

This first slice adds stable contracts and deterministic planning only. It does
not claim that any ML reconstruction backend works on the target PC yet.

## Operating rule

Models produce observations. Deterministic stages produce deliverables.
Observed, adjusted, and generated regions must remain distinguishable in every
report and package.

## Routes

- `isolated_asset`: transparent trees, castles, rocks, buildings and landmarks.
- `diorama_map`: isometric boards and map-like world concepts.
- `perspective_vista`: cinematic landscapes such as the lighthouse archipelago.
- `composite_scene`: multi-object scenes that must first produce a reviewable
  decomposition instead of pretending to be a finished world.

## Foundation delivered

- Versioned `ObservationPackage` contract with strict validation and JSON
  round-tripping.
- Deterministic measurable-signal router with explicit ambiguity reporting.
- Route-aware resumable stage DAG.
- GTX 1660 SUPER 6 GB hardware profile with one heavy GPU worker and a 5.6 GB
  ceiling.
- Unreal-compatible preview and production terrain profiles.
- Canonical benchmark identities and proof requirements for the nine supplied
  concepts.
- Unit tests for contracts, routing, stage ordering, hardware policy and
  benchmark integrity.

## Heavy-GPU policy

The target profile permits one heavy worker at a time. Segmentation, MoGe and
optional image-to-3D proxies must run in separate processes and release their
memory before the next worker starts. CPU terrain reconstruction, hydrology,
erosion, mask generation and validation follow afterward.

## Preserve-first policy

- Preserve supplied alpha channels exactly as the visual-alpha source.
- Never encode trees, towers, bridges or buildings into the terrain heightmap.
- Never allow optional generative completion to overwrite observed terrain.
- Ambiguity produces `manual_review_required=true`; it does not delete valid
  observations or source assets.
- Composite scenes stop at a decomposition review package until their parts are
  independently proven.

## Next implementation slices

1. Add the isolated MoGe-2 environment and a measured GTX 1660 SUPER hardware
   probe. No compatibility claim is allowed without peak VRAM, wall time,
   artifact hashes and process-exit proof.
2. Implement perspective point-map rasterization into visible top-down terrain,
   with sample-count, variance and visibility-confidence rasters.
3. Implement diorama board detection, homography rectification and separation
   of placed trees/rocks/structures from terrain elevation.
4. Add deterministic heightfield completion, Priority-Flood depression handling,
   D8 flow and lightweight erosion.
5. Generate Blender proof scenes, then Unreal Landscape packages only after the
   structural gates pass.

## Current proof status

- Contracts and route DAG: unit tested in an isolated source-layout harness.
- Fixture SHA-256 values, dimensions and alpha statistics: measured from the
  supplied files.
- MoGe-2 on the target GPU: **NOT PROVEN**.
- Perspective camera/world-up recovery: **NOT PROVEN**.
- Diorama rectification: **NOT PROVEN**.
- Procedural tree reconstruction: **NOT PROVEN**.
- Unreal Landscape import: **NOT PROVEN**.
