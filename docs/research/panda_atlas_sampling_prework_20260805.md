# Panda atlas sampling prework — 2026-08-05

## Scope

This lane isolates the visible black stippling and polygonal underside holes from every neural
and source-image variable. It uses the exact panda mesh, the production UV reader/rasterizer and
the production GLB binder, but replaces appearance with deterministic per-triangle colors.

The first worker run is diagnostic-only. It does not regenerate geometry, views or textures and
does not modify any preserved panda artifact.

## Why this test comes first

The newest failure affects the authoritative original-projected face and generated regions in the
same way. That makes neural-view quality an insufficient explanation. A unique-triangle atlas with
debug-magenta unowned space can distinguish:

- valid per-triangle texture support;
- zero/sub-texel UV triangles;
- UV orientation mismatch;
- CPU texel-center versus renderer sampling disagreement;
- accidental sampling of unowned atlas space;
- material/sampler leakage after fresh import.

## Primary-source research

### Conservative coverage

NVIDIA defines overestimated conservative rasterization as including every pixel whose pixel cell
has any non-empty intersection with the polygon. Standard center-sample rasterization cannot
guarantee this at any fixed resolution.

- NVIDIA GPU Gems 2, Chapter 42:
  https://developer.nvidia.com/gpugems/gpugems2/part-v-image-oriented-computing/chapter-42-conservative-rasterization

The production rasterizer currently assigns a texel only when its center lies inside a UV
triangle. That is valid for an injective ownership map, but it is not sufficient proof that
bilinear sampling of the continuous UV triangle will never reach uninitialized neighboring
texels.

### glTF sampler contract

The glTF 2.0 specification defines `9729` as LINEAR filtering and `33071` as CLAMP_TO_EDGE.
Linear filtering is a weighted sum of neighboring texels, so an atlas must provide valid same-chart
support around every sampled UV location; clamp-to-edge only controls coordinates outside the
whole image and does not protect internal UV chart boundaries.

- Khronos glTF 2.0 sampler specification:
  https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-sampler

### UV chart packing

xatlas generates unique texture coordinates and separates chart construction from chart packing,
allowing resolution, texel scale and packing options to be changed independently.

- Official xatlas repository:
  https://github.com/jpcy/xatlas

Injectivity alone does not guarantee a minimum renderable footprint for every triangle. The
diagnostic therefore records zero, one, 1–3, 4–8 and 9+ support texels per triangle.

### Classical multiview texturing

`mvs-texturing` remains a useful later comparison for view selection and seam leveling, but its
core assumption is pixel-accurate registration of images and cameras. That assumption should not
be evaluated until the atlas/UV/sampler round trip is proven independently.

- Official repository:
  https://github.com/nmoehrle/mvs-texturing

## Worker deliverable

The self-hosted Windows worker will upload one ZIP containing:

- exact input and immutable buffer hashes;
- per-triangle UV area and texel support arrays;
- unique-triangle atlas and GLB;
- orientation atlas and GLB;
- fresh-import six-view unlit renders;
- debug-magenta/black pixel counts;
- optional unlit renders of the latest local real panda diagnostic;
- contact sheets;
- logs, receipts and SHA-256 manifest.

## Decision rule

Do not change MV-Adapter, source registration, fusion or completion based on this run.

- Visible debug magenta or unexpected black in the synthetic render:
  `PANDA_ATLAS_CONTRACT = REJECTED`.
- Clean synthetic render but poor real panda:
  atlas transport is cleared and work moves to ownership, chart-local support and evidence fusion.
