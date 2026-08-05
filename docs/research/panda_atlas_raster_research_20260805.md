# Panda atlas rasterization prework — 2026-08-05

## Decision boundary

The current production owner rasterizer is exact **point sampling at texel centres**. That
contract is valuable for direct provenance because one texel resolves to one explicit triangle and
one barycentric point. It must not be silently replaced by dilation, broad neighbouring-ID
acceptance, or conservative rasterization.

The diagnostic lane added here measures a second, separate quantity: whether a UV triangle has
**positive-area intersection with a texel cell**. The second mask is occupancy evidence only. A cell
found only by the conservative pass is a `VISIBLE_SOURCE_GAP` candidate, never
`ORIGINAL_DIRECT` or `GENERATED_OBSERVED`.

## Primary-source findings

- NVIDIA's conservative-rasterization definition includes pixels whose pixel cell intersects the
  polygon. This is the right model for detecting surface footprint, but not for proving a sampled
  source observation.
- NVLabs documents nvdiffrast rasterization as point-sampled through the pixel centre and warns
  that triangles can disappear when they are tiny relative to pixels. Rendering at higher
  resolution can reduce misses, but it does not create source provenance for an atlas cell.
- xatlas exposes packing controls including padding and bilinear-aware packing. Padding protects
  chart sampling boundaries, but it does not solve triangles whose interiors contain no texel
  centre.
- Blender's seam bleed extends values outside UV islands to reduce filtering artifacts. Bleed is a
  post-ownership sampling repair and must not precede ownership/provenance classification.

## Pipeline hypothesis under test

1. Keep centre-sampled ownership unchanged for direct observations.
2. Compute conservative occupancy separately.
3. Classify `conservative && !centre_owned` as unsampled surface footprint.
4. Repair only inside the same triangle or same chart, recording the repair provenance.
5. Apply chart-aware padding/bleed after evidence fusion.
6. Keep exact generated-view triangle-ID matching; do not restore the legacy mismatch waiver.

## Bounded worker output

The self-hosted Windows workflow runs synthetic unit tests, audits the canonical panda UV mesh at
512 and 1024, freshly renders selected existing diagnostic GLBs, builds contact sheets, and uploads
one immutable ZIP. It does not regenerate neural views, UVs, geometry, cameras, or production
assets, and it does not overwrite the golden panda.
