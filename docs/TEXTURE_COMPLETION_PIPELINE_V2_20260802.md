# Low-VRAM texture completion pipeline v2 — implementation task

## Goal

Improve the current single-image texture route so side and rear surfaces preserve readable material identity without ever projecting the front face onto the back.

The immediate benchmark is the existing **sanitized 5-step tactical red panda geometry**. Do not regenerate geometry. Keep all source images, original GLBs, sanitized GLBs, and previously accepted/rejected textured artifacts immutable.

## Current proven state

The current raster projector correctly requires both depth visibility and front-facing normals. Mirrored fallback views are barred from semantic projection, and unobserved triangles receive synthesized material rather than front-view atlas reuse.

The remaining weakness is not rear-face ghosting. It is low-detail, flat, weakly separated material synthesis on side and rear surfaces.

The current comparison also exposes two atlas problems that must be fixed before adding diffusion:

- the UV stage reports a requested texture size of 1024 while the packed base-colour image is only 512x512;
- the current smart-UV candidates have approximately 14–15% estimated utilization and non-trivial overlap estimates.

## Research-derived architecture

Implement a practical low-VRAM approximation of the ideas used by TEXTure, Text2Tex, Paint3D, MVPaint, and synchronized multi-view texturing:

1. preserve already observed texels;
2. divide every rendered view into **KEEP**, **REFINE**, and **GENERATE** regions;
3. choose the next view by maximum still-unobserved visible surface area;
4. generate only the missing regions, conditioned by geometry and the source image;
5. project generated pixels back through the same depth/normal visibility gates;
6. harmonize colour and level seams in UV space;
7. preserve per-texel provenance so generated texture can never be mistaken for observed texture.

Do not copy an entire external framework. Implement the minimum shared stages in the existing pipeline.

---

# Phase A — CPU-only atlas and propagation correction

## A1. Enforce one atlas-resolution contract

The requested atlas size must equal:

- UV pack resolution;
- raster projection resolution;
- saved base-colour resolution;
- embedded GLB texture resolution;
- validation-report resolution.

For this benchmark use **1024x1024** throughout.

Fail closed with `ATLAS_RESOLUTION_CONTRACT_MISMATCH` if any stage disagrees.

Add tests proving that a requested 1024 atlas cannot silently become 512.

## A2. Restore production UV packing

Use the existing xatlas/UV-quality route rather than the current low-utilization planar/smart-UV result.

Run deterministic candidate presets and select lexicographically by:

1. zero or negligible exact polygon overlap;
2. no out-of-range UVs;
3. no material triangle loss;
4. highest atlas utilization;
5. lower chart count as tie-breaker.

Acceptance for the panda benchmark:

- exact overlap gate passes;
- utilization >= 55%;
- no UVs outside [0,1];
- no texture loss after fresh import.

Do not accept raster pixel-collision overlap as the exact overlap metric.

## A3. Replace Euclidean donor fill with surface-graph propagation

The current donor policy uses same component, normal agreement, and bounded Euclidean radius. Keep those as safety gates, but propagate low-frequency appearance over a welded triangle adjacency graph rather than direct 3D nearest-neighbour transfer.

Build weighted adjacency only across true shared mesh edges. Edge cost must increase with:

- geodesic edge length;
- normal discontinuity;
- curvature discontinuity;
- UV/material-region boundary evidence;
- crossing between disconnected welded components.

Observed triangles are fixed Dirichlet constraints. Solve or iteratively diffuse only low-frequency colour statistics onto unobserved triangles.

Use robust per-triangle statistics, not individual semantic pixels:

- median Lab colour;
- chroma range;
- luminance range;
- optional low-frequency 3x3 directional gradient summary.

This must preserve material regions better than a global or Euclidean prior while remaining incapable of copying a recognizable face to the rear.

Record per triangle:

- observed;
- graph-propagated;
- component prior;
- global emergency prior.

## A4. Add region-aware material priors

Cluster observed triangles into a small deterministic set of appearance regions using observed Lab colour plus geometry features. Do not use asset-specific labels or coordinates.

Suggested upper bound: 8 regions.

Examples the algorithm should naturally separate on the panda:

- orange tail fur;
- dark tactical cloth;
- brown/green ghillie material;
- face fur;
- dark rifle/equipment.

Unobserved triangles may inherit only from graph-reachable regions with compatible normal/curvature context. Report fallback counts.

## A5. CPU-only benchmark before diffusion

Re-texture only sanitized 5-step geometry with the improved atlas and graph propagation.

Produce:

- front / three-quarter / side / rear renders;
- provenance atlas;
- material-region atlas;
- seam-distance atlas;
- report comparing old neutral fill versus graph propagation.

This phase must not alter geometry and must not use GPU generation.

---

# Phase B — masked geometry-conditioned view completion

Run Phase B only after Phase A passes structural and visual gates.

## B1. Render canonical completion inputs

Render at 512x512 for:

- front;
- right three-quarter;
- right side;
- rear;
- left side;
- left three-quarter;
- optional top only when it reveals meaningful unseen area.

For every view save:

- current colour render;
- depth;
- world normal;
- triangle ID;
- UV coordinates;
- observed-confidence image;
- KEEP / REFINE / GENERATE trimap.

Trimap rules:

- `KEEP`: high-confidence observed texels; immutable;
- `REFINE`: narrow transition band around observed/generated boundaries;
- `GENERATE`: visible unobserved surface only;
- background and occluded triangles: forbidden.

## B2. Coverage-driven next-best-view order

At each iteration, score candidate views by the visible area of currently unobserved triangles, discounted for grazing angles and tiny fragmented regions.

Choose the highest-scoring next view rather than a fixed order. Stop when:

- no view adds meaningful new coverage; or
- a maximum of four generated views has been used.

Record the score and added coverage for every chosen/rejected view.

## B3. Low-VRAM inpainting backend

Do not use Hunyuan3D-Paint, Paint3D, MVPaint, or a new large multi-view stack for this implementation.

Use an already-installed Stable Diffusion 1.5/2.x inpainting-compatible backend only when available, with:

- depth ControlNet or equivalent geometry condition;
- source-image reference conditioning through IP-Adapter or an existing equivalent;
- model CPU offload;
- VAE tiling/slicing;
- FP16 UNet where supported;
- one 512x512 view at a time;
- deterministic seed;
- low-to-moderate denoise strength;
- no batch multi-view inference.

Do not download a new multi-gigabyte model automatically. Capability-check the existing local environments. If the required inpainting, depth-control, and image-reference components are unavailable, finish Phase A and report:

`BLOCKED_OPTIONAL_VIEW_INPAINT_BACKEND_NOT_INSTALLED`

Do not return to the previously blocked MV-Adapter debugging lane during this task.

## B4. Protect identity and observed texels

The inpainting image must start from the current render. The source image is reference conditioning, not a texture pasted onto hidden surfaces.

After generation:

- restore KEEP pixels bit-for-bit from the input render;
- allow REFINE pixels only limited colour correction;
- accept generated pixels only inside GENERATE;
- reject background or occluded pixels;
- reject source-facing pixels on rear-facing triangles through the existing depth/normal gate.

Back-project generated pixels only to the exact visible triangle IDs from that view.

Generated views must be recorded as `source_type=generated`, with confidence below the real front view.

## B5. Sequential consistency

After every accepted generated view:

1. project it into the atlas;
2. render all candidate views again from the updated atlas;
3. recompute trimaps and next-best-view scores;
4. use the updated render as the next inpainting initialization.

Do not independently generate all views from the original front image.

## B6. Reject bad generated views automatically

Reject a generated view when any of these occurs:

- KEEP-region mean absolute RGB change exceeds the configured tolerance;
- silhouette or triangle ownership changes outside GENERATE;
- near-constant output;
- insufficient colour diversity;
- major source identity drift in already visible face/equipment regions;
- generated view adds less than the minimum new surface coverage;
- front-face semantic detail appears on rear-facing regions;
- NaN/Inf or blank output.

A rejected view must not modify the atlas.

---

# Phase C — colour harmonization, seams, and high-resolution merge

## C1. Per-view colour harmonization

Before back-projection, estimate a robust affine colour transform for each generated view using only overlap with high-confidence existing texels.

Fit in Lab or linear RGB with outlier rejection. Apply the transform only to REFINE and GENERATE pixels.

Record transform parameters and overlap residuals.

## C2. UV seam leveling

Identify corresponding UV seam edges through welded 3D positions.

Run two levels:

1. global per-chart colour-offset solve across seam correspondences;
2. local seam-band gradient/Poisson or mask-normalized feather blend.

Observed high-confidence interiors remain fixed. Never blur across unrelated UV charts or material-region boundaries.

## C3. Native 1024 merge

The generative completion may run at 512, but the final atlas must be 1024:

1. upsample only the generated low-frequency regions;
2. reproject the original source image natively at 1024 over observed regions;
3. restore KEEP texels from the high-resolution projection;
4. run seam leveling;
5. embed the 1024 atlas in the GLB.

Do not claim generated 1024 detail where only 512 synthesis exists. Report observed and generated effective resolution separately.

---

# Acceptance gates

Test first on sanitized 5-step panda geometry only.

Required:

- no new geometry generation;
- geometry hash/count unchanged through texturing;
- 1024 atlas contract proven;
- exact UV overlap gate passes;
- UV utilization >= 55%;
- rear face absent;
- KEEP texels preserved within tolerance;
- observed/generated provenance embedded in reports;
- side/rear show clearer tail, ghillie, backpack, cloth, and rifle material separation than the current neutral-fill baseline;
- no black holes;
- no cross-component colour bleeding;
- fresh Blender import succeeds;
- front / three-quarter / side / rear contact sheet produced;
- runtime and peak VRAM recorded.

Classify separately:

- `UV_ATLAS_V2`
- `GRAPH_MATERIAL_PROPAGATION`
- `VIEW_COMPLETION_BACKEND`
- `GENERATED_VIEW_ACCEPTANCE`
- `SEAM_LEVELING`
- `OBSERVED_TEXEL_PRESERVATION`
- `FRESH_IMPORT`
- `FULL_AROUND_TEXTURE_BASELINE`

Only after sanitized 5-step passes, run the exact same route on sanitized 4-step for comparison. Do not change repaired v7 during implementation.

# Tests

Add focused CPU tests for:

- atlas size contract;
- exact-overlap gate integration;
- graph propagation cannot cross disconnected components;
- graph propagation respects strong normal/material boundaries;
- trimap partition is exhaustive and non-overlapping;
- KEEP pixels are immutable;
- generated pixels cannot project onto occluded or rear-facing triangles;
- generated view rejection leaves the atlas byte-identical;
- seam leveling does not cross unrelated islands;
- per-texel provenance totals equal atlas occupancy.

Run focused tests first. Do not require the unrelated nvdiffrast environment-order test to pass for this CPU/diffusers work, but report the full-suite result honestly.

# Scope limits

Do not:

- regenerate panda geometry;
- rerun frog during this implementation;
- restart MV-Adapter debugging;
- install Hunyuan3D-Paint, Paint3D, or MVPaint;
- overwrite immutable artifacts;
- add a new orchestration framework;
- perform rigging, animation, retopology, or LOD work;
- promote a result based only on metrics without fresh-import renders.

Commit shared code/tests first. Then run the sanitized 5-step panda benchmark. Push the validated commit, verify local equals remote, and leave a clean worktree.