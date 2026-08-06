# Generic surface-registration forensics — 2026-08-06

The panda face investigation exposed a general pipeline gap: appearance
landmarks can be projected onto multiple UV charts and depth layers, so a
pixel translation is not a valid ownership rule.

`workers/surface_registration_forensics.py` is the model-agnostic diagnostic
utility for this problem. It accepts:

- a GLB mesh with `POSITION`, `NORMAL`, `TEXCOORD_0`, and indices;
- a JSON camera contract containing named right/up/origin vectors;
- an optional boolean triangle mask supplied by the caller.

It derives UV chart IDs from complete shared UV edges, records selected
triangle centroids/normals/UVs, and emits projections for every supplied
camera. It contains no character names, absolute asset paths, triangle ID
tables, or semantic exceptions. A character-specific mask or landmark fixture
is external evidence, never production logic.

## Contract

1. Registration evidence must identify the semantic region and its owning
   surface triangles before texture edits.
2. A chart/triangle may only be remapped when its camera visibility, normal,
   depth, and landmark correspondence are proven.
3. Translation-only atlas edits are diagnostic experiments, not ownership
   fixes.
4. Failed character fixtures remain rejected receipts; they cannot alter the
   generic classifier.

The focused generic regression suite passes (`2 passed`).

## External fixture run

The utility was exercised against the current 2048 mesh as an external
fixture, not as production logic:

- mesh SHA256: `1c4bec32a31ce90766f12f095e2d3c9c7e57e0b34e4f8ffc79c9a29ea8e7a817`
- selected fixture triangles: `42,509`
- derived UV charts across the mesh: `112,971`
- manifest: `C:\AI\panda_support_local_2048\face_chart_manifest_fixture\surface_registration_manifest.json`

The fixture confirms that the visible face error is a depth/parallax mismatch
between the front registration and an oblique camera. It does not authorize a
texture shift. A future correction must provide explicit landmark-to-surface
correspondences for the relevant camera set.
