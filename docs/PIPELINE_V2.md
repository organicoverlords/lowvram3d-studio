# Pipeline V2

A generic, resumable, self-correcting asset pipeline. Replaces the per-asset orchestration scripts
that required a human prompt to diagnose and repair each defect.

## One command

```powershell
.\Run-AssetPipeline.ps1 -Image "C:\path\to\character.png" -Profile Auto
```

Adopt an existing master and skip generation:

```powershell
.\scripts\run-asset-pipeline.ps1 -Image src.png -Profile Auto -ExistingMaster master.glb -ToStage CLEAN
```

## Why it exists

The shaman run produced a sequence of outputs that were technically valid and visibly wrong: a
material-ID map that was per-triangle noise while reporting full coverage and zero NaN; a UV atlas
whose overlap detector timed out and returned zeroes that read as "clean"; a base colour stored in
the wrong row convention that looked like a plausible patchwork rather than like a flip; a third of
the model painted one flat grey by a "component-local" prior on a mesh that is one component. Every
one of those passed a green build and needed a human to look at a render and say no.

V2's job is to say no by itself.

## State machine

`INGEST → GENERATE → GEOMETRY_QA → CLEAN → LOD → UV → BAKE → TEXTURE → TEXTURE_QA → PARTS →
RIG_READINESS → RIG → EXPORT`

Each stage:

- reads immutable inputs and hashes them into its receipt;
- writes into `state/<STAGE>/candidate/`;
- is promoted to `state/<STAGE>/proven/` **only** after its gate passes, so a failed retry can never
  destroy the last proven result;
- writes `state/<STAGE>/receipt.json` recording input hashes, output hashes, gate measurements,
  every attempt and the final verdict;
- fails closed - a gate that could not run is a failure, never a pass;
- is skipped on resume when its recorded input hashes match and its verdict was `passed`.

## Repair policy

Each failure code maps to exactly one bounded repair recipe. At most **two** retries per stage;
after that the stage stops and marks `needs_human: true`. Retrying until something passes is how a
pipeline learns to emit technically valid rubbish, so the budget is hard.

| Failure code | Repair |
| --- | --- |
| `UV_ROW_ORIENTATION_MISMATCH` | convert the atlas out of the projector's inverted row convention |
| `FLAT_NEUTRAL_ATLAS_REGIONS` | repaint priors from nearest observed same-component donors |
| `UNFINISHED_SYNTHESIS` | re-inject cavity/AO high frequencies into synthesized regions |
| `PLASTIC_ROUGHNESS` | raise the roughness floor, cut metallic |
| `MATERIAL_ID_NOISE` | weld by position before computing component ids |
| `FLOATING_DEBRIS` | widen the debris height band and strip again |
| `UV_OVERLAP` / `UV_DEGENERATE` | xatlas route, prune offending faces in place without re-unwrapping |
| `CAMERA_LABEL_MISMATCH` | flip the yaw convention and re-render |
| `BACKGROUND_CONTAMINATION` | require a stronger alpha before a texel may be projected |
| `BAD_ORIENTATION` | re-run upright reorientation |
| `REAR_MIRRORS_FRONT` | bar non-real views from semantic projection |

Codes with **no** recipe, on purpose. They mean an input is not what it claims to be, or that a
decision belongs to a human. A retry cannot make any of them true:

| Failure code | Meaning |
| --- | --- |
| `UV_MASTER_HASH_MISMATCH` | the canonical UV master on disk is not the one that was proven |
| `UV_MASTER_GEOMETRY_MISMATCH` | its surface or triangle count changed |
| `UV_GEOMETRY_CHANGED` | a rewrap altered positions or topology, which it must never do |
| `UV_INJECTIVE_WORKER_FAILED` | the rewrap worker died; distinct from it *reporting* an overlap |
| `LOD_TOPOLOGY_REGRESSED` | decimation tore the mesh. The recipe is to decimate less, which changes a delivery target, so it is a human's call |

A note on `UV_OVERLAP`'s recipe, because it was wrong for a long time in two compounding ways.
The stage read `route` from the manifest and never from `overrides`, so the switch to the xatlas
route could not fire and the stage re-ran the identical route until its budget was gone. And
`uv_xatlas_isolated.run_one` returned `timed_out` unconditionally - its timeout `return` sat in the
outer `try` body rather than the handler - which made the statements reading the child's real status
unreachable and left `child_report` unbound on the success path, so the route reported `failed` even
when it worked. Either bug alone hides the other. Both are fixed; if this recipe stops working
again, check that pair first.

## Stage routes

`UV` and `TEXTURE` each dispatch on a manifest-declared route. The default route is unchanged.

| Stage | Route | Worker | When |
| --- | --- | --- | --- |
| UV | `fast_blender` (default) | `blender/final_pipeline_uv.py` | general assets |
| UV | `xatlas` | `workers/uv_xatlas_isolated.py` | preset search over chart budget and stretch |
| UV | `existing` | `blender/validate_existing_uv.py` | adopt a UV mesh that already carries a material and packed texture |
| UV | `injective` | `workers/uv_rewrap_injective.py` | rebuild UVs so no texel interior is claimed twice |
| TEXTURE | `raster_project` (default) | `workers/raster_project.py` and the bake chain | general assets |
| TEXTURE | `mvadapter_sixview` | `workers/injective_atlas_texture.py` | fuse six already-generated MV-Adapter views onto an injective atlas |

The `injective` route is what a single-owner atlas actually requires. An atlas resolves each texel
to one triangle, so if the layout is not injective every other claimant of a texel displays a
colour computed for a surface it is not part of. On the red panda that showed up as the front face
appearing on the back of the head, and it was not repairable by fusion: at 2048 the sum of UV
triangle areas was 1.062e8 texels against an atlas holding 4.19e6. See
`UV_REWRAP_TEXTURE_REPAIR_20260804.md`.

When `uv.master` is declared, the `injective` route adopts that canonical master instead of
unwrapping, after checking its sha256, its geometry fingerprint and its injectivity.
`validate_existing_uv.py` is deliberately not used for this: it requires a material and a packed
texture, which a UV master legitimately does not carry.

`mvadapter_sixview` requires `uv.route == "injective"` and declares `BAKE` and `TEXTURE_QA`
inapplicable — it consumes no baked maps and emits no ORM, and running those stages to a green
`passed` on absent inputs is the exact failure mode V2 exists to prevent. The runner records them
as `not_applicable` rather than `passed`.

New failure codes: `UV_MASTER_HASH_MISMATCH`, `UV_MASTER_GEOMETRY_MISMATCH`, `UV_GEOMETRY_CHANGED`.
None has a bounded repair recipe — they mean the input is not what it claims to be, which is a
human's problem, not a retry's.

`uv.max_degenerate_uv_triangles` declares a measured allowance for zero-area UV triangles. It
defaults to `0`; it is never inferred from what a packer happened to produce.

## Profiles

`humanoid`, `humanoid_complex_accessories`, `quadruped`, `flying_creature`, `static_prop`,
`vehicle`, `building`, `environment_piece`.

`Auto` detects from the source silhouette using three explainable signals: aspect ratio,
bounding-box fill, and the fraction of subject area that disappears under an erosion sized as a
fraction of subject height (so the measure means the same thing at any resolution). When confidence
is low it falls back to `humanoid_complex_accessories` - over-preserving costs triangles, while
under-preserving destroys thin geometry that cannot be recovered.

Measured on the benchmark assets:

| Asset | aspect | fill | thin area | Detected |
| --- | --- | --- | --- | --- |
| shaman | 1.25 | 0.40 | 0.357 | `humanoid_complex_accessories` |
| red panda | 1.30 | 0.66 | 0.114 | `humanoid` |
| turbo bird | 1.15 | 0.63 | 0.142 | `static_prop` |

## Which prior failures are now caught automatically

Verified by `benchmarks/run_benchmarks.py`, which replays the real artifacts:

- `UV_ROW_ORIENTATION_MISMATCH` - sampled against recorded (uv, colour) truth in both conventions;
  the wrong one was off by 61.9/255 against 3.2/255.
- `FLAT_NEUTRAL_ATLAS_REGIONS` - per-UV-island colour spread; the flat-grey atlas had 186 flat
  islands over 57,495 triangles, the repaired one 37 over 921.
- `MATERIAL_ID_NOISE` - resolved component count, not pixel statistics.
- `UV_OVERLAP` from a **timed-out** detector - zeroes from an incomplete run are treated as failure.
- `PLASTIC_ROUGHNESS` - roughness percentile and spread.
- `CAMERA_LABEL_MISMATCH` - mirror-invariant colour correlation against the source.
- `FLOATING_DEBRIS` / `BAD_ORIENTATION` - component and axis-ratio analysis.

## Per-asset declarations

Some properties are facts about an asset, not defects. They are declared in the manifest with their
measured values and never inferred from the fact that a gate happened to fail:

| Key | Meaning |
| --- | --- |
| `uv.max_degenerate_uv_triangles` | zero-area UV triangles that are known 3D slivers owning no texels (default `0`) |
| `lod.allow_topology_regression_below_lod0` | thin features may be lost at distance LODs; LOD0 stays strict (default `false`) |
| `clean.stance_repair` | a pose whose legs are connected in the source art, where the repair could only open a rigging gap by deleting garment (default `true`) |
| `generator_settings.octree_ladder` / `.steps` | generator quality ladder (defaults to the proven `384:3000,320:2000,256:1500` at 5 steps) |
| `generator_runtime.python` / `.pythonpath` | the generator does not run in the pipeline's interpreter |

When a repair is skipped by declaration the receipt records that it was skipped and why. A silent
skip is indistinguishable from a repair that ran and passed, which is the failure mode V2 exists to
prevent.

## Still requiring a human

- Final approval of the textured asset. The pipeline rejects; it does not approve.
- `MISSING_THIN_FEATURES`, `FACE_UNREADABLE` and `CROSS_COMPONENT_PROJECTION` are **advisory**:
  their detectors are weaker than the defects they describe, so they are reported and not enforced.
- Whether a generated mesh being mirrored relative to the source illustration is acceptable.
- Anything a profile cannot express: art direction, silhouette intent, "is this the right character".
