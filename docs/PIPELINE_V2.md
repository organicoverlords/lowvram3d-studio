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

## Still requiring a human

- Final approval of the textured asset. The pipeline rejects; it does not approve.
- `MISSING_THIN_FEATURES`, `FACE_UNREADABLE` and `CROSS_COMPONENT_PROJECTION` are **advisory**:
  their detectors are weaker than the defects they describe, so they are reported and not enforced.
- Whether a generated mesh being mirrored relative to the source illustration is acceptable.
- Anything a profile cannot express: art direction, silhouette intent, "is this the right character".
