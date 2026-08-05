# Panda atlas support fix — 2026-08-06

This changeset addresses the verified root cause of the black stippling: the
CPU atlas had direct texel-centre owners, while many positive-area UV
triangles had no centre sample. The GPU still sampled those continuous UV
footprints, so unwritten atlas space became visible.

## Contract implemented

- Direct centre ownership remains unchanged and is never replaced.
- UV charts are derived from complete shared indexed UV edges; 3-D proximity
  is not used to join charts.
- Positive-area triangle/pixel-cell intersections are resolved into a separate
  `CONSERVATIVE_SURFACE_SUPPORT` layer using closest-point barycentrics.
- Same-chart ties are deterministic (UV distance, then triangle ID).
- Cross-chart collisions remain unresolved and are serialized rather than
  silently assigned.
- Gutter expansion is chart-local and is never reported as provenance.
- Synthetic support atlases use green direct, orange conservative support, red
  cross-chart collision, blue gutter, and magenta unresolved classes.

## Workflow repair

`.github/workflows/panda-atlas-root-local-worker.yml` now flattens JSON
candidate manifests explicitly for Windows PowerShell 5.1, validates scalar
labels/paths, records per-candidate start/render receipts, continues after a
candidate render failure, writes `render_matrix.json`, packages partial
evidence under `if: always()`, and uploads a non-empty
`panda_atlas_support_fix_<run>.zip` with `if-no-files-found: error`.

The existing `LOWVRAM3D-KONE` runner is addressed by its exact labels and is
not re-registered or modified.

## Honest status

The support contract is diagnostic until the synthetic render gate reports no
visible magenta or black default pixels. The workflow therefore packages all
audit, support, render, test, hash, and log evidence even when a render or
synthetic gate fails. No neural views, geometry, UV master, golden asset, or
prior evidence package is overwritten.

The first worker run after this change is the source of truth for the measured
1024 support counts, chart inventory, unresolved/collision counts, render
matrix, ZIP hash, and visual verdict.
