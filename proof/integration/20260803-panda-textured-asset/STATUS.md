# Panda textured asset — in progress (2026-08-03)

Classification so far: **`PANDA_TEXTURED_ASSET_VISUAL_REPAIR_REQUIRED`** — a blocking
orientation defect was found upstream of texturing. Not `PROVEN`, and not a hard blocker.

## What passed

The canonical 384x20 run on the bar-repaired mesh with the corrected permutation is
numerically and structurally clean:

* status `PROVEN`, 20/20 steps, six outputs, finite gate passed
* structural gate passed — silhouette IoU against the controls 0.966–0.978 per view
* colour gate and semantic gate passed; filenames follow the contract
* front/rear correlation 0.046 direct, 0.249 mirrored — no rear-face duplication
* first attempt died at step 4 with a CUDA illegal memory access; the retry, identical in
  every respect, completed in 188 s. Transient device fault, recorded not hidden.

## What blocks the textured asset

**No generated view contains a recognisable panda face**, so texture gate 1 fails before any
projection is attempted. The cause is upstream of the MV-Adapter.

`CANONICAL_TRANSFORM` in `build_mvadapter_cpu_controls.py` maps mesh **Z** to the rig's up
axis. The asset's up axis is mesh **+Y**:

| evidence | value |
| --- | --- |
| longest mesh extent | Y, 1.952 (vs X 1.668, Z 1.522) |
| first principal axis | (0.298, −0.955, 0.013) — essentially ∓Y |
| orange tail centroid | (0.557, −0.572, −0.118), at 22% of the Y range |
| axis render with up=+Y | character stands upright (`orientation_probe/up_pY`) |

So the whole six-view rig orbits the wrong axis. Every view renders the character lying on
its side, and the camera that actually faces the head is handed the elevation +89.99
embedding — it is told it is looking straight down, so it generates a top-down image
instead of a face. That single error explains the sideways framing, the missing face, and
why the earlier symmetry probe could not separate the facing axis.

This supersedes the raw-index permutation question rather than answering it: with the up
axis corrected the six cameras point elsewhere entirely, so the mapping must be re-derived,
not patched.

## Required next steps

1. Re-express the mesh in canonical upright orientation (mesh +Y → rig up), or fix the rig's
   canonical transform. Geometry must be rotated, not re-meshed.
2. Rebuild the 384 controls from the reoriented mesh.
3. Re-derive raw-to-semantic from the corrected rig and confirm against landmarks.
4. Re-run 384x20.
5. Only then run `multiview_texture_projection.py`, which is written and unit-clean but has
   not been run on real views — feeding it face-less views would bake the defect into the
   atlas.

## Artifacts

* run: `C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout\sd21_repaired_384x20_retry_20260803`
* orientation probe: `…\tactical_red_panda_scout\orientation_probe\up_pY`
* repaired mesh: `…\bar_local_closure_v1\tactical_red_panda_scout_bar_repaired.glb`
  (sha256 `78c55133165e931bc8d6765610a679d1d18badcdc178820a69e31b7b32bcbfb8`)
