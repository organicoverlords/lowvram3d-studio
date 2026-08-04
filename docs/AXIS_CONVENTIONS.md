# Axis conventions, measured

Three sessions guessed at how a glTF mesh lands in Unreal, and at least one
guess was wrong in a way that cost days: a reconstruction placed at the yaw the
standard convention predicts rendered an almost-empty frame, and the workaround
was to frame on the scene's centroid at roughly yaw −50°.

Everything below was measured. Nothing here is inferred from a convention
document, and the receipts are in `evidence/axis-probe/`.

## 1. Unreal's glTF importer

`workers/make_axis_probe_glb.py` writes a mesh with one small box per axis at a
*different* distance along it (+X at 1 m, +Y at 2 m, +Z at 3 m), so each axis is
individually identifiable in both direction and sign.
`unreal/measure_axis_mapping.py` imports it and reads the vertex positions back
out of the StaticMesh.

| glTF axis | Unreal axis | measured offset | off-axis leakage |
|---|---|---|---|
| +X | **+X** | 100 cm | 0.0 cm |
| +Y | **+Z** | 200 cm | 0.0 cm |
| +Z | **+Y** | 300 cm | 0.0 cm |

**glTF (x, y, z) arrives as Unreal (x, z, y), at 100 cm per glTF metre.**

Two consequences worth stating plainly:

- The importer already applies the metre→centimetre conversion. Scaling the
  actor by 100 on top of it once put a 544 m scene at 25 km across.
- Swapping Y and Z is a reflection (determinant −1). That is *correct*: it is
  what converts right-handed Y-up glTF into left-handed Z-up Unreal. A mesh that
  is already correct in glTF needs no further correction, and one that is
  mirrored on import is mirrored in the source file.

A glTF camera looking down −Z therefore looks down Unreal **−Y**, which is
**yaw −90**, not yaw 0.

## 2. The MoGe reconstruction was upside down

`unreal/measure_reconstruction_orientation.py` recovers the image→world
orientation of an imported reconstruction without assuming anything: MoGe writes
the source image's pixel coordinates as UVs, so the vertices with v ≈ 0 came
from the image's top row and those with v ≈ 1 from its bottom row, and the world
direction between the two groups *is* the image's up direction.

Measure the bands as **directions from the camera**, not as mean positions. Mean
position mixes direction with depth, and the two differ wildly across one image:
the top band is sky hundreds of metres out while the bottom band is ground a few
metres away, so the difference of their mean positions points nowhere useful.
This gave a plausible-looking 129° first answer that was an artefact.

Measured on `barn_auto_moge.glb`:

- camera forward `(0.00, −0.99, −0.11)` → yaw **−89.9°**, matching the probe
- image right `(1.00, 0.01, −0.04)` → **+X**, as predicted
- image up `(−0.21, 0.08, −0.97)` → **−Z**: the image's up pointed *down*
- handedness dot **−0.98**: mirrored, so **no camera pose could frame it**

That last line is the whole story. A vertical mirror is not a rotation, so no
yaw would ever have worked, and scoring kept preferring a flipped render.

## 3. The fix, and the proof it is the right one

The defect is in the export, not the importer. `moge_reconstruct.py` negated
both Y and Z, which is right if MoGe returns OpenCV points (X right, Y **down**,
Z forward). It does not — its Y is already up — so the extra negation inverted
every reconstruction. Only Z is negated now, and face winding is reversed with
it because a single-axis negation is a reflection.

Verified before changing the exporter, by transforming an existing
reconstruction (`workers/reorient_reconstruction.py`) and re-rendering it
against the source across a yaw sweep (`scripts/verify_axis_mapping_prediction.py`):

| yaw | before (combined) | after (combined) | after: flip needed |
|---|---|---|---|
| 0 | 0.134 | 0.152 | identity |
| −50 (the old workaround) | **0.301** | 0.098 | vflip |
| **−90 (measured)** | −0.026 | **1.439** | **identity** |
| 90 | 0.136 | 0.136 | vflip |
| 180 | 0.088 | 0.173 | hflip |

Correlation at yaw −90 goes from −0.229 to **+0.780**, silhouette IoU from 0.150
to **0.526**, and the winning render needs no flip at all. The old −50°
workaround was the best of a set of bad options and drops to 0.098 once the real
defect is gone.

## 4. What this means for generated assets

Hunyuan3D Mini Turbo writes Y-up glTF like every other generator here, so its
output arrives upright in Unreal with no rotation applied — Y→Z is exactly the
up-axis conversion. Scale generated meshes from their own imported bounds rather
than from an assumed unit size, and do not apply a second ×100.
