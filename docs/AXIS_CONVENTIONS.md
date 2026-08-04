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

## 2. The reconstruction rendered upside down

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

## 3. The real defect was the UV convention — and the render metric endorsed the wrong fix

**trimesh flips v when it writes glTF.** It holds UVs with the origin at the
bottom left and flips on export to reach glTF's top-left origin. So authoring v
in image-row order (row 0 → v 0) exports an *inverted texture*: the sky gets
painted along the ground. `moge_reconstruct` now authors v inverted on purpose,
so the export flip lands it right way up.

The axis conversion was never wrong. MoGe returns OpenCV points with Y **down** —
confirmed against the model itself, which puts the top of the image at Y −8.18
and the ground at +1.25 — so negating both Y and Z is correct, and being two
negations it is a rotation, leaving winding alone.

### The wrong fix, and why it looked right

An earlier pass this session removed the Y negation instead. That flipped the
geometry to match the inverted texture: two wrongs that cancel *from the source
camera* and from nowhere else. It scored **1.439** against the source at yaw −90
with no flip needed — a better score than the correct mesh gets.

It scored better *because* it was wrong. With the texture inverted, the ground
geometry is painted with the image's sky, so the render's upper region is bright
cloud. The source's upper region is also bright cloud. The metric rewarded a
luminance-layout coincidence. The correct mesh honestly renders black where MoGe
masked the sky away, and scores **0.533**.

**Do not use source-view similarity as a correctness gate.** It is the
photometric pipeline's own test, and this is the second time in this project
that it has certified a mesh that was wrong in world space.

### What settles it, without rendering anything

Read the GLB's accessors directly and ask two questions of each mesh: are the
vertices carrying the *top* of the texture higher in Y than those carrying the
bottom, and where in the v range are the sky rows MoGe masked away?

| mesh | v range | texture-top verts | texture-bottom verts |
|---|---|---|---|
| **corrected** | **0.050–1.000** | **+8.16** | **−1.24** |
| geometry-flipped ("the wrong fix") | 0.000–0.950 | +1.24 | −8.16 |
| original | 0.000–0.950 | −1.24 | +8.16 |

The sky is masked, so the absent rows must sit at the *top* of the texture —
v ≈ 0. Only the corrected mesh has that, and only it puts the texture's top on
the geometry that came from the top of the image. Both facts are properties of
the file, checkable in a second, and neither can be faked by a coincidence of
brightness.

Yaw −90 remains correct, and the corrected mesh wins its sweep with `identity`:

| yaw | corrected (combined) | flip needed |
|---|---|---|
| 0 | 0.137 | vflip |
| −50 (the old workaround) | 0.159 | vflip |
| **−90 (measured)** | **0.533** | **identity** |
| 90 | 0.136 | vflip |
| 180 | 0.138 | rot180 |

## 4. What this means for generated assets

Hunyuan3D Mini Turbo writes Y-up glTF like every other generator here, so its
output arrives upright in Unreal with no rotation applied — Y→Z is exactly the
up-axis conversion. Scale generated meshes from their own imported bounds rather
than from an assumed unit size, and do not apply a second ×100.
