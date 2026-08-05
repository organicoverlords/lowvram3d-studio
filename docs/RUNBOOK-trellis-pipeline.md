# TRELLIS.2 image-to-asset pipeline — runbook

How to take a single concept image to a textured GLB on this machine, and why
each stage is the way it is. Every command here has been run end to end; the
numbers quoted are measured, not estimated.

**Machine:** GTX 1660 SUPER, 6 GB VRAM, Turing sm_75 but TU116 — **no tensor
cores**. 16 GB RAM. Windows 11.

---

## 0. Prerequisites

| thing | path |
|---|---|
| trellis-cli | `C:\AI\trellis-cpp\build-mmq\Release\trellis-cli.exe` |
| GGUF weights (7.21 GB) | `C:\AI\trellis-cpp\models` |
| Python for workers | `C:\AI\HY3D2\python_standalone\python.exe` |
| Repo workers | `workers/` in this tree |

Some workers need the repo on the path: prefix with `PYTHONPATH=src`.

### Rebuilding trellis.cpp (only if the binary is lost)

The build flags are not optional on this card. ggml selects tensor-core kernel
paths from the reported compute capability (7.5), and TU116 reports 7.5 while
lacking the silicon, so a default build produces `illegal memory access` a few
diffusion steps in. Compiling for **virtual Pascal** forces the DP4A path
instead.

```bash
cmake -B build-mmq -G "Visual Studio 17 2022" -A x64 \
  -T cuda="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" \
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  "-DCMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual" \
  -DGGML_CUDA_FORCE_MMQ=ON -DTRELLIS_WEBP=OFF
cmake --build build-mmq --config Release --parallel 4
```

**CUDA 12.6 specifically.** CMake will auto-select 13.2 if allowed, and CUDA 13
dropped Pascal: `nvcc fatal: Unsupported gpu architecture 'compute_61'`.

`CMakeLists.txt:152` also hardcodes `CUDA_ARCHITECTURES "86;120"` upstream. It is
patched here to honour the configure-time value; re-apply after any pull.

---

## 1. Matte the source image

The generator needs a background-free subject, and the input must carry alpha or
stage 2 will fall back to a keyer that cuts specular highlights into holes.

```bash
py workers/pipeline_matte.py --image source.png --out crop512.png
```

Verify alpha exists and covers roughly half the frame. An opaque image here is
the single most common cause of a bad generation.

---

## 2. Geometry + texture in one pass

```bash
cd C:\AI\trellis-cpp\build-mmq\Release
./trellis-cli.exe crop512.png out.glb \
  --models C:\AI\trellis-cpp\models \
  --res 512 --seed 12345 --atlas 1024 --no-fa --require-gpu
```

**Measured: 271 s.** Produces 145,770 faces, one 1024 UV chart, PBR base
colour/metallic/roughness/alpha. Internally: 12-step sparse-structure flow →
12-step shape SLAT flow → FlexiDualGrid decode → 3.03M faces → its own CuMesh
QEM decimation to ~147k → UV bake.

- `--no-fa` — required. FlashAttention is a tensor-core path; without this the
  run dies at diffusion step 1.
- `--require-gpu` — refuses a silent CPU fallback, which would otherwise look
  like success.
- `--decim 0` — add this for **geometry evaluation only**; it keeps the full
  2.8M-face mesh. Do not use it for a production asset.
- **`--res 1024` does not work on this GPU.** Three attempts, all
  `MUL_MAT: misaligned address` at the shape SLAT flow, including with `--f32`.
  The sparse-structure stage is nondeterministic, so each run hands the shape
  flow differently-shaped tensors and some shapes hit a kernel bug. Not a config
  error. Same for `--atlas 2048`, which fails in the same stage.

Geometry-only (`--no-texture`) takes 249 s and is useful for A/B on shape alone.

---

## 3. Composite reference artwork into the atlas

Native texturing covers everything but regresses to a muted grey-brown — the
documented weakness of native 3D texturing on flat geometry. If the concept
sheet has real orthographic elevations, paint them over the generated atlas.

```bash
R=evidence/compare/boat/refsheet/cond
py workers/hybrid_atlas_composite.py \
  --mesh out.glb --out hybrid.glb \
  --view front=$R/front.png --view right=$R/side.png \
  --view back=$R/back.png  --view left=$R/side_mirror.png \
  --contrast 0.12 --shadows 0.03 --highlights 0.10 \
  --saturation 1.15 --luma 0.55 \
  --receipt hybrid.json
```

**Those grade values are calibrated, not defaults-by-accident.** Two earlier
settings were wrong in opposite directions and both are recorded in the git
history: chroma-only transfer at `luma 0.35` kept the generated luminance, which
is itself the defect; then full transfer with `contrast 0.28 shadows 0.15`
crushed the side views to black. `luma 0.55` with a gentle S-curve holds all
seven views.

De-lighting is on by default and is what makes any luminance transfer safe: the
elevations are paintings carrying their own shadow, and projecting that raw onto
already-dark texture double-darkens. `--no-delight` disables it, and then
`--luma` must come back down.

Coverage on the boat: **65.4%** of surface texels take real art. The rest — roof,
underside, occluded interiors — keep the generated texture, which is why the
asset has no unpainted regions.

---

## 4. Add a decal for authored detail

Lettering will not survive a 1024 atlas spread across a whole vehicle. Give it
its own texture.

```bash
py workers/apply_decal.py \
  --mesh hybrid.glb --decal marquee.png --out final.glb \
  --roi 0.30,0.10,0.70,0.28 --decal-out marquee_keyed.png
```

`--roi` is `x0,y0,x1,y1` in the **front view's normalised image coordinates**,
y down. The worker selects front-facing triangles inside that rectangle, fits a
PCA plane for the UV basis, and offsets the patch fractionally along its normal.

It reuses the mesh's **own triangles** rather than adding a quad, so the decal
follows the curvature of the facade and cannot float or clip as the camera
moves. Background is keyed by connected-component analysis from the image
border, so highlights inside the sign survive while the surround is cut.

---

## 5. Preview

```bash
py workers/preview_textured_mesh.py --glb final.glb --out sheet.png
```

**Use this, not `preview_coloured_mesh.py`, for anything with a UV atlas.** The
vertex-colour preview samples one colour per vertex; on a 145k-face mesh with a
1024 atlas that discards most of the texture and makes good assets look muddy.
Several review verdicts in this project were formed on those degraded images
before the mistake was found. A 956×476 decal on 267 triangles is *invisible* at
vertex resolution.

For geometry-only meshes, `preview_generated_mesh.py` (normal-shaded) is still
the right tool.

---

## 6. Optional: LOD

```bash
py workers/lod_per_shell.py --input mesh.glb --out lod.glb --target 235000
```

**Never run a global quadric decimator on this geometry.** TRELLIS output is
non-watertight with thousands of open shells; a global pass returns torn plates
and holes, and collapses the body count (measured: 7,144 → 916). Two reasons,
both handled here: boundary vertices on an open shell have no opposing plane, so
the cheapest collapse folds the shell flat — hence `BOUNDARY_WEIGHT 12`; and a
global error budget spends itself on ornament while leaving flat areas dense —
hence per-shell budgets proportional to area.

---

## What does not work — do not retry

| attempt | outcome |
|---|---|
| `--res 1024`, `--atlas 2048` | `MUL_MAT: misaligned address`, 3 attempts, incl. `--f32` |
| Global quadric decimation | destroys the mesh |
| `workers/denoise_protected.py` | no-op: 49% of the mesh is shells under 400 faces, so protection freezes everything worth freezing and there is nothing left to smooth. The granular look **is** those shells |
| Hunyuan3D levers | steps, octree, views, capacity, triangle count, MC vs DMC — seven measured dead ends |
| Deck plan as roof texture | it is an interior floor plan |
| MV-Adapter views for TRELLIS geometry | generated from Hunyuan geometry; they encode a different shape |

---

## Timings, measured

| stage | time |
|---|---|
| geometry only | 249 s |
| geometry + PBR texture | 271 s |
| hybrid atlas composite | ~3 min (CPU) |
| decal | seconds |
| textured 7-view preview | ~8 min (CPU, pure NumPy) |

---

## 7. Complexity preflight — check before spending four minutes

`workers/trellis_run.py` wraps `trellis-cli`, watches for the structured-latent
size it prints at stage 4 of 7, and aborts before the decode if the subject is
too complex for this machine.

```bash
py workers/trellis_run.py --image matte.png --out asset.glb --receipt run.json
```

Measured, four subjects on 16 GB RAM:

| subject | latent | result |
|---|---:|---|
| castle | 82,304 | decoded |
| boat | 86,784 | decoded |
| red panda (ghillie suit) | 140,480 | **died in FlexiDualGrid decode** |
| blue tree (foliage) | 191,744 | **died in FlexiDualGrid decode** |

The latent tracks how much of the volume the subject actually occupies. Rigid
architecture activates few voxels; fur, webbing and foliage activate many. The
failures are host RAM during decode, not VRAM.

Why the wrapper is worth having: five attempts at the panda produced **three
different errors** — `CUDA error: out of memory`, a ggml host-allocation assert,
and `misaligned address` — for one underlying cause. The preflight turns that
into one diagnosis at ~140 s instead of a crash at ~260 s.

The band is a warning, not a law: four samples establish a correlation and the
gap between 87k and 140k is unsampled. `--max-latent 0` disables the abort and
records the number, which is how the boundary gets refined.

## 8. Grading a subject with no reference artwork

`hybrid_atlas_composite.py` works with **no `--view` arguments**: it then grades
the generated atlas without compositing anything. This is the normal case for a
single-image subject.

```bash
py workers/hybrid_atlas_composite.py --mesh asset.glb --out graded.glb \
  --contrast 0.18 --shadows 0.03 --highlights 0.05 \
  --saturation 1.12 --warmth 0.04
```

Those values are calibrated on the castle against its source image. Native
texturing on this stack returns **cool and grey-blue** on stone and timber
subjects. The instinct to fix that with saturation is wrong — it amplifies the
existing hue instead of moving it. `--warmth` is a luminance-preserving white
balance shift and is the correct control; a three-way comparison
(`evidence/compare/castle/castle_grade_3way.png`) shows it beating a
higher-saturation, higher-highlight grade clearly.

Subjects **with** reference elevations use the same tool with `--view` and a
lower contrast, because the projected artwork brings its own contrast:
`--contrast 0.10 --shadows 0.00 --highlights 0.18 --saturation 1.20`.

## 9. Subject classes — what to expect

Two subjects have been through the full pipeline and rated **mid-ground**: the
boat (with four reference elevations) and the castle (single image, no artwork).
Both are rigid architecture.

Predicted, not yet measured:

- **Vehicles / hard-surface**: best candidates, potentially better than
  mid-ground if not mechanically intricate.
- **Humanoids**: likely decode successfully if clothed or armoured; faces,
  hands and thin appendages weak.
- **Rock formations**: succeed at mid-ground range, generic erosion detail.
- **Furry creatures**: high failure risk on both capacity and quality.
- **Trees with real foliage**: likely to exceed the latent budget outright.
