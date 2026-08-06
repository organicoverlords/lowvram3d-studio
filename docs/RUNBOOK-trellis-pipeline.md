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
py workers/pipeline_matte.py --image source.png --output matte.png
```

Verify alpha exists. An opaque image here is the single most common cause of a
bad generation. Coverage of roughly half the frame is ideal, but a tall or wide
subject padded to square will legitimately sit near 30% — the boat and both
castles land between 29% and 47%.

**Contact shadows must be removed, and the option for it is off by default.**
A studio render of a character usually sits on a soft ground shadow. Left in,
that shadow is opaque to the generator and becomes a floor slab fused to the
feet.

```bash
py workers/pipeline_matte.py --image source.png --output matte.png \
  --shadow-tolerance 180 --shadow-from 0.78
```

`--shadow-from` is the height fraction below which the stronger tolerance
applies. **Sweep it and look at the crops.** On the shaman, 130 left the
shadow, 180 removed it cleanly with feet and claws intact, and 240 destroyed
the feet.

And do not trust a coverage number to pick the setting. "Alpha remaining in the
bottom 14% of frame" reads:

| tolerance | bottom-14% alpha | actual result |
|---:|---:|---|
| 130 | 10.82% | shadow still present |
| **180** | **7.69%** | **correct — shadow gone, subject intact** |
| 240 | 1.12% | feet destroyed |
| 300 | 0.00% | everything below the hem deleted |

The metric ranks total destruction as the best outcome, because the feet are
*in* the region it measures. Render the crops over a magenta background and
judge by eye. A number that scores catastrophe above correctness is worse than
no number at all.

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
| `workers/denoise_protected.py` | no-op, twice measured. 79.9% of vertices frozen, mean displacement 5e-06, and **zero** vertices reach the displacement cap — the filter is not restrained, it has nothing to say. Dihedral is bimodal (p50 4.5°, p99 177°) and the high tail is *not* degenerate: those faces have 1.02x median area and quality 0.60. They are thin double-sided sheets, 19.6% of surface area, and they are the spokes and railings. Do not smooth them |
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

## 7. When a run fails — retry before you diagnose

**Runs on this card fail stochastically. A single failure means almost
nothing.** Use the seed-retry script for anything large, and read a failure as
"try again" before reading it as a diagnosis:

```bash
bash workers/trellis_retry_seeds.sh matte.png out.glb 12345 777 4242 31337
```

It stops at the first seed that completes and writes it to `winning_seed.txt`.
Seed is not a quality lever — there is no reason to prefer one draw's geometry
over another's — so taking the first survivor costs nothing.

Worked example, the 4K gothic castle. **Nine attempts, one asset.** Five of them
were me changing parameters, and none of those parameters was the answer:

| attempt | what | outcome |
|---:|---|---|
| 1-5 | flag changes (`--tex-res`, `--f32`), disk cleanup | five *different* failures, no asset |
| 6 | seed 12345 | `misaligned address`, stage 4 |
| 7 | seed 777 | aborted by the latent threshold at 134,944 |
| 8 | seed 20260806 | silent kill, stage 5 |
| **9** | **seed 4242** | **success: 3,949,828 faces raw, 144,002 after decimation, 342 s** |

`workers/trellis_run.py` still reports the structured-latent size at stage 4/7,
which is worth having in the receipt. It **no longer aborts by default**, because
the threshold did not survive a bigger sample — see below.

### The latent band was wrong, and here is the sample that killed it

Every latent observed, with what happened:

| latent | subject | result |
|---:|---|---|
| 82,304 | castle 1 | decoded |
| 86,784 | boat | decoded |
| 94,528 | castle 2 | died stage 5 |
| 96,448 | castle 2 | died stage 5 |
| 106,496 | castle 2 | geometry decoded, died stage 6 |
| **111,520** | castle 2 | **full success, 3,949,828 faces** |
| 134,944 | castle 2 | aborted by the old threshold — never tested |
| 140,480 | red panda | died stage 5 |
| 191,744 | blue tree | died stage 5 |

**96,448 fails and 111,520 succeeds**, so no threshold in that range separates
them. The original four-sample band was four single draws from four images, read
as four subject properties. One image alone draws **94,528 to 134,944** — a 43%
spread — because the sparse-structure stage is nondeterministic. The latent is
not a stable property of a subject, let alone a capacity limit.

What survives: nothing at 140k+ has ever decoded, so a very large value is still
a real warning. `--max-latent` now defaults to **off**; pass it explicitly for
unattended batch work, where burning 260 s on a doomed run costs more than
skipping one that might have worked.

### The nondeterminism also causes the crashes

This dominates everything else on this card. **Different draws hand the downstream stages differently
shaped tensors, and TU116 has shape-dependent kernel faults.** The same image,
same flags, failed in five different places on five consecutive attempts:

| # | flags | died at | error |
|---:|---|---|---|
| 1 | plain | stage 6 | `cudaMalloc` OOM — but stage 5 **succeeded**: 1,514,675 voxels, 3,467,018 faces |
| 2 | `--tex-res 512` | stage 5 | `cudaMalloc` OOM |
| 3 | `--tex-res 512` | stage 3 | kernel launch, out of memory |
| 4 | `--tex-res 512` | stage 3 | `cublasGemmStridedBatchedEx(... CUBLAS_GEMM_DEFAULT_TENSOR_OP)` failed to launch |
| 5 | `--f32` | stage 4 | `misaligned address` |

Attempt 4 is the informative one: `CUBLAS_GEMM_DEFAULT_TENSOR_OP` is an explicit
request for a tensor-core kernel on a card that has none. `GGML_CUDA_FORCE_MMQ`
and `--no-fa` redirect the MMQ and FlashAttention paths; **neither covers
`mul_mat_batched_cublas`**, which reaches cuBLAS directly. `--f32` does avoid it
— stage 3 went from failing at 28.9 s to completing in 83.2 s — at roughly 3x
the time, and it then exposes the `misaligned address` fault one stage later.

So: **do not read a single failure as a verdict on the subject.** Attempt 1
proves this castle's geometry is within reach of this machine. Vary the seed and
try again before touching a parameter.

```bash
bash workers/trellis_retry_seeds.sh matte.png out.glb 12345 777 4242 31337
```

Stops at the first seed that completes and writes it to `winning_seed.txt`, so
the result stays reproducible. Seed is not a quality lever here — there is no
reason to prefer one draw's geometry over another's — so taking the first
survivor costs nothing.

Levers, in the order worth trying:

1. **Another seed.** Each attempt is an independent draw. Cheapest and most
   often sufficient for a large subject.
2. **Close other GPU consumers** if `nvidia-smi --query-gpu=memory.free` is low,
   and check commit headroom (`Win32_OperatingSystem.FreeVirtualMemory`) — CUDA
   pins host memory, so an exhausted commit limit surfaces as "out of memory"
   with VRAM sitting idle.
3. `--f32` if failures cluster at **stage 3** specifically. 3x slower.
4. `--tex-res 512` only for a **stage 6** failure. It cannot affect any earlier
   stage, and pretending otherwise wasted three attempts here.
5. `--no-texture` to bank geometry once stage 5 is clearing reliably.
6. Simplify or crop the subject. Last — it changes the deliverable.

`trellis_run.py` records which stage died, the MiB requested, and whether
geometry decoded, then exits **3** on a VRAM fault. A stage-6 failure means the
mesh existed and only the texture was lost.

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

## 9. Measuring geometry quality instead of eyeballing it

```bash
py workers/feature_edge_f1.py --teacher reference.glb --candidate asset.glb \
  --out score.json --dump-edges edges/
```

Renders both meshes to depth and normal buffers across four shared cameras,
extracts internal feature edges (occlusion steps plus creases, **silhouette
excluded**), and reports precision/recall/F1 after registering the candidate by
yaw. Read `trustworthy` before reading `mean_f1` — it is false when fewer than
three views align or when the candidate's edge density exceeds 1.6x the
teacher's, and a false there means the number is meaningless, not bad.

Measured on the boat against the online reference:

| candidate | F1 | P | R | density | trustworthy |
|---|---:|---:|---:|---:|:--:|
| teacher vs itself (self-test) | 1.000 | 1.000 | 1.000 | 1.000 | yes |
| TRELLIS.2 512 (debris removed) | **0.690** | 0.678 | 0.713 | 1.263 | yes |
| Hunyuan3D mini, octree 320 | 0.573 | 0.611 | 0.551 | 0.671 | yes |
| a different object entirely | 0.333 | 0.526 | 0.252 | 0.405 | **no** |

Two things to take from the table. **Use `--dump-edges` and look at the
overlays** — teacher red, candidate green, yellow agreement — because the
per-view numbers say *that* a view failed and the overlay says *why*. And treat
the top view separately: both generators score ~0.43 there and both exceed the
teacher's downward edge density, because they build roofs as assemblies of thin
plates whose rims each emit an edge, while the reference models one solid panel.
That is a representational difference, not a defect, and it drags the mean down
by about 0.09 on every candidate equally.

The candidate must be under roughly 1M faces. A 2.81M-face mesh exhausts 16 GB
during the render; run the debris-removed or LOD version and say so in the label.

## 10. Subject classes — what to expect

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
