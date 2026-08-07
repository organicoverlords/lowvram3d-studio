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

#### Do not use build-cublas as the production binary

There is a second build, `build-cublas`, identical except for
`-DGGML_CUDA_FORCE_MMQ=OFF -DGGML_CUDA_FORCE_CUBLAS=ON`. It exists to test
whether the residual `misaligned address` fault -- which lands in MUL_MAT,
~50% of attempts, on shapes that vary per subject -- goes away when ggml's
quantized kernels are never selected.

It may well fix the fault. Nobody has measured that yet, because it is too slow
to use. On the sparse-structure flow, which runs on a fixed 32^3 latent grid and
therefore costs the same whatever the subject:

| binary | s/step | subject |
|---|---|---|
| build-mmq | 7.3 - 7.6 | snail, boat, frog |
| build-cublas | 36.2 | frog, steady over 4 steps |

**4.8x slower.** That is inherent to how the flag works: with the quantized
kernels off, weights are dequantized to F16 and multiplied through cuBLAS GEMM,
on a card with no tensor cores. Projected to a full res-1024 run that is ~90 min
against ~21 min. A fault that costs half of all attempts is ~42 min expected, so
the always-slow binary is the worse trade even if it never faults once.

Keep retry-on-fault on `build-mmq`. `TRELLIS_CLI` overrides the binary if you
want to measure the cuBLAS fault rate properly.

#### `--f32` is unmeasured, not exonerated

`--f32` ("f32 sparse-conv compute") was tried against the fault and the frog then
went 0 for 3. That number proves nothing, because two variables moved at once:
every `--f32` run was on the frog at 18,559 HR tokens, and every run without it
was on a smaller subject. The frog is the largest res-1024 subject attempted, and
size already correlates with failure in the reference-latent table the receipts
carry. So "0 for 3" is equally consistent with `--f32` being neutral, with the
frog being too big, and with `--f32` making things actively worse.

A fault-rate claim needs one variable at a time: same subject, same seed, N
attempts with and N without. Until that exists, do not describe `--f32` as
"doesn't help" -- describe it as untested. Receipts record the full argv, so any
run that uses it is already contributing evidence.

Note also that the fault does not always announce itself the same way. On the
frog it appeared as `misaligned address` twice and as `the function failed to
launch on the GPU` once, 18 minutes in. `nvlddmkm` logged an event at each
failure timestamp and there were no Event 4101 entries, so the launch failure was
not a watchdog timeout -- same fault, different surface.

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
- **`--res 1024` works, with retries.** This entry used to say it did not. Four
  subjects cleared it in one session, each within two attempts:

    | subject | HR tokens | attempts | s/step (HR flow) |
    |---|---|---|---|
    | snail | 13,615 | 2 | 80.2 |
    | whale | -- | 2 | -- |
    | heron | -- | 2 | -- |
    | shaman | -- | 2 | -- |
    | frog | 18,559 | **0 for 4**, cleared res-512 first try | 157.0 |

  So it is a size ceiling, not a hard block. Cost scales superlinearly in HR
  tokens, which track *occupied volume*, not source resolution: the frog is 36%
  more tokens than the snail for 96% more time per step, and dies. Use `--res
  1024 --tex-res 512` with a retry loop, and fall back to 512 for the biggest
  subjects.

  Conditioning is resolution-invariant, so a bigger source image buys nothing
  here: a 659x484 input and a 4322x7680 input both produce exactly
  `cond tokens=1029 / 1024-cond tokens=4101`, because preprocess crops to the
  alpha box and DINOv3 samples at a fixed size.

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

## 9b. Hunyuan3D-Paint: keep the conditioning image at ~512

**Feed paint a ~512 conditioning image. Not the 4K or 8K matte.**

This cost four failed paint runs to find. The shaman failed on
`shaman4k_matte.png` (3072x3840) four times across three different meshes:

    TRELLIS res-512  138k faces   render 1024   illegal memory access
    TRELLIS res-1024 290k faces   render 1024   illegal memory access
    TRELLIS res-1024 290k faces   render 1024   illegal memory access
    Mini Turbo       400k faces   render 1024   illegal memory access
    Mini Turbo       400k faces   render  768   CUBLAS_STATUS_EXECUTION_FAILED

Swapping in `shaman_512.png` and changing nothing else -- same meshes, same
`--texture-size 2048`, same render size, same quiet GPU -- succeeded on the
first try, twice:

    TRELLIS res-1024 290k faces   620.6s
    Mini Turbo       400k faces   446.6s

The mechanism is in the second error, which names it. The multiview pipeline
loads as `torch_dtype=torch.float16` (`hy3dgen/texgen/utils/multiview_utils.py`
line 36), so every GEMM in it is `CUDA_R_16F`, and cuBLAS picks an algorithm per
shape. Some shapes get `CUBLAS_GEMM_DEFAULT_TENSOR_OP` -- a tensor-core path.
TU116 reports compute capability 7.5 but has no tensor cores, which is the same
silicon gap that forces `--no-fa` and the virtual-Pascal build for trellis.cpp.

So it is shape-dependent, not size-dependent. The conditioning image sets those
shapes. The whale and heron both painted fine on large mattes; the shaman did
not. That is why it looked random for hours: retrying re-rolls nothing, and
changing `--render-size` only perturbs the shapes and hopes to miss the bad
path.

Large mattes remain correct for TRELLIS geometry -- it is only the paint stage
that cares. Build the paint input by cropping the matte to its alpha box,
squaring it, and resizing to 512.

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

## 11. Texturing a geometry-only generator — the six-view route

TRELLIS bakes its own atlas, so §2 covers it. **Hunyuan3D Mini Turbo emits
geometry only**, and the whole texture path below exists for that case. It is
also the better route for any subject the generator router (§`choose_generator`)
sends to Hunyuan, i.e. anything built of cords and hanging props.

Measured on the bird-skull shaman, which is the asset this route was proven on.

### Why a single projection is not enough

`fast_texture_projection` paints from one camera. That is one side:

| | texels | share |
|---|---:|---:|
| observed from the source photo | 336,737 | **16.5%** |
| dilated invention | 1,707,626 | 83.5% |

The 83.5% reads as camouflage patchwork on every view except the photographed
one, and **no grade or reorientation fixes it** — one photograph carries one
side. The six-view route takes the same asset to **53.4% observed**, a quarter
of it seen by two or more views.

### The route

```bash
# 1. decimate — Hunyuan returns ~1.1M faces, watertight.
#    workers/decimate_mesh.py is the sanctioned worker and drives Blender
#    headless. The shaman and panda were done with open3d inline instead,
#    which is far faster. open3d lives ONLY in the standalone runtime -- it is
#    absent from both the 3.12 and 3.11 system interpreters, so this line must
#    use the same python.exe as the GPU stages:
C:/AI/HY3D2/python_standalone/python.exe -c "import trimesh,open3d as o3d,numpy as np; \
m=trimesh.load('shaman_miniturbo.glb',process=False).to_geometry(); \
d=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), \
o3d.utility.Vector3iVector(m.faces)).simplify_quadric_decimation(400000); \
trimesh.Trimesh(np.asarray(d.vertices),np.asarray(d.triangles)).export('lod.glb')"
#    1,112,498 -> 400,000 in 12.6 s, still watertight, 21 shells

# 2. unwrap — this is why shell count matters. xatlas has never completed on
#    a TRELLIS mesh here (71,043 shells); on 21 shells it takes minutes.
py workers/unwrap_mesh_uv.py --input lod.glb --output uv.glb \
   --report uv.json --resolution 2048 --padding 4
#    4,611 charts, overlap 0.0, degenerate 0.019%, PROVEN

# 3. orient — see "Two axis traps" below
# 4. controls
py workers/build_mvadapter_cpu_controls.py --mesh uv_zup.glb \
   --output-dir controls_384 --size 384

# 5. audit + relabel
py workers/audit_and_relabel_mvadapter_controls.py --mesh uv_zup.glb \
   --source-image crop512.png --source-dir controls_384 \
   --output-dir controls_384_audited

# 6. six-view inference — NOTE the interpreter and --worker, see below
PYTHONPATH="C:/AI/HY3D2/Hunyuan3D-2;C:/AI/mvadapter-upstream-inspection" \
C:/AI/HY3D2/python_standalone/python.exe workers/run_sixview_no_cudnn.py \
   --config configs/<asset>_mvadapter_ig2mv_sd21_512x20.json \
   --output-dir evidence/compare/<asset>_sixview_512 \
   --worker "$PWD/workers/mvadapter_sd21_six_view_inference.py"
#    20/20 steps, 302 s at 384; 512 costs more but fits and is better

# 7. project all six
py workers/multiview_texture_projection.py --mesh uv_zup.glb \
   --bundle controls_384_audited --views-receipt <views>.json \
   --output-dir mv --output-glb textured.glb --report mv.json --atlas-size 2048
```

### Substitute the photograph for the generated front

The front is the one direction where real data exists, and it is also where
MV-Adapter is weakest. On the shaman the generated front failed **both** QA
gates that the run failed overall:

| front view | structural IoU | foreground saturation |
|---|---:|---:|
| MV-Adapter generated | 0.677 | 0.070 |
| the photograph, bbox-registered | **0.689** | **0.413** |

Six times the colour, and a better silhouette match. Register the matte into the
front control's bbox, write it over `view_0_front.png` in a copy of the
inference receipt, and re-run step 7 against that copy. The result is visibly
warmer and carries real photographic detail in the face and robes.

### Two axis traps, both silent

**`build_mvadapter_cpu_controls.py` hardcodes `world_up = [0,0,1]`** while glTF
is Y-up. The first build put the shaman **lying on its side** in all four
horizontal views, with the standing views in the top/bottom slots. Nothing
failed; `passed: true`. Pre-rotate +90° about X into Z-up, and **always look at
the normal contact sheet** before spending a GPU sequence:

```bash
py -c "import numpy as np;from PIL import Image;\
n=['front','right','rear','left','top','bottom'];\
Image.fromarray(np.concatenate([np.asarray(Image.open(f'DIR/{x}_normal.png').convert('RGB')) for x in n],1)).save('DIR/sheet.png')"
```

**A generator has no notion of front.** Mini Turbo returned this shaman facing
152° off +Z. Silhouette IoU cannot settle it — front and back silhouettes are
near mirrors, scoring 0.689 vs 0.560. `fast_texture_projection` now reports
`canonical_orientation.rotate_about_y_degrees`, derived from the camera that
painted the atlas: the painted hemisphere is centred on `-matrix[2]`, which is
a fact about where paint landed rather than a guess about shape.

### Three more traps in the six-view stage

**The launcher hardcodes its worker path.** `run_sixview_no_cudnn.py` defaults to
the worker inside `lowvram3d-two-character-production-20260804`, which is a
different repo owned by a different agent. An edit made to this repo's
`workers/mvadapter_sd21_six_view_inference.py` — for example admitting 512 to the
resolution gate — has no effect at all unless `--worker` points at it. The
symptom is a preflight rejection that names a value you have already changed.

**The labels can be rotated, per asset.** The builder emits
`front, right, rear, left`, and on the panda the true geometric opposite of
index 0 is **index 3**, not index 2. On the whale and the shaman it is index 2,
as labelled. So this is not a fixed off-by-one to hardcode — read the camera
directions from `camera_contract.json` every time. Any check that pairs views by
label order may be comparing things 90° apart and will pass on anything.

**512 is admitted and is better than 384.** It fits in 6 GB. The gate lives in
`ALLOWED_RESOLUTIONS` in the worker — and see the `--worker` trap above, because
that is exactly the edit that silently does nothing.

### Which way is front — use paint, not silhouette

Two heuristics were tried and both chose wrong:

| method | failed on | why |
|---|---|---|
| tail-colour | shaman (6448 vs 6559), and the red panda itself | a near-tie decided by noise |
| silhouette IoU | whale (0.881 vs 0.618, chose the mirror profile) | front and back silhouettes of a symmetric subject are near-identical |

The strongest signal available *among those three* is **painted-texel
coverage**: sample the observed mask through each candidate view's stored
`triangle_ids`/`barycentric` and count texels that received paint. On the whale
the two candidates scored **5 texels vs 25,107**.
`audit_and_relabel_mvadapter_controls.py --observed-mask` does this and records
`basis`, `painted_texel_coverage` and `tail_rule_would_have_chosen`.

**It is not a front-axis detector, and on the panda it confirmed a 180° error
instead of catching it.** Painted-texel coverage answers "which camera did the
projection paint from", which is a fact about the projection. If the projection
solved the front 180° wrong, the paint is on the back and this audit agrees with
it — confidently, at 0.8364 against 0.0016. Circular by construction.

The check that actually separates them compares **where the photograph landed
against where the mesh's detail is**: render the photo-textured mesh through a
candidate camera's own `triangle_ids` and put it beside that camera's normal
render. If the photographic face sits on featureless geometry, the axis is
wrong. `evidence/compare/panda2/photo_vs_geometry.png` is that comparison for
both hemispheres, and it is the only artefact in this route that has ever caught
this class of error.

### The interpreter

GPU work needs **`C:\AI\HY3D2\python_standalone\python.exe`** with
`PYTHONPATH=C:/AI/HY3D2/Hunyuan3D-2`. The system Python 3.12 carries a phantom
`torch` whose `torch.backends` attribute does not exist, and Python 3.11 has no
scipy. Symptom of the wrong one: `AttributeError: module 'torch' has no
attribute 'backends'`, or `No module named 'hy3dgen'`.

### Config preflight

The inference worker only accepts `status: PREPARED_NOT_EXECUTED` and refuses
if `gpu_sequence_consumed` is truthy. Copying a *finished* config from another
asset carries both fields in the wrong state. The gate catches it without
spending the sequence, but it costs a round trip.

### Reviewing the result

```bash
py workers/turntable.py --glb textured.glb --out evidence/turntables/asset.webp
py workers/turntable.py --index evidence/turntables --out evidence/turntables/index.html
```

A rotating view exposes what a contact sheet hides: seams where two projected
views disagree, and surfaces that change character as they turn. WebP, not GIF —
256-colour quantisation posterises weathered cloth into flat plates that look
exactly like the projection artefact you are checking for.

**The turntable is retired.** At useful quality it took 30 minutes for one
asset, which is not worth it for a check a seven-view contact sheet mostly
answers. `workers/turntable.py` still works if you want it; nothing in the route
depends on it. Use `preview_textured_mesh.py` instead.

### What this route achieves, and its one open defect

| asset | resolution | directly observed | seen by ≥2 views |
|---|---:|---:|---:|
| sky whale courier | 512 | 69.59% | 36.27% |
| red panda | 512 | 66.43% | 29.82% |
| boat | 384 | 64.94% | 19.50% |
| bird-skull shaman | 512 | 53.85% | 26.18% |

Against 16.5% for a single projection.

**The panda's front axis is 180° wrong**: its photograph was projected onto the
back of its head, so its real sculpted face received invented camouflage. That
is a projection bug, not an MV-Adapter bug — the generator painted a face where
the geometry has one. The whale's axis is correct. The shaman and boat have not
been checked with `photo_vs_geometry`. Do that before trusting any of these
numbers as texture quality: coverage counts texels painted, not texels painted
*correctly*. Read the retraction at the top of
`docs/JANUS-six-view-defect-20260806.md` before running another sequence.
