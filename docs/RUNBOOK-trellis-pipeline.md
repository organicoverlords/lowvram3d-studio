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

## 9a. Hunyuan3D-Paint: never let it re-unwrap UVs it already has

**If the mesh carries TEXCOORD_0, skip `mesh_uv_wrap`. This is the difference
between a paint that finishes in six minutes and one that never finishes.**

`Hunyuan3DPaintPipeline.__call__` calls `mesh_uv_wrap` unconditionally, and that
is `xatlas.parametrize` over every face -- single threaded, and printing
nothing. It sits between "pipeline loaded" and the first progress bar, so while
it runs the console is blank, the GPU is idle, and three CPU cores are pinned.

On the moss titan it never completed. Three runs died in it at 25, 25 and 7
minutes. The geometry is the worst case xatlas can be handed: 286,994 faces
whose surface is thousands of thin hanging strands, each a separate chart to
pack.

It is also wasted work. TRELLIS exports `TEXCOORD_0` with one UV per vertex and
a 2048 atlas. The bake writes a new texture into whatever layout the mesh
already carries; it does not require xatlas's particular packing. Skipping the
unwrap when valid UVs exist took the same asset to **365.1 s of paint --
faster than all 21 previous paints on this machine**, on the largest mesh in
the set.

`workers/hunyuan_paint_texture.py` patches `mesh_uv_wrap` at the module level
for the duration of the run. Meshes that genuinely lack UVs still get
unwrapped, and the skip prints itself either way, so nothing silently paints
without a parameterisation.

### The diagnostic lesson, which cost more than the bug

Three wrong answers came first, each derived from watching resource *shapes*
rather than asking the process what it was doing:

| theory | why it was wrong |
|---|---|
| the mesh is too big | the heron is 295,108 faces and painted in 881 s |
| conditioning is too large | both inputs were already 512, byte-identical |
| the UNet is running on CPU | it was not -- that GPU burst was the delight model |

One command settled it:

    py-spy dump --pid <pid>

which printed `mesh_uv_wrap (hy3dgen/texgen/utils/uv_warp_utils.py:26)` in about
a second, without touching the running process. **Reach for the stack before
reaching for a theory.** Flat VRAM at 100% utilisation is a healthy sampler;
flat VRAM at ~11% with cores pinned means the work is not on the card -- but
neither tells you *which function*, and that is the only thing worth knowing.

Two supporting fixes are in the tooling now. `trellis_run.py` opens its `--log`
before the child starts and flushes per line, instead of writing it after
`process.wait()` -- a 40-minute stage used to produce an empty path for exactly
as long as it mattered. `hunyuan_paint_texture.py` wraps the vendor's named
stages (`render_normal_multiview`, `bake_from_multiview`, `texture_inpaint`) so
the phases the vendor does not announce announce themselves.

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

## 10a. TRELLIS can return a billboard and call it success

On 2026-08-08 the greentree was generated at `--res 512`. It finished in 264 s
and wrote a receipt with `success: true`, `geometry_decoded: true`,
`geometry_finalized: true`, `exit_code: 0`, 806,990 decoded voxels, a
1024 atlas, and a finalizer note listing weld, hole fill, narrow-band remesh,
winding repair, component filter and QEM. The log showed every stage running.

The asset was two crossed flat panels.

Nothing in the receipt is false. Every field it records was true of the mesh it
made. The failure is not detectable from any of them, and it is not detectable
from the face count either — 146,326 faces, because a subdivided plane has as
many triangles as you like.

**The three numbers that do detect it** (`tools/check_not_billboard.py`):

| measure | what it is | billboard | solid |
|---|---|---|---|
| `fill` | volume / bounding-box volume | 0.010 | 0.025 – 0.171 |
| `quad_ratio` | area / area of the crossed quads the box allows | **0.998** | 1.75 – 2.85 |
| `worst_spread` | vertex fraction in the two busiest bins of 20, per axis | **0.554** | 0.178 – 0.231 |

`quad_ratio` is the sharp one, and its direction is counter-intuitive enough
that the first version of the gate had it backwards. A billboard does not have
*too much* area; it has *exactly* the area of the crossed quads, so it sits at
1.0 and every solid sits **above** it. Written as a ceiling of 0.75 the test
fired on every real asset in the project and falsely aborted a good 20-minute
seal-diver run. It is a lower bound: flag `quad_ratio < 1.30`.

`fill` cannot carry the test alone. The seal diver is a genuine solid at 0.025
because flippers, ropes, an anchor and a swinging lantern inflate the bounding
box far beyond the body inside it. The gate requires **two of three** symptoms.

**Cause, and the retry that is worth running.** The sparse-structure stage runs
on a 32³ grid at `--res 512`, so one structural cell is 16 source pixels. A
banyan's aerial roots are thinner than that and lose their cells to the canopy
above them, and the subject collapses onto the two planes carrying most of its
silhouette. At `--res 1024` the grid is 64³. Every asset in this project that
came out solid was generated at 1024; the tree was the only one that was not.

So the gate belongs **between geometry and paint**, not at the end — the paint
is the forty-minute stage, and painting cardboard is the whole cost of the bug.

Corollary for §10 above: "trees with real foliage" does not fail by exceeding
the latent budget, which is what was predicted. The greentree's latent was
102,400 — squarely in the *typical* band, lower than the castle that succeeded.
It fails by collapsing to a billboard while reporting success.

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

## 12. Mini Turbo octree resolution — above 384 it bought nothing here

**Scope this claim carefully.** It is about the Mini Turbo decode grid on one
tested subject. It is NOT a statement about generator resolution in general, and
in particular it does not transfer to TRELLIS: raising TRELLIS from 512 to 1024
genuinely produced structure that was absent before -- railing gaps, wheel
spokes, roof detail. Those are different mechanisms. TRELLIS 512→1024 changes
what the latent resolves; Mini Turbo's octree only changes how finely an already
fixed latent is meshed.

The accurate phrasing of what was measured: *Mini Turbo octree above 384 did not
improve this tested latent reconstruction.*

The Mini Turbo decode grid is settable via `--octree-ladder res:chunks`. The
default ladder is `384:3000,320:2000,256:1500`. All three of the rungs above the
default were run on the same heron matte, same seed 1007, same steps:

| octree | decode grid | raw triangles | peak VRAM | seconds |
|---:|---|---:|---:|---:|
| 384 | — | 1,201,338 | 5,440 MB | 280.9 |
| 448 | `[111, 222, 444]` | 1,648,988 | 6,446 MB | 384.5 |
| 512 | `[63, 126, 252, 504]` | 2,129,470 | 7,643 MB | 722.1 |

**All three produce visually identical meshes.** This was checked on the raw,
undecimated output at 2400 px, cropped to the feet and to the head and reeds, by
two vision models and by eye. Luna on the feet: fused paddle at 384, still fused
at 448, "looks the same" at 512. Spark on the reeds: same 3–4 stalks, same
separation, no clarity gain across the ladder. The user's own read agreed.

So going from 1.2M to 2.1M triangles subdivides surfaces that already exist and
creates no new structure. The fused toes are fused in the latent, not in the
mesh -- which is exactly why this says nothing about TRELLIS, where the
resolution knob moves the latent itself. **Keep the 384 default for Mini
Turbo.** 512 costs 2.6× the wall clock and spills 1.5 GB
into system RAM for triangles carrying no extra information.

If thin structure matters, the levers are conditioning (§ "Conditioning beats
resolution") or TRELLIS, which wins on discrete structure.

### 512 fails intermittently, and the ladder will not catch it

The first octree-512 attempt died with:

    CUDA error: misaligned address

at 4,223 MB peak — a gigabyte *below* what 384 already reaches. It is an
alignment fault in the FlashVDM decode, not a memory ceiling. The identical
settings succeeded on the next attempt in a fresh process, so it is allocator
placement, not a property of the rung.

`mini_turbo_generate.py` only steps down its ladder on
`torch.cuda.OutOfMemoryError`; every other exception aborts the whole run. So a
misaligned-address fault means **the lower rungs are never attempted**. Retry in
a fresh process — the fault poisons the CUDA context, so an in-process retry
fails for a reason unrelated to the settings.

## 13. VRAM spill is real headroom, and it is taken from system RAM

Octree 512 peaked at **7,643 MB on a 6,143 MB card** and completed. Windows WDDM
grants a shared GPU memory budget of half system RAM — 15.3 GB ÷ 2 ≈ 7.6 GB — on
top of the dedicated pool, which is where Task Manager's ~13.7 GB comes from.
Reserved hit 11.4 GB on that run.

The cost is not only speed. **Shared GPU memory is system RAM.** While that run
held 11.4 GB, a concurrent Hunyuan3D-Paint process died with:

    numpy._core._exceptions._ArrayMemoryError:
    Unable to allocate 3.00 MiB for an array with shape (1, 512, 512, 3)

Failing to allocate three megabytes is not a texture-size limit — it is the
other job having eaten the machine. Treat the spill as usable for a single job
at a time and never as headroom to run two.

## 14. Texel starvation — face count must match the texture budget

> **RETRACTED IN PART, 2026-08-07.** The *visible* symptom this section reports
> -- surfaces reading as a mosaic of flat plates -- was an artifact of
> `render_textured_views.py`, which sampled textures with a nearest-neighbour
> lookup and no filtering. A face covering one or two texels took a single
> texel's colour across its whole area, and neighbouring faces landing in
> different texels met at a hard edge. Blender, which filters properly, shows
> none of it.
>
> Controlled test: the same fennec mesh baked at 1024 (1.40 median texels/face)
> and at 2048 (5.59), rendered through Blender, differ by **mean absolute
> 0.45/255, with 1.54% of pixels differing by more than 8**. A 4x atlas bought
> nothing visible.
>
> The measurements below stand. The conclusion drawn from them -- that the
> assets were visibly damaged by texel starvation -- does not. Section 18's rule
> was applied to crops and framing but never to the renderer itself, and two
> vision models confirmed the false reading because they were shown the same bad
> picture. `render_textured_views.py` now samples bilinearly.

Measured on the finished painted assets, texels per face (median):

| asset | faces | atlas | texels/face | atlas colour std |
|---|---:|---:|---:|---|
| fennec Mini Turbo | 400,000 | 2048 | 5.06 | 60 / 54 / 46 |
| heron Mini Turbo | 400,000 | 2048 | 4.50 | 21 / 19 / 17 |
| shaman TRELLIS | 289,838 | 2048 | 4.91 | — |
| frog TRELLIS 512 | 145,084 | 2048 | 9.47 | — |

A triangle with five texels holds one colour, and once chart padding takes its
share there is nothing left. Every face becomes a flat fill and the surface
reads as a mosaic.

**Every delivered asset has this.** It is invisible on low-contrast subjects —
the heron is nearly monochrome, so flat charts next to flat charts differ by
nothing — and obvious on high-contrast ones: the fennec is pale linen against
dark leather and the identical defect lands as camouflage.

The rule is an asset-class one, not a per-asset patch: face count has to be
matched to the texture budget, not chosen independently of it.

**≥32 texels per face is an engineering target, not a proven threshold.** Five
texels per triangle is demonstrably inadequate -- that much is measured. Where
the real cutoff sits, and whether it is the same for a shaggy character and a
hard-surface building, has not been A/B'd. 32 is the number that makes a chart
able to hold a gradient rather than a colour; treat it as a starting point and
replace it when the A/B exists. At texture 4096 with ~80% atlas utilisation it
works out to roughly 420k faces.

## 15. Serialising GPU jobs — use a lock, not a process check

Waiting for "no GPU processes running" before starting is a check-then-act race.
When the octree-448 job exited, two waiting scripts polled within seven seconds
of each other, both saw zero, and both started; one then starved the other to
death as described in §13.

`tools/gpu_lock.sh` replaces it, with the lock at an application-level path
(`%LOCALAPPDATA%/LowVRAM3DStudio/locks/gpu.lock`) rather than inside any one
agent's session directory -- two agents holding two different lock paths
serialise against nobody.

**Status: implemented helper, not yet end-to-end proven.** Every wrapper script
written today acquires it, but the workers themselves -- `trellis_run.py`,
`hunyuan_paint_texture.py`, `mini_turbo_generate.py`, the Blender render workers
-- do not acquire it internally. Anything invoking them directly still bypasses
mutual exclusion entirely. Wiring the lock into the entrypoints, and
demonstrating two competing jobs actually serialising, is outstanding work.

`mkdir` is atomic — the directory either
did not exist and this process created it, or it existed and mkdir fails, with no
window in between. A lock *file* would not do: `[ -f lock ] || touch lock` has
exactly the same race. The lock records its owner PID so a lock left by a killed
script is recognised as stale rather than blocking every later run.

Blender counts as a GPU job in the busy set. A Blender render during a paint run
has killed two paint runs.

## 16. Judging a render — the procedure, not the instinct

See `.claude/skills/visual-verify/SKILL.md`. Three rules, each of which exists
because it was violated:

1. **Open the image yourself before showing it or sending it anywhere.** A
   cropping bug produced a "head" sheet containing only reed tips and empty
   backdrop. It was sent to both vision models unexamined. Luna described a beak
   yawing sideways with a downward roll; Spark reported "beak fully visible in
   all 9 views, crop adequate, no occlusion". Both fabricated a detailed reading
   of an empty frame. A confident model answer is not evidence the image was
   valid.
2. **Size the evidence.** Fine features cannot be judged from a contact sheet at
   250–400 px per subject. Render one view at 2000 px+, crop, upscale, judge
   that. If the feature is not legible, the answer is "cannot tell".
3. **Two models plus your own eyes**, via `command-code` — Luna is
   `gpt-5.6-luna`, Spark is `meta/muse-spark-1.2-contributor`. Report all three
   readings including disagreement. Their contradiction on the empty crop is the
   only reason the fabrication surfaced.

### Two cropping traps on these renders

- **Contact sheets carry a white caption band.** Sampling background at `a[5,5]`
  hits the caption, so "not background" matches the entire backdrop, the bounding
  box becomes the whole image, and every relative crop lands in empty space.
  Sample from inside the render area.
- **The topmost pixel is not the top of the head.** Reeds project above the
  heron's skull, so a head band anchored to the bbox top captures reeds only.
  Verify a derived mask by printing its per-row widths — a mask covering every
  pixel of every row is a broken mask, not a large subject.

## 17. The latent cannot represent hair topology — and no setting fixes that

The weeping willow is the clearest result of the day. Mini Turbo at octree 384,
no decimation:

    tris 4,253,066   1062.5 s   peak VRAM 9,254 MB on a 6,143 MB card

3.5x the heron's triangle count at identical settings, 3.8x the time, and 3.1 GB
spilled into system RAM — the largest overflow recorded here, and it completed.

And the result is unusable. Both vision models, independently, on the raw clay:

| feature | verdict |
|---|---|
| thousands of fine hanging strands | fused; "no strand separation anywhere, even in silhouette" |
| ~20 teardrop lanterns | **zero** present; "not even approximated as bumps" |
| braided twisted trunk | a fat lumpy column; roots fused into a solid base disk |
| canopy | "smooth melted shell"; "classic minimal-surface collapse" |

Both rejected polygon budget as the cause unprompted. Spark: *"4.25M tris is
~100x more than needed to represent the braid/strands — the mesh is subdividing
a smooth blob."* The sparse-voxel latent with a diffusion prior trained on solid
Objaverse objects has no way to express hair topology or twenty separate small
instances, so it reconstructs the thinnest thing it can represent: a closed,
smoothed shell.

This closes §12 properly. Octree 384 -> 448 -> 512 changed nothing on the heron,
and the reason generalises: **decode resolution cannot add what the latent never
encoded.** More grid subdivides the blob more finely. For subjects made of thin
separated structures — foliage, hair, rope, chains, railings, small hanging
props — the generator is the wrong tool, not the settings.

Expect the same on any subject whose defining feature is thin-and-separated.
Judge those on the raw clay before spending a paint pass on them.

## 18. Vision-model verification — the three silent failures

Procedure lives in `.claude/skills/visual-verify/SKILL.md`. The traps that
produce a confident wrong answer rather than an error:

1. **Image outside the workspace.** The model reads the file with a tool; a path
   outside the working directory is refused, and the run exits **0** having
   printed only its opening sentence. `evidence/compare/<subject>/source.png`
   already exists for this reason — never point at `Downloads`. Confirm with
   `--output-format json` and look for `"stopReason": "permission_denied"` in
   the `run_end` event; the plain text output never mentions it.
2. **A preamble-only reply is a dead call, not a terse verdict.**
3. **`--effort` is Luna-only.** Spark answers `Muse Spark 1.2 Contributor has no
   adjustable reasoning effort` and does nothing else. Luna must always run at
   `--effort high` or `xhigh` — at default it confirms; at high it diagnoses.

And the failure that has nothing to do with the CLI: **an empty or wrong frame
gets answered anyway.** A crop containing only reed tips drew a detailed
description of a beak's rotation from one model and "beak fully visible in all 9
views, crop adequate, no occlusion" from the other. Open the image yourself
first. Their disagreement was the only reason the fabrication surfaced.

## 19. The atlas is confetti — thousands of independently-owned UV charts

> **THE SYMPTOM THIS EXPLAINS WAS NOT REAL, 2026-08-07.** This section was
> written to account for hard-edged plates on the painted fennec. Those plates
> came from a nearest-neighbour texture sampler in
> `render_textured_views.py`, not from the bake -- see the retraction in
> section 14. Rendered through Blender the same asset is clean.
>
> Everything measured here is still true and still worth knowing: 5,609 UV
> charts with a median of 14 faces, 36.7% atlas utilisation, 85% of the atlas
> reached by no camera, and the three discontinuities in the vendor blend. What
> is withdrawn is the causal story -- the defect/trigger/amplifier model at the
> end -- because there was no defect in the output to explain. Treat this as a
> characterisation of the unwrap, not as a diagnosis of a failure.

`tools/paint_view_provenance.py` replays the vendor's exact projection and blend
with the generated views replaced by one flat colour per camera. Nothing else
changes, so any structure in the output comes from geometry and blending alone.

On the TRELLIS fennec at 1024 it returned **85% of the atlas owned by no camera
at all**, and the owned 15% shattered into thousands of specks whose winning
camera differs from that of the speck beside it.

The cause is the unwrap, and it is universal:

| mesh | faces | UV charts | median faces/chart | atlas used |
|---|---:|---:|---:|---:|
| boat | 149,720 | 3,232 | 4 | 44.4% |
| fennec Mini Turbo | 400,000 | 7,249 | 3 | 39.6% |
| fennec TRELLIS | 284,142 | 5,609 | 14 | 36.7% |
| heron Mini Turbo | 400,000 | 9,806 | 5 | 33.1% |
| panda Mini Turbo | 400,000 | 7,382 | 15 | 36.7% |
| shaman Mini Turbo | 400,000 | 4,686 | 4 | 45.2% |
| whale Mini Turbo | 400,000 | 8,086 | 3 | 37.5% |

Two consequences follow.

**Only about a third of the atlas holds any surface**, so the naive
`width x height / faces` figure overstates density by roughly 3x. Always weight
by UV triangle area, the way section 14 does. Quoting the naive number is what
made a 1.00-texel-per-face bake look like 3.69.

**Roughly 60% of the surface that *is* unwrapped receives no projected pixel**
and is filled by `uv_inpaint`. Raising the atlas does not change this — coverage
is geometric, not resolution-dependent.

Three separate discontinuities in the vendor bake turn per-chart ownership into
hard edges rather than a gradient:

```python
cos_image[cos_image < cos_thres] = 0                 # a cut, not a falloff
project_cos_map = weight * cos_map ** self.config.bake_exp    # bake_exp 4
if painted_sum / view_sum > 0.99: continue           # whole view dropped, in fixed order
```

The camera weights are `[1, 0.1, 0.5, 0.1, 0.05, 0.05]` for front, right, back,
left, top, bottom -- so a chart facing right is coloured at a tenth the
confidence of one facing front, and the last two views can be discarded whole.

**Do not read "universal" as "harmless."** This is a latent structural weakness
present in every asset shipped so far. It stays invisible until something else
exposes it. The current causal model for the fennec:

- **defect** — thousands of tiny independently-owned charts
- **trigger** — a quarter-resolution texture budget (median 1.00 texels/face)
- **amplifier** — high-contrast cream-and-dark markings making cross-camera
  disagreement visible where the near-monochrome whale and shaman hid it
