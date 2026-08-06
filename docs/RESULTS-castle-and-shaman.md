# How the castle and shaman were produced

Two assets, both from a single concept image, both on a GTX 1660 SUPER with
6 GB VRAM and no tensor cores. This records what was actually run, what the
numbers were, and — more usefully — which of the things that *looked* like the
answer turned out not to be.

| | 4K gothic castle | bird-skull shaman |
|---|---|---|
| rating (independent vision review) | mid-ground, low end | **mid-ground, upper end** |
| ranked against prior subjects | 4th of 4 | **2nd of 4**, best single-image result |
| generation | seed 4242, 342.1 s | seed 777, 235.5 s |
| structured latent | 111,520 | **60,832** (smallest of any subject) |
| decode | 1,783,319 voxels → 3,949,828 faces | 746,035 voxels → 1,618,674 faces |
| after decimation | 144,002 | 144,728 |
| attempts needed | **9** | 4 (3 of them invalid, see below) |

---

## The recipe

```bash
# 1. Matte. Characters need the contact shadow removed or it becomes a floor slab.
py workers/pipeline_matte.py --image source.png --output matte.png \
   --shadow-tolerance 180 --shadow-from 0.78

# 2. Square it, preserving thin structures through the downsample.
py workers/prepare_input.py --image matte.png --out input512.png

# 3. Check what will be lost before spending four minutes.
py workers/feature_risk.py --image input512.png --overlay risk.png

# 4. Generate. Vary the seed; failures on this card are stochastic.
bash workers/trellis_retry_seeds.sh input512.png asset.glb 4242 777 12345 31337

# 5. Grade. These numbers came from vision review, not from guessing.
py workers/hybrid_atlas_composite.py --mesh asset.glb --out graded.glb \
   --contrast 0.22 --shadows -0.10 --highlights 0.04 --saturation 1.12 --warmth 0.04

# 6. Promote lit windows and lamps to actual emission.
py workers/emissive_from_atlas.py --mesh graded.glb --out lit.glb

# 7. Look at it.
py workers/preview_textured_mesh.py --glb lit.glb --out sheet.png
```

Castle grade differed: `--contrast 0.28 --shadows -0.09 --highlights 0.10
--saturation 1.16 --warmth 0.07`.

---

## What actually made the difference

**Varying the seed.** Not a parameter. The castle took nine attempts; the first
five were flag changes and *none of them was the answer*. Each run's
sparse-structure stage is nondeterministic, so downstream stages get differently
shaped tensors, and TU116 has shape-dependent kernel faults. The same image
failed in five different places on five consecutive runs. The sixth through
ninth varied only the seed, and the ninth worked.

**Grading from measurement, not instinct.** Every grade specified by an outside
vision review beat the one guessed locally — three times out of three. The
instinct to fix a dull atlas with saturation was wrong twice: the atlases were
*cool*, needing a hue shift, and `--warmth` did not exist until a review said so.

**Removing the contact shadow.** The shaman's studio shadow is opaque to the
generator. Left in, it becomes a floor slab welded to the feet.

**Looking at the renders instead of the metrics.** The coverage number written
to tune shadow removal ranks total destruction as the best outcome, because the
feet lie inside the region it measures:

| tolerance | "alpha remaining" | actual result |
|---:|---:|---|
| 130 | 10.82% | shadow still there |
| **180** | **7.69%** | **correct** |
| 240 | 1.12% | feet destroyed |
| 300 | 0.00% | everything below the hem deleted |

---

## The structural-cell rule

The single most useful thing learned. `trellis-cli` logs
`active voxels @res32`: occupancy is decided on a **32³ grid** before any
refinement. At a 512 px input that is **16 px per structural cell**, and a
feature that cannot claim a cell is never built.

| feature | size in input | cells | outcome |
|---|---:|---:|---|
| hanging cords | 1 px wide, ~150 px long | ~10 along the run | **survived** |
| pendants | 15–23 px | ~1–1.5 | blobbed to identical stubs |
| portcullis bars | ~3 px | 0.2 | vanished |
| crenellations | ~6 px, tightly repeated | — | merged to a flat parapet |
| beak, staff, crossbar | 60–400 px | 4–25 | clean |

It is **not thickness and not spacing**: a 1 px cord beats a 20 px pendant,
because a long feature claims cells along its length while a compact prop has
one chance at one cell. `workers/feature_risk.py` measures this with two
morphological tests — opening finds solid geometry too thin to claim a cell,
closing finds negative space that will fill in — and predicts every observed
outcome above, including that the staff ring's ~30 px hole would survive, which
it did.

---

## Things that looked like the answer and were not

**`--res 1024` does not change occupancy.** Same subject: 1901 active cells at
res-512, 1921 at res-1024. It is a cascade that refines what the 32³ grid
already claimed. It does raise the HR latent from 60,832 to 273,760, so it can
sharpen a stub — but it cannot recover a feature that never claimed a cell. It
also has never completed on this card: `--f32` clears the kernel faults
(`CUBLAS_GEMM_DEFAULT_TENSOR_OP` on a card with no tensor cores) and then stage 5
wants 4696.87 MiB of VRAM against ~4600 free.

**Widening sub-cell holes destroys the prop.** The idea was to preserve pendant
shape by enlarging holes to the cell threshold, no assembly required. The
arithmetic kills it: widening a 7 px hole to 16 px removes ~9 px of material all
round and the pendant wall is 8 px thick, so the hole eats through. Kept in
`prepare_input.py` as a documented negative result, default off.

**The latent band did not predict capacity.** 96,448 failed and 111,520
succeeded on the same image. One image draws latents from 94,528 to 134,944 — a
43% spread — so a single latent is not a property of a subject at all. The abort
threshold built on four single samples killed a run that might have worked; it
now defaults to off.

**Denoising is a no-op, twice measured.** 79.9% of vertices frozen, mean
displacement 5e-06, and zero vertices reaching the displacement cap — the filter
is not restrained, it has nothing to say. Bilateral normal weighting is designed
to refuse exactly the geometry in question, and that geometry is real: faces on
>90° adjacencies have 1.02× median area and are thin double-sided sheets holding
19.6% of surface area. They are the spokes and railings.

---

## Process failures worth not repeating

**`TaskStop` does not kill the process tree.** A stopped sweep keeps looping and
relaunches its next seed seconds after you verify nothing is running. Two
`trellis-cli` processes then share 6 GB and produce `illegal memory access` at
stage 3 — which reads exactly like a subject-specific hardware fault. **Three
shaman seeds were written off that way**, and the very next seed succeeded on a
clean card. `trellis_retry_seeds.sh` now refuses to start if another
`trellis-cli` is running.

**Check the process list before blaming hardware.** An orphaned worker
accumulated 19,578 s of CPU while its competition's failures were attributed to
the GPU.

**A preview that cannot show the thing you are checking will answer "no".**
`preview_textured_mesh.py` sampled base colour only, so an emissive texture was
invisible to the one tool used to verify emissive extraction. Same class of
error as the vertex-colour preview that made a decal look like it had failed,
and several review verdicts in this project were formed on those degraded images.
