# Handoff — castle and shaman, 2026-08-06

Two assets delivered from single concept images on a GTX 1660 SUPER (6 GB, no
tensor cores, 16 GB RAM). Full method in `docs/RESULTS-castle-and-shaman.md`;
commands in `docs/RUNBOOK-trellis-pipeline.md`. This is what the next person
needs to know that is not in either.

## 1. State

| asset | file | status |
|---|---|---|
| gothic castle, graded | `evidence/deliverable/castle2_gothic_graded.glb` | done |
| gothic castle, lit windows | `evidence/deliverable/castle2_gothic_lit.glb` | done |
| shaman, res-512 graded | `evidence/compare/shaman/shaman_graded.glb` | done, pendants are stubs |
| shaman, no decimation | `evidence/compare/shaman/shaman_full.glb` | **just generated, NOT yet reviewed** |

`shaman_full.glb` is seed 777 at `--decim 0`: V=704,752 F=1,467,620 against
144,728 for the decimated version. It exists to answer one question — whether
the pendants were destroyed by the default quadric decimation or were never
generated properly. **Nobody has looked at it yet.** Render it first:

```bash
py workers/preview_textured_mesh.py --glb evidence/compare/shaman/shaman_full.glb \
   --out evidence/compare/shaman/shaman_full_7view.png
```

## 2. One narrow regression against earlier Hunyuan runs: the pendants

**Read this as a scoped finding, not a reason to switch generators.** The user
is explicit that the TRELLIS models are better overall — the pendants are the
single thing that earlier runs did better. Do not re-open the lane decision on
the strength of one prop.

**The user has produced shamans with intact pendants before, using
Hunyuan3D-2 — not TRELLIS.** Found late, in
`C:\Users\Lauri\Desktop\lowvram3d-magicmusic-asset-systems`:

- `benchmarks/manifests/antlered_bird_shaman_anchor.json` — canonical fixture,
  source 1122x1402, same image
- `.github/workflows/shaman-1p5m-comparison.yml`, `shaman-staff-hole-repair.yml`,
  `shaman-v2-*` — a full production pipeline with texture QA gates
- generator references: **Hunyuan3D-2, "1p5m" (19 mentions), octree 256**
- `blender/shaman_*.py` — ~15 scripts including garment separation, rigging,
  antler debris cleanup

**Why this matters architecturally.** Everything derived today about why small
props are lost — the structural-cell rule, the 32^3 occupancy grid, the 16 px
threshold — is a property of **TRELLIS's sparse-voxel latents**. Hunyuan3D is a
**VecSet** model: an unordered set of latent vectors with no occupancy grid at
all. There is no cell for a pendant to fail to claim.

So "TRELLIS replaces Hunyuan for geometry" (HANDOFF-boat-sixview §10) stands as
the lane decision — it is backed by a trustworthy measurement (F1 0.690 vs
0.573) and by every subject since. The qualification is narrow: **on small
detached props specifically, VecSet has no failure mode equivalent to the
occupancy gate**, which is why the earlier runs kept the pendants.

**The useful next step is diagnostic, not a lane change:** generate this shaman
through Hunyuan3D-2 at octree 256 purely to confirm the pendants survive there.
If they do, it isolates the loss to the occupancy gate rather than to input
resolution or decimation — which tells you whether the gap is closable inside
TRELLIS (detail pass, higher HR token budget) or not at all. Do not adopt
Hunyuan for the whole asset on the strength of it.

## 3. What was measured today and is worth keeping

**The structural-cell rule.** `active voxels @res32` — occupancy on a 32^3 grid
before refinement, so 16 px per cell at a 512 input. A feature that cannot claim
a cell is never built. Predicts every observed outcome: 1 px cords survive
(~10 cells along their run), 15-23 px pendants blob (~1 cell), 3 px portcullis
bars vanish (0.2 cells). `workers/feature_risk.py` measures it, with an opening
test for thin solids and a closing test for negative space that will fill in.
Validated: it predicted the staff ring's ~30 px hole would survive, and it did.

**Applies to TRELLIS only.** See §2.

**Seed variation beats parameter tuning.** The castle took 9 attempts; the first
5 were flag changes and none was the answer. Failures on TU116 are stochastic
because the sparse-structure stage is nondeterministic and downstream kernels
are shape-sensitive. `workers/trellis_retry_seeds.sh`.

**Grading: trust the vision review.** Three for three, an outside review's grade
beat the locally guessed one. The instinct to fix a dull atlas with saturation
was wrong twice — the atlases were *cool*, needing `--warmth`, which did not
exist until a review said so.

## 4. Open

1. **Review `shaman_full.glb`** — the decimation question, above. If the
   pendants are intact at 1.47M faces, the fast recipe is
   `--decim 0 --box-uv` and it applies to every subject, including the castle's
   portcullis. `--box-uv` is insurance: xatlas on a high-shell-count mesh at
   this size has never completed in this project.
2. **Hunyuan3D-2 comparison** — §2.
3. **res-1024 is nearly reachable.** `--f32` clears the kernel faults
   (`CUBLAS_GEMM_DEFAULT_TENSOR_OP` on a card with no tensor cores; stage 3 goes
   from failing at 28.9 s to completing in 83.2 s). Stage 5 then wants
   **4696.87 MiB** of VRAM against ~4600 free — short by about 2%. Freeing
   ~200 MB of desktop VRAM would likely land it. `--max-tokens 32768` was being
   tested for the same reason and never finished. Worth it for surface fidelity;
   **it will not restore missing pendants**, since occupancy is identical
   (1901 cells at res-512 vs 1921 at res-1024).
4. **Detail passes are built but unmerged.** `prepare_input.py --crop` produces
   properly-detailed sub-assets (mobile at 2.2x magnification, pendants at 5-7
   cells instead of 1). Merging two independently-normalised TRELLIS outputs
   needs registration: image-space prior for x/y/scale, then ICP against the
   stub geometry. The user does not want an assembly step, so this is on hold.
5. **`--atlas 2048` untested** under the seed-retry regime. A vision review
   judged it not worth it — geometry is the binding constraint, not texels.

## 5. Do not repeat

- **`TaskStop` does not kill the process tree.** A stopped sweep relaunches its
  next seed seconds after you verify nothing is running. Two `trellis-cli`
  processes sharing 6 GB produce `illegal memory access` at stage 3, which reads
  as a subject-specific hardware fault. **Three shaman seeds were written off
  that way**; the next seed succeeded on a clean card. The retry script now
  refuses to start if another `trellis-cli` is live — but kill the **bash loop**
  as well as the process.
- **Widening sub-cell holes destroys the prop.** Tried, measured, failed:
  widening a 7 px hole to 16 px removes ~9 px of material and the pendant wall
  is 8 px. Documented in `prepare_input.py`, default off.
- **A coverage metric ranked total destruction as the best shadow-removal
  setting**, because the feet lie inside the region it measured. Judge crops by
  eye.
- **A preview that cannot render what you are checking will answer "no".**
  `preview_textured_mesh.py` sampled base colour only, so emissive extraction
  was invisible to its own verification tool.
