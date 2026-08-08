# Handoff — 2026-08-08, trees / billboard gate / rigging / segmentation

State of the lane after a long session. Written for whoever picks this up next,
including me.

## What is generated and where

`tools/run_asset.sh <subject> [res] [seed]` runs the whole lane for one subject:
matte → conditioning → TRELLIS → **billboard gate** → 9 views → paint → 9 views
→ segmentation. A new subject needs only `evidence/compare/<subject>/source.*`.

`tools/run_miniturbo.sh <subject>` is the alternative lane for the cases TRELLIS
cannot do. It needs **both** `--image` and `--conditioning-image`; passing only
the latter exits before touching the GPU.

New this session, all gated and rendered:

| subject | route | faces (deliverable / master) | note |
|---|---|---|---|
| seal diver | TRELLIS 1024 | 298,334 / 7.5M | rigged, walking, segmented |
| fat tree | TRELLIS 512 | 148,852 / 23.3M | largest decode in the project |
| oak hamlet | TRELLIS 512 | 142,776 / 6.8M | cleanest gate numbers |
| tree city | TRELLIS 512 | 146,938 / 9.3M | see the matte section |
| greentree | Mini Turbo | 400,000 | TRELLIS billboards this subject |

## The three silent failures found

Each of these exits 0 and produces a plausible receipt.

**1. TRELLIS can return a billboard** (runbook 10a). Two crossed cardboard
panels with `success: true`. `tools/check_not_billboard.py` measures fill,
quad_ratio and spread and fails on two of three. `quad_ratio` is a **lower**
bound — a billboard sits at exactly 1.0 and every solid sits above it. The gate
runs between geometry and paint because paint is the expensive stage.

**2. The matte can amputate the subject** (runbook 10c). rembg/u2net deleted the
tree city's whole canopy; corner alpha, coverage and the billboard gate all
passed, because the stump that remained was genuinely solid. `matte_rembg.py`
now fails when the subject starts more than 25% down the frame. Choose the matte
by backdrop: `auto_matte.py` for plain, `matte_rembg.py` for non-flat.

**3. Rigged exports lose their skin** (`lowvram3d-repo`). `export_apply=True`
consumes the Armature modifier and writes `JOINTS_0`/`WEIGHTS_0` with no `skins`
array. Bone heat reports failure through the report system rather than an
exception, so the old fallback was unreachable — it weighted **0%** of every
asset here, probably because these meshes are shattered at UV seams and it needs
a connected manifold. Gate every rig on `tools/validate_rig_export.py`.

## Hardware ceilings, measured

- **~38k tokens does not fit 6 GB.** `--res 1024` switches stage 4 into the
  LR→upsample→HR cascade; the HR half is what does not fit. Past the ceiling the
  penalty is **23x per step**, not a gradual curve, because Windows pages VRAM
  instead of raising OOM. `dwm` holds 1.4 GB of the card, so budget ~4.7 GB.
- `nvidia-smi` will not show the spill. `Get-Counter "\GPU Process Memory(*)\Shared Usage"` will.
- Longest successful run: moss titan, 2806.5 s. Elapsed time is a poor anomaly
  signal; per-step cadence is the good one.

## Monitoring

`tools/watch_trellis.sh <log> [stall_seconds]` reports PROGRESS with the gap
since the last line, STALL every 5 minutes with GPU state, and EXIT with whether
the output exists. Watch the log `trellis_run.py` writes with `--log`; the
shell's `tee` copy never receives a carriage-return progress bar.

## Open, in rough priority order

1. **The vendor paint flattens organic subjects.** Four assets in a row where
   the pre-paint TRELLIS atlas looks better — seal diver, Mini Turbo greentree,
   fat tree, oak hamlet. Broad surfaces lose their weathering and material
   separation. Either drop the stage for organics or find out why. Not
   independently verified: Luna and Spark were rate-limited all session.
2. **Rig weights.** Bone heat fails at 0%, so everything runs on the geodesic
   fallback. The titan robot walks cleanly; the seal diver's legs still tear.
   `libigl` 2.6.1 and `coacd` are installed for this — `igl.bbw` is present but
   the Python bindings ship no tetrahedraliser, so volumetric BBW needs
   fTetWild. `igl.harmonic` is the surface variant with no new dependency.
   UniRig is the right answer and cannot run here: it needs 8 GB VRAM and
   flash-attn on sm_80+.
3. **Segment before rigging.** Part boundaries already land at neck, shoulder,
   waist, knee and ankle, which is where weight transitions belong, and
   segmentation already computes the geodesics the rig recomputes.
4. **The high-poly masters are untracked.** 1.8 GB of `.ply` across the project,
   excluded by `.gitignore`, existing on one disk. They hold detail the 148k
   deliverables do not — the canopy is a shell in the GLB and separate leaf
   plates in the master.
5. `evidence/driver_validation/` holds ten runs at ~57 MB, probably dead weight.

## Traps worth not rediscovering

- A GLB duplicates vertices at every UV seam, so the index buffer is not the
  surface's connectivity: the seal diver reads as 8,031 shells until positions
  are welded, then 8. Weld for adjacency, never in place — that fuses the atlas.
- `trimesh.submesh` re-packs the whole atlas per part; 14 parts came to 6.6 GB.
- Blender 5.2 has no `Action.fcurves` (4.4 moved to layers/channelbags), renamed
  `NISHITA` to `MULTIPLE_SCATTERING`, resolves bare relative render paths against
  the drive root, and exits 0 when its Python raises.
- The inspection scene compresses display scale (`DISPLAY_EXPONENT`) so a 35x
  size range reads as ~4.5x. `REAL_SIZES.json` keeps true metres for export.
- Blender 5's default AgX view transform flattens catalogue renders; the scene
  sets Standard.
