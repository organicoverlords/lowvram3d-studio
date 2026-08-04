# Handoff — 2026-08-05 (third session)

Repo: `C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803`
Branch: `agent/scene-pipeline-smoke-20260803`
Head: `9319ff0` (this session started at `61ab886`, 18 commits)
Unreal project: `C:\Users\Lauri\Desktop\UnrealAITest58\UnrealAITest58.uproject` (UE 5.8.0)

**Start here:** `docs/DESIGN_IMPROVEMENTS.md` (read the retraction in section 1
first), then `docs/AXIS_CONVENTIONS.md`, then `docs/pipelines/README.md`. Run
`python -m uemcp doctor` from `unreal/` before diagnosing anything.

## Read this before trusting any number in here

Three defects this session were found by **looking at an image**, and all three
had survived confident, mutually consistent numeric checks:

1. a "second building" that was a tree-trunk shadow;
2. a "tree line" that is **one wind-swept tree** -- twelve instances came from a
   pixel budget, and every vegetation symptom followed from that one number;
3. a facing test that put the barn's appearance on its **rear**, which shipped
   because nothing in the project could render a texture outside an Unreal scene.

`docs/DESIGN_IMPROVEMENTS.md` records the general form: *agreement between
measurements is not evidence when they share an input.* Crop border contact,
projected scale disagreement and pairwise overlap all independently indicted the
vegetation region, and all three were downstream of the same wrong cluster count.

## What changed this session

- **Split-phase runs.** Generation and the editor cannot share a 6 GB card; the
  editor's ~2.3 GB pins every asset to octree 256. Run `--phase generate` with
  Unreal closed, then `--phase build --resume` with it open. Both assets now
  reach **384**, which had never happened before.
- **Square conditioning.** Mini Turbo letterboxes what it is given, so an
  819x266 barn filled a third of the frame. Crops are padded to square.
- **One connected mass is one object.** k-means over points is gone.
- **Welding on export.** Every delivered mesh was triangle soup: 449,917
  vertices for 149,997 triangles, 149,960 disconnected bodies. Now 74,556 and
  1 body, files 3.5x smaller. Likely the cause of the import handler timeouts.
- **Placement is optimised, not just audited.** Cross-region overlap 0.26 -> 0.05.
- **Honest texture coverage.** Faces the camera never saw are flat-filled rather
  than smeared. Barn 0.392, tree 0.426 -- i.e. most of an asset's appearance is
  invention, and the ship benchmark independently measured 0.19.
- **`workers/render_textured_views.py`** renders appearance on the CPU, and runs
  automatically for every generated asset.

## Unfinished, and why

**Nothing has been built in Unreal since any of this.** The last scene predates
welding, the facing gate, the placement objective and both current assets. That
is the obvious next step and it needs the editor open.

Multi-view texturing is the correct fix for the largest remaining defect and has
**never worked on this machine** -- see the corrected section 3 of the design
notes, and `proof/benchmarks/20260803-ship-production-texture.json`, whose own
root cause reads "5 of the 6 conditioning views are mirrored fills at confidence
0". Give it its own budget; do not slot it into a scene run.

`scripts/measure_offaxis_stability.py` is still written and never executed.

---

## 1. What actually works now

One command takes an image to a built, rendered Unreal scene **containing
generated geometry** rather than engine primitives:

```bash
PYTHONPATH=src python -m lowvram3d.image_to_scene_pipeline \
  --image "C:/Users/Lauri/Downloads/benchmarkpics/treesandbarn.png" \
  --project "C:/Users/Lauri/Desktop/UnrealAITest58/UnrealAITest58.uproject" \
  --scene-id barn_gen --input-kind scene --quality-tier preview \
  --output-root "/Game/AgentProof/BarnGen" --evidence-root evidence/barn-gen \
  --generate-assets --max-generated-assets 1
```

Ends with `PIPELINE_CLASSIFICATION=SCENE_BUILT_WITH_GENERATED_ASSETS`.

Chain: image → MoGe-2 depth → SegFormer regions **+ per-region masks + geometric
clustering** → unprojected placement → **crop → matte → Mini Turbo → decimate →
project the crop as texture → import** → spawned in Unreal → rendered. Every
stage writes a receipt.

**The classification distinguishes generated from primitive.** `SCENE_BUILT` used
to be true of a field of cubes; there are now three outcomes
(`..._WITH_GENERATED_ASSETS`, `..._FROM_PRIMITIVES`, `PARTIAL`) and the build
receipt counts the two kinds of actor separately. Do not average them.

Interpreter note: use Python312 and an **absolute** `PYTHONPATH`. `PYTHONPATH=src`
does not survive an MSYS shell.

---

## 2. Axis conventions — measured, and not what was assumed

Full detail and receipts in `docs/AXIS_CONVENTIONS.md` and `evidence/axis-probe/`.
This closes the open item that had been guessed at three times.

**Unreal's glTF importer maps glTF (x, y, z) onto Unreal (x, z, y), at 100 cm per
glTF metre**, with zero off-axis leakage. Measured with an asymmetric probe
(`workers/make_axis_probe_glb.py` + `unreal/measure_axis_mapping.py`), and
independently confirmed later when a generated mesh's Unreal extents matched its
glTF extents under that mapping exactly.

Two consequences: the importer already applies the metre→centimetre conversion,
and the Y↔Z swap is a *reflection*, which is correct — it converts right-handed
Y-up glTF into left-handed Z-up Unreal. A mesh that arrives mirrored is mirrored
in the source file.

**The reconstruction rendered upside down because trimesh flips v on glTF
export.** It holds UVs bottom-left-origin and flips on write to reach glTF's
top-left origin, so authoring v in image-row order exports an inverted texture —
the sky painted along the ground. `moge_reconstruct` now authors v inverted so
the export flip lands it right way up. The axis conversion was never wrong: MoGe
returns OpenCV points with Y **down** (confirmed against the model: image top at
Y −8.18, ground at +1.25), so negating both Y and Z is correct.

**Read this part before trusting a render score.** An earlier pass this session
removed the Y negation instead, flipping the geometry to match the inverted
texture — two wrongs that cancel from the source camera and nowhere else. It
scored **1.439** at yaw −90 with no flip, *better than the correct mesh*, because
a texture-inverted mesh paints bright cloud onto the ground and the source's
upper region is also bright cloud. The correct mesh honestly renders black where
MoGe masked the sky and scores **0.533**. Source-view similarity is the
photometric pipeline's own test and has now certified a world-space-wrong mesh
twice in this project. Do not use it as a correctness gate.

What settles it in a second, without rendering: read the GLB accessors and check
that vertices carrying the *top* of the texture sit higher in Y than those
carrying the bottom, and that the sky rows MoGe masked away are missing from the
**top** of the v range.

| mesh | v range | texture-top verts | texture-bottom verts |
|---|---|---|---|
| **corrected** | **0.050–1.000** | **+8.16** | **−1.24** |
| geometry-flipped (the wrong fix) | 0.000–0.950 | +1.24 | −8.16 |
| original | 0.000–0.950 | −1.24 | +8.16 |

The source camera for a reconstruction is **yaw −90**, not yaw 0 — that part
held up, and the corrected mesh wins its sweep with `identity`.

---

## 3. Per-object generation

`src/lowvram3d/asset_generation.py` is the stage; `workers/mini_turbo_generate.py`
is unchanged and still the only generator.

Three judgements, each deliberate:

- **Surfaces are not generated.** Terrain, water and paths are measured extents,
  not objects with a silhouette, and keep their planes.
- **Scatter regions generate once** and reuse the mesh across instances. A tree
  line is twelve placements of one subject, not twelve ten-minute generations.
  This is a cost decision and the receipt says so.
- **Failures are per-asset.** A region that fails keeps its primitive and is
  recorded as failed; failing the run would discard everything that worked.

**Matte from the segmentation masks, never rembg.** The segmentation stage
computed a per-class mask and threw it away; it now writes one PNG per region
(`--mask-dir`). rembg on this source erased half the barn — the same failure that
stripped the shaman's ornaments. Measured difference on the same crop: 99 bodies
and not watertight → **12 bodies, watertight**, and the shape went from a smooth
loaf to a barn with a ridge line, a doorway and the lean-to from the photo.

**Decimate before leaving the stage.** Mini Turbo samples a 384³ volume whatever
the subject, so one building is ~1.7 M triangles — grid resolution, not detail.
Blender reduces it to 150 k (`workers/decimate_mesh.py`; no Python decimator is
installed in this interpreter and Blender is already a declared dependency).
This takes the Unreal import from **over ten minutes to seconds**.

`workers/preview_generated_mesh.py` renders any asset on its own, on the CPU, so
"is this actually a barn?" does not require building a scene.

---

## 4. Runtime and its failure mode

Mini Turbo needs its own interpreter, `hy3dgen` on `PYTHONPATH`, and a model root
that is neither — all three checked up front by `resolve_runtime()`, which names
what is missing. If the weights are absent, look in the parallel `hub\` HF tree
before re-downloading 3.8 GB.

**The `CUDA error: misaligned address` was VRAM headroom, and it is diagnosed.**

Assets after the first kept dying with a misaligned address inside a Linear on
the second diffusion step, which reads like a driver defect. It is not. The crop
that failed all three ladder rungs as the *second* asset generated cleanly at
full resolution as the **first** asset of a cold run — 416 k triangles at a
4,597 MB peak — so the input was never the problem.

The card is 6 GB. Mini Turbo peaks at **4,597–5,017 MB** at octree 384 (measured,
three runs). **A running Unreal editor holds about 2 GB of the same card**, and
this pipeline drives both. With the editor up, `nvidia-smi` reports ~3,980 MB
free, which is less than the top rung needs. Short of memory, this GPU does not
reliably raise `OutOfMemoryError` — it faults.

So generation now reads free VRAM before each asset, waits for headroom, and
**drops ladder rungs it has measured evidence it cannot afford** — with the
editor running it starts at 320, which is the rung that actually succeeded in
that state. Only rungs with a measured peak are ever dropped; the lower ones are
left unmeasured rather than guessed. The receipt records free VRAM per asset, so
a future failure is attributable rather than mysterious.

The structural point for anyone extending this: **the two halves of this
pipeline compete for one 6 GB card.** Generating while driving the editor is the
constraint, not a flaw in Mini Turbo.

---

## 5. Bugs found and fixed this session

Each of these put real work into the project and made it look like something
else had failed.

1. **`unreal.Rotator` is `(roll, pitch, yaw)`.** `Rotator(0, yaw, 0)` sets
   *pitch* and lays every generated mesh on its side. Caught by recording
   requested against placed extents — they came back as permutations of each
   other. Use keyword arguments.
2. **Relative paths crossing into the editor.** The editor resolves a relative
   path against the *project* directory, so a generated mesh reported as "not
   found" and renders landed in `UnrealAITest58/`. Resolve before sending.
3. **A handler timeout is not a failure — but a script error is.** The import
   workaround treated every exception as "still working" and polled for thirty
   minutes for a mesh that could never appear.
4. **Polling restarted the work it was waiting for.** The poll re-ran the
   importer, which starts an import when it finds no mesh, so a two-minute import
   never finished. Polling is query-only now.
5. **Import reuse ignored the source.** A regenerated mesh keeps its filename, so
   a scene was rendered with a barn that had been replaced an hour earlier, with
   nothing in any receipt saying so. Reuse now requires a source-hash match,
   stored as asset metadata.
6. **Scale fit the smallest axis ratio**, so a 15 m tree line became 2.6 m lumps.
   Match the measured *height*: it is the dimension a single view determines.
7. **Importing inside the scene build** hid a successful ten-minute import behind
   a handler timeout and shipped primitives with a `PROVEN` receipt.
8. **`depth_receipt` was referenced before assignment** when a SceneSpec was
   supplied.

---

## 6. Open items, highest leverage first

1. **Vegetation instances, properly.** Regions that get scattered are now split
   into spatially coherent clumps by k-means over the point map
   (`cluster_region_points`), each placed at its own depth and size. That was
   necessary because a semantic class is not an object: this "tree" region's
   pixels run from 2.4 m to 21 m with a median of 13.46 m — *identical* to the
   barn's — which is how a building ended up entirely inside a tree.

   Measured effect (`unreal/audit_actor_overlaps.py`): worst barn-vs-tree
   overlap **1.00 → 0.28**, buried pairs 3 → 1, and the remaining one is the
   hovel, still a primitive. The scene now reads as a barn with a tree line.

   Canopy clumps no longer float. Segmentation fits a **sloped ground plane**
   (least squares over every horizontal surface's points — a constant would be
   wrong by metres, since this ground drops 0.176 m per metre forward), and each
   clump gets an inferred `trunk_support` from that plane to its base.

   Two things that were tried and rejected, both worth not repeating:
   - *Stretching the clump down to the ground.* The mesh is a foliage blob,
     roughly as deep as it is tall, so scaling it uniformly to a
     ground-to-canopy height makes it that wide too — overlapping pairs went
     1 → 31 and buried 0 → 9.
   - *Trunks everywhere.* A trunk descending from a canopy centroid through the
     barn is an inference the evidence does not support: crowns overhang, and a
     tree behind a building has its trunk behind it. Trunks that would pierce an
     observed solid are withdrawn (7 of 9 here), which is honest and keeps the
     overlap numbers at their best.

   Still wanting real instance segmentation: the crop for a scatter region is a
   chunk of hedge, so the mesh is a blob. Accurate to the input, not a tree.

   **Placement now takes every position and size from measured points**
   (`measured_unreal_m`, converted to Unreal's frame once, in segmentation)
   rather than unprojecting a bounding box at one depth. Mixing the two methods
   is what put the ground plane 3.6 m above the barn's base — the ground was
   measured and the barn unprojected, and MoGe's ground is not flat here (1.3 m
   below the camera at 2.3 m out, 4.9 m below at 13.5 m). Overlapping pairs
   11 → 1, buried 0.

   Regions that keep a primitive are tinted to their own mean source colour.
   Everything rendered default white before, so ground, water and every unbuilt
   volume read as the same blank slab and hid whatever stood on them.
2. **Better texturing than a single-view projection.** `workers/project_crop_texture.py`
   projects each asset's conditioning crop back onto it, along the mesh's own
   −Z, mapping the front-facing bounding box to the crop. The generator aligns
   its output to the conditioning view by construction, so this lands. UVs stay
   in *object* space deliberately: a scatter region reuses one mesh across every
   instance, and a world-space projection would need one baked copy per
   instance.

   The front of an object gets its real appearance and the sides stretch along
   the projection axis. That is inherent to one view and is what
   `offaxis_stability` exists to catch — it is not a claim the back was seen.
   Filling the matte against the crop's mean colour rather than white matters:
   white leaves a bright halo wherever the projection spills past the subject.

   Textures import with sRGB already enabled here (checked, not assumed —
   the reconstruction path had to correct for the opposite).

   Next would be multi-view: the back and sides are currently smeared.
3. **Recover the top octree rung.** The fault in §4 is understood and mitigated
   by dropping to 320 when the editor is up, which costs geometry resolution.
   Closing the editor for the generation phase, or ordering all generation
   before the pipeline first touches Unreal, would buy back octree 384 — and the
   pipeline already generates before importing, so this is mostly about not
   leaving an editor open during a run.
4. **A/splats needs a UE plugin.** Unchanged. Installing one is a project
   change, so ask first.
5. **`mcp__unreal-engine__*` correlationId mismatch** — unchanged; use `uemcp`.

---

## 7. Test suite

`536 passed / 70 failed`, and **the same 70 fail on a clean tree** — verified by
stashing every change and re-running. They are the pre-existing
`torch`-is-an-empty-namespace problem in Python312, not a regression. Three files
also fail collection for the same reason and must be `--ignore`d.

New: `tests/test_asset_generation_plan.py` covers what gets generated and from
which pixels, without touching a GPU.

---

## 8. Working habits that paid off, again

- **Measure, don't eyeball.** The axis mapping, the up-axis defect and the
  Rotator argument order were all found by comparing numbers that should have
  matched and didn't. None was visible by looking at a render.
- **Make the prediction falsifiable.** The probe measurement predicted yaw −90;
  the render *refuted* it, and the refutation is what exposed the real defect. A
  measurement that only confirms itself would have shipped the wrong fix.
- **Distrust green receipts.** Still true. This session added `PROVEN` receipts
  over a stale mesh and over primitives standing in for a failed import.
- **Report the artefact, not the exit code.** Every acceptance here checks the
  file that was supposed to be written.
