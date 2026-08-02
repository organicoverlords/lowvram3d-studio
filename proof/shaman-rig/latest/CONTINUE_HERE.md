# CONTINUE HERE — shaman rig / animation

## State

Milestone 0 (audit and truth packet) is **PROVEN**. Nothing is rigged yet.

Branch `magicmusic/parts-pose-materials-20260802`. Never work on main, never
merge, never force-push.

Canonical source (immutable, verified this run):

```
C:\AI\LowVRAM3D-benchmarks\pipeline-v2-validation\shaman_v2_validation\state\CLEAN\proven\shaman_v2_validation_stance_clean.glb
SHA256 2f712b49a88a39cb10fb08e6bfb08becef2025b153ce87103e86fde97dfb8c80
```

All generated artifacts live under:

```
C:\AI\LowVRAM3D-benchmarks\shaman-rig-animation\run-20260802-123123\
```

## The three facts that drive every later decision

1. **The source is corner-split.** 3,258,048 stored vertices for 1,086,965
   triangles (exactly 3.00 per triangle). Edges are not shared, so Blender's
   heat-diffusion automatic weights have nothing to diffuse along. Welding at
   1e-4 recovers 537,041 vertices, 1,074,083 faces, one dominant connected
   component at 98.95%, 348 boundary edges, 13 non-manifold edges.
   **Rigging must run on a welded rig base**, and that transformation must be
   gated and reported — not folded silently into "topology unchanged".

2. **The 1.65 model width is not arm span.** It is a horizontal antler bar at
   head height with hanging ornaments. Any landmark logic using raw per-slice
   min/max width will be wrong. Use the density-based body core
   (`dense_interval` in `blender/shaman_rig_audit.py`).

3. **The arms hang down at the sides**, resting against a ragged cloth cape.
   Automatic weights will very likely bleed between arm and torso. SKIN_QA has
   to prove they did not, with deformation poses — not assume it.

## Reproduce Milestone 0

```bash
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --factory-startup --python "C:\Users\Lauri\Desktop\lowvram3d-magicmusic-asset-systems\blender\shaman_rig_audit.py" -- --input "C:\AI\LowVRAM3D-benchmarks\pipeline-v2-validation\shaman_v2_validation\state\CLEAN\proven\shaman_v2_validation_stance_clean.glb" --report "<run>\audit\source-audit.json"
```

Preview renders (workbench, 5 views):

```bash
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --factory-startup --python "C:\Users\Lauri\Desktop\lowvram3d-magicmusic-asset-systems\blender\shaman_rig_preview.py" -- --input "<glb-or-blend>" --output-dir "<run>\preview" --prefix source
```

Contact sheet (needs an env with PIL — `envs\control` works, `asset-systems-test` does not):

```bash
"C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\envs\control\Scripts\python.exe" workers/shaman_contact_sheet.py --image A.png --label a --output sheet.png
```

## Milestone 1 result

- `RIG_BASE` **PROVEN** — `run-20260802-123123\rig-base\shaman_rig_base.blend`
- `STAFF_POLICY` **PROVEN** — fused staff cylinder, 9,845 verts
- `PARTS` **NOT PROVEN** — positional bands only, anatomy not established

## Fourth hard-won fact

**GLB is export-only.** Exporting the welded rig base and reimporting it
re-splits every vertex per corner (537,041 → ~3.2M) and silently undoes the
weld. Carry the mesh between rig stages as `.blend`.

## Next action

**Anatomical segmentation of a robed figure**, because SKIN is blocked on it.

The blocker is real, not a tuning problem: the character wears a full cape and
skirt, so the legs and lower torso are not visible to any purely geometric band
or density heuristic. `shaman_semantic_regions.json` reports
`safe_for_skinning: ["staff"]` and nothing else.

Options, in order of expected reliability:

1. **Interior-cavity analysis** — the skirt is a shell; ray-cast or voxelise to
   find the enclosed leg volumes rather than segmenting the outer surface.
2. **Symmetry-plane + medial-axis extraction** on the welded base to recover a
   skeleton from volume rather than from Z bands.
3. **User-supplied landmarks** — a handful of clicked positions (pelvis, knees,
   shoulders, elbows, wrists) would make the rig deterministic immediately, and
   is the cheapest path to a first animated proof.

Do **not** proceed to RIG/SKIN using the current band regions. They would bind
skirt cloth to `thigh_l`/`thigh_r` and fail the SKIN_QA gate "no major body
region moves with the wrong bone" — after burning a full skinning pass on
537,041 vertices.

## Do not

- Do not modify or move the canonical source.
- Do not promote `...\state\TEXTURE\proven\shaman_v2_validation_textured_lod0.glb`.
  Its directory says "proven"; it is a **rejected** visual baseline.
- Do not reuse the boolean staff-hole cutter approach.
- Do not claim TIER 2 finger rigging. The audit caps itself at TIER 1 and the
  right-hand density peak count (10) is cloth fringe noise, not fingers.
- Do not promote anything on an AI image judgement alone. Deterministic
  geometry / topology / skin / animation / fresh-import gates are authoritative.

## Textured review target (current)

Authoritative animation source stays `C:\AI\LowVRAM3D-benchmarks\shaman-rig-animation\run-20260802-123123\motion\shaman_animated.blend`.
Textured review target: `C:\AI\LowVRAM3D-benchmarks\shaman-rig-animation\run-20260802-123123	extured_rig\shaman_textured_rigged.blend`.
Review bundle: `C:\AI\LowVRAM3D-benchmarks\shaman-rig-animation\run-20260802-123123\shaman_textured_motion_review.zip`.

Rebuild the textured target with:

1. `blender/shaman_antler_debris_cleanup.py` (textured LOD0 -> cleaned GLB)
2. `blender/shaman_textured_bind.py` (cleaned GLB + animated blend -> textured rig)
3. `blender/shaman_textured_export_qa.py` (parity-gated GLB export)

## Next action

1. Milestone E - weight shift with both feet planted, prove pelvis transfer and
   robe response before any walk work resumes.
2. Fix export topology parity: the GLB emits 2 meshes against 1 and 80 extra
   triangles. Suspect the hidden rig-base mesh or an exporter split.
3. Milestone F - two-step contact test with explicit heel strike, foot flat,
   mid stance, heel off, toe off and swing phases.

Still outstanding from earlier instructions: PROFILE_DISCOVERY stage, the
CLOTH_PREP / CLOTH_SIM / CLOTH_QA stages, the motion visual evaluator with its
14 deterministic detectors, and the texture CUDA lane.
