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

## Next action

Milestone 1 — build the welded rig base and the deterministic semantic-region
worker, then the fused staff policy.

Order that respects the findings above:

1. `RIG_BASE` — weld at 1e-4, gate: face count within tolerance, one dominant
   component preserved, no volume change beyond 0.5%, bounds unchanged.
2. `PARTS` — density-core segmentation for torso/pelvis/head/neck/arms/hands/
   legs/feet plus staff, antlers, ornaments, cloth. Fail closed on weak
   evidence; regions must carry confidence and mesh state.
3. `STAFF_POLICY` — `fused_staff_control` only. No cutter, no hole, no separate
   mesh. `separate_staff_candidate` stays an optional experiment that can never
   block or auto-promote.

## Do not

- Do not modify or move the canonical source.
- Do not promote `...\state\TEXTURE\proven\shaman_v2_validation_textured_lod0.glb`.
  Its directory says "proven"; it is a **rejected** visual baseline.
- Do not reuse the boolean staff-hole cutter approach.
- Do not claim TIER 2 finger rigging. The audit caps itself at TIER 1 and the
  right-hand density peak count (10) is cloth fringe noise, not fingers.
- Do not promote anything on an AI image judgement alone. Deterministic
  geometry / topology / skin / animation / fresh-import gates are authoritative.
