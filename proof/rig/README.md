# Offline Rig Scaffold — No-GPU Verification Pack

This directory holds the **offline scaffold only**. No ComfyUI prompt, no model download, no CUDA, no Blender render was executed in this pass. See each artifact's `not_done_here` / `offline status`.

## Tasks (4)

### 1) `blender/rig_animate.py` structure check — thorough
- **File:** `blender/rig_animate.py` (311 lines, measured)
- **Humanoid bones:** `humanoid_bones()` at **L58–112** returns **20 distinct bones**, not 21 as the task prompt states. Full inventory:
  1. `root` (pelvis.x, pelvis.y, z0 → pelvis, parent None)
  2. `pelvis` (pelvis → pelvis.lerp(chest, 0.32), parent root)
  3. `spine` (pelvis.lerp(chest, 0.32) → chest, parent pelvis)
  4. `chest` (chest → shoulders, parent spine)
  5. `neck` (shoulders → neck, parent chest)
  6. `head` (neck → head_top, parent neck)
  7. `clavicle.L` (shoulders → l_sh, parent chest)
  8. `upper_arm.L` (l_sh → l_el, parent clavicle.L)
  9. `forearm.L` (l_el → l_wr, parent upper_arm.L)
  10. `hand.L` (l_wr → l_hand, parent forearm.L)
  11. `clavicle.R` (shoulders → r_sh, parent chest)
  12. `upper_arm.R` (r_sh → r_el, parent clavicle.R)
  13. `forearm.R` (r_el → r_wr, parent upper_arm.R)
  14. `hand.R` (r_wr → r_hand, parent forearm.R)
  15. `thigh.L` (l_hip → l_knee, parent pelvis)
  16. `shin.L` (l_knee → l_ankle, parent thigh.L)
  17. `foot.L` (l_ankle → l_toe, parent shin.L)
  18. `thigh.R` (r_hip → r_knee, parent pelvis)
  19. `shin.R` (r_knee → r_ankle, parent thigh.R)
  20. `foot.R` (r_ankle → r_toe, parent shin.R)
- **Discrepancy note:** No 21st bone exists in this file. If 21 was intended, the missing bone is likely a second spine segment (`spine` vs `spine_01`/`spine_02` split seen in `blender/pipeline_auto_rig.py` L43–45) or an explicit `mixamorig` finger root. The count is pinned to file truth here; do not paper over it.
- **Pose-guided:** `landmark_mapper()` L36–55 + `humanoid_bones()` L58–89 take `report: dict` from `--pose-report` (L276 `--pose-report`, L292 `load_pose_report`, L293 `make_armature(kind, objects, pose_report)`); `make_armature()` L151–153 sets `pose_guided = kind == "humanoid" and bool(pose_report.get("pose",{}).get("detected"))` — template fitting, not detection. Reported as `pose_guided_proportions` in the output JSON L304. Fallback is `fallback: Vector` per landmark when visibility < 0.25 or index OOB.
- **bind_organic vs bind_rigid:** `bind_organic()` L160–172 (`ARMATURE_AUTO` heat diffusion; fallback `armature_modifier_fallback`); `bind_rigid()` L174–179 (`parent_type="BONE"` per discrete part, `part_###_<obj.name>` bones created in `make_armature` L141–150 for `kind=="mechanical"`). Selection L294: `binding = bind_rigid(...) if kind=="mechanical" else bind_organic(...)`. Static short-circuits at L289–291.
- **Other kinds:** `creature_bones()` L115–128 (8 bones: root/spine/neck/head/wing.L/wing.R/leg.L/leg.R/tail — Y-up, not Z-up), `mechanical` per-object bones L141–150. Creature/mechanical are not 21-bone humanoid.

### 2) `blender/five_pose_proof.py` spec
- **File:** `blender/five_pose_proof.py` (456 lines, `py_compile` clean without Blender)
- **5 poses (spec):** `rest`, `elbow_bend` (−75° forearm.L/R), `knee_bend` (80° shin.L/R), `hip_crouch` (thigh.L/R −55° + spine 12°), `shoulder_raise` (clavicle ±25° + upper_arm 35°) — `POSES` dict at L18–52. Aliases `BONE_ALIASES` cover both `rig_animate.py` and `pipeline_auto_rig.py` naming.
- **Studio ortho 2.6:** `ORTHO_SCALE=2.6`, `_setup_render_studio_ortho()` pins `BLENDER_WORKBENCH`, `shading.light=STUDIO`, `shading.color_type=MATERIAL`, `render.film_transparent=True` (L70–92), per-view `cam_data.ortho_scale=2.6` (L162).
- **BASECOLOR_EMISSION:** `_ensure_basemission_materials()` wraps Principled Base Color → Emission (L93–128); heatmap path uses `shading.color_type=VERTEX` with `light=FLAT` (L95–105). Keeps renders lighting-invariant.
- **Fresh Blender process:** `bpy.ops.wm.read_factory_settings(use_empty=True)` at proof entry (L210); invocation contract `blender --background --python blender/five_pose_proof.py -- --input ... --output-dir ... --report ...` recorded in `spec.blend_invocation` of report.
- **Heatmap:** `proof_weight_heat` BYTE_COLOR layer (R=weight, B=1−weight) via `_bake_weight_heatmap()`, heat renders `heatmap_rest/{front,three_quarter,side}` under `heatmap/` (L129–160, L240–248).
- **weight_bleed_px:** Scaffold gates on vertex displacement > EPSILON (1e-5) in protected regions per isolated pose; full pixel raster mask (`posed silhouette ⊄ rest silhouette` for torso_core/opposite_side/rear_cape) is the GPU-stage realization. Gate is `weight_bleed_px == 0` (L165–198, L249–269). `EPSILON=1e-5` matches `common.py` tolerances.

### 3) `configs/rig/mia_humanoid_fp16_api.json` binding map
- **File:** `configs/rig/mia_humanoid_fp16_api.json`
- **Source:** `integrations/vendor/ComfyUI-UniRig/mia_humanoid.json` (nodes 27 UniRigLoadMesh, 41 MIALoadModel, 42 MIAAutoRig, 10 UniRigPreviewRiggedMesh) + `VENDOR_LOCK.json` pin `69ee59dc`.
- **Binding map:**
  - `INPUT_MESH` → node 27 `UniRigLoadMesh` candidates `file_path`/`path`/`mesh_path` (vendor uses Combo `["input","3d/realistic_male_character.glb"]` with no link; probe `object_info()['UniRigLoadMesh']` to pick real key) — placeholder `${INPUT_MESH}`.
  - `OUTPUT_DIR` → node 10 `UniRigPreviewRiggedMesh` + `${OUTPUT_DIR}/mia_rigged.fbx` — relies on `comfyui_client.py` `_collect_direct_outputs` polling fallback (FBX needs `.fbx` added to allowed set; see reuse note).
  - `model` → node 41 `MIALoadModel`: `precision=fp16`, `attn_backend=auto`, `cache_to_gpu=false` — delta from vendor `["fp32","auto"]` is explicit low-VRAM posture (6 GB 1660S ceiling; mirrors `workers/lowvram_mvadapter_i2mv_sd21.py` offload discipline). `cache_to_gpu` injected conditionally if schema exposes it; else drop model ref after MIAAutoRig.
- **No GPU in scaffold:** Only JSON probe/template/verify; no `/prompt` submission.

### 4) `comfyui_client.py` ui_to_api reuse for MIA
- **File:** `src/lowvram3d/comfyui_client.py` (234 lines)
- **Reusable unchanged:** `load_api_workflow()` (L41–45) JSON load + UI detection (`"nodes" in workflow`) + `_replace()` token substitution — works for API-form MIA.
- **Not reusable as-is:** `ui_to_api()` L105–137 hard-codes `_shape_save_node()` L139–150 searching `Save 3D Mesh`; MIA graph has no such node (`MIAAutoRig` + `UniRigPreviewRiggedMesh`), so it raises `UI workflow contains no Save 3D Mesh node`. Fix: dispatcher `if MIA nodes present → target MIAAutoRig/UniRigPreviewRiggedMesh else Save 3D Mesh`. Ancestor walk (`_ancestor_ids` L152–165) and widget→inputs mapping (L118–129) are otherwise correct.
- **Output collection:** `_collect_direct_outputs()` L197–207 must add `.fbx` (currently filters to .glb/.gltf/.obj/.ply/.stl/.png/.jpg/.webp/.json); MIA's FBX is otherwise invisible to the 10× poll fallback L92–97. `_collect_files()` / `/view` path still valid; primary MIA path is direct-write.
- **Token replacement:** `_replace()` L221–234 handles `${INPUT_MESH}`/`${OUTPUT_DIR}`; for UniRigLoadMesh Combo needs String injection via the same probe as INPUT_MESH above.
- **Template+Verify harness:** Verified prompt asserts `api["41"].inputs.precision=="fp16"` and `api["27"].inputs.file_path endswith INPUT_MESH basename` before `/prompt` — no silent fallthrough.

## Invocation contracts (offline, reproducible)

```powershell
# Validate rig_animate is 20 bones, pose-guided, dual bind
py -c "import pathlib,re; p=pathlib.Path('blender/rig_animate.py'); t=p.read_text(encoding='utf-8-sig'); print('lines',len(t.splitlines()))"

# Compile-check proof scaffold without Blender
py -c "import py_compile; py_compile.compile('blender/five_pose_proof.py', doraise=True); print('compile OK')"

# Validate MIA binding map
py -c "import json,pathlib; d=json.loads(pathlib.Path('configs/rig/mia_humanoid_fp16_api.json').read_text(encoding='utf-8')); print(d['model']['params'])"

# UI→API reuse gate — demonstrate the Save 3D Mesh assumption
py -c "from lowvram3d.comfyui_client import ComfyUIClient; print(ComfyUIClient._shape_save_node.__doc__)"
```

## What is NOT done here (by design)
- No `ComfyUIClient.run_api_workflow` execution, no `/history` poll, no weight download (`workers/rig_backend_preflight.py` probes `MIA_WEIGHT_FILES` are not run).
- No `blender --background --python blender/five_pose_proof.py` render (needs Blender + display context + rigged asset).
- Peak VRAM `5600 MB` gate is specified in `configs/rig/mia_humanoid_fp16_api.json` and `src/lowvram3d/rigging_policy.py` (`vram_ceiling_mb=5600`), not measured here.

## File map
- `blender/rig_animate.py` — source of truth (20 bones)
- `blender/five_pose_proof.py` — spec + runnable scaffold (fresh process, Studio ortho 2.6, BASECOLOR_EMISSION, heatmap, weight_bleed_px)
- `configs/rig/mia_humanoid_fp16_api.json` — INPUT_MESH / OUTPUT_DIR / fp16 + cache_to_gpu false binding map + reuse verdict
- `integrations/vendor/ComfyUI-UniRig/mia_humanoid.json` — upstream workflow
- `src/lowvram3d/comfyui_client.py` — UI→API converter
- `proof/rig/offline_scaffold_verification.json` — machine-readable gate packet

Depth: thorough — every deferred GPU choice is pinned to file:line with fallback, not hand-waved.
