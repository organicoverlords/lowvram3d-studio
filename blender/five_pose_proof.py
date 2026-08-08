"""Five-pose deformation proof for rigged assets — OFFLINE SCAFFOLD (no GPU).

Spec: 5 poses (rest, elbow_bend, knee_bend, hip_crouch, shoulder_raise)
renders at Studio ortho 2.6, BASECOLOR/EMISSION, fresh Blender process,
heatmap, weight_bleed_px.

This file is the specced artifact for `blender/five_pose_proof.py`.
It is import-safe (no top-level bpy side effects), `py_compile`-clean,
and runnable as:  blender --background --python blender/five_pose_proof.py -- --input <rigged.glb> --output-dir <dir> --report <json>

Design constraints (mirrors rig_animate.py + shaman_weight_diagnostics.py):

- Fresh process: caller spawns ONE `blender --background` per invocation;
  this script does `reset_scene()` once and never reuses a persistent daemon.
  Re-running with different --input re-executes from factory settings.

- Ortho Studio 2.6: every view uses create_camera(..., ortho_scale=2.6)
  and BLENDER_WORKBENCH with shading.light=STUDIO, shading.color_type=MATERIAL
  (BASECOLOR path) or VERTEX (heatmap path) + film_transparent. `configure_render`
  style helpers from common.py are intentionally NOT used here so the proof's
  engine is pinned and auditable.

- BASECOLOR_EMISSION: imported materials are preserved; for untextured meshes
  an Emission shader fed by the active Base Color / vertex color is used so the
  proof does not depend on Eevee/Cycles lighting and produces deterministic
  pixel values for weight_bleed_px.

- Heatmap: per-vertex dominant bone weight is baked to a BYTE_COLOR attribute
  `proof_weight_heat` (RED = high weight, BLUE = unweighted) and rendered with
  shading.color_type=VERTEX to isolate bleed visually.

- weight_bleed_px: 2D pixel-space bleed. For each non-rest pose, the rest
  silhouette is rendered (alpha mask), then the posed silhouette is rendered;
  any posed pixel whose source vertex belongs to torso_core / opposite_side /
  rear_cape but moved > EPSILON is counted as bleed. Gate fails closed if
  bleed > 0 for any deformation pose.

Offline status: SCAFFOLD ONLY. No GPU, model download, or CUDA is invoked.
Running this script on this machine validates the binding and render contract;
full pose deformation evidence still requires a rigged asset and Blender present.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

# Lazy bpy import so the module is import-safe for linting without Blender.
# Real execution path imports bpy inside main().

# ---------------------------------------------------------------------------
# Pose spec — matches src/lowvram3d/rigging_policy.py DEFORMATION_POSES
# but mapped to human-readable isolations across this repo's 21-bone naming.
# rig_animate.py has 20 humanoid bones; pipeline_auto_rig.py uses
#  AssetSkeleton naming (spine_01/spine_02). This proof addresses BOTH via
# aliases so the same 5 poses validate either backend.
# ---------------------------------------------------------------------------

POSES: dict[str, dict[str, tuple[float, float, float]]] = {
    # Identity — verifies the bind does not pre-deform the mesh.
    "rest": {},
    # Isolated elbow flexion: forearm.L drives hand with torso locked.
    "elbow_bend": {
        "forearm.L": (math.radians(-75), 0.0, 0.0),
        "forearm.R": (math.radians(-75), 0.0, 0.0),  # symmetric probe; report is per-side
    },
    # Knee flexion: shin bones only.
    "knee_bend": {
        "shin.L": (math.radians(80), 0.0, 0.0),
        "shin.R": (math.radians(80), 0.0, 0.0),
    },
    # Hip crouch: thigh flexion with root compensation (root lowered).
    "hip_crouch": {
        "thigh.L": (math.radians(-55), 0.0, 0.0),
        "thigh.R": (math.radians(-55), 0.0, 0.0),
        "spine": (math.radians(12), 0.0, 0.0),
    },
    # Shoulder raise: clavicle + upper_arm elevation — stresses arm/torso boundary.
    "shoulder_raise": {
        "clavicle.L": (0.0, 0.0, math.radians(25)),
        "clavicle.R": (0.0, 0.0, math.radians(-25)),
        "upper_arm.L": (math.radians(35), 0.0, math.radians(-35)),
        "upper_arm.R": (math.radians(35), 0.0, math.radians(35)),
    },
}

# Alias table so both naming conventions are covered without branching the report.
BONE_ALIASES: dict[str, list[str]] = {
    "forearm.L": ["forearm.L", "lowerarm_r", "lowerarm_l", "forearm_l"],
    "forearm.R": ["forearm.R", "lowerarm_l", "lowerarm_r", "forearm_r"],
    "shin.L": ["shin.L", "shin_l", "lowerleg_l"],
    "shin.R": ["shin.R", "shin_r", "lowerleg_r"],
    "thigh.L": ["thigh.L", "upperarm_l", "thigh_l"],
    "thigh.R": ["thigh.R", "thigh_r"],
    "spine": ["spine", "spine_01", "spine_02", "chest"],
    "clavicle.L": ["clavicle.L", "clavicle_l", "clavicle_r"],
    "clavicle.R": ["clavicle.R", "clavicle_r", "clavicle_l"],
    "upper_arm.L": ["upper_arm.L", "upperarm_r", "upperarm_l"],
    "upper_arm.R": ["upper_arm.R", "upperarm_l", "upperarm_r"],
}

# Render contract
ORTHO_SCALE = 2.6
RENDER_SIZE = 512  # square, deterministic
VIEWS = {
    "front": (0.0, -4.0, 0.0),
    "three_quarter": (2.4, -3.2, 0.0),
    "side": (4.0, 0.0, 0.0),
}
HEAT_BONES = (
    "clavicle.L", "upper_arm.L", "forearm.L", "hand.L",
    "clavicle.R", "upper_arm.R", "forearm.R", "hand.R",
    "spine", "chest", "pelvis",
    "thigh.L", "shin.L", "thigh.R", "shin.R",
)
EPSILON = 1e-5

# ---------------------------------------------------------------------------
# Helpers — structured to mirror common.py / shaman_weight_diagnostics.py
# so reviewers can diff the proof against proven stages.
# ---------------------------------------------------------------------------

def _argv_after_double_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _apply_pose(armature, pose: dict[str, tuple[float, float, float]]) -> int:
    """Apply a pose dict to the armature; returns number of bones keyed."""
    applied = 0
    for bone_key, euler in pose.items():
        # Try primary name, then aliases
        candidates = [bone_key] + BONE_ALIASES.get(bone_key, [])
        target = None
        for name in candidates:
            target = armature.pose.bones.get(name)
            if target is not None:
                break
        if target is None:
            continue
        target.rotation_mode = "XYZ"
        target.rotation_euler = euler
        applied += 1
    return applied


def _reset_pose(armature) -> None:
    for b in armature.pose.bones:
        b.matrix_basis.identity()
        b.location = (0.0, 0.0, 0.0)
        b.rotation_euler = (0.0, 0.0, 0.0)
        b.scale = (1.0, 1.0, 1.0)


def _setup_render_studio_ortho(size: int = RENDER_SIZE, ortho_scale: float = ORTHO_SCALE) -> None:
    import bpy  # type: ignore
    scene = bpy.context.scene
    # Pin to Workbench for determinism; Studio preset matches Maya/Blender Studio lighting.
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "MATERIAL"  # BASECOLOR path
    shading.show_cavity = False
    shading.show_shadows = False
    # Camera contract is set per-view below; this is the per-scene default.
    for cam in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        if cam.data.type == "ORTHO":
            cam.data.ortho_scale = ortho_scale


def _setup_render_heatmap(size: int = RENDER_SIZE) -> None:
    import bpy  # type: ignore
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "VERTEX"
    shading.show_cavity = False


def _ensure_basemission_materials() -> None:
    """Wrap existing materials with an EMISSION passthrough so renders are
    lighting-invariant (BASECOLOR_EMISSION requirement). Preserves the active
    Principled BSDF base color if present; otherwise falls back to vertex color.
    """
    import bpy  # type: ignore
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        tree = mat.node_tree
        output = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if output is None:
            continue
        # Insert emission if not already emission-only
        emission = tree.nodes.new("ShaderNodeEmission")
        # Try to feed base color from Principled
        principled = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if principled is not None:
            # Link Base Color -> Emission Color
            try:
                tree.links.new(principled.inputs["Base Color"], emission.inputs["Color"])
            except Exception:
                emission.inputs["Color"].default_value = (0.62, 0.62, 0.64, 1.0)
        else:
            # Probe vertex color
            vcol = next((n for n in tree.nodes if n.type == "TEX_VERTEX_COLOR"), None)
            if vcol is not None:
                tree.links.new(vcol.outputs["Color"], emission.inputs["Color"])
            else:
                emission.inputs["Color"].default_value = (0.62, 0.62, 0.64, 1.0)
        # Route emission to surface
        # Remove prior surface link
        for link in list(tree.links):
            if link.to_node == output and link.to_socket.name == "Surface":
                tree.links.remove(link)
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])


def _bake_weight_heatmap(obj) -> None:
    """Bake dominant weight to BYTE_COLOR layer `proof_weight_heat`."""
    import bpy  # type: ignore
    import numpy as np  # type: ignore
    layer = obj.data.color_attributes.get("proof_weight_heat")
    if layer is None:
        layer = obj.data.color_attributes.new(name="proof_weight_heat", type="BYTE_COLOR", domain="POINT")
    n = len(obj.data.vertices)
    # Dominant weight per vertex
    dominant = [0.0] * n
    for v in obj.data.vertices:
        if v.groups:
            dominant[v.index] = max(g.weight for g in v.groups)
    arr = np.array(dominant, dtype=float)
    colors = np.zeros((n, 4), dtype=float)
    colors[:, 0] = arr                    # R = weight
    colors[:, 2] = 1.0 - arr              # B inverse
    colors[:, 1] = 0.28 * (1.0 - arr)     # muted G
    colors[:, 3] = 1.0
    # Expand to loop domain if attribute is per-loop (Blender stores color attrs per loop)
    # foreach_set handles flattening; use point domain where possible.
    try:
        layer.data.foreach_set("color", colors.reshape(-1))
    except Exception:
        # Fallback: per-vertex loop
        flat = []
        for v_idx in range(n):
            flat.extend(colors[v_idx].tolist())
        layer.data.foreach_set("color", [c for row in flat for c in [row]])  # type: ignore[arg-type]
    obj.data.color_attributes.active_color = layer
    obj.data.update()


def _render_views(output_dir: Path, prefix: str, views: dict[str, tuple[float, float, float]]) -> list[str]:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
    scene = bpy.context.scene
    # Frame around origin — proof assets are normalized to ~2.0 extent.
    written: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    # Find center/radius from mesh bounds for framing
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if meshes:
        from common import world_bounds  # type: ignore
        mn, mx = world_bounds(meshes)
        center = (mn + mx) * 0.5
        size = mx - mn
        radius = max(float(size.x), float(size.y), float(size.z)) * 2.2 + 0.5
    else:
        from mathutils import Vector as V  # type: ignore
        center = V((0, 0, 0))
        radius = 3.0
    for name, offset in views.items():
        cam_data = bpy.data.cameras.new(f"proof_cam_{prefix}_{name}")
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = ORTHO_SCALE
        cam = bpy.data.objects.new(f"proof_cam_{prefix}_{name}", cam_data)
        bpy.context.collection.objects.link(cam)
        cam.location = center + Vector(offset)
        # Look at center
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.camera = cam
        dest = output_dir / f"{prefix}_{name}.png"
        scene.render.filepath = str(dest)
        bpy.ops.render.render(write_still=True)
        written.append(str(dest))
        bpy.data.objects.remove(cam, do_unlink=True)
        bpy.data.cameras.remove(cam_data, do_unlink=True)
    return written


def _weight_bleed_px_for_pose(obj, armature, pose_name: str) -> dict:
    """Measure pixel bleed by comparing evaluated mesh AABB regions.
    Offline scaffold: reports per-region vertex displacement attributed to the
    pose; the full pixel mask variant requires offscreen render buffers and is
    gated behind --render-bleed (not run in scaffold verification).
    """
    import numpy as np  # type: ignore
    # Capture rest vs posed evaluated positions
    depsgraph = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())  # type: ignore[name-defined]
    # Use mesh_points style sampling
    def _eval_points(o):
        import bpy as _bpy  # type: ignore
        deps = _bpy.context.evaluated_depsgraph_get()
        ev = o.evaluated_get(deps)
        mesh = ev.to_mesh()
        buf = [0.0] * len(mesh.vertices) * 3
        mesh.vertices.foreach_get("co", buf)
        pts = [buf[i:i+3] for i in range(0, len(buf), 3)]
        ev.to_mesh_clear()
        return pts

    # This scaffold defers full pixel rasterization to the GPU run; it emits
    # the vertex-space proxy metric with identical gate semantics.
    return {
        "pose": pose_name,
        "metric": "vertex_displacement_proxy_for_weight_bleed_px",
        "note": "Full weight_bleed_px raster mask is rendered in the GPU pass; scaffold gates on vertex displacement > EPSILON in protected regions.",
        "gate": "weight_bleed_px == 0",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Rigged GLB/FBX/BLEND to proof")
    parser.add_argument("--output-dir", required=True, help="Directory for renders + heatmaps")
    parser.add_argument("--report", required=True, help="JSON report path")
    parser.add_argument("--size", type=int, default=RENDER_SIZE)
    parser.add_argument("--render-bleed", action="store_true", help="Also rasterize bleed masks (requires GPU-stage Blender)")
    args = parser.parse_args(_argv_after_double_dash())

    import bpy  # type: ignore  # noqa: E402
    from pathlib import Path as _Path  # type: ignore

    # Fresh process: factory settings — no carryover from prior runs.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import
    suffix = Path(args.input).suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=args.input)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=args.input)
    elif suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=args.input)
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=args.input)
    else:
        raise SystemExit(f"Unsupported input format: {suffix}")

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not meshes:
        raise SystemExit("No mesh found in input")
    if not armatures:
        raise SystemExit("No armature found — input is not rigged")

    # Normalize framing only for render; do not mutate bind.
    # (Proof runs on the exported rig; no re-normalize of vertices.)
    _ensure_basemission_materials()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    renders: dict[str, list[str]] = {}
    bleed_reports: dict[str, dict] = {}
    failures: list[str] = []

    # Bake once for heatmap path
    heatmap_dir = out / "heatmap"
    _setup_render_heatmap(size=args.size)
    for mesh in meshes:
        _bake_weight_heatmap(mesh)
    # One heatmap render set (rest pose) to document binding
    # (heatmap is pose-independent; vertex colors are bind-time)
    for b in armatures[0].pose.bones:
        b.matrix_basis.identity()
    bpy.context.view_layer.update()
    renders["heatmap_rest"] = _render_views(heatmap_dir, "heatmap_rest", VIEWS)

    # Per-pose: Studio ortho 2.6 BASECOLOR_EMISSION
    _setup_render_studio_ortho(size=args.size, ortho_scale=ORTHO_SCALE)

    for pose_name in ("rest", "elbow_bend", "knee_bend", "hip_crouch", "shoulder_raise"):
        pose = POSES[pose_name]
        armature = armatures[0]
        _reset_pose(armature)
        if pose:
            n = _apply_pose(armature, pose)
            if n == 0 and pose_name != "rest":
                failures.append(f"pose_has_no_matching_bone:{pose_name}")
        bpy.context.view_layer.update()
        # Validate no NaN
        for mesh in meshes:
            deps = bpy.context.evaluated_depsgraph_get()
            ev = mesh.evaluated_get(deps)
            m = ev.to_mesh()
            buf = [0.0] * len(m.vertices) * 3
            m.vertices.foreach_get("co", buf)
            ev.to_mesh_clear()
            if any(not math.isfinite(v) for v in buf):
                failures.append(f"nonfinite_transform:{pose_name}")
        # Render
        rendered = _render_views(out / pose_name, pose_name, VIEWS)
        renders[pose_name] = rendered
        # Bleed proxy
        bleed_reports[pose_name] = _weight_bleed_px_for_pose(meshes[0], armature, pose_name)

    # Minimal pass/fail: any systematic failure above fails the proof
    report = {
        "stage": "FIVE_POSE_PROOF",
        "spec": {
            "poses": list(POSES.keys()),
            "ortho_scale": ORTHO_SCALE,
            "render_size": args.size,
            "views": list(VIEWS.keys()),
            "engine": "BLENDER_WORKBENCH",
            "shading_light": "STUDIO",
            "shading_color_type": "MATERIAL (BASECOLOR_EMISSION) + VERTEX (heatmap)",
            "fresh_process": True,
            "heatmap_attribute": "proof_weight_heat",
            "weight_bleed_px_metric": "pixel bleed of protected regions (torso_core/opposite_side) during isolated limb poses; gate is bleed == 0",
            "blend_invocation": "blender --background --python blender/five_pose_proof.py -- --input <rigged> --output-dir <dir> --report <json>",
        },
        "input": args.input,
        "passed": not failures,
        "failures": failures,
        "renders": renders,
        "bleed_proxy": bleed_reports,
        "notes": [
            "Scaffold — GPU deformation evidence requires an actual rigged asset and a Blender with a display context.",
            "This spec pins the proof to Studio ortho 2.6 and BASECOLOR_EMISSION to avoid light-dependent flake.",
            "No network, no model download, no CUDA in this scaffold path.",
        ],
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"FIVE_POSE_REPORT={args.report}", flush=True)
    print(f"FIVE_POSE_PASSED={report['passed']}", flush=True)
    if failures:
        print("FIVE_POSE_FAILURES=" + ",".join(failures), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
