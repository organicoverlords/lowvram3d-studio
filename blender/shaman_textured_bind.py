"""Stage 2: bind the cleaned textured mesh to the already-proven armature.

The proven armature and its skinned rig-base mesh are never rebuilt. The
textured LOD0 is a different topology (136k welded verts, UV unwrapped) so
weights are transferred by nearest-polygon interpolation from the proven bind,
then limited and normalised to the same budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash  # noqa: E402

MAX_INFLUENCES = 4


def bounds(obj) -> np.ndarray:
    count = len(obj.data.vertices)
    buffer = np.empty(count * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", buffer)
    points = buffer.reshape(-1, 3)
    matrix = np.array(obj.matrix_world.to_4x4())
    world = (matrix @ np.hstack([points, np.ones((count, 1))]).T).T[:, :3]
    return np.vstack([world.min(axis=0), world.max(axis=0)])


def weight_matrix(obj, names: list[str]) -> np.ndarray:
    lookup = {group.name: group.index for group in obj.vertex_groups}
    matrix = np.zeros((len(obj.data.vertices), len(names)), dtype=np.float64)
    columns = {lookup[name]: position for position, name in enumerate(names) if name in lookup}
    for vertex in obj.data.vertices:
        for element in vertex.groups:
            column = columns.get(element.group)
            if column is not None:
                matrix[vertex.index, column] = element.weight
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rigged-blend", required=True)
    parser.add_argument("--textured-glb", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())

    bpy.ops.wm.open_mainfile(filepath=args.rigged_blend)
    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    source = next(obj for obj in bpy.data.objects if obj.type == "MESH")
    source_groups = [group.name for group in source.vertex_groups]
    source_bounds = bounds(source)

    before_objects = {obj.name for obj in bpy.data.objects}
    bpy.ops.import_scene.gltf(filepath=args.textured_glb)
    imported = [
        obj for obj in bpy.data.objects
        if obj.name not in before_objects and obj.type == "MESH"
    ]
    if len(imported) != 1:
        raise RuntimeError(f"expected one imported textured mesh, found {len(imported)}")
    target = imported[0]
    target.name = "shaman_textured_skinned"

    # glTF import can arrive parented under an empty; bake to world space so the
    # transfer runs in the same frame as the proven bind.
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    target.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    target_bounds = bounds(target)
    alignment_delta = float(np.abs(source_bounds - target_bounds).max())

    materials = [slot.material.name for slot in target.material_slots if slot.material]
    uv_layers = [layer.name for layer in target.data.uv_layers]
    images = sorted({
        node.image.name
        for slot in target.material_slots
        if slot.material and slot.material.use_nodes
        for node in slot.material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image
    })

    for name in source_groups:
        if name not in target.vertex_groups:
            target.vertex_groups.new(name=name)

    modifier = target.modifiers.new(name="WeightTransfer", type="DATA_TRANSFER")
    modifier.object = source
    modifier.use_vert_data = True
    modifier.data_types_verts = {"VGROUP_WEIGHTS"}
    modifier.vert_mapping = "POLYINTERP_NEAREST"
    modifier.layers_vgroup_select_src = "ALL"
    modifier.layers_vgroup_select_dst = "NAME"
    bpy.ops.object.modifier_apply(modifier=modifier.name)

    bpy.ops.object.vertex_group_limit_total(limit=MAX_INFLUENCES)
    bpy.ops.object.vertex_group_normalize_all(lock_active=False)

    matrix = weight_matrix(target, source_groups)
    sums = matrix.sum(axis=1)
    influences = (matrix > 0.0).sum(axis=1)
    unweighted = int((sums <= 1e-6).sum())

    target.parent = armature
    armature_modifier = target.modifiers.new(name="Armature", type="ARMATURE")
    armature_modifier.object = armature

    # Keep the proven rig-base mesh in the file as the authoritative bind, but
    # out of the way of textured review renders.
    source.hide_render = True
    source.hide_viewport = True

    bones = sorted(bone.name for bone in armature.data.bones)
    deform_bones = sorted(bone.name for bone in armature.data.bones if bone.use_deform)
    sockets = sorted(b for b in bones if b.endswith(("_fx", "_socket")))
    actions = sorted(action.name for action in bpy.data.actions)

    failures = []
    if unweighted:
        failures.append("TEXTURED_UNWEIGHTED_VERTICES")
    if not np.isfinite(matrix).all():
        failures.append("NONFINITE_TRANSFERRED_WEIGHTS")
    if influences.max(initial=0) > MAX_INFLUENCES:
        failures.append("TOO_MANY_INFLUENCES")
    weighted = sums > 1e-6
    if weighted.any() and (np.abs(sums[weighted] - 1.0) > 1e-3).any():
        failures.append("WEIGHT_SUM_OUT_OF_RANGE")
    if not materials:
        failures.append("MATERIAL_SLOTS_LOST")
    if not uv_layers:
        failures.append("UV_LAYERS_LOST")
    if not images:
        failures.append("TEXTURE_BINDINGS_LOST")
    if alignment_delta > 0.05:
        failures.append("TEXTURED_MESH_NOT_ALIGNED_TO_RIG")

    report = {
        "stage": "TEXTURED_RIG_BIND",
        "passed": not failures,
        "failures": failures,
        "rigged_source": args.rigged_blend,
        "textured_source": args.textured_glb,
        "armature": armature.name,
        "armature_rebuilt": False,
        "proven_rig_base_preserved": True,
        "bone_count": len(bones),
        "deform_bone_count": len(deform_bones),
        "socket_count": len(sockets),
        "sockets": sockets,
        "actions": actions,
        "action_count": len(actions),
        "transfer_method": "DATA_TRANSFER_POLYINTERP_NEAREST",
        "alignment_delta": alignment_delta,
        "source_bounds": source_bounds.tolist(),
        "target_bounds": target_bounds.tolist(),
        "textured_mesh": {
            "name": target.name,
            "vertices": len(target.data.vertices),
            "polygons": len(target.data.polygons),
            "materials": materials,
            "uv_layers": uv_layers,
            "images": images,
        },
        "skin": {
            "unweighted_vertices": unweighted,
            "max_influences": int(influences.max(initial=0)),
            "mean_influences": float(influences.mean()),
            "weight_sum_min": float(sums[weighted].min()) if weighted.any() else None,
            "weight_sum_max": float(sums[weighted].max()) if weighted.any() else None,
        },
        "texture_status": "REJECTED_VISUAL_BASELINE_USED_FOR_MOTION_REVIEW_ONLY",
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"TEXTURED_VERTS={len(target.data.vertices)}", flush=True)
    print(f"ALIGNMENT_DELTA={alignment_delta:.6f}", flush=True)
    print(f"TEXTURED_UNWEIGHTED={unweighted}", flush=True)
    print(f"TEXTURED_MAX_INFLUENCES={int(influences.max(initial=0))}", flush=True)
    print(f"MATERIALS={materials} UV={uv_layers} IMAGES={len(images)}", flush=True)
    print(f"BONES={len(bones)} SOCKETS={len(sockets)} ACTIONS={len(actions)}", flush=True)

    if failures:
        print("TEXTURED_BIND_FAILED=" + ",".join(failures), flush=True)
        raise SystemExit(2)

    Path(args.output_blend).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=args.output_blend)
    print("TEXTURED_BIND_PASSED=true", flush=True)


if __name__ == "__main__":
    main()
