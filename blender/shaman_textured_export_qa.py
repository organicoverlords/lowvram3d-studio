"""Textured export with name, material, texture and animation parity gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash, reset_scene  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def survey() -> dict:
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.hide_render]
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    bones, sockets, deform = [], [], []
    for armature in armatures:
        for bone in armature.data.bones:
            bones.append(bone.name)
            if bone.use_deform:
                deform.append(bone.name)
            if bone.name.endswith(("_fx", "_socket")):
                sockets.append(bone.name)
    images = sorted({
        node.image.name
        for mesh in meshes
        for slot in mesh.material_slots
        if slot.material and slot.material.use_nodes
        for node in slot.material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image
    })
    return {
        "mesh_count": len(meshes),
        "armature_count": len(armatures),
        "vertices": sum(len(m.data.vertices) for m in meshes),
        "triangles": sum(
            sum(max(len(p.vertices) - 2, 0) for p in m.data.polygons) for m in meshes
        ),
        "bones": sorted(bones),
        "deform_bones": sorted(deform),
        "sockets": sorted(sockets),
        "actions": sorted(a.name for a in bpy.data.actions),
        "material_slots": sorted({
            slot.material.name for m in meshes for slot in m.material_slots if slot.material
        }),
        "material_count": len({
            slot.material.name for m in meshes for slot in m.material_slots if slot.material
        }),
        "uv_layers": sorted({layer.name for m in meshes for layer in m.data.uv_layers}),
        "images": images,
        "skinned_meshes": sum(
            1 for m in meshes if any(mod.type == "ARMATURE" for mod in m.modifiers)
        ),
        "animated_armatures": sum(
            1 for a in armatures if a.animation_data is not None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    glb = output / "shaman_textured_game_ready.glb"

    bpy.ops.wm.open_mainfile(filepath=args.input)
    source = survey()

    bpy.ops.export_scene.gltf(
        filepath=str(glb),
        export_format="GLB",
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_skins=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        use_visible=True,
        use_selection=False,
    )

    reset_scene()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    imported = survey()
    imported["bytes"] = glb.stat().st_size
    imported["sha256"] = sha256(glb)

    def parity(key: str) -> dict:
        expected, actual = set(source[key]), set(imported[key])
        return {
            "expected_count": len(expected),
            "actual_count": len(actual),
            "missing": sorted(expected - actual),
            "unexpected": sorted(actual - expected),
            "parity": expected == actual,
        }

    bone_parity = parity("bones")
    socket_parity = parity("sockets")
    action_parity = parity("actions")
    material_parity = parity("material_slots")
    uv_parity = parity("uv_layers")

    classifications = {
        "EXPORT_STRUCTURE_PROVEN": bool(
            imported["mesh_count"] >= 1
            and imported["armature_count"] >= 1
            and imported["skinned_meshes"] >= 1
            and imported["vertices"] > 0
        ),
        "EXPORT_NAME_PARITY_PROVEN": bool(bone_parity["parity"] and socket_parity["parity"]),
        "EXPORT_MATERIAL_PARITY_PROVEN": bool(
            material_parity["parity"] and imported["material_count"] == source["material_count"]
        ),
        "EXPORT_TEXTURE_BINDINGS_PROVEN": bool(
            len(imported["images"]) >= len(source["images"]) and len(imported["images"]) > 0
        ),
        "EXPORT_ANIMATION_PARITY_PROVEN": bool(action_parity["parity"]),
    }
    # Triangle count and mesh count must be part of the fidelity gate. Without
    # them EXPORT_FIDELITY_PROVEN can pass while the exporter silently emits an
    # extra mesh or changes topology.
    classifications["EXPORT_TOPOLOGY_PARITY_PROVEN"] = bool(
        imported["triangles"] == source["triangles"]
        and imported["mesh_count"] == source["mesh_count"]
    )
    classifications["EXPORT_FIDELITY_PROVEN"] = all(classifications.values())

    report = {
        "stage": "TEXTURED_EXPORT_QA",
        "export": str(glb),
        "classifications": classifications,
        "source": source,
        "fresh_import": imported,
        "parity": {
            "bones": bone_parity,
            "sockets": socket_parity,
            "actions": action_parity,
            "material_slots": material_parity,
            "uv_layers": uv_parity,
        },
        "vertex_expansion": {
            "source_vertices": source["vertices"],
            "imported_vertices": imported["vertices"],
            "ratio": imported["vertices"] / max(source["vertices"], 1),
            "explanation": (
                "glTF stores one vertex per unique position/normal/UV corner, so "
                "a welded mesh expands on reimport. Triangle count is the "
                "invariant to compare, not vertex count."
            ),
            "triangles_preserved": imported["triangles"] == source["triangles"],
        },
        "texture_status": "REJECTED_VISUAL_BASELINE_USED_FOR_MOTION_REVIEW_ONLY",
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, value in classifications.items():
        print(f"{name}={value}", flush=True)
    print(f"TRIANGLES_PRESERVED={report['vertex_expansion']['triangles_preserved']}", flush=True)
    print(f"TEXTURED_EXPORT={glb}", flush=True)


if __name__ == "__main__":
    main()
