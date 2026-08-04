"""Audit the reconstructed source shell before accepting any visual proof.

A manifest claiming a material was assigned is not evidence. This reads the
material actually bound to the mesh component in the loaded level, and reports
whether any visible slot falls back to an engine placeholder.

    python -m uemcp python @unreal/audit_source_shell.py --json
"""

import json

import unreal

SHELL_LABEL = "Castlegrounds_ReconstructedMesh"
PLACEHOLDER_MARKERS = ("worldgridmaterial", "defaultmaterial", "preview", "checker")


def vec(value) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
matches = [a for a in subsystem.get_all_level_actors()
           if str(a.get_actor_label()) == SHELL_LABEL]
if len(matches) != 1:
    raise RuntimeError(f"expected exactly one {SHELL_LABEL}, found {len(matches)}")

actor = matches[0]
component = actor.get_component_by_class(unreal.StaticMeshComponent)
if component is None:
    raise RuntimeError(f"{SHELL_LABEL} has no StaticMeshComponent")

mesh = component.get_editor_property("static_mesh")
transform = actor.get_actor_transform()
rotation = actor.get_actor_rotation()

audit = {
    "actor_path": str(actor.get_path_name()),
    "static_mesh": str(mesh.get_path_name()) if mesh else None,
    "actor_location": vec(transform.translation),
    "actor_rotation_pyr": [float(rotation.pitch), float(rotation.yaw), float(rotation.roll)],
    "actor_scale": vec(transform.scale3d),
    "component_relative_location": vec(component.get_editor_property("relative_location")),
    "component_relative_scale": vec(component.get_editor_property("relative_scale3d")),
    "visible": bool(component.get_editor_property("visible")),
    "hidden_in_game": bool(component.get_editor_property("hidden_in_game")),
    "slots": [],
}

try:
    origin, extent = actor.get_actor_bounds(False)
    audit["bounds_origin"] = vec(origin)
    audit["bounds_extent"] = vec(extent)
except Exception as exc:
    audit["bounds_error"] = str(exc)

if mesh is not None:
    try:
        audit["nanite_enabled"] = bool(
            mesh.get_editor_property("nanite_settings").get_editor_property("enabled"))
    except Exception:
        audit["nanite_enabled"] = None
    try:
        audit["num_lods"] = int(mesh.get_num_lods())
    except Exception:
        pass

for index in range(int(component.get_num_materials())):
    material = component.get_material(index)
    path = str(material.get_path_name()) if material else None
    slot = {"slot": index, "material": path, "textures": []}
    lowered = (path or "").lower()
    slot["placeholder"] = bool(
        path is None or any(marker in lowered for marker in PLACEHOLDER_MARKERS))

    if material is not None:
        try:
            for name in unreal.MaterialEditingLibrary.get_texture_parameter_names(material):
                texture = unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(
                    material, name)
                if texture is None:
                    continue
                slot["textures"].append({
                    "parameter": str(name),
                    "path": str(texture.get_path_name()),
                    "width": int(texture.get_editor_property("blueprint_get_size_x")
                                 if hasattr(texture, "blueprint_get_size_x") else 0),
                    "srgb": bool(texture.get_editor_property("srgb")),
                })
        except Exception as exc:
            slot["texture_error"] = str(exc)

    audit["slots"].append(slot)

audit["placeholder_slots"] = [s["slot"] for s in audit["slots"] if s["placeholder"]]
audit["classification"] = "REJECTED" if audit["placeholder_slots"] or not audit["slots"] else "PROVEN"

result = json.dumps(audit)
