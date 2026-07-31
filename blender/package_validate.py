from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import bpy
from mathutils import Vector

from common import (
    argv_after_double_dash,
    configure_render,
    create_camera,
    export_fbx,
    export_glb,
    extended_mesh_stats,
    import_mesh,
    object_center,
    reset_scene,
    save_json,
    select_only,
    triangle_count,
    welded_topology_stats,
    world_bounds,
)


def bool_arg(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def copy_if_present(source: str | Path, target: Path) -> str | None:
    source_path = Path(source)
    if not source_path.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)
    return str(target)


def add_collision(objects: list[bpy.types.Object], mode: str) -> list[str]:
    collisions = []
    candidates = sorted(objects, key=triangle_count, reverse=True)
    if mode in {"convex", "capsule_hulls"}:
        candidates = candidates[: min(4, len(candidates))]
    elif mode == "per_major_part":
        total = max(sum(triangle_count(obj) for obj in objects), 1)
        candidates = [obj for obj in candidates if triangle_count(obj) / total >= 0.01][:16]
    elif mode in {"architectural", "room_shell", "per_object", "per_cell"}:
        candidates = candidates[: min(64, len(candidates))]
    for index, obj in enumerate(candidates):
        duplicate = obj.copy()
        duplicate.data = obj.data.copy()
        bpy.context.collection.objects.link(duplicate)
        duplicate.name = f"UCX_{obj.name}_{index:02d}"
        duplicate.hide_render = True
        select_only([duplicate])
        bpy.context.view_layer.objects.active = duplicate
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.convex_hull()
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
        collisions.append(duplicate.name)
    return collisions


def add_sockets(objects: list[bpy.types.Object]) -> list[str]:
    minimum, maximum = world_bounds(objects)
    center = (minimum + maximum) * 0.5
    locations = {
        "SOCKET_root": Vector((center.x, center.y, minimum.z)),
        "SOCKET_center": center,
        "SOCKET_top": Vector((center.x, center.y, maximum.z)),
    }
    names = []
    for name, location in locations.items():
        empty = bpy.data.objects.new(name, None)
        empty.empty_display_type = "PLAIN_AXES"
        empty.location = location
        bpy.context.collection.objects.link(empty)
        names.append(name)
    return names


def render_angle(objects, path: Path, direction: tuple[float, float, float], ortho: bool = True) -> None:
    minimum, maximum = world_bounds(objects)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    scale = max(extent.x, extent.y, extent.z, 1.0)
    location = center + Vector(direction).normalized() * scale * 2.6
    camera = create_camera(f"Camera_{path.stem}", tuple(location), scale * 1.45)
    if not ortho:
        camera.data.type = "PERSP"
        camera.data.lens = 55
    direction_to_center = center - camera.location
    camera.rotation_euler = direction_to_center.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    key_data = bpy.data.lights.new(f"Key_{path.stem}", "AREA")
    key_data.energy = 1000
    key_data.size = scale * 2
    key = bpy.data.objects.new(f"Key_{path.stem}", key_data)
    bpy.context.collection.objects.link(key)
    key.location = center + Vector((-1.5, -1.5, 2.5)) * scale
    fill_data = bpy.data.lights.new(f"Fill_{path.stem}", "AREA")
    fill_data.energy = 400
    fill_data.size = scale * 2
    fill = bpy.data.objects.new(f"Fill_{path.stem}", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = center + Vector((1.5, 1.0, 0.5)) * scale
    configure_render(768, 768, transparent=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(camera, do_unlink=True)
    bpy.data.objects.remove(key, do_unlink=True)
    bpy.data.objects.remove(fill, do_unlink=True)


def export_spatial_cells(objects: list[bpy.types.Object], cells_dir: Path, divisions: int) -> dict:
    cells_dir.mkdir(parents=True, exist_ok=True)
    minimum, maximum = world_bounds(objects)
    extent = maximum - minimum
    divisions = max(1, int(divisions))
    step_x = max(extent.x / divisions, 1e-6)
    step_y = max(extent.y / divisions, 1e-6)
    groups: dict[tuple[int, int], list[bpy.types.Object]] = {}
    for obj in objects:
        center = object_center(obj)
        x = min(divisions - 1, max(0, int((center.x - minimum.x) / step_x)))
        y = min(divisions - 1, max(0, int((center.y - minimum.y) / step_y)))
        groups.setdefault((x, y), []).append(obj)
    cells = []
    for (x, y), group in sorted(groups.items()):
        path = cells_dir / f"cell_{x:02d}_{y:02d}.glb"
        select_only(group)
        export_glb(str(path), selected_only=True)
        cells.append(
            {
                "x": x,
                "y": y,
                "path": str(path),
                "objects": [obj.name for obj in group],
                "triangles": sum(triangle_count(obj) for obj in group),
            }
        )
    return {"divisions": divisions, "cells": cells}


def validate_clean(
    path: Path,
    target_min: int,
    target_max: int,
    budget_mode: str,
    export_strategy: str,
    asset_type: str,
) -> dict:
    reset_scene()
    objects = import_mesh(str(path))
    stats = extended_mesh_stats(objects)
    materials = {slot.material for obj in objects for slot in obj.material_slots if slot.material}
    texture_nodes = []
    missing_textures = []
    for material in materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not node.image:
                continue
            texture_nodes.append(f"{material.name}:{node.image.name}")
            filepath = Path(bpy.path.abspath(node.image.filepath)) if node.image.filepath else None
            if not node.image.packed_file and filepath and not filepath.is_file():
                missing_textures.append(str(filepath))
    zero_normals = 0
    sampled_vertices = 0
    for obj in objects:
        # bpy_prop_collection rejects a slice with a step, so sample by index instead.
        vertices = obj.data.vertices
        step = max(1, len(vertices) // 10000)
        for index in range(0, len(vertices), step):
            normal = vertices[index].normal
            sampled_vertices += 1
            if normal.length < 1e-8 or not all(math.isfinite(float(value)) for value in normal):
                zero_normals += 1
    triangles = stats["triangles"]
    topology = welded_topology_stats(objects)
    armatures = [obj.name for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    actions = sorted(action.name for action in bpy.data.actions)
    animated_required = export_strategy == "animated_human_avatar"
    within_tolerance = True if budget_mode == "per_object" else target_min * 0.75 <= triangles <= target_max * 1.25
    errors = []
    if not objects:
        errors.append("No meshes after clean import")
    if stats["uv_layers"] <= 0:
        errors.append("No UV layers after clean import")
    if not materials:
        errors.append("No materials after clean import")
    if not texture_nodes:
        errors.append("No image textures linked from materials")
    if missing_textures:
        errors.append("Missing external texture references")
    if not stats["finite_bounds"]:
        errors.append("Non-finite bounds")
    if zero_normals:
        errors.append("Invalid sampled vertex normals")
    if not within_tolerance:
        errors.append("Triangle count is outside the allowed tolerance")
    subject_types = {"avatar", "character", "creature"}
    if asset_type in subject_types and topology["components"] > 8:
        errors.append(
            f"Subject contains {topology['components']} welded connected components; expected at most 8"
        )
    boundary_limit = max(256, int(max(triangles, 1) * 0.01))
    if asset_type in subject_types and topology["boundary_edges"] > boundary_limit:
        errors.append(
            f"Subject contains {topology['boundary_edges']} welded boundary edges; limit is {boundary_limit}"
        )
    if animated_required and not armatures:
        errors.append("Animated avatar contains no armature after clean GLB import")
    if animated_required and not actions:
        errors.append("Animated avatar contains no animation actions after clean GLB import")
    if animated_required and not any("dance" in name.lower() for name in actions):
        errors.append("Animated avatar contains no dance action")
    return {
        "success": not errors,
        "stats": stats,
        "material_count": len(materials),
        "texture_node_count": len(texture_nodes),
        "texture_nodes": texture_nodes,
        "missing_textures": missing_textures,
        "sampled_vertices": sampled_vertices,
        "invalid_sampled_normals": zero_normals,
        "target_min": target_min,
        "budget_mode": budget_mode,
        "target_max": target_max,
        "triangle_count_within_tolerance": within_tolerance,
        "welded_topology": topology,
        "armatures": armatures,
        "animation_actions": actions,
        "animation_required": animated_required,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--high", required=True)
    parser.add_argument("--lod1", default="")
    parser.add_argument("--lod2", default="")
    parser.add_argument("--maps-dir", required=True)
    parser.add_argument("--parts-manifest", required=True)
    parser.add_argument("--uv-layout", required=True)
    parser.add_argument("--stage-reports-dir", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--pipeline-report", required=True)
    parser.add_argument("--asset-type", required=True)
    parser.add_argument("--target-min", type=int, required=True)
    parser.add_argument("--target-max", type=int, required=True)
    parser.add_argument("--budget-mode", choices=("total", "per_object"), default="total")
    parser.add_argument("--collision-mode", default="convex")
    parser.add_argument("--spatial-chunking", default="false")
    parser.add_argument("--cell-divisions", type=int, default=4)
    parser.add_argument("--export-strategy", default="static_asset")
    args = parser.parse_args(argv_after_double_dash())

    root = Path(args.output_root)
    source_dir = root / "source"
    meshes_dir = root / "meshes"
    textures_dir = root / "textures"
    previews_dir = root / "previews"
    reports_dir = root / "reports"
    cells_dir = root / "cells"
    for directory in (source_dir, meshes_dir, textures_dir, previews_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copy_if_present(args.high, source_dir / "original.glb")
    copy_if_present(args.high, meshes_dir / "high.glb")
    copy_if_present(args.lod1, meshes_dir / "lod1.glb")
    copy_if_present(args.lod2, meshes_dir / "lod2.glb")
    copy_if_present(args.parts_manifest, reports_dir / "parts.json")
    copy_if_present(args.uv_layout, previews_dir / "uv_layout.png")
    for source in Path(args.maps_dir).glob("*.png"):
        copy_if_present(source, textures_dir / source.name)
    if args.stage_reports_dir and Path(args.stage_reports_dir).is_dir():
        for source in Path(args.stage_reports_dir).glob("*.json"):
            copy_if_present(source, reports_dir / source.name)

    reset_scene()
    objects = import_mesh(args.input)
    collisions = add_collision(objects, args.collision_mode)
    sockets = add_sockets(objects)
    selected = [obj for obj in bpy.context.scene.objects if obj.type in {"MESH", "ARMATURE", "EMPTY"}]
    select_only(selected)
    final_glb = meshes_dir / "game_ready.glb"
    final_fbx = meshes_dir / "game_ready.fbx"
    candidate_glb = meshes_dir / ".game_ready.candidate.glb"
    candidate_fbx = meshes_dir / ".game_ready.candidate.fbx"
    candidate_glb.unlink(missing_ok=True)
    candidate_fbx.unlink(missing_ok=True)
    export_glb(str(candidate_glb), selected_only=True)
    export_fbx(str(candidate_fbx), selected_only=True)
    render_angle(objects, previews_dir / "front.png", (0, -1, 0))
    render_angle(objects, previews_dir / "side.png", (1, 0, 0))
    render_angle(objects, previews_dir / "rear.png", (0, 1, 0))
    render_angle(objects, previews_dir / "perspective.png", (1.6, -2.0, 1.2), ortho=False)
    chunks = export_spatial_cells(objects, cells_dir, args.cell_divisions) if bool_arg(args.spatial_chunking) else {"divisions": 0, "cells": []}
    if chunks["cells"]:
        save_json(reports_dir / "cells.json", chunks)
    save_json(
        reports_dir / "scene_manifest.json",
        {
            "asset_type": args.asset_type,
            "export_strategy": args.export_strategy,
            "objects": [
                {"name": obj.name, "triangles": triangle_count(obj), "center": list(object_center(obj))}
                for obj in objects
            ],
            "spatial_cells": chunks,
        },
    )

    validation = validate_clean(
        candidate_glb,
        args.target_min,
        args.target_max,
        args.budget_mode,
        args.export_strategy,
        args.asset_type,
    )
    if validation["success"]:
        os.replace(candidate_glb, final_glb)
        os.replace(candidate_fbx, final_fbx)
    else:
        candidate_glb.unlink(missing_ok=True)
        candidate_fbx.unlink(missing_ok=True)
    validation.update(
        {
            "asset_type": args.asset_type,
            "primary_glb": str(final_glb) if validation["success"] else None,
            "fbx": str(final_fbx) if validation["success"] else None,
            "candidate_promoted": validation["success"],
            "collisions": collisions,
            "sockets": sockets,
            "cell_count": len(chunks["cells"]),
            "export_strategy": args.export_strategy,
        }
    )
    save_json(args.validation, validation)
    map_files = sorted(str(path) for path in textures_dir.glob("*.png"))
    parts_data = json.loads((reports_dir / "parts.json").read_text(encoding="utf-8")) if (reports_dir / "parts.json").is_file() else {}
    save_json(
        args.pipeline_report,
        {
            "success": validation["success"],
            "asset_type": args.asset_type,
            "output_root": str(root),
            "high_mesh": str(meshes_dir / "high.glb"),
            "game_ready_mesh": str(final_glb) if validation["success"] else None,
            "lod1": str(meshes_dir / "lod1.glb") if (meshes_dir / "lod1.glb").is_file() else None,
            "lod2": str(meshes_dir / "lod2.glb") if (meshes_dir / "lod2.glb").is_file() else None,
            "textures": map_files,
            "previews": sorted(str(path) for path in previews_dir.glob("*.png")),
            "part_count": parts_data.get("part_count", 0),
            "cell_count": len(chunks["cells"]),
            "export_strategy": args.export_strategy,
            "validation": validation,
            "manual_review_required": args.asset_type in {"avatar", "character", "creature"},
        },
    )
    if not validation["success"]:
        raise RuntimeError(f"Clean re-import validation failed: {validation['errors']}")


if __name__ == "__main__":
    main()
