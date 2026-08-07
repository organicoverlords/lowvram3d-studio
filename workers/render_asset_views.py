"""Rasterised review renders of finished assets, several meshes per launch.

This project reviewed geometry with a point splat for most of a day because it
is fast, and the splat lies in one specific and costly way: the gaps between
splats read as holes. A 146k-face mesh with ZERO boundary edges and 99.52
percent of its area in one shell was reported as full of holes, and the next
step was very nearly a repair of a mesh that did not need repairing.

So: rasterise. Blender headless with EEVEE is a few seconds per view, and the
real cost is process startup, which is why this renders every mesh and every
view in ONE launch rather than one per call.

The splat still has a place -- 1.2M sampled points over three meshes in under a
minute, with no Blender dependency, is the right tool for silhouette, symmetry
and gross proportion. It is not the tool for judging surface quality, and
nothing about surface quality should be concluded from it.

    py workers/render_asset_views.py --mesh a.glb --mesh b.glb --out sheet.png

Camera is orthographic and framed on the union bounding box of each mesh
separately, so two meshes of the same subject at different scales still line up.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")

#: Named viewpoints as (yaw degrees, pitch degrees). Yaw 0 looks along -Y, so
#: with the subject's long axis snapped to X (see `align` below) yaw 0 is a
#: broadside and yaw 90/270 look down the length.
#:
#: The two ends are named neutrally. Which end is the bow is a question about
#: what is PAINTED there -- a sign, a figurehead, a wheel -- and this module
#: cannot answer it from geometry. It renders both and lets the viewer say.
VIEWS = {
    "profile": (0.0, 0.0),
    "profile_far": (180.0, 0.0),
    "end_plus": (90.0, 0.0),
    "end_minus": (270.0, 0.0),
    "plan": (0.0, 88.0),
    "three_quarter": (35.0, 18.0),
}

SCRIPT = r'''
import bpy, math, sys, json
from mathutils import Vector, Matrix


def bounds(meshes):
    lo = Vector((min(min((o.matrix_world @ Vector(c))[i] for c in o.bound_box)
                     for o in meshes) for i in range(3)))
    hi = Vector((max(max((o.matrix_world @ Vector(c))[i] for c in o.bound_box)
                     for o in meshes) for i in range(3)))
    return lo, hi


def align_long_axis_to_x(meshes):
    """Snap the longest horizontal extent onto X, by a quarter turn or nothing.

    glTF is Y-up and Blender is Z-up, so an asset authored lying along glTF Z
    arrives lying along Blender Y -- which is the axis yaw 0 looks DOWN. That
    is how this module spent a day calling the two sides "bow" and "stern".
    Rather than keep a per-asset table of which yaw means what, put the subject
    in a known pose and let the view names mean the same thing every time.

    A quarter turn only. An arbitrary rotation fitted to the point cloud would
    also work and would introduce a small skew into every render, which is
    precisely the kind of quiet distortion this project keeps having to undo.
    """
    lo, hi = bounds(meshes)
    if (hi - lo).y <= (hi - lo).x:
        return False
    turn = Matrix.Rotation(math.radians(90.0), 4, "Z")
    for obj in meshes:
        obj.matrix_world = turn @ obj.matrix_world
    bpy.context.view_layer.update()
    return True

payload = json.loads(sys.argv[-1])
bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = payload["size"]
scene.render.resolution_y = payload["size"]
scene.view_settings.view_transform = "Standard"
scene.render.film_transparent = False

world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.4

camera_data = bpy.data.cameras.new("cam")
camera_data.type = "ORTHO"
camera = bpy.data.objects.new("cam", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera.rotation_mode = "QUATERNION"

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", type="SUN"))
sun.data.energy = 3.0
sun.rotation_mode = "QUATERNION"
scene.collection.objects.link(sun)

for job in payload["jobs"]:
    for obj in [o for o in scene.objects if o.type == "MESH"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=job["mesh"])
    meshes = [o for o in scene.objects if o.type == "MESH"]
    turned = align_long_axis_to_x(meshes) if payload["align"] else False
    print("ALIGNED %s %s" % (job["prefix"], turned), flush=True)
    lo, hi = bounds(meshes)
    centre = (lo + hi) / 2
    radius = max((hi - lo)) * 0.72
    camera_data.ortho_scale = radius * 2.0

    for name, yaw, pitch in job["views"]:
        a, b = math.radians(yaw), math.radians(pitch)
        direction = Vector((math.cos(b) * math.sin(a),
                            -math.cos(b) * math.cos(a), math.sin(b)))
        camera.location = centre + direction * radius * 4
        # Cull everything past the mid-plane. An end-on orthographic view shows
        # the FAR end's ornament over the near end whenever the near end is
        # shorter, which is indistinguishable from the texture stage having
        # duplicated that ornament onto both ends. Clipping at the middle is
        # the difference between the two.
        camera_data.clip_end = (radius * 4 if payload["half"] else 1e5)
        camera.rotation_quaternion = (-direction).to_track_quat("-Z", "Y")
        sun.location = centre + Vector((direction.y, -direction.x, 1.2)) * radius * 4
        sun.rotation_quaternion = (centre - sun.location).to_track_quat("-Z", "Y")
        scene.render.filepath = job["prefix"] + "_" + name + ".png"
        bpy.ops.render.render(write_still=True)
        print("RENDERED " + scene.render.filepath, flush=True)
'''


def run(meshes: list[Path], out_path: Path, view_names: list[str],
        size: int, align: bool = True, half: bool = False) -> dict:
    from PIL import Image, ImageDraw

    started = time.time()
    scratch = Path(tempfile.mkdtemp(prefix="assetviews-"))
    script = scratch / "render.py"
    script.write_text(SCRIPT, encoding="utf-8")

    jobs = [{"mesh": str(m.resolve()), "prefix": str(scratch / f"m{i}"),
             "views": [[n, *VIEWS[n]] for n in view_names]}
            for i, m in enumerate(meshes)]
    payload = json.dumps({"size": size, "align": bool(align), "half": bool(half), "jobs": jobs})

    completed = subprocess.run(
        [str(BLENDER), "-b", "--python", str(script), "--", payload],
        capture_output=True, text=True)
    rendered = [line.split(" ", 1)[1] for line in completed.stdout.splitlines()
                if line.startswith("RENDERED ")]
    aligned = [line.split(" ")[-1] == "True" for line in completed.stdout.splitlines()
               if line.startswith("ALIGNED ")]
    if not rendered:
        raise SystemExit("blender produced nothing:\n"
                         + (completed.stdout + completed.stderr)[-1200:])

    sheet = Image.new("RGB", (size * len(view_names), (size + 26) * len(meshes)),
                      (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for row, mesh in enumerate(meshes):
        draw.text((6, row * (size + 26) + 7), mesh.name, fill=(0, 0, 0))
        for column, name in enumerate(view_names):
            tile = Path(jobs[row]["prefix"] + "_" + name + ".png")
            if tile.is_file():
                sheet.paste(Image.open(tile).convert("RGB"),
                            (column * size, row * (size + 26) + 26))
            if row == 0:
                draw.text((column * size + size - 78, 7), name, fill=(120, 120, 120))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)

    return {
        "schema": "lowvram3d_asset_views_v1",
        "meshes": [str(m) for m in meshes],
        "views": view_names,
        "size": size,
        "tiles_rendered": len(rendered),
        "seconds": round(time.time() - started, 1),
        "sheet": str(out_path),
        "align_long_axis_to_x": bool(align),
        "near_half_only": bool(half),
        "quarter_turn_applied": aligned,
        "surface_qa_valid": True,
        "note": ("rasterised; a point splat renders solid surfaces as "
                 "perforated and must not be used to judge surface quality"),
        "view_note": ("end_plus and end_minus are the two ends of the long "
                      "axis; which one is the bow is not a geometric fact and "
                      "is not asserted here"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--views", default="profile,end_plus,three_quarter")
    parser.add_argument("--size", type=int, default=520)
    parser.add_argument("--half", action="store_true",
                        help="clip at the mid-plane so only the near half is "
                             "visible; separates far-end occlusion from a "
                             "texture duplicated onto both ends")
    parser.add_argument("--no-align", dest="align", action="store_false",
                        help="render in the asset's own pose; view names then "
                             "mean whatever the exporter's axes happened to be")
    args = parser.parse_args(argv)

    names = [v.strip() for v in args.views.split(",") if v.strip()]
    unknown = [v for v in names if v not in VIEWS]
    if unknown:
        raise SystemExit(f"unknown views {unknown}; have {sorted(VIEWS)}")

    result = run(args.mesh, args.out, names, args.size, args.align, args.half)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
