"""Reduce a generated mesh to a triangle budget, using Blender headless.

Mini Turbo at octree 384 returns roughly 1.7 million triangles for a single
building. That is a marching-cubes artefact rather than detail -- the volume is
sampled on a 384^3 grid whatever the subject -- and it costs real time
downstream: importing one such mesh into Unreal takes over ten minutes and
outlives the editor bridge's handler timeout, and a scene of them is unusable.

Blender is used rather than a Python decimator because it is already a declared
dependency of this project and none of `fast_simplification`, `open3d` or
`pymeshlab` is installed in the pipeline interpreter. Its Decimate modifier in
COLLAPSE mode is quadric edge collapse, which preserves silhouette far better
than vertex clustering at these ratios.

The original GLB is never modified; the reduced mesh is written alongside it.

    py -3.12 workers/decimate_mesh.py --input in.glb --output out.glb \
        --target-triangles 150000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"

# Run inside Blender: import, decimate to a ratio, export. Written to a temp
# file rather than passed with --python-expr so quoting survives Windows.
BLENDER_SCRIPT = '''
import bpy, sys, json

argv = sys.argv[sys.argv.index("--") + 1:]
source, destination, ratio = argv[0], argv[1], float(argv[2])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=source)

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
before = sum(len(o.data.polygons) for o in meshes)
for obj in meshes:
    if ratio < 1.0:
        modifier = obj.modifiers.new(name="decimate", type="DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
after = sum(len(o.data.polygons) for o in bpy.context.scene.objects if o.type == "MESH")

bpy.ops.export_scene.gltf(filepath=destination, export_format="GLB")
print("DECIMATE_RESULT " + json.dumps({"before": before, "after": after}))
'''


def triangle_count(path: Path) -> int:
    import trimesh

    scene = trimesh.load(str(path), process=False)
    geometries = (list(scene.geometry.values())
                  if hasattr(scene, "geometry") else [scene])
    return sum(int(len(g.faces)) for g in geometries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-triangles", type=int, default=150_000)
    parser.add_argument("--blender", default=os.environ.get("BLENDER_PATH",
                                                            DEFAULT_BLENDER))
    parser.add_argument("--receipt", default="")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)

    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    receipt = {"schema_version": "mesh_decimation_v1", "input": str(source),
               "output": str(destination),
               "target_triangles": args.target_triangles}

    before = triangle_count(source)
    receipt["input_triangles"] = before
    if before <= args.target_triangles:
        # Already within budget: copy rather than round-trip through Blender,
        # which would re-tessellate for no benefit.
        destination.write_bytes(source.read_bytes())
        receipt.update({"classification": "PROVEN", "decimated": False,
                        "output_triangles": before, "ratio": 1.0,
                        "reason": "already within the triangle budget"})
    else:
        ratio = args.target_triangles / float(before)
        blender = Path(args.blender)
        if not blender.is_file():
            receipt.update({"classification": "UNAVAILABLE",
                            "reason": f"blender not found at {blender}"})
            _write(receipt, args, destination)
            return 1

        with tempfile.TemporaryDirectory() as work:
            script = Path(work) / "decimate.py"
            script.write_text(BLENDER_SCRIPT, encoding="utf-8")
            completed = subprocess.run(
                [str(blender), "--background", "--factory-startup",
                 "--python", str(script), "--",
                 str(source), str(destination), f"{ratio:.6f}"],
                capture_output=True, text=True, timeout=args.timeout)
        receipt["blender_exit_code"] = completed.returncode
        if not destination.is_file():
            receipt.update({"classification": "FAILED",
                            "reason": "blender produced no output",
                            "stderr_tail": (completed.stderr or "")[-800:],
                            "stdout_tail": (completed.stdout or "")[-800:]})
            _write(receipt, args, destination)
            return 1
        # Count the written file rather than trusting Blender's report: the
        # exporter can drop geometry the modifier kept.
        receipt.update({"classification": "PROVEN", "decimated": True,
                        "ratio": round(ratio, 6),
                        "output_triangles": triangle_count(destination)})

    receipt["output_bytes"] = destination.stat().st_size
    receipt["reduction"] = (
        round(1.0 - receipt["output_triangles"] / float(before), 4) if before else 0.0)
    _write(receipt, args, destination)
    return 0


def _write(receipt: dict, args, destination: Path) -> None:
    path = Path(args.receipt) if args.receipt else destination.with_suffix(".decimation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
