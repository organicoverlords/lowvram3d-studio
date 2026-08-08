"""Render the inspection lineup headless, on the GPU.

    blender.exe --background evidence/deliverables/blender/ALL_ASSETS_inspection.blend \
        --python tools/render_inspection_scene.py -- --out lineup.png --samples 32

GPU on purpose. The card is idle whenever the pipeline is in a CPU stage, and
CPU rendering is what killed three paint runs by competing for cores and RAM
with work that was already running. Rendering on the GPU while paint holds the
CPU is the one arrangement where both fit on this machine at once.

Engine name is discovered rather than hardcoded: it has been BLENDER_EEVEE,
BLENDER_EEVEE_NEXT, and back to BLENDER_EEVEE across recent versions, and a
wrong literal fails with an enum error that says nothing about why.
"""

import argparse
import sys
from pathlib import Path

import bpy


def pick_engine(preferred):
    """First engine in `preferred` the running Blender actually offers."""
    available = [item.identifier for item in
                 bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    for name in preferred:
        if name in available:
            return name, available
    return available[0], available


def enable_gpu():
    """Turn on every GPU backend that reports a device, for Cycles.

    Returns the device names actually enabled so the caller can print them --
    "GPU rendering" that silently fell back to CPU is the failure this guards.
    """
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        return []
    settings = prefs.preferences
    enabled = []
    for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            settings.compute_device_type = backend
        except TypeError:
            continue
        settings.get_devices()
        found = [d for d in settings.devices if d.type == backend]
        if found:
            for device in settings.devices:
                device.use = device.type != "CPU"
            enabled = [d.name for d in found]
            break
    return enabled


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--width", type=int, default=3200)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--engine", default="eevee", choices=("eevee", "cycles"))
    args = parser.parse_args(argv)

    scene = bpy.context.scene
    if scene.camera is None:
        raise SystemExit("RENDER_ABORT: the blend has no active camera")

    preferred = (["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"] if args.engine == "eevee"
                 else ["CYCLES"])
    engine, available = pick_engine(preferred)
    scene.render.engine = engine
    print(f"engine   : {engine}   (available: {', '.join(available)})")

    if engine == "CYCLES":
        scene.cycles.device = "GPU"
        devices = enable_gpu()
        print(f"devices  : {', '.join(devices) if devices else 'NONE -- falling back to CPU'}")
        scene.cycles.samples = args.samples
    else:
        scene.eevee.taa_render_samples = args.samples

    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out)

    objects = len(scene.objects)
    placards = len([o for o in scene.objects if o.name.startswith("placard_")])
    ground = [o for o in scene.objects if o.name == "ground_grass"]
    sky = "yes" if scene.world and scene.world.use_nodes and any(
        n.bl_idname == "ShaderNodeTexSky" for n in scene.world.node_tree.nodes) else "NO"
    print(f"objects  : {objects}   placards: {placards}   "
          f"ground: {'yes' if ground else 'NO'}   sky: {sky}")

    bpy.ops.render.render(write_still=True)
    # Blender appends nothing for a still, but the path can gain an extension.
    final = out if out.exists() else out.with_suffix(".png")
    print(f"wrote    : {final}  "
          f"{final.stat().st_size / 1e6:.1f} MB" if final.exists() else
          "RENDER_ABORT: no file was written")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
