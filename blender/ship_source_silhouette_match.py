"""Find the view of a candidate mesh that best reproduces the source matte silhouette.

The eight-view proof renders a fixed rig, which is the right thing for comparing candidates to each
other but is unfair to a generator that reconstructs into its own camera frame: a boat that came out
tilted looks like a shapeless block from every axis view, and gets rejected for its pose rather than
its geometry.

This searches yaw/pitch (and an image-plane roll, which is the same thing as rolling the camera) for
the view whose silhouette best matches the conditioning matte, and reports the intersection-over-
union of that match. Both silhouettes are cropped to their tight bounding box and resampled to a
common grid first, so the score measures shape agreement and not scale or framing.

A high best-IoU means the generator did reconstruct the subject and only the pose is off. A low
best-IoU from every direction means there is no view in which the mesh looks like the ship, which is
a geometry verdict rather than an orientation one.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
import numpy as np

from common import argv_after_double_dash, import_mesh, reset_scene, world_bounds
from shaman_texture_review import place_camera, setup_world

GRID = 128


def load_alpha(path: Path) -> np.ndarray:
    """Alpha channel of a PNG as a bool mask, via Blender's loader (no PIL in bpy)."""
    image = bpy.data.images.load(str(path))
    try:
        width, height = image.size
        pixels = np.array(image.pixels[:], np.float32).reshape(height, width, 4)
        return pixels[..., 3] > 0.5
    finally:
        bpy.data.images.remove(image)


def normalise(mask: np.ndarray) -> np.ndarray | None:
    """Crop to the tight bounding box and nearest-resample to a fixed square grid."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    cropped = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    ys = (np.linspace(0, cropped.shape[0] - 1, GRID)).round().astype(int)
    xs = (np.linspace(0, cropped.shape[1] - 1, GRID)).round().astype(int)
    return cropped[np.ix_(ys, xs)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def rotate(mask: np.ndarray, degrees: float) -> np.ndarray:
    """Nearest-neighbour image-plane rotation about the centre."""
    if degrees == 0.0:
        return mask
    size = mask.shape[0]
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    sx = cos * (xx - centre) + sin * (yy - centre) + centre
    sy = -sin * (xx - centre) + cos * (yy - centre) + centre
    sx = np.clip(sx.round().astype(int), 0, size - 1)
    sy = np.clip(sy.round().astype(int), 0, size - 1)
    return mask[sy, sx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--source-matte", required=True, help="RGBA conditioning image")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--yaw-step", type=float, default=30.0)
    parser.add_argument("--keep-best", type=int, default=4)
    args = parser.parse_args(argv_after_double_dash())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = normalise(load_alpha(Path(args.source_matte)))
    if target is None:
        raise RuntimeError("SOURCE_MATTE_EMPTY")
    # The matte is stored top-down and Blender reads bottom-up; compare against both so a match is
    # not missed on a vertical flip convention alone.
    targets = {"as_is": target, "flipped": target[::-1]}

    reset_scene()
    objects = import_mesh(str(args.glb))
    if not objects:
        raise RuntimeError("FRESH_IMPORT_NO_MESH")
    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)

    setup_world()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    camera_data = bpy.data.cameras.new("silhouette_search_camera")
    camera = bpy.data.objects.new("silhouette_search_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    scratch = output_dir / "_probe.png"
    rolls = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]
    yaws = [y * args.yaw_step for y in range(int(round(360.0 / args.yaw_step)))]
    pitches = [-30.0, -15.0, 0.0, 15.0, 30.0]

    results = []
    for yaw in yaws:
        for pitch in pitches:
            place_camera(camera, centre, radius, yaw, pitch, 1.0, centre, -1.0)
            scene.render.filepath = str(scratch)
            bpy.ops.render.render(write_still=True)
            candidate = normalise(load_alpha(scratch))
            if candidate is None:
                continue
            best = max(
                ((iou(rotate(candidate, roll), reference), roll, key)
                 for roll in rolls for key, reference in targets.items()),
                key=lambda item: item[0],
            )
            results.append({"yaw": yaw, "pitch": pitch, "roll": best[1],
                            "target_orientation": best[2], "iou": round(best[0], 4)})
            print(f"SILHOUETTE_PROBE yaw={yaw:.0f} pitch={pitch:.0f} iou={best[0]:.4f}", flush=True)

    results.sort(key=lambda item: item["iou"], reverse=True)
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    kept = []
    for rank, entry in enumerate(results[:args.keep_best]):
        place_camera(camera, centre, radius, entry["yaw"], entry["pitch"], 1.0, centre, -1.0)
        path = output_dir / f"best_{rank + 1:02d}_yaw{int(entry['yaw'])}_pitch{int(entry['pitch'])}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        kept.append({**entry, "path": str(path)})
        print(f"SILHOUETTE_BEST rank={rank + 1} iou={entry['iou']} {path}", flush=True)

    if scratch.exists():
        scratch.unlink()

    report = {
        "schema": "ship_source_silhouette_match_v1",
        "glb": str(args.glb),
        "source_matte": str(args.source_matte),
        "grid": GRID,
        "probe_resolution": args.resolution,
        "views_probed": len(results),
        "best_iou": results[0]["iou"] if results else 0.0,
        "best_views": kept,
        "all_views": results,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"SILHOUETTE_MATCH_DONE best_iou={report['best_iou']} report={args.report}", flush=True)


if __name__ == "__main__":
    main()
