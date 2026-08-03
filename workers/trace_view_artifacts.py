"""Separate geometry artifacts from texture-only artifacts inside a screen region.

A dark speckle near the feet can be two very different things: stray geometry floating in
front of the body, or a black patch in the atlas painted onto perfectly good surface. The
first needs a local mesh repair, the second needs the texel rejected during projection, and
guessing wrong makes the asset worse either way.

The distinction is decided from the bundle's own triangle-ID and depth buffers: geometry
artifacts form connected components that are detached from the body and stand off its depth
surface, while texture artifacts sit on triangles that are part of the main component at
the expected depth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mesh_io import read_glb, triangle_components
from render_control_bundle_texture import base_colour_image, file_prefix, sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--semantic", required=True,
                        help="semantic label of the view to trace, resolved via the contract")
    parser.add_argument("--region", nargs=4, type=float, required=True,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="fractional screen box, 0-1, origin top-left")
    parser.add_argument("--darkness", type=float, default=0.14,
                        help="sampled luminance below this counts as a dark texel")
    parser.add_argument("--report", required=True)
    parser.add_argument("--overlay", required=True)
    args = parser.parse_args()

    bundle = Path(args.bundle)
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))
    matches = [view for view in contract["views"]
               if str(view.get("proven_semantic") or view["semantic_name"]) == args.semantic]
    if len(matches) != 1:
        raise RuntimeError(f"TRACE_VIEW_SEMANTIC_AMBIGUOUS:{args.semantic}")
    view = matches[0]
    prefix = file_prefix(view)

    ids = np.load(bundle / f"{prefix}_triangle_ids.npy")
    depth = np.load(bundle / f"{prefix}_depth.npy")
    bary = np.load(bundle / f"{prefix}_barycentric.npy")
    size = ids.shape[0]

    positions, _normals, uv, tris = read_glb(Path(args.mesh))
    texture = np.asarray(base_colour_image(Path(args.mesh)))

    x0, y0, x1, y1 = args.region
    box = (int(x0 * size), int(y0 * size), int(x1 * size), int(y1 * size))
    region = np.zeros(ids.shape, bool)
    region[box[1]:box[3], box[0]:box[2]] = True
    visible = region & (ids >= 0)
    if not visible.any():
        raise RuntimeError("TRACE_VIEW_REGION_EMPTY")

    labels, _welded = triangle_components(positions, tris, 4e-4)
    sizes = np.bincount(labels)
    main_label = int(np.argmax(sizes))

    face_ids = ids[visible]
    component = labels[face_ids]
    detached = component != main_label

    # Depth stand-off: how far a pixel sits in front of the median depth of its neighbourhood.
    local = depth[visible]
    median_depth = float(np.median(local))
    standoff = median_depth - local

    pixel_uv = np.einsum("nc,ncd->nd", bary[visible], uv[tris[face_ids]])
    colour = sample(texture, pixel_uv).astype(np.float32) / 255.0
    luminance = colour @ np.array([0.2126, 0.7152, 0.0722])
    dark = luminance < args.darkness

    geometry_artifact = detached
    texture_artifact = dark & ~detached

    detached_components = sorted(set(int(c) for c in component[detached]))
    per_component = []
    for cid in detached_components:
        member = np.flatnonzero(labels == cid)
        vertices = np.unique(tris[member])
        per_component.append({
            "component_id": cid,
            "triangle_count": int(sizes[cid]),
            "pixels_in_region": int((component == cid).sum()),
            "bounds_min": positions[vertices].min(axis=0).tolist(),
            "bounds_max": positions[vertices].max(axis=0).tolist(),
            "face_ids": member.tolist() if sizes[cid] <= 512 else "OVER_512_NOT_LISTED",
        })

    overlay = np.zeros(ids.shape + (3,), np.uint8)
    seen = ids >= 0
    overlay[seen] = (60, 60, 66)
    canvas_dark = np.zeros(ids.shape, bool)
    canvas_detached = np.zeros(ids.shape, bool)
    canvas_dark[visible] = texture_artifact
    canvas_detached[visible] = geometry_artifact
    overlay[canvas_dark] = (255, 210, 0)
    overlay[canvas_detached] = (255, 40, 40)
    Image.fromarray(overlay).resize((size * 2, size * 2), Image.NEAREST).save(args.overlay)

    report = {
        "schema": "view_artifact_trace_v1",
        "mesh": str(args.mesh),
        "bundle": str(bundle),
        "semantic": args.semantic,
        "control_file_prefix": prefix,
        "region_pixels_xyxy": list(box),
        "region_visible_pixels": int(visible.sum()),
        "main_component_id": main_label,
        "mesh_component_count": int(len(sizes)),
        "geometry_artifact_pixels": int(geometry_artifact.sum()),
        "texture_artifact_pixels": int(texture_artifact.sum()),
        "detached_components_in_region": per_component,
        "depth": {
            "median": median_depth,
            "max_standoff_toward_camera": float(standoff.max()),
            "pixels_standing_off_over_5pct_of_span": int((standoff > 0.05).sum()),
        },
        "verdict": (
            "GEOMETRY_ARTIFACT_PRESENT" if geometry_artifact.any()
            else "TEXTURE_ONLY_ARTIFACT" if texture_artifact.any()
            else "NO_ARTIFACT_FOUND"),
        "action": (
            "remove the listed detached components locally" if geometry_artifact.any()
            else "reject the dark texels during projection; the surface itself is sound"
            if texture_artifact.any() else "none"),
        "overlay": str(args.overlay),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"VIEW_ARTIFACT_TRACE {report['verdict']} geometry={report['geometry_artifact_pixels']} "
          f"texture={report['texture_artifact_pixels']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
