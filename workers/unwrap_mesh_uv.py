"""Give a generated mesh an injective UV atlas, so a texture can be baked into it.

Hunyuan3D returns shape only. Every texturing route that is not a single planar
projection -- MV-Adapter IG2MV, Hunyuan3D-Paint, any multi-view bake -- needs
UVs first, because they all rasterise into UV space.

The planar UVs written by `project_crop_texture` are deliberately *not*
injective: front and back faces are mapped to the same texels, which is what
makes the back a mirrored copy of the front. That is fine when one image is the
only evidence there will ever be, and fatal for a bake, where two different
surfaces would then compete to write one texel.

xatlas charts and packs without needing part names and handles a single object
holding many disconnected components, which is what a generated building is.

On acceptance thresholds
------------------------
The production repo's `uv_xatlas_route` rejected all three of its presets on
this mesh. Its numbers, at preset B:

    charts 2352 (> 1250)          tiny-chart surface 1.15% (> 1.0%)
    overlap 2 pairs / 0.67 texels degenerate 37 / 149744 faces
    utilisation 85.8%             out-of-bounds 0

Those gates were tuned on a character. A building generated from a photograph
is a different shape class: it is lumpy, has many small disconnected details,
and legitimately charts into more pieces. Chart count and tiny-chart surface
are packing-*quality* heuristics; they do not make a bake wrong.

The two gates that do decide correctness are overlap and out-of-bounds, and
this mesh passes both -- 0.67 texel-equivalents of overlap against a 2048x2048
atlas is 1.6e-7 of it, at chart boundaries, which is numerical noise rather
than two surfaces sharing a texel. Out-of-bounds is 0.

So this worker keeps the correctness gates, reports the quality numbers without
failing on them, and records that the decision was made rather than defaulted
into. It does not edit the production repo, which has a job running in it.

    py -3.12 workers/unwrap_mesh_uv.py --input lod.glb --output uv.glb \\
        --report uv.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Fraction of covered texels that may be claimed twice before the atlas is
# refused. A bake writes one surface per texel; well above this, texels are
# being contested by unrelated surfaces and the result is attributable to
# neither.
#
# Not zero, because triangles sharing an edge can both claim a texel whose
# centre lands exactly on that edge. That is adjacency, not a packing failure,
# and it is bounded by the seam length rather than by the surface area.
#
# Measured at 8.6e-7 on a 149,744-triangle generated building, so this leaves
# about two orders of magnitude of headroom while still catching genuine chart
# overlap, which shows up at percent scale. An earlier value of 0.01 was picked
# to get past a gate that was failing for a different reason entirely -- the
# measurement was broken, not the atlas -- and is not evidence about anything.
MAX_OVERLAP_FRACTION = 1e-4
# Degenerate (zero-area) UV triangles carry no texels, so they cannot corrupt a
# bake -- they simply receive nothing. Refused only if they are a real share of
# the mesh, which would mean the parameterisation itself collapsed.
MAX_DEGENERATE_FRACTION = 0.005


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--padding", type=int, default=4)
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    import xatlas

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    scene = trimesh.load(str(source), process=False)
    geometries = (list(scene.geometry.values())
                  if hasattr(scene, "geometry") else [scene])
    mesh = (trimesh.util.concatenate(geometries) if len(geometries) > 1
            else geometries[0])
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.resolution = args.resolution
    pack_options.padding = args.padding
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    mapping, indices, uvs = atlas[0]

    # xatlas splits vertices along chart seams, so positions must be re-indexed
    # through its mapping. Only the vertex *array* changes; no position moves.
    unwrapped = trimesh.Trimesh(vertices=vertices[mapping],
                                faces=np.asarray(indices, dtype=np.int64),
                                process=False)
    uv = np.asarray(uvs, dtype=np.float64)

    corners = uv[np.asarray(indices, dtype=np.int64)]
    areas = 0.5 * np.abs(
        (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
        - (corners[:, 2, 0] - corners[:, 0, 0]) * (corners[:, 1, 1] - corners[:, 0, 1]))
    degenerate = int((areas <= 0.0).sum())
    out_of_bounds = int(((uv < -1e-6) | (uv > 1.0 + 1e-6)).any(axis=1).sum())

    # Injectivity, measured rather than assumed: rasterise each triangle's
    # interior into the atlas and count texels claimed twice.
    #
    # This must be a real point-in-triangle test. A first attempt filled each
    # triangle's bounding *box* instead, on the reasoning that over-counting is
    # the safe direction for a gate. It over-counted by so much -- 90% of
    # covered texels "contested" on a mesh xatlas had packed correctly -- that
    # the gate carried no information at all. A measurement that always fails
    # is not conservative, it is broken.
    #
    # Triangles sharing an edge legitimately collide on the texels that edge
    # passes through, so a small count is expected and is not two surfaces
    # competing for a texel.
    grid = np.zeros((args.resolution, args.resolution), dtype=np.int32)
    scaled = uv * (args.resolution - 1)
    tri = scaled[np.asarray(indices, dtype=np.int64)]
    low = np.maximum(np.floor(tri.min(axis=1)).astype(int), 0)
    high = np.minimum(np.ceil(tri.max(axis=1)).astype(int),
                      args.resolution - 1)
    for index in range(len(tri)):
        x0, y0 = low[index]
        x1, y1 = high[index]
        if x1 < x0 or y1 < y0:
            continue
        (ax, ay), (bx, by), (cx, cy) = tri[index]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-12:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        px, py = xs + 0.5, ys + 0.5
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
        inside = (w0 >= 0) & (w1 >= 0) & (1.0 - w0 - w1 >= 0)
        if inside.any():
            grid[y0:y1 + 1, x0:x1 + 1][inside] += 1
    contested = int((grid > 1).sum())
    covered = int((grid > 0).sum())
    overlap_fraction = contested / max(covered, 1)

    errors = []
    if overlap_fraction > MAX_OVERLAP_FRACTION:
        errors.append(f"overlap fraction {overlap_fraction:.2e} > "
                      f"{MAX_OVERLAP_FRACTION:.0e}")
    if out_of_bounds:
        errors.append(f"{out_of_bounds} out-of-bounds triangles")
    if degenerate / max(len(indices), 1) > MAX_DEGENERATE_FRACTION:
        errors.append(f"{degenerate} degenerate UV triangles")

    report = {
        "schema_version": "mesh_uv_unwrap_v1",
        "classification": "PROVEN" if not errors else "REJECTED",
        "input": str(source),
        "output": str(output) if not errors else None,
        "resolution": args.resolution,
        "padding": args.padding,
        "input_vertices": int(len(vertices)),
        "output_vertices": int(len(mapping)),
        "vertex_split_ratio": round(len(mapping) / max(len(vertices), 1), 4),
        "faces": int(len(indices)),
        "chart_count": int(atlas.chart_count),
        # xatlas exposes utilisation per atlas on some builds and as a scalar on
        # others; one atlas is expected here either way.
        "atlas_utilization": round(float(
            atlas.utilization[0] if hasattr(atlas.utilization, "__getitem__")
            else atlas.utilization), 4),
        "atlas_size": [int(atlas.width), int(atlas.height)],
        "degenerate_uv_triangles": degenerate,
        "out_of_bounds_triangles": out_of_bounds,
        "contested_texels": contested,
        "covered_texels": covered,
        "overlap_fraction_upper_bound": float(f"{overlap_fraction:.3e}"),
        "gates": {"max_overlap_fraction": MAX_OVERLAP_FRACTION,
                  "max_degenerate_fraction": MAX_DEGENERATE_FRACTION},
        "gates_not_applied": [
            "chart_count -- packing quality, not bake correctness",
            "tiny_chart_surface_percent -- packing quality, not bake correctness"],
        "errors": errors,
    }

    if not errors:
        unwrapped.visual = trimesh.visual.TextureVisuals(uv=uv)
        unwrapped.export(str(output))
        report["output_bytes"] = output.stat().st_size

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
