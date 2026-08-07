"""Trim the flat apron that overhangs the hull at the waterline. Nothing else.

Single-image reconstruction has no information about an object's underside, and
on this subject both the res-512 and res-1024 TRELLIS meshes close it off with a
flat horizontal plate that extends past the hull sides. Measured, in the bottom
2% of height with |normal . up| > 0.85:

    512    3,424 faces   2.34% of mesh   6.76% of area   2.50% of H thick
    1024  12,417 faces   4.17% of mesh   5.22% of area   2.47% of H thick

Identical thickness, so it is a sheet rather than a volume; and res-1024 spends
3.6x the faces on slightly less area, which is the sawtooth perimeter being
resolved into many small triangles instead of a few large lobes. That is the
geometry and the location. The mechanism -- the decoder closing the unobserved
underside off at the widest silhouette -- is the best explanation available and
fits every measurement, but it is inference, not something these numbers prove.

Why a dedicated worker rather than the existing cleanup gate: this mesh finally
has separated balusters and openwork valance, and every generic operation on
hand destroys exactly that. Voxel remesh at pitch 256 removed the mast. Laplacian
smoothing would round the sawteeth and soften the ornament in the same pass.
"Remove small components" does not apply at all -- the plate is attached to the
main shell, which is 99.98% of the area. So: one operation, on one geometrically
defined region, with an acceptance test that fails closed.

The region is defined by three conditions ANDed together, not by a threshold on
any one of them:

    within the bottom `band` of object height
    AND normal predominantly vertical
    AND centroid outside the hull's own footprint just above the band

The third is what keeps the hull's legitimate flat bottom. Only the overhang
goes.

    py workers/trim_waterline_apron.py --mesh in.glb --report
    py workers/trim_waterline_apron.py --mesh in.glb --out trimmed.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: Bottom fraction of object height searched for apron faces.
BAND = 0.03
#: Reference slab, as fractions of height, whose plan footprint defines "inside
#: the hull". Starts above the band so the apron cannot vote for itself.
REFERENCE = (0.04, 0.16)
#: |normal . up| above this counts as horizontal.
FLATNESS = 0.85
#: Plan-footprint raster resolution and how far outside it a face may sit before
#: it is called overhang. In units of the object's largest horizontal extent.
FOOTPRINT_GRID = 512
FOOTPRINT_MARGIN = 0.012


def load(path: Path):
    import trimesh
    scene = trimesh.load(path, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    return mesh


def footprint_mask(points_xz: np.ndarray, lo: np.ndarray, span: float,
                   margin: float):
    """Occupancy of the reference slab in plan, dilated by `margin`."""
    from scipy import ndimage
    grid = np.zeros((FOOTPRINT_GRID, FOOTPRINT_GRID), dtype=bool)
    cell = span / FOOTPRINT_GRID
    index = np.clip(((points_xz - lo) / cell).astype(int), 0,
                    FOOTPRINT_GRID - 1)
    grid[index[:, 0], index[:, 1]] = True
    # Close first: the slab is sampled at vertices, so its interior is a shell
    # of points, and an un-closed mask would call the hull's own centre
    # "outside".
    grid = ndimage.binary_closing(grid, np.ones((5, 5), bool))
    grid = ndimage.binary_fill_holes(grid)
    return ndimage.binary_dilation(
        grid, np.ones((3, 3), bool),
        iterations=max(1, int(round(margin / cell)))), cell


def select(mesh, up: int = 1) -> dict:
    vertices = mesh.vertices
    low, high = float(vertices[:, up].min()), float(vertices[:, up].max())
    height = high - low
    horizontal = [a for a in range(3) if a != up]

    centres = mesh.triangles_center
    normals = mesh.face_normals
    in_band = centres[:, up] < low + height * BAND
    flat = np.abs(normals[:, up]) > FLATNESS

    slab = ((vertices[:, up] >= low + height * REFERENCE[0]) &
            (vertices[:, up] <= low + height * REFERENCE[1]))
    reference = vertices[slab][:, horizontal]

    plan_lo = vertices[:, horizontal].min(axis=0)
    span = float(np.ptp(vertices[:, horizontal], axis=0).max()) * 1.02
    mask, cell = footprint_mask(reference, plan_lo, span,
                                FOOTPRINT_MARGIN * span)

    index = np.clip(((centres[:, horizontal] - plan_lo) / cell).astype(int),
                    0, FOOTPRINT_GRID - 1)
    outside = ~mask[index[:, 0], index[:, 1]]

    doomed = in_band & flat & outside
    return {
        "doomed": doomed,
        "in_band": in_band,
        "flat": flat,
        "outside": outside,
        "height": height,
        "low": low,
    }


def report(mesh, chosen: dict) -> dict:
    doomed = chosen["doomed"]
    return {
        "faces": int(len(mesh.faces)),
        "in_band": int(chosen["in_band"].sum()),
        "band_and_flat": int((chosen["in_band"] & chosen["flat"]).sum()),
        "selected": int(doomed.sum()),
        "selected_face_fraction": round(float(doomed.mean()), 5),
        "selected_area_fraction": round(
            float(mesh.area_faces[doomed].sum() / mesh.area), 5),
        "band": BAND,
        "reference_slab": list(REFERENCE),
        "flatness": FLATNESS,
    }


def run(mesh_path: Path, out_path: Path | None) -> dict:
    mesh = load(mesh_path)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    before = {
        "faces": int(len(mesh.faces)),
        "shells": int(mesh.body_count),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "bounds": np.round(mesh.bounds, 5).tolist(),
    }
    chosen = select(mesh)
    summary = report(mesh, chosen)

    result = {
        "schema": "lowvram3d_waterline_trim_v1",
        "mesh_in": str(mesh_path),
        "before": before,
        "selection": summary,
        "note": ("apron overhang only: bottom band AND vertical normal AND "
                 "outside the hull footprint measured above the band"),
    }
    if out_path is None:
        result["mode"] = "report-only"
        return result

    mesh.update_faces(~chosen["doomed"])
    mesh.remove_unreferenced_vertices()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    result["mode"] = "written"
    result["mesh_out"] = str(out_path)
    result["after"] = {
        "faces": int(len(mesh.faces)),
        "shells": int(mesh.body_count),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "bounds": np.round(mesh.bounds, 5).tolist(),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", action="store_true",
                        help="select and measure, write nothing")
    args = parser.parse_args(argv)

    result = run(args.mesh, None if args.report else args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
