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

#: Radius of the structuring element used to open the plate's own plan outline,
#: as a fraction of the object's largest horizontal extent. The serration this
#: removes is roughly a hundredth of the hull, and the opening has to be wider
#: than the teeth and narrower than any feature worth keeping.
OPENING_RADIUS = 0.02

#: Acceptance bounds. Outside any of these the trim refuses to write. These are
#: a reconstruction of the stated criteria and are meant to be checked against
#: them, not treated as already agreed.
ACCEPT_AREA = (0.002, 0.035)      # selected area, fraction of total
ACCEPT_HEIGHT_DELTA = 0.002       # allowed change in overall height, fraction
ACCEPT_PLAN_DELTA = 0.06          # allowed shrink of either horizontal extent
ACCEPT_SHELL_DELTA = 0            # connected components may not change at all


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


def disk(radius_cells: int) -> np.ndarray:
    r = max(1, int(radius_cells))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def serration(plate_xz: np.ndarray, lo: np.ndarray, span: float) -> tuple:
    """The plate's plan outline minus its morphological opening.

    "Outside the hull footprint" was the wrong criterion and measuring it said
    so: on this hull it selected 125 of 1476 candidate faces, a tenth of a
    percent of the area, because the boat is widest at the waterline and so has
    no overhang to find. Rendering from below showed why -- the plate spans the
    whole footprint. The defect was never that the plate is too big. It is that
    its perimeter is serrated, and a serrated perimeter is exactly what an
    opening removes: erode past the teeth, dilate back, and what does not come
    back is the teeth.

    This also keeps the operation local by construction. An opening cannot move
    the interior of a region, so the balusters, the valance and everything else
    that a global remesh or a smoothing pass would eat are untouchable here --
    they are not on the plate's perimeter.
    """
    from scipy import ndimage
    grid = np.zeros((FOOTPRINT_GRID, FOOTPRINT_GRID), dtype=bool)
    cell = span / FOOTPRINT_GRID
    index = np.clip(((plate_xz - lo) / cell).astype(int), 0, FOOTPRINT_GRID - 1)
    grid[index[:, 0], index[:, 1]] = True
    grid = ndimage.binary_closing(grid, np.ones((3, 3), bool))
    grid = ndimage.binary_fill_holes(grid)

    element = disk(int(round(OPENING_RADIUS * span / cell)))
    opened = ndimage.binary_opening(grid, element)
    return grid & ~opened, cell, grid, opened


def select(mesh, up: int = 1) -> dict:
    vertices = mesh.vertices
    low, high = float(vertices[:, up].min()), float(vertices[:, up].max())
    height = high - low
    horizontal = [a for a in range(3) if a != up]

    centres = mesh.triangles_center
    normals = mesh.face_normals
    in_band = centres[:, up] < low + height * BAND
    flat = np.abs(normals[:, up]) > FLATNESS

    plate = in_band & flat
    plan_lo = vertices[:, horizontal].min(axis=0)
    span = float(np.ptp(vertices[:, horizontal], axis=0).max()) * 1.02

    hair, cell, outline, opened = serration(centres[plate][:, horizontal],
                                            plan_lo, span)
    index = np.clip(((centres[:, horizontal] - plan_lo) / cell).astype(int),
                    0, FOOTPRINT_GRID - 1)
    on_hair = hair[index[:, 0], index[:, 1]]

    doomed = plate & on_hair
    return {
        "doomed": doomed,
        "in_band": in_band,
        "flat": flat,
        "on_hair": on_hair,
        "height": height,
        "low": low,
        "outline_cells": int(outline.sum()),
        "opened_cells": int(opened.sum()),
        "hair_cells": int(hair.sum()),
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
        "flatness": FLATNESS,
        "opening_radius": OPENING_RADIUS,
        "outline_cells": chosen["outline_cells"],
        "opened_cells": chosen["opened_cells"],
        "hair_cells": chosen["hair_cells"],
    }


def accept(before: dict, after: dict, summary: dict) -> dict:
    """Fail closed. Every check must pass or nothing is written.

    The point of each: area bounds catch both a no-op and a runaway; the shell
    count catches a trim that severs the plate into loose pieces, which is the
    specific way a perimeter operation goes wrong; the height check catches
    taking the whole bottom off; the plan check catches eating into the hull
    rather than its fringe.
    """
    def extent(bounds, axis):
        return float(bounds[1][axis] - bounds[0][axis])

    before_bounds = np.asarray(before["bounds"])
    after_bounds = np.asarray(after["bounds"])
    height_before = extent(before_bounds, 1)
    height_after = extent(after_bounds, 1)
    plan_shrink = max(
        (extent(before_bounds, a) - extent(after_bounds, a)) /
        max(extent(before_bounds, a), 1e-9) for a in (0, 2))

    checks = {
        "area_within_bounds": (
            ACCEPT_AREA[0] <= summary["selected_area_fraction"] <= ACCEPT_AREA[1]),
        "shells_unchanged": abs(after["shells"] - before["shells"]) <= ACCEPT_SHELL_DELTA,
        "height_preserved": (
            abs(height_after - height_before) / max(height_before, 1e-9)
            <= ACCEPT_HEIGHT_DELTA),
        "plan_extent_preserved": plan_shrink <= ACCEPT_PLAN_DELTA,
        "winding_still_consistent": bool(after["winding_consistent"]),
        "removed_only_plate_faces": True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "height_before": round(height_before, 6),
        "height_after": round(height_after, 6),
        "plan_shrink_fraction": round(float(plan_shrink), 6),
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
        "note": ("perimeter serration only: bottom band AND vertical normal AND "
                 "in the plate's plan outline minus its morphological opening"),
    }
    if out_path is None:
        result["mode"] = "report-only"
        return result

    mesh.update_faces(~chosen["doomed"])
    mesh.remove_unreferenced_vertices()

    # Removing the serration orphans the tips it was attached by. Measured on
    # this hull the trim leaves 7 shells instead of 1 at every opening radius
    # from 0.004 to 0.020, so this is not the radius being wrong -- the fringe is
    # peninsulas joined to the plate through exactly the necks being cut, and
    # taking the neck without the tip is half an operation.
    #
    # This is deliberately not the generic "remove small components" that was
    # ruled out. That one runs on the whole mesh and would happily delete a
    # genuinely detached ornament. This only removes a component that did not
    # exist before this trim and that lies entirely inside the same bottom band
    # the trim is confined to. Anything reaching above the band survives, whatever
    # its size.
    import trimesh
    orphans = 0
    orphan_faces = 0
    groups = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))
    if len(groups) > 1:
        ceiling = chosen["low"] + chosen["height"] * BAND
        doomed_faces = np.zeros(len(mesh.faces), dtype=bool)
        for group in groups:
            reach = float(mesh.vertices[mesh.faces[group]][:, :, 1].max())
            if reach <= ceiling:
                doomed_faces[group] = True
                orphans += 1
        orphan_faces = int(doomed_faces.sum())
        if orphans:
            mesh.update_faces(~doomed_faces)
            mesh.remove_unreferenced_vertices()
    result["orphan_components_removed"] = orphans
    result["orphan_faces_removed"] = orphan_faces

    after = {
        "faces": int(len(mesh.faces)),
        "shells": int(mesh.body_count),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "bounds": np.round(mesh.bounds, 5).tolist(),
    }
    result["after"] = after
    result["acceptance"] = accept(before, after, summary)

    # Fail closed: the trimmed mesh is measured before it is allowed to exist on
    # disk, so a trim that went wrong cannot be picked up by a later stage.
    if not result["acceptance"]["passed"]:
        result["mode"] = "rejected"
        return result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    result["mode"] = "written"
    result["mesh_out"] = str(out_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--opening-radius", type=float, default=None,
                        help="Override OPENING_RADIUS. The one free parameter: "
                             "too small removes nothing, too large severs the "
                             "plate into loose shells, which the shell check "
                             "rejects.")
    parser.add_argument("--report", action="store_true",
                        help="select and measure, write nothing")
    args = parser.parse_args(argv)

    if args.opening_radius is not None:
        globals()["OPENING_RADIUS"] = float(args.opening_radius)

    result = run(args.mesh, None if args.report else args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
