"""Raster texture route, stage 2: numpy/opencv UV-atlas projector.

Replaces the Cycles bake in blender/project_texture.py, which measured 600s+ on a moderately
dense mesh. This does the same visibility-weighted multi-view projection at roughly 1000x the
speed by rasterizing triangles directly into UV-atlas pixel space instead of round-tripping
through a renderer.

Key correctness rule, learned the hard way: only views whose view_metadata.json entry has
source_type in {"real", "generated"} may contribute real projected pixels. Mirrored/synthetic
fallback views (produced by make_fallback_views.py when only one photo exists) are barred from
semantic projection -- projecting a mirrored front view onto rear-facing polygons puts a
duplicated face on the back of the mesh. Unobserved polygons receive explicit neutral synthesis at
export, never a winning projection from a mirrored or wrapped front view.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

try:
    from .texture_contract import assert_atlas_dimensions, validate_requested_atlas_size
    from .projection_repair import (
        face_id_matches_within_radius,
        gated_sample_mask,
        rear_face_provenance_violations,
    )
except ImportError:  # direct worker execution
    from texture_contract import assert_atlas_dimensions, validate_requested_atlas_size
    from projection_repair import (
        face_id_matches_within_radius,
        gated_sample_mask,
        rear_face_provenance_violations,
    )

FACING_POWER = 3.0
FACING_MIN = 0.15
ALPHA_MIN = 0.35
HIGH_CONF = 0.45
# Unseen-surface donor constraints (see the fill section for why these are not just "nearest").
DONOR_NEIGHBOURS = 8
DONOR_MIN_NORMAL_DOT = 0.25
DONOR_MAX_RADIUS_FRACTION = 0.12
# Position-weld tolerance for donor component scoping (UV seams duplicate vertices).
WELD_TOLERANCE = 4e-4


def projection_triangle_gate(visibility: np.ndarray, facing: np.ndarray,
                             threshold: float = FACING_MIN) -> np.ndarray:
    """Return triangles eligible for source projection in one camera view.

    ``visibility`` is the precomputed depth/occlusion result for the view.  A
    triangle must pass both that result and the normal-facing test before any
    source pixel can be sampled or blended into the atlas.  Invalid values are
    rejected explicitly so a malformed visibility/normal array cannot turn
    into a permissive CUDA/NumPy truth test later in the route.
    """
    visibility = np.asarray(visibility, dtype=bool)
    facing = np.asarray(facing, dtype=np.float32)
    if visibility.ndim != 1 or facing.ndim != 1 or visibility.shape != facing.shape:
        raise ValueError(
            f"visibility/facing must be matching 1-D arrays, got "
            f"{visibility.shape} and {facing.shape}"
        )
    return visibility & np.isfinite(facing) & (facing > float(threshold))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--views-dir", required=True)
    parser.add_argument("--view-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas-size", type=int, default=1024)
    parser.add_argument("--progress", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--provenance", default="")
    parser.add_argument("--facial-mask", default="")
    parser.add_argument("--require-face-id", action="store_true")
    parser.add_argument("--face-id-radius", type=int, default=0)
    parser.add_argument("--neutral-fill-only", action="store_true")
    args = parser.parse_args()

    npz, viewdir, meta_path, outdir = Path(args.npz), Path(args.views_dir), Path(args.view_metadata), Path(args.output_dir)
    atlas_size = validate_requested_atlas_size(args.atlas_size)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    conf_by_view = {v["view"]: (v["source_type"], float(v["confidence"])) for v in meta["views"]}
    semantic_types = set(meta["policy"]["semantic_projection"])

    d = np.load(npz)
    verts, tris, uvs, normals = d["verts"], d["tris"], d["uvs"], d["normals"]
    view_names = [str(x) for x in d["view_names"]]
    view_locs, ortho = d["view_locs"], float(d["ortho_scale"])
    total_tris = len(tris)

    face_id_arrays: dict[str, np.ndarray] = {}
    for vname in view_names:
        key = f"face_id_{vname}"
        if key in d.files:
            face_id_arrays[vname] = np.asarray(d[key], dtype=np.int32)
        elif args.require_face_id:
            raise RuntimeError(f"FACE_ID_BUFFER_MISSING:{key}")

    usable = [(i, n) for i, n in enumerate(view_names)
              if conf_by_view.get(n, ("synthetic", 0.0))[0] in semantic_types
              and conf_by_view.get(n, ("synthetic", 0.0))[1] > 0.0]
    if not usable:
        raise RuntimeError("No semantic-projection views available (view_metadata.json barred all views)")

    best_conf = np.zeros((atlas_size, atlas_size), np.float32)
    best_rgb = np.zeros((atlas_size, atlas_size, 3), np.float32)
    best_view = np.full((atlas_size, atlas_size), -1, np.int16)
    # This is deliberately independent of UV-atlas occupancy.  A planar UV
    # layout can overlap front and rear polygons, so atlas pixels alone cannot
    # tell us which mesh polygon actually received a gated observation.
    triangle_observed = np.zeros(total_tris, dtype=bool)
    triangle_observed_sum = np.zeros((total_tris, 3), np.float64)
    triangle_observed_count = np.zeros(total_tris, dtype=np.int64)
    triangle_visible = np.zeros(total_tris, dtype=bool)
    triangle_front_facing = np.zeros(total_tris, dtype=bool)
    triangle_face_id_matched = np.zeros(total_tris, dtype=bool)
    triangle_mask_valid = np.zeros(total_tris, dtype=bool)
    triangle_winning_view = np.full(total_tris, -1, np.int16)
    triangle_winning_conf = np.zeros(total_tris, np.float32)
    triangle_winning_facial = np.zeros(total_tris, dtype=bool)
    gate_counts = {}

    facial_mask_by_view: dict[str, np.ndarray] = {}
    if args.facial_mask:
        facial = cv2.imread(str(args.facial_mask), cv2.IMREAD_GRAYSCALE)
        if facial is None:
            raise RuntimeError(f"FACIAL_MASK_UNREADABLE:{args.facial_mask}")
        facial_mask_by_view["front"] = facial > 0

    uv_px = uvs.copy()
    uv_px[..., 0] *= (atlas_size - 1)
    uv_px[..., 1] = (1.0 - uv_px[..., 1]) * (atlas_size - 1)

    island = np.zeros((atlas_size, atlas_size), np.uint8)
    cv2.fillPoly(island, [t.astype(np.int32) for t in uv_px], 255)
    island = cv2.dilate(island, np.ones((3, 3), np.uint8), iterations=2) > 0

    progress_path = Path(args.progress) if args.progress else None

    def write_progress(idx: int, processed: int, status: str = "running") -> None:
        if progress_path is None:
            return
        progress_path.write_text(json.dumps({
            "status": status, "view": idx, "total_views": max(len(usable), 1),
            "triangles_processed": int(processed), "triangles_total": int(total_tris),
            "coverage_percent": round(float((best_conf > 0).sum() / max(island.sum(), 1) * 100), 1),
            "elapsed_seconds": round(time.time() - t0, 1),
        }, indent=2), encoding="utf-8")

    for slot, (vi, vname) in enumerate(usable):
        src = cv2.imread(str(viewdir / f"{vname}.png"), cv2.IMREAD_UNCHANGED)
        if src is None:
            continue
        if src.shape[2] == 4:
            srgb = cv2.cvtColor(src[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
            salpha = src[:, :, 3].astype(np.float32) / 255.0
        else:
            srgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32)
            salpha = np.ones(src.shape[:2], np.float32)
        sh, sw = salpha.shape
        src_conf = conf_by_view[vname][1]
        face_id = face_id_arrays.get(vname)
        if args.require_face_id and face_id is None:
            raise RuntimeError(f"FACE_ID_BUFFER_MISSING:{vname}")
        if face_id is not None and face_id.shape != salpha.shape:
            raise RuntimeError(
                f"FACE_ID_DIMENSION_MISMATCH:{vname}:{face_id.shape}!={salpha.shape}"
            )
        facial = facial_mask_by_view.get(vname)
        if facial is not None and facial.shape != salpha.shape:
            raise RuntimeError(
                f"FACIAL_MASK_DIMENSION_MISMATCH:{vname}:{facial.shape}!={salpha.shape}"
            )

        cam = view_locs[vi]
        vdir = cam / (np.linalg.norm(cam) + 1e-9)
        vis = d[f"vis_{vname}"]
        facing = normals @ vdir
        projection_gate = projection_triangle_gate(vis, facing)
        triangle_visible |= vis
        triangle_front_facing |= np.isfinite(facing) & (facing > FACING_MIN)
        gate_counts[vname] = {
            "depth_visible_triangles": int(np.count_nonzero(vis)),
            "normal_facing_triangles": int(np.count_nonzero(facing > FACING_MIN)),
            "eligible_triangles": int(np.count_nonzero(projection_gate)),
            "face_id_buffer": bool(face_id is not None),
        }
        axis = int(np.argmax(np.abs(vdir)))
        ua, va = (0, 2) if axis == 1 else ((1, 2) if axis == 0 else (0, 1))
        flip_u = -1.0 if vdir[axis] > 0 else 1.0

        processed = 0
        for t in range(total_tris):
            if not projection_gate[t]:
                continue
            p = verts[tris[t]]
            a = uv_px[t]
            x_lo, y_lo = np.maximum(np.floor(a.min(0)).astype(int), 0)
            x_hi, y_hi = np.minimum(np.ceil(a.max(0)).astype(int), atlas_size - 1)
            if x_hi < x_lo or y_hi < y_lo:
                continue
            xs, ys = np.meshgrid(np.arange(x_lo, x_hi + 1), np.arange(y_lo, y_hi + 1))
            xs, ys = xs.ravel(), ys.ravel()
            px, py = xs + 0.5, ys + 0.5
            (x0, y0), (x1, y1), (x2, y2) = a
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-9:
                continue
            w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
            w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
            w2 = 1.0 - w0 - w1
            ins = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not ins.any():
                continue
            w0, w1, w2, xs2, ys2 = w0[ins], w1[ins], w2[ins], xs[ins], ys[ins]
            world = w0[:, None] * p[0] + w1[:, None] * p[1] + w2[:, None] * p[2]
            u = (world[:, ua] * flip_u) / ortho + 0.5
            v = 0.5 - world[:, va] / ortho
            inframe = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
            sx = np.clip((u * (sw - 1)).astype(int), 0, sw - 1)
            sy = np.clip((v * (sh - 1)).astype(int), 0, sh - 1)
            alpha = salpha[sy, sx]
            edge = np.clip(np.minimum.reduce([u, 1 - u, v, 1 - v]) * 10.0, 0.0, 1.0)
            conf = src_conf * (max(facing[t], 0.0) ** FACING_POWER) * alpha * edge
            source_mask_valid = inframe & (alpha > ALPHA_MIN)
            if face_id is not None:
                face_id_match = face_id_matches_within_radius(
                    face_id, sx, sy, int(t), args.face_id_radius
                )
            else:
                face_id_match = np.ones_like(source_mask_valid, dtype=bool)
            gated = gated_sample_mask(
                depth_visible=bool(vis[t]),
                facing_score=float(facing[t]),
                face_id_match=face_id_match,
                source_mask_valid=source_mask_valid,
                confidence=conf,
            )
            triangle_face_id_matched[t] |= bool(np.any(face_id_match & source_mask_valid))
            triangle_mask_valid[t] |= bool(np.any(source_mask_valid))
            conf = np.where(gated, conf, 0.0).astype(np.float32)
            if not (conf > 0).any():
                continue
            triangle_observed[t] = True
            win = conf > best_conf[ys2, xs2]
            if win.any():
                yi, xi = ys2[win], xs2[win]
                best_conf[yi, xi] = conf[win]
                best_rgb[yi, xi] = srgb[sy[win], sx[win]]
                best_view[yi, xi] = vi
                triangle_observed_sum[t] += srgb[sy[win], sx[win]].astype(np.float64).sum(axis=0)
                triangle_observed_count[t] += int(np.count_nonzero(win))
                winning_max = float(conf[win].max())
                if winning_max >= float(triangle_winning_conf[t]):
                    triangle_winning_conf[t] = winning_max
                    triangle_winning_view[t] = int(vi)
                    if facial is not None:
                        triangle_winning_facial[t] = bool(np.any(facial[sy[win], sx[win]]))
            processed += 1

        write_progress(slot + 1, processed)

    real_mask = best_conf > 0
    real_pct = float(real_mask.sum() / island.sum() * 100)

    colour = best_rgb.copy()
    mask = real_mask.copy()
    n_islands, island_labels = cv2.connectedComponents(island.astype(np.uint8), connectivity=8)

    # ---- unseen-surface fill, in 3D rather than in atlas space ----
    # Earlier revisions seeded uncovered UV charts from the *global* mean of every observed pixel.
    # On this subject the observed pixels are dominated by dark ghillie cloth, so unseen surfaces
    # collapsed to near-black and the side/rear read as unlit. Colour is instead inherited from the
    # nearest genuinely observed geometry in 3D, so the back of the head takes head colour and the
    # far side of the tail takes tail colour. Per-triangle flat colour is inherently low frequency,
    # so no recognisable feature (eye, muzzle, scope) can be transported onto an unseen surface.
    tri_id = np.full((atlas_size, atlas_size), -1, np.int32)
    for t in range(total_tris):
        a = uv_px[t]
        x_lo, y_lo = np.maximum(np.floor(a.min(0)).astype(int), 0)
        x_hi, y_hi = np.minimum(np.ceil(a.max(0)).astype(int), atlas_size - 1)
        if x_hi < x_lo or y_hi < y_lo:
            continue
        xs, ys = np.meshgrid(np.arange(x_lo, x_hi + 1), np.arange(y_lo, y_hi + 1))
        xs, ys = xs.ravel(), ys.ravel()
        px, py = xs + 0.5, ys + 0.5
        (x0, y0), (x1, y1), (x2, y2) = a
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9:
            continue
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
        w2 = 1.0 - w0 - w1
        ins = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if ins.any():
            tri_id[ys[ins], xs[ins]] = t

    # Use only colours won by the gated source projection for donor statistics.
    # Reconstructing observations from ``tri_id`` would reintroduce the planar
    # UV overlap bug by assigning front atlas pixels to rear polygons.
    has_observation = triangle_observed & (triangle_observed_count > 0)
    tri_colour = np.zeros((total_tris, 3), np.float32)
    tri_colour[has_observation] = (
        triangle_observed_sum[has_observation] / triangle_observed_count[has_observation, None]
    ).astype(np.float32)

    # Donor selection is deliberately constrained. Unrestricted nearest-triangle transfer would
    # let the rifle inherit fur, or the tail inherit vest, purely because they are spatially close.
    # A donor must be on the SAME connected mesh component, agree in surface-normal direction, and
    # sit within a bounded radius. Tiers degrade explicitly: component donors -> component-local
    # robust prior -> global prior (emergency only), and each triangle records which tier it used.
    centroids = verts[tris].mean(axis=1)
    scene_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    max_donor_radius = scene_diag * DONOR_MAX_RADIUS_FRACTION

    # Connected components over triangles that share a vertex -- computed on POSITION-WELDED
    # indices, never on the raw indices. A UV unwrap duplicates vertices along every seam, so raw
    # indices would split one continuous surface into per-chart islands; donor scoping would then
    # collapse to "same UV chart" and most triangles would fall through to the global prior.
    quantised = np.round(verts / WELD_TOLERANCE).astype(np.int64)
    _, welded_index = np.unique(quantised, axis=0, return_inverse=True)
    welded_tris = welded_index[tris]

    corner_rows = np.repeat(np.arange(total_tris), 3)
    corner_cols = welded_tris.reshape(-1)
    incidence = coo_matrix(
        (np.ones(corner_rows.size, np.int8), (corner_rows, corner_cols)),
        shape=(total_tris, int(welded_tris.max()) + 1),
    ).tocsr()
    _, tri_component = connected_components(incidence @ incidence.T, directed=False)

    fill_tier = np.zeros(total_tris, np.int8)  # 0 observed, 1 donor, 2 component prior, 3 global
    fill_tier[~has_observation] = 3
    global_prior = (
        np.median(tri_colour[has_observation], axis=0)
        if has_observation.any()
        else np.array([96.0, 98.0, 86.0], np.float32)
    )

    for component in np.unique(tri_component):
        in_component = tri_component == component
        donors = in_component & has_observation
        targets = in_component & ~has_observation
        if not targets.any():
            continue
        if not donors.any():
            # No observation anywhere on this component: a component-local prior is impossible,
            # so fall back to the global prior and mark the tier honestly.
            tri_colour[targets] = global_prior
            fill_tier[targets] = 3
            continue

        donor_index = np.flatnonzero(donors)
        target_index = np.flatnonzero(targets)
        tree = cKDTree(centroids[donor_index])
        neighbours = min(DONOR_NEIGHBOURS, donor_index.size)
        distance, nearest = tree.query(centroids[target_index], k=neighbours)
        distance = np.atleast_2d(distance.T).T if neighbours > 1 else distance[:, None]
        nearest = np.atleast_2d(nearest.T).T if neighbours > 1 else nearest[:, None]

        donor_ids = donor_index[nearest]
        normal_agreement = np.einsum("ijk,ik->ij", normals[donor_ids], normals[target_index])
        acceptable = (normal_agreement >= DONOR_MIN_NORMAL_DOT) & (distance <= max_donor_radius)

        weight = np.where(
            acceptable,
            (1.0 / np.maximum(distance, 1e-6)) * np.clip(normal_agreement, 0.0, None),
            0.0,
        )
        weight_total = weight.sum(axis=1)
        has_donor = weight_total > 0

        blended = np.zeros((target_index.size, 3), np.float32)
        if has_donor.any():
            normalised = weight[has_donor] / weight_total[has_donor, None]
            blended[has_donor] = (tri_colour[donor_ids[has_donor]] * normalised[..., None]).sum(axis=1)
        # Component-local robust prior for targets with no acceptable donor.
        component_prior = np.median(tri_colour[donors], axis=0)
        blended[~has_donor] = component_prior

        tri_colour[target_index] = blended
        fill_tier[target_index] = np.where(has_donor, 1, 2).astype(np.int8)

    unseen_triangles = int((~has_observation).sum())

    paintable = (tri_id >= 0) & island & (~real_mask)
    if paintable.any():
        colour[paintable] = tri_colour[tri_id[paintable]]
        mask |= paintable

    # ---- island padding: grow only inside each chart, never across charts ----
    remaining = island & (~mask)
    if remaining.any():
        for label in range(1, n_islands):
            this_island = island_labels == label
            if not (this_island & remaining).any():
                continue
            ys_i, xs_i = np.nonzero(this_island)
            y0, y1 = int(ys_i.min()), int(ys_i.max()) + 1
            x0, x1 = int(xs_i.min()), int(xs_i.max()) + 1
            sub_island = this_island[y0:y1, x0:x1]
            sub_colour = colour[y0:y1, x0:x1]
            sub_mask = mask[y0:y1, x0:x1] & sub_island
            for _ in range(12):
                if sub_mask[sub_island].all():
                    break
                mf = sub_mask.astype(np.float32)
                numerator = cv2.blur(sub_colour * mf[..., None], (9, 9))
                denominator = cv2.blur(mf, (9, 9))[..., None]
                grown = numerator / np.maximum(denominator, 1e-5)
                newly = (~sub_mask) & sub_island & (denominator[..., 0] > 1e-5)
                if not newly.any():
                    break
                sub_colour[newly] = grown[newly]
                sub_mask |= newly
            colour[y0:y1, x0:x1] = sub_colour
            mask[y0:y1, x0:x1] |= sub_mask

    filled = mask & ~real_mask
    if filled.any():
        # Normalised (mask-aware) blur. A plain GaussianBlur averages in the black void outside the
        # chart, which is what previously darkened every synthesised region towards the island edge.
        mf = mask.astype(np.float32)
        numerator = cv2.GaussianBlur(colour * mf[..., None], (0, 0), 9.0)
        denominator = cv2.GaussianBlur(mf, (0, 0), 9.0)[..., None]
        smooth = numerator / np.maximum(denominator, 1e-6)
        colour[filled] = smooth[filled]

    atlas = np.clip(colour, 0, 255).astype(np.uint8)
    # Any remaining holes are inpainted per-island so colour can never cross into an unrelated
    # UV chart (island_labels partitions the atlas; inpaint runs once per label with everything
    # outside that label's island masked out of the input entirely).
    still_holes = (~mask) & island
    if still_holes.any():
        for label in range(1, n_islands):
            this_island = island_labels == label
            local_holes = (still_holes & this_island).astype(np.uint8)
            if not local_holes.any():
                continue
            masked_atlas = atlas.copy()
            masked_atlas[~this_island] = 0
            repaired = cv2.inpaint(masked_atlas, local_holes, 3, cv2.INPAINT_TELEA)
            atlas[this_island] = repaired[this_island]

    # Inpainting is intentionally conservative around one-pixel UV cracks and can leave isolated
    # zero-valued texels inside an otherwise painted island.  Close only those residual holes by
    # copying the nearest already-painted texel within the same island.  This cannot transport
    # colour across charts, and it never invents a source pixel: a chart with no painted texel is
    # filled from the already-computed local/global material prior.
    residual_holes = island & (~mask)
    residual_island_holes = int(np.count_nonzero(residual_holes))
    if residual_holes.any():
        for label in range(1, n_islands):
            this_island = island_labels == label
            holes = residual_holes & this_island
            if not holes.any():
                continue
            known = mask & this_island
            if known.any():
                distance_input = np.where(known, 0, 255).astype(np.uint8)
                _, labels = cv2.distanceTransformWithLabels(
                    distance_input, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
                )
                source_coords = np.argwhere(known)
                hole_coords = np.argwhere(holes)
                source_labels = labels[holes].astype(np.int64) - 1
                valid = (source_labels >= 0) & (source_labels < len(source_coords))
                if valid.any():
                    src = source_coords[source_labels[valid]]
                    dst = hole_coords[valid]
                    colour[dst[:, 0], dst[:, 1]] = colour[src[:, 0], src[:, 1]]
            else:
                colour[this_island] = global_prior
            mask[this_island] = True

    if args.neutral_fill_only:
        # Do not overwrite the atlas here.  Existing UVs may have ownership
        # collisions; erasing every texel whose *last* raster owner is
        # unobserved would also erase a valid observed front texel.  The
        # exporter applies neutral material per unobserved polygon, which is
        # the ownership-safe rear protection rule.
        neutral_fill_policy = "per_polygon_neutral_material"
    else:
        neutral_fill_policy = "atlas_fill_policy"

    atlas = np.clip(colour, 0, 255).astype(np.uint8)
    basecolor_path = outdir / "basecolor.png"
    cv2.imwrite(str(basecolor_path), cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR))
    decoded = cv2.imread(str(basecolor_path), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("ATLAS_RESOLUTION_CONTRACT_MISMATCH: basecolor write failed")
    assert_atlas_dimensions(decoded.shape[:2][::-1], atlas_size, "raster_project output")
    observed_mask_path = outdir / "observed_triangles.npy"
    np.save(observed_mask_path, triangle_observed)

    rear_direction = view_locs[usable[0][0]] / (np.linalg.norm(view_locs[usable[0][0]]) + 1e-9)
    rear_dominant = (normals @ rear_direction) < -FACING_MIN
    illegal_rear = rear_face_provenance_violations(
        rear_dominant,
        triangle_winning_view,
        triangle_winning_facial,
    )
    provenance_path = Path(args.provenance) if args.provenance else outdir / "triangle_texture_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema": "triangle_texture_provenance_v2",
        "triangle_count": int(total_tris),
        "source_views": view_names,
        "winning_source_view": triangle_winning_view.astype(int).tolist(),
        "winning_confidence": triangle_winning_conf.astype(float).tolist(),
        "visible": triangle_visible.tolist(),
        "front_facing": triangle_front_facing.tolist(),
        "face_id_matched": triangle_face_id_matched.tolist(),
        "masked_valid": triangle_mask_valid.tolist(),
        "rear_dominant": rear_dominant.tolist(),
        "winning_source_is_facial": triangle_winning_facial.tolist(),
        "fallback_mode": [
            "source_view" if bool(observed) else "neutral_synthesis"
            for observed in triangle_observed
        ],
        "illegal_rear_facial_triangle_ids": np.flatnonzero(illegal_rear).astype(int).tolist(),
        "gates": {
            "depth_visible_required": True,
            "front_facing_threshold": FACING_MIN,
            "face_id_match_required": bool(args.require_face_id),
            "face_id_pixel_tolerance": max(int(args.face_id_radius), 0),
            "source_mask_alpha_min": ALPHA_MIN,
            "confidence_threshold": 0.20,
            "rear_facial_guard": True,
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if illegal_rear.any():
        raise RuntimeError(
            "REJECTED_REAR_FACE_PROJECTION: "
            f"{int(illegal_rear.sum())} rear triangles received facial provenance; "
            f"see {provenance_path}"
        )

    dbg = np.zeros((atlas_size, atlas_size, 3), np.uint8)
    dbg[island] = (60, 60, 60)
    dbg[filled] = (120, 120, 120)
    for vi, vname in usable:
        dbg[best_view == vi] = (60, 220, 90)
    cv2.imwrite(str(outdir / "debug_source_view.png"), cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(outdir / "debug_confidence.png"),
                (np.clip(best_conf / max(best_conf.max(), 1e-6), 0, 1) * 255).astype(np.uint8))
    cov = np.zeros((atlas_size, atlas_size), np.uint8)
    cov[island] = 40
    cov[filled] = 130
    cov[real_mask] = 255
    cv2.imwrite(str(outdir / "debug_coverage.png"), cov)

    write_progress(len(usable), total_tris, status="complete")

    if args.report:
        island_px = int(island.sum())
        report = {
            "success": True,
            "backend": "raster_uv_atlas_projection_cpu",
            "atlas_resolution": atlas_size,
            "atlas_resolution_contract": {
                "requested": atlas_size,
                "saved": [atlas_size, atlas_size],
                "passed": True,
            },
            "semantic_views": [n for _, n in usable],
            "barred_views": [n for n in view_names if n not in [x[1] for x in usable]],
            "uv_island_occupancy_percent": round(island_px / (atlas_size * atlas_size) * 100, 2),
            "observed_semantic_coverage_percent": round(real_pct, 2),
            "synthesized_surface_coverage_percent": round(100.0 - real_pct, 2),
            "final_filled_uv_percent": 100.0,
            "high_confidence_percent": round(float((best_conf > HIGH_CONF).sum() / island_px * 100), 2),
            "unseen_triangles": unseen_triangles,
            "observed_triangles": int(triangle_observed.sum()),
            "observed_triangle_percent": round(float(triangle_observed.mean() * 100.0), 2),
            "unobserved_triangles": int((~triangle_observed).sum()),
            "observed_triangle_mask": str(observed_mask_path),
            "visibility_gate": {
                "depth_visibility_mask_required": True,
                "normal_facing_threshold": FACING_MIN,
                "invalid_normals_rejected": True,
                "face_id_match_required": bool(args.require_face_id),
                "source_mask_required": True,
                "sample_confidence_threshold": 0.20,
                "per_view_counts": gate_counts,
            },
            "triangle_texture_provenance": str(provenance_path),
            "unseen_fill_policy": "neutral_material_for_unobserved_triangles",
            "neutral_fill_only": bool(args.neutral_fill_only),
            "neutral_fill_policy": neutral_fill_policy,
            "fill_tier_triangle_counts": {
                "observed": int((fill_tier == 0).sum()),
                "constrained_donor": int((fill_tier == 1).sum()),
                "component_local_prior": int((fill_tier == 2).sum()),
                "global_prior_emergency": int((fill_tier == 3).sum()),
            },
            "donor_constraints": {
                "neighbours": DONOR_NEIGHBOURS,
                "min_normal_dot": DONOR_MIN_NORMAL_DOT,
                "max_radius_fraction": DONOR_MAX_RADIUS_FRACTION,
                "same_component_required": True,
            },
            "residual_island_holes": residual_island_holes,
            "residual_island_holes_remaining": int(np.count_nonzero(island & (~mask))),
            "residual_hole_fill": "nearest_painted_texel_same_island_or_material_prior",
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"RASTER_PROJECT real={real_pct:.1f}% filled={float(filled.sum()/island.sum()*100):.1f}% "
          f"elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
