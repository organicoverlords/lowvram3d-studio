"""Optimise placement against the measurements the pipeline already takes.

Placement is heuristic, and until now correctness was checked *afterwards*:
`audit_actor_overlaps` reported that a barn was inside a tree, and nothing acted
on the report. The fitted ground plane, the pairwise overlap and the per-region
source bounding box are already computed every run. This turns those three
reports into a loss and moves the actors until it is small.

Four terms, all measured rather than invented:

* **reprojection** -- an actor's box, projected back into the source camera,
  should cover the pixels its region was segmented from.
* **measured anchor** -- an actor should stay near where MoGe put it. A 2D
  bounding box does not constrain depth, so without this the optimiser satisfies
  the plane by sliding actors along the view ray.
* **ground contact** -- an actor's underside should meet the fitted plane.
* **non-penetration** -- solids should not share volume.

The first two are what was observed; the last two are how a scene ought to
behave. When they disagree, the observations should win, which is what the
tolerances encode.

Deliberately *not* a term: similarity to a render of the result. Source-view
similarity has certified a world-space-wrong mesh twice in this project. It is
the evaluation, and evaluation must not be the objective.

Translation only, 3 DoF per actor. Yaw is left alone because the overlap term
here is axis-aligned, so a yaw it cannot see is a yaw it should not move.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Kinds that rest on the terrain. The terrain itself does not.
GROUNDED_KINDS = {"structure", "trunk_support", "scatter_instance", "prop"}
# Kinds excluded from the penetration term: the ground is *supposed* to be
# underneath everything, and penalising that would push the scene into the air.
SURFACE_KINDS = {"ground_plane", "water_surface"}

# Each term is divided by what it would be acceptable to be wrong by, so the
# sum is dimensionless and the three are actually comparable. Weighting them
# directly does not work: image error is in normalised units (~0.01) and ground
# error is in metres (~1), so a plain weight of 0.15 on the ground term still
# lets it outweigh reprojection by two orders of magnitude, and every actor gets
# slammed onto the terrain regardless of where it was photographed.
#
# A fifth of the image being wrong and a metre of float are then equally bad,
# which is a claim about this pipeline that can be argued with -- which is the
# point of writing it as a tolerance rather than burying it in a weight.
REPROJECTION_TOLERANCE_NORM = 0.02
# Defaulted, but overridden per scene by the plane's own fit residual: a plane
# fitted to within a metre has no business pulling actors to within a centimetre.
GROUND_TOLERANCE_M = 0.5
# Fraction of the smaller actor's volume that may be shared before it counts.
PENETRATION_TOLERANCE = 0.05
# How far an actor may drift from where MoGe measured it before that counts as
# contradicting the depth reconstruction.
#
# This term exists because a 2D bounding box cannot pin down depth: with only
# reprojection and ground contact, the optimiser discovered it could satisfy the
# plane by sliding an actor two metres *towards the camera*, paying a little
# reprojection to gain a lot of ground. The starting position is not an
# arbitrary initial guess to be improved away -- it is a depth measurement, and
# it needs a term saying so.
ANCHOR_TOLERANCE_M = 0.75
# An actor is not moved further than this *on any axis* from where it was
# measured, whatever the terms want. A metre of correction is a fix; ten is a
# fabrication. The Euclidean distance can reach sqrt(3) times this, and is
# reported separately so the cap is not mistaken for a distance bound.
MAX_SHIFT_PER_AXIS_M = 2.0


def ground_height_m(plane: dict[str, float] | None, forward: float,
                    right: float) -> float | None:
    if not plane:
        return None
    return (plane["height_at_camera_m"]
            + plane["slope_forward"] * forward
            + plane["slope_right"] * right)


# A box that straddles the camera plane has no finite image bounding box -- the
# perspective divide sends its near corners to infinity. The terrain slab does
# exactly this every scene: it is forty metres across and the camera stands on
# it. Left unbounded, one such actor contributes a cost of order 1e13 and every
# real gradient in the scene disappears underneath it.
#
# Clamp to somewhat outside the frame. Past that, "further off screen" carries
# no information worth optimising towards.
PROJECTION_CLAMP = (-1.0, 2.0)
MIN_FORWARD_M = 0.25


def project_box(centre: np.ndarray, size: np.ndarray, tan_x: float,
                tan_y: float) -> np.ndarray:
    """Project an axis-aligned box into normalised image coordinates.

    Unreal frame here is forward = +x, right = +y, up = +z, camera at the
    origin looking down +x -- the frame `_measured_extent` converts into.
    """
    half = size * 0.5
    signs = np.array([[sx, sy, sz]
                      for sx in (-1.0, 1.0)
                      for sy in (-1.0, 1.0)
                      for sz in (-1.0, 1.0)])
    corners = centre + signs * half
    forward = np.maximum(corners[:, 0], MIN_FORWARD_M)
    u = 0.5 + (corners[:, 1] / forward) / (2.0 * tan_x)
    v = 0.5 - (corners[:, 2] / forward) / (2.0 * tan_y)
    u = np.clip(u, *PROJECTION_CLAMP)
    v = np.clip(v, *PROJECTION_CLAMP)
    return np.array([u.min(), v.min(), u.max(), v.max()])


def _centre_of(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5])


def _extent_of(bbox: np.ndarray) -> np.ndarray:
    return np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])


def scale_disagreement(predicted: np.ndarray, target: np.ndarray) -> float:
    """How much larger or smaller an actor projects than the pixels it came from.

    1.0 is agreement. This is a *size* error and no amount of moving fixes it,
    so it is reported rather than optimised.
    """
    predicted_extent = _extent_of(predicted)
    target_extent = np.maximum(_extent_of(target), 1e-6)
    ratios = predicted_extent / target_extent
    return float(np.exp(np.abs(np.log(np.maximum(ratios, 1e-6))).max()))


def straddles_camera(centre: np.ndarray, size: np.ndarray) -> bool:
    """Whether the box crosses the camera plane, making its projection meaningless."""
    return bool(centre[0] - size[0] * 0.5 <= MIN_FORWARD_M)


def _overlap_volume(centre_a, size_a, centre_b, size_b) -> float:
    lo = np.maximum(centre_a - size_a * 0.5, centre_b - size_b * 0.5)
    hi = np.minimum(centre_a + size_a * 0.5, centre_b + size_b * 0.5)
    span = np.maximum(hi - lo, 0.0)
    return float(span.prod())


def _actor_state(placement: dict[str, Any]) -> tuple[list, np.ndarray, np.ndarray]:
    actors = placement.get("actors", [])
    centres = np.array([[a["location_cm"][0] / 100.0,
                         a["location_cm"][1] / 100.0,
                         a["location_cm"][2] / 100.0] for a in actors])
    sizes = np.array([[float(v) for v in a["size_m"]] for a in actors])
    return actors, centres, sizes


def refine(placement: dict[str, Any], plane: dict[str, float] | None,
           camera: dict[str, Any]) -> dict[str, Any]:
    """Move actors to satisfy the measurements, and report what moved and why."""
    from scipy.optimize import minimize

    actors, centres, sizes = _actor_state(placement)
    if not len(actors):
        return {"schema_version": "placement_refinement_v1",
                "classification": "NOT_APPLICABLE", "reason": "no actors"}

    fov_x = math.radians(float(camera.get("fov_x_deg", 90.0)))
    fov_y = math.radians(float(camera.get("fov_y_deg", 60.0)))
    tan_x, tan_y = math.tan(fov_x / 2.0), math.tan(fov_y / 2.0)

    targets = [a.get("source_bbox_norm_xyxy") for a in actors]
    # An actor the camera stands inside cannot be located by its silhouette.
    # Drop the term rather than optimise against a clamped fiction, and say how
    # many lost it so a scene held up only by the anchor is visible as such.
    unprojectable = [index for index in range(len(actors))
                     if straddles_camera(centres[index], sizes[index])]
    for index in unprojectable:
        targets[index] = None
    # An inferred actor was never photographed -- a trunk_support carries its
    # whole region's bbox while being a 0.7 m cylinder, so its "reprojection
    # error" is a comparison against pixels belonging to something else. It
    # measured 65x disagreement, which is not a defect in the trunk, it is a
    # question that should not have been asked.
    inferred = [index for index, a in enumerate(actors) if a.get("inferred")]
    for index in inferred:
        targets[index] = None
    # A surface's extent is not set by its visible pixels. Terrain runs to the
    # horizon and out of frame on every side, so its projection is clipped and
    # the ratio against a grass-region bbox is an artefact of the clamp rather
    # than a measurement -- the barn scene's ground planes read 3.0 and 23.6
    # against the barn's honest 1.12.
    #
    # On that scene this changes nothing, because both slabs already straddle
    # the camera and were dropped above. It is here for the surface that does
    # not: a distant water plane projects finitely and would otherwise be
    # optimised towards the size of whatever patch of it happened to be visible.
    surfaces = [index for index, a in enumerate(actors)
                if a.get("kind") in SURFACE_KINDS]
    for index in surfaces:
        targets[index] = None
    grounded = np.array([a.get("kind") in GROUNDED_KINDS for a in actors])
    solid = [index for index, a in enumerate(actors)
             if a.get("kind") not in SURFACE_KINDS]
    # Instances of one region are a decomposition of a single continuous mass --
    # twelve clumps of one hedge are *supposed* to abut. Penalising them as if
    # they were separate objects makes the optimiser shove a hedge apart, which
    # is why the shift cap was binding on half the scene.
    pairs = [(i, j) for position, i in enumerate(solid)
             for j in solid[position + 1:]
             if actors[i].get("region_id") != actors[j].get("region_id")]

    def unpack(flat):
        return centres + flat.reshape(centres.shape)

    # A plane fitted to within a metre should not pull actors to a centimetre.
    ground_tolerance = max(
        float((plane or {}).get("residual_p95_m") or 0.0), GROUND_TOLERANCE_M)

    def cost(flat):
        moved = unpack(flat)
        # Drift from the measured position. Cheap, vectorised, and applied to
        # every actor including those with no source bbox.
        total = float((((moved - centres) / ANCHOR_TOLERANCE_M) ** 2).sum())
        for index, target in enumerate(targets):
            if not target:
                continue
            predicted = project_box(moved[index], sizes[index], tan_x, tan_y)
            # Centre only, deliberately. Translation can correct a centre that
            # is in the wrong place; it cannot correct an extent that is the
            # wrong size, and asking it to try makes position absorb a scale
            # error -- an actor shoved metres off its measured depth so its
            # silhouette grows to match. Scale disagreement is reported instead.
            residual = (_centre_of(predicted) - _centre_of(np.asarray(target, float)))
            total += float(((residual / REPROJECTION_TOLERANCE_NORM) ** 2).sum())
        if plane:
            for index in np.flatnonzero(grounded):
                bottom = moved[index, 2] - sizes[index, 2] * 0.5
                height = ground_height_m(plane, moved[index, 0], moved[index, 1])
                total += ((bottom - height) / ground_tolerance) ** 2
        for i, j in pairs:
            shared = _overlap_volume(moved[i], sizes[i], moved[j], sizes[j])
            if shared > 0.0:
                smaller = min(sizes[i].prod(), sizes[j].prod())
                fraction = shared / max(smaller, 1e-6)
                total += (fraction / PENETRATION_TOLERANCE) ** 2
        return total

    before = cost(np.zeros(centres.size))
    bounds = [(-MAX_SHIFT_PER_AXIS_M, MAX_SHIFT_PER_AXIS_M)] * centres.size
    solution = minimize(cost, np.zeros(centres.size), method="L-BFGS-B",
                        bounds=bounds, options={"maxiter": 400})
    shifts = solution.x.reshape(centres.shape)
    after = cost(solution.x)

    # A refinement that does not improve the objective is not applied. The
    # optimiser is a tool for satisfying the measurements, not an authority
    # over them.
    improved = bool(after < before)
    if improved:
        for index, actor in enumerate(actors):
            actor["location_cm"] = [
                round(float((centres[index, axis] + shifts[index, axis]) * 100.0), 2)
                for axis in range(3)]
            distance = float(np.linalg.norm(shifts[index]))
            if distance > 1e-4:
                actor["refined_shift_m"] = round(distance, 4)

    # Size error the optimiser was explicitly not allowed to hide in position.
    final = unpack(solution.x if improved else np.zeros(centres.size))
    disagreements = [
        scale_disagreement(project_box(final[index], sizes[index], tan_x, tan_y),
                           np.asarray(target, dtype=float))
        for index, target in enumerate(targets) if target]
    for index, target in enumerate(targets):
        if target:
            actors[index]["projected_scale_ratio"] = round(scale_disagreement(
                project_box(final[index], sizes[index], tan_x, tan_y),
                np.asarray(target, dtype=float)), 3)

    distances = np.linalg.norm(shifts, axis=1)
    per_axis = np.abs(shifts).max(axis=1)
    return {
        "schema_version": "placement_refinement_v1",
        "classification": "PROVEN" if improved else "NOT_APPLICABLE",
        "applied": improved,
        "cost_before": round(float(before), 6),
        "cost_after": round(float(after), 6),
        "actor_count": len(actors),
        "moved_count": int((distances > 0.01).sum()),
        "max_shift_m": round(float(distances.max()), 4),
        "mean_shift_m": round(float(distances.mean()), 4),
        "max_shift_per_axis_m": round(float(per_axis.max()), 4),
        "shift_cap_per_axis_m": MAX_SHIFT_PER_AXIS_M,
        "at_cap_count": int((per_axis >= MAX_SHIFT_PER_AXIS_M - 1e-6).sum()),
        "tolerances": {"reprojection_norm": REPROJECTION_TOLERANCE_NORM,
                       "ground_m": round(ground_tolerance, 4),
                       "penetration_fraction": PENETRATION_TOLERANCE,
                       "measured_anchor_m": ANCHOR_TOLERANCE_M},
        "ground_plane_used": bool(plane),
        "reprojection_actor_count": int(sum(1 for t in targets if t)),
        "scale_disagreement": {
            "worst_ratio": round(max(disagreements), 3) if disagreements else None,
            "median_ratio": round(float(np.median(disagreements)), 3) if disagreements else None,
            "note": "actor extent vs the pixels it came from; translation cannot fix this",
        },
        "unprojectable_actor_count": len(unprojectable),
        "inferred_actor_count": len(inferred),
        "surface_actor_count": len(surfaces),
        "optimiser": {"method": "L-BFGS-B", "success": bool(solution.success),
                      "iterations": int(solution.nit)},
    }
