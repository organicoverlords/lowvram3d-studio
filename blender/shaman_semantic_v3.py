"""V3 semantic regions: split anatomical arm from sleeve cloth, plus triangle diagnostics.

The V2 pass proved the arm no longer drags torso or cape, but it treated the
whole sleeve as one region and bound it to the arm chain, so the draped garment
lifted as a rigid triangular sheet. A vertex-only classification cannot see that
failure: the panel is rigid precisely because every vertex in it moves together.

This module therefore does two things V2 could not:

* splits the sleeve into anatomical core, anchor and drape bands, using radial
  distance to the arm axis measured from the mesh rather than assumed;
* classifies every *triangle* by the semantic classes of its three vertices, so
  cross-region triangles and stretch/shear/flip defects are measurable.

Cut points come from the measured radial distribution of the V2 sleeve set:
radius percentiles 5/25/50/75/90/95 = 0.017 / 0.048 / 0.075 / 0.101 / 0.133 /
0.149, and chain-t percentiles 1.08-2.0 (i.e. the sleeve mass sits on the
forearm-to-hand segment, not the upper arm).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shaman_weight_diagnostics import semantic_masks  # noqa: E402

# Radial bands around the arm axis, in model units.
ARM_CORE_RADIUS = 0.045      # anatomical arm cross-section
ANCHOR_RADIUS = 0.075        # cloth sitting directly on the arm
DRAPE_UPPER_RADIUS = 0.115   # hanging cloth still following the sleeve
HAND_CORE_RADIUS = 0.060     # visible hand lobe only
ANCHOR_MAX_T = 1.35          # anchor stays near the arm root


def arm_axis_metrics(points: np.ndarray, landmarks: dict, side: str):
    """Radial distance to the arm polyline and parametric position along it."""

    joints = {j["name"]: np.array(j["position"]) for j in landmarks["joints"]}
    segments = [
        (joints[f"upperarm_{side}"], joints[f"lowerarm_{side}"]),
        (joints[f"lowerarm_{side}"], joints[f"hand_{side}"]),
    ]
    best = np.full(points.shape[0], np.inf)
    param = np.zeros(points.shape[0])
    projection = np.zeros_like(points)
    for index, (a, b) in enumerate(segments):
        ab = b - a
        length_sq = float(ab @ ab)
        if length_sq < 1e-12:
            continue
        t = np.clip(((points - a) @ ab) / length_sq, 0.0, 1.0)
        candidate = a[None, :] + t[:, None] * ab[None, :]
        distance = np.linalg.norm(points - candidate, axis=1)
        update = distance < best
        best[update] = distance[update]
        param[update] = (index + t)[update]
        projection[update] = candidate[update]
    return best, param, projection, joints


def semantic_masks_v3(points: np.ndarray, landmarks: dict, staff_mask: np.ndarray) -> dict:
    """V2 masks plus the arm-core / anchor / drape split on both sides."""

    masks = dict(semantic_masks(points, landmarks, staff_mask))

    for side in ("l", "r"):
        radius, param, _projection, joints = arm_axis_metrics(points, landmarks, side)
        envelope = masks[f"sleeve_{side}"] | masks[f"hand_{side}_region"]
        hand_distance = np.linalg.norm(points - joints[f"hand_{side}"], axis=1)

        arm_core = envelope & (radius <= ARM_CORE_RADIUS)
        hand_core = (
            envelope
            & (hand_distance <= HAND_CORE_RADIUS)
            & (radius <= ANCHOR_RADIUS)
            & ~arm_core
        )
        anchor = (
            envelope
            & (radius > ARM_CORE_RADIUS)
            & (radius <= ANCHOR_RADIUS)
            & (param <= ANCHOR_MAX_T)
            & ~hand_core
        )
        drape_upper = (
            envelope & ~arm_core & ~hand_core & ~anchor
            & (radius <= DRAPE_UPPER_RADIUS)
        )
        drape_lower = (
            envelope & ~arm_core & ~hand_core & ~anchor & ~drape_upper
        )

        masks[f"arm_core_{side}"] = arm_core
        masks[f"hand_core_{side}"] = hand_core
        masks[f"sleeve_anchor_{side}"] = anchor
        masks[f"sleeve_drape_upper_{side}"] = drape_upper
        masks[f"sleeve_drape_lower_{side}"] = drape_lower
        masks[f"sleeve_drape_{side}"] = drape_upper | drape_lower
        masks[f"arm_radius_{side}"] = radius
        masks[f"arm_param_{side}"] = param

    masks["cape_r"] = masks["side_cape"] & (points[:, 0] > landmarks["symmetry_plane_x"])
    masks["cape_l"] = masks["side_cape"] & (points[:, 0] <= landmarks["symmetry_plane_x"])
    return masks


CLASS_ORDER = (
    "staff", "torso_core", "rear_cape", "cape_l", "cape_r",
    "arm_core_l", "hand_core_l", "sleeve_anchor_l", "sleeve_drape_upper_l", "sleeve_drape_lower_l",
    "arm_core_r", "hand_core_r", "sleeve_anchor_r", "sleeve_drape_upper_r", "sleeve_drape_lower_r",
    "hanging_accessories",
)


def vertex_classes(masks: dict, count: int) -> np.ndarray:
    """One winning class per vertex, resolved by CLASS_ORDER precedence."""

    classes = np.full(count, -1, dtype=np.int32)
    for index, name in enumerate(CLASS_ORDER):
        mask = masks.get(name)
        if mask is None or not isinstance(mask, np.ndarray) or mask.dtype != bool:
            continue
        classes[(classes < 0) & mask] = index
    return classes


def triangle_metrics(rest: np.ndarray, posed: np.ndarray, triangles: np.ndarray) -> dict:
    """Per-triangle deformation: stretch, area, aspect, shear, flips, degeneracy."""

    def edges(points):
        a = points[triangles[:, 0]]
        b = points[triangles[:, 1]]
        c = points[triangles[:, 2]]
        return a, b, c

    ra, rb, rc = edges(rest)
    pa, pb, pc = edges(posed)

    rest_lengths = np.stack([
        np.linalg.norm(rb - ra, axis=1),
        np.linalg.norm(rc - rb, axis=1),
        np.linalg.norm(ra - rc, axis=1),
    ], axis=1)
    posed_lengths = np.stack([
        np.linalg.norm(pb - pa, axis=1),
        np.linalg.norm(pc - pb, axis=1),
        np.linalg.norm(pa - pc, axis=1),
    ], axis=1)

    safe_rest = np.maximum(rest_lengths, 1e-9)
    stretch = posed_lengths / safe_rest

    rest_normal = np.cross(rb - ra, rc - ra)
    posed_normal = np.cross(pb - pa, pc - pa)
    rest_area = 0.5 * np.linalg.norm(rest_normal, axis=1)
    posed_area = 0.5 * np.linalg.norm(posed_normal, axis=1)
    area_ratio = posed_area / np.maximum(rest_area, 1e-12)

    rest_unit = rest_normal / np.maximum(np.linalg.norm(rest_normal, axis=1, keepdims=True), 1e-12)
    posed_unit = posed_normal / np.maximum(np.linalg.norm(posed_normal, axis=1, keepdims=True), 1e-12)
    alignment = np.einsum("ij,ij->i", rest_unit, posed_unit)

    def aspect(lengths, area):
        longest = lengths.max(axis=1)
        return (longest * longest) / np.maximum(area, 1e-12)

    aspect_change = aspect(posed_lengths, posed_area) / np.maximum(
        aspect(rest_lengths, rest_area), 1e-12
    )

    valid = rest_area > 1e-12
    return {
        "triangles": int(triangles.shape[0]),
        "max_edge_stretch": float(stretch[valid].max()) if valid.any() else 0.0,
        "max_area_ratio": float(area_ratio[valid].max()) if valid.any() else 0.0,
        "min_area_ratio": float(area_ratio[valid].min()) if valid.any() else 0.0,
        "max_aspect_ratio_change": float(aspect_change[valid].max()) if valid.any() else 0.0,
        "flipped_normals": int((alignment < 0.0).sum()),
        "inverted_triangles": int(((alignment < 0.0) & valid).sum()),
        "degenerate_introduced": int(((posed_area <= 1e-12) & valid).sum()),
        "extreme_stretch_triangles": int((stretch.max(axis=1) > 1.5).sum()),
        "extreme_shear_triangles": int((aspect_change > 3.0).sum()),
        "longest_new_edge": float(posed_lengths.max()) if triangles.size else 0.0,
        "longest_new_stretched_edge": float(
            posed_lengths[stretch > 1.2].max()
        ) if (stretch > 1.2).any() else 0.0,
    }


def cross_region_triangles(classes: np.ndarray, triangles: np.ndarray) -> dict:
    """Count triangles whose vertices span different semantic classes."""

    tri_classes = classes[triangles]
    mixed = (tri_classes[:, 0] != tri_classes[:, 1]) | (tri_classes[:, 1] != tri_classes[:, 2])

    def index_of(name: str) -> int:
        return CLASS_ORDER.index(name) if name in CLASS_ORDER else -2

    def pair_count(first: str, second: str) -> int:
        a, b = index_of(first), index_of(second)
        has_a = (tri_classes == a).any(axis=1)
        has_b = (tri_classes == b).any(axis=1)
        return int((has_a & has_b).sum())

    return {
        "mixed_triangles": int(mixed.sum()),
        "mixed_ratio": float(mixed.sum() / max(triangles.shape[0], 1)),
        "arm_core_to_torso": pair_count("arm_core_r", "torso_core"),
        "arm_core_to_cape": pair_count("arm_core_r", "cape_r")
        + pair_count("arm_core_r", "rear_cape"),
        "hand_core_to_sleeve_drape": pair_count("hand_core_r", "sleeve_drape_upper_r")
        + pair_count("hand_core_r", "sleeve_drape_lower_r"),
        "sleeve_anchor_to_drape": pair_count("sleeve_anchor_r", "sleeve_drape_upper_r")
        + pair_count("sleeve_anchor_r", "sleeve_drape_lower_r"),
        "mixed_mask": mixed,
    }


def region_summary(masks: dict, count: int) -> dict:
    return {
        name: int(masks[name].sum())
        for name in CLASS_ORDER
        if isinstance(masks.get(name), np.ndarray) and masks[name].dtype == bool
    }


def write_report(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
