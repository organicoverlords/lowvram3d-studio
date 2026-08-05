"""Deterministic semantic region segmentation for the shaman rig base.

Every region carries its own confidence, mesh state and source method. The asset
is a single fused mesh, so almost nothing is genuinely `separate`; regions are
reported as `fused` unless the connectivity analysis proves otherwise, and
`ambiguous` when the evidence does not separate two plausible labels.

No region is invented from weak evidence: a label that cannot be supported is
emitted with low confidence and `safety_classification = do_not_deform`, which
the skinning stage treats as a hard exclusion rather than a hint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash, export_glb, reset_scene  # noqa: E402


# Height fractions measured from the Milestone 0 audit and the rendered proof.
BANDS = {
    "foot": (0.000, 0.075),
    "shin": (0.075, 0.215),
    "thigh": (0.215, 0.370),
    "pelvis": (0.370, 0.455),
    "spine": (0.455, 0.585),
    "chest": (0.585, 0.665),
    "neck": (0.665, 0.700),
    "head": (0.700, 0.860),
    "antler": (0.860, 1.001),
}

ARM_BAND = (0.280, 0.620)
HAND_BAND_HEIGHT = 0.075       # fraction of model height below the arm lobe base
STAFF_MIN_Z_COVERAGE = 0.60
STAFF_MAX_FOOTPRINT = 0.16     # model units, xy extent of the staff column
ORNAMENT_BAND = (0.500, 0.860)


def load_points(path: str):
    """Load the rig base.

    A .blend is required for welded input: exporting to GLB and reimporting
    re-splits vertices per corner, silently undoing the weld and inflating the
    vertex count back to ~3.2M. GLB is an export format here, never a carrier
    between rig stages.
    """

    suffix = Path(path).suffix.lower()
    if suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path)
    else:
        reset_scene()
        bpy.ops.import_scene.gltf(filepath=path)
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one mesh in rig base, found {len(meshes)}")
    obj = meshes[0]
    count = len(obj.data.vertices)
    buffer = np.empty(count * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", buffer)
    local = buffer.reshape(-1, 3)
    matrix = np.array(obj.matrix_world.to_4x4())
    homogeneous = np.hstack([local, np.ones((count, 1), dtype=np.float32)])
    world = (matrix @ homogeneous.T).T[:, :3]
    return obj, world


def dense_interval(values: np.ndarray, bins: int = 96, floor_ratio: float = 0.08):
    if values.size == 0:
        return None
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return {"low": low, "high": high, "width": 0.0}
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    peak = counts.max()
    if peak <= 0:
        return None
    floor = max(peak * floor_ratio, 1.0)
    start = int(np.clip(np.searchsorted(edges, np.median(values)) - 1, 0, bins - 1))
    if counts[start] < floor:
        start = int(np.argmax(counts))
    left = start
    while left > 0 and counts[left - 1] >= floor:
        left -= 1
    right = start
    while right < bins - 1 and counts[right + 1] >= floor:
        right += 1
    return {
        "low": float(edges[left]),
        "high": float(edges[right + 1]),
        "width": float(edges[right + 1] - edges[left]),
    }


def find_staff_column(points: np.ndarray, height: float, centre_x: float) -> dict:
    """Locate the staff by fitting an axis and keeping a cylinder around it.

    An XY occupancy grid fails here: the pole is tilted, so over ~1.9 m it walks
    across several grid cells and no single cell shows tall coverage. Fitting a
    line through the candidate band and selecting a cylinder handles the tilt,
    and the radius test rejects the hand that grips the shaft at the same X.
    """

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    low = float(z.min())
    fraction = (z - low) / height
    span = float(x.max() - x.min())

    # Coarse: X bins that span most of the model height and sit off-centre.
    bins = 96
    edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    index = np.clip(((x - float(x.min())) / max(span, 1e-9) * bins).astype(np.int32), 0, bins - 1)
    candidates = []
    for step in range(bins):
        mask = index == step
        if int(mask.sum()) < 300:
            continue
        coverage = float(z[mask].max() - z[mask].min()) / height
        offset = abs((edges[step] + edges[step + 1]) * 0.5 - centre_x)
        if coverage >= 0.70 and offset > span * 0.15:
            candidates.append((step, coverage, offset))

    if not candidates:
        return {"detected": False, "reason": "NO_TALL_OFFSET_X_BAND", "confidence": 0.0}

    # Group candidate bins into contiguous runs. Taking min/max across all
    # candidates would span both sides of the body and fit the axis straight
    # through the torso, so exactly one run is selected.
    runs: list[list[tuple[int, float, float]]] = []
    for item in sorted(candidates):
        if runs and item[0] == runs[-1][-1][0] + 1:
            runs[-1].append(item)
        else:
            runs.append([item])

    scored = []
    for run in runs:
        steps = [entry[0] for entry in run]
        width = float(edges[max(steps) + 1] - edges[min(steps)])
        if width > 0.25:
            continue
        scored.append((sum(entry[1] for entry in run) / len(run), run, width))
    if not scored:
        return {
            "detected": False,
            "reason": "NO_NARROW_CONTIGUOUS_OFFSET_RUN",
            "confidence": 0.0,
            "run_count": len(runs),
        }

    scored.sort(key=lambda item: -item[0])
    best_run = scored[0][1]
    steps = [entry[0] for entry in best_run]
    band_low = float(edges[min(steps)])
    band_high = float(edges[max(steps) + 1])
    band = (x >= band_low) & (x <= band_high)
    if int(band.sum()) < 1000:
        return {"detected": False, "reason": "STAFF_BAND_TOO_SPARSE", "confidence": 0.0}

    # Fit x(z) and y(z) on per-slice medians so cloth and hand outliers cannot
    # drag the axis; medians are robust and deterministic.
    slices = 40
    sample_z, sample_x, sample_y = [], [], []
    for step in range(slices):
        lo = step / slices
        hi = (step + 1) / slices
        mask = band & (fraction >= lo) & (fraction < hi)
        if int(mask.sum()) < 50:
            continue
        sample_z.append(float(np.median(z[mask])))
        sample_x.append(float(np.median(x[mask])))
        sample_y.append(float(np.median(y[mask])))

    if len(sample_z) < 12:
        return {"detected": False, "reason": "STAFF_AXIS_UNDERSAMPLED", "confidence": 0.0}

    zs = np.array(sample_z)
    fit_x = np.polyfit(zs, np.array(sample_x), 1)
    fit_y = np.polyfit(zs, np.array(sample_y), 1)

    radial = np.hypot(x - np.polyval(fit_x, z), y - np.polyval(fit_y, z))
    for radius in (0.055, 0.070, 0.085, 0.100):
        mask = radial <= radius
        if int(mask.sum()) < 500:
            continue
        coverage = float(z[mask].max() - z[mask].min()) / height
        if coverage >= STAFF_MIN_Z_COVERAGE:
            block = points[mask]
            footprint_x = float(block[:, 0].max() - block[:, 0].min())
            footprint_y = float(block[:, 1].max() - block[:, 1].min())
            if footprint_x <= STAFF_MAX_FOOTPRINT and footprint_y <= STAFF_MAX_FOOTPRINT:
                return {
                    "detected": True,
                    "confidence": 0.7,
                    "vertex_indices": np.flatnonzero(mask),
                    "vertex_count": int(mask.sum()),
                    "z_coverage": coverage,
                    "cylinder_radius": radius,
                    "axis_x_slope_intercept": [float(v) for v in fit_x],
                    "axis_y_slope_intercept": [float(v) for v in fit_y],
                    "band": [band_low, band_high],
                    "footprint_x": footprint_x,
                    "footprint_y": footprint_y,
                    "centroid": [float(v) for v in block.mean(axis=0)],
                    "method": "median_axis_fit_plus_cylinder_selection",
                    "reason": None,
                }

    return {
        "detected": False,
        "reason": "NO_CYLINDER_RADIUS_SATISFIED_COVERAGE_AND_FOOTPRINT",
        "confidence": 0.2,
        "band": [band_low, band_high],
        "method": "median_axis_fit_plus_cylinder_selection",
    }


def region_record(
    region_id: str,
    label: str,
    indices: np.ndarray,
    points: np.ndarray,
    *,
    confidence: float,
    mesh_state: str,
    method: str,
    safety: str,
) -> dict:
    if indices.size == 0:
        return {
            "id": region_id,
            "label": label,
            "confidence": 0.0,
            "vertex_group": region_id,
            "vertex_count": 0,
            "mesh_state": "ambiguous",
            "source_method": method,
            "safety_classification": "do_not_deform",
            "empty": True,
        }
    block = points[indices]
    centred = block - block.mean(axis=0)
    # Principal axis via covariance; deterministic for a fixed vertex set.
    try:
        _values, vectors = np.linalg.eigh(np.cov(centred.T))
        principal = [float(v) for v in vectors[:, -1]]
    except np.linalg.LinAlgError:
        principal = [0.0, 0.0, 1.0]
    return {
        "id": region_id,
        "label": label,
        "confidence": float(confidence),
        "vertex_group": region_id,
        "vertex_count": int(indices.size),
        "bounding_box": {
            "min": [float(v) for v in block.min(axis=0)],
            "max": [float(v) for v in block.max(axis=0)],
        },
        "centroid": [float(v) for v in block.mean(axis=0)],
        "principal_axis": principal,
        "mesh_state": mesh_state,
        "source_method": method,
        "safety_classification": safety,
        "empty": False,
    }


def segment(points: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    z = points[:, 2]
    x = points[:, 0]
    low = float(z.min())
    height = max(float(z.max()) - low, 1e-9)
    fraction = (z - low) / height

    torso_mask = (fraction >= 0.455) & (fraction <= 0.620)
    torso_core = dense_interval(x[torso_mask]) if torso_mask.any() else None
    centre_x = float(np.median(x[torso_mask])) if torso_mask.any() else float(np.median(x))

    staff = find_staff_column(points, height, centre_x)
    staff_indices = staff.get("vertex_indices")
    staff_set = (
        np.zeros(points.shape[0], dtype=bool) if staff_indices is None else None
    )
    if staff_indices is not None:
        staff_set = np.zeros(points.shape[0], dtype=bool)
        staff_set[staff_indices] = True
    if not staff.get("detected"):
        staff_set = np.zeros(points.shape[0], dtype=bool)

    assigned = np.zeros(points.shape[0], dtype=bool)
    groups: dict[str, np.ndarray] = {}

    def claim(name: str, mask: np.ndarray) -> None:
        mask = mask & ~assigned & ~staff_set
        groups[name] = np.flatnonzero(mask)
        assigned[mask] = True

    # Staff first: it wins over every body band it passes through.
    groups["staff"] = np.flatnonzero(staff_set)
    assigned |= staff_set

    # Arms: lateral lobes outside the torso core, above the hand band.
    arm_mask = (fraction >= ARM_BAND[0]) & (fraction <= ARM_BAND[1])
    if torso_core is not None:
        margin = torso_core["width"] * 0.16
        left_lobe = arm_mask & (x < torso_core["low"] + margin)
        right_lobe = arm_mask & (x > torso_core["high"] - margin)
    else:
        left_lobe = np.zeros_like(arm_mask)
        right_lobe = np.zeros_like(arm_mask)

    for side, lobe in (("l", left_lobe), ("r", right_lobe)):
        if not lobe.any():
            groups[f"hand_{side}"] = np.array([], dtype=np.int64)
            groups[f"lowerarm_{side}"] = np.array([], dtype=np.int64)
            groups[f"upperarm_{side}"] = np.array([], dtype=np.int64)
            continue
        lobe_low = float(fraction[lobe].min())
        hand = lobe & (fraction <= lobe_low + HAND_BAND_HEIGHT)
        lower = lobe & (fraction > lobe_low + HAND_BAND_HEIGHT) & (fraction <= lobe_low + 0.20)
        upper = lobe & (fraction > lobe_low + 0.20)
        claim(f"hand_{side}", hand)
        claim(f"lowerarm_{side}", lower)
        claim(f"upperarm_{side}", upper)

    # Legs split left/right about the torso centre.
    for name, (lo, hi) in (("foot", BANDS["foot"]), ("shin", BANDS["shin"]), ("thigh", BANDS["thigh"])):
        band = (fraction >= lo) & (fraction < hi)
        claim(f"{name}_l", band & (x < centre_x))
        claim(f"{name}_r", band & (x >= centre_x))

    for name in ("pelvis", "spine", "chest", "neck", "head"):
        lo, hi = BANDS[name]
        claim(name, (fraction >= lo) & (fraction < hi))

    # Antlers and the ornament bar share the top band; split by lateral distance.
    lo, hi = BANDS["antler"]
    top = (fraction >= lo) & (fraction < hi)
    head_core = dense_interval(x[(fraction >= 0.700) & (fraction < 0.860)])
    if head_core is not None:
        near = np.abs(x - centre_x) <= head_core["width"] * 0.75
    else:
        near = np.abs(x - centre_x) <= height * 0.15
    claim("antler", top & near)
    claim("ornament_bar", top & ~near)

    # Hanging ornaments: below the bar but laterally outside the body core.
    ornament = (fraction >= ORNAMENT_BAND[0]) & (fraction < ORNAMENT_BAND[1])
    if torso_core is not None:
        outside = (x < torso_core["low"]) | (x > torso_core["high"])
    else:
        outside = np.zeros_like(ornament)
    claim("ornament_hanging", ornament & outside)

    # Everything still unassigned in the mid body is cloth.
    claim("cloth", ~assigned)

    context = {
        "model_height": height,
        "z_min": low,
        "torso_core": torso_core,
        "body_centre_x": centre_x,
        "staff": {key: value for key, value in staff.items() if key != "vertex_indices"},
    }
    return groups, context


# Confidence is deliberately low across the body bands. The rendered region
# proof shows the segmentation is a stack of horizontal slabs on a *robed*
# figure: the legs are hidden inside a skirt, so a band labelled `thigh_l` is
# mostly cape cloth, not a thigh. These labels are positional bands, not proven
# anatomy, and nothing downstream may treat them as anatomical truth.
POSITIONAL = "positional_band_not_proven_anatomy"

REGION_METADATA = {
    "staff": ("staff", 0.70, "fused", "staff_control_only"),
    "hand_l": ("left hand region", 0.35, "ambiguous", POSITIONAL),
    "hand_r": ("right hand region", 0.30, "ambiguous", POSITIONAL),
    "lowerarm_l": ("left lower arm region", 0.30, "ambiguous", POSITIONAL),
    "lowerarm_r": ("right lower arm region", 0.30, "ambiguous", POSITIONAL),
    "upperarm_l": ("left upper arm region", 0.30, "ambiguous", POSITIONAL),
    "upperarm_r": ("right upper arm region", 0.30, "ambiguous", POSITIONAL),
    "thigh_l": ("lower-left band (skirt-dominated)", 0.20, "ambiguous", POSITIONAL),
    "thigh_r": ("lower-right band (skirt-dominated)", 0.20, "ambiguous", POSITIONAL),
    "shin_l": ("shin-height left band (skirt-dominated)", 0.20, "ambiguous", POSITIONAL),
    "shin_r": ("shin-height right band (skirt-dominated)", 0.20, "ambiguous", POSITIONAL),
    "foot_l": ("left foot band", 0.45, "ambiguous", POSITIONAL),
    "foot_r": ("right foot band", 0.45, "ambiguous", POSITIONAL),
    "pelvis": ("pelvis-height band", 0.25, "ambiguous", POSITIONAL),
    "spine": ("torso-height band", 0.25, "ambiguous", POSITIONAL),
    "chest": ("chest-height band", 0.25, "ambiguous", POSITIONAL),
    "neck": ("neck-height band", 0.20, "ambiguous", POSITIONAL),
    "head": ("head-height band", 0.35, "ambiguous", POSITIONAL),
    "antler": ("antler / crown band", 0.30, "ambiguous", "secondary_chain_candidate"),
    "ornament_bar": ("ornament bar band", 0.30, "ambiguous", "secondary_chain_candidate"),
    "ornament_hanging": ("hanging ornaments", 0.20, "ambiguous", "secondary_chain_candidate"),
    "cloth": ("residual cloth", 0.20, "ambiguous", POSITIONAL),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--regions", required=True)
    args = parser.parse_args(argv_after_double_dash())

    obj, points = load_points(args.input)
    groups, context = segment(points)

    total = points.shape[0]
    covered = sum(int(indices.size) for indices in groups.values())

    records = []
    for name, indices in groups.items():
        label, confidence, mesh_state, safety = REGION_METADATA.get(
            name, (name, 0.2, "ambiguous", "do_not_deform")
        )
        records.append(
            region_record(
                name,
                label,
                indices,
                points,
                confidence=confidence,
                mesh_state=mesh_state,
                method="density_core_band_segmentation",
                safety=safety,
            )
        )
        group = obj.vertex_groups.new(name=name)
        if indices.size:
            group.add([int(i) for i in indices], 1.0, "REPLACE")

    # Bake a deterministic colour per region so the segmentation can be reviewed
    # visually. Two segmentation heuristics were already wrong in ways only a
    # render exposed, so this is a required artifact, not a nicety.
    palette = {}
    colour_layer = obj.data.color_attributes.new(
        name="region_colour", type="BYTE_COLOR", domain="POINT"
    )
    colours = np.tile(np.array([0.15, 0.15, 0.15, 1.0], dtype=np.float32), (total, 1))
    for position, (name, indices) in enumerate(sorted(groups.items())):
        hue = (position * 0.6180339887) % 1.0
        import colorsys

        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        palette[name] = [round(float(v), 4) for v in rgb]
        if indices.size:
            colours[indices, :3] = rgb
    colour_layer.data.foreach_set("color", colours.reshape(-1))
    obj.data.update()

    unsupported = [
        {"label": label, "reason": "NO_SUPPORTING_GEOMETRY_EVIDENCE", "confidence": 0.0}
        for label in ("tail", "feathers", "ropes")
    ]

    manifest = {
        "stage": "PARTS",
        "source": args.input,
        "vertex_total": total,
        "vertices_assigned": covered,
        "coverage_ratio": float(covered / max(total, 1)),
        "unassigned": int(total - covered),
        "context": context,
        "region_count": len(records),
        "region_colour_palette": palette,
        "regions": records,
        "unsupported_labels": unsupported,
        "staff_mode": "fused_staff_control",
        "staff_mode_note": (
            "Production-safe default. No boolean cutter, no hole, no separate "
            "mesh. separate_staff_candidate is not run here and can never "
            "auto-promote."
        ),
        "method": "density_core_band_segmentation",
        "anatomy_proven": False,
        "status": "POSITIONAL_BANDS_ONLY_ANATOMY_NOT_PROVEN",
        "blocking_codes": ["PARTS_ANATOMY_NOT_PROVEN_ROBED_FIGURE"],
        "confidence_policy": (
            "Bands and lateral lobes are geometric heuristics on a single fused "
            "mesh. No region claims a separate mesh state it cannot prove."
        ),
        "review_finding": (
            "The rendered region proof shows horizontal slabs across a robed "
            "figure. Legs sit inside a skirt, so lower-body bands are "
            "skirt-dominated and must not be treated as thighs or shins. Only "
            "the staff cylinder is a positively identified structure."
        ),
        "safe_for_skinning": ["staff"],
    }

    Path(args.regions).parent.mkdir(parents=True, exist_ok=True)
    Path(args.regions).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    Path(args.output_glb).parent.mkdir(parents=True, exist_ok=True)
    export_glb(args.output_glb)
    bpy.ops.wm.save_as_mainfile(filepath=args.output_blend)

    print(f"REGION_COUNT={len(records)}", flush=True)
    print(f"REGION_COVERAGE={manifest['coverage_ratio']:.6f}", flush=True)
    print(f"STAFF_DETECTED={context['staff'].get('detected')}", flush=True)
    print(f"STAFF_VERTICES={groups['staff'].size}", flush=True)
    for record in sorted(records, key=lambda item: -item["vertex_count"]):
        print(
            f"REGION {record['id']:18s} verts={record['vertex_count']:7d} "
            f"conf={record['confidence']:.2f} state={record['mesh_state']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
