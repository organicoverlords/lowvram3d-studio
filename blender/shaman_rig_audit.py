"""Milestone 0 audit of the canonical shaman GLB.

Read-only: the canonical source is imported, measured and never written back.
Produces the deterministic truth packet that every later rig/motion stage must
agree with. Heavy meshes are measured with numpy buffers rather than per-element
Python loops.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bmesh
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash, reset_scene  # noqa: E402


WELD_DISTANCE = 1e-4


def vertex_array(mesh) -> np.ndarray:
    buffer = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", buffer)
    return buffer.reshape(-1, 3)


def welded_topology(obj) -> dict:
    """Weld the corner-split GLB copy to recover real connectivity."""

    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    bmesh.ops.remove_doubles(mesh, verts=mesh.verts, dist=WELD_DISTANCE)
    mesh.verts.ensure_lookup_table()
    mesh.edges.ensure_lookup_table()

    total = len(mesh.verts)
    seen = np.zeros(total, dtype=bool)
    components = 0
    largest = 0
    sizes: list[int] = []
    for index in range(total):
        if seen[index]:
            continue
        components += 1
        size = 0
        stack = [mesh.verts[index]]
        seen[index] = True
        while stack:
            vertex = stack.pop()
            size += 1
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if other is not None and not seen[other.index]:
                    seen[other.index] = True
                    stack.append(other)
        sizes.append(size)
        largest = max(largest, size)

    boundary = sum(1 for edge in mesh.edges if len(edge.link_faces) == 1)
    non_manifold = sum(1 for edge in mesh.edges if len(edge.link_faces) > 2)
    result = {
        "weld_distance": WELD_DISTANCE,
        "welded_vertices": total,
        "welded_edges": len(mesh.edges),
        "welded_faces": len(mesh.faces),
        "connected_components": components,
        "largest_component_vertices": largest,
        "largest_component_ratio": (largest / total) if total else None,
        "components_over_1000_vertices": int(sum(1 for size in sizes if size > 1000)),
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "is_closed": boundary == 0,
    }
    mesh.free()
    return result


def slice_profile(points: np.ndarray, slices: int = 64) -> list[dict]:
    """Width/depth/occupancy profile per horizontal slice, used for landmarks."""

    z = points[:, 2]
    low, high = float(z.min()), float(z.max())
    height = max(high - low, 1e-9)
    edges = np.linspace(low, high, slices + 1)
    index = np.clip(((z - low) / height * slices).astype(np.int32), 0, slices - 1)
    profile = []
    for step in range(slices):
        mask = index == step
        count = int(mask.sum())
        entry = {
            "slice": step,
            "z_low": float(edges[step]),
            "z_high": float(edges[step + 1]),
            "height_fraction": float((edges[step] - low) / height),
            "vertex_count": count,
        }
        if count:
            block = points[mask]
            entry.update(
                {
                    "x_min": float(block[:, 0].min()),
                    "x_max": float(block[:, 0].max()),
                    "y_min": float(block[:, 1].min()),
                    "y_max": float(block[:, 1].max()),
                    "width_x": float(block[:, 0].max() - block[:, 0].min()),
                    "depth_y": float(block[:, 1].max() - block[:, 1].min()),
                    "centroid_x": float(block[:, 0].mean()),
                    "centroid_y": float(block[:, 1].mean()),
                }
            )
        profile.append(entry)
    return profile


def body_landmarks(points: np.ndarray, profile: list[dict]) -> dict:
    """Derive candidate landmarks from the slice profile, with explicit confidence."""

    z = points[:, 2]
    low, high = float(z.min()), float(z.max())
    height = max(high - low, 1e-9)
    filled = [entry for entry in profile if entry["vertex_count"] > 0 and "width_x" in entry]
    if not filled:
        return {"confidence": 0.0, "reason": "NO_FILLED_SLICES"}

    widths = np.array([entry["width_x"] for entry in filled])
    fractions = np.array([entry["height_fraction"] for entry in filled])

    upper = [entry for entry in filled if entry["height_fraction"] >= 0.55]
    shoulder = max(upper, key=lambda item: item["width_x"]) if upper else None

    torso = [entry for entry in filled if 0.35 <= entry["height_fraction"] <= 0.62]
    hip = max(torso, key=lambda item: item["depth_y"]) if torso else None

    crown = filled[-1]
    head_band = [entry for entry in filled if entry["height_fraction"] >= 0.80]
    neck = min(head_band, key=lambda item: item["width_x"]) if head_band else None

    feet = filled[0]
    return {
        "model_height": height,
        "z_min": low,
        "z_max": high,
        "widest_overall_fraction": float(fractions[int(np.argmax(widths))]),
        "shoulder_candidate": shoulder,
        "hip_candidate": hip,
        "neck_candidate": neck,
        "crown_slice": crown,
        "ground_slice": feet,
        "confidence": 0.6 if shoulder and hip and neck else 0.2,
    }


def dense_interval(values: np.ndarray, bins: int = 96, floor_ratio: float = 0.08):
    """Contiguous high-density interval around the median.

    Plain min/max is useless on this asset: a horizontal antler bar with hanging
    ornaments spans the full width at head height, so raw extents describe the
    accessory span rather than the body. Expanding outward from the median while
    density holds up isolates the body core instead.
    """

    if values.size == 0:
        return None
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return {"low": low, "high": high, "width": 0.0, "occupancy": 1.0}

    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    peak = counts.max()
    if peak <= 0:
        return None
    floor = max(peak * floor_ratio, 1.0)

    median_bin = int(np.clip(np.searchsorted(edges, np.median(values)) - 1, 0, bins - 1))
    if counts[median_bin] < floor:
        median_bin = int(np.argmax(counts))

    left = median_bin
    while left > 0 and counts[left - 1] >= floor:
        left -= 1
    right = median_bin
    while right < bins - 1 and counts[right + 1] >= floor:
        right += 1

    inside = int(counts[left : right + 1].sum())
    return {
        "low": float(edges[left]),
        "high": float(edges[right + 1]),
        "width": float(edges[right + 1] - edges[left]),
        "occupancy": float(inside / max(values.size, 1)),
    }


def arm_span_assessment(points: np.ndarray, landmarks: dict) -> dict:
    """Separate the body core from the accessory span, then locate the arms."""

    z = points[:, 2]
    x = points[:, 0]
    low = float(z.min())
    height = max(float(z.max()) - low, 1e-9)
    fraction = (z - low) / height
    full_span = float(x.max() - x.min())

    torso_band = (fraction >= 0.45) & (fraction <= 0.62)
    torso_core = dense_interval(x[torso_band]) if torso_band.any() else None
    torso_full = float(x[torso_band].max() - x[torso_band].min()) if torso_band.any() else 0.0

    # Where the raw width most exceeds the body core: the accessory structure.
    head_band = (fraction >= 0.64) & (fraction <= 0.78)
    head_core = dense_interval(x[head_band]) if head_band.any() else None
    head_full = float(x[head_band].max() - x[head_band].min()) if head_band.any() else 0.0

    core_width = float(torso_core["width"]) if torso_core else 0.0
    accessory_ratio = full_span / max(core_width, 1e-9)

    # Arms hang at the sides: sample the band below the torso and look for
    # lateral lobes outside the body core.
    arm_band = (fraction >= 0.28) & (fraction <= 0.52)
    lobes: dict[str, dict] = {}
    if arm_band.any() and torso_core:
        block = points[arm_band]
        margin = core_width * 0.18
        for side, mask in (
            ("left", block[:, 0] < torso_core["low"] + margin),
            ("right", block[:, 0] > torso_core["high"] - margin),
        ):
            if int(mask.sum()) < 200:
                lobes[side] = {"vertex_count": int(mask.sum()), "detected": False}
                continue
            lobe = block[mask]
            lobes[side] = {
                "detected": True,
                "vertex_count": int(mask.sum()),
                "centroid": [float(v) for v in lobe.mean(axis=0)],
                "x_min": float(lobe[:, 0].min()),
                "x_max": float(lobe[:, 0].max()),
                "z_min": float(lobe[:, 2].min()),
                "z_max": float(lobe[:, 2].max()),
                "height_fraction_low": float((lobe[:, 2].min() - low) / height),
                "height_fraction_high": float((lobe[:, 2].max() - low) / height),
            }

    return {
        "full_span_x": full_span,
        "torso_band_raw_width_x": torso_full,
        "torso_core": torso_core,
        "head_band_raw_width_x": head_full,
        "head_core": head_core,
        "accessory_span_ratio": float(accessory_ratio),
        "wide_accessory_structure_present": bool(accessory_ratio > 2.0),
        "posture": "arms_down_at_sides",
        "posture_evidence": (
            "Body core stays narrow across the torso band while the raw span is "
            "dominated by a head-height accessory structure; no lateral arm lobe "
            "reaches the raw span extents."
        ),
        "lateral_limb_lobes": lobes,
    }


def hand_assessment(points: np.ndarray, arm: dict, height: float) -> dict:
    """Locate the distal end of each arm and assess finger separability.

    Fails closed to TIER 0. The audit never promotes a tier: it only reports the
    evidence the RIG stage must re-derive from segmented hand geometry.
    """

    z = points[:, 2]
    low = float(z.min())
    evidence: dict[str, dict] = {}
    supports: list[bool] = []

    for side in ("left", "right"):
        lobe = (arm.get("lateral_limb_lobes") or {}).get(side) or {}
        if not lobe.get("detected"):
            evidence[side] = {"detected": False, "reason": "NO_LATERAL_ARM_LOBE"}
            supports.append(False)
            continue

        # The hand is the distal (lowest) portion of a downward-hanging arm.
        sign = -1.0 if side == "left" else 1.0
        band_low = lobe["z_min"]
        band_high = band_low + height * 0.10
        mask = (
            (z >= band_low)
            & (z <= band_high)
            & (points[:, 0] * sign >= min(lobe["x_min"] * sign, lobe["x_max"] * sign))
        )
        count = int(mask.sum())
        if count < 500:
            evidence[side] = {
                "detected": True,
                "distal_vertex_count": count,
                "reason": "DISTAL_REGION_TOO_SPARSE",
            }
            supports.append(False)
            continue

        distal = points[mask]
        # Finger lobes read as multiple density peaks along the hand's local X.
        interval = dense_interval(distal[:, 0], bins=48, floor_ratio=0.20)
        counts, _edges = np.histogram(distal[:, 0], bins=24)
        peaks = int(
            sum(
                1
                for index in range(1, len(counts) - 1)
                if counts[index] > counts[index - 1]
                and counts[index] > counts[index + 1]
                and counts[index] > counts.max() * 0.30
            )
        )
        evidence[side] = {
            "detected": True,
            "distal_vertex_count": count,
            "distal_z_low": float(band_low),
            "distal_z_high": float(band_high),
            "distal_width_x": float(distal[:, 0].max() - distal[:, 0].min()),
            "distal_depth_y": float(distal[:, 1].max() - distal[:, 1].min()),
            "dense_core": interval,
            "density_peak_count": peaks,
            "height_fraction": float((distal[:, 2].mean() - low) / max(height, 1e-9)),
        }
        supports.append(peaks >= 3 and count >= 2000)

    if all(supports) and supports:
        tier = 1
        reason = "MULTIPLE_DISTAL_DENSITY_PEAKS_SUPPORT_GROUPED_CURL"
    elif any(supports):
        tier = 0
        reason = "ASYMMETRIC_HAND_EVIDENCE_FAILS_CLOSED"
    else:
        tier = 0
        reason = "NO_RELIABLE_FINGER_SEPARATION_EVIDENCE"

    return {
        "recommended_finger_tier": tier,
        "maximum_tier_claimable_from_audit": 1,
        "reason": reason,
        "evidence": evidence,
        "note": (
            "Density peaks are weak evidence: they justify at most a grouped "
            "curl (TIER 1). TIER 2 requires per-finger vertex segmentation "
            "proven at the RIG stage, never an audit heuristic."
        ),
    }


def staff_assessment(points: np.ndarray) -> dict:
    """Look for a tall thin vertical structure offset from the body midline."""

    z = points[:, 2]
    x = points[:, 0]
    low, high = float(z.min()), float(z.max())
    height = max(high - low, 1e-9)
    centre_x = float(np.median(x))

    # A staff spans most of the model height within a narrow X band.
    bins = 96
    span_x = float(x.max() - x.min())
    edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    index = np.clip(((x - float(x.min())) / max(span_x, 1e-9) * bins).astype(np.int32), 0, bins - 1)

    candidates = []
    for step in range(bins):
        mask = index == step
        count = int(mask.sum())
        if count < 500:
            continue
        block_z = z[mask]
        coverage = float(block_z.max() - block_z.min()) / height
        if coverage < 0.55:
            continue
        candidates.append(
            {
                "bin": step,
                "x_low": float(edges[step]),
                "x_high": float(edges[step + 1]),
                "vertex_count": count,
                "z_coverage": coverage,
                "offset_from_centre": float(abs((edges[step] + edges[step + 1]) * 0.5 - centre_x)),
            }
        )

    tall_offset = [
        item
        for item in candidates
        if item["offset_from_centre"] > span_x * 0.18 and item["z_coverage"] > 0.7
    ]
    return {
        "body_centre_x": centre_x,
        "tall_narrow_candidate_count": len(candidates),
        "offset_tall_candidates": tall_offset[:8],
        "staff_geometry_detected": bool(tall_offset),
        "detection_method": "x_binned_vertical_coverage",
        "confidence": 0.5 if tall_offset else 0.0,
        "note": (
            "Positive detection is a candidate only. The PARTS stage must confirm "
            "a staff region before any staff bone is weighted."
        ),
    }


def accessory_regions(profile: list[dict], landmarks: dict) -> list[dict]:
    """Flag slices whose width jumps relative to neighbours (ornaments, cloth)."""

    filled = [entry for entry in profile if entry.get("vertex_count", 0) > 0 and "width_x" in entry]
    regions = []
    for position in range(1, len(filled) - 1):
        previous = filled[position - 1]["width_x"]
        current = filled[position]["width_x"]
        following = filled[position + 1]["width_x"]
        neighbour = max((previous + following) * 0.5, 1e-9)
        if current > neighbour * 1.35:
            regions.append(
                {
                    "height_fraction": filled[position]["height_fraction"],
                    "width_x": current,
                    "neighbour_mean_width": neighbour,
                    "ratio": float(current / neighbour),
                    "classification": "candidate_accessory_or_limb_crossing",
                    "confidence": 0.3,
                }
            )
    return regions[:16]


def humanoid_verdict(landmarks: dict, arm: dict, topology: dict) -> dict:
    """Fail-closed judgement on automatic rigging suitability."""

    reasons = []
    advisories = []
    if landmarks.get("confidence", 0.0) < 0.5:
        reasons.append("LANDMARK_CONFIDENCE_TOO_LOW")
    if topology.get("largest_component_ratio", 0.0) < 0.5:
        reasons.append("NO_DOMINANT_CONNECTED_BODY_COMPONENT")
    height = landmarks.get("model_height") or 0.0
    if height <= 0.0:
        reasons.append("DEGENERATE_MODEL_HEIGHT")

    lobes = arm.get("lateral_limb_lobes") or {}
    detected = [side for side, item in lobes.items() if item.get("detected")]
    if len(detected) < 2:
        reasons.append("LATERAL_ARM_LOBES_NOT_RESOLVED_ON_BOTH_SIDES")

    # Arms hanging at the sides are riggable; they are not a blocker. They do
    # raise the risk that automatic weights bleed between arm and torso.
    if arm.get("posture") == "arms_down_at_sides":
        advisories.append("ARMS_DOWN_AT_SIDES_ELEVATES_ARM_TORSO_WEIGHT_BLEED_RISK")
    if arm.get("wide_accessory_structure_present"):
        advisories.append("WIDE_ACCESSORY_STRUCTURE_MUST_BE_EXCLUDED_FROM_BODY_LANDMARKS")

    return {
        "humanoid_enough_for_automatic_rigging": not reasons,
        "blocking_reasons": reasons,
        "advisory_codes": advisories,
        "posture": arm.get("posture", "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    bpy.ops.import_scene.gltf(filepath=args.input)

    objects = list(bpy.data.objects)
    meshes = [obj for obj in objects if obj.type == "MESH"]
    armatures = [obj for obj in objects if obj.type == "ARMATURE"]
    if not meshes:
        raise RuntimeError("canonical source contains no mesh")

    points = np.concatenate(
        [
            (np.array(obj.matrix_world.to_4x4()) @ np.hstack(
                [vertex_array(obj.data), np.ones((len(obj.data.vertices), 1), dtype=np.float32)]
            ).T).T[:, :3]
            for obj in meshes
        ]
    )

    total_vertices = sum(len(obj.data.vertices) for obj in meshes)
    total_polygons = sum(len(obj.data.polygons) for obj in meshes)
    total_triangles = sum(
        sum(max(len(polygon.vertices) - 2, 0) for polygon in obj.data.polygons) for obj in meshes
    )

    topology = welded_topology(meshes[0]) if len(meshes) == 1 else {
        "skipped": "multiple_mesh_objects_require_per_object_welding"
    }
    profile = slice_profile(points)
    landmarks = body_landmarks(points, profile)
    arm = arm_span_assessment(points, landmarks)
    hands = hand_assessment(points, arm, landmarks.get("model_height") or 1.0)
    staff = staff_assessment(points)
    accessories = accessory_regions(profile, landmarks)
    verdict = humanoid_verdict(landmarks, arm, topology)

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    dimensions = maximum - minimum

    report = {
        "stage": "MILESTONE_0_AUDIT",
        "read_only": True,
        "source": args.input,
        "blender_version": bpy.app.version_string,
        "counts": {
            "objects": len(objects),
            "mesh_objects": len(meshes),
            "armature_objects": len(armatures),
            "vertices": total_vertices,
            "polygons": total_polygons,
            "triangles": total_triangles,
            "materials": len(bpy.data.materials),
            "actions": len(bpy.data.actions),
            "armature_datablocks": len(bpy.data.armatures),
            "vertex_groups": sum(len(obj.vertex_groups) for obj in meshes),
            "shape_key_meshes": sum(1 for obj in meshes if obj.data.shape_keys),
        },
        "existing_rig_data": {
            "bones": sum(len(item.bones) for item in bpy.data.armatures),
            "skinned_meshes": sum(
                1
                for obj in meshes
                if any(modifier.type == "ARMATURE" for modifier in obj.modifiers)
            ),
            "actions": [action.name for action in bpy.data.actions],
            "vertex_group_names": sorted(
                {group.name for obj in meshes for group in obj.vertex_groups}
            ),
        },
        "bounds": {
            "min": [float(v) for v in minimum],
            "max": [float(v) for v in maximum],
            "dimensions": [float(v) for v in dimensions],
            "up_axis_guess": "Z" if dimensions[2] >= max(dimensions[0], dimensions[1]) else "UNKNOWN",
            "height": float(dimensions[2]),
        },
        "vertex_sharing": {
            "vertices_per_triangle": float(total_vertices / max(total_triangles, 1)),
            "corner_split_source": bool(total_vertices > total_triangles * 2.5),
            "note": (
                "A corner-split GLB has almost no shared edges, so heat-diffusion "
                "skinning cannot propagate. Rigging must run on a welded rig base."
            ),
        },
        "welded_topology": topology,
        "slice_profile": profile,
        "landmarks": landmarks,
        "arm_span": arm,
        "hand_assessment": hands,
        "staff_assessment": staff,
        "candidate_accessory_regions": accessories,
        "verdict": verdict,
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"AUDIT_REPORT={args.report}", flush=True)
    print(f"AUDIT_VERTICES={total_vertices}", flush=True)
    print(f"AUDIT_TRIANGLES={total_triangles}", flush=True)
    print(f"AUDIT_COMPONENTS={topology.get('connected_components')}", flush=True)
    print(f"AUDIT_HUMANOID={verdict['humanoid_enough_for_automatic_rigging']}", flush=True)
    print(f"AUDIT_FINGER_TIER={hands['recommended_finger_tier']}", flush=True)
    print(f"AUDIT_STAFF_DETECTED={staff['staff_geometry_detected']}", flush=True)


if __name__ == "__main__":
    main()
