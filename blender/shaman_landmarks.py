"""Landmark fitting for the robed shaman (PATH B).

Geometric body segmentation is REJECTED_NOT_OBSERVABLE: the figure wears a full
cape and skirt, so hidden arms and legs cannot be recovered from the outer
surface. This stage therefore fits only landmarks that are defensible from the
silhouette, the robe mass, the staff grip and anthropometric fallback.

Every joint records its position, confidence, source, uncertainty radius and
whether deformation QA is allowed to move it. Nothing here claims hidden limb
segmentation as proven.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash  # noqa: E402


# Fractions of *body* height (ground to skull top), standard humanoid, used only
# where geometry cannot observe the joint.
ANTHROPOMETRIC = {
    "ankle": 0.039,
    "knee": 0.285,
    "hip": 0.530,
    "waist": 0.630,
    "chest": 0.720,
    "shoulder": 0.818,
    "neck": 0.870,
}

SLICES = 128


def load(path: str):
    if Path(path).suffix.lower() != ".blend":
        raise RuntimeError("landmark fitting requires the welded .blend rig base")
    bpy.ops.wm.open_mainfile(filepath=path)
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one mesh, found {len(meshes)}")
    obj = meshes[0]
    count = len(obj.data.vertices)
    buffer = np.empty(count * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", buffer)
    local = buffer.reshape(-1, 3)
    matrix = np.array(obj.matrix_world.to_4x4())
    homogeneous = np.hstack([local, np.ones((count, 1), dtype=np.float32)])
    return obj, (matrix @ homogeneous.T).T[:, :3]


def dense_interval(values: np.ndarray, bins: int = 96, floor_ratio: float = 0.10):
    if values.size < 8:
        return None
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return {"low": low, "high": high, "width": 0.0, "centre": low}
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
        "centre": float((edges[left] + edges[right + 1]) * 0.5),
    }


def core_profile(points: np.ndarray) -> list[dict]:
    """Per-slice body-core interval, immune to the wide antler/ornament bar."""

    z = points[:, 2]
    low, high = float(z.min()), float(z.max())
    height = max(high - low, 1e-9)
    index = np.clip(((z - low) / height * SLICES).astype(np.int32), 0, SLICES - 1)
    profile = []
    for step in range(SLICES):
        mask = index == step
        if int(mask.sum()) < 30:
            profile.append({"slice": step, "vertex_count": int(mask.sum()), "core": None})
            continue
        block = points[mask]
        core_x = dense_interval(block[:, 0])
        core_y = dense_interval(block[:, 1])
        profile.append(
            {
                "slice": step,
                "fraction": float((step + 0.5) / SLICES),
                "z": float(low + (step + 0.5) / SLICES * height),
                "vertex_count": int(mask.sum()),
                "core": core_x,
                "core_y": core_y,
                "raw_width": float(block[:, 0].max() - block[:, 0].min()),
            }
        )
    return profile


def joint(
    name: str,
    position,
    *,
    confidence: float,
    source: str,
    uncertainty: float,
    adjustable: bool,
    note: str = "",
) -> dict:
    return {
        "name": name,
        "position": [float(v) for v in position],
        "confidence": float(confidence),
        "source": source,
        "uncertainty_radius": float(uncertainty),
        "adjustable_during_deformation_qa": bool(adjustable),
        "note": note,
    }


def fit(points: np.ndarray, staff: dict | None) -> dict:
    z = points[:, 2]
    ground = float(z.min())
    crown = float(z.max())
    total_height = crown - ground

    profile = core_profile(points)
    filled = [entry for entry in profile if entry.get("core")]

    # Global symmetry plane: median X of the dense body core across the torso.
    torso_slices = [e for e in filled if 0.35 <= e["fraction"] <= 0.65]
    symmetry_x = (
        float(np.median([e["core"]["centre"] for e in torso_slices]))
        if torso_slices
        else float(np.median(points[:, 0]))
    )
    torso_y = (
        float(np.median([e["core_y"]["centre"] for e in torso_slices if e.get("core_y")]))
        if torso_slices
        else float(np.median(points[:, 1]))
    )

    # Landmarks are read off the measured body-core width profile, not from
    # raw extents and not from human proportions applied to a stylised figure.
    #
    # The profile shows: a core minimum at the collar (neck), a widening below
    # it (cape shoulders), a strong flare where the skirt begins (hip), and a
    # second widening above the collar (head/beak). Raw width is useless above
    # f 0.55 because pendant ornaments hang from the antler bar.

    # Neck: narrowest body core in the collar window.
    neck_candidates = [
        e for e in filled if 0.72 <= e["fraction"] <= 0.88 and e["vertex_count"] >= 800
    ]
    neck_entry = min(neck_candidates, key=lambda e: e["core"]["width"]) if neck_candidates else None
    neck_z = neck_entry["z"] if neck_entry else ground + total_height * 0.80
    neck_width = neck_entry["core"]["width"] if neck_entry else total_height * 0.17

    # Head: slices above the neck that still carry mass; antler twigs are sparse.
    head_slices = [
        e for e in filled
        if e["z"] > neck_z and e["vertex_count"] >= 800 and e["fraction"] <= 0.96
    ]
    skull_top_z = max((e["z"] for e in head_slices), default=neck_z + total_height * 0.12)
    head_centre_z = (
        float(np.mean([e["z"] for e in head_slices])) if head_slices else (neck_z + skull_top_z) * 0.5
    )

    # Shoulders: first slice below the neck where the cape widens decisively.
    below_neck = sorted(
        [e for e in filled if e["z"] < neck_z and e["vertex_count"] >= 800],
        key=lambda e: -e["z"],
    )
    shoulder_entry = next(
        (e for e in below_neck if e["core"]["width"] >= neck_width * 1.45), None
    )
    if shoulder_entry is None:
        shoulder_entry = below_neck[0] if below_neck else None
    shoulder_z = shoulder_entry["z"] if shoulder_entry else neck_z - total_height * 0.06
    shoulder_width = shoulder_entry["core"]["width"] if shoulder_entry else total_height * 0.26
    shoulder_half = float(shoulder_width * 0.5)

    # Hip: where the skirt flares out from the torso, searching down from the
    # shoulder. This is the most reliable lower-body landmark on a robed figure.
    hip_entry = next(
        (
            e
            for e in below_neck
            if e["z"] < shoulder_z and e["core"]["width"] >= shoulder_width * 1.7
        ),
        None,
    )
    hip_z = hip_entry["z"] if hip_entry else ground + (shoulder_z - ground) * 0.42
    body_height = max(skull_top_z - ground, 1e-6)

    # Robe hem and foot contact: lowest slices carrying real mass.
    contact = sorted(
        [e for e in filled if e["vertex_count"] >= 800], key=lambda e: e["z"]
    )
    ankle_z = contact[0]["z"] if contact else ground + body_height * 0.04
    foot_core = contact[0]["core"] if contact else None
    foot_half = (foot_core["width"] * 0.25) if foot_core else body_height * 0.05
    hem_z = min(
        (e["z"] for e in filled if e["vertex_count"] >= 400 and e["z"] < hip_z),
        default=ground,
    )

    # Spine chain distributed between hip and neck; knee midway hip->ankle.
    waist_z = hip_z + (shoulder_z - hip_z) * 0.28
    chest_z = hip_z + (shoulder_z - hip_z) * 0.82
    knee_z = ankle_z + (hip_z - ankle_z) * 0.48
    hip_half = float(max(shoulder_half * 0.42, body_height * 0.05))

    # Arm placement must come from the measured lateral lobes, not from a
    # fraction of shoulder width. Scaling shoulder_half put the hand bones about
    # 0.2 units inboard of the visible hands - inside the robe - so the hands
    # did not move when the hand bone rotated.
    arm_band = (z >= ground + total_height * 0.26) & (z <= ground + total_height * 0.56)
    torso_core = dense_interval(points[:, 0][
        (z >= ground + total_height * 0.55) & (z <= ground + total_height * 0.72)
    ])
    lobes: dict[str, dict] = {}
    if torso_core is not None and arm_band.any():
        block = points[arm_band]
        margin = torso_core["width"] * 0.10
        for side, mask in (
            ("l", block[:, 0] < torso_core["low"] - margin),
            ("r", block[:, 0] > torso_core["high"] + margin),
        ):
            if int(mask.sum()) < 300:
                continue
            lobe = block[mask]
            distal = lobe[lobe[:, 2] <= lobe[:, 2].min() + total_height * 0.06]
            lobes[side] = {
                "vertex_count": int(mask.sum()),
                "x_mean": float(lobe[:, 0].mean()),
                "x_outer": float(lobe[:, 0].min() if side == "l" else lobe[:, 0].max()),
                "z_min": float(lobe[:, 2].min()),
                "z_max": float(lobe[:, 2].max()),
                "hand_centroid": [float(v) for v in distal.mean(axis=0)],
            }

    # The staff-holding side is often occluded by the staff and cape, so only one
    # lobe resolves. Mirror it about the symmetry plane rather than falling back
    # to a shoulder-width fraction, which put the hand bone inside the robe.
    if len(lobes) == 1:
        present = next(iter(lobes))
        other = "r" if present == "l" else "l"
        source = lobes[present]
        centroid = source["hand_centroid"]
        lobes[other] = {
            "vertex_count": 0,
            "x_mean": float(2.0 * symmetry_x - source["x_mean"]),
            "x_outer": float(2.0 * symmetry_x - source["x_outer"]),
            "z_min": source["z_min"],
            "z_max": source["z_max"],
            "hand_centroid": [float(2.0 * symmetry_x - centroid[0]), centroid[1], centroid[2]],
            "mirrored_from": present,
        }

    hand_z = hip_z - (hip_z - knee_z) * 0.25
    elbow_z = shoulder_z - (shoulder_z - hip_z) * 0.62

    def arm_x(side: str, sign: float, blend: float) -> float:
        shoulder_x = symmetry_x + sign * shoulder_half * 0.86
        lobe = lobes.get(side)
        if not lobe:
            return float(shoulder_x)
        return float(shoulder_x + (lobe["x_mean"] - shoulder_x) * blend)

    def hand_point(side: str, sign: float):
        lobe = lobes.get(side)
        if not lobe:
            return (symmetry_x + sign * shoulder_half * 0.94, torso_y, hand_z)
        centroid = lobe["hand_centroid"]
        return (centroid[0], centroid[1], centroid[2])

    joints = [
        joint("root", (symmetry_x, torso_y, ground), confidence=0.9,
              source="geometry_ground_plane", uncertainty=0.005, adjustable=False),
        joint("pelvis", (symmetry_x, torso_y, hip_z), confidence=0.55,
              source="silhouette_skirt_flare_onset", uncertainty=body_height * 0.05,
              adjustable=True, note="pelvis bone hidden; height inferred from robe flare"),
        joint("spine_01", (symmetry_x, torso_y, waist_z), confidence=0.5,
              source="interpolated_hip_to_shoulder", uncertainty=body_height * 0.04,
              adjustable=True),
        joint("spine_02", (symmetry_x, torso_y, (waist_z + chest_z) * 0.5), confidence=0.5,
              source="interpolated_hip_to_shoulder", uncertainty=body_height * 0.04,
              adjustable=True),
        joint("chest", (symmetry_x, torso_y, chest_z), confidence=0.55,
              source="interpolated_hip_to_shoulder", uncertainty=body_height * 0.035,
              adjustable=True),
        joint("neck", (symmetry_x, torso_y, neck_z), confidence=0.7,
              source="silhouette_core_minimum", uncertainty=body_height * 0.02, adjustable=True),
        joint("head", (symmetry_x, torso_y, head_centre_z), confidence=0.7,
              source="silhouette_head_mass_centre", uncertainty=body_height * 0.03,
              adjustable=False),
    ]

    for side, sign in (("l", -1.0), ("r", 1.0)):
        joints.extend([
            joint(f"clavicle_{side}", (symmetry_x + sign * shoulder_half * 0.32, torso_y, shoulder_z),
                  confidence=0.55, source="silhouette_cape_shoulder_widening",
                  uncertainty=body_height * 0.04, adjustable=True),
            joint(f"upperarm_{side}", (arm_x(side, sign, 0.0), torso_y,
                                       shoulder_z - body_height * 0.02),
                  confidence=0.5, source="silhouette_cape_shoulder_widening",
                  uncertainty=body_height * 0.05, adjustable=True,
                  note="sleeve exterior only; humerus not observable"),
            joint(f"lowerarm_{side}", (arm_x(side, sign, 0.65), torso_y, elbow_z),
                  confidence=0.4, source="measured_lateral_arm_lobe",
                  uncertainty=body_height * 0.06, adjustable=True, note="elbow hidden by sleeve"),
            joint(f"hand_{side}", hand_point(side, sign),
                  confidence=0.45, source="measured_distal_arm_lobe_centroid",
                  uncertainty=body_height * 0.05, adjustable=True),
            joint(f"thigh_{side}", (symmetry_x + sign * hip_half, torso_y, hip_z),
                  confidence=0.3, source="anthropometric_fallback_inside_robe",
                  uncertainty=body_height * 0.06, adjustable=True, note="hidden inside robe"),
            joint(f"calf_{side}", (symmetry_x + sign * hip_half, torso_y, knee_z),
                  confidence=0.3, source="anthropometric_fallback_inside_robe",
                  uncertainty=body_height * 0.06, adjustable=True, note="hidden inside robe"),
            joint(f"foot_{side}", (symmetry_x + sign * hip_half, torso_y, ankle_z),
                  confidence=0.45, source="lowest_mass_bearing_contact_slice",
                  uncertainty=body_height * 0.05, adjustable=True),
            joint(f"toe_{side}", (symmetry_x + sign * hip_half, torso_y - foot_half * 1.6, ground),
                  confidence=0.35, source="contact_slice_estimate", uncertainty=body_height * 0.04,
                  adjustable=True),
        ])

    # Staff grip: taken from the proven staff axis, never invented.
    grip = None
    if staff and staff.get("detected"):
        fit_x = staff["axis_x_slope_intercept"]
        fit_y = staff["axis_y_slope_intercept"]
        grip_z = hand_z
        grip = joint(
            "staff_grip",
            (float(np.polyval(fit_x, grip_z)), float(np.polyval(fit_y, grip_z)), grip_z),
            confidence=0.7,
            source="proven_staff_axis_at_hand_height",
            uncertainty=0.03,
            adjustable=False,
        )
        joints.append(grip)

    return {
        "stage": "LANDMARKS",
        "body_segmentation": "REJECTED_NOT_OBSERVABLE",
        "ground_z": ground,
        "crown_z": crown,
        "total_height_including_antlers": total_height,
        "skull_top_z": skull_top_z,
        "body_height_ground_to_skull": body_height,
        "symmetry_plane_x": symmetry_x,
        "torso_centreline_y": torso_y,
        "shoulder_z": shoulder_z,
        "shoulder_half_width": shoulder_half,
        "shoulder_core_width": shoulder_width,
        "neck_z": neck_z,
        "neck_core_width": neck_width,
        "head_centre_z": head_centre_z,
        "hip_z": hip_z,
        "waist_z": waist_z,
        "chest_z": chest_z,
        "knee_z": knee_z,
        "ankle_z": ankle_z,
        "hip_half_width": hip_half,
        "lateral_arm_lobes": lobes,
        "robe_hem_z": hem_z,
        "anthropometric_fractions": ANTHROPOMETRIC,
        "joint_count": len(joints),
        "joints": joints,
        "observable_joint_count": sum(1 for j in joints if j["confidence"] >= 0.6),
        "inferred_joint_count": sum(1 for j in joints if j["confidence"] < 0.6),
        "policy": (
            "Hidden limb joints are anthropometric fallback anchored on observed "
            "ground and skull-top landmarks. They are explicitly not proven "
            "anatomy and are marked adjustable so deformation QA can move them."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv_after_double_dash())

    _obj, points = load(args.input)
    regions = json.loads(Path(args.regions).read_text(encoding="utf-8"))
    staff = (regions.get("context") or {}).get("staff")

    result = fit(points, staff)
    result["source_mesh"] = args.input
    result["regions_manifest"] = args.regions

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"LANDMARK_COUNT={result['joint_count']}", flush=True)
    print(f"BODY_HEIGHT={result['body_height_ground_to_skull']:.4f}", flush=True)
    print(f"SYMMETRY_X={result['symmetry_plane_x']:.4f}", flush=True)
    print(f"SHOULDER_Z={result['shoulder_z']:.4f} HALF={result['shoulder_half_width']:.4f}", flush=True)
    print(f"NECK_Z={result['neck_z']:.4f} HIP_Z={result['hip_z']:.4f}", flush=True)
    print(f"OBSERVABLE={result['observable_joint_count']} INFERRED={result['inferred_joint_count']}", flush=True)
    print(f"STAFF_GRIP={'yes' if staff and staff.get('detected') else 'no'}", flush=True)


if __name__ == "__main__":
    main()
