"""PATH B: landmark-driven proxy anatomy and robe-first skinning.

Body segmentation is REJECTED_NOT_OBSERVABLE. This stage builds a game skeleton
from fitted landmarks, adds non-rendering internal capsule proxies, and skins the
visible robe with spatial influence volumes instead of per-vertex limb
classification.

Key policy, enforced in code:

* lower-robe vertices are never bound to individual leg bones - they follow the
  pelvis plus cloth-bone chains, so the skirt cannot split into leg-shaped lobes;
* leg bones still exist for animation compatibility, retargeting and collision
  proxies, and carry zero skin weight by design;
* staff vertices are bound to the staff deform bone only, in fused-staff-control
  mode - no cutting, no separate mesh;
* the bind pose is the source pose. No A-pose is forced on fused robe geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash  # noqa: E402
from shaman_semantic_v3 import semantic_masks_v3  # noqa: E402

MAX_INFLUENCES = 4
CLOTH_CHAINS = (("f", 0.0, -1.0), ("b", 0.0, 1.0), ("l", -1.0, 0.0), ("r", 1.0, 0.0))
CLOTH_SEGMENTS = 3

SOCKETS = (
    ("root_fx", "root", (0.0, 0.0, 0.02)),
    ("head_fx", "head", (0.0, 0.0, 0.06)),
    ("hand_l_fx", "hand_l", (0.0, 0.0, 0.0)),
    ("hand_r_fx", "hand_r", (0.0, 0.0, 0.0)),
    ("foot_l_fx", "foot_l", (0.0, 0.0, 0.0)),
    ("foot_r_fx", "foot_r", (0.0, 0.0, 0.0)),
    ("hand_l_socket", "hand_l", (0.0, 0.0, 0.0)),
    ("hand_r_socket", "hand_r", (0.0, 0.0, 0.0)),
    ("staff_socket", "staff_deform", (0.0, 0.0, 0.0)),
    ("staff_tip_fx", "staff_deform", (0.0, 0.0, 0.0)),
)


def load_mesh(path: str):
    bpy.ops.wm.open_mainfile(filepath=path)
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one mesh, found {len(meshes)}")
    return meshes[0]


def world_points(obj) -> np.ndarray:
    count = len(obj.data.vertices)
    buffer = np.empty(count * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", buffer)
    local = buffer.reshape(-1, 3)
    matrix = np.array(obj.matrix_world.to_4x4())
    return (matrix @ np.hstack([local, np.ones((count, 1), dtype=np.float32)]).T).T[:, :3]


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    length_sq = float(ab @ ab)
    if length_sq < 1e-12:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / length_sq, 0.0, 1.0)
    projection = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - projection, axis=1)


def build_skeleton(landmarks: dict, staff: dict):
    joints = {item["name"]: np.array(item["position"], dtype=np.float64) for item in landmarks["joints"]}
    ground = landmarks["ground_z"]
    hip_z = landmarks["hip_z"]
    hem_z = landmarks["robe_hem_z"]
    symmetry = landmarks["symmetry_plane_x"]
    torso_y = landmarks["torso_centreline_y"]
    body = landmarks["body_height_ground_to_skull"]

    data = bpy.data.armatures.new("shaman_rig")
    armature = bpy.data.objects.new("shaman_rig", data)
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit = data.edit_bones

    created: dict[str, object] = {}
    spec: list[tuple] = []

    def add(name, head, tail, parent, deform=True):
        spec.append((name, np.asarray(head, dtype=np.float64), np.asarray(tail, dtype=np.float64), parent, deform))

    add("root", joints["root"], joints["root"] + np.array([0, 0, body * 0.06]), None)
    add("pelvis", joints["pelvis"], joints["spine_01"], "root")
    add("spine_01", joints["spine_01"], joints["spine_02"], "pelvis")
    add("spine_02", joints["spine_02"], joints["chest"], "spine_01")
    add("chest", joints["chest"], joints["neck"], "spine_02")
    add("neck", joints["neck"], joints["neck"] + np.array([0, 0, body * 0.04]), "chest")
    add("head", joints["neck"] + np.array([0, 0, body * 0.04]), joints["head"], "neck")

    for side in ("l", "r"):
        add(f"clavicle_{side}", joints["chest"], joints[f"upperarm_{side}"], "chest")
        add(f"upperarm_{side}", joints[f"upperarm_{side}"], joints[f"lowerarm_{side}"], f"clavicle_{side}")
        add(f"lowerarm_{side}", joints[f"lowerarm_{side}"], joints[f"hand_{side}"], f"upperarm_{side}")
        add(f"hand_{side}", joints[f"hand_{side}"],
            joints[f"hand_{side}"] + np.array([0, 0, -body * 0.05]), f"lowerarm_{side}")
        add(f"thigh_{side}", joints[f"thigh_{side}"], joints[f"calf_{side}"], "pelvis")
        add(f"calf_{side}", joints[f"calf_{side}"], joints[f"foot_{side}"], f"thigh_{side}")
        add(f"foot_{side}", joints[f"foot_{side}"], joints[f"toe_{side}"], f"calf_{side}")
        add(f"toe_{side}", joints[f"toe_{side}"],
            joints[f"toe_{side}"] + np.array([0, -body * 0.03, 0]), f"foot_{side}")

    # Cloth chains hang from the pelvis to the robe hem at four azimuths.
    radius = landmarks["hip_half_width"] * 1.35
    drop = (hip_z - hem_z) / CLOTH_SEGMENTS
    for tag, dx, dy in CLOTH_CHAINS:
        parent = "pelvis"
        for segment in range(CLOTH_SEGMENTS):
            top = np.array([symmetry + dx * radius, torso_y + dy * radius, hip_z - drop * segment])
            bottom = np.array([symmetry + dx * radius, torso_y + dy * radius, hip_z - drop * (segment + 1)])
            name = f"cloth_{tag}_{segment + 1:02d}"
            add(name, top, bottom, parent)
            parent = name

    # Free-side sleeve cloth chain. The drape is three times the volume of the
    # anatomical arm, so it needs its own bones; binding it to the arm chain is
    # what produced the rigid triangular wing.
    upper_r = joints["upperarm_r"]
    lower_r = joints["lowerarm_r"]
    hand_r = joints["hand_r"]
    add("sleeve_r_anchor", upper_r, lower_r, "upperarm_r")
    drop = max((lower_r[2] - hand_r[2]) * 0.65, body * 0.05)
    previous = "sleeve_r_anchor"
    origin = np.array([lower_r[0], lower_r[1], lower_r[2]])
    for index in range(1, 4):
        top = origin + np.array([0.0, 0.0, -drop * (index - 1)])
        bottom = origin + np.array([0.0, 0.0, -drop * index])
        name = f"sleeve_r_drape_{index:02d}"
        add(name, top, bottom, previous)
        previous = name

    # Staff deform bone follows the proven staff axis.
    if staff.get("detected"):
        fit_x = staff["axis_x_slope_intercept"]
        fit_y = staff["axis_y_slope_intercept"]
        z_low = ground
        z_high = ground + landmarks["total_height_including_antlers"] * staff["z_coverage"]
        add("staff_deform",
            [float(np.polyval(fit_x, z_low)), float(np.polyval(fit_y, z_low)), z_low],
            [float(np.polyval(fit_x, z_high)), float(np.polyval(fit_y, z_high)), z_high],
            "root")

    for name, head, tail, parent, deform in spec:
        bone = edit.new(name)
        bone.head = Vector(head)
        bone.tail = Vector(tail)
        if np.linalg.norm(tail - head) < 1e-5:
            bone.tail = Vector(head + np.array([0.0, 0.0, 0.01]))
        bone.use_deform = deform
        created[name] = bone
    for name, _h, _t, parent, _d in spec:
        if parent:
            created[name].parent = created[parent]

    # Sockets: non-deforming child bones with recorded local transforms.
    socket_records = []
    for socket, parent, offset in SOCKETS:
        if parent not in created:
            socket_records.append({
                "socket": socket, "parent_bone": parent, "enabled": False,
                "reason": "PARENT_BONE_ABSENT", "confidence": 0.0,
            })
            continue
        anchor = created[parent]
        bone = edit.new(socket)
        base = np.array(anchor.tail) if socket == "staff_tip_fx" else np.array(anchor.head)
        bone.head = Vector(base + np.array(offset))
        bone.tail = Vector(base + np.array(offset) + np.array([0.0, 0.0, 0.03]))
        bone.parent = anchor
        bone.use_deform = False
        created[socket] = bone
        socket_records.append({
            "socket": socket,
            "parent_bone": parent,
            "local_position": [float(v) for v in offset],
            "local_rotation": [0.0, 0.0, 0.0],
            "local_scale": [1.0, 1.0, 1.0],
            "effect_type": "attachment" if socket.endswith("socket") else "vfx",
            "enabled": True,
            "confidence": 0.7 if parent in {"root", "head", "staff_deform"} else 0.4,
            "reason": None,
        })

    bpy.ops.object.mode_set(mode="OBJECT")
    return armature, spec, socket_records


def compute_weights(points: np.ndarray, spec, landmarks: dict, staff_mask: np.ndarray, masks: dict):
    """Robe-first spatial influence volumes.

    Returns {bone_name: (indices, weights)}. Leg bones are deliberately absent:
    lower-robe vertices follow pelvis and cloth chains so the skirt cannot be
    torn into leg-shaped lobes by per-vertex classification.
    """

    bones = {name: (head, tail) for name, head, tail, _p, deform in spec if deform}
    symmetry = landmarks["symmetry_plane_x"]
    hip_z = landmarks["hip_z"]
    neck_z = landmarks["neck_z"]
    shoulder_z = landmarks["shoulder_z"]
    shoulder_half = landmarks["shoulder_half_width"]
    hem_z = landmarks["robe_hem_z"]
    body = landmarks["body_height_ground_to_skull"]
    torso_y = landmarks["torso_centreline_y"]

    z = points[:, 2]
    x = points[:, 0]
    count = points.shape[0]
    weights = {}

    def accumulate(name, mask, value):
        if name not in weights:
            weights[name] = np.zeros(count, dtype=np.float64)
        weights[name][mask] += value[mask] if isinstance(value, np.ndarray) else value

    # 1. Staff: exclusive.
    accumulate("staff_deform", staff_mask, 1.0)
    free = ~staff_mask

    # 2. Sleeves: arm bones may only reach vertices the semantic masks classify
    #    as sleeve or hand. The previous version used a raw lateral band plus a
    #    distance falloff and then removed every touched vertex from the torso
    #    blend, so a cape vertex grazed by the arm volume with influence 0.001
    #    normalised to 100% arm weight - that is what produced the fan-shaped
    #    cape slabs. Arm reach is now bounded by classification, not distance.
    sleeve_any = np.zeros(count, dtype=bool)
    arm_share = np.zeros(count, dtype=np.float64)
    for side in ("l", "r"):
        chain_available = f"sleeve_{side}_anchor" in bones
        allowed = (
            (masks[f"arm_core_{side}"] | masks[f"hand_core_{side}"])
            if chain_available
            else (masks[f"sleeve_{side}"] | masks[f"hand_{side}_region"])
        )
        if not allowed.any():
            continue
        chain = [f"clavicle_{side}", f"upperarm_{side}", f"lowerarm_{side}", f"hand_{side}"]
        radius = float(masks["sleeve_radius"])
        radii = {
            f"clavicle_{side}": radius * 1.10,
            f"upperarm_{side}": radius * 1.10,
            f"lowerarm_{side}": radius * 1.10,
            f"hand_{side}": radius * 1.30,
        }
        for name in chain:
            head, tail = bones[name]
            distance = segment_distance(points, head, tail)
            influence = np.clip(1.0 - distance / max(radii[name], 1e-6), 0.0, 1.0) ** 2
            mask = allowed & (influence > 0.0)
            accumulate(name, mask, influence)
            sleeve_any |= mask
            arm_share[mask] += influence[mask]
    arm_share = np.clip(arm_share, 0.0, 1.0)

    # The garment is a separate semantic carrier. The hand must not drive the
    # broad hanging panel; anchor and drape vertices receive only the small
    # sleeve chain, with distance-weighted overlap between adjacent links.
    sleeve_garment = np.zeros(count, dtype=bool)
    for side in ("l", "r"):
        anchor_mask = masks[f"sleeve_anchor_{side}"] & free
        drape_mask = masks[f"sleeve_drape_{side}"] & free
        chain_available = f"sleeve_{side}_anchor" in bones
        if not chain_available:
            continue
        sleeve_garment |= anchor_mask | drape_mask
        anchor_name = f"sleeve_{side}_anchor"
        if anchor_name in bones:
            head, tail = bones[anchor_name]
            influence = np.clip(1.0 - segment_distance(points, head, tail) / max(float(masks["sleeve_radius"]), 1e-6), 0.0, 1.0) ** 2
            accumulate(anchor_name, anchor_mask, np.maximum(influence, anchor_mask.astype(np.float64) * 0.25))
        drape_bones = [f"sleeve_{side}_drape_{index:02d}" for index in range(1, 4) if f"sleeve_{side}_drape_{index:02d}" in bones]
        drape_distances = np.stack([
            segment_distance(points, bones[name][0], bones[name][1]) for name in drape_bones
        ], axis=1)
        nearest = np.argmin(drape_distances, axis=1)
        for index in range(1, 4):
            name = f"sleeve_{side}_drape_{index:02d}"
            if name not in bones:
                continue
            head, tail = bones[name]
            influence = np.clip(1.0 - segment_distance(points, head, tail) / max(float(masks["sleeve_radius"]) * 1.5, 1e-6), 0.0, 1.0) ** 2
            band = drape_mask & (np.array(drape_bones)[nearest] == name)
            accumulate(name, band, np.maximum(influence, band.astype(np.float64) * 0.25))

    # Do not let height-only robe rules reintroduce body weights into the
    # garment carrier after the arm split.
    body_free = free & ~sleeve_garment

    # Body bones still reach sleeve vertices, scaled down by the arm's share.
    # Excluding them entirely was the normalisation trap described above.
    body_share = 1.0 - arm_share

    # 3. Head and neck.
    head_mask = body_free & (z > neck_z)
    accumulate("head", head_mask, body_share)
    neck_mask = body_free & (z <= neck_z) & (z > neck_z - body * 0.05)
    blend = np.clip((z - (neck_z - body * 0.05)) / max(body * 0.05, 1e-6), 0.0, 1.0)
    accumulate("neck", neck_mask, blend * body_share)
    accumulate("chest", neck_mask, (1.0 - blend) * body_share)

    # 4. Upper robe: pelvis -> spine_01 -> spine_02 -> chest by height.
    upper = body_free & ~head_mask & ~neck_mask & (z >= hip_z)
    stops = [
        ("pelvis", hip_z),
        ("spine_01", landmarks["waist_z"]),
        ("spine_02", (landmarks["waist_z"] + landmarks["chest_z"]) * 0.5),
        ("chest", landmarks["chest_z"]),
        ("neck", neck_z - body * 0.05),
    ]
    for index in range(len(stops) - 1):
        lower_name, lower_z = stops[index]
        upper_name, upper_z = stops[index + 1]
        span = max(upper_z - lower_z, 1e-6)
        band = upper & (z >= lower_z) & (z < upper_z)
        t = np.clip((z - lower_z) / span, 0.0, 1.0)
        accumulate(lower_name, band, (1.0 - t) * body_share)
        accumulate(upper_name, band, t * body_share)

    # 5. Lower robe: pelvis support near the waist, cloth chains toward the hem.
    lower = body_free & (z < hip_z)
    if lower.any():
        span = max(hip_z - hem_z, 1e-6)
        descent = np.clip((hip_z - z) / span, 0.0, 1.0)
        pelvis_share = np.clip(1.0 - descent * 1.25, 0.0, 1.0)
        accumulate("pelvis", lower, pelvis_share * body_share)

        angle = np.arctan2(points[:, 1] - torso_y, x - symmetry)
        chain_angles = {"f": -np.pi / 2, "b": np.pi / 2, "l": np.pi, "r": 0.0}
        cloth_share = 1.0 - pelvis_share
        for tag, _dx, _dy in CLOTH_CHAINS:
            delta = np.abs(np.arctan2(np.sin(angle - chain_angles[tag]), np.cos(angle - chain_angles[tag])))
            azimuth = np.clip(1.0 - delta / (np.pi * 0.75), 0.0, 1.0) ** 2
            for segment in range(CLOTH_SEGMENTS):
                name = f"cloth_{tag}_{segment + 1:02d}"
                low = hip_z - (hip_z - hem_z) * (segment + 1) / CLOTH_SEGMENTS
                high = hip_z - (hip_z - hem_z) * segment / CLOTH_SEGMENTS
                if segment == CLOTH_SEGMENTS - 1:
                    # The hem estimate is a mass threshold, so geometry below it
                    # (feet, trailing fringe) must still be covered.
                    low = -np.inf
                band = lower & (z >= low) & (z < high + 1e-9)
                accumulate(name, band, cloth_share * azimuth * body_share)

    return weights


def nearest_bone_fallback(points, spec, weights, count):
    """Bind any vertex the volume rules missed to its nearest deform bone.

    Reported explicitly and gated: a large fallback population means the
    influence volumes are wrong, not that the fallback is working.
    """

    # Must match the small-weight cull in normalise(), or vertices carrying only
    # sub-threshold influence look covered here and are culled to nothing later.
    covered = np.zeros(count, dtype=bool)
    for values in weights.values():
        covered |= values > 1e-5
    missing = np.flatnonzero(~covered)
    if missing.size == 0:
        return 0, []

    # Arm bones are excluded as fallback targets. A vertex the semantic volume
    # rules did not classify as sleeve is by definition not a sleeve vertex, and
    # letting the fallback bind it to an arm bone reintroduces exactly the cape
    # bleed the semantic masks exist to prevent.
    candidates = [
        (name, head, tail)
        for name, head, tail, _p, deform in spec
        if deform
        and not name.startswith(("thigh", "calf", "foot", "toe", "staff"))
        and not name.startswith(("clavicle", "upperarm", "lowerarm", "hand"))
    ]
    distances = np.stack(
        [segment_distance(points[missing], head, tail) for _n, head, tail in candidates],
        axis=1,
    )
    choice = np.argmin(distances, axis=1)
    used = set()
    for position, vertex in enumerate(missing):
        name = candidates[choice[position]][0]
        used.add(name)
        if name not in weights:
            weights[name] = np.zeros(count, dtype=np.float64)
        weights[name][vertex] = 1.0
    return int(missing.size), sorted(used)


def normalise(weights: dict[str, np.ndarray], count: int):
    names = sorted(weights)
    matrix = np.stack([weights[name] for name in names], axis=1) if names else np.zeros((count, 0))
    matrix[~np.isfinite(matrix)] = 0.0
    matrix[matrix < 1e-5] = 0.0

    if matrix.shape[1] > MAX_INFLUENCES:
        keep = np.argpartition(-matrix, MAX_INFLUENCES - 1, axis=1)[:, :MAX_INFLUENCES]
        pruned = np.zeros_like(matrix)
        rows = np.arange(count)[:, None]
        pruned[rows, keep] = matrix[rows, keep]
        matrix = pruned

    totals = matrix.sum(axis=1)
    unweighted = totals <= 0.0
    return names, matrix, totals, unweighted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--landmarks", required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--rig-report", required=True)
    parser.add_argument("--skin-report", required=True)
    parser.add_argument("--sockets", required=True)
    args = parser.parse_args(argv_after_double_dash())

    landmarks = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))
    regions = json.loads(Path(args.regions).read_text(encoding="utf-8"))
    staff = (regions.get("context") or {}).get("staff") or {}

    obj = load_mesh(args.input)
    points = world_points(obj)
    count = points.shape[0]

    staff_group = obj.vertex_groups.get("staff")
    staff_mask = np.zeros(count, dtype=bool)
    if staff_group is not None:
        index = staff_group.index
        for vertex in obj.data.vertices:
            for element in vertex.groups:
                if element.group == index and element.weight > 0.5:
                    staff_mask[vertex.index] = True
                    break

    # Region groups were positional bands only; drop them so nothing downstream
    # can mistake them for skin weights.
    for group in list(obj.vertex_groups):
        obj.vertex_groups.remove(group)

    armature, spec, socket_records = build_skeleton(landmarks, staff)

    masks = semantic_masks_v3(points, landmarks, staff_mask)
    weights = compute_weights(points, spec, landmarks, staff_mask, masks)
    fallback_count, fallback_bones = nearest_bone_fallback(points, spec, weights, count)
    names, matrix, totals, unweighted = normalise(weights, count)

    safe = totals.copy()
    safe[safe <= 0.0] = 1.0
    matrix = matrix / safe[:, None]

    for column, name in enumerate(names):
        group = obj.vertex_groups.new(name=name)
        column_weights = matrix[:, column]
        selected = np.flatnonzero(column_weights > 0.0)
        for vertex in selected:
            group.add([int(vertex)], float(column_weights[vertex]), "REPLACE")

    obj.parent = armature
    modifier = obj.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature

    sums = matrix.sum(axis=1)
    weighted = sums > 0.0
    influences = (matrix > 0.0).sum(axis=1)

    deform_bones = [name for name, _h, _t, _p, deform in spec if deform]
    leg_bones = [b for b in deform_bones if b.split("_")[0] in {"thigh", "calf", "foot", "toe"}]
    leg_weighted = [b for b in leg_bones if b in names and matrix[:, names.index(b)].sum() > 0]

    failures = []
    if int(unweighted.sum()) > 0:
        failures.append("UNWEIGHTED_VERTICES")
    if not np.isfinite(matrix).all():
        failures.append("NONFINITE_WEIGHTS")
    if weighted.any() and (np.abs(sums[weighted] - 1.0) > 1e-3).any():
        failures.append("WEIGHT_SUM_OUT_OF_RANGE")
    if influences.max(initial=0) > MAX_INFLUENCES:
        failures.append("TOO_MANY_INFLUENCES")
    if leg_weighted:
        failures.append("LEG_BONES_CARRY_ROBE_WEIGHT")
    if fallback_count > count * 0.01:
        failures.append("NEAREST_BONE_FALLBACK_POPULATION_TOO_LARGE")
    if staff_mask.any():
        staff_column = names.index("staff_deform") if "staff_deform" in names else None
        if staff_column is None or matrix[staff_mask, staff_column].min() < 0.999:
            failures.append("STAFF_NOT_EXCLUSIVELY_BOUND")
        body_columns = [i for i, n in enumerate(names) if n != "staff_deform"]
        if body_columns and matrix[np.ix_(staff_mask, body_columns)].max(initial=0.0) > 1e-6:
            failures.append("STAFF_SHARED_WITH_BODY_BONES")
        if matrix[~staff_mask, staff_column].max(initial=0.0) > 1e-6:
            failures.append("BODY_FOLLOWS_STAFF_BONE")

    rig_report = {
        "stage": "RIG",
        "passed": not failures,
        "body_segmentation": "REJECTED_NOT_OBSERVABLE",
        "pose_mode": "SOURCE_POSE_RIG",
        "a_pose": "REJECTED_UNSAFE_FUSED_ROBE",
        "bone_count": len(spec) + len(SOCKETS),
        "deform_bone_count": len(deform_bones),
        "deform_bones": deform_bones,
        "leg_bones_present_but_unweighted": leg_bones,
        "cloth_bone_count": sum(1 for n in deform_bones if n.startswith("cloth_")),
        "finger_tier": 0,
        "finger_tier_reason": "HANDS_HIDDEN_BY_SLEEVES_NO_VISIBLE_FINGER_SEPARATION",
        "staff_mode": "fused_staff_control",
        "staff_bone_present": "staff_deform" in deform_bones,
        "socket_count": sum(1 for item in socket_records if item["enabled"]),
        "landmarks": args.landmarks,
    }
    skin_report = {
        "stage": "SKIN_QA",
        "passed": not failures,
        "failures": failures,
        "vertex_count": count,
        "unweighted_vertices": int(unweighted.sum()),
        "weight_sum_min": float(sums[weighted].min()) if weighted.any() else None,
        "weight_sum_max": float(sums[weighted].max()) if weighted.any() else None,
        "max_influences": int(influences.max(initial=0)),
        "mean_influences": float(influences.mean()),
        "staff_vertices": int(staff_mask.sum()),
        "semantic_region_sizes": {
            name: int(value.sum())
            for name, value in masks.items()
            if isinstance(value, np.ndarray)
        },
        "sleeve_radius": float(masks["sleeve_radius"]),
        "nearest_bone_fallback_vertices": fallback_count,
        "nearest_bone_fallback_ratio": float(fallback_count / max(count, 1)),
        "nearest_bone_fallback_bones": fallback_bones,
        "bone_weight_totals": {
            name: float(matrix[:, column].sum()) for column, name in enumerate(names)
        },
        "policy": {
            "lower_robe_bound_to": "pelvis + cloth chains",
            "leg_bones_weighted": bool(leg_weighted),
            "per_vertex_limb_classification_used": False,
        },
    }

    for path, payload in (
        (args.rig_report, rig_report),
        (args.skin_report, skin_report),
        (args.sockets, {"stage": "EFFECTS", "socket_count": len(socket_records), "sockets": socket_records}),
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if failures:
        print("RIG_FAILED=" + ",".join(failures), flush=True)
        raise SystemExit(2)

    Path(args.output_blend).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=args.output_blend)

    print(f"RIG_BONES={rig_report['bone_count']} DEFORM={len(deform_bones)}", flush=True)
    print(f"CLOTH_BONES={rig_report['cloth_bone_count']}", flush=True)
    print(f"SOCKETS={rig_report['socket_count']}", flush=True)
    print(f"SKIN_UNWEIGHTED={int(unweighted.sum())}", flush=True)
    print(f"SKIN_MAX_INFLUENCES={int(influences.max(initial=0))}", flush=True)
    print(f"STAFF_VERTICES={int(staff_mask.sum())}", flush=True)
    print("RIG_PASSED=true", flush=True)


if __name__ == "__main__":
    main()
