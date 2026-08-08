from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from common import argv_after_double_dash, export_glb, import_mesh, reset_scene, save_json, select_only, world_bounds


def add_bone(armature, name, head, tail, parent=None):
    head_v, tail_v = Vector(head), Vector(tail)
    if (tail_v - head_v).length < 1e-5:
        tail_v.z += 0.01
    bone = armature.edit_bones.new(name)
    bone.head, bone.tail = head_v, tail_v
    if parent:
        bone.parent = armature.edit_bones.get(parent)
        bone.use_connect = False
    return bone


def load_pose_report(path: str) -> dict:
    target = Path(path) if path else None
    if not target or not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def landmark_mapper(report: dict, minimum: Vector, maximum: Vector):
    landmarks = report.get("pose", {}).get("landmarks", [])
    worlds = report.get("pose", {}).get("world_landmarks", [])
    center = (minimum + maximum) * 0.5
    size = maximum - minimum

    def point(index: int, fallback: Vector) -> Vector:
        if index >= len(landmarks):
            return fallback.copy()
        item = landmarks[index]
        if float(item.get("visibility", 0.0)) < 0.25:
            return fallback.copy()
        x = minimum.x + float(item.get("x", 0.5)) * size.x
        z = maximum.z - float(item.get("y", 0.5)) * size.z
        y = center.y
        if index < len(worlds):
            y -= float(worlds[index].get("z", 0.0)) * max(size.y, size.x) * 0.35
        return Vector((x, y, z))

    return point


def estimate_leg_axes(objects, minimum: Vector, maximum: Vector):
    """Find where the legs actually are, by looking at a slice through them.

    The template placed hips at a fixed +/-9% of body width. Measured on the
    seal diver that put thigh.L at x -0.056 and thigh.R at +0.058 on a body
    0.634 wide -- 0.114 apart, both effectively in the midline, neither inside a
    leg. Near the hips every leg vertex is then roughly equidistant from both
    bones, ownership splits arbitrarily between them, and the mesh shears apart
    the moment the walk plays. Geodesic weighting does not help, because the
    weights were never the problem.

    So take a horizontal band at shin height, where two legs are two separate
    lumps of geometry, and use the mean x of each side. Where the legs are fused
    or there is only one, both means collapse toward the centre, which is the
    correct answer for that shape too.

    Returns None when there is too little geometry in the band to be worth
    trusting, so the caller keeps the old proportional guess.
    """
    size = maximum - minimum
    if size.z <= 0:
        return None

    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        raw = np.empty(len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get("co", raw)
        local = raw.reshape(-1, 3)
        matrix = np.array(obj.matrix_world)
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        points.append(world)
    if not points:
        return None
    world = np.concatenate(points)

    # Shin height: below the knee so the legs have separated, above the ankle so
    # feet and flippers -- which splay outward and would exaggerate the gap --
    # are not what gets measured.
    low = minimum.z + size.z * 0.12
    high = minimum.z + size.z * 0.24
    band = world[(world[:, 2] >= low) & (world[:, 2] <= high)]
    if len(band) < 200:
        return None

    centre = (minimum.x + maximum.x) * 0.5
    left = band[band[:, 0] < centre]
    right = band[band[:, 0] >= centre]
    if len(left) < 50 or len(right) < 50:
        return None

    left_x = float(np.median(left[:, 0]))
    right_x = float(np.median(right[:, 0]))
    # Refuse a split so wide it must be arms or gear rather than legs.
    if (right_x - left_x) > size.x * 0.75:
        return None
    print(f"[rig] leg axes measured at x {left_x:.3f} and {right_x:.3f} "
          f"(body width {size.x:.3f}, {len(band)} vertices in band)", flush=True)
    return left_x, right_x


def humanoid_bones(minimum: Vector, maximum: Vector, report: dict,
                   leg_axes=None) -> list[tuple]:
    center = (minimum + maximum) * 0.5
    size = maximum - minimum
    z0, z1 = minimum.z, maximum.z
    pelvis_f = Vector((center.x, center.y, z0 + size.z * 0.48))
    chest_f = Vector((center.x, center.y, z0 + size.z * 0.72))
    neck_f = Vector((center.x, center.y, z0 + size.z * 0.84))
    head_f = Vector((center.x, center.y, z0 + size.z * 0.96))
    point = landmark_mapper(report, minimum, maximum)

    l_sh = point(11, chest_f + Vector((-size.x * 0.16, 0, 0)))
    r_sh = point(12, chest_f + Vector((size.x * 0.16, 0, 0)))
    l_el = point(13, l_sh + Vector((-size.x * 0.20, 0, -size.z * 0.10)))
    r_el = point(14, r_sh + Vector((size.x * 0.20, 0, -size.z * 0.10)))
    l_wr = point(15, l_el + Vector((-size.x * 0.18, 0, -size.z * 0.10)))
    r_wr = point(16, r_el + Vector((size.x * 0.18, 0, -size.z * 0.10)))
    l_hand = point(19, l_wr + Vector((-size.x * 0.06, 0, -size.z * 0.02)))
    r_hand = point(20, r_wr + Vector((size.x * 0.06, 0, -size.z * 0.02)))
    # Measured leg axes beat the proportional guess whenever they are available;
    # the guess is only a fallback for a mesh too sparse to measure.
    if leg_axes is not None:
        left_x, right_x = leg_axes
        hip_l_default = Vector((left_x, pelvis_f.y, pelvis_f.z))
        hip_r_default = Vector((right_x, pelvis_f.y, pelvis_f.z))
    else:
        hip_l_default = pelvis_f + Vector((-size.x * 0.09, 0, 0))
        hip_r_default = pelvis_f + Vector((size.x * 0.09, 0, 0))
    l_hip = point(23, hip_l_default)
    r_hip = point(24, hip_r_default)
    pelvis = (l_hip + r_hip) * 0.5
    l_knee = point(25, Vector((l_hip.x, center.y, z0 + size.z * 0.25)))
    r_knee = point(26, Vector((r_hip.x, center.y, z0 + size.z * 0.25)))
    l_ankle = point(27, Vector((l_hip.x, center.y, z0 + size.z * 0.04)))
    r_ankle = point(28, Vector((r_hip.x, center.y, z0 + size.z * 0.04)))
    l_toe = point(31, l_ankle + Vector((0, -size.y * 0.12, 0)))
    r_toe = point(32, r_ankle + Vector((0, -size.y * 0.12, 0)))
    shoulders = (l_sh + r_sh) * 0.5
    chest = pelvis.lerp(shoulders, 0.72)
    neck = shoulders.lerp(point(0, head_f), 0.42)
    head = point(0, head_f)
    head_top = Vector((head.x, head.y, min(z1, head.z + size.z * 0.10)))

    return [
        ("root", Vector((pelvis.x, pelvis.y, z0)), pelvis, None),
        ("pelvis", pelvis, pelvis.lerp(chest, 0.32), "root"),
        ("spine", pelvis.lerp(chest, 0.32), chest, "pelvis"),
        ("chest", chest, shoulders, "spine"),
        ("neck", shoulders, neck, "chest"),
        ("head", neck, head_top, "neck"),
        ("clavicle.L", shoulders, l_sh, "chest"),
        ("upper_arm.L", l_sh, l_el, "clavicle.L"),
        ("forearm.L", l_el, l_wr, "upper_arm.L"),
        ("hand.L", l_wr, l_hand, "forearm.L"),
        ("clavicle.R", shoulders, r_sh, "chest"),
        ("upper_arm.R", r_sh, r_el, "clavicle.R"),
        ("forearm.R", r_el, r_wr, "upper_arm.R"),
        ("hand.R", r_wr, r_hand, "forearm.R"),
        ("thigh.L", l_hip, l_knee, "pelvis"),
        ("shin.L", l_knee, l_ankle, "thigh.L"),
        ("foot.L", l_ankle, l_toe, "shin.L"),
        ("thigh.R", r_hip, r_knee, "pelvis"),
        ("shin.R", r_knee, r_ankle, "thigh.R"),
        ("foot.R", r_ankle, r_toe, "shin.R"),
    ]


def creature_bones(minimum: Vector, maximum: Vector) -> list[tuple]:
    center = (minimum + maximum) * 0.5
    sx, sy, sz = maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z
    return [
        ("root", (center.x, minimum.y, center.z), (center.x, center.y, center.z), None),
        ("spine", (center.x, center.y, center.z), (center.x, center.y + sy * 0.25, center.z + sz * 0.08), "root"),
        ("neck", (center.x, center.y + sy * 0.25, center.z + sz * 0.08), (center.x, center.y + sy * 0.4, center.z + sz * 0.2), "spine"),
        ("head", (center.x, center.y + sy * 0.4, center.z + sz * 0.2), (center.x, maximum.y, center.z + sz * 0.25), "neck"),
        ("wing.L", (center.x, center.y, center.z + sz * 0.1), (minimum.x, center.y, center.z + sz * 0.05), "spine"),
        ("wing.R", (center.x, center.y, center.z + sz * 0.1), (maximum.x, center.y, center.z + sz * 0.05), "spine"),
        ("leg.L", (center.x - sx * 0.15, center.y, center.z), (center.x - sx * 0.15, center.y, minimum.z), "root"),
        ("leg.R", (center.x + sx * 0.15, center.y, center.z), (center.x + sx * 0.15, center.y, minimum.z), "root"),
        ("tail", (center.x, center.y - sy * 0.2, center.z), (center.x, minimum.y, center.z), "root"),
    ]


def make_armature(kind: str, objects: list[bpy.types.Object], pose_report: dict) -> tuple[bpy.types.Object, dict[str, str], bool]:
    minimum, maximum = world_bounds(objects)
    data = bpy.data.armatures.new("GameRig")
    armature = bpy.data.objects.new("GameRig", data)
    bpy.context.collection.objects.link(armature)
    select_only([armature])
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    object_bones: dict[str, str] = {}
    pose_guided = False
    if kind == "mechanical":
        center = (minimum + maximum) * 0.5
        add_bone(data, "root", (center.x, center.y, minimum.z), (center.x, center.y, center.z), None)
        scale = max((maximum - minimum).length * 0.02, 0.02)
        for index, obj in enumerate(objects):
            obj_min, obj_max = world_bounds([obj])
            pivot = (obj_min + obj_max) * 0.5
            bone_name = f"part_{index:03d}_{obj.name}"[:60]
            add_bone(data, bone_name, pivot, pivot + Vector((0, 0, scale)), "root")
            object_bones[obj.name] = bone_name
    else:
        if kind == "humanoid":
            bones = humanoid_bones(minimum, maximum, pose_report,
                                   estimate_leg_axes(objects, minimum, maximum))
        else:
            bones = creature_bones(minimum, maximum)
        pose_guided = kind == "humanoid" and bool(pose_report.get("pose", {}).get("detected"))
        for name, head, tail, parent in bones:
            add_bone(data, name, head, tail, parent)
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature, object_bones, pose_guided


def weighted_fraction(obj, bone_names: set[str]) -> float:
    """Fraction of vertices carrying a non-zero weight in some deform group.

    The number that decides whether a bind worked. `parent_set` reports a
    bone-heat failure through the operator report, not through an exception, so
    it returns {'FINISHED'} either way and the caller cannot tell. Counting the
    vertices it actually weighted can.
    """
    indices = {group.index for group in obj.vertex_groups
               if group.name in bone_names}
    if not indices:
        return 0.0
    weighted = sum(
        1 for vertex in obj.data.vertices
        if any(g.group in indices and g.weight > 1e-4 for g in vertex.groups))
    return weighted / max(len(obj.data.vertices), 1)


def welded_edges(vertices, faces, tolerance: float = 1e-6):
    """Unique mesh edges in a position-welded index space.

    A generated GLB duplicates a vertex at every UV seam, so the stored index
    buffer is not the surface's connectivity: read literally, a painted seal
    diver is 8,031 disconnected shells whose largest holds 1,450 faces. Welding
    positions recovers the real thing -- 8 shells, one body, 0.4% debris.

    This is almost certainly why bone heat weighting returns 0% on these meshes.
    It needs a connected manifold and the index buffer never gave it one.

    Returns (weld index per original vertex, unique edge pairs, edge lengths).
    """
    quantised = np.round(vertices / tolerance).astype(np.int64)
    _, weld = np.unique(quantised, axis=0, return_inverse=True)
    welded_faces = weld[faces]

    edges = np.vstack([welded_faces[:, [0, 1]],
                       welded_faces[:, [1, 2]],
                       welded_faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]

    count = int(weld.max()) + 1
    positions = np.zeros((count, 3))
    positions[weld] = vertices
    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    return weld, edges, np.maximum(lengths, 1e-9), positions


def geodesic_owner(positions, edges, lengths, seeds, seed_owner, rounds: int = 400):
    """Multi-source shortest path over the surface, vectorised.

    Blender bundles numpy but not scipy, so this is iterative edge relaxation --
    Bellman-Ford in the shape a GPU would run it -- rather than a heap-based
    Dijkstra. Each round is a handful of numpy operations over the whole edge
    array, which is far faster in practice than a Python heap over 900k
    vertices, and it converges once no edge improves.

    Sorting candidates descending before the scatter makes the smallest value
    the last write, so a single pass takes the best improvement per vertex
    rather than an arbitrary one.
    """
    count = len(positions)
    distance = np.full(count, np.inf)
    owner = np.full(count, -1, dtype=np.int64)
    distance[seeds] = 0.0
    owner[seeds] = seed_owner

    source = np.concatenate([edges[:, 0], edges[:, 1]])
    target = np.concatenate([edges[:, 1], edges[:, 0]])
    weight = np.concatenate([lengths, lengths])

    for _ in range(rounds):
        candidate = distance[source] + weight
        better = candidate < distance[target]
        if not better.any():
            break
        order = np.argsort(-candidate[better])
        hit = target[better][order]
        value = candidate[better][order]
        from_owner = owner[source[better]][order]
        keep = value < distance[hit]
        distance[hit[keep]] = value[keep]
        owner[hit[keep]] = from_owner[keep]
    return distance, owner


def bind_geodesic(obj, armature, bone_names: list[str], smoothing: int = 14) -> bool:
    """Weight vertices by distance ALONG THE SURFACE to the nearest bone.

    This is what proximity weighting cannot do. The left thigh bone is close in
    space to right-leg vertices -- the gap between two legs is only a few
    centimetres -- so Euclidean nearest-bone grabs across it and the legs shear
    into each other when the walk cycle plays. Over the surface there is no such
    shortcut: the only path from one leg to the other runs up through the
    pelvis, so the weights cannot leak.

    Hard ownership first, then Laplacian smoothing of the one-hot weights over
    the same welded graph. Smoothing gives joints a gradient to bend through,
    and because it only ever moves weight along real edges it cannot reintroduce
    the leak the hard assignment just avoided.
    """
    mesh = obj.data
    vertices = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", vertices)
    vertices = vertices.reshape(-1, 3)
    faces = np.array([polygon.vertices[:] for polygon in mesh.polygons
                      if len(polygon.vertices) == 3], dtype=np.int64)
    if len(faces) == 0:
        return False

    weld, edges, lengths, positions = welded_edges(vertices, faces)
    if len(edges) == 0:
        return False

    bones = [armature.data.bones[name] for name in bone_names
             if name in armature.data.bones]
    if not bones:
        return False

    # Seed each bone with the welded vertices nearest its segment. A handful of
    # seeds rather than one, so a single bad vertex cannot decide a whole limb,
    # but kept local so a seed cannot land on the wrong side of a gap.
    seeds, seed_owner = [], []
    for index, bone in enumerate(bones):
        head = np.array(bone.head_local)
        axis = np.array(bone.tail_local) - head
        length_squared = float(axis @ axis)
        if length_squared < 1e-12:
            closest = np.tile(head, (len(positions), 1))
        else:
            t = np.clip((positions - head) @ axis / length_squared, 0.0, 1.0)
            closest = head + t[:, None] * axis
        gap = np.linalg.norm(positions - closest, axis=1)
        picked = np.argpartition(gap, min(6, len(gap) - 1))[:6]
        seeds.extend(picked.tolist())
        seed_owner.extend([index] * len(picked))

    _, owner = geodesic_owner(positions, edges, lengths,
                              np.array(seeds), np.array(seed_owner))
    reached = (owner >= 0).mean()
    print(f"[rig] geodesic ownership reached {reached * 100:.1f}% of welded "
          f"vertices over {len(edges)} edges", flush=True)
    if reached < 0.5:
        return False

    weights = np.zeros((len(positions), len(bones)), dtype=np.float32)
    valid = owner >= 0
    weights[np.where(valid)[0], owner[valid]] = 1.0

    degree = np.zeros(len(positions))
    np.add.at(degree, edges[:, 0], 1.0)
    np.add.at(degree, edges[:, 1], 1.0)
    degree = np.maximum(degree, 1.0)
    for _ in range(smoothing):
        neighbour = np.zeros_like(weights)
        np.add.at(neighbour, edges[:, 0], weights[edges[:, 1]])
        np.add.at(neighbour, edges[:, 1], weights[edges[:, 0]])
        weights = (weights + neighbour / degree[:, None]) * 0.5
    total = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(total, 1e-8)

    per_vertex = weights[weld]
    groups = {bone.name: obj.vertex_groups.get(bone.name)
              or obj.vertex_groups.new(name=bone.name) for bone in bones}
    # Quantised so the vertices sharing a weight can go in one add() call.
    # Per-vertex calls on a 900k mesh take minutes; this takes seconds.
    levels = 48
    for index, bone in enumerate(bones):
        column = per_vertex[:, index]
        bucket = np.clip((column * levels).astype(np.int64), 0, levels)
        for step in range(1, levels + 1):
            members = np.where(bucket == step)[0]
            if len(members):
                groups[bone.name].add(members.tolist(), step / levels, "REPLACE")

    if not any(m.type == "ARMATURE" for m in obj.modifiers):
        modifier = obj.modifiers.new("Armature", "ARMATURE")
        modifier.object = armature
    obj.parent = armature
    return True


def bind_by_proximity(obj, armature, bone_names: list[str]) -> None:
    """Assign each vertex to its nearest bone segment, blended over the two
    nearest, so a failed bone-heat solve still yields a deformable mesh.

    Not as good as a heat solve -- there is no surface-geodesic term, so a
    weight can bleed between limbs that are close in space but far along the
    body. It is, however, a rig that moves, which is strictly more than the
    silent no-skin export it replaces.
    """
    bones = [armature.data.bones[name] for name in bone_names
             if name in armature.data.bones]
    if not bones:
        return
    segments = [(bone.name, bone.head_local, bone.tail_local) for bone in bones]
    groups = {name: obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
              for name, _, _ in segments}

    for vertex in obj.data.vertices:
        point = obj.matrix_world @ vertex.co
        distances = []
        for name, head, tail in segments:
            axis = tail - head
            length_squared = axis.length_squared
            if length_squared < 1e-9:
                closest = head
            else:
                t = max(0.0, min(1.0, (point - head).dot(axis) / length_squared))
                closest = head + axis * t
            distances.append(((point - closest).length, name))
        distances.sort()
        (near_d, near_n) = distances[0]
        # Blend with the runner-up so joints bend instead of shearing.
        if len(distances) > 1:
            (far_d, far_n) = distances[1]
            total = near_d + far_d
            near_w = 1.0 if total < 1e-9 else far_d / total
            groups[near_n].add([vertex.index], near_w, "REPLACE")
            groups[far_n].add([vertex.index], 1.0 - near_w, "REPLACE")
        else:
            groups[near_n].add([vertex.index], 1.0, "REPLACE")

    if not any(m.type == "ARMATURE" for m in obj.modifiers):
        modifier = obj.modifiers.new("Armature", "ARMATURE")
        modifier.object = armature
    obj.parent = armature


# A bind that weights less than this fraction of the mesh is treated as failed.
# Bone heat either solves broadly or collapses; a partial solve in between has
# not been observed, so the threshold only has to sit clear of both.
BIND_COVERAGE_FLOOR = 0.60


def bind_organic(objects, armature) -> str:
    select_only(objects + [armature])
    bpy.context.view_layer.objects.active = armature
    bone_names = [bone.name for bone in armature.data.bones]
    names = set(bone_names)

    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as error:
        print(f"[rig] parent_set raised: {error}", flush=True)

    # Verify rather than trust. `Bone Heat Weighting: failed to find solution
    # for one or more bones` is printed as a report and the operator still
    # returns FINISHED, so the only way to know is to count the weights.
    coverage = {obj.name: weighted_fraction(obj, names) for obj in objects}
    for name, fraction in coverage.items():
        print(f"[rig] {name}: {fraction * 100:.1f}% of vertices weighted",
              flush=True)

    failed = [obj for obj in objects
              if coverage[obj.name] < BIND_COVERAGE_FLOOR]
    if not failed:
        return "automatic_weights"

    # Geodesic first, proximity only if that cannot run. Euclidean nearest-bone
    # is the last resort because it reaches across gaps: the left thigh bone is
    # centimetres from right-leg vertices, so it claims them, and the legs shear
    # into each other as soon as the walk plays.
    print(f"[rig] bone heat did not weight {len(failed)} mesh(es) -- "
          f"trying geodesic weights", flush=True)
    used = []
    for obj in failed:
        if bind_geodesic(obj, armature, bone_names):
            used.append("geodesic")
        else:
            print(f"[rig] {obj.name}: geodesic bind unavailable, "
                  f"using proximity", flush=True)
            bind_by_proximity(obj, armature, bone_names)
            used.append("proximity")
        after = weighted_fraction(obj, names)
        print(f"[rig] {obj.name}: {after * 100:.1f}% weighted after "
              f"{used[-1]} fallback", flush=True)
    return ("geodesic_weights" if all(u == "geodesic" for u in used)
            else "proximity_weights")


def bind_rigid(objects, armature, object_bones: dict[str, str]) -> str:
    for obj in objects:
        obj.parent = armature
        obj.parent_type = "BONE"
        obj.parent_bone = object_bones.get(obj.name, "root")
    return "rigid_parts"


def new_action(armature, name: str, frames: int):
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    armature.animation_data_create()
    armature.animation_data.action = action
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, frames
    return action


def key_rotation(armature, bone_name: str, frame: int, xyz: tuple[float, float, float]) -> None:
    bone = armature.pose.bones.get(bone_name)
    if bone is None:
        return
    bone.rotation_mode = "XYZ"
    bone.rotation_euler = xyz
    bone.keyframe_insert("rotation_euler", frame=frame)


def key_location(armature, bone_name: str, frame: int, xyz: tuple[float, float, float]) -> None:
    bone = armature.pose.bones.get(bone_name)
    if bone is None:
        return
    bone.location = xyz
    bone.keyframe_insert("location", frame=frame)


def add_idle(armature) -> str:
    new_action(armature, "idle", 48)
    for frame, amount in ((1, 0.0), (13, 0.025), (25, 0.0), (37, -0.02), (48, 0.0)):
        key_rotation(armature, "spine", frame, (amount, 0.0, amount * 0.4))
        key_rotation(armature, "head", frame, (0.0, amount * 0.6, 0.0))
    return "idle"


def add_dance(armature) -> str:
    new_action(armature, "dance_loop", 96)
    frames = (1, 13, 25, 37, 49, 61, 73, 85, 96)
    for index, frame in enumerate(frames):
        phase = (index / 8.0) * math.tau
        sway = math.sin(phase)
        bounce = max(0.0, math.sin(phase * 2.0))
        key_location(armature, "root", frame, (sway * 0.035, 0.0, bounce * 0.035))
        key_rotation(armature, "pelvis", frame, (0.08 * bounce, 0.0, -0.22 * sway))
        key_rotation(armature, "spine", frame, (-0.08 * bounce, 0.08 * sway, 0.18 * sway))
        key_rotation(armature, "chest", frame, (0.0, -0.08 * sway, -0.22 * sway))
        key_rotation(armature, "head", frame, (0.05 * bounce, 0.0, 0.08 * sway))
        key_rotation(armature, "upper_arm.L", frame, (0.45 + 0.45 * sway, -0.10, -0.55 - 0.30 * sway))
        key_rotation(armature, "forearm.L", frame, (-0.50 + 0.20 * sway, 0.0, -0.15))
        key_rotation(armature, "upper_arm.R", frame, (0.45 - 0.45 * sway, 0.10, 0.55 - 0.30 * sway))
        key_rotation(armature, "forearm.R", frame, (-0.50 - 0.20 * sway, 0.0, 0.15))
        key_rotation(armature, "thigh.L", frame, (-0.18 * sway, 0.0, 0.08 * sway))
        key_rotation(armature, "thigh.R", frame, (0.18 * sway, 0.0, 0.08 * sway))
        key_rotation(armature, "shin.L", frame, (0.20 * max(0.0, sway), 0.0, 0.0))
        key_rotation(armature, "shin.R", frame, (0.20 * max(0.0, -sway), 0.0, 0.0))
    cycle_and_smooth(armature)
    return "dance_loop"


def action_fcurves(action):
    """The action's F-curves, on both the old and the slotted data model.

    Blender 4.4 moved animation into layers, strips and channelbags, and
    `Action.fcurves` no longer exists on 5.2 -- reading it raises
    AttributeError. Every loop here that sets interpolation or attaches a CYCLES
    modifier went through that attribute, so on 5.2 the walk crashed outright
    and `add_dance` had the same latent break. Blender still exits 0 when its
    Python raises, so this surfaced as an export that silently lacked the action
    rather than as a failure.

    Returns a flat list so callers do not care which model is in use.
    """
    curves = getattr(action, "fcurves", None)
    if curves is not None:
        return list(curves)
    collected = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                collected.extend(channelbag.fcurves)
    return collected


def cycle_and_smooth(armature) -> None:
    """Bezier interpolation on every key, and a cycle modifier per curve."""
    action = armature.animation_data.action
    for fcurve in action_fcurves(action):
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "BEZIER"
        if not any(m.type == "CYCLES" for m in fcurve.modifiers):
            fcurve.modifiers.new(type="CYCLES")


def smooth_only(armature) -> None:
    action = armature.animation_data.action
    for fcurve in action_fcurves(action):
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "BEZIER"


def key_scale(armature, bone_name: str, frame: int, xyz: tuple[float, float, float]) -> None:
    bone = armature.pose.bones.get(bone_name)
    if bone is None:
        return
    bone.scale = xyz
    bone.keyframe_insert("scale", frame=frame)


def add_breathe(armature) -> str:
    """Chest expansion on a slow asymmetric cycle.

    Rotation alone cannot read as a breath -- a ribcage gets bigger, it does not
    tip -- so the chest bone is scaled, and the shoulders are carried outward
    with it. The cycle is deliberately not a sine: a real breath is a quick
    intake and a longer release, so the peak sits at frame 36 of 96 rather than
    at the midpoint. An even cycle reads as a machine.

    Amplitudes are small on purpose. At 4% chest scale this survives being
    layered under idle or sitting; at the 15% that looks right in isolation it
    turns a diver's drysuit into a bellows.
    """
    new_action(armature, "breathe", 96)
    # frame, inflation 0..1
    for frame, amount in ((1, 0.0), (36, 1.0), (60, 0.45), (96, 0.0)):
        key_scale(armature, "chest", frame,
                  (1.0 + 0.045 * amount, 1.0 + 0.02 * amount, 1.0 + 0.035 * amount))
        key_scale(armature, "spine", frame, (1.0, 1.0 + 0.015 * amount, 1.0))
        key_rotation(armature, "chest", frame, (-0.03 * amount, 0.0, 0.0))
        key_rotation(armature, "clavicle.L", frame, (0.0, 0.0, -0.05 * amount))
        key_rotation(armature, "clavicle.R", frame, (0.0, 0.0, 0.05 * amount))
        key_rotation(armature, "head", frame, (0.012 * amount, 0.0, 0.0))
        key_location(armature, "root", frame, (0.0, 0.0, 0.004 * amount))

    cycle_and_smooth(armature)
    return "breathe"


def add_sit(armature) -> str:
    """Stand, lower into a seated pose, settle.

    Not a loop -- it plays once and holds, so it is keyed pose-to-pose with a
    settle beat at the end rather than cycled. The root drops by 0.42 of the
    thigh length rather than a fixed distance, because the same action has to
    work on a 1.9 m shaman and a 120 m titan.

    The thigh rotates to roughly a right angle and the shin takes the rest. The
    shin sign is negative for the same reason as in the walk: a knee bends one
    way, and letting the interpolator choose puts it through the joint.
    """
    thigh = armature.data.bones.get("thigh.L")
    drop = (thigh.length * 0.42) if thigh else 0.2

    new_action(armature, "sit", 72)
    #  frame, how far into the sit (0 standing, 1 seated)
    for frame, amount in ((1, 0.0), (30, 0.85), (46, 1.0), (72, 1.0)):
        key_location(armature, "root", frame, (0.0, -drop * 0.35 * amount,
                                               -drop * amount))
        key_rotation(armature, "pelvis", frame, (-0.35 * amount, 0.0, 0.0))
        key_rotation(armature, "spine", frame, (0.22 * amount, 0.0, 0.0))
        key_rotation(armature, "chest", frame, (0.10 * amount, 0.0, 0.0))
        key_rotation(armature, "head", frame, (-0.10 * amount, 0.0, 0.0))
        for side, sign in (("L", 1.0), ("R", -1.0)):
            key_rotation(armature, f"thigh.{side}", frame,
                         (1.50 * amount, 0.0, 0.10 * amount * sign))
            key_rotation(armature, f"shin.{side}", frame, (-1.45 * amount, 0.0, 0.0))
            key_rotation(armature, f"foot.{side}", frame, (-0.10 * amount, 0.0, 0.0))
            key_rotation(armature, f"upper_arm.{side}", frame,
                         (0.30 * amount, 0.0, -0.18 * amount * sign))
            key_rotation(armature, f"forearm.{side}", frame, (-0.55 * amount, 0.0, 0.0))

    smooth_only(armature)
    return "sit"


def add_walk(armature) -> str:
    """A contralateral walk cycle: opposite arm and leg swing together.

    Driven by one phase angle so the loop closes exactly at the wrap frame
    rather than approximately. The legs run on sin(phase) and the arms on
    -sin(phase), which is what makes it read as walking rather than as
    marching; a same-side swing looks wrong immediately even to an untrained
    eye. The pelvis rises on |cos| because the body lifts twice per cycle, once
    over each supporting leg, not once.

    Knees only ever bend one way: shin rotation is clamped to the negative side
    so the leg cannot hyperextend through the joint on the passing pose.
    """
    frames = (1, 9, 17, 25, 33, 41, 49)
    new_action(armature, "walk_loop", frames[-1])
    for index, frame in enumerate(frames):
        phase = (index / (len(frames) - 1)) * math.tau
        swing = math.sin(phase)
        lift = abs(math.cos(phase))

        key_location(armature, "root", frame, (0.0, 0.0, lift * 0.02))
        key_rotation(armature, "pelvis", frame, (0.0, 0.0, -0.06 * swing))
        key_rotation(armature, "spine", frame, (0.04, 0.05 * swing, 0.05 * swing))
        key_rotation(armature, "chest", frame, (0.0, -0.09 * swing, 0.0))
        key_rotation(armature, "head", frame, (0.0, 0.04 * swing, 0.0))

        # Legs: one forward while the other is back.
        key_rotation(armature, "thigh.L", frame, (0.55 * swing, 0.0, 0.0))
        key_rotation(armature, "thigh.R", frame, (-0.55 * swing, 0.0, 0.0))
        key_rotation(armature, "shin.L", frame, (-0.65 * max(0.0, -swing), 0.0, 0.0))
        key_rotation(armature, "shin.R", frame, (-0.65 * max(0.0, swing), 0.0, 0.0))
        # The foot counter-rotates against the thigh so the sole stays roughly
        # level through the stride instead of pointing wherever the shin does.
        key_rotation(armature, "foot.L", frame, (-0.30 * swing, 0.0, 0.0))
        key_rotation(armature, "foot.R", frame, (0.30 * swing, 0.0, 0.0))

        # Arms opposite the legs on the same side.
        key_rotation(armature, "upper_arm.L", frame, (-0.45 * swing, 0.0, -0.12))
        key_rotation(armature, "upper_arm.R", frame, (0.45 * swing, 0.0, 0.12))
        key_rotation(armature, "forearm.L", frame, (-0.25 - 0.20 * max(0.0, -swing), 0.0, 0.0))
        key_rotation(armature, "forearm.R", frame, (-0.25 - 0.20 * max(0.0, swing), 0.0, 0.0))

    cycle_and_smooth(armature)
    return "walk_loop"


def add_creature_walk(armature) -> str:
    """A lumbering two-leg creature walk on the creature skeleton.

    The creature rig has `leg.L`/`leg.R`, `spine`, `neck`, `head`, `tail` and a
    pair of `wing` bones, and no knees -- so this cannot be the humanoid cycle
    with different names. Each leg is a single bone that swings from the body,
    and the weight shift has to be carried by the spine roll and the body lift
    instead of by a knee bend.

    Amplitudes are deliberately smaller than the humanoid walk (0.34 rad against
    0.55). A creature rigged from a bounding box has no anatomy underneath the
    bones, so a big swing shears the mass rather than articulating it -- which
    is exactly what the humanoid rig did to the moss titan.
    """
    frames = (1, 9, 17, 25, 33, 41, 49)
    new_action(armature, "creature_walk", frames[-1])
    for index, frame in enumerate(frames):
        phase = (index / (len(frames) - 1)) * math.tau
        swing = math.sin(phase)
        lift = abs(math.cos(phase))

        key_location(armature, "root", frame, (0.0, 0.0, lift * 0.025))
        key_rotation(armature, "spine", frame, (0.05 * swing, 0.0, 0.07 * swing))
        key_rotation(armature, "neck", frame, (-0.04 * swing, 0.0, -0.05 * swing))
        key_rotation(armature, "head", frame, (0.05 * lift, 0.0, -0.04 * swing))
        key_rotation(armature, "tail", frame, (0.0, 0.0, -0.12 * swing))
        key_rotation(armature, "leg.L", frame, (0.34 * swing, 0.0, 0.0))
        key_rotation(armature, "leg.R", frame, (-0.34 * swing, 0.0, 0.0))
        key_rotation(armature, "wing.L", frame, (0.0, 0.10 * swing, 0.0))
        key_rotation(armature, "wing.R", frame, (0.0, -0.10 * swing, 0.0))

    cycle_and_smooth(armature)
    return "creature_walk"


def add_mechanical_actions(armature) -> list[str]:
    new_action(armature, "mechanical_cycle", 48)
    bones = [bone.name for bone in armature.data.bones if bone.name != "root"]
    rotating = [name for name in bones if any(token in name.lower() for token in ("wheel", "propeller", "rotor", "turbine"))] or bones[:4]
    for name in rotating:
        key_rotation(armature, name, 1, (0, 0, 0))
        key_rotation(armature, name, 48, (0, math.tau, 0))
    return ["mechanical_cycle"]


def add_actions(armature, kind: str, animation_preset: str) -> list[str]:
    if kind == "humanoid":
        actions = [add_idle(armature)]
        if animation_preset in {"walk", "all", "auto"}:
            actions.append(add_walk(armature))
        if animation_preset in {"breathe", "all", "auto"}:
            actions.append(add_breathe(armature))
        if animation_preset in {"sit", "all", "auto"}:
            actions.append(add_sit(armature))
        if animation_preset in {"dance", "all"}:
            actions.append(add_dance(armature))
        return actions
    if kind == "creature":
        new_action(armature, "creature_idle", 48)
        for frame, amount in ((1, 0.0), (24, 0.15), (48, 0.0)):
            key_rotation(armature, "head", frame, (0.0, amount, 0.0))
            key_rotation(armature, "tail", frame, (0.0, -amount, 0.0))
        actions = ["creature_idle"]
        if animation_preset in {"walk", "all", "auto"}:
            actions.append(add_creature_walk(armature))
        return actions
    return add_mechanical_actions(armature)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--kind", choices=("auto", "humanoid", "creature", "mechanical", "static"), default="auto")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--animation-preset", choices=("none", "idle", "walk", "breathe", "sit", "dance", "all", "auto"), default="auto")
    parser.add_argument("--pose-report", default="")
    args = parser.parse_args(argv_after_double_dash())
    reset_scene()
    objects = import_mesh(args.input)
    prompt, kind = args.prompt.lower(), args.kind
    if kind == "auto":
        if any(word in prompt for word in ("vehicle", "machine", "robot", "tank", "ship", "door", "wheel")):
            kind = "mechanical"
        elif any(word in prompt for word in ("human", "person", "character", "soldier", "humanoid", "avatar")):
            kind = "humanoid"
        else:
            kind = "creature"
    if kind == "static":
        export_glb(args.output)
        save_json(args.report, {"kind": kind, "binding": "none", "actions": [], "success": True})
        return
    pose_report = load_pose_report(args.pose_report)
    armature, object_bones, pose_guided = make_armature(kind, objects, pose_report)
    binding = bind_rigid(objects, armature, object_bones) if kind == "mechanical" else bind_organic(objects, armature)
    actions = add_actions(armature, kind, args.animation_preset)
    # apply_modifiers=False: applying them consumes the Armature modifier and
    # exports a skinless mesh that still carries JOINTS_0/WEIGHTS_0.
    export_glb(args.output, apply_modifiers=False)
    save_json(args.report, {
        "success": True,
        "kind": kind,
        "binding": binding,
        "bones": [bone.name for bone in armature.data.bones],
        "actions": actions,
        "animation_preset": args.animation_preset,
        "pose_guided_proportions": pose_guided,
        "deformation_proven": False,
        "warning": "Pose-guided template fitting and automatic weights require visual deformation inspection before shipping.",
    })


if __name__ == "__main__":
    main()
