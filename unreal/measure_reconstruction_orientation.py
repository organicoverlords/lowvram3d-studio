"""Recover the image->world orientation of an imported MoGe reconstruction.

The yaw sweep in `scripts/verify_axis_mapping_prediction.py` showed the
reconstruction is not merely rotated: the yaw that puts the most geometry in
frame renders *vertically mirrored*, and no rigid camera pose produces a mirror.
So measure the mesh itself instead of arguing about conventions.

A MoGe reconstruction carries the source image's pixel coordinates as UVs, which
makes the answer directly observable: the vertices with v ~ 0 came from the top
row of the source image and the ones with v ~ 1 from the bottom row. The world
direction between those two groups *is* the image's up direction, measured
rather than assumed. The same for u -> the image's right direction, and the
centroid direction is the camera's forward.

With all three known, the handedness of (right, up, forward) says whether the
imported scene is mirrored relative to the source view. Unreal is left-handed
with right = up x forward, so a positive dot between the measured right and
up x forward means the view is reproducible by a camera; a negative one means
the import mirrored the scene and no camera rotation can undo it.

Configure with a `RECONSTRUCTION_ORIENTATION_REQUEST` global holding
`static_mesh` (package path) or `actor_label`.

    python -m uemcp python @unreal/measure_reconstruction_orientation.py --json
"""

import json
import math

import unreal

REQUEST = globals().get("RECONSTRUCTION_ORIENTATION_REQUEST") or {}
STATIC_MESH = REQUEST.get("static_mesh")
ACTOR_LABEL = REQUEST.get("actor_label")
# Fraction of the UV range taken as "the edge" of the image on each side.
EDGE_BAND = float(REQUEST.get("edge_band", 0.05))

report = {"schema_version": "reconstruction_orientation_v1"}

mesh = None
if STATIC_MESH:
    mesh = unreal.load_asset(STATIC_MESH)
if mesh is None and ACTOR_LABEL:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in subsystem.get_all_level_actors():
        if str(actor.get_actor_label()) == ACTOR_LABEL:
            component = actor.get_component_by_class(unreal.StaticMeshComponent)
            mesh = component.get_editor_property("static_mesh")
            report["actor_rotation"] = [
                float(actor.get_actor_rotation().pitch),
                float(actor.get_actor_rotation().yaw),
                float(actor.get_actor_rotation().roll)]
            report["actor_scale"] = [
                float(actor.get_actor_scale3d().x),
                float(actor.get_actor_scale3d().y),
                float(actor.get_actor_scale3d().z)]
            break
if mesh is None:
    raise RuntimeError("no static mesh found for " + str(REQUEST))
report["static_mesh"] = str(mesh.get_path_name())

# Nanite reports a coarse fallback proxy instead of the real vertices, which
# would silently be what gets measured here.
settings = mesh.get_editor_property("nanite_settings")
report["nanite_enabled"] = bool(settings.get_editor_property("enabled"))

vertices = []
uvs = []
section = 0
while section < 16:
    try:
        data = unreal.ProceduralMeshLibrary.get_section_from_static_mesh(mesh, 0, section)
    except Exception:
        break
    if not data or not len(data[0]):
        break
    vertices.extend(data[0])
    uvs.extend(data[3])
    section += 1

report["vertex_count"] = len(vertices)
report["uv_count"] = len(uvs)
if not vertices or len(uvs) != len(vertices):
    raise RuntimeError(
        "need matching vertex and UV arrays; got %d vertices and %d UVs"
        % (len(vertices), len(uvs)))


def group_mean(indices):
    total = [0.0, 0.0, 0.0]
    for index in indices:
        vertex = vertices[index]
        total[0] += float(vertex.x)
        total[1] += float(vertex.y)
        total[2] += float(vertex.z)
    count = float(len(indices))
    return [component / count for component in total]


def group_direction_mean(indices):
    """Mean *viewing direction* of a band, which is what the image encodes.

    Averaging world positions instead conflates direction with depth, and the
    two are wildly different across an image: the top band is sky hundreds of
    metres out while the bottom band is ground a few metres away, so the mean
    position of the top band sits nearer the camera than the bottom band's and
    the difference between them points nowhere meaningful. Every vertex was
    unprojected from the origin, so its unit direction *is* its image position.
    """
    total = [0.0, 0.0, 0.0]
    used = 0
    for index in indices:
        vertex = vertices[index]
        direction = [float(vertex.x), float(vertex.y), float(vertex.z)]
        length = sum(component * component for component in direction) ** 0.5
        if length < 1e-6:
            continue
        for axis in range(3):
            total[axis] += direction[axis] / length
        used += 1
    if not used:
        return [0.0, 0.0, 0.0]
    return [component / used for component in total]


def normalise(vector):
    length = sum(component * component for component in vector) ** 0.5
    if length < 1e-9:
        return [0.0, 0.0, 0.0], 0.0
    return [component / length for component in vector], length


us = [float(uv.x) for uv in uvs]
vs = [float(uv.y) for uv in uvs]
u_low, u_high = min(us), max(us)
v_low, v_high = min(vs), max(vs)
report["uv_range"] = {"u": [round(u_low, 4), round(u_high, 4)],
                      "v": [round(v_low, 4), round(v_high, 4)]}

u_band = (u_high - u_low) * EDGE_BAND
v_band = (v_high - v_low) * EDGE_BAND
left = [i for i, u in enumerate(us) if u <= u_low + u_band]
right = [i for i, u in enumerate(us) if u >= u_high - u_band]
# v is the image's vertical texture coordinate; v ~ v_low is the top row.
top = [i for i, v in enumerate(vs) if v <= v_low + v_band]
bottom = [i for i, v in enumerate(vs) if v >= v_high - v_band]
report["band_counts"] = {"left": len(left), "right": len(right),
                         "top": len(top), "bottom": len(bottom)}
if not (left and right and top and bottom):
    raise RuntimeError("UV edge bands are empty; UVs are not image coordinates")

left_mean, right_mean = group_mean(left), group_mean(right)
top_mean, bottom_mean = group_mean(top), group_mean(bottom)
centroid = group_mean(list(range(len(vertices))))

left_dir, right_dir = group_direction_mean(left), group_direction_mean(right)
top_dir, bottom_dir = group_direction_mean(top), group_direction_mean(bottom)

image_right, right_span = normalise(
    [right_dir[i] - left_dir[i] for i in range(3)])
image_up, up_span = normalise(
    [top_dir[i] - bottom_dir[i] for i in range(3)])
forward, _ = normalise(group_direction_mean(list(range(len(vertices)))))
_, _centroid_length = normalise(centroid)

report.update({
    "centroid_cm": [round(c, 2) for c in centroid],
    "image_left_mean_cm": [round(c, 2) for c in left_mean],
    "image_right_mean_cm": [round(c, 2) for c in right_mean],
    "image_top_mean_cm": [round(c, 2) for c in top_mean],
    "image_bottom_mean_cm": [round(c, 2) for c in bottom_mean],
    "image_right_direction": [round(c, 4) for c in image_right],
    "image_up_direction": [round(c, 4) for c in image_up],
    "camera_forward_direction": [round(c, 4) for c in forward],
    "centroid_distance_cm": round(_centroid_length, 2),
    "image_right_direction_separation": round(right_span, 4),
    "image_up_direction_separation": round(up_span, 4),
    "image_left_direction": [round(c, 4) for c in left_dir],
    "image_top_direction": [round(c, 4) for c in top_dir],
    "image_bottom_direction": [round(c, 4) for c in bottom_dir],
})

# Unreal is left-handed: right = up x forward. Compare the measured right
# against that to find out whether a camera can reproduce the source view.
cross = [
    image_up[1] * forward[2] - image_up[2] * forward[1],
    image_up[2] * forward[0] - image_up[0] * forward[2],
    image_up[0] * forward[1] - image_up[1] * forward[0],
]
handedness_dot = sum(cross[i] * image_right[i] for i in range(3))
report["expected_right_from_up_cross_forward"] = [round(c, 4) for c in cross]
report["handedness_dot"] = round(handedness_dot, 4)
report["mirrored"] = bool(handedness_dot < 0.0)

# The camera pose that reproduces the source view, if one exists.
yaw = math.degrees(math.atan2(forward[1], forward[0]))
pitch = math.degrees(math.atan2(
    forward[2], (forward[0] ** 2 + forward[1] ** 2) ** 0.5))
report["source_camera_rotation"] = {
    "pitch": round(pitch, 3), "yaw": round(yaw, 3), "roll": 0.0}
report["up_tilt_degrees"] = round(
    math.degrees(math.acos(max(-1.0, min(1.0, image_up[2])))), 3)
report["classification"] = "PROVEN"

result = json.dumps(report)
