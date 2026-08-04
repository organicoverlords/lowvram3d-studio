"""World bounds of the actors a build owns, for framing an orbit around them.

Camera poses for an off-axis sweep have to come from the scene's own extent, not
from constants: this project's scenes range from a barn a few metres across to a
field two hundred metres out, and a fixed orbit radius would be inside one and
lost outside the other.

Excludes lights, sky and cameras, whose bounds are editor billboards rather than
anything you can look at.

Configure with a `SCENE_BOUNDS_REQUEST` global holding `owner_tag`.

    python -m uemcp python @unreal/scene_bounds.py --json
"""

import json

import unreal

REQUEST = globals().get("SCENE_BOUNDS_REQUEST") or {}
OWNER_TAG = REQUEST.get("owner_tag")
EXCLUDED = (unreal.Light, unreal.SkyAtmosphere, unreal.CameraActor)
CM_PER_M = 100.0

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

low = [float("inf")] * 3
high = [float("-inf")] * 3
counted = []
for actor in actor_subsystem.get_all_level_actors():
    tags = {str(t) for t in (actor.get_editor_property("tags") or [])}
    if (OWNER_TAG and OWNER_TAG not in tags) or isinstance(actor, EXCLUDED):
        continue
    if not actor.get_component_by_class(unreal.StaticMeshComponent):
        continue
    centre, half = actor.get_actor_bounds(False)
    for axis, (c, h) in enumerate(((centre.x, half.x), (centre.y, half.y),
                                   (centre.z, half.z))):
        low[axis] = min(low[axis], float(c) - float(h))
        high[axis] = max(high[axis], float(c) + float(h))
    counted.append(str(actor.get_actor_label()))

if not counted:
    raise RuntimeError("no owned mesh actors found for " + str(OWNER_TAG))

result = json.dumps({
    "schema_version": "scene_bounds_v1",
    "classification": "PROVEN",
    "owner_tag": OWNER_TAG,
    "actor_count": len(counted),
    "min_m": [round(v / CM_PER_M, 3) for v in low],
    "max_m": [round(v / CM_PER_M, 3) for v in high],
    "centre_m": [round((low[a] + high[a]) / 2 / CM_PER_M, 3) for a in range(3)],
    "size_m": [round((high[a] - low[a]) / CM_PER_M, 3) for a in range(3)],
})
