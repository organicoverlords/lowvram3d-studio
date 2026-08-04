"""Measure how much the scene's actors interpenetrate each other.

The assembled scene looks wrong in a way that is easy to describe loosely --
"the trees swallow the barn" -- and hard to act on. Overlap is measurable: two
actors that occupy the same cubic metres are a placement defect regardless of
how the render happens to be framed, and the number says which pair and by how
much.

Reports every pair whose world bounding boxes intersect, as a fraction of the
smaller actor's volume, so a building buried inside a tree reads as ~1.0 and two
trees brushing shoulders reads as a few per cent.

Bounding boxes overstate overlap for concave shapes, so this is an upper bound
and a screening tool, not a collision test. It is still the right instrument
for the failure it is aimed at.

Configure with an `OVERLAP_AUDIT_REQUEST` global holding `owner_tag`, or nothing
to audit every actor in the level.

    python -m uemcp python @unreal/audit_actor_overlaps.py --json
"""

import json

import unreal

REQUEST = globals().get("OVERLAP_AUDIT_REQUEST") or {}
OWNER_TAG = REQUEST.get("owner_tag")
# Below this fraction two actors are merely touching, which scenes do.
REPORT_THRESHOLD = float(REQUEST.get("threshold", 0.02))
CM_PER_M = 100.0

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

entries = []
for actor in actor_subsystem.get_all_level_actors():
    if not actor.get_component_by_class(unreal.StaticMeshComponent):
        continue
    # A CameraActor carries a StaticMeshComponent for its editor billboard, and
    # its bounds are a 20 m box that overlaps everything near the origin.
    if isinstance(actor, (unreal.CameraActor, unreal.Light, unreal.SkyAtmosphere)):
        continue
    tags = {str(t) for t in (actor.get_editor_property("tags") or [])}
    if OWNER_TAG and OWNER_TAG not in tags:
        continue
    centre, half = actor.get_actor_bounds(False)
    if min(float(half.x), float(half.y), float(half.z)) <= 0.0:
        continue
    entries.append({
        "label": str(actor.get_actor_label()),
        "source": "generated" if "generated" in tags else "primitive",
        "min": [float(centre.x) - float(half.x), float(centre.y) - float(half.y),
                float(centre.z) - float(half.z)],
        "max": [float(centre.x) + float(half.x), float(centre.y) + float(half.y),
                float(centre.z) + float(half.z)],
        "volume_m3": (8.0 * float(half.x) * float(half.y) * float(half.z)
                      / (CM_PER_M ** 3)),
    })


def intersection_volume_m3(a, b):
    spans = [max(0.0, min(a["max"][axis], b["max"][axis])
                 - max(a["min"][axis], b["min"][axis])) for axis in range(3)]
    return spans[0] * spans[1] * spans[2] / (CM_PER_M ** 3)


overlaps = []
for index, first in enumerate(entries):
    for second in entries[index + 1:]:
        volume = intersection_volume_m3(first, second)
        if volume <= 0.0:
            continue
        smaller = min(first["volume_m3"], second["volume_m3"])
        fraction = volume / smaller if smaller > 0 else 0.0
        if fraction < REPORT_THRESHOLD:
            continue
        overlaps.append({
            "a": first["label"], "b": second["label"],
            "a_source": first["source"], "b_source": second["source"],
            "overlap_m3": round(volume, 2),
            "fraction_of_smaller": round(fraction, 4),
        })

overlaps.sort(key=lambda o: -o["fraction_of_smaller"])
buried = [o for o in overlaps if o["fraction_of_smaller"] >= 0.5]

result = json.dumps({
    "schema_version": "actor_overlap_audit_v1",
    "classification": "PROVEN",
    "owner_tag": OWNER_TAG,
    "actor_count": len(entries),
    "overlapping_pair_count": len(overlaps),
    # Half an actor's volume inside another is not a scene, it is a collision.
    "buried_pair_count": len(buried),
    "worst_fraction": overlaps[0]["fraction_of_smaller"] if overlaps else 0.0,
    "overlaps": overlaps[:40],
    "actors": [{"label": e["label"], "source": e["source"],
                "size_m": [round((e["max"][axis] - e["min"][axis]) / CM_PER_M, 2)
                           for axis in range(3)],
                "centre_m": [round((e["max"][axis] + e["min"][axis]) / 2 / CM_PER_M, 2)
                             for axis in range(3)]}
               for e in entries],
})
