"""Measure how much the scene's actors interpenetrate each other.

The assembled scene looks wrong in a way that is easy to describe loosely --
"the trees swallow the barn" -- and hard to act on. Overlap is measurable: two
actors that occupy the same cubic metres are a placement defect regardless of
how the render happens to be framed, and the number says which pair and by how
much.

Reports every pair whose world bounding boxes intersect, as a fraction of the
smaller actor's volume, so a building buried inside a tree reads as ~1.0 and two
trees brushing shoulders reads as a few per cent.

Pairs are split by whether they belong to the same region. Instances of one
region are a decomposition of a single continuous mass -- twelve clumps of one
hedge are meant to abut -- so counting them alongside genuine object collisions
made the headline figure describe the wrong thing. On the barn scene the
all-pairs worst was 0.79 and all of it was hedge against hedge, while the real
cross-object defect measured 0.26. `worst_fraction` is now cross-region;
`worst_fraction_any_pair` is kept beside it so the two cannot be conflated
again by accident.

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
    region = next((tag.partition(":")[2] for tag in tags
                   if tag.startswith("region:")), None)
    entries.append({
        "label": str(actor.get_actor_label()),
        "region": region,
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
        same_region = bool(first["region"] and first["region"] == second["region"])
        overlaps.append({
            "a": first["label"], "b": second["label"],
            "a_source": first["source"], "b_source": second["source"],
            "region": first["region"] if same_region else None,
            # Instances of one region are a decomposition of a single mass:
            # twelve clumps of one hedge are meant to abut. Counting those the
            # same as a barn inside a tree is what made this audit's headline
            # figure meaningless -- on the barn scene the all-pairs worst was
            # 0.79 and every bit of it was hedge touching hedge, while the
            # genuine cross-object defect was 0.26.
            "same_region": same_region,
            "overlap_m3": round(volume, 2),
            "fraction_of_smaller": round(fraction, 4),
        })

overlaps.sort(key=lambda o: -o["fraction_of_smaller"])
cross_region = [o for o in overlaps if not o["same_region"]]
buried = [o for o in cross_region if o["fraction_of_smaller"] >= 0.5]

result = json.dumps({
    "schema_version": "actor_overlap_audit_v2",
    "classification": "PROVEN",
    "owner_tag": OWNER_TAG,
    "actor_count": len(entries),
    "overlapping_pair_count": len(overlaps),
    "cross_region_pair_count": len(cross_region),
    # Half an actor's volume inside another is not a scene, it is a collision.
    # Counted across regions only, for the reason recorded on `same_region`.
    "buried_pair_count": len(buried),
    # The figure to judge a scene by. `worst_fraction_any_pair` is kept beside
    # it so the two are never silently conflated again, and because a region
    # whose own instances sit almost entirely inside each other is still worth
    # seeing -- it means the clustering split one thing into copies.
    "worst_fraction": (cross_region[0]["fraction_of_smaller"]
                       if cross_region else 0.0),
    "worst_fraction_any_pair": (overlaps[0]["fraction_of_smaller"]
                                if overlaps else 0.0),
    "region_tags_present": bool([e for e in entries if e["region"]]),
    "overlaps": overlaps[:40],
    "actors": [{"label": e["label"], "source": e["source"],
                "size_m": [round((e["max"][axis] - e["min"][axis]) / CM_PER_M, 2)
                           for axis in range(3)],
                "centre_m": [round((e["max"][axis] + e["min"][axis]) / 2 / CM_PER_M, 2)
                             for axis in range(3)]}
               for e in entries],
})
