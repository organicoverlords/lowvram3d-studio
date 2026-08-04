"""Collapse duplicate scene lights down to one of each label.

The environment builder was not idempotent and re-ran, so the level ended up
with three identical `SunLight` directional lights and four `SkyLight`s stacked
on the level's own pair. Unreal can only pick one directional light for forward
shading, translucency, water and volumetric fog, which is the
"multiple directional lights are competing" warning in the viewport.

The extras are switched off rather than deleted: `affects_world = False`
removes them from every lighting calculation while leaving the actors in place,
so this is reversible and the level is not restructured behind the user's back.

Run through the bridge:

    python -m uemcp python @unreal/fix_duplicate_lights.py --json
"""

import json

import unreal

LIGHT_CLASSES = ("DirectionalLight", "SkyLight")

subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

seen: set[str] = set()
kept: list[dict] = []
disabled: list[dict] = []

for actor in subsystem.get_all_level_actors():
    class_name = str(actor.get_class().get_name())
    if class_name not in LIGHT_CLASSES:
        continue

    label = str(actor.get_actor_label())
    key = f"{class_name}:{label}"
    components = list(actor.get_components_by_class(unreal.LightComponentBase))
    record = {"label": label, "class": class_name,
              "path": str(actor.get_path_name())}

    if key not in seen:
        seen.add(key)
        # Make sure the survivor is actually contributing.
        for component in components:
            try:
                component.set_editor_property("affects_world", True)
            except Exception:
                pass
        kept.append(record)
        continue

    for component in components:
        try:
            component.set_editor_property("affects_world", False)
        except Exception:
            pass
    disabled.append(record)

result = json.dumps({"kept": kept, "disabled": disabled,
                     "kept_count": len(kept), "disabled_count": len(disabled)})
