"""Hide or show every actor a build owns, so a render can be differenced.

Measuring how much of a frame the scene actually fills means separating scene
pixels from background, and guessing that from colour is unreliable -- a dark
ground and a dark tree are not distinguishable from an unlit backdrop by
threshold. Rendering the same pose twice, once with the build's actors hidden,
gives the answer exactly: whatever changed is the scene.

Editor captures respect `set_is_temporarily_hidden_in_editor`, not the in-game
hidden flag, which is why that is the one used here.

Configure with a `TOGGLE_REQUEST` global holding `owner_tag` and `hidden`.

    python -m uemcp python @unreal/toggle_owned_actors.py --json
"""

import json

import unreal

REQUEST = globals().get("TOGGLE_REQUEST") or {}
OWNER_TAG = REQUEST["owner_tag"]
HIDDEN = bool(REQUEST.get("hidden", True))
# Lights and sky are the backdrop, not the scene: hiding them would change the
# exposure between the two renders and swamp the difference.
KEEP_VISIBLE = (unreal.Light, unreal.SkyAtmosphere, unreal.CameraActor)

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

toggled = []
for actor in actor_subsystem.get_all_level_actors():
    tags = {str(t) for t in (actor.get_editor_property("tags") or [])}
    if OWNER_TAG not in tags or isinstance(actor, KEEP_VISIBLE):
        continue
    actor.set_is_temporarily_hidden_in_editor(HIDDEN)
    toggled.append(str(actor.get_actor_label()))

result = json.dumps({
    "schema_version": "toggle_owned_actors_v1",
    "classification": "PROVEN",
    "owner_tag": OWNER_TAG,
    "hidden": HIDDEN,
    "toggled_count": len(toggled),
})
