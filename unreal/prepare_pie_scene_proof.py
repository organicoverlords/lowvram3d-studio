"""Place the PIE pawn for one deterministic image-to-scene proof view."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import unreal


REPO_ROOT = Path(r"C:/Users/Lauri/Desktop/lowvram3d-scene-smoke-20260803")
OUT = REPO_ROOT / "evidence" / "latest-image-to-scene" / "screenshots" / "view_state.json"

VIEWS = {
    "source_camera": ([1000.0, 1712.5, 700.0], [0.0, 90.0, 0.0]),
    "front": ([1000.0, 2100.0, 900.0], [-8.0, 90.0, 0.0]),
    "three_quarter": ([450.0, 1900.0, 850.0], [-10.0, 55.0, 0.0]),
    "side": ([1000.0, 2500.0, 850.0], [-8.0, 0.0, 0.0]),
    "rear": ([1000.0, 3600.0, 900.0], [-8.0, -90.0, 0.0]),
    "top": ([1000.0, 2200.0, 2100.0], [-35.0, 90.0, 0.0]),
    "bridge": ([0.0, 4700.0, 650.0], [-8.0, 90.0, 0.0]),
    "vegetation": ([-1800.0, 3200.0, 1300.0], [-18.0, 35.0, 0.0]),
}

name = sys.argv[1] if len(sys.argv) > 1 else "source_camera"
if name not in VIEWS:
    raise RuntimeError(f"unknown proof view: {name}")
worlds = list(unreal.EditorLevelLibrary.get_pie_worlds(False))
if not worlds:
    raise RuntimeError("PIE world is not running")
world = worlds[0]
pawn = unreal.GameplayStatics.get_player_pawn(world, 0)
if pawn is None:
    raise RuntimeError("PIE pawn is unavailable")
location, rotation = VIEWS[name]
pawn.set_actor_location(unreal.Vector(*location), False, False)
controller = pawn.get_controller()
if controller is not None:
    controller.set_control_rotation(unreal.Rotator(*rotation))
unreal.SystemLibrary.execute_console_command(world, "DisableAllScreenMessages")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"view": name, "location_cm": location, "rotation_deg": rotation, "world": str(world.get_path_name())}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
unreal.log("PIE_PROOF_VIEW=" + name)
