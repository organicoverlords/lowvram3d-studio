"""Capture one PIE proof image without the Slate-window screenshot route.

The previous bridge capture used a Slate-backed game-window screenshot. This
script uses Unreal's AutomationLibrary high-res game-viewport capture, which
does not include editor UI. It is intentionally run only after a fresh editor
session has been started and a PIE world is confirmed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import unreal


REPO_ROOT = Path(r"C:/Users/Lauri/Desktop/lowvram3d-scene-smoke-20260803")
EVIDENCE = REPO_ROOT / "evidence" / "latest-image-to-scene"
VIEWS = {"source_camera", "front", "three_quarter", "side", "rear", "top", "bridge", "vegetation"}

view = sys.argv[1] if len(sys.argv) > 1 else "source_camera"
if view not in VIEWS:
    raise RuntimeError(f"unknown proof view: {view}")
worlds = list(unreal.EditorLevelLibrary.get_pie_worlds(False))
if not worlds:
    raise RuntimeError("PIE world is not running")
world = worlds[0]
output = EVIDENCE / "screenshots" / f"{view}_automation.png"
output.parent.mkdir(parents=True, exist_ok=True)

# This API routes through the game viewport and omits Slate/editor UI. The
# returned task completes asynchronously; the receipt records the request and
# the next validation pass verifies the file before promotion.
task = unreal.AutomationLibrary.take_high_res_screenshot(960, 720, str(output), None, False, False)
result = {
    "schema_version": "pie_proof_capture_receipt_v1",
    "classification": "REQUESTED",
    "view": view,
    "world": str(world.get_path_name()),
    "output": str(output),
    "capture_api": "AutomationLibrary.take_high_res_screenshot",
    "slate_window_capture": False,
    "includes_editor_ui": False,
    "task": str(task),
}
(EVIDENCE / "screenshots" / f"{view}_automation_receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
unreal.log("PIE_AUTOMATION_CAPTURE_REQUESTED=" + view)
