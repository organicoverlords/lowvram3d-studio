"""Drive the running Unreal Editor from the pipeline, automatically.

The pipeline previously stopped at plans and manifests, and someone had to run
the Unreal half by hand. That is the difference between a pipeline and a set of
scripts, and it is why "the scene was built" kept meaning "a manifest said it
would be".

This stage builds the scene for real, and like the depth stage it degrades
visibly rather than failing the run: if no editor is reachable the receipt says
so and the pipeline continues, so a headless run still produces its plans.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# uemcp lives beside the pipeline rather than in site-packages.
UEMCP_DIR = Path(__file__).resolve().parents[2] / "unreal"
BUILDER = UEMCP_DIR / "build_structural_scene.py"


def normalise_package_root(value: str | None) -> str | None:
    """Recover an Unreal package path that a POSIX shell may have mangled.

    MSYS/Git Bash rewrites a leading-slash argument into a Windows path, so
    `--output-root /Game/AgentProof/X` arrives as
    `C:/Program Files/Git/Game/AgentProof/X` and the editor silently creates the
    map somewhere unintended. Recover the `/Game/...` tail rather than trusting
    the string.
    """
    if not value:
        return None
    text = str(value).replace("\\", "/").rstrip("/")
    marker = "/Game/"
    if text.startswith("/Game/"):
        return text
    index = text.find(marker)
    if index != -1:
        return text[index:]
    if text.startswith("Game/"):
        return "/" + text
    return None


def _bridge():
    if str(UEMCP_DIR) not in sys.path:
        sys.path.insert(0, str(UEMCP_DIR))
    from uemcp import Bridge  # noqa: PLC0415 - optional dependency by design

    return Bridge()


def build_scene(placement: dict[str, Any], scene_id: str,
                output_root: str | None = None,
                generated_assets: dict[str, Any] | None = None,
                timeout: float = 900.0) -> dict[str, Any]:
    """Spawn the structural scene in the live editor."""
    if not placement.get("actors"):
        return {"available": False, "reason": "placement produced no actors"}

    try:
        bridge = _bridge()
        if not bridge.is_ready():
            return {"available": False,
                    "reason": "editor is not ready; it may be loading or blocked "
                              "by a modal dialog"}
    except Exception as exc:
        return {"available": False,
                "reason": f"no editor bridge: {type(exc).__name__}: {exc}"}

    request: dict[str, Any] = {"placement": placement, "scene_id": scene_id}
    if generated_assets and generated_assets.get("assets"):
        # Only the fields the builder reads: the manifest also carries crops and
        # per-attempt VRAM telemetry, and the request crosses the bridge as one
        # JSON literal.
        request["generated_assets"] = {"assets": [
            {"asset_id": asset["asset_id"], "glb": asset["glb"],
             "status": asset["status"], "triangles": asset.get("triangles")}
            for asset in generated_assets["assets"]
            if asset.get("status") == "generated" and asset.get("glb")]}
    root = normalise_package_root(output_root)
    if root:
        request["package_root"] = root
        request["map_path"] = f"{root}/Maps/L_{scene_id}"

    try:
        # Assign the request in __main__ first: the builder runs with
        # ExecuteFile semantics in that same scope and reads it as a global.
        bridge.python(f"STRUCTURAL_REQUEST = {json.dumps(request)}",
                      "STRUCTURAL_REQUEST", timeout=timeout)
        receipt = bridge.python_json(
            BUILDER.read_text(encoding="utf-8"), "result", timeout=timeout)
    except Exception as exc:
        return {"available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "builder": str(BUILDER)}

    receipt["available"] = True
    return receipt


def capture_scene(output: Path, scene_id: str, fov_deg: float,
                  width: int = 1280, height: int = 720,
                  timeout: float = 600.0) -> dict[str, Any]:
    """Render the built scene from its own camera, as evidence it exists."""
    try:
        bridge = _bridge()
        output.parent.mkdir(parents=True, exist_ok=True)
        result = bridge.call("capture_scene_png", {
            "outputPath": str(output), "width": int(width), "height": int(height),
            "fov": float(fov_deg), "world": "editor",
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        }, timeout=timeout)
        result["available"] = True
        return result
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
