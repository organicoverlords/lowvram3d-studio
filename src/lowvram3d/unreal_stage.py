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
import time
from pathlib import Path
from typing import Any

# uemcp lives beside the pipeline rather than in site-packages.
UEMCP_DIR = Path(__file__).resolve().parents[2] / "unreal"
BUILDER = UEMCP_DIR / "build_structural_scene.py"
MESH_IMPORTER = UEMCP_DIR / "import_generated_mesh.py"


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


def _is_handler_timeout(exc: Exception) -> bool:
    """Did the editor run out of handler time, or did the script fail?

    The bridge reports both as the same exception type, and treating a script
    error as a timeout means waiting out the whole settle window for work that
    was never started.
    """
    return "timed out" in str(exc).lower()


def import_generated_meshes(generated_assets: dict[str, Any],
                            package_root: str,
                            timeout: float = 900.0,
                            settle_timeout: float = 1800.0) -> dict[str, Any]:
    """Import each generated GLB into the project, one call per mesh.

    A million-triangle import outlives the bridge's handler timeout while the
    editor goes on and finishes it, so a raised timeout here means "not yet",
    not "failed". Re-running the import in that state would queue a second copy
    of an import that is already running, so poll the editor instead.
    """
    receipt: dict[str, Any] = {"schema_version": "generated_mesh_import_receipt_v1",
                               "package_root": package_root, "meshes": {},
                               "failures": []}
    assets = [a for a in generated_assets.get("assets", [])
              if a.get("status") == "generated" and a.get("glb")]
    if not assets:
        receipt["classification"] = "EMPTY"
        return receipt

    try:
        bridge = _bridge()
    except Exception as exc:
        receipt["classification"] = "UNAVAILABLE"
        receipt["reason"] = f"{type(exc).__name__}: {exc}"
        return receipt

    for asset in assets:
        destination = f"{package_root}/GeneratedMeshes/{asset['asset_id']}"
        # The editor resolves a relative path against the project directory.
        request = {"glb": str(Path(asset["glb"]).resolve()),
                   "destination": destination}
        code = MESH_IMPORTER.read_text(encoding="utf-8")
        try:
            bridge.python("MESH_IMPORT_REQUEST = " + json.dumps(request),
                          "MESH_IMPORT_REQUEST", timeout=120.0)
            result = bridge.python_json(code, "result", timeout=timeout)
        except Exception as exc:
            # Only a *timeout* means "still working". Anything else is a real
            # error, and polling through it waits half an hour for a mesh that
            # will never appear -- which is exactly what a relative GLB path,
            # unresolvable from the editor's own directory, did here.
            if not _is_handler_timeout(exc):
                receipt["failures"].append({
                    "asset_id": asset["asset_id"], "glb": request["glb"],
                    "error": f"{type(exc).__name__}: {exc}"})
                continue
            # The editor is still importing. Poll in *query-only* mode: asking
            # the importer again would start a second import of the same file
            # on every poll, which is worse than waiting.
            deadline = time.monotonic() + settle_timeout
            probe = {**request, "query_only": True}
            result = None
            while time.monotonic() < deadline:
                time.sleep(20.0)
                try:
                    bridge.python("MESH_IMPORT_REQUEST = " + json.dumps(probe),
                                  "MESH_IMPORT_REQUEST", timeout=120.0)
                    result = bridge.python_json(code, "result", timeout=300.0)
                    break
                except Exception:
                    continue
            if result is None:
                receipt["failures"].append({
                    "asset_id": asset["asset_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "waited_seconds": settle_timeout})
                continue
            result["completed_after_handler_timeout"] = True
        receipt["meshes"][asset["asset_id"]] = result

    receipt["imported_count"] = len(receipt["meshes"])
    receipt["classification"] = (
        "PROVEN" if receipt["meshes"] and not receipt["failures"]
        else "PARTIAL" if receipt["meshes"] else "FAILED")
    return receipt


def build_scene(placement: dict[str, Any], scene_id: str,
                output_root: str | None = None,
                imported_meshes: dict[str, Any] | None = None,
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
    if imported_meshes and imported_meshes.get("meshes"):
        # Only the fields the builder reads: the import receipt also carries
        # bounds and telemetry, and the request crosses the bridge as one JSON
        # literal.
        request["generated_assets"] = {"meshes": {
            asset_id: {"static_mesh": entry["static_mesh"],
                       "triangles": entry.get("triangles")}
            for asset_id, entry in imported_meshes["meshes"].items()
            if entry.get("static_mesh")}}
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
                  location: tuple[float, float, float] = (0.0, 0.0, 0.0),
                  rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
                  timeout: float = 600.0) -> dict[str, Any]:
    """Render the built scene, by default from its own source camera.

    `location` is (x, y, z) in centimetres and `rotation` is (pitch, yaw, roll)
    in degrees, so a caller can also take the off-axis views that are the one
    metric both pipelines have to satisfy -- and which the source camera, by
    construction, cannot provide.
    """
    try:
        bridge = _bridge()
        output.parent.mkdir(parents=True, exist_ok=True)
        # The editor resolves a relative path against the *project* directory,
        # so a render invoked from the repo silently lands in UnrealAITest58/
        # and the evidence directory stays empty.
        output = output.resolve()
        result = bridge.call("capture_scene_png", {
            "outputPath": str(output), "width": int(width), "height": int(height),
            "fov": float(fov_deg), "world": "editor",
            "location": {"x": float(location[0]), "y": float(location[1]),
                         "z": float(location[2])},
            "rotation": {"pitch": float(rotation[0]), "yaw": float(rotation[1]),
                         "roll": float(rotation[2])},
        }, timeout=timeout)
        result["available"] = True
        return result
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
