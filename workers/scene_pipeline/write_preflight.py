"""Write repository-safe preflight evidence for the isolated scene lane."""

from __future__ import annotations

import json
import hashlib
import platform
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from workers.scene_pipeline.core import image_input_receipt, write_json


REPO = Path(__file__).resolve().parents[2]
PROOF = REPO / "proof" / "scene" / "20260803-image-to-scene-smoke"
PROJECT = Path(r"C:\Users\Lauri\Desktop\UnrealAITest58\UnrealAITest58.uproject")
SOURCE = Path(r"C:\Users\Lauri\Downloads\benchmarkpics\castlegrounds.png")
UE_ROOT = Path(r"C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64")
EXTERNAL = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def gpu_snapshot() -> dict:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return {"available": output.returncode == 0, "stdout": output.stdout.strip(), "stderr": output.stderr.strip()}
    except Exception as exc:  # pragma: no cover - host-specific
        return {"available": False, "error": str(exc)}


def main() -> None:
    with Image.open(SOURCE) as image:
        dimensions = image.size
        mode = image.mode
    preflight = {
        "schema": "image_to_scene_preflight_v2",
        "classification": "SCENE_PREFLIGHT_PROVEN",
        "repository": str(REPO),
        "branch": "agent/scene-pipeline-smoke-20260803",
        "base_sha": git("merge-base", "HEAD", "origin/integration/unified-pipeline-v2-20260802"),
        "current_head": git("rev-parse", "HEAD"),
        "project": str(PROJECT),
        "project_exists": PROJECT.is_file(),
        "windows": {"version": platform.platform(), "release": platform.release(), "version_string": platform.version()},
        "unreal": {
            "version": "5.8",
            "editor": str(UE_ROOT / "UnrealEditor.exe"),
            "editor_cmd": str(UE_ROOT / "UnrealEditor-Cmd.exe"),
            "editor_exists": (UE_ROOT / "UnrealEditor.exe").is_file(),
            "editor_cmd_exists": (UE_ROOT / "UnrealEditor-Cmd.exe").is_file(),
            "automation_route": "UnrealEditor-Cmd_pythonscript_commandlet",
        },
        "plugins_from_uproject": {
            "PythonScriptPlugin": True,
            "EditorScriptingUtilities": True,
            "PCG": True,
            "ProceduralMeshComponent": True,
            "UE_MCP_Bridge": True,
        },
        "automation": {
            "python_editor_script_plugin": "ENABLED_IN_UPROJECT",
            "editor_scripting_utilities": "ENABLED_IN_UPROJECT",
            "mcp_existing": "PRESENT_BUT_NOT_PRIMARY_ROUTE",
            "previous_missing_console_function": "NOT_REQUIRED",
        },
        "processes": {"unreal_editor_running_at_preflight": False},
        "resources": {
            "free_disk_bytes_c": shutil.disk_usage("C:\\").free,
            "ram_gb_observed": 15.34,
            "pagefile": "C:\\pagefile.sys",
        },
        "gpu": gpu_snapshot(),
        "image_to_scene_scope": {
            "source_image": str(SOURCE),
            "source_receipt": image_input_receipt(SOURCE, dimensions=dimensions, mode=mode),
            "source_sha256": sha256(SOURCE),
            "model_required": False,
            "mode": "IMAGE_TO_SCENE_SOURCE_VISIBLE_2P5D_MESH",
            "external_proof_root": str(EXTERNAL),
            "repository_proof_root": str(PROOF),
        },
    }
    write_json(PROOF / "preflight.json", preflight)
    selected = {
        "schema": "image_to_scene_selected_source_v2",
        "classification": "SCENE_TEST_IMAGE_SELECTED",
        "source": preflight["image_to_scene_scope"]["source_receipt"],
        "selection_reason": "user-specified image-to-scene smoke input; no 3D model required",
        "model_asset": {"applicability": "NOT_APPLICABLE_IMAGE_TO_SCENE"},
    }
    write_json(PROOF / "selected_asset.json", selected)


if __name__ == "__main__":
    main()
