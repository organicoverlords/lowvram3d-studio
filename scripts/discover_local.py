from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def candidates() -> list[Path]:
    home = Path.home()
    roots = [home / "Downloads", home / "Desktop", Path("C:/AI"), Path("C:/ComfyUI")]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("**/ComfyUI/main.py", "**/ComfyUI/README.md"):
            for marker in root.glob(pattern):
                folder = marker.parent
                if folder not in found:
                    found.append(folder)
                if len(found) >= 20:
                    return found
    return found


def find_blender() -> str:
    direct = shutil.which("blender")
    if direct:
        return direct
    base = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Blender Foundation"
    matches = sorted(base.glob("Blender */blender.exe"), reverse=True) if base.exists() else []
    return str(matches[0]) if matches else ""


def score_comfy(path: Path) -> int:
    score = 0
    names = [child.name.lower() for child in (path / "custom_nodes").iterdir()] if (path / "custom_nodes").exists() else []
    if any("trellis" in name for name in names): score += 4
    if any("3d" in name for name in names): score += 2
    if (path.parent / "python_embeded" / "python.exe").exists(): score += 2
    return score


def find_workflow(comfy: Path) -> str:
    roots = [comfy / "user", comfy / "workflows", comfy.parent / "workflows", comfy / "custom_nodes" / "ComfyUI-3D-Pack" / "example_workflows" / "Hunyuan3D_V2"]
    tokens = ("mini_turbo", "mini turbo", "hunyuan3d-2mini", "hunyuan3d2mini", "hunyuan3d-2mini-turbo")
    candidates: list[tuple[int, Path]] = []
    for root in roots:
        if not root.exists(): continue
        for path in root.rglob("*.json"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            if any(token in text for token in tokens):
                filename = path.name.lower().replace("-", "_")
                score = 20 if "mini_turbo" in filename else 10 if "mini" in filename and "turbo" in filename else 0
                score += 5 if "user" in {part.lower() for part in path.parts} else 0
                candidates.append((score, path))
    return str(max(candidates, key=lambda item: item[0])[1]) if candidates else ""


def main() -> None:
    comfy_list = sorted(candidates(), key=score_comfy, reverse=True)
    comfy = comfy_list[0] if comfy_list else None
    embedded = ""
    if comfy:
        for p in (comfy.parent / "python_embeded" / "python.exe", comfy.parent / "python_embedded" / "python.exe"):
            if p.exists(): embedded = str(p); break
    result = {
        "comfyui_path": str(comfy) if comfy else "",
        "comfyui_python": embedded,
        "mini_turbo_workflow": find_workflow(comfy) if comfy else "",
        "blender_path": find_blender(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
