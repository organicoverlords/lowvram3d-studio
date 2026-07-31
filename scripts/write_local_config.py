from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--discovery", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    discovery = json.loads(Path(args.discovery).read_text(encoding="utf-8-sig"))
    cfg = {
        "install_root": str(root),
        "jobs_root": str(root / "jobs"),
        "comfyui_url": "http://127.0.0.1:8188",
        "comfyui_path": discovery.get("comfyui_path", ""),
        "mini_turbo_workflow": discovery.get("mini_turbo_workflow", ""),
        "blender_path": discovery.get("blender_path", "") or "blender",
        "studio_url": "http://127.0.0.1:8311",
        "meshtools_url": "http://127.0.0.1:8200",
        "vram_ceiling_mb": 5600,
        "target_faces": 50000,
        "texture_size": 2048,
        "lane_order": ["A", "B", "C"],
        "mv_adapter_python": str(root / "envs" / "mv-adapter" / "Scripts" / "python.exe"),
        "mv_adapter_root": str(root / "thirdparty" / "MV-Adapter"),
        "sf3d_python": str(root / "envs" / "sf3d" / "Scripts" / "python.exe"),
        "sf3d_root": str(root / "thirdparty" / "stable-fast-3d"),
        "tripo_python": "",
        "tripo_root": "",
        "models_offline": False,
        "keep_intermediates": True,
        "extra": {
            "comfyui_python": discovery.get("comfyui_python", ""),
            "studio_root": str(root / "thirdparty" / "3DGenStudio"),
            "preferred_lane": "A",
            "fallback_lane_1": "B",
            "fallback_lane_2": "C"
        }
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
