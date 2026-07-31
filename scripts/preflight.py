from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except Exception:
        return ""


def main() -> None:
    report = {
        "platform": platform.platform(),
        "python": sys.version,
        "git": shutil.which("git") or "",
        "node": shutil.which("node") or "",
        "npm": shutil.which("npm") or "",
        "blender": shutil.which("blender") or "",
        "nvidia_smi": shutil.which("nvidia-smi") or "",
        "gpu": run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"]),
        "ram_bytes": None,
        "pagefile": run(["wmic", "pagefile", "get", "AllocatedBaseSize,CurrentUsage,Name", "/format:list"]) if os.name == "nt" else "",
    }
    try:
        import psutil
        report["ram_bytes"] = psutil.virtual_memory().total
    except Exception:
        pass
    output = Path(os.environ.get("LOWVRAM3D_PREFLIGHT", "preflight.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    gpu = report["gpu"].lower()
    if "1660" in gpu and "6144" not in gpu and "6" not in gpu:
        print("WARNING: GTX 1660 detected but VRAM report was unexpected.")


if __name__ == "__main__":
    main()
