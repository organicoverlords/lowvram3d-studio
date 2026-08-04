"""Scene-local Unreal and evidence path derivation."""

from __future__ import annotations

import re
from pathlib import Path


def safe_scene_id(scene_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(scene_id)).strip("_")
    if not value:
        raise ValueError("scene_id must contain at least one safe character")
    return value


def derive_scene_paths(scene_id: str, output_root: str = "/Game/GeneratedScenes", evidence_root: str | Path | None = None) -> dict[str, str]:
    safe = safe_scene_id(scene_id)
    root = output_root.rstrip("/")
    evidence = Path(evidence_root) if evidence_root is not None else Path("evidence") / "scenes" / safe
    return {
        "scene_root": f"{root}/{safe}",
        "map": f"{root}/{safe}/Maps/L_{safe}_Generated",
        "geometry": f"{root}/{safe}/Geometry",
        "materials": f"{root}/{safe}/Materials",
        "pcg": f"{root}/{safe}/PCG",
        "evidence": str(evidence),
    }
