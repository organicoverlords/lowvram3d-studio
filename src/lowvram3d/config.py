from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PipelineConfig:
    install_root: Path
    jobs_root: Path
    comfyui_url: str = "http://127.0.0.1:8188"
    comfyui_path: str = ""
    mini_turbo_workflow: str = ""
    blender_path: str = "blender"
    studio_url: str = "http://127.0.0.1:8311"
    meshtools_url: str = "http://127.0.0.1:8200"
    vram_ceiling_mb: int = 5600
    target_faces: int = 50000
    texture_size: int = 2048
    lane_order: list[str] = field(default_factory=lambda: ["A", "B", "C"])
    mv_adapter_python: str = ""
    mv_adapter_root: str = ""
    # stabilityai/stable-diffusion-2-1-base was removed from Hugging Face, so Lane A can only
    # resolve its base model from disk. A path to a diffusers layout is accepted here.
    sd21_base_model: str = "stabilityai/stable-diffusion-2-1-base"
    sf3d_python: str = ""
    sf3d_root: str = ""
    tripo_python: str = ""
    tripo_root: str = ""
    models_offline: bool = False
    keep_intermediates: bool = True
    # The Cycles-bake fallback texture route (blender/project_texture.py) measured ~10min+ on a
    # 980k-face mesh and was killed by timeout in testing. The raster route (numpy/opencv UV-atlas
    # projection, no Cycles) does the same job in ~2s. Kept switchable so the old path can still be
    # used as an explicit diagnostic fallback.
    use_raster_texture_route: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "PipelineConfig":
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        root = Path(os.path.expandvars(data.get("install_root", str(source.parent.parent)))).resolve()
        jobs = Path(os.path.expandvars(data.get("jobs_root", str(root / "jobs")))).resolve()
        return cls(
            install_root=root,
            jobs_root=jobs,
            comfyui_url=data.get("comfyui_url", "http://127.0.0.1:8188").rstrip("/"),
            comfyui_path=os.path.expandvars(data.get("comfyui_path", "")),
            mini_turbo_workflow=os.path.expandvars(data.get("mini_turbo_workflow", "")),
            blender_path=os.path.expandvars(data.get("blender_path", "blender")),
            studio_url=data.get("studio_url", "http://127.0.0.1:8311").rstrip("/"),
            meshtools_url=data.get("meshtools_url", "http://127.0.0.1:8200").rstrip("/"),
            vram_ceiling_mb=int(data.get("vram_ceiling_mb", 5600)),
            target_faces=int(data.get("target_faces", 50000)),
            texture_size=int(data.get("texture_size", 2048)),
            lane_order=[str(v).upper() for v in data.get("lane_order", ["A", "B", "C"])],
            mv_adapter_python=os.path.expandvars(data.get("mv_adapter_python", "")),
            mv_adapter_root=os.path.expandvars(data.get("mv_adapter_root", "")),
            sd21_base_model=os.path.expandvars(data.get("sd21_base_model", "stabilityai/stable-diffusion-2-1-base")),
            sf3d_python=os.path.expandvars(data.get("sf3d_python", "")),
            sf3d_root=os.path.expandvars(data.get("sf3d_root", "")),
            tripo_python=os.path.expandvars(data.get("tripo_python", "")),
            tripo_root=os.path.expandvars(data.get("tripo_root", "")),
            models_offline=bool(data.get("models_offline", False)),
            keep_intermediates=bool(data.get("keep_intermediates", True)),
            use_raster_texture_route=bool(data.get("use_raster_texture_route", True)),
            extra=data.get("extra", {}),
        )

    def ensure_dirs(self) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
