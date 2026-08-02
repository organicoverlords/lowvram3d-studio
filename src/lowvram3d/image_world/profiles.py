"""Hardware and quality profiles for the image-to-world pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ContractError


@dataclass(frozen=True)
class HardwareProfile:
    profile_id: str
    max_gpu_memory_mb: int
    max_system_memory_mb: int
    heavy_gpu_workers: int
    preview_input_long_edge: int
    default_geometry_tokens: int
    preferred_precision: str
    blender_renderer: str

    def validate(self) -> None:
        if not self.profile_id:
            raise ContractError("hardware profile id is required")
        if self.max_gpu_memory_mb <= 0 or self.max_system_memory_mb <= 0:
            raise ContractError("memory limits must be positive")
        if self.heavy_gpu_workers != 1:
            raise ContractError("the low-VRAM profile requires exactly one heavy GPU worker")
        if self.preview_input_long_edge < 256:
            raise ContractError("preview input edge is unreasonably small")
        if self.default_geometry_tokens <= 0:
            raise ContractError("geometry tokens must be positive")
        if self.preferred_precision not in {"fp16", "fp32"}:
            raise ContractError("precision must be fp16 or fp32")


@dataclass(frozen=True)
class QualityProfile:
    profile_id: str
    terrain_resolution: int
    source_long_edge: int
    erosion_iterations: int
    create_residual_meshes: bool
    export_unreal: bool

    def validate(self) -> None:
        if self.terrain_resolution not in {513, 1009, 2017, 4033, 8129}:
            raise ContractError("terrain resolution must be a supported Unreal Landscape size")
        if self.source_long_edge < 256:
            raise ContractError("source long edge is unreasonably small")
        if self.erosion_iterations < 0:
            raise ContractError("erosion iterations cannot be negative")


def gtx1660_super_6gb() -> HardwareProfile:
    profile = HardwareProfile(
        profile_id="windows_gtx1660_super_6gb",
        max_gpu_memory_mb=5600,
        max_system_memory_mb=15000,
        heavy_gpu_workers=1,
        preview_input_long_edge=768,
        default_geometry_tokens=1200,
        preferred_precision="fp16",
        blender_renderer="BLENDER_EEVEE_NEXT",
    )
    profile.validate()
    return profile


QUALITY_PROFILES = {
    "preview": QualityProfile(
        profile_id="preview",
        terrain_resolution=513,
        source_long_edge=768,
        erosion_iterations=8,
        create_residual_meshes=False,
        export_unreal=False,
    ),
    "production": QualityProfile(
        profile_id="production",
        terrain_resolution=1009,
        source_long_edge=1024,
        erosion_iterations=32,
        create_residual_meshes=True,
        export_unreal=True,
    ),
    "hero": QualityProfile(
        profile_id="hero",
        terrain_resolution=2017,
        source_long_edge=1448,
        erosion_iterations=64,
        create_residual_meshes=True,
        export_unreal=True,
    ),
}

for _profile in QUALITY_PROFILES.values():
    _profile.validate()
