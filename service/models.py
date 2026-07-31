from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    imageBase64: str = Field(alias="image_base64")
    imageMimeType: str = Field(default="image/png", alias="image_mime_type")
    imageFilename: str = Field(default="input.png", alias="image_filename")
    prompt: str = ""
    name: str = "Generated Mesh"
    projectId: str = Field(default="", alias="project_id")
    cardId: str = Field(default="", alias="card_id")

    model_config = {"populate_by_name": True}


class MeshRequest(BaseModel):
    meshBase64: str = Field(alias="mesh_base64")
    meshMimeType: str = Field(default="model/gltf-binary", alias="mesh_mime_type")
    meshFilename: str = Field(default="mesh.glb", alias="mesh_filename")
    prompt: str = ""
    name: str = "Processed Mesh"
    projectId: str = Field(default="", alias="project_id")
    cardId: str = Field(default="", alias="card_id")
    rigKind: str = Field(default="auto", alias="rig_kind")

    model_config = {"populate_by_name": True}


class PostProcessRequest(MeshRequest):
    assetType: str = Field(default="auto", alias="asset_type")
    qualityPreset: str = Field(default="gameplay", alias="quality_preset")
    separateMovableParts: bool = Field(default=True, alias="separate_movable_parts")
    textureResolution: int = Field(default=2048, alias="texture_resolution", ge=512, le=4096)
    lodEnabled: bool = Field(default=True, alias="lod_enabled")
    removeHiddenGeometry: bool = Field(default=False, alias="remove_hidden_geometry")
    experimentalSemanticSplit: bool = Field(default=False, alias="experimental_semantic_split")
    animationPreset: str = Field(default="dance", alias="animation_preset")
    resumeJobId: str = Field(default="", alias="resume_job_id")
    resumeFailedJob: bool = Field(default=True, alias="resume_failed_job")


class FullRequest(GenerateRequest):
    assetType: str = Field(default="auto", alias="asset_type")
    qualityPreset: str = Field(default="gameplay", alias="quality_preset")
    separateMovableParts: bool = Field(default=True, alias="separate_movable_parts")
    textureResolution: int = Field(default=2048, alias="texture_resolution", ge=512, le=4096)
    lodEnabled: bool = Field(default=True, alias="lod_enabled")
    removeHiddenGeometry: bool = Field(default=False, alias="remove_hidden_geometry")
    experimentalSemanticSplit: bool = Field(default=False, alias="experimental_semantic_split")
    backgroundRemoval: bool = Field(default=True, alias="background_removal")
    animationPreset: str = Field(default="dance", alias="animation_preset")
    resumeJobId: str = Field(default="", alias="resume_job_id")
    resumeFailedJob: bool = Field(default=True, alias="resume_failed_job")
