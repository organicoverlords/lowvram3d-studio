"""Declarative DAG for the reusable image-to-scene pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    version: str
    dependencies: tuple[str, ...]
    required_inputs: tuple[str, ...]
    produced_outputs: tuple[str, ...]
    resource_requirements: Mapping[str, Any]
    proof_gates: tuple[str, ...]
    invalidation_rules: tuple[str, ...]
    repair_targets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_STAGES = (
    "run_initialization", "source_image_audit", "camera_estimation", "depth_estimation",
    "surface_orientation", "semantic_segmentation", "instance_decomposition",
    "structural_feature_extraction", "material_region_analysis", "scene_type_classification",
    "scene_graph_construction", "world_scale_inference", "visibility_occlusion_analysis",
    "unseen_world_completion", "representation_selection", "scene_completeness", "asset_strategy",
    "terrain_planning", "architecture_planning", "path_water_crossing_planning", "vegetation_planning",
    "gameplay_planning", "visual_shell_generation", "terrain_generation", "architecture_generation",
    "water_generation", "crossing_generation", "hero_object_generation", "vegetation_generation",
    "world_continuation", "material_generation", "material_harmonization", "lighting_inference",
    "atmosphere_generation", "unreal_import", "unreal_assembly", "collision_generation",
    "navigation_generation", "runtime_setup", "structural_validation", "source_view_validation",
    "offset_view_validation", "gameplay_validation", "visual_defect_detection", "performance_validation",
    "automatic_repair", "final_packaging", "cross_scene_regression",
)


def _resource(stage_id: str) -> dict[str, Any]:
    unreal = stage_id.startswith(("unreal_", "collision_", "navigation_", "runtime_", "gameplay_")) or stage_id in {"terrain_generation", "architecture_generation", "water_generation", "crossing_generation", "hero_object_generation", "vegetation_generation"}
    gpu = stage_id in {"depth_estimation", "semantic_segmentation", "material_generation"}
    return {
        "requires_unreal": unreal,
        "requires_blender": stage_id in {"visual_shell_generation", "material_harmonization"},
        "cpu_inference": stage_id in {"source_image_audit", "camera_estimation", "scene_graph_construction", "asset_strategy", "visual_defect_detection"},
        "gpu_inference": gpu,
        "max_vram_mb": 6144 if gpu else 0,
        "exclusive_gpu": gpu or unreal,
    }


def _make_stages() -> tuple[StageSpec, ...]:
    stages: list[StageSpec] = []
    previous: str | None = None
    for stage_id in _STAGES:
        deps = (previous,) if previous else ()
        if stage_id == "scene_graph_construction":
            deps = ("semantic_segmentation", "instance_decomposition", "structural_feature_extraction", "material_region_analysis")
        elif stage_id == "representation_selection":
            deps = ("scene_graph_construction", "world_scale_inference", "visibility_occlusion_analysis", "unseen_world_completion")
        elif stage_id == "scene_completeness":
            deps = ("representation_selection",)
        elif stage_id == "asset_strategy":
            deps = ("scene_completeness", "representation_selection")
        elif stage_id in {"terrain_planning", "architecture_planning", "path_water_crossing_planning", "vegetation_planning", "gameplay_planning"}:
            deps = ("asset_strategy",)
        elif stage_id in {"visual_shell_generation", "terrain_generation", "architecture_generation", "water_generation", "crossing_generation", "hero_object_generation", "vegetation_generation"}:
            deps = ("scene_completeness", "asset_strategy")
        elif stage_id == "world_continuation":
            deps = ("terrain_generation", "architecture_generation", "water_generation", "crossing_generation", "vegetation_generation")
        elif stage_id == "material_generation":
            deps = ("world_continuation", "material_region_analysis")
        elif stage_id == "material_harmonization":
            deps = ("material_generation", "visual_shell_generation")
        elif stage_id == "lighting_inference":
            deps = ("scene_graph_construction",)
        elif stage_id == "atmosphere_generation":
            deps = ("lighting_inference",)
        elif stage_id == "unreal_import":
            deps = ("visual_shell_generation", "terrain_generation", "architecture_generation", "water_generation", "crossing_generation", "vegetation_generation")
        elif stage_id == "unreal_assembly":
            deps = ("unreal_import", "material_harmonization", "atmosphere_generation")
        elif stage_id == "collision_generation":
            deps = ("unreal_assembly", "gameplay_planning")
        elif stage_id == "navigation_generation":
            deps = ("collision_generation",)
        elif stage_id == "runtime_setup":
            deps = ("navigation_generation",)
        elif stage_id == "structural_validation":
            deps = ("runtime_setup",)
        elif stage_id == "source_view_validation":
            deps = ("structural_validation", "camera_estimation")
        elif stage_id == "offset_view_validation":
            deps = ("source_view_validation", "world_continuation", "unseen_world_completion")
        elif stage_id == "gameplay_validation":
            deps = ("navigation_generation", "runtime_setup")
        elif stage_id == "visual_defect_detection":
            deps = ("source_view_validation", "offset_view_validation", "gameplay_validation")
        elif stage_id == "performance_validation":
            deps = ("structural_validation", "gameplay_validation")
        elif stage_id == "automatic_repair":
            deps = ("visual_defect_detection", "performance_validation")
        elif stage_id == "final_packaging":
            deps = ("automatic_repair", "performance_validation")
        elif stage_id == "cross_scene_regression":
            deps = ("final_packaging",)
        stages.append(StageSpec(
            stage_id=stage_id,
            version="1.0.0",
            dependencies=tuple(d for d in deps if d),
            required_inputs=("scene_spec", "source_sha256"),
            produced_outputs=(f"{stage_id}.json",),
            resource_requirements=_resource(stage_id),
            proof_gates=(f"{stage_id.upper()}_PROVEN",),
            invalidation_rules=("version_or_input_hash_changed", "upstream_stage_changed"),
            repair_targets=(stage_id,),
        ))
        previous = stage_id
    return tuple(stages)


DEFAULT_STAGES = _make_stages()
STAGE_INDEX = {stage.stage_id: stage for stage in DEFAULT_STAGES}


def validate_dag(stages: Iterable[StageSpec] = DEFAULT_STAGES) -> dict[str, Any]:
    entries = tuple(stages)
    ids = {stage.stage_id for stage in entries}
    errors: list[str] = []
    for stage in entries:
        for dependency in stage.dependencies:
            if dependency not in ids:
                errors.append(f"{stage.stage_id}: missing dependency {dependency}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visiting:
            errors.append(f"cycle:{stage_id}")
            return
        if stage_id in visited:
            return
        visiting.add(stage_id)
        stage = next(item for item in entries if item.stage_id == stage_id)
        for dependency in stage.dependencies:
            if dependency in ids:
                visit(dependency)
        visiting.remove(stage_id)
        visited.add(stage_id)

    for stage in entries:
        visit(stage.stage_id)
    return {"schema_version": "pipeline_graph_v1", "classification": "PROVEN" if not errors else "REJECTED", "stage_count": len(entries), "stages": [stage.to_dict() for stage in entries], "errors": errors}


def graph_hash(stages: Iterable[StageSpec] = DEFAULT_STAGES) -> str:
    payload = json.dumps(validate_dag(stages), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def downstream_stages(stage_id: str, stages: Iterable[StageSpec] = DEFAULT_STAGES) -> list[str]:
    entries = tuple(stages)
    result: list[str] = []
    pending = [stage_id]
    while pending:
        current = pending.pop(0)
        for stage in entries:
            if current in stage.dependencies and stage.stage_id not in result:
                result.append(stage.stage_id)
                pending.append(stage.stage_id)
    return result
