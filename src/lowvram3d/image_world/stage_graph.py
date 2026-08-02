"""Route-aware stage graph for resumable image-to-world jobs."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ContractError, ImageWorldRoute


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    dependencies: tuple[str, ...] = ()
    heavy_gpu: bool = False
    optional: bool = False
    produces: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelinePlan:
    route: ImageWorldRoute
    stages: tuple[StageSpec, ...]

    def validate(self) -> None:
        ids = [stage.stage_id for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise ContractError("pipeline stage ids must be unique")
        known: set[str] = set()
        for stage in self.stages:
            missing = set(stage.dependencies) - known
            if missing:
                raise ContractError(
                    f"stage {stage.stage_id} appears before dependencies: {sorted(missing)}"
                )
            known.add(stage.stage_id)

    def stage(self, stage_id: str) -> StageSpec:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)

    @property
    def heavy_gpu_stages(self) -> tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.stages if stage.heavy_gpu)


def build_plan(route: ImageWorldRoute, *, include_experimental: bool = False) -> PipelinePlan:
    common = [
        StageSpec("source_ingest", produces=("source/original.png", "reports/source.json")),
        StageSpec(
            "route_classification",
            dependencies=("source_ingest",),
            produces=("reports/route.json",),
        ),
        StageSpec(
            "semantic_observations",
            dependencies=("route_classification",),
            heavy_gpu=True,
            produces=("masks/", "reports/segmentation.json"),
        ),
        StageSpec(
            "geometry_observations",
            dependencies=("semantic_observations",),
            heavy_gpu=True,
            produces=("geometry/", "reports/geometry-backend.json"),
        ),
    ]

    if route is ImageWorldRoute.ISOLATED_ASSET:
        stages = common + [
            StageSpec(
                "isolated_asset_analysis",
                dependencies=("geometry_observations",),
                produces=("reports/asset-analysis.json",),
            ),
            StageSpec(
                "proxy_geometry",
                dependencies=("isolated_asset_analysis",),
                heavy_gpu=True,
                optional=True,
                produces=("meshes/proxy.glb",),
            ),
            StageSpec(
                "procedural_asset_build",
                dependencies=("isolated_asset_analysis",),
                produces=("meshes/high.glb", "reports/generator.json"),
            ),
            StageSpec(
                "blender_asset_preview",
                dependencies=("procedural_asset_build",),
                produces=("previews/",),
            ),
            StageSpec(
                "asset_validation",
                dependencies=("blender_asset_preview",),
                produces=("reports/validation-report.json",),
            ),
            StageSpec(
                "asset_package",
                dependencies=("asset_validation",),
                produces=("meshes/game_ready.glb", "world_manifest.json"),
            ),
        ]
    elif route is ImageWorldRoute.DIORAMA_MAP:
        stages = common + _terrain_tail(
            visible_stage=StageSpec(
                "diorama_rectification",
                dependencies=("geometry_observations",),
                produces=("maps/rectified-source.png", "reports/rectification.json"),
            ),
            reconstruction_stage=StageSpec(
                "diorama_semantic_reconstruction",
                dependencies=("diorama_rectification",),
                produces=("maps/observed-height.exr", "placement/"),
            ),
            include_experimental=include_experimental,
        )
    elif route is ImageWorldRoute.PERSPECTIVE_VISTA:
        stages = common + [
            StageSpec(
                "camera_recovery",
                dependencies=("geometry_observations",),
                produces=("camera.json",),
            ),
            StageSpec(
                "landmark_extraction",
                dependencies=("camera_recovery",),
                produces=("landmarks.json",),
            ),
        ] + _terrain_tail(
            visible_stage=StageSpec(
                "visible_terrain_reprojection",
                dependencies=("landmark_extraction",),
                produces=("maps/observed-height.exr", "maps/visibility-confidence.exr"),
            ),
            reconstruction_stage=None,
            include_experimental=include_experimental,
        )
    else:
        stages = common + [
            StageSpec(
                "composite_scene_decomposition",
                dependencies=("geometry_observations",),
                produces=("decomposition/manifest.json",),
            ),
            StageSpec(
                "composite_review_package",
                dependencies=("composite_scene_decomposition",),
                produces=("previews/decomposition.png", "reports/validation-report.json"),
            ),
        ]

    plan = PipelinePlan(route=route, stages=tuple(stages))
    plan.validate()
    return plan


def _terrain_tail(
    *,
    visible_stage: StageSpec,
    reconstruction_stage: StageSpec | None,
    include_experimental: bool,
) -> list[StageSpec]:
    stages = [visible_stage]
    previous = visible_stage.stage_id
    if reconstruction_stage is not None:
        stages.append(reconstruction_stage)
        previous = reconstruction_stage.stage_id
    stages.extend(
        [
            StageSpec(
                "heightfield_completion",
                dependencies=(previous,),
                produces=("maps/completed-height.exr", "maps/completion-mask.png"),
            ),
            StageSpec(
                "hydrology_and_erosion",
                dependencies=("heightfield_completion",),
                produces=("maps/conditioned-height.exr", "splines/rivers.json"),
            ),
            StageSpec(
                "terrain_material_masks",
                dependencies=("hydrology_and_erosion",),
                produces=("weightmaps/",),
            ),
            StageSpec(
                "residual_mesh_extraction",
                dependencies=("terrain_material_masks",),
                produces=("residuals/",),
            ),
        ]
    )
    if include_experimental:
        stages.append(
            StageSpec(
                "experimental_terrain_completion",
                dependencies=("heightfield_completion",),
                heavy_gpu=True,
                optional=True,
                produces=("experimental/terrain-completion.exr",),
            )
        )
    stages.extend(
        [
            StageSpec(
                "blender_world_preview",
                dependencies=("residual_mesh_extraction",),
                produces=("preview.blend", "previews/"),
            ),
            StageSpec(
                "world_validation",
                dependencies=("blender_world_preview",),
                produces=("reports/validation-report.json",),
            ),
            StageSpec(
                "unreal_world_package",
                dependencies=("world_validation",),
                produces=("heightmap_16bit.png", "unreal_import.py", "world_manifest.json"),
            ),
        ]
    )
    return stages
