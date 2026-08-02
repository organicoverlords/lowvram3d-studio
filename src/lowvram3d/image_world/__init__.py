"""Image-to-world routing, contracts, and resumable stage planning."""

from .classifier import InputSignals, classify_input
from .contracts import (
    BackendReceipt,
    CameraObservation,
    ContractError,
    ImageWorldRoute,
    ObservationPackage,
    RouteDecision,
    SourceImageIdentity,
)
from .hydrology import d8_flow_direction, flow_accumulation, priority_flood_fill, stream_mask
from .profiles import QUALITY_PROFILES, HardwareProfile, QualityProfile, gtx1660_super_6gb
from .stage_graph import PipelinePlan, StageSpec, build_plan
from .surface_projection import SurfaceProjectionResult, project_moge_surface, robust_xy_bounds
from .terrain import (
    CompletedHeightfield,
    HeightfieldObservation,
    complete_heightfield,
    rasterize_point_map,
    slope_degrees,
)
from .world_frame import (
    WorldFrameEstimate,
    camera_to_world_rotation,
    estimate_world_up_from_normals,
    transform_camera_vectors,
)

__all__ = [
    "BackendReceipt",
    "CameraObservation",
    "CompletedHeightfield",
    "ContractError",
    "HardwareProfile",
    "HeightfieldObservation",
    "ImageWorldRoute",
    "InputSignals",
    "ObservationPackage",
    "PipelinePlan",
    "QUALITY_PROFILES",
    "QualityProfile",
    "RouteDecision",
    "SourceImageIdentity",
    "StageSpec",
    "SurfaceProjectionResult",
    "WorldFrameEstimate",
    "build_plan",
    "camera_to_world_rotation",
    "classify_input",
    "complete_heightfield",
    "d8_flow_direction",
    "estimate_world_up_from_normals",
    "flow_accumulation",
    "gtx1660_super_6gb",
    "priority_flood_fill",
    "project_moge_surface",
    "rasterize_point_map",
    "robust_xy_bounds",
    "slope_degrees",
    "stream_mask",
    "transform_camera_vectors",
]
