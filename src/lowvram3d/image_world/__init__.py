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
from .terrain import (
    CompletedHeightfield,
    HeightfieldObservation,
    complete_heightfield,
    rasterize_point_map,
    slope_degrees,
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
    "build_plan",
    "classify_input",
    "complete_heightfield",
    "d8_flow_direction",
    "flow_accumulation",
    "gtx1660_super_6gb",
    "priority_flood_fill",
    "rasterize_point_map",
    "slope_degrees",
    "stream_mask",
]
