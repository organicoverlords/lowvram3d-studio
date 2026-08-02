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
from .profiles import QUALITY_PROFILES, HardwareProfile, QualityProfile, gtx1660_super_6gb
from .stage_graph import PipelinePlan, StageSpec, build_plan

__all__ = [
    "BackendReceipt",
    "CameraObservation",
    "ContractError",
    "HardwareProfile",
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
    "gtx1660_super_6gb",
]
