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
from .moge_probe import (
    DEFAULT_MOGE_MODEL,
    MogeOutputSummary,
    MogeProbeReport,
    MogeProbeSettings,
    save_moge_maps,
    validate_moge_output,
)
from .profiles import QUALITY_PROFILES, HardwareProfile, QualityProfile, gtx1660_super_6gb
from .stage_graph import PipelinePlan, StageSpec, build_plan

__all__ = [
    "BackendReceipt",
    "CameraObservation",
    "ContractError",
    "DEFAULT_MOGE_MODEL",
    "HardwareProfile",
    "ImageWorldRoute",
    "InputSignals",
    "MogeOutputSummary",
    "MogeProbeReport",
    "MogeProbeSettings",
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
    "save_moge_maps",
    "validate_moge_output",
]
