"""Stable contracts for image-to-world observations and route decisions.

The ML backends only produce observations. Deterministic downstream stages
consume these contracts and must preserve the distinction between observed,
adjusted, and generated data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when an image-to-world contract is invalid."""


class ImageWorldRoute(str, Enum):
    ISOLATED_ASSET = "isolated_asset"
    DIORAMA_MAP = "diorama_map"
    PERSPECTIVE_VISTA = "perspective_vista"
    COMPOSITE_SCENE = "composite_scene"


@dataclass(frozen=True)
class SourceImageIdentity:
    filename: str
    sha256: str
    width: int
    height: int
    has_alpha: bool

    def validate(self) -> None:
        if not self.filename or PurePosixPath(self.filename).name != self.filename:
            raise ContractError("source filename must be a basename")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ContractError("source sha256 must be 64 lowercase hexadecimal characters")
        if self.width <= 0 or self.height <= 0:
            raise ContractError("source dimensions must be positive")


@dataclass(frozen=True)
class RouteDecision:
    selected: ImageWorldRoute
    confidence: float
    alternatives: Mapping[ImageWorldRoute, float]
    manual_review_required: bool = False
    reasons: tuple[str, ...] = ()

    def validate(self) -> None:
        _validate_probability(self.confidence, "route confidence")
        if not self.alternatives:
            raise ContractError("route alternatives cannot be empty")
        for route, score in self.alternatives.items():
            if not isinstance(route, ImageWorldRoute):
                raise ContractError("route alternative keys must be ImageWorldRoute values")
            _validate_probability(score, f"route score for {route.value}")
        if self.selected not in self.alternatives:
            raise ContractError("selected route must be present in alternatives")
        selected_score = self.alternatives[self.selected]
        if abs(selected_score - self.confidence) > 1e-9:
            raise ContractError("route confidence must equal the selected alternative score")
        if any(score > selected_score + 1e-12 for score in self.alternatives.values()):
            raise ContractError("selected route must have the highest alternative score")


@dataclass(frozen=True)
class CameraObservation:
    intrinsics: tuple[tuple[float, float, float], ...]
    world_up: tuple[float, float, float]
    estimated_fov_degrees: float
    confidence: float

    def validate(self) -> None:
        if len(self.intrinsics) != 3 or any(len(row) != 3 for row in self.intrinsics):
            raise ContractError("camera intrinsics must be a 3x3 matrix")
        if len(self.world_up) != 3:
            raise ContractError("camera world_up must have three components")
        if not 1.0 <= self.estimated_fov_degrees <= 179.0:
            raise ContractError("estimated FOV must be between 1 and 179 degrees")
        _validate_probability(self.confidence, "camera confidence")


@dataclass(frozen=True)
class BackendReceipt:
    backend: str
    version: str
    model: str | None = None
    model_sha256: str | None = None
    precision: str | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.backend.strip() or not self.version.strip():
            raise ContractError("backend and version are required")
        if self.model_sha256 is not None and not _SHA256_RE.fullmatch(self.model_sha256):
            raise ContractError("model sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True)
class ObservationPackage:
    source: SourceImageIdentity
    route: RouteDecision
    geometry: Mapping[str, str] = field(default_factory=dict)
    masks: Mapping[str, str] = field(default_factory=dict)
    camera: CameraObservation | None = None
    world: Mapping[str, float | int] = field(default_factory=dict)
    provenance: tuple[BackendReceipt, ...] = ()
    observed_fraction: float = 0.0
    completed_fraction: float = 0.0
    seed: int = 0
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError(f"unsupported observation schema version: {self.schema_version}")
        self.source.validate()
        self.route.validate()
        if self.camera is not None:
            self.camera.validate()
        for collection_name, collection in (("geometry", self.geometry), ("masks", self.masks)):
            for key, path in collection.items():
                if not key.strip():
                    raise ContractError(f"{collection_name} artifact key cannot be empty")
                _validate_relative_artifact_path(path, f"{collection_name}.{key}")
        for receipt in self.provenance:
            receipt.validate()
        _validate_probability(self.observed_fraction, "observed fraction")
        _validate_probability(self.completed_fraction, "completed fraction")
        if self.observed_fraction + self.completed_fraction > 1.0 + 1e-9:
            raise ContractError("observed and completed fractions cannot sum above one")
        if self.seed < 0:
            raise ContractError("seed must be non-negative")
        for key, value in self.world.items():
            if not key.strip():
                raise ContractError("world setting key cannot be empty")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(f"world setting {key!r} must be numeric")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _jsonable(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservationPackage":
        source_data = _required_mapping(data, "source")
        route_data = _required_mapping(data, "route")
        alternatives_data = _required_mapping(route_data, "alternatives")
        package = cls(
            schema_version=int(data.get("schema_version", 1)),
            source=SourceImageIdentity(
                filename=str(source_data["filename"]),
                sha256=str(source_data["sha256"]),
                width=int(source_data["width"]),
                height=int(source_data["height"]),
                has_alpha=bool(source_data["has_alpha"]),
            ),
            route=RouteDecision(
                selected=ImageWorldRoute(route_data["selected"]),
                confidence=float(route_data["confidence"]),
                alternatives={
                    ImageWorldRoute(route): float(score)
                    for route, score in alternatives_data.items()
                },
                manual_review_required=bool(route_data.get("manual_review_required", False)),
                reasons=tuple(str(reason) for reason in route_data.get("reasons", ())),
            ),
            geometry={str(key): str(value) for key, value in data.get("geometry", {}).items()},
            masks={str(key): str(value) for key, value in data.get("masks", {}).items()},
            camera=_camera_from_dict(data.get("camera")),
            world={str(key): value for key, value in data.get("world", {}).items()},
            provenance=tuple(_receipt_from_dict(item) for item in data.get("provenance", ())),
            observed_fraction=float(data.get("observed_fraction", 0.0)),
            completed_fraction=float(data.get("completed_fraction", 0.0)),
            seed=int(data.get("seed", 0)),
        )
        package.validate()
        return package

    @classmethod
    def from_json(cls, payload: str) -> "ObservationPackage":
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ContractError("observation JSON root must be an object")
        return cls.from_dict(data)


def _camera_from_dict(data: Any) -> CameraObservation | None:
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise ContractError("camera must be an object")
    return CameraObservation(
        intrinsics=tuple(tuple(float(value) for value in row) for row in data["intrinsics"]),
        world_up=tuple(float(value) for value in data["world_up"]),
        estimated_fov_degrees=float(data["estimated_fov_degrees"]),
        confidence=float(data["confidence"]),
    )


def _receipt_from_dict(data: Any) -> BackendReceipt:
    if not isinstance(data, Mapping):
        raise ContractError("provenance entries must be objects")
    return BackendReceipt(
        backend=str(data["backend"]),
        version=str(data["version"]),
        model=None if data.get("model") is None else str(data["model"]),
        model_sha256=None if data.get("model_sha256") is None else str(data["model_sha256"]),
        precision=None if data.get("precision") is None else str(data["precision"]),
        settings=dict(data.get("settings", {})),
    )


def _required_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ContractError(f"{key} must be an object")
    return value


def _validate_probability(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ContractError(f"{label} must be between zero and one")


def _validate_relative_artifact_path(path: str, label: str) -> None:
    if not path:
        raise ContractError(f"{label} path cannot be empty")
    pure = PurePosixPath(path.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ContractError(f"{label} must be a package-relative path")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            (_jsonable(key) if isinstance(key, Enum) else str(key)): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
