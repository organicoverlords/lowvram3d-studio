"""Stable backend-neutral contract for V4 semantic view generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json


BACKENDS = {"existing_mvadapter", "external_multiview_worker", "manual_registered_view"}


@dataclass(frozen=True)
class ViewRequest:
    request_id: str
    view_name: str
    camera: dict
    camera_hash: str
    mesh_hash: str
    uv_hash: str
    provenance_hash: str
    target_semantic_class: str
    expected_useful_coverage_percent: float
    protected_regions: list[str] = field(default_factory=lambda: ["front_face"])
    forbidden_content: list[str] = field(default_factory=lambda: ["eyes", "beak", "front_face_plate", "front_mask"])
    conditioning_inputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ViewResult:
    request_id: str
    backend: str
    image: str
    mask: str
    camera_hash: str
    source_class: str
    confidence: float
    generation_seed: int | None
    model_identity: str
    model_hash: str
    settings: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    status: str = "GENERATED_VIEW_ACCEPTED"

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f"unsupported backend: {self.backend}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


def contract_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_request(path: str | Path, request: ViewRequest) -> None:
    Path(path).write_text(json.dumps(request.to_dict(), indent=2), encoding="utf-8")


def write_result(path: str | Path, result: ViewResult) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def read_request(path: str | Path) -> ViewRequest:
    return ViewRequest(**json.loads(Path(path).read_text(encoding="utf-8")))


def read_result(path: str | Path) -> ViewResult:
    return ViewResult(**json.loads(Path(path).read_text(encoding="utf-8")))
