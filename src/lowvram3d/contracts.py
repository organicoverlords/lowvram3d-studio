from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(slots=True)
class StageReceipt:
    stage: str
    status: str
    started_at: int
    finished_at: int | None = None
    command: list[str] = field(default_factory=list)
    exit_code: int | None = None
    peak_vram_mb: int | None = None
    peak_ram_mb: int | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    failure_class: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class JobReceipt:
    job_id: str
    operation: str
    status: str = "queued"
    selected_lane: str | None = None
    requested_lanes: list[str] = field(default_factory=list)
    input_files: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    stages: list[StageReceipt] = field(default_factory=list)
    started_at: int = field(default_factory=now_ms)
    finished_at: int | None = None
    hardware: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def create(cls, operation: str, lanes: list[str], inputs: dict[str, str]) -> "JobReceipt":
        return cls(job_id=str(uuid.uuid4()), operation=operation, requested_lanes=lanes, input_files=inputs)

    @classmethod
    def load(cls, path: Path) -> "JobReceipt":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        data["stages"] = [StageReceipt(**item) for item in data.get("stages", [])]
        return cls(**data)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


REQUIRED_GAME_OUTPUTS = (
    "asset_glb",
    "preview_png",
    "validation_json",
    "job_receipt_json",
)
