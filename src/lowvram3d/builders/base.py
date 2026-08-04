"""Builder contract primitives; no scene-specific implementation is allowed here."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BuilderContract:
    builder_id: str
    semantic_classes: tuple[str, ...]
    representations: tuple[str, ...]
    input_contract: tuple[str, ...]
    output_types: tuple[str, ...]
    collision_policy: str
    navigation_policy: str
    material_policy: str
    proof_gates: tuple[str, ...]
    fallback: str
    resource_budget: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_budget(contract: BuilderContract, region_count: int = 1) -> dict[str, int]:
    count = max(1, int(region_count))
    return {key: int(value) * count for key, value in contract.resource_budget.items()}
