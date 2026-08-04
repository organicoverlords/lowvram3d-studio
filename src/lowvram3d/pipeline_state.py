"""Hash-addressed state and targeted invalidation for pipeline stages."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from .pipeline_graph import DEFAULT_STAGES, StageSpec, downstream_stages, graph_hash


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def stage_input_hash(stage: StageSpec, inputs: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> str:
    return canonical_hash({"stage_id": stage.stage_id, "stage_version": stage.version, "inputs": inputs, "config": config or {}})


def new_state(scene_id: str, source_sha256: str, output_map: str, run_id: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "pipeline_state_v2",
        "scene_id": scene_id,
        "run_id": run_id,
        "source_sha256": source_sha256,
        "output_map": output_map,
        "graph_hash": graph_hash(),
        "config_hash": canonical_hash(config or {}),
        "started_at": now,
        "updated_at": now,
        "stages": {},
        "classification": "PARTIAL",
    }


def reusable_stage(state: Mapping[str, Any], stage: StageSpec, input_hash: str, config_hash: str) -> bool:
    record = state.get("stages", {}).get(stage.stage_id, {})
    return bool(record.get("classification") == "PROVEN" and record.get("stage_version") == stage.version and record.get("input_hash") == input_hash and record.get("config_hash") == config_hash)


def record_stage(state: dict[str, Any], stage: StageSpec, input_hash: str, config_hash: str, result: Mapping[str, Any], output: str | None = None) -> None:
    state.setdefault("stages", {})[stage.stage_id] = {
        "classification": result.get("classification", "NOT_PROVEN"),
        "stage_version": stage.version,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "output": output,
        "proof_gates": result.get("proof_gates", stage.proof_gates),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state["updated_at"] = datetime.now(timezone.utc).isoformat()


def invalidate_from(state: dict[str, Any], stage_id: str, stages: tuple[StageSpec, ...] = DEFAULT_STAGES) -> list[str]:
    invalidated = [stage_id, *downstream_stages(stage_id, stages)]
    for target in invalidated:
        if target in state.get("stages", {}):
            state["stages"][target]["classification"] = "INVALIDATED"
            state["stages"][target]["invalidated_by"] = stage_id
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    return invalidated


def repair_routes() -> dict[str, str]:
    return {
        "source_silhouette_mismatch": "camera_estimation|representation_selection|terrain_generation|architecture_generation",
        "floating_architecture": "scene_graph_construction|architecture_generation",
        "grey_default_material": "material_generation|material_harmonization",
        "vegetation_in_water": "vegetation_planning|vegetation_generation",
        "navmesh_crossing_water": "gameplay_planning|navigation_generation",
        "offset_view_hole": "unseen_world_completion|world_continuation",
        "incorrect_object_classification": "semantic_segmentation|instance_decomposition",
    }
