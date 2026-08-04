"""Write honest CPU-side final receipts after live proof collection."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EVIDENCE = ROOT / "evidence" / "latest-image-to-scene"
GEN_EVIDENCE = ROOT / "evidence" / "generalization"


def read(name: str) -> dict:
    path = IMAGE_EVIDENCE / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def classification(name: str) -> str:
    if name.startswith("../"):
        path = IMAGE_EVIDENCE / name
        return str(json.loads(path.read_text(encoding="utf-8")).get("classification", "NOT_PROVEN")) if path.is_file() else "NOT_PROVEN"
    return str(read(name).get("classification", "NOT_PROVEN"))


visual = "BLOCKED_UNREAL_EDITOR_SLATE_ASSERT" if (IMAGE_EVIDENCE / "editor_crash_receipt.json").is_file() else "NOT_PROVEN"
existing_state = read("pipeline_state.json")
gates = {
    "scene_spec_and_builder_plan": classification("scene_completeness_receipt.json"),
    "complete_scene_layers": classification("layer_validation_receipt.json"),
    "navigation": classification("../latest-scene-navigation/navigation_validation_receipt.json"),
    "gameplay": classification("gameplay_validation_receipt.json"),
    "fresh_map_reload": classification("fresh_reload_receipt.json"),
    "live_contract": classification("../latest-scene-live-review/live_review_receipt.json"),
    "visual_proof": visual,
}
castlegrounds = "IMAGE_TO_SCENE_SMOKE_PROVEN" if all(value == "PROVEN" for value in gates.values()) else "IMAGE_TO_SCENE_SMOKE_PARTIAL"
generic = "GENERIC_ONE_IMAGE_TO_SCENE_PIPELINE_PARTIAL"

report = [
    "# Image-to-Scene Evidence Summary",
    "",
    "## Castlegrounds fixture",
    "",
    f"- Classification: `{castlegrounds}`",
    "- The source image was used only as the first integration fixture.",
    "- The source map remained protected; independent gameplay layers, water exclusions, bridge traversal, navigation bounds, and fresh map reload are separately receipted.",
    f"- Visual proof: `{visual}`.",
    "- The live editor terminated during the last read-only PIE audit. No automatic retry or editor restart was performed.",
    "",
    "### Gates",
    "",
]
for name, value in gates.items():
    report.append(f"- `{name}`: `{value}`")
report += [
    "",
    "## Generic pipeline",
    "",
    f"- Classification: `{generic}`",
    "- Selection is driven by SceneSpec layer semantics and a capability registry.",
    "- Scene-local map/evidence paths and resource budgets are enforced.",
    "- `treesandbarn` and `landscape` completed CPU bootstrap/resume dry runs with camera/depth still `REQUIRES_ANALYSIS`; they are not end-to-end Unreal scene proofs.",
    "- `baatti.jpg` and `panda.jpg` are explicitly recorded as `input_kind=object` and are excluded from scene-generalization evidence.",
    "- A second real end-to-end materially different scene has not been proven.",
    "",
    "## Protected source",
    "",
    "- Output map: `/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1`",
    "- Source map SHA-256: `39547be52ab21f3f6b0d99c0f2a2f93103a5c0ebf9da56435e37feae04cc15f9`",
    "- No GPU work was requested by the generic CPU lane.",
]
(IMAGE_EVIDENCE / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
(IMAGE_EVIDENCE / "pipeline_state.json").write_text(json.dumps({
    "schema_version": "image_to_scene_pipeline_state_v2",
    "scene_id": "castlegrounds",
    "classification": castlegrounds,
    "next_action": "USER_REVIEW_REQUIRED_UNREAL_EDITOR_RESTART" if visual.startswith("BLOCKED") else "USER_REVIEW_REQUIRED",
    "source_map_protected": True,
    "gpu_work_requested": False,
    "graph_hash": existing_state.get("graph_hash"),
    "input_hashes": existing_state.get("input_hashes", {}),
    "stages": existing_state.get("stages", {}),
    "repair_routes": existing_state.get("repair_routes", {}),
    "gates": gates,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(GEN_EVIDENCE / "SUMMARY.md").write_text("# Generalization Evidence\n\n`GENERIC_ONE_IMAGE_TO_SCENE_PIPELINE_PARTIAL`\n\nObject images `baatti.jpg` and `panda.jpg` are excluded from scene evidence. `treesandbarn` and `landscape` are CPU bootstrap/resume dry runs only.\n", encoding="utf-8")
print("CASTGROUND_FINAL=" + castlegrounds)
print("GENERIC_FINAL=" + generic)
