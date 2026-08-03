"""Validation for the reusable SceneSpec v1 contract.

The validator intentionally uses only the Python standard library so the scene
contract can be checked before Blender, Unreal, or any neural environment is
started. JSON Schema remains the interchange contract; this module adds the
cross-reference and safety checks that JSON Schema alone cannot express.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "scene_spec_v1"
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "scene_id",
    "intent",
    "source",
    "coordinate_system",
    "camera",
    "regions",
    "assets",
    "pcg",
    "budgets",
    "proof",
}


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _add_type_error(errors: list[dict[str, str]], path: str, expected: str) -> None:
    errors.append({"code": "TYPE_ERROR", "path": path, "message": f"expected {expected}"})


def _require_mapping(
    value: Any, path: str, errors: list[dict[str, str]]
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        _add_type_error(errors, path, "object")
        return None
    return value


def _require_sequence(
    value: Any, path: str, errors: list[dict[str, str]]
) -> Sequence[Any] | None:
    if not _is_sequence(value):
        _add_type_error(errors, path, "array")
        return None
    return value


def _collect_ids(
    items: Any,
    path: str,
    errors: list[dict[str, str]],
) -> tuple[set[str], list[Mapping[str, Any]]]:
    sequence = _require_sequence(items, path, errors)
    if sequence is None:
        return set(), []

    ids: set[str] = set()
    mappings: list[Mapping[str, Any]] = []
    for index, item in enumerate(sequence):
        item_path = f"{path}[{index}]"
        mapping = _require_mapping(item, item_path, errors)
        if mapping is None:
            continue
        mappings.append(mapping)
        item_id = mapping.get("id")
        if not isinstance(item_id, str) or not item_id:
            errors.append(
                {"code": "MISSING_ID", "path": f"{item_path}.id", "message": "non-empty id required"}
            )
            continue
        if item_id in ids:
            errors.append(
                {"code": "DUPLICATE_ID", "path": f"{item_path}.id", "message": item_id}
            )
        ids.add(item_id)
    return ids, mappings


def _validate_camera_view(
    view: Any,
    path: str,
    errors: list[dict[str, str]],
) -> None:
    mapping = _require_mapping(view, path, errors)
    if mapping is None:
        return

    near = mapping.get("near_m")
    far = mapping.get("far_m")
    if not isinstance(near, (int, float)) or near <= 0:
        errors.append({"code": "INVALID_CAMERA_NEAR", "path": f"{path}.near_m", "message": "must be > 0"})
    if not isinstance(far, (int, float)) or far <= 0:
        errors.append({"code": "INVALID_CAMERA_FAR", "path": f"{path}.far_m", "message": "must be > 0"})
    if isinstance(near, (int, float)) and isinstance(far, (int, float)) and far <= near:
        errors.append(
            {
                "code": "INVALID_CAMERA_RANGE",
                "path": path,
                "message": "far_m must be greater than near_m",
            }
        )

    projection = mapping.get("projection")
    if projection == "perspective":
        fov = mapping.get("field_of_view_deg")
        if not isinstance(fov, (int, float)) or not 0 < fov < 180:
            errors.append(
                {
                    "code": "INVALID_CAMERA_FOV",
                    "path": f"{path}.field_of_view_deg",
                    "message": "perspective camera requires 0 < field_of_view_deg < 180",
                }
            )
    elif projection == "orthographic":
        scale = mapping.get("ortho_scale_m")
        if not isinstance(scale, (int, float)) or scale <= 0:
            errors.append(
                {
                    "code": "INVALID_ORTHO_SCALE",
                    "path": f"{path}.ortho_scale_m",
                    "message": "orthographic camera requires ortho_scale_m > 0",
                }
            )
    else:
        errors.append(
            {
                "code": "INVALID_CAMERA_PROJECTION",
                "path": f"{path}.projection",
                "message": "must be perspective or orthographic",
            }
        )


def validate_scene_spec(spec: Any) -> dict[str, Any]:
    """Return a deterministic validation report for one decoded SceneSpec."""

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    root = _require_mapping(spec, "$", errors)
    if root is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "classification": "REJECTED",
            "scene_spec_valid": False,
            "error_count": len(errors),
            "warning_count": 0,
            "errors": errors,
            "warnings": warnings,
        }

    for key in sorted(_REQUIRED_TOP_LEVEL - root.keys()):
        errors.append({"code": "MISSING_REQUIRED_KEY", "path": f"$.{key}", "message": "required"})

    if root.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            {
                "code": "SCHEMA_VERSION_MISMATCH",
                "path": "$.schema_version",
                "message": f"expected {SCHEMA_VERSION}",
            }
        )

    scene_id = root.get("scene_id")
    if not isinstance(scene_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", scene_id):
        errors.append(
            {
                "code": "INVALID_SCENE_ID",
                "path": "$.scene_id",
                "message": "use 3-64 lowercase alphanumeric, underscore, or hyphen characters",
            }
        )

    source = _require_mapping(root.get("source"), "$.source", errors)
    if source is not None:
        sha256 = source.get("sha256")
        if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
            errors.append(
                {
                    "code": "INVALID_SOURCE_SHA256",
                    "path": "$.source.sha256",
                    "message": "expected 64 lowercase hexadecimal characters",
                }
            )

    coordinate_system = _require_mapping(root.get("coordinate_system"), "$.coordinate_system", errors)
    if coordinate_system is not None:
        expected = {
            "units": "meters",
            "up_axis": "Z",
            "handedness": "right",
            "unreal_scale_cm_per_unit": 100,
        }
        for key, expected_value in expected.items():
            if coordinate_system.get(key) != expected_value:
                errors.append(
                    {
                        "code": "UNSUPPORTED_COORDINATE_SYSTEM",
                        "path": f"$.coordinate_system.{key}",
                        "message": f"expected {expected_value!r}",
                    }
                )

    intent = _require_mapping(root.get("intent"), "$.intent", errors)
    if intent is not None and intent.get("target_engine") != "unreal_5_8":
        errors.append(
            {
                "code": "UNSUPPORTED_TARGET_ENGINE",
                "path": "$.intent.target_engine",
                "message": "expected unreal_5_8",
            }
        )

    camera = _require_mapping(root.get("camera"), "$.camera", errors)
    camera_view_ids: set[str] = set()
    if camera is not None:
        _validate_camera_view(camera.get("source_camera"), "$.camera.source_camera", errors)
        required_views = _require_sequence(camera.get("required_views"), "$.camera.required_views", errors)
        if required_views is not None:
            if not required_views:
                errors.append(
                    {
                        "code": "EMPTY_REQUIRED_VIEWS",
                        "path": "$.camera.required_views",
                        "message": "at least one required view is needed",
                    }
                )
            for index, view in enumerate(required_views):
                path = f"$.camera.required_views[{index}]"
                _validate_camera_view(view, path, errors)
                if isinstance(view, Mapping):
                    view_id = view.get("id")
                    if not isinstance(view_id, str) or not view_id:
                        errors.append(
                            {"code": "MISSING_ID", "path": f"{path}.id", "message": "non-empty id required"}
                        )
                    elif view_id in camera_view_ids:
                        errors.append(
                            {"code": "DUPLICATE_ID", "path": f"{path}.id", "message": view_id}
                        )
                    else:
                        camera_view_ids.add(view_id)

    depth_band_ids, depth_bands = _collect_ids(root.get("depth_bands", []), "$.depth_bands", errors)
    for index, band in enumerate(depth_bands):
        near = band.get("near_m")
        far = band.get("far_m")
        if not isinstance(near, (int, float)) or not isinstance(far, (int, float)) or far <= near:
            errors.append(
                {
                    "code": "INVALID_DEPTH_BAND",
                    "path": f"$.depth_bands[{index}]",
                    "message": "far_m must be greater than near_m",
                }
            )

    region_ids, regions = _collect_ids(root.get("regions"), "$.regions", errors)
    for index, region in enumerate(regions):
        depth_band = region.get("depth_band")
        if depth_band is not None and depth_band not in depth_band_ids:
            errors.append(
                {
                    "code": "MISSING_REFERENCE",
                    "path": f"$.regions[{index}].depth_band",
                    "message": str(depth_band),
                }
            )

    asset_ids, assets = _collect_ids(root.get("assets"), "$.assets", errors)
    for index, asset in enumerate(assets):
        region_id = asset.get("region_id")
        if region_id is not None and region_id not in region_ids:
            errors.append(
                {
                    "code": "MISSING_REFERENCE",
                    "path": f"$.assets[{index}].region_id",
                    "message": str(region_id),
                }
            )
        if asset.get("representation") == "gameplay_proxy" and asset.get("collision") == "none":
            errors.append(
                {
                    "code": "UNSAFE_GAMEPLAY_PROXY",
                    "path": f"$.assets[{index}].collision",
                    "message": "gameplay_proxy requires collision",
                }
            )

    population_ids, populations = _collect_ids(root.get("populations", []), "$.populations", errors)
    for index, population in enumerate(populations):
        region_id = population.get("region_id")
        if region_id not in region_ids:
            errors.append(
                {
                    "code": "MISSING_REFERENCE",
                    "path": f"$.populations[{index}].region_id",
                    "message": str(region_id),
                }
            )
        asset_refs = _require_sequence(
            population.get("asset_refs"), f"$.populations[{index}].asset_refs", errors
        )
        if asset_refs is not None:
            for ref_index, asset_ref in enumerate(asset_refs):
                if asset_ref not in asset_ids:
                    errors.append(
                        {
                            "code": "MISSING_REFERENCE",
                            "path": f"$.populations[{index}].asset_refs[{ref_index}]",
                            "message": str(asset_ref),
                        }
                    )
        if population.get("execution") == "gpu_pcg" and (
            population.get("requires_collision") is True
            or population.get("requires_navigation") is True
        ):
            errors.append(
                {
                    "code": "UNSAFE_GPU_PCG",
                    "path": f"$.populations[{index}]",
                    "message": "GPU PCG populations cannot require collision or navigation",
                }
            )

    spline_ids, _ = _collect_ids(root.get("splines", []), "$.splines", errors)
    pcg = _require_mapping(root.get("pcg"), "$.pcg", errors)
    pcg_layer_ids: set[str] = set()
    if pcg is not None:
        if pcg.get("policy") != "reference_driven_incremental":
            errors.append(
                {
                    "code": "UNSUPPORTED_PCG_POLICY",
                    "path": "$.pcg.policy",
                    "message": "expected reference_driven_incremental",
                }
            )
        pcg_layer_ids, layers = _collect_ids(pcg.get("layers"), "$.pcg.layers", errors)
        known_inputs = region_ids | asset_ids | population_ids | spline_ids
        gameplay_tokens = {"gameplay", "collision", "navigation", "bridge"}
        for index, layer in enumerate(layers):
            ref = layer.get("reference_graph")
            if not isinstance(ref, str) or not ref:
                errors.append(
                    {
                        "code": "MISSING_REFERENCE_GRAPH",
                        "path": f"$.pcg.layers[{index}].reference_graph",
                        "message": "reference-driven PCG layer requires a reference graph",
                    }
                )
            inputs = _require_sequence(layer.get("inputs"), f"$.pcg.layers[{index}].inputs", errors)
            if inputs is not None:
                for input_index, item in enumerate(inputs):
                    if item not in known_inputs and item not in {"buildable_mask"}:
                        warnings.append(
                            {
                                "code": "UNRESOLVED_DERIVED_INPUT",
                                "path": f"$.pcg.layers[{index}].inputs[{input_index}]",
                                "message": str(item),
                            }
                        )
            if layer.get("execution") == "gpu":
                if layer.get("pattern") != "small_scatter":
                    errors.append(
                        {
                            "code": "UNSAFE_GPU_PCG_LAYER",
                            "path": f"$.pcg.layers[{index}].pattern",
                            "message": "SceneSpec v1 limits GPU execution to small_scatter",
                        }
                    )
                outputs = _require_sequence(layer.get("outputs"), f"$.pcg.layers[{index}].outputs", errors)
                if outputs is not None:
                    for output_index, output in enumerate(outputs):
                        normalized = str(output).lower()
                        if any(token in normalized for token in gameplay_tokens):
                            errors.append(
                                {
                                    "code": "GPU_GAMEPLAY_OUTPUT",
                                    "path": f"$.pcg.layers[{index}].outputs[{output_index}]",
                                    "message": str(output),
                                }
                            )

    budgets = _require_mapping(root.get("budgets"), "$.budgets", errors)
    if budgets is not None:
        if budgets.get("allow_concurrent_neural_and_unreal_gpu") is not False:
            errors.append(
                {
                    "code": "UNSAFE_GPU_CONCURRENCY",
                    "path": "$.budgets.allow_concurrent_neural_and_unreal_gpu",
                    "message": "must be false on the target 6 GB GPU",
                }
            )
        gpu_vram_mb = budgets.get("gpu_vram_mb")
        if not isinstance(gpu_vram_mb, int) or not 256 <= gpu_vram_mb <= 6144:
            errors.append(
                {
                    "code": "INVALID_GPU_BUDGET",
                    "path": "$.budgets.gpu_vram_mb",
                    "message": "must be an integer between 256 and 6144",
                }
            )

    proof = _require_mapping(root.get("proof"), "$.proof", errors)
    if proof is not None:
        gates = _require_sequence(proof.get("required_gates"), "$.proof.required_gates", errors)
        if gates is not None and not gates:
            errors.append(
                {
                    "code": "EMPTY_PROOF_GATES",
                    "path": "$.proof.required_gates",
                    "message": "at least one proof gate is required",
                }
            )

    classification = "PROVEN" if not errors else "REJECTED"
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "scene_spec_valid": not errors,
        "scene_id": scene_id if isinstance(scene_id, str) else None,
        "counts": {
            "camera_views": len(camera_view_ids),
            "depth_bands": len(depth_band_ids),
            "regions": len(region_ids),
            "assets": len(asset_ids),
            "populations": len(population_ids),
            "splines": len(spline_ids),
            "pcg_layers": len(pcg_layer_ids),
        },
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def validate_scene_spec_file(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "classification": "REJECTED",
            "scene_spec_valid": False,
            "scene_id": None,
            "counts": {},
            "error_count": 1,
            "warning_count": 0,
            "errors": [
                {
                    "code": "SCENE_SPEC_READ_FAILED",
                    "path": str(spec_path),
                    "message": str(exc),
                }
            ],
            "warnings": [],
        }
    report = validate_scene_spec(spec)
    report["scene_spec_path"] = str(spec_path.resolve())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LowVRAM3D SceneSpec v1")
    parser.add_argument("--scene-spec", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    report = validate_scene_spec_file(args.scene_spec)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    status = "PROVEN" if report["scene_spec_valid"] else "REJECTED"
    print(f"SCENE_SPEC_VALID={status}")
    print(f"ERROR_COUNT={report['error_count']}")
    print(f"WARNING_COUNT={report['warning_count']}")
    return 0 if report["scene_spec_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
