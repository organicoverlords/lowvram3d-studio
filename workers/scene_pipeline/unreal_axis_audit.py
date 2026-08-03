"""Import axis marker GLB through Unreal Interchange and measure bounds centers."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
CONTENT = "/Game/AgentProof/ImageToSceneSmoke_20260803/TransformAudit"
GLB = ROOT / "axis_fixture_markers.glb"


def write_json(name: str, payload: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def import_assets() -> list[object]:
    task = unreal.AssetImportTask()
    task.filename = str(GLB)
    task.destination_path = CONTENT
    task.automated = True
    task.save = True
    task.replace_existing = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    paths = list(task.imported_object_paths)
    assets = [unreal.EditorAssetLibrary.load_asset(path) for path in paths]
    meshes = [asset for asset in assets if isinstance(asset, unreal.StaticMesh)]
    if not meshes:
        raise RuntimeError("UNREAL_AXIS_FIXTURE_IMPORT_NO_MESHES:" + repr(paths))
    return meshes


def bounds_center(mesh: object) -> list[float]:
    bounds = mesh.get_bounds()
    origin = bounds.origin
    return [float(origin.x), float(origin.y), float(origin.z)]


def main() -> None:
    meshes = import_assets()
    expected = json.loads((ROOT / "axis_fixture_markers_input.json").read_text(encoding="utf-8"))["centers"]
    observed = {}
    for mesh in meshes:
        name = mesh.get_name().lower()
        key = next((key for key in expected if key.lower() in name), name)
        observed[key] = bounds_center(mesh)
    source = []
    target = []
    for key, center in expected.items():
        if key in observed:
            source.append(center)
            target.append(observed[key])
    origin = observed["axis_origin"]
    plus_x = observed["axis_plus_x"]
    plus_y = observed["axis_plus_y"]
    plus_z = observed["axis_plus_z"]
    matrix = [
        [plus_x[0] - origin[0], plus_y[0] - origin[0], plus_z[0] - origin[0], origin[0]],
        [plus_x[1] - origin[1], plus_y[1] - origin[1], plus_z[1] - origin[1], origin[1]],
        [plus_x[2] - origin[2], plus_y[2] - origin[2], plus_z[2] - origin[2], origin[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    residuals = []
    for key, center in expected.items():
        if key in observed:
            x, y, z = center
            predicted = [matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3], matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3], matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]]
            residuals.append(sum((predicted[i] - observed[key][i]) ** 2 for i in range(3)) ** 0.5)
    report = {
        "schema": "moge_glb_unreal_transform_audit_v1",
        "classification": "MOGE_GLB_UNREAL_TRANSFORM_PROVEN" if len(source) == len(expected) and max(residuals, default=float("inf")) < 1e-3 else "MOGE_GLB_UNREAL_TRANSFORM_REJECTED",
        "glb": str(GLB),
        "content_root": CONTENT,
        "imported_mesh_count": len(meshes),
        "expected_marker_centers": expected,
        "observed_mesh_bounds_centers": observed,
        "M_glb_to_unreal": matrix,
        "max_residual": max(residuals, default=None),
        "determinant": matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]),
        "actor_transform": "IDENTITY_NOT_APPLIED_TO_ASSET_BOUNDS",
        "scale_conversion": "measured_from_marker_bounds",
    }
    write_json("unreal_axis_transform_audit.json", report)
    unreal.SystemLibrary.quit_editor()


main()
