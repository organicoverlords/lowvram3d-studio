"""Create the small GLB fixture used only for coordinate audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")


def main() -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    path = ROOT / "axis_fixture.glb"
    mesh.export(path)
    (ROOT / "axis_fixture_input.json").write_text(json.dumps({
        "schema": "axis_fixture_input_v1",
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
        "glb": str(path),
    }, indent=2) + "\n", encoding="utf-8")
    marker_scene = trimesh.Scene()
    centers = {"axis_origin": [0.0, 0.0, 0.0], "axis_plus_x": [1.0, 0.0, 0.0], "axis_plus_y": [0.0, 1.0, 0.0], "axis_plus_z": [0.0, 0.0, 1.0], "axis_xyz": [1.0, 1.0, 1.0]}
    for name, center in centers.items():
        marker = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        marker.apply_translation(center)
        marker_scene.add_geometry(marker, geom_name=name)
    marker_path = ROOT / "axis_fixture_markers.glb"
    marker_scene.export(marker_path)
    (ROOT / "axis_fixture_markers_input.json").write_text(json.dumps({"schema": "axis_fixture_markers_input_v1", "centers": centers, "glb": str(marker_path)}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
