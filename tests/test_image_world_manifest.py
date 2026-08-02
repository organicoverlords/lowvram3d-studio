from pathlib import Path
import json

from workers.image_world.build_lighthouse_world_manifest import main


def test_manifest_declares_missing_geometry_without_faking_completion(tmp_path, monkeypatch):
    geometry = tmp_path / "geometry"
    surface = tmp_path / "surface"
    output = tmp_path / "output"
    geometry.mkdir()
    surface.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_lighthouse_world_manifest.py",
            "--geometry",
            str(geometry),
            "--surface",
            str(surface),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    data = json.loads((output / "lighthouse-world-manifest.json").read_text())
    assert data["status"] == "READY_FOR_DCC_RECONSTRUCTION"
    assert data["classification"] == "IMAGE_WORLD_OBSERVATION_NOT_FINAL_WORLD"
    assert all(
        item["exists"] is False
        for item in data["source_contract"]["geometry_files"].values()
    )
    assert "semantic terrain segmentation" in data["not_proven"]
