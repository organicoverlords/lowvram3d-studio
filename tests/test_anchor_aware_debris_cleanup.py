from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from lowvram3d.thin_feature_anchors import discover_thin_feature_anchors, serialize_anchor_receipt
from thin_feature_anchor_fixtures import detached_singleton
from workers.mesh_io import vertex_normals, write_glb


SOURCE_HASH = "7" * 64
ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(root: Path) -> tuple[Path, Path]:
    mesh = detached_singleton()
    positions = mesh.vertices.astype("float32")
    triangles = mesh.faces.astype("int64")
    source = root / "lod0.glb"
    write_glb(source, positions, vertex_normals(positions, triangles), None, triangles)
    receipt = root / "anchor_receipt.json"
    receipt.write_bytes(serialize_anchor_receipt(
        discover_thin_feature_anchors(mesh, source_mesh_sha256=SOURCE_HASH)
    ))
    return source, receipt


def test_registered_singleton_is_a_hard_cleanup_failure(tmp_path: Path) -> None:
    source, receipt = _write_fixture(tmp_path)
    output = tmp_path / "clean.glb"
    report = tmp_path / "cleanup.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "workers" / "pipeline_debris_strip.py"),
            "--input", str(source),
            "--output", str(output),
            "--report", str(report),
            "--anchor-receipt", str(receipt),
            "--source-hash", SOURCE_HASH,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not output.exists()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert "ANCHOR_INTERSECTION" in data["detail"]
    assert data["anchor_gate"]["before_sha256"]
    assert data["anchor_gate"]["after_sha256"]
