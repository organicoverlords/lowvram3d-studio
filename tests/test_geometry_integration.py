from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_component_cleanup_welds_before_classification():
    source = (ROOT / "blender" / "raster_cleanup_extract.py").read_text(encoding="utf-8")
    assert source.index("bmesh.ops.remove_doubles") < source.index("components = component_faces")
    assert "topology_gate(" in source
    assert "fill_small_holes(" in source
    assert "REMOVE_DETACHED_SINGLE_SUBJECT" not in source  # decision belongs to pure policy module


def test_postprocess_projection_uses_raster_gate():
    appearance = (ROOT / "src" / "lowvram3d" / "appearance.py").read_text(encoding="utf-8")
    pipeline = (ROOT / "src" / "lowvram3d" / "pipeline.py").read_text(encoding="utf-8")
    route = (ROOT / "src" / "lowvram3d" / "raster_route.py").read_text(encoding="utf-8")
    assert "return engine._texture_projection_raster" in appearance
    assert "return run_raster_texture_route" in pipeline
    assert '"geometry_repair_extract_v2"' in route
    assert '"geometry_quality_validate_v2"' in route
    assert "candidate_v2" in route
    assert "os.replace(temporary, destination)" in route


def test_final_package_export_is_atomic_and_topology_gated():
    package = (ROOT / "blender" / "package_validate.py").read_text(encoding="utf-8")
    common = (ROOT / "blender" / "common.py").read_text(encoding="utf-8")
    assert ".game_ready.candidate.glb" in package
    assert "os.replace(candidate_glb, final_glb)" in package
    assert 'validation["candidate_promoted"]' not in package
    assert '"candidate_promoted": validation["success"]' in package
    assert "welded_topology_stats" in package
    assert "def welded_topology_stats" in common
