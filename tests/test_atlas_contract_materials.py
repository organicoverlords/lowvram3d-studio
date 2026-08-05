"""Synthetic guards for atlas material and triangle-coverage contracts."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from fast_texture_projection import (  # noqa: E402
    _read_glb,
    bind_texture,
    clean_pbr_material,
    triangle_coverage_mask,
)


def test_clean_pbr_material_does_not_inherit_non_basecolor_slots() -> None:
    material = clean_pbr_material("ProjectedAtlasProvenance", texture_index=7)

    assert material == {
        "name": "ProjectedAtlasProvenance",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.85,
            "baseColorTexture": {"index": 7},
        },
    }
    assert set(material) == {"name", "pbrMetallicRoughness"}
    assert set(material["pbrMetallicRoughness"]) == {
        "baseColorFactor", "metallicFactor", "roughnessFactor", "baseColorTexture"
    }


def test_clean_pbr_material_rejects_non_rgba_factor() -> None:
    with pytest.raises(ValueError, match="four components"):
        clean_pbr_material("bad", base_color_factor=(1.0, 1.0, 1.0))  # type: ignore[arg-type]


def _write_minimal_indexed_glb(path: Path, source_sampler: dict) -> None:
    indices = struct.pack("<6H", 0, 1, 2, 2, 3, 0)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(indices)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(indices)}],
        "accessors": [{"bufferView": 0, "componentType": 5123, "count": 6,
                       "type": "SCALAR"}],
        "images": [{"uri": "source.png"}],
        "samplers": [dict(source_sampler)],
        "textures": [{"sampler": 0, "source": 0}],
        "materials": [{
            "name": "SourceMasked",
            "alphaMode": "MASK",
            "normalTexture": {"index": 0},
            "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
        }],
        "meshes": [{"primitives": [{"indices": 0, "material": 0}]}],
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = indices + b"\x00" * ((4 - len(indices) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        + struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes)


def test_bind_texture_uses_dedicated_linear_clamped_opaque_atlas(tmp_path: Path) -> None:
    source_sampler = {"magFilter": 9728, "minFilter": 9987,
                      "wrapS": 10497, "wrapT": 10497}
    source = tmp_path / "source.glb"
    output = tmp_path / "textured.glb"
    _write_minimal_indexed_glb(source, source_sampler)

    bind_texture(source, output, b"atlas-png", np.asarray([True, False]))
    gltf, _blob = _read_glb(output)

    assert gltf["samplers"][0] == source_sampler
    assert gltf["samplers"][1] == {
        "magFilter": 9729, "minFilter": 9729,
        "wrapS": 33071, "wrapT": 33071,
    }
    atlas_texture = gltf["textures"][-1]
    assert atlas_texture["sampler"] == 1
    assert gltf["materials"][0]["alphaMode"] == "MASK"
    atlas_material = next(m for m in gltf["materials"]
                           if m.get("name") == "ProjectedAtlasProvenance")
    assert atlas_material["alphaMode"] == "OPAQUE"
    assert "alphaCutoff" not in atlas_material


def test_bind_texture_accepts_explicit_neutral_fallback_factor(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "textured.glb"
    _write_minimal_indexed_glb(source, {"magFilter": 9729, "minFilter": 9729})
    bind_texture(source, output, b"atlas-png", np.asarray([True, False]),
                 neutral_factor=(0.31, 0.27, 0.22, 1.0))
    gltf, _blob = _read_glb(output)
    neutral = next(m for m in gltf["materials"] if m.get("name") == "NeutralUnobservedSurface")
    assert neutral["pbrMetallicRoughness"]["baseColorFactor"] == [0.31, 0.27, 0.22, 1.0]


def test_triangle_coverage_mask_is_closed_over_invalid_owner_ids() -> None:
    owner = np.asarray([
        [-1, 0, 2, 99],
        [3, 0, -4, 1],
    ], dtype=np.int64)

    assert triangle_coverage_mask(owner, 4).tolist() == [True, True, True, True]


def test_triangle_coverage_mask_preserves_unowned_triangles() -> None:
    owner = np.asarray([[-1, 4], [-1, -1]], dtype=np.int32)

    assert triangle_coverage_mask(owner, 5).tolist() == [False, False, False, False, True]


def test_triangle_coverage_mask_rejects_negative_triangle_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        triangle_coverage_mask(np.zeros((1, 1), dtype=np.int32), -1)
