from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path


class _Vector:
    def __init__(self, values):
        self.values = [float(value) for value in values]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __add__(self, other):
        return _Vector(a + b for a, b in zip(self.values, other.values))

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        return _Vector(a - b for a, b in zip(self.values, other.values))

    def __mul__(self, scalar):
        return _Vector(value * scalar for value in self.values)

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return _Vector(value / scalar for value in self.values)

    def dot(self, other):
        return sum(a * b for a, b in zip(self.values, other.values))

    @property
    def length(self):
        return math.sqrt(self.dot(self))

    def cross(self, other):
        a, b, c = self.values
        d, e, f = other.values
        return _Vector((b * f - c * e, c * d - a * f, a * e - b * d))

    def normalize(self):
        length = math.sqrt(self.dot(self))
        self.values[:] = [value / length for value in self.values]


def _load_lod_module():
    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(Object=object)
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = _Vector
    common = types.ModuleType("common")
    common.argv_after_double_dash = lambda: []
    common.import_mesh = lambda *_args: []
    common.reset_scene = lambda: None
    common.save_json = lambda *_args: None
    old = {name: sys.modules.get(name) for name in ("bpy", "mathutils", "common")}
    sys.modules.update({"bpy": bpy, "mathutils": mathutils, "common": common})
    try:
        path = Path(__file__).resolve().parents[1] / "blender" / "final_pipeline_lods.py"
        spec = importlib.util.spec_from_file_location("final_pipeline_lods_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _Identity:
    def __matmul__(self, value):
        return value


class _Vertex:
    def __init__(self, index, position):
        self.index = index
        self.co = _Vector(position)


class _Polygon:
    def __init__(self, vertices):
        self.vertices = vertices


def _mesh_object(points, faces):
    data = types.SimpleNamespace(
        vertices=[_Vertex(index, point) for index, point in enumerate(points)],
        polygons=[_Polygon(face) for face in faces],
    )
    return types.SimpleNamespace(data=data, matrix_world=_Identity())


def test_nearby_body_vertex_cannot_satisfy_vanished_silhouette_feature():
    lod = _load_lod_module()
    # The only vertex near the anchor seed belongs to the body triangle.  The anchor's detached
    # feature triangles are gone, so seed identity alone must not make the LOD pass.
    obj = _mesh_object(
        [(-1.0, -1.0, -0.8), (1.0, -1.0, -0.8), (0.8, 0.0, 0.0)],
        [(0, 1, 2)],
    )
    anchor = {
        "anchor_id": "tfa-vanished",
            "seeds": [[0.23094, 0.0, 0.0]],
            "bounds_normalized": {"min": [0.22, -0.01, -0.01], "max": [0.24, 0.01, 0.01]},
        "supported_views": ["front"],
        "survival_floor": {
            "exclusive_pixel_retention_ratio": 0.60,
            "per_view_exclusive_pixels": {"front": 2},
        },
    }
    bounds = (_Vector((-1.0, -1.0, -1.0)), _Vector((1.0, 1.0, 1.0)))
    result = lod.evaluate_anchor_survival(obj, [anchor], {}, bounds, [
        {"name": "front", "direction": [0.0, -1.0, 0.0]},
    ], 32)
    record = result["anchors"][0]
    assert record["retained_seeds"] == 1
    assert record["support"]["front"]["exclusive_pixels"] == 0
    assert record["under_floor_views"] == ["front"]
    assert result["missing_ids"] == ["tfa-vanished"]
