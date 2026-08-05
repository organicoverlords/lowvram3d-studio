"""Perform one bounded UV chart-separation pass for proven front/rear conflicts.

Only the rear triangles listed by ``uv_targeted_conflict.py`` are split from their
old UV ownership.  Their positions, normals, face order, and geometry are copied
exactly; only duplicated corner vertices and TEXCOORD_0 values are added.  The
affected triangles are placed in one automatically selected atlas rectangle that
has zero positive-area overlap with the front-observed UV triangles.

This is intentionally not a general unwrap and never invokes xatlas or Blender.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "workers"))

from lowvram3d.uv_overlap import AREA_EPSILON_UV, _clip_convex, _polygon_area
from uv_exact_validate import _accessor


def _read_glb(path: Path) -> tuple[dict, bytes, bytes]:
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise RuntimeError(f"not a GLB: {path}")
    json_len = struct.unpack_from("<I", data, 12)[0]
    json_start = 20
    gltf = json.loads(data[json_start : json_start + json_len])
    offset = json_start + json_len
    offset += (-json_len) % 4
    bin_len, bin_kind = struct.unpack_from("<II", data, offset)
    if bin_kind != 0x004E4942:
        raise RuntimeError("GLB has no BIN chunk")
    return gltf, data[offset + 8 : offset + 8 + bin_len], data


def _read_accessor(gltf: dict, blob: bytes, index: int) -> np.ndarray:
    return _accessor(gltf, blob, index).copy()


def _uv_face_triangles(gltf: dict, blob: bytes) -> tuple[np.ndarray, list[tuple[int, int]]]:
    triangles: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for mesh in gltf["meshes"]:
        for primitive in mesh["primitives"]:
            attrs = primitive.get("attributes", {})
            if "TEXCOORD_0" not in attrs or "indices" not in primitive:
                continue
            uv = _read_accessor(gltf, blob, attrs["TEXCOORD_0"]).astype(np.float64)
            indices = _read_accessor(gltf, blob, primitive["indices"]).astype(np.int64).reshape(-1, 3)
            tri = uv[indices]
            triangles.append(tri)
            ranges.append((cursor, cursor + len(tri)))
            cursor += len(tri)
    if not triangles:
        raise RuntimeError("GLB has no indexed TEXCOORD_0 primitive")
    return np.concatenate(triangles), ranges


def _positive_overlap(a: np.ndarray, b: np.ndarray) -> float:
    if (
        a[:, 0].max() < b[:, 0].min()
        or b[:, 0].max() < a[:, 0].min()
        or a[:, 1].max() < b[:, 1].min()
        or b[:, 1].max() < a[:, 1].min()
    ):
        return 0.0
    return _polygon_area(_clip_convex(a.copy(), b.copy()))


def _find_empty_target(front_uv: np.ndarray, front_ids: np.ndarray, size_pixels: int = 64) -> tuple[float, float, float]:
    """Find a square with exact zero-area overlap against front UV triangles."""
    size = float(size_pixels) / 1024.0
    low = front_uv[front_ids].min(axis=1)
    high = front_uv[front_ids].max(axis=1)
    # Scan a deterministic coarse grid first, then a one-pixel grid if needed.
    for step_pixels in (32, 8, 1):
        for y_px in range(0, 1024 - size_pixels + 1, step_pixels):
            for x_px in range(0, 1024 - size_pixels + 1, step_pixels):
                x0, y0 = x_px / 1024.0, y_px / 1024.0
                x1, y1 = x0 + size, y0 + size
                rect = np.asarray(
                    [[[x0, y0], [x1, y0], [x0, y1]], [[x1, y0], [x1, y1], [x0, y1]]],
                    dtype=np.float64,
                )
                conflict = False
                # Bounding-box prefilter keeps the exact check bounded.
                candidates = front_ids[
                    (high[:, 0] >= x0)
                    & (low[:, 0] <= x1)
                    & (high[:, 1] >= y0)
                    & (low[:, 1] <= y1)
                ]
                for triangle_id in candidates.tolist():
                    if any(_positive_overlap(front_uv[triangle_id], r) > AREA_EPSILON_UV for r in rect):
                        conflict = True
                        break
                if not conflict:
                    return x0, y0, size
        # A smaller rectangle is still a valid separated chart as long as its area is nonzero.
        size_pixels = max(8, size_pixels // 2)
        size = float(size_pixels) / 1024.0
    raise RuntimeError("could not find an atlas rectangle disjoint from front-observed UVs")


def _append_buffer(blob: bytearray, payload: bytes) -> tuple[int, int]:
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    blob.extend(payload)
    return offset, len(payload)


def _make_accessor(
    gltf: dict, blob: bytearray, array: np.ndarray, *, target: int | None, component_type: int, kind: str
) -> int:
    dtype = np.dtype("<f4") if component_type == 5126 else np.dtype("<u4")
    values = np.asarray(array, dtype=dtype)
    offset, length = _append_buffer(blob, values.tobytes(order="C"))
    view: dict = {"buffer": 0, "byteOffset": offset, "byteLength": length}
    if target is not None:
        view["target"] = target
    gltf.setdefault("bufferViews", []).append(view)
    accessor: dict = {
        "bufferView": len(gltf["bufferViews"]) - 1,
        "componentType": component_type,
        "count": int(len(values)),
        "type": kind,
    }
    if kind != "SCALAR":
        accessor["min"] = values.min(axis=0).astype(float).tolist()
        accessor["max"] = values.max(axis=0).astype(float).tolist()
    gltf.setdefault("accessors", []).append(accessor)
    return len(gltf["accessors"]) - 1


def _geometry_face_hash(gltf: dict, blob: bytes) -> str:
    digest = hashlib.sha256()
    for mesh in gltf["meshes"]:
        for primitive in mesh["primitives"]:
            attrs = primitive.get("attributes", {})
            if "POSITION" not in attrs or "indices" not in primitive:
                continue
            positions = _read_accessor(gltf, blob, attrs["POSITION"]).astype("<f4")
            indices = _read_accessor(gltf, blob, primitive["indices"]).astype(np.int64).reshape(-1, 3)
            digest.update(positions[indices].tobytes(order="C"))
    return digest.hexdigest()


def separate(input_path: Path, output_path: Path, conflict_report_path: Path, report_path: Path) -> dict:
    gltf, original_blob, _ = _read_glb(input_path)
    conflict = json.loads(conflict_report_path.read_text(encoding="utf-8"))
    if not conflict.get("success") or int(conflict.get("positive_overlap_pair_count", 0)) <= 0:
        raise RuntimeError("chart separation requires a proven positive-area conflict")
    affected = set(int(value) for value in conflict["conflicting_rear_triangle_ids"])
    if not affected:
        raise RuntimeError("conflict report has no affected rear triangles")

    all_uv, ranges = _uv_face_triangles(gltf, original_blob)
    front_ids = np.asarray(conflict["conflicting_front_triangle_ids"], dtype=np.int64)
    x0, y0, size = _find_empty_target(all_uv, front_ids)
    neutral_uv = np.asarray(
        [[x0 + size * 0.1, y0 + size * 0.1], [x0 + size * 0.9, y0 + size * 0.1], [x0 + size * 0.1, y0 + size * 0.9]],
        dtype=np.float32,
    )

    before_face_hash = _geometry_face_hash(gltf, original_blob)
    before_uv_hash = hashlib.sha256(all_uv.astype("<f4").tobytes(order="C")).hexdigest()
    new_gltf = copy.deepcopy(gltf)
    new_blob = bytearray(original_blob)
    global_triangle = 0
    separated = 0
    for mesh_index, mesh in enumerate(new_gltf["meshes"]):
        for primitive_index, primitive in enumerate(mesh["primitives"]):
            attrs = primitive.get("attributes", {})
            if "TEXCOORD_0" not in attrs or "indices" not in primitive:
                continue
            old_positions = _read_accessor(gltf, original_blob, attrs["POSITION"])
            old_uv = _read_accessor(gltf, original_blob, attrs["TEXCOORD_0"])
            old_indices = _read_accessor(gltf, original_blob, primitive["indices"]).astype(np.int64).reshape(-1, 3)
            old_normals = _read_accessor(gltf, original_blob, attrs["NORMAL"]) if "NORMAL" in attrs else None
            new_values: dict[str, np.ndarray] = {}
            for name, accessor_index in attrs.items():
                values = _read_accessor(gltf, original_blob, accessor_index)
                new_values[name] = values.copy()
            new_indices = old_indices.copy()
            additions: dict[str, list[np.ndarray]] = {name: [] for name in attrs}
            for local_face, face in enumerate(old_indices):
                global_face = global_triangle + local_face
                if global_face not in affected:
                    continue
                base = len(new_values["POSITION"]) + len(additions["POSITION"])
                for corner in range(3):
                    source_vertex = int(face[corner])
                    for name in attrs:
                        if name == "TEXCOORD_0":
                            additions[name].append(neutral_uv[corner])
                        else:
                            additions[name].append(new_values[name][source_vertex])
                    new_indices[local_face, corner] = base + corner
                separated += 1
            for name, extra in additions.items():
                if extra:
                    new_values[name] = np.concatenate([new_values[name], np.asarray(extra, dtype=new_values[name].dtype)])

            # Rebuild only this primitive's attribute/index accessors; old unused accessors remain harmless.
            updated_attrs = {}
            for name, values in new_values.items():
                component_type = 5126
                kind = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[values.shape[1] if values.ndim == 2 else 1]
                updated_attrs[name] = _make_accessor(
                    new_gltf,
                    new_blob,
                    values,
                    target=34962,
                    component_type=component_type,
                    kind=kind,
                )
            primitive["attributes"] = updated_attrs
            primitive["indices"] = _make_accessor(
                new_gltf, new_blob, new_indices.reshape(-1), target=34963, component_type=5125, kind="SCALAR"
            )
            global_triangle += len(old_indices)

    if separated != len(affected):
        raise RuntimeError(f"affected triangle count mismatch: separated={separated} expected={len(affected)}")

    new_gltf["buffers"][0]["byteLength"] = len(new_blob)
    json_bytes = json.dumps(new_gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = bytes(new_blob) + b"\x00" * ((4 - len(new_blob) % 4) % 4)
    glb = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_bytes) + 8 + len(bin_bytes))
    glb += struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    glb += struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(glb)

    after_gltf, after_blob, _ = _read_glb(output_path)
    after_uv, _ = _uv_face_triangles(after_gltf, after_blob)
    report = {
        "schema": "targeted_uv_chart_separation_report_v1",
        "input": str(input_path),
        "output": str(output_path),
        "conflict_report": str(conflict_report_path),
        "affected_rear_triangle_count": len(affected),
        "separated_triangle_count": separated,
        "target_rectangle_uv": [x0, y0, size, size],
        "neutral_triangle_uv": neutral_uv.astype(float).tolist(),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "source_geometry_face_hash": before_face_hash,
        "output_geometry_face_hash": _geometry_face_hash(after_gltf, after_blob),
        "geometry_face_hash_preserved": before_face_hash == _geometry_face_hash(after_gltf, after_blob),
        "source_uv_hash": before_uv_hash,
        "output_uv_hash": hashlib.sha256(after_uv.astype("<f4").tobytes(order="C")).hexdigest(),
        "uv_changed": not np.array_equal(all_uv.astype("<f4"), after_uv.astype("<f4")),
        "policy": "separate_only_proven_front_rear_conflicts; no_xatlas; no_geometry_generation",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--conflict-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = separate(Path(args.input), Path(args.output), Path(args.conflict_report), Path(args.report))
    print(
        f"UV_CHART_SEPARATION separated={report['separated_triangle_count']} "
        f"geometry_preserved={report['geometry_face_hash_preserved']} output={report['output']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
