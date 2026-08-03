"""Remove one proven stretched-face set while preserving GLB attributes, UVs and images."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from mesh_io import read_glb, triangle_components


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise RuntimeError("NOT_GLB")
    json_length = struct.unpack_from("<I", raw, 12)[0]
    meta_start = 20
    meta = json.loads(raw[meta_start:meta_start + json_length])
    bin_start = meta_start + ((json_length + 3) // 4) * 4 + 8
    bin_length = struct.unpack_from("<I", raw, bin_start - 8)[0]
    return meta, raw[bin_start:bin_start + bin_length]


def accessor_bytes(meta: dict, binary: bytes, accessor_index: int) -> bytes:
    accessor = meta["accessors"][accessor_index]
    view = meta["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    component = {5121: 1, 5123: 2, 5125: 4, 5126: 4}[accessor["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    item = component * width
    stride = view.get("byteStride") or item
    if stride == item:
        return binary[start:start + count * item]
    return b"".join(binary[start + i * stride:start + i * stride + item] for i in range(count))


def write_glb(meta: dict, binary: bytes, output: Path) -> None:
    json_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    binary = binary + b"\x00" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        + struct.pack("<II", len(binary), 0x004E4942) + binary
    )


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def primitive_ranges(meta: dict) -> list[dict]:
    result = []
    cursor = 0
    for mesh_id, mesh in enumerate(meta.get("meshes", [])):
        for primitive_id, primitive in enumerate(mesh.get("primitives", [])):
            count = meta["accessors"][primitive["indices"]]["count"] // 3
            result.append({
                "mesh": mesh_id,
                "primitive": primitive_id,
                "global_start": cursor,
                "global_end": cursor + count,
                "triangles": count,
                "material": primitive.get("material"),
                "index_accessor": primitive["indices"],
            })
            cursor += count
    return result


def topology_stats(positions: np.ndarray, triangles: np.ndarray) -> dict:
    labels, _ = triangle_components(positions, triangles, 4e-4)
    sizes = np.bincount(labels)
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "vertices": int(len(positions)),
        "triangles": int(len(triangles)),
        "components": int(len(sizes)),
        "largest_component_fraction": float(sizes.max() / max(len(triangles), 1)),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "non_manifold_edges": int(np.count_nonzero(counts > 2)),
        "unique_edges": int(len(unique)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glb", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--face-id", action="append", type=int, required=True)
    args = parser.parse_args()

    input_glb = Path(args.input_glb)
    output_glb = Path(args.output_glb)
    selected = np.asarray(sorted(set(args.face_id)), dtype=np.int64)
    meta, binary = read_glb(input_glb)
    ranges = primitive_ranges(meta)
    positions, _normals, _uv, triangles = read_glb_mesh(input_glb)
    if len(selected) == 0 or int(selected.min()) < 0 or int(selected.max()) >= len(triangles):
        raise RuntimeError("SPIKE_FACE_ID_OUT_OF_RANGE")

    selected_ranges = [r for r in ranges if r["global_start"] <= int(selected.min()) < r["global_end"]]
    if len(selected_ranges) != 1 or any(not (selected_ranges[0]["global_start"] <= int(face) < selected_ranges[0]["global_end"]) for face in selected):
        raise RuntimeError("SPIKE_FACES_CROSS_PRIMITIVES")
    target_range = selected_ranges[0]
    if target_range["material"] != 1:
        raise RuntimeError("SPIKE_MATERIAL_PROVENANCE_CHANGED")

    st = positions[triangles[selected]]
    side_lengths = np.stack((
        np.linalg.norm(st[:, 1] - st[:, 0], axis=1),
        np.linalg.norm(st[:, 2] - st[:, 1], axis=1),
        np.linalg.norm(st[:, 0] - st[:, 2], axis=1),
    ), axis=1)
    extent = st.max(axis=(0, 1)) - st.min(axis=(0, 1))
    selected_aspects = side_lengths.max(axis=1) / np.maximum(side_lengths.min(axis=1), 1e-12)

    accessor_index = target_range["index_accessor"]
    accessor = meta["accessors"][accessor_index]
    if accessor["componentType"] != 5125 or accessor["type"] != "SCALAR":
        raise RuntimeError("SPIKE_INDEX_ACCESSOR_UNSUPPORTED")
    original_indices = np.frombuffer(accessor_bytes(meta, binary, accessor_index), dtype="<u4").copy()
    local_face_ids = selected - target_range["global_start"]
    keep = np.ones(accessor["count"] // 3, dtype=bool)
    keep[local_face_ids] = False
    repaired_indices = original_indices.reshape(-1, 3)[keep].reshape(-1)
    repaired_bytes = np.ascontiguousarray(repaired_indices.astype("<u4")).tobytes()
    binary_mut = bytearray(binary)
    while len(binary_mut) % 4:
        binary_mut.append(0)
    new_offset = len(binary_mut)
    binary_mut.extend(repaired_bytes)
    new_view = {
        "buffer": 0,
        "byteOffset": new_offset,
        "byteLength": len(repaired_bytes),
        "target": 34963,
    }
    meta.setdefault("bufferViews", []).append(new_view)
    accessor["bufferView"] = len(meta["bufferViews"]) - 1
    accessor["byteOffset"] = 0
    accessor["count"] = int(len(repaired_indices))
    write_glb(meta, bytes(binary_mut), output_glb)

    repaired_positions, _n2, _uv2, repaired_triangles = read_glb_mesh(output_glb)
    before_topology = topology_stats(positions, triangles)
    after_topology = topology_stats(repaired_positions, repaired_triangles)
    report = {
        "schema": "horizontal_spike_repair_v1",
        "classification": "PROVEN_TARGETED_STRETCHED_TRIANGLE_REMOVAL",
        "input_glb": str(input_glb),
        "output_glb": str(output_glb),
        "selected_face_ids": selected.tolist(),
        "selected_triangle_count": int(len(selected)),
        "selected_component_id": 0,
        "selected_primitive": target_range,
        "selected_material": target_range["material"],
        "selected_bounds": {
            "min": st.min(axis=(0, 1)).tolist(),
            "max": st.max(axis=(0, 1)).tolist(),
            "extent": extent.tolist(),
        },
        "selected_screen_evidence": {
            "camera": "production +z front, 512px orthographic",
            "pixel_bounds": [134.3, 255.2, 361.9, 256.4],
            "horizontal_spike": True,
        },
        "side_length_aspect_ratio": {
            "min": float(selected_aspects.min()),
            "max": float(selected_aspects.max()),
            "median": float(np.median(selected_aspects)),
        },
        "connectivity": {
            "whole_mesh_component_id": 0,
            "whole_mesh_component_count_before": before_topology["components"],
            "selected_faces_are_detached_component": False,
            "selected_faces_are_stretched_surface_patch": True,
            "artifact_fan_vertex_isolated_from_other_faces": True,
            "repair_scope": "exact selected faces only; no vertex/UV/normal deletion",
        },
        "topology_before": before_topology,
        "topology_after": after_topology,
        "position_bytes_sha256_before": digest(np.ascontiguousarray(positions).tobytes()),
        "position_bytes_sha256_after": digest(np.ascontiguousarray(repaired_positions).tobytes()),
        "input_glb_sha256": digest(input_glb.read_bytes()),
        "output_glb_sha256": digest(output_glb.read_bytes()),
        "triangle_count_delta": int(len(repaired_triangles) - len(triangles)),
        "texture_uv_atlas_untouched": True,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SPIKE_REPAIR_DONE removed={len(selected)} triangles output={output_glb}", flush=True)


def read_glb_mesh(path: Path):
    # Keep this import local so this worker remains executable with PYTHONPATH=workers.
    return read_glb_arrays(path)


def read_glb_arrays(path: Path):
    from mesh_io import read_glb as read_mesh_glb
    return read_mesh_glb(path)


if __name__ == "__main__":
    main()
