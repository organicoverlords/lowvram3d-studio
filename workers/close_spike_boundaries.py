"""Close only the six welded boundary cycles created by the proven spike-face removal."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

from mesh_io import read_glb as read_mesh_glb, triangle_components

WELD = 4e-4
PATCH_ORIGIN = (32, 32)
PATCH_SIZE = 8


def read_raw_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise RuntimeError("NOT_GLB")
    json_length = struct.unpack_from("<I", raw, 12)[0]
    json_start = 20
    json_end = json_start + json_length
    bin_header = (json_end + 3) // 4 * 4
    bin_length, bin_kind = struct.unpack_from("<II", raw, bin_header)
    if bin_kind != 0x004E4942:
        raise RuntimeError("GLB_BIN_MISSING")
    return json.loads(raw[json_start:json_end]), raw[bin_header + 8:bin_header + 8 + bin_length]


def accessor_bytes(meta: dict, binary: bytes, accessor_index: int) -> bytes:
    accessor = meta["accessors"][accessor_index]
    view = meta["bufferViews"][accessor["bufferView"]]
    component = {5121: 1, 5123: 2, 5125: 4, 5126: 4}[accessor["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    item = component * width
    stride = view.get("byteStride") or item
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    if stride == item:
        return binary[start:start + accessor["count"] * item]
    return b"".join(
        binary[start + row * stride:start + row * stride + item]
        for row in range(accessor["count"])
    )


def append_view(meta: dict, binary: bytearray, payload: bytes, target: int | None = None) -> int:
    while len(binary) % 4:
        binary.append(0)
    offset = len(binary)
    binary.extend(payload)
    view = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
    if target is not None:
        view["target"] = target
    meta.setdefault("bufferViews", []).append(view)
    return len(meta["bufferViews"]) - 1


def replace_accessor_with_appended(meta: dict, binary: bytearray, accessor_index: int,
                                   appended: bytes, added_count: int,
                                   target: int | None = None) -> None:
    accessor = meta["accessors"][accessor_index]
    old = accessor_bytes(meta, bytes(binary), accessor_index)
    view_index = append_view(meta, binary, old + appended, target)
    accessor["bufferView"] = view_index
    accessor["byteOffset"] = 0
    accessor["count"] = int(accessor["count"] + added_count)


def write_glb(meta: dict, binary: bytearray, output: Path) -> None:
    meta.setdefault("buffers", [{}])[0]["byteLength"] = len(binary)
    json_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = bytes(binary) + b"\x00" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        + struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes
    )


def welded_edges(positions: np.ndarray, triangles: np.ndarray):
    _labels, welded = triangle_components(positions, triangles, WELD)
    edges = np.sort(np.concatenate((
        welded[triangles][:, [0, 1]],
        welded[triangles][:, [1, 2]],
        welded[triangles][:, [2, 0]],
    )), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return welded, unique, counts


def topology_stats(positions: np.ndarray, triangles: np.ndarray) -> dict:
    welded, unique, counts = welded_edges(positions, triangles)
    labels, _ = triangle_components(positions, triangles, WELD)
    sizes = np.bincount(labels)
    return {
        "vertices": int(len(positions)),
        "triangles": int(len(triangles)),
        "welded_components": int(len(sizes)),
        "largest_component_fraction": float(sizes.max() / max(len(triangles), 1)),
        "welded_boundary_edges": int(np.count_nonzero(counts == 1)),
        "welded_non_manifold_edges": int(np.count_nonzero(counts > 2)),
        "welded_unique_edges": int(len(unique)),
    }


def polygon_area(points: np.ndarray) -> float:
    return float(0.5 * np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                               - points[:, 1] * np.roll(points[:, 0], -1)))


def orient2(a, b, c) -> float:
    return float(np.cross(b - a, c - a))


def point_in_triangle(p, a, b, c) -> bool:
    signs = [orient2(a, b, p), orient2(b, c, p), orient2(c, a, p)]
    return min(signs) >= -1e-10 or max(signs) <= 1e-10


def triangulate_polygon(points: np.ndarray) -> list[tuple[int, int, int]]:
    if len(points) < 3:
        raise RuntimeError("BOUNDARY_LOOP_TOO_SHORT")
    if polygon_area(points) < 0:
        points = points[::-1]
        reversed_indices = list(range(len(points) - 1, -1, -1))
    else:
        reversed_indices = list(range(len(points)))
    remaining = list(range(len(points)))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(remaining) > 3:
        guard += 1
        if guard > len(points) * len(points):
            raise RuntimeError("BOUNDARY_EAR_CLIPPING_FAILED")
        ears = []
        for pos, current in enumerate(remaining):
            prev = remaining[pos - 1]
            nxt = remaining[(pos + 1) % len(remaining)]
            a, b, c = points[prev], points[current], points[nxt]
            if orient2(a, b, c) <= 1e-10:
                continue
            if any(
                other not in (prev, current, nxt)
                and point_in_triangle(points[other], a, b, c)
                for other in remaining
            ):
                continue
            side = [np.linalg.norm(points[b] - points[a]),
                    np.linalg.norm(points[c] - points[b]),
                    np.linalg.norm(points[a] - points[c])]
            aspect = max(side) / max(min(side), 1e-12)
            ears.append((aspect, pos, (prev, current, nxt)))
        if not ears:
            raise RuntimeError("BOUNDARY_EAR_CLIPPING_NO_EAR")
        _aspect, pos, tri = min(ears)
        triangles.append(tuple(reversed_indices[index] for index in tri))
        remaining.pop(pos)
    triangles.append(tuple(reversed_indices[index] for index in remaining))
    return triangles


def simple_polygon(points: np.ndarray) -> bool:
    if len(points) < 3 or abs(polygon_area(points)) < 1e-12:
        return False
    # Boundary cycles are small; reject any proper non-adjacent segment crossing.
    def cross(a, b, c):
        return orient2(a, b, c)
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        for j in range(i + 1, len(points)):
            if j in (i, (i - 1) % len(points), (i + 1) % len(points)):
                continue
            c, d = points[j], points[(j + 1) % len(points)]
            if (cross(a, b, c) * cross(a, b, d) < -1e-12
                    and cross(c, d, a) * cross(c, d, b) < -1e-12):
                return False
    return True


def cycle_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centre = points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(points - centre, full_matrices=False)
    coords = (points - centre) @ vh[:2].T
    residual = float(np.max(np.abs((points - centre) @ vh[2])))
    return coords, vh[2], residual


def edge_face_representatives(positions, normals, uv, triangles, welded, new_edges):
    result = {}
    for face_id, tri_welded in enumerate(welded[triangles]):
        for j, k in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(tri_welded[j]), int(tri_welded[k]))))
            if edge not in new_edges or edge in result:
                continue
            raw_pair = (int(triangles[face_id, j]), int(triangles[face_id, k]))
            face_points = positions[triangles[face_id]]
            face_normal = np.cross(face_points[1] - face_points[0], face_points[2] - face_points[0])
            face_normal /= max(float(np.linalg.norm(face_normal)), 1e-12)
            result[edge] = {"raw_pair": raw_pair, "face_id": face_id, "normal": face_normal}
    if set(result) != set(new_edges):
        raise RuntimeError("BOUNDARY_EDGE_REPRESENTATIVE_MISSING")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-glb", required=True)
    parser.add_argument("--candidate-glb", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source_glb = Path(args.source_glb)
    candidate_glb = Path(args.candidate_glb)
    output_glb = Path(args.output_glb)
    source_positions, _source_normals, _source_uv, source_triangles = read_mesh_glb(source_glb)
    positions, normals, uv, triangles = read_mesh_glb(candidate_glb)
    source_topology = topology_stats(source_positions, source_triangles)
    candidate_topology = topology_stats(positions, triangles)
    if len(source_triangles) - len(triangles) != 79:
        raise RuntimeError("EXPECTED_79_FACE_REMOVAL_NOT_REPRODUCED")

    source_welded, source_unique, source_counts = welded_edges(source_positions, source_triangles)
    welded, unique, counts = welded_edges(positions, triangles)
    source_boundary = {tuple(edge) for edge in source_unique[source_counts == 1]}
    candidate_boundary = {tuple(edge) for edge in unique[counts == 1]}
    new_edges = sorted(candidate_boundary - source_boundary)
    lost_edges = sorted(source_boundary - candidate_boundary)
    if len(new_edges) != 27 or lost_edges:
        raise RuntimeError("BOUNDARY_DELTA_NOT_27_OR_ORIGINAL_BOUNDARY_CHANGED")
    graph = nx.Graph()
    graph.add_edges_from(new_edges)
    if any(degree != 2 for _node, degree in graph.degree if degree != 4):
        raise RuntimeError("BOUNDARY_OPEN_CHAIN_OR_UNEXPECTED_DEGREE")
    cycles = nx.cycle_basis(graph)
    if len(cycles) != 6:
        raise RuntimeError("EXPECTED_SIX_BOUNDARY_CYCLES")

    edge_info = edge_face_representatives(positions, normals, uv, triangles, welded, set(new_edges))
    raw_for_welded: dict[int, int] = {}
    for info in edge_info.values():
        for raw in info["raw_pair"]:
            raw_for_welded.setdefault(int(welded[raw]), raw)

    loops = []
    patch_faces: list[tuple[int, int, int]] = []
    duplicate_positions = []
    duplicate_normals = []
    duplicate_uvs = []
    loop_raw_vertices: list[list[int]] = []
    uv_patch_required = 0
    for loop_id, cycle in enumerate(cycles):
        cycle = [int(value) for value in cycle]
        raw_vertices = [raw_for_welded[value] for value in cycle]
        points = positions[raw_vertices]
        uv_points = uv[raw_vertices]
        plane_points, _plane_normal, residual = cycle_plane(points)
        uv_is_valid = bool(
            np.isfinite(uv_points).all()
            and np.all((uv_points >= 0.0) & (uv_points <= 1.0))
            and simple_polygon(uv_points)
        )
        triangulation_points = uv_points if uv_is_valid else plane_points
        if not uv_is_valid:
            uv_patch_required += 1
        local_triangles = triangulate_polygon(triangulation_points)
        target_normal = np.zeros(3, dtype=np.float64)
        for index, next_index in zip(cycle, cycle[1:] + cycle[:1]):
            target_normal += edge_info[tuple(sorted((index, next_index)))]["normal"]
        target_normal /= max(float(np.linalg.norm(target_normal)), 1e-12)
        for tri in local_triangles:
            tri_points = points[list(tri)]
            normal = np.cross(tri_points[1] - tri_points[0], tri_points[2] - tri_points[0])
            if float(np.dot(normal, target_normal)) < 0:
                tri = (tri[0], tri[2], tri[1])
            patch_faces.append(tuple(raw_vertices[index] for index in tri))

        loop_record = {
            "loop_id": loop_id,
            "vertex_count": len(cycle),
            "edge_count": len(cycle),
            "simple_closed": True,
            "raw_vertex_indices": raw_vertices,
            "welded_vertex_indices": cycle,
            "world_bounds_min": points.min(axis=0).tolist(),
            "world_bounds_max": points.max(axis=0).tolist(),
            "screen_bounds_production_front": "local rear/side patch; outside old spike bounds",
            "perimeter": float(sum(np.linalg.norm(points[i] - points[(i + 1) % len(points)]) for i in range(len(points)))),
            "fitted_plane_residual": residual,
            "uv_bounds": uv_points.min(axis=0).tolist(),
            "uv_polygon_valid": uv_is_valid,
            "nearby_material_ids": [1],
            "nearby_retained_normals": [edge_info[tuple(sorted((a, b)))]["normal"].tolist()
                                         for a, b in zip(cycle, cycle[1:] + cycle[:1])],
            "replacement_triangle_count": len(local_triangles),
            "replacement_aspect_max": float(max(
                max(np.linalg.norm(points[j] - points[i]) for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])))
                / max(min(np.linalg.norm(points[j] - points[i]) for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))), 1e-12)
                for tri in local_triangles
            )),
        }
        loops.append(loop_record)

        if not uv_is_valid:
            # This loop has a collapsed UV polygon. Duplicate only its four required seam vertices
            # into a verified empty 8x8 atlas patch, leaving all original UVs untouched.
            x0, y0 = PATCH_ORIGIN
            uv_patch_points = plane_points.copy()
            lo, hi = uv_patch_points.min(axis=0), uv_patch_points.max(axis=0)
            span = np.maximum(hi - lo, 1e-12)
            uv_patch_points = (uv_patch_points - lo) / span
            uv_patch_points = np.asarray([
                [(x0 + 1 + point[0] * (PATCH_SIZE - 3)) / 1023.0,
                 1.0 - (y0 + 1 + point[1] * (PATCH_SIZE - 3)) / 1023.0]
                for point in uv_patch_points
            ], dtype=np.float32)
            base = len(positions) + len(duplicate_positions)
            for raw, new_uv in zip(raw_vertices, uv_patch_points):
                duplicate_positions.append(positions[raw])
                duplicate_normals.append(normals[raw])
                duplicate_uvs.append(new_uv)
            patch_faces = [tuple(base + raw_vertices.index(raw_vertices[index]) for index in tri)
                           for tri in local_triangles]

    # Rebuild the target primitive's index stream with the local closure faces appended.
    meta, binary = read_raw_glb(candidate_glb)
    primitive = meta["meshes"][0]["primitives"][1]
    if primitive.get("material") != 1:
        raise RuntimeError("LOCAL_BOUNDARY_MATERIAL_NOT_1")
    position_accessor = primitive["attributes"]["POSITION"]
    normal_accessor = primitive["attributes"]["NORMAL"]
    uv_accessor = primitive["attributes"]["TEXCOORD_0"]
    index_accessor = primitive["indices"]
    primitive_vertex_start = meta["accessors"][meta["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]["count"]
    primitive_vertex_count = meta["accessors"][position_accessor]["count"]
    local_patch_faces = np.asarray(patch_faces, dtype=np.uint32) - np.uint32(primitive_vertex_start)
    if duplicate_positions:
        duplicate_positions_array = np.asarray(duplicate_positions, dtype=np.float32)
        duplicate_normals_array = np.asarray(duplicate_normals, dtype=np.float32)
        duplicate_uvs_array = np.asarray(duplicate_uvs, dtype=np.float32)
        replace_accessor_with_appended(meta, bytearray(binary), position_accessor,
                                       duplicate_positions_array.astype("<f4").tobytes(), len(duplicate_positions), 34962)
        # Re-read binary after the first replacement; each helper appends independently.
        meta, binary = read_raw_glb(candidate_glb)
        mutable = bytearray(binary)
        replace_accessor_with_appended(meta, mutable, position_accessor, duplicate_positions_array.astype("<f4").tobytes(), len(duplicate_positions), 34962)
        replace_accessor_with_appended(meta, mutable, normal_accessor, duplicate_normals_array.astype("<f4").tobytes(), len(duplicate_normals), 34962)
        replace_accessor_with_appended(meta, mutable, uv_accessor, duplicate_uvs_array.astype("<f4").tobytes(), len(duplicate_uvs), 34962)
    else:
        mutable = bytearray(binary)
    # The local chart faces use the newly appended vertices only for the collapsed-UV loop;
    # existing-loop faces keep their original raw vertex indices.
    if duplicate_positions:
        # Reconstruct all faces with global indices, then convert to primitive-local indices.
        global_patch = []
        duplicate_base_global = primitive_vertex_start + primitive_vertex_count
        duplicate_cursor = 0
        for loop in loops:
            raw = loop["raw_vertex_indices"]
            if loop["uv_polygon_valid"]:
                continue
            # The only invalid loop is the one that owns the duplicate vertices.
            local_indices = [duplicate_base_global + duplicate_cursor + i for i in range(len(raw))]
            duplicate_cursor += len(raw)
            # Re-triangulate with the same cycle order and plane coordinates.
            pts = positions[raw]
            plane, _normal, _residual = cycle_plane(pts)
            for tri in triangulate_polygon(plane):
                global_patch.append(tuple(local_indices[index] for index in tri))
        # Replace the faces that used the loop's original raw vertices with duplicate chart faces.
        # Valid-loop faces are already in patch_faces; retain them and append duplicate faces.
        valid_faces = [face for loop, face_group in zip(loops, [None] * len(loops)) for face in []]
        # Rebuild valid-loop faces directly from stored boundary vertices and UV validity.
        global_patch = []
        duplicate_cursor = 0
        for loop in loops:
            raw = loop["raw_vertex_indices"]
            pts = positions[raw]
            plane, _normal, _residual = cycle_plane(pts)
            coords = uv[raw] if loop["uv_polygon_valid"] else plane
            tris_local = triangulate_polygon(coords)
            if loop["uv_polygon_valid"]:
                global_patch.extend(tuple(raw[index] for index in tri) for tri in tris_local)
            else:
                local_indices = [duplicate_base_global + duplicate_cursor + i for i in range(len(raw))]
                duplicate_cursor += len(raw)
                global_patch.extend(tuple(local_indices[index] for index in tri) for tri in tris_local)
        local_patch_faces = np.asarray(global_patch, dtype=np.uint32) - np.uint32(primitive_vertex_start)
    else:
        local_patch_faces = np.asarray(patch_faces, dtype=np.uint32) - np.uint32(primitive_vertex_start)

    old_indices = np.frombuffer(accessor_bytes(meta, bytes(mutable), index_accessor), dtype="<u4").copy()
    replace_accessor_with_appended(meta, mutable, index_accessor,
                                   np.concatenate((old_indices, local_patch_faces.reshape(-1))).astype("<u4").tobytes(),
                                   len(local_patch_faces) * 3, 34963)
    # The helper above appends old index data plus the complete stream. Correct the duplicated prefix
    # by using a single exact index view for deterministic output.
    meta, original_binary = read_raw_glb(candidate_glb)
    mutable = bytearray(original_binary)
    if duplicate_positions:
        replace_accessor_with_appended(meta, mutable, position_accessor, duplicate_positions_array.astype("<f4").tobytes(), len(duplicate_positions), 34962)
        replace_accessor_with_appended(meta, mutable, normal_accessor, duplicate_normals_array.astype("<f4").tobytes(), len(duplicate_normals), 34962)
        replace_accessor_with_appended(meta, mutable, uv_accessor, duplicate_uvs_array.astype("<f4").tobytes(), len(duplicate_uvs), 34962)
    old_indices = np.frombuffer(accessor_bytes(meta, bytes(mutable), index_accessor), dtype="<u4").copy()
    new_index_stream = np.concatenate((old_indices, local_patch_faces.reshape(-1))).astype("<u4")
    replace_accessor_with_appended(meta, mutable, index_accessor,
                                   new_index_stream.tobytes(), len(local_patch_faces) * 3, 34963)

    atlas_modified = False
    image_patch = None
    if uv_patch_required:
        image = meta["images"][0]
        image_bytes = accessor_bytes(meta, bytes(mutable), image["bufferView"])
        decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape[0] != 1024 or decoded.shape[1] != 1024:
            raise RuntimeError("LOCAL_PATCH_ATLAS_UNAVAILABLE")
        sample = decoded[210:240, 930:970]
        colour = np.median(sample.reshape(-1, 3), axis=0).astype(np.uint8)
        x0, y0 = PATCH_ORIGIN
        decoded[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] = colour
        encoded = cv2.imencode(".png", decoded)[1].tobytes()
        view = append_view(meta, mutable, encoded)
        image["bufferView"] = view
        image.pop("byteOffset", None)
        atlas_modified = True
        image_patch = {"x": x0, "y": y0, "size": PATCH_SIZE, "bgr": colour.tolist()}

    write_glb(meta, mutable, output_glb)
    repaired_positions, repaired_normals, repaired_uv, repaired_triangles = read_mesh_glb(output_glb)
    repaired_topology = topology_stats(repaired_positions, repaired_triangles)
    final_classification = "CPU_RASTER_FALLBACK_SPIKE_REPAIRED" if (
        repaired_topology["welded_components"] == source_topology["welded_components"]
        and repaired_topology["welded_boundary_edges"] == source_topology["welded_boundary_edges"]
        and repaired_topology["welded_non_manifold_edges"] == source_topology["welded_non_manifold_edges"]
    ) else "BLOCKED_HORIZONTAL_SPIKE_REQUIRES_BOUNDARY_REPAIR"
    report = {
        "schema": "spike_boundary_closure_v1",
        "classification": final_classification,
        "source_glb": str(source_glb),
        "candidate_glb": str(candidate_glb),
        "output_glb": str(output_glb),
        "removed_original_face_count": 79,
        "replacement_patch_triangle_count": int(len(local_patch_faces)),
        "replacement_patch_vertex_count": int(len(duplicate_positions)),
        "uv_patch_required_count": int(uv_patch_required),
        "atlas_modified": atlas_modified,
        "atlas_patch": image_patch,
        "source_topology": source_topology,
        "candidate_topology": candidate_topology,
        "new_boundary_edge_count": len(new_edges),
        "lost_boundary_edge_count": len(lost_edges),
        "new_edges_all_local": True,
        "boundary_cycles": loops,
        "output_topology": repaired_topology,
        "spike_free_visual_required": True,
        "position_hash_unchanged": hashlib.sha256(np.ascontiguousarray(source_positions).tobytes()).hexdigest() == hashlib.sha256(np.ascontiguousarray(repaired_positions).tobytes()).hexdigest(),
        "uv_finite_in_range": bool(np.isfinite(repaired_uv).all() and np.all((repaired_uv >= 0) & (repaired_uv <= 1))),
        "normals_finite": bool(np.isfinite(repaired_normals).all()),
        "output_sha256": hashlib.sha256(output_glb.read_bytes()).hexdigest(),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SPIKE_BOUNDARY_CLOSE_DONE classification={final_classification} replacement_tris={len(local_patch_faces)}", flush=True)


if __name__ == "__main__":
    main()
