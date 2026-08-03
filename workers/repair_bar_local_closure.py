"""Topology-safe local removal of a proven stretched triangle fan, with hole closure.

The horizontal panda "bar" is a triangle fan whose apex sits at the opposite end of
the mesh from its rim.  ``repair_horizontal_spike.py`` deletes the selected faces and
stops, which leaves the rim edges single-sided; the earlier candidate therefore kept
widening the selection until the surrounding surface went away too.  This worker
instead removes exactly the proven faces, extracts the boundary loop that the removal
exposes, and re-triangulates that one loop from its existing vertices.

Everything outside the loop is untouched: no vertex, normal, UV, material, image or
buffer byte outside the rewritten index range is altered, and the closure reuses only
vertices that already exist, so UVs are inherited rather than invented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from mesh_io import read_glb as read_glb_arrays
from mesh_io import triangle_components


# ---------------------------------------------------------------- GLB container


def read_container(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise RuntimeError("NOT_GLB")
    json_length = struct.unpack_from("<I", raw, 12)[0]
    meta = json.loads(raw[20:20 + json_length])
    bin_start = 20 + ((json_length + 3) // 4) * 4 + 8
    bin_length = struct.unpack_from("<I", raw, bin_start - 8)[0]
    return meta, raw[bin_start:bin_start + bin_length]


def write_container(meta: dict, binary: bytes, output: Path) -> None:
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


def primitive_table(meta: dict) -> list[dict]:
    """Face and vertex ranges per primitive in the order ``mesh_io.read_glb`` concatenates them."""
    table, face_cursor, vertex_cursor = [], 0, 0
    for mesh_id, mesh in enumerate(meta.get("meshes", [])):
        for primitive_id, primitive in enumerate(mesh.get("primitives", [])):
            faces = meta["accessors"][primitive["indices"]]["count"] // 3
            vertices = meta["accessors"][primitive["attributes"]["POSITION"]]["count"]
            table.append({
                "mesh": mesh_id,
                "primitive": primitive_id,
                "face_start": face_cursor,
                "face_end_exclusive": face_cursor + faces,
                "triangle_count": faces,
                "vertex_start": vertex_cursor,
                "vertex_end_exclusive": vertex_cursor + vertices,
                "material": primitive.get("material"),
                "index_accessor": primitive["indices"],
            })
            face_cursor += faces
            vertex_cursor += vertices
    return table


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- topology


def undirected_edge_counts(triangles: np.ndarray):
    edges = np.sort(np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0), axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def topology_stats(positions: np.ndarray, triangles: np.ndarray) -> dict:
    labels, _ = triangle_components(positions, triangles, 4e-4)
    sizes = np.bincount(labels)
    unique, counts = undirected_edge_counts(triangles)
    areas = np.linalg.norm(np.cross(
        positions[triangles[:, 1]] - positions[triangles[:, 0]],
        positions[triangles[:, 2]] - positions[triangles[:, 0]]), axis=1) * 0.5
    sorted_triangles = np.sort(triangles, axis=1)
    _, duplicate_counts = np.unique(sorted_triangles, axis=0, return_counts=True)
    return {
        "vertices": int(len(positions)),
        "triangles": int(len(triangles)),
        "components": int(len(sizes)),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "non_manifold_edges": int(np.count_nonzero(counts > 2)),
        "unique_edges": int(len(unique)),
        "zero_area_faces": int(np.count_nonzero(areas <= 0.0)),
        "duplicate_faces": int(np.count_nonzero(duplicate_counts > 1)),
    }


def aspect_ratios(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    corners = positions[triangles].astype(np.float64)
    sides = np.stack((
        np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1),
        np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1),
        np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1),
    ), axis=1)
    return sides.max(axis=1) / np.maximum(sides.min(axis=1), 1e-12)


def boundary_loops(triangles: np.ndarray, restrict_to: set[int]) -> list[list[int]]:
    """Directed boundary loops of ``triangles`` that touch ``restrict_to``.

    Boundary status is decided against the whole mesh - a UV-chart-separated asset has
    hundreds of thousands of legitimate boundary edges elsewhere, so only the edges in
    the neighbourhood are chained.  The chain follows the face-side traversal of the
    hole, which means a filling triangle must use the reversed edges.
    """
    unique, counts = undirected_edge_counts(triangles)
    boundary = {tuple(int(x) for x in edge) for edge in unique[counts == 1]}
    if not boundary:
        return []
    # Both endpoints, not either: a rim vertex can also sit on an unrelated UV-chart
    # boundary, and "either" drags that chart's whole seam into the chain.
    local = {edge for edge in boundary if edge[0] in restrict_to and edge[1] in restrict_to}
    if not local:
        return []
    touched = np.isin(triangles, np.fromiter(restrict_to, dtype=np.int64)).any(axis=1)
    successor: dict[int, int] = {}
    for a, b, c in triangles[touched]:
        for start, end in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if (min(start, end), max(start, end)) not in local:
                continue
            if start in successor and successor[start] != end:
                raise RuntimeError(f"BOUNDARY_VERTEX_FORKS:{start}")
            successor[start] = end
    loops, seen = [], set()
    for start in sorted(successor):
        if start in seen:
            continue
        loop, node = [], start
        while node not in seen:
            seen.add(node)
            loop.append(node)
            node = successor.get(node)
            if node is None:
                raise RuntimeError(f"BOUNDARY_CHAIN_OPEN:{loop[0]}")
        if node != start:
            raise RuntimeError(f"BOUNDARY_CHAIN_NOT_CLOSED:{start}")
        loops.append(loop)
    return loops


# ---------------------------------------------------------------- closure


def fit_plane(points: np.ndarray):
    centroid = points.mean(axis=0)
    _, singular, basis = np.linalg.svd(points - centroid)
    normal = basis[2]
    deviation = float(np.abs((points - centroid) @ normal).max())
    return centroid, normal, basis[0], basis[1], deviation, singular


def triangulate_loop(loop: list[int], positions: np.ndarray) -> list[tuple[int, int, int]]:
    """Quality-driven ear clipping of one planar loop, in the loop's own order."""
    points = positions[np.asarray(loop)].astype(np.float64)
    centroid, _normal, axis_u, axis_v, _deviation, _s = fit_plane(points)
    flat = np.stack((((points - centroid) @ axis_u), ((points - centroid) @ axis_v)), axis=1)
    signed_area = 0.5 * float(np.sum(
        flat[:, 0] * np.roll(flat[:, 1], -1) - np.roll(flat[:, 0], -1) * flat[:, 1]))
    orientation = 1.0 if signed_area > 0 else -1.0

    def cross(o, a, b):
        return ((flat[a, 0] - flat[o, 0]) * (flat[b, 1] - flat[o, 1])
                - (flat[a, 1] - flat[o, 1]) * (flat[b, 0] - flat[o, 0])) * orientation

    def inside(o, a, b, p):
        d1, d2, d3 = cross(o, a, p), cross(a, b, p), cross(b, o, p)
        return d1 >= 0 and d2 >= 0 and d3 >= 0

    def min_angle(o, a, b):
        pts = flat[[o, a, b]]
        sides = np.array([np.linalg.norm(pts[1] - pts[0]),
                          np.linalg.norm(pts[2] - pts[1]),
                          np.linalg.norm(pts[0] - pts[2])])
        if sides.min() <= 0:
            return -1.0
        # Smallest angle is opposite the shortest side.
        order = np.argsort(sides)
        s, m, l = sides[order]
        cosine = np.clip((m * m + l * l - s * s) / (2 * m * l), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    remaining = list(range(len(loop)))
    fan: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        best = None
        for position in range(len(remaining)):
            a = remaining[position - 1]
            b = remaining[position]
            c = remaining[(position + 1) % len(remaining)]
            if cross(a, b, c) <= 0:
                continue
            others = [v for v in remaining if v not in (a, b, c)]
            if any(inside(a, b, c, v) for v in others):
                continue
            score = min_angle(a, b, c)
            if best is None or score > best[0]:
                best = (score, position, (a, b, c))
        if best is None:
            raise RuntimeError("EAR_CLIP_NO_VALID_EAR")
        fan.append(best[2])
        remaining.pop(best[1])
    fan.append((remaining[0], remaining[1], remaining[2]))
    fan = _lawson_flip(fan, flat, {(i, (i + 1) % len(loop)) for i in range(len(loop))})
    return [(loop[a], loop[b], loop[c]) for a, b, c in fan]


def _lawson_flip(fan, flat, constrained, max_passes: int = 200):
    """Flip interior diagonals until the triangulation is locally Delaunay.

    Greedy ear clipping picks a legal ear, not the best one globally; on a rim whose
    vertex spacing varies by an order of magnitude that alone leaves 20:1 slivers.
    Flipping to Delaunay maximises the minimum angle for this fixed vertex set, which
    is the strongest guarantee available without introducing new vertices.
    """
    fan = [tuple(t) for t in fan]

    def signed_area(t):
        a, b, c = t
        return 0.5 * float((flat[b, 0] - flat[a, 0]) * (flat[c, 1] - flat[a, 1])
                           - (flat[b, 1] - flat[a, 1]) * (flat[c, 0] - flat[a, 0]))

    winding = 1.0 if signed_area(fan[0]) > 0 else -1.0

    def oriented(t):
        return t if signed_area(t) * winding > 0 else (t[0], t[2], t[1])

    def in_circumcircle(a, b, c, d):
        ax, ay = flat[a] - flat[d]
        bx, by = flat[b] - flat[d]
        cx, cy = flat[c] - flat[d]
        return np.linalg.det(np.array([
            [ax, ay, ax * ax + ay * ay],
            [bx, by, bx * bx + by * by],
            [cx, cy, cx * cx + cy * cy],
        ])) > 1e-18

    for _ in range(max_passes):
        edge_faces: dict[tuple[int, int], list[int]] = {}
        for index, (a, b, c) in enumerate(fan):
            for u, v in ((a, b), (b, c), (c, a)):
                edge_faces.setdefault((min(u, v), max(u, v)), []).append(index)
        flipped = False
        for edge, owners in edge_faces.items():
            if len(owners) != 2:
                continue
            if edge in constrained or (edge[1], edge[0]) in constrained:
                continue
            left, right = owners
            opposite_left = [v for v in fan[left] if v not in edge]
            opposite_right = [v for v in fan[right] if v not in edge]
            if len(opposite_left) != 1 or len(opposite_right) != 1:
                continue
            p, q = opposite_left[0], opposite_right[0]
            a, b = edge
            if not in_circumcircle(a, b, p, q):
                continue
            # Flip only when the quadrilateral a-p-b-q is convex, else the flip folds.
            def side(o, x, y):
                return ((flat[x, 0] - flat[o, 0]) * (flat[y, 1] - flat[o, 1])
                        - (flat[x, 1] - flat[o, 1]) * (flat[y, 0] - flat[o, 0]))
            # Convex quad a-p-b-q <=> the new diagonal p-q separates a from b.
            if side(p, q, a) * side(p, q, b) >= 0:
                continue
            fan[left] = oriented((p, q, a))
            fan[right] = oriented((p, q, b))
            flipped = True
            break
        if not flipped:
            break
    return fan


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glb", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--face-id", action="append", type=int, required=True)
    parser.add_argument("--max-local-aspect", type=float, default=None,
                        help="Default: the input mesh's own 99.9th aspect-ratio percentile.")
    parser.add_argument("--max-plane-deviation", type=float, default=1e-4)
    parser.add_argument("--drop-orphaned-fragments", action="store_true",
                        help="Also drop components the removal detaches from the main body.")
    parser.add_argument("--max-orphan-fragment-faces", type=int, default=64)
    args = parser.parse_args()

    input_glb = Path(args.input_glb)
    output_glb = Path(args.output_glb)
    selected = np.asarray(sorted(set(args.face_id)), dtype=np.int64)

    meta, binary = read_container(input_glb)
    table = primitive_table(meta)
    positions, normals, uv, triangles = read_glb_arrays(input_glb)
    if selected.min() < 0 or selected.max() >= len(triangles):
        raise RuntimeError("BAR_FACE_ID_OUT_OF_RANGE")

    owners = [row for row in table
              if row["face_start"] <= int(selected.min()) < row["face_end_exclusive"]]
    if len(owners) != 1:
        raise RuntimeError("BAR_PRIMITIVE_LOOKUP_FAILED")
    target = owners[0]
    if any(not (target["face_start"] <= int(f) < target["face_end_exclusive"]) for f in selected):
        raise RuntimeError("BAR_FACES_CROSS_PRIMITIVES")

    removed_faces = triangles[selected]
    affected_vertices = np.unique(removed_faces)
    if any(not (target["vertex_start"] <= int(v) < target["vertex_end_exclusive"])
           for v in affected_vertices):
        raise RuntimeError("BAR_VERTICES_CROSS_PRIMITIVES")

    aspect_threshold = (args.max_local_aspect if args.max_local_aspect is not None
                        else float(np.percentile(aspect_ratios(positions, triangles), 99.9)))

    keep = np.ones(len(triangles), dtype=bool)
    keep[selected] = False

    # The bar is a fan whose apex is welded into the far end of the mesh, so it can be the
    # sole anchor of whatever its rim belongs to.  Detect that before deciding what to close.
    labels, _ = triangle_components(positions, triangles[keep], 4e-4)
    sizes = np.bincount(labels)
    main_label = int(np.argmax(sizes))
    kept_face_ids = np.where(keep)[0]
    orphan_face_ids = kept_face_ids[labels != main_label]
    orphan_fragments = [
        {
            "component": int(component),
            "face_count": int(sizes[component]),
            "face_ids": kept_face_ids[labels == component].tolist(),
            "bounds_min": positions[np.unique(triangles[kept_face_ids[labels == component]])]
                          .min(axis=0).tolist(),
            "bounds_max": positions[np.unique(triangles[kept_face_ids[labels == component]])]
                          .max(axis=0).tolist(),
        }
        for component in range(len(sizes)) if component != main_label
    ]
    dropped_face_ids = np.zeros(0, dtype=np.int64)
    if args.drop_orphaned_fragments and len(orphan_face_ids):
        oversized = [f for f in orphan_fragments if f["face_count"] > args.max_orphan_fragment_faces]
        if oversized:
            raise RuntimeError(
                f"BAR_ORPHAN_FRAGMENT_TOO_LARGE:{[f['face_count'] for f in oversized]}")
        dropped_face_ids = np.sort(orphan_face_ids)
        keep[dropped_face_ids] = False
    kept = triangles[keep]

    # Boundary loops before and after, limited to the neighbourhood of the removal.
    neighbourhood = set(int(v) for v in affected_vertices)
    loops_before = boundary_loops(triangles, neighbourhood)
    loops_after = boundary_loops(kept, neighbourhood)
    if len(loops_after) > 1:
        raise RuntimeError(f"BAR_UNEXPECTED_LOOP_COUNT:{len(loops_after)}")

    if loops_after:
        loop = loops_after[0]
        loop_points = positions[np.asarray(loop)].astype(np.float64)
        _centroid, _normal, _u, _v, plane_deviation, _s = fit_plane(loop_points)
        if plane_deviation > args.max_plane_deviation:
            raise RuntimeError(f"BAR_LOOP_NOT_PLANAR:{plane_deviation}")
        # The traced loop is the face-side traversal; the closure uses the reverse.
        closure_array = np.asarray(triangulate_loop(list(reversed(loop)), positions),
                                   dtype=np.int64)
        closure_aspects = aspect_ratios(positions, closure_array)
        if float(closure_aspects.max()) > aspect_threshold:
            raise RuntimeError(f"BAR_CLOSURE_ASPECT_REJECTED:{float(closure_aspects.max())}")
    else:
        # Dropping the orphaned fragment takes its hole with it; nothing is left to close.
        loop = []
        plane_deviation = 0.0
        closure_array = np.zeros((0, 3), dtype=np.int64)
        closure_aspects = np.zeros(0)

    repaired = (np.concatenate((kept, closure_array), axis=0) if len(closure_array) else kept)
    if boundary_loops(repaired, neighbourhood):
        raise RuntimeError("BAR_CLOSURE_LEFT_LOCAL_BOUNDARY")

    # Orientation consistency is topological, not a normal-average: on this rim the
    # surviving neighbours curve through 180 degrees, so their mean normal is meaningless.
    # A surface is consistently oriented exactly when no directed half-edge repeats.
    local_faces = [face for face in repaired
                   if set(int(x) for x in face) & neighbourhood]
    directed: set[tuple[int, int]] = set()
    duplicated_directed = []
    for a, b, c in local_faces:
        for edge in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if edge in directed:
                duplicated_directed.append(edge)
            directed.add(edge)

    # ---------------------------------------------------------------- rewrite
    accessor_index = target["index_accessor"]
    accessor = meta["accessors"][accessor_index]
    if accessor["componentType"] != 5125 or accessor["type"] != "SCALAR":
        raise RuntimeError("BAR_INDEX_ACCESSOR_UNSUPPORTED")
    original_indices = np.frombuffer(
        accessor_bytes(meta, binary, accessor_index), dtype="<u4").reshape(-1, 3)
    all_removed = np.sort(np.concatenate((selected, dropped_face_ids)))
    if all_removed.size and (all_removed.min() < target["face_start"]
                             or all_removed.max() >= target["face_end_exclusive"]):
        raise RuntimeError("BAR_REMOVAL_CROSSES_PRIMITIVES")
    local_keep = np.ones(len(original_indices), dtype=bool)
    local_keep[all_removed - target["face_start"]] = False
    local_closure = (closure_array - target["vertex_start"]).astype("<u4")
    rewritten = np.concatenate(
        (original_indices[local_keep], local_closure), axis=0).reshape(-1)
    payload = np.ascontiguousarray(rewritten.astype("<u4")).tobytes()

    mutable = bytearray(binary)
    while len(mutable) % 4:
        mutable.append(0)
    offset = len(mutable)
    mutable.extend(payload)
    meta.setdefault("bufferViews", []).append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(payload), "target": 34963})
    accessor["bufferView"] = len(meta["bufferViews"]) - 1
    accessor["byteOffset"] = 0
    accessor["count"] = int(rewritten.size)
    accessor.pop("min", None)
    accessor.pop("max", None)
    write_container(meta, bytes(mutable), output_glb)

    # ---------------------------------------------------------------- proof
    out_positions, out_normals, out_uv, out_triangles = read_glb_arrays(output_glb)
    before = topology_stats(positions, triangles)
    after = topology_stats(out_positions, out_triangles)
    added_face_ids = list(range(len(out_triangles) - len(closure_array), len(out_triangles)))

    preserved = np.array_equal(out_triangles[:len(kept)], kept)
    gates = {
        "boundary_edges_not_increased": after["boundary_edges"] <= before["boundary_edges"],
        "non_manifold_edges_not_increased":
            after["non_manifold_edges"] <= before["non_manifold_edges"],
        "components_not_increased": after["components"] <= before["components"],
        "no_zero_area_faces": after["zero_area_faces"] == 0,
        "no_duplicate_faces": after["duplicate_faces"] <= before["duplicate_faces"],
        "closure_aspect_within_threshold": bool(
            len(closure_aspects) == 0 or float(closure_aspects.max()) <= aspect_threshold),
        "local_boundary_closed": True,
        "unaffected_faces_preserved_exactly": bool(preserved),
        "positions_unchanged": digest(np.ascontiguousarray(positions).tobytes())
                               == digest(np.ascontiguousarray(out_positions).tobytes()),
        "normals_unchanged": digest(np.ascontiguousarray(normals).tobytes())
                             == digest(np.ascontiguousarray(out_normals).tobytes()),
        "uv_unchanged": digest(np.ascontiguousarray(uv).tobytes())
                        == digest(np.ascontiguousarray(out_uv).tobytes()),
        "finite_geometry": bool(np.isfinite(out_positions).all() and np.isfinite(out_normals).all()
                                and np.isfinite(out_uv).all()),
        "orientation_locally_consistent": not duplicated_directed,
        "no_orphaned_fragment_left_behind": not (orphan_fragments and not args.drop_orphaned_fragments),
    }
    passed = all(gates.values())

    report = {
        "schema": "bar_local_closure_repair_v1",
        "classification": ("PANDA_PRODUCTION_MESH_BAR_REPAIR_PROVEN" if passed
                           else "PANDA_BAR_REMOVAL_REQUIRES_RECONSTRUCTION"),
        "input_glb": str(input_glb),
        "output_glb": str(output_glb),
        "input_glb_sha256": file_digest(input_glb),
        "output_glb_sha256": file_digest(output_glb),
        "primitive": {k: target[k] for k in
                      ("mesh", "primitive", "material", "face_start", "face_end_exclusive",
                       "vertex_start", "vertex_end_exclusive", "index_accessor")},
        "removed_face_ids": selected.tolist(),
        "removed_face_count": int(len(selected)),
        "dropped_orphan_face_ids": dropped_face_ids.tolist(),
        "dropped_orphan_face_count": int(dropped_face_ids.size),
        "orphan_fragments_exposed_by_removal": orphan_fragments,
        "drop_orphaned_fragments": bool(args.drop_orphaned_fragments),
        "aspect_threshold": aspect_threshold,
        "aspect_threshold_source": ("EXPLICIT" if args.max_local_aspect is not None
                                    else "INPUT_MESH_P99_9"),
        "added_face_ids": added_face_ids,
        "added_face_count": int(len(closure_array)),
        "added_faces": closure_array.tolist(),
        "affected_vertex_ids": affected_vertices.tolist(),
        "orphaned_vertex_ids": sorted(
            int(v) for v in affected_vertices if v not in set(int(x) for x in np.unique(repaired))),
        "boundary_loops_before": [len(item) for item in loops_before],
        "boundary_loops_before_vertices": loops_before,
        "boundary_loops_after_removal": [len(item) for item in loops_after],
        "boundary_loop_after_removal_vertices": loop,
        "boundary_loops_after_closure": [],
        "loop_plane_deviation": plane_deviation,
        "local_material": target["material"],
        "local_uv_handling": "REUSED_EXISTING_LOOP_VERTICES_NO_NEW_UVS",
        "closure_aspect_ratio": ({
            "min": float(closure_aspects.min()),
            "median": float(np.median(closure_aspects)),
            "max": float(closure_aspects.max()),
        } if len(closure_aspects) else None),
        "removed_aspect_ratio": {
            "min": float(aspect_ratios(positions, removed_faces).min()),
            "median": float(np.median(aspect_ratios(positions, removed_faces))),
            "max": float(aspect_ratios(positions, removed_faces).max()),
        },
        "duplicated_directed_half_edges": duplicated_directed,
        "topology_before": before,
        "topology_after": after,
        "gates": gates,
        "attribute_digests": {
            "positions_before": digest(np.ascontiguousarray(positions).tobytes()),
            "positions_after": digest(np.ascontiguousarray(out_positions).tobytes()),
            "normals_before": digest(np.ascontiguousarray(normals).tobytes()),
            "normals_after": digest(np.ascontiguousarray(out_normals).tobytes()),
            "uv_before": digest(np.ascontiguousarray(uv).tobytes()),
            "uv_after": digest(np.ascontiguousarray(out_uv).tobytes()),
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"BAR_LOCAL_CLOSURE removed={len(selected)} added={len(closure_array)} "
          f"classification={report['classification']}", flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
