"""Third bounded panda face-surface candidate pass.

v2 was rejected because it selected a broad depth-window union (67 patches,
1,341 triangles, 5,514 writable texels) whose edge bled into the hood. v3
abandons depth-window unions entirely:

  * the face domain comes from the raycast hit pixels themselves: rays are
    cast through the source-space fixture face polygon mask, and a triangle is
    a candidate exactly when it is hit (within rank/facing limits) by one of
    those rays.  No UV centroids are ever tested against the polygon - the
    polygon lives in source image pixels and is consumed only as the raycast
    mask;
  * the seven landmark anchors are the front-most front-facing hit inside the
    polygon near each landmark source point;
  * a minimal connected surface is grown by multi-source Dijkstra over the
    welded 3D-mesh adjacency (so growth crosses atlas seams), with edge weight
    = 3D centroid distance, stopped at a bounded target triangle count.

The result is deliberately much smaller than v2 while still covering nose,
muzzle, eyes, chin and forehead.  Geometry, normals, UVs and indices of the
baseline GLB are never modified; a writable texel mask is computed from the
exact atlas rasteriser.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .atlas_raster import rasterise
    from .face_patch_texture import build_face_patch_atlas
    from .face_surface_candidate import anchors_from_layers, camera_from_projection_fit, camera_to_dict
    from .face_surface_ownership import Camera, hit_ranks, trace_mask_layers
    from .face_surface_candidate_v2 import border_connected_foreground, fixture_face_mask, load_source_fixture
    from .fast_texture_projection import bind_texture, fit_camera, immutable_buffer_hashes
    from .mesh_io import read_glb
except ImportError:  # pragma: no cover
    from atlas_raster import rasterise
    from face_patch_texture import build_face_patch_atlas
    from face_surface_candidate import anchors_from_layers, camera_from_projection_fit, camera_to_dict
    from face_surface_ownership import Camera, hit_ranks, trace_mask_layers
    from face_surface_candidate_v2 import border_connected_foreground, fixture_face_mask, load_source_fixture
    from fast_texture_projection import bind_texture, fit_camera, immutable_buffer_hashes
    from mesh_io import read_glb

EPS = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def points_in_polygon(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Ray-crossing point-in-polygon test, vectorised over points."""
    polygon = np.asarray(polygon, np.float64)
    points = np.asarray(points, np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("FACE_POLYGON_INVALID")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("FACE_POINTS_INVALID")
    x, y = points[:, 0], points[:, 1]
    count = len(polygon)
    inside = np.zeros(len(points), dtype=bool)
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crossing = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-30) + xi
        )
        inside ^= crossing
        j = i
    return inside


def candidate_ids_from_layers(layers: dict[str, np.ndarray], *, max_rank: int = 6,
                               minimum_facing: float = 0.05) -> np.ndarray:
    """Unique face-domain triangle ids from raycast hits inside the polygon.

    Rays were already restricted to the fixture face polygon mask, so every
    triangle here is a polygon-interior candidate.  Rank and facing limits
    reject far/oblique surfaces without any global depth window.
    """
    if len(layers["triangle_ids"]) == 0:
        return np.empty(0, np.int64)
    ranks = hit_ranks(layers["offsets"])
    eligible = (ranks < int(max_rank)) & (layers["normal_facing"] > float(minimum_facing))
    return np.unique(layers["triangle_ids"][eligible])


def anchor_from_layers(layers: dict[str, np.ndarray], landmark: dict[str, Any],
                       *, minimum_facing: float = 0.05, neighbour_rays: int = 96) -> dict[str, Any]:
    """Front-most front-facing hit near a landmark source point.

    Nearest rays (mergesort distance ordering) are searched in ascending
    distance; the first ray that owns a qualifying hit is used, and among the
    hits of that ray the front-most (smallest depth, tie-broken by triangle
    id) is selected.  Deterministic by construction.
    """
    pixels = np.asarray(layers["pixels_xy"], np.float64)
    offsets = np.asarray(layers["offsets"], np.int64)
    triangle_ids = np.asarray(layers["triangle_ids"], np.int64)
    barycentric = np.asarray(layers["barycentric"], np.float64)
    depth = np.asarray(layers["depth"], np.float64)
    facing = np.asarray(layers["normal_facing"], np.float64)
    source_xy = np.asarray(landmark["source_xy"], np.float64)
    order = np.argsort(np.sum((pixels - source_xy[None, :]) ** 2, axis=1), kind="mergesort")
    for ray_index in order[:max(1, int(neighbour_rays))]:
        start, end = int(offsets[ray_index]), int(offsets[ray_index + 1])
        if start >= end:
            continue
        hits = np.arange(start, end)
        valid = facing[hits] > float(minimum_facing)
        if not np.any(valid):
            continue
        hits = hits[valid]
        ordered = hits[np.lexsort((triangle_ids[hits], depth[hits]))]
        hit_index = int(ordered[0])
        return {
            "name": str(landmark["name"]),
            "source_xy": source_xy.tolist(),
            "sampled_ray_xy": pixels[ray_index].tolist(),
            "triangle_id": int(triangle_ids[hit_index]),
            "barycentric": barycentric[hit_index].tolist(),
            "depth": float(depth[hit_index]),
            "facing": float(facing[hit_index]),
        }
    raise RuntimeError(f"FACE_ANCHOR_NOT_FOUND:{landmark['name']}")


def face_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(
        positions[triangles[:, 1]] - positions[triangles[:, 0]],
        positions[triangles[:, 2]] - positions[triangles[:, 0]],
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def weld_mesh_vertices(positions: np.ndarray, *, tolerance: float | None = None) -> np.ndarray:
    """Deterministic welded vertex index from quantised positions."""
    positions = np.asarray(positions, np.float64)
    diagonal = float(np.linalg.norm(np.ptp(positions, axis=0)))
    tol = float(tolerance if tolerance is not None else max(diagonal * 2e-4, 1e-6))
    quantised = np.rint(positions / tol).astype(np.int64)
    _, inverse = np.unique(quantised, axis=0, return_inverse=True)
    return inverse


def weld_face_adjacency(triangles: np.ndarray, welded: np.ndarray,
                        candidate_ids: np.ndarray) -> dict[int, set[int]]:
    """Triangle-triangle adjacency over complete welded 3D edges.

    This is 3D adjacency (shared welded vertex pairs), deliberately not UV
    adjacency, so geodesic growth crosses atlas seams correctly.
    """
    candidate = np.unique(np.asarray(candidate_ids, np.int64))
    adjacency: dict[int, set[int]] = {int(t): set() for t in candidate}
    edge_owner: dict[tuple[int, int], int] = {}
    for triangle_id in candidate.tolist():
        corners = [int(welded[v]) for v in triangles[triangle_id]]
        for a, b in ((corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])):
            edge = (a, b) if a <= b else (b, a)
            other = edge_owner.get(edge)
            if other is None:
                edge_owner[edge] = int(triangle_id)
            elif other != triangle_id:
                adjacency[int(triangle_id)].add(other)
                adjacency[other].add(int(triangle_id))
    return adjacency


def build_edge_costs(adjacency: dict[int, set[int]], centroids: np.ndarray,
                     face_normals_array: np.ndarray, projected_depths: np.ndarray,
                     chart_ids: np.ndarray | None = None,
                     classification: np.ndarray | None = None, *,
                     depth_penalty: float = 0.0,
                     normal_penalty: float = 0.0,
                     boundary_penalty: float = 0.0,
                     chart_penalty: float = 0.0) -> dict[tuple[int, int], float]:
    """Deterministic geodesic edge cost between adjacent triangles.

    Base weight is 3D centroid distance.  Optional penalties are additive and
    documented for the gate tests: depth discontinuity (absolute projected
    depth difference, LOCAL to the edge - never a global window), normal
    discontinuity (1 - dot), face-boundary band use and UV-chart transition.
    Costs are keyed by the ordered pair (min, max) of triangle ids.
    """
    edges: dict[tuple[int, int], float] = {}
    for triangle_id in sorted(adjacency):
        for neighbour in sorted(adjacency[triangle_id]):
            key = (triangle_id, neighbour) if triangle_id < neighbour else (neighbour, triangle_id)
            if key in edges:
                continue
            geodesic = float(np.linalg.norm(centroids[triangle_id] - centroids[neighbour]))
            cost = geodesic
            if depth_penalty:
                cost += float(depth_penalty) * abs(
                    float(projected_depths[triangle_id]) - float(projected_depths[neighbour]))
            if normal_penalty:
                gap = 1.0 - float(np.dot(face_normals_array[triangle_id], face_normals_array[neighbour]))
                cost += float(normal_penalty) * float(np.clip(gap, 0.0, 2.0))
            if boundary_penalty:
                class_a = 0 if classification is None else int(classification[triangle_id])
                class_b = 0 if classification is None else int(classification[neighbour])
                if class_a == 1 or class_b == 1:
                    cost += float(boundary_penalty)
            if chart_penalty and chart_ids is not None:
                if int(chart_ids[triangle_id]) != int(chart_ids[neighbour]):
                    cost += float(chart_penalty)
            edges[key] = float(cost)
    return edges


def grow_geodesic_surface(anchor_ids: list[int], adjacency: dict[int, set[int]],
                          edge_costs: dict[tuple[int, int], float],
                          target_count: int) -> set[int]:
    """Multi-source Dijkstra from the anchors over the welded mesh graph.

    Edge weight is 3D centroid distance.  The heap pops in non-decreasing
    distance with triangle-id tie-breaks, so the surface is deterministic:
    the ``target_count`` geodesically closest triangles reachable from the
    anchors.  OUTSIDE/infinite edges are never relaxed.
    """
    selected: set[int] = set()
    distances: dict[int, float] = {}
    heap: list[tuple[float, int]] = []
    for anchor in sorted(set(int(a) for a in anchor_ids)):
        if anchor not in selected:
            selected.add(anchor)
            distances[anchor] = 0.0
            heapq.heappush(heap, (0.0, anchor))
    while heap and len(selected) < int(target_count):
        distance, node = heapq.heappop(heap)
        if node not in selected:
            selected.add(node)
            distances[node] = distance
        for neighbour in adjacency[node]:
            if neighbour in selected:
                continue
            key = (node, neighbour) if node < neighbour else (neighbour, node)
            weight = edge_costs.get(key, np.inf)
            if not np.isfinite(weight):
                continue
            candidate = distance + weight
            if candidate < distances.get(neighbour, np.inf) - 1e-12:
                distances[neighbour] = candidate
                heapq.heappush(heap, (candidate, neighbour))
    return selected


def largest_connected_component(selected: set[int],
                                 adjacency: dict[int, set[int]]) -> set[int]:
    """Keep only the largest welded component of a triangle set."""
    if not selected:
        return set()
    remaining = set(selected)
    best: set[int] = set()
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            component.add(current)
            for neighbour in adjacency[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        if len(component) > len(best):
            best = component
    return best


def _dijkstra_bridge_path(sources: set[int], targets: set[int],
                          full_adjacency: dict[int, set[int]],
                          full_edge_costs: dict[tuple[int, int], float]) -> list[int] | None:
    if sources & targets:
        return [min(sources & targets)]
    distances: dict[int, float] = {}
    previous: dict[int, int] = {}
    heap: list[tuple[float, int]] = []
    for s in sorted(sources):
        distances[int(s)] = 0.0
        heapq.heappush(heap, (0.0, int(s)))
    visited: set[int] = set()
    target_set = set(int(t) for t in targets)
    best_target: int | None = None
    while heap:
        dist, node = heapq.heappop(heap)
        if node in visited:
            continue
        if dist > distances.get(node, float("inf")) + 1e-12:
            continue
        visited.add(node)
        if node in target_set:
            best_target = node
            break
        for neighbour in sorted(full_adjacency.get(node, ())):
            if neighbour in visited:
                continue
            key = (node, neighbour) if node < neighbour else (neighbour, node)
            w = full_edge_costs.get(key, float("inf"))
            if not np.isfinite(w):
                continue
            cand = float(dist + w)
            prev = distances.get(neighbour, float("inf"))
            if cand < prev - 1e-12:
                distances[neighbour] = cand
                previous[neighbour] = node
                heapq.heappush(heap, (cand, neighbour))
    if best_target is None:
        return None
    path = [best_target]
    cur = best_target
    while cur not in sources:
        cur = previous[cur]
        path.append(cur)
    path.reverse()
    return path


def anchor_aware_component(selected: set[int], adjacency: dict[int, set[int]],
                           anchor_ids: list[int]) -> set[int]:
    """Prefer the welded component containing anchors; fall back to largest."""
    if not selected:
        return set()
    anchors = set(int(a) for a in anchor_ids if int(a) in selected)
    if not anchors:
        return largest_connected_component(selected, adjacency)
    remaining = set(selected)
    components: list[set[int]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        comp: set[int] = set()
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nb in adjacency[cur]:
                if nb in remaining:
                    remaining.remove(nb)
                    stack.append(nb)
        components.append(comp)
    anchored = [c for c in components if anchors & c]
    if not anchored:
        return max(components, key=len)
    containing = [c for c in anchored if len(anchors & c) == len(anchors)]
    if containing:
        return max(containing, key=len)
    return max(anchored, key=lambda c: (len(anchors & c), len(c)))


def connect_anchors_into_core(core: set[int], adjacency: dict[int, set[int]],
                               edge_costs: dict[tuple[int, int], float],
                               anchor_ids: list[int]) -> set[int]:
    """Bounded Steiner augmentation: cheapest welded paths between anchor islands."""
    if not core or not anchor_ids:
        return set(core)
    missing = [int(a) for a in anchor_ids if int(a) not in core and int(a) in adjacency]
    if not missing:
        return set(core)
    result = set(core)
    for target in sorted(set(missing)):
        distances: dict[int, float] = {}
        prev: dict[int, int | None] = {}
        heap: list[tuple[float, int]] = []
        for seed in sorted(result):
            distances[seed] = 0.0
            prev[seed] = None
            heapq.heappush(heap, (0.0, seed))
        visited: set[int] = set()
        found = False
        while heap:
            dist, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node == target:
                found = True
                break
            for nb in sorted(adjacency[node]):
                key = (node, nb) if node < nb else (nb, node)
                w = edge_costs.get(key)
                if w is None or not np.isfinite(w):
                    continue
                cand = dist + float(w)
                if cand < distances.get(nb, np.inf) - 1e-12:
                    distances[nb] = cand
                    prev[nb] = node
                    heapq.heappush(heap, (cand, nb))
        if not found or target not in prev and target not in distances:
            continue
        cur: int | None = target
        while cur is not None:
            result.add(cur)
            cur = prev.get(cur)
    return result


def bounded_fill_to_floor(core: set[int], adjacency: dict[int, set[int]],
                          min_triangles: int, max_triangles: int) -> set[int]:
    """BFS expansion by increasing geodesic distance to reach min_triangles if needed."""
    if len(core) >= int(min_triangles):
        return set(core)
    if len(core) > int(max_triangles):
        return set(core)
    frontier: list[int] = sorted(core)
    visited = set(core)
    queue = list(frontier)
    idx = 0
    order: list[int] = []
    seen_queue = set(queue)
    while idx < len(queue) and len(visited) + len(order) < int(min_triangles):
        cur = queue[idx]
        idx += 1
        for nb in sorted(adjacency[cur]):
            if nb in visited or nb in seen_queue:
                continue
            seen_queue.add(nb)
            queue.append(nb)
            order.append(nb)
            if len(visited) + len(order) >= int(min_triangles):
                break
    needed = int(min_triangles) - len(core)
    result = set(core)
    for tid in order[:max(0, needed)]:
        if len(result) >= int(max_triangles):
            break
        result.add(tid)
    return result


def clamp_triangle_count(selected: set[int], *, min_triangles: int,
                         max_triangles: int) -> set[int]:
    count = len(selected)
    if count < int(min_triangles):
        raise RuntimeError(
            f"FACE_SURFACE_UNDERSIZED:{count}<{min_triangles}")
    if count > int(max_triangles):
        # Trimming only the highest-cost leaves keeps the surface connected
        # when the grow target overran the clamp; not exercised on real data.
        raise RuntimeError(
            f"FACE_SURFACE_OVERSIZED:{count}>{max_triangles}")
    return selected


def fill_enclosed_interior(core: set[int], adjacency: dict[int, set[int]],
                           classification: np.ndarray, *, max_additions: int = 0) -> set[int]:
    """Add enclosed STRICT_INTERIOR triangles surrounded by selected neighbours.

    A triangle is added only when it is STRICT_INTERIOR and at least two of its
    in-domain welded neighbours are already selected, so growth fills enclosed
    interior holes rather than dilating freely.  ``max_additions`` bounds the
    amount of filling; ``0`` disables filling.
    """
    if max_additions <= 0:
        return set(core)
    result = set(core)
    core_size = len(core)
    changed = True
    while changed and len(result) - core_size < max_additions:
        changed = False
        candidates = []
        for triangle_id in sorted(adjacency):
            if triangle_id in result or int(classification[triangle_id]) != 0:
                continue
            neighbours = adjacency[triangle_id]
            in_domain = [n for n in neighbours if n in adjacency]
            selected_neighbours = sum(1 for n in in_domain if n in result)
            if selected_neighbours >= 2:
                candidates.append((selected_neighbours, triangle_id))
        if not candidates:
            break
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for _count, triangle_id in candidates:
            if len(result) - core_size >= max_additions:
                break
            if triangle_id not in result:
                result.add(triangle_id)
                changed = True
    return result


def writable_texel_mask(uv: np.ndarray, triangles: np.ndarray, selected_ids: np.ndarray,
                        size: int) -> np.ndarray:
    """Boolean atlas mask of texels owned by a selected triangle."""
    owner, _weights = rasterise(uv, triangles, size)
    selected = np.zeros(len(triangles), dtype=bool)
    ids = np.unique(np.asarray(selected_ids, np.int64))
    if ids.size:
        selected[ids] = True
    return (owner >= 0) & selected[np.maximum(owner, 0)]


def score_map(selected_ids: np.ndarray, adjacency: dict[int, set[int]],
              edge_costs: dict[tuple[int, int], float]) -> np.ndarray:
    """Per-triangle score array: mean incident geodesic edge cost."""
    scores: dict[int, float] = {}
    for triangle_id in np.asarray(selected_ids, np.int64):
        total = 0.0
        count = 0
        for neighbour in adjacency[int(triangle_id)]:
            key = (int(triangle_id), neighbour) if int(triangle_id) < neighbour else (neighbour, int(triangle_id))
            weight = edge_costs.get(key, np.inf)
            if np.isfinite(weight):
                total += float(weight)
                count += 1
        scores[int(triangle_id)] = total / max(count, 1)
    return scores


def build_face_surface_candidate(
    baseline_glb: Path,
    baseline_atlas: Path,
    source_image: Path,
    source_fixture: Path,
    output_dir: Path,
    *,
    ray_stride: int = 3,
    max_rank: int = 6,
    minimum_facing: float = 0.05,
    grow_target: int = 350,
    min_triangles: int = 200,
    max_triangles: int = 400,
    neighbour_rays: int = 96,
    seed: int = 20260806,
    minimum_alpha: float = 0.30,
    tps_regularization: float = 1e-3,
    build_textured: bool = False,
) -> dict[str, Any]:
    """Build one minimal geodesic face-surface candidate (no depth window)."""
    np.random.seed(int(seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    positions, normals, uv, triangles = read_glb(baseline_glb)
    if uv is None:
        raise RuntimeError("FACE_CANDIDATE_REQUIRES_UV")
    source_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    atlas_bgr = cv2.imread(str(baseline_atlas), cv2.IMREAD_COLOR)
    if source_bgr is None or atlas_bgr is None:
        raise RuntimeError("FACE_CANDIDATE_IMAGE_MISSING")
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    baseline_rgb = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGB)
    fixture = load_source_fixture(source_fixture, source_image, source_rgb.shape[:2])
    landmarks = fixture["landmarks"]
    polygon = np.asarray(fixture["face_polygon"], np.float64)
    shape = (int(source_rgb.shape[0]), int(source_rgb.shape[1]))
    foreground, source_alpha, foreground_report = border_connected_foreground(source_rgb)
    face_mask = fixture_face_mask(shape, fixture)
    cv2.imwrite(str(output_dir / "foreground_mask.png"), foreground.astype(np.uint8) * 255)
    cv2.imwrite(str(output_dir / "face_mask.png"), face_mask.astype(np.uint8) * 255)
    (output_dir / "source_fixture_used.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")

    camera_fit = fit_camera(np.asarray(positions, np.float64), np.asarray(triangles, np.int64), foreground)
    camera = camera_from_projection_fit(camera_fit, positions, shape[1], shape[0])
    (output_dir / "camera_contract.json").write_text(
        json.dumps(camera_to_dict(camera, camera_fit), indent=2), encoding="utf-8")

    layers = trace_mask_layers(positions, normals, triangles, camera, face_mask,
                               stride=ray_stride, max_hits=24, leaf_size=16)
    np.savez_compressed(output_dir / "all_ray_hits.npz", **layers)
    candidate_ids = candidate_ids_from_layers(layers, max_rank=max_rank,
                                              minimum_facing=minimum_facing)

    # ---- anchors: front-most front-facing hit per landmark -----------------
    anchors = []
    anchor_triangle_ids: list[int] = []
    for landmark in landmarks:
        row = anchor_from_layers(layers, landmark, minimum_facing=minimum_facing,
                                 neighbour_rays=neighbour_rays)
        anchors.append(row)
        anchor_triangle_ids.append(int(row["triangle_id"]))
    (output_dir / "auto_target_anchors.json").write_text(json.dumps(anchors, indent=2), encoding="utf-8")

    # ---- welded 3D mesh graph over the candidate domain --------------------
    welded = weld_mesh_vertices(positions)
    adjacency = weld_face_adjacency(triangles, welded, candidate_ids)
    centroids = positions[triangles].mean(axis=1)
    face_normals_array = face_normals(positions, triangles)
    projected_centroid, projected_depth = camera.project(centroids)
    edge_costs = build_edge_costs(adjacency, centroids, face_normals_array, projected_depth)

    if len(candidate_ids) and len(anchor_triangle_ids):
        comp_of: dict[int, int] = {}
        visited_comp: set[int] = set()
        components: list[set[int]] = []
        for seed in sorted(adjacency):
            if seed in visited_comp:
                continue
            stack = [seed]
            visited_comp.add(seed)
            comp: set[int] = set()
            while stack:
                cur = stack.pop()
                comp.add(cur)
                for nb in sorted(adjacency[cur]):
                    if nb not in visited_comp:
                        visited_comp.add(nb)
                        stack.append(nb)
            for tid in comp:
                comp_of[tid] = len(components)
            components.append(comp)
        anchor_comp_ids = {comp_of.get(int(a)) for a in anchor_triangle_ids if int(a) in comp_of}
        anchor_comp_ids.discard(None)
        if len(anchor_comp_ids) > 1:
            anchor_components = [components[cid] for cid in sorted(anchor_comp_ids)]  # type: ignore[arg-type]
            anchor_components = sorted(anchor_components, key=lambda s: min(s))
            full_adjacency = weld_face_adjacency(triangles, welded, np.arange(len(triangles), dtype=np.int64))
            full_edge_costs = build_edge_costs(full_adjacency, centroids, face_normals_array, projected_depth)
            super_nodes: set[int] = set(anchor_components[0])
            remaining_comps: list[set[int]] = anchor_components[1:]
            bridge_nodes: set[int] = set()
            while remaining_comps:
                best_path: list[int] | None = None
                best_tuple: tuple | None = None
                best_target: set[int] | None = None
                for comp in sorted(remaining_comps, key=lambda s: min(s)):
                    path = _dijkstra_bridge_path(super_nodes, comp, full_adjacency, full_edge_costs)
                    if path is None:
                        continue
                    cost = 0.0
                    for u, v in zip(path, path[1:]):
                        key = (u, v) if u < v else (v, u)
                        cost += float(full_edge_costs.get(key, float("inf")))
                    tup = (float(cost), len(path), tuple(path))
                    if best_path is None or tup < best_tuple:  # type: ignore[operator]
                        best_path = path
                        best_tuple = tup
                        best_target = comp
                if best_path is None or best_target is None:
                    break
                bridge_nodes.update(best_path)
                super_nodes.update(best_path)
                super_nodes.update(best_target)
                remaining_comps = [c for c in remaining_comps if c is not best_target]
            if bridge_nodes:
                augmented_ids = np.unique(np.concatenate([np.asarray(candidate_ids, dtype=np.int64), np.asarray(sorted(bridge_nodes), dtype=np.int64)]))
                adjacency = weld_face_adjacency(triangles, welded, augmented_ids)
                edge_costs = build_edge_costs(adjacency, centroids, face_normals_array, projected_depth)

    # ---- geodesic grow from the anchors ------------------------------------
    grown = grow_geodesic_surface(anchor_triangle_ids, adjacency, edge_costs,
                                   target_count=grow_target)
    core = anchor_aware_component(grown, adjacency, anchor_triangle_ids)
    core = connect_anchors_into_core(core, adjacency, edge_costs, anchor_triangle_ids)
    if len(core) < int(min_triangles):
        core = bounded_fill_to_floor(core, adjacency, min_triangles, max_triangles)
    selected = clamp_triangle_count(core, min_triangles=min_triangles, max_triangles=max_triangles)
    selected_ids = np.asarray(sorted(selected), np.int64)
    np.save(output_dir / "selected_face_triangles.npy", selected_ids)
    np.save(output_dir / "anchor_geodesic_core.npy", np.asarray(sorted(core), np.int64))

    # ---- atlas writable mask ------------------------------------------------
    texture_selected_ids = np.asarray(
        sorted(set(selected_ids.tolist()) & set(candidate_ids.tolist())), np.int64
    )
    writable = writable_texel_mask(uv, triangles, texture_selected_ids, baseline_rgb.shape[0])
    np.save(output_dir / "writable_texel_mask.npy", writable)
    cv2.imwrite(str(output_dir / "writable_texel_mask.png"), writable.astype(np.uint8) * 255)

    scores = score_map(selected_ids, adjacency, edge_costs)
    (output_dir / "patch_scores.json").write_text(json.dumps({
        "schema": "face_v3_patch_scores_v1",
        "per_triangle_mean_geodesic_edge_cost": {
            int(triangle_id): float(scores[triangle_id])
            for triangle_id in selected_ids.tolist()
        },
    }, indent=2), encoding="utf-8")
    np.save(output_dir / "triangle_geodesic_scores.npy",
            np.asarray([scores.get(int(t), np.inf) for t in range(len(triangles))], np.float64))

    # ---- gate checks ---------------------------------------------------------
    selected_set = set(selected_ids.tolist())
    outside_hit_rank = ranks = hit_ranks(layers["offsets"])
    _ = outside_hit_rank  # ranks used only for the report below
    anchors_in_candidate = all(int(anchor["triangle_id"]) in candidate_ids.tolist() for anchor in anchors)
    anchor_domain = all(anchor["triangle_id"] in selected_set for anchor in anchors)
    connected_components = 0
    remaining = set(selected_set)
    while remaining:
        connected_components += 1
        stack = [min(remaining)]
        remaining.remove(min(remaining))
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
    rear_visible = 0
    for triangle_id in selected_ids.tolist():
        if float(face_normals_array[triangle_id] @ (-camera.forward)) < -0.2:
            rear_visible += 1
    candidate_membership = np.isin(selected_ids, candidate_ids)
    bridge_triangle_count = int((~candidate_membership).sum())
    bridge_rear_visible = int(sum(
        (not bool(candidate_membership[index])) and float(face_normals_array[triangle_id] @ (-camera.forward)) < -0.2
        for index, triangle_id in enumerate(selected_ids.tolist())
    ))

    hard_gate_failures: list[str] = []
    if connected_components != 1:
        hard_gate_failures.append(f"CONNECTED_COMPONENTS:{connected_components}")
    if not anchor_domain:
        hard_gate_failures.append("ANCHORS_NOT_IN_SELECTED")
    if not anchors_in_candidate:
        hard_gate_failures.append("ANCHORS_NOT_IN_CANDIDATE_DOMAIN")
    if not (int(min_triangles) <= len(selected_ids) <= int(max_triangles)):
        hard_gate_failures.append(f"TRIANGLE_COUNT:{len(selected_ids)}")

    selection = {
        "schema": "face_v3_selection_decision_v1",
        "classification": "CANDIDATE_REQUIRES_VISUAL_REVIEW",
        "selected_triangle_count": int(len(selected_ids)),
        "writable_texel_count": int(writable.sum()),
        "texture_selected_triangle_count": int(len(texture_selected_ids)),
        "anchor_count": int(len(anchors)),
        "anchor_triangle_ids": {anchor["name"]: anchor["triangle_id"] for anchor in anchors},
        "connected_components": connected_components,
        "outside_face_selected": bridge_triangle_count,
        "bridge_triangle_count": bridge_triangle_count,
        "rear_visible_selected": rear_visible,
        "rear_visible_bridge_triangles": bridge_rear_visible,
        "anchors_in_candidate_domain": bool(anchors_in_candidate),
        "anchors_in_selected": bool(anchor_domain),
        "geodesic_core_triangles": int(len(core)),
        "grow_target": int(grow_target),
        "max_rank": int(max_rank),
        "minimum_facing": float(minimum_facing),
        "ray_stride": int(ray_stride),
        "no_depth_window": True,
        "vs_v2_selected_triangles": 1341,
        "vs_v2_writable_texels": 5514,
        "smaller_than_v2": bool(len(selected_ids) < 1341 and int(writable.sum()) < 5514),
        "hard_gate_failures": hard_gate_failures,
    }
    if hard_gate_failures:
        selection["classification"] = "REJECTED_HARD_GATE"
    (output_dir / "selection_decision.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    if hard_gate_failures:
        raise RuntimeError("FACE_SURFACE_HARD_GATE:" + ",".join(hard_gate_failures))

    report = {
        "schema": "face_surface_candidate_v3",
        "classification": "CANDIDATE_REQUIRES_VISUAL_REVIEW",
        "baseline_glb": str(baseline_glb),
        "baseline_glb_sha256": sha256(baseline_glb),
        "baseline_atlas": str(baseline_atlas),
        "baseline_atlas_sha256": sha256(baseline_atlas),
        "source_image": str(source_image),
        "source_image_sha256": sha256(source_image),
        "source_fixture": str(source_fixture),
        "foreground": foreground_report,
        "camera": camera_to_dict(camera, camera_fit),
        "candidate_domain": {
            "candidate_triangle_count": int(len(candidate_ids)),
            "ray_count": int(len(layers["pixels_xy"])),
            "hit_count": int(len(layers["triangle_ids"])),
            "max_rank": int(max_rank),
            "minimum_facing": float(minimum_facing),
        },
        "selection": selection,
        "anchors": anchors,
        "writable_texel_count": int(writable.sum()),
        "selected_triangle_count": int(len(selected_ids)),
        "seed": int(seed),
        "promotion_authorized": False,
    }

    if build_textured:
        target_points, source_points = [], []
        for anchor in anchors:
            bary = np.asarray(anchor["barycentric"], np.float64)
            point_3d = bary @ positions[triangles[int(anchor["triangle_id"])]]
            target_xy, _ = camera.project(point_3d[None, :])
            target_points.append(target_xy[0])
            source_points.append(anchor["source_xy"])
        candidate_atlas, texture_report, texture_writable = build_face_patch_atlas(
            baseline_rgb, source_rgb, source_alpha, positions, uv, triangles,
            texture_selected_ids, camera, np.asarray(target_points), np.asarray(source_points),
            minimum_alpha=minimum_alpha, tps_regularization=tps_regularization,
        )
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(candidate_atlas, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("FACE_CANDIDATE_ATLAS_ENCODE_FAILED")
        atlas_path = output_dir / "atlas_face_surface_owned_2048.png"
        glb_path = output_dir / "panda_face_surface_owned_2048.glb"
        atlas_path.write_bytes(encoded.tobytes())
        before_hashes = immutable_buffer_hashes(baseline_glb)
        bind_texture(baseline_glb, glb_path, encoded.tobytes(),
                     textured_triangles=np.ones(len(triangles), bool))
        after_hashes = immutable_buffer_hashes(glb_path)
        if before_hashes != after_hashes:
            raise RuntimeError("FACE_CANDIDATE_GEOMETRY_UV_INDEX_CHANGED")
        non_face_changed = int(np.any(
            candidate_atlas[~texture_writable] != baseline_rgb[~texture_writable], axis=1).sum())
        if non_face_changed:
            raise RuntimeError("FACE_CANDIDATE_NON_FACE_CHANGED")
        report.update({
            "texture": texture_report,
            "texture_selected_triangle_count": int(len(texture_selected_ids)),
            "non_face_atlas_pixels_changed": non_face_changed,
            "immutable_hashes_before": before_hashes,
            "immutable_hashes_after": after_hashes,
            "output_atlas": str(atlas_path),
            "output_atlas_sha256": sha256(atlas_path),
            "output_glb": str(glb_path),
            "output_glb_sha256": sha256(glb_path),
        })

    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-glb", type=Path, required=True)
    parser.add_argument("--baseline-atlas", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--source-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ray-stride", type=int, default=3)
    parser.add_argument("--max-rank", type=int, default=6)
    parser.add_argument("--minimum-facing", type=float, default=0.05)
    parser.add_argument("--grow-target", type=int, default=350)
    parser.add_argument("--min-triangles", type=int, default=200)
    parser.add_argument("--max-triangles", type=int, default=400)
    parser.add_argument("--neighbour-rays", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--minimum-alpha", type=float, default=0.30)
    parser.add_argument("--build-textured", action="store_true")
    args = parser.parse_args()
    report = build_face_surface_candidate(
        args.baseline_glb, args.baseline_atlas, args.source_image, args.source_fixture,
        args.output_dir, ray_stride=args.ray_stride, max_rank=args.max_rank,
        minimum_facing=args.minimum_facing, grow_target=args.grow_target,
        min_triangles=args.min_triangles, max_triangles=args.max_triangles,
        neighbour_rays=args.neighbour_rays, seed=args.seed,
        minimum_alpha=args.minimum_alpha, build_textured=args.build_textured,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
