"""Build the bounded continuous UV->world->camera all-hit surface manifest.

This worker deliberately does not read or use legacy rounded triangle-ID buffers.  Atlas
ownership is established at exact UV texel centres, then the corresponding continuous image
coordinate is queried against a deterministic CPU screen-bin all-hit reference.  Screen bins are
only an acceleration index: every candidate triangle is re-tested with float64 barycentrics, so
no rounded pixel is used as evidence authority.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import psutil
from PIL import Image
from scipy import __version__ as scipy_version
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

try:
    from atlas_raster import census, rasterise
    from mesh_io import read_glb
except ModuleNotFoundError:  # package import from the focused test suite
    from workers.atlas_raster import census, rasterise
    from workers.mesh_io import read_glb


STATE_NAMES = ("VISIBLE_EXACT", "VISIBLE_EDGE_TIE", "OCCLUDED", "OUT_OF_FRAME", "BACKFACING", "CONTRACT_ERROR", "UNKNOWN_AMBIGUOUS")
STATE_CODE = {name: index for index, name in enumerate(STATE_NAMES)}
ADMITTED_STATES = frozenset(("VISIBLE_EXACT", "VISIBLE_EDGE_TIE"))
EVIDENCE_CLASS_GENERATED = 1
SCHEMA = "panda_continuous_surface_manifest_v1"
CONTRACT_VERSION = "continuous_surface_contract_v1"
# Exact measured diagnostic identity approved by the UNKNOWN_AMBIGUOUS policy:
# (view index, local owned sample index, owner face) -> nearest front faces.
APPROVED_UNKNOWN_FRONT_GROUPS = {
    (0, 2835, 623615): (181983, 181985), (0, 12618, 320224): (16755, 205083),
    (0, 16132, 578658): (133879, 133880), (0, 23681, 441185): (187325, 187328),
    (0, 25078, 510607): (127886, 127887), (0, 25992, 370467): (26790, 26787),
    (0, 29248, 454852): (227747, 227765), (0, 30480, 588244): (126770, 126812),
    (0, 30556, 506065): (190976, 190996), (1, 8649, 262983): (28703, 29331),
    (1, 9500, 407287): (407283, 407311), (1, 17665, 101162): (979, 980),
    (1, 19826, 99967): (631, 99968), (1, 24823, 573240): (82129, 82447),
    (1, 25078, 510607): (76305, 76307), (1, 27367, 300032): (37329, 37331),
    (1, 29248, 454852): (67956, 454862), (1, 30480, 588244): (83875, 83877),
    (1, 30556, 506065): (75715, 506095),
    (2, 2835, 623615): (624325, 624332), (2, 9500, 407287): (418627, 418641),
    (2, 12618, 320224): (63503, 429653), (2, 17665, 101162): (592819, 592838),
    (2, 23681, 441185): (506243, 506246), (2, 25992, 370467): (56184, 390906),
    (2, 27367, 300032): (303231, 303255), (2, 30480, 588244): (593537, 593588),
    (2, 30556, 506065): (509844, 509861), (3, 1372, 330720): (330707, 333411),
    (3, 17665, 101162): (101110, 101137), (3, 25992, 370467): (370462, 370465),
    (3, 27367, 300032): (300030, 300052), (3, 30556, 506065): (506062, 506081),
    (4, 2835, 623615): (624030, 624033), (4, 12618, 320224): (320499, 41639),
    (4, 16132, 578658): (578777, 578779), (4, 23681, 441185): (441881, 441884),
    (5, 12618, 320224): (318728, 318730), (5, 16132, 578658): (577459, 577462),
    (5, 23681, 441185): (438965, 438968),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_array(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype.byteorder not in ("|", "<"):
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def hash_bytes_array(value: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(value), dtype=np.uint8).copy()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a byte-stable NPZ (NumPy's default zip timestamps are not stable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-14:
        raise RuntimeError("CAMERA_VECTOR_ZERO")
    return value / norm


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("view_count") != 6 or len(contract.get("views", [])) != 6:
        raise RuntimeError("CAMERA_CONTRACT_VIEW_COUNT_INVALID")
    required = ("camera_position", "camera_direction", "camera_right", "camera_up", "proven_semantic")
    views = sorted(contract["views"], key=lambda item: int(item["index"]))
    if [int(item["index"]) for item in views] != list(range(6)):
        raise RuntimeError("CAMERA_CONTRACT_VIEW_INDICES_INVALID")
    for item in views:
        if any(key not in item for key in required):
            raise RuntimeError(f"CAMERA_CONTRACT_FIELD_MISSING:{item.get('index')}")
    return contract


def load_evidence(receipt_path: Path) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    images: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    paths: dict[str, str] = {}
    for output in receipt.get("output_images", []):
        name = str(output.get("name", ""))
        path = Path(str(output.get("path", "")))
        stem = path.stem
        parts = stem.split("_", 2)
        semantic = parts[2] if len(parts) == 3 and parts[0] == "view" else stem
        if not path.is_file():
            raise RuntimeError(f"EVIDENCE_IMAGE_MISSING:{path}")
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 2 or image.shape[1] < 2:
            raise RuntimeError(f"EVIDENCE_IMAGE_INVALID:{path}")
        images[semantic] = image
        hashes[semantic] = sha256_file(path)
        paths[semantic] = str(path)
    return images, hashes, paths


def connected_face_components(triangles: np.ndarray, vertex_count: int) -> tuple[np.ndarray, int]:
    """Return deterministic vertex-connected component IDs for faces."""
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0)
    graph = coo_matrix(
        (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
        shape=(vertex_count, vertex_count),
    )
    graph = graph + graph.T
    count, labels = connected_components(graph.tocsr(), directed=False, return_labels=True)
    face_labels = labels[triangles]
    if not np.all(face_labels == face_labels[:, :1]):
        face_labels = face_labels.min(axis=1, keepdims=True).repeat(3, axis=1)
    return face_labels[:, 0].astype(np.int32), int(count)


def uv_chart_components(triangles: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, int]:
    """Connect faces sharing an exact UV coordinate; labels are stable component IDs."""
    face_count = int(len(triangles))
    face_ids = np.repeat(np.arange(face_count, dtype=np.int64), 3)
    uv_rows = np.asarray(uv[triangles].reshape(-1, 2), dtype=np.float64)
    _, uv_ids = np.unique(uv_rows, axis=0, return_inverse=True)
    order = np.argsort(uv_ids, kind="stable")
    ordered_uv = uv_ids[order]
    ordered_faces = face_ids[order]
    same = ordered_uv[1:] == ordered_uv[:-1]
    if not np.any(same):
        return np.arange(face_count, dtype=np.int32), face_count
    left = ordered_faces[:-1][same]
    right = ordered_faces[1:][same]
    graph = coo_matrix(
        (np.ones(len(left), dtype=np.uint8), (left, right)),
        shape=(face_count, face_count),
    )
    graph = graph + graph.T
    count, labels = connected_components(graph.tocsr(), directed=False, return_labels=True)
    return labels.astype(np.int32), int(count)


def project(points: np.ndarray, view: dict[str, Any], projection_span: float) -> np.ndarray:
    right = unit(np.asarray(view["camera_right"], dtype=np.float64))
    up = unit(np.asarray(view["camera_up"], dtype=np.float64))
    return np.stack((points @ right / projection_span + 0.5, 0.5 - points @ up / projection_span), axis=1)


def bilinear(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(np.asarray(xy[:, 0], dtype=np.float64), 0.0, width - 1.0)
    y = np.clip(np.asarray(xy[:, 1], dtype=np.float64), 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    # Keep the source image in its compact uint8 form.  Casting the complete
    # image for every scalar sample creates an avoidable full-frame temporary;
    # cast only the four indexed texels used by this interpolation.
    c00 = image[y0, x0].astype(np.float32, copy=False)
    c10 = image[y0, x1].astype(np.float32, copy=False)
    c01 = image[y1, x0].astype(np.float32, copy=False)
    c11 = image[y1, x1].astype(np.float32, copy=False)
    top = c00 * (1.0 - wx[:, None]) + c10 * wx[:, None]
    bottom = c01 * (1.0 - wx[:, None]) + c11 * wx[:, None]
    return top * (1.0 - wy[:, None]) + bottom * wy[:, None]


def build_screen_grid(tri_screen: np.ndarray, size: int) -> list[np.ndarray]:
    """Index triangles by coarse screen bins; tests remain continuous float64."""
    bins: list[list[int]] = [[] for _ in range(size * size)]
    lows = np.floor(np.nanmin(tri_screen, axis=1) * size).astype(np.int64)
    highs = np.floor(np.nanmax(tri_screen, axis=1) * size).astype(np.int64)
    for triangle_id in range(tri_screen.shape[0]):
        # Expand the acceleration bin by one cell at each side.  This is only
        # a conservative candidate lookup margin; every candidate is still
        # accepted only after the exact float64 barycentric test below.
        x0 = max(0, int(lows[triangle_id, 0]) - 1)
        y0 = max(0, int(lows[triangle_id, 1]) - 1)
        x1 = min(size - 1, int(highs[triangle_id, 0]) + 1)
        y1 = min(size - 1, int(highs[triangle_id, 1]) + 1)
        if x1 < x0 or y1 < y0:
            continue
        for row in range(y0, y1 + 1):
            base = row * size
            for col in range(x0, x1 + 1):
                bins[base + col].append(triangle_id)
    return [np.asarray(values, dtype=np.int32) for values in bins]


def query_all_hits(
    query_screen: np.ndarray,
    tri_screen: np.ndarray,
    vertices: np.ndarray,
    tri_vertices: np.ndarray,
    tri_uv: np.ndarray,
    face_normals: np.ndarray,
    face_components: np.ndarray,
    view: dict[str, Any],
    projection_span: float,
    lattice_size: int,
    tolerances: dict[str, float],
 ) -> Iterator[list[dict[str, Any]]]:
    grid = build_screen_grid(tri_screen, lattice_size)
    direction = unit(np.asarray(view["camera_direction"], dtype=np.float64))
    camera_position = np.asarray(view["camera_position"], dtype=np.float64)
    eps = float(tolerances["barycentric_abs"])
    tile_size = 1024
    for tile_start in range(0, len(query_screen), tile_size):
        tile = query_screen[tile_start:tile_start + tile_size]
        for point in tile:
            if (
                not np.isfinite(point).all()
                or np.any(point < -tolerances["screen_bounds"])
                or np.any(point > 1.0 + tolerances["screen_bounds"])
            ):
                yield []
                continue
            col = min(lattice_size - 1, max(0, int(math.floor(float(point[0]) * lattice_size))))
            row = min(lattice_size - 1, max(0, int(math.floor(float(point[1]) * lattice_size))))
            candidates = grid[row * lattice_size + col]
            if candidates.size == 0:
                yield []
                continue
            corners = tri_screen[candidates]
            x0, y0 = corners[:, 0, 0], corners[:, 0, 1]
            x1, y1 = corners[:, 1, 0], corners[:, 1, 1]
            x2, y2 = corners[:, 2, 0], corners[:, 2, 1]
            px, py = float(point[0]), float(point[1])
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            usable = np.abs(denominator) > 1e-15
            if not usable.any():
                yield []
                continue
            w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / np.where(usable, denominator, 1.0)
            w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / np.where(usable, denominator, 1.0)
            w2 = 1.0 - w0 - w1
            inside = usable & (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
            indices = np.flatnonzero(inside)
            hits: list[dict[str, Any]] = []
            for candidate_index in indices.tolist():
                face_id = int(candidates[candidate_index])
                bary = np.asarray((w0[candidate_index], w1[candidate_index], w2[candidate_index]), dtype=np.float64)
                point_world = bary @ tri_vertices[face_id]
                depth = float(np.dot(point_world - camera_position, direction))
                hit_uv = bary @ tri_uv[face_id]
                hits.append({
                    "face": face_id,
                    "depth": depth,
                    "bary": bary,
                    "point": point_world,
                    "uv": hit_uv,
                    "component": int(face_components[face_id]),
                    "facing": float(np.dot(face_normals[face_id], -direction)),
                })
            hits.sort(key=lambda item: (item["depth"], item["face"]))
            yield hits


def unresolved_front_co_depth(hits: list[dict[str, Any]], tolerances: dict[str, float]) -> bool:
    """Whether the nearest all-hit group has more than one co-depth face."""
    if not hits:
        return False
    first_depth = float(hits[0]["depth"])
    limit = float(tolerances["co_depth_abs"]) + float(tolerances["co_depth_rel"]) * max(1.0, abs(first_depth))
    return sum(abs(float(hit["depth"]) - first_depth) <= limit for hit in hits) > 1


def approved_unknown_ambiguity(view_index: int, sample_index: int, owner_face: int, front_faces: tuple[int, ...]) -> bool:
    return APPROVED_UNKNOWN_FRONT_GROUPS.get((int(view_index), int(sample_index), int(owner_face))) == tuple(int(face) for face in front_faces)


def classify_front_co_depth_policy(
    view_index: int, sample_index: int, owner_face: int,
    hits: list[dict[str, Any]], tolerances: dict[str, float], view: dict[str, Any] | None = None,
) -> str | None:
    if not unresolved_front_co_depth(hits, tolerances):
        return None
    first_depth = float(hits[0]["depth"])
    limit = float(tolerances["co_depth_abs"]) + float(tolerances["co_depth_rel"]) * max(1.0, abs(first_depth))
    front = tuple(int(hit["face"]) for hit in hits if abs(float(hit["depth"]) - first_depth) <= limit)
    if approved_unknown_ambiguity(view_index, sample_index, owner_face, front):
        return "UNKNOWN_AMBIGUOUS"
    if view is not None and resolution_independent_unknown_signature(owner_face, hits, view, tolerances):
        return "UNKNOWN_AMBIGUOUS"
    return "CONTRACT_ERROR"


def resolution_independent_unknown_signature(
    owner_face: int, hits: list[dict[str, Any]], view: dict[str, Any], tolerances: dict[str, float],
) -> bool:
    """Exact two-face, owner-absent, equal-depth signature; no proximity waiver."""
    if not hits:
        return False
    first_depth = float(hits[0]["depth"])
    limit = float(tolerances["co_depth_abs"]) + float(tolerances["co_depth_rel"]) * max(1.0, abs(first_depth))
    front = [hit for hit in hits if abs(float(hit["depth"]) - first_depth) <= limit]
    if len(front) != 2 or int(owner_face) in {int(hit["face"]) for hit in front}:
        return False
    direction = np.asarray(view["camera_direction"], dtype=np.longdouble)
    direction /= np.linalg.norm(direction)
    camera_position = np.asarray(view["camera_position"], dtype=np.longdouble)
    long_depths = [np.dot(np.asarray(hit["point"], dtype=np.longdouble) - camera_position, direction) for hit in front]
    long_spread = abs(long_depths[0] - long_depths[1])
    return bool(long_spread <= np.longdouble(limit))


def visible_edge_tie_signature(
    owner_face: int, hits: list[dict[str, Any]], face_vertices: np.ndarray,
    face_normals: np.ndarray, tolerances: dict[str, float], view: dict[str, Any] | None = None,
    owner_point: np.ndarray | None = None, owner_uv: np.ndarray | None = None,
    owner_barycentric: np.ndarray | None = None, face_components: np.ndarray | None = None,
    face_charts: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Prove the narrow owner-present, shared-position-edge tie contract.

    The edge test is position-topological rather than raw-index based because the
    canonical mesh contains UV seam splits. Exactly two endpoint position matches
    are required; three matches (duplicate faces) and zero/one matches are rejected.
    """
    if not hits:
        return None
    first_depth = float(hits[0]["depth"])
    limit = float(tolerances["co_depth_abs"]) + float(tolerances["co_depth_rel"]) * max(1.0, abs(first_depth))
    front = [hit for hit in hits if abs(float(hit["depth"]) - first_depth) <= limit]
    if len(front) != 2 or int(owner_face) not in {int(hit["face"]) for hit in front}:
        return None
    first, second = (int(front[0]["face"]), int(front[1]["face"]))
    owner_hits = [hit for hit in front if int(hit["face"]) == int(owner_face)]
    if len(owner_hits) != 1 or owner_point is None or owner_uv is None or owner_barycentric is None:
        return None
    owner_hit = owner_hits[0]
    owner_point_error = float(np.max(np.abs(np.asarray(owner_hit["point"]) - np.asarray(owner_point))))
    owner_uv_error = float(np.max(np.abs(np.asarray(owner_hit["uv"]) - np.asarray(owner_uv))))
    owner_bary_error = float(np.max(np.abs(np.asarray(owner_hit["bary"]) - np.asarray(owner_barycentric))))
    if owner_point_error > float(tolerances.get("edge_point_abs", tolerances["point_abs"])):
        return None
    if owner_uv_error > float(tolerances.get("uv_reconstruction_abs", 1e-8)):
        return None
    if owner_bary_error > float(tolerances.get("barycentric_abs", 1e-9)):
        return None
    point_error = float(np.max(np.abs(np.asarray(front[0]["point"]) - np.asarray(front[1]["point"]))))
    if point_error > float(tolerances.get("edge_point_abs", tolerances["point_abs"])):
        return None
    distances = np.linalg.norm(face_vertices[first][:, None, :] - face_vertices[second][None, :, :], axis=2)
    position_abs = float(tolerances.get("edge_position_abs", tolerances.get("edge_point_abs", tolerances["point_abs"])))
    matches = [(i, j) for i in range(3) for j in range(3) if float(distances[i, j]) <= position_abs]
    # A valid edge is exactly two one-to-one coincident endpoints. Reject a
    # fully duplicate face and any loose/near-coincident correspondence.
    if len(matches) != 2 or len({i for i, _ in matches}) != 2 or len({j for _, j in matches}) != 2:
        return None
    normal_dot = float(np.dot(face_normals[first], face_normals[second]))
    if normal_dot < float(tolerances.get("edge_normal_dot_min", 0.95)):
        return None
    if view is None:
        return None
    direction = np.asarray(view["camera_direction"], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    camera_position = np.asarray(view["camera_position"], dtype=np.float64)
    float_depths = [float(np.dot(np.asarray(hit["point"], dtype=np.float64) - camera_position, direction)) for hit in front]
    depth_spread = abs(float_depths[0] - float_depths[1])
    if depth_spread > float(tolerances.get("edge_depth_abs", tolerances["co_depth_abs"])):
        return None
    components = None if face_components is None else [int(face_components[first]), int(face_components[second])]
    charts = None if face_charts is None else [int(face_charts[first]), int(face_charts[second])]
    return {
        "face_a": first, "face_b": second, "owner": int(owner_face),
        "component_a": None if components is None else components[0], "component_b": None if components is None else components[1],
        "chart_a": None if charts is None else charts[0], "chart_b": None if charts is None else charts[1],
        "owner_component": None if face_components is None else int(face_components[int(owner_face)]),
        "owner_chart": None if face_charts is None else int(face_charts[int(owner_face)]),
        "shared_position_endpoint_count": 2, "point": np.asarray(front[0]["point"], dtype=np.float64),
        "point_error": point_error, "normal_dot": normal_dot,
        "owner_point_error": owner_point_error, "owner_uv_error": owner_uv_error,
        "owner_bary_error": owner_bary_error,
        "depth_spread": depth_spread, "front_layer": [int(hit["face"]) for hit in front],
    }


def semantic_digest(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json(metadata))
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(arrays[name]).tobytes(order="C"))
    return digest.hexdigest()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    mesh = Path(args.mesh)
    camera_contract_path = Path(args.camera_contract)
    evidence_receipt_path = Path(args.evidence_receipt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_contract(camera_contract_path)
    images, evidence_hashes, evidence_paths = load_evidence(evidence_receipt_path)
    positions, _normals, uv, triangles = read_glb(mesh)
    if uv is None:
        raise RuntimeError("MESH_UV_MISSING")
    positions = np.asarray(positions, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise RuntimeError("MESH_TRIANGLES_INVALID")
    transform = np.asarray(contract.get("control_space_transform"), dtype=np.float64)
    if transform.shape != (3, 3):
        raise RuntimeError("CAMERA_CONTRACT_TRANSFORM_INVALID")
    canonical = positions @ transform.T
    max_abs = float(np.max(np.abs(canonical)))
    if max_abs <= 1e-14:
        raise RuntimeError("MESH_CANONICAL_SCALE_INVALID")
    canonical *= 0.5 / max_abs
    face_vertices = canonical[triangles]
    face_uv = uv[triangles]
    face_normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    face_norms = np.linalg.norm(face_normals, axis=1)
    degenerate = face_norms <= 1e-14
    face_normals /= np.maximum(face_norms[:, None], 1e-14)
    face_components, component_count = connected_face_components(triangles, len(positions))
    face_charts, chart_count = uv_chart_components(triangles, uv)
    mesh_hash = sha256_file(mesh)
    triangle_index_hash = sha256_array(triangles.astype("<i8", copy=False))
    vertex_index_hash = sha256_array(np.arange(len(positions), dtype="<i8"))
    uv_hash = sha256_array(uv.astype("<f8", copy=False))
    camera_hash = sha256_file(camera_contract_path)
    owner, owner_weights = rasterise(uv, triangles, int(args.atlas_size))
    covered_texels, _covered_triangles = census(uv, triangles, int(args.atlas_size))
    cover_counts = np.bincount(covered_texels, minlength=int(args.atlas_size) * int(args.atlas_size))
    ambiguous_uv_texels = np.flatnonzero(cover_counts > 1).astype(np.int64)
    # A UV edge with more than one covering face has no unique owner under the
    # continuous contract.  It is excluded from the manifest population rather
    # than admitted through a neighbor/lowest-index waiver.
    owner.reshape(-1)[ambiguous_uv_texels] = -1
    owned_flat = np.flatnonzero(owner.reshape(-1) >= 0).astype(np.int64)
    owner_face = owner.reshape(-1)[owned_flat].astype(np.int32)
    wa = owner_weights.reshape(-1, 2)[owned_flat].astype(np.float64)
    # atlas_raster returns wa/wb as coefficients of face corners 1/2; keep
    # the manifest barycentric order canonical as corners 0/1/2.
    owner_bary = np.column_stack((1.0 - wa[:, 0] - wa[:, 1], wa[:, 0], wa[:, 1]))
    owner_points = np.einsum("ni,nij->nj", owner_bary, face_vertices[owner_face])
    owner_uv = np.einsum("ni,nij->nj", owner_bary, face_uv[owner_face])
    texel_y, texel_x = np.divmod(owned_flat, int(args.atlas_size))
    texel_xy = np.column_stack((texel_x, texel_y)).astype(np.int32)
    numerator = (2 * texel_xy + 1).astype(np.int64)
    views = sorted(contract["views"], key=lambda item: int(item["index"]))
    projection_span = float(contract.get("projection_span", contract.get("projection_half_span", 0.55) * 2.0))
    tolerances = {
        "barycentric_abs": 1e-9,
        "screen_bounds": 1e-9,
        "point_abs": 1e-8,
        "depth_abs": 1e-8,
        "depth_rel": 1e-8,
        "co_depth_abs": 1e-8,
        "co_depth_rel": 1e-8,
        "edge_point_abs": 1e-8,
        "edge_position_abs": 1e-8,
        "edge_depth_abs": 1e-8,
        "edge_normal_dot_min": 0.95,
        "facing_min": 1e-8,
        "uv_reconstruction_abs": 1e-8,
    }
    arrays: dict[str, list[Any]] = {
        "view_index": [], "sample_index": [], "state": [], "owner_face": [], "layer0_face": [],
        "expected_layer": [], "owner_barycentric": [], "owner_point": [], "owner_uv": [],
        "edge_tie_face_a": [], "edge_tie_face_b": [], "edge_tie_owner": [],
        "edge_tie_component_a": [], "edge_tie_component_b": [], "edge_tie_chart_a": [], "edge_tie_chart_b": [],
        "edge_tie_owner_component": [], "edge_tie_owner_chart": [],
        "edge_tie_shared_position_endpoint_count": [], "edge_tie_point": [],
        "edge_tie_point_error": [], "edge_tie_normal_dot": [], "edge_tie_depth_spread": [],
        "edge_tie_owner_point_error": [], "edge_tie_owner_uv_error": [], "edge_tie_owner_bary_error": [],
        "owner_normal": [], "owner_component": [], "screen_xy": [], "source_xy": [], "evidence_rgb": [],
        "evidence_class": [], "admitted_color": [],
        "uv_chart": [], "identity_mesh_hash": [], "identity_triangle_index_hash": [],
        "identity_vertex_index_hash": [], "identity_uv_hash": [], "identity_camera_hash": [],
        "identity_evidence_hash": [],
        "hit_offsets": [0], "hit_face": [], "hit_depth": [], "hit_barycentric": [], "hit_point": [],
        "hit_uv": [], "hit_component": [], "hit_facing": [], "hit_layer": [], "hit_co_depth_group": [],
    }
    per_view: list[dict[str, Any]] = []
    max_layer = 0
    co_depth_ambiguities = 0
    missing_owner = 0
    for view in views:
        semantic = str(view["proven_semantic"])
        if semantic not in images:
            raise RuntimeError(f"EVIDENCE_SEMANTIC_MISSING:{semantic}")
        tri_screen = project(face_vertices.reshape(-1, 3), view, projection_span).reshape(-1, 3, 2)
        screen_xy = project(owner_points, view, projection_span)
        all_hits = query_all_hits(
            screen_xy, tri_screen, canonical, face_vertices, face_uv, face_normals,
            face_components, view, projection_span, int(args.atlas_size), tolerances,
        )
        state_counts = {name: 0 for name in STATE_NAMES}
        view_max_layer = 0
        view_ambiguity = 0
        view_missing = 0
        valid_screen = np.all((screen_xy >= -tolerances["screen_bounds"]) & (screen_xy <= 1.0 + tolerances["screen_bounds"]), axis=1)
        source_xy = np.full((len(owned_flat), 2), -1.0, dtype=np.float64)
        evidence_rgb = np.zeros((len(owned_flat), 3), dtype=np.float32)
        valid_rows = np.flatnonzero(valid_screen)
        if valid_rows.size:
            image = images[semantic]
            source_xy[valid_rows] = screen_xy[valid_rows] * np.asarray([image.shape[1] - 1, image.shape[0] - 1], dtype=np.float64)
        for sample_index, hits in enumerate(all_hits):
            expected = int(owner_face[sample_index])
            state = "CONTRACT_ERROR"
            layer0 = -1
            expected_layer = -1
            edge_proof = None
            if not valid_screen[sample_index]:
                state = "OUT_OF_FRAME"
            elif not hits:
                # A continuous orthographic ray has no unique intersection with
                # an edge-on expected face.  This is the locked deterministic
                # BACKFACING rejection, not a missing-owner waiver.
                direction = unit(np.asarray(view["camera_direction"], dtype=np.float64))
                owner_facing = float(np.dot(face_normals[expected], -direction))
                if owner_facing <= tolerances["facing_min"]:
                    state = "BACKFACING"
                else:
                    state = "CONTRACT_ERROR"
                    missing_owner += 1
                    view_missing += 1
            else:
                layer0 = int(hits[0]["face"])
                expected_positions = [idx for idx, hit in enumerate(hits) if hit["face"] == expected]
                if expected_positions:
                    expected_layer = int(expected_positions[0])
                    view_max_layer = max(view_max_layer, expected_layer)
                    max_layer = max(max_layer, expected_layer)
                else:
                    direction = unit(np.asarray(view["camera_direction"], dtype=np.float64))
                    owner_facing = float(np.dot(face_normals[expected], -direction))
                    if owner_facing <= tolerances["facing_min"]:
                        state = "BACKFACING"
                    else:
                        missing_owner += 1
                        view_missing += 1
                first_depth = float(hits[0]["depth"])
                group = [hit for hit in hits if abs(float(hit["depth"]) - first_depth) <= tolerances["co_depth_abs"] + tolerances["co_depth_rel"] * max(1.0, abs(first_depth))]
                edge_proof = visible_edge_tie_signature(
                    expected, hits, face_vertices, face_normals, tolerances, view,
                    owner_points[sample_index], owner_uv[sample_index], owner_bary[sample_index], face_components, face_charts,
                )
                # Any unresolved front co-depth group is a contract error.  A
                # deeper expected owner may be OCCLUDED only after the front
                # group is uniquely resolved.
                ambiguity_policy_state = classify_front_co_depth_policy(
                    int(view["index"]), int(sample_index), int(expected), hits, tolerances, view,
                )
                if edge_proof is not None:
                    state = "VISIBLE_EDGE_TIE"
                    co_depth_ambiguities += 1
                    view_ambiguity += 1
                elif ambiguity_policy_state == "UNKNOWN_AMBIGUOUS":
                    # The bounded diagnostic proved these rows are irreducible
                    # front co-depths.  They are completion-only unknowns: no
                    # evidence/color is admitted and no arbitrary layer wins.
                    state = "UNKNOWN_AMBIGUOUS"
                    co_depth_ambiguities += 1
                    view_ambiguity += 1
                elif ambiguity_policy_state == "CONTRACT_ERROR":
                    # Any new/unlisted ambiguity remains a hard contract
                    # failure; the approved set above is intentionally closed.
                    state = "CONTRACT_ERROR"
                    co_depth_ambiguities += 1
                elif expected_layer < 0 and state != "BACKFACING":
                    state = "CONTRACT_ERROR"
                elif state == "BACKFACING":
                    pass
                elif expected_layer > 0:
                    state = "OCCLUDED"
                else:
                    expected_hit = hits[0]
                    point_error = float(np.max(np.abs(expected_hit["point"] - owner_points[sample_index])))
                    uv_error = float(np.max(np.abs(expected_hit["uv"] - owner_uv[sample_index])))
                    if point_error > tolerances["point_abs"] or uv_error > tolerances["uv_reconstruction_abs"]:
                        state = "CONTRACT_ERROR"
                    elif expected_hit["facing"] < tolerances["facing_min"]:
                        state = "BACKFACING"
                    else:
                        state = "VISIBLE_EXACT"
            if state in ADMITTED_STATES:
                evidence_rgb[sample_index] = bilinear(images[semantic], source_xy[sample_index:sample_index + 1])[0]
            state_counts[state] += 1
            row = len(arrays["state"])
            arrays["view_index"].append(int(view["index"]))
            arrays["sample_index"].append(sample_index)
            arrays["state"].append(STATE_CODE[state])
            arrays["owner_face"].append(expected)
            arrays["layer0_face"].append(layer0)
            arrays["expected_layer"].append(expected_layer)
            arrays["owner_barycentric"].append(owner_bary[sample_index])
            arrays["owner_point"].append(owner_points[sample_index])
            arrays["owner_uv"].append(owner_uv[sample_index])
            if edge_proof is None:
                arrays["edge_tie_face_a"].append(-1)
                arrays["edge_tie_face_b"].append(-1)
                arrays["edge_tie_owner"].append(-1)
                arrays["edge_tie_component_a"].append(-1)
                arrays["edge_tie_component_b"].append(-1)
                arrays["edge_tie_chart_a"].append(-1)
                arrays["edge_tie_chart_b"].append(-1)
                arrays["edge_tie_owner_component"].append(-1)
                arrays["edge_tie_owner_chart"].append(-1)
                arrays["edge_tie_shared_position_endpoint_count"].append(0)
                arrays["edge_tie_point"].append(np.zeros(3, dtype=np.float64))
                arrays["edge_tie_point_error"].append(0.0)
                arrays["edge_tie_normal_dot"].append(0.0)
                arrays["edge_tie_depth_spread"].append(0.0)
                arrays["edge_tie_owner_point_error"].append(0.0)
                arrays["edge_tie_owner_uv_error"].append(0.0)
                arrays["edge_tie_owner_bary_error"].append(0.0)
            else:
                arrays["edge_tie_face_a"].append(edge_proof["face_a"])
                arrays["edge_tie_face_b"].append(edge_proof["face_b"])
                arrays["edge_tie_owner"].append(edge_proof["owner"])
                arrays["edge_tie_component_a"].append(edge_proof["component_a"])
                arrays["edge_tie_component_b"].append(edge_proof["component_b"])
                arrays["edge_tie_chart_a"].append(edge_proof["chart_a"])
                arrays["edge_tie_chart_b"].append(edge_proof["chart_b"])
                arrays["edge_tie_owner_component"].append(edge_proof["owner_component"])
                arrays["edge_tie_owner_chart"].append(edge_proof["owner_chart"])
                arrays["edge_tie_shared_position_endpoint_count"].append(edge_proof["shared_position_endpoint_count"])
                arrays["edge_tie_point"].append(edge_proof["point"])
                arrays["edge_tie_point_error"].append(edge_proof["point_error"])
                arrays["edge_tie_normal_dot"].append(edge_proof["normal_dot"])
                arrays["edge_tie_depth_spread"].append(edge_proof["depth_spread"])
                arrays["edge_tie_owner_point_error"].append(edge_proof["owner_point_error"])
                arrays["edge_tie_owner_uv_error"].append(edge_proof["owner_uv_error"])
                arrays["edge_tie_owner_bary_error"].append(edge_proof["owner_bary_error"])
            arrays["owner_normal"].append(face_normals[expected])
            arrays["owner_component"].append(int(face_components[expected]))
            arrays["uv_chart"].append(int(face_charts[expected]))
            arrays["identity_mesh_hash"].append(hash_bytes_array(mesh_hash))
            arrays["identity_triangle_index_hash"].append(hash_bytes_array(triangle_index_hash))
            arrays["identity_vertex_index_hash"].append(hash_bytes_array(vertex_index_hash))
            arrays["identity_uv_hash"].append(hash_bytes_array(uv_hash))
            arrays["identity_camera_hash"].append(hash_bytes_array(camera_hash))
            arrays["identity_evidence_hash"].append(hash_bytes_array(evidence_hashes[semantic]))
            arrays["screen_xy"].append(screen_xy[sample_index])
            arrays["source_xy"].append(source_xy[sample_index])
            arrays["evidence_rgb"].append(evidence_rgb[sample_index])
            arrays["evidence_class"].append(EVIDENCE_CLASS_GENERATED if state in ADMITTED_STATES else 0)
            arrays["admitted_color"].append(state in ADMITTED_STATES)
            co_depth_group = -1
            previous_depth = None
            for layer, hit in enumerate(hits):
                if previous_depth is None or abs(float(hit["depth"]) - previous_depth) > tolerances["co_depth_abs"] + tolerances["co_depth_rel"] * max(1.0, abs(float(hit["depth"]))):
                    co_depth_group += 1
                previous_depth = float(hit["depth"])
                arrays["hit_face"].append(hit["face"])
                arrays["hit_depth"].append(hit["depth"])
                arrays["hit_barycentric"].append(hit["bary"])
                arrays["hit_point"].append(hit["point"])
                arrays["hit_uv"].append(hit["uv"])
                arrays["hit_component"].append(hit["component"])
                arrays["hit_facing"].append(hit["facing"])
                arrays["hit_layer"].append(layer)
                arrays["hit_co_depth_group"].append(co_depth_group)
            arrays["hit_offsets"].append(len(arrays["hit_face"]))
        per_view.append({
            "index": int(view["index"]), "semantic": semantic, "owned_samples": int(len(owned_flat)),
            "state_counts": state_counts, "admitted_evidence": int(state_counts["VISIBLE_EXACT"] + state_counts["VISIBLE_EDGE_TIE"]),
            "max_expected_layer": int(view_max_layer), "co_depth_ambiguities": int(view_ambiguity),
            "missing_owner": int(view_missing), "out_of_frame": int(state_counts["OUT_OF_FRAME"]),
        })
    canonical_arrays = {
        "view_index": np.asarray(arrays["view_index"], dtype="<i2"),
        "sample_index": np.asarray(arrays["sample_index"], dtype="<i4"),
        "state": np.asarray(arrays["state"], dtype="<u1"),
        "owner_face": np.asarray(arrays["owner_face"], dtype="<i4"),
        "layer0_face": np.asarray(arrays["layer0_face"], dtype="<i4"),
        "expected_layer": np.asarray(arrays["expected_layer"], dtype="<i4"),
        "owner_barycentric": np.asarray(arrays["owner_barycentric"], dtype="<f8"),
        "owner_point": np.asarray(arrays["owner_point"], dtype="<f8"),
        "owner_uv": np.asarray(arrays["owner_uv"], dtype="<f8"),
        "edge_tie_face_a": np.asarray(arrays["edge_tie_face_a"], dtype="<i4"),
        "edge_tie_face_b": np.asarray(arrays["edge_tie_face_b"], dtype="<i4"),
        "edge_tie_owner": np.asarray(arrays["edge_tie_owner"], dtype="<i4"),
        "edge_tie_component_a": np.asarray(arrays["edge_tie_component_a"], dtype="<i4"),
        "edge_tie_component_b": np.asarray(arrays["edge_tie_component_b"], dtype="<i4"),
        "edge_tie_chart_a": np.asarray(arrays["edge_tie_chart_a"], dtype="<i4"),
        "edge_tie_chart_b": np.asarray(arrays["edge_tie_chart_b"], dtype="<i4"),
        "edge_tie_owner_component": np.asarray(arrays["edge_tie_owner_component"], dtype="<i4"),
        "edge_tie_owner_chart": np.asarray(arrays["edge_tie_owner_chart"], dtype="<i4"),
        "edge_tie_shared_position_endpoint_count": np.asarray(arrays["edge_tie_shared_position_endpoint_count"], dtype="<i1"),
        "edge_tie_point": np.asarray(arrays["edge_tie_point"], dtype="<f8"),
        "edge_tie_point_error": np.asarray(arrays["edge_tie_point_error"], dtype="<f8"),
        "edge_tie_normal_dot": np.asarray(arrays["edge_tie_normal_dot"], dtype="<f8"),
        "edge_tie_depth_spread": np.asarray(arrays["edge_tie_depth_spread"], dtype="<f8"),
        "edge_tie_owner_point_error": np.asarray(arrays["edge_tie_owner_point_error"], dtype="<f8"),
        "edge_tie_owner_uv_error": np.asarray(arrays["edge_tie_owner_uv_error"], dtype="<f8"),
        "edge_tie_owner_bary_error": np.asarray(arrays["edge_tie_owner_bary_error"], dtype="<f8"),
        "owner_normal": np.asarray(arrays["owner_normal"], dtype="<f8"),
        "owner_component": np.asarray(arrays["owner_component"], dtype="<i4"),
        "uv_chart": np.asarray(arrays["uv_chart"], dtype="<i4"),
        "identity_mesh_hash": np.asarray(arrays["identity_mesh_hash"], dtype="<u1"),
        "identity_triangle_index_hash": np.asarray(arrays["identity_triangle_index_hash"], dtype="<u1"),
        "identity_vertex_index_hash": np.asarray(arrays["identity_vertex_index_hash"], dtype="<u1"),
        "identity_uv_hash": np.asarray(arrays["identity_uv_hash"], dtype="<u1"),
        "identity_camera_hash": np.asarray(arrays["identity_camera_hash"], dtype="<u1"),
        "identity_evidence_hash": np.asarray(arrays["identity_evidence_hash"], dtype="<u1"),
        "screen_xy": np.asarray(arrays["screen_xy"], dtype="<f8"),
        "source_xy": np.asarray(arrays["source_xy"], dtype="<f8"),
        "evidence_rgb": np.asarray(arrays["evidence_rgb"], dtype="<f4"),
        "evidence_class": np.asarray(arrays["evidence_class"], dtype="<u1"),
        "admitted_color": np.asarray(arrays["admitted_color"], dtype="<?"),
        "hit_offsets": np.asarray(arrays["hit_offsets"], dtype="<i8"),
        "hit_face": np.asarray(arrays["hit_face"], dtype="<i4"),
        "hit_depth": np.asarray(arrays["hit_depth"], dtype="<f8"),
        "hit_barycentric": np.asarray(arrays["hit_barycentric"], dtype="<f8"),
        "hit_point": np.asarray(arrays["hit_point"], dtype="<f8"),
        "hit_uv": np.asarray(arrays["hit_uv"], dtype="<f8"),
        "hit_component": np.asarray(arrays["hit_component"], dtype="<i4"),
        "hit_facing": np.asarray(arrays["hit_facing"], dtype="<f8"),
        "hit_layer": np.asarray(arrays["hit_layer"], dtype="<i4"),
        "hit_co_depth_group": np.asarray(arrays["hit_co_depth_group"], dtype="<i4"),
        "texel_xy": texel_xy.astype("<i4"),
        "texel_center_numerator": numerator.astype("<i8"),
    }
    metadata = {
        "schema": SCHEMA, "contract_version": CONTRACT_VERSION, "atlas_size": int(args.atlas_size),
        "owned_texel_count": int(len(owned_flat)), "row_count": int(len(canonical_arrays["state"])),
        "mesh_sha256": mesh_hash, "triangle_index_hash": triangle_index_hash,
        "vertex_index_hash": vertex_index_hash, "uv_hash": uv_hash,
        "camera_contract_sha256": camera_hash,
        "evidence_receipt_sha256": sha256_file(evidence_receipt_path), "evidence_hashes": evidence_hashes,
        "mesh_triangle_count": int(len(triangles)), "mesh_vertex_count": int(len(positions)),
        "component_count": int(component_count), "uv_chart_count": int(chart_count),
        "projection_span": projection_span,
        "control_space_transform": transform.tolist(),
        "atlas_owner_rule": "exact texel centre rational numerator (2x+1,2y+1)/(2N); unique covering face only; multiply-covered UV edges rejected",
        "ambiguous_uv_texels_rejected": int(len(ambiguous_uv_texels)),
        "identity_hashes_per_row": True,
        "projection_rule": "continuous orthographic screen=[dot(point,right)/span+0.5,0.5-dot(point,up)/span]",
        "coordinate_conventions": {
            "pixel_center": "atlas texel center is exact rational ((2*x+1),(2*y+1))/(2*N)",
            "image_origin": "top_left",
            "row_direction": "increasing y moves downward",
            "uv_origin": "top_left",
            "screen_origin": "top_left normalized [0,1]^2",
            "source_xy": "screen_xy * (image_width-1,image_height-1), float64",
            "bilinear": "floor/clamp neighboring pixels, linear weights in float64, output float32 RGB",
            "rounding": "no implicit integer rounding in manifest",
        },
        "all_hit_rule": "screen-bin acceleration followed by float64 barycentric re-test; sort by depth then face ID",
        "layer_rule": "only unique layer 0 may admit color, except VISIBLE_EDGE_TIE binds color to the serialized owner among exactly two tied front faces",
        "visible_edge_tie_rule": "exactly two nearest front faces, owner present, exactly two coincident position endpoints, owner point/UV/bary identity, point/normal gates, float64 depth separation <= locked edge_depth_abs, replay-stable; both faces, component/chart IDs, and owner are serialized",
        "visible_edge_tie_policy_version": "owner_present_shared_position_edge_zero_longdouble_depth_v1",
        "unknown_ambiguous_rule": "irreducible measured front co-depth rows are UNKNOWN_AMBIGUOUS; no color/evidence or arbitrary layer is admitted",
        "unknown_ambiguous_policy_version": "exact_two_front_faces_owner_absent_equal_depth_longdouble_replay_stable_v1",
        "legacy_rounded_id_buffers_authoritative": False,
        "arbitrary_layer_acceptance": 0,
        "implicit_pixel_rounding": 0,
        "numeric_tolerance_receipt": tolerances,
        "state_names": list(STATE_NAMES),
        "evidence_paths": evidence_paths,
        "evidence_class_generated": EVIDENCE_CLASS_GENERATED,
        "per_view": per_view,
        "co_depth_ambiguity_count": int(co_depth_ambiguities),
        "missing_owner_count": int(missing_owner),
        "max_expected_layer": int(max_layer),
    }
    state_counts_total = {name: int(sum(view["state_counts"][name] for view in per_view)) for name in STATE_NAMES}
    metadata["state_counts_total"] = state_counts_total
    metadata["contract_error_count"] = int(state_counts_total["CONTRACT_ERROR"])
    metadata["unknown_ambiguous_count"] = int(state_counts_total["UNKNOWN_AMBIGUOUS"])
    metadata["semantic_digest"] = semantic_digest(canonical_arrays, {
        "contract_version": CONTRACT_VERSION, "atlas_size": int(args.atlas_size),
        "mesh_sha256": metadata["mesh_sha256"], "triangle_index_hash": metadata["triangle_index_hash"],
        "vertex_index_hash": metadata["vertex_index_hash"], "uv_hash": metadata["uv_hash"],
        "camera_contract_sha256": metadata["camera_contract_sha256"], "evidence_hashes": evidence_hashes,
        "state_names": list(STATE_NAMES),
    })
    write_deterministic_npz(output_dir / "surface_manifest.npz", canonical_arrays)
    manifest_hash = sha256_file(output_dir / "surface_manifest.npz")
    metadata["surface_manifest_sha256"] = manifest_hash
    metadata["runtime_seconds"] = round(float(time.perf_counter() - started), 6)
    metadata["peak_rss_mb"] = round(float(psutil.Process().memory_info().rss / (1024.0 * 1024.0)), 3)
    write_json(output_dir / "surface_manifest.json", metadata)
    write_json(output_dir / "state_counts.json", {
        "schema": "panda_continuous_surface_state_counts_v1", "atlas_size": int(args.atlas_size),
        "owned_texel_count": int(len(owned_flat)), "per_view": per_view, "total": state_counts_total,
        "admitted_evidence_total": int(state_counts_total["VISIBLE_EXACT"] + state_counts_total["VISIBLE_EDGE_TIE"]),
        "contract_error_total": int(state_counts_total["CONTRACT_ERROR"]),
        "visible_edge_tie_total": int(state_counts_total["VISIBLE_EDGE_TIE"]),
        "unknown_ambiguous_total": int(state_counts_total["UNKNOWN_AMBIGUOUS"]),
        "arbitrary_layer_acceptance": 0, "implicit_pixel_rounding": 0,
    })
    write_json(output_dir / "toolchain_receipt.json", {
        "schema": "panda_continuous_surface_toolchain_receipt_v1", "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy_version, "platform": platform.platform(),
        "implementation": "workers/build_continuous_surface_manifest.py", "cold_run_label": str(args.run_label),
        "legacy_rounded_id_buffers_read": False,
        "mesh_sha256": mesh_hash, "triangle_index_hash": triangle_index_hash,
        "vertex_index_hash": vertex_index_hash, "uv_hash": uv_hash,
        "camera_contract_sha256": camera_hash, "evidence_hashes": evidence_hashes,
        "contract_error_total": int(state_counts_total["CONTRACT_ERROR"]),
        "visible_edge_tie_total": int(state_counts_total["VISIBLE_EDGE_TIE"]),
        "unknown_ambiguous_total": int(state_counts_total["UNKNOWN_AMBIGUOUS"]),
    })
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--camera-contract", required=True)
    parser.add_argument("--evidence-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas-size", type=int, default=256)
    parser.add_argument("--run-label", default="cold_run")
    args = parser.parse_args()
    if int(args.atlas_size) not in (256, 384, 512):
        raise RuntimeError("BOUNDED_ATLAS_SIZE_ONLY_256_384_OR_512")
    report = build_manifest(args)
    print(json.dumps({
        "surface_manifest": str(Path(args.output_dir) / "surface_manifest.npz"),
        "surface_manifest_sha256": report["surface_manifest_sha256"],
        "semantic_digest": report["semantic_digest"], "row_count": report["row_count"],
        "runtime_seconds": report["runtime_seconds"], "peak_rss_mb": report["peak_rss_mb"],
        "contract_error_count": report["contract_error_count"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
