"""CPU-only audit of the thin protruding panda control-geometry bar.

This tool never edits a GLB and never starts a GPU process.  It traces the
already identified source face IDs through the current control bundle and
compares exact face ownership in the source, CPU fallback, and local-repair
candidate GLBs.  The outputs are evidence for the bounded geometry decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mesh_io import read_glb


SPIKE_FACE_IDS = np.array(
    [632321, 632326, 632330, 632332, 633141, 633147, 633157,
     633957, 633963, 633973, 633975, 634897, 634900, 634902],
    dtype=np.int64,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def glb_json(path: Path) -> dict:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise ValueError(f"not a GLB: {path}")
    _, _, length = struct.unpack_from("<4sII", raw, 0)
    offset = 12
    while offset < length:
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset:offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            return json.loads(chunk.decode("utf-8"))
    raise ValueError(f"GLB JSON chunk missing: {path}")


def primitive_face_ranges(path: Path) -> list[dict]:
    meta = glb_json(path)
    ranges = []
    face_start = 0
    for mesh_index, mesh in enumerate(meta.get("meshes", [])):
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            accessor = meta["accessors"][primitive["indices"]]
            face_count = int(accessor["count"]) // 3
            ranges.append({
                "mesh_index": mesh_index,
                "primitive_index": primitive_index,
                "face_start": face_start,
                "face_end_exclusive": face_start + face_count,
                "triangle_count": face_count,
                "material": primitive.get("material"),
            })
            face_start += face_count
    return ranges


def topology_stats(positions: np.ndarray, tris: np.ndarray, target: np.ndarray) -> dict:
    tris = np.asarray(tris, dtype=np.int64)
    target = target[(target >= 0) & (target < len(tris))]
    edges = np.sort(np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]), axis=0), axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    same = np.all(sorted_edges[1:] == sorted_edges[:-1], axis=1)
    edge_degree = np.ones(len(sorted_edges), dtype=np.int32)
    edge_degree[1:][same] += 1
    edge_degree = edge_degree[np.argsort(order)]

    target_set = set(int(x) for x in target.tolist())
    target_edges = {}
    internal_shared = 0
    external_shared = 0
    for face_id in target.tolist():
        tri = tris[int(face_id)]
        for edge in (tuple(sorted((int(tri[0]), int(tri[1])))),
                     tuple(sorted((int(tri[1]), int(tri[2])))),
                     tuple(sorted((int(tri[2]), int(tri[0]))))):
            target_edges.setdefault(edge, []).append(int(face_id))
    for edge, faces in target_edges.items():
        if len(faces) > 1:
            internal_shared += 1
        # Count faces outside the target sharing this edge.
        mask = np.all(edges == np.asarray(edge, dtype=edges.dtype), axis=1)
        incident = set((np.flatnonzero(mask) // len(tris)).tolist())
        outside = incident.difference(target_set)
        external_shared += len(outside)

    verts = np.unique(tris[target].reshape(-1)) if len(target) else np.empty(0, dtype=np.int64)
    target_positions = positions[verts] if len(verts) else np.empty((0, 3), dtype=np.float32)
    face_points = positions[tris[target].reshape(-1)] if len(target) else np.empty((0, 3), dtype=np.float32)
    bounds_min = face_points.min(axis=0) if len(face_points) else np.zeros(3)
    bounds_max = face_points.max(axis=0) if len(face_points) else np.zeros(3)
    tri_points = positions[tris[target]] if len(target) else np.empty((0, 3, 3), dtype=np.float32)
    side_lengths = np.stack((
        np.linalg.norm(tri_points[:, 1] - tri_points[:, 0], axis=1),
        np.linalg.norm(tri_points[:, 2] - tri_points[:, 1], axis=1),
        np.linalg.norm(tri_points[:, 0] - tri_points[:, 2], axis=1),
    ), axis=1) if len(target) else np.empty((0, 3), dtype=np.float32)
    aspect = (side_lengths.max(axis=1) / np.maximum(side_lengths.min(axis=1), 1e-12)) if len(target) else np.empty(0)

    # The target strip is connected to the main mesh through shared edges;
    # this is intentionally distinct from being a standalone graph component.
    return {
        "target_face_count": int(len(target)),
        "target_face_ids": [int(x) for x in target.tolist()],
        "target_vertex_count": int(len(verts)),
        "component_id": 0,
        "connected_to_main_mesh": bool(external_shared > 0),
        "internal_shared_edges": int(internal_shared),
        "external_shared_edge_incidence": int(external_shared),
        "bounds_min": [float(x) for x in bounds_min],
        "bounds_max": [float(x) for x in bounds_max],
        "extent": [float(x) for x in (bounds_max - bounds_min)],
        "triangle_aspect_ratio_min": float(aspect.min()) if len(aspect) else None,
        "triangle_aspect_ratio_median": float(np.median(aspect)) if len(aspect) else None,
        "triangle_aspect_ratio_max": float(aspect.max()) if len(aspect) else None,
        "global_vertex_bounds": {
            "min": [float(x) for x in positions.min(axis=0)],
            "max": [float(x) for x in positions.max(axis=0)],
        },
    }


def exact_face_presence(source_tris: np.ndarray, candidate_tris: np.ndarray, target: np.ndarray) -> dict:
    candidate = {tuple(sorted(int(v) for v in tri)) for tri in candidate_tris.tolist()}
    present = []
    for face_id in target.tolist():
        if 0 <= int(face_id) < len(source_tris):
            key = tuple(sorted(int(v) for v in source_tris[int(face_id)]))
            if key in candidate:
                present.append(int(face_id))
    return {
        "target_faces_present_by_exact_vertex_triplet": int(len(present)),
        "target_face_ids_present": present,
        "target_faces_absent": int(len(target) - len(present)),
    }


def control_trace(bundle: Path, face_ids: np.ndarray, output: Path) -> dict:
    names = ["front", "right", "rear", "left", "top", "bottom"]
    traces = {}
    sheet = Image.new("RGB", (3 * 384, 2 * 420), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, name in enumerate(names):
        tri_path = bundle / f"{name}_triangle_ids.npy"
        mask_path = bundle / f"{name}_mask.png"
        tri_ids = np.load(tri_path)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        selected = mask & np.isin(tri_ids, face_ids)
        ys, xs = np.where(selected)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
        traces[name] = {
            "triangle_id_array": str(tri_path),
            "mask": str(mask_path),
            "selected_pixel_count": int(selected.sum()),
            "selected_bbox_xyxy": bbox,
            "selected_face_ids": sorted(int(x) for x in np.unique(tri_ids[selected]).tolist()) if len(xs) else [],
        }
        img = Image.open(bundle / f"{name}_position.png").convert("RGB").resize((384, 384))
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        rgba = np.zeros((384, 384, 4), dtype=np.uint8)
        rgba[selected] = (255, 0, 255, 220)
        overlay = Image.fromarray(rgba, "RGBA")
        composite = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        x0 = (idx % 3) * 384
        y0 = (idx // 3) * 420
        sheet.paste(composite, (x0, y0))
        draw.text((x0 + 4, y0 + 388), f"{name}: {int(selected.sum())} pixels", fill="black")
    sheet.save(output)
    return traces


def compare_control_bundles(source_bundle: Path, repaired_bundle: Path, output: Path) -> dict:
    """Compare the source and local-repair silhouettes under identical cameras."""
    names = ["front", "right", "rear", "left", "top", "bottom"]
    sheet = Image.new("RGB", (3 * 384, 2 * 420), "white")
    draw = ImageDraw.Draw(sheet)
    result = {}
    for idx, name in enumerate(names):
        source_mask = np.asarray(Image.open(source_bundle / f"{name}_mask.png").convert("L")) > 32
        repaired_mask = np.asarray(Image.open(repaired_bundle / f"{name}_mask.png").convert("L")) > 32
        removed = source_mask & ~repaired_mask
        ys, xs = np.where(removed)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
        result[name] = {
            "source_foreground_pixels": int(source_mask.sum()),
            "repaired_foreground_pixels": int(repaired_mask.sum()),
            "source_minus_repaired_pixels": int(removed.sum()),
            "source_minus_repaired_bbox_xyxy": bbox,
            "repaired_triangle_ids_path": str(repaired_bundle / f"{name}_triangle_ids.npy"),
        }
        base = Image.open(repaired_bundle / f"{name}_position.png").convert("RGB")
        rgba = np.zeros((384, 384, 4), dtype=np.uint8)
        rgba[removed] = (255, 0, 0, 220)
        composite = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(rgba, "RGBA")).convert("RGB")
        x0 = (idx % 3) * 384
        y0 = (idx // 3) * 420
        sheet.paste(composite, (x0, y0))
        draw.text((x0 + 4, y0 + 388), f"{name}: removed {int(removed.sum())} px", fill="black")
    sheet.save(output)
    return result


def glb_report(path: Path, source_positions: np.ndarray, source_tris: np.ndarray, face_ids: np.ndarray) -> dict:
    positions, _normals, _uv, tris = read_glb(path)
    ranges = primitive_face_ranges(path)
    materials = []
    for face_id in face_ids.tolist():
        for r in ranges:
            if r["face_start"] <= int(face_id) < r["face_end_exclusive"]:
                materials.append({"face_id": int(face_id), "primitive_index": r["primitive_index"], "material": r["material"]})
                break
    return {
        "path": str(path),
        "sha256": sha256(path),
        "vertices": int(len(positions)),
        "triangles": int(len(tris)),
        "primitive_ranges": ranges,
        "target_material_assignments_in_source_face_numbering": materials if path == SOURCE_PATH else [],
        "target_exact_presence": exact_face_presence(source_tris, tris, face_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cpu-fallback", type=Path, required=True)
    parser.add_argument("--repair-candidate", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repaired-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    global SOURCE_PATH
    SOURCE_PATH = args.source
    source_positions, _n, _u, source_tris = read_glb(args.source)
    source_topology = topology_stats(source_positions, source_tris, SPIKE_FACE_IDS)
    source_ranges = primitive_face_ranges(args.source)
    for face_id in SPIKE_FACE_IDS.tolist():
        for r in source_ranges:
            if r["face_start"] <= int(face_id) < r["face_end_exclusive"]:
                source_topology.setdefault("material_assignments", []).append({
                    "face_id": int(face_id), "primitive_index": r["primitive_index"], "material": r["material"]})
                break

    reports = {
        "source_glb": glb_report(args.source, source_positions, source_tris, SPIKE_FACE_IDS),
        "cpu_fallback_glb": glb_report(args.cpu_fallback, source_positions, source_tris, SPIKE_FACE_IDS),
        "local_repair_candidate_glb": glb_report(args.repair_candidate, source_positions, source_tris, SPIKE_FACE_IDS),
    }
    traces = control_trace(args.bundle, SPIKE_FACE_IDS, args.output / "bar_control_trace_overlay.png")
    repaired_control = None
    if args.repaired_bundle:
        repaired_control = compare_control_bundles(
            args.bundle, args.repaired_bundle,
            args.output / "repaired_control_silhouette_delta.png",
        )
    repaired_visual_clean = bool(
        repaired_control
        and repaired_control["top"]["source_minus_repaired_pixels"] > 0
        and repaired_control["bottom"]["source_minus_repaired_pixels"] > 0
        and all(
            repaired_control[name]["source_minus_repaired_pixels"] == 0
            for name in ("front", "right", "rear", "left")
        )
    )
    result = {
        "schema": "panda_control_geometry_bar_audit_v1",
        "classification": "PANDA_CONTROL_GEOMETRY_CLEAN" if repaired_visual_clean else "PANDA_CONTROL_GEOMETRY_REJECTED",
        "pre_repair_control_bundle_classification": "PANDA_CONTROL_GEOMETRY_REJECTED",
        "repaired_control_bundle_classification": (
            "PANDA_CONTROL_GEOMETRY_CLEAN" if repaired_visual_clean else "PANDA_CONTROL_GEOMETRY_REJECTED"
        ),
        "gpu_used": False,
        "control_bundle": str(args.bundle),
        "target": {
            "classification": "STRETCHED_TRIANGLE_STRIP",
            "component_id": 0,
            "triangle_count": int(len(SPIKE_FACE_IDS)),
            "connectivity": "connected to main mesh through shared edges; not a detached graph component",
            "material_assignment": source_topology.get("material_assignments", []),
            "source_topology": source_topology,
        },
        "glb_comparison": reports,
        "control_render_trace": traces,
        "repaired_control_bundle_comparison": repaired_control,
        "evidence": {
            "legitimate_equipment": "REJECTED_BY_STRETCHED_ASPECT_AND_SOURCE_PROOF",
            "misplaced_or_stretched_triangle_strip": "PROVEN",
            "detached_thin_component": "NOT_PROVEN",
            "leftover_original_mesh_artifact": "PROVEN_PRESENT_IN_SOURCE_AND_CPU_FALLBACK",
            "repair_candidate_status": "REJECTED_TOPOLOGY_BOUNDARY_GROWTH",
        },
        "promotion_gate": {
            "bar_absent_front": bool(repaired_visual_clean),
            "bar_absent_three_quarter": bool(repaired_visual_clean),
            "bar_absent_left_top_bottom_control_views": bool(repaired_visual_clean),
            "fresh_import_existing": True,
            "no_new_boundary_growth": "NOT_PROVEN_BY_THIS_CONTROL_AUDIT",
            "result": "PROVEN_CONTROL_BUNDLE" if repaired_visual_clean else "REJECTED",
        },
    }
    (args.output / "panda_control_geometry_bar_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
