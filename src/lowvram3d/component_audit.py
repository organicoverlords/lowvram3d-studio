"""Generic deterministic debris audit for a clean high-resolution geometry master.

Runs before any retopology or decimation.  Decisions combine connected-component geometry,
scale-relative proximity, fourteen projected views, screen-space isolation, source-mask support and
depth separation.  Face count alone is never sufficient.  Suspicious-but-unproven components fail
closed instead of being silently kept or deleted.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .geometry_compare import VIEW_DIRECTIONS, load_mesh, sample_surface, topology_counts
from .quality_ladder import AssetFamily, family_for_asset_type


@dataclass(frozen=True, slots=True)
class AuditConfig:
    render_size: int = 384
    total_samples: int = 220_000
    min_component_samples: int = 128
    max_component_samples: int = 8192
    max_passes: int = 4
    attach_distance_diag: float = 0.0015
    detached_distance_diag: float = 0.0020
    hover_distance_diag: float = 0.0040
    hover_depth_gap_diag: float = 0.0060
    meaningful_fraction: float = 0.03
    source_keep_percent: float = 10.0
    outboard_percent: float = 85.0
    outboard_views: int = 3
    gap_views: int = 3
    hover_views: int = 5
    hover_max_area_fraction: float = 0.015
    internal_max_area_fraction: float = 0.002


@dataclass(frozen=True, slots=True)
class Component:
    component_id: int
    face_ids: np.ndarray
    faces: int
    face_fraction: float
    area: float
    area_fraction: float
    centroid: np.ndarray
    extent: np.ndarray
    elongation: float
    flatness: float
    nearest_distance_diag: float
    signature: str


def _clean_topology(mesh: trimesh.Trimesh) -> None:
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    elif hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    elif hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()


def _connected_faces(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    nodes = np.arange(len(mesh.faces), dtype=np.int64)
    groups = trimesh.graph.connected_components(mesh.face_adjacency, nodes=nodes, min_len=1)
    return [np.asarray(group, np.int64) for group in groups]


def _signature(faces: int, area_fraction: float, centroid: np.ndarray, extent: np.ndarray) -> str:
    payload = "|".join(
        [
            str(faces),
            f"{area_fraction:.8f}",
            *(f"{float(value):.6f}" for value in centroid),
            *(f"{float(value):.6f}" for value in extent),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _component_records(
    mesh: trimesh.Trimesh,
    groups: list[np.ndarray],
    model_diagonal: float,
) -> tuple[list[Component], int]:
    faces = np.asarray(mesh.faces, np.int64)
    vertices = np.asarray(mesh.vertices, np.float64)
    area_faces = np.asarray(mesh.area_faces, np.float64)
    total_area = max(float(area_faces.sum()), 1e-12)
    total_faces = max(len(faces), 1)
    records: list[Component] = []
    for component_id, face_ids in enumerate(groups):
        vertex_ids = np.unique(faces[face_ids])
        points = vertices[vertex_ids]
        low, high = points.min(axis=0), points.max(axis=0)
        extent = high - low
        ordered = np.sort(extent)[::-1]
        area = float(area_faces[face_ids].sum())
        area_fraction = area / total_area
        centroid = points.mean(axis=0)
        records.append(
            Component(
                component_id=component_id,
                face_ids=face_ids,
                faces=int(len(face_ids)),
                face_fraction=len(face_ids) / total_faces,
                area=area,
                area_fraction=area_fraction,
                centroid=centroid,
                extent=extent,
                elongation=float(ordered[0] / max(ordered[2], model_diagonal * 1e-9)),
                flatness=float(ordered[1] / max(ordered[2], model_diagonal * 1e-9)),
                nearest_distance_diag=0.0,
                signature=_signature(len(face_ids), area_fraction, centroid, extent),
            )
        )
    main_id = max(records, key=lambda item: item.faces).component_id
    main_vertex_ids = np.unique(faces[records[main_id].face_ids])
    main_tree = cKDTree(vertices[main_vertex_ids])
    updated: list[Component] = []
    for record in records:
        if record.component_id == main_id:
            nearest = 0.0
        else:
            vertex_ids = np.unique(faces[record.face_ids])
            points = vertices[vertex_ids]
            step = max(1, len(points) // 256)
            distance, _ = main_tree.query(points[::step], k=1, workers=-1)
            nearest = float(np.min(distance)) / model_diagonal
        updated.append(replace(record, nearest_distance_diag=nearest))
    return updated, main_id


def _component_points(
    mesh: trimesh.Trimesh,
    component: Component,
    config: AuditConfig,
    seed: int,
) -> np.ndarray:
    count = round(config.total_samples * component.area_fraction)
    count = min(config.max_component_samples, max(config.min_component_samples, count))
    submesh = mesh.submesh([component.face_ids], append=True, repair=False)
    return sample_surface(submesh, count, seed + component.component_id).points


def _basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(direction, np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    up_hint = np.asarray((0.0, 0.0, 1.0), np.float64)
    if abs(float(np.dot(axis, up_hint))) > 0.92:
        up_hint = np.asarray((0.0, 1.0, 0.0), np.float64)
    right = np.cross(up_hint, axis)
    right /= max(np.linalg.norm(right), 1e-12)
    return right, np.cross(axis, right), axis


def _project(
    points: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    half_extent: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    right, up, axis = _basis(direction)
    relative = points - center
    px = np.rint(((relative @ right) / half_extent * 0.5 + 0.5) * (size - 1)).astype(np.int32)
    py = np.rint((1.0 - ((relative @ up) / half_extent * 0.5 + 0.5)) * (size - 1)).astype(np.int32)
    depth = relative @ axis
    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    flat = py[valid] * size + px[valid]
    depth_buffer = np.full(size * size, np.nan, np.float64)
    if len(flat):
        order = np.argsort(flat)
        sorted_flat = flat[order]
        sorted_depth = depth[valid][order]
        unique, starts = np.unique(sorted_flat, return_index=True)
        depth_buffer[unique] = np.maximum.reduceat(sorted_depth, starts)
    depth_buffer = depth_buffer.reshape((size, size))
    raw_mask = np.isfinite(depth_buffer)
    mask = cv2.dilate(raw_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    return mask, depth_buffer


def _read_source_mask(source_image: str | Path | None) -> np.ndarray | None:
    if not source_image:
        return None
    image = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 4:
        return None
    mask = image[..., 3] > 40
    coverage = float(mask.mean())
    return mask if 0.02 <= coverage <= 0.95 else None


def _source_support(
    component_mask: np.ndarray,
    complete_mask: np.ndarray,
    source_mask: np.ndarray | None,
) -> float:
    if source_mask is None or not component_mask.any() or not complete_mask.any():
        return 0.0
    model_y, model_x = np.nonzero(complete_mask)
    source_y, source_x = np.nonzero(source_mask)
    if not len(model_x) or not len(source_x):
        return 0.0
    mx0, mx1 = model_x.min(), model_x.max() + 1
    my0, my1 = model_y.min(), model_y.max() + 1
    sx0, sx1 = source_x.min(), source_x.max() + 1
    sy0, sy1 = source_y.min(), source_y.max() + 1
    crop = component_mask[my0:my1, mx0:mx1].astype(np.uint8)
    resized = cv2.resize(crop, (max(1, sx1 - sx0), max(1, sy1 - sy0)), interpolation=cv2.INTER_NEAREST) > 0
    target = source_mask[sy0:sy1, sx0:sx1]
    pixels = int(resized.sum())
    return float((resized & target).sum() / pixels * 100.0) if pixels else 0.0


def _projection_metrics(
    samples: dict[int, np.ndarray],
    main_id: int,
    center: np.ndarray,
    half_extent: float,
    model_diagonal: float,
    source_mask: np.ndarray | None,
    config: AuditConfig,
) -> dict[int, dict]:
    metrics = {
        component_id: {
            "visible_views": 0,
            "island_views": 0,
            "gap_views": 0,
            "overlap_views": 0,
            "depth_separated_views": 0,
            "visible_pixels": 0,
            "outside_pixels": 0,
            "depth_gaps": [],
            "source_support_percent": 0.0,
        }
        for component_id in samples
        if component_id != main_id
    }
    front_index = 3  # VIEW_DIRECTIONS[3] == (0, -1, 0)
    for view_index, direction in enumerate(VIEW_DIRECTIONS):
        main_mask, main_depth = _project(
            samples[main_id], direction, center, half_extent, config.render_size
        )
        dilated_main = cv2.dilate(main_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        distance_to_main = cv2.distanceTransform((~main_mask).astype(np.uint8), cv2.DIST_L2, 3)
        projected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        complete_mask = main_mask.copy()
        for component_id, points in samples.items():
            if component_id == main_id:
                continue
            component_mask, component_depth = _project(
                points, direction, center, half_extent, config.render_size
            )
            projected[component_id] = component_mask, component_depth
            complete_mask |= component_mask

        for component_id, (component_mask, component_depth) in projected.items():
            if not component_mask.any():
                continue
            item = metrics[component_id]
            visible = int(component_mask.sum())
            item["visible_views"] += 1
            item["visible_pixels"] += visible
            item["outside_pixels"] += int((component_mask & ~dilated_main).sum())
            if float(distance_to_main[component_mask].min()) >= 2.0:
                item["gap_views"] += 1

            overlap = component_mask & main_mask
            if overlap.any():
                item["overlap_views"] += 1
                finite = overlap & np.isfinite(component_depth) & np.isfinite(main_depth)
                if finite.any():
                    gap = float(np.median(np.abs(component_depth[finite] - main_depth[finite]))) / model_diagonal
                    item["depth_gaps"].append(gap)
                    if gap >= config.hover_depth_gap_diag:
                        item["depth_separated_views"] += 1

            labels = cv2.connectedComponents((component_mask | main_mask).astype(np.uint8), 8)[1]
            component_labels = set(np.unique(labels[component_mask])) - {0}
            main_labels = set(np.unique(labels[main_mask])) - {0}
            if component_labels.isdisjoint(main_labels):
                item["island_views"] += 1
            if view_index == front_index:
                item["source_support_percent"] = _source_support(
                    component_mask, complete_mask, source_mask
                )

    for item in metrics.values():
        visible = max(int(item["visible_pixels"]), 1)
        item["aggregate_outside_percent"] = item["outside_pixels"] / visible * 100.0
        item["median_depth_gap_diag"] = (
            float(np.median(item["depth_gaps"])) if item["depth_gaps"] else 0.0
        )
        del item["outside_pixels"]
        del item["depth_gaps"]
    return metrics


def _decision(
    component: Component,
    projection: dict,
    main_id: int,
    family: AssetFamily,
    config: AuditConfig,
) -> dict:
    if component.component_id == main_id:
        action, reason = "KEEP_CONFIRMED", "largest connected surface"
    else:
        attached = component.nearest_distance_diag <= config.attach_distance_diag
        meaningful = (
            component.area_fraction >= config.meaningful_fraction
            or component.face_fraction >= config.meaningful_fraction
        )
        structured = component.elongation >= 6.0
        support_admissible = not (
            projection["island_views"] >= config.outboard_views
            and projection["aggregate_outside_percent"] >= 80.0
        )
        source_support = projection["source_support_percent"] if support_admissible else 0.0
        if attached:
            action, reason = "KEEP_CONFIRMED", "within scale-relative attachment distance"
        elif meaningful:
            action, reason = "KEEP_CONFIRMED", "meaningful share of master surface"
        elif source_support >= config.source_keep_percent:
            action, reason = "KEEP_CONFIRMED", "admissible source-foreground support"
        elif structured and component.nearest_distance_diag < 0.03:
            action, reason = "KEEP_CONFIRMED", "nearby elongated or structured component"
        else:
            outboard = (
                component.nearest_distance_diag >= config.detached_distance_diag
                and source_support < 1.0
                and projection["island_views"] >= config.outboard_views
                and projection["aggregate_outside_percent"] >= config.outboard_percent
                and projection["gap_views"] >= config.gap_views
            )
            strict_outboard = (
                component.nearest_distance_diag >= config.detached_distance_diag
                and source_support < 1.0
                and projection["island_views"] >= 2
                and projection["aggregate_outside_percent"] >= 95.0
                and projection["gap_views"] >= 2
                and component.area_fraction <= 0.01
            )
            hover = (
                component.nearest_distance_diag >= config.hover_distance_diag
                and source_support < 1.0
                and projection["depth_separated_views"] >= config.hover_views
                and projection["overlap_views"] >= config.hover_views
                and projection["median_depth_gap_diag"] >= config.hover_depth_gap_diag
                and component.area_fraction <= config.hover_max_area_fraction
                and not structured
            )
            internal = (
                family in {AssetFamily.ORGANIC, AssetFamily.MIXED}
                and projection["visible_views"] == 0
                and component.nearest_distance_diag >= config.detached_distance_diag
                and component.area_fraction <= config.internal_max_area_fraction
            )
            if outboard or strict_outboard:
                action, reason = "REMOVE_CONFIRMED_DEBRIS", "detached outboard island in multiple views"
            elif hover:
                action, reason = "REMOVE_CONFIRMED_DEBRIS", "depth-separated surface hovering over main body"
            elif internal:
                action, reason = "REMOVE_CONFIRMED_DEBRIS", "small detached internal component invisible in every view"
            else:
                suspicious = (
                    projection["island_views"] >= 2
                    or projection["depth_separated_views"] >= 3
                    or projection["aggregate_outside_percent"] >= 70.0
                )
                action = "AUDIT_REQUIRED" if suspicious else "KEEP_AMBIGUOUS"
                reason = "visible evidence inconclusive; fail closed" if suspicious else "no destructive rule matched"
    return {
        "component_id": component.component_id,
        "signature": component.signature,
        "faces": component.faces,
        "face_fraction": component.face_fraction,
        "area_fraction": component.area_fraction,
        "elongation": component.elongation,
        "flatness": component.flatness,
        "nearest_distance_diag": component.nearest_distance_diag,
        "action": action,
        "reason": reason,
        "projection": projection,
    }


def audit_pass(
    mesh: trimesh.Trimesh,
    family: AssetFamily,
    source_mask: np.ndarray | None,
    config: AuditConfig,
    seed: int,
) -> tuple[dict, np.ndarray]:
    bounds = np.asarray(mesh.bounds, np.float64)
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    model_diagonal = max(float(np.linalg.norm(extent)), 1e-12)
    half_extent = max(float(extent.max()) * 0.60, 1e-9)
    groups = _connected_faces(mesh)
    components, main_id = _component_records(mesh, groups, model_diagonal)
    samples = {
        component.component_id: _component_points(mesh, component, config, seed)
        for component in components
    }
    projections = _projection_metrics(
        samples, main_id, center, half_extent, model_diagonal, source_mask, config
    )
    zero_projection = {
        "visible_views": 0,
        "island_views": 0,
        "gap_views": 0,
        "overlap_views": 0,
        "depth_separated_views": 0,
        "visible_pixels": 0,
        "source_support_percent": 0.0,
        "aggregate_outside_percent": 0.0,
        "median_depth_gap_diag": 0.0,
    }
    decisions = [
        _decision(
            component,
            projections.get(component.component_id, zero_projection),
            main_id,
            family,
            config,
        )
        for component in components
    ]
    remove_ids = {
        decision["component_id"]
        for decision in decisions
        if decision["action"] == "REMOVE_CONFIRMED_DEBRIS"
    }
    removal = (
        np.concatenate([components[component_id].face_ids for component_id in sorted(remove_ids)])
        if remove_ids
        else np.empty(0, np.int64)
    )
    report = {
        "faces": int(len(mesh.faces)),
        "components": len(components),
        "main_component_id": main_id,
        "main_component_faces": components[main_id].faces,
        "model_diagonal": model_diagonal,
        "removed_component_ids": sorted(remove_ids),
        "removed_faces": int(len(removal)),
        "audit_required_count": sum(
            decision["action"] == "AUDIT_REQUIRED" for decision in decisions
        ),
        "decisions": decisions,
    }
    return report, removal


def audit_and_cleanup(
    input_path: str | Path,
    output_path: str | Path,
    *,
    asset_type: str,
    source_image: str | Path | None = None,
    config: AuditConfig = AuditConfig(),
    seed: int = 0,
) -> dict:
    source = load_mesh(input_path)
    _clean_topology(source)
    family = family_for_asset_type(asset_type)
    source_mask = _read_source_mask(source_image)
    before = topology_counts(source)
    current = source.copy()
    passes = []
    removed_total = 0
    initial_main_faces: int | None = None

    for index in range(config.max_passes):
        report, removal = audit_pass(current, family, source_mask, config, seed + index * 1000)
        report["pass"] = index + 1
        passes.append(report)
        if initial_main_faces is None:
            initial_main_faces = int(report["main_component_faces"])
        if not len(removal):
            break
        keep = np.ones(len(current.faces), dtype=bool)
        keep[removal] = False
        current = current.submesh([np.flatnonzero(keep)], append=True, repair=False)
        current.remove_unreferenced_vertices()
        removed_total += int(len(removal))

    final_audit, remaining_removal = audit_pass(
        current, family, source_mask, config, seed + 99_000
    )
    after = topology_counts(current)
    errors = []
    if len(remaining_removal):
        errors.append("cleanup did not converge within max_passes")
    if initial_main_faces is not None and final_audit["main_component_faces"] != initial_main_faces:
        errors.append(
            f"main component changed {initial_main_faces} -> {final_audit['main_component_faces']}"
        )
    if after["boundary_edges"] > before["boundary_edges"]:
        errors.append("boundary-edge count increased")
    if after["non_manifold_edges"] > before["non_manifold_edges"]:
        errors.append("non-manifold-edge count increased")
    if final_audit["audit_required_count"]:
        errors.append(
            f"{final_audit['audit_required_count']} visible components remain audit-required"
        )

    output = Path(output_path)
    if not errors:
        output.parent.mkdir(parents=True, exist_ok=True)
        current.export(output)
    return {
        "success": not errors,
        "input": str(input_path),
        "output": str(output),
        "asset_type": asset_type,
        "asset_family": family.value,
        "config": {
            field: getattr(config, field) for field in config.__dataclass_fields__
        },
        "topology_before": before,
        "topology_after": after,
        "faces_removed": removed_total,
        "faces_removed_percent": removed_total / max(before["faces"], 1) * 100.0,
        "main_component_faces_before": initial_main_faces,
        "main_component_faces_after": final_audit["main_component_faces"],
        "passes": passes,
        "final_audit": final_audit,
        "errors": errors,
    }
