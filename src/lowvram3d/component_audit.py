"""Generic deterministic connected-component audit for clean high-resolution masters.

The audit runs before decimation.  It combines 3D proximity, component geometry, fourteen projected
views, outboard-island evidence and depth separation.  It never removes a component from face count
alone and fails closed when a visible component remains suspicious but unproven.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
    total_surface_samples: int = 220_000
    minimum_component_samples: int = 128
    maximum_component_samples: int = 8192
    max_passes: int = 4
    attach_distance_diag: float = 0.0015
    minimum_detached_distance_diag: float = 0.0020
    hover_distance_diag: float = 0.0040
    hover_depth_gap_diag: float = 0.0060
    meaningful_area_fraction: float = 0.03
    meaningful_face_fraction: float = 0.03
    source_support_keep_percent: float = 10.0
    outboard_outside_percent: float = 85.0
    outboard_island_views: int = 3
    outboard_gap_views: int = 3
    hover_separated_views: int = 5
    hover_max_area_fraction: float = 0.015
    internal_max_area_fraction: float = 0.002


@dataclass(frozen=True, slots=True)
class ComponentGeometry:
    component_id: int
    face_ids: np.ndarray
    face_count: int
    face_fraction: float
    area: float
    area_fraction: float
    centroid: np.ndarray
    bbox_extent: np.ndarray
    bbox_diagonal: float
    elongation: float
    flatness: float
    compactness: float
    nearest_distance_diag: float
    signature: str


@dataclass(frozen=True, slots=True)
class ProjectedComponent:
    visible_views: int
    island_views: int
    gap_views: int
    overlap_views: int
    depth_separated_views: int
    aggregate_outside_percent: float
    median_depth_gap_diag: float
    source_support_percent: float
    total_visible_pixels: int


@dataclass(frozen=True, slots=True)
class ComponentDecision:
    component_id: int
    signature: str
    action: str
    reason: str
    faces: int
    area_fraction: float
    nearest_distance_diag: float
    projected: ProjectedComponent

    @property
    def removable(self) -> bool:
        return self.action == "REMOVE_CONFIRMED_DEBRIS"

    @property
    def risky_ambiguous(self) -> bool:
        return self.action == "AUDIT_REQUIRED"

    def as_dict(self) -> dict:
        return {
            "component_id": self.component_id,
            "signature": self.signature,
            "action": self.action,
            "reason": self.reason,
            "faces": self.faces,
            "area_fraction": self.area_fraction,
            "nearest_distance_diag": self.nearest_distance_diag,
            "projected": {
                "visible_views": self.projected.visible_views,
                "island_views": self.projected.island_views,
                "gap_views": self.projected.gap_views,
                "overlap_views": self.projected.overlap_views,
                "depth_separated_views": self.projected.depth_separated_views,
                "aggregate_outside_percent": self.projected.aggregate_outside_percent,
                "median_depth_gap_diag": self.projected.median_depth_gap_diag,
                "source_support_percent": self.projected.source_support_percent,
                "total_visible_pixels": self.projected.total_visible_pixels,
            },
        }


def _face_components(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    nodes = np.arange(len(mesh.faces), dtype=np.int64)
    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=nodes,
        min_len=1,
    )
    return [np.asarray(component, np.int64) for component in components]


def _component_signature(
    face_count: int,
    area_fraction: float,
    centroid: np.ndarray,
    extent: np.ndarray,
) -> str:
    payload = "|".join(
        [
            str(face_count),
            f"{area_fraction:.8f}",
            *(f"{value:.6f}" for value in centroid),
            *(f"{value:.6f}" for value in extent),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _component_geometry(
    mesh: trimesh.Trimesh,
    components: list[np.ndarray],
    model_diagonal: float,
) -> tuple[list[ComponentGeometry], int]:
    area_faces = np.asarray(mesh.area_faces, np.float64)
    total_area = max(float(area_faces.sum()), 1e-12)
    total_faces = max(len(mesh.faces), 1)
    records = []
    for component_id, face_ids in enumerate(components):
        vertex_ids = np.unique(np.asarray(mesh.faces, np.int64)[face_ids])
        points = np.asarray(mesh.vertices, np.float64)[vertex_ids]
        low, high = points.min(axis=0), points.max(axis=0)
        extent = high - low
        sorted_extent = np.sort(extent)[::-1]
        area = float(area_faces[face_ids].sum())
        area_fraction = area / total_area
        centroid = points.mean(axis=0)
        bbox_diagonal = float(np.linalg.norm(extent))
        elongation = float(sorted_extent[0] / max(sorted_extent[2], model_diagonal * 1e-9))
        flatness = float(sorted_extent[1] / max(sorted_extent[2], model_diagonal * 1e-9))
        compactness = float(area / max(bbox_diagonal * bbox_diagonal, 1e-12))
        records.append(
            ComponentGeometry(
                component_id=component_id,
                face_ids=face_ids,
                face_count=int(len(face_ids)),
                face_fraction=len(face_ids) / total_faces,
                area=area,
                area_fraction=area_fraction,
                centroid=centroid,
                bbox_extent=extent,
                bbox_diagonal=bbox_diagonal,
                elongation=elongation,
                flatness=flatness,
                compactness=compactness,
                nearest_distance_diag=0.0,
                signature=_component_signature(len(face_ids), area_fraction, centroid, extent),
            )
        )
    main_id = max(records, key=lambda record: record.face_count).component_id
    main_faces = records[main_id].face_ids
    main_vertex_ids = np.unique(np.asarray(mesh.faces, np.int64)[main_faces])
    main_tree = cKDTree(np.asarray(mesh.vertices, np.float64)[main_vertex_ids])
    updated = []
    for record in records:
        if record.component_id == main_id:
            nearest = 0.0
        else:
            vertex_ids = np.unique(np.asarray(mesh.faces, np.int64)[record.face_ids])
            points = np.asarray(mesh.vertices, np.float64)[vertex_ids]
            step = max(1, len(points) // 256)
            distance, _ = main_tree.query(points[::step], k=1, workers=-1)
            nearest = float(np.min(distance)) / model_diagonal
        updated.append(
            ComponentGeometry(
                **{
                    **record.__dict__,
                    "nearest_distance_diag": nearest,
                }
            )
        )
    return updated, main_id


def _sample_component(
    mesh: trimesh.Trimesh,
    geometry: ComponentGeometry,
    config: AuditConfig,
    seed: int,
) -> np.ndarray:
    sample_count = round(config.total_surface_samples * geometry.area_fraction)
    sample_count = min(config.maximum_component_samples, max(config.minimum_component_samples, sample_count))
    submesh = mesh.submesh([geometry.face_ids], append=True, repair=False)
    return sample_surface(submesh, sample_count, seed + geometry.component_id).points


def _camera_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_axis = np.asarray(direction, np.float64)
    camera_axis /= max(np.linalg.norm(camera_axis), 1e-12)
    up_hint = np.asarray((0.0, 0.0, 1.0), np.float64)
    if abs(float(np.dot(camera_axis, up_hint))) > 0.92:
        up_hint = np.asarray((0.0, 1.0, 0.0), np.float64)
    right = np.cross(up_hint, camera_axis)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(camera_axis, right)
    return right, up, camera_axis


def _project_points(
    points: np.ndarray,
    *,
    direction: np.ndarray,
    center: np.ndarray,
    half_extent: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    right, up, camera_axis = _camera_basis(direction)
    relative = points - center
    px = np.rint(((relative @ right) / half_extent * 0.5 + 0.5) * (size - 1)).astype(np.int32)
    py = np.rint((1.0 - ((relative @ up) / half_extent * 0.5 + 0.5)) * (size - 1)).astype(np.int32)
    depth = relative @ camera_axis
    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    flat = py[valid] * size + px[valid]
    depth_buffer = np.full(size * size, np.nan, np.float64)
    if flat.size:
        order = np.argsort(flat)
        sorted_flat = flat[order]
        sorted_depth = depth[valid][order]
        unique, starts = np.unique(sorted_flat, return_index=True)
        # Camera lies in +direction; the point with maximum axis depth is closest to it.
        maxima = np.maximum.reduceat(sorted_depth, starts)
        depth_buffer[unique] = maxima
    depth_buffer = depth_buffer.reshape((size, size))
    mask = np.isfinite(depth_buffer)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    return mask, depth_buffer


def _source_mask(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 4:
        return None
    alpha = image[..., 3]
    mask = alpha > 40
    coverage = float(mask.mean())
    return mask if 0.02 <= coverage <= 0.95 else None


def _fit_source_support(
    component_mask: np.ndarray,
    complete_mask: np.ndarray,
    source_mask: np.ndarray | None,
) -> float:
    if source_mask is None or not component_mask.any() or not complete_mask.any():
        return 0.0
    ys, xs = np.nonzero(complete_mask)
    sy, sx = np.nonzero(source_mask)
    if not len(xs) or not len(sx):
        return 0.0
    model_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    source_box = (sx.min(), sy.min(), sx.max() + 1, sy.max() + 1)
    crop = component_mask[model_box[1]:model_box[3], model_box[0]:model_box[2]].astype(np.uint8)
    width = max(1, source_box[2] - source_box[0])
    height = max(1, source_box[3] - source_box[1])
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    target = source_mask[source_box[1]:source_box[3], source_box[0]:source_box[2]]
    pixels = int(resized.sum())
    return float((resized & target).sum() / pixels * 100.0) if pixels else 0.0


def _projected_metrics(
    samples: dict[int, np.ndarray],
    main_id: int,
    *,
    center: np.ndarray,
    half_extent: float,
    model_diagonal: float,
    config: AuditConfig,
    source_mask: np.ndarray | None,
) -> dict[int, ProjectedComponent]:
    accumulator = {
        component_id: {
            "visible_views": 0,
            "island_views": 0,
            "gap_views": 0,
            "overlap_views": 0,
            "depth_separated_views": 0,
            "outside_pixels": 0,
            "visible_pixels": 0,
            "depth_gaps": [],
            "source_support": 0.0,
        }
        for component_id in samples
        if component_id != main_id
    }
    front_index = 3  # direction (0, -1, 0)
    for view_index, direction in enumerate(VIEW_DIRECTIONS):
        main_mask, main_depth = _project_points(
            samples[main_id],
            direction=direction,
            center=center,
            half_extent=half_extent,
            size=config.render_size,
        )
        dilated_main = cv2.dilate(main_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        distance = cv2.distanceTransform((~main_mask).astype(np.uint8), cv2.DIST_L2, 3)
        complete_mask = main_mask.copy()
        projected = {}
        for component_id, points in samples.items():
            if component_id == main_id:
                continue
            component_mask, component_depth = _project_points(
                points,
                direction=direction,
                center=center,
                half_extent=half_extent,
                size=config.render_size,
            )
            projected[component_id] = (component_mask, component_depth)
            complete_mask |= component_mask

        for component_id, (component_mask, component_depth) in projected.items():
            data = accumulator[component_id]
            visible = int(component_mask.sum())
            if visible == 0:
                continue
            data["visible_views"] += 1
            data["visible_pixels"] += visible
            outside = component_mask & ~dilated_main
            data["outside_pixels"] += int(outside.sum())
            overlap = component_mask & main_mask
            if overlap.any():
                data["overlap_views"] += 1
                both_depth = overlap & np.isfinite(component_depth) & np.isfinite(main_depth)
                if both_depth.any():
                    gaps = np.abs(component_depth[both_depth] - main_depth[both_depth]) / model_diagonal
                    median_gap = float(np.median(gaps))
                    data["depth_gaps"].append(median_gap)
                    if median_gap >= config.hover_depth_gap_diag:
                        data["depth_separated_views"] += 1
            if float(distance[component_mask].min(initial=0.0)) >= 2.0:
                data["gap_views"] += 1
            connected = cv2.connectedComponents((component_mask | main_mask).astype(np.uint8), 8)[1]
            component_labels = set(np.unique(connected[component_mask])) - {0}
            main_labels = set(np.unique(connected[main_mask])) - {0}
            if component_labels.isdisjoint(main_labels):
                data["island_views"] += 1
            if view_index == front_index:
                data["source_support"] = _fit_source_support(component_mask, complete_mask, source_mask)

    result = {}
    for component_id, data in accumulator.items():
        visible = max(int(data["visible_pixels"]), 1)
        result[component_id] = ProjectedComponent(
            visible_views=int(data["visible_views"]),
            island_views=int(data["island_views"]),
            gap_views=int(data["gap_views"]),
            overlap_views=int(data["overlap_views"]),
            depth_separated_views=int(data["depth_separated_views"]),
            aggregate_outside_percent=float(data["outside_pixels"]) / visible * 100.0,
            median_depth_gap_diag=float(np.median(data["depth_gaps"])) if data["depth_gaps"] else 0.0,
            source_support_percent=float(data["source_support"]),
            total_visible_pixels=int(data["visible_pixels"]),
        )
    return result


def _classify(
    geometry: ComponentGeometry,
    projected: ProjectedComponent,
    *,
    main_id: int,
    family: AssetFamily,
    config: AuditConfig,
) -> ComponentDecision:
    if geometry.component_id == main_id:
        return ComponentDecision(
            geometry.component_id,
            geometry.signature,
            "KEEP_CONFIRMED",
            "largest connected surface",
            geometry.face_count,
            geometry.area_fraction,
            geometry.nearest_distance_diag,
            projected,
        )

    attached = geometry.nearest_distance_diag <= config.attach_distance_diag
    meaningful = (
        geometry.area_fraction >= config.meaningful_area_fraction
        or geometry.face_fraction >= config.meaningful_face_fraction
    )
    structured = geometry.elongation >= 6.0
    front_support_admissible = not (
        projected.island_views >= config.outboard_island_views
        and projected.aggregate_outside_percent >= 80.0
    )
    effective_source_support = (
        projected.source_support_percent if front_support_admissible else 0.0
    )

    if attached:
        action, reason = "KEEP_CONFIRMED", "surface is within the scale-relative attachment distance"
    elif meaningful:
        action, reason = "KEEP_CONFIRMED", "component is a meaningful share of the clean master"
    elif effective_source_support >= config.source_support_keep_percent:
        action, reason = "KEEP_CONFIRMED", "component has admissible source-foreground support"
    elif structured and geometry.nearest_distance_diag < 0.03:
        action, reason = "KEEP_CONFIRMED", "nearby elongated/structured component"
    else:
        outboard = (
            geometry.nearest_distance_diag >= config.minimum_detached_distance_diag
            and effective_source_support < 1.0
            and projected.island_views >= config.outboard_island_views
            and projected.aggregate_outside_percent >= config.outboard_outside_percent
            and projected.gap_views >= config.outboard_gap_views
        )
        strict_outboard = (
            geometry.nearest_distance_diag >= config.minimum_detached_distance_diag
            and effective_source_support < 1.0
            and projected.island_views >= 2
            and projected.aggregate_outside_percent >= 95.0
            and projected.gap_views >= 2
            and geometry.area_fraction <= 0.01
        )
        hover = (
            geometry.nearest_distance_diag >= config.hover_distance_diag
            and effective_source_support < 1.0
            and projected.depth_separated_views >= config.hover_separated_views
            and projected.overlap_views >= config.hover_separated_views
            and projected.median_depth_gap_diag >= config.hover_depth_gap_diag
            and geometry.area_fraction <= config.hover_max_area_fraction
            and not structured
        )
        removable_internal = (
            family in {AssetFamily.ORGANIC, AssetFamily.MIXED}
            and projected.visible_views == 0
            and geometry.nearest_distance_diag >= config.minimum_detached_distance_diag
            and geometry.area_fraction <= config.internal_max_area_fraction
        )
        if outboard or strict_outboard:
            action, reason = "REMOVE_CONFIRMED_DEBRIS", "detached outboard island in multiple views"
        elif hover:
            action, reason = "REMOVE_CONFIRMED_DEBRIS", "detached depth-separated surface hovering over the main body"
        elif removable_internal:
            action, reason = "REMOVE_CONFIRMED_DEBRIS", "small detached internal component invisible in all audit views"
        else:
            suspicious = (
                projected.island_views >= 2
                or projected.depth_separated_views >= 3
                or projected.aggregate_outside_percent >= 70.0
            )
            action = "AUDIT_REQUIRED" if suspicious else "KEEP_AMBIGUOUS"
            reason = (
                "visible detached evidence is inconclusive; fail closed"
                if suspicious
                else "no destructive rule matched"
            )

    return ComponentDecision(
        geometry.component_id,
        geometry.signature,
        action,
        reason,
        geometry.face_count,
        geometry.area_fraction,
        geometry.nearest_distance_diag,
        projected,
    )


def audit_once(
    mesh: trimesh.Trimesh,
    *,
    family: AssetFamily,
    source_mask: np.ndarray | None,
    config: AuditConfig,
    seed: int,
) -> tuple[list[ComponentDecision], np.ndarray, dict]:
    bounds = np.asarray(mesh.bounds, np.float64)
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    model_diagonal = max(float(np.linalg.norm(extent)), 1e-12)
    half_extent = max(float(extent.max()) * 0.60, 1e-9)
    components = _face_components(mesh)
    geometry, main_id = _component_geometry(mesh, components, model_diagonal)
    samples = {
        record.component_id: _sample_component(mesh, record, config, seed)
        for record in geometry
    }
    projected = _projected_metrics(
        samples,
        main_id,
        center=center,
        half_extent=half_extent,
        model_diagonal=model_diagonal,
        config=config,
        source_mask=source_mask,
    )
    zero = ProjectedComponent(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0)
    decisions = [
        _classify(
            record,
            projected.get(record.component_id, zero),
            main_id=main_id,
            family=family,
            config=config,
        )
        for record in geometry
    ]
    remove_ids = {decision.component_id for decision in decisions if decision.removable}
    remove_faces = np.concatenate(
        [geometry[component_id].face_ids for component_id in sorted(remove_ids)]
    ) if remove_ids else np.empty(0, np.int64)
    report = {
        "faces": int(len(mesh.faces)),
        "component_count": len(components),
        "main_component_id": main_id,
        "main_component_faces": geometry[main_id].face_count,
        "model_diagonal": model_diagonal,
        "removed_component_ids": sorted(remove_ids),
        "removed_faces": int(len(remove_faces)),
        "risky_ambiguous_count": sum(decision.risky_ambiguous for decision in decisions),
        "decisions": [decision.as_dict() for decision in decisions],
    }
    return decisions, remove_faces, report


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
    source.merge_vertices(merge_tex=True, merge_norm=True)
    source.remove_degenerate_faces()
    source.remove_duplicate_faces()
    source.remove_unreferenced_vertices()
    family = family_for_asset_type(asset_type)
    alpha_mask = _source_mask(source_image)
    source_topology = topology_counts(source)
    source_main_faces = None
    current = source.copy()
    passes = []
    removed_total = 0

    for pass_index in range(config.max_passes):
        decisions, remove_faces, report = audit_once(
            current,
            family=family,
            source_mask=alpha_mask,
            config=config,
            seed=seed + pass_index * 1000,
        )
        report["pass"] = pass_index + 1
        passes.append(report)
        if source_main_faces is None:
            source_main_faces = report["main_component_faces"]
        if len(remove_faces) == 0:
            break
        keep = np.ones(len(current.faces), bool)
        keep[remove_faces] = False
        current = current.submesh([np.flatnonzero(keep)], append=True, repair=False)
        current.remove_unreferenced_vertices()
        removed_total += int(len(remove_faces))

    final_decisions, final_remove, final_report = audit_once(
        current,
        family=family,
        source_mask=alpha_mask,
        config=config,
        seed=seed + 99_000,
    )
    final_topology = topology_counts(current)
    final_main_faces = final_report["main_component_faces"]
    risky = [decision for decision in final_decisions if decision.risky_ambiguous]
    errors = []
    if len(final_remove):
        errors.append("cleanup did not converge within the bounded pass count")
    if source_main_faces is not None and final_main_faces != source_main_faces:
        errors.append(f"main component changed {source_main_faces} -> {final_main_faces}")
    if final_topology["boundary_edges"] > source_topology["boundary_edges"]:
        errors.append("boundary-edge count increased")
    if final_topology["non_manifold_edges"] > source_topology["non_manifold_edges"]:
        errors.append("non-manifold-edge count increased")
    if risky:
        errors.append(f"{len(risky)} visible components remain audit-required")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not errors:
        current.export(output)
    return {
        "success": not errors,
        "input": str(input_path),
        "output": str(output),
        "asset_type": asset_type,
        "asset_family": family.value,
        "config": {field: getattr(config, field) for field in config.__dataclass_fields__},
        "topology_before": source_topology,
        "topology_after": final_topology,
        "faces_removed": removed_total,
        "faces_removed_percent": removed_total / max(source_topology["faces"], 1) * 100.0,
        "main_component_faces_before": source_main_faces,
        "main_component_faces_after": final_main_faces,
        "passes": passes,
        "final_audit": final_report,
        "risky_components": [decision.as_dict() for decision in risky],
        "errors": errors,
    }
