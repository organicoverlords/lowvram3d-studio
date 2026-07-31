"""Dense reference implementation of the projection metrics, for tests only.

This is the pre-rewrite production code. It materialises one HxW mask and one HxW float64
depth image per component simultaneously, which exhausts memory on real high-resolution
masters (roughly 276 GiB resident for the 223,679-component bird master), so it is kept OUT
of ``src`` and off every production path. It exists purely as the equivalence oracle for the
sparse implementation, on tiny synthetic meshes.
"""
from __future__ import annotations

import cv2
import numpy as np

from lowvram3d.component_audit import AuditConfig, _project, _source_support
from lowvram3d.geometry_compare import VIEW_DIRECTIONS


def _projection_metrics_dense(
    samples: dict[int, np.ndarray],
    main_id: int,
    center: np.ndarray,
    half_extent: float,
    model_diagonal: float,
    source_mask: np.ndarray | None,
    config: AuditConfig,
) -> dict[int, dict]:
    """Reference implementation retained for equivalence tests only.

    NOT on any production path: it holds one HxW mask and one HxW float64 depth image per
    component simultaneously, which exhausts memory on real high-resolution masters. Use only
    on tiny synthetic meshes, as the equivalence oracle for ``_projection_metrics``.
    """
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
    front_index = 3
    for view_index, direction in enumerate(VIEW_DIRECTIONS):
        main_mask, main_depth = _project(samples[main_id], direction, center, half_extent, config.render_size)
        dilated_main = cv2.dilate(main_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        distance_to_main = cv2.distanceTransform((~main_mask).astype(np.uint8), cv2.DIST_L2, 3)
        projected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        complete_mask = main_mask.copy()
        for component_id, points in samples.items():
            if component_id == main_id:
                continue
            component_mask, component_depth = _project(points, direction, center, half_extent, config.render_size)
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
                item["source_support_percent"] = _source_support(component_mask, complete_mask, source_mask)

    for item in metrics.values():
        visible = max(int(item["visible_pixels"]), 1)
        item["aggregate_outside_percent"] = item["outside_pixels"] / visible * 100.0
        item["median_depth_gap_diag"] = float(np.median(item["depth_gaps"])) if item["depth_gaps"] else 0.0
        del item["outside_pixels"]
        del item["depth_gaps"]
    return metrics
