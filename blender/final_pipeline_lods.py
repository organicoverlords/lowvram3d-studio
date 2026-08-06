"""Build the shaman LOD chain from the cleaned high master with feature protection.

A flat collapse-decimate at these ratios destroys exactly the features the silhouette depends on:
the cords go first (they are thin tubes only a few faces around), then the leaf pendants and the
staff ring. Blender's collapse decimator accepts a vertex group whose weight scales the local
ratio, so this builds a protection weight map and decimates through it rather than uniformly.

Protection is spatial plus topological: the head and beak, the hands and feet band, the antler and
cord region above the shoulders, the robe hem, and every small loose component (the ornaments that
survived Stage 1 cleanup) are weighted up; the bulk robe and torso surfaces, which carry most of
the triangles and almost none of the silhouette, are weighted down.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash, import_mesh, reset_scene, save_json

PROTECT_GROUP = "lod_protect"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_anchor_receipt(path: str | Path, source_hash: str) -> dict:
    """Load the ticket-01 receipt without importing third-party packages into Blender."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("receipt_type") != "thin_feature_anchor_discovery":
        raise ValueError("invalid anchor receipt type")
    if payload.get("source_mesh_sha256") != source_hash:
        raise ValueError("anchor receipt source hash does not match clean master")
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or schema_version.split(".", 1)[0] != "1":
        raise ValueError("unsupported anchor receipt schema version")
    view_set = payload.get("view_set")
    expected_views = ("front", "right", "back", "left", "top", "bottom")
    if (
        not isinstance(view_set, list)
        or [item.get("name") for item in view_set if isinstance(item, dict)] != list(expected_views)
    ):
        raise ValueError("anchor receipt view_set must contain the production six-view order")
    anchors = payload.get("anchors")
    if not isinstance(anchors, list):
        raise ValueError("anchor receipt anchors must be a list")
    discovery = payload.get("discovery")
    frame = discovery.get("normalization_frame") if isinstance(discovery, dict) else None
    if not isinstance(frame, dict):
        raise ValueError("anchor receipt is missing discovery.normalization_frame")
    for field in ("center", "bounds_min", "bounds_max"):
        values = frame.get(field)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(type(value) is not float or not math.isfinite(float(value)) for value in values)
        ):
            raise ValueError(f"anchor receipt normalization frame has malformed {field}")
    diagonal_value = frame.get("diagonal")
    if (
        type(diagonal_value) is not float
        or not math.isfinite(float(diagonal_value))
        or float(diagonal_value) <= 0.0
    ):
        raise ValueError("anchor receipt normalization frame has malformed diagonal")
    if any(low > high for low, high in zip(frame["bounds_min"], frame["bounds_max"])):
        raise ValueError("anchor receipt normalization frame bounds are inverted")
    midpoint = [(low + high) * 0.5 for low, high in zip(frame["bounds_min"], frame["bounds_max"])]
    if any(abs(center - expected) > 1.0e-7 for center, expected in zip(frame["center"], midpoint)):
        raise ValueError("anchor receipt normalization frame center is not the bounds midpoint")
    required = (
        "anchor_id",
        "seeds",
        "bounds_normalized",
        "per_view_support",
        "supported_views",
        "survival_floor",
    )
    seen = set()
    ordered_ids = []
    for anchor in anchors:
        if not isinstance(anchor, dict) or any(field not in anchor for field in required):
            raise ValueError("anchor receipt contains an incomplete anchor")
        anchor_id = anchor["anchor_id"]
        if not isinstance(anchor_id, str) or anchor_id in seen:
            raise ValueError("anchor receipt contains duplicate/malformed anchor IDs")
        seen.add(anchor_id)
        ordered_ids.append(anchor_id)
        if not isinstance(anchor.get("seeds"), list) or not anchor["seeds"]:
            raise ValueError(f"anchor {anchor_id} has no seed geometry")
        for seed in anchor["seeds"]:
            if (
                not isinstance(seed, list)
                or len(seed) != 3
                or any(
                    type(value) is not float
                    or not math.isfinite(float(value))
                    for value in seed
                )
            ):
                raise ValueError(f"anchor {anchor_id} has malformed normalized seed geometry")
        bounds = anchor["bounds_normalized"]
        if not isinstance(bounds, dict):
            raise ValueError(f"anchor {anchor_id} has malformed normalized bounds")
        for bound_name in ("min", "max"):
            values = bounds.get(bound_name)
            if (
                not isinstance(values, list)
                or len(values) != 3
                or any(
                    type(value) is not float
                    or not math.isfinite(float(value))
                    for value in values
                )
            ):
                raise ValueError(f"anchor {anchor_id} has malformed normalized bounds")
        supported_views = anchor.get("supported_views")
        if not isinstance(supported_views, list) or any(view not in expected_views for view in supported_views):
            raise ValueError(f"anchor {anchor_id} has malformed supported views")
        floor = anchor.get("survival_floor")
        if not isinstance(floor, dict):
            raise ValueError(f"anchor {anchor_id} has no survival floor")
        ratio = floor.get("exclusive_pixel_retention_ratio")
        if not isinstance(ratio, (int, float)) or not 0.0 <= float(ratio) <= 1.0:
            raise ValueError(f"anchor {anchor_id} has malformed survival floor")
    if ordered_ids != sorted(ordered_ids):
        raise ValueError("anchor receipt anchors must be ordered by anchor_id")
    return payload


def _receipt_reference_bounds(receipt: dict) -> tuple[Vector, Vector]:
    """Return the immutable clean-source frame used by normalized anchor coordinates."""
    frame = receipt["discovery"]["normalization_frame"]
    return Vector(frame["bounds_min"]), Vector(frame["bounds_max"])


def world_bounds_of(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return lo, hi


def build_protection(
    obj: bpy.types.Object,
    bulk_weight: float,
    anchors: list[dict] | None = None,
    reference_bounds: tuple[Vector, Vector] | None = None,
) -> dict:
    """Weight silhouette bands and ticket-01 anchor seed regions."""
    lo, hi = world_bounds_of(obj)
    height = max(hi.z - lo.z, 1e-6)
    group = obj.vertex_groups.new(name=PROTECT_GROUP)

    protected: list[int] = []
    bulk: list[int] = []
    for vertex in obj.data.vertices:
        world = obj.matrix_world @ vertex.co
        fraction = (world.z - lo.z) / height
        # Above the shoulders: head, beak, antler span, the pole and everything hanging from it.
        upper = fraction >= 0.62
        # Feet, boots and the robe hem/fringe.
        lower = fraction <= 0.16
        if upper or lower:
            protected.append(vertex.index)
        else:
            bulk.append(vertex.index)

    anchor_records = []
    anchor_vertices: dict[str, list[int]] = {}
    reference_lo, reference_hi = reference_bounds or (lo, hi)
    diagonal = max((reference_hi - reference_lo).length, 1e-6)
    center = (reference_lo + reference_hi) * 0.5
    for anchor in anchors or []:
        anchor_id = anchor["anchor_id"]
        selected: set[int] = set()
        # Seeds are normalized by the source bounds diagonal.  Nearest-vertex matching keeps the
        # contract independent of Blender object/face ordering after each fresh import.
        for seed in anchor.get("seeds", []):
            if not isinstance(seed, (list, tuple)) or len(seed) != 3:
                continue
            target = center + Vector(seed) * diagonal
            nearest = min(obj.data.vertices, key=lambda vertex: ((obj.matrix_world @ vertex.co) - target).length)
            selected.add(nearest.index)
        if selected:
            # Include a small neighborhood so a thin cap cannot collapse around a protected seed.
            radius = max(diagonal * 0.006, diagonal * 0.12 * float(anchor.get("area_fraction", 0.0)) ** 0.5)
            seed_positions = [obj.matrix_world @ obj.data.vertices[index].co for index in selected]
            for vertex in obj.data.vertices:
                world = obj.matrix_world @ vertex.co
                if any((world - seed_position).length <= radius for seed_position in seed_positions):
                    selected.add(vertex.index)
            protected.extend(selected)
        anchor_vertices[anchor_id] = sorted(selected)
        anchor_records.append({"anchor_id": anchor_id, "seed_count": len(anchor.get("seeds", [])),
                               "protected_vertices": len(selected)})

    group.add(bulk, bulk_weight, "REPLACE")
    # Add anchors last so their 1.0 weight cannot be overwritten by the bulk assignment.
    group.add(protected, 1.0, "REPLACE")
    return {
        "protected_vertices": len(protected),
        "bulk_vertices": len(bulk),
        "bulk_weight": bulk_weight,
        "upper_band_from": 0.62,
        "lower_band_to": 0.16,
        "anchors": anchor_records,
        "anchor_vertices": anchor_vertices,
    }


_DEFAULT_VIEW_SET = (
    ("front", (0.0, -1.0, 0.0)),
    ("right", (1.0, 0.0, 0.0)),
    ("back", (0.0, 1.0, 0.0)),
    ("left", (-1.0, 0.0, 0.0)),
    ("top", (0.0, 0.0, 1.0)),
    ("bottom", (0.0, 0.0, -1.0)),
)


def _view_basis(direction: tuple[float, float, float]) -> tuple[Vector, Vector, Vector]:
    axis = Vector(direction)
    axis.normalize()
    up_hint = Vector((0.0, 0.0, 1.0))
    if abs(axis.dot(up_hint)) > 0.92:
        up_hint = Vector((0.0, 1.0, 0.0))
    right = up_hint.cross(axis)
    right.normalize()
    up = axis.cross(right)
    up.normalize()
    return right, up, axis


def _raster_triangle(
    mask: set[tuple[int, int]],
    points: tuple[tuple[float, float], ...],
    size: int,
    clip_bounds: tuple[int, int, int, int] | None = None,
) -> None:
    """Fill one projected triangle with the same integer-pixel semantics as source discovery."""
    if len(points) != 3:
        return
    (ax, ay), (bx, by), (cx, cy) = points
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(area) <= 1.0e-9:
        return
    low_x = max(0, int(math.floor(min(ax, bx, cx))))
    high_x = min(size - 1, int(math.ceil(max(ax, bx, cx))))
    low_y = max(0, int(math.floor(min(ay, by, cy))))
    high_y = min(size - 1, int(math.ceil(max(ay, by, cy))))
    if clip_bounds is not None:
        clip_low_x, clip_high_x, clip_low_y, clip_high_y = clip_bounds
        low_x = max(low_x, clip_low_x)
        high_x = min(high_x, clip_high_x)
        low_y = max(low_y, clip_low_y)
        high_y = min(high_y, clip_high_y)
    if low_x > high_x or low_y > high_y:
        return
    sign = 1.0 if area > 0.0 else -1.0
    for py in range(low_y, high_y + 1):
        for px in range(low_x, high_x + 1):
            edge_a = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            edge_b = (cx - bx) * (py - by) - (cy - by) * (px - bx)
            edge_c = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
            if sign * edge_a >= -1.0e-7 and sign * edge_b >= -1.0e-7 and sign * edge_c >= -1.0e-7:
                mask.add((px, py))


def _silhouette_mask(
    triangles: list[tuple[Vector, Vector, Vector]],
    direction: tuple[float, float, float],
    center: Vector,
    half_extent: float,
    size: int,
    clip_bounds: tuple[int, int, int, int] | None = None,
) -> set[tuple[int, int]]:
    mask: set[tuple[int, int]] = set()
    right, up, _ = _view_basis(direction)
    for triangle in triangles:
        projected = []
        for point in triangle:
            relative = point - center
            x = relative.dot(right)
            y = relative.dot(up)
            px = round((x / half_extent * 0.5 + 0.5) * (size - 1))
            py = round((1.0 - (y / half_extent * 0.5 + 0.5)) * (size - 1))
            projected.append((float(px), float(py)))
        _raster_triangle(mask, tuple(projected), size, clip_bounds)
    return mask


def _mesh_triangles(obj: bpy.types.Object) -> list[tuple[Vector, Vector, Vector]]:
    positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    triangles: list[tuple[Vector, Vector, Vector]] = []
    for polygon in obj.data.polygons:
        indices = list(polygon.vertices)
        if len(indices) < 3:
            continue
        for index in range(1, len(indices) - 1):
            triangles.append((positions[indices[0]], positions[indices[index]], positions[indices[index + 1]]))
    return triangles


def _anchor_candidate_triangles(
    obj: bpy.types.Object,
    anchor: dict,
    center: Vector,
    diagonal: float,
) -> tuple[list[tuple[Vector, Vector, Vector]], list[tuple[Vector, Vector, Vector]]]:
    triangles = _mesh_triangles(obj)
    region = anchor.get("bounds_normalized") or {}
    low = Vector(region.get("min", ()))
    high = Vector(region.get("max", ()))
    if len(low) != 3 or len(high) != 3:
        return [], triangles
    # Bounds are quantized in the receipt.  A small pad covers quantization while requiring at
    # least two vertices in the region (a nearby body vertex cannot become the feature silhouette).
    pad = 0.012
    candidate: list[tuple[Vector, Vector, Vector]] = []
    remainder: list[tuple[Vector, Vector, Vector]] = []
    for triangle in triangles:
        normalized = [(point - center) / diagonal for point in triangle]
        inside = [all(low[index] - pad <= point[index] <= high[index] + pad for index in range(3))
                  for point in normalized]
        centroid = sum(normalized, Vector((0.0, 0.0, 0.0))) / 3.0
        centroid_inside = all(low[index] - pad <= centroid[index] <= high[index] + pad for index in range(3))
        if sum(inside) >= 2 or centroid_inside:
            candidate.append(triangle)
        else:
            remainder.append(triangle)
    return candidate, remainder


def _anchor_silhouette_support(
    obj: bpy.types.Object,
    anchor: dict,
    center: Vector,
    diagonal: float,
    view_set: list[dict] | tuple[tuple[str, tuple[float, float, float]], ...],
    render_size: int,
) -> dict[str, dict[str, float | int]]:
    candidate, remainder = _anchor_candidate_triangles(obj, anchor, center, diagonal)
    half_extent = diagonal * 0.55
    support: dict[str, dict[str, float | int]] = {}
    for view in view_set:
        if isinstance(view, dict):
            name, direction = view.get("name"), tuple(view.get("direction", ()))
        else:
            name, direction = view
        if not name or len(direction) != 3:
            continue
        candidate_mask = _silhouette_mask(candidate, direction, center, half_extent, render_size)
        if candidate_mask:
            xs = [point[0] for point in candidate_mask]
            ys = [point[1] for point in candidate_mask]
            clip_bounds = (min(xs), max(xs), min(ys), max(ys))
            remainder_mask = _silhouette_mask(
                remainder, direction, center, half_extent, render_size, clip_bounds
            )
        else:
            remainder_mask = set()
        candidate_pixels = len(candidate_mask)
        exclusive_pixels = len(candidate_mask - remainder_mask)
        support[name] = {
            "candidate_pixels": candidate_pixels,
            "exclusive_pixels": exclusive_pixels,
            "support_ratio": round(exclusive_pixels / max(candidate_pixels, 1), 8),
        }
    return support


def evaluate_anchor_survival(
    obj: bpy.types.Object,
    anchors: list[dict],
    protection: dict,
    reference_bounds: tuple[Vector, Vector] | None = None,
    view_set: list[dict] | None = None,
    render_size: int = 192,
) -> dict:
    """Gate identity and recomputed per-view silhouette support against the receipt floors."""
    reference_lo, reference_hi = reference_bounds or world_bounds_of(obj)
    diagonal = max((reference_hi - reference_lo).length, 1e-6)
    center = (reference_lo + reference_hi) * 0.5
    tolerance = diagonal * 0.08
    present_ids: list[str] = []
    missing_ids: list[str] = []
    records = []
    vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    for anchor in anchors:
        anchor_id = anchor["anchor_id"]
        retained = 0
        seeds = anchor.get("seeds", [])
        for seed in seeds:
            target = center + Vector(seed) * diagonal
            if vertices and min((point - target).length for point in vertices) <= tolerance:
                retained += 1
        identity_ratio = retained / max(len(seeds), 1)
        floor = float((anchor.get("survival_floor") or {}).get("exclusive_pixel_retention_ratio", 0.0))
        floor_pixels = (anchor.get("survival_floor") or {}).get("per_view_exclusive_pixels") or {}
        source_views = anchor.get("supported_views") or []
        support = _anchor_silhouette_support(
            obj, anchor, center, diagonal, view_set or _DEFAULT_VIEW_SET, render_size
        )
        support_ratios = {view: float(metrics["support_ratio"]) for view, metrics in support.items()}
        under_floor = [
            view for view in source_views
            if (
                view not in support_ratios
                or support_ratios[view] < floor
                or (
                    view in floor_pixels
                    and int(support[view]["exclusive_pixels"]) < int(floor_pixels[view])
                )
            )
        ]
        present = bool(seeds) and retained == len(seeds) and not under_floor
        (present_ids if present else missing_ids).append(anchor_id)
        records.append({"anchor_id": anchor_id, "retained_seeds": retained,
                        "seed_count": len(seeds), "identity_ratio": round(identity_ratio, 8),
                        "support": support, "support_ratios": support_ratios,
                        "minimum_support_floor": floor, "under_floor_views": under_floor,
                        "present": present})
    return {"present_ids": present_ids, "missing_ids": missing_ids,
            "anchors": records, "all_present": not missing_ids}


def decimate_to(obj: bpy.types.Object, target_triangles: int, use_group: bool) -> dict:
    source = len(obj.data.polygons)
    ratio = min(1.0, max(0.002, target_triangles / max(source, 1)))
    modifier = obj.modifiers.new(name="lod_decimate", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = True
    if use_group and PROTECT_GROUP in {g.name for g in obj.vertex_groups}:
        modifier.vertex_group = PROTECT_GROUP
        # Weighted collapse: protected vertices resist collapse, bulk yields first.
        modifier.vertex_group_factor = 1.0
        modifier.invert_vertex_group = False
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return {"requested_ratio": ratio, "source_triangles": source}


def export(obj: bpy.types.Object, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=path, export_format="GLB", use_selection=True, export_yup=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--targets", default="220000,90000,40000,15000")
    parser.add_argument("--bulk-weight", type=float, default=0.25)
    parser.add_argument("--merge-distance", type=float, default=1e-4)
    # Optional for backwards-compatible direct CLI use.  The production V2 route always supplies
    # the ticket-01 receipt and therefore runs the hard anchor gate.
    parser.add_argument("--anchor-receipt")
    args = parser.parse_args(argv_after_double_dash())

    targets = [int(v) for v in args.targets.split(",")]
    clean_hash_before = _sha256_file(args.input)
    anchor_receipt = None
    anchors = []
    if args.anchor_receipt:
        anchor_receipt = _load_anchor_receipt(args.anchor_receipt, clean_hash_before)
        anchors = anchor_receipt["anchors"]
    output_dir = Path(args.output_dir)
    results = []
    reference_bounds = _receipt_reference_bounds(anchor_receipt) if anchor_receipt else None
    receipt_view_set = anchor_receipt.get("view_set") if anchor_receipt else None
    receipt_render_size = int(
        ((anchor_receipt or {}).get("discovery") or {}).get("parameters", {}).get("render_size", 192)
    )
    failed_lods = []

    for index, target in enumerate(targets):
        # Each LOD is decimated from the clean master in a fresh scene, so errors do not compound
        # down the chain the way successive decimation of the previous LOD would.
        reset_scene()
        objects = import_mesh(args.input)
        if not objects:
            raise RuntimeError(f"No mesh in {args.input}")
        if len(objects) > 1:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = objects[0]
            bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active or objects[0]

        # glTF stores per-corner vertices, so a re-imported mesh has every triangle as its own
        # island. Collapsing that shreds the surface - it cannot merge across the seams - and
        # yields tens of thousands of components. Weld first so the decimator sees real topology.
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=args.merge_distance)
        bpy.ops.object.mode_set(mode="OBJECT")
        welded_vertices = len(obj.data.vertices)

        if reference_bounds is None:
            reference_bounds = world_bounds_of(obj)
        protection = build_protection(obj, args.bulk_weight, anchors, reference_bounds)
        decimation = decimate_to(obj, target, use_group=True)

        anchor_survival = evaluate_anchor_survival(
            obj,
            anchors,
            protection,
            reference_bounds,
            receipt_view_set,
            receipt_render_size,
        )
        failure_reasons = []
        if not anchor_survival["all_present"]:
            failure_reasons.append("ANCHOR_MISSING_OR_UNDER_FLOOR")

        path = output_dir / f"shaman_lod{index}.glb"
        export(obj, str(path))

        achieved = len(obj.data.polygons)
        lo, hi = world_bounds_of(obj)
        if achieved <= 0 or achieved > max(target * 1.20, target + 5000):
            failure_reasons.append("TRIANGLE_TARGET_GATE")
        results.append(
            {
                "lod": index,
                "path": str(path),
                "target_triangles": target,
                "achieved_triangles": achieved,
                "vertices": len(obj.data.vertices),
                "welded_source_vertices": welded_vertices,
                "bytes": path.stat().st_size,
                "bounds_min": [lo.x, lo.y, lo.z],
                "bounds_max": [hi.x, hi.y, hi.z],
                "extent": [hi.x - lo.x, hi.y - lo.y, hi.z - lo.z],
                "protection": protection,
                "anchor_survival": anchor_survival,
                "anchor_ids": sorted(anchor.get("anchor_id") for anchor in anchors),
                "input_sha256": clean_hash_before,
                "output_sha256": _sha256_file(path),
                "passed": not failure_reasons,
                "failure_reasons": failure_reasons,
                **decimation,
            }
        )
        if failure_reasons:
            failed_lods.append(index)
        print(
            f"LOD{index}: target={target} achieved={achieved} verts={len(obj.data.vertices)}",
            flush=True,
        )

    clean_hash_after = _sha256_file(args.input)
    save_json(args.report, {
        "input": args.input,
        "clean_master_sha256_before": clean_hash_before,
        "clean_master_sha256_after": clean_hash_after,
        "clean_master_unchanged": clean_hash_before == clean_hash_after,
        "anchor_receipt": args.anchor_receipt,
        "anchor_receipt_sha256": _sha256_file(args.anchor_receipt) if args.anchor_receipt else None,
        "anchor_ids": sorted(anchor.get("anchor_id") for anchor in anchors),
        "failed_lods": failed_lods,
        "lods": results,
    })
    if clean_hash_before != clean_hash_after:
        raise RuntimeError("clean master changed during LOD generation")
    if failed_lods:
        raise RuntimeError(f"LOD anchor gate failed for LODs: {failed_lods}")


if __name__ == "__main__":
    main()

