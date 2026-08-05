"""Repair occupied atlas texels that are visible in an assigned view but missed a sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lowvram3d.texture_provenance import EvidenceState, FrequencyAuthority, Lineage, SourceClass, load_npz, save_npz


FACING_MIN = 0.15
ALPHA_MIN = 0.35


def _same_view_sample(image: np.ndarray, alpha: np.ndarray, face_id: np.ndarray | None,
                      u: float, v: float, triangle: int, radius: int = 2):
    h, w = alpha.shape
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    x0 = int(round(u * (w - 1))); y0 = int(round(v * (h - 1)))
    candidates = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = x0 + dx, y0 + dy
            if not (0 <= x < w and 0 <= y < h):
                continue
            if alpha[y, x] <= ALPHA_MIN:
                continue
            if face_id is not None and int(face_id[y, x]) != int(triangle):
                continue
            candidates.append((dx * dx + dy * dy, x, y))
    if not candidates:
        return None
    _, x, y = min(candidates)
    return image[y, x], np.array([x, y], np.int32)


def repair(projection_npz: Path, views_dir: Path, basecolor: Path,
           atlas_provenance: Path, assignment_path: Path,
           output: Path, output_provenance: Path, report_path: Path) -> dict:
    data = np.load(projection_npz, allow_pickle=False)
    names = [str(x) for x in data["view_names"]]
    positions = np.asarray(data["verts"], np.float32)
    triangles = np.asarray(data["tris"], np.int32)
    uv_owner = load_npz(atlas_provenance)
    owner = np.asarray(uv_owner["triangle_id"], np.int32)
    occupied = np.asarray(uv_owner["atlas_occupied_mask"], bool)
    direct = np.asarray(uv_owner["direct_observed_texel_mask"], bool)
    if owner.shape != occupied.shape or owner.shape != direct.shape:
        raise RuntimeError("PER_TEXEL_PROVENANCE_DIMENSION_MISMATCH")
    assigned = np.asarray(np.load(assignment_path), np.int32)
    if assigned.shape != (len(triangles),):
        raise RuntimeError("SOURCE_ASSIGNMENT_SHAPE_MISMATCH")

    image_bgr = cv2.imread(str(basecolor), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("BASECOLOR_UNREADABLE")
    # The projector's internal rows are vertically opposite to the canonical glTF PNG.
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)[::-1].copy()
    initial = {key: np.array(uv_owner[key], copy=True) for key in (
        "triangle_id", "direct_observed_texel_mask", "source_view", "source_pixel",
        "barycentric", "confidence", "evidence_state",
    ) if key in uv_owner}
    gap = np.zeros_like(direct)
    unobserved = np.zeros_like(direct)
    visible_gap_cross_view = 0
    attempted = 0
    repaired = 0
    occupied_black = 0
    coords = np.argwhere(occupied & ~direct)
    for y, x in coords:
        t = int(owner[y, x])
        if t < 0 or t >= len(triangles):
            continue
        vi = int(assigned[t])
        if vi < 0 or vi >= len(names):
            unobserved[y, x] = True
            continue
        name = names[vi]
        visible_triangles = np.asarray(data[f"vis_{name}"], bool)
        if not bool(visible_triangles[t]):
            unobserved[y, x] = True
            continue
        # Use the exact UV-owner barycentric coordinate emitted by the projector.
        weights = np.asarray(uv_owner["barycentric"][y, x], np.float32)
        if not np.isfinite(weights).all() or weights.sum() < 0.99 or weights.sum() > 1.01:
            unobserved[y, x] = True
            continue
        world = (weights[:, None] * positions[triangles[t]]).sum(axis=0)
        cam = np.asarray(data["view_locs"][vi], np.float32)
        direction = cam / max(float(np.linalg.norm(cam)), 1e-12)
        normal = np.cross(positions[triangles[t, 1]] - positions[triangles[t, 0]],
                          positions[triangles[t, 2]] - positions[triangles[t, 0]])
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        facing = float(normal @ direction)
        attempted += 1
        if not np.isfinite(facing) or facing <= FACING_MIN:
            unobserved[y, x] = True
            continue
        axis = int(np.argmax(np.abs(direction)))
        ua, va = (0, 2) if axis == 1 else ((1, 2) if axis == 0 else (0, 1))
        flip_u = -1.0 if direction[axis] > 0 else 1.0
        u = float((world[ua] * flip_u) / float(data["ortho_scale"]) + 0.5)
        v = float(0.5 - world[va] / float(data["ortho_scale"]))
        source_bgr = cv2.imread(str(views_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if source_bgr is None:
            raise RuntimeError("VIEW_IMAGE_MISSING:" + name)
        if source_bgr.ndim == 3 and source_bgr.shape[2] >= 4:
            source = cv2.cvtColor(source_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
            alpha = source_bgr[:, :, 3].astype(np.float32) / 255.0
        else:
            source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            alpha = np.ones(source.shape[:2], np.float32)
        face = np.asarray(data.get(f"face_id_{name}"), np.int32) if f"face_id_{name}" in data.files else None
        sample = _same_view_sample(source, alpha, face, u, v, t)
        if sample is None:
            unobserved[y, x] = True
            continue
        rgb, pixel = sample
        image[y, x] = rgb
        gap[y, x] = True
        repaired += 1
        uv_owner["evidence_state"][y, x] = np.uint8(EvidenceState.VISIBLE_SOURCE_GAP)
        uv_owner["source_class"][y, x] = np.uint8(SourceClass.ORIGINAL_NONFACE)
        uv_owner["lineage"][y, x] = np.uint16(Lineage.ORIGINAL_NONFACE)
        uv_owner["lineage_bits"][y, x] = np.uint16(Lineage.ORIGINAL_NONFACE)
        uv_owner["source_view"][y, x] = vi
        uv_owner["primary_view"][y, x] = vi
        uv_owner["source_pixel"][y, x] = pixel
        uv_owner["visibility"][y, x] = True
        uv_owner["facing"][y, x] = facing
        uv_owner["face_id_match"][y, x] = face is None or int(face[pixel[1], pixel[0]]) == t
        uv_owner["source_mask_valid"][y, x] = True
        uv_owner["confidence"][y, x] = max(0.0, min(1.0, facing))
        uv_owner["frequency_authority"][y, x] = np.uint8(FrequencyAuthority.LOW_AND_MEDIUM)
        uv_owner["completion_method"][y, x] = "same_view_visible_gap_repair"

    uv_owner["visible_source_gap_mask"] = gap
    uv_owner["visible_source_gap"] = gap
    uv_owner["unobserved_surface_mask"] = unobserved
    uv_owner["unobserved_surface"] = unobserved
    uv_owner["unresolved_mask"] = unobserved
    uv_owner["unresolved"] = unobserved
    uv_owner["direct_observed"] = direct
    uv_owner["direct_observed_texel_mask"] = direct
    occupied_black = int(np.count_nonzero(occupied & (direct | gap) & np.all(image == 0, axis=2)))
    preserved = all(np.array_equal(initial[key], uv_owner[key]) for key in initial)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(image[::-1], cv2.COLOR_RGB2BGR))
    save_npz(output_provenance, uv_owner)
    report = {
        "schema": "visible_source_gap_repair_v1",
        "occupied_texels": int(occupied.sum()),
        "direct_texels": int(direct.sum()),
        "visible_gap_candidates": int(coords.shape[0]),
        "visible_gap_attempted": int(attempted),
        "visible_gap_repaired": int(repaired),
        "unobserved_surface_texels": int(unobserved.sum()),
        "visible_gap_cross_view_transfer": int(visible_gap_cross_view),
        "occupied_visible_black_texels": occupied_black,
        "direct_provenance_preserved": bool(preserved),
        "source_view_authority": "same_assigned_view_only",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-npz", required=True)
    parser.add_argument("--views-dir", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--atlas-provenance", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-provenance", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    repair(Path(args.projection_npz), Path(args.views_dir), Path(args.basecolor),
           Path(args.atlas_provenance), Path(args.assignment), Path(args.output),
           Path(args.output_provenance), Path(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
