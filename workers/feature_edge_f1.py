"""Score a candidate mesh against a teacher mesh on internal feature edges.

Every geometry experiment in this project so far has been judged by rendering
four flat-shaded views and looking at them -- by eye, or by handing the sheet to
a vision model. That does not scale, it is not reproducible, and it is why
several dead ends took a full run each to reject. The receipts that do exist
report triangles, bodies, watertightness and silhouette coverage, and **none of
those discriminates the failure we actually have**: a mesh can carry 1.63M
triangles, be watertight, match the silhouette, and still be melted. Triangle
density, curvature energy and Euler characteristic are all gamed by
tessellation or by noise.

What separates "crisp" from "melted" is *internal* structure -- the window
recesses, railing gaps, deck lines and panel breaks that live inside the
silhouette. So that is what this measures:

    multi-view internal feature-edge precision / recall / F1

For each shared camera, both meshes are rendered to a depth buffer and a normal
buffer. Feature edges are the union of depth discontinuities (an occlusion step,
i.e. one surface in front of another) and normal-angle discontinuities (a
crease, where the surface turns sharply). The **outer silhouette is excluded**,
because silhouette agreement is a different question that is already measured
separately and would otherwise dominate the score -- two blobs of the same
outline would look like a match.

Candidate edge pixels are matched to teacher edge pixels within a small pixel
tolerance, via a distance transform, and vice versa:

    precision = fraction of candidate edges that land on a teacher edge
    recall    = fraction of teacher edges that the candidate reproduces

This is the metric that would have rejected the earlier experiments cheaply.
Melted geometry scores near zero recall, because smooth dense triangles produce
almost no internal feature edges at all. Lumpy noise is punished on precision,
because invented edges are not in the teacher. Neither failure can be hidden by
adding triangles.

Two implementation points that are not optional:

- **Depth and normal passes, not shaded RGB.** Edge-detecting a flat-shaded
  render finds triangle boundaries, so a denser mesh scores higher for being
  denser. That measures tessellation, which is exactly the confound being
  avoided.
- **Both meshes are normalised to a unit bounding box.** Generators disagree
  about scale and origin -- the diso DMC path emits at half the scale of the
  marching-cubes path from the identical field -- and an unnormalised comparison
  measures that disagreement rather than shape.

Alignment is *assumed*, not solved. Both meshes must already share a canonical
orientation. `silhouette_iou` is reported per view as a guard: if it is low, the
meshes are not aligned and the F1 numbers are meaningless rather than bad. The
gate is deliberately reported and not auto-corrected, because silently rotating
a mesh to improve its own score is how a metric becomes decorative.

    py feature_edge_f1.py --teacher online.glb --candidate mini.glb \
        --out report.json --size 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Shared cameras, matching workers/preview_generated_mesh.py so the numbers can
#: be read against previews that already exist. (forward, up) in a Y-up frame.
VIEWS = {
    "front": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "three_quarter": ((-0.7, -0.2, -0.7), (0.0, 1.0, 0.0)),
    "side": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "top": ((0.0, -1.0, -0.001), (0.0, 0.0, -1.0)),
}

#: A depth step counts as an occlusion edge past this fraction of the object's
#: own depth range. Relative, so it survives the unit-box normalisation and does
#: not need retuning per subject.
DEPTH_STEP = 0.012

#: Crease angle, degrees. Below roughly 25 the smooth curvature of a hull reads
#: as a crease and every mesh scores as highly detailed; well above it, genuine
#: panel breaks are missed.
CREASE_DEGREES = 30.0

#: Silhouette pixels are excluded by dilating the background this many pixels
#: inward, so the outline itself cannot contribute to the score.
SILHOUETTE_GUARD = 2

#: Match tolerance in pixels. Two renders of the same feature will not land on
#: exactly the same pixel; this is the "close enough" radius.
MATCH_TOLERANCE = 2.0

#: Below this silhouette IoU the two meshes are not describing the same pose,
#: and the feature scores are reported but flagged as untrustworthy.
MIN_TRUSTWORTHY_IOU = 0.55

#: Yaw angles tried when registering the candidate to the teacher. Generators
#: do not agree on which way a subject faces, and the first run of this metric
#: failed its own negative control precisely because nothing was registered:
#: silhouette IoU came in at 0.40-0.69 and every comparison tripped the trust
#: gate. 72 steps is 5 degrees, fine enough that residual yaw error is well
#: under the match tolerance at 384 px.
YAW_STEPS = 72

#: Points sampled per mesh for registration. Registration is done on point
#: clouds rather than renders because it must be cheap enough to try 72 poses,
#: and because Chamfer distance is a *geometric* criterion -- it cannot see the
#: feature-edge score it is about to enable. Choosing a pose by the score itself
#: would be a metric optimising its own output.
REGISTRATION_POINTS = 20000

#: A candidate whose edge density exceeds the teacher's by more than this factor
#: is flagged. High recall is trivially purchased by covering the frame in
#: edges: the negative control scored recall 0.54 at 2-3x the teacher's density
#: while precision sat at 0.17-0.53, and averaging the two into F1 let the noise
#: win. Density is reported and gated, not folded into the score.
MAX_EDGE_DENSITY_RATIO = 1.6


def load_normalised(path, up_axis="auto"):
    """Load a GLB flattened through its scene graph, in a unit bounding box.

    Concatenating `scene.geometry.values()` discards node transforms, which has
    already produced two wrong conclusions in this project. Flatten properly.
    """
    import numpy as np
    import trimesh

    scene = trimesh.load(str(path), process=False)
    if hasattr(scene, "geometry"):
        mesh = (scene.to_geometry() if hasattr(scene, "to_geometry")
                else scene.dump(concatenate=True))
    else:
        mesh = scene
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Up-axis guess, same rule as the preview worker: a generator rests its
    # subject on the ground, so the up axis has its minimum at zero while the
    # others straddle it.
    if up_axis == "auto":
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        span = np.maximum(high - low, 1e-9)
        grounded = np.abs(low) / span
        centred = np.abs(low + high) / span
        candidates = [i for i in (1, 2)
                      if grounded[i] < 0.02 and centred[i] > 0.5]
        up_axis = {1: "y", 2: "z"}.get(candidates[0], "y") if len(candidates) == 1 else "y"
    if up_axis == "z":
        vertices = np.column_stack(
            [vertices[:, 0], vertices[:, 2], -vertices[:, 1]])

    # Unit box, centred. Scale by the largest extent so proportions survive.
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    vertices = (vertices - (low + high) * 0.5) / max(float((high - low).max()), 1e-9)
    return vertices, faces


def sample_surface(vertices, faces, count, seed=0):
    """Area-weighted points on the surface.

    Vertices alone are a biased sample: a mesh with 2.8M vertices concentrated
    on ornament would register against its ornament rather than its shape.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = areas.sum()
    if total <= 0:
        return vertices[rng.integers(0, len(vertices), count)]
    picked = rng.choice(len(faces), size=count, p=areas / total)
    u = rng.random((count, 1))
    v = rng.random((count, 1))
    over = (u + v) > 1.0
    u[over] = 1.0 - u[over]
    v[over] = 1.0 - v[over]
    return a[picked] + u * (b[picked] - a[picked]) + v * (c[picked] - a[picked])


def register_yaw(teacher_points, candidate_points, steps=YAW_STEPS):
    """Best yaw for the candidate, by symmetric Chamfer distance.

    Rotation about Y only. Generators agree on which way is up -- they rest the
    subject on the ground -- but not on which way it faces, and yaw is the axis
    that actually differs. Tilt and roll are deliberately not searched: adding
    free parameters lets a wrong mesh contort itself into a better score, and
    the residual would be indistinguishable from genuine shape disagreement.

    Chamfer is computed on point clouds and is blind to feature edges, so the
    pose is chosen without reference to the quantity it will be used to measure.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    teacher_tree = cKDTree(teacher_points)
    best = (None, np.inf)
    for index in range(steps):
        angle = 2.0 * np.pi * index / steps
        cos, sin = np.cos(angle), np.sin(angle)
        rotated = candidate_points @ np.array(
            [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
        forward_d, _ = teacher_tree.query(rotated, workers=-1)
        backward_d, _ = cKDTree(rotated).query(teacher_points, workers=-1)
        chamfer = float(forward_d.mean() + backward_d.mean())
        if chamfer < best[1]:
            best = (angle, chamfer)
    return best


def rotate_y(vertices, angle):
    import numpy as np

    cos, sin = np.cos(angle), np.sin(angle)
    return vertices @ np.array(
        [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def render_depth_normal(vertices, faces, forward, up, size):
    """Orthographic depth + face-normal buffers.

    Rasterises near-to-far with an occupancy mask rather than far-to-near with a
    z-test. Both are correct for opaque geometry, but near-to-far lets a
    triangle be skipped entirely once its bounding box is already covered, which
    on a 1M-triangle mesh is most of them after the first depth layer. The
    painter's-order version in the preview worker takes minutes per view; this
    matters because the metric renders two meshes across four views.
    """
    import numpy as np

    forward = np.asarray(forward, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right /= max(np.linalg.norm(right), 1e-9)
    true_up = np.cross(right, forward)
    basis = np.stack([right, true_up, forward])

    camera = vertices @ basis.T
    low, high = camera[:, :2].min(axis=0), camera[:, :2].max(axis=0)
    span = float(max(high - low)) or 1.0
    margin = size * 0.06
    scale = (size - 2 * margin) / span
    centre = (low + high) * 0.5
    screen = (camera[:, :2] - centre) * scale + size * 0.5
    screen[:, 1] = size - screen[:, 1]

    edge1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    normals = np.cross(edge1, edge2)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(lengths, 1e-9, None)

    # Orient every normal toward the camera. Generated meshes are not reliably
    # wound, and a crease detector that sees a sign flip where the winding flips
    # would score winding noise as architectural detail.
    facing = normals @ forward
    normals = np.where((facing > 0)[:, None], -normals, normals)

    triangles = screen[faces]
    depth = camera[faces][:, :, 2].mean(axis=1)

    depth_buffer = np.full((size, size), np.inf)
    normal_buffer = np.zeros((size, size, 3))
    covered = np.zeros((size, size), dtype=bool)

    for index in np.argsort(depth):          # near to far
        tri = triangles[index]
        x0, y0 = np.floor(tri.min(axis=0)).astype(int)
        x1, y1 = np.ceil(tri.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, size), min(y1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        window = covered[y0:y1, x0:x1]
        if window.all():                     # fully occluded already
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        px, py = xs + 0.5, ys + 0.5
        ax, ay = tri[0]
        bx, by = tri[1]
        cx, cy = tri[2]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-12:
            continue
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1) & ~window
        if not inside.any():
            continue
        depth_buffer[y0:y1, x0:x1][inside] = depth[index]
        normal_buffer[y0:y1, x0:x1][inside] = normals[index]
        window |= inside
    return depth_buffer, normal_buffer, covered


def feature_edges(depth, normals, covered):
    """Internal feature edges: occlusion steps and creases, silhouette removed."""
    import numpy as np
    from scipy import ndimage

    # Normalise depth over the object only, so the thresholds are relative.
    finite = covered & np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(covered), covered.copy()
    span = float(depth[finite].max() - depth[finite].min()) or 1.0
    normalised = np.where(finite, (depth - depth[finite].min()) / span, 0.0)

    # Depth discontinuity: max absolute step to a 4-neighbour.
    step = np.zeros_like(normalised)
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        rolled = np.roll(normalised, shift, axis=axis)
        valid = np.roll(finite, shift, axis=axis) & finite
        step = np.maximum(step, np.where(valid, np.abs(normalised - rolled), 0.0))
    depth_edge = finite & (step > DEPTH_STEP)

    # Crease: max angle between this normal and a 4-neighbour's.
    cosine = np.ones(normals.shape[:2])
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        rolled = np.roll(normals, shift, axis=axis)
        valid = np.roll(finite, shift, axis=axis) & finite
        dot = np.clip((normals * rolled).sum(axis=2), -1.0, 1.0)
        cosine = np.where(valid, np.minimum(cosine, dot), cosine)
    crease_edge = finite & (cosine < np.cos(np.radians(CREASE_DEGREES)))

    # Drop the outer silhouette: anything within the guard band of background.
    interior = ndimage.binary_erosion(
        covered, np.ones((3, 3), bool), iterations=SILHOUETTE_GUARD + 1)
    return (depth_edge | crease_edge) & interior, covered


def score(teacher_edges, candidate_edges, tolerance):
    """Precision/recall/F1 by nearest-edge distance within a tolerance."""
    import numpy as np
    from scipy import ndimage

    if not teacher_edges.any() and not candidate_edges.any():
        return {"precision": None, "recall": None, "f1": None,
                "teacher_edge_pixels": 0, "candidate_edge_pixels": 0,
                "note": "neither mesh has internal feature edges"}
    # Distance from every pixel to the nearest edge of the other mesh.
    to_teacher = ndimage.distance_transform_edt(~teacher_edges)
    to_candidate = ndimage.distance_transform_edt(~candidate_edges)

    precision = (float((to_teacher[candidate_edges] <= tolerance).mean())
                 if candidate_edges.any() else 0.0)
    recall = (float((to_candidate[teacher_edges] <= tolerance).mean())
              if teacher_edges.any() else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "teacher_edge_pixels": int(teacher_edges.sum()),
        "candidate_edge_pixels": int(candidate_edges.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True,
                        help="Reference mesh, e.g. the online-service GLB.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=MATCH_TOLERANCE)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--align", choices=("yaw", "none"), default="yaw",
        help="Register the candidate to the teacher before measuring. 'yaw' "
             "searches rotations about the up axis and picks the one minimising "
             "symmetric Chamfer distance on sampled surface points. 'none' "
             "assumes the meshes already share a pose, which they generally do "
             "not.")
    parser.add_argument("--dump-edges", default="",
                        help="Directory for per-view edge PNGs, for eyeballing "
                             "what the metric actually fired on.")
    args = parser.parse_args(argv)

    import numpy as np

    teacher_v, teacher_f = load_normalised(args.teacher)
    candidate_v, candidate_f = load_normalised(args.candidate)

    # Register before measuring. Without this the metric compares two meshes
    # through a misregistration and reports the misalignment as missing detail.
    registration = {"mode": args.align}
    if args.align == "yaw":
        teacher_points = sample_surface(teacher_v, teacher_f, REGISTRATION_POINTS)
        candidate_points = sample_surface(candidate_v, candidate_f,
                                          REGISTRATION_POINTS)
        angle, chamfer = register_yaw(teacher_points, candidate_points)
        candidate_v = rotate_y(candidate_v, angle)
        registration.update({
            "yaw_degrees": round(float(np.degrees(angle)), 2),
            "chamfer": round(chamfer, 5),
            "points": REGISTRATION_POINTS,
            "criterion": "symmetric chamfer on surface points; blind to the "
                         "feature score it enables",
        })

    views = {}
    for name, (forward, up) in VIEWS.items():
        t_depth, t_normal, t_cov = render_depth_normal(
            teacher_v, teacher_f, forward, up, args.size)
        c_depth, c_normal, c_cov = render_depth_normal(
            candidate_v, candidate_f, forward, up, args.size)
        t_edges, _ = feature_edges(t_depth, t_normal, t_cov)
        c_edges, _ = feature_edges(c_depth, c_normal, c_cov)

        union = (t_cov | c_cov).sum()
        iou = float((t_cov & c_cov).sum() / union) if union else 0.0
        entry = score(t_edges, c_edges, args.tolerance)
        entry["silhouette_iou"] = round(iou, 4)
        entry["aligned"] = bool(iou >= MIN_TRUSTWORTHY_IOU)
        # Edge density normalises out how much of the frame the subject fills,
        # so a small render and a large one are comparable.
        entry["teacher_edge_density"] = round(
            float(t_edges.sum() / max(t_cov.sum(), 1)), 5)
        entry["candidate_edge_density"] = round(
            float(c_edges.sum() / max(c_cov.sum(), 1)), 5)
        views[name] = entry

        if args.dump_edges:
            from PIL import Image
            out_dir = Path(args.dump_edges)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Teacher red, candidate green: yellow is agreement, and the two
            # kinds of failure are visually distinct rather than both "wrong".
            rgb = np.zeros((args.size, args.size, 3), dtype=np.uint8)
            rgb[..., 0] = t_edges * 255
            rgb[..., 1] = c_edges * 255
            Image.fromarray(rgb).save(out_dir / f"edges_{name}.png")

    scored = [v for v in views.values() if v.get("f1") is not None]
    mean_f1 = round(float(np.mean([v["f1"] for v in scored])), 4) if scored else None
    mean_recall = round(float(np.mean([v["recall"] for v in scored])), 4) if scored else None
    mean_precision = round(float(np.mean([v["precision"] for v in scored])), 4) if scored else None
    aligned_views = sum(1 for v in views.values() if v["aligned"])

    # Edge-density ratio, reported alongside the score rather than inside it.
    # Recall alone is buyable: cover the frame in edges and some will land near
    # a teacher edge by luck. The first negative control did exactly that --
    # a different object at 2-3x the teacher's edge density out-scored every
    # real candidate once F1 averaged its poor precision away.
    ratios = [v["candidate_edge_density"] / v["teacher_edge_density"]
              for v in views.values() if v["teacher_edge_density"] > 0]
    density_ratio = round(float(np.mean(ratios)), 3) if ratios else None
    density_inflated = bool(density_ratio and density_ratio > MAX_EDGE_DENSITY_RATIO)

    report = {
        "schema_version": "feature_edge_f1_v1",
        "label": args.label,
        "teacher": str(Path(args.teacher).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "size": args.size,
        "tolerance_px": args.tolerance,
        "thresholds": {
            "depth_step": DEPTH_STEP,
            "crease_degrees": CREASE_DEGREES,
            "silhouette_guard_px": SILHOUETTE_GUARD,
        },
        "registration": registration,
        "views": views,
        "mean_f1": mean_f1,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "edge_density_ratio": density_ratio,
        "density_inflated": density_inflated,
        "aligned_views": aligned_views,
        # Alignment is now solved, but by a criterion that cannot see the score
        # -- and the residual is still reported rather than assumed away. A
        # comparison is trustworthy only if the poses actually agree AND the
        # candidate is not simply drowning the frame in edges.
        "trustworthy": bool(aligned_views >= 3 and not density_inflated),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
