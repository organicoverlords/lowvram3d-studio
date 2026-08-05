"""Remove high-frequency speckle without erasing one-triangle-wide features.

TRELLIS surfaces carry a fine granular noise -- visible as speckle across the
hull and facade -- that reads as low quality next to the concept art. The
obvious response, a feature-preserving smoothing pass, was already measured on
this project and failed at every setting: on Hunyuan meshes the "lumps" *were*
the geometry, and on this mesh a rail is one or two triangles wide, which is
numerically indistinguishable from noise to any generic filter.

So the filter is not the interesting part. The protection is.

The method is bilateral normal filtering (Zheng et al. / Sun et al.): smooth the
*face normal field* rather than vertex positions, weighting each neighbour by
both spatial proximity and normal similarity, then move vertices to agree with
the filtered normals. Normals are the right domain because a crease is a
discontinuity in the normal field, so a bilateral weight can preserve it,
whereas Laplacian position smoothing cannot see it at all.

On top of that, four classes of geometry are frozen outright and never move:

- **Boundary vertices.** Every open shell edge. Moving them tears the shell.
- **Thin shells.** A rail is a long thin box; smoothing rounds it to a ribbon
  and then to nothing. Measured by smallest-over-largest extent.
- **Small shells.** Ornament, spokes, posts -- too few faces for smoothing to
  be anything but destruction.
- **High-curvature vertices.** Creases and corners, by dihedral angle.

And the displacement of everything else is capped at a fraction of local edge
length, so even an unprotected vertex cannot travel far enough to remove a
feature the classifier missed. That cap is the safety net for the classifier
being wrong, which on a mesh with 7,144 shells it sometimes will be.

    py denoise_protected.py --input mesh.glb --out clean.glb --iterations 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Bilateral normal weight: how much a neighbouring face's normal must agree
#: before it influences this one. Smaller preserves sharper creases; too small
#: and nothing is smoothed at all.
SIGMA_NORMAL = 0.35

#: Dihedral angle, degrees, above which a vertex is treated as a crease and
#: frozen. The mesh's genuine architecture -- deck edges, panel breaks -- sits
#: above this; the speckle sits well below.
CREASE_DEGREES = 42.0

#: Shells with fewer faces, or thinner than this ratio, are frozen entirely.
#: Same reasoning and same numbers as workers/lod_per_shell.py, deliberately:
#: the two stages should agree about what counts as a thin feature.
MIN_FACES_TO_SMOOTH = 400
MIN_THICKNESS_RATIO = 0.06

#: Maximum vertex travel per run, as a fraction of that vertex's mean incident
#: edge length. The backstop for a misclassified feature.
MAX_DISPLACEMENT = 0.20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=3,
                        help="Normal-filter iterations. 2-4 is the useful "
                             "range; more converges toward a flat surface.")
    parser.add_argument("--vertex-iterations", type=int, default=8)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh

    scene = trimesh.load(args.input, process=False)
    source = (scene.to_geometry() if hasattr(scene, "geometry") else scene)
    mesh = trimesh.Trimesh(vertices=np.asarray(source.vertices),
                           faces=np.asarray(source.faces), process=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    original = vertices.copy()
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # --- classify what must not move -------------------------------------
    frozen = np.zeros(len(vertices), dtype=bool)

    # Boundary: an edge used by exactly one face.
    edges = mesh.edges_sorted.reshape(-1, 2)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]
    frozen[np.unique(boundary_edges)] = True
    boundary_count = int(frozen.sum())

    # Small or thin shells, frozen wholesale.
    #
    # Label faces by connected component directly from the adjacency graph
    # rather than calling mesh.split(). split() materialises a separate Trimesh
    # per shell -- 2,904 of them here -- and finding which original vertices
    # each one owns then needs a proximity query per shell against every
    # vertex. That is quadratic and it exhausted 16 GB of RAM on the first
    # attempt. The component labels are already implicit in face_adjacency.
    thin_or_small = 0
    labels = trimesh.graph.connected_component_labels(
        mesh.face_adjacency, node_count=len(faces))
    for label in np.unique(labels):
        member_faces = np.flatnonzero(labels == label)
        member_vertices = np.unique(faces[member_faces].reshape(-1))
        points = vertices[member_vertices]
        extents = np.sort(points.max(axis=0) - points.min(axis=0))
        thin = extents[0] / max(extents[2], 1e-9) < MIN_THICKNESS_RATIO
        if len(member_faces) >= MIN_FACES_TO_SMOOTH and not thin:
            continue
        frozen[member_vertices] = True
        thin_or_small += len(member_vertices)

    # Creases, by dihedral angle across adjacent faces.
    adjacency = mesh.face_adjacency
    angles = mesh.face_adjacency_angles
    sharp = adjacency[np.degrees(angles) > CREASE_DEGREES]
    if len(sharp):
        frozen[np.unique(faces[sharp.reshape(-1)].reshape(-1))] = True
    crease_count = int(frozen.sum()) - boundary_count - thin_or_small

    movable = ~frozen

    # --- bilateral normal filtering --------------------------------------
    normals = np.asarray(mesh.face_normals, dtype=np.float64).copy()
    centroids = vertices[faces].mean(axis=1)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    left, right = adjacency[:, 0], adjacency[:, 1]
    separation = np.linalg.norm(centroids[left] - centroids[right], axis=1)
    sigma_space = max(float(separation.mean()), 1e-9)

    for _ in range(args.iterations):
        difference = np.linalg.norm(normals[left] - normals[right], axis=1)
        spatial = np.exp(-(separation ** 2) / (2 * sigma_space ** 2))
        angular = np.exp(-(difference ** 2) / (2 * SIGMA_NORMAL ** 2))
        weight = areas[right] * spatial * angular

        accumulated = np.zeros_like(normals)
        total = np.zeros(len(normals))
        # Symmetric: each adjacency contributes in both directions.
        np.add.at(accumulated, left, normals[right] * weight[:, None])
        np.add.at(total, left, weight)
        np.add.at(accumulated, right, normals[left] * weight[:, None])
        np.add.at(total, right, weight)
        accumulated += normals * areas[:, None]
        total += areas

        normals = accumulated / np.clip(total, 1e-12, None)[:, None]
        normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)

    # --- move vertices to agree with the filtered normals -----------------
    # Sun et al.: push each vertex along each incident face normal by the
    # residual between the face plane and the vertex.
    corner = faces.reshape(-1)
    face_of_corner = np.repeat(np.arange(len(faces)), 3)
    for _ in range(args.vertex_iterations):
        centroids = vertices[faces].mean(axis=1)
        offset = centroids[face_of_corner] - vertices[corner]
        projection = (offset * normals[face_of_corner]).sum(axis=1)
        delta = normals[face_of_corner] * projection[:, None]
        accumulated = np.zeros_like(vertices)
        count = np.zeros(len(vertices))
        np.add.at(accumulated, corner, delta)
        np.add.at(count, corner, 1.0)
        step = accumulated / np.clip(count, 1.0, None)[:, None]
        step[~movable] = 0.0
        vertices += step / 3.0

    # --- cap total displacement ------------------------------------------
    edge_vectors = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    edge_length = np.linalg.norm(edge_vectors, axis=1)
    mean_edge = np.zeros(len(vertices))
    edge_count = np.zeros(len(vertices))
    np.add.at(mean_edge, edges[:, 0], edge_length)
    np.add.at(edge_count, edges[:, 0], 1.0)
    np.add.at(mean_edge, edges[:, 1], edge_length)
    np.add.at(edge_count, edges[:, 1], 1.0)
    mean_edge /= np.clip(edge_count, 1.0, None)

    travel = vertices - original
    distance = np.linalg.norm(travel, axis=1)
    limit = MAX_DISPLACEMENT * mean_edge
    excessive = distance > limit
    scale = np.ones(len(vertices))
    scale[excessive] = limit[excessive] / np.clip(distance[excessive], 1e-12, None)
    vertices = original + travel * scale[:, None]
    clamped = int(excessive.sum())

    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.export(str(out))

    final_travel = float(np.linalg.norm(vertices - original, axis=1).mean())
    receipt = {
        "schema_version": "denoise_protected_v1",
        "input": str(Path(args.input).resolve()),
        "output": str(out.resolve()),
        "triangles": int(len(faces)),
        "vertices": int(len(vertices)),
        "frozen_vertices": int(frozen.sum()),
        "frozen_fraction": round(float(frozen.mean()), 4),
        "frozen_boundary": boundary_count,
        "frozen_thin_or_small": thin_or_small,
        "frozen_crease": max(crease_count, 0),
        "clamped_vertices": clamped,
        "mean_displacement": round(final_travel, 6),
        "iterations": args.iterations,
        "sigma_normal": SIGMA_NORMAL,
        "crease_degrees": CREASE_DEGREES,
        "max_displacement_ratio": MAX_DISPLACEMENT,
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
