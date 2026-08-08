"""Segment a generated mesh into parts, as a pipeline stage.

    py workers/segment_mesh.py --mesh asset.glb --out-dir parts/ --parts 12
    py workers/segment_mesh.py --mesh asset.glb --out-dir parts/ --backend partfield

Produces, from one mesh:

  labels.npy        one part id per face
  <name>_part##.glb one mesh per part
  <name>_hulls.glb  a convex hull per part, for collision
  segment.json      the receipt, including the backend's licence

Two backends behind one contract, chosen with --backend:

  geometry   scipy + trimesh, already installed, no weights, no licence
             encumbrance. Concavity-aware: it cuts where the surface bends
             inward, which is where a limb meets a torso.

  partfield  nv-tlabs/PartField. Better parts, and semantic rather than merely
             geometric. NVIDIA licence: **non-commercial research and education
             only**. Every receipt it writes carries licence_encumbered: true so
             an asset that inherited that restriction can be found later.

The geometry backend exists because it is the one that can run on anything. It
does not name parts -- it will not tell you a region is an arm -- and it is not
trying to. What it guarantees is that a part is a *connected* region of surface,
which is the property the rigging fix actually needs: the only path from one leg
to the other runs up through the pelvis, so weights cannot leak across the gap
the way Euclidean proximity lets them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

BACKENDS = ("geometry", "partfield")


def load_mesh(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_mesh() if hasattr(loaded, "to_mesh") else loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise SystemExit(f"SEGMENT_ABORT: {path} did not load as a single mesh")
    return mesh


def welded_face_adjacency(mesh: trimesh.Trimesh, tolerance: float = 1e-6) -> np.ndarray:
    """Face adjacency computed on welded POSITIONS, not on stored indices.

    A GLB duplicates a vertex wherever the UV atlas has a seam, so two triangles
    that meet along a seam share no vertex index even though they share an edge
    in space. Read literally, this seal diver is 8,031 disconnected shells whose
    largest is 1,450 faces -- a shell soup with no body in it. Segmenting that
    put 97.7% of the mesh into a single 'debris' part, which is a true statement
    about the index buffer and a false one about the object.

    It is also, almost certainly, why bone heat weighting returns 0% on these
    meshes: the solver needs a connected manifold and the index buffer does not
    give it one.

    So positions are rounded to a tolerance and de-duplicated, faces are
    rewritten in that welded index space, and adjacency comes from edges shared
    there. The mesh itself is never modified -- welding it for real would fuse
    the UV seams and ruin the atlas the paint stage just produced.
    """
    quantised = np.round(mesh.vertices / tolerance).astype(np.int64)
    _, weld = np.unique(quantised, axis=0, return_inverse=True)
    faces = weld[mesh.faces]

    # Each triangle contributes three undirected edges; an edge seen by exactly
    # two faces joins them.
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    owner = np.tile(np.arange(len(faces)), 3)

    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    pairs = np.column_stack([owner[:-1][same], owner[1:][same]])
    return pairs[pairs[:, 0] != pairs[:, 1]]


def dual_graph(mesh: trimesh.Trimesh, concavity_penalty: float):
    """Face adjacency as a weighted graph, cheap to cut where the surface is concave.

    The weight between two neighbouring faces is their centroid distance, scaled
    down when the shared edge is concave. A concave crease is where a limb joins
    a body, so making those edges cheap means a shortest-path frontier prefers to
    stop there -- which is what puts a part boundary at the shoulder rather than
    halfway down the arm.

    `face_adjacency_convex` is trimesh's per-edge convexity flag; the angle is
    the dihedral. Convex edges keep full cost, concave edges are divided by the
    penalty, so a higher penalty cuts more eagerly at creases.
    """
    pairs = welded_face_adjacency(mesh)
    if len(pairs) == 0:
        raise SystemExit("SEGMENT_ABORT: mesh has no face adjacency -- is it a point soup?")
    centroids = mesh.triangles_center
    lengths = np.linalg.norm(centroids[pairs[:, 0]] - centroids[pairs[:, 1]], axis=1)
    lengths = np.maximum(lengths, 1e-9)

    # Computed here rather than taken from mesh.face_adjacency_angles, because
    # those are indexed against trimesh's own adjacency and this graph is built
    # on welded positions -- the two arrays do not line up.
    normals = mesh.face_normals
    normal_a, normal_b = normals[pairs[:, 0]], normals[pairs[:, 1]]
    angles = np.arccos(np.clip(np.einsum("ij,ij->i", normal_a, normal_b), -1.0, 1.0))
    # A fold is convex when the neighbour sits behind this face's plane.
    offset = centroids[pairs[:, 1]] - centroids[pairs[:, 0]]
    convex = np.einsum("ij,ij->i", normal_a, offset) < 0.0
    # Deeper creases are cheaper still: a 90-degree concave fold should cut
    # before a 10-degree one does.
    depth = np.where(convex, 0.0, np.clip(angles / np.pi, 0.0, 1.0))
    weights = lengths / (1.0 + concavity_penalty * depth)

    count = len(mesh.faces)
    graph = coo_matrix(
        (np.concatenate([weights, weights]),
         (np.concatenate([pairs[:, 0], pairs[:, 1]]),
          np.concatenate([pairs[:, 1], pairs[:, 0]]))),
        shape=(count, count)).tocsr()
    return graph


def farthest_point_seeds(graph, count: int, start: int = 0) -> list[int]:
    """Pick seeds that are far apart *along the surface*, not in space.

    Euclidean farthest-point sampling would happily place two seeds on opposite
    sides of a thin gap -- the two legs again. Geodesic sampling cannot: the
    distance it measures is the walk over the surface.
    """
    seeds = [start]
    distance = dijkstra(graph, indices=start, directed=False)
    finite = np.isfinite(distance)
    while len(seeds) < count:
        candidate = int(np.argmax(np.where(finite, distance, -1.0)))
        if not finite[candidate] or distance[candidate] <= 0:
            break  # nothing reachable left to be far from
        seeds.append(candidate)
        fresh = dijkstra(graph, indices=candidate, directed=False)
        distance = np.minimum(distance, fresh)
        finite = np.isfinite(distance)
    return seeds


def segment_geometry(mesh: trimesh.Trimesh, parts: int, concavity_penalty: float,
                     min_fraction: float) -> tuple[np.ndarray, dict]:
    graph = dual_graph(mesh, concavity_penalty)

    # Disconnected shells are parts by definition -- no amount of clustering
    # should ever merge two things that do not touch.
    shell_count, shell_label = connected_components(graph, directed=False)
    labels = np.full(len(mesh.faces), -1, dtype=np.int64)
    next_label = 0
    notes = [f"{shell_count} connected shell(s)"]

    sizes = np.bincount(shell_label, minlength=shell_count)

    # A generated mesh is not one closed surface. This seal diver has 8,031
    # connected shells, 7,334 of them under 100 faces -- a third of the mesh is
    # debris. Giving every shell its own part produced 8,031 parts and, because
    # a disconnected shell shares no face adjacency with anything, the small-part
    # merger could never reach them: its `mask.any()` test is False forever. The
    # run wrote 6.6 GB of part files before it was stopped.
    #
    # So shells are triaged by size first. Only shells big enough to be a real
    # component get their own budget; everything below the floor is collected
    # into a single debris part, kept rather than deleted because dropping
    # geometry silently is how a lantern goes missing.
    debris_floor = max(8, int(len(mesh.faces) * min_fraction))
    big = [s for s in np.argsort(sizes)[::-1] if sizes[s] >= debris_floor]
    small = [s for s in range(shell_count) if sizes[s] < debris_floor and sizes[s] > 0]
    big_faces = int(sum(sizes[s] for s in big))
    notes.append(f"{len(big)} shell(s) over {debris_floor} faces; "
                 f"{len(small)} smaller shell(s) collected as one debris part")

    for shell in big:
        member = np.where(shell_label == shell)[0]
        if len(member) == 0:
            continue
        # Budget the requested part count across the big shells by face share,
        # and never let the total exceed what was asked for.
        remaining = max(1, parts - next_label)
        share = max(1, int(round(parts * len(member) / max(big_faces, 1))))
        share = min(share, remaining)
        if share == 1 or len(member) < 8:
            labels[member] = next_label
            next_label += 1
            continue

        sub = graph[member][:, member]
        seeds = farthest_point_seeds(sub, share)
        # One Dijkstra per seed, then argmin: the geodesic Voronoi cell of each
        # seed, with the concavity discount already baked into the weights.
        spread = dijkstra(sub, indices=seeds, directed=False)
        spread = np.where(np.isfinite(spread), spread, np.inf)
        owner = np.argmin(spread, axis=0)
        for index in range(len(seeds)):
            labels[member[owner == index]] = next_label + index
        next_label += len(seeds)

    if small:
        debris = np.isin(shell_label, small)
        labels[debris] = next_label
        next_label += 1
        notes.append(f"debris part holds {int(debris.sum())} faces "
                     f"({debris.sum() / len(mesh.faces) * 100:.1f}% of the mesh)")

    if (labels < 0).any():
        raise SystemExit(f"SEGMENT_ABORT: {int((labels < 0).sum())} faces unassigned")

    labels, merged = merge_small_parts(mesh, graph, labels, min_fraction)
    notes.append(f"{merged} part(s) merged as smaller than "
                 f"{min_fraction * 100:.2f}% of the mesh")
    return labels, {"notes": notes, "parts": int(labels.max()) + 1}


def merge_small_parts(mesh, graph, labels: np.ndarray, min_fraction: float):
    """Fold slivers into the neighbour they share the most boundary with.

    Geodesic Voronoi on a noisy generated surface throws off small fragments at
    crease junctions. Left alone they become one-face 'parts' that are useless
    as rig segments and worse as collision hulls.
    """
    threshold = max(4, int(len(mesh.faces) * min_fraction))
    pairs = welded_face_adjacency(mesh)
    merged = 0
    for _ in range(12):  # a merge can leave its target below threshold too
        counts = np.bincount(labels, minlength=labels.max() + 1)
        small = [i for i, c in enumerate(counts) if 0 < c < threshold]
        if not small:
            break
        for part in small:
            faces = np.where(labels == part)[0]
            if len(faces) == 0:
                continue
            mask = np.isin(pairs[:, 0], faces) ^ np.isin(pairs[:, 1], faces)
            if not mask.any():
                continue
            border = pairs[mask]
            others = np.where(np.isin(border[:, 0], faces),
                              border[:, 1], border[:, 0])
            others = others[labels[others] != part]
            if len(others) == 0:
                continue
            target = np.bincount(labels[others]).argmax()
            labels[faces] = target
            merged += 1
    # Compact the ids so they run 0..n-1 with no holes.
    unique, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64), merged


def segment_partfield(mesh_path: str, parts: int, out_dir: Path) -> tuple[np.ndarray, dict]:
    """PartField backend. Kept behind an explicit flag and an explicit licence.

    Not vendored: PartField is a separate checkout with its own CUDA-pinned
    environment, and copying it in would make this repo inherit the NVIDIA
    non-commercial terms wholesale. This shells out to whatever checkout the
    caller points at, and records that it did.
    """
    import os
    import subprocess

    root = os.environ.get("PARTFIELD_ROOT")
    if not root or not Path(root).is_dir():
        raise SystemExit(
            "SEGMENT_ABORT: --backend partfield needs PARTFIELD_ROOT set to a "
            "PartField checkout (https://github.com/nv-tlabs/PartField). "
            "Its licence is non-commercial research and education only.")
    python = os.environ.get("PARTFIELD_PYTHON", sys.executable)

    features = out_dir / "partfield_features"
    features.mkdir(parents=True, exist_ok=True)
    subprocess.run([python, str(Path(root) / "partfield_inference.py"),
                    "--input", mesh_path, "--output", str(features)],
                   check=True, cwd=root)
    subprocess.run([python, str(Path(root) / "run_part_clustering.py"),
                    "--input", str(features), "--output", str(features),
                    "--num_clusters", str(parts)],
                   check=True, cwd=root)

    produced = sorted(features.rglob("*.npy"))
    if not produced:
        raise SystemExit("SEGMENT_ABORT: PartField wrote no labels")
    labels = np.load(produced[-1]).astype(np.int64)
    return labels, {"notes": [f"PartField labels from {produced[-1].name}"],
                    "parts": int(labels.max()) + 1}


def slice_faces(mesh: trimesh.Trimesh, faces: np.ndarray) -> trimesh.Trimesh:
    """A subset of the mesh that keeps its UVs and SHARES its material.

    Not `mesh.submesh()`. That routes through `visual.concatenate`, which calls
    `material.pack()` and re-packs the whole 2048 atlas for every part. On a
    14-part seal diver the segmentation itself finished in seconds and the
    export was still running ten minutes later, entirely inside pack() -- a
    py-spy dump named the frame immediately.

    Sharing one material object across the parts is correct here anyway. Every
    part came off the same atlas and addresses it through its own UVs, so
    repacking produces fourteen near-identical 2048 textures where one will do.
    """
    face_subset = mesh.faces[faces]
    unique, inverse = np.unique(face_subset.reshape(-1), return_inverse=True)
    piece = trimesh.Trimesh(
        vertices=mesh.vertices[unique],
        faces=inverse.reshape((-1, 3)),
        process=False,
        validate=False)

    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    if uv is not None and len(uv) == len(mesh.vertices):
        piece.visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(uv)[unique], material=visual.material)
    else:
        colours = getattr(visual, "vertex_colors", None)
        if colours is not None and len(colours) == len(mesh.vertices):
            piece.visual = trimesh.visual.ColorVisuals(
                mesh=piece, vertex_colors=np.asarray(colours)[unique])
    return piece


def write_parts(mesh: trimesh.Trimesh, labels: np.ndarray, out_dir: Path,
                stem: str, write_hulls: bool, separate_files: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labels.npy", labels)

    written, hulls, pieces, failed_hulls = [], [], [], 0
    for part in range(int(labels.max()) + 1):
        faces = np.where(labels == part)[0]
        if len(faces) == 0:
            continue
        piece = slice_faces(mesh, faces)
        pieces.append(piece)
        entry = {"part": part, "faces": int(len(faces))}
        if separate_files:
            # Off by default. Each GLB embeds the whole 2048 atlas, so writing
            # one per part cost 6.6 GB for a single seal diver -- the parts are
            # 3.3 MB of texture and a few hundred KB of geometry apiece.
            target = out_dir / f"{stem}_part{part:02d}.glb"
            piece.export(target)
            entry["file"] = target.name
        written.append(entry)
        if write_hulls:
            try:
                # A convex hull per part is the standard collision proxy: the
                # union of per-part hulls approximates a concave body far better
                # than one hull over the whole mesh, which for a standing figure
                # is a solid block including the space between the legs.
                hull = piece.convex_hull
                if hull.volume > 0:
                    hulls.append(hull)
            except Exception:
                failed_hulls += 1

    # One scene holding every part as its own node. This is the default output
    # because it carries the same information as N separate files -- an importer
    # sees N meshes -- while embedding the shared atlas once instead of N times.
    parts_file = out_dir / f"{stem}_parts.glb"
    scene = trimesh.Scene()
    for entry, piece in zip(written, pieces):
        scene.add_geometry(piece, node_name=f"part{entry['part']:02d}")
    scene.export(parts_file)

    hull_file = None
    if write_hulls and hulls:
        hull_file = out_dir / f"{stem}_hulls.glb"
        hull_scene = trimesh.Scene()
        for index, hull in enumerate(hulls):
            hull_scene.add_geometry(hull, node_name=f"hull{index:02d}")
        hull_scene.export(hull_file)

    return {"parts_written": written,
            "parts_file": parts_file.name,
            "separate_files": separate_files,
            "hulls": len(hulls),
            "hulls_failed": failed_hulls,
            "hull_file": hull_file.name if hull_file else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="geometry")
    parser.add_argument("--parts", type=int, default=12)
    parser.add_argument("--concavity-penalty", type=float, default=6.0,
                        help="how much cheaper a concave edge is to cut; "
                             "higher puts boundaries harder into creases")
    parser.add_argument("--min-fraction", type=float, default=0.004,
                        help="parts smaller than this share of the mesh are "
                             "merged into their largest neighbour")
    parser.add_argument("--no-hulls", action="store_true")
    parser.add_argument("--separate-files", action="store_true",
                        help="also write one GLB per part; each embeds the "
                             "whole atlas, so this is off by default")
    parser.add_argument("--receipt")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    mesh = load_mesh(args.mesh)
    stem = Path(args.mesh).stem
    print(f"[segment] {stem}: {len(mesh.faces)} faces, backend {args.backend}",
          flush=True)

    if args.backend == "partfield":
        labels, detail = segment_partfield(args.mesh, args.parts, out_dir)
        licence = "NVIDIA non-commercial research and education only"
        encumbered = True
    else:
        labels, detail = segment_geometry(mesh, args.parts,
                                          args.concavity_penalty,
                                          args.min_fraction)
        licence = "none -- scipy and trimesh only"
        encumbered = False

    if len(labels) != len(mesh.faces):
        raise SystemExit(f"SEGMENT_ABORT: backend returned {len(labels)} labels "
                         f"for {len(mesh.faces)} faces")

    for note in detail["notes"]:
        print(f"[segment] {note}", flush=True)
    sizes = np.bincount(labels)
    print(f"[segment] {detail['parts']} parts, face counts "
          f"min {sizes.min()} median {int(np.median(sizes))} max {sizes.max()}",
          flush=True)

    written = write_parts(mesh, labels, out_dir, stem,
                          not args.no_hulls, args.separate_files)

    receipt = {
        "schema": "lowvram3d_segment_v1",
        "mesh": str(Path(args.mesh).resolve()),
        "backend": args.backend,
        "licence": licence,
        "licence_encumbered": encumbered,
        "faces": int(len(mesh.faces)),
        "requested_parts": args.parts,
        "parts": detail["parts"],
        "concavity_penalty": args.concavity_penalty,
        "face_counts": [int(v) for v in sizes],
        "notes": detail["notes"],
        **written,
    }
    path = Path(args.receipt) if args.receipt else out_dir / "segment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"[segment] wrote {len(written['parts_written'])} part meshes, "
          f"{written['hulls']} hulls, receipt {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
