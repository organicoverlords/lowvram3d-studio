"""Turn one image into a textured depth mesh with MoGe-2.

This is the stage the scene pipeline was missing. Without it the analysis
bundle reports `depth.confidence: 0.0` and `requires_depth_estimator`, and the
asset strategy falls through to `source_projection` -- a flat picture mapped
onto a shell, which reproduces the source view perfectly and carries almost no
real geometry.

Two things matter for output quality:

*Depth-edge culling.* A naive grid triangulation connects foreground to
background across every depth discontinuity, producing the long radial smears
that make an oblique view look shredded. Triangles whose vertices disagree in
depth by more than `edge_rtol` (relative to their own depth) are dropped, which
leaves a hole where the occluded background genuinely was not observed.

*Axis convention.* MoGe returns points in OpenCV camera space (X right, Y down,
Z forward). glTF is Y up, Z toward the viewer, so Y and Z are negated on
export. Skipping this is what leaves a reconstruction rotated 180 degrees about
the view axis once it reaches Unreal.

Run with the MoGe environment:

    .../envs/image-world-moge/Scripts/python.exe -m lowvram3d.moge_reconstruct \\
        --image in.png --output out.glb --receipt out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Ruicheng/moge-2-vitb-normal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_mesh(points, colors, mask, edge_rtol: float):
    """Grid-triangulate a points map, dropping triangles that span depth edges."""
    import numpy as np

    height, width = mask.shape
    index = np.full((height, width), -1, dtype=np.int64)
    valid = mask.reshape(-1)
    index.reshape(-1)[valid] = np.arange(int(valid.sum()))

    vertices = points.reshape(-1, 3)[valid]
    vertex_colors = colors.reshape(-1, 3)[valid]

    # Per-pixel UVs so the source image can be applied as a texture later.
    us, vs = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    uvs = np.stack([us, vs], axis=-1).reshape(-1, 2)[valid]

    top_left = index[:-1, :-1]
    top_right = index[:-1, 1:]
    bottom_left = index[1:, :-1]
    bottom_right = index[1:, 1:]

    # Masked-out pixels carry NaN; substitute a finite sentinel so the depth
    # comparison below stays warning-free. Those quads are rejected by quad_ok.
    depth = np.nan_to_num(points[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    quad_ok = (top_left >= 0) & (top_right >= 0) & (bottom_left >= 0) & (bottom_right >= 0)

    quad_depth = np.stack([depth[:-1, :-1], depth[:-1, 1:],
                           depth[1:, :-1], depth[1:, 1:]], axis=-1)
    span = quad_depth.max(axis=-1) - quad_depth.min(axis=-1)
    reference = np.clip(quad_depth.min(axis=-1), 1e-6, None)
    continuous = (span / reference) < edge_rtol

    keep = quad_ok & continuous
    faces = np.concatenate([
        np.stack([top_left[keep], bottom_left[keep], bottom_right[keep]], axis=-1),
        np.stack([top_left[keep], bottom_right[keep], top_right[keep]], axis=-1),
    ], axis=0)

    dropped = int(quad_ok.sum() - keep.sum())
    return vertices, vertex_colors, uvs, faces, dropped, int(quad_ok.sum())


def reconstruct(image_path: Path, output: Path, receipt_path: Path,
                model_name: str = DEFAULT_MODEL, edge_rtol: float = 0.05,
                max_triangles: int = 1_500_000) -> dict[str, Any]:
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from moge.model.v2 import MoGeModel

    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(model_name).to(device).eval()

    tensor = torch.tensor(array / 255.0, dtype=torch.float32,
                          device=device).permute(2, 0, 1)
    with torch.no_grad():
        prediction = model.infer(tensor)

    points = prediction["points"].cpu().numpy().astype(np.float64)
    mask = prediction["mask"].cpu().numpy().astype(bool)
    intrinsics = prediction["intrinsics"].cpu().numpy().astype(np.float64)

    # Normalised intrinsics: fx is a fraction of image width.
    fov_x = float(math.degrees(2.0 * math.atan(0.5 / intrinsics[0, 0])))
    fov_y = float(math.degrees(2.0 * math.atan(0.5 / intrinsics[1, 1])))

    # Bound the triangle budget by subsampling the point grid. Two triangles per
    # quad, so a stride of s costs roughly 2*H*W/s^2 triangles. This keeps the
    # budget without depending on a decimation backend, and unlike decimation it
    # never welds across the depth edges the culling below is there to preserve.
    stride = 1
    while 2 * (height // stride) * (width // stride) > max_triangles:
        stride += 1
    if stride > 1:
        points = points[::stride, ::stride]
        mask = mask[::stride, ::stride]
        array = array[::stride, ::stride]

    vertices, colors, uvs, faces, dropped, total = build_mesh(
        points, array.astype(np.float64) / 255.0, mask, edge_rtol)

    # OpenCV camera space -> glTF: negate Y (down->up) and Z (forward->back).
    vertices = vertices * np.array([1.0, -1.0, -1.0])

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces,
        vertex_colors=(colors * 255).astype(np.uint8), process=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output)

    receipt = {
        "schema_version": "moge_reconstruction_receipt_v1",
        "classification": "PROVEN",
        "image": str(image_path),
        "image_sha256": sha256(image_path),
        "image_dimensions": [width, height],
        "model": model_name,
        "device": device,
        "output_glb": str(output),
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "quads_considered": total,
        "quads_dropped_at_depth_edges": dropped,
        "depth_edge_rtol": edge_rtol,
        "grid_stride": stride,
        "masked_pixel_fraction": float(mask.mean()),
        "camera": {
            "fov_x_deg": fov_x,
            "fov_y_deg": fov_y,
            "aspect_ratio": width / height,
            "projection": "perspective",
            "convention": "gltf_y_up_z_back",
        },
        "depth_range": [float(points[..., 2][mask].min()),
                        float(points[..., 2][mask].max())],
    }

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--edge-rtol", type=float, default=0.05,
                        help="relative depth jump above which a quad is dropped")
    parser.add_argument("--max-triangles", type=int, default=1_500_000)
    args = parser.parse_args(argv)

    receipt = reconstruct(Path(args.image), Path(args.output), Path(args.receipt),
                          args.model, args.edge_rtol, args.max_triangles)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
