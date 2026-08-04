"""Export a MoGe reconstruction as 3D Gaussian Splats instead of a mesh.

Why this exists: a mesh has connectivity, and connectivity is what fails on a
single-view reconstruction. Triangles spanning a depth discontinuity stretch
foreground into background, which is the smearing that makes an oblique view
look shredded. Culling those triangles removes the smears but leaves holes.
Either way the artefacts come from edges that were never observed.

Splats have no connectivity, so that entire failure mode disappears. Each
observed pixel becomes an independent oriented disc; unobserved regions are
simply empty rather than being bridged or torn.

This is a depth-initialised splat set, not an optimised one. A single view
cannot supervise a fit -- there is no second viewpoint to disagree with -- so
positions and colours come straight from the point map and only the extents are
estimated. It is the correct starting point for later multi-view refinement,
and already renders without the mesh artefacts.

Output is the standard INRIA 3DGS PLY that viewers and engine plugins expect.

    .../envs/image-world-moge/Scripts/python.exe -m lowvram3d.splat_export \\
        --image in.png --output scene.ply --receipt scene.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

# Zeroth-order spherical harmonic. 3DGS stores colour as SH coefficients, so a
# plain RGB value has to be converted rather than written directly.
SH_C0 = 0.28209479177387814


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _neighbour_spacing(points, mask):
    """Distance to the nearest in-plane neighbour, per pixel.

    A splat should be about as wide as the gap to its neighbours: smaller and
    the surface shows through, larger and it bleeds over depth edges. Spacing
    grows with depth, so this scales naturally across the frame.
    """
    import numpy as np

    spacing = np.full(mask.shape, np.nan)
    right = np.linalg.norm(np.diff(points, axis=1), axis=-1)
    down = np.linalg.norm(np.diff(points, axis=0), axis=-1)

    candidates = np.full(mask.shape + (4,), np.nan)
    candidates[:, :-1, 0] = right
    candidates[:, 1:, 1] = right
    candidates[:-1, :, 2] = down
    candidates[1:, :, 3] = down

    with np.errstate(invalid="ignore"):
        spacing = np.nanmin(candidates, axis=-1)
    return spacing


def write_ply(path: Path, xyz, rgb, scales, opacity, rotation) -> None:
    """Write the INRIA 3DGS PLY layout (SH degree 0)."""
    import numpy as np

    count = xyz.shape[0]
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {count}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1",
        "property float rot_2", "property float rot_3",
        "end_header", "",
    ]

    fields = [xyz, np.zeros_like(xyz), rgb, opacity[:, None], scales, rotation]
    data = np.concatenate(fields, axis=1).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(data.tobytes())


def export(image_path: Path, output: Path, receipt_path: Path,
           model_name: str = "Ruicheng/moge-2-vitb-normal",
           max_splats: int = 2_000_000, opacity_value: float = 0.99,
           thickness: float = 0.25) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image
    from moge.model.v2 import MoGeModel

    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(model_name).to(device).eval()
    tensor = torch.tensor(array / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        prediction = model.infer(tensor)

    points = prediction["points"].cpu().numpy().astype(np.float64)
    mask = prediction["mask"].cpu().numpy().astype(bool)
    intrinsics = prediction["intrinsics"].cpu().numpy().astype(np.float64)
    fov_x = float(math.degrees(2.0 * math.atan(0.5 / intrinsics[0, 0])))
    fov_y = float(math.degrees(2.0 * math.atan(0.5 / intrinsics[1, 1])))

    spacing = _neighbour_spacing(points, mask)
    valid = mask & np.isfinite(spacing) & np.isfinite(points).all(axis=-1)

    stride = 1
    while int(valid[::stride, ::stride].sum()) > max_splats:
        stride += 1
    if stride > 1:
        points, array_s, valid_s, spacing_s = (points[::stride, ::stride],
                                               array[::stride, ::stride],
                                               valid[::stride, ::stride],
                                               spacing[::stride, ::stride] * stride)
    else:
        array_s, valid_s, spacing_s = array, valid, spacing

    xyz = points[valid_s]
    colours = array_s[valid_s].astype(np.float64) / 255.0
    radius = np.clip(spacing_s[valid_s], 1e-6, None) * 0.5

    # OpenCV camera space -> the Y-up, Z-back convention viewers expect.
    xyz = xyz * np.array([1.0, -1.0, -1.0])

    # Splats are discs: wide in the surface plane, thin along the view ray.
    # Storing a genuine thickness rather than a sphere keeps depth edges crisp.
    scales = np.stack([radius, radius, radius * thickness], axis=-1)

    result = {
        "xyz": xyz.astype(np.float32),
        # 3DGS stores colour as an SH DC coefficient, not as RGB.
        "rgb": ((colours - 0.5) / SH_C0).astype(np.float32),
        # Opacity and scale are stored pre-activation: logit and log.
        "opacity": np.full(xyz.shape[0], math.log(opacity_value / (1 - opacity_value)),
                           dtype=np.float32),
        "scales": np.log(np.clip(scales, 1e-8, None)).astype(np.float32),
        # Identity quaternion, w first.
        "rotation": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                            (xyz.shape[0], 1)),
    }

    write_ply(output, result["xyz"], result["rgb"], result["scales"],
              result["opacity"], result["rotation"])

    receipt = {
        "schema_version": "splat_export_receipt_v1",
        "classification": "PROVEN",
        "representation": "3d_gaussian_splats",
        "image": str(image_path),
        "image_sha256": sha256(image_path),
        "image_dimensions": [width, height],
        "model": model_name,
        "device": device,
        "output_ply": str(output),
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        "splat_count": int(xyz.shape[0]),
        "grid_stride": stride,
        "sh_degree": 0,
        "opacity": opacity_value,
        "disc_thickness_ratio": thickness,
        "masked_pixel_fraction": float(mask.mean()),
        "depth_range": [float(points[..., 2][np.isfinite(points[..., 2])].min()),
                        float(points[..., 2][np.isfinite(points[..., 2])].max())],
        "camera": {"fov_x_deg": fov_x, "fov_y_deg": fov_y,
                   "aspect_ratio": width / height, "projection": "perspective"},
        "notes": "depth-initialised, not multi-view optimised",
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
    parser.add_argument("--model", default="Ruicheng/moge-2-vitb-normal")
    parser.add_argument("--max-splats", type=int, default=2_000_000)
    parser.add_argument("--thickness", type=float, default=0.25)
    args = parser.parse_args(argv)

    receipt = export(Path(args.image), Path(args.output), Path(args.receipt),
                     args.model, args.max_splats, thickness=args.thickness)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
