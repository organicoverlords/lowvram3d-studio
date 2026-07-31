from __future__ import annotations

import argparse
import json
import site
from pathlib import Path


SHIM_SOURCE = r'''from __future__ import annotations

import numpy as np
import torch
from skimage import measure


def marching_cubes(volume: torch.Tensor, threshold: float):
    """CPU-compatible replacement for torchmcubes.marching_cubes.

    TripoSR only requires vertices and triangle indices. The density volume is
    moved to CPU for scikit-image's Lewiner marching-cubes implementation, then
    returned as CPU torch tensors. TripoSR moves the result back to its active
    device after extraction.
    """
    if not isinstance(volume, torch.Tensor):
        volume = torch.as_tensor(volume)
    array = np.ascontiguousarray(volume.detach().float().cpu().numpy())
    level = float(threshold)
    if array.ndim != 3:
        raise ValueError(f"marching_cubes expects a 3D volume, got shape {array.shape}")
    minimum = float(array.min())
    maximum = float(array.max())
    if not minimum <= level <= maximum:
        empty_vertices = torch.empty((0, 3), dtype=torch.float32)
        empty_faces = torch.empty((0, 3), dtype=torch.int64)
        return empty_vertices, empty_faces
    vertices, faces, _normals, _values = measure.marching_cubes(
        array,
        level=level,
        method="lewiner",
        allow_degenerate=False,
    )
    return (
        torch.from_numpy(np.ascontiguousarray(vertices, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int64)),
    )
'''


def resolve_site_packages() -> Path:
    candidates = [Path(item) for item in site.getsitepackages()]
    if not candidates:
        raise RuntimeError("No site-packages directory was reported by Python.")
    return candidates[0]


def verify() -> dict[str, object]:
    import torch
    from torchmcubes import marching_cubes

    axis = torch.linspace(-1.0, 1.0, 24)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    sphere = x.square() + y.square() + z.square() - 0.45
    vertices, faces = marching_cubes(sphere, 0.0)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("CPU marching-cubes verification produced no vertices.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("CPU marching-cubes verification produced no faces.")
    return {
        "backend": "scikit-image-lewiner-cpu",
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof", default="")
    parser.add_argument("--target", default="")
    args = parser.parse_args()

    package_dir = (Path(args.target).resolve() if args.target else resolve_site_packages()) / "torchmcubes"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(SHIM_SOURCE, encoding="utf-8")

    if args.target:
        import sys
        sys.path.insert(0, str(Path(args.target).resolve()))
    result = verify()
    result["package"] = str(package_dir)
    payload = json.dumps(result, indent=2)
    if args.proof:
        proof = Path(args.proof)
        proof.parent.mkdir(parents=True, exist_ok=True)
        proof.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
