"""Create winding-corrected copies of fresh comparison GLBs only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds\moge_comparison")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python make_moge_comparison_winding.py NAME")
    folder = ROOT / sys.argv[1]
    mesh = trimesh.load(folder / "official.glb", force="mesh", process=False)
    mesh.faces = np.asarray(mesh.faces)[:, [0, 2, 1]]
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.fix_normals()
    out = folder / "official_winding.glb"
    mesh.export(out, file_type="glb")
    print(out)


if __name__ == "__main__":
    main()
