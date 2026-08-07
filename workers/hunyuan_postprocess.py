"""Hunyuan's own mesh postprocessors, which this project never ran.

TRELLIS was judged on raw decoder output because `--no-texture` silently
skipped the geometry finalizer, and the finalized mesh turned out to be an
entirely different class of object. That was a tooling failure wearing a model
failure's clothes, and the same shape of mistake is available here: Hunyuan's
generators emit a dense marching-cubes mesh, its own app runs FloaterRemover,
DegenerateFaceRemover and FaceReducer before anything downstream sees it, and
our workers have done

    mesh = pipeline(...)[0]
    mesh.export(...)

since the first Mini Turbo run. So every Mini Turbo verdict on this project was
recorded against unfinished output.

That does not mean the verdicts were wrong. A decimator cannot straighten a
hull that was reconstructed skewed, and the Mini Turbo boat's problems were
read as gross-proportion problems. This worker exists to find out which, by
separating the two questions completely: same master mesh, no new diffusion,
only the postprocess varied.

    py workers/hunyuan_postprocess.py --mesh in.glb --faces 500000 --out out.glb

Needs the HY3D2 standalone interpreter with hy3dgen on PYTHONPATH -- pymeshlab
lives there, not in the control venv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def run(mesh_path: Path, out_path: Path, faces: int,
        floaters: bool = True, degenerates: bool = True) -> dict:
    import trimesh
    from hy3dgen.shapegen import (DegenerateFaceRemover, FaceReducer,
                                  FloaterRemover)

    started = time.time()
    scene = trimesh.load(mesh_path, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    steps = [{"step": "load", "faces": int(len(mesh.faces)),
              "seconds": round(time.time() - started, 1)}]

    if floaters:
        mark = time.time()
        mesh = FloaterRemover()(mesh)
        steps.append({"step": "floater_remover", "faces": int(len(mesh.faces)),
                      "seconds": round(time.time() - mark, 1)})
    if degenerates:
        mark = time.time()
        mesh = DegenerateFaceRemover()(mesh)
        steps.append({"step": "degenerate_face_remover",
                      "faces": int(len(mesh.faces)),
                      "seconds": round(time.time() - mark, 1)})
    if faces:
        mark = time.time()
        mesh = FaceReducer()(mesh, max_facenum=faces)
        steps.append({"step": "face_reducer", "faces": int(len(mesh.faces)),
                      "seconds": round(time.time() - mark, 1)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)

    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    return {
        "schema": "lowvram3d_hunyuan_postprocess_v1",
        "mesh_in": str(mesh_path),
        "mesh_out": str(out_path),
        "target_faces": faces,
        "steps": steps,
        "faces_out": int(len(welded.faces)),
        "shells_welded": int(welded.body_count),
        "winding_consistent": bool(welded.is_winding_consistent),
        "watertight": bool(welded.is_watertight),
        "seconds": round(time.time() - started, 1),
        "note": ("Hunyuan's own postprocessors, which the generator workers on "
                 "this project have never called; every prior Mini Turbo "
                 "verdict was recorded against unfinished output"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--faces", type=int, default=500_000,
                        help="0 to skip decimation entirely")
    parser.add_argument("--no-floaters", dest="floaters", action="store_false")
    parser.add_argument("--no-degenerates", dest="degenerates",
                        action="store_false")
    args = parser.parse_args(argv)

    result = run(args.mesh, args.out, args.faces, args.floaters,
                 args.degenerates)
    args.out.with_suffix(".postprocess.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
