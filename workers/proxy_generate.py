from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def newest_glb(folder: Path) -> Path | None:
    files = list(folder.rglob("*.glb"))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def run_sf3d(python: str, root: Path, image: Path, output: Path) -> bool:
    if not python or not (root / "run.py").is_file():
        return False
    work = output.parent / "sf3d"
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SF3D_USE_CPU"] = "1"
    subprocess.check_call([
        python, str(root / "run.py"), str(image), "--output-dir", str(work),
        "--texture-resolution", "1024", "--device", "cpu",
    ], cwd=str(root), env=env)
    result = newest_glb(work)
    if result:
        shutil.copy2(result, output)
        return True
    return False


def run_triposr(python: str, root: Path, image: Path, output: Path) -> bool:
    if not python or not (root / "run.py").is_file():
        return False
    work = output.parent / "triposr"
    work.mkdir(parents=True, exist_ok=True)
    # --bake-texture routes the export through xatlas.export(), which always writes Wavefront
    # OBJ even when the filename ends in .glb, producing a file no glTF reader accepts. The
    # Blender stages bake PBR maps from the source image anyway, so the vertex-coloured
    # trimesh GLB export is both valid and sufficient as the high-poly source.
    # Marching-cubes resolution drives the face count of every downstream stage. 192 remains the
    # production fallback default, while bounded geometry experiments may raise it explicitly.
    mc_resolution = os.environ.get("LOWVRAM3D_TRIPOSR_MC", "192")
    chunk_size = os.environ.get("LOWVRAM3D_TRIPOSR_CHUNK", "1024")
    subprocess.check_call([
        python, str(root / "run.py"), str(image), "--output-dir", str(work),
        "--chunk-size", chunk_size, "--mc-resolution", mc_resolution,
        "--model-save-format", "glb",
    ], cwd=str(root))
    result = newest_glb(work)
    if result:
        shutil.copy2(result, output)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sf3d-python", default="")
    parser.add_argument("--sf3d-root", default="")
    parser.add_argument("--tripo-python", default="")
    parser.add_argument("--tripo-root", default="")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Path(args.input_image)

    preference = os.environ.get("LOWVRAM3D_PROXY_BACKEND", "auto").strip().lower()
    if preference not in {"auto", "sf3d", "triposr"}:
        raise RuntimeError(
            "LOWVRAM3D_PROXY_BACKEND must be one of auto, sf3d or triposr; "
            f"got {preference!r}"
        )

    if preference == "triposr":
        if run_triposr(args.tripo_python, Path(args.tripo_root), image, output):
            return
        if run_sf3d(args.sf3d_python, Path(args.sf3d_root), image, output):
            return
    elif preference == "sf3d":
        if run_sf3d(args.sf3d_python, Path(args.sf3d_root), image, output):
            return
        if run_triposr(args.tripo_python, Path(args.tripo_root), image, output):
            return
    else:
        if run_sf3d(args.sf3d_python, Path(args.sf3d_root), image, output):
            return
        if run_triposr(args.tripo_python, Path(args.tripo_root), image, output):
            return

    raise RuntimeError(
        f"No usable SF3D or TripoSR fallback installation was found for preference {preference}"
    )


if __name__ == "__main__":
    main()
