"""Focused recovery entrypoint: produce one textured baseline GLB from an existing job.

This exists because a full pipeline re-run costs an hour of self-hosted GPU time to recover a
result whose expensive inputs -- the UV unwrap and the projection views -- are already on disk and
already valid. It drives the three existing raster-route stages directly:

    raster_cleanup_extract.py  ->  raster_project.py  ->  raster_export.py

and adds nothing to the route itself. In particular there is no new "baseline mode": the stages run
with their production arguments, and the only choices made here are which views directory to read
and which atlas size to request.

The views directory is selected EXPLICITLY rather than discovered. A job may carry several view
sets of differing validity -- job 26a37e41 carries a views/mv_adapter set whose six PNGs are
byte-identical pure black, because the SD2.1 latents went non-finite on Turing fp16 and diffusers
cast NaN to 0 on the way to uint8. Auto-discovery would happily texture the mesh with those and
emit a structurally perfect, entirely black GLB. Naming views/projection explicitly, and verifying
its pixels before use, is the whole point of this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
# raster_cleanup_extract.py only rasterises these four; top/bottom are never sampled.
REQUIRED_VIEWS = ("front", "right", "back", "left")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_image(path: Path) -> dict:
    """Content statistics sufficient to distinguish a real render from a NaN-to-black placeholder."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {"readable": False, "path": str(path)}
    rgb = image[:, :, :3].astype(np.float64)
    alpha = image[:, :, 3].astype(np.float64) if image.shape[2] == 4 else None
    flat = rgb.reshape(-1, rgb.shape[-1])
    return {
        "readable": True,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "resolution": [int(image.shape[1]), int(image.shape[0])],
        "all_finite": bool(np.isfinite(rgb).all()),
        "rgb_min": float(rgb.min()),
        "rgb_max": float(rgb.max()),
        "rgb_mean": round(float(rgb.mean()), 4),
        "rgb_std": round(float(rgb.std()), 4),
        "unique_colours": int(len(np.unique(flat, axis=0))),
        "alpha_mean": round(float(alpha.mean()), 4) if alpha is not None else None,
        "nonzero_pixel_percent": round(float((rgb.max(axis=2) > 0).mean() * 100), 4),
    }


def assert_views_usable(views_dir: Path, metadata: Path) -> dict:
    """Refuse to texture from a view set that is black, constant, or duplicated.

    This runs before any Blender work so the failure is cheap and the reason is explicit, rather
    than surfacing an hour later as a black model that passes every structural check.
    """
    if not metadata.is_file():
        raise SystemExit(f"BLOCKER: view metadata missing: {metadata}")
    policy = json.loads(metadata.read_text(encoding="utf-8"))
    semantic = set(policy["policy"]["semantic_projection"])
    contributing = [
        entry["view"]
        for entry in policy["views"]
        if entry["source_type"] in semantic and float(entry["confidence"]) > 0.0
    ]
    if not contributing:
        raise SystemExit(
            f"BLOCKER: {metadata} bars every view from semantic projection; nothing would be painted"
        )

    stats: dict[str, dict] = {}
    digests: dict[str, list[str]] = {}
    for name in REQUIRED_VIEWS:
        path = views_dir / f"{name}.png"
        if not path.is_file():
            raise SystemExit(f"BLOCKER: required view missing: {path}")
        info = describe_image(path)
        if not info["readable"]:
            raise SystemExit(f"BLOCKER: unreadable view: {path}")
        stats[name] = info
        digests.setdefault(info["sha256"], []).append(name)

    duplicated = {sha: names for sha, names in digests.items() if len(names) > 1}
    if duplicated:
        raise SystemExit(f"BLOCKER: views are byte-identical (NaN-to-black signature): {duplicated}")

    for name in contributing:
        info = stats.get(name)
        if info is None:
            continue
        if not info["all_finite"]:
            raise SystemExit(f"BLOCKER: view {name} contains non-finite pixels")
        if info["unique_colours"] <= 1 or info["rgb_std"] <= 1.0:
            raise SystemExit(
                f"BLOCKER: contributing view {name} is constant/black "
                f"(unique={info['unique_colours']} std={info['rgb_std']})"
            )

    return {"contributing_views": contributing, "view_stats": stats}


def run(
    label: str,
    command: list[str],
    env: dict | None = None,
    cwd: Path | None = None,
    artifacts: dict[str, Path] | None = None,
) -> dict:
    """Run a stage and verify it actually produced what it promised.

    Exit code alone is not evidence. Blender reports an uncaught Python exception on stderr and
    still exits 0, so a stage can abort mid-way and look clean; the declared artifacts are the
    only trustworthy signal that the stage did its job.
    """
    printable = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in command)
    print(f"\n=== {label} ===\n{printable}\n", flush=True)
    started = time.time()
    merged = {**os.environ, **(env or {})}
    completed = subprocess.run(
        [str(c) for c in command], env=merged, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    elapsed = round(time.time() - started, 1)
    tail = "\n".join((completed.stdout or "").strip().splitlines()[-15:])
    print(tail, flush=True)
    def fail(reason: str) -> None:
        print("--- stderr tail ---", flush=True)
        print("\n".join((completed.stderr or "").strip().splitlines()[-30:]), flush=True)
        raise SystemExit(f"BLOCKER: stage '{label}' {reason} after {elapsed}s\ncommand: {printable}")

    if completed.returncode != 0:
        fail(f"exited {completed.returncode}")
    for name, path in (artifacts or {}).items():
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            fail(f"exited 0 but artifact '{name}' is missing or empty: {path}")
    print(f"[{label}] ok in {elapsed}s", flush=True)
    return {"stage": label, "command": printable, "exit_code": 0, "elapsed_seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--views-subdir", default="views/projection",
                        help="Explicitly chosen, pixel-verified view set. Never auto-discovered.")
    parser.add_argument("--mesh", default="uv/game_ready_uv.glb",
                        help="Existing UV-unwrapped mesh; reused as-is, never regenerated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="shaman_textured_baseline")
    parser.add_argument("--atlas-size", type=int, default=512)
    parser.add_argument("--cleanup-mode", default="single_subject_strict",
                        choices=("conservative", "single_subject_strict"))
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    args = parser.parse_args()

    job = Path(args.job_dir)
    views = job / args.views_subdir
    mesh = job / args.mesh
    metadata = views / "view_metadata.json"
    outdir = Path(args.output_dir)
    work = outdir / "work"
    work.mkdir(parents=True, exist_ok=True)

    if not mesh.is_file():
        raise SystemExit(f"BLOCKER: UV mesh missing: {mesh}")

    print(f"mesh   : {mesh} ({mesh.stat().st_size} bytes)")
    print(f"views  : {views}")
    verification = assert_views_usable(views, metadata)
    print(f"contributing views: {verification['contributing_views']}")

    pythonpath = os.pathsep.join((str(REPO_ROOT / "blender"), str(REPO_ROOT / "src")))
    blender_env = {"PYTHONPATH": pythonpath}
    stages = []

    cleaned = work / "mesh_clean.glb"
    npz = work / "mesh_clean.npz"
    cleanup_report = work / "geometry_cleanup_report.json"
    stages.append(run("geometry_repair_extract", [
        args.blender, "--background", "--python-use-system-env",
        "--python", REPO_ROOT / "blender" / "raster_cleanup_extract.py", "--",
        "--input", mesh, "--output-glb", cleaned, "--output-npz", npz,
        "--report", cleanup_report, "--cleanup-mode", args.cleanup_mode,
    ], env=blender_env, artifacts={"mesh": cleaned, "npz": npz, "report": cleanup_report}))

    project_report = work / "raster_report.json"
    observed_triangles = work / "observed_triangles.npy"
    stages.append(run("raster_project", [
        sys.executable, REPO_ROOT / "workers" / "raster_project.py",
        "--npz", npz, "--views-dir", views, "--view-metadata", metadata,
        "--output-dir", work, "--atlas-size", str(args.atlas_size),
        "--progress", work / "raster-progress.json", "--report", project_report,
    ], artifacts={"basecolor": work / "basecolor.png", "report": project_report,
                  "observed_triangles": observed_triangles}))

    atlas = work / "basecolor.png"
    glb = outdir / f"{args.name}.glb"
    texture = outdir / f"{args.name}_basecolor.png"
    export_report = work / "raster_export_report.json"
    stages.append(run("raster_export", [
        args.blender, "--background", "--python-use-system-env",
        "--python", REPO_ROOT / "blender" / "raster_export.py", "--",
        "--cleaned-mesh", cleaned, "--atlas", atlas,
        "--output", glb, "--texture", texture, "--atlas-size", str(args.atlas_size),
        "--observed-triangles", observed_triangles, "--report", export_report,
    ], env=blender_env, artifacts={"glb": glb, "texture": texture, "report": export_report}))

    summary = {
        "success": True,
        "glb": str(glb),
        "glb_bytes": glb.stat().st_size,
        "glb_sha256": sha256_of(glb),
        "basecolor": str(texture),
        "atlas_size": args.atlas_size,
        "cleanup_mode": args.cleanup_mode,
        "source_job": str(job),
        "source_mesh": str(mesh),
        "views_dir": str(views),
        "view_verification": verification,
        "raster_report": json.loads(project_report.read_text(encoding="utf-8")),
        "cleanup_report_path": str(cleanup_report),
        "stages": stages,
    }
    (outdir / f"{args.name}_recovery.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nRECOVERY_BASELINE_WRITTEN {glb} bytes={summary['glb_bytes']} sha256={summary['glb_sha256']}", flush=True)


if __name__ == "__main__":
    main()
