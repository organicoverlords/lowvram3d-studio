"""ROUTE_D_GEOM: TripoSR geometry only, as the third single-image backend in the ship benchmark.

Stock `run.py` either bakes a texture atlas or writes vertex colours; both are texturing decisions
the benchmark has not reached yet. This worker stops after `extract_mesh` with vertex colour off,
so what is compared is marching-cubes geometry and nothing else.

Background removal is skipped for the same reason as the SF3D route: the input is the pipeline's
own proven matte, and re-cutting it with rembg would compare a different subject. The matte is
composited onto the 0.5 grey plate exactly as `run.py` does after its own rembg pass, so TripoSR
sees the conditioning it expects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="pre-matted RGBA source")
    parser.add_argument("--output", required=True, help="destination .glb")
    parser.add_argument("--report", required=True)
    parser.add_argument("--triposr-root", default=r"C:\AI\TripoSR")
    parser.add_argument("--pretrained-model", default="stabilityai/TripoSR")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    parser.add_argument("--conditioning-copy", default=None,
                        help="optional path to write the exact grey-plate conditioning fed to TSR")
    args = parser.parse_args()

    sys.path.insert(0, args.triposr_root)

    import numpy as np
    import torch
    from PIL import Image

    from tsr.system import TSR
    from tsr.utils import resize_foreground

    report: dict = {
        "route": "ROUTE_D_TRIPOSR_GEOMETRY_ONLY",
        "backend": "triposr",
        "scope": "geometry_only",
        "vertex_colour": False,
        "texture_bake": False,
        "image": args.image,
        "image_sha256": sha256(Path(args.image)),
        "pretrained_model": args.pretrained_model,
        "background_removal": "skipped: input is the pipeline matte",
        "chunk_size": args.chunk_size,
        "mc_resolution": args.mc_resolution,
        "foreground_ratio": args.foreground_ratio,
        "success": False,
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; TripoSR benchmark requires the GPU")
        device = "cuda:0"
        report["gpu"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = list(torch.cuda.get_device_capability(0))
        report["torch"] = torch.__version__
        report["python"] = sys.version.split()[0]

        source = Image.open(args.image).convert("RGBA")
        if source.getchannel("A").getextrema()[0] == 255:
            raise RuntimeError("input has no transparency; a matted RGBA source is required")
        source = resize_foreground(source, args.foreground_ratio)
        # Identical compositing to run.py's post-rembg branch: premultiply onto a 0.5 grey plate.
        rgba = np.array(source).astype(np.float32) / 255.0
        plate = rgba[:, :, :3] * rgba[:, :, 3:4] + (1 - rgba[:, :, 3:4]) * 0.5
        conditioning = Image.fromarray((plate * 255.0).astype(np.uint8))
        if args.conditioning_copy:
            Path(args.conditioning_copy).parent.mkdir(parents=True, exist_ok=True)
            conditioning.save(args.conditioning_copy)
            report["conditioning_copy"] = args.conditioning_copy
            report["conditioning_copy_sha256"] = sha256(Path(args.conditioning_copy))

        model = TSR.from_pretrained(
            args.pretrained_model, config_name="config.yaml", weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(args.chunk_size)
        model.to(device)

        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        with torch.no_grad():
            scene_codes = model([conditioning], device=device)
            meshes = model.extract_mesh(scene_codes, False, resolution=args.mc_resolution)
        report["seconds"] = round(time.time() - started, 1)
        report["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)

        mesh = meshes[0]
        if len(mesh.vertices) == 0:
            raise RuntimeError("TripoSR produced an empty mesh")

        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(destination))

        report.update({
            "success": True,
            "output": str(destination),
            "output_sha256": sha256(destination),
            "output_bytes": destination.stat().st_size,
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.faces)),
            "bounds_min": mesh.bounds[0].tolist(),
            "bounds_max": mesh.bounds[1].tolist(),
        })
    except Exception as error:  # noqa: BLE001 - the receipt must record why, not just that
        report["error"] = f"{type(error).__name__}: {error}"[:1200]

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["success"]:
        print(f"ROUTE_D_GEOM_OK triangles={report['triangles']} "
              f"peak_vram={report.get('peak_vram_mb')}MB {report['seconds']}s", flush=True)
    else:
        print(f"ROUTE_D_GEOM_FAILED {report.get('error')}", flush=True)
    raise SystemExit(0 if report["success"] else 2)


if __name__ == "__main__":
    main()
