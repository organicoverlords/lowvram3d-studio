"""ROUTE_C: single-image geometry from the locally installed Stable Fast 3D, as a comparator.

A second, materially different single-image backend, so geometry quality is not judged only against
more samples from the one generator that already failed. Nothing here becomes the production
generator; this exists to answer whether a different backend does better on the same picture.

Two deviations from the stock `run.py`, both required on this machine:

* background removal is skipped. The input is the pipeline's own matte, and rembg re-cut on this
  particular source drops every hanging ornament and its cord and blackens the staff ring, so
  running it would compare a different subject.
* autocast uses float16 below compute capability 8.0. The stock script asks for bfloat16, which
  Turing does not implement, so on a 1660 Super it either falls over or silently crawls.
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
    parser.add_argument("--sf3d-root", default=r"C:\AI\StableFast3D")
    parser.add_argument("--pretrained-model", default=r"C:\AI\StableFast3D\models\stable-fast-3d")
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    parser.add_argument("--texture-resolution", type=int, default=1024)
    parser.add_argument("--remesh", default="none", choices=("none", "triangle", "quad"))
    parser.add_argument("--target-vertex-count", type=int, default=-1)
    args = parser.parse_args()

    sys.path.insert(0, args.sf3d_root)
    from contextlib import nullcontext

    import torch
    from PIL import Image

    from sf3d.system import SF3D
    from sf3d.utils import resize_foreground

    report: dict = {
        "route": "ROUTE_C_SF3D",
        "backend": "stable-fast-3d",
        "image": args.image,
        "image_sha256": sha256(Path(args.image)),
        "pretrained_model": args.pretrained_model,
        "background_removal": "skipped: input is the pipeline matte",
        "success": False,
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; SF3D baking requires the GPU")
        device = "cuda"
        capability = torch.cuda.get_device_capability(0)
        report["gpu"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = list(capability)
        # bfloat16 needs SM80. Turing reports it as available through autocast and then either
        # errors in the kernels or falls back to something unusably slow.
        dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        report["autocast_dtype"] = str(dtype).replace("torch.", "")

        image = Image.open(args.image).convert("RGBA")
        if image.getchannel("A").getextrema()[0] == 255:
            raise RuntimeError("input has no transparency; a matted RGBA source is required")
        image = resize_foreground(image, args.foreground_ratio)

        model = SF3D.from_pretrained(
            args.pretrained_model, config_name="config.yaml", weight_name="model.safetensors",
        )
        model.to(device)
        model.eval()

        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=dtype):
                mesh, _ = model.run_image(
                    [image],
                    bake_resolution=args.texture_resolution,
                    remesh=args.remesh,
                    vertex_count=args.target_vertex_count,
                )
        report["seconds"] = round(time.time() - started, 1)
        report["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)

        if isinstance(mesh, list):
            mesh = mesh[0]
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(destination), include_normals=True)

        report.update({
            "success": True,
            "output": str(destination),
            "output_sha256": sha256(destination),
            "output_bytes": destination.stat().st_size,
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.faces)),
        })
    except Exception as error:  # noqa: BLE001 - the receipt must record why, not just that
        report["error"] = f"{type(error).__name__}: {error}"[:1200]

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["success"]:
        print(f"ROUTE_C_OK triangles={report['triangles']} "
              f"peak_vram={report.get('peak_vram_mb')}MB {report['seconds']}s", flush=True)
    else:
        print(f"ROUTE_C_FAILED {report.get('error')}", flush=True)
    raise SystemExit(0 if report["success"] else 2)


if __name__ == "__main__":
    main()
