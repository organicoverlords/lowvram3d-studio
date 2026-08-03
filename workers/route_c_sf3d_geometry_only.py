"""ROUTE_C_GEOM: Stable Fast 3D geometry only, stopping before UV unwrap and texture bake.

The existing `route_c_sf3d_generate.py` calls `SF3D.run_image`, which always unwraps UVs and runs
the compiled texture baker. The ship benchmark judges geometry before any texturing decision, so
this worker reproduces `run_image`'s conditioning batch exactly and then stops after
`triplane_to_meshes` -- no `unwrap_uv`, no `TextureBaker`, no illumination estimation.

Same two machine-specific deviations as the full route:

* background removal is skipped; the input is the pipeline's own proven matte.
* autocast uses float16 below compute capability 8.0, because Turing has no bfloat16 kernels.
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
    args = parser.parse_args()

    sys.path.insert(0, args.sf3d_root)

    import numpy as np
    import torch
    import trimesh
    from PIL import Image

    from sf3d.system import SF3D
    from sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w, resize_foreground

    report: dict = {
        "route": "ROUTE_C_SF3D_GEOMETRY_ONLY",
        "backend": "stable-fast-3d",
        "scope": "geometry_only",
        "uv_unwrap": False,
        "texture_bake": False,
        "illumination_estimation": False,
        "image": args.image,
        "image_sha256": sha256(Path(args.image)),
        "pretrained_model": args.pretrained_model,
        "background_removal": "skipped: input is the pipeline matte",
        "foreground_ratio": args.foreground_ratio,
        "success": False,
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; SF3D requires the GPU")
        device = "cuda"
        capability = torch.cuda.get_device_capability(0)
        report["gpu"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = list(capability)
        report["torch"] = torch.__version__
        report["python"] = sys.version.split()[0]
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
        report["isosurface_resolution"] = int(model.cfg.isosurface_resolution)
        report["cond_image_size"] = int(model.cfg.cond_image_size)

        torch.cuda.reset_peak_memory_stats()
        started = time.time()

        # Same batch construction as SF3D.run_image, kept in step with the stock code path so the
        # geometry compared here is the geometry the full route would have textured.
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=dtype):
                mask_cond, rgb_cond = model.prepare_image(image)
                c2w_cond = default_cond_c2w(model.cfg.default_distance).to(device)
                intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_deg(
                    model.cfg.default_fovy_deg,
                    model.cfg.cond_image_size,
                    model.cfg.cond_image_size,
                )
                batch = {
                    "rgb_cond": rgb_cond,
                    "mask_cond": mask_cond,
                    "c2w_cond": c2w_cond.view(1, 1, 4, 4),
                    "intrinsic_cond": intrinsic.to(device).view(1, 1, 3, 3),
                    "intrinsic_normed_cond": intrinsic_normed_cond.to(device).view(1, 1, 3, 3),
                }
                batch["rgb_cond"] = model.image_processor(batch["rgb_cond"], model.cfg.cond_image_size)
                batch["mask_cond"] = model.image_processor(batch["mask_cond"], model.cfg.cond_image_size)
                scene_codes, _direct = model.get_scene_codes(batch)

            # generate_mesh disables autocast around isosurfacing; marching tetrahedra needs Float.
            meshes = model.triplane_to_meshes(scene_codes)

        report["seconds"] = round(time.time() - started, 1)
        report["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)

        mesh = meshes[0]
        if mesh.v_pos.shape[0] == 0:
            raise RuntimeError("SF3D produced an empty mesh")
        vertices = mesh.v_pos.detach().float().cpu().numpy().astype(np.float64)
        faces = mesh.t_pos_idx.detach().long().cpu().numpy()
        report["raw_vertices"] = int(vertices.shape[0])
        report["raw_triangles"] = int(faces.shape[0])

        exported = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        exported.export(str(destination))

        report.update({
            "success": True,
            "output": str(destination),
            "output_sha256": sha256(destination),
            "output_bytes": destination.stat().st_size,
            "vertices": int(len(exported.vertices)),
            "triangles": int(len(exported.faces)),
            "bounds_min": exported.bounds[0].tolist(),
            "bounds_max": exported.bounds[1].tolist(),
        })
    except Exception as error:  # noqa: BLE001 - the receipt must record why, not just that
        report["error"] = f"{type(error).__name__}: {error}"[:1200]

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["success"]:
        print(f"ROUTE_C_GEOM_OK triangles={report['triangles']} "
              f"peak_vram={report.get('peak_vram_mb')}MB {report['seconds']}s", flush=True)
    else:
        print(f"ROUTE_C_GEOM_FAILED {report.get('error')}", flush=True)
    raise SystemExit(0 if report["success"] else 2)


if __name__ == "__main__":
    main()
