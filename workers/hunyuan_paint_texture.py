"""Texture a mesh with Hunyuan3D-Paint -- the vendor's own texture stage.

This project spent a long time hand-rolling the texturing step: fit an
orthographic camera to the conditioning matte by silhouette IoU, project the
source photograph through it, then fill the unobserved hemisphere with
MV-Adapter. Every part of that was ours, including the failure modes -- a
silhouette objective that compares handedness rather than facing, receipts that
claim work that did not happen, and a long chain of criteria that each answered
a question next to the real one.

Hunyuan3D ships the stage we were reimplementing. `hunyuan3d-paint-v2-0-turbo`
takes a mesh and the source image and produces view-consistent multiview colour
which it bakes to a UV atlas, with `hunyuan3d-delight-v2-0` removing baked
lighting from the input first. It never asks anyone to fit a camera, because the
generator already knows the frame its own geometry was produced in.

Both checkpoints were already on this machine, in the same HF tree as the Mini
Turbo DiT, and had never been run.

Two machine-specific notes, both previously measured, neither model-specific:

  * cuDNN's FP16 convolution path is defective on this GPU (TU116, no tensor
    cores) and produces roughly 25 percent NaN. It is disabled here.
  * 9.6 GB of paint plus 4.0 GB of delight does not fit in 6 GB, so model CPU
    offload is mandatory rather than an optimisation.

    py workers/hunyuan_paint_texture.py --mesh in.glb --image src.png \
       --out textured.glb
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

#: The HF tree that already holds the paint and delight checkpoints alongside
#: the Mini Turbo DiT. Pointing at the snapshot directly avoids a network call.
DEFAULT_MODEL_ROOT = Path(
    r"C:\AI\HY3D2\HuggingFaceHub\models--tencent--Hunyuan3D-2\snapshots"
    r"\9cd649ba6913f7a852e3286bad86bfa9a2d83dcf"
)
PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0-turbo"


def run(mesh_path: Path, image_path: Path, out_path: Path,
        model_root: Path, offload: bool) -> dict:
    # torch must be imported before custom_rasterizer_kernel: the extension links
    # against torch's CUDA DLLs and cannot find them until torch has put its own
    # directory on the DLL search path. Importing it the other way round fails
    # with a bare "DLL load failed", which reads like a broken build and is not.
    import torch

    torch.backends.cudnn.enabled = False

    import trimesh
    from PIL import Image
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    started = time.time()
    pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        str(model_root), subfolder=PAINT_SUBFOLDER)
    if offload:
        pipeline.enable_model_cpu_offload()
    loaded = time.time()

    scene = trimesh.load(mesh_path, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene

    image = Image.open(image_path)
    image = image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")

    textured = pipeline(mesh, image=image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    textured.export(out_path)
    finished = time.time()

    return {
        "schema": "lowvram3d_hunyuan_paint_v1",
        "mesh_in": str(mesh_path),
        "image": str(image_path),
        "mesh_out": str(out_path),
        "paint_checkpoint": PAINT_SUBFOLDER,
        "model_cpu_offload": bool(offload),
        "cudnn_enabled": False,
        "load_seconds": round(loaded - started, 1),
        "paint_seconds": round(finished - loaded, 1),
        "note": ("vendor texture stage; no fitted camera, no silhouette "
                 "objective, no hand-rolled projection"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--no-offload", action="store_true",
                        help="keep everything resident; needs far more than 6 GB")
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run(args.mesh, args.image, args.out, args.model_root,
                 offload=not args.no_offload)
    receipt = args.receipt or args.out.with_suffix(".paint.json")
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
