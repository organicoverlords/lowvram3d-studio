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

#: The vendor defaults are 2048 for both. Multiview diffusion survives them on
#: 6 GB; the bake does not. `back_project` scatters every view into the atlas
#: through `scatter_add_nd_with_count`, so the transient is driven by
#: views * render_size^2 alongside a live texture_size^2 accumulator, and at
#: 2048/2048 it dies with "fatal: Memory allocation failure" followed by an
#: opaque CUDA unknown error -- after the expensive part has already succeeded.
#: Halving both quarters that transient.
DEFAULT_RENDER_SIZE = 1024
DEFAULT_TEXTURE_SIZE = 1024


def run(mesh_path: Path, image_path: Path, out_path: Path,
        model_root: Path, offload: bool,
        render_size: int = DEFAULT_RENDER_SIZE,
        texture_size: int = DEFAULT_TEXTURE_SIZE,
        cpu_rng: bool = True) -> dict:
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

    # The renderer is built inside Hunyuan3DTexGenPipeline.__init__, which runs
    # during from_pretrained -- not lazily at __call__ as this code previously
    # assumed. Setting config afterwards therefore changed nothing: every run
    # silently used the vendor defaults of 2048/2048, and --render-size and
    # --texture-size were decorative. A "4096" bake came out with a 2048 atlas
    # and byte-for-byte the same texel density as the 2048 run, which is what
    # exposed it.
    #
    # So the renderer has to be rebuilt after the config is changed. It is a
    # plain object holding rasteriser state, not something the loaded weights
    # point back into, so replacing it is safe.
    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

    pipeline.config.render_size = int(render_size)
    pipeline.config.texture_size = int(texture_size)
    pipeline.render = MeshRender(
        default_resolution=pipeline.config.render_size,
        texture_size=pipeline.config.texture_size)

    # Fail loudly rather than emit another file whose name lies about its atlas.
    # MeshRender stores this as a (width, height) tuple, not the int it was
    # given, so read it back the way it is actually held.
    actual = pipeline.render.texture_size
    actual_side = int(actual[0]) if isinstance(actual, (tuple, list)) else int(actual)
    if actual_side != int(texture_size):
        raise RuntimeError(
            f"renderer texture_size is {actual}, asked for {texture_size}")
    loaded = time.time()

    scene = trimesh.load(mesh_path, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene

    image = Image.open(image_path)
    image = image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")

    # Move the multiview sampler's RNG off the GPU.
    #
    # hy3dgen/texgen/utils/multiview_utils.py:69 builds
    #     torch.Generator(device=self.pipeline.device).manual_seed(0)
    # which is a CUDA generator, so every randn in the diffusion loop runs on
    # the card. That is exactly where the bluetree paint died --
    # diffusers/schedulers/scheduling_euler_ancestral_discrete.py:427 calling
    # randn_tensor -- alongside an nvlddmkm Id 13 in the system log.
    #
    # diffusers' randn_tensor keys off the generator's device: give it a CPU
    # generator and the noise is drawn on the host and copied over, removing
    # those launches from the GPU entirely. It does not fix the underlying
    # driver fault, and it is not a claim that it will; it removes one class of
    # kernel launch from the window where the fault happens.
    #
    # Patched around the call rather than edited into the vendor tree, so the
    # checkout stays pristine and this reverts by deleting these lines. The
    # swap is global to torch for the duration, which is why it is scoped as
    # tightly as possible and restored in the finally.
    real_generator = torch.Generator

    def cpu_generator(*args, **kwargs):
        kwargs.pop("device", None)
        return real_generator(*args, device="cpu", **kwargs)

    if cpu_rng:
        torch.Generator = cpu_generator
    # The vendor call is opaque -- one function in, a textured mesh out -- so
    # these two lines are the only visibility this worker can offer. They matter
    # because of what falls between them: the vendor preprocesses the mesh on the
    # CPU before any diffusion starts, and on a large undecimated mesh that phase
    # runs for many minutes emitting nothing. Its own progress bars only appear
    # once sampling begins, so "loaded, then silence" is not a hang -- it is
    # preprocessing, and the face count below is what predicts how long it lasts.
    faces = getattr(getattr(mesh, "faces", None), "__len__", lambda: None)()
    print(f"[paint] entering vendor pipeline: {faces} faces, "
          f"render {render_size}, texture {texture_size}", flush=True)

    # Stage boundaries, taken from the vendor pipeline itself.
    #
    # Hunyuan3DPaintPipeline.__call__ is one opaque call from outside, and its
    # own progress bars only exist for the diffusion step. Everything around
    # that -- multiview normal and position rendering, the bake, the inpaint --
    # runs silently, and on this machine those phases have run for tens of
    # minutes with nothing on screen. Wrapping the named methods turns "blank
    # console" into "which phase, and how long it has been in it", which is the
    # difference between waiting and knowing.
    #
    # Instance-level, so nothing about the vendor package on disk changes.
    def instrument(target, names):
        for name in names:
            original = getattr(target, name, None)
            if not callable(original):
                continue

            def wrapper(*args, _name=name, _original=original, **kwargs):
                print(f"[paint] >> {_name}", flush=True)
                begun = time.time()
                try:
                    return _original(*args, **kwargs)
                finally:
                    print(f"[paint] << {_name} took {time.time() - begun:.1f}s",
                          flush=True)

            setattr(target, name, wrapper)

    instrument(pipeline, ("render_normal_multiview", "render_position_multiview",
                          "bake_from_multiview", "texture_inpaint",
                          "recenter_image"))

    # Do not re-unwrap a mesh that already has UVs.
    #
    # pipelines.__call__ calls mesh_uv_wrap unconditionally, and that is
    # xatlas.parametrize over every face, single threaded, printing nothing.
    # On the moss titan -- 286,994 faces whose surface is thousands of thin
    # hanging strands, each its own chart to pack -- it ran past 25 minutes
    # with the GPU idle and no output, twice, and was mistaken for a hung run
    # and for the UNet executing on the CPU. It was neither.
    #
    # The work is also redundant here: TRELLIS exports TEXCOORD_0 with a 2048
    # atlas, so a valid parameterisation already exists. The bake writes a new
    # texture into whatever layout the mesh carries; it does not require
    # xatlas's particular packing. Meshes that genuinely lack UVs still get
    # unwrapped, so nothing silently paints without a parameterisation.
    from hy3dgen.texgen import pipelines as vendor_pipelines

    vendor_unwrap = vendor_pipelines.mesh_uv_wrap

    def unwrap_only_if_needed(target):
        uv = getattr(getattr(target, "visual", None), "uv", None)
        if uv is not None and len(uv) == len(target.vertices):
            print(f"[paint] >> mesh_uv_wrap SKIPPED: mesh already carries "
                  f"{len(uv)} UVs", flush=True)
            return target
        print("[paint] >> mesh_uv_wrap: no usable UVs, running xatlas "
              "(slow on high face counts)", flush=True)
        begun = time.time()
        result = vendor_unwrap(target)
        print(f"[paint] << mesh_uv_wrap took {time.time() - begun:.1f}s",
              flush=True)
        return result

    vendor_pipelines.mesh_uv_wrap = unwrap_only_if_needed
    entered = time.time()
    try:
        textured = pipeline(mesh, image=image)
    finally:
        torch.Generator = real_generator
        print(f"[paint] vendor pipeline returned after "
              f"{time.time() - entered:.1f}s", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    textured.export(out_path)
    finished = time.time()

    # What the atlas actually came out as, read off the exported material rather
    # than echoed back from the arguments. Receipts that only repeated the
    # request said render_size 1024 / texture_size 2048 for every asset today
    # while the pipeline was silently using its own 2048/2048 defaults.
    # The vendor sometimes returns a Scene rather than a Trimesh, and a Scene has
    # no .visual -- so this silently logged null for a bake whose atlas was in
    # fact exactly the size requested. Resolve to the geometry first. A null here
    # now means the texture is genuinely missing, which is worth seeing.
    baked = None
    try:
        geometry = textured
        if hasattr(geometry, "geometry"):
            geometry = next(iter(geometry.geometry.values()))
        baked = list(geometry.visual.material.baseColorTexture.size)
    except Exception:  # pragma: no cover - depends on the vendor's return type
        pass

    return {
        "schema": "lowvram3d_hunyuan_paint_v1",
        "mesh_in": str(mesh_path),
        "image": str(image_path),
        "mesh_out": str(out_path),
        "paint_checkpoint": PAINT_SUBFOLDER,
        "baked_texture_size": baked,
        "renderer_texture_size": list(actual) if isinstance(actual, (tuple, list)) else actual,
        "model_cpu_offload": bool(offload),
        "render_size": int(render_size),
        "texture_size": int(texture_size),
        "cudnn_enabled": False,
        "cpu_rng": bool(cpu_rng),
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
    parser.add_argument("--render-size", type=int, default=DEFAULT_RENDER_SIZE)
    parser.add_argument("--texture-size", type=int, default=DEFAULT_TEXTURE_SIZE)
    parser.add_argument("--gpu-rng", action="store_true",
                        help="keep the multiview sampler's RNG on the GPU "
                             "(vendor default). Off by default: the CUDA "
                             "generator's randn is where the bluetree paint "
                             "faulted, with a matching nvlddmkm Id 13")
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run(args.mesh, args.image, args.out, args.model_root,
                 offload=not args.no_offload,
                 render_size=args.render_size, texture_size=args.texture_size,
                 cpu_rng=not args.gpu_rng)
    receipt = args.receipt or args.out.with_suffix(".paint.json")
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
