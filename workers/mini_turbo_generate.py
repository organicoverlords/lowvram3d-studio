"""Run one Hunyuan3D-2 Mini Turbo geometry generation directly against the installed weights.

This is the Mini Turbo lane without the ComfyUI transport. The ComfyUI install on this machine
has no Python environment, no built ComfyUI-3D-Pack extensions and no Hunyuan3D weights in its
model tree, so the HTTP route cannot reach the model at all. The weights themselves are present
and healthy under the standalone Hunyuan3D install, so this worker loads
``hunyuan3d-dit-v2-mini-turbo`` directly.

Hard rules enforced here:

* Mini Turbo is the only generator. There is no fallback path of any kind.
* Any failure raises and stops the run.
* The generated GLB is written once and preserved byte-for-byte; nothing downstream mutates it.
* Only a file created during this exact execution is accepted, and only if it is a real glTF
  binary rather than an empty file, a renamed OBJ or a stale previous output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

GLB_MAGIC = b"glTF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_real_glb(path: Path, started_at: float) -> None:
    """Reject anything that is not a genuine, freshly written GLB."""
    if not path.is_file():
        raise RuntimeError(f"Mini Turbo produced no mesh at {path}")
    size = path.stat().st_size
    if size < 4096:
        raise RuntimeError(f"Mini Turbo mesh is empty or too small ({size} bytes): {path}")
    # Requirement: only accept a file created/modified during this prompt execution.
    mtime = path.stat().st_mtime
    if mtime < started_at - 1.0:
        raise RuntimeError(
            f"Mini Turbo mesh at {path} predates this run "
            f"(mtime={mtime}, run started {started_at}); refusing stale output"
        )
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != GLB_MAGIC:
        raise RuntimeError(
            f"Mini Turbo output is not a binary glTF (magic={magic!r}): {path}. "
            "Refusing OBJ/PLY/preview renamed to .glb"
        )


def load_conditioning_image(image_path: Path):
    from PIL import Image

    image = Image.open(image_path)
    if image.mode == "RGBA" and image.getchannel("A").getextrema()[0] < 255:
        return image
    # Mini Turbo is trained on background-free subjects; an opaque input must be matted first.
    from hy3dgen.rembg import BackgroundRemover

    return BackgroundRemover()(image.convert("RGB"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    # Pre-matted RGBA input. rembg destroys this source (drops every hanging ornament and its cord,
    # blackens the staff ring), so the caller mattes it with workers/shaman_matte.py instead.
    parser.add_argument("--conditioning-image", default="")
    parser.add_argument("--expected-image-sha256", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-mini-turbo")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    # Descending ladder of geometry settings. Only VRAM pressure moves us down it; the generator
    # never changes. Each entry is (octree_resolution, num_chunks).
    parser.add_argument("--octree-ladder", default="384:3000,320:2000,256:1500")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output = Path(args.output).resolve()
    result_path = Path(args.result_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "success": False,
        "selected_generator": "mini_turbo",
        "generator_locked": True,
        "fallback_generators_allowed": [],
        "transport": "direct-local-weights",
        "comfyui_url": None,
        "comfyui_workflow": None,
        "model_root": args.model_root,
        "model_subfolder": args.subfolder,
        "source_image": str(image_path),
        "prompt": args.prompt,
        "target_fbx_used": False,
        "texture_pipeline_run": False,
        "rig_pipeline_run": False,
    }

    try:
        if not image_path.is_file():
            raise RuntimeError(f"Source image missing: {image_path}")
        image_hash = sha256(image_path)
        payload["source_image_sha256"] = image_hash
        expected = args.expected_image_sha256.strip().lower()
        if expected and image_hash.lower() != expected:
            raise RuntimeError(f"Source image hash mismatch: expected {expected}, got {image_hash}")

        model_root = Path(args.model_root)
        weights = model_root / args.subfolder / "model.fp16.safetensors"
        if not weights.is_file():
            raise RuntimeError(f"Mini Turbo weights not found: {weights}")
        payload["weights"] = str(weights)
        payload["weights_bytes"] = weights.stat().st_size

        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; Mini Turbo requires the GPU on this machine")
        payload["torch"] = torch.__version__
        payload["gpu"] = torch.cuda.get_device_name(0)

        if args.conditioning_image:
            from PIL import Image

            supplied = Path(args.conditioning_image).resolve()
            if not supplied.is_file():
                raise RuntimeError(f"Supplied conditioning image missing: {supplied}")
            conditioning = Image.open(supplied).convert("RGBA")
            if conditioning.getchannel("A").getextrema()[0] == 255:
                raise RuntimeError(
                    f"Supplied conditioning image has no transparency: {supplied}"
                )
            payload["conditioning_source"] = str(supplied)
            payload["matte"] = "shaman_matte.py"
        else:
            conditioning = load_conditioning_image(image_path)
            payload["matte"] = "rembg"
        matted = output.parent / "mini_turbo_conditioning.png"
        conditioning.save(matted)
        payload["conditioning_image"] = str(matted)

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(model_root),
            subfolder=args.subfolder,
            use_safetensors=True,
            variant="fp16",
            device="cuda",
        )
        # FlashVDM keeps volume decoding inside a 6 GB budget without changing the generator.
        try:
            pipeline.enable_flashvdm(topk_mode="merge")
            payload["flashvdm"] = True
        except Exception as exc:  # pragma: no cover - depends on installed hy3dgen build
            payload["flashvdm"] = False
            payload["flashvdm_error"] = str(exc)

        ladder = []
        for entry in args.octree_ladder.split(","):
            resolution, _, chunks = entry.strip().partition(":")
            ladder.append((int(resolution), int(chunks or 8000)))

        started_at = time.time()
        attempts = []
        mesh = None
        for octree_resolution, num_chunks in ladder:
            attempt = {"octree_resolution": octree_resolution, "num_chunks": num_chunks}
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                generated = pipeline(
                    image=conditioning,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    octree_resolution=octree_resolution,
                    num_chunks=num_chunks,
                    generator=torch.manual_seed(args.seed),
                    output_type="trimesh",
                )
                mesh = generated[0]
                attempt["status"] = "ok"
                attempt["peak_vram_mb"] = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
                attempts.append(attempt)
                payload["octree_resolution"] = octree_resolution
                payload["num_chunks"] = num_chunks
                break
            except torch.cuda.OutOfMemoryError as exc:
                attempt["status"] = "cuda_oom"
                attempt["error"] = str(exc)[:400]
                attempts.append(attempt)
                mesh = None
                torch.cuda.empty_cache()
                continue
        payload["attempts"] = attempts
        if mesh is None:
            raise RuntimeError(
                "Mini Turbo exhausted its geometry-setting ladder without fitting in VRAM. "
                "No fallback generator is permitted."
            )

        if isinstance(mesh, list):
            mesh = mesh[0]
        mesh.export(str(output))
        verify_real_glb(output, started_at)

        payload.update(
            {
                "success": True,
                "raw_glb": str(output),
                "raw_glb_sha256": sha256(output),
                "raw_glb_bytes": output.stat().st_size,
                "raw_vertices": int(len(mesh.vertices)),
                "raw_triangles": int(len(mesh.faces)),
                "generation_seconds": round(time.time() - started_at, 1),
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
            }
        )
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"MINI_TURBO_GENERATED glb={output} verts={payload['raw_vertices']} "
            f"tris={payload['raw_triangles']} octree={payload['octree_resolution']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        import traceback

        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc()
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"MINI_TURBO_FAILED error={exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
