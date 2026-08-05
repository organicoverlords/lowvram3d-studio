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
CONSOLE_HEALTHY = "HEALTHY"
CONSOLE_DEGRADED = "DEGRADED"
_console_state = CONSOLE_HEALTHY
_console_exit_message_sent = False


def _reset_console_state():
    global _console_state, _console_exit_message_sent
    _console_state = CONSOLE_HEALTHY
    _console_exit_message_sent = False


def _tensor_summary(name, value, diagnostic=False):
    """Return cheap production metadata, with scans only in explicit diagnostic mode."""
    import torch

    if not isinstance(value, torch.Tensor):
        return None
    probe = value.detach()
    summary = {
        "name": name,
        "shape": list(probe.shape),
        "dtype": str(probe.dtype),
        "device": str(probe.device),
        "numel": int(probe.numel()),
    }
    if diagnostic and value.is_cuda:
        torch.cuda.synchronize(value.device)
    if diagnostic and probe.numel():
        finite = bool(torch.isfinite(probe).all().item())
        minimum = float(probe.float().amin().item())
        maximum = float(probe.float().amax().item())
        summary.update({"finite": finite, "min": minimum, "max": maximum})
    elif diagnostic:
        finite, minimum, maximum = True, None, None
        summary.update({"finite": finite, "min": minimum, "max": maximum})
    return summary


def _cuda_stats(torch, synchronize=False, boundary_name=None):
    if synchronize:
        if not boundary_name:
            raise ValueError("diagnostic CUDA synchronization requires a named boundary")
        torch.cuda.synchronize()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _safe_console_write(stream, message, flush=True):
    """Best-effort bounded console sink; telemetry and artifacts never depend on it."""
    global _console_state
    if _console_state == CONSOLE_DEGRADED:
        return None
    message = " ".join(str(message).replace("\r", " ").replace("\n", " ").split())[:512] + "\n"
    try:
        stream.write(message)
        if flush:
            stream.flush()
    except (OSError, UnicodeError, ValueError, BrokenPipeError) as exc:
        _console_state = CONSOLE_DEGRADED
        return {
            "type": type(exc).__name__,
            "errno": getattr(exc, "errno", None),
            "message": str(exc),
        }
    return None


def _emit_console_exit_status():
    """Allow one compact ASCII status after a degraded run, at process exit only."""
    global _console_exit_message_sent
    if _console_state != CONSOLE_DEGRADED or _console_exit_message_sent:
        return
    _console_exit_message_sent = True
    # A degraded stdout is skipped by _safe_console_write; stderr remains best effort.
    previous = _console_state
    globals()["_console_state"] = CONSOLE_HEALTHY
    _safe_console_write(sys.stderr, "MINI_TURBO_CONSOLE_DEGRADED")
    globals()["_console_state"] = previous


def _write_json_artifact(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _mesh_summary(mesh):
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
    }


def _mesh_boundary_summary(mesh):
    """Describe mesh arrays at a serialization boundary without changing them."""
    import numpy as np

    vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype="<f4"))
    faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype="<u4").reshape(-1, 3))
    duplicate = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "duplicate_index_faces": int(duplicate.sum()),
        "duplicate_index_percent": round(float(duplicate.mean() * 100.0), 6) if len(faces) else 0.0,
        "vertex_array_sha256": hashlib.sha256(vertices.tobytes()).hexdigest(),
        "index_array_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
    }


def _sanitize_decoded_mesh(mesh):
    """Drop exact duplicate-index faces immediately after decoder output.

    Mini Turbo's decoder can return a trimesh object containing a large duplicate-index face
    population. Those faces are zero-area and Blender silently drops them later, which makes the
    corruption look like a GLB/import problem. Sanitize the returned mesh before any downstream
    component, UV, or texture stage and keep all non-duplicate faces unchanged.
    """
    import numpy as np

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise RuntimeError("decoded mesh has invalid or non-finite vertices")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("decoded mesh has invalid face indices")
    duplicate = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    removed = int(duplicate.sum())
    if removed:
        mesh.update_faces(~duplicate)
        mesh.remove_unreferenced_vertices()
    return mesh, removed


def _classify_generation_failure(exc: Exception) -> str | None:
    text = str(exc).lower()
    if "empty active point" in text:
        return "EMPTY_ACTIVE_POINT_SET"
    if "expected reduction dim 0 to have non-zero size" in text:
        return "EXPECTED_REDUCTION_DIM_NON_ZERO"
    return None


def _trace(
    trace_path,
    operation,
    torch=None,
    tensors=None,
    artifact_callback=None,
    diagnostic=False,
    boundary_name=None,
    **fields,
):
    record = {
        "time": time.time(),
        "operation": operation,
        "telemetry_mode": "diagnostic" if diagnostic else "production",
        **fields,
    }
    if torch is not None and torch.cuda.is_available():
        record["cuda"] = _cuda_stats(torch, synchronize=diagnostic, boundary_name=boundary_name)
    if tensors:
        record["tensors"] = [summary for summary in tensors if summary]
    # Force JSON-safe representation before any sink is attempted.
    record = json.loads(json.dumps(record, default=str))
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    if artifact_callback is not None:
        artifact_callback(record)
    console_error = _safe_console_write(sys.stdout, "MINI_TURBO_TRACE " + json.dumps(record))
    if console_error:
        # The durable event already exists; append a sink diagnostic as its own durable event.
        sink_record = {"time": time.time(), "operation": "console_sink_failure", "error": console_error}
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sink_record) + "\n")
    return record


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


#: View names MVImageProcessorV2 accepts, from its own view2idx table.
MULTIVIEW_NAMES = {"front", "left", "back", "right"}


def _conditioning_dimensions(conditioning):
    """Sizes of the conditioning, whether it is one image or a dict of views.

    Trivial, and it exists as a named function because getting it wrong cost
    two runs and an incorrect diagnosis. `conditioning` is rebound from a PIL
    image to a {view: image} dict on the multiview path, and the telemetry line
    a few statements later still read `.size` off it. The resulting "'dict'
    object has no attribute 'size'" was blamed on hy3dgen and on the mv
    checkpoint in turn, when it never reached either -- the pipeline had not
    been constructed yet. Telemetry that crashes the run it is measuring is
    worse than no telemetry, so this branch is tested rather than inlined.
    """
    if isinstance(conditioning, dict):
        return {name: list(view.size) for name, view in conditioning.items()}
    return list(conditioning.size)


def load_conditioning_image(image_path: Path):
    from PIL import Image

    image = Image.open(image_path)
    if image.mode == "RGBA" and image.getchannel("A").getextrema()[0] < 255:
        return image
    # Mini Turbo is trained on background-free subjects; an opaque input must be matted first.
    from hy3dgen.rembg import BackgroundRemover

    return BackgroundRemover()(image.convert("RGB"))


def _surface_extractor(name: str):
    """Build a named surface extractor, failing loudly rather than silently.

    `set_surface_extractor` on the pipeline is deprecated and logs a warning,
    so the extractor is assigned onto the VAE directly, which is what that
    warning tells callers to do. Import is deferred because the extractor
    module pulls in the whole hy3dgen autoencoder package.
    """
    from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors

    if name not in SurfaceExtractors:
        raise ValueError(
            f"unknown extractor {name!r}, have {sorted(SurfaceExtractors)}")
    extractor = SurfaceExtractors[name]()
    if name == "dmc":
        # diso is an optional dependency and DMCSurfaceExtractor only imports
        # it on first run, deep inside volume decoding. Import it here so a
        # missing package costs a second rather than a full diffusion pass.
        import diso  # noqa: F401
    return extractor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    # Pre-matted RGBA input. rembg destroys this source (drops every hanging ornament and its cord,
    # blackens the staff ring), so the caller mattes it with workers/shaman_matte.py instead.
    parser.add_argument("--conditioning-image", default="")
    parser.add_argument(
        "--view", action="append", default=[],
        help="Extra conditioning view as NAME=PATH, repeatable. NAME is one of "
             "front/left/back/right. Requires an mv checkpoint; the ordinary "
             "checkpoints take a single image and will fail on a dict.")
    parser.add_argument("--expected-image-sha256", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-mini-turbo")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--diagnostic-telemetry", action="store_true")
    # Descending ladder of geometry settings. Only VRAM pressure moves us down it; the generator
    # never changes. Each entry is (octree_resolution, num_chunks).
    parser.add_argument("--octree-ladder", default="384:3000,320:2000,256:1500")
    parser.add_argument(
        "--mc-algo", choices=("mc", "dmc"), default="",
        help="Surface extractor. Empty keeps the pipeline default ('mc', "
             "Lewiner marching cubes). 'dmc' is diso's differentiable dual "
             "marching cubes, which places one vertex per cell instead of one "
             "per edge crossing and so can represent a crease.")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output = Path(args.output).resolve()
    result_path = Path(args.result_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = result_path.with_name("generation_trace.jsonl")
    trace_path.unlink(missing_ok=True)

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
        "diagnostics_trace": str(trace_path),
        "seed": args.seed,
        "steps": args.steps,
        "octree_ladder": args.octree_ladder,
        "diagnostic_telemetry": args.diagnostic_telemetry,
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
        _trace(trace_path, "cuda_ready", torch, diagnostic=args.diagnostic_telemetry, boundary_name="cuda_ready", device=str(torch.cuda.current_device()))

        # Hoisted out of the branch below. It used to be imported only when a
        # conditioning image was supplied, which left the name unbound on the
        # multiview path -- `--view` without `--conditioning-image` failed with
        # "cannot access local variable 'Image'", a message that points at the
        # multiview code rather than at the import that is actually missing.
        from PIL import Image

        if args.conditioning_image:
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
        payload["image_dimensions"] = list(conditioning.size)

        # Multiview conditioning. The mv checkpoints declare MVImageProcessorV2
        # in their own config.yaml, so loading one selects the multiview path
        # automatically and the pipeline then expects a {view: image} dict
        # rather than a single image. Views are named for the *asset*, not the
        # rig -- 'front' is the face the conditioning camera saw.
        #
        # A single-view checkpoint given a dict, or an mv checkpoint given one
        # image, both fail deep inside the preprocessor, so the mismatch is
        # checked here where the message can say which it was.
        # Detect the checkpoint's own expectation rather than trusting the
        # caller. An mv checkpoint handed a single image dies as
        # "'Image' object has no attribute 'items'" several frames inside the
        # preprocessor, which names neither the checkpoint nor the argument.
        config_path = model_root / args.subfolder / "config.yaml"
        multiview_checkpoint = (
            config_path.is_file()
            and "MVImageProcessorV2" in config_path.read_text(encoding="utf-8"))
        payload["multiview_checkpoint"] = multiview_checkpoint
        if args.view and not multiview_checkpoint:
            raise RuntimeError(
                f"MULTIVIEW_VIEWS_ON_SINGLE_VIEW_CHECKPOINT:{args.subfolder} "
                "declares ImageProcessorV2 and takes one image")

        if args.view or multiview_checkpoint:
            views = {}
            for spec in args.view:
                name, _, path = spec.partition("=")
                name = name.strip().lower()
                if name not in MULTIVIEW_NAMES:
                    raise RuntimeError(
                        f"MULTIVIEW_NAME_INVALID:{name}; expected one of "
                        f"{sorted(MULTIVIEW_NAMES)}")
                view_path = Path(path).resolve()
                if not view_path.is_file():
                    raise RuntimeError(f"MULTIVIEW_IMAGE_MISSING:{view_path}")
                views[name] = Image.open(view_path).convert("RGBA")
            if "front" not in views:
                views["front"] = conditioning
            conditioning = views
            payload["multiview"] = True
            payload["multiview_views"] = sorted(views)
            _trace(trace_path, "multiview_conditioning", torch,
                   diagnostic=args.diagnostic_telemetry,
                   boundary_name="multiview_conditioning",
                   views=sorted(views))
        else:
            payload["multiview"] = False
        _trace(trace_path, "conditioning_loaded", torch, diagnostic=args.diagnostic_telemetry, boundary_name="conditioning_loaded", image_dimensions=_conditioning_dimensions(conditioning))

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(model_root),
            subfolder=args.subfolder,
            use_safetensors=True,
            variant="fp16",
            device="cuda",
        )
        _trace(trace_path, "pipeline_loaded", torch, diagnostic=args.diagnostic_telemetry, boundary_name="pipeline_loaded", model_root=str(model_root), subfolder=args.subfolder)
        # Surface extraction. The pipeline default is 'mc' -- skimage's Lewiner
        # marching cubes -- and every mesh this project has produced used it,
        # including all four rows of the capacity comparison. Marching cubes
        # places one vertex per grid edge crossing, so it cannot represent a
        # crease: a sharp edge becomes a staircase, and the fix for a staircase
        # is more grid, which is the octree lever already measured and
        # eliminated.
        #
        # 'dmc' is dual marching cubes via diso's DiffDMC, which is already
        # installed. It places one vertex per cell and is free to put it where
        # the field says the feature is, so creases survive at a given
        # resolution. That is the one untested thing in the extraction path.
        #
        # It is not a detail generator. If the decoded field is genuinely
        # smooth where a window should be, no extractor recovers the window.
        # The testable prediction is narrower: less tessellation noise and
        # crisper existing edges at the same octree resolution, or no change --
        # and 'no change' is itself the useful answer, because it moves the
        # blame upstream of extraction to the field the VAE decodes.
        # FlashVDM keeps volume decoding inside a 6 GB budget without changing the generator.
        try:
            pipeline.enable_flashvdm(topk_mode="merge")
            payload["flashvdm"] = True
            _trace(trace_path, "flashvdm_enabled", torch, diagnostic=args.diagnostic_telemetry, boundary_name="flashvdm_enabled", topk_mode="merge")
        except Exception as exc:  # pragma: no cover - depends on installed hy3dgen build
            payload["flashvdm"] = False
            payload["flashvdm_error"] = str(exc)

        # Surface extraction. MUST come after enable_flashvdm: that call swaps
        # in the turbo VAE (`replace_vae=True`) and then sets the extractor from
        # its own `mc_algo='mc'` default, so an extractor assigned before it is
        # discarded along with the VAE object it was attached to. Setting it
        # first is silent -- the run completes and reports success with the
        # wrong extractor, which is exactly what happened on the first attempt
        # at this experiment.
        #
        # The pipeline default is 'mc', skimage's Lewiner marching cubes, and
        # every mesh this project has produced used it, including all four rows
        # of the capacity comparison. Marching cubes places one vertex per grid
        # edge crossing, so it cannot represent a crease: a sharp edge becomes a
        # staircase, and the fix for a staircase is more grid -- the octree
        # lever already measured and eliminated.
        #
        # 'dmc' is diso's differentiable dual marching cubes. It places one
        # vertex per cell and is free to put it where the field says the feature
        # is, so creases survive at a given resolution.
        #
        # It is not a detail generator. If the decoded field is smooth where a
        # window should be, no extractor recovers the window. The testable
        # prediction is narrow: less tessellation noise and crisper existing
        # edges at the same octree resolution, or no change at all -- and no
        # change is itself worth knowing, because it moves the blame upstream of
        # extraction to the field the VAE decodes.
        if args.mc_algo:
            try:
                pipeline.vae.surface_extractor = _surface_extractor(args.mc_algo)
            except Exception as exc:
                raise SystemExit(f"MC_ALGO_UNAVAILABLE: {args.mc_algo}: {exc}")
        # Record the class actually in place rather than the flag that was
        # requested. The two disagreed once already, and a receipt that repeats
        # the request back cannot detect that.
        payload["mc_algo"] = args.mc_algo or "mc"
        payload["surface_extractor"] = type(pipeline.vae.surface_extractor).__name__

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
                _trace(
                    trace_path,
                    "generation_attempt_begin",
                    torch,
                    diagnostic=args.diagnostic_telemetry,
                    boundary_name="generation_attempt_begin",
                    octree_resolution=octree_resolution,
                    num_chunks=num_chunks,
                    seed=args.seed,
                    steps=args.steps,
                )

                def generation_callback(step_idx, timestep, outputs):
                    tensors = []
                    if hasattr(outputs, "items"):
                        tensors.extend(_tensor_summary(name, value, diagnostic=args.diagnostic_telemetry) for name, value in outputs.items())
                    else:
                        tensors.append(_tensor_summary("prev_sample", getattr(outputs, "prev_sample", None), diagnostic=args.diagnostic_telemetry))
                    _trace(
                        trace_path,
                        "diffusion_step_complete",
                        torch,
                        diagnostic=args.diagnostic_telemetry,
                        boundary_name=f"diffusion_step_{int(step_idx) + 1}",
                        tensors=tensors,
                        step_number=int(step_idx) + 1,
                        timestep=int(timestep.item()) if hasattr(timestep, "item") else int(timestep),
                        chunk_range=None,
                        last_successful_operation="scheduler.step",
                    )

                generated = pipeline(
                    image=conditioning,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    octree_resolution=octree_resolution,
                    num_chunks=num_chunks,
                    generator=torch.manual_seed(args.seed),
                    output_type="trimesh",
                    callback=generation_callback,
                    callback_steps=1,
                )
                _trace(trace_path, "generation_complete", torch, diagnostic=args.diagnostic_telemetry, boundary_name="generation_complete", last_successful_operation="pipeline.__call__")
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
        decoded_boundary = _mesh_boundary_summary(mesh)
        _trace(
            trace_path,
            "mesh_decoded",
            torch,
            diagnostic=args.diagnostic_telemetry,
            boundary_name="mesh_decode",
            last_successful_operation="mesh_decode",
            **decoded_boundary,
        )
        mesh, removed_degenerate_faces = _sanitize_decoded_mesh(mesh)
        sanitized_boundary = _mesh_boundary_summary(mesh)
        _trace(
            trace_path,
            "mesh_sanitized",
            torch,
            diagnostic=args.diagnostic_telemetry,
            boundary_name="mesh_sanitize",
            last_successful_operation="mesh_sanitize",
            removed_degenerate_faces=removed_degenerate_faces,
            **sanitized_boundary,
        )
        _trace(
            trace_path,
            "mesh_ready",
            torch,
            diagnostic=args.diagnostic_telemetry,
            boundary_name="mesh_ready",
            last_successful_operation="mesh_sanitize",
            **sanitized_boundary,
        )
        mesh.export(str(output))
        _trace(
            trace_path,
            "mesh_exported",
            torch,
            diagnostic=args.diagnostic_telemetry,
            boundary_name="mesh_exported",
            last_successful_operation="mesh.export",
            output=str(output),
            **sanitized_boundary,
        )
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
        _write_json_artifact(result_path, payload)
        _safe_console_write(
            sys.stdout,
            f"MINI_TURBO_GENERATED glb={output} verts={payload['raw_vertices']} "
            f"tris={payload['raw_triangles']} octree={payload['octree_resolution']}\n",
        )
        _emit_console_exit_status()
        return 0
    except Exception as exc:
        import traceback

        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc()
        failure_code = _classify_generation_failure(exc)
        if failure_code:
            payload["failure_code"] = failure_code
        payload["last_successful_operation"] = "see generation_trace.jsonl"
        _write_json_artifact(result_path, payload)
        _safe_console_write(sys.stderr, f"MINI_TURBO_FAILED error={exc}")
        _emit_console_exit_status()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
