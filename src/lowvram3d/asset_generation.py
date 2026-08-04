"""Generate a real mesh per placed region with Hunyuan3D Mini Turbo.

Until this stage existed, the structural pipeline's output was engine
primitives: a cube for the barn, cylinders for the trees. Placement already
recovers where each region is and how big it is, so the only thing missing was
the object itself -- and the source image still holds the pixels each region was
measured from, which is exactly a generator's input.

So: crop the source image to a region's own bounding box, hand the crop to Mini
Turbo, and keep the GLB. The rest of the pipeline swaps it in for the primitive.

Three decisions worth stating, because each one is a judgement rather than an
obvious default:

*Not every class gets a mesh.* Terrain, water and paths are surfaces measured as
extents, not objects with a silhouette; a generated blob would be worse than the
plane they already get. Only object-like classes are generated.

*Scatter classes generate once.* A vegetation region's bbox covers a whole tree
line. Placement slices it into per-instance windows, so one instance's crop is a
single tree -- generate that one and reuse the mesh across the instances rather
than spending five GPU-minutes per tree to produce twelve near-identical ones.
The receipt says so; this is a cost decision, not a claim that the trees differ.

*Failures are per-asset and visible.* A region whose generation fails keeps its
primitive and is recorded as failed. The alternative -- failing the run -- would
throw away every asset that did work, and the alternative to *that* is silently
shipping primitives while the receipt says PROVEN, which is the failure mode
this project keeps rediscovering.

Mini Turbo needs its own interpreter and source tree; none of it is discoverable
from the model root, so the runtime is resolved and checked up front and the
receipt names what was missing rather than failing on the first crop.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "workers" / "mini_turbo_generate.py"
DECIMATOR = REPO / "workers" / "decimate_mesh.py"
PREVIEWER = REPO / "workers" / "preview_generated_mesh.py"

# Mini Turbo samples a 384^3 volume whatever the subject, so a single building
# comes back at ~1.7 M triangles. That is grid resolution, not detail, and it
# costs more than it looks: importing one such mesh into Unreal takes over ten
# minutes and outlives the editor bridge's handler timeout, so a slow import is
# indistinguishable from a failed one. Reduce before the mesh leaves this stage.
DEFAULT_TRIANGLE_BUDGET = 150_000

# Observed on this GTX 1660 SUPER, across two independent runs: the first asset
# of a run generates cleanly and every asset after it dies with `CUDA error:
# misaligned address` inside a Linear on the second diffusion step. Each asset
# is already its own process, so this is not a leaked context -- it looks like
# the card itself needing to settle after ten minutes at 100%. This GPU has
# form: fp16 cuDNN convolution on it produces NaN often enough to be disabled
# elsewhere in the project.
#
# A pause between assets is the cheap thing to try. It is not a fix, and the
# retry ladder still carries the load if it does not help.
DEFAULT_SETTLE_SECONDS = 45.0

# Mini Turbo is installed standalone: its own Python, hy3dgen as a source tree
# on PYTHONPATH, and weights under a model root that is not either of those.
# Overridable so a different install does not need a code change.
DEFAULT_RUNTIME = {
    "interpreter": r"C:\AI\HY3D2\python_standalone\python.exe",
    "source_tree": r"C:\AI\HY3D2\Hunyuan3D-2",
    "model_root": r"C:\AI\HY3D2\HuggingFaceHub\hunyuan3d-2mini-direct",
    "subfolder": "hunyuan3d-dit-v2-mini-turbo",
}

# Classes with a silhouette worth reconstructing. Surfaces are excluded above.
GENERATED_LAYERS = {"architecture", "vegetation", "prop", "object"}
# Classes placed as many instances of one region.
SCATTER_LAYERS = {"vegetation", "prop"}
# A crop tighter than this is too few pixels to condition on.
MIN_CROP_PIXELS = 64
# Scatter slices are sized for *placement*, not for conditioning: splitting a
# tree line into twelve gives a 1:6.6 sliver of foliage that no generator can
# read as a subject. Widen the conditioning window about the slice centre to at
# least this width-to-height ratio, clamped to the region, so the crop shows a
# whole tree rather than a vertical stripe of one.
MIN_CROP_ASPECT = 1.0 / 3.0
# A mask covering less of the crop than this leaves too little subject to
# condition on, so fall back rather than generate from near-empty pixels.
MIN_MASK_COVERAGE = 0.04
# Breathing room around a region so its silhouette is not cut off at the edge.
CROP_PADDING = 0.06


def resolve_runtime(overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Locate the Mini Turbo install and say precisely what is missing."""
    runtime = dict(DEFAULT_RUNTIME)
    for key in runtime:
        value = os.environ.get("MINI_TURBO_" + key.upper())
        if value:
            runtime[key] = value
    runtime.update({k: v for k, v in (overrides or {}).items() if v})

    weights = Path(runtime["model_root"]) / runtime["subfolder"] / "model.fp16.safetensors"
    checks = {
        "interpreter": Path(runtime["interpreter"]).is_file(),
        "source_tree": (Path(runtime["source_tree"]) / "hy3dgen").is_dir(),
        "model_root": Path(runtime["model_root"]).is_dir(),
        "weights": weights.is_file(),
    }
    runtime["checks"] = checks
    runtime["weights"] = str(weights)
    runtime["available"] = all(checks.values())
    if not runtime["available"]:
        missing = sorted(name for name, ok in checks.items() if not ok)
        runtime["reason"] = "missing: " + ", ".join(missing)
        if not checks["weights"]:
            # The DiT weights live in a second HuggingFace tree and were
            # hardlinked into the model root; that is the first place to look
            # before re-downloading 3.8 GB.
            runtime["repair"] = (
                "Mini Turbo's DiT weights ship in a separate HF snapshot "
                "(HuggingFaceHub/hub/models--tencent--Hunyuan3D-2mini/...). Link "
                f"model.fp16.safetensors into {weights.parent} rather than "
                "re-downloading it.")
    return runtime


def _widen_to_aspect(bbox: list[float], bounds: list[float]) -> list[float]:
    """Widen a crop window about its centre until it is not a sliver."""
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if height <= 0 or width / height >= MIN_CROP_ASPECT:
        return list(bbox)
    target = height * MIN_CROP_ASPECT
    centre = (bbox[0] + bbox[2]) * 0.5
    low = max(bounds[0], centre - target * 0.5)
    high = min(bounds[2], centre + target * 0.5)
    # Clamping one side must not shrink the window; push the other side out.
    if high - low < target:
        if low <= bounds[0]:
            high = min(bounds[2], low + target)
        else:
            low = max(bounds[0], high - target)
    return [low, bbox[1], high, bbox[3]]


def plan(placement: dict[str, Any]) -> list[dict[str, Any]]:
    """Decide which assets to generate, and which actors each one serves."""
    jobs: dict[str, dict[str, Any]] = {}
    for index, actor in enumerate(placement.get("actors", [])):
        layer = actor.get("layer_type")
        bbox = actor.get("source_bbox_norm_xyxy")
        if layer not in GENERATED_LAYERS or not bbox:
            continue
        region_id = str(actor["region_id"])
        job = jobs.get(region_id)
        if job is None:
            job = {
                "asset_id": region_id,
                "region_id": region_id,
                "layer_type": layer,
                "semantic_label": actor.get("semantic_label"),
                "crop_bbox_norm_xyxy": [float(v) for v in bbox],
                "shared_across_instances": layer in SCATTER_LAYERS,
                "actor_indices": [],
            }
            jobs[region_id] = job
        job["actor_indices"].append(index)

    for job in jobs.values():
        if not job["shared_across_instances"]:
            continue
        # Condition on a middle instance: the ends of a scatter region are the
        # most likely to be clipped by the region's own bounding box.
        members = job["actor_indices"]
        chosen = members[len(members) // 2]
        actor = placement["actors"][chosen]
        region_bbox = [float(v) for v in actor.get(
            "region_bbox_norm_xyxy", job["crop_bbox_norm_xyxy"])]
        job["crop_bbox_norm_xyxy"] = _widen_to_aspect(
            [float(v) for v in actor["source_bbox_norm_xyxy"]], region_bbox)
        job["conditioned_on_actor_index"] = chosen
        job["instance_count"] = len(members)
    return sorted(jobs.values(), key=lambda job: job["asset_id"])


def _crop(image_path: Path, bbox: list[float], destination: Path,
          mask_path: Path | None = None) -> dict[str, Any]:
    """Crop the region, and matte it with its own segmentation mask if there is one.

    Mini Turbo is trained on background-free subjects, so the crop needs an
    alpha channel. Its own fallback is rembg, which on this project's sources is
    actively destructive: a dark barn against dark trees came back with half the
    building erased, and the shaman source lost every ornament. The segmentation
    stage already decided which pixels belong to this region, so use that.
    """
    from PIL import Image, ImageFilter

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    pad_x = (bbox[2] - bbox[0]) * CROP_PADDING
    pad_y = (bbox[3] - bbox[1]) * CROP_PADDING
    box = (
        max(0, int(round((bbox[0] - pad_x) * width))),
        max(0, int(round((bbox[1] - pad_y) * height))),
        min(width, int(round((bbox[2] + pad_x) * width))),
        min(height, int(round((bbox[3] + pad_y) * height))),
    )
    crop = image.crop(box)
    destination.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "crop_png": str(destination),
        "crop_box_px": list(box),
        "crop_size_px": list(crop.size),
        "too_small": bool(min(crop.size) < MIN_CROP_PIXELS),
        "matte": "rembg",
    }
    if mask_path is not None and Path(mask_path).is_file():
        mask = Image.open(mask_path).convert("L").crop(box)
        # A hard label boundary cuts through antialiased edges and leaves a
        # stair-stepped silhouette, which the generator reproduces as faceting.
        mask = mask.filter(ImageFilter.GaussianBlur(1.0))
        covered = sum(mask.histogram()[128:]) / float(max(1, mask.size[0] * mask.size[1]))
        if covered >= MIN_MASK_COVERAGE:
            matted = crop.convert("RGBA")
            matted.putalpha(mask)
            matted.save(destination)
            result.update({"matte": "segmentation_mask",
                           "mask_png": str(mask_path),
                           "mask_coverage": round(covered, 4)})
            return result
        # Too little of the crop survives the mask to condition on -- usually a
        # region whose bbox is mostly other things. Say so rather than handing
        # the generator a nearly empty image.
        result.update({"mask_png": str(mask_path),
                       "mask_coverage": round(covered, 4),
                       "mask_rejected": "coverage below threshold"})

    crop.save(destination)
    return result


def _is_retryable_cuda_fault(error: Any) -> bool:
    """Is this a CUDA fault worth retrying lower down the ladder?

    Observed on this GPU: a generation dies with `CUDA error: misaligned
    address` inside a Linear during the second diffusion step. The worker's own
    ladder cannot help, because it only steps down on OutOfMemoryError and
    because the CUDA context is unusable afterwards -- so the retry has to be a
    fresh process, which is what this layer already spawns per asset.

    Deliberately narrow: a mesh that fails for a *content* reason should fail
    once and be reported, not retried three times at fifteen minutes each.
    """
    text = str(error or "").lower()
    if not text:
        return False
    retryable = ("misaligned address", "illegal memory access",
                 "unspecified launch failure", "cublas_status")
    return any(marker in text for marker in retryable)


def _ladder_retries(octree_ladder: str) -> list[str]:
    """The ladder, then the same ladder with its top rungs removed.

    Each retry is a fresh process at a coarser resolution: whatever alignment
    the kernel tripped over is shape-dependent, so a smaller grid is the cheap
    thing to try before giving up on the asset.
    """
    rungs = [rung.strip() for rung in octree_ladder.split(",") if rung.strip()]
    return [",".join(rungs[index:]) for index in range(len(rungs))]


def _decimate(glb: Path, budget: int) -> tuple[Path, dict[str, Any]]:
    """Reduce a generation to the triangle budget, keeping the original.

    A decimation failure is not fatal: the full-resolution mesh is still a
    correct mesh, just an expensive one, so fall back to it and say so.
    """
    reduced = glb.with_name(glb.stem + "_lod.glb")
    receipt_path = glb.with_name(glb.stem + "_decimation.json")
    completed = subprocess.run(
        [sys.executable, str(DECIMATOR), "--input", str(glb),
         "--output", str(reduced), "--target-triangles", str(budget),
         "--receipt", str(receipt_path)],
        capture_output=True, text=True)
    receipt: dict[str, Any] = {}
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if completed.returncode == 0 and reduced.is_file() and reduced.stat().st_size > 4096:
        return reduced, receipt
    return glb, {**receipt, "classification": receipt.get("classification", "FAILED"),
                 "used": "full_resolution_mesh",
                 "stderr_tail": (completed.stderr or "")[-400:]}


def _preview(glb: Path, destination: Path) -> dict[str, Any]:
    """Render the asset on its own so it can be judged apart from the scene.

    Never fatal: a missing preview costs a look, not an asset.
    """
    receipt_path = destination.with_suffix(".json")
    completed = subprocess.run(
        [sys.executable, str(PREVIEWER), "--glb", str(glb),
         "--out", str(destination), "--receipt", str(receipt_path)],
        capture_output=True, text=True)
    if completed.returncode != 0 or not receipt_path.is_file():
        return {"preview_error": (completed.stderr or "")[-300:]}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {"preview_png": str(destination),
            "mesh_bodies": receipt.get("body_count"),
            "mesh_watertight": receipt.get("watertight"),
            "mesh_extent": receipt.get("extent"),
            "view_coverage": receipt.get("view_coverage")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(placement: dict[str, Any], image: Path, output_dir: Path,
             runtime_overrides: dict[str, str] | None = None,
             max_assets: int | None = None,
             steps: int = 5,
             octree_ladder: str = "384:3000,320:2000,256:1500",
             mask_dir: Path | None = None,
             triangle_budget: int = DEFAULT_TRIANGLE_BUDGET,
             settle_seconds: float = DEFAULT_SETTLE_SECONDS,
             timeout: float = 1800.0) -> dict[str, Any]:
    """Crop, generate and verify one mesh per eligible region."""
    # Absolute from here on. These paths are handed to the Unreal editor, which
    # resolves a relative one against its *own* project directory -- so a
    # perfectly good mesh reports as "not found" from a different working
    # directory than the one that wrote it.
    output_dir = Path(output_dir).resolve()
    image = Path(image).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def mask_for(region_id: str) -> Path | None:
        candidate = Path(mask_dir) / f"{region_id}.png"
        return candidate if candidate.is_file() else None
    runtime = resolve_runtime(runtime_overrides)
    jobs = plan(placement)
    if max_assets is not None and max_assets >= 0:
        jobs = jobs[:max_assets]

    receipt: dict[str, Any] = {
        "schema_version": "generated_asset_manifest_v1",
        "generator": "hunyuan3d_mini_turbo",
        "runtime": {k: v for k, v in runtime.items() if k != "checks"},
        "runtime_checks": runtime["checks"],
        "source_image": str(Path(image).resolve()),
        "planned_asset_count": len(jobs),
        "triangle_budget": triangle_budget,
        "steps": steps,
        "octree_ladder": octree_ladder,
        "assets": [],
    }

    if not runtime["available"]:
        receipt["classification"] = "UNAVAILABLE"
        receipt["reason"] = runtime.get("reason")
        receipt["assets"] = [
            {**job, "status": "skipped", "reason": "mini turbo runtime unavailable"}
            for job in jobs]
        return receipt

    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        runtime["source_tree"] + (os.pathsep + existing if existing else ""))

    for position, job in enumerate(jobs):
        if position and settle_seconds > 0:
            time.sleep(settle_seconds)
        asset_dir = output_dir / job["asset_id"]
        crop = _crop(Path(image), job["crop_bbox_norm_xyxy"],
                     asset_dir / "crop.png",
                     mask_for(job["region_id"]) if mask_dir else None)
        entry = {**job, **crop}
        if crop["too_small"]:
            entry["status"] = "skipped"
            entry["reason"] = (
                f"crop is {crop['crop_size_px']} px, below the {MIN_CROP_PIXELS} px "
                "minimum to condition on")
            receipt["assets"].append(entry)
            continue

        glb = asset_dir / f"{job['asset_id']}.glb"
        result_json = asset_dir / "mini_turbo_result.json"
        worker_result: dict[str, Any] = {}
        attempts: list[dict[str, Any]] = []
        timed_out = False
        for ladder in _ladder_retries(octree_ladder):
            command = [
                runtime["interpreter"], str(WORKER),
                "--image", str(asset_dir / "crop.png"),
                "--output", str(glb),
                # The worker mattes with rembg unless handed a pre-matted RGBA
                # image, and it rejects one with no transparency, so only pass
                # the crop through when the segmentation mask actually applied.
                *(("--conditioning-image", str(asset_dir / "crop.png"))
                  if crop["matte"] == "segmentation_mask" else ()),
                "--result-json", str(result_json),
                "--model-root", runtime["model_root"],
                "--subfolder", runtime["subfolder"],
                "--steps", str(steps),
                "--octree-ladder", ladder,
            ]
            try:
                completed = subprocess.run(
                    command, env=environment, capture_output=True, text=True,
                    timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                attempts.append({"octree_ladder": ladder, "outcome": "timeout"})
                break

            worker_result = {}
            if result_json.is_file():
                worker_result = json.loads(result_json.read_text(encoding="utf-8"))
            attempts.append({
                "octree_ladder": ladder,
                "exit_code": completed.returncode,
                "outcome": "ok" if worker_result.get("success") else "failed",
                "error": str(worker_result.get("error") or "")[:200] or None,
            })
            entry["exit_code"] = completed.returncode
            entry["stderr_tail"] = (completed.stderr or "")[-600:]
            if worker_result.get("success"):
                break
            if not _is_retryable_cuda_fault(worker_result.get("error")):
                break
            if settle_seconds > 0:
                time.sleep(settle_seconds)

        entry["generation_attempts"] = attempts
        if timed_out:
            entry["status"] = "failed"
            entry["reason"] = f"generation exceeded {timeout:.0f}s"
            receipt["assets"].append(entry)
            continue
        # Trust the artefact, not the exit code: this project has repeatedly
        # found green receipts over missing or empty outputs.
        if glb.is_file() and glb.stat().st_size > 4096 and worker_result.get("success"):
            usable, decimation = _decimate(glb, triangle_budget)
            entry.update({
                "status": "generated",
                "glb": str(usable),
                "raw_glb": str(glb),
                "decimation": decimation,
                "glb_bytes": usable.stat().st_size,
                "glb_sha256": _sha256(usable),
                "vertices": worker_result.get("raw_vertices"),
                "triangles": decimation.get("output_triangles",
                                            worker_result.get("raw_triangles")),
                "raw_triangles": worker_result.get("raw_triangles"),
                "octree_resolution": worker_result.get("octree_resolution"),
                "peak_vram_mb": max(
                    (a.get("peak_vram_mb") or 0)
                    for a in worker_result.get("attempts", [{}])) or None,
                "generation_seconds": worker_result.get("generation_seconds"),
                **_preview(usable, asset_dir / "preview.png"),
            })
        else:
            entry["status"] = "failed"
            entry["reason"] = worker_result.get("error") or "no usable GLB produced"
            entry["failure_code"] = worker_result.get("failure_code")
        receipt["assets"].append(entry)

    generated = [a for a in receipt["assets"] if a["status"] == "generated"]
    receipt["generated_count"] = len(generated)
    receipt["failed_count"] = sum(
        1 for a in receipt["assets"] if a["status"] == "failed")
    receipt["skipped_count"] = sum(
        1 for a in receipt["assets"] if a["status"] == "skipped")
    receipt["actors_served"] = sum(len(a["actor_indices"]) for a in generated)
    receipt["classification"] = (
        "PROVEN" if generated and not receipt["failed_count"]
        else "PARTIAL" if generated
        else "EMPTY")
    return receipt


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--max-assets", type=int, default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--octree-ladder", default="384:3000,320:2000,256:1500")
    parser.add_argument("--mask-dir", default=None,
                        help="per-region masks from the segmentation stage")
    parser.add_argument("--triangle-budget", type=int,
                        default=DEFAULT_TRIANGLE_BUDGET)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)

    placement = json.loads(Path(args.placement).read_text(encoding="utf-8"))
    if args.plan_only:
        result = {"schema_version": "generated_asset_plan_v1",
                  "runtime": resolve_runtime(),
                  "jobs": plan(placement)}
    else:
        result = generate(placement, Path(args.image), Path(args.output_dir),
                          max_assets=args.max_assets, steps=args.steps,
                          octree_ladder=args.octree_ladder,
                          mask_dir=Path(args.mask_dir) if args.mask_dir else None,
                          triangle_budget=args.triangle_budget)

    receipt = Path(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "assets"},
                     indent=2, sort_keys=True))
    for asset in result.get("assets", []):
        print(f"  {asset['asset_id']:<28} {asset['status']:<10} "
              f"{asset.get('triangles') or asset.get('reason') or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
