from __future__ import annotations

import base64
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from lowvram3d.config import PipelineConfig
from lowvram3d.contracts import JobReceipt, now_ms
from lowvram3d.pipeline import PipelineEngine
from service.context_store import ContextStore
from service.models import FullRequest, GenerateRequest, MeshRequest, PostProcessRequest

CONFIG_PATH = Path(os.environ.get("LOWVRAM3D_CONFIG", PACKAGE_ROOT / "config" / "local.json"))
config = PipelineConfig.load(CONFIG_PATH)
engine = PipelineEngine(config)
contexts = ContextStore(config.install_root / "contexts")
app = FastAPI(title="LowVRAM 3D Studio Worker", version="0.6.0")


def decode_file(encoded: str, filename: str, folder: Path, *, resume_candidate: bool = False) -> Path:
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    safe_name = Path(filename).name or "input.bin"
    if resume_candidate:
        safe_name = f"resume_candidate_{uuid.uuid4().hex}_{safe_name}"
    target = folder / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_bytes(base64.b64decode(encoded, validate=True))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 input: {exc}") from exc
    return target


def receipt_path(job_id: str) -> Path:
    return config.jobs_root / job_id / "proof" / "job_receipt.json"


def load_receipt(job_id: str) -> JobReceipt | None:
    path = receipt_path(job_id)
    try:
        return JobReceipt.load(path) if path.is_file() else None
    except Exception:
        return None


def mark_job_failed(job_id: str, exc: Exception) -> None:
    receipt = load_receipt(job_id) if job_id else None
    if receipt is None or receipt.status == "passed":
        return
    receipt.status = "failed"
    receipt.error = str(exc)
    receipt.finished_at = now_ms()
    receipt.write(receipt_path(job_id))


def binary_mesh(path: Path, receipt: JobReceipt, job_dir: Path) -> FileResponse:
    if not path.is_file() or path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="Pipeline returned no mesh")
    proof = job_dir / "proof" / "job_receipt.json"
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=path.name,
        headers={
            "X-LowVRAM3D-Receipt": str(proof),
            "X-LowVRAM3D-Job": receipt.job_id,
            "X-LowVRAM3D-Resume-Job": receipt.job_id,
        },
    )


def raise_job_error(exc: Exception, job_id: str = "") -> None:
    detail: dict[str, Any] = {"message": str(exc)}
    headers: dict[str, str] = {}
    if job_id:
        detail.update({
            "job_id": job_id,
            "resume_job_id": job_id,
            "receipt": str(receipt_path(job_id)),
            "resume_endpoint": f"/v1/jobs/{job_id}/resume",
        })
        headers.update({"X-LowVRAM3D-Job": job_id, "X-LowVRAM3D-Resume-Job": job_id})
    raise HTTPException(status_code=500, detail=detail, headers=headers) from exc


def failed_context_job(project_id: str, card_id: str, operation: str, enabled: bool) -> str:
    if not enabled or not project_id or not card_id:
        return ""
    context = contexts.get_exact(project_id, card_id)
    job_id = str(context.get("job_id", ""))
    receipt = load_receipt(job_id) if job_id else None
    if receipt and receipt.status == "failed" and receipt.operation == operation:
        return job_id
    return ""


def locate_saved_source(receipt: JobReceipt, job_dir: Path, kind: str) -> Path:
    value = receipt.input_files.get(kind, "")
    if value and Path(value).is_file():
        return Path(value)
    if kind == "mesh":
        candidates = sorted((job_dir / "source").glob("original.*"))
    else:
        candidates = sorted(
            path for path in (job_dir / "input").glob("*")
            if path.is_file() and not path.name.startswith("resume_candidate_")
        )
    if not candidates:
        raise RuntimeError(f"Saved {kind} input is missing for resumable job {receipt.job_id}")
    return candidates[0]


def run_full_request(request: FullRequest, resume_job_id: str = "") -> tuple[Path, JobReceipt, Path]:
    if resume_job_id:
        job_dir = config.jobs_root / resume_job_id
        image = decode_file(request.imageBase64, request.imageFilename, job_dir / "input", resume_candidate=True)
        return engine.full(
            image,
            request.prompt,
            asset_type=request.assetType,
            quality=request.qualityPreset,
            separate_movable_parts=request.separateMovableParts,
            texture_resolution=request.textureResolution,
            lod_enabled=request.lodEnabled,
            remove_hidden_geometry=request.removeHiddenGeometry,
            experimental_semantic_split=request.experimentalSemanticSplit,
            background_removal=request.backgroundRemoval,
            animation_preset=request.animationPreset,
            resume_job_id=resume_job_id,
        )
    receipt, job_dir = engine.new_job("full", {})
    contexts.put(request.projectId, request.cardId, {
        "source_image": "", "prompt": request.prompt, "job_id": receipt.job_id, "operation": "full",
    })
    image = decode_file(request.imageBase64, request.imageFilename, job_dir / "input")
    receipt.input_files["image"] = str(image)
    engine._write_receipt(receipt, job_dir)
    contexts.put(request.projectId, request.cardId, {
        "source_image": str(image), "prompt": request.prompt, "job_id": receipt.job_id, "operation": "full",
    })
    return engine.full(
        image,
        request.prompt,
        asset_type=request.assetType,
        quality=request.qualityPreset,
        separate_movable_parts=request.separateMovableParts,
        texture_resolution=request.textureResolution,
        lod_enabled=request.lodEnabled,
        remove_hidden_geometry=request.removeHiddenGeometry,
        experimental_semantic_split=request.experimentalSemanticSplit,
        background_removal=request.backgroundRemoval,
        animation_preset=request.animationPreset,
        receipt=receipt,
        job_dir=job_dir,
    )


def run_postprocess_request(request: PostProcessRequest, resume_job_id: str = "") -> tuple[Path, JobReceipt, Path]:
    inherited = contexts.get(request.projectId, request.cardId)
    source_image = (
        Path(inherited["source_image"])
        if inherited.get("source_image") and Path(inherited["source_image"]).is_file()
        else None
    )
    if resume_job_id:
        job_dir = config.jobs_root / resume_job_id
        mesh = decode_file(request.meshBase64, request.meshFilename, job_dir / "input", resume_candidate=True)
    else:
        receipt, job_dir = engine.new_job("postprocess", {})
        resume_job_id = receipt.job_id
        contexts.put(request.projectId, request.cardId, {
            "source_mesh": "", "source_image": str(source_image) if source_image else "",
            "prompt": request.prompt, "job_id": receipt.job_id, "operation": "postprocess",
        })
        mesh = decode_file(request.meshBase64, request.meshFilename, job_dir / "input")
        receipt.input_files["mesh"] = str(mesh)
        engine._write_receipt(receipt, job_dir)
        contexts.put(request.projectId, request.cardId, {
            "source_mesh": str(mesh), "source_image": str(source_image) if source_image else "",
            "prompt": request.prompt, "job_id": receipt.job_id, "operation": "postprocess",
        })
    return engine.postprocess(
        mesh,
        asset_type=request.assetType,
        quality=request.qualityPreset,
        separate_movable_parts=request.separateMovableParts,
        texture_resolution=request.textureResolution,
        lod_enabled=request.lodEnabled,
        remove_hidden_geometry=request.removeHiddenGeometry,
        experimental_semantic_split=request.experimentalSemanticSplit,
        prompt=request.prompt,
        source_image=source_image,
        animation_preset=request.animationPreset,
        resume_job_id=resume_job_id,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "config": str(CONFIG_PATH),
        "lanes": config.lane_order,
        "vram_ceiling_mb": config.vram_ceiling_mb,
        "studio_url": config.studio_url,
        "resume": {"installer": True, "asset_jobs": True, "automatic_failed_card_resume": True},
    }


@app.post("/v1/generate")
def generate(request: GenerateRequest):
    job_id = ""
    try:
        receipt, job_dir = engine.new_job("generate", {})
        job_id = receipt.job_id
        image = decode_file(request.imageBase64, request.imageFilename, job_dir / "input")
        receipt.input_files["image"] = str(image)
        engine._write_receipt(receipt, job_dir)
        contexts.put(request.projectId, request.cardId, {
            "source_image": str(image), "prompt": request.prompt, "job_id": job_id, "operation": "generate",
        })
        mesh, receipt, job_dir = engine.generate(image, request.prompt, receipt, job_dir)
        return binary_mesh(mesh, receipt, job_dir)
    except HTTPException as exc:
        mark_job_failed(job_id, exc)
        raise
    except Exception as exc:
        mark_job_failed(job_id, exc)
        raise_job_error(exc, job_id)


@app.post("/v1/texture")
def texture(request: MeshRequest):
    job_id = ""
    try:
        receipt, job_dir = engine.new_job("texture", {})
        job_id = receipt.job_id
        mesh = decode_file(request.meshBase64, request.meshFilename, job_dir / "input")
        context = contexts.get(request.projectId, request.cardId)
        source = Path(context["source_image"]) if context.get("source_image") and Path(context["source_image"]).is_file() else None
        textured, receipt, job_dir = engine.texture(mesh, request.prompt or context.get("prompt", ""), source, receipt, job_dir)
        return binary_mesh(textured, receipt, job_dir)
    except HTTPException as exc:
        mark_job_failed(job_id, exc)
        raise
    except Exception as exc:
        mark_job_failed(job_id, exc)
        raise_job_error(exc, job_id)


@app.post("/v1/rig")
def rig(request: MeshRequest):
    job_id = ""
    try:
        receipt, job_dir = engine.new_job("rig", {})
        job_id = receipt.job_id
        mesh = decode_file(request.meshBase64, request.meshFilename, job_dir / "input")
        final, receipt, job_dir = engine.rig_game_ready(mesh, request.prompt, request.rigKind, receipt, job_dir)
        return binary_mesh(final, receipt, job_dir)
    except HTTPException as exc:
        mark_job_failed(job_id, exc)
        raise
    except Exception as exc:
        mark_job_failed(job_id, exc)
        raise_job_error(exc, job_id)


@app.post("/v1/postprocess")
def postprocess(request: PostProcessRequest):
    job_id = request.resumeJobId or failed_context_job(
        request.projectId, request.cardId, "postprocess", request.resumeFailedJob,
    )
    try:
        final, receipt, job_dir = run_postprocess_request(request, job_id)
        return binary_mesh(final, receipt, job_dir)
    except HTTPException as exc:
        candidate = job_id or str(contexts.get_exact(request.projectId, request.cardId).get("job_id", ""))
        mark_job_failed(candidate, exc)
        raise
    except Exception as exc:
        candidate = job_id or str(contexts.get_exact(request.projectId, request.cardId).get("job_id", ""))
        mark_job_failed(candidate, exc)
        raise_job_error(exc, candidate)


@app.post("/v1/full")
def full(request: FullRequest):
    job_id = request.resumeJobId or failed_context_job(
        request.projectId, request.cardId, "full", request.resumeFailedJob,
    )
    try:
        final, receipt, job_dir = run_full_request(request, job_id)
        return binary_mesh(final, receipt, job_dir)
    except HTTPException as exc:
        candidate = job_id or str(contexts.get_exact(request.projectId, request.cardId).get("job_id", ""))
        mark_job_failed(candidate, exc)
        raise
    except Exception as exc:
        candidate = job_id or str(contexts.get_exact(request.projectId, request.cardId).get("job_id", ""))
        mark_job_failed(candidate, exc)
        raise_job_error(exc, candidate)


@app.post("/v1/jobs/{job_id}/resume")
def resume_job(job_id: str):
    receipt = load_receipt(job_id)
    if receipt is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = config.jobs_root / job_id
    try:
        if receipt.operation == "full":
            image = locate_saved_source(receipt, job_dir, "image")
            params = receipt.parameters.get("full_request", {})
            final, receipt, job_dir = engine.full(image, resume_job_id=job_id, **params)
        elif receipt.operation == "postprocess":
            mesh = locate_saved_source(receipt, job_dir, "mesh")
            params = receipt.parameters.get("postprocess_request", {})
            final, receipt, job_dir = engine.postprocess(mesh, resume_job_id=job_id, **params)
        else:
            raise RuntimeError(f"Operation '{receipt.operation}' does not yet support no-upload resume")
        return binary_mesh(final, receipt, job_dir)
    except HTTPException as exc:
        mark_job_failed(job_id, exc)
        raise
    except Exception as exc:
        mark_job_failed(job_id, exc)
        raise_job_error(exc, job_id)


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str):
    receipt = receipt_path(job_id)
    if not receipt.is_file():
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(json.loads(receipt.read_text(encoding="utf-8-sig")))
