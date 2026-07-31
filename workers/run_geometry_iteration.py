"""Run one bounded geometry-first shaman iteration through the real pipeline.

This intentionally stops after geometry generation, high-detail preparation, fresh Blender
round-trip validation and neutral proof renders. Texture generation, UV projection, map baking,
part splitting, rigging and animation are not executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lowvram3d.avatar import preprocess_subject
from lowvram3d.config import PipelineConfig
from lowvram3d.pipeline import PipelineEngine


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty {label}: {path}")
    return path


def copy_required(source: Path, destination: Path, label: str) -> str:
    require_file(source, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    require_file(destination, f"copied {label}")
    return str(destination)


def copy_optional(source: Path, destination: Path) -> str | None:
    if not source.is_file() or source.stat().st_size <= 0:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-image-sha256", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--preserve-faces", type=int, default=350000)
    args = parser.parse_args()

    image = Path(args.image).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "geometry_iteration_result.json"

    payload: dict = {
        "success": False,
        "classification": "NOT_PROVEN",
        "source_image": str(image),
        "config": str(config_path),
        "preserve_faces": int(args.preserve_faces),
    }

    try:
        require_file(image, "source image")
        require_file(config_path, "pipeline config")
        image_hash = sha256(image)
        expected = args.expected_image_sha256.strip().lower()
        if expected and image_hash.lower() != expected:
            raise RuntimeError(
                f"Input image hash mismatch: expected {expected}, got {image_hash}"
            )

        config = PipelineConfig.load(config_path)
        # Preserve useful generated detail. The geometry master is always copied byte-for-byte;
        # this only raises the downstream preparation ceiling above the old 50k default.
        config.target_faces = max(50000, min(int(args.preserve_faces), 1000000))
        engine = PipelineEngine(config)
        receipt, job_dir = engine.new_job("geometry_iteration", {"image": str(image)})
        receipt.parameters["geometry_iteration"] = {
            "source_image_sha256": image_hash,
            "preserve_faces": config.target_faces,
            "scope": (
                "geometry generation + preparation + fresh roundtrip validation only; "
                "no texture, UV projection, map bake, split, rig, A-pose or animation"
            ),
        }
        engine._write_receipt(receipt, job_dir)

        payload.update(
            {
                "source_image_sha256": image_hash,
                "job_id": receipt.job_id,
                "job_dir": str(job_dir),
            }
        )

        generation_image = preprocess_subject(
            engine, image, receipt, job_dir, strict_avatar=False
        )
        receipt.outputs["geometry_iteration_input"] = str(generation_image)
        engine._write_receipt(receipt, job_dir)

        prepared, receipt, job_dir = engine.generate(
            generation_image, args.prompt, receipt, job_dir
        )
        raw_value = receipt.outputs.get("raw_mesh")
        raw_mesh = Path(raw_value) if raw_value else job_dir / "raw" / "mesh.glb"
        require_file(raw_mesh, "generated geometry master")
        require_file(prepared, "prepared geometry candidate")

        proof_dir = job_dir / "proof" / "geometry_iteration"
        preview_dir = proof_dir / "previews"
        roundtrip = proof_dir / "geometry_roundtrip.glb"
        validation = proof_dir / "geometry_validation.json"
        preview_front = preview_dir / "front.png"
        preview_three_quarter = preview_dir / "three_quarter.png"
        preview_side = preview_dir / "side.png"
        preview_back = preview_dir / "back.png"

        engine._blender_stage(
            "geometry_iteration_validate",
            "geometry_iteration_validate.py",
            [
                "--input",
                raw_mesh,
                "--roundtrip-output",
                roundtrip,
                "--validation",
                validation,
                "--preview-dir",
                preview_dir,
            ],
            {
                "roundtrip": roundtrip,
                "validation": validation,
                "preview_front": preview_front,
                "preview_three_quarter": preview_three_quarter,
                "preview_side": preview_side,
                "preview_back": preview_back,
            },
            receipt,
            job_dir,
            timeout=1200,
        )

        validation_data = json.loads(
            require_file(validation, "geometry validation report").read_text(
                encoding="utf-8-sig"
            )
        )
        if not validation_data.get("success"):
            raise RuntimeError(
                f"Fresh Blender geometry validation failed: {validation_data}"
            )

        deliverable = output_dir / "deliverable"
        deliverable.mkdir(parents=True, exist_ok=True)
        copied = {
            "geometry_master": copy_required(
                raw_mesh,
                deliverable / "shaman_geometry_master.glb",
                "generated geometry master",
            ),
            "geometry_working": copy_required(
                prepared,
                deliverable / "shaman_geometry_working.glb",
                "prepared geometry candidate",
            ),
            "geometry_roundtrip": copy_required(
                roundtrip,
                deliverable / "shaman_geometry_roundtrip.glb",
                "fresh Blender roundtrip geometry",
            ),
            "validation": copy_required(
                validation,
                deliverable / "geometry_validation.json",
                "geometry validation report",
            ),
            "preview_front": copy_required(
                preview_front, deliverable / "preview_front.png", "front preview"
            ),
            "preview_three_quarter": copy_required(
                preview_three_quarter,
                deliverable / "preview_three_quarter.png",
                "three-quarter preview",
            ),
            "preview_side": copy_required(
                preview_side, deliverable / "preview_side.png", "side preview"
            ),
            "preview_back": copy_required(
                preview_back, deliverable / "preview_back.png", "back preview"
            ),
        }
        copy_optional(
            job_dir / "preprocess" / "subject.png",
            deliverable / "pipeline_subject_input.png",
        )
        copy_optional(
            job_dir / "preprocess" / "subject_mask.png",
            deliverable / "pipeline_subject_mask.png",
        )
        copy_optional(
            job_dir / "proof" / "prepare_stats.json",
            deliverable / "prepare_stats.json",
        )

        receipt.operation = "geometry_iteration"
        receipt.status = "passed"
        receipt.error = None
        receipt.outputs.update(
            {
                "geometry_master": str(raw_mesh),
                "geometry_working": str(prepared),
                "geometry_roundtrip": str(roundtrip),
                "geometry_validation": str(validation),
                "geometry_preview_front": str(preview_front),
                "geometry_preview_three_quarter": str(preview_three_quarter),
                "geometry_preview_side": str(preview_side),
                "geometry_preview_back": str(preview_back),
            }
        )
        engine._write_receipt(receipt, job_dir)
        copy_required(
            job_dir / "proof" / "job_receipt.json",
            deliverable / "job_receipt.json",
            "job receipt",
        )

        master_delivery = Path(copied["geometry_master"])
        working_delivery = Path(copied["geometry_working"])
        roundtrip_delivery = Path(copied["geometry_roundtrip"])
        payload.update(
            {
                "success": True,
                "classification": "GEOMETRY_ITERATION_PASSED_REQUIRES_VISUAL_REVIEW",
                "selected_lane": receipt.selected_lane,
                "validation": validation_data,
                "deliverable_dir": str(deliverable),
                "geometry_master": str(master_delivery),
                "geometry_master_sha256": sha256(master_delivery),
                "geometry_master_bytes": master_delivery.stat().st_size,
                "geometry_working": str(working_delivery),
                "geometry_working_sha256": sha256(working_delivery),
                "geometry_working_bytes": working_delivery.stat().st_size,
                "geometry_roundtrip": str(roundtrip_delivery),
                "geometry_roundtrip_sha256": sha256(roundtrip_delivery),
                "geometry_roundtrip_bytes": roundtrip_delivery.stat().st_size,
                "copied_outputs": copied,
                "target_fbx_used": False,
                "texture_pipeline_run": False,
                "rig_pipeline_run": False,
            }
        )
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            "GEOMETRY_ITERATION_PASSED "
            f"job_id={receipt.job_id} lane={receipt.selected_lane} "
            f"master={master_delivery} result={result_path}",
            flush=True,
        )
        return 0
    except Exception as exc:
        payload.update(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"GEOMETRY_ITERATION_FAILED error={exc} result={result_path}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
