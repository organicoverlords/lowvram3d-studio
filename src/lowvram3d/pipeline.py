from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from .config import PipelineConfig
from .contracts import JobReceipt, StageReceipt, now_ms
from .runner import StageFailure, artifact_is_valid, run_stage
from .presets import get_profile
from .resume import build_contract, find_existing_input, sha256_file, store_or_validate_contract
from .raster_route import run_raster_texture_route
from .avatar import preprocess_subject


class PipelineEngine:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.ensure_dirs()
        self.package_root = Path(__file__).resolve().parents[2]
        self.python = sys.executable
    def new_job(self, operation: str, inputs: dict[str, str]) -> tuple[JobReceipt, Path]:
        receipt = JobReceipt.create(operation, self.config.lane_order, inputs)
        job_dir = self.config.jobs_root / receipt.job_id
        for name in ("input", "source", "preprocess", "analysis", "raw", "prepared", "split", "optimized", "uv", "controls", "views", "textured", "mapped", "parts", "rigged", "final", "output", "proof", "reports", "logs"):
            (job_dir / name).mkdir(parents=True, exist_ok=True)
        receipt.status = "running"
        self._write_receipt(receipt, job_dir)
        return receipt, job_dir
    def generate(self, image: Path, prompt: str, receipt: JobReceipt | None = None, job_dir: Path | None = None) -> tuple[Path, JobReceipt, Path]:
        if receipt is None or job_dir is None:
            receipt, job_dir = self.new_job("generate", {"image": str(image)})
        receipt.status = "running"
        receipt.error = None
        receipt.finished_at = None
        receipt.input_files.setdefault("image", str(image))
        raw_mesh = job_dir / "raw" / "mesh.glb"
        prepared = job_dir / "prepared" / "mesh.glb"
        stats = job_dir / "proof" / "prepare_stats.json"

        geometry_stages = {"mini_turbo", "proxy_geometry", "geometry_resume"}
        raw_reusable = artifact_is_valid(raw_mesh) and any(
            item.stage in geometry_stages and item.status in {"passed", "reused"}
            for item in receipt.stages
        )
        geometry_error: str | None = None
        if raw_reusable:
            receipt.stages.append(
                StageReceipt(
                    "geometry_resume", "reused", now_ms(), finished_at=now_ms(),
                    artifacts={"mesh": str(raw_mesh)},
                    notes=["Reused existing validated high-poly geometry."],
                )
            )
            self._write_receipt(receipt, job_dir)
        else:
            mini_attempted = False
            for lane in self.config.lane_order:
                if lane in {"A", "B"} and not mini_attempted:
                    mini_attempted = True
                    try:
                        self._mini_turbo(image, prompt, raw_mesh, receipt, job_dir)
                        receipt.selected_lane = lane
                        break
                    except StageFailure as exc:
                        geometry_error = exc.receipt.error
                        self._record_stage_failure(receipt, job_dir, exc.receipt)
                        receipt.status = "running"
                        receipt.error = None
                if lane == "C":
                    try:
                        self._proxy_geometry(image, raw_mesh, receipt, job_dir)
                        receipt.selected_lane = "C"
                        break
                    except StageFailure as exc:
                        geometry_error = exc.receipt.error
                        self._record_stage_failure(receipt, job_dir, exc.receipt)
                        receipt.status = "running"
                        receipt.error = None
            if not artifact_is_valid(raw_mesh):  # is_file() would pass a corrupt lane output to Blender
                return self._fail(receipt, job_dir, geometry_error or "All geometry lanes failed")

        self._reusable_blender_stage(
            "prepare",
            "prepare.py",
            ["--input", raw_mesh, "--output", prepared, "--stats", stats, "--target-faces", self.config.target_faces],
            {"mesh": prepared, "stats": stats},
            receipt,
            job_dir,
        )
        receipt.outputs.update({"raw_mesh": str(raw_mesh), "prepared_mesh": str(prepared), "prepare_stats": str(stats)})
        self._write_receipt(receipt, job_dir)
        return prepared, receipt, job_dir
    def texture(
        self,
        mesh: Path,
        prompt: str,
        source_image: Path | None = None,
        receipt: JobReceipt | None = None,
        job_dir: Path | None = None,
    ) -> tuple[Path, JobReceipt, Path]:
        if receipt is None or job_dir is None:
            inputs = {"mesh": str(mesh)}
            if source_image:
                inputs["source_image"] = str(source_image)
            receipt, job_dir = self.new_job("texture", inputs)
        prepared = job_dir / "prepared" / "texture_input.glb"
        stats = job_dir / "proof" / "texture_prepare_stats.json"
        self._blender_stage(
            "texture_prepare",
            "prepare.py",
            ["--input", mesh, "--output", prepared, "--stats", stats, "--target-faces", self.config.target_faces],
            {"mesh": prepared, "stats": stats},
            receipt,
            job_dir,
        )
        last_error = "No texture lane attempted"
        textured: Path | None = None
        for lane in self.config.lane_order:
            try:
                if lane == "A":
                    textured = self._texture_mv_adapter(prepared, prompt, source_image, receipt, job_dir)
                elif lane == "B":
                    textured = self._texture_projection(prepared, prompt, source_image, receipt, job_dir)
                elif lane == "C" and source_image:
                    textured = job_dir / "textured" / "proxy_textured.glb"
                    self._proxy_geometry(source_image, textured, receipt, job_dir, stage="texture_proxy")
                else:
                    continue
                receipt.selected_lane = lane
                break
            except StageFailure as exc:
                last_error = exc.receipt.error or f"Lane {lane} failed"
                self._record_stage_failure(receipt, job_dir, exc.receipt)
                receipt.status = "running"
                receipt.error = None
        if textured is None or not textured.is_file():
            return self._fail(receipt, job_dir, last_error)
        mapped = job_dir / "mapped" / "asset.glb"
        maps_dir = job_dir / "mapped" / "maps"
        map_args: list[object] = ["--input", textured, "--output", mapped, "--maps-dir", maps_dir, "--size", self.config.texture_size, "--material-hint", prompt]
        high_poly = receipt.outputs.get("raw_mesh")
        if high_poly and Path(high_poly).is_file():
            map_args += ["--high-poly", high_poly]
        self._blender_stage(
            "bake_maps",
            "bake_maps.py",
            map_args,
            {
                "mesh": mapped,
                "basecolor": maps_dir / "basecolor.png",
                "normal": maps_dir / "normal.png",
                "roughness": maps_dir / "roughness.png",
                "metallic": maps_dir / "metallic.png",
                "ao": maps_dir / "ambient_occlusion.png",
            },
            receipt,
            job_dir,
        )
        receipt.outputs.update({"textured_mesh": str(mapped), "maps_dir": str(maps_dir)})
        self._write_receipt(receipt, job_dir)
        return mapped, receipt, job_dir
    def rig_game_ready(
        self,
        mesh: Path,
        prompt: str,
        rig_kind: str = "auto",
        receipt: JobReceipt | None = None,
        job_dir: Path | None = None,
    ) -> tuple[Path, JobReceipt, Path]:
        if receipt is None or job_dir is None:
            receipt, job_dir = self.new_job("rig", {"mesh": str(mesh)})
        split_mesh = job_dir / "parts" / "combined.glb"
        parts_manifest = job_dir / "parts" / "parts.json"
        self._blender_stage(
            "split_parts",
            "split_parts.py",
            [
                "--input", mesh,
                "--output", split_mesh,
                "--parts-dir", job_dir / "parts" / "individual",
                "--manifest", parts_manifest,
                "--asset-type", "vehicle" if rig_kind == "mechanical" else "creature",
                "--max-parts", 96,
                "--merge-small",
                "--separate-movable",
            ],
            {"mesh": split_mesh, "manifest": parts_manifest},
            receipt,
            job_dir,
        )
        rigged = job_dir / "rigged" / "asset.glb"
        rig_report = job_dir / "proof" / "rig_report.json"
        try:
            self._blender_stage(
                "rig_animate",
                "rig_animate.py",
                ["--input", split_mesh, "--output", rigged, "--report", rig_report, "--kind", rig_kind, "--prompt", prompt],
                {"mesh": rigged, "report": rig_report},
                receipt,
                job_dir,
            )
        except StageFailure as exc:
            self._record_stage_failure(receipt, job_dir, exc.receipt)
            receipt.status = "running"
            receipt.error = None
            # Rigid-part fallback still produces an animatable hierarchy for machines
            # and stylized multipart creatures when automatic skinning fails.
            self._blender_stage(
                "rig_rigid_fallback",
                "rig_animate.py",
                ["--input", split_mesh, "--output", rigged, "--report", rig_report, "--kind", "mechanical", "--prompt", prompt],
                {"mesh": rigged, "report": rig_report},
                receipt,
                job_dir,
            )
        final_dir = job_dir / "final"
        validation = job_dir / "proof" / "validation.json"
        preview = job_dir / "proof" / "preview.png"
        game_manifest = job_dir / "proof" / "game_manifest.json"
        self._blender_stage(
            "export_validate",
            "export_validate.py",
            ["--input", rigged, "--output-dir", final_dir, "--validation", validation, "--preview", preview, "--manifest", game_manifest],
            {
                "glb": final_dir / "asset.glb",
                "fbx": final_dir / "asset.fbx",
                "preview": preview,
                "validation": validation,
                "game_manifest": game_manifest,
            },
            receipt,
            job_dir,
        )
        receipt.status = "passed"
        receipt.finished_at = now_ms()
        receipt.outputs.update({
            "asset_glb": str(final_dir / "asset.glb"),
            "asset_fbx": str(final_dir / "asset.fbx"),
            "parts_manifest": str(parts_manifest),
            "rig_report": str(rig_report),
            "preview_png": str(preview),
            "validation_json": str(validation),
            "game_manifest_json": str(game_manifest),
            "job_receipt_json": str(job_dir / "proof" / "job_receipt.json"),
        })
        self._write_receipt(receipt, job_dir)
        return final_dir / "asset.glb", receipt, job_dir
    def postprocess(
        self,
        mesh: Path,
        *,
        asset_type: str = "auto",
        quality: str = "gameplay",
        separate_movable_parts: bool = True,
        texture_resolution: int | None = None,
        lod_enabled: bool = True,
        remove_hidden_geometry: bool = False,
        experimental_semantic_split: bool = False,
        prompt: str = "",
        source_image: Path | None = None,
        animation_preset: str = "dance",
        resume_job_id: str = "",
    ) -> tuple[Path, JobReceipt, Path]:
        from .postprocess import run_postprocess

        return run_postprocess(
            self,
            mesh,
            asset_type=asset_type,
            quality=quality,
            separate_movable_parts=separate_movable_parts,
            texture_resolution=texture_resolution,
            lod_enabled=lod_enabled,
            remove_hidden_geometry=remove_hidden_geometry,
            experimental_semantic_split=experimental_semantic_split,
            prompt=prompt,
            source_image=source_image,
            animation_preset=animation_preset,
            resume_job_id=resume_job_id,
        )
    def full(
        self,
        image: Path,
        prompt: str,
        *,
        asset_type: str = "auto",
        quality: str = "gameplay",
        separate_movable_parts: bool = True,
        texture_resolution: int | None = None,
        lod_enabled: bool = True,
        remove_hidden_geometry: bool = False,
        experimental_semantic_split: bool = False,
        background_removal: bool = True,
        animation_preset: str = "dance",
        receipt: JobReceipt | None = None,
        job_dir: Path | None = None,
        resume_job_id: str = "",
    ) -> tuple[Path, JobReceipt, Path]:
        if resume_job_id:
            job_dir = self.config.jobs_root / resume_job_id
            receipt_path = job_dir / "proof" / "job_receipt.json"
            if not receipt_path.is_file():
                raise RuntimeError(f"Resume job does not exist: {resume_job_id}")
            receipt = JobReceipt.load(receipt_path)
        elif receipt is None or job_dir is None:
            receipt, job_dir = self.new_job("full", {"image": str(image)})

        full_parameters = {
            "prompt": prompt,
            "asset_type": asset_type,
            "quality": quality,
            "separate_movable_parts": separate_movable_parts,
            "texture_resolution": texture_resolution,
            "lod_enabled": lod_enabled,
            "remove_hidden_geometry": remove_hidden_geometry,
            "experimental_semantic_split": experimental_semantic_split,
            "background_removal": background_removal,
            "animation_preset": animation_preset,
        }
        contract_source = image
        existing_input = find_existing_input(job_dir)
        if resume_job_id and existing_input is not None:
            if sha256_file(existing_input) != sha256_file(image):
                raise RuntimeError(
                    "Resume image mismatch: the requested image differs from the image that created this job. "
                    "Start a new job rather than reusing geometry from another image."
                )
            contract_source = existing_input
        full_contract = build_contract("full", contract_source, full_parameters)
        store_or_validate_contract(receipt, "full_contract", full_contract)
        receipt.parameters["full_request"] = full_parameters
        receipt.input_files["image"] = str(contract_source)
        receipt.operation = "full"
        receipt.status = "running"
        receipt.error = None
        receipt.finished_at = None
        self._write_receipt(receipt, job_dir)
        generation_image = contract_source
        resolved_profile = get_profile(asset_type, quality, texture_resolution, lod_enabled, prompt=prompt, filename=image.name)
        remove_subject_background = background_removal and resolved_profile.asset_type.value not in {"building", "room", "scene", "level"}
        if remove_subject_background or resolved_profile.asset_type.value == "avatar":
            generation_image = preprocess_subject(
                self, contract_source, receipt, job_dir, strict_avatar=resolved_profile.asset_type.value == "avatar"
            )
            receipt.outputs["preprocessed_image"] = str(generation_image)
            self._write_receipt(receipt, job_dir)
        generation_prompt = prompt
        if resolved_profile.asset_type.value == "avatar":
            generation_prompt = (
                f"{prompt}. Photorealistic full-body human avatar, preserve the person's face, hair, body proportions, "
                "clothing colors and visible accessories; anatomically coherent, neutral game-ready body, no duplicate limbs."
            ).strip(". ")
        _prepared, receipt, job_dir = self.generate(generation_image, generation_prompt, receipt, job_dir)
        receipt.parameters["geometry_lane"] = receipt.selected_lane
        self._write_receipt(receipt, job_dir)
        raw_mesh_value = receipt.outputs.get("raw_mesh")
        raw_mesh = Path(raw_mesh_value) if raw_mesh_value else _prepared
        if not raw_mesh.is_file():
            return self._fail(receipt, job_dir, "Geometry stage produced no high-poly source mesh")
        return self.postprocess(
            raw_mesh,
            asset_type=asset_type,
            quality=quality,
            separate_movable_parts=separate_movable_parts,
            texture_resolution=texture_resolution,
            lod_enabled=lod_enabled,
            remove_hidden_geometry=remove_hidden_geometry,
            experimental_semantic_split=experimental_semantic_split,
            prompt=prompt,
            source_image=generation_image,
            animation_preset=animation_preset,
            resume_job_id=receipt.job_id,
        )


    def _mini_turbo(self, image: Path, prompt: str, output: Path, receipt: JobReceipt, job_dir: Path) -> None:
        workflow = Path(self.config.mini_turbo_workflow)
        if not workflow.is_file():
            dummy = StageReceipt("mini_turbo", "failed", now_ms(), finished_at=now_ms(), error="Mini Turbo API workflow is not configured")
            raise StageFailure(dummy.error or "Mini Turbo unavailable", dummy)
        command = [
            self.python, str(self.package_root / "workers" / "comfyui_mini_turbo.py"),
            "--comfy-url", self.config.comfyui_url,
            "--workflow", str(workflow),
            "--input-image", str(image),
            "--output", str(output),
            "--prompt", prompt,
        ]
        self._command_stage("mini_turbo", command, {"mesh": output}, receipt, job_dir, timeout=3600)

    def _proxy_geometry(self, image: Path, output: Path, receipt: JobReceipt, job_dir: Path, stage: str = "proxy_geometry") -> None:
        command = [
            self.python, str(self.package_root / "workers" / "proxy_generate.py"),
            "--input-image", str(image), "--output", str(output),
            "--sf3d-python", self.config.sf3d_python, "--sf3d-root", self.config.sf3d_root,
            "--tripo-python", self.config.tripo_python, "--tripo-root", self.config.tripo_root,
        ]
        self._command_stage(stage, command, {"mesh": output}, receipt, job_dir, timeout=7200)

    def _texture_mv_adapter(self, mesh: Path, prompt: str, source: Path | None, receipt: JobReceipt, job_dir: Path) -> Path:
        controls = job_dir / "controls"
        metadata = controls / "cameras.json"
        self._blender_stage(
            "render_controls", "render_controls.py",
            ["--input", mesh, "--output-dir", controls, "--metadata", metadata, "--size", 512],
            {"metadata": metadata, "front_normal": controls / "front_normal.png", "front_position": controls / "front_position.png"},
            receipt, job_dir,
        )
        mv_python = Path(self.config.mv_adapter_python)
        mv_root = Path(self.config.mv_adapter_root)
        if not mv_python.is_file() or not mv_root.is_dir():
            dummy = StageReceipt("mv_adapter", "failed", now_ms(), finished_at=now_ms(), error="MV-Adapter environment is not installed")
            raise StageFailure(dummy.error or "MV-Adapter unavailable", dummy)
        views = job_dir / "views" / "mv_adapter"
        command = [
            str(mv_python), str(self.package_root / "workers" / "mv_adapter_from_controls.py"),
            "--repo", str(mv_root), "--controls-dir", str(controls), "--output-dir", str(views), "--prompt", prompt,
            "--base-model", self.config.sd21_base_model,
        ]
        if self.config.models_offline:
            command.append("--offline")
        self._command_stage(
            "mv_adapter", command,
            {"contact_sheet": views / "contact_sheet.png", "front": views / "front.png", "back": views / "back.png"},
            receipt, job_dir, timeout=3600,
        )
        output = job_dir / "textured" / "mv_adapter.glb"
        texture = job_dir / "textured" / "basecolor.png"
        project_args: list[object] = ["--input", mesh, "--views-dir", views, "--output", output, "--texture", texture, "--size", self.config.texture_size]
        if source:
            project_args += ["--source-image", source]
        self._blender_stage(
            "project_mv_views", "project_texture.py", project_args,
            {"mesh": output, "basecolor": texture}, receipt, job_dir,
        )
        return output

    def _texture_projection(self, mesh: Path, prompt: str, source: Path | None, receipt: JobReceipt, job_dir: Path) -> Path:
        views = job_dir / "views" / "projection"
        command = [
            self.python, str(self.package_root / "workers" / "make_fallback_views.py"),
            "--prompt", prompt, "--output-dir", str(views), "--size", "512",
        ]
        if source:
            command += ["--source-image", str(source)]
        self._command_stage(
            "fallback_views", command,
            {"contact_sheet": views / "contact_sheet.png", "front": views / "front.png", "back": views / "back.png"},
            receipt, job_dir,
        )
        if self.config.use_raster_texture_route:
            output, _texture = self._texture_projection_raster(mesh, views, receipt, job_dir)
            return output
        output = job_dir / "textured" / "projection.glb"
        texture = job_dir / "textured" / "basecolor.png"
        args: list[object] = ["--input", mesh, "--views-dir", views, "--output", output, "--texture", texture, "--size", self.config.texture_size]
        if source:
            args += ["--source-image", source]
        self._blender_stage("project_fallback", "project_texture.py", args, {"mesh": output, "basecolor": texture}, receipt, job_dir)
        return output

    def _texture_projection_raster(
        self,
        mesh: Path,
        views: Path,
        receipt: JobReceipt,
        job_dir: Path,
    ) -> tuple[Path, Path]:
        return run_raster_texture_route(self, mesh, views, receipt, job_dir)

    def _reusable_blender_stage(
        self,
        stage: str,
        script: str,
        args: list[object],
        artifacts: dict[str, Path | str],
        receipt: JobReceipt,
        job_dir: Path,
        timeout: int = 7200,
    ) -> None:
        normalized = {name: Path(path) for name, path in artifacts.items()}
        previous_ok = any(item.stage == stage and item.status in {"passed", "reused"} for item in receipt.stages)
        artifacts_ok = all(artifact_is_valid(path) for path in normalized.values())
        if previous_ok and artifacts_ok:
            receipt.stages.append(
                StageReceipt(
                    stage=stage,
                    status="reused",
                    started_at=now_ms(),
                    finished_at=now_ms(),
                    artifacts={name: str(path) for name, path in normalized.items()},
                    notes=["Reused validated stage outputs from the existing job directory."],
                )
            )
            self._write_receipt(receipt, job_dir)
            return
        self._blender_stage(stage, script, args, artifacts, receipt, job_dir, timeout=timeout)

    def _blender_stage(
        self,
        stage: str,
        script: str,
        args: list[object],
        artifacts: dict[str, Path | str],
        receipt: JobReceipt,
        job_dir: Path,
        timeout: int = 7200,
    ) -> None:
        # Blender ignores the inherited PYTHONPATH without --python-use-system-env, and every
        # blender/*.py script then dies on "No module named 'common'" before doing any work.
        command = [self.config.blender_path, "--background", "--python-use-system-env", "--python", str(self.package_root / "blender" / script), "--"]
        command.extend(str(value) for value in args)
        pythonpath = os.pathsep.join((str(self.package_root / "blender"), str(self.package_root / "src")))
        self._command_stage(stage, command, artifacts, receipt, job_dir, timeout=timeout, env={"PYTHONPATH": pythonpath})

    def _command_stage(
        self,
        stage: str,
        command: list[str],
        artifacts: dict[str, Path | str],
        receipt: JobReceipt,
        job_dir: Path,
        timeout: int = 3600,
        env: dict[str, str] | None = None,
    ) -> None:
        normalized = {name: str(path) for name, path in artifacts.items()}
        try:
            stage_receipt = run_stage(
                stage, command, self.package_root, job_dir / "logs", normalized,
                self.config.vram_ceiling_mb, env=env, timeout_seconds=timeout,
            )
        except StageFailure as exc:
            self._record_stage_failure(receipt, job_dir, exc.receipt)
            raise
        receipt.stages.append(stage_receipt)
        receipt.status = "running"
        receipt.error = None
        self._write_receipt(receipt, job_dir)

    def _record_stage_failure(self, receipt: JobReceipt, job_dir: Path, stage_receipt: StageReceipt) -> None:
        duplicate = any(
            item.stage == stage_receipt.stage and item.started_at == stage_receipt.started_at
            for item in receipt.stages
        )
        if not duplicate:
            receipt.stages.append(stage_receipt)
        receipt.status = "failed"
        receipt.error = stage_receipt.error or f"Stage {stage_receipt.stage} failed"
        self._write_receipt(receipt, job_dir)

    def _write_receipt(self, receipt: JobReceipt, job_dir: Path) -> None:
        receipt.outputs["job_receipt_json"] = str(job_dir / "proof" / "job_receipt.json")
        receipt.write(job_dir / "proof" / "job_receipt.json")

    def _fail(self, receipt: JobReceipt, job_dir: Path, error: str):
        receipt.status = "failed"
        receipt.error = error
        receipt.finished_at = now_ms()
        self._write_receipt(receipt, job_dir)
        raise RuntimeError(error)
