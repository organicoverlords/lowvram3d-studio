from __future__ import annotations

import base64
import importlib
import json
import os
import struct
import sys
import threading
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lowvram3d.config import PipelineConfig
from lowvram3d.contracts import JobReceipt, StageReceipt, now_ms
from lowvram3d.pipeline import PipelineEngine
from lowvram3d.presets import get_profile
from lowvram3d import runner as runner_module
from lowvram3d.runner import StageFailure, artifact_is_valid, ensure_artifacts, run_stage
from service.context_store import ContextStore
from service.models import FullRequest, GenerateRequest, MeshRequest, PostProcessRequest




def write_fake_artifact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text("{}", encoding="utf-8")
    elif path.suffix == ".png":
        from PIL import Image
        Image.new("RGBA", (2, 2), (127, 127, 127, 255)).save(path)
    elif path.suffix == ".glb":
        document = json.dumps({"asset": {"version": "2.0"}, "scenes": [{"nodes": []}], "scene": 0}).encode("utf-8")
        padding = (-len(document)) % 4
        document += b" " * padding
        total = 12 + 8 + len(document)
        path.write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(document), 0x4E4F534A) + document)
    else:
        path.write_bytes(b"artifact")


class ConfigEncodingTests(unittest.TestCase):
    def test_pipeline_config_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "local.json"
            payload = {
                "install_root": td,
                "jobs_root": str(Path(td) / "jobs"),
                "studio_url": "http://127.0.0.1:8311",
            }
            path.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))
            loaded = PipelineConfig.load(path)
            self.assertEqual(loaded.studio_url, "http://127.0.0.1:8311")


class RunnerTests(unittest.TestCase):
    def test_success_requires_nonempty_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "out.bin"
            receipt = run_stage(
                "make", [sys.executable, "-c", f"from pathlib import Path; Path(r'{output}').write_bytes(b'ok')"],
                root, root / "logs", {"mesh": str(output)}, 999999, timeout_seconds=30,
            )
            self.assertEqual(receipt.status, "passed")
            self.assertTrue(output.is_file())

    def test_avatar_preprocess_uses_the_same_gpu_lock(self):
        from lowvram3d.runner import is_gpu_heavy_stage

        self.assertTrue(is_gpu_heavy_stage("subject_preprocess"))
        self.assertTrue(is_gpu_heavy_stage("birefnet_avatar_preprocess"))

    def test_process_launch_failure_is_receipted_and_classified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(StageFailure) as caught:
                run_stage(
                    "missing-exe", [str(root / "does-not-exist.exe")], root, root / "logs", {}, 999999,
                    timeout_seconds=10,
                )
            self.assertEqual(caught.exception.receipt.status, "failed")
            self.assertEqual(caught.exception.receipt.failure_class, "process_launch")
            self.assertTrue(any(note.startswith("stderr=") for note in caught.exception.receipt.notes))

    def test_cuda_oom_is_classified_from_logs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            command = [sys.executable, "-c", "import sys; print('CUDA out of memory', file=sys.stderr); raise SystemExit(1)"]
            with self.assertRaises(StageFailure) as caught:
                run_stage("oom", command, root, root / "logs", {}, 999999, timeout_seconds=10)
            self.assertEqual(caught.exception.receipt.failure_class, "cuda_oom")
            self.assertIn("CUDA out of memory", " ".join(caught.exception.receipt.notes))

    def test_gpu_heavy_stages_are_serialized(self):
        active = 0
        peak = 0
        guard = threading.Lock()
        receipts = []

        def fake_impl(stage, command, cwd, logs_dir, required_artifacts, vram_ceiling_mb, env=None, timeout_seconds=None):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with guard:
                active -= 1
            return StageReceipt(stage, "passed", now_ms(), finished_at=now_ms())

        def invoke(name):
            receipts.append(runner_module.run_stage(name, [], Path('.'), Path('.'), {}, 5600))

        with patch.object(runner_module, "_run_stage_impl", side_effect=fake_impl):
            threads = [threading.Thread(target=invoke, args=(name,)) for name in ("mini_turbo_a", "mv_adapter_b")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(peak, 1)
        self.assertEqual(len(receipts), 2)
        self.assertTrue(all(any(note.startswith("gpu_lock_wait_ms=") for note in item.notes) for item in receipts))

    def test_truncated_glb_is_not_a_valid_checkpoint_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "broken.glb"
            path.write_bytes(b"glTF")
            self.assertFalse(artifact_is_valid(path))
            self.assertEqual(ensure_artifacts({"mesh": str(path)}), ["mesh"])

    def test_glb_requires_a_parseable_json_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "header-only.glb"
            path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12))
            self.assertFalse(artifact_is_valid(path))
            write_fake_artifact(path)
            self.assertTrue(artifact_is_valid(path))

    def test_truncated_png_is_not_a_valid_checkpoint_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "broken.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            self.assertFalse(artifact_is_valid(path))
            write_fake_artifact(path)
            self.assertTrue(artifact_is_valid(path))

    def test_failed_json_report_is_not_reusable(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "validation.json"
            path.write_text(json.dumps({"success": False, "errors": ["bad bake"]}), encoding="utf-8")
            self.assertFalse(artifact_is_valid(path))
            path.write_text(json.dumps({"success": True}), encoding="utf-8")
            self.assertTrue(artifact_is_valid(path))

    def test_zero_exit_missing_artifact_is_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(StageFailure) as ctx:
                run_stage("missing", [sys.executable, "-c", "pass"], root, root / "logs", {"mesh": str(root / "none.glb")}, 999999, timeout_seconds=30)
            self.assertIn("Missing artifacts", ctx.exception.receipt.error or "")


class GpuBudgetTests(unittest.TestCase):
    def test_stage_budget_is_not_charged_for_other_processes(self):
        with patch.object(runner_module, "total_gpu_memory_mb", return_value=6144):
            quiet, _ = runner_module.gpu_budget(5600, 200)
            busy, _ = runner_module.gpu_budget(5600, 2300)
        # nvidia-smi reports whole-card usage and Windows WDDM gives no per-process figure, so a
        # stage is budgeted against memory actually free rather than an absolute ceiling the
        # desktop has already eaten into.
        self.assertEqual(quiet, 6144 - 200 - runner_module.GPU_RESERVE_MB)
        self.assertEqual(busy, 6144 - 2300 - runner_module.GPU_RESERVE_MB)
        self.assertGreater(quiet, busy)

    def test_hard_cap_still_catches_a_runaway(self):
        with patch.object(runner_module, "total_gpu_memory_mb", return_value=6144):
            _, hard_cap = runner_module.gpu_budget(5600, 1100)
        self.assertEqual(hard_cap, 6144 - runner_module.GPU_RESERVE_MB)

    def test_absolute_ceiling_is_used_when_the_card_is_unknown(self):
        with patch.object(runner_module, "total_gpu_memory_mb", return_value=None):
            budget, hard_cap = runner_module.gpu_budget(5600, 1100)
        self.assertEqual(budget, 5600)
        self.assertIsNone(hard_cap)


class ContextTests(unittest.TestCase):
    def test_project_fallback_recovers_source_image(self):
        with tempfile.TemporaryDirectory() as td:
            store = ContextStore(Path(td))
            store.put("p1", "generation-card", {"source_image": "bird.png"})
            recovered = store.get("p1", "new-texture-card")
            self.assertEqual(recovered["source_image"], "bird.png")


class ModelTests(unittest.TestCase):
    def test_studio_alias_payloads(self):
        generation = GenerateRequest.model_validate({"image_base64": "YQ==", "image_filename": "x.png"})
        mesh = MeshRequest.model_validate({"mesh_base64": "YQ==", "mesh_filename": "x.glb"})
        self.assertEqual(generation.imageFilename, "x.png")
        self.assertEqual(mesh.meshFilename, "x.glb")
        post = PostProcessRequest.model_validate({
            "mesh_base64": "YQ==",
            "mesh_filename": "x.glb",
            "asset_type": "vehicle",
            "quality_preset": "hero",
            "texture_resolution": 4096,
        })
        self.assertEqual(post.assetType, "vehicle")
        self.assertEqual(post.qualityPreset, "hero")
        self.assertEqual(post.textureResolution, 4096)
        full = FullRequest.model_validate({
            "image_base64": "YQ==",
            "asset_type": "creature",
            "quality_preset": "hero",
            "texture_resolution": 4096,
            "separate_movable_parts": False,
            "background_removal": True,
            "animation_preset": "dance",
        })
        self.assertEqual(full.assetType, "creature")
        self.assertEqual(full.qualityPreset, "hero")
        self.assertFalse(full.separateMovableParts)
        self.assertTrue(full.resumeFailedJob)
        self.assertTrue(full.backgroundRemoval)
        self.assertEqual(full.animationPreset, "dance")
        self.assertEqual(full.resumeJobId, "")
        self.assertTrue(post.resumeFailedJob)


class FakeEngine(PipelineEngine):
    def __init__(self, config):
        super().__init__(config)
        self.mini_calls = 0

    def _mini_turbo(self, image, prompt, output, receipt, job_dir):
        self.mini_calls += 1
        failure = StageReceipt("mini_turbo", "failed", now_ms(), finished_at=now_ms(), error="not available")
        raise StageFailure("not available", failure)

    def _proxy_geometry(self, image, output, receipt, job_dir, stage="proxy_geometry"):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        write_fake_artifact(Path(output))
        receipt.stages.append(StageReceipt(stage, "passed", now_ms(), finished_at=now_ms(), artifacts={"mesh": str(output)}))

    def _blender_stage(self, stage, script, args, artifacts, receipt, job_dir, timeout=7200):
        for path in artifacts.values():
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_fake_artifact(target)
        receipt.stages.append(StageReceipt(stage, "passed", now_ms(), finished_at=now_ms(), artifacts={k: str(v) for k, v in artifacts.items()}))

    def _command_stage(self, stage, command, artifacts, receipt, job_dir, timeout=3600, env=None):
        for path in artifacts.values():
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_fake_artifact(target)
        receipt.stages.append(StageReceipt(stage, "passed", now_ms(), finished_at=now_ms(), artifacts={k: str(v) for k, v in artifacts.items()}))
        self._write_receipt(receipt, job_dir)


class FailOnceEngine(FakeEngine):
    def __init__(self, config, fail_stage="bake"):
        super().__init__(config)
        self.fail_stage = fail_stage
        self.failed_once = False
        self.geometry_calls = 0

    def _proxy_geometry(self, image, output, receipt, job_dir, stage="proxy_geometry"):
        self.geometry_calls += 1
        return super()._proxy_geometry(image, output, receipt, job_dir, stage)

    def _blender_stage(self, stage, script, args, artifacts, receipt, job_dir, timeout=7200):
        if stage == self.fail_stage and not self.failed_once:
            self.failed_once = True
            failure = StageReceipt(
                stage, "failed", now_ms(), finished_at=now_ms(), error=f"synthetic {stage} failure"
            )
            self._record_stage_failure(receipt, job_dir, failure)
            raise StageFailure(failure.error or "synthetic failure", failure)
        return super()._blender_stage(stage, script, args, artifacts, receipt, job_dir, timeout)




class ServiceResumeTests(unittest.TestCase):
    def test_same_failed_studio_card_auto_resumes_one_job(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "install_root": str(root),
                "jobs_root": str(root / "jobs"),
                "blender_path": "missing",
                "lane_order": ["C"],
            }), encoding="utf-8")
            previous = os.environ.get("LOWVRAM3D_CONFIG")
            os.environ["LOWVRAM3D_CONFIG"] = str(config_path)
            sys.modules.pop("service.app", None)
            try:
                module = importlib.import_module("service.app")
                cfg = PipelineConfig.load(config_path)
                fake = FailOnceEngine(cfg)
                module.config = cfg
                module.engine = fake
                module.contexts = ContextStore(root / "contexts")
                client = TestClient(module.app)
                payload = {
                    "image_base64": base64.b64encode(b"benchmark image").decode("ascii"),
                    "image_filename": "benchmark.png",
                    "prompt": "armoured creature",
                    "project_id": "project-1",
                    "card_id": "card-1",
                    "asset_type": "creature",
                    "quality_preset": "gameplay",
                }
                first = client.post("/v1/full", json=payload)
                self.assertEqual(first.status_code, 500)
                job_id = first.json()["detail"]["resume_job_id"]
                self.assertEqual(first.headers["x-lowvram3d-resume-job"], job_id)

                second = client.post("/v1/full", json=payload)
                self.assertEqual(second.status_code, 200, second.text)
                self.assertEqual(second.headers["x-lowvram3d-job"], job_id)
                self.assertEqual(fake.geometry_calls, 1)
                self.assertEqual(len([path for path in cfg.jobs_root.iterdir() if path.is_dir()]), 1)
            finally:
                sys.modules.pop("service.app", None)
                if previous is None:
                    os.environ.pop("LOWVRAM3D_CONFIG", None)
                else:
                    os.environ["LOWVRAM3D_CONFIG"] = previous


    def test_no_upload_resume_endpoint_continues_saved_job(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "install_root": str(root), "jobs_root": str(root / "jobs"),
                "blender_path": "missing", "lane_order": ["C"],
            }), encoding="utf-8")
            previous = os.environ.get("LOWVRAM3D_CONFIG")
            os.environ["LOWVRAM3D_CONFIG"] = str(config_path)
            sys.modules.pop("service.app", None)
            try:
                module = importlib.import_module("service.app")
                cfg = PipelineConfig.load(config_path)
                fake = FailOnceEngine(cfg)
                module.config = cfg
                module.engine = fake
                module.contexts = ContextStore(root / "contexts")
                client = TestClient(module.app)
                first = client.post("/v1/full", json={
                    "image_base64": base64.b64encode(b"benchmark image").decode("ascii"),
                    "image_filename": "benchmark.png", "prompt": "creature",
                    "project_id": "p", "card_id": "c", "asset_type": "creature",
                })
                self.assertEqual(first.status_code, 500)
                job_id = first.json()["detail"]["job_id"]
                resumed = client.post(f"/v1/jobs/{job_id}/resume")
                self.assertEqual(resumed.status_code, 200, resumed.text)
                self.assertEqual(resumed.headers["x-lowvram3d-job"], job_id)
                self.assertEqual(fake.geometry_calls, 1)
            finally:
                sys.modules.pop("service.app", None)
                if previous is None:
                    os.environ.pop("LOWVRAM3D_CONFIG", None)
                else:
                    os.environ["LOWVRAM3D_CONFIG"] = previous

    def test_invalid_upload_is_recorded_as_failed_job(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "install_root": str(root), "jobs_root": str(root / "jobs"), "blender_path": "missing",
            }), encoding="utf-8")
            previous = os.environ.get("LOWVRAM3D_CONFIG")
            os.environ["LOWVRAM3D_CONFIG"] = str(config_path)
            sys.modules.pop("service.app", None)
            try:
                module = importlib.import_module("service.app")
                module.contexts = ContextStore(root / "contexts")
                client = TestClient(module.app)
                response = client.post("/v1/full", json={
                    "image_base64": "not-valid-base64",
                    "project_id": "project-2", "card_id": "card-2",
                })
                self.assertEqual(response.status_code, 400)
                context = module.contexts.get_exact("project-2", "card-2")
                receipt = JobReceipt.load(root / "jobs" / context["job_id"] / "proof" / "job_receipt.json")
                self.assertEqual(receipt.status, "failed")
                self.assertIsNotNone(receipt.finished_at)
            finally:
                sys.modules.pop("service.app", None)
                if previous is None:
                    os.environ.pop("LOWVRAM3D_CONFIG", None)
                else:
                    os.environ["LOWVRAM3D_CONFIG"] = previous


class StudioMeshToolTests(unittest.TestCase):
    def test_sse_parser_accepts_progress_heartbeat_and_done(self):
        import base64
        from workers.studio_meshtools import parse_sse

        class Response:
            def iter_lines(self, decode_unicode=True):
                return iter([
                    "data: {\"type\":\"progress\",\"stage\":\"start\",\"frac\":0}",
                    ": keepalive",
                    "",
                    "data: {\"type\":\"done\",\"mesh_b64\":\"" + base64.b64encode(b"glb").decode("ascii") + "\",\"stats\":{\"face_count\":42}}",
                ])

        terminal, events = parse_sse(Response())
        self.assertEqual(terminal["type"], "done")
        self.assertEqual(terminal["stats"]["face_count"], 42)
        self.assertEqual(len(events), 2)

    def test_sse_parser_rejects_error_event(self):
        from workers.studio_meshtools import parse_sse

        class Response:
            def iter_lines(self, decode_unicode=True):
                return iter(["data: {\"type\":\"error\",\"detail\":\"retopo failed\"}"])

        with self.assertRaisesRegex(RuntimeError, "retopo failed"):
            parse_sse(Response())


class LaneTests(unittest.TestCase):
    def test_geometry_does_not_repeat_mini_turbo_for_texture_lane_b(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "input.png"
            image.write_bytes(b"image")
            cfg = PipelineConfig(install_root=root, jobs_root=root / "jobs", blender_path="missing", lane_order=["A", "B", "C"])
            engine = FakeEngine(cfg)
            mesh, receipt, _ = engine.generate(image, "bird")
            self.assertTrue(mesh.is_file())
            self.assertEqual(engine.mini_calls, 1)
            self.assertEqual(receipt.selected_lane, "C")

    def test_geometry_reaches_lane_c(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "input.png"
            image.write_bytes(b"image")
            cfg = PipelineConfig(install_root=root, jobs_root=root / "jobs", blender_path="missing", lane_order=["A", "B", "C"])
            mesh, receipt, _ = FakeEngine(cfg).generate(image, "bird")
            self.assertTrue(mesh.is_file())
            self.assertEqual(receipt.selected_lane, "C")

    def test_legacy_resume_rejects_changed_profile_before_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mesh = root / "input.glb"
            mesh.write_bytes(b"legacy mesh")
            cfg = PipelineConfig(install_root=root, jobs_root=root / "jobs", blender_path="missing")
            engine = FakeEngine(cfg)
            receipt, job_dir = engine.new_job("postprocess", {"mesh": str(mesh)})
            receipt.status = "failed"
            receipt.parameters = {
                "profile": get_profile("creature", "gameplay", 2048, True).to_dict(),
                "separate_movable_parts": True,
                "remove_hidden_geometry": False,
                "experimental_semantic_split": False,
                "prompt": "creature",
            }
            engine._write_receipt(receipt, job_dir)
            with self.assertRaisesRegex(RuntimeError, "Legacy resume settings mismatch"):
                engine.postprocess(
                    mesh, asset_type="vehicle", quality="gameplay", texture_resolution=2048,
                    prompt="vehicle", resume_job_id=receipt.job_id,
                )

    def test_postprocess_runs_complete_static_slice(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mesh = root / "input.glb"
            mesh.write_bytes(b"mesh")
            cfg = PipelineConfig(install_root=root, jobs_root=root / "jobs", blender_path="missing")
            output, receipt, job_dir = FakeEngine(cfg).postprocess(
                mesh,
                asset_type="vehicle",
                quality="gameplay",
                texture_resolution=2048,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(receipt.status, "passed")
            self.assertEqual(receipt.parameters["profile"]["asset_type"], "vehicle")
            stages = [item.stage for item in receipt.stages]
            self.assertEqual(
                stages,
                ["ingest_validate", "analyse", "split", "retopologize_blender", "uv_blender", "bake", "rig", "export_validate"],
            )
            self.assertTrue((job_dir / "proof" / "job_receipt.json").is_file())


class FullPipelineTests(unittest.TestCase):
    def test_failed_full_job_resumes_without_regenerating_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "input.png"
            image.write_bytes(b"same benchmark image")
            cfg = PipelineConfig(install_root=root, jobs_root=root / "jobs", blender_path="missing", lane_order=["C"])
            engine = FailOnceEngine(cfg)
            with self.assertRaisesRegex(StageFailure, "synthetic bake failure"):
                engine.full(image, "benchmark creature", asset_type="creature", quality="gameplay")
            jobs = [path for path in cfg.jobs_root.iterdir() if path.is_dir()]
            self.assertEqual(len(jobs), 1)
            job_id = jobs[0].name
            failed = JobReceipt.load(jobs[0] / "proof" / "job_receipt.json")
            self.assertEqual(failed.status, "failed")
            self.assertTrue(any(stage.stage == "bake" and stage.status == "failed" for stage in failed.stages))

            output, resumed, resumed_dir = engine.full(
                image, "benchmark creature", asset_type="creature", quality="gameplay", resume_job_id=job_id,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(resumed.status, "passed")
            self.assertEqual(resumed_dir.name, job_id)
            self.assertEqual(engine.geometry_calls, 1)
            self.assertEqual(len([path for path in cfg.jobs_root.iterdir() if path.is_dir()]), 1)
            self.assertTrue(any(stage.stage == "geometry_resume" and stage.status == "reused" for stage in resumed.stages))
            self.assertTrue(any(stage.status == "reused" for stage in resumed.stages))

    def test_resume_rejects_changed_settings_and_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "input.png"
            image.write_bytes(b"source A")
            cfg = PipelineConfig(install_root=root, jobs_root=root / "jobs", blender_path="missing", lane_order=["C"])
            engine = FailOnceEngine(cfg)
            with self.assertRaises(StageFailure):
                engine.full(image, "benchmark creature", asset_type="creature", quality="gameplay")
            job_id = next(cfg.jobs_root.iterdir()).name
            with self.assertRaisesRegex(RuntimeError, "Resume contract mismatch"):
                engine.full(image, "benchmark creature", asset_type="creature", quality="hero", resume_job_id=job_id)
            changed = root / "changed.png"
            changed.write_bytes(b"source B")
            with self.assertRaisesRegex(RuntimeError, r"Resume (image|contract) mismatch"):
                engine.full(changed, "benchmark creature", asset_type="creature", quality="gameplay", resume_job_id=job_id)

    def test_full_uses_raw_geometry_and_class_postprocess(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "input.png"
            image.write_bytes(b"image")
            cfg = PipelineConfig(install_root=root, jobs_root=root / "jobs", blender_path="missing", lane_order=["C"])
            engine = FakeEngine(cfg)
            output, receipt, job_dir = engine.full(
                image,
                "armoured raccoon creature",
                asset_type="creature",
                quality="gameplay",
            )
            self.assertTrue(output.is_file())
            self.assertEqual(receipt.operation, "full")
            self.assertEqual(receipt.parameters["profile"]["asset_type"], "creature")
            self.assertTrue((job_dir / "source" / "original.glb").is_file())
            self.assertIn("rig", [stage.stage for stage in receipt.stages])


if __name__ == "__main__":
    unittest.main()

class WorkflowConversionTests(unittest.TestCase):
    def test_ui_workflow_is_pruned_to_shape_ancestors(self):
        from lowvram3d.comfyui_client import ComfyUIClient
        ui = {
            "nodes": [
                {"id": 1, "type": "LoadImage", "inputs": [], "widgets_values": ["old.png", "image"]},
                {"id": 2, "type": "ShapeLoader", "inputs": [], "widgets_values": ["mini", True]},
                {"id": 3, "type": "ShapeGen", "inputs": [{"name": "pipe", "link": 10}, {"name": "image", "link": 11}], "widgets_values": [5, 256]},
                {"id": 4, "type": "[Comfy3D] Save 3D Mesh", "inputs": [{"name": "mesh", "link": 12}], "widgets_values": ["mesh_shape.glb"]},
                {"id": 9, "type": "PaintModel", "inputs": [], "widgets_values": []},
            ],
            "links": [[10, 2, 0, 3, 0, "PIPE"], [11, 1, 0, 3, 1, "IMAGE"], [12, 3, 0, 4, 0, "MESH"]],
        }
        info = {
            "LoadImage": {"input": {"required": {"image": [], "upload": []}}},
            "ShapeLoader": {"input": {"required": {"model": [], "low_vram": []}}},
            "ShapeGen": {"input": {"required": {"pipe": [], "image": [], "steps": [], "resolution": []}}},
            "[Comfy3D] Save 3D Mesh": {"input": {"required": {"mesh": [], "save_path": []}}},
            "PaintModel": {"input": {"required": {}}},
        }
        api = ComfyUIClient("http://unused").ui_to_api(ui, info)
        self.assertNotIn("9", api)
        self.assertEqual(api["1"]["inputs"]["image"], "${INPUT_IMAGE}")
        self.assertEqual(api["3"]["inputs"]["pipe"], ["2", 0])
        self.assertIn("mini_turbo_mesh.glb", api["4"]["inputs"]["save_path"])
