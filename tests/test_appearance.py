from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lowvram3d.config import PipelineConfig
from lowvram3d.contracts import StageReceipt, now_ms
from tests.test_core import FakeEngine, write_fake_artifact


class AppearanceIntegrationTests(unittest.TestCase):
    def test_source_appearance_is_resolved_before_bake_and_forwarded(self):
        class RecordingEngine(FakeEngine):
            def __init__(self, config):
                super().__init__(config)
                self.blender_calls = []

            def _blender_stage(self, stage, script, args, artifacts, receipt, job_dir, timeout=7200):
                self.blender_calls.append((stage, script, [str(value) for value in args]))
                return super()._blender_stage(stage, script, args, artifacts, receipt, job_dir, timeout)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mesh = root / "input.glb"
            source = root / "source.png"
            write_fake_artifact(mesh)
            write_fake_artifact(source)
            cfg = PipelineConfig(install_root=root, jobs_root=root / "jobs", blender_path="missing", lane_order=["B"] )
            engine = RecordingEngine(cfg)

            def fake_appearance(_engine, uv_mesh, source_reference, prompt, receipt, job_dir, texture_size, padding_px):
                self.assertTrue(source_reference.is_file())
                self.assertEqual(source_reference.parent.name, "source")
                appearance = job_dir / "textured" / "projection" / "mesh.glb"
                basecolor = job_dir / "textured" / "projection" / "basecolor.png"
                write_fake_artifact(appearance)
                write_fake_artifact(basecolor)
                receipt.stages.append(StageReceipt(
                    "project_fallback_views", "passed", now_ms(), finished_at=now_ms(),
                    artifacts={"mesh": str(appearance), "basecolor": str(basecolor)},
                ))
                return appearance, basecolor, "B"

            with patch("lowvram3d.postprocess.resolve_source_appearance", side_effect=fake_appearance):
                output, receipt, job_dir = engine.postprocess(
                    mesh, asset_type="vehicle", quality="gameplay", source_image=source, prompt="painted vehicle",
                )

            self.assertTrue(output.is_file())
            stages = [stage for stage, _script, _args in engine.blender_calls]
            self.assertIn("bake", stages)
            bake_index = stages.index("bake")
            self.assertTrue(any(item.stage == "project_fallback_views" for item in receipt.stages[:]))
            bake_args = engine.blender_calls[bake_index][2]
            self.assertIn("--basecolor-image", bake_args)
            basecolor_index = bake_args.index("--basecolor-image") + 1
            self.assertEqual(Path(bake_args[basecolor_index]), job_dir / "textured" / "projection" / "basecolor.png")
            self.assertEqual(receipt.parameters["texture_lane"], "B")
            self.assertEqual(Path(receipt.outputs["appearance_mesh"]), job_dir / "textured" / "projection" / "mesh.glb")
            self.assertTrue((job_dir / "source" / "reference_image.png").is_file())



if __name__ == "__main__":
    unittest.main()
