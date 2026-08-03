"""Low-VRAM MV-Adapter pipeline: component registration, offload and route locks.

Written as stdlib ``unittest`` so the torch-bearing MV-Adapter environment can
run it with ``python -m unittest`` (it has no pytest), while the pytest-driven
control environment collects and skips it cleanly (it has no torch).
"""
from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

UPSTREAM = Path(r"C:\AI\mvadapter-upstream-inspection")
if UPSTREAM.is_dir() and str(UPSTREAM) not in sys.path:
    # Appended, never prepended: the upstream checkout ships its own top-level
    # ``scripts`` package that would otherwise shadow this repository's.
    sys.path.append(str(UPSTREAM))

try:  # torch/diffusers only exist in the MV-Adapter environment
    import torch  # noqa: F401
    import lowvram_mvadapter_i2mv_sd21 as lowvram

    HAS_STACK = True
    STACK_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on environment
    lowvram = None
    HAS_STACK = False
    STACK_ERROR = f"{type(exc).__name__}: {exc}"

PREFLIGHT_RECEIPT = Path(
    r"C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803"
    r"\tactical_red_panda_scout\sd21_preflight_v3\sd21_load_only_receipt.json"
)


@unittest.skipUnless(HAS_STACK, f"torch/diffusers stack unavailable: {STACK_ERROR}")
class CondEncoderRegistrationTests(unittest.TestCase):
    def test_cond_encoder_is_in_the_constructor_signature(self) -> None:
        parameters = inspect.signature(lowvram.LowVRAMMVAdapterI2MVSDPipeline.__init__).parameters
        self.assertIn("cond_encoder", parameters)

    def test_cond_encoder_is_an_expected_component(self) -> None:
        """This is the exact mechanism diffusers uses to build ``pipe.components``."""
        pipeline_class = lowvram.LowVRAMMVAdapterI2MVSDPipeline
        expected, optional = pipeline_class._get_signature_keys(pipeline_class)
        self.assertIn("cond_encoder", expected)
        self.assertNotIn("cond_encoder", optional)
        for name in ("text_encoder", "unet", "vae"):
            self.assertIn(name, expected)

    def test_upstream_pipeline_does_not_expose_cond_encoder(self) -> None:
        """Guards the reason this subclass exists."""
        from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline

        expected, _optional = MVAdapterI2MVSDPipeline._get_signature_keys(MVAdapterI2MVSDPipeline)
        self.assertNotIn("cond_encoder", expected)

    def test_required_components_cover_the_cond_encoder(self) -> None:
        self.assertIn("cond_encoder", lowvram.REQUIRED_MODEL_COMPONENTS)
        for name in ("text_encoder", "unet", "vae"):
            self.assertIn(name, lowvram.REQUIRED_MODEL_COMPONENTS)


@unittest.skipUnless(HAS_STACK, f"torch/diffusers stack unavailable: {STACK_ERROR}")
class RouteRestrictionTests(unittest.TestCase):
    def test_text_only_adapter_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            lowvram.assert_image_geometry_adapter("mvadapter_t2mv_sd21.safetensors")
        self.assertIn("TEXT_CONDITIONED_ADAPTER_FORBIDDEN", str(ctx.exception))

    def test_text_plus_geometry_adapter_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            lowvram.assert_image_geometry_adapter("mvadapter_tg2mv_sd21.safetensors")
        self.assertIn("TEXT_CONDITIONED_ADAPTER_FORBIDDEN", str(ctx.exception))

    def test_unrelated_adapter_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            lowvram.assert_image_geometry_adapter("mvadapter_i2mv_sdxl.safetensors")

    def test_image_geometry_adapter_is_accepted(self) -> None:
        lowvram.assert_image_geometry_adapter(lowvram.REQUIRED_ADAPTER_NAME)

    def test_nvdiffrast_is_not_imported(self) -> None:
        self.assertFalse(
            any(name == "nvdiffrast" or name.startswith("nvdiffrast.") for name in sys.modules)
        )


@unittest.skipUnless(PREFLIGHT_RECEIPT.is_file(), "preflight receipt not produced on this machine")
class PreflightReceiptTests(unittest.TestCase):
    """Integration evidence from the real bounded preflight run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt = json.loads(PREFLIGHT_RECEIPT.read_text(encoding="utf-8"))

    def test_preflight_succeeded_with_the_local_subclass(self) -> None:
        self.assertTrue(self.receipt["success"])
        self.assertEqual(self.receipt["pipeline_class"], "LowVRAMMVAdapterI2MVSDPipeline")
        self.assertEqual(self.receipt["adapter_file"], "mvadapter_ig2mv_sd21.safetensors")

    def test_cond_encoder_registered_and_hooked(self) -> None:
        self.assertTrue(self.receipt["cond_encoder_registered"])
        self.assertTrue(self.receipt["cond_encoder_hooked"])
        cond = self.receipt["component_inventory"]["cond_encoder"]
        self.assertTrue(cond["is_torch_module"])
        self.assertGreater(cond["parameter_count"], 0)
        self.assertEqual(cond["dtypes"], ["float16"])

    def test_every_required_component_has_a_hook_and_is_not_resident(self) -> None:
        hooks = self.receipt["component_hooks_after_offload"]
        for name in ("text_encoder", "unet", "vae", "cond_encoder"):
            self.assertIn(name, hooks, name)
            self.assertTrue(hooks[name]["hook_installed"], name)
            self.assertGreaterEqual(hooks[name]["submodule_hook_count"], 1)
            self.assertIn("cuda:0", hooks[name]["execution_devices"], name)
            self.assertFalse(hooks[name]["resident_on_cuda"], name)

    def test_offload_and_attention_settings(self) -> None:
        self.assertEqual(self.receipt["offload_mode"], "SEQUENTIAL")
        self.assertEqual(self.receipt["attention_backend"], "PYTORCH_SDPA")
        self.assertEqual(self.receipt["attention_slicing"], "DISABLED")
        self.assertEqual(self.receipt["vae_slicing"], "ENABLED")
        self.assertEqual(self.receipt["attention_report"]["sliced_processor_classes"], [])

    def test_adapter_keys_all_accounted_for(self) -> None:
        report = self.receipt["adapter_report"]
        self.assertEqual(report["adapter_missing_keys"], [])
        self.assertEqual(report["adapter_unexpected_keys"], [])
        self.assertEqual(report["cond_encoder_missing_keys"], [])
        self.assertGreater(report["cond_encoder_keys_loaded"], 0)
        self.assertGreater(report["unet_adapter_keys_loaded"], 0)
        self.assertEqual(
            report["adapter_loaded_key_count"],
            report["unet_adapter_keys_loaded"] + report["cond_encoder_keys_loaded"],
        )
        self.assertTrue(report["cond_encoder_weights_changed"])

    def test_device_path_smoke_test_ran_on_cuda_and_released(self) -> None:
        smoke = self.receipt["device_path_smoke_test"]
        self.assertTrue(smoke["passed"])
        self.assertEqual(smoke["cond_encoder_execution_device"], "cuda:0")
        self.assertEqual(smoke["cond_encoder_parameter_device_during_forward"], "cuda:0")
        self.assertGreater(smoke["cond_encoder_output_level_count"], 0)
        self.assertTrue(smoke["cond_encoder_output_finite"])
        self.assertFalse(smoke["cond_encoder_resident_on_cuda_after"])
        self.assertTrue(smoke["cuda_memory_released"])

    def test_no_generation_happened(self) -> None:
        smoke = self.receipt["device_path_smoke_test"]
        self.assertFalse(smoke["unet_denoising_called"])
        self.assertFalse(smoke["reference_unet_pass_called"])
        self.assertFalse(smoke["scheduler_step_called"])
        self.assertFalse(smoke["vae_decode_called"])
        self.assertEqual(smoke["output_images"], 0)
        self.assertFalse(smoke["gpu_sequence_consumed"])
        self.assertFalse(self.receipt["denoising_called"])
        self.assertFalse(self.receipt["reference_unet_pass_called"])
        self.assertEqual(self.receipt["output_images"], [])
        self.assertFalse(self.receipt["nvdiffrast_imported"])


if __name__ == "__main__":
    unittest.main()
