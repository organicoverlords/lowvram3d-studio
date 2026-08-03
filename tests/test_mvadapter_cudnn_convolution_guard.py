"""Convolution self-test and the UNet cuDNN guard it decides.

The defect this guards against is specific to one machine: on this GTX 1660 SUPER with this
PyTorch/CUDA/cuDNN build, ``down_blocks.0.resnets.0.conv1`` in FP16 returns a partly non-finite
tensor from entirely finite inputs when cuDNN is enabled, and agrees with FP32 when it is not. The
claim is therefore measured at startup rather than assumed, and these tests cover the decision, the
scope of the workaround, and the fact that nothing else in the pipeline is slowed down by it.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

UPSTREAM = Path(r"C:\AI\mvadapter-upstream-inspection")
if UPSTREAM.is_dir() and str(UPSTREAM) not in sys.path:
    sys.path.append(str(UPSTREAM))

try:  # torch/diffusers only exist in the MV-Adapter environment
    import torch
    import lowvram_mvadapter_i2mv_sd21 as lowvram

    HAS_STACK = True
    STACK_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on environment
    torch = None
    lowvram = None
    HAS_STACK = False
    STACK_ERROR = f"{type(exc).__name__}: {exc}"

SELF_TEST_FIELDS = (
    "gpu_name",
    "compute_capability",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "fp16_cudnn_finite_fraction",
    "fp16_no_cudnn_finite_fraction",
    "fp32_cudnn_finite_fraction",
    "max_error_fp16_no_cudnn_vs_fp32",
)


def _fake_unet():
    unet = torch.nn.Conv2d(2, 2, 3, padding=1)
    return unet


def _fake_pipe():
    return SimpleNamespace(unet=_fake_unet())


@unittest.skipUnless(HAS_STACK, f"torch stack unavailable: {STACK_ERROR}")
class CudnnFlagScopeTests(unittest.TestCase):
    def test_flags_helper_disables_cudnn_only_inside_the_block(self) -> None:
        before = torch.backends.cudnn.enabled
        with lowvram._cudnn_flags(False):
            self.assertFalse(torch.backends.cudnn.enabled)
        self.assertEqual(torch.backends.cudnn.enabled, before)

    def test_flags_helper_leaves_unrelated_backend_settings_alone(self) -> None:
        """``cudnn.flags`` has its own defaults for these; passing only ``enabled`` would reset them."""
        backend = torch.backends.cudnn
        original = (backend.benchmark, backend.deterministic, backend.allow_tf32)
        backend.benchmark, backend.deterministic, backend.allow_tf32 = True, True, False
        try:
            with lowvram._cudnn_flags(False):
                self.assertTrue(backend.benchmark)
                self.assertTrue(backend.deterministic)
                self.assertFalse(backend.allow_tf32)
        finally:
            backend.benchmark, backend.deterministic, backend.allow_tf32 = original


@unittest.skipUnless(HAS_STACK, f"torch stack unavailable: {STACK_ERROR}")
class GuardInstallationTests(unittest.TestCase):
    def test_guard_is_not_installed_when_the_fp16_cudnn_path_is_sound(self) -> None:
        pipe = _fake_pipe()

        report = lowvram.install_unet_cudnn_guard(
            pipe, {"unet_cudnn_disabled": False, "verdict_reason": "FP16_CUDNN_CONSISTENT"}
        )

        self.assertFalse(report["installed"])
        # ``forward`` is untouched: no instance attribute shadows the bound class method.
        self.assertNotIn("forward", vars(pipe.unet))
        self.assertFalse(getattr(pipe.unet, "_lowvram_cudnn_guard_installed", False))

    def test_guard_runs_the_unet_forward_with_cudnn_disabled(self) -> None:
        pipe = _fake_pipe()
        observed: list[bool] = []
        pipe.unet.forward = lambda sample: observed.append(torch.backends.cudnn.enabled) or sample

        report = lowvram.install_unet_cudnn_guard(
            pipe, {"unet_cudnn_disabled": True, "verdict_reason": "FP16_CUDNN_NON_FINITE"}
        )
        pipe.unet.forward(torch.zeros((1, 2, 4, 4)))

        self.assertTrue(report["installed"])
        self.assertEqual(observed, [False])

    def test_cudnn_is_restored_after_the_unet_forward_returns(self) -> None:
        pipe = _fake_pipe()
        before = torch.backends.cudnn.enabled
        lowvram.install_unet_cudnn_guard(pipe, {"unet_cudnn_disabled": True})

        pipe.unet.forward(torch.zeros((1, 2, 4, 4)))

        self.assertEqual(torch.backends.cudnn.enabled, before)

    def test_cudnn_is_restored_even_when_the_unet_forward_raises(self) -> None:
        pipe = _fake_pipe()

        def explode(_sample):
            raise RuntimeError("unet failed")

        pipe.unet.forward = explode
        lowvram.install_unet_cudnn_guard(pipe, {"unet_cudnn_disabled": True})
        before = torch.backends.cudnn.enabled

        with self.assertRaises(RuntimeError):
            pipe.unet.forward(torch.zeros((1, 2, 4, 4)))

        self.assertEqual(torch.backends.cudnn.enabled, before)

    def test_guard_is_installed_once(self) -> None:
        pipe = _fake_pipe()
        lowvram.install_unet_cudnn_guard(pipe, {"unet_cudnn_disabled": True})
        guarded = pipe.unet.forward

        second = lowvram.install_unet_cudnn_guard(pipe, {"unet_cudnn_disabled": True})

        self.assertTrue(second["already_installed"])
        self.assertIs(pipe.unet.forward, guarded)

    def test_guard_scope_is_the_unet_alone(self) -> None:
        """The FP32 VAE and FP32 condition encoder keep cuDNN; the defect is FP16-only."""
        pipe = _fake_pipe()
        report = lowvram.install_unet_cudnn_guard(pipe, {"unet_cudnn_disabled": True})

        self.assertEqual(report["scope"], "unet.forward")
        self.assertEqual(report["covers"], ["reference_unet_pass", "denoising_unet_pass"])
        for component in ("vae", "cond_encoder"):
            self.assertIn(component, report["cudnn_left_enabled_for"])


@unittest.skipUnless(HAS_STACK, f"torch stack unavailable: {STACK_ERROR}")
class SelfTestReportTests(unittest.TestCase):
    def test_self_test_records_the_full_runtime_identity_and_all_three_paths(self) -> None:
        record = lowvram.convolution_self_test(force=True)

        for field in SELF_TEST_FIELDS:
            self.assertIn(field, record)
        self.assertEqual(record["probe"]["module"], "down_blocks.0.resnets.0.conv1")
        self.assertIn(record["verdict_reason"] if record.get("executed") else "CUDA_UNAVAILABLE",
                      {"FP16_CUDNN_NON_FINITE", "FP16_CUDNN_INCONSISTENT_WITH_FP32",
                       "FP16_CUDNN_CONSISTENT", "CUDA_UNAVAILABLE"})

    @unittest.skipUnless(HAS_STACK and torch is not None and torch.cuda.is_available(),
                         "CUDA required")
    def test_the_selected_fallback_is_itself_finite_and_close_to_fp32(self) -> None:
        record = lowvram.convolution_self_test(force=True)

        self.assertTrue(record["executed"])
        self.assertEqual(record["fp16_no_cudnn_finite_fraction"], 1.0)
        self.assertEqual(record["fp32_cudnn_finite_fraction"], 1.0)
        self.assertLessEqual(record["max_error_fp16_no_cudnn_vs_fp32"],
                             lowvram.CONVOLUTION_MAX_ERROR_TOLERANCE)

    @unittest.skipUnless(HAS_STACK and torch is not None and torch.cuda.is_available(),
                         "CUDA required")
    def test_the_verdict_follows_from_the_measurements(self) -> None:
        record = lowvram.convolution_self_test(force=True)
        non_finite = record["fp16_cudnn_finite_fraction"] != 1.0
        error = record["max_error_fp16_cudnn_vs_fp32"]
        inconsistent = error is None or error > lowvram.CONVOLUTION_MAX_ERROR_TOLERANCE

        self.assertEqual(record["unet_cudnn_disabled"], bool(non_finite or inconsistent))
        self.assertEqual(record["cudnn_fp16_convolution_safe"], not (non_finite or inconsistent))

    def test_the_verdict_is_cached_so_it_is_measured_once_per_process(self) -> None:
        first = lowvram.convolution_self_test(force=True)
        second = lowvram.convolution_self_test()

        self.assertIs(first, second)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
