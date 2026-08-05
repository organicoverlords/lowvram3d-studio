"""Run the production six-view worker with cuDNN off, as a controlled experiment.

The first barn six-view run completed all twenty steps and produced six pure
black images. Every input was clean -- initial latent absolute max 4.07,
reference latent 9.65, control features finite with max 1.0, condition
residuals finite -- and the latents were 100% NaN immediately after the first
UNet forward.

The pipeline's own cuDNN guard did not install, because its self-test reported
`fp16_cudnn_finite_fraction: 1.0` and the guard is gated on that test tripping.
A 40-trial repeat of the same probe here also produced 0 failures, so the probe
does not reproduce the defect -- but the probe builds a *fresh random* Conv2d,
and this GPU's recorded fp16 convolution defect is weight-dependent. A probe
that cannot reproduce it is not evidence that the real UNet is unaffected.

So this changes exactly one thing and re-runs: cuDNN off process-wide. If the
NaN disappears, the cause is convolution and the guard's gate is unsound. If it
survives, cuDNN is exonerated and the cause is elsewhere -- which is worth as
much, and is why this is a launcher rather than a fix.

It deliberately does not edit the production repo: another agent has a job
running there, and a shared working tree is not the place for an experiment.

    py -3.12 workers/run_sixview_no_cudnn.py --config C.json --output-dir D
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

WORKER = Path(r"C:\Users\Lauri\Desktop\lowvram3d-two-character-production-20260804"
              r"\workers\mvadapter_sd21_six_view_inference.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attempt", default="primary")
    parser.add_argument("--worker", default=str(WORKER))
    parser.add_argument(
        "--guidance-scale", type=float, default=None,
        help="Override the worker's hardcoded guidance_scale=1.0. Above 1 this "
             "enables real CFG, which doubles the UNet batch.")
    parser.add_argument(
        "--negative-prompt", default=None,
        help="Only meaningful with --guidance-scale > 1.")
    args = parser.parse_args()

    import torch

    # Before the worker imports anything that builds a module. cudnn.enabled is
    # read at each convolution call, so setting it here covers the whole run --
    # UNet, VAE and condition encoder alike. Broader than the pipeline's own
    # guard, which wraps only the UNet; that is intentional, because the point
    # is to answer "is it convolution at all", not to ship this setting.
    torch.backends.cudnn.enabled = False
    print(f"cudnn.enabled={torch.backends.cudnn.enabled} "
          f"torch={torch.__version__}", flush=True)

    if args.guidance_scale is not None:
        _override_guidance(args.guidance_scale, args.negative_prompt)

    sys.argv = [args.worker,
                "--config", args.config,
                "--output-dir", args.output_dir,
                "--attempt", args.attempt]
    runpy.run_path(args.worker, run_name="__main__")
    return 0


def _override_guidance(guidance_scale: float, negative_prompt: str | None) -> None:
    """Force a guidance scale the production worker does not expose.

    That worker passes `guidance_scale=1.0` and `negative_prompt=None` as
    literals, so there is no config route to classifier-free guidance -- and at
    exactly 1.0 CFG is *off*, which is a documented cause of washed-out,
    desaturated SD output. The boat's six views came back neutral grey to within
    three levels across channels while the conditioning photograph is warm brown,
    which looks far more like the reference being ignored than like an exposure
    error. Testing that costs one run; assuming it costs a wrong texture.

    The patch lands on the upstream base class rather than the LowVRAM subclass,
    which does not override `__call__`, and it is applied here rather than in the
    production tree because another agent has a job running there.

    Note this is a real cost, not a free knob: CFG concatenates the unconditional
    and conditional batches, so the UNet runs at double width. The last run peaked
    at 2359 MB of 6144, so there is room, but not unlimited room.
    """
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline

    original = MVAdapterI2MVSDPipeline.__call__

    def patched(self, *call_args, **kwargs):
        kwargs["guidance_scale"] = guidance_scale
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        return original(self, *call_args, **kwargs)

    MVAdapterI2MVSDPipeline.__call__ = patched
    print(f"guidance_scale override -> {guidance_scale} "
          f"(cfg_enabled={guidance_scale > 1}) negative_prompt={negative_prompt!r}",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
