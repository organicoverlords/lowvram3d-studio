"""Hammer the call that faults, and count faults per thousand launches.

Every driver/clock change gets judged by the same number instead of by whether
one long job happened to survive. The fault is stochastic -- the same seed on the
same image passed 5 times and failed 7 -- so "it worked once" proves nothing and
a rate is the only honest measure.

The workload is deliberately the exact call the bluetree paint died in:
`torch.randn` on CUDA, as issued by
diffusers/schedulers/scheduling_euler_ancestral_discrete.py:427 through
randn_tensor, plus a matmul to keep the SMs busy the way a real diffusion step
does.

Run it before any change to get a baseline, then after each single change:

    python tools/gpu_fault_probe.py --seconds 120

A CUDA fault poisons the context and cannot be caught and retried in-process, so
this reports the launch count reached before the first fault. Compare runs by
launches-survived, and cross-check against the driver's own log:

    Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddMinutes(-10)} |
      Where-Object { $_.ProviderName -match 'nvlddmkm' } | Group-Object Id

An nvlddmkm Id 13 or 153 at the same second is what makes it a GPU fault rather
than a host problem -- the fennec SIGSEGV at 22:23:01 had no such event and was
host-side heap corruption, an entirely separate bug.
"""

import argparse
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--size", type=int, default=1024,
                    help="side of the square tensors; the default keeps the "
                         "footprint small so this measures stability, not VRAM")
    ap.add_argument("--label", default="", help="which change is under test")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")

    torch.backends.cudnn.enabled = False
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)

    n = args.size
    a = torch.randn(n, n, device=device)
    torch.cuda.synchronize()

    launches = 0
    started = time.time()
    fault = None
    try:
        while time.time() - started < args.seconds:
            for _ in range(50):
                # The faulting call itself.
                noise = torch.randn(n, n, device=device,
                                    generator=torch.Generator(device=device))
                a = torch.tanh(a @ noise * 1e-3)
                launches += 2
            # Synchronise outside the inner loop so faults are attributed to a
            # window rather than serialising every launch -- CUDA_LAUNCH_BLOCKING
            # is the tool for pinpointing, not for measuring a rate.
            torch.cuda.synchronize()
    except Exception as exc:            # noqa: BLE001 - the fault is the result
        fault = f"{type(exc).__name__}: {str(exc)[:300]}"

    elapsed = time.time() - started
    result = {
        "schema": "lowvram3d_gpu_fault_probe_v1",
        "label": args.label,
        "device": name,
        "seconds_requested": args.seconds,
        "seconds_elapsed": round(elapsed, 1),
        "launches": launches,
        "launches_per_second": round(launches / max(elapsed, 1e-9), 1),
        "faulted": fault is not None,
        "fault": fault,
        "verdict": ("FAULTED after %d launches" % launches) if fault
                   else ("clean for %d launches" % launches),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")


main()
