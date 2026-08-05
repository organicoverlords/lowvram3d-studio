"""Run trellis-cli with a complexity preflight, so a doomed decode aborts early.

Four subjects were run through this pipeline on a 16 GB machine:

    castle       latent 82,304   succeeded
    boat         latent 86,784   succeeded
    red panda    latent 140,480  died in FlexiDualGrid decode
    blue tree    latent 191,744  died in FlexiDualGrid decode

The split is not random and it is not the GPU. It is host RAM during decode, and
it tracks the size of the structured latent, which is a proxy for how much of the
volume the subject actually occupies. Rigid architecture activates few voxels;
fur, webbing and foliage activate many.

The useful part is *when* that number appears. `trellis-cli` prints it at stage 4
of 7 -- after roughly 90 s of flow, but **before** the decode that fails. So the
run can be stopped there with a clear reason instead of burning four minutes and
returning an ambiguous crash. Five separate panda attempts produced three
different errors (CUDA OOM, a ggml host-allocation assert, and a misaligned
address) for what turned out to be one cause.

The threshold is deliberately a warning band rather than a hard line: four
samples establish a correlation, not a boundary. `--max-latent 0` disables the
abort and just records the number, which is how the band gets refined -- run a
subject that ought to fail and see whether it does.

    py trellis_run.py --image matte.png --out asset.glb
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CLI = r"C:\AI\trellis-cpp\build-mmq\Release\trellis-cli.exe"
MODELS = r"C:\AI\trellis-cpp\models"

#: Above this, the decode has failed every time on 16 GB. Below it, every run
#: has succeeded. The gap between 87k and 140k is unsampled, so the default sits
#: near the top of the known-good range rather than in the middle of the void.
LATENT_WARN = 100_000
LATENT_ABORT = 130_000

#: Printed by trellis-cli at stage 4/7, e.g.
#:     [stats] slat (res32) n=86336 mean=0.3836 ...
LATENT_PATTERN = re.compile(r"slat \(res\d+\) n=(\d+)")

#: Flags that are not optional on this card, and why, are in
#: docs/RUNBOOK-trellis-pipeline.md. Summarised: --no-fa avoids the
#: tensor-core FlashAttention path that TU116 cannot execute, and
#: --require-gpu refuses a silent CPU fallback that would look like success.
FIXED_FLAGS = ["--no-fa", "--require-gpu"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Pre-matted RGBA input.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--res", default="512", choices=("512", "1024", "1536"))
    parser.add_argument("--atlas", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-texture", action="store_true")
    parser.add_argument("--max-latent", type=int, default=LATENT_ABORT,
                        help="Abort once the latent exceeds this. 0 disables "
                             "the abort and only records the value.")
    parser.add_argument("--log", default="")
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    image = Path(args.image).resolve()
    if not image.is_file():
        raise SystemExit(f"IMAGE_MISSING:{image}")

    command = [CLI, str(image), str(Path(args.out).resolve()),
               "--models", MODELS, "--res", args.res,
               "--atlas", str(args.atlas), "--seed", str(args.seed),
               *FIXED_FLAGS]
    if args.no_texture:
        command.append("--no-texture")

    started = time.time()
    lines: list[str] = []
    latent = None
    aborted = False

    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", bufsize=1)
    for line in process.stdout:
        lines.append(line.rstrip("\n"))
        sys.stdout.write(line)
        sys.stdout.flush()
        if latent is None:
            found = LATENT_PATTERN.search(line)
            if found:
                latent = int(found.group(1))
                if args.max_latent and latent > args.max_latent:
                    # Kill before the decode rather than after it. The failure
                    # downstream is a crash with a misleading message; this is a
                    # diagnosis.
                    aborted = True
                    process.kill()
                    break
    process.wait()
    elapsed = round(time.time() - started, 1)

    output = Path(args.out)
    produced = output.is_file() and output.stat().st_size > 4096
    if args.log:
        Path(args.log).write_text("\n".join(lines) + "\n", encoding="utf-8")

    band = ("unknown" if latent is None
            else "safe" if latent <= LATENT_WARN
            else "marginal" if latent <= LATENT_ABORT
            else "over")

    receipt = {
        "schema_version": "trellis_run_v1",
        "image": str(image),
        "output": str(output.resolve()),
        "command": command,
        "res": args.res,
        "atlas": args.atlas,
        "seed": args.seed,
        "seconds": elapsed,
        "latent_size": latent,
        "latent_band": band,
        "latent_warn": LATENT_WARN,
        "latent_abort": args.max_latent,
        "aborted_on_latent": aborted,
        "exit_code": process.returncode,
        "success": bool(produced and not aborted),
        "output_bytes": output.stat().st_size if output.is_file() else 0,
        "reference_latents": {"castle": 82304, "boat": 86784,
                              "red_panda": 140480, "blue_tree": 191744},
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))

    if aborted:
        # Report the band, not a verdict. The threshold is a policy the caller
        # set; whether this particular subject would have decoded is a separate
        # question, and saying otherwise turns a configurable limit into a
        # false claim about the hardware.
        observed = ("above every latent that has decoded successfully here"
                    if latent > LATENT_WARN else
                    "inside the range that has decoded successfully here, so "
                    "this abort reflects --max-latent rather than a known limit")
        print(f"\nTRELLIS_ABORTED_ON_COMPLEXITY latent={latent} > "
              f"--max-latent {args.max_latent}. That is {observed}. "
              f"Known-good up to {LATENT_WARN}; both failures so far were "
              f"{LATENT_ABORT}+. Options: simplify or crop the subject, or "
              f"raise --max-latent to probe the boundary.", file=sys.stderr)
        return 2
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
