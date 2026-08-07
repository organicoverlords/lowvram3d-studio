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

#: Latent size is reported, and by default no longer aborts. It was worth a try
#: and it did not survive contact with a bigger sample:
#:
#:      82,304  castle 1        decoded
#:      86,784  boat            decoded
#:      94,528  castle 2        died stage 5
#:      96,448  castle 2        died stage 5
#:     106,496  castle 2        decoded geometry, died stage 6
#:     111,520  castle 2        FULL SUCCESS, 3,949,828 faces
#:     134,944  castle 2        aborted by this threshold -- never tested
#:     140,480  red panda       died stage 5
#:     191,744  blue tree       died stage 5
#:
#: 96,448 fails and 111,520 succeeds, so there is no threshold in that range
#: that separates them. The earlier four-sample "band" was four single draws
#: from one image each, read as four subject properties. One image alone draws
#: 94,528 to 134,944 -- a 43% spread -- because the sparse-structure stage is
#: nondeterministic, so the latent is not even a stable property of a subject.
#:
#: What it is still good for: 140k+ has never decoded, so a very large value is
#: a genuine warning. Pass --max-latent explicitly to re-enable the abort for
#: unattended batch work, where burning 260 s on a doomed run costs more than
#: skipping a run that might have worked.
LATENT_WARN = 120_000
LATENT_ABORT = 0

#: Printed by trellis-cli at stage 4/7, e.g.
#:     [stats] slat (res32) n=86336 mean=0.3836 ...
LATENT_PATTERN = re.compile(r"slat \(res\d+\) n=(\d+)")

#: Stage 5 reports what the geometry decode actually produced. Captured because
#: the *second* failure mode is sized by this, not by the latent.
VOXEL_PATTERN = re.compile(r"decoded voxels @res\d+ = (\d+)")
MESH_PATTERN = re.compile(r"mesh V=(\d+) F=(\d+)")

#: There are two distinct capacity walls and they are not the same wall.
#:
#:   host RAM, stage 5  -- FlexiDualGrid shape decode. Predicted by latent size:
#:                         panda 140,480 and blue tree 191,744 both died here.
#:   VRAM, stage 6      -- the PBR decode compute graph, whose size tracks the
#:                         *decoded voxel count*, not the latent. A 4K gothic
#:                         castle at latent 106,496 cleared stage 5 comfortably
#:                         (3.47M faces) and then asked for a 2110.87 MiB CUDA
#:                         buffer with roughly 4.3 GB free, and lost.
#:
#: So a "safe" latent does not imply the run will finish, and a run that reaches
#: stage 6 has already banked a usable mesh. Both are reported.
VRAM_OOM_PATTERN = re.compile(
    r"allocating ([\d.]+) MiB on device \d+: cudaMalloc failed: out of memory")

#: Flags that are not optional on this card, and why, are in
#: docs/RUNBOOK-trellis-pipeline.md. Summarised: --no-fa avoids the
#: tensor-core FlashAttention path that TU116 cannot execute, and
#: --require-gpu refuses a silent CPU fallback that would look like success.
FIXED_FLAGS = ["--no-fa", "--require-gpu"]


def missing_weights(res: str, tex_res: int | None, no_texture: bool) -> list[str]:
    """GGUFs this configuration will need, that are not on disk.

    A res-1024 run reached stage 6 after 610 s of HR flow and a successful
    3.56M-voxel decode, and then died on

        failed to open GGUF file '.../tex_flow_1024.gguf' (No such file)

    Ten minutes of the most expensive compute this machine does, discarded for
    a missing file that could have been checked in a millisecond. The fetch
    script pulled the res-512 texture weights and the res-1024 SHAPE weights,
    so geometry at 1024 works and texture at 1024 cannot -- which is invisible
    until the run is nearly over.
    """
    needed = ["dinov3.gguf", "ss_flow.gguf", "ss_dec.gguf",
              f"shape_flow_{res}.gguf", "shape_dec.gguf"]
    if not no_texture:
        needed += [f"tex_flow_{tex_res or res}.gguf", "tex_dec.gguf"]
    return [n for n in needed if not (Path(MODELS) / n).is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Pre-matted RGBA input.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--res", default="512", choices=("512", "1024", "1536"))
    parser.add_argument("--atlas", type=int, default=0,
                        help="0 lets trellis-cli pick by resolution "
                             "(2048 at res 1024, 1024 at res 512)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-texture", action="store_true",
                        help="UNFINISHED GEOMETRY. This does not merely skip "
                             "colour: trellis-cli only enters its geometry "
                             "finalizer when PBR exists, so without a texture "
                             "it writes the RAW FlexiDualGrid decode and skips "
                             "weld, narrow-band remesh, winding repair, "
                             "component filtering and QEM simplification. "
                             "Measured on seed 1006: 2,035,434 faces, 51,502 "
                             "shells, 116,785 boundary edges against 146,076 / "
                             "161 / 0 with stage 6. Do not publish this.")
    parser.add_argument("--tex-res", type=int, choices=(512, 1024), default=None,
                        help="PBR decode volume resolution. The stage-6 CUDA "
                             "graph is the second capacity wall; dropping this "
                             "is the only lever that shrinks it without "
                             "abandoning the texture.")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="Everything after this is passed to trellis-cli "
                             "verbatim. Last argument.")
    parser.add_argument("--max-latent", type=int, default=LATENT_ABORT,
                        help="Abort once the latent exceeds this. 0 disables "
                             "the abort and only records the value.")
    parser.add_argument("--log", default="")
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    image = Path(args.image).resolve()
    if not image.is_file():
        raise SystemExit(f"IMAGE_MISSING:{image}")

    missing = missing_weights(args.res, args.tex_res, args.no_texture)
    if missing:
        raise SystemExit("WEIGHTS_MISSING:" + ",".join(missing))

    command = [CLI, str(image), str(Path(args.out).resolve()),
               "--models", MODELS, "--res", args.res,
               "--seed", str(args.seed), *FIXED_FLAGS]
    # Only pass --atlas when the caller asked for one. trellis-cli picks by
    # resolution (2048 at 1024, 1024 at 512); forwarding this worker's own
    # default silently overrode that on every cascade run ever made here.
    if args.atlas:
        command += ["--atlas", str(args.atlas)]
    if args.no_texture:
        command.append("--no-texture")
    if args.tex_res:
        command += ["--tex-res", str(args.tex_res)]
    command += list(args.extra)

    started = time.time()
    lines: list[str] = []
    latent = None
    aborted = False
    voxels = None
    mesh_faces = None
    vram_oom_mib = None

    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", bufsize=1)
    for line in process.stdout:
        lines.append(line.rstrip("\n"))
        sys.stdout.write(line)
        sys.stdout.flush()
        if voxels is None:
            found = VOXEL_PATTERN.search(line)
            if found:
                voxels = int(found.group(1))
        if mesh_faces is None:
            found = MESH_PATTERN.search(line)
            if found:
                mesh_faces = int(found.group(2))
        if vram_oom_mib is None:
            found = VRAM_OOM_PATTERN.search(line)
            if found:
                vram_oom_mib = float(found.group(1))
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
            else "typical" if latent <= LATENT_WARN
            else "large")

    failure_stage = None
    if process.returncode != 0:
        # Which stage it died in, so a failed attempt is data rather than a
        # blank. Stage 3 faults ("misaligned address", "illegal instruction")
        # are the shape-dependent TU116 kernel family and are expected at
        # roughly half of attempts; stage 5 is the host-RAM decode wall and
        # stage 6 is the VRAM one. They need different responses.
        stages = re.findall(r"\[(\d)/\d\]\s*(.+)", "\n".join(lines))
        failure_stage = stages[-1][1].strip()[:80] if stages else "before stage 1"

    receipt = {
        "schema_version": "trellis_run_v2",
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
        "decoded_voxels": voxels,
        "decoded_faces": mesh_faces,
        # Non-null means stage 5 finished and stage 6 ran out of VRAM. The
        # geometry was real; only the texture was lost.
        "vram_oom_mib": vram_oom_mib,
        "geometry_decoded": voxels is not None,
        # The single most important field in this receipt. False means the mesh
        # is the raw decoder output with none of the vendor finalization, and
        # is not a deliverable -- see --no-texture.
        "geometry_finalized": not args.no_texture,
        "finalizer_note": (
            "raw FlexiDualGrid decode; vendor stage 6 skipped"
            if args.no_texture else
            "vendor stage 6: weld, hole fill, narrow-band remesh, winding "
            "repair, component filter, QEM to 150k @512"),
        # Survival is per ATTEMPT, not per seed. Seed 1006 completed and then
        # died with "illegal instruction" on an identical rerun the same day.
        # Nothing here should be read as reproducible.
        "seed_reproducible": False,
        "failure_stage": failure_stage,
        "tex_res": args.tex_res,
        "exit_code": process.returncode,
        "success": bool(produced and not aborted),
        "output_bytes": output.stat().st_size if output.is_file() else 0,
        # Every latent observed, with what happened. Kept as a list of
        # observations rather than a threshold, because the threshold did not
        # survive: 96,448 failed and 111,520 succeeded on the same image.
        "reference_latents": [
            [82304, "castle1", "decoded"],
            [86784, "boat", "decoded"],
            [94528, "castle2", "died stage 5"],
            [96448, "castle2", "died stage 5"],
            [106496, "castle2", "geometry decoded, died stage 6"],
            [111520, "castle2", "full success, 3949828 faces"],
            [140480, "red_panda", "died stage 5"],
            [191744, "blue_tree", "died stage 5"],
        ],
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
        observed = ("larger than any latent that has decoded here (highest "
                    "success: 111,520)" if latent > 111_520 else
                    "inside the range that has decoded here, so this abort is "
                    "your --max-latent policy and not a known limit")
        print(f"\nTRELLIS_ABORTED_ON_COMPLEXITY latent={latent:,} > "
              f"--max-latent {args.max_latent:,}. That is {observed}. Note "
              f"96,448 has failed and 111,520 has succeeded on the same image, "
              f"so latent is a weak predictor; a different seed changes it by "
              f"tens of thousands. Prefer workers/trellis_retry_seeds.sh over "
              f"raising this.", file=sys.stderr)
        return 2
    if vram_oom_mib is not None:
        # Which stage matters, and it is not fixed. The same subject at the same
        # seed has failed at stage 6 with latent 106,496 and at stage 5 with
        # latent 94,528, minutes apart -- so this is contention for the card
        # with whatever else is running, not a property of the subject. Report
        # what actually happened rather than inferring a wall from one sample.
        if voxels is None:
            where = ("stage 5 (FlexiDualGrid shape decode). No geometry was "
                     "produced")
            advice = ("close other GPU consumers and retry; --no-texture "
                      "lowers peak but does not help this stage; simplify or "
                      "crop the subject")
        else:
            where = (f"stage 6 (PBR decode). Geometry HAD decoded: "
                     f"{voxels:,} voxels, {mesh_faces:,} faces")
            advice = ("--tex-res 512; close other GPU consumers; --no-texture "
                      "to bank the geometry and texture it separately")
        print(f"\nTRELLIS_VRAM_OOM at {where}. Asked for {vram_oom_mib:.0f} MiB "
              f"and failed; latent was "
              f"{latent:,} ({band}). This is a VRAM failure, not the latent "
              f"wall -- a smaller latent does not make it go away. Options: "
              f"{advice}.", file=sys.stderr)
        return 3
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
