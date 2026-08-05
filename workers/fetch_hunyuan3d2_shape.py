"""Fetch the full Hunyuan3D-2 shape model, and only that.

The local stack runs Hunyuan3D-2**mini** turbo: 0.6B parameters, DiT depth 8 with
16 single blocks. Online services produce clean, fully-3D geometry of the same
subjects from a single photograph, so the limiting factor here is model capacity
rather than the number of conditioning views, the octree resolution, or the step
count -- all three of which were measured and ruled out.

The full Hunyuan3D-2 shape model is 1.1B, the same class as the mv checkpoint
that already runs on this 6 GB card at a 4.97 GB peak. That is the argument for
expecting it to fit.

Only two subfolders are fetched, because the repository is 22.95 GB for the
shape DiT alone once its ckpt and fp32 duplicates are counted, and this machine
has under 10 GB free:

    hunyuan3d-dit-v2-0-turbo   4.59 GB   fp16 safetensors + config
    hunyuan3d-vae-v2-0-turbo   0.76 GB   matching VAE

The **turbo** variant is deliberate. Its scheduler is consistency-distilled, and
a measurement today showed 5 steps and 30 steps produce visually equivalent
geometry on the mv checkpoint -- 4.5x the runtime for nothing. The non-distilled
`hunyuan3d-dit-v2-0` would need tens of steps at roughly 90 s each.

    py fetch_hunyuan3d2_shape.py --output C:\\AI\\HY3D2\\HuggingFaceHub\\hunyuan3d-2-direct
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = "tencent/Hunyuan3D-2"

#: Everything needed to build a shape pipeline, and nothing else. Note the
#: exclusion of *.ckpt: the repo ships each weight twice, once as a pickle and
#: once as safetensors, and pulling both would double the cost for no benefit.
PATTERNS = [
    "hunyuan3d-dit-v2-0-turbo/config.yaml",
    "hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors",
    "hunyuan3d-vae-v2-0-turbo/*",
]

#: Refuse to start if the destination drive cannot hold the download plus a
#: working margin. A part-written 4.6 GB file that fills the system drive is a
#: worse outcome than not starting.
REQUIRED_FREE_GB = 6.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", default="")
    parser.add_argument("--allow-ckpt", action="store_true",
                        help="Also fetch the .ckpt duplicates. Doubles the size.")
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(output.drive + "\\").free / 1024 ** 3
    if free_gb < REQUIRED_FREE_GB:
        raise SystemExit(
            f"INSUFFICIENT_DISK: {free_gb:.1f} GB free on {output.drive}, "
            f"need {REQUIRED_FREE_GB} GB")

    patterns = list(PATTERNS)
    if args.allow_ckpt:
        patterns.append("hunyuan3d-dit-v2-0-turbo/*.ckpt")

    print(f"free before: {free_gb:.1f} GB", flush=True)
    print(f"fetching {REPO} -> {output}", flush=True)
    for pattern in patterns:
        print(f"  {pattern}", flush=True)

    snapshot_download(
        repo_id=REPO,
        local_dir=str(output),
        allow_patterns=patterns,
        max_workers=4,
    )

    fetched = sorted(
        (p for p in output.rglob("*") if p.is_file()),
        key=lambda p: -p.stat().st_size)
    total = sum(p.stat().st_size for p in fetched)
    after_gb = shutil.disk_usage(output.drive + "\\").free / 1024 ** 3

    receipt = {
        "schema_version": "hunyuan3d2_shape_fetch_v1",
        "repo": REPO,
        "output": str(output),
        "patterns": patterns,
        "files": [{"path": str(p.relative_to(output)),
                   "gb": round(p.stat().st_size / 1024 ** 3, 3)}
                  for p in fetched[:12]],
        "total_gb": round(total / 1024 ** 3, 3),
        "free_gb_after": round(after_gb, 2),
        "model_root": str(output),
        "subfolder": "hunyuan3d-dit-v2-0-turbo",
        "usage": ("workers/mini_turbo_generate.py --model-root <output> "
                  "--subfolder hunyuan3d-dit-v2-0-turbo --steps 5"),
        "note": ("single-image checkpoint: ImageProcessorV2, not "
                 "MVImageProcessorV2, so --view is not accepted"),
    }
    if args.receipt:
        Path(args.receipt).write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
