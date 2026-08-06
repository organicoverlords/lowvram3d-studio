"""Run TRELLIS across several seeds and score each mesh on proportion.

The sparse-structure stage is nondeterministic, so seed is a real variable
rather than a formality: one image has drawn latents from 94,528 to 134,944, a
43 percent spread, and the meshes differ accordingly. Running one seed and
reading the result as a property of the generator is how this project has been
wrong before.

Scoring is against measured proportions rather than a look. The hosted reference
for the Lucky Drown is length:height 1.74 and length:beam 2.79; every Hunyuan
multiview mesh we produced came in at 1.04-1.12 and 1.82-1.89, and TRELLIS at
1.51-1.57 and 2.00-2.27 without anyone having checked. Proportion is the one
property a wrong result cannot hide behind detail, and it is cheap.

Targets are a per-CLASS constant, not per asset -- a vessel is long and low, a
tower is not -- so they are passed in, and the receipt records them.

    py workers/trellis_seed_tournament.py --image matte.png --out-dir tour \
       --seeds 8 --target-lh 1.74 --target-lb 2.79
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

WORKER = Path(__file__).with_name("trellis_run.py")


#: Percentile trimmed off each end of each axis before measuring proportion.
#: The bounding box is the wrong instrument here: TRELLIS output carries
#: floating debris, and a speck off the port bow widens the beam without
#: widening the boat. Measured on the same meshes, bounding-box L:B reads 1.81
#: for a fresh seed against 2.26 for a debris-cleaned one -- a gap that is
#: entirely cleanup, and which made cleaned meshes look like better draws.
EXTENT_TRIM_PERCENT = 0.5


def measure(path: Path) -> dict:
    """Proportions, and the winding check that decides if a mesh is exportable."""
    import numpy as np
    import trimesh

    scene = trimesh.load(path, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    extents = np.sort(mesh.extents)[::-1]

    points, _ = trimesh.sample.sample_surface(mesh, 300_000, seed=7)
    low = np.percentile(points, EXTENT_TRIM_PERCENT, axis=0)
    high = np.percentile(points, 100 - EXTENT_TRIM_PERCENT, axis=0)
    robust = np.sort(high - low)[::-1]

    return {
        "faces": int(len(mesh.faces)),
        "length_height": round(float(robust[0] / robust[1]), 4),
        "length_beam": round(float(robust[0] / robust[2]), 4),
        "bbox_length_height": round(float(extents[0] / extents[1]), 4),
        "bbox_length_beam": round(float(extents[0] / extents[2]), 4),
        # TRELLIS output has come back with inconsistent winding, which renders
        # as a black mesh and breaks texturing. It is a repair, not a defect,
        # but it must be seen rather than discovered downstream.
        "winding_consistent": bool(mesh.is_winding_consistent),
        "volume": round(float(mesh.volume), 6),
    }


def score(measured: dict, target_lh: float, target_lb: float) -> float:
    """Relative proportion error, symmetric in over- and under-shoot."""
    import math
    return round(
        abs(math.log(measured["length_height"] / target_lh))
        + abs(math.log(measured["length_beam"] / target_lb)), 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--res", default="512", choices=("512", "1024", "1536"))
    parser.add_argument("--target-lh", type=float, required=True)
    parser.add_argument("--target-lb", type=float, required=True)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in range(args.seeds):
        seed = args.first_seed + index
        out = args.out_dir / f"seed{seed}.glb"
        started = time.time()
        completed = subprocess.run(
            [sys.executable, str(WORKER), "--image", str(args.image),
             "--out", str(out), "--res", args.res, "--seed", str(seed),
             "--no-texture", "--log", str(args.out_dir / f"seed{seed}.log")],
            capture_output=True, text=True)
        entry = {"seed": seed, "seconds": round(time.time() - started, 1),
                 "returncode": completed.returncode, "mesh": str(out)}
        if completed.returncode == 0 and out.exists():
            entry.update(measure(out))
            entry["proportion_error"] = score(entry, args.target_lh, args.target_lb)
        else:
            # A failed seed is data, not an interruption. On TU116 the
            # sparse-structure stage is nondeterministic and the card has
            # shape-dependent kernel faults -- "misaligned address" and
            # tensor-op launch failures -- that no flag avoids, so each seed is
            # an independent draw and the survival rate is a measurement.
            #
            # Read the CLI log, not the subprocess pipes: trellis_run.py exits
            # non-zero but still prints its own JSON receipt on stdout, so
            # capturing stdout here records a latent-size table where the actual
            # fault line should be.
            log = args.out_dir / f"seed{seed}.log"
            tail = log.read_text(encoding="utf-8", errors="replace")[-600:] \
                if log.exists() else (completed.stderr or "")[-600:]
            entry["error"] = tail.strip().splitlines()[-6:]
        entries.append(entry)
        print(json.dumps(entry), flush=True)

    finished = [e for e in entries if "proportion_error" in e]
    finished.sort(key=lambda e: e["proportion_error"])
    result = {
        "schema": "lowvram3d_trellis_seed_tournament_v1",
        "image": str(args.image),
        "resolution": args.res,
        "target_length_height": args.target_lh,
        "target_length_beam": args.target_lb,
        "seeds_run": len(entries),
        "seeds_completed": len(finished),
        "ranked": finished,
        "failed": [e for e in entries if "proportion_error" not in e],
    }
    (args.out_dir / "tournament.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "ranked"}, indent=2))
    for entry in finished[:3]:
        print("  seed %-6d err %.4f  L:H %.2f  L:B %.2f  %d faces" % (
            entry["seed"], entry["proportion_error"], entry["length_height"],
            entry["length_beam"], entry["faces"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
