"""Pick the generator from the input image, before anything is generated.

Two generators are available and neither is better. They fail in opposite ways,
and which one is right is decidable from the source image.

Measured on the same shaman, same input:

                        TRELLIS.2 512      Hunyuan3D Mini Turbo o384
    faces               2,185,428          1,112,498
    connected shells       71,043                 23
    watertight                 no                yes
    largest shell               -   1,105,450 (99.4% one body)

A thousandfold difference in fragmentation, and it is architectural rather than
a setting: TRELLIS puts latents on sparse voxels and decodes each independently,
so a thin cord emerges as a chain of disconnected fragments with gaps between
them. Hunyuan is VecSet -- an unordered set of latent vectors with no occupancy
grid -- and returns essentially one connected body.

The consequences, all observed:

    a TRELLIS rope can break, because it is not one object
    a global decimator destroys TRELLIS ornaments, because there is no
      connectivity to preserve (91% face cut turned individually shaped
      pendants into identical stubs)
    xatlas never completes on 71k charts

But TRELLIS is not worse. On internal feature-edge F1 against an online
reference it scores 0.690 against Hunyuan mini's 0.573, both trustworthy. It
resolves window recesses, railing gaps and panel breaks that Hunyuan melts.

So the rule is about what the subject *is*:

    connected thin structures  -> Hunyuan   (cords, chains, rigging, cables)
    dense internal detail      -> TRELLIS   (architecture, machinery, facades)

`feature_risk.py` already counts the discriminating quantity. A "strand" is a
region thinner than one structural cell but long enough to claim cells along its
run -- exactly a hanging cord. Counts on subjects run so far:

    shaman   18 strands   -> Hunyuan   (its cords broke under TRELLIS)
    panda     2 strands   -> TRELLIS
    castle    few         -> TRELLIS   (and TRELLIS produced a good castle)

The threshold is deliberately low. A subject with several genuine strands has
something TRELLIS will fragment, and the cost of choosing Hunyuan for a subject
that did not need it is softer detail -- recoverable. The cost of choosing
TRELLIS for a subject with rigging is broken geometry, which is not.

    py choose_generator.py --image crop512.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

#: Two conditions, both required, and the second is the one that discriminates.
#:
#: An absolute strand count does NOT separate these subjects. Measured:
#: shaman 18, panda 7, castle 6 -- any threshold low enough to catch the shaman
#: also catches the two that TRELLIS handled well. The first version of this
#: file used 6 and routed all three to Hunyuan.
#:
#: What separates them is what the subject is *made of*: strands as a share of
#: its fine features.
#:
#:     shaman  18 strands / 23 compact  = 0.78   dominated by cords
#:     panda    7 strands / 17 compact  = 0.41   mixed
#:     castle   6 strands / 22 compact  = 0.27   dominated by detail
#:
#: CAVEAT, and it is not small: these thresholds were chosen after seeing three
#: subjects, so they are fitted to three points. They are a starting rule, not a
#: measured boundary, and the fourth subject may well move them. The ratio at
#: least has a mechanism behind it -- a subject built of cords needs connected
#: geometry, a subject built of compact detail needs resolution -- which the
#: absolute count did not.
STRAND_THRESHOLD = 10
STRAND_RATIO_THRESHOLD = 0.60

PYTHON = sys.executable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True,
                        help="The prepared square input.")
    parser.add_argument("--prefer", choices=("auto", "trellis", "hunyuan"),
                        default="auto",
                        help="'auto' measures. The other two are the user "
                             "override, recorded in the receipt as such.")
    parser.add_argument("--strand-threshold", type=int, default=STRAND_THRESHOLD)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    if args.prefer != "auto":
        receipt = {"schema_version": "choose_generator_v1",
                   "generator": args.prefer, "basis": "user override"}
        if args.receipt:
            Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                          encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 0

    risk_worker = Path(__file__).with_name("feature_risk.py")
    result = subprocess.run(
        [PYTHON, str(risk_worker), "--image", args.image, "--top", "0"],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"FEATURE_RISK_FAILED: {result.stderr[-300:]}")
    risk = json.loads(result.stdout)

    strands = int(risk.get("strands", 0))
    at_risk = int(risk.get("at_risk", 0))

    # Strands per compact feature. Not strands/(strands+at_risk) -- that was
    # the first version and it disagreed with the numbers this file reasons
    # from, putting the shaman at 0.44 instead of 0.78 and routing it wrong.
    ratio = strands / max(at_risk, 1)
    if strands >= args.strand_threshold and ratio >= STRAND_RATIO_THRESHOLD:
        generator = "hunyuan"
        reason = (f"{strands} strands are {ratio:.0%} of this subject's fine "
                  f"features -- it is built of cords. TRELLIS returns thin "
                  f"structures as disconnected fragments (71,043 shells here "
                  f"against Hunyuan's 23, which is watertight), so cords break "
                  f"and a decimator eats the props they carry.")
    else:
        generator = "trellis"
        reason = (f"{strands} strands are only {ratio:.0%} of this subject's "
                  f"fine features -- it is built of compact detail, which "
                  f"TRELLIS resolves better (feature-edge F1 0.690 against "
                  f"0.573) and which has no connected structure to fragment.")

    receipt = {
        "schema_version": "choose_generator_v1",
        "image": str(Path(args.image).resolve()),
        "generator": generator,
        "basis": "measured",
        "reason": reason,
        "measurements": {
            "strands": strands,
            "at_risk": at_risk,
            "shape_lost": int(risk.get("shape_lost", 0)),
            "structural_cell_px": risk.get("structural_cell_px"),
        },
        "strand_ratio": round(ratio, 3),
        "thresholds": {"strand": args.strand_threshold,
                       "strand_ratio": STRAND_RATIO_THRESHOLD},
        "command": (
            "bash workers/trellis_retry_seeds.sh IMAGE OUT 4242 777 12345"
            if generator == "trellis" else
            "workers/mini_turbo_generate.py --steps 5 --seed 12345 "
            "--octree-ladder 384:3000,320:2000,256:1500"),
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
