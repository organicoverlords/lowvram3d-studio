"""Generate several Mini Turbo candidates and keep the one that best matches the source.

The pipeline used to accept whatever the generator returned on its first attempt. Nothing downstream
ever asked whether the mesh resembled the picture, so a head reconstructed as a smooth rounded mass
where the source has a narrow bird skull passed every gate and was only caught by a human looking at
a texture render eight stages later. Generation from a single image is stochastic; the fix is to
draw a few samples and let the source silhouette decide, not to hope the first one is good.

The generator is unchanged - still Mini Turbo, still the same weights. Only the seed and the
sampling settings vary between candidates, and the selection is made by an explicit recorded score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# A selection is only meaningful if the winner beats the field by more than the scorer's own noise.
# Measured on this asset, changing the seed moves the score by about 0.005 while the distance to a
# passing score is 0.22: picking the maximum of that field is picking noise and recording it as a
# decision, which looks exactly like a working automated pipeline.
MIN_MEANINGFUL_SPREAD = 0.02


def run(command: list[str]) -> tuple[int, str]:
    process = subprocess.run([str(part) for part in command], capture_output=True, text=True)
    return process.returncode, (process.stdout or "") + (process.stderr or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--conditioning-image", default="")
    parser.add_argument("--matte", required=True, help="matted source used to score silhouettes")
    parser.add_argument("--output", required=True, help="winning mesh")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--seeds", default="12345,777,20260801")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--octree-ladder", default="384:3000,320:2000,256:1500")
    parser.add_argument("--front-direction", default="+z")
    args = parser.parse_args()

    workers = Path(args.workers_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()][:args.candidates]

    # Scores are only comparable when they came from the same scorer. Editing the scorer while a
    # sweep was running once produced a receipt whose candidates had been measured by two different
    # versions, and the resulting "winner" and its convincing 0.085 spread were both artifacts of
    # the edit rather than differences between the meshes.
    scorer = workers / "pipeline_source_fidelity_qa.py"
    scorer_version = hashlib.sha256(scorer.read_bytes()).hexdigest()[:16]

    attempts = []
    for index, seed in enumerate(seeds):
        mesh = work / f"candidate_{index}_seed{seed}.glb"
        result = work / f"candidate_{index}_seed{seed}_generate.json"
        command = [
            args.python, workers / "mini_turbo_generate.py",
            "--image", args.image, "--output", mesh, "--result-json", result,
            "--model-root", args.model_root, "--seed", str(seed),
            "--steps", str(args.steps), "--octree-ladder", args.octree_ladder,
        ]
        if args.conditioning_image:
            command += ["--conditioning-image", args.conditioning_image]
        code, out = run(command)
        record = {"seed": seed, "mesh": str(mesh), "generate_exit": code}
        if code != 0 or not mesh.exists():
            record.update({"scored": False, "detail": out[-600:]})
            attempts.append(record)
            continue

        fidelity = work / f"candidate_{index}_seed{seed}_fidelity.json"
        overlay = work / f"candidate_{index}_seed{seed}_overlay.png"
        code, out = run([
            args.python, workers / "pipeline_source_fidelity_qa.py",
            "--mesh", mesh, "--source", args.matte, "--report", fidelity,
            "--debug-image", overlay, "--front-direction", args.front_direction,
        ])
        # Exit 2 means the gate ran and failed, which is still a usable score. Only a missing report
        # means the measurement itself did not happen, and an unmeasured candidate cannot win.
        if not fidelity.exists():
            record.update({"scored": False, "detail": out[-600:]})
            attempts.append(record)
            continue
        data = json.loads(fidelity.read_text(encoding="utf-8"))
        record.update({
            "scored": True,
            "scorer_version": scorer_version,
            "score": data.get("score", 0.0),
            "silhouette_iou": data.get("silhouette_iou", {}),
            "failure_codes": data.get("failure_codes", []),
            "fidelity_report": str(fidelity),
            "overlay": str(overlay),
        })
        attempts.append(record)
        print(f"CANDIDATE seed={seed} score={record['score']:.4f} "
              f"iou={record['silhouette_iou']}", flush=True)

    scored = [a for a in attempts if a.get("scored")]
    if not scored:
        report = {"attempts": attempts, "selected": None, "passed": False,
                  "detail": "no candidate could be generated and scored"}
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("GENERATE_BEST_OF failed: no scored candidate", flush=True)
        raise SystemExit(2)

    versions = {a["scorer_version"] for a in scored}
    if len(versions) > 1:
        raise SystemExit(f"scores came from {len(versions)} scorer versions and are not comparable")

    winner = max(scored, key=lambda a: a["score"])
    spread = round(max(a["score"] for a in scored) - min(a["score"] for a in scored), 5)
    # Report indistinguishability rather than dressing it up as a choice. When every candidate lands
    # inside the noise, the honest output is that this lever does not work on this asset, so the
    # caller escalates to one that might instead of believing the problem is solved.
    meaningful = spread >= MIN_MEANINGFUL_SPREAD

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_bytes(Path(winner["mesh"]).read_bytes())

    report = {
        "candidates_requested": args.candidates,
        "candidates_scored": len(scored),
        "scorer_version": scorer_version,
        "selection_basis": "0.6 * overall silhouette IoU + 0.4 * upper-band IoU",
        "attempts": attempts,
        "selected": winner,
        "output": args.output,
        "score_spread": spread,
        "minimum_meaningful_spread": MIN_MEANINGFUL_SPREAD,
        "selection_is_meaningful": meaningful,
        "selection_note": None if meaningful else
            "candidates are indistinguishable within scorer noise; this lever does not "
            "discriminate on this asset and the winner is arbitrary",
        "passed": True,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"GENERATE_BEST_OF selected seed={winner['seed']} score={winner['score']:.4f} "
          f"from {len(scored)} candidates (spread {spread:.4f}, "
          f"meaningful={meaningful})", flush=True)


if __name__ == "__main__":
    main()
