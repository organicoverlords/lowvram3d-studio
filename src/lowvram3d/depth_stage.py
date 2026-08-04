"""Run the MoGe depth reconstruction as a bounded, optional pipeline stage.

MoGe needs its own interpreter (CUDA torch plus the `moge` package), which is
almost never the interpreter the pipeline itself runs under. Rather than force
one environment to satisfy both, this shells out to the configured environment
and degrades cleanly: when the environment is missing or the run fails, the
caller keeps the existing CPU bootstrap instead of aborting the pipeline.

That distinction is deliberate. A pipeline that silently produced a flat
`source_projection` shell -- an image mapped onto geometry that reproduces the
source view perfectly and carries no depth -- is exactly what made earlier runs
look successful while proving nothing. Callers get `available: False` and a
reason, so the shortfall is visible in the receipt.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_ENV = Path(
    r"C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\envs\image-world-moge\Scripts\python.exe")


def resolve_interpreter(explicit: str | Path | None = None) -> Path | None:
    """Pick the interpreter that can import `moge`, or None."""
    candidates = [explicit, os.environ.get("LOWVRAM3D_MOGE_PYTHON"), DEFAULT_ENV]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def reconstruct_depth(image: Path, work_dir: Path, scene_id: str,
                      max_triangles: int = 1_500_000, edge_rtol: float = 0.05,
                      interpreter: str | Path | None = None,
                      timeout: float = 1800.0) -> dict[str, Any]:
    """Produce a textured depth mesh for `image`, or explain why it could not."""
    python = resolve_interpreter(interpreter)
    if python is None:
        return {"available": False,
                "reason": "no MoGe interpreter found; set LOWVRAM3D_MOGE_PYTHON",
                "searched": str(DEFAULT_ENV)}

    work_dir.mkdir(parents=True, exist_ok=True)
    glb = work_dir / f"{scene_id}_moge.glb"
    receipt_path = work_dir / f"{scene_id}_moge_receipt.json"

    command = [
        str(python), "-m", "lowvram3d.moge_reconstruct",
        "--image", str(image),
        "--output", str(glb),
        "--receipt", str(receipt_path),
        "--max-triangles", str(int(max_triangles)),
        "--edge-rtol", str(float(edge_rtol)),
    ]
    env = dict(os.environ)
    # The MoGe environment does not have this package installed; run from source.
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": f"MoGe timed out after {timeout}s"}
    except OSError as exc:
        return {"available": False, "reason": f"could not launch MoGe: {exc}"}

    if completed.returncode != 0 or not receipt_path.is_file():
        return {"available": False,
                "reason": f"MoGe exited {completed.returncode}",
                "stderr": (completed.stderr or "")[-2000:]}

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["available"] = True
    receipt["interpreter"] = str(python)
    return receipt


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--max-triangles", type=int, default=1_500_000)
    args = parser.parse_args(argv)

    receipt = reconstruct_depth(Path(args.image), Path(args.work_dir),
                                args.scene_id, args.max_triangles)
    json.dump(receipt, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0 if receipt.get("available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
