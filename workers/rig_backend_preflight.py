"""CPU-only availability preflight for external rigging backends.

The preflight intentionally does not import torch, initialize CUDA, download
weights or run inference.  It is safe to execute while another GPU job is
running.  Its output is a machine-readable input to a later rig execution
worker; absence of a backend is reported rather than silently changing routes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from lowvram3d.rigging_policy import build_rigging_plan, pipeline_stage_order


MIA_WEIGHT_FILES = (
    "bw.pth",
    "bw_normal.pth",
    "joints.pth",
    "joints_coarse.pth",
    "pose.pth",
)


def _first_existing(root: Path, relative_paths: Iterable[str]) -> Path | None:
    for relative in relative_paths:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def probe_mia(root: Path | None) -> dict:
    if root is None:
        return {"available": False, "reason": "mia_root_not_configured"}
    root = root.expanduser().resolve()
    weight_roots = (root, root / "output" / "best" / "new", root / "models" / "mia")
    found: dict[str, str] = {}
    missing: list[str] = []
    for filename in MIA_WEIGHT_FILES:
        hit = next((base / filename for base in weight_roots if (base / filename).is_file()), None)
        if hit is None:
            missing.append(filename)
        else:
            found[filename] = str(hit)
    source = _first_existing(root, ("app.py", "nodes/mia_auto_rig.py", "nodes/mia_inference.py"))
    return {
        "available": source is not None and not missing,
        "root": str(root),
        "source_marker": str(source) if source else None,
        "weights": found,
        "missing_weights": missing,
        "reason": None if source is not None and not missing else "missing_source_or_weights",
    }


def probe_puppeteer(root: Path | None) -> dict:
    if root is None:
        return {"available": False, "reason": "puppeteer_root_not_configured"}
    root = root.expanduser().resolve()
    skeleton = root / "skeleton"
    skinning = root / "skinning"
    required = {
        "skeleton": skeleton / "README.md",
        "skinning": skinning / "README.md",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    return {
        "available": not missing,
        "root": str(root),
        "missing_components": missing,
        "attention_required": "sdpa",
        "note": "Runtime adapter must override the upstream FlashAttention-2 assumption on sm75.",
        "reason": None if not missing else "missing_puppeteer_components",
    }


def probe_unirig(root: Path | None) -> dict:
    if root is None:
        return {"available": False, "reason": "unirig_root_not_configured"}
    root = root.expanduser().resolve()
    markers = (
        root / "nodes" / "auto_rig.py",
        root / "nodes" / "load_model.py",
        root / "nodes" / "skinning.py",
    )
    missing = [path.name for path in markers if not path.is_file()]
    return {
        "available": not missing,
        "root": str(root),
        "missing_components": missing,
        "reason": None if not missing else "missing_unirig_components",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-type", required=True)
    parser.add_argument("--rig-kind", default="auto")
    parser.add_argument("--vram-ceiling-mb", type=int, default=5600)
    parser.add_argument("--preferred-backend", default="")
    parser.add_argument("--mia-root", type=Path)
    parser.add_argument("--puppeteer-root", type=Path)
    parser.add_argument("--unirig-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = build_rigging_plan(
        args.asset_type,
        rig_kind=args.rig_kind,
        vram_ceiling_mb=args.vram_ceiling_mb,
        preferred_backend=args.preferred_backend or None,
    )
    probes = {
        "mia": probe_mia(args.mia_root),
        "puppeteer": probe_puppeteer(args.puppeteer_root),
        "unirig": probe_unirig(args.unirig_root),
        "legacy_rigid": {"available": True, "reason": None},
        "none": {"available": True, "reason": None},
    }
    selected_probe = probes[plan.backend]
    result = {
        "status": "READY" if selected_probe["available"] else "BLOCKED_BACKEND_UNAVAILABLE",
        "gpu_work_started": False,
        "plan": plan.to_dict(),
        "stage_order": list(pipeline_stage_order(plan)),
        "selected_backend_probe": selected_probe,
        "backend_probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if selected_probe["available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
