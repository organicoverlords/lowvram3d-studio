"""ROUTE_B feasibility probe: can a multiview geometry route run on this machine at all?

Written as a probe rather than a pipeline because the answer turned out to be no, and the useful
artifact is the evidence for that rather than a half-route that produces images nothing can consume.

Route B needs two things: a generator that turns one picture into consistent novel views, and a
shape backend that consumes several views. This checks for both and reports exactly which files are
missing, so the conclusion can be re-tested automatically once anything changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MVADAPTER_VARIANTS = {
    "i2mv_sd21": "mvadapter_i2mv_sd21.safetensors",
    "i2mv_sdxl": "mvadapter_i2mv_sdxl.safetensors",
    "ig2mv_sd21": "mvadapter_ig2mv_sd21.safetensors",
    "tg2mv_sd21": "mvadapter_tg2mv_sd21.safetensors",
}


def probe_view_generator(search_roots: list[Path], mvadapter_repo: Path) -> dict:
    found: dict[str, dict] = {}
    for name, filename in MVADAPTER_VARIANTS.items():
        hits = []
        for root in search_roots:
            if not root.exists():
                continue
            hits += [p for p in root.rglob(filename) if p.is_file() and p.stat().st_size > 1 << 20]
        found[name] = {
            "present": bool(hits),
            "path": str(hits[0]) if hits else None,
            "bytes": hits[0].stat().st_size if hits else 0,
        }

    package_ok, package_detail = False, None
    if mvadapter_repo.exists():
        sys.path.insert(0, str(mvadapter_repo))
        try:
            import mvadapter  # noqa: F401
            package_ok = True
            package_detail = str(mvadapter_repo)
        except Exception as error:  # noqa: BLE001
            package_detail = f"{type(error).__name__}: {error}"[:300]

    return {
        "mvadapter_package_importable": package_ok,
        "mvadapter_package": package_detail,
        "weights": found,
        # Image-only conditioning is the only variant that can propose a *different* shape. The
        # geometry-conditioned variants render views consistent with a mesh you already have, so
        # feeding their output back into a shape model reproduces the shape it started from - which
        # is precisely the head this route exists to replace.
        "image_only_conditioning_available": found["i2mv_sd21"]["present"] or found["i2mv_sdxl"]["present"],
    }


def probe_shape_backend(hunyuan_repo: Path, search_roots: list[Path]) -> dict:
    multiview_classes: list[str] = []
    single_image_only = None
    if hunyuan_repo.exists():
        sys.path.insert(0, str(hunyuan_repo))
        try:
            from hy3dgen import shapegen

            exported = [n for n in dir(shapegen) if "Pipeline" in n]
            multiview_classes = [n for n in exported if "MV" in n.upper() or "Multi" in n]
            import inspect

            pipeline = getattr(shapegen, "Hunyuan3DDiTFlowMatchingPipeline", None)
            if pipeline is not None:
                params = list(inspect.signature(pipeline.__call__).parameters)
                single_image_only = "image" in params and not any(
                    p in params for p in ("images", "mv_image", "multiview")
                )
        except Exception as error:  # noqa: BLE001
            multiview_classes = [f"import failed: {type(error).__name__}: {error}"[:200]]

    checkpoints = []
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.rglob("hunyuan3d-dit-*"):
            if not candidate.is_dir():
                continue
            weights = [p for p in candidate.glob("*.safetensors") if p.stat().st_size > 1 << 20]
            checkpoints.append({
                "name": candidate.name,
                "path": str(candidate),
                "has_weights": bool(weights),
                "weight_bytes": weights[0].stat().st_size if weights else 0,
            })

    return {
        "multiview_pipeline_classes": multiview_classes,
        "single_image_signature_only": single_image_only,
        "shape_checkpoints": checkpoints,
        "multiview_checkpoint_present": any(
            "mv" in c["name"].split("-")[-1] and c["has_weights"] for c in checkpoints),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", action="append", required=True)
    parser.add_argument("--mvadapter-repo", required=True)
    parser.add_argument("--hunyuan-repo", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    roots = [Path(r) for r in args.search_root]
    views = probe_view_generator(roots, Path(args.mvadapter_repo))
    shape = probe_shape_backend(Path(args.hunyuan_repo), roots)

    blockers = []
    if not views["image_only_conditioning_available"]:
        blockers.append("NO_IMAGE_ONLY_MULTIVIEW_WEIGHTS")
    if not shape["multiview_checkpoint_present"] and not shape["multiview_pipeline_classes"]:
        blockers.append("NO_MULTIVIEW_SHAPE_BACKEND")

    report = {
        "route": "ROUTE_B_MULTIVIEW",
        "view_generator": views,
        "shape_backend": shape,
        "blockers": blockers,
        "runnable": not blockers,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ROUTE_B_PROBE runnable={report['runnable']} blockers={blockers}", flush=True)
    raise SystemExit(0 if report["runnable"] else 2)


if __name__ == "__main__":
    main()
