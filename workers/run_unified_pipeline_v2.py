"""Run the canonical static Pipeline V2 stage range."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run_asset_pipeline import Pipeline, build_manifest
from unified_pipeline_v2 import CANONICAL_STAGES, normalize_manifest, register_unified_stages, _seed_existing_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--asset-id", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--blender", default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
    parser.add_argument("--from-stage", default="INGEST")
    parser.add_argument("--to-stage", default="EXPORT_QA")
    parser.add_argument("--existing-master", default="")
    parser.add_argument("--existing-clean", default="",
                        help="adopt an already-proven CLEAN geometry boundary")
    parser.add_argument("--existing-matte", default="",
                        help="adopt an already-proven INGEST matte boundary")
    parser.add_argument("--existing-uv", default="",
                        help="adopt and validate an existing UV mesh without unwrapping")
    parser.add_argument("--existing-textured", default="",
                        help="adopt an already-proven textured GLB for EXPORT_QA only")
    parser.add_argument("--write-manifest-only", action="store_true")
    args = parser.parse_args()

    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    else:
        if not args.image:
            raise SystemExit("--image or --manifest is required")
        output_root = Path(args.output_root or (Path(__file__).resolve().parents[1] / "pipeline-v2-runs"))
        manifest = build_manifest(Path(args.image), args.profile, output_root, args.asset_id or None)
    manifest = normalize_manifest(
        manifest,
        stage_from=args.from_stage,
        stage_to=args.to_stage,
        existing_master=args.existing_master,
    )
    if args.existing_uv:
        manifest["uv_mesh"] = str(Path(args.existing_uv))

    root = Path(manifest["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.write_manifest_only:
        return 0

    pipeline = Pipeline(manifest, root, args.python, args.blender)
    start, stop = args.from_stage.upper(), args.to_stage.upper()
    stages = register_unified_stages(pipeline, manifest, existing_master=args.existing_master)
    if args.existing_clean:
        clean_source = Path(args.existing_clean)
        _seed_existing_receipt(pipeline, "GENERATE", "master", clean_source, [clean_source])
        _seed_existing_receipt(pipeline, "CLEAN", "clean", clean_source, [clean_source])
    elif args.existing_master and start == "CLEAN":
        _seed_existing_receipt(pipeline, "GENERATE", "master", Path(args.existing_master), [Path(args.existing_master)])
    elif args.existing_master and start == "UV":
        _seed_existing_receipt(pipeline, "LOD", "lod0", Path(args.existing_master), [Path(args.existing_master)])
    if args.existing_textured:
        _seed_existing_receipt(
            pipeline, "TEXTURE", "textured_glb", Path(args.existing_textured),
            [Path(args.existing_textured)],
        )
    if args.existing_uv:
        uv_source = Path(args.existing_uv)
        if not uv_source.is_file() or uv_source.stat().st_size == 0:
            raise SystemExit(f"EXISTING_UV_INVALID: {uv_source}")
        _seed_existing_receipt(pipeline, "UV", "uv_mesh", uv_source, [uv_source])
    if args.existing_matte:
        matte_source = Path(args.existing_matte)
        _seed_existing_receipt(pipeline, "INGEST", "matte", matte_source, [matte_source])

    summary = {}
    exit_code = 0
    for stage in CANONICAL_STAGES[CANONICAL_STAGES.index(start):CANONICAL_STAGES.index(stop) + 1]:
        receipt = stages[stage]()
        summary[stage] = {
            "status": receipt["status"],
            "failure_codes": receipt.get("failure_codes", []),
            "needs_human": bool(receipt.get("needs_human")),
        }
        if receipt["status"] != "passed":
            exit_code = 1
            break
    payload = {
        "asset_id": manifest["asset_id"],
        "pipeline": manifest["canonical_pipeline"],
        "stage_order": list(CANONICAL_STAGES),
        "stages": summary,
    }
    (root / "pipeline_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
