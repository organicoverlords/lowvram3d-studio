"""Complete Pipeline V2 entrypoint.

Keeps the proven core implementation intact, merges in production stage adapters, and treats a
missing stage as a software defect instead of silently printing `not_implemented` and exiting zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run_asset_pipeline import Pipeline, STAGES, build_manifest
from pipeline_v2_stages import register_stages as register_core_stages
from pipeline_v2_production_stages import register_production_stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete Pipeline V2 asset runner")
    parser.add_argument("--image", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--asset-id", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--blender", default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
    parser.add_argument("--from-stage", default="INGEST")
    parser.add_argument("--to-stage", default="EXPORT")
    parser.add_argument("--existing-master", default="", help="skip GENERATE and adopt this GLB")
    parser.add_argument("--write-manifest-only", action="store_true")
    args = parser.parse_args()

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    else:
        if not args.image:
            raise SystemExit("--image or --manifest is required")
        image = Path(args.image)
        output_root = Path(args.output_root or (Path(__file__).resolve().parents[1] / "pipeline-v2-runs"))
        manifest = build_manifest(image, args.profile, output_root, args.asset_id or None)

    root = Path(manifest["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"MANIFEST {manifest_path}", flush=True)
    print(f"ASSET {manifest['asset_id']} profile={manifest['profile']}", flush=True)
    if args.write_manifest_only:
        return

    pipeline = Pipeline(manifest, root, args.python, args.blender)
    stages = register_core_stages(pipeline, manifest, existing_master=args.existing_master)
    stages.update(register_production_stages(pipeline, manifest))

    start_name = args.from_stage.upper()
    stop_name = args.to_stage.upper()
    if start_name not in STAGES or stop_name not in STAGES:
        raise SystemExit(f"unknown stage range {start_name}..{stop_name}")
    start, stop = STAGES.index(start_name), STAGES.index(stop_name)
    if stop < start:
        raise SystemExit(f"to-stage {stop_name} precedes from-stage {start_name}")

    summary = {}
    exit_code = 0
    for stage in STAGES[start: stop + 1]:
        runner = stages.get(stage)
        if runner is None:
            summary[stage] = {
                "status": "failed",
                "failure_codes": ["STAGE_NOT_IMPLEMENTED"],
                "needs_human": False,
            }
            print(f"[{stage}] SOFTWARE DEFECT: stage not implemented", flush=True)
            exit_code = 2
            break
        receipt = runner()
        summary[stage] = {
            "status": receipt["status"],
            "failure_codes": receipt.get("failure_codes", []),
            "needs_human": bool(receipt.get("needs_human", False)),
        }
        if receipt["status"] != "passed":
            exit_code = 1
            break

    payload = {"asset_id": manifest["asset_id"], "profile": manifest["profile"], "stages": summary}
    (root / "pipeline_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed = [name for name, value in summary.items() if value["status"] != "passed"]
    print(
        f"PIPELINE_RESULT asset={manifest['asset_id']} "
        f"{'FAILED at ' + failed[0] if failed else 'COMPLETED'} stages={json.dumps(summary)}",
        flush=True,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
