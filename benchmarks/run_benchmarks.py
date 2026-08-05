"""Pipeline V2 benchmark pack.

Two kinds of case:

* **Regression cases** replay artifacts that the shaman run actually produced - including the ones
  that were technically valid and visibly wrong - and assert the evaluator now reaches the verdict
  a human had to reach by hand. Each case names the exact failure codes it must raise, so a change
  that stops detecting a defect fails here rather than in a future asset.
* **Geometry cases** run the generic geometry gate over real masters (bird, red panda, shaman) and
  assert it does not regress on assets that are known good.

A benchmark that only asserted "passes" would be useless: the point is that the known-bad inputs
are still rejected, for the stated reason.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(r"C:\AI\LowVRAM3D-benchmarks")
SHAMAN = BENCH / "outputs" / "antlered_bird_shaman_anchor" / "final-pipeline"
PROJECTION = SHAMAN / "texture" / "projection"


def evaluator_case(name, expect_pass, expect_codes, arguments, out_dir, python):
    report = out_dir / f"{name}.json"
    command = [python, str(REPO_ROOT / "workers" / "visual_evaluator.py"),
               "--report", str(report)] + [str(a) for a in arguments]
    process = subprocess.run(command, capture_output=True, text=True,
                             env=_env(), cwd=str(REPO_ROOT))
    data = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    got_codes = set(data.get("blocking_codes", []))
    passed = bool(data.get("passed"))
    ok = (passed == expect_pass) and set(expect_codes) <= got_codes
    return {"case": name, "kind": "evaluator", "ok": ok,
            "expected_pass": expect_pass, "actual_pass": passed,
            "expected_codes": sorted(expect_codes), "actual_codes": sorted(got_codes),
            "stdout": (process.stdout or "").strip().splitlines()[-1:] or [""]}


def geometry_case(name, mesh, expect_pass, out_dir, python, max_axis_ratio=8.0):
    report = out_dir / f"{name}.json"
    process = subprocess.run(
        [python, str(REPO_ROOT / "workers" / "pipeline_geometry_qa.py"),
         "--mesh", str(mesh), "--report", str(report),
         "--max-axis-ratio", str(max_axis_ratio)],
        capture_output=True, text=True, env=_env(), cwd=str(REPO_ROOT))
    data = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    passed = bool(data.get("passed"))
    return {"case": name, "kind": "geometry", "ok": passed == expect_pass,
            "expected_pass": expect_pass, "actual_pass": passed,
            "actual_codes": data.get("failure_codes", []),
            "axis_ratio": data.get("axis_ratio"),
            "components": data.get("components"),
            "stdout": (process.stdout or "").strip().splitlines()[-1:] or [""]}


def _env():
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "blender"), str(REPO_ROOT / "src"), str(REPO_ROOT / "workers"), str(REPO_ROOT)])
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmarks" / "results"))
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    renders = SHAMAN / "reports" / "texture_review"
    swapped = SHAMAN / "reports" / "texture_review_swapped"
    truth = SHAMAN / "reports" / "orientation_truth.json"
    source = BENCH / "outputs" / "antlered_bird_shaman_anchor" / "mini-turbo-iterations" / "20260801-013528" / "shaman_matte.png"

    results = []

    # --- regression: defects the shaman run shipped past a green build ---------------------
    results.append(evaluator_case(
        "shaman_atlas_inverted_rows", False, {"UV_ROW_ORIENTATION_MISMATCH"},
        ["--render-dir", renders, "--basecolor", PROJECTION / "atlas" / "basecolor.png",
         "--coverage", PROJECTION / "atlas" / "debug_coverage.png", "--orientation-truth", truth],
        out_dir, args.python))

    results.append(evaluator_case(
        "shaman_flat_grey_priors", False, {"FLAT_NEUTRAL_ATLAS_REGIONS"},
        ["--render-dir", renders,
         "--basecolor", PROJECTION / "atlas" / "basecolor_gltf_preretouch.png",
         "--coverage", PROJECTION / "atlas" / "debug_coverage_gltf.png",
         "--orientation-truth", truth,
         "--region-report", SHAMAN / "reports" / "region_all.json"],
        out_dir, args.python))

    results.append(evaluator_case(
        "shaman_swapped_camera_labels", False, {"CAMERA_LABEL_MISMATCH"},
        ["--render-dir", swapped, "--basecolor", SHAMAN / "textures" / "shaman_basecolor_4k.png",
         "--coverage", PROJECTION / "atlas" / "debug_coverage_gltf.png",
         "--orientation-truth", truth, "--source-image", source,
         "--region-report", SHAMAN / "reports" / "region_all_after.json"],
        out_dir, args.python))

    results.append(evaluator_case(
        "shaman_material_id_per_triangle_noise", False, {"MATERIAL_ID_NOISE"},
        ["--render-dir", renders, "--material-id-components", "27097"],
        out_dir, args.python))

    results.append(evaluator_case(
        "shaman_uv_detector_timed_out", False, {"UV_OVERLAP"},
        ["--render-dir", renders, "--uv-report", REPO_ROOT / "benchmarks" / "fixtures" / "uv_timed_out.json"],
        out_dir, args.python))

    results.append(evaluator_case(
        "shaman_plastic_roughness", False, {"PLASTIC_ROUGHNESS"},
        ["--render-dir", renders, "--orm", REPO_ROOT / "benchmarks" / "fixtures" / "orm_plastic.png"],
        out_dir, args.python))

    # --- the accepted candidate must still pass -------------------------------------------
    results.append(evaluator_case(
        "shaman_accepted_candidate", True, set(),
        ["--render-dir", renders, "--basecolor", SHAMAN / "textures" / "shaman_basecolor_4k.png",
         "--orm", SHAMAN / "textures" / "shaman_orm_4k.png",
         "--coverage", PROJECTION / "atlas" / "debug_coverage_gltf.png",
         "--orientation-truth", truth, "--source-image", source,
         "--region-report", SHAMAN / "reports" / "region_all_after.json",
         "--material-id", SHAMAN / "textures" / "shaman_material_id_4k.png",
         "--material-id-components", "62",
         "--uv-report", SHAMAN / "reports" / "uv_quality_exact.json"],
        out_dir, args.python))

    # --- geometry gate over real masters ---------------------------------------------------
    for name, mesh, expect in (
        ("geometry_shaman_master", BENCH / "outputs" / "antlered_bird_shaman_anchor" /
         "geometry-latest" / "shaman_geometry_master.glb", None),
        ("geometry_turbo_bird_master", BENCH / "masters" / "turbo_bird_master.glb", None),
        ("geometry_red_panda_master", BENCH / "masters" / "red_panda_master.glb", None),
        ("geometry_shaman_lod0_cleaned", SHAMAN / "textured" / "shaman_lod0_uv_clean.glb", True),
    ):
        if not Path(mesh).exists():
            results.append({"case": name, "kind": "geometry", "ok": None, "skipped": f"missing {mesh}"})
            continue
        # expect None means "record the outcome, do not assert" - these masters are raw generator
        # output and are expected to carry debris; the assertion is on the cleaned mesh.
        outcome = geometry_case(name, mesh, expect if expect is not None else True, out_dir, args.python)
        if expect is None:
            outcome["ok"] = None
            outcome["note"] = "recorded, not asserted: raw generator output"
        results.append(outcome)

    asserted = [r for r in results if r.get("ok") is not None]
    failed = [r for r in asserted if not r["ok"]]
    summary = {
        "total_cases": len(results),
        "asserted": len(asserted),
        "passed": len(asserted) - len(failed),
        "failed": [r["case"] for r in failed],
        "results": results,
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for record in results:
        flag = {True: "PASS", False: "FAIL", None: "note"}[record.get("ok")]
        print(f"  [{flag}] {record['case']}: expected_pass={record.get('expected_pass')} "
              f"actual_pass={record.get('actual_pass')} codes={record.get('actual_codes')}", flush=True)
    print(f"BENCHMARK {summary['passed']}/{summary['asserted']} asserted cases passed; "
          f"failed={summary['failed']}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
