"""Run the high-resolution-master geometry phase with bounded descending candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from lowvram3d.benchmark_suite import fixture_by_id
from lowvram3d.presets import get_profile
from lowvram3d.quality_ladder import candidate_ladder, family_for_asset_type


STATUS_INTERVAL_SECONDS = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tail(path: Path, lines: int = 30) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def run_process(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    logs_dir: Path,
    timeout: int,
    accepted_exit_codes: set[int] | None = None,
) -> int:
    accepted = accepted_exit_codes or {0}
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{name}.stdout.log"
    stderr_path = logs_dir / f"{name}.stderr.log"
    started = time.monotonic()
    print(f"STATUS stage={name} state=starting command={command[0]}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=stdout, stderr=stderr, text=True)
        next_status = started + STATUS_INTERVAL_SECONDS
        while process.poll() is None:
            now = time.monotonic()
            if now - started > timeout:
                process.kill()
                process.wait()
                raise TimeoutError(f"{name} exceeded {timeout}s")
            if now >= next_status:
                print(
                    f"STATUS stage={name} state=running elapsed_seconds={int(now - started)} pid={process.pid}",
                    flush=True,
                )
                next_status += STATUS_INTERVAL_SECONDS
            time.sleep(1.0)
    elapsed = time.monotonic() - started
    code = int(process.returncode or 0)
    print(f"STATUS stage={name} state=finished exit={code} elapsed_seconds={elapsed:.1f}", flush=True)
    if code not in accepted:
        raise RuntimeError(
            f"{name} failed with exit {code}\nSTDOUT:\n{tail(stdout_path)}\nSTDERR:\n{tail(stderr_path)}"
        )
    return code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw high-resolution GLB from the geometry model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--asset-type", default="auto")
    parser.add_argument("--quality", choices=("background", "gameplay", "hero"), default="hero")
    parser.add_argument("--source-image", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--benchmark-fixture", default="")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--silhouette-size", type=int, default=384)
    parser.add_argument("--max-candidates", type=int, default=7)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    logs = output / "logs"
    geometry_dir = output / "geometry"
    candidate_dir = geometry_dir / "candidates"
    reports_dir = output / "reports"
    for directory in (logs, geometry_dir, candidate_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source = Path(args.input).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_image = Path(args.source_image).resolve() if args.source_image else None
    if source_image and not source_image.is_file():
        raise FileNotFoundError(source_image)

    fixture = fixture_by_id(args.benchmark_fixture) if args.benchmark_fixture else None
    profile = get_profile(
        args.asset_type,
        args.quality,
        lod_enabled=True,
        prompt=args.prompt,
        filename=source.name,
    )
    asset_type = profile.asset_type.value
    family = family_for_asset_type(asset_type)
    pythonpath = os.pathsep.join((str(repo / "src"), str(repo), str(repo / "blender")))
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    clean_master = geometry_dir / "clean_master.glb"
    cleanup_report = reports_dir / "component_audit.json"
    cleanup_command = [
        sys.executable,
        str(repo / "workers" / "component_audit_cleanup.py"),
        "--input", str(source),
        "--output", str(clean_master),
        "--report", str(cleanup_report),
        "--asset-type", asset_type,
        "--samples", str(max(50_000, args.samples)),
    ]
    if source_image:
        cleanup_command += ["--source-image", str(source_image)]
    run_process(
        "clean_high_resolution_master",
        cleanup_command,
        cwd=repo,
        env=env,
        logs_dir=logs,
        timeout=7200,
    )
    cleanup = json.loads(cleanup_report.read_text(encoding="utf-8"))
    if not cleanup.get("success") or not clean_master.is_file():
        raise RuntimeError("high-resolution master cleanup did not pass")

    master_faces = int(cleanup["topology_after"]["faces"])
    plans = candidate_ladder(
        master_faces,
        args.quality,
        family,
        max_candidates=max(1, min(args.max_candidates, 9)),
    )
    budgets = [plan.target_faces for plan in plans]
    if not budgets:
        shutil.copy2(clean_master, geometry_dir / "selected_lod0.glb")
        raise RuntimeError("clean master is already below the minimum candidate floor; no ladder generated")

    candidate_manifest = reports_dir / "candidate_generation.json"
    blender_command = [
        str(Path(args.blender).resolve()),
        "--background",
        "--python-use-system-env",
        "--python", str(repo / "blender" / "optimize_ladder_candidates.py"),
        "--",
        "--input", str(clean_master),
        "--output-dir", str(candidate_dir),
        "--manifest", str(candidate_manifest),
        "--asset-type", asset_type,
        "--budgets", ",".join(str(value) for value in budgets),
        "--budget-mode", profile.budget_mode,
        "--per-object-target", str(profile.per_object_target),
        "--planar-angle-deg", str(profile.planar_angle_deg),
    ]
    run_process(
        "generate_lod_candidates",
        blender_command,
        cwd=repo,
        env=env,
        logs_dir=logs,
        timeout=14_400,
    )
    generated = json.loads(candidate_manifest.read_text(encoding="utf-8"))

    evaluations = []
    passed_once = False
    consecutive_failures = 0
    for item in generated["candidates"]:
        candidate_path = Path(item["path"])
        name = candidate_path.stem
        comparison_report = reports_dir / f"compare_{name}.json"
        compare_command = [
            sys.executable,
            str(repo / "workers" / "geometry_compare.py"),
            "--master", str(clean_master),
            "--candidate", str(candidate_path),
            "--report", str(comparison_report),
            "--asset-family", family.value,
            "--quality", args.quality,
            "--samples", str(max(20_000, args.samples)),
            "--silhouette-size", str(max(128, args.silhouette_size)),
            "--name", name,
        ]
        code = run_process(
            f"compare_{name}",
            compare_command,
            cwd=repo,
            env=env,
            logs_dir=logs,
            timeout=7200,
            accepted_exit_codes={0, 2},
        )
        comparison = json.loads(comparison_report.read_text(encoding="utf-8"))
        evaluation = comparison["evaluation"]
        evaluation["path"] = str(candidate_path)
        evaluation["report"] = str(comparison_report)
        evaluations.append(evaluation)
        if code == 0 and evaluation["valid"]:
            passed_once = True
            consecutive_failures = 0
        else:
            consecutive_failures += 1
        if passed_once and consecutive_failures >= 2:
            print("STATUS stage=compare_candidates state=stopped reason=two_consecutive_quality_failures", flush=True)
            break

    valid = [evaluation for evaluation in evaluations if evaluation["valid"]]
    if valid:
        selected = min(valid, key=lambda evaluation: int(evaluation["face_count"]))
        selected_source = Path(selected["path"])
        selection = "lowest_passing_candidate"
    else:
        selected = {
            "name": "clean_master",
            "face_count": master_faces,
            "path": str(clean_master),
            "valid": True,
            "errors": [],
        }
        selected_source = clean_master
        selection = "clean_master_fallback_no_candidate_passed"

    selected_lod0 = geometry_dir / "selected_lod0.glb"
    shutil.copy2(selected_source, selected_lod0)
    lod1 = geometry_dir / "lod1.glb"
    lod2 = geometry_dir / "lod2.glb"
    final_stats = reports_dir / "selected_lod_generation.json"
    final_mesh = geometry_dir / "game_ready_untextured.glb"
    actual_faces = int(selected["face_count"])
    final_command = [
        str(Path(args.blender).resolve()),
        "--background",
        "--python-use-system-env",
        "--python", str(repo / "blender" / "optimize_asset.py"),
        "--",
        "--input", str(selected_lod0),
        "--output", str(final_mesh),
        "--stats", str(final_stats),
        "--asset-type", asset_type,
        "--target-triangles", str(actual_faces),
        "--target-min", str(max(1, round(actual_faces * 0.95))),
        "--target-max", str(max(1, round(actual_faces * 1.05))),
        "--budget-mode", profile.budget_mode,
        "--per-object-target", str(profile.per_object_target),
        "--planar-angle-deg", "0.0",
        "--lod1", str(lod1),
        "--lod2", str(lod2),
        "--lod-ratios", ",".join(str(value) for value in profile.lod_ratios),
        "--lod-count", str(profile.lod_count),
    ]
    run_process(
        "materialize_selected_lods",
        final_command,
        cwd=repo,
        env=env,
        logs_dir=logs,
        timeout=7200,
    )

    result = {
        "success": True,
        "benchmark_fixture": fixture.as_dict() if fixture else None,
        "source": str(source),
        "source_sha256": sha256(source),
        "asset_type": asset_type,
        "asset_family": family.value,
        "quality": args.quality,
        "clean_master": str(clean_master),
        "clean_master_sha256": sha256(clean_master),
        "master_faces": master_faces,
        "cleanup_report": str(cleanup_report),
        "candidate_budgets": budgets,
        "evaluations": evaluations,
        "selection_policy": selection,
        "selected": selected,
        "selected_lod0": str(selected_lod0),
        "selected_lod0_sha256": sha256(selected_lod0),
        "game_ready_untextured": str(final_mesh),
        "game_ready_untextured_sha256": sha256(final_mesh),
        "lod1": str(lod1) if lod1.is_file() else None,
        "lod2": str(lod2) if lod2.is_file() else None,
        "external_inference_calls": 0,
        "manual_review_required": False,
    }
    final_report = reports_dir / "highres_geometry_ladder.json"
    final_report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "HIGHRES_GEOMETRY_LADDER_PASSED "
        f"master_faces={master_faces} selected_faces={actual_faces} "
        f"selection={selection} output={final_mesh}",
        flush=True,
    )


if __name__ == "__main__":
    main()
