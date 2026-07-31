"""Inventory the local benchmark pack and emit deterministic commands for available masters.

Large source images and GLBs are intentionally not committed. Expected layout:

  <root>/images/<fixture.source_image_name>
  <root>/masters/<fixture.master_reference_name or fixture_id.glb>
  <root>/outputs/<fixture_id>/...

This module never executes generated commands. It only writes a machine-readable plan.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lowvram3d.benchmark_suite import FIXTURES, BenchmarkFixture


ANCHORS = {"turbo_bird_high_detail", "red_panda_character"}


def asset_type_for_fixture(fixture: BenchmarkFixture) -> str:
    if fixture.group == "generated_character_examples" or fixture.fixture_id == "red_panda_character":
        return "character"
    if fixture.group == "generated_building_examples":
        return "building"
    if fixture.group == "generated_vehicle_examples":
        return "vehicle"
    if fixture.group == "generated_natural_examples":
        return "prop"
    if fixture.group == "generated_creature_examples":
        return "creature"
    return {
        "organic": "creature",
        "hard_surface": "vehicle",
        "architectural": "building",
        "natural": "prop",
        "mixed": "prop",
        "unknown": "prop",
    }[fixture.family.value]


def fixture_paths(root: Path, fixture: BenchmarkFixture) -> tuple[Path, Path, Path]:
    image = root / "images" / fixture.source_image_name
    master_name = fixture.master_reference_name or f"{fixture.fixture_id}.glb"
    master = root / "masters" / master_name
    output = root / "outputs" / fixture.fixture_id
    return image, master, output


def command_for_fixture(
    repo: Path,
    blender: str,
    fixture: BenchmarkFixture,
    image: Path,
    master: Path,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(repo / "workers" / "highres_geometry_ladder.py"),
        "--input", str(master),
        "--output-dir", str(output),
        "--blender", blender,
        "--asset-type", asset_type_for_fixture(fixture),
        "--quality", fixture.quality,
        "--benchmark-fixture", fixture.fixture_id,
        "--prompt", ", ".join(fixture.prompt_tags),
    ]
    if image.is_file():
        command += ["--source-image", str(image)]
    return command


def build_inventory(root: Path, blender: str, selected_ids: set[str]) -> dict:
    repo = Path(__file__).resolve().parents[1]
    fixtures = [fixture for fixture in FIXTURES if not selected_ids or fixture.fixture_id in selected_ids]
    unknown = selected_ids - {fixture.fixture_id for fixture in fixtures}
    if unknown:
        raise KeyError(f"unknown benchmark fixtures: {sorted(unknown)}")

    items = []
    for fixture in fixtures:
        image, master, output = fixture_paths(root, fixture)
        items.append(
            {
                "fixture_id": fixture.fixture_id,
                "title": fixture.title,
                "group": fixture.group,
                "family": fixture.family.value,
                "asset_type": asset_type_for_fixture(fixture),
                "quality": fixture.quality,
                "image": str(image),
                "image_present": image.is_file(),
                "master": str(master),
                "master_present": master.is_file(),
                "output": str(output),
                "anchor": fixture.fixture_id in ANCHORS,
                "command": command_for_fixture(repo, blender, fixture, image, master, output)
                if master.is_file() and blender
                else None,
            }
        )
    return {
        "schema_version": 1,
        "root": str(root),
        "fixture_count": len(items),
        "available_images": sum(item["image_present"] for item in items),
        "available_masters": sum(item["master_present"] for item in items),
        "missing_anchor_masters": [
            item["fixture_id"] for item in items if item["anchor"] and not item["master_present"]
        ],
        "items": items,
        "external_inference_calls": 0,
        "commands_executed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Local benchmark-pack directory")
    parser.add_argument("--report", default="")
    parser.add_argument("--blender", default="")
    parser.add_argument("--fixture", action="append", default=[])
    parser.add_argument("--require-anchors", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = build_inventory(root, args.blender, set(args.fixture))
    report = Path(args.report).resolve() if args.report else root / "benchmark_inventory.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"BENCHMARK_INVENTORY fixtures={payload['fixture_count']} "
        f"images={payload['available_images']} masters={payload['available_masters']} report={report}",
        flush=True,
    )
    if args.require_anchors and payload["missing_anchor_masters"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
