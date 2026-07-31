"""Inventory the local benchmark pack and enforce the mandatory primary-anchor gate.

This worker never executes generated commands. It emits a deterministic plan and blocks later
fixtures until the priority-0 anchor has a recorded PROVEN result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lowvram3d.benchmark_suite import (
    ANCHOR_IDS,
    PRIMARY_ANCHOR_ID,
    BenchmarkFixture,
    ordered_fixtures,
)


PROVEN = "PROVEN"
BLOCKING_PRIMARY_STATES = {"FAILED", "BLOCKED", "NOT_RUN", "NOT_PROVEN"}


def asset_type_for_fixture(fixture: BenchmarkFixture) -> str:
    if fixture.group == "generated_character_examples" or fixture.fixture_id in {
        "red_panda_character",
        PRIMARY_ANCHOR_ID,
    }:
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


def primary_status_path(root: Path) -> Path:
    return root / "outputs" / PRIMARY_ANCHOR_ID / "benchmark_status.json"


def read_primary_status(root: Path) -> str:
    path = primary_status_path(root)
    if not path.is_file():
        return "NOT_RUN"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return "NOT_PROVEN"
    status = str(payload.get("classification", payload.get("status", "NOT_PROVEN"))).upper()
    return status if status in BLOCKING_PRIMARY_STATES | {PROVEN} else "NOT_PROVEN"


def command_for_fixture(
    repo: Path,
    blender: str,
    fixture: BenchmarkFixture,
    image: Path,
    master: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(repo / "workers" / "highres_geometry_ladder.py"),
        "--input", str(master),
        "--output-dir", str(output),
        "--blender", blender,
        "--asset-type", asset_type_for_fixture(fixture),
        "--quality", fixture.quality,
        "--benchmark-fixture", fixture.fixture_id,
        "--prompt", ", ".join(fixture.prompt_tags),
        "--source-image", str(image),
    ]


def build_inventory(
    root: Path,
    blender: str,
    selected_ids: set[str],
    *,
    allow_primary_bypass: bool = False,
) -> dict:
    repo = Path(__file__).resolve().parents[1]
    all_fixtures = ordered_fixtures()
    known_ids = {fixture.fixture_id for fixture in all_fixtures}
    unknown = selected_ids - known_ids
    if unknown:
        raise KeyError(f"unknown benchmark fixtures: {sorted(unknown)}")
    fixtures = [fixture for fixture in all_fixtures if not selected_ids or fixture.fixture_id in selected_ids]
    primary_status = read_primary_status(root)

    items = []
    for fixture in fixtures:
        image, master, output = fixture_paths(root, fixture)
        files_ready = image.is_file() and master.is_file()
        blocked_by_primary = (
            fixture.fixture_id != PRIMARY_ANCHOR_ID
            and primary_status != PROVEN
            and not allow_primary_bypass
        )
        command = (
            command_for_fixture(repo, blender, fixture, image, master, output)
            if files_ready and blender and not blocked_by_primary
            else None
        )
        items.append(
            {
                "fixture_id": fixture.fixture_id,
                "title": fixture.title,
                "group": fixture.group,
                "family": fixture.family.value,
                "asset_type": asset_type_for_fixture(fixture),
                "quality": fixture.quality,
                "priority": fixture.priority,
                "required_first": fixture.required_first,
                "image": str(image),
                "image_present": image.is_file(),
                "master": str(master),
                "master_present": master.is_file(),
                "files_ready": files_ready,
                "output": str(output),
                "anchor": fixture.fixture_id in ANCHOR_IDS,
                "blocked_by_primary_anchor": blocked_by_primary,
                "command": command,
            }
        )

    primary_item = next(
        (item for item in items if item["fixture_id"] == PRIMARY_ANCHOR_ID),
        None,
    )
    if primary_item is None:
        primary_image, primary_master, _ = fixture_paths(
            root,
            next(fixture for fixture in all_fixtures if fixture.fixture_id == PRIMARY_ANCHOR_ID),
        )
        primary_files_ready = primary_image.is_file() and primary_master.is_file()
    else:
        primary_files_ready = bool(primary_item["files_ready"])

    if primary_status == PROVEN:
        overall_gate = "PRIMARY_PROVEN"
    elif primary_files_ready:
        overall_gate = "READY_TO_RUN_PRIMARY"
    else:
        overall_gate = "BLOCKED_PRIMARY_ANCHOR_SOURCE_NOT_VERIFIED"

    return {
        "schema_version": 2,
        "root": str(root),
        "fixture_count": len(items),
        "available_images": sum(item["image_present"] for item in items),
        "available_masters": sum(item["master_present"] for item in items),
        "primary_anchor_id": PRIMARY_ANCHOR_ID,
        "primary_anchor_status": primary_status,
        "primary_anchor_files_ready": primary_files_ready,
        "primary_anchor_status_path": str(primary_status_path(root)),
        "overall_gate": overall_gate,
        "missing_anchor_images": [
            item["fixture_id"] for item in items if item["anchor"] and not item["image_present"]
        ],
        "missing_anchor_masters": [
            item["fixture_id"] for item in items if item["anchor"] and not item["master_present"]
        ],
        "items": items,
        "external_inference_calls": 0,
        "commands_executed": 0,
        "primary_bypass_used": allow_primary_bypass,
        "overall_pass_permitted": primary_status == PROVEN and not allow_primary_bypass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Local benchmark-pack directory")
    parser.add_argument("--report", default="")
    parser.add_argument("--blender", default="")
    parser.add_argument("--fixture", action="append", default=[])
    parser.add_argument("--require-anchors", action="store_true")
    parser.add_argument("--require-primary-anchor", action="store_true")
    parser.add_argument("--allow-primary-anchor-bypass", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = set(args.fixture)
    payload = build_inventory(
        root,
        args.blender,
        selected,
        allow_primary_bypass=args.allow_primary_anchor_bypass,
    )
    report = Path(args.report).resolve() if args.report else root / "benchmark_inventory.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"BENCHMARK_INVENTORY fixtures={payload['fixture_count']} "
        f"images={payload['available_images']} masters={payload['available_masters']} "
        f"primary={payload['primary_anchor_status']} gate={payload['overall_gate']} report={report}",
        flush=True,
    )

    default_full_run = not selected
    if (args.require_primary_anchor or default_full_run) and not payload["primary_anchor_files_ready"]:
        raise SystemExit(3)
    if args.require_anchors and (payload["missing_anchor_images"] or payload["missing_anchor_masters"]):
        raise SystemExit(2)
    later_requested = selected and PRIMARY_ANCHOR_ID not in selected
    if later_requested and payload["primary_anchor_status"] != PROVEN and not args.allow_primary_anchor_bypass:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
