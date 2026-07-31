"""Create a non-destructive, debris-audited high-resolution geometry master."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lowvram3d.component_audit import AuditConfig, audit_and_cleanup


def config_for_asset_type(
    asset_type: str,
    *,
    render_size: int,
    samples: int,
    max_passes: int,
) -> AuditConfig:
    """Return conservative scale-relative cleanup limits for each asset family."""
    kind = str(asset_type).strip().lower()
    common = {
        "render_size": max(192, render_size),
        "total_samples": max(50_000, samples),
        "max_passes": max(1, min(max_passes, 6)),
    }
    if kind in {"building", "room"}:
        return AuditConfig(
            **common,
            outboard_max_area_fraction=0.003,
            hover_max_area_fraction=0.002,
            internal_max_area_fraction=0.0,
            preserve_area_fraction=0.01,
            preserve_face_fraction=0.05,
        )
    if kind in {"scene", "level"}:
        return AuditConfig(
            **common,
            outboard_max_area_fraction=0.0015,
            hover_max_area_fraction=0.001,
            internal_max_area_fraction=0.0,
            preserve_area_fraction=0.005,
            preserve_face_fraction=0.02,
        )
    if kind in {"vehicle", "prop"}:
        return AuditConfig(
            **common,
            outboard_max_area_fraction=0.015,
            hover_max_area_fraction=0.008,
            internal_max_area_fraction=0.0005,
            preserve_area_fraction=0.03,
            preserve_face_fraction=0.10,
        )
    if kind in {"natural", "vegetation"}:
        return AuditConfig(
            **common,
            outboard_max_area_fraction=0.012,
            hover_max_area_fraction=0.006,
            internal_max_area_fraction=0.0005,
            preserve_area_fraction=0.02,
            preserve_face_fraction=0.08,
        )
    return AuditConfig(**common)


def _ambiguity_only(result: dict) -> bool:
    """True when cleanup failed only because evidence was inconclusive."""
    errors = [str(item) for item in result.get("errors", [])]
    if not errors:
        return False
    for error in errors:
        if error == "cleanup did not converge within max_passes":
            continue
        words = error.split()
        if (
            len(words) >= 5
            and words[0].isdigit()
            and error.endswith("visible components remain audit-required")
        ):
            continue
        return False
    return True


def _topology_safe(result: dict) -> bool:
    """Reject pass-through when the audit observed real topology damage."""
    before = result.get("topology_before", {}) or {}
    after = result.get("topology_after", {}) or {}
    for key in ("boundary_edges", "non_manifold_edges"):
        if int(after.get(key, 0)) > int(before.get(key, 0)):
            return False
    main_before = result.get("main_component_faces_before")
    main_after = result.get("main_component_faces_after")
    if main_before is not None and main_after is not None and int(main_before) != int(main_after):
        return False
    return True


def _preserve_original(
    result: dict,
    input_path: str,
    output_path: str,
) -> dict:
    """Preserve a valid generated mesh when cleanup confidence is insufficient.

    The component audit remains evidence, but ambiguity is not allowed to destroy a usable
    TurboBird-class result or stop the rest of the production pipeline.
    """
    source = Path(input_path)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)

    original_errors = [str(item) for item in result.get("errors", [])]
    before = dict(result.get("topology_before", {}) or {})
    result["success"] = True
    result["output"] = str(output)
    result["errors"] = []
    result["warnings"] = [
        *[str(item) for item in result.get("warnings", [])],
        "component cleanup was inconclusive; preserved the original generated mesh byte-for-byte",
        *[f"audit: {item}" for item in original_errors],
    ]
    result["topology_after"] = before
    result["faces_removed"] = 0
    result["faces_removed_percent"] = 0.0
    result["main_component_faces_after"] = result.get("main_component_faces_before")
    result["manual_review_required"] = True
    result["audit_policy"] = {
        "selected": "preserve_original_on_audit_ambiguity",
        "hard_failure": False,
        "original_errors": original_errors,
        "reason": (
            "The generated mesh had no demonstrated topology regression. Ambiguous component "
            "classification is recorded for review but does not block downstream processing."
        ),
    }
    return result


def run_cleanup(
    input_path: str,
    output_path: str,
    *,
    asset_type: str,
    source_image: str | None,
    config: AuditConfig,
    seed: int,
) -> dict:
    """Run cleanup once and fail only on demonstrated damage or process errors."""
    result = audit_and_cleanup(
        input_path,
        output_path,
        asset_type=asset_type,
        source_image=source_image,
        config=config,
        seed=seed,
    )
    if result.get("success"):
        result["manual_review_required"] = False
        result["audit_policy"] = {"selected": "audited_cleanup", "hard_failure": False}
        return result
    if _ambiguity_only(result) and _topology_safe(result):
        return _preserve_original(result, input_path, output_path)
    result["audit_policy"] = {
        "selected": "hard_failure",
        "hard_failure": True,
        "reason": "cleanup reported a process error or demonstrated topology regression",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--asset-type", default="prop")
    parser.add_argument("--source-image", default="")
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--samples", type=int, default=220_000)
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = config_for_asset_type(
        args.asset_type,
        render_size=args.render_size,
        samples=args.samples,
        max_passes=args.max_passes,
    )
    result = run_cleanup(
        args.input,
        args.output,
        asset_type=args.asset_type,
        source_image=args.source_image or None,
        config=config,
        seed=args.seed,
    )
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    policy = result.get("audit_policy", {}).get("selected", "unknown")
    print(
        "COMPONENT_AUDIT_CLEANUP "
        f"success={result['success']} "
        f"faces={result['topology_before']['faces']}->{result['topology_after']['faces']} "
        f"removed={result['faces_removed_percent']:.4f}% "
        f"boundary={result['topology_before']['boundary_edges']}->{result['topology_after']['boundary_edges']} "
        f"passes={len(result['passes'])} policy={policy}",
        flush=True,
    )
    if not result["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
