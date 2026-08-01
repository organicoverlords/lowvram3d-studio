"""Fail closed when source-to-mesh registration is too weak for projection.

Atlas orientation proves that texels sample the intended UV coordinates. It does not prove that the
source illustration was warped onto the correct geometric regions. This gate reads the projection
view builder's silhouette-registration evidence and rejects a texture before bake/export when the
source-to-mesh warp is too weak or unstable. Pipeline repair policy version 2 pairs this gate with
source-supported post-LOD debris classification, so small valid ornaments are not sacrificed merely
to make a geometry metric pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

POLICY_VERSION = 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-final-iou", type=float, default=0.58)
    parser.add_argument("--max-flow-cap-saturation", type=float, default=0.98)
    args = parser.parse_args()

    source = Path(args.registration_report)
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    dense = data.get("dense_registration") or {}
    accepted = bool(dense.get("accepted"))
    affine_iou = float(
        dense.get("affine_iou", data.get("silhouette_iou_refined_affine", 0.0)) or 0.0
    )
    dense_iou = float(dense.get("dense_iou", affine_iou) or 0.0)
    final_iou = dense_iou if accepted else affine_iou
    p95 = float(dense.get("flow_p95_pixels", 0.0) or 0.0)
    cap = float(dense.get("flow_cap_pixels", 0.0) or 0.0)
    cap_saturation = p95 / cap if cap > 0 else 0.0
    visible = float(data.get("visible_percent", 0.0) or 0.0)

    failures = []
    if not bool(data.get("registration_gate_passed", False)):
        failures.append("registration worker rejected its own result")
    if final_iou < args.min_final_iou:
        failures.append(f"final silhouette IoU {final_iou:.4f} < {args.min_final_iou:.4f}")
    if cap > 0 and cap_saturation > args.max_flow_cap_saturation:
        failures.append(
            f"95th-percentile warp uses {cap_saturation:.3f} of the hard flow cap; alignment is unstable"
        )
    if visible <= 5.0:
        failures.append(f"only {visible:.3f}% of triangles are directly visible")

    result = {
        "policy_version": POLICY_VERSION,
        "registration_report": str(source),
        "passed": not failures,
        "failure_code": None if not failures else "TEXTURE_MISREGISTRATION",
        "affine_iou": affine_iou,
        "dense_iou": dense_iou,
        "dense_accepted": accepted,
        "final_iou": final_iou,
        "flow_p95_pixels": p95,
        "flow_cap_pixels": cap,
        "flow_cap_saturation": cap_saturation,
        "visible_percent": visible,
        "limits": {
            "min_final_iou": args.min_final_iou,
            "max_flow_cap_saturation": args.max_flow_cap_saturation,
            "min_visible_percent": 5.0,
        },
        "failures": failures,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"TEXTURE_REGISTRATION policy={POLICY_VERSION} passed={result['passed']} "
        f"final_iou={final_iou:.4f} dense={accepted} "
        f"flow_cap_saturation={cap_saturation:.3f}",
        flush=True,
    )
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
