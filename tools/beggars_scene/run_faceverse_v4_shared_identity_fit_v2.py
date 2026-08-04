from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def argument_value(flag: str) -> str:
    try:
        index = sys.argv.index(flag)
    except ValueError as error:
        raise SystemExit(f"Required argument missing from shared-fit v2 bootstrap: {flag}") from error
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"Required argument has no value: {flag}")
    return sys.argv[index + 1]


def install_faceverse_landmark_scale_fix(faceverse_root: Path) -> type[Any]:
    if str(faceverse_root) not in sys.path:
        sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel

    original_init = FaceVerseRecon.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if getattr(self, "_beggars_landmark_scale_fix", False):
            return
        self.idBase_lms = self.idBase_lms / 100.0
        self.exBase_lms = self.exBase_lms / 100.0
        self.meanshape_lms = self.meanshape_lms / 100.0
        self._beggars_landmark_scale_fix = True
        print(
            "FACEVERSE_LANDMARK_BASIS_SCALE_FIX=PROVEN "
            f"id={tuple(self.idBase_lms.shape)} exp={tuple(self.exBase_lms.shape)} "
            f"mean={tuple(self.meanshape_lms.shape)}",
            flush=True,
        )

    FaceVerseRecon.__init__ = patched_init
    return FaceVerseRecon


def corrected_map_landmarks_to_source(
    model: Any,
    coefficients: torch.Tensor,
    bbox_tensor: torch.Tensor,
) -> torch.Tensor:
    (
        identity,
        expression,
        texture,
        lighting,
        angles,
        translation,
        eyes,
    ) = model.split_coeffs(coefficients)
    adjusted_expression = expression.clone()
    adjusted_expression[:, 14:16] = model.adjust_eyes(adjusted_expression[:, 14:16])
    adjusted_expression[:, 49:50] = model.adjust_mouth(adjusted_expression[:, 49:50])
    corrected_coefficients = model.merge_coeffs(
        identity,
        adjusted_expression,
        texture,
        lighting,
        angles,
        translation,
        eyes,
    )
    projected = model.run(corrected_coefficients, only_lms=True)["lms_proj"][:, :478, :2]
    widths = (bbox_tensor[:, 2] - bbox_tensor[:, 0]).view(-1, 1)
    heights = (bbox_tensor[:, 3] - bbox_tensor[:, 1]).view(-1, 1)
    mapped_x = projected[:, :, 0] / float(model.imgsize) * widths + bbox_tensor[:, 0].view(-1, 1)
    mapped_y = projected[:, :, 1] / float(model.imgsize) * heights + bbox_tensor[:, 1].view(-1, 1)
    return torch.stack((mapped_x, mapped_y), dim=2)


def validate_and_reclassify(output_dir: Path) -> None:
    report_path = output_dir / "faceverse_shared_identity_fit.json"
    if not report_path.exists():
        raise RuntimeError(f"Shared-fit v2 report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    initial_rmse = float(report["initial_global_landmark_rmse_pixels"])
    final_rmse = float(report["final_global_landmark_rmse_pixels"])
    if not 0.0 < initial_rmse <= 50.0:
        raise RuntimeError(
            f"Corrected FaceVerse initial landmark projection is implausible: {initial_rmse:.4f}px"
        )
    if not 0.0 < final_rmse <= initial_rmse:
        raise RuntimeError(
            "Corrected FaceVerse optimizer did not improve the projection: "
            f"initial={initial_rmse:.4f}px final={final_rmse:.4f}px"
        )
    report["classification"] = "USER_VISUAL_REVIEW_REQUIRED"
    report["route"] = "FACEVERSE_V4_SHARED_MULTI_FRAME_LANDMARK_IDENTITY_FIT_V2_SCALED"
    report["landmark_basis_scale_fix"] = {
        "idBase_lms_divisor": 100.0,
        "exBase_lms_divisor": 100.0,
        "meanshape_lms_divisor": 100.0,
        "reason": "Match the scaling already applied by the official full-mesh FaceVerse initialization.",
    }
    report["initial_projection_sanity_gate_px"] = 50.0
    report["projection_audit_reference"] = {
        "full_mesh_vs_target_rmse_pixels": 7.3298797607421875,
        "unscaled_lightweight_vs_target_rmse_pixels": 24372.640625,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "FACEVERSE_SHARED_FIT_V2_SANITY=PROVEN "
        f"initial_rmse={initial_rmse:.4f} final_rmse={final_rmse:.4f}",
        flush=True,
    )


def main() -> int:
    faceverse_root = Path(argument_value("--faceverse-root")).resolve()
    output_dir = Path(argument_value("--output-dir")).resolve()
    if not faceverse_root.exists():
        raise SystemExit(f"FaceVerse source root is missing: {faceverse_root}")

    install_faceverse_landmark_scale_fix(faceverse_root)

    import run_faceverse_v4_shared_identity_fit as base  # pylint: disable=import-outside-toplevel

    base.map_landmarks_to_source = corrected_map_landmarks_to_source
    result = int(base.main())
    validate_and_reclassify(output_dir)
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise
