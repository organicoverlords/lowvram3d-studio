"""Measure whether a scene survives being looked at from somewhere else.

`docs/pipelines/README.md` names `offaxis_stability` as the one metric both
pipelines must satisfy, and the only one that would have caught the flat
textured shell that passed as a scene for several sessions. It had never been
implemented. This implements it.

The failure it exists to catch: a shell reproduces its source view perfectly and
vanishes when the camera moves, because there is nothing behind the image. Real
geometry keeps filling the frame from any angle.

*How coverage is measured.* Each pose is rendered twice, once with the build's
own actors hidden, and the scene's contribution is the difference. Thresholding
colour instead would not work here -- a dark tree against an unlit backdrop is
not separable by brightness, and this project's renders are mostly dark.

*What the number means.* Coverage is the fraction of the frame the scene fills
at a pose. The score is the worst off-axis coverage divided by the coverage from
the scene's own front, so 1.0 means the scene holds up equally from everywhere
and values near 0 mean it disappears when you step aside. A flat shell scores
near zero by construction.

Angles are relative to the scene's front, and the orbit radius comes from the
scene's own bounds -- a fixed radius would be inside a barn and lost outside a
two-hundred-metre field.

    py -3.12 scripts/measure_offaxis_stability.py --scene-id barn_gen \
        --out evidence/barn-auto/offaxis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UNREAL = REPO / "unreal"
# Yaw offsets from the scene's front. 0 is the source view; the rest step away.
DEFAULT_ANGLES = [0.0, 30.0, 60.0, 90.0, -30.0, -60.0, -90.0]
# A pixel must change by more than this for the scene to count as covering it,
# which keeps compression noise and faint bounce light out of the measurement.
DIFFERENCE_THRESHOLD = 8


def _bridge():
    if str(UNREAL) not in sys.path:
        sys.path.insert(0, str(UNREAL))
    from uemcp import Bridge

    return Bridge()


def _assign(bridge, name, value):
    bridge.python(f"import json\n{name} = json.loads({json.dumps(value)!r})",
                  name, timeout=120.0)


def coverage(scene_png: Path, empty_png: Path) -> float:
    """Fraction of the frame the scene's own actors account for."""
    import numpy as np
    from PIL import Image

    scene = np.asarray(Image.open(scene_png).convert("RGB"), dtype=np.int16)
    empty = np.asarray(Image.open(empty_png).convert("RGB"), dtype=np.int16)
    if scene.shape != empty.shape:
        raise RuntimeError("render pair differs in size")
    return float((np.abs(scene - empty).max(axis=2) > DIFFERENCE_THRESHOLD).mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--owner-tag", default=None)
    parser.add_argument("--fov", type=float, default=75.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--angles", default=None,
                        help="comma-separated yaw offsets from the scene front")
    parser.add_argument("--pitch", type=float, default=-12.0)
    parser.add_argument("--radius-scale", type=float, default=1.4)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO / "src"))
    from lowvram3d.unreal_stage import capture_scene

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    owner_tag = args.owner_tag or f"structural_build_{args.scene_id}"
    angles = ([float(a) for a in args.angles.split(",")] if args.angles
              else DEFAULT_ANGLES)

    bridge = _bridge()
    _assign(bridge, "SCENE_BOUNDS_REQUEST", {"owner_tag": owner_tag})
    bounds = bridge.python_json(
        (UNREAL / "scene_bounds.py").read_text(encoding="utf-8"), "result",
        timeout=300.0)

    centre = bounds["centre_m"]
    # Radius from the scene's own footprint, and far enough back that the whole
    # thing fits the field of view at every angle.
    span = max(bounds["size_m"][0], bounds["size_m"][1], 1.0)
    radius = args.radius_scale * 0.5 * span / math.tan(math.radians(args.fov) * 0.5)
    height = centre[2] + max(bounds["size_m"][2], 1.0) * 0.6

    def toggle(hidden: bool) -> None:
        _assign(bridge, "TOGGLE_REQUEST",
                {"owner_tag": owner_tag, "hidden": hidden})
        bridge.python_json(
            (UNREAL / "toggle_owned_actors.py").read_text(encoding="utf-8"),
            "result", timeout=300.0)

    views = []
    try:
        for angle in angles:
            # The scene's front is -X looking toward +X, so an offset orbits
            # around the centre at that yaw.
            theta = math.radians(180.0 + angle)
            location = (centre[0] + radius * math.cos(theta),
                        centre[1] + radius * math.sin(theta),
                        height)
            yaw = math.degrees(math.atan2(centre[1] - location[1],
                                          centre[0] - location[0]))
            tag = f"{int(angle):+04d}"
            scene_png = out / f"view_{tag}.png"
            empty_png = out / f"view_{tag}_empty.png"

            toggle(False)
            shot = capture_scene(scene_png, args.scene_id, args.fov,
                                 width=args.width, height=args.height,
                                 location=location,
                                 rotation=(args.pitch, yaw, 0.0))
            toggle(True)
            capture_scene(empty_png, args.scene_id, args.fov,
                          width=args.width, height=args.height,
                          location=location, rotation=(args.pitch, yaw, 0.0))

            views.append({
                "angle_deg": angle,
                "location_m": [round(v, 2) for v in location],
                "yaw_deg": round(yaw, 2),
                "render": str(scene_png),
                "captured": bool(shot.get("success")),
                "coverage": round(coverage(scene_png, empty_png), 5)
                if scene_png.is_file() and empty_png.is_file() else None,
            })
    finally:
        # Never leave the scene hidden, whatever went wrong above.
        toggle(False)

    scored = [v for v in views if isinstance(v.get("coverage"), float)]
    front = next((v["coverage"] for v in scored if v["angle_deg"] == 0.0), None)
    offaxis = [v["coverage"] for v in scored if v["angle_deg"] != 0.0]
    stability = (min(offaxis) / front if front and offaxis and front > 1e-6
                 else None)

    receipt = {
        "schema_version": "offaxis_stability_v1",
        "scene_id": args.scene_id,
        "owner_tag": owner_tag,
        "bounds": bounds,
        "orbit_radius_m": round(radius, 2),
        "views": views,
        "front_coverage": front,
        "worst_offaxis_coverage": min(offaxis) if offaxis else None,
        "mean_offaxis_coverage": (round(sum(offaxis) / len(offaxis), 5)
                                  if offaxis else None),
        "offaxis_stability": round(stability, 4) if stability is not None else None,
        # A shell scores near zero by construction; real geometry stays visible.
        "classification": (
            "PROVEN" if stability is not None and stability >= 0.5
            else "WEAK" if stability is not None and stability >= 0.2
            else "NOT_PROVEN"),
    }
    (out / "offaxis_stability.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "views"},
                     indent=2, sort_keys=True))
    for view in views:
        print(f"  yaw {view['angle_deg']:+6.1f}  coverage {view['coverage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
