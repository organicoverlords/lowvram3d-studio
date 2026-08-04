"""Test the measured glTF->Unreal axis mapping against a real render.

`unreal/measure_axis_mapping.py` measures the importer's mapping from a probe
mesh. That measurement makes a falsifiable prediction about a MoGe
reconstruction, which is what makes it worth trusting:

MoGe exports glTF with Y up and Z toward the viewer, so its camera looks down
glTF -Z. Under the measured mapping (gltf X->+X, Y->+Z, Z->+Y) that direction
becomes Unreal -Y, which is **yaw -90**, not the yaw 0 the standard convention
predicts and that this project assumed.

So: build the reconstruction, render it from a sweep of yaws, and score each
render against the source image. If the measurement is right, -90 wins clearly.
If something else wins, the measurement is incomplete and must not be built on.

    py -3.12 scripts/verify_axis_mapping_prediction.py --glb <moge.glb> \
        --source <image.png> --out evidence/axis-probe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UNREAL = REPO / "unreal"
YAW_CANDIDATES = [0.0, -50.0, -90.0, 90.0, 180.0]


def _bridge():
    if str(UNREAL) not in sys.path:
        sys.path.insert(0, str(UNREAL))
    from uemcp import Bridge

    return Bridge()


def score(render: Path, source: Path, out: Path) -> dict:
    """Delegate to the existing comparison so scoring stays one implementation."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compare_source_render.py"),
         "--render", str(render), "--source", str(source), "--out", str(out)],
        capture_output=True, text=True)
    receipt = out / "source_camera_comparison.json"
    if not receipt.is_file():
        return {"error": (result.stderr or result.stdout)[-800:]}
    return json.loads(receipt.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene-id", default="axis_check")
    parser.add_argument("--fov", type=float, default=92.873)
    parser.add_argument("--aspect", type=float, default=4.0 / 3.0)
    parser.add_argument("--skip-build", action="store_true",
                        help="the scene is already built in the editor")
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    glb = Path(args.glb).resolve()
    source = Path(args.source).resolve()

    bridge = _bridge()
    request = {
        "glb": str(glb),
        "scene_id": args.scene_id,
        "package_root": f"/Game/AgentProof/{args.scene_id}",
        "fov_deg": args.fov,
        "aspect_ratio": args.aspect,
    }
    if args.skip_build:
        build = {"skipped": True}
    else:
        bridge.python("SCENE_REQUEST = " + json.dumps(request), "SCENE_REQUEST",
                      timeout=1800)
        try:
            build = bridge.python_json(
                (UNREAL / "build_reconstructed_scene.py").read_text(encoding="utf-8"),
                "result", timeout=1800)
        except Exception as exc:
            # A large import routinely exceeds the handler timeout while still
            # completing, so ask the editor what exists rather than retrying
            # and stacking a second copy of the scene.
            build = {"handler_error": f"{type(exc).__name__}: {exc}"}
            build["actors_after"] = bridge.python_json(
                "import unreal, json\n"
                "sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)\n"
                "result = json.dumps({'actors': [str(a.get_actor_label())"
                " for a in sub.get_all_level_actors()]})\n", "result", timeout=300)
            if not any("ReconstructedMesh" in label for label
                       in build["actors_after"].get("actors", [])):
                raise
    (out / "axis_check_build_receipt.json").write_text(
        json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    results = []
    for yaw in YAW_CANDIDATES:
        render = out / f"yaw_{int(yaw):+04d}.png"
        shot = bridge.call("capture_scene_png", {
            "outputPath": str(render), "width": 1024, "height": 768,
            "fov": args.fov, "world": "editor",
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"pitch": 0.0, "yaw": yaw, "roll": 0.0},
        }, timeout=600)
        entry = {"yaw": yaw, "render": str(render),
                 "render_bytes": render.stat().st_size if render.is_file() else 0,
                 "capture": shot}
        if render.is_file():
            comparison = score(render, source, out / f"score_yaw_{int(yaw):+04d}")
            best = comparison.get("best_orientation")
            entry["best_transform"] = best
            # Score the *unflipped* render: a yaw is only correct if it matches
            # the source without also needing a mirror or a 180 rotation.
            identity = (comparison.get("orientation_scores") or {}).get("identity", {})
            entry["correlation"] = identity.get("combined")
            entry["identity_scores"] = identity
            entry["comparison"] = comparison
        results.append(entry)

    scored = [r for r in results if isinstance(r.get("correlation"), (int, float))]
    winner = max(scored, key=lambda r: r["correlation"]) if scored else None
    receipt = {
        "schema_version": "axis_mapping_prediction_v1",
        "prediction": "yaw -90 reproduces the source view",
        "glb": str(glb),
        "source": str(source),
        "candidates": [
            {k: v for k, v in r.items() if k != "comparison"} for r in results],
        "winning_yaw": winner["yaw"] if winner else None,
        "winning_correlation": winner["correlation"] if winner else None,
        "winning_transform": winner.get("best_transform") if winner else None,
        "classification": (
            "PROVEN" if winner and winner["yaw"] == -90.0 else
            "REFUTED" if winner else "NOT_PROVEN"),
    }
    (out / "axis_mapping_prediction.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
