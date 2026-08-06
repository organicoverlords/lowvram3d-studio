"""Decide whether a reconstruction is left-right reversed. No per-asset tuning.

Three wrong answers on the red panda came from the same mistake: comparing a
measurement taken in one camera convention against a measurement taken in
another. A yaw in the projector's convention, a panel in a preview tool's
convention and a texel count in atlas space are three different frames, and
"the tail is on the right" means something different in each.

A fourth wrong answer was nearly added by fixing the panda specifically. The
obvious chirality test is to find the subject's one saturated lateral feature --
the rust tail -- and see which side it falls on. That works, and it is worthless:
it needs a hue window per subject, and this pipeline also has to handle a sky
whale, a boat, a bird-skull shaman and a castle. A criterion that must be
re-tuned for every asset is not a criterion.

What generalises is that CHIRALITY IS CARRIED BY THE GEOMETRY. A rifle held
across the chest, a tail to one side, an asymmetric rig -- the generator sculpts
these, and a mirrored mesh sculpts them backwards. So:

    render the bare mesh at the resolved front yaw, then ask whether the SOURCE
    or the SOURCE MIRRORED is the better match to that render.

If the mirrored source wins, the mesh is reversed relative to the source. There
is no colour model, no threshold per subject and nothing to hand-tune. It is
also non-circular in the way that matters: the render carries no texture, so the
projection whose correctness is in question contributes nothing to the answer.

The comparison runs in DINOv2 feature space for the same reason the front-axis
resolver does -- a self-supervised ViT is dominated by shape and part layout,
which is what survives the gap between a painted illustration and grey clay.
Raw pixels do not survive it; that was measured, not assumed.

    py workers/measure_chirality.py --mesh uv.glb --source crop.png \
       --yaw 178.333 --out chirality.json --sheet chirality.png

Reports MIRRORED or MATCHES_SOURCE only when the two candidates separate by
MIN_SEPARATION. A near-tie means the subject is close to bilaterally symmetric
from this view, in which case chirality is genuinely unobservable here and the
honest answer is UNRESOLVED -- the same fail-closed posture the true-rear
verdict takes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from resolve_front_axis_dino import MODEL, RENDER, clay_render, embed

#: A source and its mirror are the same picture to any shape-blind statistic, so
#: the two scores are close by construction and the threshold has to be modest.
#: But it cannot be zero: a subject that really is bilaterally symmetric from the
#: front will produce a coin flip, and calling that a verdict is how the earlier
#: wrong answers were reached. Below this, say UNRESOLVED.
MIN_SEPARATION = 0.01


def resolve(mesh_path: Path, source_path: Path, yaw: float, sheet: Path | None):
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    scene = trimesh.load(mesh_path, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    vertices = np.asarray(mesh.vertices, np.float64)
    vertices = vertices - vertices.mean(axis=0)
    vertices /= max(float(np.abs(vertices).max()), 1e-9)
    faces = np.asarray(mesh.faces, np.int64)
    normals = np.asarray(mesh.vertex_normals, np.float64)

    # Composited onto white, because the clay render is on white and the
    # background must not be the thing the features agree about.
    source = Image.open(source_path).convert("RGBA")
    flat = Image.new("RGB", source.size, (255, 255, 255))
    flat.paste(source, mask=source.split()[3])
    flipped = flat.transpose(Image.FLIP_LEFT_RIGHT)

    render = clay_render(vertices, faces, normals, float(yaw), size=RENDER)

    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    vectors = embed([render, flat, flipped], model, processor, torch)
    as_is = float(vectors[0] @ vectors[1])
    mirrored = float(vectors[0] @ vectors[2])

    separation = abs(as_is - mirrored)
    if separation < MIN_SEPARATION:
        verdict = "UNRESOLVED"
    elif mirrored > as_is:
        verdict = "MIRRORED"
    else:
        verdict = "MATCHES_SOURCE"

    result = {
        "schema": "lowvram3d_chirality_v2",
        "mesh": str(mesh_path),
        "source": str(source_path),
        "front_yaw_deg": float(yaw),
        "model": MODEL,
        "score_source_as_is": round(as_is, 6),
        "score_source_mirrored": round(mirrored, 6),
        "separation": round(separation, 6),
        "min_separation": MIN_SEPARATION,
        "verdict": verdict,
        "note": ("bare geometry versus source and mirrored source; no texture, "
                 "no atlas, no colour model, no per-asset constant. See the "
                 "docstring for why a tail-colour test was rejected."),
    }

    if sheet is not None:
        from PIL import ImageDraw

        tile, bar = 320, 24
        canvas = Image.new("RGB", (tile * 3, tile + bar), (24, 24, 28))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, tile * 3, bar], fill=(12, 12, 14))
        draw.text((6, 6), f"{verdict}   as-is {as_is:.4f}   mirrored "
                          f"{mirrored:.4f}   sep {separation:.4f}",
                  fill=(255, 255, 120))
        panels = [(Image.fromarray(render).convert("RGB"), "geometry"),
                  (flat, "source"), (flipped, "source mirrored")]
        for i, (image, label) in enumerate(panels):
            canvas.paste(image.resize((tile, tile), Image.LANCZOS), (i * tile, bar))
            draw.text((i * tile + 6, bar + 4), label, fill=(255, 255, 255))
        sheet.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(sheet)
        result["sheet"] = str(sheet)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--yaw", required=True, type=float,
                        help="resolved front yaw, from resolve_front_axis_dino")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sheet", type=Path, default=None)
    args = parser.parse_args(argv)

    result = resolve(args.mesh, args.source, args.yaw, args.sheet)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
