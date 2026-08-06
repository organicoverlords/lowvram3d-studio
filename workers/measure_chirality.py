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

BUT NOT THE CLS VECTOR. The first working version of this file compared pooled
CLS embeddings and returned MATCHES_SOURCE for the real mesh at separation
0.0144, which looked like a result. Run as a control against a deliberately
mirrored copy of the same mesh, it returned MATCHES_SOURCE again, at a *larger*
separation of 0.0155. The criterion does not detect mirroring at all.

The reason is not subtle in hindsight: DINOv2's self-supervised training uses
random horizontal flip as an augmentation, so its pooled features are trained to
be invariant to precisely the thing being measured here. Asking a flip-invariant
embedding about handedness cannot work.

The spatial tokens are not flip-invariant, because they are position-indexed: a
horizontal flip permutes the patch grid. So this compares the PATCH TOKEN GRIDS
with spatial correspondence -- token (i,j) of the render against token (i,j) of
the source -- rather than two pooled summaries. The invariance that defeats the
CLS comparison is exactly what the registration removes.

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

from resolve_front_axis_dino import MODEL, RENDER, clay_render

#: A source and its mirror are the same picture to any shape-blind statistic, so
#: the two scores are close by construction and the threshold has to be modest.
#: But it cannot be zero: a subject that really is bilaterally symmetric from the
#: front will produce a coin flip, and calling that a verdict is how the earlier
#: wrong answers were reached. Below this, say UNRESOLVED.
#:
#: Class-level tuning belongs here. A character or a hand-held item has sculpted
#: handedness and should clear this comfortably; a building or a landscape is
#: often genuinely symmetric from the front, and for those UNRESOLVED is the
#: correct answer rather than a failure. Per-asset tuning does not belong here
#: and never will.
MIN_SEPARATION = 0.01


def embed_grid(images, model, processor, torch):
    """Spatially-registered DINOv2 patch-token grids, L2 normalised per token.

    The CLS token is dropped deliberately. See the module docstring: pooled
    DINOv2 features are trained to be horizontal-flip invariant, so they cannot
    answer a handedness question. The patch grid can, because flipping the image
    permutes it.
    """
    import numpy as np
    from PIL import Image

    batch = [image.convert("RGB") if isinstance(image, Image.Image)
             else Image.fromarray(image).convert("RGB") for image in images]
    inputs = processor(images=batch, return_tensors="pt")
    with torch.no_grad():
        tokens = model(**inputs).last_hidden_state[:, 1:]
    grids = tokens.cpu().numpy().astype(np.float64)
    grids /= np.maximum(np.linalg.norm(grids, axis=-1, keepdims=True), 1e-12)
    return grids


def registered_similarity(a, b):
    """Mean cosine between patch tokens at the SAME grid position."""
    import numpy as np

    return float(np.mean(np.sum(a * b, axis=-1)))


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
    grids = embed_grid([render, flat, flipped], model, processor, torch)
    as_is = registered_similarity(grids[0], grids[1])
    mirrored = registered_similarity(grids[0], grids[2])

    separation = abs(as_is - mirrored)
    if separation < MIN_SEPARATION:
        verdict = "UNRESOLVED"
    elif mirrored > as_is:
        verdict = "MIRRORED"
    else:
        verdict = "MATCHES_SOURCE"

    result = {
        "schema": "lowvram3d_chirality_v3",
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
