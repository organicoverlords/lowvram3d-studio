"""Decide which way a generated mesh faces, without looking at any paint.

Every criterion this project has used answers a question next to the real one:

    tail colour            which side is the tail on IN THE SOURCE IMAGE
    silhouette IoU         which yaw's OUTLINE matches the source
    painted-texel coverage which camera THE PROJECTION painted from

The third is circular -- it reports where paint landed, so a projection solved
180 degrees backwards is confirmed rather than caught, and on the red panda it
was, 0.8364 against 0.0016, while the photographic face sat on the back of the
character's head. The second is a coin flip on anything close to bilaterally
symmetric: the panda's fit scored 0.915 IoU at a yaw that turned out to be
exactly wrong, because a front outline and a back outline of the same character
are near-identical. Getting a confident number from a symmetric objective is not
evidence.

The asymmetry that actually exists between a front and a back is *shape*: a face
protrudes, a hood does not. So compare the source image against a SHADED RENDER
OF THE GEOMETRY at each candidate yaw, in a feature space that survives the
enormous appearance gap between a painted illustration and a grey clay render.
Raw pixels and gradient energy both fail at this -- gradient energy was tried and
scored the wrong panda 0.081 and the correct whale 0.124, no separation.

DINOv2 does survive it. It is a self-supervised ViT whose features are dominated
by shape and part layout rather than colour or texture, which is exactly the
invariance needed to match "illustration of a face" to "clay render of a face".
The weights are already in the local HF cache, it runs on CPU in seconds, and
nothing about it touches the atlas.

    py workers/resolve_front_axis_dino.py --mesh uv.glb --source crop.png \
       --out front_axis.json --sheet front_axis.png

Validated on two assets whose answer is known independently: the red panda,
whose axis is 180 degrees wrong, and the sky whale, whose axis is correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Yaws to score. The 180 degree ambiguity is the failure being targeted, but
#: sweeping the circle costs seconds and catches a subject that is off by some
#: other angle -- which the silhouette fit would also have got wrong, silently.
DEFAULT_YAWS = tuple(range(0, 360, 15))

#: Render size for the clay pass. DINOv2 resizes to 224 anyway; going much above
#: this only costs rasterisation time.
RENDER = 512

#: Clay shading. Flat enough that DINOv2 sees shape rather than a lighting rig,
#: bright enough that a muzzle reads as a muzzle.
AMBIENT = 0.35
DIFFUSE = 0.65

MODEL = "facebook/dinov2-large"


def _rotation_y(degrees: float):
    import numpy as np

    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def clay_render(vertices, faces, normals, yaw: float, size: int = RENDER):
    """A grey shaded render of the geometry from `yaw`, on a white field."""
    import numpy as np

    rotation = _rotation_y(yaw)
    v = vertices @ rotation.T
    n = normals @ rotation.T

    span = float(np.abs(v[:, :2]).max()) or 1.0
    scale = (size * 0.44) / span
    screen = np.empty((len(v), 2))
    screen[:, 0] = v[:, 0] * scale + size * 0.5
    screen[:, 1] = size * 0.5 - v[:, 1] * scale

    image = np.ones((size, size), np.float64)
    zbuffer = np.full((size, size), -np.inf)

    # Camera looks down -Z after rotation, so +Z is toward the viewer.
    shade = np.clip(np.abs(n[:, 2]), 0.0, 1.0) * DIFFUSE + AMBIENT
    tri = screen[faces]
    depth = v[faces][:, :, 2]
    tri_shade = shade[faces].mean(axis=1)

    order = np.argsort(depth.mean(axis=1))
    for index in order:
        t = tri[index]
        x0, y0 = np.floor(t.min(axis=0)).astype(int)
        x1, y1 = np.ceil(t.max(axis=0)).astype(int) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, size), min(y1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        px, py = xs + 0.5, ys + 0.5
        ax, ay = t[0]
        bx, by = t[1]
        cx, cy = t[2]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-12:
            continue
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * depth[index, 0] + w1 * depth[index, 1] + w2 * depth[index, 2]
        window = zbuffer[y0:y1, x0:x1]
        write = inside & (z > window)
        if not write.any():
            continue
        window[write] = z[write]
        image[y0:y1, x0:x1][write] = tri_shade[index]
    return (np.clip(image, 0, 1) * 255).astype("uint8")


def embed(images, model, processor, torch):
    """DINOv2 CLS embeddings, L2 normalised."""
    import numpy as np
    from PIL import Image

    batch = [Image.fromarray(a).convert("RGB") if not isinstance(a, Image.Image)
             else a.convert("RGB") for a in images]
    inputs = processor(images=batch, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[:, 0]
    vectors = out.cpu().numpy().astype(np.float64)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors


def resolve(mesh_path: Path, source_path: Path, yaws, sheet: Path | None):
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

    # The source is composited onto white so the background cannot be the thing
    # DINOv2 matches on -- the clay renders are on white too.
    source = Image.open(source_path).convert("RGBA")
    flat = Image.new("RGB", source.size, (255, 255, 255))
    flat.paste(source, mask=source.split()[3])

    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()

    renders = [clay_render(vertices, faces, normals, float(y)) for y in yaws]
    source_vec = embed([flat], model, processor, torch)[0]
    render_vecs = embed(renders, model, processor, torch)
    scores = render_vecs @ source_vec

    order = np.argsort(-scores)
    best = int(order[0])
    runner = int(order[1])

    # The number that matters is the separation from the ANTIPODE, not from the
    # runner-up. The failure being detected is a 180 degree error, and the
    # runner-up is almost always the winner's neighbour -- on the panda, 345 beat
    # 0 by 0.0045, which reads as agonising ambiguity and is nothing of the kind:
    # both show the face. Against its antipode the same winner leads 0.4224 to
    # 0.2288. Reporting the runner-up margin here would have retired a criterion
    # that works.
    yaw_array = np.asarray([float(y) % 360.0 for y in yaws])
    antipode = int(np.argmin(np.abs(((yaw_array - (yaw_array[best] + 180.0)) + 180.0) % 360.0 - 180.0)))
    result = {
        "schema": "lowvram3d_front_axis_dino_v1",
        "mesh": str(mesh_path),
        "source": str(source_path),
        "model": MODEL,
        "best_yaw_deg": float(yaws[best]),
        "best_score": round(float(scores[best]), 6),
        "runner_up_yaw_deg": float(yaws[runner]),
        "runner_up_score": round(float(scores[runner]), 6),
        "neighbour_margin": round(float(scores[best] - scores[runner]), 6),
        "antipode_yaw_deg": float(yaws[antipode]),
        "antipode_score": round(float(scores[antipode]), 6),
        "antipode_margin": round(float(scores[best] - scores[antipode]), 6),
        "front_back_separated": bool(scores[best] - scores[antipode] > 0.05),
        "scores": [{"yaw": float(y), "score": round(float(s), 6)}
                   for y, s in zip(yaws, scores)],
        "note": ("geometry-vs-source similarity; no atlas, no paint, no "
                 "silhouette. See docstring for why those three cannot answer "
                 "this."),
    }

    if sheet is not None:
        from PIL import ImageDraw

        tile, bar = 200, 22
        columns = min(8, len(yaws))
        rows = (len(yaws) + columns - 1) // columns
        canvas = Image.new("RGB", (tile * columns, (tile + bar) * rows + tile + bar),
                           (24, 24, 28))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, tile * columns, bar], fill=(12, 12, 14))
        draw.text((6, 6), f"SOURCE  |  best yaw {yaws[best]} "
                          f"(score {scores[best]:.4f}, antipode margin "
                          f"{result['antipode_margin']:.4f})",
                  fill=(255, 255, 120))
        canvas.paste(flat.resize((tile, tile), Image.LANCZOS), (0, bar))
        for i, (y, render) in enumerate(zip(yaws, renders)):
            column, row = i % columns, i // columns
            x, top = column * tile, tile + bar + row * (tile + bar)
            colour = (40, 90, 40) if i == best else (12, 12, 14)
            draw.rectangle([x, top, x + tile, top + bar], fill=colour)
            draw.text((x + 5, top + 5), f"{y}deg  {scores[i]:.4f}",
                      fill=(255, 255, 255))
            canvas.paste(Image.fromarray(render).convert("RGB").resize((tile, tile)),
                         (x, top + bar))
        sheet.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(sheet)
        result["sheet"] = str(sheet)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sheet", type=Path, default=None)
    parser.add_argument("--yaws", type=str, default="",
                        help="comma-separated degrees; default sweeps 0..345 by 15")
    args = parser.parse_args(argv)

    yaws = ([float(v) for v in args.yaws.split(",")] if args.yaws
            else list(DEFAULT_YAWS))
    result = resolve(args.mesh, args.source, yaws, args.sheet)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "scores"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
