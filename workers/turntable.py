"""Spin a textured GLB into an animated WebP, and index the results as one page.

Looking at these assets meant opening Blender, which is slow and which nobody
does casually. A seven-view contact sheet is cheaper but it answers the wrong
question: it shows seven frozen angles, and the failures that matter here --
a seam where two projected views disagree, a pendant textured from the wrong
side, an atlas that is coherent from the front and patchwork everywhere else --
are all things you notice when the model *moves* and a surface changes character
as it rotates.

So: a turntable. It reuses `preview_textured_mesh`'s rasteriser rather than
adding a renderer, which keeps this honest -- what you see here is what that
tool sees, including its emissive compositing.

WebP rather than GIF. GIF quantises to 256 colours, and these atlases are
weathered cloth and bone in narrow tonal bands, which is precisely the content
a 256-colour palette destroys: it posterises the robe into flat plates that look
exactly like the projection artefact you are checking for. WebP is lossy but it
is not palettised, so a smooth gradient stays smooth.

`--index` then writes a single self-contained HTML page pointing at whatever
turntables exist, so the whole set can be reviewed in a browser with no server,
no CDN and no build step.

    py turntable.py --glb asset.glb --out asset.webp
    py turntable.py --index evidence/compare --out evidence/compare/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Frames in a full revolution. 24 is a 15 degree step.
#:
#: 36 frames at 420 px took THIRTY MINUTES on the shaman, because
#: `preview_textured_mesh` rasterises triangle by triangle in a Python loop, so
#: cost is linear in faces and there are 400,000 of them. Frame count and
#: resolution barely help against that -- `--preview-faces` does, and it is on
#: by default. Nobody inspects a turntable at full geometric fidelity; they are
#: looking at the texture.
#: A turntable that takes longer than two minutes does not get made, so the
#: defaults are chosen to fit that budget rather than to look their best.
DEFAULT_FRAMES = 20

#: Milliseconds per frame. 80 ms spun the shaman fast enough to be useless for
#: inspection. 160 is ~6 fps, which reads as a deliberate turn.
DEFAULT_DURATION = 160

#: Decimate to this many faces before rendering. At a few hundred pixels the
#: silhouette difference is invisible and the render is roughly 5x faster.
#: 0 disables it.
DEFAULT_PREVIEW_FACES = 30_000

#: Shading, overriding `preview_textured_mesh`'s own constants for the duration
#: of the render.
#:
#: That module uses AMBIENT 0.60 with DIFFUSE 0.50, so a surface facing the
#: camera reaches 1.10 and clips. On a contact sheet that is a deliberate choice
#: -- it asks "what colour is on this surface", and flat bright lighting answers
#: it. On a turntable it washes the asset out, and the shaman's bone-and-cloth
#: palette is already pale. Ambient down and diffuse up keeps the peak at 1.0
#: and lets the form read as it turns.
PREVIEW_AMBIENT = 0.42
PREVIEW_DIFFUSE = 0.58

#: A gentle downward tilt. A dead-level camera hides the top of a head and the
#: upper surfaces of shoulders, which is where projection seams tend to sit.
DEFAULT_ELEVATION = -0.18


def _decimate(scene, target: int):
    """Reduce face count for preview only. Returns the scene unchanged on failure."""
    import numpy as np
    import trimesh

    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    if target <= 0 or len(mesh.faces) <= target:
        return scene, len(mesh.faces), len(mesh.faces)
    original = len(mesh.faces)
    try:
        reduced = mesh.simplify_quadric_decimation(face_count=target)
    except Exception:
        # Not fatal: a slow turntable still beats no turntable.
        return scene, original, original
    if getattr(mesh.visual, "uv", None) is not None and len(reduced.faces):
        # trimesh carries UVs through when it can; if it did not, keep the
        # original rather than render an untextured spin.
        if getattr(reduced.visual, "uv", None) is None:
            return scene, original, original
    return reduced, original, len(reduced.faces)


def render_turntable(glb: Path, out: Path, size: int, frames: int,
                     duration: int, elevation: float,
                     preview_faces: int = DEFAULT_PREVIEW_FACES) -> dict:
    import numpy as np
    import trimesh
    from PIL import Image

    import preview_textured_mesh as preview

    preview.AMBIENT, preview.DIFFUSE = PREVIEW_AMBIENT, PREVIEW_DIFFUSE

    scene, faces_before, faces_after = _decimate(
        trimesh.load(glb, process=False), preview_faces)
    parts = preview.collect(scene)
    if not parts:
        raise SystemExit(f"NO_GEOMETRY: {glb}")

    images = []
    for step in range(frames):
        angle = 2.0 * np.pi * step / frames
        forward = (float(-np.sin(angle)), elevation, float(-np.cos(angle)))
        pixels, _covered = preview.render(parts, forward, (0.0, 1.0, 0.0), size)
        images.append(Image.fromarray(
            (np.clip(pixels, 0.0, 1.0) * 255).astype(np.uint8)))

    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=duration, loop=0, quality=82, method=4)
    return {
        "glb": str(glb),
        "out": str(out),
        "frames": frames,
        "size": size,
        "duration_ms": duration,
        "elevation": elevation,
        "faces_rendered": faces_after,
        "faces_original": faces_before,
        "ambient": PREVIEW_AMBIENT,
        "diffuse": PREVIEW_DIFFUSE,
        "bytes": out.stat().st_size,
    }


PAGE = """<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 2rem;
          background: #14141a; color: #e8e8ee; }}
  h1 {{ font-size: 1.3rem; font-weight: 600; margin: 0 0 .3rem; }}
  p.sub {{ margin: 0 0 2rem; opacity: .6; }}
  .grid {{ display: grid; gap: 1.5rem;
           grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }}
  figure {{ margin: 0; background: #1e1e26; border-radius: 10px;
            overflow: hidden; }}
  img {{ display: block; width: 100%; height: auto; background: #fff; }}
  figcaption {{ padding: .7rem .9rem; font-size: .85rem; }}
  .name {{ font-weight: 600; }}
  .meta {{ opacity: .55; font-size: .78rem; margin-top: .15rem; }}
</style>
<h1>{title}</h1>
<p class="sub">{count} turntable(s). Click an image to open it full size.</p>
<div class="grid">
{cards}
</div>
"""

CARD = """  <figure>
    <a href="{href}"><img src="{href}" alt="{name}" loading="lazy"></a>
    <figcaption><div class="name">{name}</div>
      <div class="meta">{meta}</div></figcaption>
  </figure>"""


#: Turntables first, then contact sheets. A turntable is the better artefact but
#: it costs minutes per asset, and a review page that waits for all of them is a
#: page nobody looks at.
INDEX_PATTERNS = ("*.webp", "*7view.png", "*contact*.png")


def build_index(root: Path, out: Path, title: str) -> dict:
    turntables = []
    for pattern in INDEX_PATTERNS:
        turntables.extend(sorted(root.rglob(pattern)))
    cards = []
    for path in turntables:
        try:
            rel = path.relative_to(out.parent)
        except ValueError:
            rel = path.resolve()
        size_kb = path.stat().st_size / 1024.0
        cards.append(CARD.format(
            href=html.escape(str(rel).replace("\\", "/")),
            name=html.escape(path.stem),
            meta=f"{size_kb:,.0f} KB"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.format(title=html.escape(title), count=len(cards),
                               cards="\n".join(cards)), encoding="utf-8")
    return {"index": str(out), "turntables": len(cards),
            "files": [str(p) for p in turntables]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", help="Textured GLB to spin.")
    parser.add_argument("--index", help="Directory to scan for turntables "
                                        "instead of rendering one.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="LowVRAM3D turntables")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    parser.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION)
    parser.add_argument("--preview-faces", type=int, default=DEFAULT_PREVIEW_FACES,
                        help="Decimate to this many faces before rendering. "
                             "0 renders full geometry, which on 400k faces is "
                             "tens of minutes.")
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    if bool(args.glb) == bool(args.index):
        raise SystemExit("Pass exactly one of --glb or --index.")

    if args.index:
        receipt = build_index(Path(args.index), Path(args.out), args.title)
    else:
        receipt = render_turntable(Path(args.glb), Path(args.out), args.size,
                                   args.frames, args.duration, args.elevation,
                                   args.preview_faces)

    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
