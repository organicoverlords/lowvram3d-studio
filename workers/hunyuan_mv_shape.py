"""Generate geometry from several named views with Hunyuan3D-2mv Turbo.

The Lucky Drown boat came with a reference sheet carrying a 3/4 hero view, plus
labelled FRONT, SIDE, BACK, a cutaway section and a deck plan. Everything
downstream was built from a crop of the hero panel alone -- a strongly oblique
view, which is the worst single view to hand a single-image reconstructor if you
want a hull. The result was a skewed parallelogram hull with a featureless lid
where the deck should be. Texturing was never the problem; the geometry was.

`hunyuan3d-dit-v2-mv-turbo` takes named views rather than one image, and it was
already sitting in the local HF tree, untested. The orthographic FRONT, SIDE and
BACK panels are exactly what it wants, and they were sitting unused in the same
reference sheet the hero crop came from.

From the operator's side this is still one picture in: the panels are extracted
from the sheet automatically.

    py workers/hunyuan_mv_shape.py --front f.png --left s.png --back b.png \
       --out boat_mv.glb

View tags come from MVImageProcessorV2: front=0, left=1 (front clockwise 90),
back=2, right=3. Which physical side of an asymmetric subject a given SIDE panel
represents is genuinely ambiguous from the sheet, so it is a flag, not a guess
buried in the code.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL_ROOT = Path(r"C:\AI\HY3D2\HuggingFaceHub\hunyuan3d-2mv-direct")
SUBFOLDER = "hunyuan3d-dit-v2-mv-turbo"

#: Tencent's own Turbo defaults. Deliberately not tuned before a first result
#: exists -- this project has lost more time to premature tuning than to any
#: model limitation.
STEPS = 5
GUIDANCE = 5.0
OCTREE = 256
CHUNKS = 1500


def run(views: dict, out_path: Path, steps: int, guidance: float,
        octree: int, chunks: int, seed: int,
        flashvdm: bool = False) -> dict:
    import torch

    torch.backends.cudnn.enabled = False  # FP16 cuDNN convolution is defective here

    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    started = time.time()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(MODEL_ROOT), subfolder=SUBFOLDER, use_safetensors=True, device="cuda")
    # FlashVDM's hierarchical decoder empties its candidate set on this subject
    # and dies in volume_decoders.py with "Expected reduction dim 0 to have
    # non-zero size" -- after the diffusion has already been paid for. Off by
    # default; it is a decode-speed optimisation, not a quality setting.
    if flashvdm:
        pipeline.enable_flashvdm(topk_mode="merge")
    loaded = time.time()

    images = {tag: Image.open(path).convert("RGBA") for tag, path in views.items()}
    mesh = pipeline(
        image=images,
        num_inference_steps=steps,
        guidance_scale=guidance,
        octree_resolution=octree,
        num_chunks=chunks,
        generator=torch.manual_seed(seed),
    )[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    finished = time.time()

    return {
        "schema": "lowvram3d_hunyuan_mv_shape_v1",
        "checkpoint": SUBFOLDER,
        "views": {k: str(v) for k, v in views.items()},
        "mesh_out": str(out_path),
        "steps": steps, "guidance_scale": guidance,
        "octree_resolution": octree, "num_chunks": chunks, "seed": seed, "flashvdm": bool(flashvdm),
        "faces": int(len(mesh.faces)),
        "load_seconds": round(loaded - started, 1),
        "shape_seconds": round(finished - loaded, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--left", type=Path, default=None)
    parser.add_argument("--back", type=Path, default=None)
    parser.add_argument("--right", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--guidance", type=float, default=GUIDANCE)
    parser.add_argument("--octree", type=int, default=OCTREE)
    parser.add_argument("--chunks", type=int, default=CHUNKS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--flashvdm", action="store_true",
                        help="faster decode; empties its candidate set on some subjects")
    args = parser.parse_args(argv)

    views = {tag: getattr(args, tag) for tag in ("front", "left", "back", "right")
             if getattr(args, tag) is not None}
    result = run(views, args.out, args.steps, args.guidance,
                 args.octree, args.chunks, args.seed, flashvdm=args.flashvdm)
    args.out.with_suffix(".mv.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
