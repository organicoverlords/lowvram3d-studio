"""Score a render against its source image with small local vision models.

Pixel correlation is a poor judge of a reconstruction. It is dominated by
exposure, it collapses when the render is darker than the source, and it cannot
tell "the same scene, slightly wrong geometry" from "a different scene at the
same brightness". Two small models already cached locally do much better, and
both run on CPU in a couple of seconds:

``DINO ViT-B/16``
    Self-supervised features that respond to structure and layout. This is the
    primary signal for "is the geometry right".

``CLIP ViT-B/32``
    Semantic embedding. Answers "is this still the same scene", and via text
    prompts can flag specific failure modes -- a black frame, a flat billboard,
    stretched smearing -- without a hand-written detector for each.

Neither is a ground-truth metric. They are a cheap, repeatable critic that can
rank candidate reconstructions inside a loop, which is what iterating needs.

    py -3.12 -m lowvram3d.perceptual_critic --render r.png --source s.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DINO_MODEL = "facebook/dino-vitb16"
CLIP_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

# Probes for the failure modes this pipeline actually produces. Scored as a
# softmax over the whole list, so they compete rather than firing in isolation.
DEFECT_PROMPTS = [
    "a detailed three-dimensional rendered scene",
    "a completely black image",
    "a flat photograph pasted onto a surface",
    "a stretched smeared distorted mesh",
    "an empty grey background",
]


# Loading both checkpoints costs seconds; scoring one image pair costs
# milliseconds. Cache them so a sweep or a per-stage gate stays cheap.
_MODELS: dict[str, Any] = {}


def _dino(device: str):
    if "dino" not in _MODELS:
        from transformers import AutoImageProcessor, AutoModel

        _MODELS["dino"] = (AutoImageProcessor.from_pretrained(DINO_MODEL),
                           AutoModel.from_pretrained(DINO_MODEL).to(device).eval())
    return _MODELS["dino"]


def _clip(device: str):
    if "clip" not in _MODELS:
        from transformers import CLIPModel, CLIPProcessor

        _MODELS["clip"] = (CLIPProcessor.from_pretrained(CLIP_MODEL),
                           CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval())
    return _MODELS["clip"]


def _load_images(render: Path, source: Path):
    from PIL import Image

    render_image = Image.open(render).convert("RGB")
    source_image = Image.open(source).convert("RGB")
    if source_image.size != render_image.size:
        source_image = source_image.resize(render_image.size, Image.LANCZOS)
    return render_image, source_image


def _cosine(a, b) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


def score(render: Path, source: Path, device: str = "cpu") -> dict[str, Any]:
    import time

    import torch

    started = time.perf_counter()

    render_image, source_image = _load_images(render, source)
    report: dict[str, Any] = {
        "schema_version": "perceptual_critic_v1",
        "render": str(render),
        "source": str(source),
    }

    with torch.no_grad():
        try:
            processor, model = _dino(device)
            batch = processor(images=[render_image, source_image], return_tensors="pt").to(device)
            # CLS token: a global descriptor of structure and layout.
            features = model(**batch).last_hidden_state[:, 0]
            report["dino_similarity"] = _cosine(features[0:1], features[1:2])
        except Exception as exc:
            report["dino_error"] = f"{type(exc).__name__}: {exc}"

        try:
            processor, model = _clip(device)
            images = processor(images=[render_image, source_image], return_tensors="pt").to(device)
            embeddings = model.get_image_features(**images)
            # transformers 5.x returns a model output here rather than a tensor.
            if not isinstance(embeddings, torch.Tensor):
                embeddings = getattr(embeddings, "image_embeds", None) \
                    or getattr(embeddings, "pooler_output")
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            report["clip_similarity"] = _cosine(embeddings[0:1], embeddings[1:2])

            probe = processor(text=DEFECT_PROMPTS, images=render_image,
                              return_tensors="pt", padding=True).to(device)
            logits = model(**probe).logits_per_image.softmax(dim=-1)[0]
            report["defect_probabilities"] = {
                prompt: round(float(value), 4)
                for prompt, value in zip(DEFECT_PROMPTS, logits)
            }
            report["most_likely_description"] = DEFECT_PROMPTS[int(logits.argmax())]
        except Exception as exc:
            report["clip_error"] = f"{type(exc).__name__}: {exc}"

    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    dino = report.get("dino_similarity")
    clip = report.get("clip_similarity")
    if dino is not None and clip is not None:
        # DINO weighted higher: geometry is what a reconstruction must get right.
        report["combined_score"] = round(0.6 * dino + 0.4 * clip, 4)
        report["verdict"] = (
            "GOOD" if report["combined_score"] >= 0.75 else
            "USABLE" if report["combined_score"] >= 0.55 else
            "POOR"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--receipt", default=None)
    args = parser.parse_args(argv)

    report = score(Path(args.render), Path(args.source))
    if args.receipt:
        path = Path(args.receipt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
