"""Discover and index candidate benchmark assets, read-only.

Calibrating a detector against the one asset it was written to reject is circular: the threshold
lands wherever it must to produce the verdict already decided on. This walks the machine's existing
output for assets produced by other tools and earlier runs, records what can be established about
each, and - critically - records how confidently a model can be tied to the image it was made from.

Nothing here is modified, renamed, moved or converted. The scan opens files for reading only, and
every report is written outside the scanned roots.

Pairing is evidence-based and deliberately pessimistic. Two files sharing a folder is not evidence:
a folder of twenty generations and twenty unrelated reference pictures would otherwise produce
twenty confident and entirely fictional pairs, and those fictions would then set the thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

# This tool's own reports name every model and every source it found, so re-scanning one turns it
# into apparent evidence for everything in it. Run two of the scan paired all 146 meshes to a single
# image on the strength of run one's manifest alone.
SELF_REPORT_NAMES = {"benchmark_manifest.json", "source_fidelity_calibration.json"}

MODEL_EXTENSIONS = {".glb", ".gltf", ".fbx", ".obj", ".ply", ".stl", ".blend"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
METADATA_EXTENSIONS = {".json", ".txt", ".yaml", ".yml", ".toml", ".log"}

# Standing prohibition: shaman.fbx may not be used for generation, reconstruction, cleanup, retopo,
# UV, texture, rigging or validation. Benchmarking is validation, so it is excluded here rather than
# left to be remembered later.
FORBIDDEN_BASENAMES = {"shaman.fbx"}

PROVEN, HIGH, POSSIBLE, UNPAIRED = "PROVEN", "HIGH", "POSSIBLE", "UNPAIRED"


def sha256_file(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            if limit and handle.tell() > limit:
                break
    return digest.hexdigest()


def glb_counts(path: Path) -> dict:
    """Triangle and accessor counts straight from the glTF JSON chunk, without loading buffers."""
    try:
        with path.open("rb") as handle:
            magic, _, _ = struct.unpack("<III", handle.read(12))
            if magic != 0x46546C67:
                return {"parsed": False, "reason": "not a binary glTF"}
            length, kind = struct.unpack("<II", handle.read(8))
            if kind != 0x4E4F534A:
                return {"parsed": False, "reason": "first chunk is not JSON"}
            document = json.loads(handle.read(length).decode("utf-8"))
    except (OSError, ValueError, struct.error) as error:
        return {"parsed": False, "reason": str(error)[:200]}

    accessors = document.get("accessors", [])
    triangles = 0
    primitives = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            primitives += 1
            index = primitive.get("indices")
            if index is not None and index < len(accessors):
                triangles += accessors[index].get("count", 0) // 3
    images = document.get("images", [])
    return {
        "parsed": True,
        "triangles": triangles,
        "meshes": len(document.get("meshes", [])),
        "primitives": primitives,
        "materials": len(document.get("materials", [])),
        "textures": len(document.get("textures", [])),
        "images": len(images),
        "has_texture": bool(images),
        "generator": (document.get("asset") or {}).get("generator"),
    }


def classify_asset(name: str) -> str:
    lowered = name.lower()
    table = [
        (("tree", "forest", "shrub", "rock", "landscape", "brick", "ruin", "wall", "trench"),
         "environment_piece"),
        (("hevonen", "hepo", "horse", "lepakko", "bat", "elukat"), "creature"),
        (("bird", "lintu", "shaman"), "humanoid"),
        (("panda",), "quadruped"),
        (("door", "arch", "pack"), "prop"),
    ]
    for keys, label in table:
        if any(key in lowered for key in keys):
            return label
    return "unknown"


def read_metadata(path: Path) -> dict | None:
    try:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return {"_text": path.read_text(encoding="utf-8", errors="replace")[:20000]}
    except (OSError, ValueError):
        return None


# Names a pipeline gives its own products. A document that mentions a model and a normal map is
# describing one asset's outputs, not recording what it was generated from.
DERIVED_ARTEFACT_MARKERS = (
    "normal", "basecolor", "base_color", "orm", "roughness", "metallic", "cavity", "_ao",
    "ambient", "material_id", "overlay", "debug", "coverage", "matte", "atlas", "lod",
    "render", "preview", "thumbnail", "texture", "diffuse", "specular", "displacement", "bump",
)

SOURCE_KEYS = ("source_image", "input_image", "prompt_image", "conditioning_image",
               "reference_image", "source", "input", "src_image")


def _mentions(node, needle: str) -> bool:
    if isinstance(node, str):
        return needle in node.lower()
    if isinstance(node, dict):
        return any(_mentions(v, needle) for v in node.values()) or \
               any(needle in k.lower() for k in node)
    if isinstance(node, list):
        return any(_mentions(v, needle) for v in node)
    return False


def scoped_source_reference(metadata, model_name: str,
                            image_index: dict[str, list[Path]]) -> tuple[str, Path] | None:
    """Find a source image recorded in the same record as this model, not merely the same file.

    Scope is the whole point. Searching a document globally means one manifest that lists forty
    meshes and names a single source pairs all forty to it - which is how every model on this
    machine came back "PROVEN" against one picture. A pair is only evidence when the source and the
    output are described together.
    """
    needle = model_name.lower()

    def descend(node):
        if not isinstance(node, (dict, list)):
            return None
        children = node.values() if isinstance(node, dict) else node
        # The deepest record mentioning the model is the one describing it, and it is the only
        # scope that counts. Falling back to an ancestor when that record carries no source of its
        # own is what let a manifest's single top-level source reach every mesh listed beneath it -
        # the leak this function exists to close, so there is deliberately no fallback here.
        mentioning = [c for c in children if isinstance(c, (dict, list)) and _mentions(c, needle)]
        if mentioning:
            for child in mentioning:
                result = descend(child)
                if result:
                    return result
            return None
        if isinstance(node, dict) and _mentions(node, needle):
            return find_source_reference(node, image_index)
        return None

    return descend(metadata)


def find_source_reference(metadata: dict, image_index: dict[str, list[Path]]) -> tuple[str, Path] | None:
    """Find an explicit source-image reference, refusing anything that looks like an output.

    Only values sitting under an explicitly source-named key qualify. An earlier version also
    accepted any bare string that happened to end in an image extension, which turned every
    receipt's own normal map into a "PROVEN" source and produced 109 confident pairings out of
    44 distinct meshes - a corpus that would have calibrated the thresholds onto the very asset it
    was meant to judge.
    """
    found: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and key.lower() in SOURCE_KEYS:
                    found.append(value)
                elif isinstance(value, dict) and key.lower() in SOURCE_KEYS:
                    path_value = value.get("path")
                    if isinstance(path_value, str):
                        found.append(path_value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(metadata)
    for reference in found:
        lowered = reference.lower()
        if not any(lowered.endswith(e) for e in IMAGE_EXTENSIONS):
            continue
        if any(marker in Path(lowered).name for marker in DERIVED_ARTEFACT_MARKERS):
            continue
        candidate = Path(reference)
        if candidate.is_file():
            return "explicit_path", candidate
        matches = image_index.get(candidate.name.lower(), [])
        if len(matches) == 1:
            return "explicit_name", matches[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True,
                        help="directory to scan read-only; repeatable")
    parser.add_argument("--output", required=True, help="manifest path, written outside the roots")
    parser.add_argument("--max-depth", type=int, default=12)
    args = parser.parse_args()

    roots = [Path(r) for r in args.root if Path(r).exists()]
    output_path = Path(args.output).resolve()
    models: list[Path] = []
    images: list[Path] = []
    metadata_files: list[Path] = []

    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            if depth >= args.max_depth:
                dirnames[:] = []
            for filename in filenames:
                path = Path(dirpath) / filename
                suffix = path.suffix.lower()
                if suffix in MODEL_EXTENSIONS:
                    models.append(path)
                elif suffix in IMAGE_EXTENSIONS:
                    images.append(path)
                elif suffix in METADATA_EXTENSIONS:
                    if filename.lower() in SELF_REPORT_NAMES or path == output_path:
                        continue
                    metadata_files.append(path)

    image_index: dict[str, list[Path]] = {}
    for image in images:
        image_index.setdefault(image.name.lower(), []).append(image)

    # Index every metadata document once, keyed by the model filenames it mentions.
    references: dict[str, list[tuple[Path, dict]]] = {}
    for meta_path in metadata_files:
        data = read_metadata(meta_path)
        if data is None:
            continue
        blob = json.dumps(data) if not isinstance(data, dict) or "_text" not in data else data["_text"]
        for model in models:
            if model.name.lower() in blob.lower():
                references.setdefault(model.name.lower(), []).append((meta_path, data))

    entries = []
    seen_hashes: dict[str, str] = {}
    for model in sorted(models):
        if model.name.lower() in FORBIDDEN_BASENAMES:
            entries.append({
                "path": str(model),
                "excluded": True,
                "exclusion_reason": "standing prohibition: shaman.fbx may not be used for any "
                                    "purpose including validation",
            })
            continue

        try:
            size = model.stat().st_size
        except OSError:
            continue
        digest = sha256_file(model)
        duplicate_of = seen_hashes.get(digest)
        seen_hashes.setdefault(digest, str(model))

        entry = {
            "path": str(model),
            "sha256": digest,
            "bytes": size,
            "format": model.suffix.lower().lstrip("."),
            "duplicate_of": duplicate_of,
            "asset_class_guess": classify_asset(model.name + " " + model.parent.name),
            "excluded": False,
        }
        if model.suffix.lower() == ".glb":
            entry["geometry"] = glb_counts(model)

        pairing = {"confidence": UNPAIRED, "source_image": None, "evidence": None}
        for meta_path, data in references.get(model.name.lower(), []):
            reference = scoped_source_reference(data, model.name, image_index)
            if reference:
                basis, image_path = reference
                pairing = {"confidence": PROVEN, "source_image": str(image_path),
                           "evidence": f"{meta_path.name} ({basis})"}
                break

        if pairing["confidence"] == UNPAIRED:
            # A tool that writes its input beside its output is real evidence of a pair; a folder
            # that merely contains both a model and some pictures is not.
            siblings = [p for p in model.parent.iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS] if model.parent.is_dir() else []
            inputs = [p for p in siblings if p.stem.lower() in ("input", "source", "reference", "image")]
            stem_matches = [p for p in siblings if p.stem.lower() == model.stem.lower()]
            if len(stem_matches) == 1:
                pairing = {"confidence": HIGH, "source_image": str(stem_matches[0]),
                           "evidence": "unique matching basename in the same folder"}
            elif len(inputs) == 1 and len(siblings) <= 3:
                pairing = {"confidence": HIGH, "source_image": str(inputs[0]),
                           "evidence": f"tool output folder containing a single {inputs[0].name}"}
            elif siblings:
                pairing = {"confidence": POSSIBLE, "source_image": None,
                           "evidence": f"{len(siblings)} images share the folder; no linking evidence"}

        entry["pairing"] = pairing
        entries.append(entry)

    usable = [e for e in entries if not e.get("excluded")
              and e["pairing"]["confidence"] in (PROVEN, HIGH)]
    counts: dict[str, int] = {}
    for entry in entries:
        if entry.get("excluded"):
            counts["EXCLUDED"] = counts.get("EXCLUDED", 0) + 1
        else:
            key = entry["pairing"]["confidence"]
            counts[key] = counts.get(key, 0) + 1

    manifest = {
        "roots": [str(r) for r in roots],
        "read_only": True,
        "models_found": len(models),
        "images_found": len(images),
        "metadata_files_found": len(metadata_files),
        "pairing_counts": counts,
        "calibration_usable_pairs": len(usable),
        "note": "Only PROVEN and HIGH pairs may calibrate source-fidelity metrics. POSSIBLE pairs "
                "are listed for review and must not affect thresholds.",
        "entries": entries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"BENCHMARK_DISCOVER models={len(models)} images={len(images)} "
          f"metadata={len(metadata_files)} pairing={counts} "
          f"calibration_usable={len(usable)}", flush=True)


if __name__ == "__main__":
    main()
