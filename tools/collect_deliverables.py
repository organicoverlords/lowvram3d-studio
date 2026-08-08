"""Collect the finished painted assets into one folder under consistent names.

Two constraints shape this.

The disk has 1.2 GB free of 476 GB, and the thirteen finished assets are about
160 MB. Copying them would spend an eighth of the remaining headroom to store
bytes that already exist, so these are NTFS hard links: a second directory entry
pointing at the same data. They open in any tool as ordinary files and cost
nothing. Deleting one does not touch the other; both names have to go before the
data is freed.

The second is naming. evidence/compare holds whale_final.glb, whale_final2.glb,
whale_final512.glb, whale_deliverable.glb, whale_deliverable_512.glb and
whale_deliverable_v2.glb, and none of those is the painted asset -- they all
predate the paint stage. A name that says which generator and which texture size
cannot lie the way "final" did, so that is what goes in the name, and the
manifest keeps the original path so nothing becomes untraceable.

The manifest also records texels per face, because that is the number that
decides whether a bake can hold detail at all, and three of these are below five.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(r"C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803")
OUT = REPO / "evidence" / "deliverables"

# (source, subject, generator, variant). Variant disambiguates two assets that
# would otherwise land on the same name -- an earlier Mini Turbo paint pass and
# the current one share subject and generator. The first version of this list
# had no variant field and the older shaman silently overwrote the newer one:
# thirteen entries produced twelve files, and the one that survived was the one
# that happened to be later in the list.
ASSETS = [
    ("snail/snail_1024_hypaint.glb",        "snail",  "trellis1024",  ""),
    ("titan/titan_1024_hypaint.glb",        "titan",  "trellis1024",  ""),
    ("whale/whale_t1024_hypaint.glb",       "whale",  "trellis1024",  ""),
    ("heron/heron_t1024_hypaint.glb",       "heron",  "trellis1024",  ""),
    ("shaman/shaman_t1024_hypaint.glb",     "shaman", "trellis1024",  ""),
    ("frog/frog_512_hypaint.glb",           "frog",   "trellis512",   ""),
    ("shaman/shaman_mt_hypaint2048.glb",    "shaman", "miniturbo384", ""),
    ("heron/heron_mt_hypaint2048.glb",      "heron",  "miniturbo384", ""),
    ("fennec/fennec_mt_hypaint2048.glb",    "fennec", "miniturbo384", ""),
    ("boat/stage6_1024_hypaint2048.glb",    "boat",   "trellis1024",  ""),
    ("panda2/panda_hypaint.glb",            "panda",  "miniturbo384", ""),
    ("castle_new/cpu_projection_2048/textured.glb", "castle", "trellis512_tu116_cpu_projection", ""),
    ("whale/whale_hypaint.glb",             "whale",  "miniturbo384", "earlypaint"),
    ("shaman/shaman_hypaint.glb",           "shaman", "miniturbo384", "earlypaint"),
    # "mosstitan", not "moss_titan" -- real_sizes.subject_of() splits the
    # filename on the first underscore, so moss_titan_* would look up "moss",
    # miss the table, and fall through to the 2 m default. This one is 120 m.
    # It is also a different creature from the existing "titan" entry above,
    # which stays at 40 m; sharing the subject name would collide on filename
    # and the collector would refuse the whole run.
    ("moss_titan/titan_t1024_paint2048.glb", "mosstitan", "trellis1024", ""),
    # The same titan before the vendor paint, carrying only TRELLIS's own
    # atlas. Kept as a deliverable so the lineup shows the two side by side:
    # this is the one pair in the set where the only difference is the paint
    # stage, on identical geometry from an identical seed.
    ("moss_titan/titan_t1024.glb",           "mosstitan", "trellis1024", "prepaint"),
    # First asset generated with the billboard gate in the lane. The greentree
    # ahead of it came back from TRELLIS as two crossed cardboard panels with a
    # receipt that said success, which is why check_not_billboard.py now runs
    # between geometry and paint.
    ("sealdiver/sealdiver_t1024_paint2048.glb", "sealdiver", "trellis1024", ""),
    ("sealdiver/sealdiver_t1024.glb",           "sealdiver", "trellis1024", "prepaint"),
]


def measure(path: Path) -> dict:
    """Faces, texture size, and the texels each face actually gets.

    Texels per face is the one number that predicts whether a bake reads as
    surface or as mosaic. Below about five a triangle holds one colour, and once
    chart padding takes its share there is nothing left to hold anything else.
    """
    scene = trimesh.load(path, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    row: dict = {"faces": int(len(mesh.faces))}

    texture = getattr(getattr(mesh.visual, "material", None),
                      "baseColorTexture", None)
    uv = getattr(mesh.visual, "uv", None)
    if texture is None or uv is None:
        row["texture"] = None
        return row

    side = int(texture.size[0])
    corners = np.asarray(uv)[mesh.faces] * side
    edge_a = corners[:, 1] - corners[:, 0]
    edge_b = corners[:, 2] - corners[:, 0]
    area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    row["texture"] = side
    row["texels_per_face_median"] = round(float(np.median(area)), 2)
    row["texels_per_face_p90"] = round(float(np.percentile(area, 90)), 2)
    # The threshold is a judgement, not a measurement, so it is named as one.
    row["texel_starved"] = bool(np.median(area) < 16.0)
    return row


def link(source: Path, target: Path) -> str:
    """Hard link source to target, falling back to a copy across volumes."""
    if target.exists():
        target.unlink()
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         f"New-Item -ItemType HardLink -Path '{target}' -Target '{source}' "
         f"| Out-Null"],
        capture_output=True, text=True)
    if result.returncode == 0 and target.exists():
        return "hardlink"
    target.write_bytes(source.read_bytes())
    return "copy"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    compare = REPO / "evidence" / "compare"
    manifest = []
    bytes_saved = 0

    claimed: dict[str, str] = {}
    for relative, subject, generator, variant in ASSETS:
        source = compare / relative
        if not source.exists():
            print(f"MISSING {relative}")
            continue
        row = measure(source)
        suffix = f"_tex{row['texture']}" if row.get("texture") else ""
        tag = f"_{variant}" if variant else ""
        name = f"{subject}_{generator}{suffix}{tag}.glb"
        # Fail loudly rather than overwrite. A silent collision here deletes a
        # finished asset from the delivery set and leaves the count looking right.
        if name in claimed:
            raise SystemExit(
                f"name collision: {relative} and {claimed[name]} both -> {name}. "
                f"Give one of them a variant tag.")
        claimed[name] = relative
        mode = link(source, OUT / name)
        if mode == "hardlink":
            bytes_saved += source.stat().st_size
        manifest.append({
            "name": name,
            "subject": subject,
            "generator": generator,
            "source": str(source.relative_to(REPO)).replace("\\", "/"),
            "bytes": source.stat().st_size,
            "link": mode,
            **row,
        })
        flag = "  <-- texel starved" if row.get("texel_starved") else ""
        print(f"{name:<44} {mode:<9} faces {row['faces']:>7} "
              f"tex {str(row.get('texture')):>5} "
              f"texels/face {row.get('texels_per_face_median')}{flag}")

    (OUT / "MANIFEST.json").write_text(
        json.dumps({"schema": "lowvram3d_deliverables_v1",
                    "note": ("Hard links into evidence/compare, not copies. "
                             "The source field is the authoritative path."),
                    "assets": manifest}, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} assets, {bytes_saved/1e6:.0f} MB of disk not spent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
