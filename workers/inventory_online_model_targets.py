"""Inventory and safely copy the downloaded ~1.5M-triangle online GLB targets."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

DOWNLOADS = Path(r"C:\Users\Lauri\Downloads")
TARGET = Path(r"C:\AI\LowVRAM3D-benchmarks\production\panda_online_model_targets_20260805")
REPORT = TARGET / "online_15m_model_inventory.json"

LABELS = {
    "20260730004905_ab3519e2.glb": ("fox_scout", "fox-like scout with rifle and tail; not a panda"),
    "besgu.glb": ("online_generated_target", "user-selected online-generated target model; identity label intentionally neutral"),
    "shaman33.glb": ("pale_hanging_shaman_variant", "same pale hanging shaman design, separate file/hash; not panda"),
    "ukelo.glb": ("pale_hanging_shaman_duplicate", "same visual asset as 20260730010641_ff737cfe.glb by byte hash"),
    "20260730010641_ff737cfe.glb": ("pale_hanging_shaman", "pale hanging creature/shaman; not panda"),
    "20260803013637_d0c5e5c0.glb": ("white_cat", "low quadruped cat/rodent-like creature; not panda"),
    "20260729221745_18e55b4b.glb": ("ornate_tower", "architectural tower; not character"),
    "20260729225240_853e26c9.glb": ("worm_character", "cartoon worm character"),
    "20260729232401_489f406b.glb": ("monkey_soldier", "monkey/ape soldier with rifle"),
    "20260730002405_cce2c202.glb": ("frog_astronaut", "frog/amphibian astronaut"),
    "jatti.glb": ("white_mount", "white horse-like mount; front render is rear/ambiguous"),
    "lepakko.glb": ("bat_creature", "flying bat-like creature"),
    "mustehevonen.glb": ("skeletal_mount", "skeletal horse/creature"),
    "valkohepo.glb": ("white_horse", "white horse-like mount"),
    "junu.glb": ("robot", "small armored robot"),
    "20260731003822_5a04b9c3.glb": ("ice_humanoid", "white icy humanoid"),
    "20260731110846_2c0ee02d.glb": ("armored_humanoid", "armored humanoid with rifle"),
}


def glb_json(path: Path) -> dict:
    blob = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", blob, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glTF 2.0 binary")
    pos = 12
    doc = None
    while pos < total:
        size, kind = struct.unpack_from("<II", blob, pos)
        pos += 8
        chunk = blob[pos : pos + size]
        pos += size
        if kind == 0x4E4F534A:
            doc = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\0"))
    if doc is None:
        raise ValueError("missing JSON chunk")
    return doc


def stats(path: Path) -> dict:
    doc = glb_json(path)
    accessors = doc.get("accessors", [])
    triangles = vertices = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "indices" in prim:
                triangles += int(accessors[prim["indices"]].get("count", 0)) // 3
            if "POSITION" in prim.get("attributes", {}):
                vertices += int(accessors[prim["attributes"]["POSITION"]].get("count", 0))
    return {
        "triangles": triangles,
        "vertices": vertices,
        "embedded_images": len(doc.get("images", [])),
        "materials": len(doc.get("materials", [])),
    }


def main() -> None:
    candidates = []
    for path in sorted(DOWNLOADS.rglob("*.glb")):
        try:
            info = stats(path)
        except Exception:
            continue
        if not (1_450_000 <= info["triangles"] <= 1_510_000 and info["embedded_images"] >= 3):
            continue
        category, identity = LABELS.get(path.name, ("unclassified_15m_model", "1.5M textured online GLB; visual identity not yet reviewed"))
        candidates.append({
            "source_path": str(path),
            "source_name": path.name,
            "category": category,
            "identity": identity,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            **info,
        })
    TARGET.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in candidates:
        source = Path(item["source_path"])
        destination = TARGET / f"{category}__{source.name}"
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
            destination = TARGET / f"{source.stem}__{item['sha256'][:8]}{source.suffix}"
        shutil.copy2(source, destination)
        item["target_path"] = str(destination)
        copied.append(item)
    report = {
        "purpose": "Inventory and safe copy of online-generated ~1.5M-triangle textured GLBs",
        "source_root": str(DOWNLOADS),
        "target_root": str(TARGET),
        "originals_preserved": True,
        "selection": "1450000 <= triangle_count <= 1510000 and embedded_images >= 3",
        "count": len(copied),
        "models": copied,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(copied), "report": str(REPORT), "target": str(TARGET)}, indent=2))


if __name__ == "__main__":
    main()
