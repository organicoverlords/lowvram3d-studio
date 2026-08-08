"""Fail a rigged GLB that only looks rigged.

The frog rig candidate exported with everything a rigged asset is supposed to
have -- nine named bone nodes in a parented hierarchy, JOINTS_0 and WEIGHTS_0 on
the mesh, and an idle animation with channels driving those bones -- and no
`skins` array at all. Nothing binds the mesh to the skeleton, so the bones are
inert transforms and the animation moves nodes that no vertex follows.

Every cheap check passed it. The rig report said `success: true`, binding
`automatic_weights`, nine bones, one action. Only the five-pose proof caught it,
and only indirectly, by failing to find an armature after re-import -- a glTF
importer creates an Armature from a *skin*, not from nodes that happen to be
named after bones.

So the check has to be on the binding itself, and it is a JSON-level property of
the file: no Blender, no GPU, milliseconds. Run it on anything claiming to be
rigged before the word "rigged" appears in a receipt.

    py tools/validate_rig_export.py --mesh rigged.glb
    py tools/validate_rig_export.py --mesh rigged.glb --json report.json

Exit status is 0 only when the asset is genuinely skinned.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def read_gltf(path: Path) -> tuple[dict, bytes]:
    """Return the JSON chunk and the binary chunk of a GLB."""
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path} is not a GLB")
    json_length = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + json_length])
    return gltf, data[20 + json_length + 8:]


def accessor_values(gltf: dict, blob: bytes, index: int) -> list:
    """Decode a scalar/vec accessor far enough to sanity-check weights.

    Only the component types glTF actually uses for JOINTS_0/WEIGHTS_0 are
    handled; anything else returns empty rather than guessing.
    """
    accessor = gltf["accessors"][index]
    counts = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    per = counts.get(accessor["type"])
    formats = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
    fmt = formats.get(accessor["componentType"])
    if per is None or fmt is None or "bufferView" not in accessor:
        return []
    view = gltf["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    code, size = fmt
    stride = view.get("byteStride") or per * size
    out = []
    for i in range(accessor["count"]):
        offset = start + i * stride
        out.append(struct.unpack_from("<" + code * per, blob, offset))
    return out


def validate(path: Path) -> dict:
    gltf, blob = read_gltf(path)
    nodes = gltf.get("nodes", [])
    skins = gltf.get("skins", [])
    meshes = gltf.get("meshes", [])
    animations = gltf.get("animations", [])

    failures: list[str] = []
    notes: list[str] = []

    # 1. A skin must exist. This alone is what the frog was missing.
    if not skins:
        failures.append(
            "no `skins` array: the file has no mesh-to-skeleton binding, so any "
            "bone nodes in it are inert and no importer will build an armature")

    # 2. Some node must actually use one. A skin nothing references is dead too.
    skinned_nodes = [i for i, node in enumerate(nodes)
                     if "skin" in node and "mesh" in node]
    if skins and not skinned_nodes:
        failures.append("a skin exists but no node with a mesh references it")

    # 3. Every skinned mesh needs the per-vertex attributes to go with it, and
    #    conversely attributes without a skin are the frog's exact signature.
    attributed = []
    for index, mesh in enumerate(meshes):
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            if "JOINTS_0" in attributes and "WEIGHTS_0" in attributes:
                attributed.append((index, attributes))
    if attributed and not skins:
        notes.append(
            f"{len(attributed)} primitive(s) carry JOINTS_0/WEIGHTS_0 with no skin "
            "-- the weights survived the export and only the binding was lost")
    for node_index in skinned_nodes:
        mesh_index = nodes[node_index]["mesh"]
        for primitive in meshes[mesh_index].get("primitives", []):
            attributes = primitive.get("attributes", {})
            missing = [a for a in ("JOINTS_0", "WEIGHTS_0") if a not in attributes]
            if missing:
                failures.append(
                    f"node {node_index} is skinned but its mesh lacks {', '.join(missing)}")

    # 4. Joints must resolve, and must not be empty.
    for index, skin in enumerate(skins):
        joints = skin.get("joints", [])
        if not joints:
            failures.append(f"skin {index} has no joints")
        out_of_range = [j for j in joints if j >= len(nodes)]
        if out_of_range:
            failures.append(f"skin {index} references nodes that do not exist: {out_of_range}")

    # 5. Weights that sum to zero bind nothing even with a valid skin. Sampled,
    #    not exhaustive -- this is a gate, not a mesh validator.
    weight_note = None
    for node_index in skinned_nodes[:1]:
        for primitive in meshes[nodes[node_index]["mesh"]].get("primitives", []):
            attributes = primitive.get("attributes", {})
            if "WEIGHTS_0" not in attributes:
                continue
            values = accessor_values(gltf, blob, attributes["WEIGHTS_0"])[:4096]
            if not values:
                break
            scale = 255.0 if max(max(v) for v in values) > 1.5 else 1.0
            sums = [sum(v) / scale for v in values]
            unbound = sum(1 for s in sums if s < 1e-6)
            weight_note = {"sampled": len(sums),
                           "mean_sum": round(sum(sums) / len(sums), 4),
                           "zero_weight_vertices": unbound}
            if unbound > len(sums) * 0.01:
                failures.append(
                    f"{unbound}/{len(sums)} sampled vertices have zero total weight")
            break

    # 6. An animation whose channels drive nodes outside the skin moves nothing.
    joint_set = {j for skin in skins for j in skin.get("joints", [])}
    for index, animation in enumerate(animations):
        targets = {channel["target"]["node"]
                   for channel in animation.get("channels", [])
                   if "node" in channel.get("target", {})}
        if targets and joint_set and not (targets & joint_set):
            failures.append(
                f"animation {index} drives only non-joint nodes -- nothing deforms")
        if targets and not joint_set:
            notes.append(
                f"animation {index} drives {len(targets)} node(s), none of which are "
                "joints of any skin")

    return {
        "schema": "lowvram3d_rig_export_validation_v1",
        "mesh": str(path),
        "nodes": len(nodes),
        "skins": len(skins),
        "skinned_mesh_nodes": len(skinned_nodes),
        "primitives_with_joint_weights": len(attributed),
        "animations": len(animations),
        "weights": weight_note,
        "notes": notes,
        "failures": failures,
        # The word this whole file exists to protect.
        "skinned": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = validate(args.mesh)
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["skinned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
