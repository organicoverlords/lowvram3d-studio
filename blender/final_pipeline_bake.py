"""High-to-low bake for the shaman: tangent normal, AO, cavity and material-ID support map.

Bakes from the untouched clean high master onto the UV-unwrapped LOD0 via Cycles selected-to-active
with a controlled ray distance. Every output is validated rather than assumed: an atlas that is
entirely black, carries non-finite pixels, or covers almost none of its UV space is a failed bake
even though Blender exits zero.

Cavity has no native bake type, so it is driven through geometry pointiness into an emission
shader; the material ID map is a per-connected-component colour, which is what makes it possible to
see body/staff/ornament cross-projection in the result.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import struct
from pathlib import Path

import bpy
import numpy as np

from common import argv_after_double_dash, import_mesh, reset_scene, save_json

COMPONENT_SIZE = {5121: 1, 5123: 2, 5125: 4, 5126: 4}
COMPONENT_DTYPE = {5121: "<u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}


def read_glb_uv(path: str):
    """Pull POSITION/TEXCOORD_0/indices straight out of a GLB.

    Needed because the UV atlas is written as a bare mesh with no material, and Blender's glTF
    importer only materialises the TEXCOORD sets some material actually samples - so a valid,
    fully-populated TEXCOORD_0 silently arrives as an object with zero UV layers.
    """
    data = Path(path).read_bytes()
    offset, chunk_json, binary = 12, None, None
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        payload = data[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            chunk_json = json.loads(payload)
        elif kind == 0x004E4942:
            binary = payload
        offset += 8 + length
    if chunk_json is None or binary is None:
        raise RuntimeError(f"{path} is not a self-contained GLB")

    def accessor(index):
        acc = chunk_json["accessors"][index]
        view = chunk_json["bufferViews"][acc["bufferView"]]
        count, width = acc["count"], TYPE_COUNT[acc["type"]]
        item = COMPONENT_SIZE[acc["componentType"]] * width
        stride = view.get("byteStride") or item
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        raw = np.frombuffer(binary, np.uint8, count=stride * (count - 1) + item, offset=start)
        if stride != item:
            raw = np.lib.stride_tricks.as_strided(raw, (count, item), (stride, 1)).copy()
        return raw.reshape(-1).view(COMPONENT_DTYPE[acc["componentType"]]).reshape(count, width)

    primitive = chunk_json["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    if "TEXCOORD_0" not in attributes:
        raise RuntimeError(f"{path} carries no TEXCOORD_0")
    positions = accessor(attributes["POSITION"]).astype(np.float64)
    uv = accessor(attributes["TEXCOORD_0"]).astype(np.float64)
    indices = accessor(primitive["indices"]).reshape(-1).astype(np.int64)
    return positions, uv, indices


def activate_uv_layer(obj) -> None:
    """Mark the first UV layer active for both edit and render.

    A freshly imported mesh can carry a UV layer with no active index set, and Cycles reports that
    as "No active UV layer found" - which reads like missing UVs but is only a missing flag.
    """
    layers = obj.data.uv_layers
    if not layers:
        return
    layers.active_index = 0
    layers[0].active = True
    layers[0].active_render = True


def ensure_uv_layer(obj, path: str) -> str:
    if obj.data.uv_layers:
        activate_uv_layer(obj)
        return "imported"
    positions, uv, indices = read_glb_uv(path)
    mesh = obj.data
    if len(mesh.vertices) != len(positions) or len(mesh.loops) != len(indices):
        raise RuntimeError(
            f"cannot restore UVs: GLB has {len(positions)} verts/{len(indices)} corners, "
            f"Blender imported {len(mesh.vertices)}/{len(mesh.loops)}"
        )
    loop_vertices = np.empty(len(mesh.loops), np.int64)
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    if not np.array_equal(loop_vertices, indices):
        raise RuntimeError("cannot restore UVs: the importer reordered the index buffer")
    imported = np.empty(len(mesh.vertices) * 3, np.float64)
    mesh.vertices.foreach_get("co", imported)
    drift = float(np.abs(imported.reshape(-1, 3) - positions).max())
    if drift > 1e-4:
        raise RuntimeError(f"cannot restore UVs: vertex positions drifted by {drift:.6f}")
    layer = mesh.uv_layers.new(name="UVMap")
    # glTF puts the UV origin top-left, Blender bottom-left; flipping V here reproduces exactly
    # what the importer would have done, so the bake round-trips back out unchanged.
    flipped = uv[indices].copy()
    flipped[:, 1] = 1.0 - flipped[:, 1]
    target = layer.uv if hasattr(layer, "uv") else layer.data
    target.foreach_set("vector" if hasattr(layer, "uv") else "uv", flipped.reshape(-1))
    mesh.update()
    activate_uv_layer(obj)
    return "restored_from_glb"


def write_png16(path: Path, rgb: np.ndarray) -> None:
    """Write a 16-bit RGB PNG straight from the float bake buffer.

    Neither image.save() nor image.save_render(color_depth='16') produced a 16-bit file here, and
    an 8-bit tangent normal map bands visibly on smooth surfaces. Encoding the buffer directly is
    both shorter than fighting the exporter and independent of which save path Blender picks.
    """
    import struct
    import zlib

    height, width, _ = rgb.shape
    data = np.clip(rgb, 0.0, 1.0) * 65535.0
    rows = data.astype(">u2").tobytes()
    stride = width * 6
    raw = b"".join(b"\x00" + rows[y * stride : (y + 1) * stride] for y in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def join_all(objects, name):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = name
    return joined


def component_colours(obj, weld_distance: float = 1e-5) -> int:
    """Colour each connected component distinctly, for the material-ID support map.

    The connectivity pass runs over vertices welded by position rather than the mesh as imported:
    glTF emits one vertex per face corner wherever normals split, so raw edge connectivity sees
    every triangle as its own component and the map degenerates into per-triangle noise. The weld
    is only a lookup - the mesh itself is never modified, so the normal and AO bakes read exactly
    the geometry and split normals they were given.
    """
    mesh = obj.data
    count = len(mesh.vertices)
    coordinates = np.empty(count * 3, np.float64)
    mesh.vertices.foreach_get("co", coordinates)
    quantised = np.round(coordinates.reshape(-1, 3) / weld_distance).astype(np.int64)
    _, representative = np.unique(quantised, axis=0, return_inverse=True)
    representative = representative.astype(np.int64)

    parent = list(range(int(representative.max()) + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = np.empty(len(mesh.edges) * 2, np.int32)
    mesh.edges.foreach_get("vertices", edges)
    for first, second in representative[edges.reshape(-1, 2)]:
        a, b = find(int(first)), find(int(second))
        if a != b:
            parent[a] = b

    roots: dict[int, int] = {}
    slots = np.empty(count, np.int64)
    for index in range(count):
        root = find(int(representative[index]))
        slots[index] = roots.setdefault(root, len(roots))

    attribute = mesh.color_attributes.new(name="component_id", type="FLOAT_COLOR", domain="POINT")
    palette = np.empty((len(roots), 4), np.float32)
    for slot in range(len(roots)):
        # Golden-ratio hue stepping keeps neighbouring component ids visually distinct.
        hue = (slot * 0.61803398875) % 1.0
        palette[slot] = (*colorsys.hsv_to_rgb(hue, 0.85, 1.0), 1.0)
    attribute.data.foreach_set("color", palette[slots].reshape(-1))
    mesh.update()
    return len(roots)


def emission_material(name: str, source: str):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    if source == "pointiness":
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        ramp = tree.nodes.new("ShaderNodeValToRGB")
        # Pointiness sits in a narrow band around 0.5; widen it so cavity detail is visible.
        ramp.color_ramp.elements[0].position = 0.42
        ramp.color_ramp.elements[1].position = 0.58
        tree.links.new(geometry.outputs["Pointiness"], ramp.inputs["Fac"])
        tree.links.new(ramp.outputs["Color"], emission.inputs["Color"])
    elif source == "component":
        attribute = tree.nodes.new("ShaderNodeVertexColor")
        attribute.layer_name = "component_id"
        tree.links.new(attribute.outputs["Color"], emission.inputs["Color"])
    return material


def target_material(low, image):
    material = bpy.data.materials.new("bake_target")
    material.use_nodes = True
    tree = material.node_tree
    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    tree.nodes.active = node
    low.data.materials.clear()
    low.data.materials.append(material)
    return node


def validate_image(image, name: str, background=(0.0, 0.0, 0.0)) -> dict:
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(-1, 4)
    rgb = pixels[:, :3]
    finite = bool(np.isfinite(pixels).all())
    distance = np.linalg.norm(rgb - np.array(background, np.float32), axis=1)
    covered = distance > 0.02
    coverage = float(covered.mean())
    return {
        "name": name,
        "finite": finite,
        "nan_pixels": int((~np.isfinite(pixels)).sum()),
        "mean_rgb": [float(v) for v in rgb.mean(axis=0)],
        "max_rgb": [float(v) for v in rgb.max(axis=0)],
        "coverage_fraction": coverage,
        "all_black": bool(rgb.max() <= 0.02),
    }


def bake_map(scene, low, high, kind: str, image, cage_extrusion: float, ray_distance: float) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    settings = scene.render.bake
    settings.use_selected_to_active = True
    settings.cage_extrusion = cage_extrusion
    settings.max_ray_distance = ray_distance
    settings.use_clear = True
    if kind == "NORMAL":
        scene.render.bake.normal_space = "TANGENT"
        bpy.ops.object.bake(type="NORMAL")
    elif kind == "AO":
        bpy.ops.object.bake(type="AO")
    else:
        bpy.ops.object.bake(type="EMIT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", required=True)
    parser.add_argument("--low", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--cage-extrusion", type=float, default=0.02)
    parser.add_argument("--max-ray-distance", type=float, default=0.05)
    parser.add_argument("--suffix", default="4k")
    parser.add_argument("--max-components", type=int, default=500)
    parser.add_argument("--only", default="", help="comma-separated map subset to rebake")
    args = parser.parse_args(argv_after_double_dash())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reset_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False

    # import_mesh reports every mesh in the scene, not just the file it just read, so the second
    # import has to be diffed against the first - otherwise the low join swallows the high poly
    # source and the bake runs with a single object and a dangling reference.
    high_objects = import_mesh(args.high)
    high = join_all(high_objects, "HIGH")
    before = set(bpy.context.scene.objects)
    import_mesh(args.low)
    low_objects = [o for o in bpy.context.scene.objects if o.type == "MESH" and o not in before]
    if not low_objects:
        raise RuntimeError(f"no mesh objects were imported from {args.low}")
    low = join_all(low_objects, "LOW")
    uv_source = ensure_uv_layer(low, args.low)
    print(f"UV_LAYER {uv_source} layers={len(low.data.uv_layers)}", flush=True)

    component_count = component_colours(high)
    print(f"HIGH_COMPONENTS {component_count}", flush=True)

    maps = [
        ("normal", "NORMAL", None, (0.5, 0.5, 1.0)),
        ("ao", "AO", None, (0.0, 0.0, 0.0)),
        ("cavity", "EMIT", "pointiness", (0.0, 0.0, 0.0)),
        ("material_id", "EMIT", "component", (0.0, 0.0, 0.0)),
    ]
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        maps = [m for m in maps if m[0] in wanted]
        if not maps:
            raise RuntimeError(f"--only {args.only} selected no maps")
    results = []
    for name, kind, source, background in maps:
        image = bpy.data.images.new(
            f"shaman_{name}", width=args.resolution, height=args.resolution, alpha=False,
            float_buffer=(name == "normal"),
        )
        if name == "normal":
            image.colorspace_settings.name = "Non-Color"
        target_material(low, image)
        if source:
            high.data.materials.clear()
            high.data.materials.append(emission_material(f"high_{name}", source))
        bake_map(scene, low, high, kind, image, args.cage_extrusion, args.max_ray_distance)

        path = output_dir / f"shaman_{name}_{args.suffix}.png"
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        if name == "normal":
            pixels = np.array(image.pixels[:], dtype=np.float32).reshape(args.resolution, args.resolution, 4)
            # Blender's buffer runs bottom-up; PNG rows run top-down.
            write_png16(path, pixels[::-1, :, :3])
        else:
            image.save()
        stats = validate_image(image, name, background)
        stats["path"] = str(path)
        results.append(stats)
        print(
            f"BAKED {name}: coverage={stats['coverage_fraction']*100:.2f}% "
            f"black={stats['all_black']} nan={stats['nan_pixels']} mean={stats['mean_rgb']}",
            flush=True,
        )

    failures = []
    for entry in results:
        if not entry["finite"]:
            failures.append(f"{entry['name']}: non-finite pixels")
        if entry["all_black"]:
            failures.append(f"{entry['name']}: atlas is entirely black")
        if entry["coverage_fraction"] < 0.15:
            failures.append(
                f"{entry['name']}: only {entry['coverage_fraction']*100:.2f}% of the atlas received pixels"
            )
    # A material-ID map is only useful if it names parts. When connectivity collapses - the failure
    # mode where a glTF import leaves every triangle detached - the count explodes and the map
    # bakes as per-triangle noise, which still passes every coverage and finiteness check.
    if component_count > args.max_components:
        failures.append(
            f"material_id: high poly split into {component_count} components "
            f"(limit {args.max_components}); the map is per-triangle noise, not per-part"
        )
    normal = next((e for e in results if e["name"] == "normal"), None)
    # A tangent-space normal map should sit around (0.5, 0.5, 1.0); a blue mean far below the red
    # and green means the projection collapsed.
    if normal is not None and normal["mean_rgb"][2] < 0.3:
        failures.append(f"normal: blue channel mean {normal['mean_rgb'][2]:.3f} suggests bad tangent basis")

    report = {
        "high": args.high,
        "low": args.low,
        "resolution": args.resolution,
        "samples": args.samples,
        "cage_extrusion": args.cage_extrusion,
        "max_ray_distance": args.max_ray_distance,
        "uv_layer_source": uv_source,
        "high_component_count": component_count,
        "maps": results,
        "failures": failures,
        "passed": not failures,
    }
    save_json(args.report, report)
    print(f"BAKE_RESULT passed={not failures} failures={failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
