"""Measure, rather than assume, how the glTF importer maps axes into Unreal.

Every previous statement about this convention in the project was a guess, and
at least one was wrong -- a mesh placed at the yaw the standard convention
predicts rendered an almost-empty frame. So import a probe whose three axes sit
at three *different* distances (see `workers/make_axis_probe_glb.py`), read the
imported vertex positions straight back out of the StaticMesh, and solve the
signed axis permutation from where each marker actually landed.

Reports the unit conversion too: the importer already applies metres ->
centimetres, and scaling on top of that once put a 544 m scene at 25 km across.

Configure with an `AXIS_PROBE_REQUEST` global holding `glb` and, optionally,
`package_root`.

    python -m uemcp python @unreal/measure_axis_mapping.py --json
"""

import json
import os

import unreal

REQUEST = globals().get("AXIS_PROBE_REQUEST") or {}
GLB_PATH = REQUEST["glb"]
PACKAGE_ROOT = REQUEST.get("package_root", "/Game/AgentProof/AxisProbe")
# Marker name -> its position in the source glTF, in metres.
MARKERS = REQUEST.get("markers") or {
    "origin": [0.0, 0.0, 0.0],
    "gltf_x_plus": [1.0, 0.0, 0.0],
    "gltf_y_plus": [0.0, 2.0, 0.0],
    "gltf_z_plus": [0.0, 0.0, 3.0],
}
CLUSTER_RADIUS_CM = 30.0

report = {"schema_version": "axis_mapping_measurement_v1", "glb": GLB_PATH}

if not os.path.isfile(GLB_PATH):
    raise RuntimeError("axis probe not found: " + str(GLB_PATH))

# -- import -----------------------------------------------------------------
task = unreal.AssetImportTask()
task.set_editor_property("filename", GLB_PATH)
task.set_editor_property("destination_path", PACKAGE_ROOT + "/Meshes")
task.set_editor_property("automated", True)
task.set_editor_property("replace_existing", True)
task.set_editor_property("save", True)
unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

imported = [str(p) for p in (task.get_editor_property("imported_object_paths") or [])]
report["imported_object_paths"] = imported

mesh = None
for path in imported:
    asset = unreal.load_asset(path)
    if isinstance(asset, unreal.StaticMesh):
        mesh = asset
        report["static_mesh"] = path
        break
if mesh is None:
    raise RuntimeError("probe import produced no StaticMesh; got " + str(imported))

# Nanite replaces the reported geometry with a coarse fallback proxy, which
# would be measured here instead of the real vertices.
settings = mesh.get_editor_property("nanite_settings")
settings.set_editor_property("enabled", False)
mesh.set_editor_property("nanite_settings", settings)
unreal.EditorAssetLibrary.save_loaded_asset(mesh)

# -- read the vertices back -------------------------------------------------
vertices = []
section = 0
while True:
    try:
        data = unreal.ProceduralMeshLibrary.get_section_from_static_mesh(mesh, 0, section)
    except Exception:
        break
    if not data or not len(data[0]):
        break
    vertices.extend([[float(v.x), float(v.y), float(v.z)] for v in data[0]])
    section += 1
    if section > 8:
        break

report["vertex_count"] = len(vertices)
if not vertices:
    raise RuntimeError("could not read vertices back from the imported probe")

# -- cluster ----------------------------------------------------------------
clusters = []
for vertex in vertices:
    for cluster in clusters:
        centre = cluster["sum"]
        count = cluster["count"]
        distance = sum((vertex[i] - centre[i] / count) ** 2 for i in range(3)) ** 0.5
        if distance <= CLUSTER_RADIUS_CM:
            for i in range(3):
                cluster["sum"][i] += vertex[i]
            cluster["count"] += 1
            break
    else:
        clusters.append({"sum": list(vertex), "count": 1})

centroids = [[c["sum"][i] / c["count"] for i in range(3)] for c in clusters]
report["cluster_count"] = len(centroids)
report["cluster_centroids_cm"] = [[round(v, 3) for v in c] for c in centroids]

# -- solve the mapping ------------------------------------------------------
# Each source marker lies on exactly one glTF axis at a known distance, so the
# cluster whose distance from the origin matches identifies where that axis
# went, and the dominant component's sign says whether it was flipped.
AXIS_NAMES = ["X", "Y", "Z"]
origin_centroid = min(centroids, key=lambda c: sum(v * v for v in c) ** 0.5)
report["origin_centroid_cm"] = [round(v, 3) for v in origin_centroid]

mapping = {}
scales = []
for name, source in MARKERS.items():
    if name == "origin":
        continue
    source_axis = max(range(3), key=lambda i: abs(source[i]))
    source_distance_m = abs(float(source[source_axis]))
    relative = [
        min(centroids, key=lambda c: abs(
            sum((c[i] - origin_centroid[i]) ** 2 for i in range(3)) ** 0.5
            - source_distance_m * 100.0))
    ][0]
    offset = [relative[i] - origin_centroid[i] for i in range(3)]
    target_axis = max(range(3), key=lambda i: abs(offset[i]))
    magnitude = abs(offset[target_axis])
    sign = 1 if offset[target_axis] >= 0 else -1
    off_axis = max(abs(offset[i]) for i in range(3) if i != target_axis)
    scales.append(magnitude / source_distance_m)
    mapping["gltf_" + AXIS_NAMES[source_axis]] = {
        "unreal_axis": ("+" if sign > 0 else "-") + AXIS_NAMES[target_axis],
        "source_distance_m": source_distance_m,
        "measured_offset_cm": [round(v, 3) for v in offset],
        "measured_distance_cm": round(magnitude, 3),
        "off_axis_leakage_cm": round(off_axis, 3),
        "clean": bool(off_axis < 1.0),
    }

report["mapping"] = mapping
report["unreal_cm_per_gltf_metre"] = round(sum(scales) / len(scales), 4) if scales else None
targets = [entry["unreal_axis"][1] for entry in mapping.values()]
report["is_signed_permutation"] = bool(len(set(targets)) == len(targets))
report["all_axes_clean"] = bool(all(entry["clean"] for entry in mapping.values()))
report["classification"] = (
    "PROVEN"
    if report["is_signed_permutation"] and report["all_axes_clean"]
    and len(mapping) == 3 and len(centroids) == len(MARKERS)
    else "NOT_PROVEN"
)
report["convention"] = " ".join(
    source + "->" + entry["unreal_axis"] for source, entry in sorted(mapping.items()))

result = json.dumps(report)
