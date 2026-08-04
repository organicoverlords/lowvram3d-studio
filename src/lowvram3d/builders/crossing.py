from .base import BuilderContract


def contract() -> BuilderContract:
    return BuilderContract("crossing", ("crossing", "road_or_path"), ("spline_structure", "procedural_mesh", "gameplay_proxy"), ("regions", "scene_graph"), ("StaticMeshActor",), "blocking", "walkable", "wood_or_local", ("geometry", "collision", "navigation"), "unresolved", {"triangles": 50000})
