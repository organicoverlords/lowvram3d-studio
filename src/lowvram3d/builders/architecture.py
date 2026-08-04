from .base import BuilderContract


def contract() -> BuilderContract:
    return BuilderContract("architecture", ("architecture", "interior_structure", "wall", "roof", "tower", "opening"), ("modular_architecture", "editable_mesh"), ("regions", "scene_graph"), ("StaticMeshActor",), "blocking_by_region", "walkable_openings", "local_architecture", ("geometry", "collision"), "conservative_blocker", {"triangles": 200000})
