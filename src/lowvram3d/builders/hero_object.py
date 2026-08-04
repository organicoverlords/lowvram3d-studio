from .base import BuilderContract


def contract() -> BuilderContract:
    return BuilderContract("hero_object", ("hero_object",), ("editable_mesh", "visual_shell"), ("regions", "scene_graph"), ("StaticMeshActor",), "local", "local", "object_local", ("geometry", "material"), "visual_shell", {"triangles": 100000})
