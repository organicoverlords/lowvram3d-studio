from .base import BuilderContract


def contract() -> BuilderContract:
    return BuilderContract("terrain", ("ground", "terrain", "cliff"), ("terrain", "procedural_mesh"), ("regions", "visibility"), ("StaticMeshActor",), "blocking", "walkable", "rock_or_ground", ("geometry", "collision"), "visual_shell", {"triangles": 100000})
