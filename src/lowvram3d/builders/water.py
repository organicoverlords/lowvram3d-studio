from .base import BuilderContract


def contract() -> BuilderContract:
    return BuilderContract("water", ("water",), ("water_surface", "spline_structure"), ("regions", "visibility"), ("StaticMeshActor",), "none", "excluded", "water", ("visibility", "exclusion"), "visual_shell", {"triangles": 20000})
