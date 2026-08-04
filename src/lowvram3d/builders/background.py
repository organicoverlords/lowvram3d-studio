from .base import BuilderContract


def contract() -> BuilderContract:
    return BuilderContract("background", ("background_geometry",), ("background_card", "visual_shell"), ("regions", "visibility"), ("StaticMeshActor",), "none", "ignored", "background", ("coverage",), "visual_shell", {"triangles": 100000})
