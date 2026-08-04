from .base import BuilderContract


def contract() -> BuilderContract:
    return BuilderContract("vegetation", ("vegetation",), ("procedural_population", "procedural_mesh"), ("regions", "visibility"), ("StaticMeshActor",), "none", "ignored", "biome_local", ("population", "exclusion"), "visual_shell", {"instances": 1000})
