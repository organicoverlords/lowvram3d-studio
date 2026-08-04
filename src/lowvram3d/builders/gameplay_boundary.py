from .base import BuilderContract


def contract() -> BuilderContract:
    return BuilderContract("gameplay_boundary", ("gameplay_boundary",), ("gameplay_proxy", "collision_only", "navigation_only"), ("regions", "scene_graph"), ("BlockingVolume",), "blocking", "blocked", "debug", ("collision",), "bounded_box", {"actors": 32})
