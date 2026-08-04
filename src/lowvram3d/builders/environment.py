from .base import BuilderContract


def contract() -> BuilderContract:
    return BuilderContract("environment", ("sky_or_ceiling", "lighting_source"), ("sky_environment", "visual_shell"), ("regions", "camera"), ("SkyAtmosphere", "DirectionalLight", "ExponentialHeightFog"), "none", "ignored", "environment", ("environment", "lighting"), "flat_environment", {"draw_calls": 10})
