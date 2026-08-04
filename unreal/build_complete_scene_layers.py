"""Compatibility entry point for the manifest-owned scene composition pass.

All layout, geometry, transforms, and material choices come from
scene_content_manifest.json.  This file intentionally contains no scene
coordinates or actor-specific repair logic.
"""

from apply_scene_composition import DEFAULT_MANIFEST, apply_manifest


if __name__ == "__main__":
    apply_manifest(DEFAULT_MANIFEST)
