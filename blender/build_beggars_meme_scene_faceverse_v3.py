from __future__ import annotations

import sys
from pathlib import Path

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_beggars_meme_scene_faceverse_v2 as v2  # noqa: E402


def configure_render_compat(
    scene: bpy.types.Scene,
    config: dict,
    engine: str,
) -> None:
    width, height = config["creative_target"]["resolution"]
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.fps = int(config["creative_target"]["fps"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False

    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 96
        scene.cycles.use_denoising = True
        selected_engine = "CYCLES"
    else:
        selected_engine = None
        for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                scene.render.engine = candidate
                selected_engine = candidate
                break
            except TypeError:
                continue
        if selected_engine is None:
            raise RuntimeError(
                "No compatible Eevee render-engine enum is available; "
                f"current={scene.render.engine!r}"
            )
        scene.render.use_file_extension = True

    scene.render.image_settings.color_depth = "8"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    scene.world.color = (0.003, 0.001, 0.0005)
    print(f"BLENDER_RENDER_ENGINE_COMPAT=PROVEN ENGINE={selected_engine}")


def main() -> int:
    v2.base.configure_render = configure_render_compat
    return int(v2.main())


if __name__ == "__main__":
    raise SystemExit(main())
