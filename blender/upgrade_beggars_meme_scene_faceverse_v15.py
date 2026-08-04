from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy


IMAGE_NODE_NAME = "BeggarsSourceHeadOverlayV15"
SELECTED_LABEL = "scale_200"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overlay-dir", required=True)
    parser.add_argument("--overlay-report", required=True)
    parser.add_argument("--cutout-report", required=True)
    parser.add_argument("--base-receipt", required=True)
    parser.add_argument("--frame", type=int, default=48)
    return parser.parse_args(argv)


def hide_rejected_head_geometry() -> list[str]:
    exact_names = {
        "CHAR_Antinous",
        "CHAR_Antinous_HairCap",
        "CHAR_Antinous_FittedHairShell",
        "CHAR_Antinous_Neck",
    }
    prefixes = ("HAIR_", "FACIALHAIR_", "EYE_", "MOUTH_")
    hidden: list[str] = []
    for obj in bpy.data.objects:
        if obj.name in exact_names or obj.name.startswith(prefixes):
            obj.hide_render = True
            obj.hide_set(True)
            hidden.append(obj.name)
    if "CHAR_Antinous" not in hidden:
        raise RuntimeError("V15 could not hide the rejected synthetic face")
    return sorted(hidden)


def load_overlay(path: Path, pack: bool = False) -> bpy.types.Image:
    image = bpy.data.images.load(str(path), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    image.alpha_mode = "STRAIGHT"
    if pack:
        image.pack()
    return image


def configure_compositor(selected_image: bpy.types.Image) -> bpy.types.Node:
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.render.use_compositing = True
    tree = scene.node_tree
    if tree is None:
        raise RuntimeError("V15 compositor node tree is unavailable")
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    render_layers = nodes.new("CompositorNodeRLayers")
    render_layers.name = "BeggarsRenderLayersV15"
    image_node = nodes.new("CompositorNodeImage")
    image_node.name = IMAGE_NODE_NAME
    image_node.image = selected_image
    alpha_over = nodes.new("CompositorNodeAlphaOver")
    alpha_over.name = "BeggarsSourceHeadAlphaOverV15"
    alpha_over.inputs[0].default_value = 1.0
    if hasattr(alpha_over, "premul"):
        alpha_over.premul = 1.0
    composite = nodes.new("CompositorNodeComposite")
    composite.name = "BeggarsCompositeV15"

    links.new(render_layers.outputs["Image"], alpha_over.inputs[1])
    links.new(image_node.outputs["Image"], alpha_over.inputs[2])
    links.new(alpha_over.outputs["Image"], composite.inputs["Image"])
    return image_node


def ensure_render(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 50000:
        raise RuntimeError(f"V15 render missing or implausibly small: {path}")


def render_variants(
    output_dir: Path,
    overlay_dir: Path,
    overlay_report: dict[str, Any],
    image_node: bpy.types.Node,
    frame: int,
) -> list[dict[str, Any]]:
    scene = bpy.context.scene
    camera = bpy.data.objects.get("CAM_Hero")
    if camera is None:
        raise RuntimeError("V15 hero camera missing")
    scene.camera = camera
    scene.frame_set(frame)

    rendered: list[dict[str, Any]] = []
    for variant in overlay_report["variants"]:
        overlay_path = overlay_dir / str(variant["file"])
        if not overlay_path.is_file():
            raise RuntimeError(f"V15 overlay missing: {overlay_path}")
        image = load_overlay(overlay_path, pack=False)
        image_node.image = image
        output_path = output_dir / f"composite_head_{variant['label']}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        ensure_render(output_path)
        rendered.append(
            {
                **variant,
                "render": output_path.name,
                "render_bytes": output_path.stat().st_size,
            }
        )
        print(
            "BLENDER_FACEVERSE_V15_COMPOSITE_VARIANT=PROVEN "
            f"LABEL={variant['label']}"
        )

    selected_variant = next(
        variant for variant in overlay_report["variants"] if variant["label"] == SELECTED_LABEL
    )
    selected_path = overlay_dir / str(selected_variant["file"])
    selected_image = load_overlay(selected_path, pack=True)
    image_node.image = selected_image
    return rendered


def patch_receipt(
    output_dir: Path,
    base_receipt_path: Path,
    cutout_report_path: Path,
    overlay_report_path: Path,
    hidden: list[str],
    variants: list[dict[str, Any]],
) -> None:
    receipt = json.loads(base_receipt_path.read_text(encoding="utf-8"))
    cutout_report = json.loads(cutout_report_path.read_text(encoding="utf-8"))
    overlay_report = json.loads(overlay_report_path.read_text(encoding="utf-8"))
    receipt["classification"] = "USER_VISUAL_REVIEW_REQUIRED"
    receipt["character_variant"] = "FACEVERSE_BODY_PLUS_COMPOSITOR_HEAD_V15"
    receipt["claim_policy"] = (
        "The v15 Blender compositor overlay, hidden synthetic head, procedural body/set, renders "
        "and save/reload are machine-proven. Visual likeness and final meme quality remain "
        "NOT_PROVEN until direct review."
    )
    receipt["visual_changes_v15"] = {
        "eevee_transparent_material_route_rejected": True,
        "opaque_card_route_rejected": True,
        "full_frame_rgba_blender_compositor": True,
        "synthetic_face_hair_and_neck_hidden": True,
        "hidden_objects": hidden,
        "source_head_image_packed": True,
        "selected_overlay": overlay_report["selected"],
        "procedural_torso_set_lighting_preserved": True,
        "animation_rendered": False,
        "cutout_report": cutout_report,
    }
    receipt["v15_variants"] = variants
    (output_dir / "scene_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "v15_visual_report.json").write_text(
        json.dumps(
            {
                "classification": "USER_VISUAL_REVIEW_REQUIRED",
                "machine_status": "PROVEN",
                "visual_status": "NOT_PROVEN",
                "compositor_image_node": IMAGE_NODE_NAME,
                "selected_label": SELECTED_LABEL,
                "hidden_objects": hidden,
                "variants": variants,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    overlay_dir = Path(args.overlay_dir).resolve()
    overlay_report_path = Path(args.overlay_report).resolve()
    cutout_report_path = Path(args.cutout_report).resolve()
    base_receipt_path = Path(args.base_receipt).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (overlay_report_path, cutout_report_path, base_receipt_path):
        if not path.is_file():
            raise RuntimeError(f"V15 input missing: {path}")
    overlay_report = json.loads(overlay_report_path.read_text(encoding="utf-8"))
    selected = overlay_report["selected"]
    selected_path = overlay_dir / str(selected["file"])
    if not selected_path.is_file():
        raise RuntimeError(f"V15 selected overlay missing: {selected_path}")

    hidden = hide_rejected_head_geometry()
    selected_image = load_overlay(selected_path, pack=True)
    image_node = configure_compositor(selected_image)
    variants = render_variants(
        output_dir, overlay_dir, overlay_report, image_node, args.frame
    )
    patch_receipt(
        output_dir,
        base_receipt_path,
        cutout_report_path,
        overlay_report_path,
        hidden,
        variants,
    )

    blend_path = output_dir / "beggars_photoreal_recreation.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if not blend_path.is_file() or blend_path.stat().st_size < 1000000:
        raise RuntimeError("V15 Blender file missing or implausibly small")

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    scene = bpy.context.scene
    if not scene.use_nodes or scene.node_tree is None:
        raise RuntimeError("V15 compositor did not survive save/reload")
    reloaded_node = scene.node_tree.nodes.get(IMAGE_NODE_NAME)
    face = bpy.data.objects.get("CHAR_Antinous")
    camera = bpy.data.objects.get("CAM_Hero")
    if reloaded_node is None or reloaded_node.image is None:
        raise RuntimeError("V15 source overlay node did not survive save/reload")
    if reloaded_node.image.packed_file is None:
        raise RuntimeError("V15 selected overlay is not packed after reload")
    if face is None or not face.hide_render:
        raise RuntimeError("V15 hidden synthetic face did not survive save/reload")
    if camera is None:
        raise RuntimeError("V15 hero camera missing after reload")

    scene.camera = camera
    scene.frame_set(args.frame)
    final_path = output_dir / "hero_composited_head_render.png"
    scene.render.filepath = str(final_path)
    bpy.ops.render.render(write_still=True)
    ensure_render(final_path)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    print(
        "BLENDER_FACEVERSE_V15_COMPOSITOR_HEAD=PROVEN "
        f"SELECTED={SELECTED_LABEL} HIDDEN={len(hidden)}"
    )
    print("BLENDER_FACEVERSE_V15_SAVE_RELOAD_STILL=PROVEN")
    print("BEGGARS_FACEVERSE_V15=USER_VISUAL_REVIEW_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
