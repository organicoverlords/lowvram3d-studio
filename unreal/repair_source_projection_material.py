"""Wire the source-projection material's texture into Emissive Color.

M_CastlegroundsSourceProjection is MSM_UNLIT, so its entire visible output
comes from Emissive Color. The builder created three TextureSample nodes
pointing at T_CastlegroundsSource but never connected any of them to a material
output, so the shell rendered pure black in every capture regardless of
lighting, camera or visibility profile.

Connect one sample to Emissive Color, recompile and save.

    python -m uemcp python @unreal/repair_source_projection_material.py --json
"""

import json

import unreal

MATERIAL_PATH = ("/Game/AgentProof/ImageToSceneSmoke_20260803/HybridSourceShell/"
                 "Materials/M_CastlegroundsSourceProjection")

material = unreal.load_asset(MATERIAL_PATH)
if material is None:
    raise RuntimeError(f"could not load {MATERIAL_PATH}")

expressions = list(unreal.MaterialEditingLibrary.get_material_expressions(material))
samples = [e for e in expressions
           if str(e.get_class().get_name()) == "MaterialExpressionTextureSample"]
if not samples:
    raise RuntimeError("material has no TextureSample expression to connect")

sample = samples[0]
texture = sample.get_editor_property("texture")

before = bool(unreal.MaterialEditingLibrary.get_material_property_input_node(
    material, unreal.MaterialProperty.MP_EMISSIVE_COLOR))

connected = unreal.MaterialEditingLibrary.connect_material_property(
    sample, "RGB", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

unreal.MaterialEditingLibrary.recompile_material(material)
saved = unreal.EditorAssetLibrary.save_loaded_asset(material)

after = bool(unreal.MaterialEditingLibrary.get_material_property_input_node(
    material, unreal.MaterialProperty.MP_EMISSIVE_COLOR))

result = json.dumps({
    "material": MATERIAL_PATH,
    "texture": str(texture.get_path_name()) if texture else None,
    "texture_sample_count": len(samples),
    "emissive_connected_before": before,
    "emissive_connected_after": after,
    "connect_call_returned": bool(connected),
    "saved": bool(saved),
    "classification": "REPAIRED" if after else "STILL_UNCONNECTED",
})
