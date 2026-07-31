r"""Run from Unreal Editor's Python console after setting ASSET_MANIFEST.

Example:
    import os
    os.environ['ASSET_MANIFEST'] = r'C:\...\proof\game_manifest.json'
    exec(open(r'C:\...\integrations\unreal\import_lowvram_asset.py').read())
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import unreal

manifest_path = Path(os.environ["ASSET_MANIFEST"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
source = manifest.get("skeletal_or_static_fbx") or manifest["primary_glb"]
destination = os.environ.get("ASSET_DESTINATION", "/Game/Generated/LowVRAM3D")

task = unreal.AssetImportTask()
task.filename = source
task.destination_path = destination
task.automated = True
task.save = True
task.replace_existing = False
options = unreal.FbxImportUI()
options.import_mesh = True
options.import_animations = True
options.import_materials = True
options.import_textures = True
options.create_physics_asset = True
task.options = options
unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
print({"imported": task.imported_object_paths, "manifest": str(manifest_path)})
