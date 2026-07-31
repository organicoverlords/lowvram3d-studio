# Unreal import

The pipeline exports `asset.fbx`, `asset.glb`, separate LOD GLBs, UCX collision names, sockets and `game_manifest.json`.

For a first import, drag `asset.fbx` into Unreal or run `import_lowvram_asset.py` from Unreal's Python environment after setting `ASSET_MANIFEST`. Automatic LOD assignment and deformation quality still require engine-side visual validation.
