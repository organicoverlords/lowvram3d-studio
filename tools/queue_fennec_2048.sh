#!/bin/sh
# Rebake the fennec at 2048, changing nothing else.
#
# This is a controlled single-variable test of the density hypothesis. The 1024
# bake left ~1.36 occupied texels per face (36.7% atlas utilisation over 284,142
# faces); every asset previously judged acceptable effectively got 2048, which
# is 5.42. So the mesh, the conditioning image, the cameras, the view weights,
# the projection and inpainting parameters and the UV coordinates are all held
# fixed and reused byte-for-byte from the 1024 run.
#
# --render-size stays 512 deliberately: the instruction was to move exactly one
# variable. Note that this leaves render_size as a second, untested difference
# from the assets that passed -- they ran the vendor default of 2048 there too.
# If 2048 does not resolve the shards, render_size is the next confound to
# eliminate, not evidence that density is exonerated.
#
# The 1024 outputs are never overwritten: this writes ...hypaint2048.glb beside
# ...hypaint1024.glb so the A/B survives.
set -e
cd "$(dirname "$0")/.."

REPO="$(pwd)"
SUBJ="evidence/compare/fennec"
PY="C:/AI/HY3D2/python_standalone/python.exe"

# Wait for the TRELLIS bluetree job to finish rather than competing with it.
# Polling for the process is deliberate: the mkdir lock's recorded owner PID has
# not matched the live script PID, so its stale-lock detection cannot be trusted
# to release if that job dies.
echo "[queue] waiting for GPU to free"
while tasklist //FI "IMAGENAME eq trellis-cli.exe" 2>/dev/null | grep -qi trellis-cli; do
    sleep 30
done
echo "[queue] GPU free at $(date +%H:%M:%S), starting 2048 bake"

PYTHONPATH=C:/AI/HY3D2/Hunyuan3D-2 "$PY" workers/hunyuan_paint_texture.py \
    --mesh "$SUBJ/fennec_t_uv.glb" \
    --image "$SUBJ/matte_512.png" \
    --out "$SUBJ/fennec_t_hypaint2048.glb" \
    --render-size 512 \
    --texture-size 2048 \
    > "$SUBJ/paint_t2048.log" 2>&1

echo "[queue] bake done at $(date +%H:%M:%S), rendering the same four views"

# Same renderer and same angles as the 1024 sheet, so the A/B differs only in
# the atlas. CPU device: a Blender render has starved a concurrent paint of
# system RAM before now.
C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe \
    workers/render_textured_views.py \
    --glb "$SUBJ/fennec_t_hypaint2048.glb" \
    --out "$SUBJ/views_fennec_t2048.png" \
    >> "$SUBJ/paint_t2048.log" 2>&1

echo "[queue] complete at $(date +%H:%M:%S)"
