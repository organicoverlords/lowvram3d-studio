#!/bin/sh
# Seed 777 is the last TRELLIS attempt. Then paint the Mini Turbo trees and stop.
#
# Rationale for each choice here:
#
#   no more seeds     the CUDA fault hits at res 512 as hard as at res 1024, so
#                     more seeds is not a bet worth making tonight.
#   Mini Turbo source geometry is already on disk, so this needs no generation
#                     and no CUDA lottery -- only decimate, unwrap, paint.
#   texture 2048      1024 gives ~1.4 texels/face, the starved configuration
#                     already shown to fail. 2048 gives ~5.4, matching every
#                     asset previously judged acceptable. Paint cost is driven
#                     by render_size (the diffusion pass), not texture_size, so
#                     this is close to free in wall-clock.
#   render_size 512   the cheap end, where the time actually goes.
#
# Mini Turbo GLBs carry POSITION only -- no COLOR_0, no UVs, no texture -- so
# paint is the ONLY route to appearance for them. Clay is all they can otherwise
# ever be.
set -e
cd "$(dirname "$0")/.."

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
HY="C:/AI/HY3D2/python_standalone/python.exe"

count() {
    powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='$1'\"$2).Count" \
        2>/dev/null | tr -d '\r\n '
}

echo "[finish] waiting for the last TRELLIS seed to end $(date +%H:%M:%S)"
while [ "$(count trellis-cli.exe)" != "0" ]; do sleep 15; done
echo "[finish] TRELLIS done $(date +%H:%M:%S)"

# Stop trees_fast before it can start seed 1006 or move on to greentree.
powershell -NoProfile -Command \
  "Get-CimInstance Win32_Process -Filter \"Name='sh.exe'\" | Where-Object { \$_.CommandLine -like '*trees_fast*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" \
  >/dev/null 2>&1 || true
sleep 2
echo "[finish] retry chain cut"

render() {   # mesh, out-prefix
    for mode in native clay; do
        "$PY" workers/render_asset_views.py --mesh "$1" --out "${2}_${mode}.png" \
            --views profile,three_quarter,end_plus --size 900 --$mode >/dev/null 2>&1 \
            && echo "[finish] rendered ${2}_${mode}.png" \
            || echo "[finish] render failed ${2}_${mode}.png"
    done
}

# Whatever the last seed produced, look at it.
if [ -s evidence/compare/bluetree/bluetree_t512.glb ]; then
    echo "[finish] === bluetree TRELLIS landed, rendering $(date +%H:%M:%S)"
    render evidence/compare/bluetree/bluetree_t512.glb \
           evidence/compare/bluetree/views_bluetree_t512
else
    echo "[finish] bluetree TRELLIS produced nothing; continuing to paint"
fi

########## paint the Mini Turbo trees at a density that has worked
for name in bluetree greentree; do
    D=evidence/compare/$name
    src=$D/${name}_miniturbo.glb
    [ -s "$src" ] || { echo "[finish] no $src, skipping"; continue; }

    if [ ! -s $D/${name}_mt_uv.glb ]; then
        echo "[finish] === $name decimate $(date +%H:%M:%S)"
        PYTHONPATH="C:/AI/HY3D2/Hunyuan3D-2" "$HY" workers/hunyuan_postprocess.py \
            --mesh "$src" --out $D/${name}_mt_lod.glb --faces 400000 2>&1 | tail -1
        [ -s $D/${name}_mt_lod.glb ] || { echo "[finish] $name decimate FAILED"; continue; }

        echo "[finish] === $name unwrap $(date +%H:%M:%S)"
        "$PY" workers/unwrap_mesh_uv.py --input $D/${name}_mt_lod.glb \
            --output $D/${name}_mt_uv.glb --report $D/${name}_mt_uv.json 2>&1 | tail -1
        [ -s $D/${name}_mt_uv.glb ] || { echo "[finish] $name unwrap FAILED"; continue; }
    fi

    echo "[finish] === $name paint 512/2048 $(date +%H:%M:%S)"
    PYTHONPATH="C:/AI/HY3D2/Hunyuan3D-2" "$HY" workers/hunyuan_paint_texture.py \
        --mesh $D/${name}_mt_uv.glb --image $D/matte_512.png \
        --out $D/${name}_mt_paint2048.glb \
        --render-size 512 --texture-size 2048 \
        --receipt $D/${name}_mt_paint2048.paint.json \
        > $D/paint_mt2048.log 2>&1 || true
    echo "[finish] === $name paint exit $? $(date +%H:%M:%S)"

    if [ -s $D/${name}_mt_paint2048.glb ]; then
        render $D/${name}_mt_paint2048.glb $D/views_${name}_mt_paint2048
    else
        echo "[finish] $name paint produced nothing:"
        grep -v "it/s\]" $D/paint_mt2048.log 2>/dev/null | tail -6
    fi
done

echo "############ FINISH-TREES-PAINT DONE $(date +%H:%M:%S)"
