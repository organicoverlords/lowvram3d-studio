#!/bin/sh
# Trees, one at a time, at the fastest settings that still produce geometry.
#
# What changed against trees_trellis_and_paint.sh, and why:
#
#   --res 1024 -> 512   the worker's own default. Measured on the same subjects,
#                       512 runs 2-4x faster (whale 129.7s vs 177.7-494.8s).
#   paint phase removed at 512/1024 the four tree paints would have landed at
#                       ~1.4 texels/face -- the same starved configuration the
#                       fennec is being A/B'd against. An hour spent producing
#                       textures we can already predict are unusable. Geometry
#                       first; paint once the fennec A/B says what density works.
#   one tree at a time  each finishes and is rendered before the next starts, so
#                       there is something to look at early rather than only at
#                       the end.
#
# Renders go through render_asset_views.py -- Blender headless, EEVEE, seconds
# per view, all views in one launch. --native shades from the mesh's own
# appearance (Mini Turbo COLOR_0 via a Vertex Color node, or a baked texture);
# without it the worker forces clay and the generator's colours are discarded.
set -e
cd "$(dirname "$0")/.."

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"

busy() {
    n=$(powershell -NoProfile -Command \
        "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'hunyuan_paint' }).Count" \
        2>/dev/null | tr -d '\r\n ')
    [ "$n" != "0" ] && [ -n "$n" ]
}

echo "[trees-fast] waiting for the fennec bake to release the GPU"
while busy; do sleep 20; done
echo "[trees-fast] GPU free at $(date +%H:%M:%S)"

render() {   # mesh, out-prefix
    for mode in native clay; do
        "$PY" workers/render_asset_views.py --mesh "$1" --out "${2}_${mode}.png" \
            --views profile,three_quarter,end_plus --size 900 --$mode 2>&1 | tail -1
    done
}

for name in bluetree greentree; do
    D=evidence/compare/$name
    [ -s $D/matte.png ] || { echo "[trees-fast] $name has no matte, skipping"; continue; }

    # Mini Turbo geometry already exists for both trees; render it now so there
    # is something on screen before TRELLIS has produced anything at all.
    if [ -s $D/${name}_miniturbo.glb ]; then
        echo "[trees-fast] === $name mini turbo render $(date +%H:%M:%S)"
        render $D/${name}_miniturbo.glb $D/views_${name}_miniturbo
    fi

    if [ ! -s $D/${name}_t512.glb ]; then
        for seed in 1007 777 1006; do
            [ -s $D/${name}_t512.glb ] && break
            echo "[trees-fast] === $name TRELLIS res512 seed $seed $(date +%H:%M:%S)"
            "$PY" workers/trellis_run.py --image $D/matte.png \
                --out $D/${name}_t512.glb --res 512 --seed $seed --tex-res 512 \
                --receipt $D/${name}_t512_s$seed.json \
                --log $D/trellis512_s$seed.log 2>&1 | tail -3
        done
    fi

    if [ -s $D/${name}_t512.glb ]; then
        echo "[trees-fast] === $name TRELLIS render $(date +%H:%M:%S)"
        render $D/${name}_t512.glb $D/views_${name}_t512
    else
        echo "[trees-fast] $name TRELLIS failed on all three seeds"
    fi
done

echo "############ TREES FAST DONE $(date +%H:%M:%S)"
