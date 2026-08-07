#!/bin/sh
# When the trees finish: render them native + clay FIRST, then the fennec 2048 bake.
#
# Ordering is the point. The trees script still has greentree TRELLIS and four
# decimate/unwrap/paint passes to run after bluetree, so anything that polls for
# a single output file fires into the gap between stages and lands on top of a
# live GPU job -- which is what the previous fennec queue would have done. So
# this waits for the whole script to exit, not for a file to appear.
#
# render_asset_views.py is the fast one: Blender headless, EEVEE, a few seconds
# per view, every mesh and view in one launch. --native shades from whatever
# appearance the mesh already carries (TRELLIS baked texture, or Mini Turbo
# COLOR_0 through a Vertex Color node); without it the worker forces clay on
# anything with no image texture and the vertex colours are thrown away.
set -e
cd "$(dirname "$0")/.."

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
HY="C:/AI/HY3D2/python_standalone/python.exe"

# `ps -W` lists sh.exe with no arguments, so grepping it for the script name
# matches nothing and the wait falls straight through onto the live GPU job.
# Only the WMI command line carries the script name, so ask PowerShell.
running() {
    n=$(powershell -NoProfile -Command \
        "@(Get-CimInstance Win32_Process -Filter \"Name='sh.exe'\" | Where-Object { \$_.CommandLine -like '*trees_trellis_and_paint*' }).Count" \
        2>/dev/null | tr -d '\r\n ')
    [ "$n" != "0" ] && [ -n "$n" ]
}

echo "[after-trees] waiting for trees_trellis_and_paint.sh to exit"
if ! running; then
    echo "[after-trees] ERROR: trees script not detected as running; refusing to"
    echo "[after-trees] start, since that is indistinguishable from a broken check"
    exit 1
fi
while running; do
    sleep 30
done
echo "[after-trees] trees done at $(date +%H:%M:%S)"

########## trees first, native + clay, both generators
for name in bluetree greentree; do
    D=evidence/compare/$name
    for gen in miniturbo t1024; do
        src=$D/${name}_${gen}.glb
        [ -s "$src" ] || { echo "[after-trees] missing $src"; continue; }
        for mode in native clay; do
            echo "[after-trees] $name $gen $mode $(date +%H:%M:%S)"
            "$PY" workers/render_asset_views.py \
                --mesh "$src" \
                --out $D/views_${name}_${gen}_${mode}.png \
                --views profile,three_quarter,end_plus \
                --size 900 --$mode 2>&1 | tail -2
        done
    done
done
echo "[after-trees] tree renders complete $(date +%H:%M:%S)"

########## then the controlled 2048 fennec rebake
# Everything held fixed against the 1024 run -- same mesh, same conditioning,
# same cameras, same UVs -- so texture_size is the only variable. --render-size
# stays 512 deliberately; note it remains a second untested difference from the
# assets that passed, which ran the vendor default of 2048 there too.
SUBJ=evidence/compare/fennec
echo "[after-trees] fennec 2048 bake $(date +%H:%M:%S)"
PYTHONPATH=C:/AI/HY3D2/Hunyuan3D-2 "$HY" workers/hunyuan_paint_texture.py \
    --mesh "$SUBJ/fennec_t_uv.glb" \
    --image "$SUBJ/matte_512.png" \
    --out "$SUBJ/fennec_t_hypaint2048.glb" \
    --render-size 512 --texture-size 2048 \
    --receipt "$SUBJ/fennec_t_hypaint2048.paint.json" \
    > "$SUBJ/paint_t2048.log" 2>&1
echo "[after-trees] bake exit $? $(date +%H:%M:%S)"

if [ -s "$SUBJ/fennec_t_hypaint2048.glb" ]; then
    "$PY" workers/render_asset_views.py \
        --mesh "$SUBJ/fennec_t_hypaint2048.glb" \
        --out "$SUBJ/views_fennec_t2048_native.png" \
        --views profile,three_quarter,end_plus --size 900 --native 2>&1 | tail -2
else
    tail -12 "$SUBJ/paint_t2048.log"
fi
echo "############ AFTER-TREES DONE $(date +%H:%M:%S)"
