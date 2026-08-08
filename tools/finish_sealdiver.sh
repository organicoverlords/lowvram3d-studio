#!/bin/bash
# Resume the seal diver after the geometry stage, which is already on disk.
#
# A separate file rather than a re-run of run_sealdiver.sh, for two reasons.
# Re-running would redo twenty minutes of TRELLIS to reach a mesh that already
# exists; and editing the original while a shell is executing it corrupts the
# run, because a shell reads a script by byte offset (run_moss_titan.sh emitted
# `line 44: ceipt: command not found` that way).
#
# The gate that stopped the first run was miscalibrated by me, not tripped by
# the asset: `quad_ratio` was written as a ceiling when a billboard sits at
# exactly 1.0 and every solid sits above it. Fixed in check_not_billboard.py,
# re-verified against four meshes of known truth, and re-run here.
set -e
cd "$(dirname "$0")/.."

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
HY="C:/AI/HY3D2/python_standalone/python.exe"
D=evidence/compare/sealdiver
VIEWS="profile,three_quarter,end_plus,three_quarter_rear,profile_far,three_quarter_far,end_minus,plan,below"
LIVE=$D/run_live.log

stage() { printf '\n[%s] === %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$LIVE"; }
run()   { "$@" 2>&1 | tee -a "$LIVE" | tail -4; return "${PIPESTATUS[0]}"; }

render() {
    [ -s "$1" ] || return 0
    for mode in native clay; do
        "$PY" workers/render_asset_views.py --mesh "$1" --out "${2}_${mode}.png" \
            --views "$VIEWS" --size 2000 --$mode >/dev/null 2>&1 || continue
        "$PY" workers/reflow_view_sheet.py --sheet "${2}_${mode}.png" \
            --views 9 --columns 3 >/dev/null 2>&1 || true
        echo "    rendered ${2}_${mode}_grid.png" | tee -a "$LIVE"
    done
}

stage "billboard gate (recalibrated)"
run "$PY" tools/check_not_billboard.py --mesh $D/sealdiver_t1024.glb \
    --receipt $D/sealdiver_t1024.billboard.json

stage "geometry views"
render $D/sealdiver_t1024.glb $D/views_sealdiver_t1024

stage "vendor paint 2048 (512 conditioning)"
PYTHONPATH="C:/AI/HY3D2/Hunyuan3D-2" run "$HY" workers/hunyuan_paint_texture.py \
    --mesh $D/sealdiver_t1024.glb --image $D/sealdiver_input_512.png \
    --out $D/sealdiver_t1024_paint2048.glb \
    --render-size 1024 --texture-size 2048 \
    --receipt $D/sealdiver_t1024_paint2048.paint.json

stage "painted views"
render $D/sealdiver_t1024_paint2048.glb $D/views_sealdiver_paint2048

stage "############ SEAL DIVER DONE"
ls -la $D/*_grid.png 2>/dev/null | awk '{print "  ", $NF, $5}'
