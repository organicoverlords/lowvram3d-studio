#!/bin/bash
# Moss titan through the lane that produced the thirteen working assets.
#
# Vendor texture stage, not a hand-rolled projection. The castle was moved off
# hunyuan_paint_texture.py onto a single-view CPU projection, which observes
# under 10% of the atlas and has to invent the rest; that is the failure mode
# hunyuan_paint_texture.py exists to retire. This run does not repeat it.
#
# Nine views, clay AND native, after every stage that changes the asset -- not
# only at the end. Clay shows form, native shows what the asset actually
# carries, and a stage that breaks one usually leaves the other looking fine.
set -e
cd "$(dirname "$0")/.."

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
HY="C:/AI/HY3D2/python_standalone/python.exe"
D=evidence/compare/moss_titan
VIEWS="profile,three_quarter,end_plus,three_quarter_rear,profile_far,three_quarter_far,end_minus,plan,below"

#: Everything a stage prints goes here as it prints it, so a run in progress can
#: be read instead of guessed at. `| tail -n` cannot emit anything until the
#: process exits, so piping a 20-minute stage into tail makes it invisible by
#: construction -- which is exactly what the first version of this script did.
#: tee writes through; the tail is only for the terminal summary.
LIVE=$D/run_live.log
: > "$LIVE"

stage() {    # label -- one timestamped line into the live log and stdout
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$LIVE"
}

run() {      # any command, streamed to the live log as it goes
    "$@" 2>&1 | tee -a "$LIVE" | tail -3
    return "${PIPESTATUS[0]}"
}

render() {   # mesh, prefix
    [ -s "$1" ] || return 0
    for mode in native clay; do
        "$PY" workers/render_asset_views.py --mesh "$1" --out "${2}_${mode}.png" \
            --views "$VIEWS" --size 2000 --$mode >/dev/null 2>&1 || continue
        "$PY" workers/reflow_view_sheet.py --sheet "${2}_${mode}.png" \
            --views 9 --columns 3 >/dev/null 2>&1 || true
        echo "    rendered ${2}_${mode}_grid.png"
    done
}

stage "conditioning"
run "$PY" workers/prepare_input.py --image $D/matte.png --out $D/paint_input_1024.png \
    --res 1024 --receipt $D/paint_input_1024.json

# --tex-res stays 512. models/fetch_receipt.json lists tex_flow_1024 under
# "excluded" -- it was never fetched, so --tex-res 1024 exits at once with
# WEIGHTS_MISSING. Every asset named trellis1024 is shape 1024 / texture 512,
# and the vendor paint stage supplies the 2048 atlas afterwards, so the texture
# flow resolution is not the ceiling on final appearance.
stage "TRELLIS 1024 seed 12345"
run "$PY" workers/trellis_run.py --image $D/paint_input_1024.png \
    --out $D/titan_t1024.glb --res 1024 --seed 12345 --tex-res 512 \
    --receipt $D/titan_t1024.json --log $D/titan_t1024.log

stage "geometry views"
render $D/titan_t1024.glb $D/views_titan_t1024

# Conditioning is 512 here even though the geometry stage used 1024. The paint
# pipeline runs float16 and cuBLAS picks a GEMM algorithm per tensor shape;
# shapes from a large conditioning image land on a tensor-core path TU116 lacks.
# Shape-dependent, not size-dependent -- see f5459a5. Only paint cares.
stage "vendor paint 2048 (512 conditioning)"
PYTHONPATH="C:/AI/HY3D2/Hunyuan3D-2" run "$HY" workers/hunyuan_paint_texture.py \
    --mesh $D/titan_t1024.glb --image $D/paint_input_512.png \
    --out $D/titan_t1024_paint2048.glb \
    --render-size 1024 --texture-size 2048 \
    --receipt $D/titan_t1024_paint2048.paint.json

stage "painted views"
render $D/titan_t1024_paint2048.glb $D/views_titan_paint2048

stage "############ MOSS TITAN DONE"
ls -la $D/*_grid.png 2>/dev/null | awk '{print "  ", $NF, $5}'
