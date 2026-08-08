#!/bin/bash
# One subject through the whole lane: matte -> conditioning -> TRELLIS -> gate
# -> views -> paint -> views.
#
#     tools/run_asset.sh fattree
#
# Expects evidence/compare/<subject>/source.(png|jpg) to exist. Everything else
# is derived, so a new subject needs a directory and an image and nothing else.
# This replaces the per-subject copies (run_moss_titan.sh, run_sealdiver.sh,
# run_greentree_1024.sh) that had drifted into three nearly identical files.
#
# The billboard gate sits between geometry and paint on purpose. TRELLIS can
# return two crossed cardboard panels and report success -- it did so for the
# greentree at both 512 and 1024 -- and paint is the expensive stage. See
# runbook 10a.
set -e
cd "$(dirname "$0")/.."

SUBJECT="${1:?usage: run_asset.sh <subject> [res] [seed]}"
RES="${2:-1024}"
SEED="${3:-12345}"

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
HY="C:/AI/HY3D2/python_standalone/python.exe"
D="evidence/compare/$SUBJECT"
VIEWS="profile,three_quarter,end_plus,three_quarter_rear,profile_far,three_quarter_far,end_minus,plan,below"
LIVE="$D/run_live.log"
: > "$LIVE"

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

SOURCE=""
for candidate in "$D/source.png" "$D/source.jpg"; do
    [ -f "$candidate" ] && SOURCE="$candidate" && break
done
[ -n "$SOURCE" ] || { echo "no source image in $D"; exit 1; }

# rembg/u2net rather than the colour-distance matte. auto_matte.py measures
# texture energy to separate a smooth shadow from a textured subject, which is
# right for a studio plate on a flat backdrop and wrong for anything else: on
# the seal diver's foggy teal background it selected tolerance 0 and kept 73%
# of the frame, which would have gone into TRELLIS as a slab.
stage "matte"
if [ ! -s "$D/matte.png" ]; then
    run "$HY" tools/matte_rembg.py --image "$SOURCE" --out "$D/matte.png" \
        --receipt "$D/matte.json" --max-side 2048
else
    echo "    matte.png already present, keeping it" | tee -a "$LIVE"
fi

stage "conditioning $RES and 512"
run "$PY" workers/prepare_input.py --image "$D/matte.png" \
    --out "$D/${SUBJECT}_input_${RES}.png" --res "$RES" \
    --receipt "$D/${SUBJECT}_input_${RES}.json"
run "$PY" workers/prepare_input.py --image "$D/matte.png" \
    --out "$D/${SUBJECT}_input_512.png" --res 512 \
    --receipt "$D/${SUBJECT}_input_512.json"

stage "TRELLIS $RES seed $SEED"
run "$PY" workers/trellis_run.py --image "$D/${SUBJECT}_input_${RES}.png" \
    --out "$D/${SUBJECT}_t${RES}.glb" --res "$RES" --seed "$SEED" --tex-res 512 \
    --receipt "$D/${SUBJECT}_t${RES}.json" --log "$D/${SUBJECT}_t${RES}.log"

stage "billboard gate"
run "$PY" tools/check_not_billboard.py --mesh "$D/${SUBJECT}_t${RES}.glb" \
    --receipt "$D/${SUBJECT}_t${RES}.billboard.json"

stage "geometry views"
render "$D/${SUBJECT}_t${RES}.glb" "$D/views_${SUBJECT}_t${RES}"

# 512 conditioning for paint even at higher geometry resolution: the paint
# pipeline runs float16 and cuBLAS picks a GEMM per tensor shape, and shapes
# from a large conditioning image land on a tensor-core path TU116 lacks.
stage "vendor paint 2048 (512 conditioning)"
PYTHONPATH="C:/AI/HY3D2/Hunyuan3D-2" run "$HY" workers/hunyuan_paint_texture.py \
    --mesh "$D/${SUBJECT}_t${RES}.glb" --image "$D/${SUBJECT}_input_512.png" \
    --out "$D/${SUBJECT}_t${RES}_paint2048.glb" \
    --render-size 1024 --texture-size 2048 \
    --receipt "$D/${SUBJECT}_t${RES}_paint2048.paint.json"

stage "painted views"
render "$D/${SUBJECT}_t${RES}_paint2048.glb" "$D/views_${SUBJECT}_paint2048"

stage "segmentation"
run "$PY" workers/segment_mesh.py --mesh "$D/${SUBJECT}_t${RES}_paint2048.glb" \
    --out-dir "$D/parts" --parts 14 --receipt "$D/segment.json"

stage "############ $SUBJECT DONE"
ls -la "$D"/*_grid.png 2>/dev/null | awk '{print "  ", $NF, $5}'
