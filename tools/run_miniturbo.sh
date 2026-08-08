#!/bin/bash
# One subject through Mini Turbo, for the cases TRELLIS cannot do.
#
#     tools/run_miniturbo.sh treecity
#
# TRELLIS collapses a subject whose silhouette is carried by structures thinner
# than one sparse-structure cell -- the greentree's aerial roots became two
# crossed cardboard panels at both 512 and 1024. Mini Turbo built a correct
# 4M-face banyan from the same image. This is that lane.
#
# Parameters are the ones that produced the working greentree, read from
# evidence/compare/greentree/greentree_miniturbo_a1.json rather than guessed:
# 5 steps, guidance 5.0, seed 1007, and an octree ladder that steps 384 -> 320
# -> 256 so an OOM at the top rung retries lower instead of failing the run.
set -e
cd "$(dirname "$0")/.."

SUBJECT="${1:?usage: run_miniturbo.sh <subject>}"
SEED="${2:-1007}"

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
HY="C:/AI/HY3D2/python_standalone/python.exe"
D="evidence/compare/$SUBJECT"
VIEWS="profile,three_quarter,end_plus,three_quarter_rear,profile_far,three_quarter_far,end_minus,plan,below"
LIVE="$D/run_live_mt.log"
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

[ -s "$D/mini_turbo_conditioning.png" ] || {
    echo "no conditioning at $D/mini_turbo_conditioning.png"; exit 1; }

stage "Mini Turbo (octree ladder 384 -> 320 -> 256, seed $SEED)"
# Both --image and --conditioning-image: the worker requires --image (it is the
# provenance record, hashed into the receipt) and uses --conditioning-image as
# what the model actually sees. Passing only the latter exits on a missing
# argument before touching the GPU.
run "$PY" workers/mini_turbo_generate.py \
    --image "$D/matte.png" \
    --conditioning-image "$D/mini_turbo_conditioning.png" \
    --output "$D/${SUBJECT}_miniturbo.glb" \
    --result-json "$D/${SUBJECT}_miniturbo.json" \
    --model-root "C:/AI/HY3D2/HuggingFaceHub/hunyuan3d-2mini-direct" \
    --steps 5 --guidance-scale 5.0 --seed "$SEED" \
    --octree-ladder "384:3000,320:2000,256:1500" --mc-algo mc

stage "billboard gate"
run "$PY" tools/check_not_billboard.py --mesh "$D/${SUBJECT}_miniturbo.glb" \
    --matte "$D/matte.png" --receipt "$D/${SUBJECT}_miniturbo.billboard.json"

stage "geometry views"
render "$D/${SUBJECT}_miniturbo.glb" "$D/views_${SUBJECT}_miniturbo"

stage "############ $SUBJECT MINI TURBO DONE"
ls -la "$D"/views_${SUBJECT}_miniturbo*_grid.png 2>/dev/null | awk '{print "  ", $NF, $5}'
