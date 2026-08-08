#!/bin/bash
# The moss titan's post-TRELLIS stages, runnable on their own.
#
# run_moss_titan.sh was edited at 03:55:51 while an `sh` was still executing it.
# A shell reads a script by byte offset, and that edit changed the length of
# lines above the running position, so if that shell re-reads from its saved
# offset the remainder -- geometry views, vendor paint, painted views -- can be
# mangled into garbage or skipped silently. The TRELLIS stage is the expensive
# one and it is already 40 minutes in; losing the four cheap stages after it to
# a shell-offset bug would mean paying that cost twice.
#
# So this file is the same tail of the pipeline with no dependency on that
# shell. It is idempotent: every stage checks for its own output first, so it
# is safe to run whether the original script completed the stage, mangled it,
# or never reached it.
#
#     bash tools/finish_moss_titan.sh
#
# The one thing it will not do is regenerate the geometry. If titan_t1024.glb
# is absent it stops and says so, rather than quietly producing views of
# nothing.
set -e
cd "$(dirname "$0")/.."

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
HY="C:/AI/HY3D2/python_standalone/python.exe"
D=evidence/compare/moss_titan
VIEWS="profile,three_quarter,end_plus,three_quarter_rear,profile_far,three_quarter_far,end_minus,plan,below"

LIVE=$D/finish_live.log
: >> "$LIVE"

stage() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$LIVE"; }

# Nine views, clay AND native, after every stage that changes the asset. Clay
# shows form and native shows what the asset actually carries; a stage that
# breaks one routinely leaves the other looking fine, which is why one sheet is
# never enough to judge a stage by.
render() {   # mesh, prefix
    if [ ! -s "$1" ]; then
        stage "  skip views: $1 absent"
        return 0
    fi
    for mode in native clay; do
        if [ -s "${2}_${mode}_grid.png" ]; then
            stage "  have ${2}_${mode}_grid.png"
            continue
        fi
        "$PY" workers/render_asset_views.py --mesh "$1" --out "${2}_${mode}.png" \
            --views "$VIEWS" --size 2000 --$mode 2>&1 | tee -a "$LIVE" || continue
        "$PY" workers/reflow_view_sheet.py --sheet "${2}_${mode}.png" \
            --views 9 --columns 3 2>&1 | tee -a "$LIVE" || true
        stage "  rendered ${2}_${mode}_grid.png"
    done
}

if [ ! -s $D/titan_t1024.glb ]; then
    stage "titan_t1024.glb absent -- TRELLIS has not finished. Nothing to do."
    exit 2
fi

stage "geometry views"
render $D/titan_t1024.glb $D/views_titan_t1024

# The vendor texture stage, not a hand-rolled projection. A single-view CPU
# projection observes under 10% of the atlas and has to invent the rest; this
# is the stage that exists to retire that failure mode.
if [ -s $D/titan_t1024_paint2048.glb ]; then
    stage "vendor paint 2048 -- already present, skipping"
else
    # Conditioning is 512, NOT the 1024 the geometry stage used. The multiview
    # paint pipeline runs float16; cuBLAS picks a GEMM algorithm per tensor
    # shape, and shapes derived from a large conditioning image land on a
    # tensor-core path TU116 does not have. It is shape-dependent rather than
    # size-dependent, which is why it reads as random: the whale and heron
    # painted fine on large mattes and the shaman failed four times across three
    # meshes until the only change made was swapping in a 512 input.
    # Large images stay correct for TRELLIS geometry -- only paint cares.
    stage "vendor paint 2048 (512 conditioning)"
    PYTHONPATH="C:/AI/HY3D2/Hunyuan3D-2" "$HY" workers/hunyuan_paint_texture.py \
        --mesh $D/titan_t1024.glb --image $D/paint_input_512.png \
        --out $D/titan_t1024_paint2048.glb \
        --render-size 1024 --texture-size 2048 \
        --receipt $D/titan_t1024_paint2048.paint.json 2>&1 | tee -a "$LIVE"
fi

stage "painted views"
render $D/titan_t1024_paint2048.glb $D/views_titan_paint2048

stage "############ MOSS TITAN DONE"
ls -la $D/*_grid.png 2>/dev/null | awk '{print "  ", $NF, $5}' | tee -a "$LIVE"
