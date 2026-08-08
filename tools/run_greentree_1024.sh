#!/bin/bash
# Greentree retry at res 1024, with the billboard gate.
#
# The 512 attempt finished in 264 s and produced two crossed cardboard panels:
# fill 0.010, surface area 0.998 of the crossed quads the bounding box allows,
# 55% of vertices piled on a single plane on each of two axes. The receipt said
# success, the finalizer ran every stage, and nothing short of rendering it
# showed the problem. check_not_billboard.py is that check as a number now.
#
# Resolution is the reason to expect a different answer rather than the same one
# more slowly. The sparse-structure stage runs on a 32^3 grid at --res 512, so
# one structural cell is 16 source pixels; a banyan's aerial roots are thinner
# than that and lose the cell to the canopy above them, which is how the whole
# subject collapses onto the two planes that carry most of its silhouette. At
# 1024 the grid is 64^3 and a root gets cells of its own. Every asset in this
# project that came out solid was generated at 1024; the tree is the only one
# that was not.
#
# Not a guarantee. If it collapses again the gate stops it before the paint, and
# the answer for this subject is Mini Turbo, whose greentree is already a
# correct 4M-face tree with hanging roots.
set -e
cd "$(dirname "$0")/.."

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
HY="C:/AI/HY3D2/python_standalone/python.exe"
D=evidence/compare/greentree
VIEWS="profile,three_quarter,end_plus,three_quarter_rear,profile_far,three_quarter_far,end_minus,plan,below"
LIVE=$D/run_live_1024.log
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

stage "conditioning 1024 and 512"
run "$PY" workers/prepare_input.py --image $D/matte.png \
    --out $D/greentree_input_1024.png --res 1024 \
    --receipt $D/greentree_input_1024.json
run "$PY" workers/prepare_input.py --image $D/matte.png \
    --out $D/greentree_input_512.png --res 512 \
    --receipt $D/greentree_input_512.json

stage "TRELLIS 1024 seed 12345"
run "$PY" workers/trellis_run.py --image $D/greentree_input_1024.png \
    --out $D/greentree_t1024.glb --res 1024 --seed 12345 --tex-res 512 \
    --receipt $D/greentree_t1024.json --log $D/greentree_t1024.log

stage "billboard gate"
run "$PY" tools/check_not_billboard.py --mesh $D/greentree_t1024.glb \
    --receipt $D/greentree_t1024.billboard.json

stage "geometry views"
render $D/greentree_t1024.glb $D/views_greentree_t1024

stage "vendor paint 2048 (512 conditioning)"
PYTHONPATH="C:/AI/HY3D2/Hunyuan3D-2" run "$HY" workers/hunyuan_paint_texture.py \
    --mesh $D/greentree_t1024.glb --image $D/greentree_input_512.png \
    --out $D/greentree_t1024_paint2048.glb \
    --render-size 1024 --texture-size 2048 \
    --receipt $D/greentree_t1024_paint2048.paint.json

stage "painted views"
render $D/greentree_t1024_paint2048.glb $D/views_greentree_paint2048

stage "############ GREENTREE 1024 DONE"
ls -la $D/*_grid.png 2>/dev/null | awk '{print "  ", $NF, $5}'
