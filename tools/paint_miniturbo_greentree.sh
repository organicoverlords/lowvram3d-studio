#!/bin/bash
# Paint the Mini Turbo greentree -- the tree that is actually a tree.
#
# TRELLIS produced a billboard for this subject at both 512 and 1024, so the
# geometry here comes from Mini Turbo instead: 4,031,944 faces of layered canopy
# and hanging aerial roots, fill 0.171, quad_ratio 2.85.
#
# The input is greentree_mt_uv.glb, not greentree_miniturbo.glb. That is the
# 400k-face LOD carrying 296,493 UVs from the earlier unwrap, and it matters for
# two reasons: 4M faces will not survive the paint stage, and the UVs let
# mesh_uv_wrap skip xatlas entirely. Re-parameterising a mesh that already has
# valid UVs is what made three paint runs take 25 minutes without finishing
# (see runbook 9a) -- and this asset is exactly the shape that is worst for it,
# thousands of thin separate root strands, each its own chart.
#
# Watch the log for:
#   [paint] >> mesh_uv_wrap SKIPPED: mesh already carries 296493 UVs
# If it instead says it is running xatlas, the UVs did not survive load and the
# run should be stopped rather than left for an hour.
set -e
cd "$(dirname "$0")/.."

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
HY="C:/AI/HY3D2/python_standalone/python.exe"
D=evidence/compare/greentree
VIEWS="profile,three_quarter,end_plus,three_quarter_rear,profile_far,three_quarter_far,end_minus,plan,below"
LIVE=$D/run_live_mtpaint.log
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

stage "billboard gate on the Mini Turbo geometry"
run "$PY" tools/check_not_billboard.py --mesh $D/greentree_mt_uv.glb \
    --receipt $D/greentree_mt_uv.billboard.json

stage "vendor paint 2048 (512 conditioning)"
PYTHONPATH="C:/AI/HY3D2/Hunyuan3D-2" run "$HY" workers/hunyuan_paint_texture.py \
    --mesh $D/greentree_mt_uv.glb --image $D/greentree_input_512.png \
    --out $D/greentree_mt_paint2048.glb \
    --render-size 1024 --texture-size 2048 \
    --receipt $D/greentree_mt_paint2048.paint.json

stage "painted views"
render $D/greentree_mt_paint2048.glb $D/views_greentree_mt_paint2048

stage "############ MINI TURBO GREENTREE PAINT DONE"
ls -la $D/views_greentree_mt_paint2048*_grid.png 2>/dev/null | awk '{print "  ", $NF, $5}'
