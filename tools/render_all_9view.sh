#!/bin/sh
# One 9-view sheet per finished asset, at a size that can actually be judged.
#
# Not one combined sheet: 13 assets x 9 views at 1100 px would be ~140
# megapixels, and every panel in the previous 13-asset contact sheet came out at
# 640 px -- below the size at which anything but gross shape is legible. The
# standing rule is that a feature judgement needs the subject at 2000 px or
# more, so a catalogue view cannot double as evidence.
#
# Nine views rather than three, including both rear three-quarters, which are
# where a paint has the least conditioning and is most likely to be wrong.
# reflow_view_sheet.py turns the renderer's single 9-wide row into a 3x3 grid so
# the result is square instead of 9900 px across.
#
# Blender/EEVEE, one launch per asset -- a few seconds each. --native shades
# from the asset's own baked texture.
set -e
cd "$(dirname "$0")/.."

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
VIEWS="profile,three_quarter,end_plus,three_quarter_rear,profile_far,three_quarter_far,end_minus,plan,below"
SIZE=2000
OUT=evidence/deliverables/views9
mkdir -p "$OUT"

for glb in evidence/deliverables/*.glb; do
    name=$(basename "$glb" .glb)
    echo "=== $name $(date +%H:%M:%S)"
    "$PY" workers/render_asset_views.py --mesh "$glb" \
        --out "$OUT/${name}_9view.png" \
        --views "$VIEWS" --size $SIZE --native >/dev/null 2>&1 \
        || { echo "  render FAILED"; continue; }
    "$PY" workers/reflow_view_sheet.py --sheet "$OUT/${name}_9view.png" \
        --views 9 --columns 3 >/dev/null 2>&1 \
        || echo "  reflow failed (single-row sheet kept)"
    ls -la "$OUT/${name}_9view"*.png 2>/dev/null | awk '{print "  ", $NF, $5}'
done
echo "############ 9-VIEW SHEETS DONE $(date +%H:%M:%S)"
