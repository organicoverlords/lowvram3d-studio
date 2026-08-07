#!/bin/sh
# Retry the fennec 2048 bake once the tree paints release the GPU.
#
# Serial, deliberately. Two Hunyuan paint jobs on a 6 GB card contend for both
# VRAM and system RAM, and the first attempt at this bake died at 22:23:01 with
# a bare exit 139 under exactly that kind of pressure.
#
# The first attempt also left no way to tell a crash from an external kill: the
# log ended mid-diffusion with no Python traceback, which is equally consistent
# with a segfault and with another process calling Stop-Process. So this run
# records its own PID, the machine state going in, and the exit code, giving the
# next post-mortem something to work with instead of a guess.
set -e
cd "$(dirname "$0")/.."

SUBJ=evidence/compare/fennec
HY="C:/AI/HY3D2/python_standalone/python.exe"
PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
FORENSIC=$SUBJ/paint_t2048_forensic.txt

running() {
    powershell -NoProfile -Command \
        "@(Get-CimInstance Win32_Process -Filter \"Name='sh.exe'\" | Where-Object { \$_.CommandLine -like '*finish_trees_paint*' }).Count" \
        2>/dev/null | tr -d '\r\n '
}

echo "[fennec2048] waiting for the tree paints to finish $(date +%H:%M:%S)"
while [ "$(running)" != "0" ]; do sleep 20; done

# Nothing else may hold the GPU when this starts.
while [ "$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'hunyuan_paint' }).Count" 2>/dev/null | tr -d '\r\n ')" != "0" ]; do
    sleep 20
done
echo "[fennec2048] GPU free $(date +%H:%M:%S)"

{
    echo "=== fennec 2048 bake, started $(date +%H:%M:%S)"
    powershell -NoProfile -Command \
      "\$os=Get-CimInstance Win32_OperatingSystem; 'free_ram_gb {0:N2}' -f (\$os.FreePhysicalMemory/1MB)" 2>/dev/null
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
} > "$FORENSIC"

PYTHONPATH=C:/AI/HY3D2/Hunyuan3D-2 "$HY" workers/hunyuan_paint_texture.py \
    --mesh "$SUBJ/fennec_t_uv.glb" \
    --image "$SUBJ/matte_512.png" \
    --out "$SUBJ/fennec_t_hypaint2048.glb" \
    --render-size 512 --texture-size 2048 \
    --receipt "$SUBJ/fennec_t_hypaint2048.paint.json" \
    > "$SUBJ/paint_t2048.log" 2>&1 &
BAKE_PID=$!
echo "bake_pid $BAKE_PID" >> "$FORENSIC"
echo "[fennec2048] bake pid $BAKE_PID"

wait $BAKE_PID && status=0 || status=$?
{
    echo "exit_status $status at $(date +%H:%M:%S)"
    powershell -NoProfile -Command \
      "\$os=Get-CimInstance Win32_OperatingSystem; 'free_ram_gb_at_exit {0:N2}' -f (\$os.FreePhysicalMemory/1MB)" 2>/dev/null
    # A Python-level OOM leaves a traceback; a segfault or an external kill does
    # not. This is the line that separates them next time.
    if grep -qE "Traceback|MemoryError" "$SUBJ/paint_t2048.log" 2>/dev/null; then
        echo "python_traceback_present yes  -> died inside Python (likely OOM)"
    else
        echo "python_traceback_present no   -> native crash or external kill"
    fi
} >> "$FORENSIC"

echo "[fennec2048] exit $status $(date +%H:%M:%S)"
cat "$FORENSIC"

if [ -s "$SUBJ/fennec_t_hypaint2048.glb" ]; then
    "$PY" workers/render_asset_views.py \
        --mesh "$SUBJ/fennec_t_hypaint2048.glb" \
        --out "$SUBJ/views_fennec_t2048_native.png" \
        --views profile,three_quarter,end_plus --size 900 --native 2>&1 | tail -1
    echo "[fennec2048] rendered"
else
    echo "[fennec2048] no output; log tail:"
    grep -v "it/s\]" "$SUBJ/paint_t2048.log" 2>/dev/null | tail -8
fi
echo "############ FENNEC 2048 RETRY DONE $(date +%H:%M:%S)"
