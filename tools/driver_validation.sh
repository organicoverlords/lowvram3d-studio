#!/bin/sh
# Measure the GPU fault rate against the real workload, after a driver change.
#
# gpu_fault_probe.py is not the instrument for this. It hammered torch.randn on
# CUDA for 104,900 launches without a single fault, so it cannot distinguish a
# fixed machine from a broken one -- reporting it green would mean nothing.
#
# The only thing that ever reproduced the fault is a full TRELLIS run, so that
# is what gets repeated. Fixed image, fixed seed, fixed resolution: the same
# configuration the user measured at 7 failures in 12 attempts (58%). Ten clean
# runs at that base rate is p = 0.42^10 ~ 0.0002; one clean run is noise.
#
# Every run records whether the driver logged an nvlddmkm 13 (graphics
# exception) or 153 (engine reset) inside its own window. That is what separates
# a GPU fault from a host-side failure -- the fennec SIGSEGV at 22:23:01 had no
# driver event and was heap corruption, an unrelated bug.
#
#     sh tools/driver_validation.sh 10 "debug-mode + hags-off + 610.88"
set -e
cd "$(dirname "$0")/.."

RUNS=${1:-10}
LABEL=${2:-unlabelled}

PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
IMAGE=evidence/compare/panda/panda_matte.png
SEED=12345
OUT=evidence/driver_validation
mkdir -p "$OUT"
LOG="$OUT/runs.jsonl"

[ -s "$IMAGE" ] || { echo "missing $IMAGE"; exit 1; }

# nvlddmkm 13/153 raised since the given time, as a bare count.
faults_since() {
    powershell -NoProfile -Command "
        try {
          @(Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=[datetime]'$1'} -ErrorAction Stop |
            Where-Object { \$_.ProviderName -match 'nvlddmkm' -and (\$_.Id -eq 13 -or \$_.Id -eq 153) }).Count
        } catch { 0 }" 2>/dev/null | tr -d '\r\n '
}

echo "[validate] $RUNS runs of TRELLIS 512, seed $SEED, panda_matte -- label: $LABEL"
echo "[validate] baseline for this exact configuration: 7 failures in 12 (58%)"

pass=0
fail=0
gpu_faults=0

i=1
while [ "$i" -le "$RUNS" ]; do
    mark=$(powershell -NoProfile -Command "(Get-Date).AddSeconds(-2).ToString('o')" 2>/dev/null | tr -d '\r\n')
    started=$(date +%s)
    printf '[validate] run %d/%d  %s ... ' "$i" "$RUNS" "$(date +%H:%M:%S)"

    glb="$OUT/run${i}.glb"
    rm -f "$glb"
    if "$PY" workers/trellis_run.py --image "$IMAGE" --out "$glb" \
            --res 512 --seed $SEED --tex-res 512 \
            --receipt "$OUT/run${i}.json" \
            --log "$OUT/run${i}.log" >/dev/null 2>&1 && [ -s "$glb" ]; then
        ok=true
    else
        ok=false
    fi
    elapsed=$(( $(date +%s) - started ))

    # Let the driver finish writing its event before asking for it.
    sleep 3
    ev=$(faults_since "$mark")
    [ -n "$ev" ] || ev=0

    if [ "$ok" = true ]; then
        pass=$((pass + 1)); verdict=pass
    else
        fail=$((fail + 1)); verdict=fail
        [ "$ev" != "0" ] && gpu_faults=$((gpu_faults + 1))
    fi
    echo "$verdict  ${elapsed}s  nvlddmkm=$ev"

    printf '{"label":"%s","run":%d,"verdict":"%s","seconds":%d,"nvlddmkm_13_153":%s}\n' \
        "$LABEL" "$i" "$verdict" "$elapsed" "$ev" >> "$LOG"

    # Keep only the geometry, not ten copies of it.
    [ "$verdict" = pass ] && [ "$i" != "1" ] && rm -f "$glb"
    i=$((i + 1))
done

echo "############ $LABEL: $pass pass / $fail fail out of $RUNS"
echo "############ failures carrying an nvlddmkm 13/153: $gpu_faults"
echo "############ baseline was 58% fail -- 0/$RUNS here is p~0.0002"
