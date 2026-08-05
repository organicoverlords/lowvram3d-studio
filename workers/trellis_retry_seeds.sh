#!/usr/bin/env bash
# Retry TRELLIS across seeds until one completes.
#
# Not a hack. On TU116 the sparse-structure stage is nondeterministic, so every
# run hands the downstream stages differently-shaped tensors, and this card has
# shape-dependent kernel faults (`misaligned address`, tensor-op launch failure)
# that no flag reliably avoids. A large subject rolls those dice more times than
# a small one, so it fails more often -- but each attempt is independent, and a
# subject that fails four times can succeed on the fifth with nothing changed
# but the draw.
#
# Seed is the cleanest way to force a different draw. It is not a quality lever
# here: there is no reason to prefer one seed's geometry over another's, so
# taking the first that survives costs nothing.
#
# Stops at the first success and reports which seed won, so the run is
# reproducible afterwards.
set -u
PY="C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe"
IMAGE="$1"; OUT="$2"; DIR="$(dirname "$OUT")"
shift 2
SEEDS="${*:-12345 777 20260806 4242 31337 8675309}"

for seed in $SEEDS; do
  echo "=== seed $seed ==="
  "$PY" workers/trellis_run.py \
      --image "$IMAGE" --out "$OUT" \
      --res 512 --atlas 1024 --seed "$seed" \
      --receipt "$DIR/run_seed$seed.json" \
      --log "$DIR/run_seed$seed.log" > "$DIR/stdout_seed$seed.txt" 2>&1
  status=$?
  faces=$(grep -oE 'mesh V=[0-9]+ F=[0-9]+' "$DIR/run_seed$seed.log" 2>/dev/null | tail -1)
  if [ $status -eq 0 ] && [ -s "$OUT" ]; then
    echo "SUCCESS seed=$seed  $faces"
    echo "$seed" > "$DIR/winning_seed.txt"
    exit 0
  fi
  echo "failed seed=$seed status=$status  ${faces:-no geometry}  $(tail -3 "$DIR/run_seed$seed.log" | tr '\n' ' ')"
done
echo "ALL SEEDS FAILED"
exit 1
