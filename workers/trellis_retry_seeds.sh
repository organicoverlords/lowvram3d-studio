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

# Overridable so a setting listed as "does not work" can be re-tested under the
# retry regime. Several such entries were written from single attempts, before
# it was clear that a single failure here means almost nothing.
ATLAS="${ATLAS:-1024}"
RES="${RES:-512}"
TAG="${TAG:-}"
#: Extra flags passed straight to trellis-cli, e.g. EXTRA="--f32". Kept as a
#: single string because the only current use is one or two flags.
EXTRA="${EXTRA:-}"

# Refuse to start if another trellis-cli already holds the card.
#
# This has cost real work twice. TaskStop and Ctrl-C kill the shell, not the
# process tree, so a "stopped" sweep keeps looping and launches its next seed
# seconds after you verify that nothing is running. Two runs then share 6 GB,
# and the resulting `illegal memory access` / `unknown error` at stage 3 looks
# exactly like a hardware fault attributable to the subject. Three shaman seeds
# were written off that way before anyone looked at the process list.
if command -v tasklist >/dev/null 2>&1; then
  running=$(tasklist //FI "IMAGENAME eq trellis-cli.exe" //NH 2>/dev/null \
            | grep -ci trellis-cli || true)
  if [ "${running:-0}" -gt 0 ]; then
    echo "REFUSING TO START: $running trellis-cli process(es) already running." >&2
    echo "A concurrent run on a 6 GB card produces failures that look like" >&2
    echo "subject-specific hardware faults. Kill them, or wait." >&2
    exit 4
  fi
fi

for seed in $SEEDS; do
  echo "=== seed $seed ==="
  "$PY" workers/trellis_run.py \
      --image "$IMAGE" --out "$OUT" \
      --res "$RES" --atlas "$ATLAS" --seed "$seed" \
      --receipt "$DIR/run$TAG-seed$seed.json" \
      --log "$DIR/run$TAG-seed$seed.log" \
      ${EXTRA:+--extra $EXTRA} > "$DIR/stdout$TAG-seed$seed.txt" 2>&1
  status=$?
  faces=$(grep -oE 'mesh V=[0-9]+ F=[0-9]+' "$DIR/run$TAG-seed$seed.log" 2>/dev/null | tail -1)
  if [ $status -eq 0 ] && [ -s "$OUT" ]; then
    echo "SUCCESS seed=$seed res=$RES atlas=$ATLAS  $faces"
    echo "$seed" > "$DIR/winning_seed$TAG.txt"
    exit 0
  fi
  echo "failed seed=$seed status=$status  ${faces:-no geometry}  $(tail -3 "$DIR/run$TAG-seed$seed.log" | tr '\n' ' ')"
done
echo "ALL SEEDS FAILED"
exit 1
