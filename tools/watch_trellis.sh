#!/bin/sh
# Watch a TRELLIS run and say something useful whether it progresses OR stops.
#
#     tools/watch_trellis.sh evidence/compare/fattree/fattree_t512.log 420
#
# Watching for success markers only is what let the fat tree burn 40 minutes:
# the run was "fine" by every signal being checked -- GPU at 100%, VRAM steady,
# process alive, CPU climbing -- and had not completed one step of twelve. None
# of those signals distinguish working from thrashing. The only signal that does
# is the job's own output, and the only way to use it is to notice its ABSENCE.
#
# So this reports three things, not one:
#
#   PROGRESS  a new line reached the log, with the seconds since the last one,
#             which makes a slowdown visible while it is still cheap to fix
#   STALL     nothing written for --stall seconds; repeats, escalating, so a
#             wedged run cannot be mistaken for a quiet one
#   EXIT      the process is gone, with whether the output actually exists
#
# The log must be the one trellis_run.py writes with --log. That file is flushed
# per line; the shell's tee copy is not a substitute, because a progress bar
# redrawing with carriage returns never ends a line and so never appears there.
LOG="${1:?usage: watch_trellis.sh <log> [stall_seconds]}"
STALL="${2:-420}"

# Wait for the log to exist rather than failing: the caller usually starts this
# at the same moment as the run.
waited=0
while [ ! -f "$LOG" ]; do
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge 300 ]; then
        echo "NO LOG at $LOG after ${waited}s -- did the run start?"
        exit 1
    fi
done

previous=$(wc -c < "$LOG")
quiet=0
warned=0

while true; do
    sleep 20
    current=$(wc -c < "$LOG" 2>/dev/null || echo "$previous")

    if [ "$current" != "$previous" ]; then
        line=$(tail -1 "$LOG" | tr -d '\r')
        echo "PROGRESS +${quiet}s | $line"
        previous=$current
        quiet=0
        warned=0
    else
        quiet=$((quiet + 20))
        # Escalating: once at the threshold, then every 5 minutes, so a long
        # stall keeps saying so instead of being reported once and forgotten.
        if [ "$quiet" -ge "$STALL" ] && [ $((quiet % 300)) -lt 20 ]; then
            warned=$((warned + 1))
            used=$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
                   --format=csv,noheader 2>/dev/null | tr -d '\n')
            echo "STALL #${warned}: no output for ${quiet}s (threshold ${STALL}s). GPU ${used}."
            echo "  If VRAM is near total, the driver is paging rather than failing -- steps crawl, nothing errors."
        fi
    fi

    if ! tasklist 2>/dev/null | grep -qi trellis-cli; then
        target=$(echo "$LOG" | sed 's/\.log$/.glb/')
        if [ -s "$target" ]; then
            echo "EXIT: trellis-cli gone, output written ($(wc -c < "$target") bytes)"
        else
            echo "EXIT: trellis-cli gone, NO OUTPUT at $target -- the run failed"
        fi
        exit 0
    fi
done
