# Mutual exclusion for GPU work. Source this, then call gpu_acquire / gpu_release.
#
# The process-absence check it replaces was a check-then-act race, and it cost a
# run tonight: when the octree-448 job exited, two waiting scripts polled within
# seven seconds of each other, both saw zero GPU processes, and both started. One
# of them reserved 11.4 GB of shared GPU memory -- which is system RAM -- and the
# other then died unable to allocate three megabytes.
#
# mkdir is the fix because it is atomic: the directory either did not exist and
# this process created it, or it existed and mkdir fails. There is no window
# between the test and the claim for a second process to slip through. A lock
# file would not do -- `[ -f lock ] || touch lock` has exactly the same race the
# process check had.
#
# The lock records its owner PID so a lock left behind by a killed script can be
# recognised as stale rather than blocking every later run forever.
GPU_LOCK="${GPU_LOCK:-C:/Users/Lauri/AppData/Local/Temp/claude/C--Users-Lauri-Desktop/bef7e8c6-36b0-437d-85a9-2492519bc896/scratchpad/gpu.lock}"

gpu_busy() {
  # Anything already running that this lock did not start. Blender counts: a
  # render during a paint run has killed two paint runs today.
  powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { (\$_.Name -eq 'python.exe' -and (\$_.CommandLine -match 'hunyuan_paint' -or \$_.CommandLine -match 'mini_turbo_generate' -or \$_.CommandLine -match 'avatar_preprocess')) -or \$_.Name -eq 'blender.exe' -or \$_.Name -eq 'trellis-cli.exe' }).Count" 2>/dev/null | tr -d '\r\n '
}

gpu_acquire() {
  label="$1"
  while true; do
    if mkdir "$GPU_LOCK" 2>/dev/null; then
      echo "$$ $label" > "$GPU_LOCK/owner"
      # Holding the lock is necessary but not sufficient: a job started before
      # the lock existed still owns the GPU. Wait it out, keeping the lock so
      # nobody queues behind us in the meantime.
      while true; do
        n=$(gpu_busy)
        case "$n" in ''|*[!0-9]*) sleep 15; continue ;; esac
        [ "$n" -eq 0 ] && break
        sleep 15
      done
      echo "=== lock acquired by $label $(date +%H:%M:%S)"
      sleep 5
      return 0
    fi
    owner=$(cat "$GPU_LOCK/owner" 2>/dev/null)
    pid=${owner%% *}
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "=== stale lock from pid $pid, clearing $(date +%H:%M:%S)"
      rm -rf "$GPU_LOCK"
      continue
    fi
    sleep 15
  done
}

gpu_release() {
  rm -rf "$GPU_LOCK"
  echo "=== lock released $(date +%H:%M:%S)"
}
