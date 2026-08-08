# A window that shows where a run actually is, refreshed in place.
#
# Written because a 25-minute stage was invisible to both of us. Two causes, and
# this fixes the second while run_moss_titan.sh fixes the first:
#
#   1. `command | tail -3` cannot print anything until the command exits, so
#      piping a long stage into tail hides every line it emits. The runner now
#      tees to $Dir/run_live.log instead.
#   2. Backgrounded work had no visible surface at all. This is that surface.
#
# Open it beside the work:
#
#   powershell -NoProfile -File tools\watch_run.ps1 -Dir evidence\compare\moss_titan
#
# It reads only -- it never touches the run, so closing the window is safe.

param(
    [string]$Dir = "evidence\compare\moss_titan",
    [int]$IntervalSeconds = 5
)

$ErrorActionPreference = "SilentlyContinue"
$start = Get-Date
$host.UI.RawUI.WindowTitle = "LowVRAM3D run watch - $Dir"

while ($true) {
    $lines = @()
    $lines += "=== LowVRAM3D run watch ============================================"
    $lines += ("dir      : {0}" -f $Dir)
    $lines += ("watching : {0:hh\:mm\:ss}   now {1}" -f ((Get-Date) - $start), (Get-Date -Format "HH:mm:ss"))
    $lines += ""

    # GPU. The interesting columns are the ones that told us a run was alive
    # without any log output at all: utilisation pinned at 100 and VRAM creeping
    # upward is a decoder making progress, not a hang.
    $gpu = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw `
                        --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        $f = $gpu -split "," | ForEach-Object { $_.Trim() }
        $lines += ("GPU      : {0} / {1} MiB   util {2}%   {3} C   {4} W" -f $f[0], $f[1], $f[2], $f[3], $f[4])
    } else {
        $lines += "GPU      : nvidia-smi unavailable"
    }

    $os = Get-CimInstance Win32_OperatingSystem
    $lines += ("RAM free : {0:N1} GB of {1:N1} GB" -f ($os.FreePhysicalMemory / 1MB), ($os.TotalVisibleMemorySize / 1MB))

    # Driver faults are what separate a GPU fault from a host-side failure, so
    # they belong on the same screen as the run rather than in a later autopsy.
    $faults = @(Get-WinEvent -FilterHashtable @{LogName = 'System'; StartTime = $start } -ErrorAction SilentlyContinue |
        Where-Object { $_.ProviderName -match 'nvlddmkm' -and ($_.Id -eq 13 -or $_.Id -eq 153) })
    $lines += ("nvlddmkm : {0} fault(s) since this watch started" -f $faults.Count)

    $procs = Get-CimInstance Win32_Process -Filter "Name='trellis-cli.exe' or Name='python.exe' or Name='blender.exe'" |
        Where-Object { $_.CommandLine -match 'trellis|hunyuan|render_asset|prepare_input' }
    $lines += ""
    $lines += ("stage    : {0}" -f $(if ($procs) {
        ($procs | ForEach-Object {
            switch -Regex ($_.CommandLine) {
                'trellis'       { "TRELLIS"; break }
                'hunyuan'       { "vendor paint"; break }
                'render_asset'  { "rendering views"; break }
                'prepare_input' { "conditioning"; break }
                default         { $_.Name }
            }
        } | Select-Object -Unique) -join ", "
    } else { "no pipeline process running" }))

    # The engine itself. A run with no log at all -- because trellis_run.py used
    # to write its log only after the child exited, or because the runner
    # predates the tee fix -- is still fully readable here:
    #
    #   CPU climbing ~1s per wall second  = one core pinned, work happening
    #   CPU flat                          = actually stuck
    #   host RAM step change              = a stage transition
    #   VRAM flat + util 100              = a sampler running fixed-shape steps,
    #                                       which is the HEALTHY signature
    #   VRAM creeping to the card limit
    #     with util jittering             = Windows paging VRAM, the slow death
    $cli = Get-Process -Name 'trellis-cli' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cli) {
        $cpu = [math]::Round($cli.CPU)
        $wall = ((Get-Date) - $cli.StartTime).TotalSeconds
        $rate = if ($script:lastCpu -ne $null) { $cpu - $script:lastCpu } else { $null }
        $script:lastCpu = $cpu
        $lines += ("engine   : trellis-cli pid {0}  alive {1:hh\:mm\:ss}  CPU {2}s ({3:P0} of one core)" -f `
            $cli.Id, [TimeSpan]::FromSeconds($wall), $cpu, ($cpu / [math]::Max($wall, 1)))
        $lines += ("           host RAM {0:N0} MB{1}" -f ($cli.WorkingSet64 / 1MB),
            $(if ($rate -ne $null) { "   +{0}s CPU since last refresh{1}" -f $rate,
                $(if ($rate -le 0) { "   <-- NOT PROGRESSING" } else { "" }) } else { "" }))
    }

    $live = Join-Path $Dir "run_live.log"
    $lines += ""
    if (Test-Path $live) {
        $lines += "--- run_live.log (last 14) -----------------------------------------"
        $lines += (Get-Content $live -Tail 14)
    } else {
        $lines += "--- no run_live.log yet ---------------------------------------------"
        $lines += "  (a run started before the tee fix writes nothing here; the newest"
        $lines += "   output files below are the progress signal in that case)"
    }

    $lines += ""
    $lines += "--- newest files ----------------------------------------------------"
    $lines += (Get-ChildItem $Dir -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 6 |
        ForEach-Object { "  {0:HH:mm:ss}  {1,10:N0}  {2}" -f $_.LastWriteTime, $_.Length, $_.Name })

    Clear-Host
    $lines | ForEach-Object { Write-Host $_ }
    Start-Sleep -Seconds $IntervalSeconds
}
