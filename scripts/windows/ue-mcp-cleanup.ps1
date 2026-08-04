<#
.SYNOPSIS
    Kill orphaned Unreal MCP server processes and report what is still healthy.

.DESCRIPTION
    Every `npx -y ultimate-unreal-engine-mcp` launch spawns a three-process chain
    (npx -> cmd -> node). When an agent client exits without reaping its child,
    the chain survives and keeps reconnecting to the editor bridge, which shows
    up as a reconnect storm in the editor log and as duplicate MCP servers in
    the client. This removes chains whose owning client is gone.

    Run with -WhatIf first if you want to see the list without killing anything.

.EXAMPLE
    pwsh -File scripts/windows/ue-mcp-cleanup.ps1
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$IncludeLive
)

$ErrorActionPreference = 'Stop'

function Get-McpChains {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match 'ultimate-unreal-engine-mcp' }
}

$chains = Get-McpChains
Write-Host "Found $($chains.Count) processes in ultimate-unreal-engine-mcp chains."

$liveClients = @()
Get-Process claude, opencode, codex -ErrorAction SilentlyContinue |
    ForEach-Object { $liveClients += $_.Id }

foreach ($proc in $chains) {
    # Walk up to the owning client; an orphan is one whose ancestry no longer
    # reaches a running agent client.
    $ancestor = $proc
    $owned = $false
    for ($depth = 0; $depth -lt 6 -and $ancestor; $depth++) {
        if ($liveClients -contains $ancestor.ParentProcessId) { $owned = $true; break }
        $ancestor = Get-CimInstance Win32_Process -Filter "ProcessId=$($ancestor.ParentProcessId)" -ErrorAction SilentlyContinue
    }

    if ($owned -and -not $IncludeLive) {
        Write-Host "  keep  pid=$($proc.ProcessId) $($proc.Name) (owned by a running client)"
        continue
    }

    if ($PSCmdlet.ShouldProcess("pid=$($proc.ProcessId) $($proc.Name)", 'Stop-Process')) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
            Write-Host "  killed pid=$($proc.ProcessId) $($proc.Name)"
        } catch {
            Write-Host "  skip   pid=$($proc.ProcessId) (already gone)"
        }
    }
}

Write-Host ''
Write-Host 'Remaining editor-side surfaces:'
$editor = Get-Process UnrealEditor -ErrorAction SilentlyContinue
if (-not $editor) {
    Write-Host '  UnrealEditor is not running.'
} else {
    foreach ($e in $editor) {
        $ports = Get-NetTCPConnection -State Listen -OwningProcess $e.Id -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty LocalPort | Sort-Object -Unique
        Write-Host "  UnrealEditor pid=$($e.Id) listening on: $($ports -join ', ')"
    }
}
