[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$shares = [ordered]@{
  model = 'https://1drv.ms/u/c/b8eab7b1820a6fa4/EWJOsgGxPMZDkl8xJ_QZB30BpcjNoMVGK9mnUPq5n9-lyw?e=4GvEs9'
  checkpoint = 'https://1drv.ms/u/c/b8eab7b1820a6fa4/ETfT_C9Oz1FFlykJdtj3h6MBR1KvQb5BYwesxFykH-7BZA?e=7ti1yj'
}
$out = "C:\AI\LowVRAM3D-benchmarks\beggars-scene\onedrive-probe-$env:GITHUB_RUN_ID"
New-Item -ItemType Directory -Path $out -Force | Out-Null
$curl = (Get-Command curl.exe -ErrorAction Stop).Source
$ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151.0.0.0 Safari/537.36'
$rows = @()

function Invoke-Probe {
  param([string]$Name,[string]$Url,[string]$Mode,[string[]]$Extra)
  $safe = "$Name-$Mode" -replace '[^A-Za-z0-9_.-]','_'
  $headers = Join-Path $out "$safe.headers.txt"
  $body = Join-Path $out "$safe.body.bin"
  Remove-Item $headers,$body -Force -ErrorAction SilentlyContinue
  $args = @('-sS','--connect-timeout','30','--max-time','90','-A',$ua,'-e','https://github.com/LizhenWangT/FaceVerse_v4/','-D',$headers,'-o',$body,'-w','%{http_code}|%{url_effective}|%{content_type}|%{size_download}') + $Extra + @($Url)
  $summary = (& $curl @args 2>&1 | Out-String).Trim()
  $exit = $LASTEXITCODE
  $length = if (Test-Path $body) { (Get-Item $body).Length } else { 0 }
  $hash = if ($length -gt 0) { (Get-FileHash -Algorithm SHA256 -LiteralPath $body).Hash.ToLowerInvariant() } else { $null }
  $headerText = if (Test-Path $headers) { Get-Content -LiteralPath $headers -Raw } else { '' }
  $locations = @([regex]::Matches($headerText,'(?im)^location:\s*(.+)$') | ForEach-Object { $_.Groups[1].Value.Trim() })
  $locationHosts = @($locations | ForEach-Object { try { ([Uri]$_).Host } catch { 'invalid' } } | Select-Object -Unique)
  $rows += [ordered]@{
    name=$Name; mode=$Mode; curl_exit=$exit; summary=$summary; bytes=$length; sha256=$hash;
    location_count=$locations.Count; location_hosts=$locationHosts;
    begins_with_html = if ($length -gt 0) { ([Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($body)[0..([Math]::Min($length-1,31))]) -match '<!DOCTYPE|<html') } else { $false }
  }
}

foreach ($entry in $shares.GetEnumerator()) {
  Invoke-Probe $entry.Key $entry.Value 'head-no-follow' @('-I')
  Invoke-Probe $entry.Key $entry.Value 'get-follow' @('-L','--fail-with-body')
  Invoke-Probe $entry.Key ($entry.Value + '&download=1') 'download-follow' @('-L','--fail-with-body')
  Invoke-Probe $entry.Key ($entry.Value -replace '\?e=.*$','?download=1') 'download-replace-query' @('-L','--fail-with-body')
}

$report = [ordered]@{
  classification='PROBE_COMPLETE'; workflow_run_id=$env:GITHUB_RUN_ID; source='official FaceVerse v4 README OneDrive shares'; probes=$rows
}
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $out 'probe.json') -Encoding utf8
"FACEVERSE_ONEDRIVE_PROBE_OUTPUT=$out" | Add-Content -LiteralPath $env:GITHUB_ENV
Write-Host "FACEVERSE_ONEDRIVE_PROBE=COMPLETE"
Write-Host "REPORT=$(Join-Path $out 'probe.json')"
