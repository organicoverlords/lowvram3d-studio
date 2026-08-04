[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'scripts\run-beggars-faceverse-v4-preflight.ps1'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse preflight script is missing: $target"
}

$text = Get-Content -LiteralPath $target -Raw

$newGetUv = @'
function Get-Uv {
    $command = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($command -and (Test-Path -LiteralPath $command.Source)) {
        return $command.Source
    }
    foreach ($candidate in @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:LOCALAPPDATA\Programs\uv\uv.exe"
    )) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    $wingetRoot = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
    if (Test-Path -LiteralPath $wingetRoot) {
        $match = Get-ChildItem -LiteralPath $wingetRoot -Filter 'uv.exe' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($match) { return $match.FullName }
    }
    return $null
}
'@

$newResolve = @'
function Resolve-Python39 {
    $uv = Get-Uv
    if ($uv) {
        Invoke-Native -FilePath $uv -ArgumentList @('python','install','3.9.21') -FailureMessage 'uv could not install Python 3.9.21'
        $lines = @(& $uv python find 3.9 2>$null)
        $resolved = if ($lines.Count -gt 0) { ([string]$lines[-1]).Trim() } else { '' }
        if ($LASTEXITCODE -eq 0 -and $resolved -and (Test-Path -LiteralPath $resolved)) {
            Write-Host "FACEVERSE_PYTHON39_SOURCE=UV PATH=$resolved"
            return $resolved
        }
    }

    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $probeFile = Join-Path $env:TEMP "faceverse-python39-$PID.txt"
        Remove-Item -LiteralPath $probeFile -Force -ErrorAction SilentlyContinue
        $process = Start-Process -FilePath $launcher.Source -ArgumentList @('-3.9','-c','import sys; print(sys.executable)') -RedirectStandardOutput $probeFile -RedirectStandardError (Join-Path $env:TEMP "faceverse-python39-$PID.err") -Wait -PassThru -WindowStyle Hidden
        if ($process.ExitCode -eq 0 -and (Test-Path -LiteralPath $probeFile)) {
            $resolved = ([string](Get-Content -LiteralPath $probeFile | Select-Object -Last 1)).Trim()
            if ($resolved -and (Test-Path -LiteralPath $resolved)) {
                Write-Host "FACEVERSE_PYTHON39_SOURCE=PY_LAUNCHER PATH=$resolved"
                return $resolved
            }
        }
    }

    throw 'Python 3.9 is unavailable: neither uv provisioning nor the Python launcher produced a valid executable.'
}
'@

$getUvPattern = '(?s)function Get-Uv \{.*?\r?\n\}\r?\n\r?\nfunction Resolve-Python39'
if (-not [regex]::IsMatch($text, $getUvPattern)) {
    throw 'Could not locate the FaceVerse Get-Uv function for compatibility replacement.'
}
$text = [regex]::Replace($text, $getUvPattern, $newGetUv.TrimEnd() + "`r`n`r`nfunction Resolve-Python39", 1)

$resolvePattern = '(?s)function Resolve-Python39 \{.*?\r?\n\}\r?\n\r?\nfunction Get-OneDriveApiUrl'
if (-not [regex]::IsMatch($text, $resolvePattern)) {
    throw 'Could not locate the FaceVerse Resolve-Python39 function for compatibility replacement.'
}
$text = [regex]::Replace($text, $resolvePattern, $newResolve.TrimEnd() + "`r`n`r`nfunction Get-OneDriveApiUrl", 1)

[System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $target), $text, [System.Text.UTF8Encoding]::new($false))

$tokens = $null
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path -LiteralPath $target), [ref]$tokens, [ref]$errors)
if (@($errors).Count -gt 0) {
    throw "Patched FaceVerse preflight failed PowerShell parsing: $($errors[0].Message)"
}
Write-Host 'FACEVERSE_PYTHON39_DISCOVERY_COMPAT=PROVEN'
