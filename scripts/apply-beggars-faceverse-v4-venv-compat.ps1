[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'scripts\run-beggars-faceverse-v4-preflight.ps1'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse preflight script is missing: $target"
}

$text = Get-Content -LiteralPath $target -Raw

$newInvokeNative = @'
function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [Parameter(Mandatory)][string]$FailureMessage,
        [string]$WorkingDirectory
    )
    $previousLocation = (Get-Location).Path
    $previousPreference = $ErrorActionPreference
    $output = @()
    $exitCode = -1
    try {
        if ($WorkingDirectory) {
            Set-Location -LiteralPath $WorkingDirectory
        }
        $ErrorActionPreference = 'SilentlyContinue'
        $output = @(& $FilePath @ArgumentList 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
        Set-Location -LiteralPath $previousLocation
    }
    foreach ($line in $output) {
        Write-Host ([string]$line)
    }
    if ($exitCode -ne 0) {
        throw "$FailureMessage (exit $exitCode)"
    }
}
'@

$invokePattern = '(?s)function Invoke-Native \{.*?\r?\n\}\r?\n\r?\nfunction Get-Curl'
if (-not [regex]::IsMatch($text, $invokePattern)) {
    throw 'Could not locate Invoke-Native in the FaceVerse preflight.'
}
$text = [regex]::Replace(
    $text,
    $invokePattern,
    $newInvokeNative.TrimEnd() + "`r`n`r`nfunction Get-Curl",
    1
)

$oldVenv = @'
    $basePython = Resolve-Python39
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        New-Item -ItemType Directory -Path (Split-Path $VenvRoot) -Force | Out-Null
        Invoke-Native -FilePath $basePython -ArgumentList @('-m','venv',$VenvRoot) -FailureMessage 'Could not create FaceVerse Python 3.9 environment'
    }
'@
$newVenv = @'
    $basePython = Resolve-Python39
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        if (Test-Path -LiteralPath $VenvRoot) {
            Remove-Item -LiteralPath $VenvRoot -Recurse -Force
        }
        New-Item -ItemType Directory -Path (Split-Path $VenvRoot) -Force | Out-Null
        $uv = Get-Uv
        if (-not $uv) {
            throw 'uv.exe is unavailable for FaceVerse environment creation.'
        }
        Invoke-Native -FilePath $uv -ArgumentList @('venv','--python','3.9.21',$VenvRoot) -FailureMessage 'uv could not create the FaceVerse Python 3.9 environment'
        if (-not (Test-Path -LiteralPath $VenvPython)) {
            throw "uv reported success but the FaceVerse environment Python is missing: $VenvPython"
        }
        Write-Host "FACEVERSE_VENV_CREATION=PROVEN PYTHON=$VenvPython"
    }
'@

if ($text.Contains($oldVenv)) {
    $text = $text.Replace($oldVenv, $newVenv)
}
elseif ($text.Contains('FACEVERSE_VENV_CREATION=PROVEN')) {
    Write-Host 'FACEVERSE_UV_VENV_ROUTE=ALREADY_APPLIED'
}
else {
    throw 'Could not locate the FaceVerse environment creation block.'
}

[System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $target), $text, [System.Text.UTF8Encoding]::new($false))

$tokens = $null
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path -LiteralPath $target), [ref]$tokens, [ref]$errors)
if (@($errors).Count -gt 0) {
    throw "Patched FaceVerse preflight failed PowerShell parsing: $($errors[0].Message)"
}
Write-Host 'FACEVERSE_UV_VENV_COMPAT=PROVEN'
