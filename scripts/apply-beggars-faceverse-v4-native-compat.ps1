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
    $exitCode = -1
    try {
        if ($WorkingDirectory) {
            Set-Location -LiteralPath $WorkingDirectory
        }
        $ErrorActionPreference = 'Continue'
        & $FilePath @ArgumentList
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
        Set-Location -LiteralPath $previousLocation
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

$importOld = '& $VenvPython -c "import sys; sys.path.insert(0,r''$SourceRoot''); import Sim3DR,faceversev4; print(''FACEVERSE_SOURCE_IMPORT=PROVEN'')"'
$importNew = $importOld + ' 2>$null'
$text = $text.Replace($importOld, $importNew)

$extractOld = '& $VenvPython -c $extractCode'
$extractNew = @'
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $VenvPython -c $extractCode
        $extractExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
'@.TrimEnd()
if ($text.Contains($extractOld)) {
    $text = $text.Replace($extractOld, $extractNew)
    $text = $text.Replace(
        'if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $keyframePath)) {',
        'if ($extractExitCode -ne 0 -or -not (Test-Path -LiteralPath $keyframePath)) {'
    )
}

[System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $target), $text, [System.Text.UTF8Encoding]::new($false))

$tokens = $null
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path -LiteralPath $target), [ref]$tokens, [ref]$errors)
if (@($errors).Count -gt 0) {
    throw "Patched FaceVerse preflight failed PowerShell parsing: $($errors[0].Message)"
}
Write-Host 'FACEVERSE_NATIVE_PROCESS_COMPAT=PROVEN'
