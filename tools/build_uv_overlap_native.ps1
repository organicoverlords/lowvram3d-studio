param(
    [string]$Output = "$PSScriptRoot\native\uv_overlap_native.exe"
)
$ErrorActionPreference = 'Stop'
$source = Join-Path $PSScriptRoot 'uv_overlap_native.cpp'
$outputPath = [System.IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($outputPath)) | Out-Null
$cl = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio' -Recurse -Filter cl.exe |
    Where-Object { $_.FullName -match 'Hostx64\\x64\\cl\.exe$' } |
    Sort-Object FullName -Descending | Select-Object -First 1
if (-not $cl) { throw 'MSVC x64 cl.exe was not found.' }
$vcvars = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio' -Recurse -Filter vcvars64.bat |
    Select-Object -First 1 -ExpandProperty FullName
if (-not (Test-Path -LiteralPath $vcvars)) { throw "vcvars64.bat was not found: $vcvars" }
$args = "/nologo /O2 /EHsc /std:c++17 `"$source`" /Fe:`"$outputPath`""
cmd.exe /d /s /c "call `"$vcvars`" && `"$($cl.FullName)`" $args"
if ($LASTEXITCODE -ne 0) { throw "Native overlap detector build failed with exit code $LASTEXITCODE" }
Write-Output $outputPath
