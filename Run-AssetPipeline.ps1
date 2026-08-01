<#
.SYNOPSIS
    Pipeline V2 entrypoint. One command, one image, one approval at the end.

.EXAMPLE
    .\Run-AssetPipeline.ps1 -Image "C:\path\to\character.png" -Profile Auto
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Image,
    [string]$Profile = "Auto",
    [string]$AssetId = "",
    [string]$OutputRoot = "",
    [string]$FromStage = "INGEST",
    [string]$ToStage = "TEXTURE_QA",
    [string]$ExistingMaster = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$inner = Join-Path $PSScriptRoot "scripts\run-asset-pipeline.ps1"
if (-not (Test-Path -LiteralPath $inner)) { throw "Missing $inner" }

& $inner -Image $Image -Profile $Profile -AssetId $AssetId -OutputRoot $OutputRoot `
         -FromStage $FromStage -ToStage $ToStage -ExistingMaster $ExistingMaster
exit $LASTEXITCODE
