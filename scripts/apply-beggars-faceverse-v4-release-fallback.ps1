[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'scripts\run-beggars-faceverse-v4-preflight.ps1'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse preflight script is missing: $target"
}

$text = Get-Content -LiteralPath $target -Raw

$anchor = @'
$LandmarkerUrl = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
'@.Trim()
$insertion = @'
$FaceVerseReleaseBase = 'https://github.com/Mrkomiljon/faceverse-onnx/releases/latest/download'
$FaceVerseModelReleaseUrl = "$FaceVerseReleaseBase/faceverse_v4_2.npy"
$FaceVerseCheckpointReleaseUrl = "$FaceVerseReleaseBase/faceverse_resnet50.pth"
'@.TrimEnd()
if ($text.Contains($anchor) -and -not $text.Contains('$FaceVerseModelReleaseUrl')) {
    $text = $text.Replace($anchor, $anchor + "`r`n" + $insertion)
}
elseif (-not $text.Contains('$FaceVerseModelReleaseUrl')) {
    throw 'Could not locate the FaceVerse landmarker URL anchor.'
}

$oldModelDownload = 'Download-File -Urls @("$FaceVerseModelShare&download=1", $apiUrl) -Destination $ModelPath -MinimumBytes 1000000 -Label ''FaceVerse v4 model'''
$newModelDownload = 'Download-File -Urls @("$FaceVerseModelShare&download=1", $apiUrl, $FaceVerseModelReleaseUrl) -Destination $ModelPath -MinimumBytes 1000000 -Label ''FaceVerse v4 model'''
if ($text.Contains($oldModelDownload)) {
    $text = $text.Replace($oldModelDownload, $newModelDownload)
}
elseif (-not $text.Contains('$FaceVerseModelReleaseUrl) -Destination $ModelPath')) {
    throw 'Could not locate the FaceVerse model download call.'
}

$oldCheckpointDownload = 'Download-File -Urls @("$FaceVerseCheckpointShare&download=1", $apiUrl) -Destination $CheckpointPath -MinimumBytes 1000000 -Label ''FaceVerse v4 ResNet50 checkpoint'''
$newCheckpointDownload = 'Download-File -Urls @("$FaceVerseCheckpointShare&download=1", $apiUrl, $FaceVerseCheckpointReleaseUrl) -Destination $CheckpointPath -MinimumBytes 1000000 -Label ''FaceVerse v4 ResNet50 checkpoint'''
if ($text.Contains($oldCheckpointDownload)) {
    $text = $text.Replace($oldCheckpointDownload, $newCheckpointDownload)
}
elseif (-not $text.Contains('$FaceVerseCheckpointReleaseUrl) -Destination $CheckpointPath')) {
    throw 'Could not locate the FaceVerse checkpoint download call.'
}

$proofAnchor = "    Write-Host 'FACEVERSE_MODEL_FILES=PROVEN'"
$compatibilityProof = @'
    $compatibilityCode = @"
import sys, torch, numpy as np
sys.path.insert(0, r'$SourceRoot')
from faceversev4 import FaceVerseRecon
model = FaceVerseRecon(r'$ModelPath', r'$CheckpointPath', torch.device('cpu'))
head_channels = [int(layer.out_channels) for layer in model.reconnet.final_layers]
assert head_channels == [156, 177, 251, 27, 3, 2, 1, 4], head_channels
assert sum(head_channels) == 621
assert model.id_dims == 156
assert model.exp_dims == 177
assert model.tex_dims == 251
assert int(np.asarray(model.fvd['meanshape']).reshape(-1, 3).shape[0]) > 10000
assert int(np.asarray(model.fvd['tri']).reshape(-1, 3).shape[0]) > 10000
print('FACEVERSE_MODEL_CHECKPOINT_PAIR=PROVEN', head_channels, sum(head_channels), model.id_dims, model.exp_dims, model.tex_dims)
"@
    Invoke-Native -FilePath $VenvPython -ArgumentList @('-c',$compatibilityCode) -FailureMessage 'FaceVerse model/checkpoint pair failed exact official-loader compatibility validation'
    Write-Host 'FACEVERSE_RELEASE_FALLBACK_VALIDATION=PROVEN'
'@.TrimEnd()
if ($text.Contains($proofAnchor) -and -not $text.Contains('FACEVERSE_MODEL_CHECKPOINT_PAIR=PROVEN')) {
    $text = $text.Replace($proofAnchor, $proofAnchor + "`r`n" + $compatibilityProof)
}
elseif (-not $text.Contains('FACEVERSE_MODEL_CHECKPOINT_PAIR=PROVEN')) {
    throw 'Could not locate the FaceVerse model-files proof anchor.'
}

[System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $target), $text, [System.Text.UTF8Encoding]::new($false))

$tokens = $null
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path -LiteralPath $target), [ref]$tokens, [ref]$errors)
if (@($errors).Count -gt 0) {
    throw "Patched FaceVerse preflight failed PowerShell parsing: $($errors[0].Message)"
}
Write-Host 'FACEVERSE_RELEASE_FALLBACK_PATCH=PROVEN'
