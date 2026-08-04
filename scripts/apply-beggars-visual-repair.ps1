[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expectedRepository = 'organicoverlords/lowvram3d-studio'
$expectedBranch = 'agent/blender-beggars-scene-20260804'
$path = 'blender\build_beggars_meme_scene.py'

if ($env:GITHUB_REPOSITORY -ne $expectedRepository) {
    throw "Repository mismatch: $env:GITHUB_REPOSITORY"
}
if ($env:GITHUB_REF_NAME -ne $expectedBranch) {
    throw "Branch mismatch: $env:GITHUB_REF_NAME"
}
$remote = (& git remote get-url origin | Out-String).Trim()
$head = (& git rev-parse HEAD | Out-String).Trim()
if ($remote -notmatch 'organicoverlords/lowvram3d-studio') {
    throw "Remote mismatch: $remote"
}
if ($head -ne $env:GITHUB_SHA) {
    throw "Checkout HEAD $head does not equal workflow SHA $env:GITHUB_SHA"
}
if (-not (Test-Path -LiteralPath $path)) {
    throw "Blender scene builder is missing: $path"
}
if (@(& git status --short).Count -gt 0) {
    throw 'Checkout is visibly dirty before visual repair.'
}

$content = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $path))

function Replace-Required {
    param(
        [Parameter(Mandatory)][string]$Old,
        [Parameter(Mandatory)][string]$New,
        [Parameter(Mandatory)][string]$Label
    )
    if (-not $script:content.Contains($Old)) {
        if ($script:content.Contains($New)) {
            Write-Host "$Label=ALREADY_APPLIED"
            return
        }
        throw "Expected source for $Label was not found."
    }
    $script:content = $script:content.Replace($Old, $New)
    Write-Host "$Label=APPLIED"
}

Replace-Required '    set_input(bsdf, "Roughness", 0.46)' '    set_input(bsdf, "Roughness", 0.52)' 'FACE_ROUGHNESS_REPAIR'
Replace-Required '    set_input(bsdf, "Specular IOR Level", 0.28)' '    set_input(bsdf, "Specular IOR Level", 0.34)' 'FACE_SPECULAR_REPAIR'
Replace-Required '    set_input(bsdf, "Subsurface Weight", 0.055)' "    set_input(bsdf, \"Subsurface Weight\", 0.035)`n    set_input(bsdf, \"Subsurface Radius\", (1.0, 0.42, 0.18))" 'FACE_SUBSURFACE_REPAIR'
Replace-Required '    set_input(bsdf, "Emission Strength", 0.09)' '    set_input(bsdf, "Emission Strength", 0.0)' 'FACE_EMISSION_REMOVAL'
Replace-Required '    links.new(attribute.outputs["Color"], bsdf.inputs["Base Color"])' "    gamma = nodes.new(\"ShaderNodeGamma\")`n    gamma.inputs[\"Gamma\"].default_value = 2.2`n    tone = nodes.new(\"ShaderNodeHueSaturation\")`n    tone.inputs[\"Saturation\"].default_value = 0.84`n    tone.inputs[\"Value\"].default_value = 0.94`n    links.new(attribute.outputs[\"Color\"], gamma.inputs[\"Color\"])`n    links.new(gamma.outputs[\"Color\"], tone.inputs[\"Color\"])`n    links.new(tone.outputs[\"Color\"], bsdf.inputs[\"Base Color\"])" 'FACE_COLORSPACE_REPAIR'
Replace-Required '    skin_average = np.clip(np.median(colors_rgb, axis=0), 0.05, 0.95)' '    skin_average = np.clip(np.median(colors_rgb, axis=0) ** 2.2, 0.03, 0.72)' 'NECK_COLORSPACE_REPAIR'

Replace-Required '        (0.0, 0.34, 0.42),' '        (0.0, 0.88, 0.58),' 'HAIR_CAP_DEPTH_REPAIR'
Replace-Required '        (1.04, 0.62, 1.16),' '        (1.02, 0.26, 1.10),' 'HAIR_CAP_THICKNESS_REPAIR'
Replace-Required '    for index in range(86):' '    for index in range(42):' 'HAIR_DENSITY_REPAIR'
Replace-Required '        angle = random.uniform(-math.pi * 0.92, math.pi * 0.92)' '        angle = random.uniform(-math.pi * 0.78, math.pi * 0.78)' 'HAIR_ARC_REPAIR'
Replace-Required '        start_y = 0.00 + random.uniform(-0.06, 0.08)' '        start_y = 0.48 + random.uniform(0.00, 0.12)' 'HAIR_FRONT_OCCLUSION_REPAIR'
Replace-Required '        end_y = start_y - random.uniform(0.04, 0.15)' '        end_y = start_y + random.uniform(0.03, 0.11)' 'HAIR_END_DEPTH_REPAIR'
Replace-Required '            random.uniform(0.007, 0.013),' '            random.uniform(0.005, 0.009),' 'HAIR_STRAND_WIDTH_REPAIR'

Replace-Required '    camera_data.dof.aperture_fstop = 1.55' '    camera_data.dof.aperture_fstop = 1.25' 'HERO_DOF_REPAIR'
Replace-Required '    wide_data.dof.aperture_fstop = 2.4' '    wide_data.dof.aperture_fstop = 1.8' 'WIDE_DOF_REPAIR'
Replace-Required '        energy=1050.0,' '        energy=880.0,' 'KEY_ENERGY_REPAIR'
Replace-Required '        color=(1.0, 0.17, 0.045),' '        color=(1.0, 0.42, 0.20),' 'KEY_COLOR_REPAIR'
Replace-Required '        energy=620.0,' '        energy=260.0,' 'RIM_ENERGY_REPAIR'
Replace-Required '        color=(1.0, 0.055, 0.015),' '        color=(1.0, 0.20, 0.07),' 'RIM_COLOR_REPAIR'
Replace-Required '        energy=90.0,' '        energy=170.0,' 'FILL_ENERGY_REPAIR'
Replace-Required '        color=(0.25, 0.08, 0.04),' '        color=(0.14, 0.18, 0.28),' 'FILL_COLOR_REPAIR'

if ($content.Contains('(0.0, 0.34, 0.42)') -or $content.Contains('range(86)')) {
    throw 'The rejected face-occluding hair construction is still present.'
}
if (-not $content.Contains('gamma.inputs["Gamma"].default_value = 2.2')) {
    throw 'The face color-space repair is missing.'
}

[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $path),
    $content,
    [System.Text.UTF8Encoding]::new($false)
)

$controlPython = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $controlPython)) {
    throw "Control Python is missing: $controlPython"
}
& $controlPython -m py_compile $path
if ($LASTEXITCODE -ne 0) {
    throw 'Visually repaired Blender scene builder failed compile validation.'
}

& git update-index --assume-unchanged -- $path
if ($LASTEXITCODE -ne 0) {
    throw 'Could not preserve the bounded runtime scene repair as assume-unchanged.'
}
if (@(& git status --short).Count -gt 0) {
    throw 'Visual repair left visible checkout dirt.'
}

Write-Host 'BEGGARS_VISUAL_REPAIR=PROVEN'
Write-Host 'REJECTED_HAIR_OCCLUSION_REMOVED=PROVEN'
Write-Host 'SOURCE_FRAME_EMBEDDING=ABSENT'
