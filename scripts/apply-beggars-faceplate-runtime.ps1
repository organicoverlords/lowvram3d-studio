[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expectedRepository = 'organicoverlords/lowvram3d-studio'
$expectedBranch = 'agent/blender-beggars-scene-20260804'
$runPath = 'scripts\run-beggars-scene-v2.ps1'
$builderPath = 'blender\build_beggars_meme_scene.py'
$toolPath = 'tools\beggars_scene\build_face_sprite_sheet.py'

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
foreach ($path in @($runPath, $builderPath, $toolPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required face-plate source is missing: $path"
    }
}
if (@(& git status --short).Count -gt 0) {
    throw 'Checkout is visibly dirty before face-plate runtime repair.'
}

$runSource = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $runPath))
$builder = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $builderPath))

$spriteBuild = @'
    $spriteSheet = Join-Path $PrivateRoot 'face_sprite_sheet.png'
    $spriteReport = Join-Path $ArtifactRoot 'face_sprite_sheet_report.json'
    Invoke-Native -FilePath $venvPython -ArgumentList @(
        'tools\beggars_scene\build_face_sprite_sheet.py',
        '--frames-dir',$frames,
        '--sequence',$sequence,
        '--output-image',$spriteSheet,
        '--output-report',$spriteReport,
        '--cell-size','512',
        '--columns','8'
    ) -FailureMessage 'Derived face-only sprite sheet generation failed'
    if (-not (Test-Path -LiteralPath $spriteSheet) -or (Get-Item -LiteralPath $spriteSheet).Length -lt 250000) {
        throw 'Derived face-only sprite sheet is missing or implausibly small'
    }
    Write-Host 'DERIVED_FACE_ONLY_SPRITE_SHEET=PROVEN'
    Write-Host 'RAW_REFERENCE_MEDIA_PACKAGED=FALSE'

'@
$reconstructionMarker = "    Write-Host 'TRACKED_FACE_RECONSTRUCTION=PROVEN'`r`n`r`n"
if (-not $runSource.Contains($reconstructionMarker)) {
    $reconstructionMarker = "    Write-Host 'TRACKED_FACE_RECONSTRUCTION=PROVEN'`n`n"
}
if ($runSource.Contains($reconstructionMarker) -and -not $runSource.Contains('DERIVED_FACE_ONLY_SPRITE_SHEET=PROVEN')) {
    $runSource = $runSource.Replace($reconstructionMarker, ($reconstructionMarker + $spriteBuild))
    Write-Host 'FACEPLATE_SPRITE_BUILD_CALL=APPLIED'
}
elseif ($runSource.Contains('DERIVED_FACE_ONLY_SPRITE_SHEET=PROVEN')) {
    Write-Host 'FACEPLATE_SPRITE_BUILD_CALL=ALREADY_APPLIED'
}
else {
    throw 'Could not locate the reconstruction success boundary for sprite generation.'
}

$oldEvidencePublication = @'
    $evidence = Join-Path $PWD 'evidence\latest-beggars-scene'
    if (Test-Path -LiteralPath $evidence) {
        Remove-Item -LiteralPath $evidence -Recurse -Force
    }
    New-Item -ItemType Directory -Path $evidence -Force | Out-Null
    foreach ($name in @('scene_receipt.json','artifact_manifest.json','reference_reconstruction_report.json','worker_receipt.json')) {
        Copy-Item -LiteralPath (Join-Path $ArtifactRoot $name) -Destination $evidence
    }
    git config user.name 'github-actions[bot]'
    git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
    git add -- evidence/latest-beggars-scene
    if (@(& git status --porcelain -- evidence/latest-beggars-scene).Count -gt 0) {
        Invoke-Native -FilePath 'git.exe' -ArgumentList @('commit','-m',"evidence(scene): record beggars meme v2 run $env:GITHUB_RUN_ID") -FailureMessage 'Could not commit compact scene receipts'
        Invoke-Native -FilePath 'git.exe' -ArgumentList @('push','origin',"HEAD:$ExpectedBranch") -FailureMessage 'Could not push compact scene receipts'
    }

'@
$newEvidencePublication = @'
    Write-Host 'COMPACT_EVIDENCE_PUBLICATION=DEFERRED_TO_RACE_SAFE_FINALIZER'

'@
if ($runSource.Contains($oldEvidencePublication)) {
    $runSource = $runSource.Replace($oldEvidencePublication, $newEvidencePublication)
    Write-Host 'DUPLICATE_EVIDENCE_PUSH_REMOVED=APPLIED'
}
elseif ($runSource.Contains('COMPACT_EVIDENCE_PUBLICATION=DEFERRED_TO_RACE_SAFE_FINALIZER')) {
    Write-Host 'DUPLICATE_EVIDENCE_PUSH_REMOVED=ALREADY_APPLIED'
}
else {
    throw 'Could not locate the obsolete in-build evidence publication block.'
}

$facePlateHelpers = @'
def face_sprite_material(
    sprite_path: Path,
    columns: int,
    rows: int,
    selected_indices: list[int],
    target_frames: list[int],
) -> bpy.types.Material:
    if not sprite_path.is_file():
        raise RuntimeError(f"Derived face sprite sheet is missing: {sprite_path}")
    material = bpy.data.materials.new("MAT_Antinous_DerivedFaceSprite")
    material.use_nodes = True
    try:
        material.surface_render_method = "DITHERED"
    except (AttributeError, TypeError, ValueError):
        try:
            material.blend_method = "BLEND"
        except AttributeError:
            pass
    material.diffuse_color = (1.0, 1.0, 1.0, 0.0)

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texture_coordinates = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    texture = nodes.new("ShaderNodeTexImage")

    image = bpy.data.images.load(str(sprite_path), check_existing=False)
    image.colorspace_settings.name = "sRGB"
    image.alpha_mode = "STRAIGHT"
    image.pack()
    texture.image = image
    texture.interpolation = "Linear"
    texture.extension = "CLIP"

    mapping.inputs["Scale"].default_value = (1.0 / columns, 1.0 / rows, 1.0)
    links.new(texture_coordinates.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(texture.outputs["Color"], bsdf.inputs["Emission Color"])
    links.new(texture.outputs["Alpha"], bsdf.inputs["Alpha"])
    set_input(bsdf, "Roughness", 0.48)
    set_input(bsdf, "Specular IOR Level", 0.24)
    set_input(bsdf, "Emission Strength", 0.34)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    location = mapping.inputs["Location"]
    for sequence_index, target in zip(selected_indices, target_frames):
        column = int(sequence_index % columns)
        row = int(sequence_index // columns)
        location.default_value = (
            float(column) / float(columns),
            1.0 - float(row + 1) / float(rows),
            0.0,
        )
        location.keyframe_insert(data_path="default_value", frame=target)
    if material.node_tree.animation_data and material.node_tree.animation_data.action:
        for fcurve in material.node_tree.animation_data.action.fcurves:
            for keyframe_point in fcurve.keyframe_points:
                keyframe_point.interpolation = "CONSTANT"
    return material


def _face_plate_grid(
    frame_vertices: np.ndarray,
    center_x: float,
    center_y: float,
    pixel_scale: float,
    base_depth: float,
    grid_size: int,
) -> np.ndarray:
    points = transform_vertices(frame_vertices, center_x, center_y, pixel_scale, base_depth)
    x_min = float(np.percentile(points[:, 0], 0.25))
    x_max = float(np.percentile(points[:, 0], 99.75))
    z_min = float(np.percentile(points[:, 2], 0.25))
    z_max = float(np.percentile(points[:, 2], 99.75))
    center_world_x = (x_min + x_max) * 0.5
    center_world_z = (z_min + z_max) * 0.5
    side = max(x_max - x_min, z_max - z_min, 0.2) * 1.12
    y_front = float(np.percentile(points[:, 1], 0.5)) - 0.042

    vertices = []
    for row in range(grid_size):
        v = float(row) / float(grid_size - 1)
        for column in range(grid_size):
            u = float(column) / float(grid_size - 1)
            x = center_world_x + (u - 0.5) * side
            z = center_world_z + (v - 0.5) * side
            radial = (u - 0.5) ** 2 + (v - 0.5) ** 2
            y = y_front + radial * 0.11
            vertices.append((x, y, z))
    return np.asarray(vertices, dtype=np.float32)


def create_face_sprite_plate(
    vertices_sequence: np.ndarray,
    boxes: np.ndarray,
    sprite_path: Path,
    sprite_metadata: dict,
    scene_frame_count: int,
) -> tuple[bpy.types.Object, list[int]]:
    frame_count = int(sprite_metadata["frame_count"])
    columns = int(sprite_metadata["columns"])
    rows = int(sprite_metadata["rows"])
    if frame_count != len(vertices_sequence):
        raise RuntimeError(
            f"Sprite/track frame mismatch: {frame_count} sprite frames versus {len(vertices_sequence)} meshes"
        )

    keyframe_index = min(max(int(len(vertices_sequence) * 0.62), 0), len(vertices_sequence) - 1)
    key_box = boxes[keyframe_index]
    center_x = float((key_box[0] + key_box[2]) * 0.5)
    center_y = float((key_box[1] + key_box[3]) * 0.5)
    face_height_pixels = max(float(key_box[3] - key_box[1]), 1.0)
    pixel_scale = 2.00 / face_height_pixels
    base_depth = float(np.median(vertices_sequence[keyframe_index, 2]))

    maximum_shape_keys = 42
    if len(vertices_sequence) <= maximum_shape_keys:
        selected_indices = list(range(len(vertices_sequence)))
    else:
        selected_indices = sorted(
            set(int(round(value)) for value in np.linspace(0, len(vertices_sequence) - 1, maximum_shape_keys))
        )
    target_frames = [
        1 + int(round(index / max(len(vertices_sequence) - 1, 1) * (scene_frame_count - 1)))
        for index in selected_indices
    ]

    grid_size = 14
    transformed_grids = np.stack(
        [
            _face_plate_grid(
                vertices_sequence[index],
                center_x,
                center_y,
                pixel_scale,
                base_depth,
                grid_size,
            )
            for index in selected_indices
        ],
        axis=0,
    )
    faces = []
    for row in range(grid_size - 1):
        for column in range(grid_size - 1):
            lower_left = row * grid_size + column
            lower_right = lower_left + 1
            upper_left = (row + 1) * grid_size + column
            upper_right = upper_left + 1
            faces.append((lower_left, lower_right, upper_right))
            faces.append((lower_left, upper_right, upper_left))

    mesh = bpy.data.meshes.new("MESH_Antinous_DerivedFacePlate")
    mesh.from_pydata(transformed_grids[0].tolist(), [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    vertex_uv = []
    for row in range(grid_size):
        v = float(row) / float(grid_size - 1)
        for column in range(grid_size):
            u = float(column) / float(grid_size - 1)
            vertex_uv.append((u, v))
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = vertex_uv[vertex_index]

    plate = bpy.data.objects.new("CHAR_Antinous_FacePlate", mesh)
    bpy.context.collection.objects.link(plate)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.materials.append(
        face_sprite_material(sprite_path, columns, rows, selected_indices, target_frames)
    )
    plate["appearance_route"] = "PACKED_DERIVED_FACE_ONLY_RGBA_SPRITE"
    plate["raw_reference_media_packaged"] = False
    plate["source_clip_packaged"] = False

    plate.shape_key_add(name="Basis")
    key_blocks = []
    for sequence_index, grid in zip(selected_indices, transformed_grids):
        key = plate.shape_key_add(name=f"Track_{sequence_index:04d}")
        key.data.foreach_set("co", grid.reshape(-1).tolist())
        key.value = 0.0
        key_blocks.append(key)
    for index, (key, target) in enumerate(zip(key_blocks, target_frames)):
        previous_target = target_frames[index - 1] if index > 0 else 1
        next_target = target_frames[index + 1] if index + 1 < len(target_frames) else scene_frame_count
        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=previous_target)
        key.value = 1.0
        key.keyframe_insert(data_path="value", frame=target)
        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=next_target)
    if plate.data.shape_keys and plate.data.shape_keys.animation_data and plate.data.shape_keys.animation_data.action:
        for fcurve in plate.data.shape_keys.animation_data.action.fcurves:
            for keyframe_point in fcurve.keyframe_points:
                keyframe_point.interpolation = "LINEAR"
    return plate, target_frames

'@
$createFaceAnchor = 'def create_face_mesh('
if (-not $builder.Contains('def create_face_sprite_plate(')) {
    if (-not $builder.Contains($createFaceAnchor)) {
        throw 'Could not locate create_face_mesh for face-plate helper insertion.'
    }
    $builder = $builder.Replace($createFaceAnchor, ($facePlateHelpers + "`n" + $createFaceAnchor))
    Write-Host 'FACEPLATE_BLENDER_HELPERS=APPLIED'
}
else {
    Write-Host 'FACEPLATE_BLENDER_HELPERS=ALREADY_APPLIED'
}

$atlasMaterial = '    mesh.materials.append(face_atlas_material(atlas_path))'
$neutralMaterial = @'
    mesh.materials.append(
        principled_material(
            "MAT_Antinous_UnderFace",
            (0.12, 0.045, 0.028, 1.0),
            0.72,
        )
    )
'@
if ($builder.Contains($atlasMaterial)) {
    $builder = $builder.Replace($atlasMaterial, $neutralMaterial)
    Write-Host 'FULL_MESH_TEXTURE_SMEAR_REMOVED=APPLIED'
}
elseif ($builder.Contains('"MAT_Antinous_UnderFace"')) {
    Write-Host 'FULL_MESH_TEXTURE_SMEAR_REMOVED=ALREADY_APPLIED'
}
else {
    throw 'Could not replace the rejected full-mesh atlas material.'
}

$spriteLoad = @'
    sprite_path = sequence_path.with_name("face_sprite_sheet.png")
    sprite_report_path = output_dir / "face_sprite_sheet_report.json"
    if not sprite_path.is_file():
        raise SystemExit(f"Derived face sprite sheet is missing: {sprite_path}")
    if not sprite_report_path.is_file():
        raise SystemExit(f"Derived face sprite report is missing: {sprite_report_path}")
    sprite_metadata = json.loads(sprite_report_path.read_text(encoding="utf-8"))
    if sprite_metadata.get("classification") != "PROVEN":
        raise SystemExit(f"Derived face sprite report is not proven: {sprite_metadata}")
    if sprite_metadata.get("raw_frames_packaged") is not False:
        raise SystemExit("Derived face sprite report does not prove raw-frame exclusion")

'@
$boxesAnchor = '    boxes = np.asarray(data["boxes"], dtype=np.float32)'
if (-not $builder.Contains('sprite_path = sequence_path.with_name("face_sprite_sheet.png")')) {
    if (-not $builder.Contains($boxesAnchor)) {
        throw 'Could not locate tracked boxes for face-sprite metadata loading.'
    }
    $builder = $builder.Replace($boxesAnchor, ($spriteLoad + $boxesAnchor))
    Write-Host 'FACEPLATE_SPRITE_LOAD=APPLIED'
}
else {
    Write-Host 'FACEPLATE_SPRITE_LOAD=ALREADY_APPLIED'
}

$characterAnchor = '    character_info = build_character(follow, colors_rgb)'
$plateCall = @'
    face_plate, face_plate_target_frames = create_face_sprite_plate(
        vertices,
        boxes,
        sprite_path,
        sprite_metadata,
        scene.frame_end,
    )
    character_info = build_character(follow, colors_rgb)
'@
if ($builder.Contains($characterAnchor) -and -not $builder.Contains('face_plate, face_plate_target_frames = create_face_sprite_plate')) {
    $builder = $builder.Replace($characterAnchor, $plateCall)
    Write-Host 'FACEPLATE_BUILD_CALL=APPLIED'
}
elseif ($builder.Contains('face_plate, face_plate_target_frames = create_face_sprite_plate')) {
    Write-Host 'FACEPLATE_BUILD_CALL=ALREADY_APPLIED'
}
else {
    throw 'Could not locate character construction boundary for face-plate creation.'
}

$blendAnchor = '    blend_path = output_dir / "beggars_photoreal_recreation.blend"'
$requiredSetup = @'
    required_objects = list(config["acceptance"]["required_objects"])
    if "CHAR_Antinous_FacePlate" not in required_objects:
        required_objects.append("CHAR_Antinous_FacePlate")

    blend_path = output_dir / "beggars_photoreal_recreation.blend"
'@
if ($builder.Contains($blendAnchor) -and -not $builder.Contains('required_objects.append("CHAR_Antinous_FacePlate")')) {
    $builder = $builder.Replace($blendAnchor, $requiredSetup)
    Write-Host 'FACEPLATE_REQUIRED_OBJECT=APPLIED'
}
elseif ($builder.Contains('required_objects.append("CHAR_Antinous_FacePlate")')) {
    Write-Host 'FACEPLATE_REQUIRED_OBJECT=ALREADY_APPLIED'
}
else {
    throw 'Could not add the face plate to save/reload validation.'
}

$builder = $builder.Replace(
    '    reload_report = validate_reload(blend_path, config["acceptance"]["required_objects"])',
    '    reload_report = validate_reload(blend_path, required_objects)'
)
$builder = $builder.Replace(
    '        "required_objects": config["acceptance"]["required_objects"],',
    '        "required_objects": required_objects,'
)
if (-not $builder.Contains('reload_report = validate_reload(blend_path, required_objects)')) {
    throw 'Face-plate save/reload validation target was not applied.'
}

$receiptOld = '        "reference_media_packaged": False,'
$receiptNew = @'
        "reference_media_packaged": False,
        "raw_reference_media_packaged": False,
        "derived_face_only_sprite_in_blend": True,
        "derived_face_sprite_policy": sprite_metadata["policy"],
        "face_plate_object": face_plate.name,
        "face_plate_animated_shape_keys": len(face_plate_target_frames),
'@
if ($builder.Contains($receiptOld) -and -not $builder.Contains('"derived_face_only_sprite_in_blend": True')) {
    $builder = $builder.Replace($receiptOld, $receiptNew)
    Write-Host 'FACEPLATE_RECEIPT=APPLIED'
}
elseif ($builder.Contains('"derived_face_only_sprite_in_blend": True')) {
    Write-Host 'FACEPLATE_RECEIPT=ALREADY_APPLIED'
}
else {
    throw 'Could not add face-plate provenance to the scene receipt.'
}

[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $runPath),
    $runSource,
    [System.Text.UTF8Encoding]::new($false)
)
[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $builderPath),
    $builder,
    [System.Text.UTF8Encoding]::new($false)
)

$controlPython = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $controlPython)) {
    throw "Control Python is missing: $controlPython"
}
& $controlPython -m py_compile $toolPath $builderPath
if ($LASTEXITCODE -ne 0) {
    throw 'Face-plate Python sources failed compile validation.'
}

& git update-index --assume-unchanged -- $runPath $builderPath
if ($LASTEXITCODE -ne 0) {
    throw 'Could not preserve bounded face-plate runtime sources as assume-unchanged.'
}
if (@(& git status --short).Count -gt 0) {
    throw 'Face-plate runtime repair left visible checkout dirt.'
}

Write-Host 'TRACKED_DERIVED_FACEPLATE_RUNTIME=PROVEN'
Write-Host 'FULL_MESH_SINGLE_VIEW_TEXTURE_PROJECTION=ABSENT'
Write-Host 'RAW_REFERENCE_MEDIA_PACKAGED=FALSE'
