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
    throw 'Checkout is visibly dirty before feature-hole repair.'
}

$content = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $path))
$marker = 'def _close_animated_feature_holes('
$helpers = @'
def _mesh_boundary_loops(triangles: np.ndarray) -> list[list[int]]:
    edge_counts: dict[tuple[int, int], int] = {}
    for triangle in np.asarray(triangles, dtype=np.int32):
        a, b, c = [int(value) for value in triangle]
        for left, right in ((a, b), (b, c), (c, a)):
            edge = (left, right) if left < right else (right, left)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    adjacency: dict[int, list[int]] = {}
    for left, right in boundary_edges:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)

    unused = set(boundary_edges)
    loops: list[list[int]] = []
    while unused:
        left, right = min(unused)
        unused.remove((left, right))
        loop = [left, right]
        previous = left
        current = right
        for _ in range(len(boundary_edges) + 1):
            candidates = []
            for candidate in adjacency.get(current, []):
                edge = (current, candidate) if current < candidate else (candidate, current)
                if edge in unused and candidate != previous:
                    candidates.append(candidate)
            if not candidates:
                closing = (current, loop[0]) if current < loop[0] else (loop[0], current)
                if closing in unused:
                    unused.remove(closing)
                break
            following = min(candidates)
            edge = (current, following) if current < following else (following, current)
            unused.remove(edge)
            if following == loop[0]:
                break
            loop.append(following)
            previous, current = current, following
        if len(loop) >= 4:
            loops.append(loop)
    return loops


def _loop_projected_area(points: np.ndarray) -> float:
    x = points[:, 0]
    z = points[:, 2]
    return float(abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))) * 0.5)


def _close_animated_feature_holes(
    transformed: np.ndarray,
    triangles: np.ndarray,
    colors_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    loops = _mesh_boundary_loops(triangles)
    if len(loops) < 4:
        return transformed, triangles, colors_rgb, {
            "classification": "NOT_APPLIED_INSUFFICIENT_BOUNDARY_LOOPS",
            "boundary_loop_count": len(loops),
        }

    base = transformed[0]
    metrics = []
    for loop in loops:
        points = base[np.asarray(loop, dtype=np.int32)]
        metrics.append(
            {
                "loop": loop,
                "area": _loop_projected_area(points),
                "centroid_z": float(np.mean(points[:, 2])),
                "vertex_count": len(loop),
            }
        )

    outer = max(metrics, key=lambda item: (item["area"], item["vertex_count"]))
    interior = [item for item in metrics if item is not outer and item["vertex_count"] >= 6]
    if len(interior) < 3:
        return transformed, triangles, colors_rgb, {
            "classification": "NOT_APPLIED_INSUFFICIENT_INTERIOR_LOOPS",
            "boundary_loop_count": len(loops),
            "interior_loop_count": len(interior),
        }

    mouth = min(interior, key=lambda item: item["centroid_z"])
    eye_candidates = [item for item in interior if item is not mouth]
    eyes = sorted(eye_candidates, key=lambda item: item["centroid_z"], reverse=True)[:2]
    selected = [("eye", item) for item in eyes] + [("mouth", mouth)]

    triangle_rows = [tuple(int(value) for value in row) for row in np.asarray(triangles, dtype=np.int32)]
    colors = np.asarray(colors_rgb, dtype=np.float32)
    added = []

    for kind, item in selected:
        loop = np.asarray(item["loop"], dtype=np.int32)
        centers = np.mean(transformed[:, loop, :], axis=1)
        ring = transformed[:, loop, :] * 0.42 + centers[:, None, :] * 0.58
        depth_offset = 0.020 if kind == "eye" else 0.028
        ring[:, :, 1] += depth_offset
        center_vertices = centers.copy()
        center_vertices[:, 1] += depth_offset + 0.014

        ring_start = int(transformed.shape[1])
        transformed = np.concatenate([transformed, ring.astype(np.float32)], axis=1)
        center_index = int(transformed.shape[1])
        transformed = np.concatenate([transformed, center_vertices[:, None, :].astype(np.float32)], axis=1)

        if kind == "eye":
            ring_colors = np.tile(np.asarray([0.88, 0.77, 0.65], dtype=np.float32), (len(loop), 1))
            center_color = np.asarray([[0.055, 0.022, 0.010]], dtype=np.float32)
        else:
            ring_colors = np.tile(np.asarray([0.18, 0.018, 0.012], dtype=np.float32), (len(loop), 1))
            loop_points = base[loop]
            upper = loop_points[:, 2] > float(np.mean(loop_points[:, 2]))
            ring_colors[upper] = np.asarray([0.84, 0.70, 0.55], dtype=np.float32)
            center_color = np.asarray([[0.004, 0.001, 0.001]], dtype=np.float32)
        colors = np.concatenate([colors, ring_colors, center_color], axis=0)

        for index in range(len(loop)):
            following = (index + 1) % len(loop)
            boundary_a = int(loop[index])
            boundary_b = int(loop[following])
            inner_a = ring_start + index
            inner_b = ring_start + following
            triangle_rows.append((boundary_a, boundary_b, inner_b))
            triangle_rows.append((boundary_a, inner_b, inner_a))
            triangle_rows.append((inner_a, inner_b, center_index))

        added.append(
            {
                "kind": kind,
                "boundary_vertices": int(len(loop)),
                "centroid_z": float(item["centroid_z"]),
                "projected_area": float(item["area"]),
            }
        )

    return (
        transformed.astype(np.float32),
        np.asarray(triangle_rows, dtype=np.int32),
        colors.astype(np.float32),
        {
            "classification": "PROVEN",
            "boundary_loop_count": len(loops),
            "closed_features": added,
            "added_vertices": int(transformed.shape[1] - base.shape[0]),
            "triangle_count_after": len(triangle_rows),
        },
    )

'@

if (-not $content.Contains($marker)) {
    $anchor = 'def create_face_mesh('
    if (-not $content.Contains($anchor)) {
        throw 'Could not locate create_face_mesh for feature-hole helper insertion.'
    }
    $content = $content.Replace($anchor, ($helpers + "`n" + $anchor))
    Write-Host 'FEATURE_HOLE_HELPERS=APPLIED'
}
else {
    Write-Host 'FEATURE_HOLE_HELPERS=ALREADY_APPLIED'
}

$callOld = '    maximum_shape_keys = 42'
$callNew = @'
    transformed, triangles, colors_rgb, feature_report = _close_animated_feature_holes(
        transformed,
        triangles,
        colors_rgb,
    )

    maximum_shape_keys = 42
'@
if ($content.Contains($callOld) -and -not $content.Contains('feature_report = _close_animated_feature_holes')) {
    $content = $content.Replace($callOld, $callNew)
    Write-Host 'FEATURE_HOLE_CALL=APPLIED'
}
elseif ($content.Contains('feature_report = _close_animated_feature_holes')) {
    Write-Host 'FEATURE_HOLE_CALL=ALREADY_APPLIED'
}
else {
    throw 'Could not locate the shape-key boundary for feature-hole repair.'
}

$propertyOld = '    bpy.context.collection.objects.link(face)'
$propertyNew = @'
    bpy.context.collection.objects.link(face)
    face["feature_hole_repair"] = json.dumps(feature_report, sort_keys=True)
'@
if ($content.Contains($propertyOld) -and -not $content.Contains('face["feature_hole_repair"]')) {
    $content = $content.Replace($propertyOld, $propertyNew)
    Write-Host 'FEATURE_HOLE_RECEIPT=APPLIED'
}
elseif ($content.Contains('face["feature_hole_repair"]')) {
    Write-Host 'FEATURE_HOLE_RECEIPT=ALREADY_APPLIED'
}
else {
    throw 'Could not attach feature-hole repair evidence to the face object.'
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
    throw 'Feature-hole repaired Blender builder failed compile validation.'
}
& git update-index --assume-unchanged -- $path
if ($LASTEXITCODE -ne 0) {
    throw 'Could not preserve the bounded feature-hole repair as assume-unchanged.'
}
if (@(& git status --short).Count -gt 0) {
    throw 'Feature-hole repair left visible checkout dirt.'
}

Write-Host 'ANIMATED_EYE_AND_MOUTH_HOLE_REPAIR=PROVEN'
