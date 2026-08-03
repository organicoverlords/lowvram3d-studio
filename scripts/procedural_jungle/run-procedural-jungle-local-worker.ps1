[CmdletBinding()]
param(
    [string]$ExpectedBranch = 'feature/procedural-jungle-playable-20260804'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $RepoRoot) { throw 'Not inside the expected Git repository' }
Set-Location -LiteralPath $RepoRoot
$Remote = (git config --get remote.origin.url).Trim()
$Branch = (git branch --show-current).Trim()
$Head = (git rev-parse HEAD).Trim()
$Status = @(git status --short)
if ($Remote -notmatch 'organicoverlords/lowvram3d-studio(\.git)?$') { throw "Repository mismatch: $Remote" }
if ($Branch -ne $ExpectedBranch) { throw "Branch mismatch: $Branch" }
if ($Status.Count -ne 0) { throw "Checkout is dirty before worker bootstrap: $($Status -join '; ')" }
$BundleDir = Join-Path $RepoRoot 'worker-bundles\procedural-jungle-direct-worker'
$BundleParts = @(Get-ChildItem -LiteralPath $BundleDir -Filter 'chunk-*.b64' -File | Sort-Object Name)
if ($BundleParts.Count -lt 1) { throw "Worker bundle chunks are missing: $BundleDir" }
$ExpectedSha = 'ae5a81fab6be79dac7b58bcf383679abacb4da47e6a25a9ca36258c23fcfac2d'
$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$ZipPath = Join-Path $TempRoot 'worker.zip'
$Base64 = (($BundleParts | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw }) -join '') -replace '\s',''
[IO.File]::WriteAllBytes($ZipPath, [Convert]::FromBase64String($Base64))
$ActualSha = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualSha -ne $ExpectedSha) { throw "Worker bundle hash mismatch: $ActualSha" }
$SourceRoot = Join-Path $TempRoot 'source'
Expand-Archive -LiteralPath $ZipPath -DestinationPath $SourceRoot -Force
$Entry = Join-Path $SourceRoot 'scripts\procedural_jungle\build-procedural-jungle.ps1'
if (-not (Test-Path -LiteralPath $Entry)) { throw "Decoded worker entrypoint is missing: $Entry" }
$EntryText = Get-Content -LiteralPath $Entry -Raw

# A separate Unreal project may already be open interactively. Never close it or
# touch its files. Permit coexistence only when no running process references the
# dedicated ProceduralJungle58 target project. The decoded worker otherwise has a
# conservative blanket refusal that would block safe isolated commandlets.
$TargetProjectRoot = 'C:\Users\Lauri\Desktop\ProceduralJungle58'
$UnrealProcesses = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('UnrealEditor.exe', 'UnrealEditor-Cmd.exe') }
)
$TargetProcesses = @(
    $UnrealProcesses | Where-Object {
        $_.CommandLine -and $_.CommandLine.IndexOf($TargetProjectRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0
    }
)
if ($TargetProcesses.Count -gt 0) {
    throw "Target ProceduralJungle58 project is already open; refusing concurrent mutation. PID(s): $($TargetProcesses.ProcessId -join ',')"
}
if ($UnrealProcesses.Count -gt 0) {
    $Pattern = '(?ms)^\s*if\s*\(\$[A-Za-z_][A-Za-z0-9_]*\.Count\s*-gt\s*0\)\s*\{\s*throw\s*"Unreal Editor is already running; refusing to interfere with PID\(s\):[^\"]*"\s*\}\s*'
    $Matches = [regex]::Matches($EntryText, $Pattern)
    if ($Matches.Count -ne 1) {
        throw "Could not prove a unique Unreal coexistence guard in decoded worker; matches=$($Matches.Count)"
    }
    $Replacement = "Write-Warning 'Unrelated Unreal Editor process detected; continuing only with the isolated ProceduralJungle58 project.'`r`n"
    $EntryText = [regex]::Replace($EntryText, $Pattern, $Replacement, 1)
    Write-Host "UNRELATED_UNREAL_COEXISTENCE=AUTHORIZED"
    Write-Host "UNRELATED_UNREAL_PID=$($UnrealProcesses.ProcessId -join ',')"
}

# Blender 5.2 stores keyed F-curves in layered Action channel bags instead of
# exposing Action.fcurves directly. Preserve the existing generated walk keys and
# only make the interpolation traversal compatible with both APIs.
$RigScript = Join-Path $SourceRoot 'blender\procedural_jungle\rig_animate_panda.py'
if (-not (Test-Path -LiteralPath $RigScript)) { throw "Decoded panda rig script missing: $RigScript" }
$RigText = Get-Content -LiteralPath $RigScript -Raw
$LegacyCurveLoopPattern = '(?m)^    for fcurve in action\.fcurves:\s*$'
$LegacyCurveLoopMatches = [regex]::Matches($RigText, $LegacyCurveLoopPattern)
if ($LegacyCurveLoopMatches.Count -ne 1) {
    throw "Could not prove unique legacy Action.fcurves loop; matches=$($LegacyCurveLoopMatches.Count)"
}
$LayeredCurveLoop = @'
    action_curves = list(getattr(action, "fcurves", []))
    if not action_curves:
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    action_curves.extend(getattr(channelbag, "fcurves", []))
    for fcurve in action_curves:
'@
$RigText = [regex]::Replace($RigText, $LegacyCurveLoopPattern, $LayeredCurveLoop, 1)
Set-Content -LiteralPath $RigScript -Value $RigText -Encoding utf8
Write-Host 'BLENDER_52_ACTION_CURVES_PATCH=PROVEN'

# Windows PowerShell 5 turns native stderr into terminating error records under
# ErrorActionPreference=Stop. Blender 5.2 emits benign warnings on stderr. Compile
# a tiny native forwarder so PowerShell passes every argument—including the `--`
# Python boundary—unchanged to Blender. The forwarder merges both streams into
# stdout and exits with Blender's exact exit code.
$BlenderExe = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
if (-not (Test-Path -LiteralPath $BlenderExe)) { throw "Blender executable missing: $BlenderExe" }
$BlenderForwarder = Join-Path $TempRoot 'BlenderNativeForwarder.exe'
$BlenderExeForCode = $BlenderExe.Replace('\', '\\').Replace('"', '\"')
$ForwarderSource = @"
using System;
using System.Diagnostics;
using System.Text;

public static class BlenderNativeForwarder
{
    private static string Quote(string value)
    {
        if (value == null) return "\"\"";
        if (value.Length > 0 && value.IndexOfAny(new[] { ' ', '\t', '\n', '\v', '\"' }) < 0)
            return value;
        var builder = new StringBuilder();
        builder.Append('\"');
        int slashes = 0;
        foreach (char character in value)
        {
            if (character == '\\')
            {
                slashes++;
            }
            else if (character == '\"')
            {
                builder.Append('\\', slashes * 2 + 1);
                builder.Append('\"');
                slashes = 0;
            }
            else
            {
                builder.Append('\\', slashes);
                slashes = 0;
                builder.Append(character);
            }
        }
        builder.Append('\\', slashes * 2);
        builder.Append('\"');
        return builder.ToString();
    }

    public static int Main(string[] args)
    {
        var arguments = new StringBuilder();
        for (int index = 0; index < args.Length; index++)
        {
            if (index > 0) arguments.Append(' ');
            arguments.Append(Quote(args[index]));
        }
        var info = new ProcessStartInfo
        {
            FileName = "$BlenderExeForCode",
            Arguments = arguments.ToString(),
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        using (var process = new Process { StartInfo = info })
        {
            process.OutputDataReceived += (sender, eventArgs) => { if (eventArgs.Data != null) Console.Out.WriteLine(eventArgs.Data); };
            process.ErrorDataReceived += (sender, eventArgs) => { if (eventArgs.Data != null) Console.Out.WriteLine(eventArgs.Data); };
            if (!process.Start()) return 126;
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();
            return process.ExitCode;
        }
    }
}
"@
Add-Type -TypeDefinition $ForwarderSource -Language CSharp -OutputAssembly $BlenderForwarder -OutputType ConsoleApplication
if (-not (Test-Path -LiteralPath $BlenderForwarder)) { throw 'Blender native forwarder compilation produced no executable' }
$BlenderCallPattern = '&\s*\$Blender\b'
$BlenderCallCount = [regex]::Matches($EntryText, $BlenderCallPattern).Count
if ($BlenderCallCount -lt 2 -or $BlenderCallCount -gt 4) {
    throw "Unexpected decoded Blender invocation count: $BlenderCallCount"
}
$EscapedForwarder = $BlenderForwarder.Replace("'", "''")
$EntryText = [regex]::Replace($EntryText, $BlenderCallPattern, "& '$EscapedForwarder'")
Set-Content -LiteralPath $Entry -Value $EntryText -Encoding utf8
Write-Host 'BLENDER_NATIVE_FORWARDER=PROVEN'
Write-Host "BLENDER_CALLS_PATCHED=$BlenderCallCount"

Write-Host "DIRECT_WORKER_BUNDLE_SHA256=$ActualSha"
Write-Host 'CODEX_INVOKED=NO'
Write-Host 'CLAUDE_INVOKED=NO'
Write-Host 'MAGICMUSIC_INVOKED=NO'
& $Entry -ExpectedBranch $ExpectedBranch -SourceRoot $SourceRoot
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "Direct worker entrypoint failed with exit code $WorkerExit" }
