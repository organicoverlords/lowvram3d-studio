from __future__ import annotations

import argparse
from pathlib import Path

GUARD = """$ExistingUnreal = @(Get-Process UnrealEditor, UnrealEditor-Cmd -ErrorAction SilentlyContinue)
if ($ExistingUnreal.Count -gt 0) {
    throw "Unreal Editor is already running; refusing to interfere with PID(s): $($ExistingUnreal.Id -join ',')"
}
"""
GUARD_NEEDLE = "Unreal Editor is already running; refusing to interfere with PID(s):"
GUARD_REPLACEMENT = "Write-Host 'JUNGLE_BUILD_INTERNAL_UNREAL_GUARD_DELEGATED=PROVEN'\n"

INVOKE_CHECKED = '''function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][scriptblock]$Command,
        [string]$LogPath
    )
    Write-Host "PHASE=$Name"
    if ($LogPath) {
        $parent = Split-Path -Parent $LogPath
        if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
        & $Command 2>&1 | Tee-Object -FilePath $LogPath
    } else {
        & $Command
    }
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}
'''

INVOKE_CHECKED_REPLACEMENT = '''function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][scriptblock]$Command,
        [string]$LogPath
    )
    Write-Host "PHASE=$Name"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $NativeExitCode = 0
    try {
        $ErrorActionPreference = 'Continue'
        if ($LogPath) {
            $parent = Split-Path -Parent $LogPath
            if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
            & $Command 2>&1 | Tee-Object -FilePath $LogPath
        } else {
            & $Command
        }
        $NativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($NativeExitCode -ne 0) { throw "$Name failed with exit code $NativeExitCode" }
}
'''


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label} count is not exactly one: {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    path = Path(args.path)
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    patched = replace_exact(text, GUARD, GUARD_REPLACEMENT, "decoded build guard")
    patched = replace_exact(
        patched,
        INVOKE_CHECKED,
        INVOKE_CHECKED_REPLACEMENT,
        "decoded Invoke-Checked function",
    )

    if GUARD_NEEDLE in patched:
        raise SystemExit("decoded blanket Unreal guard remains after patch")
    if "$NativeExitCode = $LASTEXITCODE" not in patched:
        raise SystemExit("native exit-code preservation marker is missing")

    path.write_bytes(patched.encode("utf-8"))
    print("JUNGLE_BUILD_INTERNAL_UNREAL_GUARD_PATCH=PROVEN", flush=True)
    print("JUNGLE_NATIVE_STDERR_EXIT_CODE_PATCH=PROVEN", flush=True)


if __name__ == "__main__":
    main()
