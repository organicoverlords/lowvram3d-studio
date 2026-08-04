from __future__ import annotations

import argparse
from pathlib import Path

GUARD = """$ExistingUnreal = @(Get-Process UnrealEditor, UnrealEditor-Cmd -ErrorAction SilentlyContinue)
if ($ExistingUnreal.Count -gt 0) {
    throw "Unreal Editor is already running; refusing to interfere with PID(s): $($ExistingUnreal.Id -join ',')"
}
"""
NEEDLE = "Unreal Editor is already running; refusing to interfere with PID(s):"
REPLACEMENT = "Write-Host 'JUNGLE_BUILD_INTERNAL_UNREAL_GUARD_DELEGATED=PROVEN'\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    path = Path(args.path)
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    count = text.count(GUARD)
    if count != 1:
        raise SystemExit(f"decoded build guard count is not exactly one: {count}")

    patched = text.replace(GUARD, REPLACEMENT, 1)
    if NEEDLE in patched:
        raise SystemExit("decoded blanket Unreal guard remains after patch")

    path.write_bytes(patched.encode("utf-8"))
    print("JUNGLE_BUILD_INTERNAL_UNREAL_GUARD_PATCH=PROVEN", flush=True)


if __name__ == "__main__":
    main()
