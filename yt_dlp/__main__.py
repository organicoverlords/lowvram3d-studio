from __future__ import annotations

"""Run the installed yt-dlp while normalizing its max-download success code.

The pinned worker intentionally asks for one download. yt-dlp can report exit code
101 after reaching that limit even though the requested media was written. The
GitHub workflow previously treated that as a failed download and deleted the valid
file before inspecting it.
"""

import importlib
import sys
from pathlib import Path


_LOCAL_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_ORIGINAL_SYS_PATH = list(sys.path)


def _resolved(path_entry: str) -> Path | None:
    try:
        return Path(path_entry or ".").resolve()
    except OSError:
        return None


# Remove the repository shim from resolution and import the installed distribution.
sys.modules.pop("yt_dlp", None)
sys.path = [
    entry
    for entry in sys.path
    if _resolved(entry) != _LOCAL_REPOSITORY_ROOT
]
try:
    real_yt_dlp = importlib.import_module("yt_dlp")
finally:
    sys.path = _ORIGINAL_SYS_PATH

try:
    real_yt_dlp.main()
except SystemExit as error:
    code = error.code if isinstance(error.code, int) else 1
    if code == 101:
        code = 0
    raise SystemExit(code) from None
