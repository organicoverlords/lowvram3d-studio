"""Generic compatibility entry point for building registered projection inputs.

The implementation is shared with the proven CPU registration worker; the production stage does
not encode an asset or anatomy-specific policy here.
"""
from shaman_texture_views_oriented import main  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(main())
