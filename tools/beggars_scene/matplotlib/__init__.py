"""Headless compatibility shim for the pinned 3DDFA utility imports.

The Beggars reconstruction lane does not call 3DDFA's plotting helpers, but
3DDFA imports matplotlib.pyplot unconditionally from utils.functions.  The
worker environment intentionally excludes the full plotting stack.  Keeping
this tiny local package makes that optional import explicit and deterministic.
"""

from . import pyplot

__all__ = ["pyplot"]
