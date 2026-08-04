"""Minimal pyplot surface for non-visual 3DDFA reconstruction.

The reconstruction path imports 3DDFA's utils.functions for crop and ROI
helpers only.  Plotting is deliberately unsupported on the worker.  Any
unexpected plotting call fails closed with a precise error instead of silently
producing incomplete evidence.
"""

from __future__ import annotations


def _unsupported(*_args, **_kwargs):
    raise RuntimeError(
        "matplotlib plotting is unavailable in the headless Beggars reconstruction lane"
    )


figure = _unsupported
subplots_adjust = _unsupported
axis = _unsupported
imshow = _unsupported
show = _unsupported
plot = _unsupported
scatter = _unsupported
savefig = _unsupported
close = _unsupported
