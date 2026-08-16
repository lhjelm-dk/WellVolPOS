"""Make a trial export publishable by moving it, without changing what it says.

A prospect's *location* is the identifying part of a Monte Carlo export. Depths tie a
file to a basin, a block and usually a well; the volume and area distributions on their
own describe a shape, not a place.

So :func:`offset_depths` shifts every depth column by one constant and touches nothing
else. That is a rigid translation of the structure:

* column heights, ``contact - crest`` and ``spill - contact``, are differences and are
  unchanged;
* ``A(z)`` is the same curve, moved -- so its gradient, its apex *height above the
  deepest contact*, and every volume derived from it are identical;
* ``r_location`` at a well moved by the same offset is identical, because it counts
  contacts either side of a depth and both sides moved together;
* the resource, area and pay distributions are untouched, so every percentile,
  ``POS_trials``, and the proven / attic split all come out the same.

**What this does and does not protect.** It anonymises *where*. It does not anonymise
*how much*: the volume distribution is the original's, and someone holding the source
data could recognise it. That is a deliberate, stated limit -- the alternative, scaling
volumes, breaks the row-level consistency the QC gate checks (``GRV = area x pay``) and
would make the file a worse test of the importer than a real one.

Use it for a demo file. Do not use it to publish something whose *volumes* are the
sensitive part.
"""

from __future__ import annotations

import pandas as pd

#: Column names, as they appear in a GeoX export, that carry a depth in m TVDSS.
#: Matched case-insensitively on substrings, because the exporter's names vary and a
#: missed depth column is the one failure mode that matters here -- it would leave the
#: real depth in the file beside the shifted ones, which is worse than not shifting at
#: all.
DEPTH_HINTS = ("contact", "crest", "spill", "depth", "tvdss")


def depth_columns(frame: pd.DataFrame) -> list[str]:
    """Every column that looks like a depth, by name."""
    return [c for c in frame.columns
            if any(h in str(c).lower() for h in DEPTH_HINTS)]


def offset_depths(frame: pd.DataFrame, *, offset: float | None = None,
                  shallowest: float | None = None) -> tuple[pd.DataFrame, float]:
    """Return ``(shifted frame, offset applied)``.

    Give ``offset`` directly, or give ``shallowest`` to have the offset chosen so the
    shallowest depth in the file lands there. The second is usually what you want: a
    round number well away from any real prospect reads as obviously synthetic, which
    is the point -- a shifted file that still looks plausible invites being taken for
    a real location.
    """
    cols = depth_columns(frame)
    if not cols:
        raise ValueError(
            "no depth-like column found; refusing to write an 'anonymised' file that "
            f"is identical to its source. Looked for {DEPTH_HINTS} in {list(frame.columns)}"
        )
    if (offset is None) == (shallowest is None):
        raise ValueError("give exactly one of offset= or shallowest=")

    out = frame.copy()
    if offset is None:
        lo = min(float(out[c].min()) for c in cols)
        offset = float(shallowest) - lo
    for c in cols:
        out[c] = out[c] + offset
    return out, float(offset)
