"""Excel-exact reimplementations of the statistical functions the source workbook uses.

The original tool is `WELL Location POS and Resources V10052017_prospect A.xlsx`.
Its numbers are the specification for this port, so these functions have to match
Excel bit-for-bit rather than merely closely.

Two behaviours are easy to get wrong and both matter:

1. ``PERCENTILE.EXC`` is not ``numpy.percentile``. Excel places the k-th
   percentile at position ``k*(n+1)`` in the sorted array (1-indexed) and is
   undefined outside ``1/(n+1) <= k <= n/(n+1)``.

2. ``PERCENTRANK`` **truncates its result to 3 significant digits by default**
   (the optional ``significance`` argument). This is not a display artefact --
   the truncated value propagates into every downstream cell. It is precisely
   why ``Results!V15`` equals 0.7708238778159993: the raw rank at 3500 m is
   0.5424124254241237, Excel truncates it to 0.542, and ``(1-0.542)**(1/3)``
   reproduces the cell to all 16 digits.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "percentile_inc",
    "percentile_exc",
    "percentrank_exc",
    "percentrank_inc",
    "truncate_to_significant_digits",
]


def _clean(a) -> np.ndarray:
    """Excel ignores blanks, text and logicals inside a range."""
    a = np.asarray(a, dtype=float).ravel()
    return np.sort(a[np.isfinite(a)])


def truncate_to_significant_digits(value: float, significance: int = 3) -> float:
    """Truncate (not round) towards zero at ``significance`` significant digits.

    Excel's PERCENTRANK does this to its result. ``truncate_to_significant_digits(0.5424124, 3)``
    returns 0.542, not 0.542412 and not 0.543.
    """
    if not np.isfinite(value) or value == 0.0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    factor = 10.0 ** (significance - 1 - exponent)
    return math.trunc(value * factor) / factor


def percentile_inc(a, p: float) -> float:
    """Excel PERCENTILE.INC / PERCENTILE. Identical to numpy's linear method.

    ``p`` is a fraction in [0, 1]. Note the orientation trap that runs through
    this whole domain: a P90 *volume* in petroleum usage is the value exceeded
    90 % of the time, i.e. ``percentile_inc(v, 0.10)``. This function takes the
    statistical percentile, not the petroleum one -- callers convert.
    """
    arr = _clean(a)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p * 100.0, method="linear"))


def percentile_exc(a, p: float) -> float:
    """Excel PERCENTILE.EXC.

    Returns NaN outside the valid range, matching Excel's #NUM! -- the workbook
    is littered with #NUM! for exactly this reason and the port must reproduce
    the condition rather than silently extrapolate.
    """
    arr = _clean(a)
    n = arr.size
    if n == 0:
        return float("nan")
    if p < 1.0 / (n + 1) or p > n / (n + 1):
        return float("nan")
    pos = p * (n + 1)          # 1-indexed position
    lo = int(math.floor(pos))
    frac = pos - lo
    if lo < 1:
        return float(arr[0])
    if lo >= n:
        return float(arr[-1])
    return float(arr[lo - 1] + frac * (arr[lo] - arr[lo - 1]))


def _percentrank(arr: np.ndarray, x: float, exclusive: bool) -> float:
    n = arr.size
    if n == 0:
        return float("nan")
    if exclusive:
        if x < arr[0] or x > arr[-1]:
            return float("nan")
    else:
        if x < arr[0] or x > arr[-1]:
            return float("nan")

    # index of the first element >= x
    k = int(np.searchsorted(arr, x, side="left"))
    if k < n and arr[k] == x:
        # exact hit: Excel uses the first matching position
        return (k + 1) / (n + 1) if exclusive else k / (n - 1) if n > 1 else 1.0
    if k == 0:
        return float("nan") if exclusive else 0.0
    lo = k - 1
    span = arr[k] - arr[lo]
    frac = 0.0 if span == 0 else (x - arr[lo]) / span
    if exclusive:
        return (lo + 1 + frac) / (n + 1)
    return (lo + frac) / (n - 1) if n > 1 else 1.0


def percentrank_exc(a, x: float, significance: int = 3) -> float:
    """Excel PERCENTRANK.EXC, including the 3-significant-digit truncation.

    Pass ``significance=None`` to skip the truncation and get the raw rank --
    useful when you want the statistically clean value rather than Excel parity.
    """
    arr = _clean(a)
    raw = _percentrank(arr, float(x), exclusive=True)
    if significance is None or not np.isfinite(raw):
        return raw
    return truncate_to_significant_digits(raw, significance)


def percentrank_inc(a, x: float, significance: int = 3) -> float:
    """Excel PERCENTRANK.INC / PERCENTRANK, including the truncation."""
    arr = _clean(a)
    raw = _percentrank(arr, float(x), exclusive=False)
    if significance is None or not np.isfinite(raw):
        return raw
    return truncate_to_significant_digits(raw, significance)
