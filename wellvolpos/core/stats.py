"""Sampling error: how much of a curve is signal and how much is 10 000 trials.

Every conditional number in this tool -- proven mean at a location, attic mean
given a dry hole, P(proven > MEFS) -- is computed over a *subset* of the trials,
and that subset shrinks as the well moves down-dip. At entry 3500 m the
discovery group holds 4 576 of 10 000 trials; at 3677 m it holds **8**. The
arithmetic is identical at both depths and so is the line width, which is the
problem this module exists to fix.

Two separate jobs, deliberately not conflated:

``bootstrap_mean_ci``
    How precisely is this statistic known, given the trials actually in the
    subset? Resampling with replacement inside the subset answers that and
    nothing else.

``support_mask`` / ``describe_support``
    Is the subset large enough to be worth drawing at all? A confidence
    interval computed from 8 trials is itself estimated from 8 trials, so a
    band is not a substitute for saying "this end of the curve is thin".

Neither addresses model error. A bootstrap band says how well the *trial file*
pins the statistic down; it says nothing about whether the contact distribution
that produced the file is right. That is the HCWC Builder's question, and
It is kept out of this tool deliberately: `r_location` already carries the
depth-dependent risk, so modelling it again would double-count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this many trials a conditional statistic is reported but not drawn as
# though it were supported. 30 is the conventional small-sample threshold and is
# a presentation choice, not a statistical law -- so it is a default here and an
# argument everywhere it is used, never a constant buried in a figure.
MIN_SUPPORT = 30


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 400,
    alpha: float = 0.10,
    seed: int | None = 0,
    min_n: int = 2,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``.

    Returns ``(lo, hi)`` at the **nominal** ``1 - alpha`` level, or
    ``(nan, nan)`` when there is too little to resample. ``seed`` is fixed by
    default: a band that moved every time a slider was nudged would read as new
    information rather than as Monte Carlo noise in the band itself.

    Nominal, not actual. The percentile method under-covers for small samples
    from a skewed distribution, and resource distributions are exactly that:
    measured coverage of the nominal 90 % interval on a lognormal with
    sigma = 0.8 is about 0.82 at n = 30, 0.85 at n = 100 and 0.87 at n = 1000.
    Since the band's whole purpose is the depths where the conditional group has
    thinned to tens of trials, that shortfall lands precisely where the band is
    being relied on -- so it is stated rather than papered over, and the figures
    label the level "nominal". BCa would tighten the coverage; it is not used
    because the band is drawn at 40-odd depths per sweep and the extra accuracy
    would not move a placement decision, whereas an unlabelled 90 % that is
    really 82 % might.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < max(int(min_n), 1):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    # One (n_boot, n) draw of indices: vectorised, so the whole sweep's bands
    # cost a fraction of the trial-splitting they annotate.
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = v[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_proportion_ci(
    flags: np.ndarray, *, n_boot: int = 400, alpha: float = 0.10, seed: int | None = 0,
    min_n: int = 2,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a proportion, e.g. P(proven > MEFS).

    Kept separate from :func:`bootstrap_mean_ci` only for clarity at the call
    site; a proportion is the mean of a boolean, and this delegates.
    """
    return bootstrap_mean_ci(
        np.asarray(flags, dtype=float), n_boot=n_boot, alpha=alpha, seed=seed, min_n=min_n
    )


def support_mask(counts: np.ndarray, min_support: int = MIN_SUPPORT) -> np.ndarray:
    """True where a step rests on at least ``min_support`` trials."""
    return np.asarray(counts) >= int(min_support)


def thin(values: np.ndarray, counts: np.ndarray, min_support: int = MIN_SUPPORT) -> np.ndarray:
    """``values`` with the under-supported entries blanked to NaN.

    Blanking rather than deleting keeps the array aligned with its depth grid,
    and NaN is what both matplotlib and plotly treat as a gap -- so an
    unsupported stretch of curve simply is not drawn, instead of being drawn as
    confidently as the rest.
    """
    out = np.array(values, dtype=float, copy=True)
    out[~support_mask(counts, min_support)] = np.nan
    return out


@dataclass
class Support:
    """Where a swept curve is well supported, and where it is not."""

    name: str
    n_min: int
    n_max: int
    n_below: int              # steps under the threshold
    n_steps: int
    min_support: int
    shallowest_thin: float | None
    deepest_thin: float | None

    @property
    def all_supported(self) -> bool:
        return self.n_below == 0

    def message(self) -> str:
        if self.all_supported:
            return (
                f"Every {self.name} step rests on at least {self.n_min:,} trials "
                f"(threshold {self.min_support})."
            )
        # The direction is derived, not assumed. The discovery group thins with
        # depth; the dry-with-attic group thins the other way, because near the
        # crest almost nothing is dry. Hardcoding "downward" was wrong for half
        # the curves this describes.
        where = ""
        if self.shallowest_thin is not None and self.deepest_thin is not None:
            if self.n_below == 1:
                where = f", at {self.shallowest_thin:.0f} m TVDSS"
            else:
                where = f", between {self.shallowest_thin:.0f} and {self.deepest_thin:.0f} m TVDSS"
        return (
            f"{self.n_below} of {self.n_steps} {self.name} steps rest on fewer than "
            f"{self.min_support} trials — as few as {self.n_min:,}{where}. Those steps are "
            f"left undrawn rather than shown as firmly as the rest."
        )


def describe_support(
    counts: np.ndarray, z: np.ndarray, min_support: int = MIN_SUPPORT, name: str = "",
) -> Support:
    """Summarise the support behind a swept curve, for the UI to state plainly.

    ``name`` names the conditional group, because a sweep has more than one and
    they thin at opposite ends -- reporting only the discovery group's support
    while the attic curve is missing from the top of the panel says nothing
    about the gap the reader can actually see.
    """
    c = np.asarray(counts)
    ok = support_mask(c, min_support)
    thin_z = np.asarray(z, dtype=float)[~ok]
    return Support(
        name=name or "swept",
        n_min=int(c.min()) if c.size else 0,
        n_max=int(c.max()) if c.size else 0,
        n_below=int((~ok).sum()),
        n_steps=int(c.size),
        min_support=int(min_support),
        shallowest_thin=float(thin_z.min()) if thin_z.size else None,
        deepest_thin=float(thin_z.max()) if thin_z.size else None,
    )
