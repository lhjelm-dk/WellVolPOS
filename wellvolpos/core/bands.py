"""Resource percentiles within contact-depth bands.

The arithmetic behind figure 3.12, which is Schneider et al. (2023) Figure 9 with
its parameterisation changed. The poster's figure draws one distribution per
**productive-area increment**; Lars asked for one per **contact-depth interval**
instead (2026-08-12), which is the same idea read in the variable the well
actually controls -- you cannot choose an area, you choose a depth, and
``A(z)`` turns the one into the other.

Why depth is the better parameter here, not merely the more convenient one:
area is a *consequence* of the contact depth (``core.structure.AreaDepth`` fits
it at R2 = 0.9999999987 on the reference file), so banding by area and banding by
depth partition the same trials -- but only the depth bands can be read against a
well. "If the contact turns out to lie between 3450 and 3520 m, this is the
resource distribution, and this is the part my well would have proven" is a
sentence about a decision. The area version is a sentence about a map.

Two families come back per band, and the distinction is the one this project
draws everywhere:

* **total** -- the whole resource of every success trial whose contact falls in
  the band. Conditional on the prospect working *and* the contact landing there.
* **proven** -- the part the well at ``z_entry``/``z_exit`` would prove, over the
  band's **discovery** trials only. Conditional again, on the same events plus
  the well being in hydrocarbons.

The proven family is deliberately *not* averaged over the band's dry trials, and
**no band is allowed to straddle the well entry** -- the entry is always a band
boundary, in both modes. Mixing the two populations would put zeros into a
distribution drawn on a logarithmic axis, where a zero cannot be drawn at all and
would silently become "the smallest thing on the plot" rather than "nothing"; and
taking ``total`` over a straddling band while taking ``proven`` over its deeper
discovery subset put the dotted curve *to the right* of its own solid curve, a
well appearing to prove more than the band contains. With the entry as an edge,
``n_discovery`` is either zero or the whole band and ``proven <= total`` holds by
construction.

**Percentiles are exceedance probabilities**, this project's convention
throughout (Milkov 2021: *"P90 is defined as 90 % probability of exceeding the
P90 estimated value"*). So P99 is a small volume and P1 a large one. Schneider's
own figure runs the other way, cumulative-less-than; the numbers are the same
distribution read from the other end, and the figure says which is on screen.

A percentile is only reported where the band has the trials to support it: the
ladder is gated so that at least :data:`MIN_TAIL` trials lie beyond each
reported point, and it is gated **once, on the smallest series drawn**, so every
band reports the same ladder. Bands that report different percentiles cannot be
compared by eye, and comparing them is the entire purpose of the figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .stats import MIN_SUPPORT

if TYPE_CHECKING:  # pragma: no cover
    # Annotation-only, and deliberately so. ``core.classes`` reaches out to
    # ``io.adapters.base`` for TrialSet, and ``io`` reaches back into
    # ``core.classes`` for the area/pay check -- a cycle that only resolves
    # because ``core.__init__`` happens to warm ``io`` first. Importing these for
    # real from here put this module ahead of that in the load order and broke it.
    # Nothing below needs the classes at runtime; it reads attributes off
    # instances the caller already built.
    from .classes import VolumeClasses
    from .groups import Groups
    from ..io.adapters.base import TrialSet

#: How the contact range is cut into bands.
BAND_MODES = ("equal_count", "equal_width")

BAND_MODE_LABELS = {
    "equal_count": "equal trial count",
    "equal_width": "equal depth interval",
}

#: The exceedance percentiles the figure would like to draw, richest first. The
#: ladder actually drawn is this filtered by :func:`supported_percentiles`.
BAND_PERCENTILES = (99, 95, 90, 75, 50, 25, 10, 5, 1)

#: Trials that must lie beyond a percentile before it is reported. Two is the
#: minimum that makes the point an observation rather than an extrapolation of
#: the interpolator between the two most extreme trials.
MIN_TAIL = 2

#: Default number of bands. Six is as many families as the sequential scale can
#: be told apart at, and on 10 000 trials it leaves ~1 250 per band.
DEFAULT_N_BANDS = 6


def supported_percentiles(
    n: int, percentiles: tuple[int, ...] = BAND_PERCENTILES, *, min_tail: int = MIN_TAIL
) -> tuple[int, ...]:
    """The subset of ``percentiles`` that ``n`` trials can actually support.

    A percentile is kept when at least ``min_tail`` trials fall beyond it, so P99
    needs 200 trials and P50 needs four. Below that, ``np.percentile`` is
    interpolating between the two most extreme trials in the sample and the
    answer is a property of the interpolator rather than of the distribution.
    """
    return tuple(
        p for p in percentiles if n * min(p, 100 - p) / 100.0 >= float(min_tail)
    )


def _exceedance(values: np.ndarray, percentiles: tuple[int, ...]) -> np.ndarray:
    """Volumes at exceedance percentiles ``percentiles``.

    P90 = the value exceeded in 90 % of cases = the 10th ordinary percentile.
    """
    if not percentiles:
        return np.empty(0, dtype=float)
    return np.percentile(values, [100 - p for p in percentiles])


def _exceedance_of(values: np.ndarray, x: float) -> float:
    """The exceedance probability of ``x`` in ``values``, as a percentage."""
    v = np.asarray(values, dtype=float)
    return float(np.count_nonzero(v > x) / v.size * 100.0) if v.size else float("nan")


@dataclass(frozen=True)
class DepthBand:
    """One contact-depth interval and the two resource distributions in it."""

    z_top: float
    z_base: float
    n: int
    #: Volumes at the reported exceedance percentiles, over every success trial
    #: in the band.
    total: np.ndarray = field(repr=False)
    #: The mean total, and the exceedance probability at which it sits. A mean is
    #: not a percentile; carrying its own probability is what lets it be drawn on
    #: the same axes without pretending to be one.
    total_mean: float = 0.0
    total_mean_p: float = float("nan")
    #: The same for the proven part, over the band's discovery trials only.
    #: ``None`` when the band holds no discoveries -- entirely above the well.
    n_discovery: int = 0
    proven: np.ndarray | None = field(default=None, repr=False)
    proven_mean: float = 0.0
    proven_mean_p: float = float("nan")

    @property
    def label(self) -> str:
        return f"{self.z_top:.0f}-{self.z_base:.0f} m"

    @property
    def mid(self) -> float:
        return 0.5 * (self.z_top + self.z_base)

    @property
    def discovery_fraction(self) -> float:
        return self.n_discovery / self.n if self.n else 0.0


@dataclass(frozen=True)
class BandedPercentiles:
    """Every band, on one shared percentile ladder."""

    bands: tuple[DepthBand, ...]
    percentiles: tuple[int, ...]
    mode: str
    z_entry: float
    z_exit: float
    n_success: int
    n_banded: int
    interval_m: float | None = None
    #: Success trials dropped for a non-positive resource, which cannot be drawn
    #: on a logarithmic axis. Zero on well-formed GeoX output; carried so that a
    #: file where it is not zero says so rather than quietly losing trials.
    n_nonpositive: int = 0
    #: Bands discarded for holding fewer than ``min_support`` trials.
    n_bands_dropped: int = 0

    @property
    def volume_range(self) -> tuple[float, float]:
        lo = min(
            float(np.min(b.total))
            for b in self.bands
            if b.total.size
        )
        hi = max(
            float(np.max(b.total))
            for b in self.bands
            if b.total.size
        )
        return lo, hi


def banded_percentiles(
    ts: "TrialSet",
    groups: "Groups",
    vc: "VolumeClasses",
    *,
    z_entry: float,
    z_exit: float,
    mode: str = "equal_count",
    n_bands: int = DEFAULT_N_BANDS,
    interval_m: float | None = None,
    percentiles: tuple[int, ...] = BAND_PERCENTILES,
    min_support: int = MIN_SUPPORT,
) -> BandedPercentiles:
    """Cut the contact range into bands and take resource percentiles in each.

    **The well entry is always a band boundary**, in both modes, and that is a
    correctness requirement rather than a tidiness one. A band straddling the entry
    holds dry trials and discoveries together; its ``total`` is then taken over all
    of them while its ``proven`` is taken over the deeper discovery subset only --
    and since the deeper trials hold more, the dotted curve came out *to the right*
    of its own solid curve, reading as a well proving more than the band contains.
    Measured on the reference file before the fix: 1.06x at the P50 of the
    3493-3510 m band. Making the entry an edge removes the mixed population
    entirely, so ``n_discovery`` is either zero or the whole band and
    ``proven <= total`` holds by construction.

    ``mode="equal_count"`` gives every band about the same number of trials, so the
    percentile ladder is uniformly supported and no band is drawn from a handful of
    trials; the depth intervals then vary, and each band's own interval is in its
    label. The bands are shared out between the dry side and the discovery side in
    proportion to their trial counts, then cut into equal counts *within* each side,
    so "about" is the price of the entry boundary.

    ``mode="equal_width"`` gives every band the same interval -- ``interval_m`` if
    supplied, otherwise the range divided by ``n_bands`` -- which is easier to read
    against a structural section, but the contact distribution is not uniform so the
    shallow and deep bands come out thin. Its grid is **anchored on the well entry**
    and stepped outward from there, which is what makes the entry an edge while
    keeping every width identical; the shallowest interval may therefore start above
    the shallowest sampled contact, since it is an interval and not a data extent.

    Neither mode is the default in any hidden sense: it is a stated setting
    (non-negotiable 5) and an unknown one raises rather than falling back.
    """
    if mode not in BAND_MODES:
        raise ValueError(
            f"unknown band mode {mode!r}; expected one of {BAND_MODES}"
        )

    contact = ts.col("contact")
    resource = ts.col("resource")
    ok = np.asarray(groups.success, dtype=bool)
    n_success = int(ok.sum())
    positive = ok & (resource > 0.0)
    n_nonpositive = n_success - int(positive.sum())

    z = contact[positive]
    v = resource[positive]
    proven_all = np.asarray(vc.proven, dtype=float)[positive]
    is_discovery = np.asarray(groups.discovery, dtype=bool)[positive]

    edges = _band_edges(z, mode=mode, n_bands=n_bands, interval_m=interval_m,
                        z_entry=float(z_entry))

    # First pass: which trials belong to which band, and how big is the smallest
    # series that will be drawn. The ladder is gated on that, once, so every band
    # reports the same percentiles.
    members: list[tuple[float, float, np.ndarray]] = []
    dropped = 0
    for z_top, z_base, first in _spans(edges):
        # Bands are **(top, base]**, open at the shallow end, because that is
        # exactly ``group_trials``' rule: a discovery is ``contact > z_entry``. With
        # the entry as an edge, the band below it then holds precisely the dry
        # trials and the band above it precisely the discoveries. Closing the
        # shallow end instead put a prospect-B trial whose contact sits *exactly* on
        # the entry into the discovery band while the engine called it dry -- one
        # trial in 10 000, and enough to make that band's proven percentiles exceed
        # its own totals. The shallowest band also takes its own lower edge, or the
        # shallowest contact in the file would belong to no band at all.
        m = (z >= z_top) if first else (z > z_top)
        m = m & (z <= z_base)
        if int(m.sum()) < min_support:
            dropped += 1
            continue
        members.append((z_top, z_base, m))

    if not members:
        raise ValueError(
            f"no contact-depth band holds {min_support} trials; "
            f"try fewer bands than {n_bands} or a wider interval"
        )

    smallest = min(int(m.sum()) for _, _, m in members)
    for _, _, m in members:
        n_disc = int((m & is_discovery).sum())
        if n_disc >= min_support:
            smallest = min(smallest, n_disc)
    ladder = supported_percentiles(smallest, percentiles)

    bands: list[DepthBand] = []
    for z_top, z_base, m in members:
        vals = v[m]
        mean = float(vals.mean())
        d = m & is_discovery
        n_disc = int(d.sum())
        if n_disc >= min_support:
            pv = proven_all[d]
            pv = pv[pv > 0.0]
        else:
            pv = np.empty(0, dtype=float)
        bands.append(
            DepthBand(
                z_top=float(z_top),
                z_base=float(z_base),
                n=int(m.sum()),
                total=_exceedance(vals, ladder),
                total_mean=mean,
                total_mean_p=_exceedance_of(vals, mean),
                n_discovery=n_disc,
                proven=_exceedance(pv, ladder) if pv.size else None,
                proven_mean=float(pv.mean()) if pv.size else 0.0,
                proven_mean_p=_exceedance_of(pv, pv.mean()) if pv.size else float("nan"),
            )
        )

    return BandedPercentiles(
        bands=tuple(bands),
        percentiles=ladder,
        mode=mode,
        z_entry=float(z_entry),
        z_exit=float(z_exit),
        n_success=n_success,
        n_banded=int(sum(b.n for b in bands)),
        interval_m=float(interval_m) if interval_m else None,
        n_nonpositive=n_nonpositive,
        n_bands_dropped=dropped,
    )


def _spans(edges: np.ndarray):
    """Yield ``(top, base, is_first)`` for consecutive edges."""
    for i in range(len(edges) - 1):
        yield float(edges[i]), float(edges[i + 1]), i == 0


def _band_edges(
    z: np.ndarray, *, mode: str, n_bands: int, interval_m: float | None, z_entry: float
) -> np.ndarray:
    """Band boundaries in contact depth, shallow first, with ``z_entry`` among them.

    See :func:`banded_percentiles` for why the entry has to be an edge. Both modes
    achieve it differently: equal-count splits its quota between the two sides of the
    entry, equal-width steps its grid outward *from* the entry.
    """
    lo, hi = float(z.min()), float(z.max())
    if mode == "equal_width":
        step = float(interval_m) if interval_m else (hi - lo) / max(int(n_bands), 1)
        if step <= 0:
            raise ValueError("interval_m must be positive")
        # Anchored on the entry, stepped both ways to cover the data. floor/ceil
        # rather than round, so the outermost intervals contain lo and hi rather
        # than stopping just short of them.
        k0 = int(np.floor((lo - z_entry) / step))
        k1 = int(np.ceil((hi - z_entry) / step))
        return z_entry + step * np.arange(k0, k1 + 1)

    n_bands = max(int(n_bands), 1)
    if not (lo < z_entry < hi):
        # The entry is outside the sampled contacts, so every trial is on one side
        # of it and there is nothing to split.
        edges = np.unique(np.percentile(z, np.linspace(0.0, 100.0, n_bands + 1)))
        edges[0], edges[-1] = lo, hi
        return edges

    above, below = z[z < z_entry], z[z >= z_entry]
    # Shared out in proportion to trial count, at least one band a side, so the
    # counts stay comparable across the boundary instead of one side being cut
    # finely and the other coarsely.
    n_above = int(round(n_bands * above.size / z.size))
    n_above = min(max(n_above, 1), n_bands - 1)
    n_below = n_bands - n_above
    interior_above = np.percentile(above, np.linspace(0.0, 100.0, n_above + 1))[1:-1]
    interior_below = np.percentile(below, np.linspace(0.0, 100.0, n_below + 1))[1:-1]
    edges = np.concatenate(
        [[lo], interior_above, [z_entry], interior_below, [hi]]
    )
    # Equal-count edges can repeat where the contact distribution has an atom;
    # collapsing them here keeps empty bands out of the count.
    return np.unique(edges)
