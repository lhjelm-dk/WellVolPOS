"""**Not surfaced in the app** (Lars, 2026-08-11): the minimum-column-height mapping
was removed from tab ⑤ and from the export. The module is kept because decision 6 is
a decision of record and the arithmetic here is tested and correct -- what was
removed is a *panel*, not a finding -- and because applying the threshold rather
than mapping it still needs a decision nobody has made: whether a sub-minimum trial
becomes a chance failure (lowering POS) or leaves the population (renormalising it).
Delete this module only along with that decision.

The assessment minimum, expressed as a minimum column height.

Hood (2024) argues the assessment minimum "can be effectively linked to seal
capacity if based on a minimum column height and not a minimum hydrocarbon
volume", which is why column height is the primary control here rather than the
P99.5 volume floor the source workbook used.

The two are close but **not identical**, and this module exists to show the
difference rather than assume it away. On the reference dataset:

* At a *fixed* contact depth -- hence a fixed column height -- the resource still
  spans roughly 3x, because area is pinned but gross pay and recovery yield are
  not.
* Cutting at the P98 level, the two definitions disagree on about 1 % of the
  excluded set. They diverge further when area and net pay are correlated.
* Both were already applied upstream: no realisation in the file sits below the
  simulator's own assessment minimum, so a looser threshold here changes nothing
  and the UI says so instead of silently doing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.adapters.base import TrialSet
from .structure import AreaDepth


@dataclass
class ThresholdMapping:
    min_column_height: float
    apex: float
    min_contact_depth: float
    min_area: float | None
    equivalent_volume: float | None       # smallest resource surviving the cut
    equivalent_percentile: float | None   # its petroleum percentile in the success set
    n_excluded: int
    frac_excluded: float
    binds: bool
    message: str


@dataclass(frozen=True)
class ColumnCheck:
    """What the trials say about column height, against the apex and a minimum."""

    apex: float
    n_success: int
    #: Success trials whose contact is at or above the apex: a positive volume with a
    #: zero or negative column, which cannot both be true.
    n_no_column: int
    min_column: float
    max_column: float
    #: Success trials below ``min_column_height``, and what treating them as chance
    #: failures would do to POS. ``None`` when no minimum is set.
    min_column_height: float | None = None
    n_sub_minimum: int = 0
    pos_before: float = float("nan")
    pos_after: float = float("nan")

    @property
    def contradicts(self) -> bool:
        return self.n_no_column > 0

    @property
    def binds(self) -> bool:
        return self.n_sub_minimum > 0

    def message(self) -> str:
        if self.contradicts:
            return (f"**{self.n_no_column:,} success trials have their contact at or above "
                    f"the apex ({self.apex:,.0f} m)** — a positive volume with no column, "
                    f"which cannot both be true. Either the apex extrapolation has "
                    f"overshot or the export is inconsistent; treat every column-height "
                    f"figure with suspicion until it is resolved.")
        if self.min_column_height is None:
            return (f"No trial has its contact at or above the apex, so every success case "
                    f"carries a real column — {self.min_column:,.0f} m at the thinnest.")
        if not self.binds:
            return (f"A {self.min_column_height:,.0f} m minimum excludes nothing: the "
                    f"thinnest column in the trials is {self.min_column:,.0f} m, so the cut "
                    f"does not bind and POS is unchanged.")
        return (f"A {self.min_column_height:,.0f} m minimum would reclassify "
                f"**{self.n_sub_minimum:,} trials** as chance failures — too thin to flow "
                f"is a failed well, not a small success — taking POS from "
                f"{self.pos_before:.4f} to **{self.pos_after:.4f}**.")


def check_column_heights(
    ts: TrialSet, apex: float, min_column_height: float | None = None
) -> ColumnCheck:
    """Column heights implied by the trials, against the apex and an optional minimum.

    Two separate questions, and only the first is a defect:

    1. **Is any success trial at or above the apex?** That is a positive volume with a
       zero or negative column -- a contradiction rather than a thin accumulation.
       Zero on both demo files, which is what a clean file looks like.
    2. **How many trials would a minimum column height exclude, and what would that do
       to POS?** Sub-minimum trials **lower POS** (Lars, 2026-08-13): they become
       chance failures rather than leaving the population, because an accumulation too
       thin to flow is a failed well and belongs in the denominator.

    Reported, never applied. Nothing in the app filters on this -- it is here so a
    minimum can be argued about with the count in front of you.
    """
    res = np.asarray(ts.col("resource"), dtype=float)
    contact = np.asarray(ts.col("contact"), dtype=float)
    success = res > 0.0
    n_success = int(success.sum())
    columns = contact[success] - float(apex)
    n_no_column = int((columns <= 0.0).sum())

    out = dict(
        apex=float(apex), n_success=n_success, n_no_column=n_no_column,
        min_column=float(columns.min()) if columns.size else float("nan"),
        max_column=float(columns.max()) if columns.size else float("nan"),
    )
    if min_column_height is None:
        return ColumnCheck(**out)

    n_sub = int((columns < float(min_column_height)).sum())
    n_total = res.size
    pos_before = n_success / n_total if n_total else float("nan")
    pos_after = (n_success - n_sub) / n_total if n_total else float("nan")
    return ColumnCheck(
        **out, min_column_height=float(min_column_height), n_sub_minimum=n_sub,
        pos_before=pos_before, pos_after=pos_after,
    )


def apply_min_column_height(
    ts: TrialSet, ad: AreaDepth | None, apex: float, min_column_height: float
) -> ThresholdMapping:
    """Map a minimum column height onto its area and volume equivalents."""
    res = ts.col("resource")
    contact = ts.col("contact")
    succ = res > 0.0
    z_min = apex + float(min_column_height)

    kept = succ & (contact >= z_min)
    n_excluded = int((succ & ~kept).sum())
    frac = n_excluded / max(int(succ.sum()), 1)

    min_area = float(ad.area_at(z_min)) if ad is not None else None
    if kept.any():
        v = res[kept]
        equiv_vol = float(v.min())
        equiv_pct = float((res[succ] >= equiv_vol).mean())
    else:
        equiv_vol = equiv_pct = None

    binds = n_excluded > 0
    if not binds:
        message = (
            f"A minimum column height of {min_column_height:g} m (contact at {z_min:.1f} m) "
            f"excludes nothing — the simulator's own assessment minimum is already stricter. "
            f"The shallowest sampled contact is {float(contact[succ].min()):.1f} m."
        )
    else:
        message = (
            f"A minimum column height of {min_column_height:g} m puts the shallowest admissible "
            f"contact at {z_min:.1f} m, excluding {n_excluded:,} of {int(succ.sum()):,} success "
            f"trials ({frac:.2%})."
        )
    return ThresholdMapping(
        min_column_height=float(min_column_height), apex=float(apex),
        min_contact_depth=z_min, min_area=min_area,
        equivalent_volume=equiv_vol, equivalent_percentile=equiv_pct,
        n_excluded=n_excluded, frac_excluded=frac, binds=binds, message=message,
    )


def volume_percentile_threshold(ts: TrialSet, percentile: float = 0.995) -> float:
    """The source workbook's alternative: a volume floor at P99.5 of the success case."""
    res = ts.col("resource")
    v = res[res > 0.0]
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, (1.0 - percentile) * 100.0))


def compare_definitions(ts: TrialSet, apex: float, min_column_height: float) -> dict:
    """Quantify how far the column-height and volume-percentile cuts disagree.

    They are two different operations on the same intuition. This returns the
    overlap so the user can see whether, for *their* prospect, treating them as
    interchangeable is safe.
    """
    res = ts.col("resource")
    contact = ts.col("contact")
    succ = res > 0.0
    z_min = apex + float(min_column_height)

    by_depth = succ & (contact >= z_min)
    n_keep = int(by_depth.sum())
    if n_keep == 0 or succ.sum() == 0:
        return {"comparable": False}

    # the volume cut that keeps the same number of trials
    thr = float(np.sort(res[succ])[max(int(succ.sum()) - n_keep, 0)])
    by_volume = succ & (res >= thr)
    agree = int((by_depth & by_volume).sum())
    excluded = max(n_keep, 1)
    return {
        "comparable": True,
        "n_kept_by_depth": n_keep,
        "n_kept_by_volume": int(by_volume.sum()),
        "n_agree": agree,
        "disagreement_frac": 1.0 - agree / excluded,
        "volume_threshold": thr,
    }


def spread_at_fixed_column(ts: TrialSet, apex: float, column_height: float, band_m: float = 20.0) -> dict:
    """Resource spread among trials at (approximately) one column height.

    If this ratio is far from 1, a column-height cut and a volume cut cannot be
    the same operation, because area is fixed while pay and yield still vary.
    """
    res, contact = ts.col("resource"), ts.col("contact")
    z = apex + column_height
    m = (res > 0) & (contact >= z) & (contact < z + band_m)
    if m.sum() < 5:
        return {"n": int(m.sum()), "ratio": float("nan")}
    v = res[m]
    return {
        "n": int(m.sum()),
        "min": float(v.min()),
        "max": float(v.max()),
        "ratio": float(v.max() / v.min()),
    }
