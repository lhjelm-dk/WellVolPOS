"""Per-trial volume classes: proven, possible, attic.

Schneider et al. (2023) name this decomposition but do not compute it:

    "the downdip EUR distribution represents the range of EUR from the prospect
     crest to the base of hydrocarbons in the well, plus the remaining EUR
     distribution from the base of the hydrocarbons in the well to the
     hydrocarbon water contact. Additional complexity -- for example,
     incorporating a range of column heights -- can be assessed using the
     techniques discussed in this paper."

This module is that additional complexity, implemented. It is an *extension* to
the reference grouping in :mod:`wellvolpos.core.groups`, not a correction of it.

Definitions, per the project's decision of record:

``proven``
    From the reservoir entry depth down to the lowest known hydrocarbon in the
    well, which is the *shallower* of the contact and the reservoir exit depth.

``possible``
    Whatever lies below the reservoir exit. Explicitly **not** proven -- a well
    that leaves the reservoir still in hydrocarbons has not established how far
    down they go.

``attic``
    The whole accumulation, in the trials where hydrocarbons are present but
    entirely up-dip of the well. This is the volume left behind by a dry hole,
    and the number that gets quoted when somebody argues for a sidetrack.

The split apportions a trial's resource by *area*, which assumes gross pay and
recovery yield are uniform across the closure within a trial. On the reference
dataset gross pay is independent of contact depth (r = -0.002), so the
assumption holds there. Where area and net pay are correlated it does not, and
:func:`check_area_pay_correlation` exists to say so out loud.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.adapters.base import TrialSet
from .groups import Groups
from .structure import AreaDepth


@dataclass
class VolumeClasses:
    proven: np.ndarray
    possible: np.ndarray
    attic: np.ndarray
    discovery_total: np.ndarray
    lkh: np.ndarray          # lowest known hydrocarbon, per trial (NaN if dry)


def split_trials(
    ts: TrialSet, ad: AreaDepth, groups: Groups, z_entry: float, z_exit: float
) -> VolumeClasses:
    if z_exit < z_entry:
        raise ValueError(
            f"reservoir exit ({z_exit} m) is shallower than entry ({z_entry} m); "
            "the well would leave the reservoir before entering it"
        )
    res = ts.col("resource")
    contact = ts.col("contact")

    disc = groups.discovery
    lkh = np.where(disc, np.minimum(contact, z_exit), np.nan)

    # Both areas come from the same fitted curve. Using the stored area for the
    # denominator instead would leave a residual-sized error, so a well that
    # logs the contact would report a sliver of "possible" volume below a depth
    # it demonstrably reached.
    a_lkh = np.where(disc, ad.area_at(np.where(disc, lkh, z_entry)), 0.0)
    a_contact = np.where(disc, ad.area_at(contact), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(a_contact > 0, a_lkh / a_contact, 0.0)
    frac = np.clip(np.nan_to_num(frac), 0.0, 1.0)

    proven = np.where(disc, res * frac, 0.0)
    possible = np.where(disc, res - proven, 0.0)
    attic = np.where(groups.dry_with_attic, res, 0.0)
    discovery_total = np.where(disc, res, 0.0)
    return VolumeClasses(proven, possible, attic, discovery_total, lkh)


def class_summary(vc: VolumeClasses, groups: Groups) -> dict[str, dict[str, float]]:
    """Percentiles and mean for each class, over the trials where it is defined."""
    def stat(v: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        x = v[mask]
        if x.size == 0:
            return {k: float("nan") for k in ("n", "p90", "p50", "mean", "p10")}
        return {
            "n": float(x.size),
            "p90": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)),
            "mean": float(x.mean()),
            "p10": float(np.percentile(x, 90)),
        }

    return {
        "discovery": stat(vc.discovery_total, groups.discovery),
        "proven": stat(vc.proven, groups.discovery),
        "possible": stat(vc.possible, groups.discovery),
        "attic_dry_hole": stat(vc.attic, groups.dry_with_attic),
    }


def check_area_pay_correlation(ts: TrialSet) -> tuple[str, str, float]:
    """Is the uniform-yield assumption behind the split defensible here?

    Returns (level, message, r). A strong positive area/net-pay correlation
    means the down-dip part of a closure carries thicker pay than the up-dip
    part, so apportioning resource by area alone understates the deep volume.
    Schneider et al. (2023) show this materially changes the conclusion.
    """
    if not (ts.has("area") and ts.has("gross_pay")):
        return "warn", "Cannot check area/net-pay correlation: area or gross pay not exported.", float("nan")
    a, g, res = ts.col("area"), ts.col("gross_pay"), ts.col("resource")
    m = (res > 0) & np.isfinite(a) & np.isfinite(g)
    if m.sum() < 30:
        return "warn", "Too few success trials to assess area/net-pay correlation.", float("nan")
    r = float(np.corrcoef(a[m], g[m])[0, 1])
    if abs(r) < 0.2:
        return "pass", f"Area and gross pay are effectively independent (r = {r:+.3f}); the uniform-yield split is sound.", r
    if abs(r) < 0.5:
        return "warn", f"Area and gross pay are moderately correlated (r = {r:+.3f}); the split understates the depth dependence of pay.", r
    return "fail", f"Area and gross pay are strongly correlated (r = {r:+.3f}); apportioning resource by area alone is not defensible. Use the reference grouping engine.", r
