"""Three named quantities from the Rose & Associates Pwell poster.

Schneider, M. & Cook, D. (2017, long form May 2021), *Drilling a Downdip
Location: Effect on Updip and Downdip Resource Estimates and Commercial Chance*,
AAPG Search & Discovery 42102. The PDF is in the project folder and was read
directly rather than recalled.

These sit apart from :mod:`wellvolpos.core.chance` and
:mod:`wellvolpos.core.classes` because their definitions are *the poster's*, not
this tool's, and the difference matters in two places where they do not agree
with what the app computes by default. Both differences are stated here rather
than smoothed over.

**Terminology.** The poster says MCFS, minimum commercial field size; the app's
sidebar says MEFS, minimum economic field size. They are the same threshold under
two names. Distinct from both is the *assessment minimum* — the minimum column
height of :mod:`wellvolpos.core.threshold` — which is the smallest accumulation
worth carrying in the assessment at all, not the smallest worth developing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.adapters.base import TrialSet
from .groups import Groups
from .structure import AreaDepth


@dataclass
class NoRegrets:
    """Rose's "No Regrets" volume, and the workbook's probabilistic answer to it.

    ``deterministic`` is the poster's definition (p. 10): *"the area associated
    with the well location multiplied by the mean average net pay and mean oil
    recovery yield"*. One number, no distribution.

    ``at_entry_mean`` is the source workbook's `Results!G8`, "Entry depth asso.
    vol." — the mean resource of the trials whose contact sits within
    ``window_m`` of the entry depth. Same idea, but it lets pay and yield vary
    as the model sampled them instead of collapsing both to their means. On the
    reference data these come out close, and where they diverge the probabilistic
    one is the better answer.

    Neither is a substitute for the regret *curve*. The poster is explicit that
    No Regrets is *"useful to the decision-maker, however, it is an
    oversimplification"*, because *"for most downdip well locations, there
    remains a chance the updip volume will exceed MCFS"*. That chance is what
    B2 draws.
    """

    deterministic: float
    at_entry_mean: float
    at_entry_n: int
    mean_pay: float
    mean_yield: float
    area_at_entry: float
    window_m: float

    def message(self) -> str:
        return (
            f"No Regrets volume at this location: **{self.deterministic:.2f} MMboe** "
            f"(Rose: {self.area_at_entry:.2f} km² × {self.mean_pay:.1f} m mean pay × "
            f"{self.mean_yield:.3f} MMboe/km²·m mean yield). The trials' own answer, "
            f"averaging the {self.at_entry_n:,} realisations whose contact falls within "
            f"±{self.window_m:.0f} m of the entry, is {self.at_entry_mean:.2f} MMboe."
        )


def no_regrets(
    ts: TrialSet, ad: AreaDepth, z_entry: float, *, window_m: float = 2.0
) -> NoRegrets:
    """Rose's No Regrets volume, plus the workbook's probabilistic counterpart.

    Yield is recovered as resource per unit HC-bearing gross rock volume, which
    is what makes the poster's "recovery yield" computable from a GeoX export
    without being told it separately.
    """
    res = ts.col("resource")
    contact = ts.col("contact")
    success = res > 0.0
    if not success.any():
        raise ValueError("no successful trials")

    pay = ts.col("gross_pay")[success] if ts.has("gross_pay") else np.array([])
    if ts.has("hc_grv"):
        grv = ts.col("hc_grv")[success]
    elif ts.has("area") and ts.has("gross_pay"):
        grv = ts.col("area")[success] * pay
    else:
        raise ValueError("need HC gross rock volume, or area and gross pay, for the yield")
    ok = grv > 0
    mean_yield = float(np.mean(res[success][ok] / grv[ok])) if ok.any() else float("nan")
    mean_pay = float(np.mean(pay)) if pay.size else float("nan")
    area_entry = float(ad.area_at(z_entry))

    near = success & (np.abs(contact - z_entry) <= window_m)
    return NoRegrets(
        deterministic=area_entry * mean_pay * mean_yield,
        at_entry_mean=float(res[near].mean()) if near.any() else float("nan"),
        at_entry_n=int(near.sum()),
        mean_pay=mean_pay, mean_yield=mean_yield, area_at_entry=area_entry,
        window_m=float(window_m),
    )


@dataclass
class CommercialChance:
    """Rose's `Pmcfs(well)` and `Pc(well)` — Equation 2 and Table 2.

    ``p_mcfs_downdip`` is the poster's quantity: the chance a discovery's
    **whole** EUR exceeds the threshold, conditional on the discovery. That is
    *not* what B2 draws — B2 conditions on the entry-to-exit *proven* split
    instead, which is a different and smaller number. Both are legitimate; only
    one is Rose's, and the two are carried side by side so neither gets quoted
    as the other.

    ``pc_well = p_well × p_mcfs_downdip`` is the chance of a commercial
    discovery at this location, and the number the poster says to use in an EMV
    calculation. It is a *chance*, not an economic value, so it is inside this
    tool's scope even though economics is not.
    """

    mcfs: float
    p_well: float
    p_mcfs_downdip: float
    p_mcfs_proven: float
    pc_well: float
    n_discovery: int

    def message(self) -> str:
        return (
            f"At MCFS {self.mcfs:.1f} MMboe: a discovery exceeds it with probability "
            f"**{self.p_mcfs_downdip:.1%}** on the whole well-associated volume (Rose's "
            f"Pmcfs(well)), or {self.p_mcfs_proven:.1%} on the proven volume alone. "
            f"Commercial chance at this location, Pc(well) = P_well × Pmcfs(well) = "
            f"**{self.pc_well:.1%}**."
        )


def commercial_chance(
    ts: TrialSet, groups: Groups, proven: np.ndarray, p_well: float, mcfs: float
) -> CommercialChance:
    """Rose Equation 2, with our own proven-based variant beside it."""
    res = ts.col("resource")
    disc = groups.discovery
    n = int(disc.sum())
    if n == 0:
        return CommercialChance(float(mcfs), float(p_well), float("nan"), float("nan"),
                                float("nan"), 0)
    p_downdip = float((res[disc] > mcfs).mean())
    p_proven = float((np.asarray(proven)[disc] > mcfs).mean())
    return CommercialChance(
        mcfs=float(mcfs), p_well=float(p_well), p_mcfs_downdip=p_downdip,
        p_mcfs_proven=p_proven, pc_well=float(p_well) * p_downdip, n_discovery=n,
    )
