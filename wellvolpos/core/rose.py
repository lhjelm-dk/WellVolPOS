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

from dataclasses import dataclass, field

import numpy as np

from ..io.adapters.base import TrialSet
from .groups import Groups
from .structure import AreaDepth


#: Default half-width for "the contact lands on the well". It reproduces the
#: original calculation exactly, which is why it is the default -- but it is a
#: tuning constant with a result attached, so the app exposes it rather than
#: burying it (Lars, 2026-08-12).
AT_WELL_WINDOW_M = 2.0


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
    ts: TrialSet, ad: AreaDepth, z_entry: float, *, window_m: float = AT_WELL_WINDOW_M
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
    #: How many *success* trials are both a discovery here and above the threshold,
    #: and how many success trials there are. See :meth:`pc_interval`.
    n_commercial: int = 0
    n_success: int = 0
    #: The POS in force, so ``pc_interval`` can scale the proportion back up.
    pos_prospect: float = 1.0

    def pc_interval(self, confidence: float = 0.90) -> tuple[float, float]:
        """Sampling interval on ``pc_well``, from the **one** proportion it reduces to.

        The identity that makes this simple, and it is exact rather than an
        approximation::

            Pc = P_well x Pmcfs
               = POS x (n_discovery / n_success) x (n_commercial / n_discovery)
               = POS x (n_commercial / n_success)

        ``n_discovery`` cancels. So ``Pc`` is ``POS_prospect`` times a *single*
        binomial proportion -- the share of success trials that are both a discovery
        at this depth and above the threshold -- and one Wilson interval on that
        proportion, scaled by ``POS``, is the interval on ``Pc``. Building it from two
        intervals and multiplying them would be both harder and wrong, since the two
        factors are computed from overlapping trials and are not independent.

        **``POS_prospect`` is not in the interval**, deliberately. It comes from the
        chance table, which is a judgement rather than a sample, so it has no sampling
        error to report -- and the fan on 3.4 is where its uncertainty already lives.
        """
        from .stats import wilson_interval

        lo, hi = wilson_interval(self.n_commercial, self.n_success,
                                 confidence=confidence)
        return (float(self.pos_prospect) * lo, float(self.pos_prospect) * hi)

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
    # The counts behind ``pc_interval``. Taken here rather than in the property so the
    # dataclass stays a value object -- and taken from the *same masks* the chances
    # are, because a count that selects differently from the chance beside it is how
    # an identity that is nearly true gets quoted as true.
    success = np.asarray(res, dtype=float) > 0.0
    n_success = int(success.sum())
    n_commercial = int((np.asarray(disc, dtype=bool) & (np.asarray(res, dtype=float) > mcfs)).sum())
    # POS recovered from ``p_well`` and the sampled r_location, so the caller does not
    # have to pass a number it has already implied. Where there are no success trials
    # there is nothing to scale.
    r = (n / n_success) if n_success else float("nan")
    pos = (float(p_well) / r) if r else float("nan")
    if n == 0:
        return CommercialChance(float(mcfs), float(p_well), float("nan"), float("nan"),
                                float("nan"), 0, 0, n_success, 1.0)
    p_downdip = float((res[disc] > mcfs).mean())
    p_proven = float((np.asarray(proven)[disc] > mcfs).mean())
    return CommercialChance(
        mcfs=float(mcfs), p_well=float(p_well), p_mcfs_downdip=p_downdip,
        p_mcfs_proven=p_proven, pc_well=float(p_well) * p_downdip, n_discovery=n,
        n_commercial=n_commercial, n_success=n_success, pos_prospect=pos,
    )


# ------------------------------------------------- the volume *at* the well
@dataclass(frozen=True)
class RosePartition:
    """Schneider & Cook's updip / downdip split of the closure, at the well.

    They partition the accumulation **at the well location**, which is a different cut
    from this app's, and the two are easy to confuse because both use the word
    "below". Ours splits at the *penetrated interval* -- entry to exit -- because a
    well proves what it drills through; theirs splits at a single point, because their
    well is a map location rather than a trajectory.

    ==================  ============================================================
    Rose updip          crest -> well.  What a dry hole leaves behind. His
                        deterministic *"No Regrets"* volume is this, evaluated at the
                        means rather than per trial.
    Rose downdip        well -> contact.  The extra volume the well opens up by
                        being deeper.
    ==================  ============================================================

    ``updip + downdip`` is the whole accumulation given a discovery, so it equals this
    app's **well-associated** volume exactly. Measured on prospect B at entry 2205 m,
    exit 2255 m: 122.05 + 49.65 = 171.69 MMboe.

    The relation to our own classes, which is what stops the two vocabularies being
    mixed:

    * Rose updip is our **proven** *only when exit = entry*; with a real penetration
      our proven reaches deeper by the entry-to-exit slice (30.49 MMboe there).
    * Rose downdip is our **possible below exit** plus that same slice.

    Both are means over the discovery trials, since neither quantity exists for a dry
    hole.
    """

    updip: np.ndarray = field(repr=False)
    downdip: np.ndarray = field(repr=False)
    n_discovery: int
    z_entry: float

    @property
    def updip_mean(self) -> float:
        return float(self.updip.mean()) if self.updip.size else float("nan")

    @property
    def downdip_mean(self) -> float:
        return float(self.downdip.mean()) if self.downdip.size else float("nan")

    @property
    def total_mean(self) -> float:
        """Their sum -- which must equal the well-associated mean."""
        return self.updip_mean + self.downdip_mean


def rose_partition(
    ts: "TrialSet", ad: "AreaDepth", z_entry: float, *,
    thickness=None, apex: float | None = None,
) -> RosePartition:
    """Split each discovery trial at the well, Schneider & Cook's way.

    Implemented by running this app's own per-trial split with **exit = entry**, which
    is exactly what their partition is: with no penetrated interval, "proven" collapses
    to crest-to-well and "below_lkh" becomes well-to-contact. So the two vocabularies
    share one piece of arithmetic and cannot drift apart -- the wedge apportionment,
    the thickness inversion and the apex are all the same ones the rest of the app
    uses.
    """
    from .classes import split_trials
    from .groups import group_trials

    groups = group_trials(ts, float(z_entry), float(z_entry))
    vc = split_trials(ts, ad, groups, float(z_entry), float(z_entry),
                      thickness=thickness, apex=apex)
    disc = np.asarray(groups.discovery, dtype=bool)
    return RosePartition(
        updip=np.asarray(vc.proven, dtype=float)[disc],
        downdip=np.asarray(vc.below_lkh, dtype=float)[disc],
        n_discovery=int(disc.sum()),
        z_entry=float(z_entry),
    )


def at_the_well_volume(
    ts: TrialSet, z_entry: float, *, window_m: float = 2.0
) -> tuple[float, int]:
    """Mean resource of the trials whose contact sits **at** the well.

    The source workbook's ``Results!G8``, *"Entry depth asso. vol."*, and the third
    of the three things the source workbook had that this app did
    not. Returns ``(mean, n)``; the count matters because a narrow window on a small
    trial file can leave nothing to average.

    **What it is.** Not the discovery case and not the attic, but the boundary
    between them: the accumulation you get when the hydrocarbon-water contact lands
    on the well rather than above or below it. It therefore sits between the two --
    on prospect A, 11.67 MMboe against an attic mean of 9.09 and a discovery mean of
    16.52.

    **Why it is worth having.** It is a probabilistic version of Rose's
    deterministic *"No Regrets"* volume (his Figures 7 and 19), and arguably a better
    one: his is ``A(z_entry) x mean(net pay) x mean(yield)``, a single arithmetic
    product, while this is the mean of the trials that actually landed there and so
    carries the model's own correlations. The poster is candid that the
    deterministic version is *"an oversimplification"*, because *"there remains a
    chance the updip volume will exceed MCFS"* -- which is what 3.6's regret curve
    answers, and why this number is quoted beside it rather than instead of it.

    ``window_m`` is +-2 m to match the workbook. It is not a sensitive choice: on the
    reference file +-2, +-5 and +-10 m give 11.67, 11.67 and 11.75 MMboe, because the
    quantity varies slowly with depth. Widen it on a sparse file rather than accept a
    mean of a handful of trials.
    """
    res = np.asarray(ts.col("resource"), dtype=float)
    contact = np.asarray(ts.col("contact"), dtype=float)
    m = (res > 0.0) & (np.abs(contact - float(z_entry)) <= float(window_m))
    n = int(m.sum())
    return (float(res[m].mean()) if n else float("nan")), n


# --------------------------------------------- how far the two outcomes overlap
def outcome_overlap(vc, groups) -> dict[str, float]:
    """Put a number on Schneider's *"surprising overlap"*.

    A6 draws the overlap between what a dry hole leaves up-dip and what a discovery
    proves, and the whole point of the figure is that it is larger than anyone
    expects. It never said *how* large. Schneider et al. quote 68 % for their
    example (their Figure 16, "the 68% overlap ... P100 to P32 of the Downdip
    Distribution"); this returns the same family of statements for the loaded trials.

    Three numbers, because they answer three different questions:

    ``proven_below_max_attic``
        The share of discoveries whose proven volume is no larger than the *best*
        possible attic. Schneider's framing, and the largest of the three.
    ``attic_above_min_proven``
        The mirror statement, from the attic's side.
    ``p_attic_beats_proven``
        **The decision-relevant one**: draw one dry outcome and one discovery
        independently, and this is the chance the volume left behind is bigger than
        the volume that would have been proved. On the demo files it is 6.8 % and
        8.3 % -- small, but not negligible, and it is the number that makes the
        overlap concrete rather than visual.

    The third is computed exactly rather than sampled: for each attic value, count
    the proven values below it. That is O(n log n) by sorting, and exact beats a
    Monte Carlo estimate of a quantity we already have every sample of.
    """
    attic = np.asarray(vc.attic[groups.dry_with_attic], dtype=float)
    proven = np.asarray(vc.proven[groups.discovery], dtype=float)
    out = {"n_attic": float(attic.size), "n_proven": float(proven.size)}
    if not attic.size or not proven.size:
        return {**out, "proven_below_max_attic": float("nan"),
                "attic_above_min_proven": float("nan"),
                "p_attic_beats_proven": float("nan")}
    out["proven_below_max_attic"] = float((proven <= attic.max()).mean())
    out["attic_above_min_proven"] = float((attic >= proven.min()).mean())
    # P(attic > proven) over independent draws, exactly: for every attic value, how
    # many proven values fall strictly below it.
    ordered = np.sort(proven)
    below = np.searchsorted(ordered, attic, side="left")
    out["p_attic_beats_proven"] = float(below.mean() / ordered.size)
    return out
