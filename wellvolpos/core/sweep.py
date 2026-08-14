"""Location sweep -- evaluate both engines across every possible entry depth.

Everything downstream of the two engines already answers "what happens at
this well location": :func:`wellvolpos.core.chance.p_well` gives the chance,
:func:`wellvolpos.core.groups.group_trials` gives the outcome split. This
module is just that, evaluated many times over a depth grid -- plus one more
figure, the Haskett (2003, SPE 84241) uncertainty-reduction curve, that turns
"what happens at this location" into "which location is best to test".

Haskett frames appraisal placement as a value-of-information problem: pick
the location whose drilling outcome (discovery vs not) most reduces the
uncertainty about ultimate recoverable resource, weighted by the chance of
each outcome. That is built here directly from the two engines already in
the app -- no separate statistical machinery::

    reduction(z) = spread(prospect)
                 - [ P(discovery|z) * spread(resource | discovery, z)
                     + P(no discovery|z) * spread(resource | no discovery, z) ]

``spread`` is the same P10-P90 range already used throughout the tool (the
petroleum convention: P90 is the *low* value, P10 the *high* one -- see
:func:`wellvolpos.core.groups.group_summary`), reported as a percentage of
the prospect's own spread so the curve sits on a fixed 0-100 % scale
regardless of prospect size. "No discovery" folds chance failures in with
dry-with-attic trials, because both look identical from the well bore:
nothing found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.adapters.base import TrialSet
from .chance import ReferenceContour
from .chance import p_well as _p_well
from .classes import split_trials
from .groups import group_trials
from .reservoir import thickness_from_pay
from .stats import MIN_SUPPORT, bootstrap_mean_ci, thin
from .structure import AreaDepth


#: The extra inter-percentile ranges 3.3 draws beside Haskett's P10-P90, narrowest
#: first. P1-P99 leans on the very tails, so on a thin conditional group it is the
#: first to become noise -- which is itself worth being able to see.
REDUCTION_RANGES = ((20.0, 80.0), (5.0, 95.0), (1.0, 99.0))


def _spread(values: np.ndarray, lo: float = 10.0, hi: float = 90.0) -> float:
    """Inter-percentile range, defaulting to P10-P90 in the petroleum convention.

    ``lo``/``hi`` are ordinary percentiles, so the default 10/90 is the petroleum
    P90-P10. A narrower pair (20/80) is offered on 3.3 because the answer should not
    depend much on where the range is cut -- and if it does, that is worth knowing
    rather than hiding behind one choice.
    """
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, hi) - np.percentile(values, lo))


def _sweep_grid(
    ts: TrialSet, z_min: float | None, z_max: float | None, n: int
) -> np.ndarray:
    """The depth grid both sweeps run on, so they cannot disagree about it.

    Defaults run from just above the shallowest successful contact -- so the
    chance curves visibly saturate towards 1 rather than starting mid-rise --
    down to the deepest. Explicit bounds are taken literally and never padded,
    which is what lets a caller zoom a sweep to exactly the interval the sliders
    are showing.
    """
    res, contact = ts.col("resource"), ts.col("contact")
    success = res > 0.0
    if not success.any():
        raise ValueError("no successful trials to sweep")
    pad_lo = z_min is None
    lo = float(contact[success].min()) if z_min is None else float(z_min)
    hi = float(contact[success].max()) if z_max is None else float(z_max)
    pad = 0.03 * (hi - lo) if pad_lo and hi > lo else 0.0
    return np.linspace(lo - pad, hi, max(int(n), 1))


@dataclass
class Sweep:
    """One evaluation of both engines at every depth in ``z``."""

    z: np.ndarray
    r_location: np.ndarray
    p_well: np.ndarray
    uncertainty_reduction: np.ndarray
    pos_prospect: float
    reference: ReferenceContour
    z_optimum: float
    reduction_optimum: float
    # Outcome tree vs location (drives A2). The four shares partition every
    # trial at every depth, and they are risked onto ``pos_prospect`` rather
    # than onto the trial file's own zero count -- see run_sweep. A hypothetical
    # exit at z + z_gap is used only to split the discovery mass into
    # contact-seen / HC-to-exit; it does not touch discovery itself
    # (group_trials' discovery mask ignores z_exit), so it cannot change
    # r_location, p_well or the Haskett optimum above.
    z_gap: float
    share_chance_failure: float
    share_dry_with_attic: np.ndarray
    share_contact_seen: np.ndarray
    share_hc_to_exit: np.ndarray

    #: The same curve over **every** trial rather than the success cases -- what
    #: 3.3 reported before 2026-08-12. Carried so the figure can draw both and show
    #: what the chance failures were contributing; identical to
    #: ``uncertainty_reduction`` on a file that has none. Defaulted, so it goes at
    #: the **end** of the dataclass -- a defaulted field inserted mid-class is a
    #: TypeError, and this is the second time that trap has been sprung here.
    uncertainty_reduction_all: np.ndarray | None = field(default=None, repr=False)
    #: The same measure on other inter-percentile ranges, keyed by ``(lo, hi)`` -- see
    #: :data:`REDUCTION_RANGES`. Haskett's P10-P90 is a convention rather than a
    #: result, so these are how much of the answer rests on it: peaking together means
    #: the recommendation is robust, peaking apart means the tails are carrying it.
    uncertainty_reduction_ranges: dict | None = field(default=None, repr=False)

def run_sweep(
    ts: TrialSet,
    pos_prospect: float,
    *,
    reference: ReferenceContour = ReferenceContour.CREST,
    reference_percentile: float = 0.90,
    z_min: float | None = None,
    z_max: float | None = None,
    n: int = 200,
    z_gap: float = 50.0,
) -> Sweep:
    """Sweep the reservoir entry depth and evaluate both engines at each step.

    The default range runs from just above the shallowest successful contact
    (so the chance curves visibly saturate towards 1) to the deepest one.
    Pass explicit ``z_min`` / ``z_max`` -- e.g. the current entry/exit slider
    bounds -- to zoom the sweep to where the user is actually looking; no
    padding is added when both are given explicitly.

    ``z_gap`` is a hypothetical entry-to-exit spacing used only to split the
    discovery group into contact-seen / HC-to-exit for the A2 outcome tree.
    It cannot affect ``r_location``, ``p_well`` or the Haskett optimum: the
    discovery mask in :func:`wellvolpos.core.groups.group_trials` depends on
    ``z_entry`` alone.

    The A2 outcome shares are risked onto ``pos_prospect``, so they always
    partition to 1.0 and their discovery mass always equals ``p_well``. The
    uncertainty-reduction curve, by contrast, is computed from the trial masks
    directly: it is a statement about how much the *trial set's* spread collapses on
    each outcome, which the entered POS does not change.

    That curve is taken over the **success cases only**, for the same reason
    ``r_location`` is: a chance failure is a property of the prospect rather than of
    where the well goes. Over every trial, a file with 23.9 % failures has a parent
    P90 of exactly zero, and the curve then mostly measures "we learned it was not a
    chance failure" -- which any depth tells you equally. On the reference file that
    moved the optimum 92 m up-dip; conditioned, it lands on the median contact.
    """
    res = ts.col("resource")
    z = _sweep_grid(ts, z_min, z_max, n)
    # **The uncertainty-reduction curve is computed on the success cases only.**
    # Taken over every trial, a file with chance failures has a parent P90 of exactly
    # zero, which inflates the parent range (13.46 -> 19.42 MMboe on the reference
    # file) and makes the curve mostly measure "we learned it was not a chance
    # failure" -- something a well at any depth tells you equally, so not a location
    # signal. It moved the reported optimum 92 m toward the crest, and the reduction
    # from 26.4 % to 50.4 %.
    #
    # Conditioning here is the same decision `r_location` already makes: a chance
    # failure is a property of the prospect, not of where the well goes, and it is
    # carried by POS_prospect. A file with no zero-volume trials is unaffected.
    learn = res > 0.0
    res_learn = res[learn]
    prospect_spread = _spread(res_learn)
    # The unconditional reading is computed alongside, not instead: 3.3 draws both so
    # the difference between them can be seen rather than taken on trust (Lars,
    # 2026-08-12). On a file with no chance failures the two coincide exactly.
    prospect_spread_all = _spread(res)
    # The same measure on a narrower range (Lars, 2026-08-13). Haskett's choice of
    # P10-P90 is a convention, not a result, so drawing P20-P80 beside it shows how
    # much of the answer depends on that choice.
    #: Percentile pairs the reduction curve is computed at, widest last. P10-P90 is
    #: Haskett's; the rest are here because his choice is a convention and the answer
    #: should not depend much on it. Where they peak together the recommendation is
    #: robust; where they do not, the tails are doing the work and it is fragile.
    prospect_spread_extra = {
        pair: _spread(res_learn, *pair) for pair in REDUCTION_RANGES
    }

    r = np.empty(z.size)
    pw = np.empty(z.size)
    reduction_pct = np.empty(z.size)
    reduction_pct_all = np.empty(z.size)
    reduction_extra = {pair: np.empty(z.size) for pair in REDUCTION_RANGES}
    share_dry = np.empty(z.size)
    share_seen = np.empty(z.size)
    share_past = np.empty(z.size)
    for i, zi in enumerate(z):
        chance = _p_well(
            ts, float(zi), pos_prospect,
            reference=reference, reference_percentile=reference_percentile,
        )
        r[i] = chance.r_location
        pw[i] = chance.p_well

        groups_i = group_trials(ts, float(zi), float(zi) + z_gap)
        discovery = groups_i.discovery

        # The outcome tree is risked onto the *entered* POS via risked_shares,
        # not onto the trial file's own zero count -- see Groups.risked_shares
        # for why: the two coincide only when pos_prospect is POS_trials, and
        # under any real chance table they do not.
        risked = groups_i.risked_shares(pos_prospect, pw[i])
        share_seen[i] = risked["contact_seen"]
        share_past[i] = risked["hc_to_exit"]
        share_dry[i] = risked["dry_with_attic"]

        # Both branches over the success cases, so the split is "is the contact
        # deeper than the well" and nothing else. `discovery` is already a subset of
        # the success cases; `~discovery & learn` is the dry-but-charged group.
        disc_learn = discovery[learn]
        p_disc = float(disc_learn.mean()) if disc_learn.size else 0.0
        disc_spread = _spread(res_learn[disc_learn])
        no_disc_spread = _spread(res_learn[~disc_learn])
        expected_post = p_disc * disc_spread + (1.0 - p_disc) * no_disc_spread
        for pair in REDUCTION_RANGES:
            parent = prospect_spread_extra[pair]
            post = (p_disc * _spread(res_learn[disc_learn], *pair)
                    + (1.0 - p_disc) * _spread(res_learn[~disc_learn], *pair))
            reduction_extra[pair][i] = (
                100.0 * (parent - post) / parent if parent > 0 else float("nan")
            )
        p_all = float(discovery.mean())
        expected_post_all = (p_all * _spread(res[discovery])
                             + (1.0 - p_all) * _spread(res[~discovery]))
        reduction_pct_all[i] = (
            100.0 * (prospect_spread_all - expected_post_all) / prospect_spread_all
            if prospect_spread_all > 0 else float("nan")
        )
        reduction_pct[i] = (
            100.0 * (prospect_spread - expected_post) / prospect_spread
            if prospect_spread > 0 else float("nan")
        )

    finite = np.isfinite(reduction_pct)
    i_opt = int(np.nanargmax(reduction_pct)) if finite.any() else 0

    return Sweep(
        z=z, r_location=r, p_well=pw, uncertainty_reduction=reduction_pct,
        uncertainty_reduction_all=reduction_pct_all,
        uncertainty_reduction_ranges=dict(reduction_extra),
        pos_prospect=float(pos_prospect), reference=reference,
        z_optimum=float(z[i_opt]), reduction_optimum=float(reduction_pct[i_opt]),
        z_gap=float(z_gap), share_chance_failure=1.0 - float(pos_prospect),
        share_dry_with_attic=share_dry, share_contact_seen=share_seen,
        share_hc_to_exit=share_past,
    )


@dataclass
class VolumeSweep:
    """The proven / possible / attic split, evaluated at every depth in ``z``.

    Companion to :class:`Sweep`, kept separate because it needs the recovered
    area-depth curve (:class:`wellvolpos.core.structure.AreaDepth`) that the
    reference-engine sweep does not. Drives B1 (volume split vs location),
    B2 (chance vs regret) and B6 (the inverse);
    ``p_proven_exceeds_mefs`` / ``p_attic_exceeds_mefs`` are ``None`` unless a
    ``mefs`` threshold is supplied.

    ``n_discovery`` / ``n_dry`` are the sample sizes each step's conditional
    statistics rest on, and they collapse down-dip: on the reference data the
    discovery group falls from 4 576 trials to 8. Carrying them means a figure
    can decline to draw what it cannot support, instead of every depth getting
    the same line width -- see :mod:`wellvolpos.core.stats`.

    ``pos_prospect`` and ``reference`` are carried for the same reason
    :class:`Sweep` carries them: the conventions that produced a curve must
    travel with it, or a figure cannot state which reference contour it used
    (non-negotiable 5).
    """

    z: np.ndarray
    z_exit: np.ndarray
    z_gap: float
    p_well: np.ndarray
    proven_mean: np.ndarray
    below_lkh_mean: np.ndarray
    attic_mean: np.ndarray
    mefs: float | None
    p_proven_exceeds_mefs: np.ndarray | None
    p_attic_exceeds_mefs: np.ndarray | None
    n_discovery: np.ndarray
    n_dry: np.ndarray
    pos_prospect: float
    reference: ReferenceContour
    # Bootstrap band on the proven mean, or None when n_boot was 0.
    proven_mean_lo: np.ndarray | None = None
    proven_mean_hi: np.ndarray | None = None
    #: Mean of the *whole* well-associated volume given a discovery -- Rose's
    #: "Downdip". Carried in its own right rather than left as proven + possible,
    #: because it is what his trade-off chart plots against ``p_well`` and what his
    #: ``Pmcfs(well)`` conditions on. Defaulted so the field could be added without
    #: breaking the constructor's positional order.
    discovery_mean: np.ndarray | None = None
    #: ``P(well-associated volume > MEFS | discovery)`` -- Rose's ``Pmcfs(well)``,
    #: which conditions on the whole downdip EUR rather than on the entry-to-exit
    #: proven split that ``p_proven_exceeds_mefs`` uses. Two different numbers, both
    #: legitimate; see :mod:`wellvolpos.core.rose`.
    p_discovery_exceeds_mefs: np.ndarray | None = None
    #: Conditional percentiles of the **proven** volume at each depth: the success
    #: case, given a discovery. P90 is the low case. Carried so B1 can show the
    #: spread around its mean and B9 can weight the spread by chance -- both asked
    #: for on 2026-08-11 -- without either figure re-deriving them and drifting.
    #: ``proven_p99`` and ``proven_p1`` were added 2026-08-11 for B9's grey family.
    #: They are the extremes of the same conditional distribution, and they are worth
    #: naming as such: on a right-skewed resource distribution P1 runs a long way
    #: above P10, so a figure that stops at P10 understates the upside it is drawn to
    #: show. All five come from one ``np.percentile`` call, so they cannot drift.
    proven_p99: np.ndarray | None = None
    proven_p90: np.ndarray | None = None
    proven_p50: np.ndarray | None = None
    proven_p10: np.ndarray | None = None
    proven_p1: np.ndarray | None = None
    alpha: float | None = None
    #: ``P(possible > MEFS | discovery)`` and ``P(possible > 0 | discovery)`` --
    #: added 2026-08-12 on Lars's question of whether the possible-below-exit
    #: probability could be shown against depth. It can, and the two readings answer
    #: different questions:
    #:
    #: * ``p_below_lkh_exceeds_mefs`` is the **material** one: given a discovery, the
    #:   chance the untested volume below the exit is on its own worth a threshold
    #:   field. It sits on B2 beside proven and attic, on the same conditioning and
    #:   the same threshold, so the three are comparable.
    #: * ``p_well_exits_in_hc`` is the **geometric** one: given a discovery, the
    #:   chance the well leaves the reservoir still in hydrocarbons at all, i.e. that
    #:   any possible volume exists. It is the fraction of discovery trials whose
    #:   contact is deeper than the exit, and it is what the exit depth controls
    #:   directly -- deepen the exit and it falls to zero.
    #:
    #: Both are conditional on a discovery, like the proven curve, so none of them
    #: may be read against ``p_well`` as if on one scale.
    p_below_lkh_exceeds_mefs: np.ndarray | None = None
    p_well_exits_in_hc: np.ndarray | None = None
    #: The **size of the upside when there is one**: mean possible volume over only the
    #: discovery trials whose contact is deeper than the exit (Lars, 2026-08-14).
    #:
    #: ``below_lkh_mean`` above is the mean over *every* discovery trial, and a
    #: discovery whose contact falls inside the penetrated interval contributes exactly
    #: zero -- the well saw the whole accumulation. Those zeros are 81 % of the
    #: discovery group on prospect A, so the unconditional mean reads as "the upside is
    #: tiny" when what it reports is the upside averaged over the cases that have none.
    #:
    #: The two are related exactly, and it is the same shape as ``P_well = POS x r``:
    #:
    #:     below_lkh_mean = p_well_exits_in_hc x below_lkh_mean_if_any
    #:
    #: Both are kept because each is wrong alone. The unconditional one is *additive*
    #: -- proven + possible = well associated -- which is what makes the volume classes
    #: a decomposition. The conditional one is the prize, and quoting it without
    #: ``p_well_exits_in_hc`` overstates that prize the way quoting a success-case
    #: volume without POS does.
    below_lkh_mean_if_any: np.ndarray | None = None
    #: P90 / P50 / P10 of that same conditional distribution -- the spread of the
    #: upside, not just its mean (Lars, 2026-08-14). Conditioned on the well leaving
    #: the reservoir in hydrocarbons, like the mean above, because a percentile of a
    #: population 41 % of which is an exact zero is not a percentile of anything.
    below_lkh_p90_if_any: np.ndarray | None = None
    below_lkh_p50_if_any: np.ndarray | None = None
    below_lkh_p10_if_any: np.ndarray | None = None
    #: The same ladder for the **attic** (over the dry-but-charged trials) and for the
    #: **at-the-well** volume (over the trials whose contact lands within the window).
    #: Added 2026-08-14 so that every volume on 3.5 carries its spread rather than only
    #: proven -- a bold mean with no percentiles beside it invites being read as the
    #: answer, and on a skewed distribution the mean is not even the middle.
    #: The **well-associated** ladder -- the whole accumulation given a discovery,
    #: crest to contact. Added 2026-08-14 at Lars's request so 3.8's frontier can be
    #: read as a range rather than a single mean line: a frontier drawn only through
    #: means says what an average discovery buys and nothing about whether a poor one
    #: still clears the bar, which is the question a location argument turns on.
    discovery_p90: np.ndarray | None = None
    discovery_p50: np.ndarray | None = None
    discovery_p10: np.ndarray | None = None
    attic_p90: np.ndarray | None = None
    attic_p50: np.ndarray | None = None
    attic_p10: np.ndarray | None = None
    at_well_p90: np.ndarray | None = None
    at_well_p50: np.ndarray | None = None
    at_well_p10: np.ndarray | None = None
    #: Mean resource of the trials whose contact sits **at** the entry, within
    #: ``at_well_window`` metres -- the workbook's ``Results!G8`` swept over depth,
    #: which is what its charts 5 and 16 plot and what Rose's Figures 7 and 19 show as
    #: the "No Regrets" curve. It is the boundary between the attic and the discovery
    #: case, so it runs between ``attic_mean`` and ``discovery_mean`` at every depth.
    at_well_mean: np.ndarray | None = None
    at_well_n: np.ndarray | None = None
    at_well_window: float = 2.0


def run_volume_sweep(
    ts: TrialSet,
    ad: AreaDepth,
    pos_prospect: float,
    *,
    z_gap: float = 50.0,
    mefs: float | None = None,
    reference: ReferenceContour = ReferenceContour.CREST,
    reference_percentile: float = 0.90,
    z_min: float | None = None,
    z_max: float | None = None,
    n: int = 60,
    n_boot: int = 0,
    at_well_window: float = 2.0,
    alpha: float = 0.10,
    seed: int | None = 0,
) -> VolumeSweep:
    """Sweep entry depth with a fixed entry-to-exit spacing, splitting each step.

    A lower ``n`` than :func:`run_sweep`'s default is deliberate: each step
    here also runs :func:`wellvolpos.core.classes.split_trials`, so this is
    the more expensive of the two sweeps.

    ``n_boot`` > 0 adds a percentile bootstrap band around the proven mean,
    resampled *within* each step's discovery group -- so the band widens by
    itself where the group is small, which is exactly where the curve deserves
    less trust. It is off by default because the band costs another resampling
    pass per step and B1 does not need it.
    """
    z = _sweep_grid(ts, z_min, z_max, n)
    # The exit is not clipped to the deepest sampled contact. Clipping bought
    # nothing numerically -- an exit at or below every sampled contact gives
    # LKH = contact either way -- and it broke two things: a sweep whose z_max
    # ran past the deepest contact produced an exit *shallower* than its own
    # entry, which split_trials rightly rejects, and B1's stated fixed gap
    # silently stopped being fixed at the deep end.
    z_exit = z + z_gap
    contact = np.asarray(ts.col("contact"), dtype=float)

    pw = np.empty(z.size)
    proven_mean = np.full(z.size, np.nan)
    below_lkh_mean = np.full(z.size, np.nan)
    below_lkh_if_any = np.full(z.size, np.nan)
    poss_p90 = np.full(z.size, np.nan)
    dis_p90 = np.full(z.size, np.nan)
    dis_p50 = np.full(z.size, np.nan)
    dis_p10 = np.full(z.size, np.nan)
    att_p90 = np.full(z.size, np.nan)
    att_p50 = np.full(z.size, np.nan)
    att_p10 = np.full(z.size, np.nan)
    atw_p90 = np.full(z.size, np.nan)
    atw_p50 = np.full(z.size, np.nan)
    atw_p10 = np.full(z.size, np.nan)
    poss_p50 = np.full(z.size, np.nan)
    poss_p10 = np.full(z.size, np.nan)
    attic_mean = np.full(z.size, np.nan)
    discovery_mean = np.full(z.size, np.nan)
    p99 = np.full(z.size, np.nan)
    p90 = np.full(z.size, np.nan)
    p50 = np.full(z.size, np.nan)
    p10 = np.full(z.size, np.nan)
    p1 = np.full(z.size, np.nan)
    n_disc = np.zeros(z.size, dtype=int)
    n_dry = np.zeros(z.size, dtype=int)
    p_proven_ex = np.full(z.size, np.nan) if mefs is not None else None
    p_attic_ex = np.full(z.size, np.nan) if mefs is not None else None
    p_below_lkh_ex = np.full(z.size, np.nan) if mefs is not None else None
    p_exits_hc = np.full(z.size, np.nan)
    at_well = np.full(z.size, np.nan)
    at_well_n = np.zeros(z.size, dtype=int)
    p_disc_ex = np.full(z.size, np.nan) if mefs is not None else None
    boot_lo = np.full(z.size, np.nan) if n_boot > 0 else None
    boot_hi = np.full(z.size, np.nan) if n_boot > 0 else None

    # Recovered **once**, outside the loop. Neither the reservoir thickness nor the
    # apex depends on where the well goes, and the thickness inversion loops over
    # every success trial -- redoing it at each of sixty depths would dominate the
    # sweep while returning the same array every time.
    _apex = float(ad.apex_estimate())
    try:
        _thickness = thickness_from_pay(ts, ad, apex=_apex).thickness
    except ValueError:
        _thickness = None          # no pay column; split_trials falls back to area

    for i, zi in enumerate(z):
        zx = float(z_exit[i])
        chance = _p_well(
            ts, float(zi), pos_prospect,
            reference=reference, reference_percentile=reference_percentile,
        )
        pw[i] = chance.p_well

        # The volume when the contact lands ON the well, not above or below it.
        m_at = (np.asarray(ts.col("resource"), dtype=float) > 0.0) & (
            np.abs(contact - float(zi)) <= float(at_well_window))
        at_well_n[i] = int(m_at.sum())
        if at_well_n[i]:
            _atw = np.asarray(ts.col("resource"), dtype=float)[m_at]
            at_well[i] = float(_atw.mean())
            # Petroleum orientation, one call so the three cannot drift: P90 is the
            # low case and therefore the 10th percentile of the values.
            atw_p90[i], atw_p50[i], atw_p10[i] = (
                float(v) for v in np.percentile(_atw, [10.0, 50.0, 90.0]))

        groups_i = group_trials(ts, float(zi), zx)
        vc = split_trials(ts, ad, groups_i, float(zi), zx,
                          thickness=_thickness, apex=_apex)
        n_disc[i] = int(groups_i.discovery.sum())
        n_dry[i] = int(groups_i.dry_with_attic.sum())

        if n_disc[i]:
            # Does the well leave the reservoir still in hydrocarbons? That is
            # exactly "is there any possible volume", and it is what the exit depth
            # controls: push the exit below the deepest contact and it is zero.
            p_exits_hc[i] = float((contact[groups_i.discovery] > zx).mean())
            proven = vc.proven[groups_i.discovery]
            associated = vc.discovery_total[groups_i.discovery]
            proven_mean[i] = float(proven.mean())
            _poss = vc.below_lkh[groups_i.discovery]
            below_lkh_mean[i] = float(_poss.mean())
            # **Conditioned on the same event the chance counts**: the well leaving
            # the reservoir still in hydrocarbons, `contact > z_exit`. Selecting on
            # `possible > 0` instead looked equivalent and was not -- the wedge
            # integral rounds a hair-thin interval to exactly zero, so the two masks
            # disagreed on a handful of trials and the identity
            # `below_lkh_mean = p_well_exits_in_hc x this` came out 1e-2 off instead of
            # exact. An identity that is nearly true is the kind that gets quoted as
            # true, so the masks are now one mask.
            _exits = contact[groups_i.discovery] > zx
            if _exits.any():
                _pv = _poss[_exits]
                below_lkh_if_any[i] = float(_pv.mean())
                # Petroleum orientation, one call so the three cannot drift: P90 is
                # the low case and therefore the 10th percentile of the values.
                poss_p90[i], poss_p50[i], poss_p10[i] = (
                    float(v) for v in np.percentile(_pv, [10.0, 50.0, 90.0])
                )
            discovery_mean[i] = float(associated.mean())
            # One call, petroleum orientation: P90 is the low case and therefore the
            # 10th percentile of the values. Same population as `discovery_mean`, so
            # the mean cannot end up outside its own ladder.
            dis_p90[i], dis_p50[i], dis_p10[i] = (
                float(v) for v in np.percentile(associated, [10.0, 50.0, 90.0])
            )
            # Petroleum orientation: P90 is the low case, so it is the 10th
            # percentile of the values.
            # One call, five outputs, petroleum orientation throughout: P99 is the
            # low case and therefore the 1st percentile of the values. Deriving them
            # separately is how two of them come to disagree.
            p99[i], p90[i], p50[i], p10[i], p1[i] = (
                float(v) for v in np.percentile(proven, [1, 10, 50, 90, 99])
            )
            if mefs is not None:
                p_proven_ex[i] = float((proven > mefs).mean())
                p_below_lkh_ex[i] = float((vc.below_lkh[groups_i.discovery] > mefs).mean())
                p_disc_ex[i] = float((associated > mefs).mean())
            if n_boot > 0:
                boot_lo[i], boot_hi[i] = bootstrap_mean_ci(
                    proven, n_boot=n_boot, alpha=alpha, seed=seed
                )
        if n_dry[i]:
            _att = vc.attic[groups_i.dry_with_attic]
            attic_mean[i] = float(_att.mean())
            att_p90[i], att_p50[i], att_p10[i] = (
                float(v) for v in np.percentile(_att, [10.0, 50.0, 90.0]))
            if mefs is not None:
                p_attic_ex[i] = float((vc.attic[groups_i.dry_with_attic] > mefs).mean())

    return VolumeSweep(
        z=z, z_exit=z_exit, z_gap=float(z_gap), p_well=pw,
        proven_mean=proven_mean, below_lkh_mean=below_lkh_mean, attic_mean=attic_mean,
        discovery_mean=discovery_mean,
        mefs=mefs, p_proven_exceeds_mefs=p_proven_ex, p_attic_exceeds_mefs=p_attic_ex,
        p_below_lkh_exceeds_mefs=p_below_lkh_ex, p_well_exits_in_hc=p_exits_hc,
        below_lkh_mean_if_any=below_lkh_if_any,
        below_lkh_p90_if_any=poss_p90, below_lkh_p50_if_any=poss_p50,
        below_lkh_p10_if_any=poss_p10,
        discovery_p90=dis_p90, discovery_p50=dis_p50, discovery_p10=dis_p10,
        attic_p90=att_p90, attic_p50=att_p50, attic_p10=att_p10,
        at_well_p90=atw_p90, at_well_p50=atw_p50, at_well_p10=atw_p10,
        at_well_mean=at_well, at_well_n=at_well_n, at_well_window=float(at_well_window),
        p_discovery_exceeds_mefs=p_disc_ex,
        proven_p99=p99, proven_p90=p90, proven_p50=p50, proven_p10=p10,
        proven_p1=p1,
        n_discovery=n_disc, n_dry=n_dry,
        pos_prospect=float(pos_prospect), reference=reference,
        proven_mean_lo=boot_lo, proven_mean_hi=boot_hi,
        alpha=float(alpha) if n_boot > 0 else None,
    )


# ------------------------------------------------------------------ inverse
@dataclass
class InverseResult:
    """B6: the depth a well must reach to prove a given volume, and its cost.

    ``achievable`` is False when no location on the swept structure proves the
    target -- asking for more than the closure holds is a legitimate question
    with "nowhere" as its legitimate answer, and returning the deepest depth
    instead would quietly answer a different one.

    ``z_lo`` / ``z_hi`` bracket the requirement using the bootstrap band on the
    proven mean: the optimistic edge of the band reaches the target shallower,
    the pessimistic edge deeper. They are ``None`` unless the sweep was run with
    ``n_boot`` > 0.
    """

    target: float
    achievable: bool
    z_required: float | None
    p_well_at: float | None
    z_lo: float | None = None
    z_hi: float | None = None
    n_discovery_at: int | None = None
    # False when the target is already met at the shallowest swept location, so
    # it imposes no requirement at all. Distinguished from achievable=True
    # because "anywhere on the structure does this" is a different answer from
    # "you must get down to 3500 m".
    binds: bool = True

    def message(self) -> str:
        if not self.achievable:
            return (
                f"No location on this structure proves {self.target:.2f} MMboe on the mean — "
                f"the deepest well-supported entry does not reach it."
            )
        if not self.binds:
            return (
                f"{self.target:.2f} MMboe is already proven at the shallowest swept location "
                f"({self.z_required:.0f} m TVDSS), so it imposes no constraint on the well."
            )
        band = ""
        if self.z_lo is not None and self.z_hi is not None and np.isfinite([self.z_lo, self.z_hi]).all():
            band = f" (band {self.z_lo:.0f}–{self.z_hi:.0f} m)"
        return (
            f"Proving {self.target:.2f} MMboe on the mean needs entry at "
            f"{self.z_required:.0f} m TVDSS{band}, where P_well is {self.p_well_at:.1%}."
        )


def _required_depth(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """Shallowest depth from which ``y`` never again falls below ``level``.

    Not the first touch of the level, which is the obvious reading and the wrong
    one. "Deeper proves more" is a geological monotonicity, but a *sampled*
    proven-mean curve violates it wherever the discovery group is small -- even
    after under-supported steps are dropped, the reference data still steps
    down by 0.45 MMboe in places. Inverting the first crossing then returns a
    depth that deeper locations contradict, which is no basis for a well
    proposal.

    Taking the last crossing of the running minimum from the deep end instead
    makes the answer a guarantee: at the depth returned, and at every deeper
    depth still supported by trials, the proven mean is at least ``level``.
    That is conservative -- it can sit a step deeper than the first touch -- and
    conservative is the right direction for a requirement.
    """
    ok = np.isfinite(y)
    if not ok.any():
        return None
    xs, ys = np.asarray(x, dtype=float)[ok], np.asarray(y, dtype=float)[ok]
    # Running minimum looking deeper: rev_min[i] = min(ys[i:])
    rev_min = np.minimum.accumulate(ys[::-1])[::-1]
    holds = rev_min >= level
    if not holds.any():
        return None
    j = int(np.argmax(holds))
    if j == 0:
        return float(xs[0])
    # Interpolate on the guarantee curve, so the reported depth is the point at
    # which the guarantee actually starts rather than the grid node after it.
    y0, y1 = rev_min[j - 1], rev_min[j]
    if y1 == y0:
        return float(xs[j])
    frac = (level - y0) / (y1 - y0)
    return float(xs[j - 1] + frac * (xs[j] - xs[j - 1]))


#: What "the volume to prove" may be measured by. ``mean`` is the default and the
#: only one the source workbook offers; the percentiles were added 2026-08-11 after
#: Lars asked whether P50/P10/P90 could be used instead.
#:
#: They answer materially different questions, and the difference is not a nuance:
#:
#: * **mean** -- the average proven volume over the discovery group. Additive across
#:   prospects, which is why portfolios run on it, but on a right-skewed resource
#:   distribution it sits above the median and is pulled by a tail of large cases.
#: * **P50** -- the median discovery. "Half the discoveries prove at least this."
#:   The one most people picture when they say "a typical outcome".
#: * **P90** -- the low case. Requiring a depth at which even a *poor* discovery
#:   proves the target, so it demands a deeper well than the mean does.
#: * **P10** -- the high case. Satisfied by a shallow well, because it only asks
#:   that a *good* discovery would prove the target.
#:
#: So P90 is the conservative reading and P10 the optimistic one, and they can differ
#: by a hundred metres or more of required entry. Naming which is on screen is the
#: whole reason this is an explicit setting rather than a default buried in code
#: (non-negotiable 5).
TARGET_STATISTICS = ("mean", "p90", "p50", "p10")

#: Labels for the four, so the app and both backends cannot word them differently.
TARGET_STATISTIC_LABELS = {
    "mean": "mean proven",
    "p90": "P90 proven (low case)",
    "p50": "P50 proven (median discovery)",
    "p10": "P10 proven (high case)",
}


def _supported_proven(
    vsweep: VolumeSweep, min_support: int, statistic: str = "mean"
) -> np.ndarray:
    """The proven-volume curve to invert, with under-supported steps removed.

    The inverse reads the same curve the figures draw. Inverting the raw curve
    instead let B6 answer at 3688 m from a mean of two trials, in a region B1
    and B2 decline to draw at all -- and gave sampling noise at the deep end
    the appearance of the structure running out of volume.

    ``statistic`` selects which proven-volume curve is inverted; see
    :data:`TARGET_STATISTICS`. A sweep run before the percentiles were carried has
    them as ``None``, and asking for one then is an error rather than a silent
    fallback to the mean -- a fallback would answer a different question under the
    label the caller chose.
    """
    if statistic not in TARGET_STATISTICS:
        raise ValueError(
            f"unknown target statistic {statistic!r}; expected one of {TARGET_STATISTICS}"
        )
    curve = vsweep.proven_mean if statistic == "mean" else getattr(vsweep, f"proven_{statistic}")
    if curve is None:
        raise ValueError(
            f"this sweep carries no proven_{statistic} curve, so the inverse cannot be "
            f"taken on it; re-run run_volume_sweep"
        )
    return thin(curve, vsweep.n_discovery, min_support)


def invert_volume_target(
    vsweep: VolumeSweep,
    target: float,
    *,
    min_support: int = MIN_SUPPORT,
    ts: TrialSet | None = None,
    reference_percentile: float = 0.90,
    statistic: str = "mean",
) -> InverseResult:
    """Given a volume to prove, where must the well go and what does it cost?

    The fourth question in CLAUDE.md's list, and the source workbook's H38-H40
    block as a curve. Inverts the proven-mean-versus-depth relationship: deeper
    entry proves more, because a deeper reservoir entry means the well's lowest
    known hydrocarbon sits further down the area-depth curve -- and costs
    chance, because ``r_location`` falls monotonically with depth. Both halves
    of that trade are reported together; the depth alone would be half an answer.

    Pass ``ts`` to have ``P_well`` evaluated exactly at the required depth by
    :func:`wellvolpos.core.chance.p_well`, using the conventions the sweep
    already carries. Without it the value is interpolated off the sweep grid,
    which on a 60-step grid disagrees with tab 4's own figure by up to 0.2
    percentage points -- small, but a visible disagreement about one well.
    """
    curve = _supported_proven(vsweep, min_support, statistic)
    z_req = _required_depth(vsweep.z, curve, float(target))
    if z_req is None:
        return InverseResult(target=float(target), achievable=False, z_required=None, p_well_at=None)

    if ts is not None:
        p_at = _p_well(
            ts, z_req, vsweep.pos_prospect,
            reference=vsweep.reference, reference_percentile=reference_percentile,
        ).p_well
    else:
        p_at = float(np.interp(z_req, vsweep.z, vsweep.p_well))

    # The support behind the answer is the *deeper* bracketing step, not an
    # average across the bracket: that is the step whose thinness would make
    # the crossing untrustworthy.
    j = int(np.searchsorted(vsweep.z, z_req))
    n_at = int(vsweep.n_discovery[min(j, vsweep.n_discovery.size - 1)])

    # The optimistic edge of the band (hi) reaches the target shallower, so it
    # supplies the shallow end of the requirement.
    z_lo = z_hi = None
    if vsweep.proven_mean_hi is not None and vsweep.proven_mean_lo is not None:
        z_lo = _required_depth(
            vsweep.z, thin(vsweep.proven_mean_hi, vsweep.n_discovery, min_support), float(target)
        )
        z_hi = _required_depth(
            vsweep.z, thin(vsweep.proven_mean_lo, vsweep.n_discovery, min_support), float(target)
        )
    binds = z_req > float(vsweep.z[np.isfinite(curve)][0]) + 1e-9 if np.isfinite(curve).any() else True
    return InverseResult(
        target=float(target), achievable=True, z_required=z_req, p_well_at=float(p_at),
        z_lo=z_lo, z_hi=z_hi, n_discovery_at=n_at, binds=binds,
    )


def volume_target_curve(
    vsweep: VolumeSweep,
    targets: np.ndarray | None = None,
    n: int = 40,
    *,
    min_support: int = MIN_SUPPORT,
    ts: TrialSet | None = None,
    statistic: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The whole inverse as a curve: (targets, required depth, P_well there).

    B6 draws this rather than a single answer, because the interesting content
    is the *shape* of the trade -- how fast chance is given up per extra MMboe
    demanded, and where the curve turns vertical because the structure has run
    out of volume to prove. The default target range spans the *supported*
    proven means only, so the curve never offers a volume the tool would refuse
    to draw elsewhere.
    """
    curve = _supported_proven(vsweep, min_support, statistic)
    finite = curve[np.isfinite(curve)]
    if finite.size == 0:
        return np.array([]), np.array([]), np.array([])
    if targets is None:
        # The ceiling is the *deepest* supported proven mean, not the curve's
        # peak. Under the guarantee convention in _required_depth nothing above
        # that can be promised: if the curve peaks and then dips, the peak is a
        # volume some single location happens to prove, not one the well can be
        # relied on to prove. Generating targets up to the peak would put points
        # on B6 that the inverse then reports as unachievable.
        targets = np.linspace(float(finite.min()), float(finite[-1]), max(int(n), 2))
    targets = np.asarray(targets, dtype=float)

    z_req = np.full(targets.size, np.nan)
    p_at = np.full(targets.size, np.nan)
    for i, t in enumerate(targets):
        res = invert_volume_target(vsweep, float(t), min_support=min_support, ts=ts,
                                   statistic=statistic)
        if res.achievable:
            z_req[i] = res.z_required
            p_at[i] = res.p_well_at
    return targets, z_req, p_at


def volume_target_band(
    vsweep: VolumeSweep,
    targets: np.ndarray,
    *,
    min_support: int = MIN_SUPPORT,
) -> tuple[np.ndarray, np.ndarray]:
    """Required-depth band implied by the bootstrap band on the proven mean.

    Lives here rather than in either figure because it is arithmetic, and both
    backends must invert the band by exactly the same code path as the curve --
    two subtly different implementations of an inversion is how a matplotlib
    export and an on-screen plot come to disagree.
    """
    if vsweep.proven_mean_lo is None or vsweep.proven_mean_hi is None:
        empty = np.full(np.asarray(targets).size, np.nan)
        return empty, empty.copy()
    hi_curve = thin(vsweep.proven_mean_hi, vsweep.n_discovery, min_support)
    lo_curve = thin(vsweep.proven_mean_lo, vsweep.n_discovery, min_support)
    z_lo = np.full(np.asarray(targets).size, np.nan)
    z_hi = np.full(np.asarray(targets).size, np.nan)
    for i, t in enumerate(np.asarray(targets, dtype=float)):
        a = _required_depth(vsweep.z, hi_curve, float(t))
        b = _required_depth(vsweep.z, lo_curve, float(t))
        if a is not None:
            z_lo[i] = a
        if b is not None:
            z_hi[i] = b
    return z_lo, z_hi


def find_crossing(
    z: np.ndarray, a: np.ndarray, b: np.ndarray
) -> float | None:
    """Depth where curves ``a`` and ``b`` cross, or None if they never do.

    B2's crossings are the argument the figure exists to make -- the depth at
    which the chance of success stops outweighing the regret of leaving volume
    up-dip -- so they are found rather than eyeballed, the same way the Haskett
    optimum is an argmax rather than a reading off a chart.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return None
    zs, d = np.asarray(z)[ok], np.asarray(a)[ok] - np.asarray(b)[ok]
    sign_change = np.signbit(d[:-1]) != np.signbit(d[1:])
    if not sign_change.any():
        return None
    j = int(np.argmax(sign_change))
    d0, d1 = d[j], d[j + 1]
    if d1 == d0:
        return float(zs[j])
    return float(zs[j] - d0 * (zs[j + 1] - zs[j]) / (d1 - d0))


# ------------------------------------- entry-depth percentiles per volume
#: The percentiles the entry-depth band reports, matching the workbook's columns
#: ``BB`` (P99), ``BD`` (P90), ``BE`` (P50) and ``BC`` (P10).
#: P1 added 2026-08-11 (Lars). Without it the family stopped at P10 and the shaded
#: range therefore stopped there too, which understates the deep tail: a volume that
#: only the largest accumulations hold is consistent with contacts well below P10.
ENTRY_DEPTH_PERCENTILES = (99, 90, 50, 10, 1)


def entry_depth_percentiles(
    ts: TrialSet,
    targets: np.ndarray,
    *,
    percentiles: tuple[int, ...] = ENTRY_DEPTH_PERCENTILES,
    min_support: int = MIN_SUPPORT,
) -> dict[int, np.ndarray]:
    """For each target volume, the spread of contact depths that deliver it.

    The 2018 macro workbook's ``BB``-``BE`` block, which the app had no equivalent
    of: given a volume you want, at what depth do the trials that actually contain
    that much sit? Returns ``{percentile: depths}``, each array the length of
    ``targets``, NaN where too few trials qualify to say.

    **This is a different question from** :func:`invert_volume_target`, and the
    difference is worth stating because the two are easy to conflate:

    * The inverse asks "how deep must the well go for the *mean proven volume* to
      reach this target". One depth per target, from the proven-mean curve, and it
      answers a *guarantee*.
    * This asks "among the trials that hold at least this much, where is the
      contact". A *distribution* per target, from the trials directly, and it
      answers a *spread*.

    So the band here is the range of contact depths consistent with a volume, while
    B6's bootstrap band is sampling error on one estimate. Both are honest, they are
    not the same band, and neither should be drawn as the other.

    The percentiles are in the petroleum orientation: **P99 is the shallow end**,
    exceeded by 99 % of the qualifying contacts, and P10 the deep end. That keeps
    them consistent with every other percentile in this codebase, where P99 is the
    low case -- here "low" meaning up-dip.

    Rose's Figure 4 is the warning that makes this worth having as a spread rather
    than a mean: *"The EUR of 9.4 MMBO is associated with productive areas from 200
    to 1500 acres."* The workbook's own ``BA`` column averages those contacts into
    one number, and averaging over a sevenfold area range is not a required depth.
    """
    res = np.asarray(ts.col("resource"), dtype=float)
    contact = np.asarray(ts.col("contact"), dtype=float)
    success = res > 0.0
    out = {q: np.full(np.asarray(targets).size, np.nan) for q in percentiles}
    # The **mean** contact among the qualifying trials, under the key ``"mean"``
    # (Lars, 2026-08-12). Not a percentile, so it gets a string key rather than an
    # integer one -- which is deliberately ugly, because a mean quietly filed among
    # P99..P1 is exactly the kind of thing that later gets read as one.
    #
    # It is the number the workbook's own ``BA`` column computes, and Rose's Figure 4
    # is the argument against quoting it alone: "The EUR of 9.4 MMBO is associated
    # with productive areas from 200 to 1500 acres." Drawn *with* the spread it is
    # useful; drawn instead of it, it is the mistake.
    out["mean"] = np.full(np.asarray(targets).size, np.nan)
    for i, target in enumerate(np.asarray(targets, dtype=float)):
        qualifies = success & (res >= target)
        n = int(qualifies.sum())
        if n < min_support:
            continue
        depths = contact[qualifies]
        for q in percentiles:
            # P99 = shallow end, so it is the 1st percentile of the depths.
            out[q][i] = float(np.percentile(depths, 100 - q))
        out["mean"][i] = float(depths.mean())
    return out
