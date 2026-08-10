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

from dataclasses import dataclass

import numpy as np

from ..io.adapters.base import TrialSet
from .chance import ReferenceContour
from .chance import p_well as _p_well
from .classes import split_trials
from .groups import group_trials
from .structure import AreaDepth


def _spread(values: np.ndarray) -> float:
    """P10-P90 range in the petroleum convention (P90 low, P10 high)."""
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 90.0) - np.percentile(values, 10.0))


@dataclass
class Sweep:
    """One evaluation of both engines at every depth in ``z``."""

    z: np.ndarray
    r_location: np.ndarray
    p_well: np.ndarray
    uncertainty_reduction: np.ndarray  # % of the prospect's own P10-P90 spread
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
    Haskett curve, by contrast, is computed from the trial masks directly: it
    is a statement about how much the *trial set's* spread collapses on each
    outcome, which the entered POS does not change.
    """
    res = ts.col("resource")
    contact = ts.col("contact")
    success = res > 0.0
    if not success.any():
        raise ValueError("no successful trials to sweep")

    pad_lo = z_min is None
    lo = float(contact[success].min()) if z_min is None else float(z_min)
    hi = float(contact[success].max()) if z_max is None else float(z_max)
    pad = 0.03 * (hi - lo) if pad_lo and hi > lo else 0.0
    z = np.linspace(lo - pad, hi, max(int(n), 1))

    prospect_spread = _spread(res)

    r = np.empty(z.size)
    pw = np.empty(z.size)
    reduction_pct = np.empty(z.size)
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

        p_disc = float(discovery.mean())
        disc_spread = _spread(res[discovery])
        no_disc_spread = _spread(res[~discovery])
        expected_post = p_disc * disc_spread + (1.0 - p_disc) * no_disc_spread
        reduction_pct[i] = (
            100.0 * (prospect_spread - expected_post) / prospect_spread
            if prospect_spread > 0 else float("nan")
        )

    finite = np.isfinite(reduction_pct)
    i_opt = int(np.nanargmax(reduction_pct)) if finite.any() else 0

    return Sweep(
        z=z, r_location=r, p_well=pw, uncertainty_reduction=reduction_pct,
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
    reference-engine sweep does not. Drives B1 (volume split vs location) and
    B2 (chance vs regret); ``p_proven_exceeds_mefs`` / ``p_attic_exceeds_mefs``
    are ``None`` unless a ``mefs`` threshold is supplied.
    """

    z: np.ndarray
    z_exit: np.ndarray
    z_gap: float
    p_well: np.ndarray
    proven_mean: np.ndarray
    possible_mean: np.ndarray
    attic_mean: np.ndarray
    mefs: float | None
    p_proven_exceeds_mefs: np.ndarray | None
    p_attic_exceeds_mefs: np.ndarray | None


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
) -> VolumeSweep:
    """Sweep entry depth with a fixed entry-to-exit spacing, splitting each step.

    A lower ``n`` than :func:`run_sweep`'s default is deliberate: each step
    here also runs :func:`wellvolpos.core.classes.split_trials`, so this is
    the more expensive of the two sweeps.
    """
    res = ts.col("resource")
    contact = ts.col("contact")
    success = res > 0.0
    if not success.any():
        raise ValueError("no successful trials to sweep")

    pad_lo = z_min is None
    lo = float(contact[success].min()) if z_min is None else float(z_min)
    hi = float(contact[success].max()) if z_max is None else float(z_max)
    pad = 0.03 * (hi - lo) if pad_lo and hi > lo else 0.0
    z = np.linspace(lo - pad, hi, max(int(n), 1))
    # The exit is not clipped to the deepest sampled contact. Clipping bought
    # nothing numerically -- an exit at or below every sampled contact gives
    # LKH = contact either way -- and it broke two things: a sweep whose z_max
    # ran past the deepest contact produced an exit *shallower* than its own
    # entry, which split_trials rightly rejects, and B1's stated fixed gap
    # silently stopped being fixed at the deep end.
    z_exit = z + z_gap

    pw = np.empty(z.size)
    proven_mean = np.full(z.size, np.nan)
    possible_mean = np.full(z.size, np.nan)
    attic_mean = np.full(z.size, np.nan)
    p_proven_ex = np.full(z.size, np.nan) if mefs is not None else None
    p_attic_ex = np.full(z.size, np.nan) if mefs is not None else None

    for i, zi in enumerate(z):
        zx = float(z_exit[i])
        chance = _p_well(
            ts, float(zi), pos_prospect,
            reference=reference, reference_percentile=reference_percentile,
        )
        pw[i] = chance.p_well

        groups_i = group_trials(ts, float(zi), zx)
        vc = split_trials(ts, ad, groups_i, float(zi), zx)

        if groups_i.discovery.any():
            proven_mean[i] = float(vc.proven[groups_i.discovery].mean())
            possible_mean[i] = float(vc.possible[groups_i.discovery].mean())
            if mefs is not None:
                p_proven_ex[i] = float((vc.proven[groups_i.discovery] > mefs).mean())
        if groups_i.dry_with_attic.any():
            attic_mean[i] = float(vc.attic[groups_i.dry_with_attic].mean())
            if mefs is not None:
                p_attic_ex[i] = float((vc.attic[groups_i.dry_with_attic] > mefs).mean())

    return VolumeSweep(
        z=z, z_exit=z_exit, z_gap=float(z_gap), p_well=pw,
        proven_mean=proven_mean, possible_mean=possible_mean, attic_mean=attic_mean,
        mefs=mefs, p_proven_exceeds_mefs=p_proven_ex, p_attic_exceeds_mefs=p_attic_ex,
    )
