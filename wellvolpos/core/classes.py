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

    **Never worse than ``warn``, decided 2026-08-10.** It was a ``fail``, and any
    fail closes the analysis tabs -- so a strongly correlated export locked the
    whole app, including the *reference* engine, which apportions nothing and is
    entirely unaffected by this assumption. Blocking there hides a result that is
    still valid. The correct behaviour is to disqualify the extension loudly and
    let the reference engine be read, which is what the message says to do. The
    synthetic correlated file in :mod:`wellvolpos.io.synthetic` is what made this
    visible, and prospect B triggers it for real -- see below.

    **The message also reports the partial correlation controlling for contact
    depth**, because the two readings call for different words and prospect B is
    the case that showed it. There, raw r = +0.87 but partial r = +0.04: area and
    pay are correlated only because both grow with fill depth, since the reservoir
    is a deterministic 50 m and the pay is therefore a wedge. That still
    disqualifies apportioning by area -- the deep part of the closure genuinely
    holds more pay per unit area than the rim -- but it is a *geometric* fact about
    a wedge, not a rock-property correlation, and a reader deserves to be told
    which. On prospect A both are ~0.
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

    why = _explain_correlation(ts, m, r)
    if abs(r) < 0.5:
        return "warn", (
            f"Area and gross pay are moderately correlated (r = {r:+.3f}); the split understates "
            f"the depth dependence of pay. {why}"
        ), r
    return (
        "warn",
        f"Area and gross pay are strongly correlated (r = {r:+.3f}); apportioning resource by "
        f"area alone is not defensible on this data. **Read the reference grouping engine and "
        f"disregard the proven/possible split** — the reference engine groups whole trials and "
        f"apportions nothing, so it does not rest on this assumption. {why}",
        r,
    )


def _explain_correlation(ts: TrialSet, m: np.ndarray, r: float) -> str:
    """Say whether the correlation survives controlling for contact depth.

    A one-line linear partialling-out, not a model: both quantities are regressed
    on the contact and the residuals correlated. Enough to separate "pay grows as
    the closure fills" from "thicker pay happens to come with bigger area", which
    are the two stories a reader might otherwise have to guess between.
    """
    if not ts.has("contact"):
        return ""
    a, g, c = (np.asarray(ts.col(k), dtype=float)[m] for k in ("area", "gross_pay", "contact"))
    if np.ptp(c) <= 0:
        return ""

    def residual(y: np.ndarray) -> np.ndarray:
        design = np.vstack([c, np.ones_like(c)]).T
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        return y - design @ coef

    ra, rg = residual(a), residual(g)
    if not (np.std(ra) > 0 and np.std(rg) > 0):
        return ""
    partial = float(np.corrcoef(ra, rg)[0, 1])
    if abs(partial) < 0.2:
        return (
            f"Controlling for contact depth it drops to r = {partial:+.3f}, so this is the "
            f"*wedge*: both grow as the closure fills, rather than thicker pay happening to come "
            f"with bigger area."
        )
    return (
        f"It survives controlling for contact depth (r = {partial:+.3f}), so pay and area are "
        f"related beyond their shared dependence on fill — a rock-property correlation, not just "
        f"wedge geometry."
    )


# --------------------------------------------------------- risked distributions
#: The percentiles the volume-class table reports, in petroleum orientation:
#: **P99 is the low case** (exceeded 99 % of the time) and P1 is the high case.
REPORT_PERCENTILES = (99, 90, 50, 10, 1)


def risked_exceedance(values: np.ndarray, chance: float):
    """The risked exceedance curve of a conditional distribution.

    Returns ``(sorted_values, percent)`` where ``percent`` is
    ``chance x P(X >= x | the outcome happens)``, in per cent.

    **Built from the chance, not by padding with zeros.** Zero-padding a
    distribution with the trial file's own non-occurrences -- ``np.where(discovery,
    res, 0)`` -- gives a curve that starts at the *file's* implied chance, which
    equals the entered chance only when the risking convention says the trials are
    already risked. On a success-case-only export with a chance table entered on
    top, the two differ by the whole of the chance table, and the figure then
    contradicts its own caption.

    That is the fourth time this codebase has drawn an unrisked number under a
    risked label (after A2, tab ④'s tree and B4), and it is why the arithmetic
    lives here instead of in the figure: a curve built this way *cannot* start
    anywhere but at ``chance``.
    """
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return v, np.array([])
    conditional = (n - np.arange(n)) / n
    return v, 100.0 * float(chance) * conditional


def class_percentiles(values: np.ndarray, chance: float,
                      percentiles=REPORT_PERCENTILES) -> dict[str, float]:
    """One class's percentiles, its mean, and the risked chance of exceeding each.

    Two readings of the same distribution, side by side, because a decision needs
    both and they answer different questions:

    *Unrisked* (conditional) -- the size **if this outcome happens**. Its P50 is
    exceeded half the time *given* the outcome, so its percentiles are the
    familiar ones and the shape is the shape of the accumulation.

    *Risked* -- the same volumes, with the chance of the outcome folded in. The
    P50 volume is then exceeded ``chance x 50 %`` of the time overall. Nothing
    about the volumes changes; what changes is the probability attached to them,
    and that is the number a decision is made against.

    The mean is reported with the exceedance probability it actually falls at,
    which on a right-skewed distribution is well above 50 % of the way down the
    curve -- typically near P35. That gap is why "the mean" and "the middle" are
    not interchangeable words here.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out: dict[str, float] = {"n": float(v.size), "chance": float(chance)}
    if v.size == 0:
        for q in percentiles:
            out[f"p{q}"] = float("nan")
        out["mean"] = float("nan")
        out["mean_at"] = float("nan")
        return out
    for q in percentiles:
        # P90 is the *low* case: exceeded 90 % of the time, so it is the 10th
        # percentile of the values. Getting this backwards is the classic error.
        out[f"p{q}"] = float(np.percentile(v, 100 - q))
    mean = float(np.mean(v))
    out["mean"] = mean
    # Where the mean sits on this distribution's own curve, conditionally.
    out["mean_at"] = 100.0 * float(np.mean(v >= mean))
    return out
