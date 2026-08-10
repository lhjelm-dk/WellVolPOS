"""Two synthetic trial files, for the cases the real data cannot exercise.

The demo data is one GeoX run of one prospect, and two of this tool's branches
never execute on it:

* **The area / net-pay correlation warning.** The per-trial split in
  :mod:`wellvolpos.core.classes` assumes gross pay and yield are uniform across
  the closure, so that a trial's proven share can be taken as its *area* share
  between entry and exit. On the reference file that assumption is sound
  (area/net-pay r = -0.002) and ``check_area_pay_correlation`` therefore never
  fires. A file where thicker pay comes with larger area makes it fire, which is
  the only way to know the guard works.
* **The risking branch.** ``POS_trials`` is readable only because GeoX writes
  chance failures into the export as all-zero trials. A success-case-only file
  has none, so the app must fall through to the chance table -- and that path is
  the one CLAUDE.md's "POS provenance" section is about.

Both generators emit **GeoX-shaped CSV**, with the same column headers and unit
conventions as ``data/demo_prospectA_reduced.csv``, so a synthetic file goes in
through the real adapter rather than around it. A generator that built a
``TrialSet`` directly would test the maths while skipping the import layer that
does most of the work.

The geometry is a cone: ``A(z) = k * (z - apex)^2`` up to a spill depth. That
gives an exactly known area-depth curve, so a test can check what
:class:`~wellvolpos.core.structure.AreaDepth` recovers against the truth rather
than against itself. Everything else is drawn from lognormals, which is what
GeoX does.

Neither file is a model of anything. They are fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

#: The reduced demo's headers, in its order. Matched exactly so the synthetic
#: files exercise the same adapter path, including the ``Recoverable.`` prefix
#: that the duplicate-column resolver depends on.
COLUMNS = [
    "TrialNumber",
    "Average gross pay",
    "HC bearing gross rock volume",
    "HC pore volume",
    "HC water contact - result",
    "Productive area",
    "Recoverable.Accumulation size Total Resources",
]

#: The sentinel depth GeoX stamps on a chance failure: above any possible crest,
#: so it can never be mistaken for a contact. The value the reference file uses.
FAILURE_SENTINEL = 3166.59


def _lognormal(rng, mean: float, p90_over_p10: float, n: int) -> np.ndarray:
    """A lognormal with the given mean and P10/P90 spread.

    Parameterised by mean and spread rather than by mu and sigma because that is
    how a prospect is actually described -- "mean 45 m, a factor of three between
    P90 and P10" is a sentence a geoscientist says.
    """
    sigma = np.log(p90_over_p10) / (2.0 * 1.2815515655446004)   # z at P90
    mu = np.log(mean) - 0.5 * sigma ** 2
    return rng.lognormal(mu, sigma, n)


def make_trials(
    n: int = 10_000,
    *,
    apex: float = 3200.0,
    spill: float = 3700.0,
    area_at_spill: float = 6.0,
    mean_pay: float = 45.0,
    mean_yield: float = 0.06,
    pos: float = 0.76,
    area_pay_correlation: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A GeoX-shaped trial table over a conical closure.

    ``pos`` of 1.0 gives a success-case-only file -- no zero trials at all, so
    the failure detector finds nothing and the app must use the chance table.
    Any lower value stamps the remainder as chance failures the way GeoX does:
    every hydrocarbon quantity exactly zero and the contact set to
    :data:`FAILURE_SENTINEL`, while the contact *distribution* keeps being
    sampled for the successes.

    ``area_pay_correlation`` induces rank correlation between productive area and
    gross pay, via a Gaussian copula. At 0.0 the two are independent, as in the
    reference file; at 0.8 the uniform-pay assumption behind the per-trial split
    is badly violated and ``check_area_pay_correlation`` should say so. The copula
    changes only the *dependence*, so a correlated file and an independent one
    drawn on the same seed share their contact and area distributions exactly and
    are therefore comparable. Their pay distributions agree except where the
    closure-volume cap below bites, which it does on a few per cent of trials --
    the cap has to be applied after the pairing, because what a trial can hold
    depends on which contact it drew.

    Areas come from contact depths through the cone, not the reverse, so
    ``A(z)`` is a deterministic function of the contact exactly as GeoX makes it
    -- which is the property that lets a trial be split at the well.
    """
    if not 0.0 < pos <= 1.0:
        raise ValueError(f"pos must be in (0, 1]; got {pos}")
    if spill <= apex:
        raise ValueError(f"spill ({spill} m) must be deeper than the apex ({apex} m)")
    rng = np.random.default_rng(seed)

    n_success = int(round(n * pos))
    # Contacts uniform in *volume* rather than in depth, so the deep half of the
    # closure is not oversampled -- a flat depth prior would put most trials near
    # spill where the cone is widest and make the resource distribution wrong.
    u = rng.uniform(0.0, 1.0, n_success)
    contact = apex + (spill - apex) * u ** (1.0 / 3.0)

    k = area_at_spill / (spill - apex) ** 2
    area = k * (contact - apex) ** 2

    # A Gaussian copula on (area rank, pay rank): correlate the normal scores,
    # then map back through the pay marginal. This leaves both marginals exactly
    # as specified and changes only the dependence, which is what makes the
    # correlated case comparable with the independent one.
    pay = _lognormal(rng, mean_pay, 3.0, n_success)
    rho = float(np.clip(area_pay_correlation, -0.99, 0.99))
    if rho != 0.0:
        z_area = np.argsort(np.argsort(area)).astype(float)
        z_area = (z_area + 0.5) / n_success
        g = rho * ndtri(z_area) + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(n_success)
        pay = np.sort(pay)[np.argsort(np.argsort(g))]

    # Gross pay is capped by the *closure*, not by the column height. GeoX's
    # gross pay is area-averaged, so ``GRV = area x pay`` exactly; that product
    # cannot exceed the closure volume above the contact, or the wedge inversion
    # in core.reservoir is handed a trial no thickness can produce and reports it
    # as inconsistent -- correctly, since it is.
    #
    # For a cone the closure volume above z is A(z)(z - apex)/3, so the bound is
    # a third of the column height. Capping at the column height instead lets a
    # cylinder of pay stand over a cone of rock, which is where the first draft
    # of this generator produced 91 impossible trials in 4 000.
    pay = np.minimum(pay, (contact - apex) / 3.0)
    grv = area * pay
    yield_ = _lognormal(rng, mean_yield, 2.0, n_success)
    resource = grv * yield_
    pore = grv * 0.18

    frame = pd.DataFrame({
        "TrialNumber": np.arange(1, n_success + 1),
        "Average gross pay": pay,
        "HC bearing gross rock volume": grv,
        "HC pore volume": pore,
        "HC water contact - result": contact,
        "Productive area": area,
        "Recoverable.Accumulation size Total Resources": resource,
    })

    n_fail = n - n_success
    if n_fail:
        failures = pd.DataFrame({
            "TrialNumber": np.arange(n_success + 1, n + 1),
            "Average gross pay": 0.0,
            "HC bearing gross rock volume": 0.0,
            "HC pore volume": 0.0,
            "HC water contact - result": FAILURE_SENTINEL,
            "Productive area": 0.0,
            "Recoverable.Accumulation size Total Resources": 0.0,
        }, index=range(n_fail))
        frame = pd.concat([frame, failures], ignore_index=True)
        # Shuffled, because a real export does not group its failures. Anything
        # that depends on row order -- and `TrialNumber` is already known not to
        # be a reliable key in a GeoX export -- should fail on this file.
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return frame[COLUMNS]


def correlated_area_pay(n: int = 10_000, *, rho: float = 0.8, seed: int = 1) -> pd.DataFrame:
    """The case that breaks the uniform-pay assumption behind the per-trial split.

    Thicker pay where the area is larger, so a trial's area share between entry
    and exit is no longer its volume share. The split still runs -- it has to,
    since nothing better is available from a trial file -- but
    ``core.classes.check_area_pay_correlation`` must say out loud that it is
    leaning on an assumption this data does not support.
    """
    return make_trials(n, area_pay_correlation=rho, seed=seed)


def success_case_only(n: int = 10_000, *, seed: int = 2) -> pd.DataFrame:
    """A file with no chance failures, to exercise the risking branch.

    ``POS_trials`` is unreadable here, so the app must take POS from the chance
    table and say so in its provenance stamp. Every quantity is a success-case
    quantity, which is the convention most operators' own exports follow.
    """
    return make_trials(n, pos=1.0, seed=seed)


def write(frame: pd.DataFrame, path) -> None:
    """Write a synthetic table where the adapter can read it."""
    frame.to_csv(path, index=False)
