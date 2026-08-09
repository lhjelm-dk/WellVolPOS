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
from .groups import group_trials


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


def run_sweep(
    ts: TrialSet,
    pos_prospect: float,
    *,
    reference: ReferenceContour = ReferenceContour.CREST,
    reference_percentile: float = 0.90,
    z_min: float | None = None,
    z_max: float | None = None,
    n: int = 200,
) -> Sweep:
    """Sweep the reservoir entry depth and evaluate both engines at each step.

    The default range runs from just above the shallowest successful contact
    (so the chance curves visibly saturate towards 1) to the deepest one.
    Pass explicit ``z_min`` / ``z_max`` -- e.g. the current entry/exit slider
    bounds -- to zoom the sweep to where the user is actually looking; no
    padding is added when both are given explicitly.
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
    for i, zi in enumerate(z):
        chance = _p_well(
            ts, float(zi), pos_prospect,
            reference=reference, reference_percentile=reference_percentile,
        )
        r[i] = chance.r_location
        pw[i] = chance.p_well

        discovery = group_trials(ts, float(zi)).discovery
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
    )
