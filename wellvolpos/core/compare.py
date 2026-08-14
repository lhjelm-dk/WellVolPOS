"""Several well locations, evaluated the same way, side by side.

The arithmetic behind tab ④'s comparison. It lives in ``core`` for the reason all the
arithmetic does: a table that computed its own numbers could disagree with the figures
beside it, and a comparison whose columns are computed differently from one another is
worse than no comparison at all. Every well here goes through exactly the functions a
single well goes through -- :func:`~wellvolpos.core.groups.group_trials`,
:func:`~wellvolpos.core.classes.split_trials`, :func:`~wellvolpos.core.chance.p_well`
-- with the same thickness, the same apex and the same reference contour.

**Percentiles are conditional**, on the success case of whichever concept they belong
to, and the risked figures are kept in their own block. Mixing an unrisked percentile
with a risked mean in one row is how a comparison table becomes the thing this project
keeps catching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .chance import ReferenceContour, p_well
from .classes import split_trials
from .groups import group_summary, group_trials
from .reservoir import thickness_from_pay

if TYPE_CHECKING:  # pragma: no cover
    from .structure import AreaDepth
    from ..io.adapters.base import TrialSet

#: The ladder every volume in the comparison is reported at. Petroleum orientation:
#: P90 is the low case, exceeded 90 % of the time.
COMPARE_PERCENTILES = (90, 50, 10)


def _stats(values: np.ndarray) -> dict[str, float]:
    """P90 / P50 / mean / P10 of a conditional distribution, empty-safe."""
    v = np.asarray(values, dtype=float)
    if not v.size:
        return {"p90": float("nan"), "p50": float("nan"),
                "mean": float("nan"), "p10": float("nan"), "n": 0.0}
    return {
        "p90": float(np.percentile(v, 10.0)),
        "p50": float(np.percentile(v, 50.0)),
        "mean": float(v.mean()),
        "p10": float(np.percentile(v, 90.0)),
        "n": float(v.size),
    }


@dataclass(frozen=True)
class WellComparison:
    """One candidate, evaluated. Everything a well proposal argues about."""

    label: str
    entry: float
    exit: float

    pos_prospect: float
    r_location: float
    p_well: float

    #: Conditional on a discovery: what the well would prove, and the whole
    #: accumulation it would have access to.
    proven: dict[str, float] = field(default_factory=dict)
    well_associated: dict[str, float] = field(default_factory=dict)
    #: Conditional on a dry hole that is nonetheless charged.
    attic: dict[str, float] = field(default_factory=dict)
    #: Conditional on a discovery: what sits below the reservoir exit, untested.
    below_lkh: dict[str, float] = field(default_factory=dict)

    n_discovery: int = 0

    @property
    def gap(self) -> float:
        return self.exit - self.entry

    @property
    def expected_proven(self) -> float:
        """Risked. Kept as a property so it cannot be confused with a percentile."""
        return self.proven.get("mean", float("nan")) * self.p_well

    @property
    def expected_well_associated(self) -> float:
        return self.well_associated.get("mean", float("nan")) * self.p_well


def compare_wells(
    ts: "TrialSet",
    ad: "AreaDepth | None",
    wells: Sequence[tuple[str, float, float]],
    *,
    pos_prospect: float,
    reference: ReferenceContour = ReferenceContour.CREST,
    apportionment: str = "wedge",
) -> tuple[WellComparison, ...]:
    """Evaluate every candidate the way the app evaluates one.

    ``wells`` is ``(label, entry, exit)`` per candidate. The thickness inversion and
    the apex are computed **once** and reused, so the candidates differ only in where
    the well is -- which is the entire point of putting them side by side. Doing it
    per well would let two locations disagree about the geometry of the same closure.
    """
    thickness = None
    apex = None
    if ad is not None:
        try:
            thickness = thickness_from_pay(ts, ad).thickness
            apex = float(ad.apex_estimate())
        except Exception:
            thickness, apex = None, None

    out: list[WellComparison] = []
    for label, entry, exit_ in wells:
        entry = float(entry)
        exit_ = float(max(exit_, entry))
        groups = group_trials(ts, entry, exit_)
        chance = p_well(ts, entry, pos_prospect, reference=reference)
        gs = group_summary(ts, groups)

        proven = well_assoc = attic = possible = {}
        n_disc = int(np.asarray(groups.discovery).sum())
        if ad is not None:
            vc = split_trials(ts, ad, groups, entry, exit_,
                              apportionment=apportionment,
                              thickness=thickness, apex=apex)
            disc = np.asarray(groups.discovery, dtype=bool)
            dry = np.asarray(groups.dry_with_attic, dtype=bool)
            proven = _stats(vc.proven[disc])
            possible = _stats(vc.below_lkh[disc])
            well_assoc = _stats(vc.discovery_total[disc])
            attic = _stats(vc.attic[dry])
        else:
            # No area column: the reference engine still works, the split does not.
            well_assoc = {k: gs["discovery"][k] for k in ("p90", "p50", "mean", "p10")}
            well_assoc["n"] = gs["discovery"]["n"]
            attic = {k: gs["attic_dry_hole"][k] for k in ("p90", "p50", "mean", "p10")}
            attic["n"] = gs["attic_dry_hole"]["n"]

        out.append(WellComparison(
            label=label, entry=entry, exit=exit_,
            pos_prospect=float(chance.pos_prospect),
            r_location=float(chance.r_location),
            p_well=float(chance.p_well),
            proven=proven, well_associated=well_assoc, attic=attic, below_lkh=possible,
            n_discovery=n_disc,
        ))
    return tuple(out)


def chance_table(rows: Sequence[WellComparison]) -> list[dict]:
    """The risk block: one row per well, chance only."""
    return [{
        "Well": r.label,
        "Entry (m)": r.entry,
        "Exit (m)": r.exit,
        "POS prospect": r.pos_prospect,
        "r location": r.r_location,
        "P well": r.p_well,
        "Discovery trials": r.n_discovery,
    } for r in rows]


def volume_table(rows: Sequence[WellComparison], concept: str = "proven") -> list[dict]:
    """The volumetric block for one concept: **conditional**, in MMboe.

    Conditional on the outcome the concept belongs to -- a discovery for proven,
    possible and well-associated, a charged dry hole for the attic -- which is where
    percentiles are defined. The chance is in :func:`chance_table` and the product in
    :func:`risked_table`; keeping the three apart is what stops a risked number being
    read as a percentile.
    """
    label = {"proven": "Proven", "well_associated": "Well associated",
             "attic": "Attic (dry & charged)", "below_lkh": "Unproven below LKH"}[concept]
    out = []
    for r in rows:
        s = getattr(r, concept)
        out.append({
            "Well": r.label,
            "Concept": label,
            "P90": s.get("p90", float("nan")),
            "P50": s.get("p50", float("nan")),
            "Mean": s.get("mean", float("nan")),
            "P10": s.get("p10", float("nan")),
        })
    return out


def risked_table(rows: Sequence[WellComparison]) -> list[dict]:
    """The risked block: mean x chance, and nothing that looks like a percentile.

    A risked *percentile* is not reported, deliberately. Risking scales the
    probability attached to a volume, not the volume, so "the risked P50" is a
    category error -- the P50 volume does not change, its exceedance probability does.
    """
    return [{
        "Well": r.label,
        "P well": r.p_well,
        "Expected proven": r.expected_proven,
        "Expected well associated": r.expected_well_associated,
    } for r in rows]
