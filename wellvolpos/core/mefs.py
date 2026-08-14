"""Every volume concept read against the MEFS / MCFS line.

Lars, 2026-08-14: *"maybe I need to know/show the probability of the P10, P50, P90,
Pmean exceeding the MCFS/MEFS."*

**A percentile is a fixed volume, so it does not have a probability of exceeding a
threshold** -- it either clears the line or it does not, and that is 0 or 1. The
quantity the question is reaching for already has a name and is already computed:
``P(volume > MEFS)`` is the *exceedance probability at MEFS*, which is the percentile
the line itself sits at. One number per concept, not four.

So each concept reports both halves of the reading, and they are different kinds of
thing:

* the **ladder** -- P90 / P50 / mean / P10, each with a clears-or-not flag. This says
  *between which percentiles* the threshold falls, which is what a reader actually
  scans for.
* ``p_exceeds`` -- the exact chance, **conditional on the concept's own outcome**, the
  same conditioning :func:`wellvolpos.core.classes.class_summary` uses. The attic's is
  conditional on a charged dry hole, the unproven volume's on the well leaving the
  reservoir in hydrocarbons; they are *not* on one footing and must never be summed or
  compared without their conditions.

The two agree by construction, and :mod:`tests.test_mefs` asserts it: if the P50 clears
the line then ``p_exceeds >= 0.5``. A ladder that disagrees with its own probability is
the failure this module exists to make impossible, and it is cheap to check.

**MEFS is never applied to the distributions.** Per Longley (2026) a volume cut-off
*raises* the unrisked mean while *lowering* commercial chance and the two do not
cancel, so the threshold is a line to read against and nothing else. Everything here
reads; nothing filters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The concepts read against the line, in nesting order, with the mask each is
#: conditional on and the palette role it is drawn in. The key is the one
#: ``class_summary`` uses, so the two cannot drift apart.
CONCEPTS = (
    ("discovery", "Well associated", "discovery", "well_associated"),
    ("proven", "Proven", "discovery", "proven"),
    ("below_lkh", "Unproven below LKH", "hc_to_exit", "below_lkh"),
    ("attic_dry_hole", "Attic", "dry_with_attic", "up_dip"),
)

#: The rungs, in the order they are read. ``mean`` sits between P50 and P10 because
#: that is where a right-skewed distribution puts it -- but it is **not** a percentile
#: and the label says so wherever this is rendered.
RUNGS = ("p90", "p50", "mean", "p10")


@dataclass(frozen=True)
class ConceptReadout:
    """One volume concept, read against the threshold."""

    key: str
    label: str
    #: What the concept is conditional on, named for display. Without this the four
    #: rows look comparable and they are not.
    condition: str
    role: str
    volumes: dict[str, float]
    #: ``P(volume > MEFS | the concept's own event)``. The exceedance probability at
    #: the line, which is the only genuine probability in this object.
    p_exceeds: float
    #: How many trials it rests on, so a reader can tell an estimate from a rumour.
    n: int

    def clears(self, rung: str) -> bool:
        """Does this rung sit above the threshold? Binary, by construction."""
        return bool(np.isfinite(self.volumes[rung]) and self.volumes[rung] > self._mefs)

    _mefs: float = 0.0

    def bracket(self) -> str:
        """Which two rungs the threshold falls between, in words."""
        clearing = [r for r in RUNGS if self.clears(r)]
        if not clearing:
            return "below every rung — the threshold is above P10"
        if len(clearing) == len(RUNGS):
            return "above every rung — the threshold is below P90"
        first = clearing[0]
        prior = RUNGS[RUNGS.index(first) - 1]
        return f"between {prior.upper()} and {first.upper()}"


@dataclass(frozen=True)
class MefsReadout:
    """Every concept against one threshold."""

    mefs: float
    concepts: tuple[ConceptReadout, ...]

    def by_key(self, key: str) -> ConceptReadout | None:
        for c in self.concepts:
            if c.key == key:
                return c
        return None


def mefs_readout(vc, groups, summary: dict[str, dict[str, float]],
                 mefs: float) -> MefsReadout:
    """Read every volume concept against the MEFS / MCFS line.

    ``summary`` is :func:`class_summary`'s output, passed in rather than recomputed so
    the volumes here are byte-for-byte the ones already on screen. Recomputing them is
    how a table comes to disagree with the figure beside it.
    """
    values = {
        "discovery": vc.discovery_total,
        "proven": vc.proven,
        "below_lkh": vc.below_lkh,
        "attic_dry_hole": vc.attic,
    }
    out: list[ConceptReadout] = []
    for key, label, mask_name, role in CONCEPTS:
        stats = summary.get(key)
        if stats is None:
            continue
        mask = np.asarray(getattr(groups, mask_name), dtype=bool)
        x = np.asarray(values[key], dtype=float)[mask]
        p = float((x > mefs).mean()) if x.size else float("nan")
        out.append(ConceptReadout(
            key=key, label=label, condition=_CONDITIONS[mask_name], role=role,
            volumes={r: float(stats.get(r, float("nan"))) for r in RUNGS},
            p_exceeds=p, n=int(x.size), _mefs=float(mefs),
        ))
    return MefsReadout(mefs=float(mefs), concepts=tuple(out))


#: Spelled out, because "conditional on its own outcome" is exactly the phrase a reader
#: skips. Each row of the readout says which event it lives in.
_CONDITIONS = {
    "discovery": "a discovery",
    "hc_to_exit": "the well leaving the reservoir in hydrocarbons",
    "dry_with_attic": "a charged dry hole",
}
