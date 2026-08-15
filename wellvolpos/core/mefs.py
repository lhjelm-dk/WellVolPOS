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


# --------------------------------------------------------------- 4.2's four cases
@dataclass(frozen=True)
class Crossing:
    """Where one of 4.2's curves crosses the MEFS line, on both readings."""

    name: str
    role: str
    #: ``P(volume > MEFS | the case happens)`` -- the solid curve's height at the line.
    conditional: float
    #: The chance the case happens at all. The dashed curve is the solid one scaled by
    #: this, so it starts here rather than at 100 %.
    chance: float
    n: int

    @property
    def short(self) -> str:
        """The name with the words a bar chart does not need taken off.

        4.2 needs the long form: its legend sits beside three other curves and has to
        say which volume each is. A bar chart's rows are already the only thing on the
        axis, and *"resource potential"* appears on two of the four, so it separates
        nothing while making the labels wide enough to be clipped -- which is what
        happened (Lars, 2026-08-15).
        """
        return (self.name
                .replace(" resource potential", "")
                .replace("Resource tested by well", "Tested by well"))

    @property
    def risked(self) -> float:
        """The dashed curve's height at the line: ``chance × conditional``.

        Risking scales the **probability** and never the volume, which is why this is
        a product rather than a second pass over different trials.
        """
        return self.chance * self.conditional


def c2_cases(ts, groups, vc, pos_prospect: float, p_well: float):
    """The four concepts 4.2 draws, with the chance each one is risked by.

    **One definition, used by the figure and by the caption that quotes its numbers.**
    They were about to be two lists in two modules, which is how a caption comes to
    assert something the figure beside it denies -- a mistake this codebase has made
    twice with numbers written against one demo file.

    The up-dip case needs its **own** chance: dry but charged is
    ``POS_prospect − P_well``, not ``P_well``. That is the one entry a second copy of
    this list would most likely get wrong, because the other three look like they take
    the obvious chance and this one does not.
    """
    res = ts.col("resource")
    disc, dry = groups.discovery, groups.dry_with_attic
    return [
        ("Prospect resource potential", res[res > 0], float(pos_prospect), "prospect"),
        ("Well associated resource potential", res[disc], float(p_well), "well_associated"),
        ("Resource tested by well", vc.proven[disc], float(p_well), "tested"),
        ("Up-dip volume", res[dry], float(max(pos_prospect - p_well, 0.0)), "up_dip"),
    ]


def c2_crossings(ts, groups, vc, pos_prospect: float, p_well: float,
                 mefs: float) -> tuple[Crossing, ...]:
    """``P(> MEFS)`` for each of 4.2's concepts, conditional and risked.

    Lars, 2026-08-14: *"can I get a probability curve in 4.2 for exceedance MEFS,
    risked and unrisked."* Each exceedance curve on 4.2 already **is** a probability
    curve, so the threshold's probability is a *crossing* rather than a new series --
    eight of them, four concepts on two readings. The figure marks them; this returns
    the numbers, because eight labels along one vertical line overlap.
    """
    out = []
    for name, values, chance_of, role in c2_cases(ts, groups, vc, pos_prospect, p_well):
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        out.append(Crossing(
            name=name, role=role,
            conditional=float((x > mefs).mean()) if x.size else float("nan"),
            chance=float(chance_of), n=int(x.size),
        ))
    return tuple(out)
