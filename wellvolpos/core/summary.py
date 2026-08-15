"""The headline: what this well gives you, in five numbers and one sentence.

Lars, 2026-08-15, from the design review: *"the tool computes everything and
concludes nothing."* It had `P_well`, `Pc`, the proven mean, the expectation peak,
the commercial peak and the required depth, and never assembled them. A reader had to
hold six figures in their head to answer "so what does this well give me".

**The assembly lives here, not in the tab.** A tab that computes is a tab that can
disagree with the figure under it -- which has happened in this codebase twice, both
times because a number was re-derived beside the thing that already had it. Everything
below is read from objects the app has already built (`ChanceResult`, `class_summary`,
`CommercialChance`); nothing is recomputed from the trials.

**Two chances and three volumes, and the wording keeps them apart.** The recurring bug
in this project is an unrisked number under a risked label, and a headline block is
exactly where that would do the most damage, because it is the part a reader quotes.
So :meth:`Headline.sentence` names the conditioning of every volume it prints.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Headline:
    """One well, summarised. Every field is read from an object that already had it."""

    entry: float
    exit_: float

    #: The chance of finding hydrocarbons at all. ``POS_prospect x r_location``.
    p_well: float
    #: The two factors, carried so the sentence can show the multiplication rather
    #: than assert the product -- see CLAUDE.md's one idea everything rests on.
    pos_prospect: float
    r_location: float

    #: Rose's commercial chance, ``P_well x Pmcfs(well)``. ``None`` when there is no
    #: area column, since the split the threshold is read against needs one.
    pc_well: float | None
    mefs: float | None

    #: Conditional on a discovery.
    proven_mean: float | None
    well_associated_mean: float | None
    #: Conditional on a charged dry hole. A different event, hence a separate field
    #: rather than a fourth entry in a volume list.
    attic_mean: float | None

    n_discovery: int
    n_dry_with_attic: int
    #: The trial count, carried rather than assumed. The first draft of
    #: :meth:`sentence` wrote "of 10,000 trials" as a literal -- which is prospect A's
    #: count and a claim about a file nobody may be looking at. Same failure the two
    #: captions in CLAUDE.md's "asserted what the data denied" section made.
    n_total: int

    @property
    def gap(self) -> float:
        return self.exit_ - self.entry

    def sentence(self) -> str:
        """The headline as one line of markdown, with the conditioning named.

        Written to be quoted. That is the whole risk: a reader lifts this into a
        well proposal, so every volume in it says which outcome it belongs to and
        neither chance is left to look like the other.
        """
        head = (f"A well entering at **{self.entry:,.0f} m** and leaving at "
                f"**{self.exit_:,.0f} m TVDSS** has a **{self.p_well:.1%}** chance of "
                f"finding hydrocarbons")
        if self.pc_well is not None and self.mefs is not None:
            head += (f", and a **{self.pc_well:.1%}** chance of finding more than "
                     f"{self.mefs:,.1f} MMboe")
        head += "."
        if self.proven_mean is None:
            return head
        return (head + f" If it works it proves **{self.proven_mean:,.1f} MMboe** on "
                f"average, out of {self.well_associated_mean:,.1f} MMboe in the "
                f"accumulation it would have found. If it is dry there is still "
                f"**{self.attic_mean:,.1f} MMboe** up-dip, in the "
                f"{self.n_dry_with_attic:,} of {self.n_total:,} trials that are charged but "
                f"shallower than the well.")


def headline(*, entry: float, exit_: float, chance, groups,
             class_stats: dict | None = None, commercial=None,
             mefs: float | None = None) -> Headline:
    """Assemble the headline from what the app has already computed.

    ``class_stats`` is :func:`wellvolpos.core.classes.class_summary`'s output and
    ``commercial`` a :class:`wellvolpos.core.rose.CommercialChance`; both are optional
    because a trial file with no productive-area column supports neither. The chance
    half is always available, so the block never disappears entirely -- a tab that
    sometimes has a headline and sometimes does not is worse than one that always has
    the part it can stand behind.
    """
    import numpy as np

    def stat(key: str) -> float | None:
        if not class_stats or key not in class_stats:
            return None
        v = class_stats[key].get("mean")
        return None if v is None or not np.isfinite(v) else float(v)

    return Headline(
        entry=float(entry), exit_=float(exit_),
        p_well=float(chance.p_well),
        pos_prospect=float(chance.pos_prospect),
        r_location=float(chance.r_location),
        pc_well=None if commercial is None else float(commercial.pc_well),
        mefs=None if mefs is None else float(mefs),
        proven_mean=stat("proven"),
        well_associated_mean=stat("discovery"),
        attic_mean=stat("attic_dry_hole"),
        n_discovery=int(np.asarray(groups.discovery).sum()),
        n_dry_with_attic=int(np.asarray(groups.dry_with_attic).sum()),
        n_total=int(np.asarray(groups.discovery).size),
    )
