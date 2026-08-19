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
    #: than assert the product: `P_well = POS_prospect x r_location`, two numbers
    #: with two meanings, never multiplied into one for reporting.
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
    #: two captions that quoted prospect A's numbers above prospect B once made.
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


# ------------------------------------------------------------- candidate depths
@dataclass(frozen=True)
class Candidate:
    """One depth the sweep says is optimal, by one measure."""

    key: str
    #: What this depth is best *at*, in the reader's words.
    label: str
    depth: float
    #: The value that makes it optimal, already formatted with its unit.
    value: str
    #: Which figure shows it, as a numbering key, so the panel can point at it.
    figure: str
    note: str = ""
    #: One clause naming the quantity this row maximises, for the table itself. The
    #: longer ``note`` explains *why* it matters and is a click away; this is what the
    #: reader needs in order to tell two rows apart at a glance.
    maximises: str = ""
    #: What kind of advice this row is. **Three different things were being listed
    #: under one heading** (Lars, 2026-08-18: *"the entry depth on this table does not
    #: give any guidance to an ideal depth to go for"*) -- and he was right, because
    #: every row was an *optimum* of some criterion and none of them bounded the
    #: answer. A floor and a ceiling do more work than a fourth optimum: together they
    #: are a window, and an optimum outside it is a warning rather than a target.
    #:
    #: ``optimum``  -- maximises something
    #: ``floor``    -- go at least this deep
    #: ``ceiling``  -- go no deeper than this
    kind: str = "optimum"
    #: The depth range over which the measure stays within :data:`PLATEAU_TOL` of its
    #: maximum. ``argmax`` on a nearly flat curve returns whichever grid point wins by
    #: a hair, and the winner moves with the grid: prospect B's commercial optimum came
    #: out at 2064 m on the app's sweep and 2115 m on a coarser one, both at Pc 21.9 %.
    #: Reporting one of them alone is false precision, so the range travels with it.
    plateau: tuple[float, float] | None = None

    @property
    def is_flat(self) -> bool:
        """Is the optimum weak -- a plateau rather than a peak?"""
        return (self.plateau is not None
                and (self.plateau[1] - self.plateau[0]) > 1.0)

    def describe_depth(self) -> str:
        """The depth, widened to its plateau where the maximum is weak.

        Kept for prose, where a range reads correctly inside a sentence. **The table
        does not use it** -- see :meth:`describe_plateau`.
        """
        if not self.is_flat:
            return f"{self.depth:,.0f} m"
        lo, hi = self.plateau
        return f"{lo:,.0f}–{hi:,.0f} m"

    def describe_plateau(self) -> str:
        """The span of equally good depths, for a column beside the depth.

        **The depth column and the value column must agree** (Lars, 2026-08-18). A
        range under *Entry* against a single number under *Value there* invites the
        reading that the value holds across the range, and it does not: the value is
        the one at ``depth``, the ``argmax``. So the table reports the depth the value
        belongs to and puts the plateau in its own column, where "equally good" is
        stated rather than implied.
        """
        if not self.is_flat:
            return "—"
        lo, hi = self.plateau
        return f"{lo:,.0f}–{hi:,.0f} m"


#: How close to the maximum still counts as "as good as the best", relatively. A
#: chance or an expectation within a fiftieth of the peak is not distinguishable given
#: the sampling error the sweep already reports, so the plateau is drawn at 2 %.
PLATEAU_TOL = 0.02


def shallowest_argmax(values, tol: float = 1e-9) -> int:
    """Index of the maximum, resolving a tie **towards the crest**.

    Lars, 2026-08-18: *"the entry depths are all the same — is this correct?"* It is,
    and the reason is worth stating rather than leaving as a coincidence.

    **Above the shallowest sampled contact every success trial is a discovery.** So
    ``r_location`` is exactly 1 there, ``P_well`` is flat at ``POS_prospect``, and the
    volume conditional on a discovery is the whole prospect distribution. Every
    criterion that does not involve MEFS is therefore *indifferent* across that whole
    band, and several of them are flat well past it. The sweep grid also starts 3 %
    above the shallowest contact by design, so the curves visibly saturate rather than
    beginning mid-rise -- which puts a genuinely degenerate stretch at the top of
    every one of them.

    ``np.nanargmax`` returned the first maximum, which happened to be the shallowest.
    That was the right answer arrived at by accident, and an accident is not a rule:
    a later change to the grid, or to argmax's tie behaviour, would silently move
    every reported optimum. So the tie-break is written down.

    **Shallowest, because shallower never costs chance.** ``P_well`` falls
    monotonically down-dip, so among depths where the criterion cannot tell the
    difference, the shallowest one is weakly better on the one thing every criterion
    here shares. The *width* of the indifference is reported separately, by
    :func:`plateau_span`.
    """
    import numpy as np

    v = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(v)):
        raise ValueError("no finite values to maximise")
    peak = float(np.nanmax(v))
    scale = max(abs(peak), 1e-30)
    near = np.isfinite(v) & (v >= peak - tol * scale)
    return int(np.argmax(near))


def plateau_span(values, z, i: int, tol: float = PLATEAU_TOL):
    """The depth span over which ``values`` stays within ``tol`` of its maximum.

    **The whole span, not the contiguous run around the peak.** A sampled curve wiggles
    at the percent level, so the near-max set is often broken by a grid point that dips
    just under the tolerance -- and a contiguous rule then reports a one-cell plateau
    for a curve that is flat over 50 m. Prospect B's commercial optimum showed exactly
    that: the peak moved 2064 -> 2115 m between two grid resolutions, both at Pc 21.9 %,
    while a contiguous plateau claimed the maximum was sharp.

    The span can therefore contain depths that are *not* within tolerance. That is the
    honest reading of "the best is somewhere in here": it is a statement about how far
    apart equally good locations lie, not a promise about every metre between them.
    """
    import numpy as np

    v = np.asarray(values, dtype=float)
    peak = float(v[i])
    if not np.isfinite(peak) or peak <= 0:
        return None
    near = np.isfinite(v) & (v >= peak * (1.0 - tol))
    if not near.any():
        return None
    zz = np.asarray(z, dtype=float)[near]
    return (float(zz.min()), float(zz.max()))


def _guaranteed_crossing(values, z, target: float, counts, min_support: int):
    """Shallowest depth from which ``values`` stays at or above ``target`` downward.

    A *guarantee*, not a first touch -- the same rule ``_required_depth`` and
    :func:`~wellvolpos.core.utility.constrained_best` use, and for the same reason: a
    sampled curve dips wherever the group behind it is small, so inverting the first
    crossing returns depths that deeper locations contradict.

    Both quantities this is used on -- the proven mean and the attic mean -- rise
    monotonically down-dip in principle, so on clean data the guarantee and the first
    crossing coincide. The rule matters exactly where the data are not clean, which is
    where a reader would otherwise be misled without noticing.
    """
    import numpy as np

    from .stats import thin

    v = thin(values, counts, min_support)
    zz = np.asarray(z, dtype=float)
    running = np.inf
    ok = np.zeros(zz.size, dtype=bool)
    for i in range(zz.size - 1, -1, -1):
        if np.isfinite(v[i]):
            running = min(running, float(v[i]))
            ok[i] = running >= float(target)
    if not ok.any():
        return None
    return float(zz[int(np.argmax(ok))])


def candidate_depths(vsweep, *, min_support: int = 30,
                     constrained=None, risk_adjusted=None,
                     required_depth: float | None = None,
                     required_target: float | None = None,
                     required_statistic: str = "mean",
                     sweep=None) -> tuple[Candidate, ...]:
    """The depths the sweep already identifies as optimal, gathered in one place.

    Lars, 2026-08-15, from the design review: the tool finds three optima on three
    different figures and never names them together, so a reader comparing them has to
    remember two while looking at the third.

    **The arithmetic mirrors the figures exactly**, including the support thinning --
    ``thin(...)`` before ``nanargmax``, so a peak can never be reported from a region
    B8 or B9 declined to plot. It is deliberately a *second* implementation rather than
    a refactor of the two figures, because moving their starred markers is a bigger
    change than this warrants; ``test_summary`` cross-checks that the two agree, which
    is the guarantee that matters.

    **The best-chance row is degenerate and is labelled as such.** ``P_well`` falls
    monotonically down-dip, so its maximum is always the shallowest supported depth.
    That is not a recommendation, it is the shallow end of the sweep -- and stating it
    is what makes the other rows read as a trade rather than as a menu.
    """
    import numpy as np

    from .stats import thin

    z = np.asarray(vsweep.z, dtype=float)
    n = vsweep.n_discovery
    out: list[Candidate] = []

    pw = thin(vsweep.p_well, n, min_support)
    if np.any(np.isfinite(pw)):
        i = shallowest_argmax(pw)
        out.append(Candidate(
            key="chance", label="Best chance of finding hydrocarbons", maximises="well POS",
            depth=float(z[i]), value=f"P_well {pw[i]:.1%}", figure="a3",
            plateau=plateau_span(pw, z, i),
            note="The shallowest supported depth, by construction — P_well only "
                 "falls as the well goes down-dip.",
        ))

    # B9's own arithmetic: P_well x the well-associated mean, thinned the same way.
    if vsweep.discovery_mean is not None:
        weighted = pw * thin(vsweep.discovery_mean, n, min_support)
        if np.any(np.isfinite(weighted)):
            i = shallowest_argmax(weighted)
            out.append(Candidate(
                key="expected", label="Most chance-weighted volume", maximises="well POS × the accumulation it would find",
                depth=float(z[i]), value=f"{weighted[i]:,.1f} MMboe expected",
                figure="b9", plateau=plateau_span(weighted, z, i),
                note="P_well x the well-associated mean. The one a portfolio adds up, "
                     "and not a volume anyone finds.",
            ))

    # B8's: Pc = P_well x P(discovery exceeds MEFS).
    if vsweep.p_discovery_exceeds_mefs is not None:
        pc = pw * thin(vsweep.p_discovery_exceeds_mefs, n, min_support)
        if np.any(np.isfinite(pc)):
            i = shallowest_argmax(pc)
            out.append(Candidate(
                key="commercial", label="Best commercial chance", maximises="Pc — well POS × the chance the discovery clears the threshold",
                depth=float(z[i]), value=f"Pc {pc[i]:.1%}", figure="b8",
                plateau=plateau_span(pc, z, i),
                note="A rising conditional times a falling P_well, so this one has an "
                     "interior maximum. Rose's number for an EMV calculation.",
            ))

    # **Two optima the expectation cannot express** -- see core/utility.py. Both are
    # passed in rather than computed here: one needs a confidence the user chose and
    # the other a risk tolerance, and neither is a property of the sweep alone.
    if constrained is not None and constrained.feasible:
        out.append(Candidate(
            key="constrained",
            label=f"Best odds at {constrained.confidence:.0%} commercial confidence",
            maximises="well POS, subject to the discovery clearing the threshold "
                      "at least that often",
            depth=constrained.depth,
            value=f"P_well {constrained.p_well_at:.1%}", figure="b8",
            note="The shallowest depth from which a discovery stays that likely to "
                 "clear MEFS. A constraint, not an optimum -- the best chance "
                 "available once the hurdle is met.",
        ))
    if risk_adjusted is not None and risk_adjusted.best is not None:
        out.append(Candidate(
            key="risk_adjusted", label="Best risk-adjusted volume",
            maximises="the same expectation under exponential utility, at the risk "
                      "tolerance set above",
            depth=risk_adjusted.best_depth,
            value=f"{risk_adjusted.ce[risk_adjusted.best]:,.1f} MMboe certain-equivalent",
            figure="b9", plateau=plateau_span(risk_adjusted.ce, risk_adjusted.z,
                                              risk_adjusted.best),
            note=f"Exponential utility at a risk tolerance of "
                 f"{risk_adjusted.rho:,.0f} MMboe. Risk aversion penalises the "
                 f"low-chance, high-volume tail, so this never sits deeper than the "
                 f"expectation peak.",
        ))

    # ---------------------------------------------------------------------------
    # **Chance-weighted PROVEN volume** (Lars, 2026-08-18). The expectation row above
    # weights the *well-associated* volume -- everything the discovery holds, whether
    # this well demonstrates it or not. Weighting the **proven** volume instead asks
    # what the well is expected to *establish*, and the two peak in different places:
    # 3,350 m against 3,461 m on prospect A. The second is the exploration question
    # when the well has to stand up on what it drilled.
    if vsweep.proven_mean is not None:
        weighted_proven = pw * thin(vsweep.proven_mean, n, min_support)
        if np.any(np.isfinite(weighted_proven)):
            i = shallowest_argmax(weighted_proven)
            out.append(Candidate(
                key="expected_proven", label="Most chance-weighted proven volume", maximises="well POS × the volume it would demonstrate",
                depth=float(z[i]),
                value=f"{weighted_proven[i]:,.1f} MMboe expected",
                figure="b9", plateau=plateau_span(weighted_proven, z, i),
                note="P_well x the proven mean — what the well is expected to "
                     "*establish*, as against what the accumulation holds.",
            ))

    # **Haskett's appraisal optimum**, from the uncertainty-reduction curve. It needs
    # the reference-engine sweep, which is why one is passed in: the volume sweep does
    # not carry it, and recomputing it here would be a second implementation of a
    # curve the tab already draws.
    if sweep is not None and getattr(sweep, "uncertainty_reduction", None) is not None:
        red = np.asarray(sweep.uncertainty_reduction, dtype=float)
        if np.any(np.isfinite(red)):
            i = shallowest_argmax(red)
            out.append(Candidate(
                key="learning", label="Most uncertainty resolved", maximises="the expected narrowing of the prospect's volume range",
                depth=float(np.asarray(sweep.z, dtype=float)[i]),
                # ``uncertainty_reduction`` is **already per cent** -- 39.4, not
                # 0.394 -- so ``:.0%`` multiplied it again and the panel read
                # "3938% of the P90-P10 range". 3.3 draws the same array against
                # an axis labelled "(%)", which is what made the units visible
                # there and invisible here.
                value=f"{red[i]:.0f}% of the P90–P10 range",
                figure="b3", plateau=plateau_span(red, np.asarray(sweep.z, float), i),
                note="Haskett's appraisal criterion: what the well is worth is what "
                     "it *resolves*, not what it finds. Independent of MEFS.",
            ))

    # ---------------------------------------------------------------------------
    # **The two bounds, which are what actually narrow the answer.**
    #
    # A floor and a ceiling do more work than a fourth optimum. The optima above all
    # maximise something and none of them says a depth is *wrong*; these two do, and
    # between them they are a window a reader can defend in a meeting.
    if vsweep.mefs is not None:
        if vsweep.proven_mean is not None:
            zc = _guaranteed_crossing(vsweep.proven_mean, z, vsweep.mefs, n, min_support)
            if zc is not None:
                out.append(Candidate(
                    key="proven_mefs", kind="floor", maximises="a bound, not an optimum",
                    label="Shallowest that proves MEFS on average",
                    depth=zc, value=f"proven mean ≥ {vsweep.mefs:,.1f} MMboe",
                    figure="b1",
                    note="Shallower than this and an average discovery does not "
                         "demonstrate a commercial volume, however likely it is.",
                ))
        if vsweep.proven_p90 is not None:
            zc = _guaranteed_crossing(vsweep.proven_p90, z, vsweep.mefs, n, min_support)
            if zc is not None:
                out.append(Candidate(
                    key="proven_p90_mefs", kind="floor", maximises="a bound, not an optimum",
                    label="Shallowest that proves MEFS even in a poor discovery",
                    depth=zc, value=f"proven P90 ≥ {vsweep.mefs:,.1f} MMboe",
                    figure="b1",
                    note="The conservative reading of the row above: not the average "
                         "discovery but a poor one still clears the threshold.",
                ))
        if vsweep.attic_mean is not None:
            zc = _guaranteed_crossing(vsweep.attic_mean, z, vsweep.mefs, n, min_support)
            if zc is not None:
                out.append(Candidate(
                    key="attic_mefs", kind="ceiling", maximises="a bound, not an optimum",
                    label="Deeper than this, a dry hole leaves MEFS up-dip",
                    depth=zc, value=f"attic mean ≥ {vsweep.mefs:,.1f} MMboe",
                    figure="b1",
                    note="The attic grows as the well goes down-dip. Past this depth "
                         "a dry hole would have left a commercial volume untested "
                         "above it — the regret case, and the reason not to chase "
                         "volume indefinitely.",
                ))

    if required_depth is not None and np.isfinite(required_depth):
        out.append(Candidate(
            key="required", label="Shallowest depth that proves the target",
            maximises="nothing — it inverts the target volume to a depth",
            depth=float(required_depth),
            value=(f"{required_target:,.1f} MMboe {required_statistic}"
                   if required_target is not None else "the current target"),
            figure="b6",
            note="A guarantee, not a first touch: the statistic stays at or above the "
                 "target from here all the way down.",
        ))
    return tuple(out)


def drilling_window(candidates) -> tuple[float | None, float | None]:
    """The deepest floor and the shallowest ceiling, from a set of candidates.

    **This is the answer the table was not giving** (Lars, 2026-08-18). Five optima
    that all land on the shallow end tell a reader nothing about where to drill; a
    bracket does, and any optimum falling outside it is then visibly a warning rather
    than a target.

    Either end can be ``None`` -- on prospect A the attic mean never reaches MEFS at
    any depth, so there is no ceiling and the honest answer is that nothing bounds the
    well from below. An empty window (floor deeper than ceiling) is also possible and
    is left for the caller to report as such: it means no depth both proves the
    threshold and avoids leaving it up-dip, which is a real finding about the prospect
    rather than an error.
    """
    floors = [c.depth for c in candidates if c.kind == "floor"]
    ceilings = [c.depth for c in candidates if c.kind == "ceiling"]
    return (max(floors) if floors else None,
            min(ceilings) if ceilings else None)
