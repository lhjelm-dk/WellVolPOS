"""Every volume concept read against the MEFS / MCFS line.

The point of the module under test is that a **percentile has no probability of
exceeding a threshold** -- it clears the line or it does not. What has a probability is
the concept, and that probability is the exceedance percentile the line sits at. These
tests pin the two together, because a ladder that disagrees with its own probability is
the one failure the module exists to prevent and it is invisible on screen.
"""

import numpy as np
import pytest

from wellvolpos.core import (
    AreaDepth,
    MEFS_RUNGS,
    class_summary,
    group_trials,
    mefs_readout,
    split_trials,
)
from wellvolpos.core.rose import commercial_chance

from .conftest import ENTRY, EXIT


@pytest.fixture(scope="module")
def readout(reduced):
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    g = group_trials(reduced, ENTRY, EXIT)
    vc = split_trials(reduced, ad, g, ENTRY, EXIT)
    return mefs_readout(vc, g, class_summary(vc, g), 14.0), g, vc


def test_the_ladder_and_the_probability_cannot_contradict_each_other(readout):
    """If the P50 clears the line, the chance of clearing it is at least a half.

    The ladder and ``p_exceeds`` are computed from the same trials by two different
    routes -- percentiles of the values against a mean of a boolean -- so agreement is
    a real check rather than a restatement. A wrong conditioning mask on either side
    breaks it, which is exactly how the comparison table came to disagree with tab ④
    on 2026-08-14.
    """
    r, _, _ = readout
    assert r.concepts, "no concepts read"
    for c in r.concepts:
        # A rung that clears the line is a volume at least this likely to be exceeded.
        for rung, floor in (("p90", 0.90), ("p50", 0.50), ("p10", 0.10)):
            if c.clears(rung):
                assert c.p_exceeds >= floor - 1e-9, (
                    f"{c.label}: {rung.upper()} clears MEFS but P(>MEFS) is "
                    f"{c.p_exceeds:.3f}, below {floor}")
            else:
                assert c.p_exceeds <= floor + 1e-9, (
                    f"{c.label}: {rung.upper()} does not clear MEFS but P(>MEFS) is "
                    f"{c.p_exceeds:.3f}, above {floor}")


def test_the_rungs_are_in_petroleum_order_and_the_mean_sits_inside_them(readout):
    r, _, _ = readout
    for c in r.concepts:
        v = c.volumes
        assert v["p90"] <= v["p50"] <= v["p10"] + 1e-9, (c.label, v)
        assert v["p90"] <= v["mean"] <= v["p10"] + 1e-9, (c.label, v)


def test_the_volumes_are_the_ones_already_on_screen(reduced):
    """Passed in from ``class_summary``, never recomputed.

    Recomputing them is how a table comes to disagree with the figure beside it, and
    the difference would be in the *conditioning* rather than the arithmetic -- so it
    would look like a rounding issue and be a population issue.
    """
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    g = group_trials(reduced, ENTRY, EXIT)
    vc = split_trials(reduced, ad, g, ENTRY, EXIT)
    cs = class_summary(vc, g)
    r = mefs_readout(vc, g, cs, 14.0)
    for c in r.concepts:
        for rung in MEFS_RUNGS:
            assert c.volumes[rung] == pytest.approx(cs[c.key][rung], abs=0, rel=0)


def test_each_concept_carries_its_own_conditioning_population(readout):
    """The four rows are not on one footing, and the counts prove it.

    The attic lives in charged dry holes and the unproven volume in the trials where
    the well left the reservoir in hydrocarbons -- different events, different
    denominators. A reader who sums or ranks these without the condition is comparing
    probabilities of different things.
    """
    r, g, _ = readout
    counts = {c.key: c.n for c in r.concepts}
    assert counts["discovery"] == int(np.asarray(g.discovery).sum())
    assert counts["proven"] == int(np.asarray(g.discovery).sum())
    assert counts["below_lkh"] == int(np.asarray(g.hc_to_exit).sum())
    assert counts["attic_dry_hole"] == int(np.asarray(g.dry_with_attic).sum())
    # And they really are different, so the distinction is not academic here.
    assert len(set(counts.values())) > 1, counts


def test_the_well_associated_reading_is_roses_pmcfs(reduced):
    """``P(well associated > MEFS | discovery)`` is exactly Rose's ``Pmcfs(well)``.

    Two modules compute it -- ``core.rose`` for the commercial chance and ``core.mefs``
    for the readout -- and if they ever disagree, one of tab ④'s two new blocks is
    quoting a number the other denies.
    """
    from wellvolpos.core import p_well as p_well_fn

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    g = group_trials(reduced, ENTRY, EXIT)
    vc = split_trials(reduced, ad, g, ENTRY, EXIT)
    chance = p_well_fn(reduced, ENTRY, 0.7605)

    r = mefs_readout(vc, g, class_summary(vc, g), 14.0)
    cc = commercial_chance(reduced, g, vc.proven, chance.p_well, 14.0)

    assert r.by_key("discovery").p_exceeds == pytest.approx(cc.p_mcfs_downdip, abs=1e-12)
    assert r.by_key("proven").p_exceeds == pytest.approx(cc.p_mcfs_proven, abs=1e-12)
    # And Pc is the product, not a third independent estimate.
    assert cc.pc_well == pytest.approx(chance.p_well * cc.p_mcfs_downdip, abs=1e-12)


def test_the_bracket_names_the_two_rungs_the_threshold_falls_between(readout):
    r, _, _ = readout
    assert "between P90 and P50" in r.by_key("discovery").bracket()
    # The attic never clears 14 MMboe on this file, and the wording says so rather
    # than naming a pair of rungs that do not bracket anything.
    assert "below every rung" in r.by_key("attic_dry_hole").bracket()


def test_the_four_c2_crossings_agree_with_the_curves_the_figure_draws(reduced):
    """4.2's marked crossings and the table under it come from one definition.

    Lars, 2026-08-14. Two lists in two modules is how a caption comes to assert
    something the figure beside it denies, so ``c2_cases`` is shared and this checks
    the crossings really are the heights of the curves at the line.
    """
    import wellvolpos.viz.interactive as I
    from wellvolpos.core import c2_cases, c2_crossings
    from wellvolpos.core.classes import risked_exceedance

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    g = group_trials(reduced, ENTRY, EXIT)
    vc = split_trials(reduced, ad, g, ENTRY, EXIT)
    pos, pw, mefs = 0.7605, 0.4576, 14.0

    cx = c2_crossings(reduced, g, vc, pos, pw, mefs)
    assert len(cx) == 4

    # The risked height is the conditional one scaled by the case's own chance --
    # risking moves the probability, never the volume.
    for c in cx:
        assert c.risked == pytest.approx(c.chance * c.conditional, abs=1e-12)
        assert 0.0 <= c.conditional <= 1.0

    # The up-dip case carries its own chance: dry but charged, POS - P_well.
    updip = next(c for c in cx if c.name == "Up-dip volume")
    assert updip.chance == pytest.approx(max(pos - pw, 0.0), abs=1e-12)
    assert updip.chance != pytest.approx(pw, abs=1e-6)

    # And each number is the height of the curve the figure actually plots.
    fig = I.pfig_c2_exceedance(reduced, g, vc, pos_prospect=pos, p_well=pw, mefs=mefs)
    marks = {t.name: t for t in fig.data
             if t.name and t.name.endswith("at MEFS") and t.mode == "markers"}
    assert len(marks) >= 1
    values = {n: vals for n, vals, _, _ in c2_cases(reduced, g, vc, pos, pw)}
    for c in cx:
        v, pct = risked_exceedance(values[c.name], 1.0)
        assert float(np.interp(mefs, v, pct)) == pytest.approx(c.conditional * 100.0, abs=0.5)


def test_c3_draws_the_same_eight_numbers_the_table_reports(reduced):
    """4.3's bars and the table under it are one calculation, both backends.

    The figure exists because 4.2 cannot label these crossings -- three of the four
    conditional ones sit within half a point of each other -- so the risk it carries
    is being a *second* calculation that drifts from the marks it was built to
    replace. It reads ``c2_crossings`` and nothing else.
    """
    import wellvolpos.viz.figures as F
    import wellvolpos.viz.interactive as I
    from wellvolpos.core import c2_crossings

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    g = group_trials(reduced, ENTRY, EXIT)
    vc = split_trials(reduced, ad, g, ENTRY, EXIT)
    pos, pw, mefs = 0.7605, 0.4576, 14.0
    kw = dict(pos_prospect=pos, p_well=pw, mefs=mefs)

    cx = c2_crossings(reduced, g, vc, pos, pw, mefs)
    fig = I.pfig_c3_mefs_bars(reduced, g, vc, **kw)
    assert len(fig.data) == 2, [t.name for t in fig.data]

    # Reversed on the figure, so the widest concept sits at the top of a horizontal
    # bar chart -- the nesting reads outside-in the way 4.2's braces do.
    expected = list(reversed(cx))
    unrisked, risked = fig.data
    assert list(unrisked.y) == [c.name for c in expected]
    for drawn, c in zip(unrisked.x, expected):
        assert float(drawn) == pytest.approx(c.conditional * 100.0, abs=1e-9)
    for drawn, c in zip(risked.x, expected):
        assert float(drawn) == pytest.approx(c.risked * 100.0, abs=1e-9)

    # A probability axis is pinned, so two prospects can be compared by eye.
    assert tuple(fig.layout.xaxis.range) == (0, 108)
    # Grouped, never stacked: the risked bar is not a *part* of the unrisked one, so
    # a stack would draw a sum that means nothing.
    assert fig.layout.barmode == "group"

    # And the export twin draws the same eight.
    _f, ax = F.fig_c3_mefs_bars(reduced, g, vc, **kw)
    widths = sorted(round(p.get_width(), 6) for p in ax.patches)
    wanted = sorted(round(v * 100.0, 6)
                    for c in cx for v in (c.conditional, c.risked))
    assert widths == pytest.approx(wanted, abs=1e-6)
