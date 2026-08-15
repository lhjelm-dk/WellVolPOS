"""The commercial volume class, and 3.1's log share axis.

The commercial class is the accumulation **given it clears MEFS** — the distribution
that belongs to Rose's ``Pc``. It is the one place in this app where a threshold
touches a distribution, so these tests pin the two things that keeps honest: the four
existing classes stay untruncated, and the new one's chance really is ``Pc``.
"""

import numpy as np
import pytest

from wellvolpos.core import (
    AreaDepth,
    c2_cases,
    c2_crossings,
    class_summary,
    group_trials,
    p_well,
    run_sweep,
    split_trials,
)
from wellvolpos.core.rose import commercial_chance
import wellvolpos.viz.figures as F
import wellvolpos.viz.interactive as I

from .conftest import ENTRY, EXIT

POS, MEFS = 0.7605, 14.0


@pytest.fixture(scope="module")
def parts(reduced):
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    g = group_trials(reduced, ENTRY, EXIT)
    vc = split_trials(reduced, ad, g, ENTRY, EXIT)
    ch = p_well(reduced, ENTRY, POS)
    cc = commercial_chance(reduced, g, vc.proven, ch.p_well, MEFS)
    cs = class_summary(vc, g, mefs=MEFS, resource=reduced.col("resource"))
    return reduced, g, vc, ch, cc, cs


def test_the_commercial_chance_is_exactly_pc(parts):
    """Its share of the discovery group times P_well is Rose's Pc, to the bit.

    If these two ever disagree, tab ④ is showing a distribution under a chance that
    does not belong to it — which is the recurring bug of this project wearing a new
    hat.
    """
    _ts, _g, _vc, ch, cc, cs = parts
    share = cs["commercial"]["n"] / cs["discovery"]["n"]
    assert share == pytest.approx(cc.p_mcfs_downdip, abs=1e-12)
    assert share * ch.p_well == pytest.approx(cc.pc_well, abs=1e-12)


def test_the_threshold_raises_the_surviving_mean(parts):
    """Longley's point, made visible rather than hidden.

    A volume cut-off raises the unrisked mean while lowering the chance. Both halves
    have to be on screen together, and this asserts the direction so a future change
    cannot quietly report a commercial mean *below* the well-associated one.
    """
    *_, cs = parts
    assert cs["commercial"]["mean"] > cs["discovery"]["mean"]
    assert cs["commercial"]["p90"] > cs["discovery"]["p90"]
    assert cs["commercial"]["n"] < cs["discovery"]["n"]


def test_the_other_four_classes_are_not_truncated(reduced):
    """The rule is that MEFS is never applied to the existing distributions.

    Adding a class conditional on clearing it must not change any of them, so this
    compares the summary with and without the threshold, key by key.
    """
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    g = group_trials(reduced, ENTRY, EXIT)
    vc = split_trials(reduced, ad, g, ENTRY, EXIT)
    plain = class_summary(vc, g)
    withm = class_summary(vc, g, mefs=MEFS, resource=reduced.col("resource"))
    assert set(withm) - set(plain) == {"commercial"}
    for key, stats in plain.items():
        for stat, value in stats.items():
            assert withm[key][stat] == pytest.approx(value, abs=0), (key, stat)


def test_the_commercial_class_stays_off_the_mefs_bar_chart(parts):
    """Its own chance of exceeding MEFS is 1.0 by construction, so a row would say
    nothing. ``c2_crossings`` must keep to the four nesting concepts."""
    ts, g, vc, ch, cc, _cs = parts
    assert len(c2_cases(ts, g, vc, POS, ch.p_well)) == 4
    assert len(c2_cases(ts, g, vc, POS, ch.p_well,
                        mefs=MEFS, pc_well=cc.pc_well)) == 5
    assert len(c2_crossings(ts, g, vc, POS, ch.p_well, MEFS)) == 4


def test_4_2_and_4_5_draw_the_commercial_class_in_both_backends(parts):
    ts, g, vc, ch, cc, _cs = parts
    kw = dict(pos_prospect=POS, p_well=ch.p_well, mefs=MEFS, pc_well=cc.pc_well)
    names = [t.name for t in I.pfig_c2_exceedance(ts, g, vc, **kw).data if t.name]
    assert any("Commercial accumulation" in n for n in names), names
    labels = [t.get_label() for t in
              F.fig_c2_exceedance(ts, g, vc, **kw)[1].lines]
    assert any("Commercial accumulation" in str(l) for l in labels), labels

    a6 = I.pfig_a6_overlap(vc, g, ts=ts, mefs=MEFS)
    assert any("Commercial" in (t.name or "") for t in a6.data), [t.name for t in a6.data]


def test_the_two_readings_on_4_2_toggle_independently(parts):
    """Plotly's legend groups by concept, so it could only hide both at once."""
    ts, g, vc, ch, cc, _cs = parts
    kw = dict(pos_prospect=POS, p_well=ch.p_well, mefs=MEFS, pc_well=cc.pc_well)
    both = I.pfig_c2_exceedance(ts, g, vc, **kw)
    cond = I.pfig_c2_exceedance(ts, g, vc, show_unconditional=False, **kw)
    uncond = I.pfig_c2_exceedance(ts, g, vc, show_conditional=False, **kw)
    lines = lambda f: [t for t in f.data if t.mode == "lines" and t.name]
    assert len(lines(cond)) < len(lines(both))
    assert len(lines(uncond)) < len(lines(both))
    assert all("risked" not in (t.name or "") for t in lines(cond))
    # And the risked-only view keeps the same number of concepts.
    assert len(lines(cond)) == len(lines(uncond))


# ------------------------------------------------------------- 3.1's log axis
def test_3_1_on_a_log_axis_gives_up_stacking_and_says_so(reduced):
    """Cumulative bands are addition, which a log scale does not preserve.

    So the log mode draws each outcome's own share instead. That is a different
    reading of the same numbers, and the subtitle has to carry it or the figure
    silently changes meaning between two settings of one radio button.
    """
    sweep = run_sweep(reduced, POS, z_gap=50.0)
    lin = I.pfig_a2_outcome_tree(sweep, current_z=ENTRY, share_scale="linear")
    log = I.pfig_a2_outcome_tree(sweep, current_z=ENTRY, share_scale="log")

    assert lin.layout.xaxis.type != "log"
    assert tuple(lin.layout.xaxis.range) == (0, 100)

    assert log.layout.xaxis.type == "log"
    lo, hi = log.layout.xaxis.range
    assert 10 ** lo == pytest.approx(1.0, rel=1e-6)
    assert 10 ** hi == pytest.approx(110.0, rel=1e-6)
    assert "cannot stack" in log.layout.title.text
    assert "log scale" in log.layout.xaxis.title.text

    # The log traces are shares, not a running total: none may exceed 100 %.
    for t in log.data:
        if t.x is not None and t.name:
            assert np.nanmax(np.asarray(t.x, dtype=float)) <= 100.0 + 1e-9, t.name


def test_an_unknown_share_scale_raises(reduced):
    """A silent fallback would answer a different question under the label chosen."""
    sweep = run_sweep(reduced, POS, z_gap=50.0)
    with pytest.raises(ValueError, match="share_scale"):
        I.pfig_a2_outcome_tree(sweep, share_scale="probit")
