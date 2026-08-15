"""The commercial volume class, and 3.1's log share axis.

The commercial class is the accumulation **given it clears MEFS** — the distribution
that belongs to Rose's ``Pc``. It is the one place in this app where a threshold
touches a distribution, so these tests pin the two things that keeps honest: the four
existing classes stay untruncated, and the new one's chance really is ``Pc``.
"""

import matplotlib
import numpy as np
import pytest

# Agg, as in every other test that draws: the default backend tries Tk and
# fails on a machine with no display.
matplotlib.use("Agg")

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


# ------------------------------------------------------------------- the wedge
def test_the_wedge_stands_at_full_thickness_up_dip_and_pinches_to_zero():
    """C4 draws the geometry ``core/reservoir.py`` inverts, and it has to be right.

    A schematic can still be wrong. The two claims the figure makes -- full reservoir
    thickness up-dip, zero where the top surface meets the contact -- are exactly the
    ones the whole proven / unproven split rests on, so they are measured off the
    drawn polygon rather than assumed from the code that drew it.
    """
    import numpy as np
    import wellvolpos.viz.interactive as I

    T, zc = 50.0, 2255.0
    fig = I.pfig_c4_wedge(thickness=T, z_contact=zc, z_entry=2205.0, z_exit=2255.0,
                          apex=2064.0)
    wedge = next(t for t in fig.data if "wedge" in (t.name or ""))
    thick = np.asarray(wedge.customdata, dtype=float)

    assert np.nanmax(thick) == pytest.approx(T, abs=1e-6), "not full thickness up-dip"
    assert np.nanmin(thick) >= 0.0
    assert np.nanmin(thick) < 0.02 * T, "the wedge never pinches out"

    # Area-averaged pay is strictly less than T, which is the figure's whole point.
    bars = {t.name: t for t in fig.data if t.name and " = " in t.name}
    pay = next(v for k, v in bars.items() if "Area-averaged" in k)
    tee = next(v for k, v in bars.items() if "thickness" in k)
    pay_m = abs(float(pay.y[0]) - float(pay.y[1]))
    t_m = abs(float(tee.y[0]) - float(tee.y[1]))
    assert t_m == pytest.approx(T, abs=1e-6)
    assert 0.0 < pay_m < t_m, (pay_m, t_m)


def test_the_wedge_obeys_the_depth_rule():
    """Depth on y, increasing downward — it is a section, so the rule applies."""
    import wellvolpos.viz.interactive as I

    fig = I.pfig_c4_wedge(thickness=50.0, z_contact=2255.0, apex=2064.0)
    lo, hi = fig.layout.yaxis.range
    assert lo > hi, "depth must increase downward"
    assert "TVDSS" in fig.layout.yaxis.title.text


def test_the_wedge_has_an_export_twin_that_draws_the_same_thing():
    import numpy as np
    import wellvolpos.viz.figures as F
    import wellvolpos.viz.interactive as I

    kw = dict(thickness=50.0, z_contact=2255.0, z_entry=2205.0, z_exit=2255.0,
              apex=2064.0)
    _fig, ax = F.fig_c4_wedge(**kw)
    labels = " | ".join(str(t.get_label()) for t in ax.lines + list(ax.collections))
    assert "Area-averaged pay" in labels and "Reservoir thickness" in labels, labels
    lo, hi = ax.get_ylim()
    assert lo > hi
    # Both backends report the same averaged pay, to the metre.
    p_int = next(t for t in I.pfig_c4_wedge(**kw).data
                 if t.name and "Area-averaged" in t.name)
    assert f"{abs(float(p_int.y[0]) - float(p_int.y[1])):,.0f} m" in labels
