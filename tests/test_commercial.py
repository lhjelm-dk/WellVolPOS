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
def test_3_1_stacks_on_both_scales_and_says_what_the_log_axis_costs(reduced):
    """**A log axis changes the scale, not the quantity** (corrected 2026-08-18).

    This test used to assert the opposite, and the reasoning behind it was wrong:
    stacking is addition and a log scale does not preserve addition — true, and
    beside the point, because a band spans the *interval* between two cumulative
    boundaries and an interval is perfectly well defined on a log axis. What the log
    scale destroys is only that a band's width on screen equals its share.

    Drawing each outcome's own share instead meant a reader toggling an axis got a
    different quantity, which is exactly what Lars reported seeing.
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
    assert "log scale" in log.layout.xaxis.title.text
    # The caveat that replaced the wrong one: widths are not shares here.
    assert "read the boundaries" in log.layout.title.text

    # Four stacked bands on both, with the same names in the same order.
    def bands(fig):
        return [t.name for t in fig.data if t.fill == "toself" and t.name]

    assert bands(log) == bands(lin)
    assert len(bands(log)) == 4

    # The outermost boundary is the running total and reaches 100 % on both.
    for fig in (lin, log):
        outer = [t for t in fig.data if t.fill == "toself"][-1]
        assert np.nanmax(np.asarray(outer.x, dtype=float)) == pytest.approx(100.0, abs=1e-6)

    # The one real compromise: the shallowest band is clipped at the axis floor,
    # because it starts at zero and a log axis has no zero.
    inner = [t for t in log.data if t.fill == "toself"][0]
    assert np.nanmin(np.asarray(inner.x, dtype=float)) == pytest.approx(1.0)


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

    # **The thickness panel makes the same two claims as a graph** (Lars's sketch,
    # 2026-08-18), so they are measured there too. The charged curve equals T up-dip
    # and reaches zero at the contact, and it does so by falling *monotonically*:
    # a curve that wandered would not be the wedge.
    curve = next(t for t in fig.data if t.name == "Charged thickness at this depth")
    cy = np.asarray(curve.y, dtype=float)
    cx = np.asarray(curve.x, dtype=float)
    order = np.argsort(cy)
    cx = cx[order]
    cy = cy[order]
    assert cx[0] == pytest.approx(T, abs=1e-9), "not full thickness up-dip"
    assert cx[-1] == pytest.approx(0.0, abs=1e-9), "does not pinch out"
    assert np.all(np.diff(cx) <= 1e-9), "the charged thickness is not monotonic"
    # It leaves T exactly one thickness above the contact -- that is the taper.
    departs = float(cy[np.argmax(cx < T - 1e-9)])
    assert departs == pytest.approx(zc - T, abs=(cy[1] - cy[0]) * 1.01)

    # Area-averaged pay is strictly less than T, which is the figure's whole point.
    rules = {t.name: t for t in fig.data if t.name and " = " in t.name}
    pay = next(v for k, v in rules.items() if "pay" in k.lower())
    tee = next(v for k, v in rules.items() if "Reservoir thickness" in k)
    pay_m = float(np.asarray(pay.x, dtype=float)[0])
    t_m = float(np.asarray(tee.x, dtype=float)[0])
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
    _fig, axes = F.fig_c4_wedge(**kw)
    # Both panels: the section on the left, the thickness graph on the right.
    assert len(axes) == 2
    labels = " | ".join(str(t.get_label())
                        for ax in axes
                        for t in list(ax.lines) + list(ax.collections))
    assert "pay" in labels.lower() and "Reservoir thickness" in labels, labels
    assert "Charged thickness at this depth" in labels, labels
    for ax in axes:
        lo, hi = ax.get_ylim()
        assert lo > hi, "depth axis not inverted"

    # Both backends report the same averaged pay, to the metre.
    p_int = next(t for t in I.pfig_c4_wedge(**kw).data
                 if t.name and "pay" in t.name.lower())
    assert f"{float(np.asarray(p_int.x, dtype=float)[0]):,.0f} m" in labels


# --------------------------------------------------------- C5, restored and fixed
def test_c5_refuses_a_dry_hole_rather_than_drawing_half_a_figure(reduced):
    """The bug that made C5 look broken, now an error.

    It was called with the median *successful* contact — 2203.3 m on prospect B against
    a 2205 m entry — so the figure was drawn for a dry hole, nothing sat below either
    cut, and only the upper half of each panel appeared. It looked like a styling
    problem and was a data one.
    """
    import wellvolpos.viz.interactive as I

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    with pytest.raises(ValueError, match="dry hole"):
        I.pfig_c5_partitions(ad, z_entry=3500.0, z_exit=3550.0, z_contact=3499.0)


def test_c5_draws_all_four_regions_on_a_discovery_contact(reduced):
    import numpy as np
    import wellvolpos.viz.interactive as I

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    c, r = reduced.col("contact"), reduced.col("resource")
    zc = float(np.median(c[(r > 0) & (c > 3500.0)]))
    # On prospect A the median discovery contact is 3,527.7 m, *above* the 3,550 m
    # exit — so LKH = the contact and there is genuinely nothing unproven below it.
    # Three regions is the correct answer there, and the figure must not invent a
    # fourth. Rose still gets two, because his cut is the entry.
    fig = I.pfig_c5_partitions(ad, z_entry=3500.0, z_exit=3550.0, z_contact=zc)
    names = [t.name for t in fig.data if t.name]
    assert names[:3] == ["Rose updip", "Rose downdip", "Proven"]
    assert ("Unproven below LKH" in names) == (zc > 3550.0)

    # A contact below the exit does produce all four.
    deep = I.pfig_c5_partitions(ad, z_entry=3500.0, z_exit=3550.0, z_contact=3620.0)
    assert [t.name for t in deep.data if t.name] == [
        "Rose updip", "Rose downdip", "Proven", "Unproven below LKH"]

    # Every region drawn has real extent — the failure mode was one drawing nothing.
    for t in deep.data:
        y = np.asarray(t.y, dtype=float)
        assert y.max() - y.min() > 1.0, t.name


def test_4_2_carries_a_brace_for_the_commercial_class(reduced):
    """Lars: "there are 4 bars but the commercial is missing". It was in ``spans`` and
    dropped by a hardcoded order list."""
    import wellvolpos.viz.interactive as I
    from wellvolpos.core import p_well as p_well_fn
    from wellvolpos.core.rose import commercial_chance

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    g = group_trials(reduced, ENTRY, EXIT)
    vc = split_trials(reduced, ad, g, ENTRY, EXIT)
    ch = p_well_fn(reduced, ENTRY, POS)
    cc = commercial_chance(reduced, g, vc.proven, ch.p_well, MEFS)

    fig = I.pfig_c2_exceedance(reduced, g, vc, pos_prospect=ch.pos_prospect,
                               p_well=ch.p_well, mefs=MEFS, pc_well=cc.pc_well)
    labels = [a.text.strip() for a in fig.layout.annotations if a.text]
    for name in ("Commercial accumulation", "Up-dip volume", "Resource tested by well",
                 "Well associated resource potential", "Prospect resource potential"):
        assert name in labels, (name, labels)
    # And the axis grew to hold five braces rather than clipping the last.
    assert fig.layout.yaxis.range[0] < -40


def test_the_commercial_chance_is_p_well_times_the_conditional(reduced, groups, area_depth):
    """``Pc = P_well x P(>MEFS | discovery)``, and the table's row must say so.

    The commercial row on tab ④ carries its own chance, computed as P_well times the
    share of *discoveries* that clear MEFS. Computing it as the share of all trials
    would give the same number only under the "trials are risked" convention, and
    would be the recurring bug — an unrisked count under a risked label — the sixth
    time.
    """
    res = np.asarray(reduced.col("resource"), dtype=float)
    disc = np.asarray(groups.discovery, dtype=bool)
    mefs = 14.0

    commercial = disc & (res > mefs)
    p_mcfs = commercial.sum() / disc.sum()
    p_well = 0.7605 * (disc.sum() / (res > 0).sum())
    pc = p_well * p_mcfs

    # The same number the other way round: the share of *all* trials that are both a
    # discovery here and above MEFS, scaled from the file's own POS onto the app's.
    direct = commercial.sum() / (res > 0).sum() * 0.7605
    assert pc == pytest.approx(direct, rel=1e-12)

    # And it is strictly below P_well, because clearing MEFS is a further condition.
    assert 0.0 < pc < p_well


def test_the_commercial_mean_exceeds_the_well_associated_mean(reduced, groups, area_depth):
    """Longley's point, made visible rather than hidden.

    A volume cut-off raises the unrisked mean while lowering commercial chance. That
    is precisely why this app never truncates the other distributions — and why the
    commercial class, which *is* conditional on clearing the threshold, must come out
    higher. If it ever did not, the class would be selecting the wrong trials.
    """
    res = np.asarray(reduced.col("resource"), dtype=float)
    disc = np.asarray(groups.discovery, dtype=bool)
    mefs = 14.0
    commercial = disc & (res > mefs)
    assert commercial.sum() > 10
    assert res[commercial].mean() > res[disc].mean()
