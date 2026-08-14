"""The plotly figures: the depth rule, colour by meaning, and level rows.

The non-negotiables in CLAUDE.md are backend-independent, so they have to be
enforced on the interactive path as well as the export path -- otherwise the
half the user actually looks at is the unchecked half. ``test_axes.py`` covers
the matplotlib side of the same rules.

Numerical correctness lives in test_sweep.py / test_chance.py / test_groups.py;
these figures are drawing, not arithmetic.
"""

import numpy as np
import pytest

from wellvolpos.core.chance import r_location
from wellvolpos.core.classes import (
    class_percentiles,
    conditional_exceedance,
    risked_exceedance,
    split_trials,
)
from wellvolpos.core.sweep import run_sweep, run_volume_sweep
from wellvolpos.viz import interactive as I
from wellvolpos.viz.theme import (
    PANEL_HEIGHT,
    colour,
    is_depth_axis_correct_plotly,
    palette,
)

from .conftest import ENTRY, EXIT

POS = 0.7605
TABLE = {"charge": 0.92, "trap": 0.94, "reservoir": 0.95, "retention": 0.93}
ZROW = (3350.0, 3700.0)


@pytest.fixture(scope="module")
def sweep(reduced):
    return run_sweep(reduced, POS, n=30, z_gap=EXIT - ENTRY)


@pytest.fixture(scope="module")
def vsweep(reduced, area_depth):
    return run_volume_sweep(reduced, area_depth, POS, n=15, mefs=14.0, z_gap=EXIT - ENTRY)


@pytest.fixture(scope="module")
def vc(reduced, area_depth, groups):
    return split_trials(reduced, area_depth, groups, ENTRY, EXIT)


@pytest.fixture(scope="module")
def r(reduced):
    return r_location(reduced, ENTRY)[0]


def _traces(fig, name):
    return [t for t in fig.data if getattr(t, "name", None) == name]


def _line_colour(fig, name):
    tr = _traces(fig, name)
    assert tr, f"no trace named {name!r}; have {[t.name for t in fig.data]}"
    return tr[0].line.color


# --------------------------------------------------------------- depth rule
def _depth_figures(ad, sweep, vsweep, reduced):
    """Every plotly figure that carries a depth, with a shared zlim applied."""
    return {
        "A1": I.pfig_a1_area_depth(ad, current_entry=ENTRY, zlim=ZROW),
        "A2": I.pfig_a2_outcome_tree(sweep, current_z=ENTRY, zlim=ZROW),
        "A3": I.pfig_a3_chance_decomposition(sweep, pos_prospect=POS, zlim=ZROW),
        "A4": I.pfig_a4_resource_vs_depth(reduced, current_entry=ENTRY, zlim=ZROW),
        "B0": I.pfig_b0_section(ad, z_entry=ENTRY, z_exit=EXIT, zlim=ZROW),
        "B1": I.pfig_b1_volume_split(vsweep, zlim=ZROW),
        "B2": I.pfig_b2_chance_vs_regret(vsweep, zlim=ZROW),
        "B3": I.pfig_b3_uncertainty_reduction(sweep, zlim=ZROW),
    }


def test_every_depth_carrying_figure_puts_depth_on_y_inverted(area_depth, sweep, vsweep, reduced):
    for name, fig in _depth_figures(area_depth, sweep, vsweep, reduced).items():
        assert is_depth_axis_correct_plotly(fig), name


def test_a_shared_zlim_is_honoured_exactly_by_every_panel(area_depth, sweep, vsweep, reduced):
    """The point of the rule: same height on the page is the same depth.

    Passing one range must give byte-identical axis ranges, or a row cannot be
    read across even though each panel individually looks right.
    """
    ranges = {
        name: tuple(fig.layout.yaxis.range)
        for name, fig in _depth_figures(area_depth, sweep, vsweep, reduced).items()
    }
    assert len(set(ranges.values())) == 1, ranges
    assert set(ranges.values()) == {(ZROW[1], ZROW[0])}     # descending = inverted


def test_a_figures_bottom_margin_grows_with_its_legend(area_depth, sweep, vsweep, reduced):
    """The reserved space is sized to the legend, not fixed (Lars, 2026-08-12).

    A fixed 125 px clipped real legends: measured in the app, B1's six entries wrapped
    to six rows in a three-column layout and lost 75 px off the bottom. Streamlit
    charts are width-responsive, so how many rows a horizontal legend takes is decided
    in the browser and cannot be known here -- which is why
    :func:`~wellvolpos.viz.theme.legend_margin` reserves for the worst case of one
    entry per row.

    The cost is paid in figure *height*, not in plot area, so a figure with twelve
    series is taller than one with two and both have the same room to draw in.
    """
    from wellvolpos.viz.theme import _has_colourbar, legend_entries, legend_margin

    figs = _depth_figures(area_depth, sweep, vsweep, reduced)
    for name, fig in figs.items():
        n = legend_entries(fig)
        want = legend_margin(n, colourbar=_has_colourbar(fig))
        assert fig.layout.margin.b == want, name
        # ...and the figure grew to pay for it, rather than the plot area shrinking.
        assert fig.layout.height == PANEL_HEIGHT + max(0, want - legend_margin(3)), name

    # Since the margin now depends on the series count, panels genuinely differ --
    # which is the reason level_row exists rather than a defect.
    assert len({fig.layout.margin.b for fig in figs.values()}) > 1


def test_level_row_makes_a_row_share_one_plot_area(area_depth, sweep, vsweep, reduced):
    """Equal ranges are not enough — the plot *areas* have to match too.

    An identical y-range still lands a given depth on a different pixel row if one
    panel's axes are inset further than another's. That used to be guaranteed by
    fixing every margin to the same number; now the margin follows the legend, so the
    sharing has to be *imposed* on a row after its figures are built.

    ``level_row`` is that step, and it is called from the same place as ``row_zlim``
    for the same reason. This asserts the mechanism rather than an accident: build
    three panels that differ, level them, and every one of the three properties a
    level row needs must hold.
    """
    from wellvolpos.viz.theme import level_row

    figs = list(_depth_figures(area_depth, sweep, vsweep, reduced).values())[:3]
    assert len({f.layout.margin.b for f in figs}) > 1, "need panels that differ to test this"

    level_row(*figs)
    margins = {(m.l, m.r, m.t, m.b, m.autoexpand)
               for m in (f.layout.margin for f in figs)}
    assert len(margins) == 1, margins
    assert next(iter(margins))[-1] is False, \
        "autoexpand must be off or a legend can shift the axes"
    assert len({f.layout.height for f in figs}) == 1
    # The row takes the largest of each, so nothing that fitted before is clipped now.
    assert next(iter(margins))[3] == max(
        __import__("wellvolpos.viz.theme", fromlist=["legend_margin"]).legend_margin(
            __import__("wellvolpos.viz.theme", fromlist=["legend_entries"]).legend_entries(f)
        ) for f in figs
    )


def test_every_legend_sits_below_the_x_axis_at_the_shared_height(
    area_depth, sweep, vsweep, reduced
):
    """Legends go **below the x-axis title**, at one shared height (Lars, 2026-08-12).

    They used to sit inside the axes, top-right, where they covered curves. Moving
    them out is only safe because ``apply_plotly`` reserves the room on *every*
    figure: ``autoexpand`` is off, so plotly will not grow a margin to fit a legend,
    and a panel that acquired one would otherwise shrink its own plot area and take
    the row out of level. Reserving it uniformly is what keeps a given depth on the
    same pixel row across a row of panels.

    It is **anchored to the figure, not to the plot area**. With plotly's default
    ``yref="paper"`` the y is a fraction of the *plot* height, so the gap below the
    axis grew with the plot and a legend that fitted on a short figure ran off a tall
    one -- three still clipped after the margin was made legend-aware.
    ``yref="container"`` measures from the figure edge, so the legend sits at the
    bottom and grows upward into the reserved margin. That is what makes it
    impossible to clip, and it is what this asserts.
    """
    for name, fig in _depth_figures(area_depth, sweep, vsweep, reduced).items():
        lg = fig.layout.legend
        assert lg.orientation == "h", f"{name} legend is not horizontal"
        assert lg.yref == "container", f"{name} legend yref={lg.yref!r}, expected 'container'"
        assert lg.yanchor == "bottom", f"{name} legend yanchor={lg.yanchor!r}"
        assert 0.0 < lg.y < 0.1, f"{name} legend y={lg.y} is not near the figure bottom"
        # ...and the margin actually reserves the space it now needs.
        assert fig.layout.margin.b >= 100, f"{name} bottom margin {fig.layout.margin.b} is too small"
        assert fig.layout.margin.autoexpand is False, name


def test_repeated_depth_labels_can_be_suppressed_for_later_panels(sweep):
    first = I.pfig_a3_chance_decomposition(sweep, zlim=ZROW)
    later = I.pfig_a3_chance_decomposition(sweep, zlim=ZROW, show_depth_labels=False)
    assert first.layout.yaxis.showticklabels is not False
    assert later.layout.yaxis.showticklabels is False
    # ...but the range still matches, so hiding labels cannot desynchronise a row
    assert tuple(first.layout.yaxis.range) == tuple(later.layout.yaxis.range)


def test_figures_without_a_depth_do_not_pretend_to_have_one(reduced, groups, vc, r):
    for name, fig in {
        "A5": I.pfig_a5_exceedance(reduced, groups, vc),
        "A6": I.pfig_a6_overlap(vc, groups),
        "B4": I.pfig_b4_chance_waterfall(TABLE, r, POS),
        "B5": I.pfig_b5_allocation_dumbbell(TABLE, r),
    }.items():
        assert not is_depth_axis_correct_plotly(fig), name


def test_row_zlim_is_the_envelope_and_ignores_absent_panels():
    assert I.row_zlim((3400.0, 3600.0), (3350.0, 3500.0)) == (3350.0, 3600.0)
    assert I.row_zlim((3400.0, 3600.0), None) == (3400.0, 3600.0)
    lo, hi = I.row_zlim((3400.0, 3600.0), pad_frac=0.1)
    assert (lo, hi) == pytest.approx((3380.0, 3620.0))
    with pytest.raises(ValueError):
        I.row_zlim(None)


# ------------------------------------------------------------ colour by role
def test_a3_draws_both_chances_in_blue_separated_by_dash(sweep):
    fig = I.pfig_a3_chance_decomposition(sweep, pos_prospect=POS)
    p_well = _traces(fig, "P<sub>well</sub> = POS × r")[0]
    r_trace = _traces(fig, "r = P(contact deeper | HC)")[0]
    assert p_well.line.color == colour("discovery")
    assert r_trace.line.color == colour("discovery")
    assert p_well.line.dash != r_trace.line.dash


def test_a5_draws_the_prospect_only_in_both_readings(reduced, groups, vc):
    """A5 is the *Prospect* tab's figure and now carries only the prospect (Lars,
    2026-08-11). The other three series say again what C2 draws and what tab 3's
    table tabulates, and three places for one set of numbers is three places to
    disagree -- their populations were verified identical before they were removed.

    Both readings stay: solid conditional from 100 %, dashed unconditional from
    POS_prospect.
    """
    fig = I.pfig_a5_exceedance(reduced, groups, vc, mefs=14.0,
                               pos_prospect=POS, p_well=0.4576)
    curves = [t for t in fig.data if t.mode == "lines" and t.name]
    assert len(curves) == 2
    assert all(t.line.color == colour("prospect") for t in curves)
    tops = sorted(float(np.nanmax(np.asarray(t.y, dtype=float))) for t in curves)
    assert tops == pytest.approx([POS * 100.0, 100.0], abs=0.5)

def test_b1_uses_the_class_colours_and_gives_each_volume_a_ladder(vsweep):
    """Bold mean, dotted P90/P50/P10, one colour per concept (Lars, 2026-08-14).

    The ladder used to be proven's alone, which made a bare bold mean look like the
    answer wherever it stood by itself -- and on a skewed distribution a mean is not
    even the middle.
    """
    fig = I.pfig_b1_volume_split(vsweep)
    assert _line_colour(fig, "Proven | discovery") == colour("proven")
    assert _line_colour(fig, "Attic | dry hole") == colour("attic")
    names = [t.name for t in fig.data if t.name]
    for base in ("Proven", "Attic", "At the well"):
        for tag in ("P90", "P50", "P10"):
            assert any(n.startswith(base) and f" {tag}" in n for n in names), (base, tag)
    # The below-exit volume is NOT here: it is conditional on a different event and
    # has its own figure.
    assert not any("below exit" in n.lower() for n in names), names


def test_b13_draws_the_below_exit_volume_conditionally_with_its_own_ladder(vsweep):
    """Split out of B1 because it is conditional on the well leaving the reservoir in
    hydrocarbons -- so its curves were never on the same footing as proven's."""
    fig = I.pfig_b13_below_exit(vsweep)
    names = [t.name for t in fig.data if t.name]
    assert any(n.startswith("Mean") for n in names), names
    for tag in ("P90", "P50", "P10"):
        assert any(n.startswith(tag) for n in names), tag
    # Every series says what it is conditional on, in its own name.
    assert all("HC seen to the exit" in n for n in names), names
    assert _line_colour(fig, next(n for n in names if n.startswith("Mean")))         == colour("below_lkh")


def test_b2_names_its_conditioning_and_keeps_the_regret_colour(vsweep):
    fig = I.pfig_b2_chance_vs_regret(vsweep)
    names = [t.name for t in fig.data if t.name]
    attic = next(n for n in names if n.startswith("P(attic"))
    assert "charged" in attic
    assert _line_colour(fig, attic) == colour("attic")


def test_a2_bands_use_the_outcome_colours_and_reach_100_percent(sweep):
    """The fills are translucent (Lars, 2026-08-10), so the colour is checked
    through the alpha rather than as a hex string -- the *role* must still be the
    canonical one, which is what non-negotiable 3 is about."""
    from wellvolpos.viz.theme import rgba

    fig = I.pfig_a2_outcome_tree(sweep)
    fills = {t.name: t.fillcolor for t in fig.data if t.fillcolor}
    for name, role in (("Dry, with attic", "attic"),
                       ("Discovery, contact seen", "tested"),
                       ("Discovery, HC to exit", "below_lkh"),
                       ("Chance failure", "muted")):
        assert fills[name] == rgba(role, 0.55), name
        assert "rgba" in fills[name] and "0.55" in fills[name]
    # The band *outlines* keep the solid role colour, so the boundaries stay crisp.
    lines = {t.name: t.line.color for t in fig.data if t.fillcolor}
    assert lines["Dry, with attic"] == colour("attic")
    top = max(float(np.nanmax(t.x)) for t in fig.data if t.x is not None and len(t.x))
    assert top == pytest.approx(100.0, abs=1e-6)


def fig_bars(fig):
    """The waterfall's Bar traces, ignoring the rule and its label."""
    return [t for t in fig.data if t.type == "bar"]


def test_b4_colours_bars_by_element_and_patterns_the_location_share(r):
    """Colour is the element, pattern is the location share (Lars's card, 2026-08-12).

    The standalone location bar under ``scheme="none"`` belongs to no element, so it
    keeps the ``p_well`` colour -- which is also what tells it apart from the ones
    that do.
    """
    from wellvolpos.viz.theme import element_colour

    bars = [t for t in fig_bars(I.pfig_b4_chance_waterfall(TABLE, r, POS, scheme="none"))]
    patterned = [t for t in bars
                 if getattr(t.marker, "pattern", None) and t.marker.pattern.shape == "/"]
    assert len(patterned) == 1
    assert patterned[0].marker.color == colour("p_well")
    faces = [t.marker.color for t in bars]
    for el in ("charge", "trap", "reservoir", "retention"):
        assert element_colour(el) in faces, el


def test_b4_draws_p_well_inside_its_own_log_axis(r):
    """The label must name P_well *and land where P_well is*.

    This used to be an ``add_hline``, and 5.1's y-axis is logarithmic: the rule was
    added before ``type="log"`` was set, so 0.2030 was stored as 0.2030 and then read
    as the exponent -- 10^0.203 = 1.60, above the axis ceiling of 1.2. The line and
    its label were off-scale and never rendered, on the one figure whose whole purpose
    is to total to P_well. The matplotlib twin was correct all along, so nothing
    compared them and noticed.

    The old test asserted only that the *text* existed, which it did. Checking that the
    coordinate is inside the axis range is what would have caught it.
    """
    fig = I.pfig_b4_chance_waterfall(TABLE, r, POS)
    total = POS * r
    said = " ".join(
        "".join(t.text) if isinstance(t.text, (list, tuple)) else (t.text or "")
        for t in fig.data if getattr(t, "mode", None) == "text"
    )
    assert f"{total:.4f}" in said
    assert f"{float(np.prod(list(TABLE.values()))) * r:.4f}" not in said

    rules = [t for t in fig.data
             if getattr(t, "mode", None) == "lines" and t.y is not None and len(t.y) == 2]
    assert len(rules) == 1, "expected exactly one P_well rule"
    lo, hi = (10.0 ** float(v) for v in fig.layout.yaxis.range)
    assert fig.layout.yaxis.type == "log"
    assert lo <= float(rules[0].y[0]) <= hi, "the P_well rule is off its own axis"
    assert float(rules[0].y[0]) == pytest.approx(total)
    # Nothing left that a later axis change could reinterpret.
    assert not [sh for sh in fig.layout.shapes if sh.y0 is not None]


def test_b4_says_so_when_r_is_zero(r):
    fig = I.pfig_b4_chance_waterfall(TABLE, 0.0, POS)
    said = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "r = 0" in said


# ------------------------------------------------------------------- hover
def test_every_curve_carries_a_hovertemplate_in_domain_units(reduced, groups, vc, sweep, vsweep):
    """Reading a probability off a curve is the whole reason for the
    interactive path, so a bare 'trace 0' tooltip is a defect."""
    for name, fig in {
        "A3": I.pfig_a3_chance_decomposition(sweep, pos_prospect=POS),
        "A5": I.pfig_a5_exceedance(reduced, groups, vc),
        "B1": I.pfig_b1_volume_split(vsweep),
        "B2": I.pfig_b2_chance_vs_regret(vsweep),
    }.items():
        shown = [t for t in fig.data if getattr(t, "hoverinfo", None) != "skip"]
        assert shown, name
        for t in shown:
            assert t.hovertemplate, f"{name}: {t.name} has no hovertemplate"


def test_b5_panels_share_one_x_range_so_the_schemes_are_comparable(r):
    fig = I.pfig_b5_allocation_dumbbell(TABLE, r, pos_prospect=POS)
    ranges = {
        tuple(getattr(fig.layout, ax).range)
        for ax in dir(fig.layout)
        if ax.startswith("xaxis") and getattr(fig.layout, ax).range is not None
    }
    assert len(ranges) == 1


# ------------------------------------------------------------------- B6
@pytest.fixture(scope="module")
def vsweep_banded(reduced, area_depth):
    return run_volume_sweep(reduced, area_depth, POS, n=25, mefs=14.0, z_gap=50.0, n_boot=150)


def test_pb6_puts_required_depth_on_y_inverted(vsweep_banded):
    fig = I.pfig_b6_inverse(vsweep_banded, target=20.0)
    assert is_depth_axis_correct_plotly(fig)
    assert "TVDSS" in fig.layout.yaxis.title.text


def _required_entry(fig):
    """B6's requirement curve, found by prefix.

    Its full name carries the reading -- "for a target MEAN PROVEN volume" -- which
    is load-bearing on a figure whose x-axis also carries per-trial volumes, so the
    tests match the prefix rather than pinning the whole label and discouraging it
    from ever saying more.
    """
    return next(t for t in fig.data if (t.name or "").startswith("Required entry"))


def test_pb6_carries_p_well_as_marker_colour_with_a_scale(vsweep_banded):
    """No dual y-axes, so the cost side of the trade is the colour."""
    fig = I.pfig_b6_inverse(vsweep_banded)
    curve = _required_entry(fig)
    assert curve.marker.colorscale is not None
    assert curve.marker.color is not None
    # The rule is "no dual y-axes", which in plotly means no y-axis *overlaying*
    # another. Asserting `"yaxis2" not in layout` instead would flag any figure
    # that ever gains a second panel while leaving a genuine twinned axis
    # undetected -- the wrong thing on both counts.
    for key in fig.layout:
        if key.startswith("yaxis"):
            assert fig.layout[key].overlaying is None, key


def test_pb6_hover_gives_volume_depth_and_chance_together(vsweep_banded):
    fig = I.pfig_b6_inverse(vsweep_banded)
    curve = _required_entry(fig)
    assert "MMboe" in curve.hovertemplate
    assert "TVDSS" in curve.hovertemplate
    assert "P<sub>well</sub>" in curve.hovertemplate


def test_pb6_band_only_appears_when_the_sweep_carried_one(reduced, area_depth, vsweep_banded):
    # It is named "CI on the ..." rather than "band" now: B6 carries *two* shaded
    # regions -- this bootstrap interval, which is sampling error on one estimate,
    # and the P90-P10 contact spread, which is geological range. Calling both a
    # "band" was how they came to be read as the same kind of thing.
    banded = I.pfig_b6_inverse(vsweep_banded)
    assert any("CI on the" in (t.name or "") for t in banded.data)
    plain = I.pfig_b6_inverse(run_volume_sweep(reduced, area_depth, POS, n=15, z_gap=50.0))
    assert not any("band" in (t.name or "") for t in plain.data)


def test_pb6_says_so_when_there_is_nothing_to_invert(reduced, area_depth):
    empty = run_volume_sweep(reduced, area_depth, POS, z_min=4000.0, z_max=4100.0, n=5, z_gap=50.0)
    fig = I.pfig_b6_inverse(empty)
    said = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "invert" in said


def test_pb1_and_pb2_leave_undersupported_steps_undrawn(vsweep_banded):
    b1 = I.pfig_b1_volume_split(vsweep_banded)
    proven = next(t for t in b1.data if t.name == "Proven | discovery")
    assert np.isnan(np.asarray(proven.x, dtype=float)).any()

    b2 = I.pfig_b2_chance_vs_regret(vsweep_banded)
    p_well = next(t for t in b2.data if t.name == "P<sub>well</sub>")
    assert not np.isnan(np.asarray(p_well.x, dtype=float)).any()   # unconditional
    conditional = next(t for t in b2.data if (t.name or "").startswith("P(proven"))
    assert np.isnan(np.asarray(conditional.x, dtype=float)).any()


def test_pb2_names_the_curves_that_meet(vsweep_banded):
    fig = I.pfig_b2_chance_vs_regret(vsweep_banded)
    said = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "chance = regret" not in said
    assert "dry & charged" in said


def test_no_figure_leaves_plotlys_placeholder_annotation_behind(reduced, area_depth, sweep, vsweep):
    """add_hline with annotation_text=None still creates an annotation, and
    plotly fills it with "new text" -- which appeared on every unlabelled
    entry/exit rule in the app."""
    figs = [
        I.pfig_a1_area_depth(area_depth, current_entry=ENTRY, current_exit=EXIT),
        I.pfig_a2_outcome_tree(sweep, current_z=ENTRY),
        I.pfig_a3_chance_decomposition(sweep, pos_prospect=POS, current_z=ENTRY),
        I.pfig_a4_resource_vs_depth(reduced, current_entry=ENTRY, current_exit=EXIT, mefs=14.0),
        I.pfig_b1_volume_split(vsweep, current_z=ENTRY),
        I.pfig_b2_chance_vs_regret(vsweep, current_z=ENTRY),
        I.pfig_b3_uncertainty_reduction(sweep, current_z=ENTRY),
    ]
    for fig in figs:
        for ann in fig.layout.annotations:
            assert ann.text not in (None, "", "new text"), fig.layout.title.text


def test_a4_names_both_well_rules(reduced):
    fig = I.pfig_a4_resource_vs_depth(reduced, current_entry=ENTRY, current_exit=EXIT, mefs=14.0)
    said = [a.text for a in fig.layout.annotations]
    assert "well entry" in said
    assert "well exit" in said


# --------------------------------------------------------------- map view
def test_map_view_is_plan_view_with_no_depth_axis(area_depth):
    """Depth is the *contour label* here, not an axis, so the depth rule does
    not apply -- but equal aspect does, or a contour enclosing twice the area
    would not look it."""
    apex = area_depth.apex_estimate()
    fig = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY, z_exit=EXIT)
    assert not is_depth_axis_correct_plotly(fig)
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1


def test_map_view_puts_the_well_on_its_own_entry_contour(area_depth):
    apex = area_depth.apex_estimate()
    fig = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY, z_exit=EXIT, well_azimuth_deg=0.0)
    well = next(t for t in fig.data if t.name == "Well")
    r_expected = area_depth.radius_at(ENTRY, apex)
    r_drawn = float(np.hypot(well.x[0], well.y[0]))
    assert r_drawn == pytest.approx(r_expected, rel=1e-9)


def test_map_view_marks_the_deepest_sampled_contact(area_depth):
    apex = area_depth.apex_estimate()
    fig = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY, interval=50.0)
    said = fig.layout.title.text
    assert f"{area_depth.deepest:.0f}" in said
    # The outer ring is the deepest contour, and nothing is drawn outside it.
    outer = area_depth.radius_at(area_depth.deepest, apex)
    for t in fig.data:
        if t.x is not None and len(t.x):
            assert np.nanmax(np.abs(np.asarray(t.x, dtype=float))) <= outer + 1e-9


def test_map_view_contour_interval_is_honoured(area_depth):
    apex = area_depth.apex_estimate()
    coarse = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY, interval=100.0)
    fine = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY, interval=25.0)
    assert len(fine.data) > len(coarse.data)


def test_map_view_shades_the_three_areas_the_well_divides_the_closure_into(area_depth):
    """Potential attic up-dip of entry, potentially proven between entry and exit,
    unproven below LKH -- the same split B0 draws in section, so the two figures
    colour-key identically."""
    apex = area_depth.apex_estimate()
    fig = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY, z_exit=EXIT)
    named = {t.name: t for t in fig.data if t.name}
    attic = next(t for n, t in named.items() if n.startswith("Potential attic"))
    proven = next(t for n, t in named.items() if n.startswith("Potentially proven"))
    possible = next(t for n, t in named.items() if n.startswith("Unproven below LKH"))
    assert attic.line.color == colour("attic")
    assert attic.fill == "toself"
    for band in (proven, possible):
        assert band.fill == "toself"
    # Fill colours come from the roles, not from literals.
    assert "rgba" in proven.fillcolor and "rgba" in possible.fillcolor


def test_map_view_areas_sum_to_the_closure_at_the_base(area_depth):
    """The three shaded areas partition the deepest contour's area, so the
    numbers in the legend add up to what the closure holds."""
    apex = area_depth.apex_estimate()
    fig = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY, z_exit=EXIT)
    import re

    shown = [
        float(re.search(r"\(([\d.]+) km", t.name).group(1))
        for t in fig.data if t.name and "km²" in t.name
    ]
    assert len(shown) == 3
    total = np.pi * area_depth.radius_at(area_depth.deepest, apex) ** 2
    assert sum(shown) == pytest.approx(total, abs=0.02)


def test_map_view_uses_the_requested_markers(area_depth):
    """circle-open-dot for the well, x-thin for the apex, per Lars."""
    apex = area_depth.apex_estimate()
    fig = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY, z_exit=EXIT)
    well = next(t for t in fig.data if t.name == "Well")
    assert well.marker.symbol == "circle-open-dot"
    apex_marker = next(
        t for t in fig.data
        if t.showlegend is False and t.mode and "markers" in t.mode and len(t.x) == 1
    )
    assert apex_marker.marker.symbol == "x-thin"


def test_map_view_shows_its_y_axis_values(area_depth):
    """It is a map: both axes are distances and both should be readable."""
    apex = area_depth.apex_estimate()
    fig = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY)
    assert fig.layout.yaxis.showticklabels is not False
    assert "km" in fig.layout.yaxis.title.text


# ------------------------------------------------- A1 / A4 percentile family
def test_a1_shows_the_area_percentile_family_thin_and_grey(area_depth, reduced):
    """Lars's convention: the mean keeps the colour and the weight because it is
    the number that gets quoted; P90/P50/P10 are thin and grey."""
    fig = I.pfig_a1_area_depth(area_depth, ts=reduced, current_entry=ENTRY)
    named = {t.name: t for t in fig.data if t.name}
    assert {"P90", "P50", "P10", "Mean area"} <= set(named)
    assert named["Mean area"].line.color == colour("prospect")
    for k in ("P90", "P50", "P10"):
        assert named[k].line.color == palette()["muted"]
        assert named[k].line.width < named["Mean area"].line.width


def test_a1_says_area_is_deterministic_rather_than_implying_uncertainty(area_depth, reduced):
    """The binned P90-P10 spread is 20 % of the mean on this file, but that is
    the depth range inside each bin -- the isotonic residual is 1e-5 of the mean.
    Reporting the binned figure would invent uncertainty the model lacks."""
    fig = I.pfig_a1_area_depth(area_depth, ts=reduced)
    assert "deterministic function of contact depth" in fig.layout.title.text


def test_a4_uses_the_same_percentile_convention_as_a1(reduced):
    fig = I.pfig_a4_resource_vs_depth(reduced, current_entry=ENTRY, current_exit=EXIT)
    named = {t.name: t for t in fig.data if t.name}
    assert {"P90", "P50", "P10", "Mean"} <= set(named)
    assert named["Mean"].line.color == colour("prospect")
    for k in ("P90", "P50", "P10"):
        assert named[k].line.color == palette()["muted"]


# ------------------------------------------------- the concepts teaching figure
@pytest.fixture(scope="module")
def concepts(full):
    """(C1 section, C2 exceedance, chance) for the full export.

    Built from the *full* export because C1 needs the reservoir thickness column
    and the 7-column paste does not carry it. Returns both figures since the old
    composite was split into two on 2026-08-11 -- tests that used to look for a
    section trace and a curve trace in one figure now say which figure they mean.
    """
    from wellvolpos.core import AreaDepth, group_trials, split_trials
    from wellvolpos.core import p_well as p_well_fn

    ad = AreaDepth.from_trials(full.col("contact"), full.col("area"))
    g = group_trials(full, ENTRY, EXIT)
    vcl = split_trials(full, ad, g, ENTRY, EXIT)
    ch = p_well_fn(full, ENTRY, POS)
    c1 = I.pfig_a1_area_depth(ad, ts=full, current_entry=ENTRY, current_exit=EXIT)
    c2 = I.pfig_c2_exceedance(full, g, vcl, pos_prospect=POS, p_well=ch.p_well, mefs=14.0)
    return c1, c2, ch


def test_a1_draws_the_reservoir_band_from_a_real_thickness(concepts):
    """Top reservoir is A(z); the base is the same curve shifted down by the
    thickness recovered from pay -- drawn four times, P90/P50/mean/P10, because that
    thickness is a distribution and one base line implied a surface the trials do
    not support.

    Both are real quantities. An earlier version used sqrt(area) as a pretend
    lateral width, which is not in the data at all. The band lives on **A1** now:
    it and C1 drew the same A(z), so C1 kept only the unlabelled thumbnail.
    """
    fig, _c2, _ = concepts
    named = " ".join(str(t.name or "") for t in fig.data)
    assert "Top reservoir" in named
    for q in ("Base P90", "Base P50", "Base mean", "Base P10"):
        assert q in named, q
    assert "thickness from pay" in fig.layout.title.text
    # The three volume classes, shaded between top and base.
    assert sum(1 for t in fig.data if t.fillcolor) >= 3

def test_concepts_draws_the_base_on_the_seven_column_paste_too(reduced, groups, vc, area_depth):
    """The payoff from back-calculating thickness instead of reading a column.

    The everyday 7-column export has no reservoir-thickness column, so the base
    reservoir used to be omitted entirely — the figure lost its wedges on the
    default data set. Inverting the wedge needs only area, pay and contact, all
    of which that export does carry.
    """
    assert not reduced.has("thickness")
    fig = I.pfig_a1_area_depth(area_depth, ts=reduced, current_entry=ENTRY, current_exit=EXIT)
    said = " ".join(str(t.name or "") for t in fig.data)
    assert "Base P50" in said and "Top reservoir" in said
    # The thickness note moved off the figure title and onto the section panel's
    # own x-axis label, where it belongs and where it stopped colliding with the
    # first subplot heading.
    assert "thickness from pay" in fig.layout.title.text
    assert sum(1 for d in fig.data if d.fillcolor) >= 3      # all three wedges



def _concept_curves(fig, concept):
    """The (conditional, unconditional) traces for one concept in C2.

    C2 draws two curves per concept, named "<concept> - conditional" and
    "- unconditional", so tests match on the prefix rather than an exact name.
    """
    out = {}
    for t in fig.data:
        name = str(t.name or "")
        if name.startswith(concept + " \u2014 "):
            out[name.rsplit("\u2014 ", 1)[1]] = t
    return out.get("conditional"), out.get("unconditional")


def test_concepts_risked_curves_start_at_their_own_chance(concepts):
    """The whole trick of the figure, and the reason it teaches the decomposition.

    Plotting the *risked* distribution -- zeros for the outcomes that do not
    happen -- makes each curve begin at its own chance rather than at 100 %. The
    two POS values are then where the curves physically start, and the gap
    between them is the location penalty rather than a caption.
    """
    _c1, fig, ch = concepts
    starts = {}
    for concept in ("Prospect resource potential", "Well associated resource potential",
                    "Up-dip volume"):
        cond, uncond = _concept_curves(fig, concept)
        # The conditional twin always reaches 100 %; the unconditional one carries
        # the chance, and that pair is the contrast the figure exists to make.
        assert float(np.nanmax(np.asarray(cond.y, dtype=float))) == pytest.approx(100.0, abs=1e-6)
        starts[concept] = float(np.nanmax(np.asarray(uncond.y, dtype=float)))
    assert starts["Prospect resource potential"] == pytest.approx(100 * POS, abs=1e-6)
    assert starts["Well associated resource potential"] == pytest.approx(100 * ch.p_well, abs=1e-6)
    # Up-dip starts at P(dry and charged) = POS - P_well.
    assert starts["Up-dip volume"] == pytest.approx(100 * (POS - ch.p_well), abs=1e-6)


def test_concepts_marks_both_pos_values_where_the_curves_start(concepts):
    _c1, fig, ch = concepts
    said = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "Asso. Final Prospect POS" in said
    assert "Asso. Well POS" in said
    assert f"{POS:.0%}" in said
    assert f"{ch.p_well:.0%}" in said


def test_concepts_braces_nest_from_narrowest_to_widest(concepts):
    """up-dip inside tested inside well associated inside prospect. Drawn widest
    at the bottom, so the containment is visible rather than asserted."""
    _c1, fig, _ = concepts
    order = ["Up-dip volume", "Resource tested by well",
             "Well associated resource potential", "Prospect resource potential"]
    # Each brace is a horizontal 2-point segment below the 0 % line.
    braces = {}
    for t in fig.data:
        if t.x is not None and len(t.x) == 2 and t.y is not None and len(t.y) == 2:
            y0, y1 = float(t.y[0]), float(t.y[1])
            if y0 == y1 and y0 < 0:
                braces[round(y0, 3)] = (float(t.x[0]), float(t.x[1]))
    assert len(braces) == len(order)
    rows = [braces[k] for k in sorted(braces, reverse=True)]   # shallowest brace first
    widths = [hi - lo for lo, hi in rows]
    assert widths == sorted(widths), "braces must widen downward"


def test_concepts_uses_one_colour_per_concept_across_both_panels(concepts):
    """The pairing only teaches if the section and the curves agree on colour."""
    c1, fig, _ = concepts
    for concept, role in (("Prospect resource potential", "prospect"),
                          ("Well associated resource potential", "well_associated"),
                          ("Resource tested by well", "tested"),
                          ("Up-dip volume", "up_dip")):
        cond, uncond = _concept_curves(fig, concept)
        # Both readings of one concept share its colour; line style, not hue,
        # separates them (non-negotiable 3).
        assert cond.line.color == colour(role) == uncond.line.color, concept
        assert cond.line.dash == "solid" and uncond.line.dash == "dash"
    # The section's bands carry the same roles, as translucent fills -- in C1 now.
    fills = [t.fillcolor for t in c1.data if t.fillcolor]
    for role in ("up_dip", "tested", "below_lkh"):
        assert any(rgba_of(role) in (f or "") for f in fills), role


def rgba_of(role):
    from wellvolpos.viz.theme import rgba

    return rgba(role, 0.55).rsplit(",", 1)[0]


def test_concepts_section_keeps_depth_on_y_inverted(concepts):
    """It is a section, so non-negotiable 2 applies to its left panel."""
    fig, _c2, _ = concepts
    assert is_depth_axis_correct_plotly(fig, "yaxis")


def test_concepts_probability_axis_leaves_room_for_the_braces(concepts):
    _c1, fig, _ = concepts
    # C2 is a standalone figure now, so its probability axis is `yaxis`, not the
    # second axis of a subplot grid.
    lo, hi = fig.layout.yaxis.range
    assert hi >= 100.0
    assert lo < 0.0, "the braces are drawn below the 0 % line"


# ---------------------------------------------------------------- area scales
# GeoX plots area-depth against area squared, so the app offers that axis too.
# The transform is presentation only -- every computed number stays in km2 --
# which is exactly the sort of claim that quietly stops being true.


def test_the_three_area_scales_transform_the_axis_and_say_so(area_depth, reduced):
    """Each scale must move the data *and* relabel. A transform applied without
    the label, or a label without the transform, is a mislabelled axis."""
    xs, labels = {}, {}
    for key in I.AREA_SCALES:
        fig = I.pfig_a1_area_depth(area_depth, ts=reduced, current_entry=ENTRY,
                                   current_exit=EXIT, area_scale=key)
        data = np.concatenate([np.asarray(t.x, float) for t in fig.data if t.x is not None])
        xs[key] = float(np.nanmax(data))
        labels[key] = fig.layout.xaxis.title.text
    assert xs["area²"] == pytest.approx(xs["area"] ** 2, rel=1e-6)
    assert xs["√area"] == pytest.approx(np.sqrt(xs["area"]), rel=1e-6)
    assert len(set(labels.values())) == 3
    assert "km²" in labels["area"] and "km⁴" in labels["area²"] and "km)" in labels["√area"]


def test_every_area_scale_is_monotone_so_the_ordering_survives(area_depth):
    """A scale that reordered the curve would make the plot say something the
    numbers do not."""
    a = np.linspace(0.0, 10.0, 50)
    for _, transform in I.AREA_SCALES.values():
        assert np.all(np.diff(transform(a)) > 0)


def test_the_depth_rule_holds_under_every_area_scale(area_depth, reduced):
    """Non-negotiable 2 is about the y-axis, so changing the x-axis must not
    disturb it."""
    for key in I.AREA_SCALES:
        fig = I.pfig_a1_area_depth(area_depth, ts=reduced, area_scale=key)
        assert is_depth_axis_correct_plotly(fig)


def test_an_unknown_scale_falls_back_on_both_the_label_and_the_data(area_depth, reduced):
    """The scale arrives as a UI string, so a fallback is right -- but it has to
    fall back on the transform and the label together, or a typo produces a
    squared axis labelled km²."""
    bad = I.pfig_a1_area_depth(area_depth, ts=reduced, area_scale="not a scale")
    plain = I.pfig_a1_area_depth(area_depth, ts=reduced, area_scale="area")
    assert bad.layout.xaxis.title.text == plain.layout.xaxis.title.text
    for t_bad, t_plain in zip(bad.data, plain.data):
        if t_bad.x is not None:
            assert np.allclose(np.asarray(t_bad.x, float), np.asarray(t_plain.x, float),
                               equal_nan=True)


def test_the_concepts_section_honours_the_area_scale_too(reduced, area_depth, groups, vc):
    """A1 and the concepts figure draw the same A(z); they must not be readable
    against different axes in the same session."""
    sq = I.pfig_a1_area_depth(area_depth, ts=reduced, current_entry=ENTRY,
                              current_exit=EXIT, area_scale="area²")
    assert "km⁴" in sq.layout.xaxis.title.text


# ------------------------------------------------------------------ colour key
# Drawn rather than written because Streamlit strips inline style out of
# markdown, which turned the HTML version into seven labels with no colours.


def test_the_colour_key_draws_a_swatch_per_concept_in_the_palette(reduced):
    for dark in (False, True):
        fig = I.pfig_colour_key(dark=dark)
        swatches = [s.fillcolor for s in fig.layout.shapes]
        assert swatches == [colour(role, dark) for role, _, _ in I.CONCEPT_KEY]
        assert len(swatches) == len(I.CONCEPT_KEY)


def test_no_two_concepts_share_a_swatch(reduced):
    """A key with a repeated colour cannot do its job. This is the one place the
    palette's distinctness is asserted as a requirement rather than a nicety."""
    for dark in (False, True):
        assert len({colour(role, dark) for role, _, _ in I.CONCEPT_KEY}) == len(I.CONCEPT_KEY)


def test_the_key_names_every_concept_and_carries_the_nesting(reduced):
    """Ordered narrowest first, so the key teaches containment as well as
    mapping: minimum inside tested inside well-associated inside prospect."""
    labels = [lab for _, lab, _ in I.CONCEPT_KEY]
    order = [role for role, _, _ in I.CONCEPT_KEY]
    assert order.index("minimum") < order.index("tested") < order.index("well_associated") \
        < order.index("prospect")
    fig = I.pfig_colour_key()
    texts = [a.text for a in fig.layout.annotations]
    assert len(texts) == len(labels)
    for lab, text in zip(labels, texts):
        assert lab in text


def test_the_key_covers_every_role_the_figures_colour_by(reduced):
    """The point of the key is that a colour means the same thing in every
    figure. If a figure starts using a role the key does not explain, the
    guarantee is gone -- so the key is checked against theme.CO_OCCURRING."""
    from wellvolpos.viz.theme import CO_OCCURRING, ROLES

    explained = {ROLES[role] for role, _, _ in I.CONCEPT_KEY}
    used = {ROLES[r] for roles in CO_OCCURRING.values() for r in roles if r in ROLES}
    assert used <= explained | {"muted"}       # muted is context, not a concept


def test_the_key_has_no_axes_to_read(reduced):
    """It is a legend, not a plot; visible axes would invite reading a scale off
    it. Exempt from the depth rule for the same reason."""
    fig = I.pfig_colour_key()
    assert fig.layout.xaxis.visible is False and fig.layout.yaxis.visible is False


# ------------------------------------------------- risked vs unrisked reading
# The fourth instance of this codebase's recurring mistake, and the reason the
# arithmetic now lives in core.classes rather than in a figure: the concepts
# figure risked its curves by zero-padding with the trial file's own masks, so a
# curve started at the *file's* implied chance instead of the entered one. On
# prospect A under "trials are risked" the two coincide, which is why three tests
# and two reviews missed it. On prospect B -- success-case only, POS from the
# chance table -- they differ by the whole table.


def test_a_risked_curve_starts_at_its_chance_by_construction():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    for chance in (1.0, 0.5, 0.2):
        v, pct = risked_exceedance(values, chance)
        assert float(pct.max()) == pytest.approx(chance * 100.0)
        assert np.all(np.diff(pct) < 0)
        # The *volumes* are untouched by risking; only the probabilities move.
        assert np.array_equal(v, values)


def test_risking_scales_the_probability_and_never_the_volume():
    values = np.array([5.0, 10.0, 20.0])
    v_un, p_un = risked_exceedance(values, 1.0)
    v_r, p_r = risked_exceedance(values, 0.4)
    assert np.array_equal(v_un, v_r)
    assert np.allclose(p_r, 0.4 * p_un)


def test_an_empty_class_gives_an_empty_curve_rather_than_raising():
    v, pct = risked_exceedance(np.array([]), 0.5)
    assert v.size == 0 and pct.size == 0


def test_class_percentiles_puts_p99_at_the_low_end(reduced):
    """P99 is exceeded 99 % of the time, so it is the *small* volume. Getting this
    backwards is the classic error in this domain and it would invert every table
    in the app."""
    res = reduced.col("resource")
    s = class_percentiles(res[res > 0], 0.4576)
    assert s["p99"] < s["p90"] < s["p50"] < s["p10"] < s["p1"]


def test_class_percentiles_reports_where_the_mean_actually_falls(reduced):
    """The mean is not a percentile. On a right-skewed resource distribution it
    sits above the P50 -- and the table says at which exceedance, because "mean"
    and "middle" get used interchangeably and they are not."""
    res = reduced.col("resource")
    s = class_percentiles(res[res > 0], 1.0)
    assert s["mean"] > s["p50"]
    assert 25.0 < s["mean_at"] < 50.0


def test_the_concepts_curves_follow_the_entered_pos_not_the_trial_file(
    reduced, area_depth, groups, vc
):
    """The bug itself, pinned. Two different entered POS values must move the
    curves; before the fix both drew identically because the zero-padding came
    from the trial masks."""
    starts = {}
    for pos in (0.7605, 0.40):
        r = r_location(reduced, ENTRY)[0]
        pw = pos * r
        fig = I.pfig_c2_exceedance(reduced, groups, vc, pos_prospect=pos, p_well=pw, mefs=14.0)
        heights = {}
        for concept in ("Prospect resource potential", "Well associated resource potential",
                        "Up-dip volume"):
            _cond, uncond = _concept_curves(fig, concept)
            heights[concept] = float(np.nanmax(np.asarray(uncond.y, dtype=float)))
        starts[pos] = heights
        assert heights["Prospect resource potential"] == pytest.approx(pos * 100.0, abs=0.5)
        assert heights["Well associated resource potential"] == pytest.approx(pw * 100.0, abs=0.5)
        assert heights["Up-dip volume"] == pytest.approx((pos - pw) * 100.0, abs=0.5)
    assert starts[0.7605] != starts[0.40]


def test_the_gap_between_the_two_curve_starts_is_the_location_penalty(
    reduced, area_depth, groups, vc
):
    """The argument the figure exists to make, as an assertion: the vertical
    distance between where the prospect curve starts and where the well-associated
    curve starts is POS - P_well, the chance the prospect has something this well
    would miss."""
    pos, pw = 0.7605, 0.4576
    fig = I.pfig_c2_exceedance(reduced, groups, vc, pos_prospect=pos, p_well=pw, mefs=14.0)
    tops = {}
    for concept in ("Prospect resource potential", "Well associated resource potential"):
        _cond, uncond = _concept_curves(fig, concept)
        tops[concept] = float(np.nanmax(np.asarray(uncond.y, dtype=float)))
    penalty = tops["Prospect resource potential"] - tops["Well associated resource potential"]
    assert penalty == pytest.approx((pos - pw) * 100.0, abs=0.5)


# ------------------------------------------------------ exceedance markers
def test_exceedance_marks_are_labelled_by_value_and_sit_on_the_curve(reduced, groups, vc):
    """Lars asked for a marker at P90/P50/mean/P10 carrying the *value*, not the
    percentile name -- the percentile is already the axis."""
    fig = I.pfig_a5_exceedance(reduced, groups, vc, mefs=14.0,
                               pos_prospect=POS, p_well=0.4576)
    marks = [t for t in fig.data if t.mode and "markers" in t.mode]
    # Four statistics on each of the prospect's two readings, since A5 is
    # prospect-only now.
    assert len(marks) == 8
    for t in marks:
        assert t.text is not None and t.text[0].strip().replace(",", "").replace(".", "").isdigit()
        # The label is the volume, and the statistic's name is in the hover.
        assert any(k in t.hovertemplate for k in ("P90", "P50", "P10", "Mean"))


def test_exceedance_marks_use_the_petroleum_orientation(reduced):
    """P90 low, P10 high -- and each marker's height is read off the curve, so it
    lands *on* the line rather than at a nominal percentile."""
    from wellvolpos.viz.figures import exceedance_marks

    res = reduced.col("resource")
    marks = dict((label, (value, pct)) for label, value, pct in exceedance_marks(res[res > 0]))
    assert marks["P90"][0] < marks["P50"][0] < marks["P10"][0]
    assert marks["P90"][1] == pytest.approx(90.0, abs=1.0)
    assert marks["P10"][1] == pytest.approx(10.0, abs=1.0)
    assert marks["Mean"][1] < 50.0                           # right-skewed


# ------------------------------------------------------------- the map view
def test_the_map_draws_dashed_contours_and_one_solid_entry_ring(area_depth):
    """Lars, 2026-08-10: line style now carries one meaning only -- is this the
    well? Every contour is dashed; the entry contour is the only solid ring."""
    apex = area_depth.apex_estimate()
    fig = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY, z_exit=EXIT, interval=50.0)
    dashes = [t.line.dash for t in fig.data if t.mode == "lines" and t.line is not None]
    assert dashes.count("solid") == 1
    assert set(dashes) - {"solid", None} == {"dash"}


def test_every_map_contour_carries_a_depth_label(area_depth):
    """So the map reads like a depth map instead of by hovering. Round depths are
    what makes labelling in place worth doing -- a legend of fifteen depths is a
    lookup table."""
    apex = area_depth.apex_estimate()
    fig = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY, z_exit=EXIT, interval=50.0)
    labels = [a.text for a in fig.layout.annotations]
    contours = area_depth.contour_radii(apex, interval=50.0, z_max=area_depth.deepest)
    for depth in contours.depths:
        assert f"{depth:.0f}" in " ".join(labels)
    assert any("well entry" in t for t in labels)


# ------------------------------------------------- B7 and B8, from the workbook
# Both come from the 2018 macro workbook, which carries four charts the 2017 one
# did not. These two are the ones the app had no equivalent of.


def test_b7_plots_chance_against_volume_not_against_depth(vsweep):
    """The trade-off, stated directly. Neither axis is a depth, so B7 joins A5, A6,
    B4 and B5 in the depth-rule exemption -- depth appears as labels along the
    curve instead."""
    fig = I.pfig_b7_frontier(vsweep, current_z=ENTRY)
    assert "MMboe" in fig.layout.xaxis.title.text
    assert "P_well" in fig.layout.yaxis.title.text
    assert not is_depth_axis_correct_plotly(fig)          # deliberately exempt
    labels = [a.text for a in fig.layout.annotations]
    assert any(str(int(z)) in " ".join(labels) for z in vsweep.z[::4])


def test_b7_moving_down_dip_buys_volume_with_chance(vsweep):
    """The whole argument of the tool, as an assertion: along the frontier, chance
    falls as volume rises. If this ever came out flat or positively sloped, either
    the sweep or the figure would be wrong."""
    fig = I.pfig_b7_frontier(vsweep)
    assoc = next(t for t in fig.data if t.name == "Well associated mean")
    x = np.asarray(assoc.x, dtype=float)
    y = np.asarray(assoc.y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    assert ok.sum() > 5
    # Rank correlation, because the frontier is curved, not linear.
    from scipy.stats import spearmanr

    rho = spearmanr(x[ok], y[ok]).statistic
    assert rho < -0.9, rho


def test_b7_marks_the_current_well_on_the_frontier(vsweep):
    fig = I.pfig_b7_frontier(vsweep, current_z=ENTRY)
    here = next(t for t in fig.data if t.name == "This well")
    assert f"{ENTRY:.0f}" in here.text[0]


def test_b8_is_a_depth_figure_and_multiplies_to_the_commercial_chance(vsweep):
    """``Pc(well) = P_well x Pmcfs(well)``, checked as arithmetic rather than
    against the figure's own drawing -- the defence CLAUDE.md asks for after B4."""
    fig = I.pfig_b8_commercial_chance(vsweep, current_z=ENTRY)
    assert is_depth_axis_correct_plotly(fig)
    named = {str(t.name).split(" —")[0]: np.asarray(t.x, dtype=float) for t in fig.data if t.name}
    pmcfs, pw, pc = named["Pmcfs(well)"], named["P_well"], named["Pc(well)"]
    ok = np.isfinite(pmcfs) & np.isfinite(pw) & np.isfinite(pc)
    assert ok.sum() > 5
    assert np.allclose(pc[ok], pw[ok] * pmcfs[ok] / 100.0, atol=1e-9)
    # And it never exceeds either factor: a commercial discovery needs both.
    assert np.all(pc[ok] <= pw[ok] + 1e-9) and np.all(pc[ok] <= pmcfs[ok] + 1e-9)


def test_b8_separates_the_conditional_from_the_unconditional_by_line_style(vsweep):
    """The convention this session settled on, applied here too: conditional solid,
    unconditional dashed."""
    fig = I.pfig_b8_commercial_chance(vsweep)
    dashes = {str(t.name).split(" —")[0]: t.line.dash for t in fig.data if t.name and t.line}
    assert dashes["Pmcfs(well)"] == "solid"
    assert dashes["Pc(well)"] == "dash"


def test_b8_finds_the_interior_maximum_of_the_commercial_chance(vsweep):
    """A rising curve times a falling one, so the product usually peaks in between,
    and that peak is the decision the figure supports."""
    fig = I.pfig_b8_commercial_chance(vsweep)
    star = [t for t in fig.data if t.marker and t.marker.symbol == "star"]
    assert len(star) == 1
    pc = next(np.asarray(t.x, dtype=float) for t in fig.data
              if t.name and str(t.name).startswith("Pc(well)"))
    assert float(star[0].x[0]) == pytest.approx(float(np.nanmax(pc)), abs=1e-9)


def test_b8_says_so_rather_than_drawing_nothing_without_a_mefs(reduced, area_depth):
    from wellvolpos.core.sweep import run_volume_sweep

    bare = run_volume_sweep(reduced, area_depth, POS, n=6, mefs=None, z_gap=EXIT - ENTRY)
    fig = I.pfig_b8_commercial_chance(bare)
    assert "needs a MEFS" in fig.layout.title.text


# --------------------------------------------- the conditional/unconditional pair
def test_the_two_readings_are_named_by_both_of_their_industry_words():
    """Half the industry says "conditional/unconditional" and half says
    "unrisked/risked". The app says both halves of each pair every time, so a
    reader never has to guess which is meant."""
    from wellvolpos.core.classes import READING_LABELS

    assert "Conditional" in READING_LABELS["conditional"]
    assert "success case" in READING_LABELS["conditional"]
    assert "Unconditional" in READING_LABELS["unconditional"]
    assert "risked" in READING_LABELS["unconditional"]


def test_the_unconditional_curve_meets_schneiders_p99_anchor(reduced):
    """Schneider et al. (2023) define Pg as "the chance of making a discovery equal
    to or exceeding the P99 EUR". So the unconditional curve's height at the
    conditional P99 should be the chance itself, to within a per cent -- a free
    check that the two definitions agree, and that our construction is theirs.
    """
    res = reduced.col("resource")
    values = res[res > 0]
    chance = 0.4576
    v, pct = risked_exceedance(values, chance)
    p99 = float(np.percentile(values, 1.0))
    at_p99 = float(np.interp(p99, v, pct[::-1][::-1])) if False else float(
        pct[np.searchsorted(v, p99, side="left")]
    )
    assert at_p99 == pytest.approx(chance * 100.0, abs=1.0)


def test_a_conditional_curve_is_the_unconditional_one_at_chance_one(reduced):
    res = reduced.col("resource")
    values = res[res > 0]
    v1, p1 = conditional_exceedance(values)
    v2, p2 = risked_exceedance(values, 1.0)
    assert np.array_equal(v1, v2) and np.allclose(p1, p2)


def test_c1_draws_the_well_as_a_vertical_track_on_the_structure(reduced, area_depth):
    """C1 shows the well, not only the two depths at which it meets the reservoir.

    Horizontal rules at entry and exit say *where* the reservoir is cut; they never
    draw the borehole, and on a structural panel that is the first thing the eye
    looks for (Lars, 2026-08-11).

    The x anchor is the assertion that matters. On an area axis x is not a spatial
    coordinate, so "vertical" is nominal and the placement has to be argued: the
    track sits at ``A(z_entry)``, the area of the contour the well enters the
    reservoir on, which is exactly where the entry rule meets the top-reservoir
    curve. Anywhere else and the well floats beside the structure instead of
    starting on it. Note this is the opposite of B0, where the well is at x = 0 --
    there zero is the crest line, here zero area *is* the apex.
    """
    fig = I.pfig_c1_section(area_depth, reduced, z_entry=ENTRY, z_exit=EXIT)
    track = [t for t in fig.data if getattr(t, "name", None) == "the well"]
    assert len(track) == 1
    xs, ys = list(track[0].x), list(track[0].y)
    assert xs[0] == xs[1] == pytest.approx(float(area_depth.area_at(ENTRY)), rel=1e-9)
    assert sorted(map(float, ys)) == sorted([ENTRY, EXIT])

    # Its foot is inside the closure, which is the geometric check that the anchor
    # is the shallow one: the closure is wider at the exit than at the entry.
    assert float(area_depth.area_at(EXIT)) > xs[0]


def test_b6_names_both_readings_on_both_axes(reduced, vsweep):
    """One pair of axes carrying two quantities, so both axes must say so.

    B6 draws two families that do **not** measure the same thing: the required-entry
    curve is a *target mean proven volume* against a *required entry depth*, while
    the contact lines are a *volume held by one trial* against a *sampled contact*.
    On prospect A that is 33.9-277.7 MMboe against 2.2-482.1.

    Lars asked for them on one graph after seeing them side by side, and this is the
    condition that makes that honest: the axis titles name both readings, so nobody
    reads one family off the other's definition. Unlabelled, this is exactly the
    figure he reported as unreadable -- so the assertion is the fix, not decoration.
    """
    fig = I.pfig_b6_inverse(vsweep, target=14.0, ts=reduced, mefs=14.0)
    x, y = fig.layout.xaxis.title.text, fig.layout.yaxis.title.text
    assert "mean proven" in x and "one trial" in x
    assert "entry" in y and "contact" in y
    assert "deeper" in y                       # the guarantee, not a first touch
    assert fig.layout.yaxis.range[0] > fig.layout.yaxis.range[1]   # still inverted

    # And the two families stay visually separable: the requirement carries the
    # P_well colour scale, the spread is the neutral muted grey.
    assert _required_entry(fig).marker.colorscale is not None
    greys = [t for t in fig.data if (t.name or "").startswith("P50 contact")]
    assert greys and greys[0].line.color == palette(False)["muted"]


def test_b9_carries_the_chance_weighted_tails_as_grey_lines(vsweep):
    """B9 draws P99/P90/P10/P1 of the proven volume, each weighted by P_well.

    Grey and thin rather than widening the fill (Lars, 2026-08-11): on a
    right-skewed resource distribution P1 runs a long way above P10, and filling
    out to it would swamp the two mean lines the figure is actually about.

    Every one is ``P_well x`` the conditional percentile, so it cannot be an
    unrisked number under a risked label -- the fifth-and-counting instance of that
    bug in this codebase.
    """
    fig = I.pfig_b9_chance_weighted(vsweep)
    names = {getattr(t, "name", None) for t in fig.data}
    for q in ("P99", "P90", "P10", "P1"):
        assert f"Proven {q} × P_well" in names, sorted(n for n in names if n)

    series = {t.name: np.asarray(t.x, dtype=float) for t in fig.data
              if getattr(t, "name", None)}
    p90, p10 = series["Proven P90 × P_well"], series["Proven P10 × P_well"]
    mean = series["Proven MEAN × P_well"]
    ok = np.isfinite(p90) & np.isfinite(p10) & np.isfinite(mean)
    assert ok.any()
    # P90 low, P10 high, mean between them: the weighting is applied to all three
    # identically, so the ordering of the conditional percentiles survives it.
    assert np.all(p90[ok] <= mean[ok] + 1e-9) and np.all(mean[ok] <= p10[ok] + 1e-9)


def test_no_colourbar_sits_outside_its_axes(reduced, vsweep):
    """A colourbar outside the plot area is clipped away and never seen.

    The depth-row rule fixes the margins with ``autoexpand=False`` -- see
    ``test_panels_in_a_row_share_one_plot_area``, which is what makes a row readable
    across -- and the two are in direct conflict: plotly's default colourbar position
    is *outside* the axes on the right, where a 25 px margin cuts it off entirely.

    It went unnoticed on **both** figures that have one. A4 is a trial-count grid, so
    without a scale its colour says only "more or less"; B6 encodes ``P_well``, the
    entire cost side of the trade it exists to show. Neither had a readable scale.

    The fix was first "inside the axes"; on 2026-08-12 Lars asked for legends and
    colourbars alike to sit **below the x-axis title**, which is better -- inside the
    axes they covered data in whichever corner they were parked.

    The first attempt at that placed the two independently -- the legend against the
    figure, the colourbar against a fraction of the plot -- and on A4 they landed on
    top of each other. There is **one** band of reserved space, so one function
    divides it: ``theme.apply_plotly`` puts the legend at the bottom, sized by its
    entry count, and the colourbar just above whatever that comes to. The figures
    themselves declare only that a colourbar exists and what it says.

    So the rule is now: every colourbar is anchored to the container, horizontal, and
    sitting above the legend rather than over it.
    """
    figs = {
        "A4": I.pfig_a4_resource_vs_depth(reduced, render="grid"),
        "B6": I.pfig_b6_inverse(vsweep, target=14.0, ts=reduced),
    }
    for name, fig in figs.items():
        bars = [t.marker.colorbar for t in fig.data
                if getattr(getattr(t, "marker", None), "colorbar", None) is not None
                and t.marker.colorbar.x is not None]
        bars += [t.colorbar for t in fig.data
                 if getattr(t, "colorbar", None) is not None and t.colorbar.x is not None]
        assert bars, f"{name}: expected a positioned colourbar"
        legend_y = fig.layout.legend.y
        for cb in bars:
            assert cb.orientation == "h", f"{name}: colourbar is not horizontal"
            assert cb.yref == "container", f"{name}: colourbar yref={cb.yref!r}"
            assert cb.yanchor == "bottom", f"{name}: colourbar yanchor={cb.yanchor!r}"
            # Above the legend, not on it. Both are container fractions from the
            # bottom, so "above" is simply a larger y.
            assert cb.y > legend_y, (
                f"{name}: colourbar y={cb.y} is not above the legend at y={legend_y}"
            )
            assert cb.y < 0.5, f"{name}: colourbar y={cb.y} has climbed into the plot"
        assert fig.layout.margin.b >= 100, f"{name}: bottom margin leaves no room"


def test_c2_labels_the_conditional_curves_only_but_marks_both(reduced, groups, vc):
    """Markers on both readings, value labels on the conditional one only.

    Both were labelled until 2026-08-12. The argument for it was that seeing one
    volume twice at two heights taught the risking; the argument against, which won,
    is that **risking scales the probability and never the volume** -- so the second
    copy of each number carried no information and doubled the text on the busiest
    figure in the app.

    The risked curve keeps its *markers*, because where it sits is the entire point:
    the same P50 volume at a lower height is the location penalty made visible. So
    this asserts an asymmetry, not a removal, and it would fail just as loudly if
    someone dropped the risked markers as if they put the labels back.
    """
    fig = I.pfig_c2_exceedance(reduced, groups, vc, pos_prospect=POS, p_well=0.4576)
    # Each statistic is its own single-point trace, and ``show_text`` decides the
    # mode -- so "markers+text" is a labelled mark and "markers" is a bare one.
    labelled = [t for t in fig.data if t.mode == "markers+text"]
    bare = [t for t in fig.data if t.mode == "markers"]

    # Four classes x four statistics, each way round.
    assert len(labelled) == 16, len(labelled)
    assert len(bare) == 16, len(bare)

    # The labelled marks sit on the conditional curves: their P90 reaches 90 %.
    assert max(float(t.y[0]) for t in labelled) == pytest.approx(90.0, abs=0.1)
    # The bare ones are risked, so every one of them is pulled below its twin --
    # the highest risked mark is POS_prospect x 0.90, not 90 %.
    assert max(float(t.y[0]) for t in bare) == pytest.approx(90.0 * POS, abs=0.6)
