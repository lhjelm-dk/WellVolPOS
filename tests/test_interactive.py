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
from wellvolpos.core.classes import split_trials
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


def test_panels_in_a_row_share_one_height(area_depth, sweep, vsweep, reduced):
    figs = _depth_figures(area_depth, sweep, vsweep, reduced)
    assert {fig.layout.height for fig in figs.values()} == {PANEL_HEIGHT}


def test_panels_in_a_row_share_one_plot_area(area_depth, sweep, vsweep, reduced):
    """Equal ranges are not enough — the plot *areas* have to match too.

    An identical y-range still lands a given depth on a different pixel row if
    one panel's axes are inset further than another's. Plotly does exactly that
    when a legend or colour bar sits outside the axes and ``autoexpand`` is
    left on: the margin grows to fit it. Same height plus same fixed margins
    plus same range is what actually makes a row level.
    """
    figs = _depth_figures(area_depth, sweep, vsweep, reduced)
    margins = {
        (m.l, m.r, m.t, m.b, m.autoexpand)
        for m in (fig.layout.margin for fig in figs.values())
    }
    assert len(margins) == 1, margins
    assert next(iter(margins))[-1] is False, "autoexpand must be off or a legend can shift the axes"


def test_no_depth_figure_puts_its_legend_outside_the_axes(area_depth, sweep, vsweep, reduced):
    """The other half of the same rule: an outside legend needs margin space
    its row-mates do not have."""
    for name, fig in _depth_figures(area_depth, sweep, vsweep, reduced).items():
        lg = fig.layout.legend
        if lg.x is not None:
            assert 0.0 <= lg.x <= 1.0, f"{name} legend x={lg.x} is outside the axes"
        if lg.y is not None:
            assert 0.0 <= lg.y <= 1.0, f"{name} legend y={lg.y} is outside the axes"


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


def test_a5_maps_the_four_series_onto_the_four_canonical_roles(reduced, groups, vc):
    fig = I.pfig_a5_exceedance(reduced, groups, vc)
    assert _line_colour(fig, "Prospect (all trials)") == colour("prospect")
    assert _line_colour(fig, "Discovery case") == colour("discovery")
    assert _line_colour(fig, "Proven at well") == colour("proven")
    assert _line_colour(fig, "Attic | dry hole") == colour("attic")


def test_b1_uses_the_class_colours(vsweep):
    fig = I.pfig_b1_volume_split(vsweep)
    assert _line_colour(fig, "Proven | discovery") == colour("proven")
    assert _line_colour(fig, "Possible below exit | discovery") == colour("possible")
    assert _line_colour(fig, "Attic | dry hole") == colour("attic")


def test_b2_names_its_conditioning_and_keeps_the_regret_colour(vsweep):
    fig = I.pfig_b2_chance_vs_regret(vsweep)
    names = [t.name for t in fig.data if t.name]
    attic = next(n for n in names if n.startswith("P(attic"))
    assert "charged" in attic
    assert _line_colour(fig, attic) == colour("attic")


def test_a2_bands_use_the_outcome_colours_and_reach_100_percent(sweep):
    fig = I.pfig_a2_outcome_tree(sweep)
    fills = {t.name: t.fillcolor for t in fig.data if t.fillcolor}
    assert fills["Dry, with attic"] == colour("attic")
    assert fills["Discovery, contact seen"] == colour("discovery")
    assert fills["Discovery, HC to exit"] == colour("possible")
    assert fills["Chance failure"] == palette()["muted"]
    top = max(float(np.nanmax(t.x)) for t in fig.data if t.x is not None and len(t.x))
    assert top == pytest.approx(100.0, abs=1e-6)


def test_b4_keeps_location_blue_and_separates_it_by_pattern(r):
    """Same reasoning as the matplotlib B4: r is a chance and A3 draws it blue,
    so hatching carries the distinction rather than a second colour."""
    fig = I.pfig_b4_chance_waterfall(TABLE, r, POS, scheme="none")
    patterned = [t for t in fig.data if t.marker.pattern and t.marker.pattern.shape == "/"]
    assert len(patterned) == 1
    assert patterned[0].marker.color == colour("p_well")


def test_b4_annotates_p_well_not_the_tables_own_product(r):
    fig = I.pfig_b4_chance_waterfall(TABLE, r, POS)
    said = " ".join(a.text or "" for a in fig.layout.annotations)
    assert f"{POS * r:.4f}" in said
    assert f"{float(np.prod(list(TABLE.values()))) * r:.4f}" not in said


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
