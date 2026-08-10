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
    assert fills["Discovery, contact seen"] == colour("tested")
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


# ------------------------------------------------------------------- B6
@pytest.fixture(scope="module")
def vsweep_banded(reduced, area_depth):
    return run_volume_sweep(reduced, area_depth, POS, n=25, mefs=14.0, z_gap=50.0, n_boot=150)


def test_pb6_puts_required_depth_on_y_inverted(vsweep_banded):
    fig = I.pfig_b6_inverse(vsweep_banded, target=20.0)
    assert is_depth_axis_correct_plotly(fig)
    assert "TVDSS" in fig.layout.yaxis.title.text


def test_pb6_carries_p_well_as_marker_colour_with_a_scale(vsweep_banded):
    """No dual y-axes, so the cost side of the trade is the colour."""
    fig = I.pfig_b6_inverse(vsweep_banded)
    curve = next(t for t in fig.data if t.name == "Required entry")
    assert curve.marker.colorscale is not None
    assert curve.marker.color is not None
    assert "yaxis2" not in fig.layout


def test_pb6_hover_gives_volume_depth_and_chance_together(vsweep_banded):
    fig = I.pfig_b6_inverse(vsweep_banded)
    curve = next(t for t in fig.data if t.name == "Required entry")
    assert "MMboe" in curve.hovertemplate
    assert "TVDSS" in curve.hovertemplate
    assert "P<sub>well</sub>" in curve.hovertemplate


def test_pb6_band_only_appears_when_the_sweep_carried_one(reduced, area_depth, vsweep_banded):
    banded = I.pfig_b6_inverse(vsweep_banded)
    assert any("band" in (t.name or "") for t in banded.data)
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
    possible below exit -- the same split B0 draws in section, so the two figures
    colour-key identically."""
    apex = area_depth.apex_estimate()
    fig = I.pfig_map_view(area_depth, apex=apex, z_entry=ENTRY, z_exit=EXIT)
    named = {t.name: t for t in fig.data if t.name}
    attic = next(t for n, t in named.items() if n.startswith("Potential attic"))
    proven = next(t for n, t in named.items() if n.startswith("Potentially proven"))
    possible = next(t for n, t in named.items() if n.startswith("Possible"))
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
def concepts(area_depth, reduced, groups, vc):
    from wellvolpos.core import p_well as p_well_fn

    ch = p_well_fn(reduced, ENTRY, POS)
    return I.pfig_concepts(
        area_depth, reduced, groups, vc, z_entry=ENTRY, z_exit=EXIT,
        pos_prospect=POS, p_well=ch.p_well, mefs=14.0,
    ), ch


def test_concepts_risked_curves_start_at_their_own_chance(concepts):
    """The whole trick of the figure, and the reason it teaches the decomposition.

    Plotting the *risked* distribution -- zeros for the outcomes that do not
    happen -- makes each curve begin at its own chance rather than at 100 %. The
    two POS values are then where the curves physically start, and the gap
    between them is the location penalty rather than a caption.
    """
    fig, ch = concepts
    starts = {}
    for t in fig.data:
        if t.name in ("Prospect resource potential", "Well associated resource potential",
                      "Up-dip volume"):
            x = np.asarray(t.x, dtype=float)
            y = np.asarray(t.y, dtype=float)
            starts[t.name] = float(y[x > 0].max())
    assert starts["Prospect resource potential"] == pytest.approx(100 * POS, abs=1e-6)
    assert starts["Well associated resource potential"] == pytest.approx(100 * ch.p_well, abs=1e-6)
    # Up-dip starts at P(dry and charged) = POS - P_well.
    assert starts["Up-dip volume"] == pytest.approx(100 * (POS - ch.p_well), abs=1e-6)


def test_concepts_marks_both_pos_values_where_the_curves_start(concepts):
    fig, ch = concepts
    said = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "Asso. Final Prospect POS" in said
    assert "Asso. Well POS" in said
    assert f"{POS:.0%}" in said
    assert f"{ch.p_well:.0%}" in said


def test_concepts_braces_nest_from_narrowest_to_widest(concepts):
    """up-dip inside tested inside well associated inside prospect. Drawn widest
    at the bottom, so the containment is visible rather than asserted."""
    fig, _ = concepts
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
    fig, _ = concepts
    curve_colour = {t.name: t.line.color for t in fig.data if t.name and t.line}
    assert curve_colour["Prospect resource potential"] == colour("prospect")
    assert curve_colour["Well associated resource potential"] == colour("well_associated")
    assert curve_colour["Resource tested by well"] == colour("tested")
    assert curve_colour["Up-dip volume"] == colour("up_dip")
    # The section's bands carry the same roles, as translucent fills.
    fills = [t.fillcolor for t in fig.data if t.fillcolor]
    for role in ("up_dip", "tested", "possible"):
        assert any(rgba_of(role) in (f or "") for f in fills), role


def rgba_of(role):
    from wellvolpos.viz.theme import rgba

    return rgba(role, 0.55).rsplit(",", 1)[0]


def test_concepts_section_keeps_depth_on_y_inverted(concepts):
    """It is a section, so non-negotiable 2 applies to its left panel."""
    fig, _ = concepts
    assert is_depth_axis_correct_plotly(fig, "yaxis")


def test_concepts_probability_axis_leaves_room_for_the_braces(concepts):
    fig, _ = concepts
    lo, hi = fig.layout.yaxis2.range
    assert hi >= 100.0
    assert lo < 0.0, "the braces are drawn below the 0 % line"
