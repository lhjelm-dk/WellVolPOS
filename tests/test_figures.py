"""Phase 1 figures: depth-axis compliance and colour-role compliance.

Correctness of the underlying numbers is covered by test_sweep.py,
test_groups.py and test_classes.py; this file only checks that the figures
honour the two non-negotiables from CLAUDE.md -- depth always on y and
inverted, colour assigned by meaning -- since those are easy to silently
break while iterating on a plot.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from wellvolpos.core.classes import split_trials
from wellvolpos.core.sweep import run_sweep, run_volume_sweep
from wellvolpos.viz import figures
from wellvolpos.viz.theme import colour, is_depth_axis_correct, palette

from .conftest import ENTRY, EXIT

POS = 0.7605


def _rgb(c):
    return to_rgba(c)[:3]


def _bands_by_label(ax) -> dict[str, tuple]:
    """Label -> face colour for filled bands, keyed by name rather than draw order."""
    handles, labels = ax.get_legend_handles_labels()
    return {lab: _rgb(h.get_facecolor()[0]) for lab, h in zip(labels, handles)}


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    """Every test in this file creates at least one figure via Agg; none of
    them are ever shown, so nothing needs them to survive past the test.
    Without this, matplotlib accumulates them for the life of the process
    and pytest's RuntimeWarning-as-error setting turns the 21st into a
    failure unrelated to whatever that test actually checks."""
    yield
    plt.close("all")


def _hist_containers(ax) -> dict:
    """Label -> BarContainer, so histogram order is never assumed."""
    _, labels = ax.get_legend_handles_labels()
    return {lab: cont for lab, cont in zip(labels, ax.containers)}


@pytest.fixture(scope="module")
def sweep(reduced):
    return run_sweep(reduced, POS, n=40)


@pytest.fixture(scope="module")
def vc(reduced, area_depth, groups):
    return split_trials(reduced, area_depth, groups, ENTRY, EXIT)


@pytest.fixture(scope="module")
def volume_sweep(reduced, area_depth):
    return run_volume_sweep(reduced, area_depth, POS, n=20, mefs=14.0)


def test_a3_depth_axis_and_chance_colour(sweep):
    fig, ax = figures.fig_a3_chance_decomposition(sweep, pos_trials=POS, current_z=ENTRY)
    assert is_depth_axis_correct(ax)
    lines = {line.get_label(): line for line in ax.get_lines()}
    p_well_line = next(v for k, v in lines.items() if "P_{well}" in k)
    r_line = next(v for k, v in lines.items() if k.startswith("r ="))
    assert p_well_line.get_color() == colour("discovery")
    assert r_line.get_color() == colour("discovery")
    assert p_well_line.get_linestyle() != r_line.get_linestyle()


def test_a4_depth_axis_and_prospect_colour(reduced):
    fig, ax = figures.fig_a4_resource_vs_depth(reduced, current_entry=ENTRY, mefs=14.0)
    assert is_depth_axis_correct(ax)
    trend_lines = {line.get_label(): line for line in ax.get_lines() if line.get_label() in ("P50", "P90", "P10")}
    assert len(trend_lines) == 3
    for line in trend_lines.values():
        assert line.get_color() == colour("prospect")


def test_a5_has_no_depth_axis_and_draws_the_prospect_only(reduced, groups, vc):
    """A5 carries the prospect and nothing else now; the other three classes live in
    C2 and in tab 3's table."""
    fig, ax = figures.fig_a5_exceedance(reduced, groups, vc, mefs=14.0,
                                        pos_prospect=POS, p_well=0.4576)
    assert not is_depth_axis_correct(ax)
    curves = [l for l in ax.get_lines() if len(l.get_xdata()) > 10]
    assert len(curves) == 2
    assert {l.get_color() for l in curves} == {colour("prospect")}

def test_b3_depth_axis_and_optimum_is_plotted_inside_the_swept_range(sweep):
    fig, ax = figures.fig_b3_uncertainty_reduction(sweep, current_z=ENTRY)
    assert is_depth_axis_correct(ax)
    assert sweep.z.min() <= sweep.z_optimum <= sweep.z.max()


def test_c2_draws_both_readings_and_labels_which_is_which(reduced, groups, vc):
    """A5 draws each case twice (Lars, 2026-08-11): solid conditional from 100 %,
    dashed unconditional from that case's own chance.

    It used to be accidentally *mixed* -- the prospect series carried the
    chance-failure zeros and so ran down to POS_trials while the other three
    started at 100 %, with nothing on the figure saying so. Now the mixture is
    deliberate, labelled, and each dashed curve carries its own chance, which
    matters because the four chances are different numbers.
    """
    pos, pw = POS, 0.4576
    fig, ax = figures.fig_c2_exceedance(reduced, groups, vc,
                                        pos_prospect=pos, p_well=pw)
    assert "conditional" in ax.get_title() and "unconditional" in ax.get_title()

    solid = [l for l in ax.get_lines()
             if l.get_linestyle() == "-" and not l.get_label().startswith("_")]
    dashed = [l for l in ax.get_lines()
              if l.get_linestyle() == "--" and len(l.get_xdata()) > 10]
    assert len(solid) == 4 and len(dashed) == 4
    for l in solid:
        assert float(np.nanmax(l.get_ydata())) == pytest.approx(100.0, abs=1e-6), l.get_label()
    # POS for the prospect, P_well twice, and POS - P_well for the attic.
    assert sorted(round(float(np.nanmax(l.get_ydata())), 2) for l in dashed) == pytest.approx(
        sorted([pos * 100, pw * 100, pw * 100, (pos - pw) * 100]), abs=0.6
    )


def test_a5_omits_the_risked_curves_when_no_chance_is_given(reduced, groups, vc):
    """The chances are arguments, never taken from the trial file's own zero count.
    Without them there is no honest unconditional curve to draw, so none is."""
    fig, ax = figures.fig_a5_exceedance(reduced, groups, vc)
    assert not [l for l in ax.get_lines()
                if l.get_linestyle() == "--" and len(l.get_xdata()) > 10]


# ------------------------------------------------------------------- A1
def test_a1_depth_axis_and_prospect_and_well_colours(area_depth):
    fig, ax = figures.fig_a1_area_depth(area_depth, current_entry=ENTRY, current_exit=EXIT)
    assert is_depth_axis_correct(ax)
    lines = {line.get_label(): line for line in ax.get_lines()}
    curve = next(ln for ln in ax.get_lines() if ln.get_label() not in ("Entry", "Exit"))
    assert curve.get_color() == colour("prospect")
    assert lines["Entry"].get_color() == colour("well")
    assert lines["Exit"].get_color() == colour("well")


# ------------------------------------------------------------------- A2
def test_a2_depth_axis_and_outcome_colours(sweep):
    fig, ax = figures.fig_a2_outcome_tree(sweep, current_z=ENTRY)
    assert is_depth_axis_correct(ax)
    bands = _bands_by_label(ax)
    # Chance failure is not one of the four canonical roles -- it is the same
    # everywhere the well could go -- so it takes the neutral muted grey.
    assert bands["Chance failure"] == pytest.approx(_rgb(palette()["muted"]))
    assert bands["Dry, with attic"] == pytest.approx(_rgb(colour("attic")))
    # Contact seen is what the well *tested*; HC continuing past the exit is the
    # untested remainder. Under the volume-concept palette those are `tested` and
    # `possible` -- not `discovery`, which is the whole well-associated case.
    assert bands["Discovery, contact seen"] == pytest.approx(_rgb(colour("tested")))
    assert bands["Discovery, HC to exit"] == pytest.approx(_rgb(colour("possible")))


# ------------------------------------------------------------------- A6
def test_a6_has_no_depth_axis_and_draws_all_four_classes(reduced, groups, vc):
    """A6 gained the well-associated and prospect distributions (Lars, 2026-08-11),
    so Schneider's attic/proven pair is now seen against the two larger
    distributions it is carved out of. Still no depth on either axis.

    Opacity had to come down with four series: at 0.6 the fourth histogram hid the
    first, and what shows through what is the entire content of this figure.
    """
    fig, ax = figures.fig_a6_overlap(vc, groups, ts=reduced, mefs=14.0)
    assert not is_depth_axis_correct(ax)
    labels = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    for expected in ("Prospect", "Well associated", "Attic", "Proven"):
        assert expected in labels, expected
    alphas = {round(float(p.get_alpha() or 1.0), 2) for p in ax.patches}
    assert alphas == {0.45}

def test_a6_densities_each_integrate_to_one(groups, vc):
    """What makes the two series comparable despite n = 4576 vs 3029.

    A6 exists to show the *overlap* in shape, so the curves have to be
    densities on shared bin edges. Plot counts instead and the taller series
    is just the more numerous one.
    """
    fig, ax = figures.fig_a6_overlap(vc, groups)
    for label, container in _hist_containers(ax).items():
        area = sum(p.get_width() * p.get_height() for p in container.patches)
        assert area == pytest.approx(1.0, abs=1e-9), label


# ------------------------------------------------------------------- B0
def test_b0_depth_axis_well_marker_and_bands_sit_where_they_belong(area_depth):
    """The bands are identified by depth, not by draw order.

    B0's whole claim is that each colour sits where that volume physically is,
    so swapping attic and possible-below must fail -- which a bare count of
    collections would not notice.
    """
    fig, ax = figures.fig_b0_section(area_depth, z_entry=ENTRY, z_exit=EXIT)
    assert is_depth_axis_correct(ax)
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert lines["Well"].get_color() == colour("well")

    expected = {"attic": _rgb(colour("attic")), "proven": _rgb(colour("proven")),
                "possible": _rgb(colour("possible"))}
    seen = {}
    for coll in ax.collections:
        zs = np.concatenate([pth.vertices[:, 1] for pth in coll.get_paths()])
        rgb = _rgb(coll.get_facecolor()[0])
        if zs.max() <= ENTRY + 1e-6:
            seen["attic"] = rgb
        elif zs.min() >= ENTRY - 1e-6 and zs.max() <= EXIT + 1e-6:
            seen["proven"] = rgb
        elif zs.min() >= EXIT - 1e-6:
            seen["possible"] = rgb
    assert seen == pytest.approx(expected)


def test_b0_names_every_band_in_the_figure(area_depth):
    """Three semantically coloured bands and no key is not a colour key."""
    fig, ax = figures.fig_b0_section(area_depth, z_entry=ENTRY, z_exit=EXIT)
    said = " ".join(t.get_text().lower() for t in ax.texts)
    assert "attic" in said
    assert "proven" in said
    assert "possible" in said


def test_b0_x_axis_claims_no_unit_for_the_sqrt_area_proxy(area_depth):
    # sqrt(area) is neither a radius nor a diameter, so labelling it km would
    # be a unit claim the number cannot support (non-negotiable 4).
    fig, ax = figures.fig_b0_section(area_depth, z_entry=ENTRY, z_exit=EXIT)
    label = ax.get_xlabel()
    assert "km" not in label
    assert "√area" in label


def test_b0_title_is_overridable_for_the_live_section(area_depth):
    fig, ax = figures.fig_b0_section(area_depth, z_entry=ENTRY, z_exit=EXIT, title="Live section")
    assert ax.get_title() == "Live section"


# ------------------------------------------------------------------- B1
def test_b1_depth_axis_and_class_colours(volume_sweep):
    fig, ax = figures.fig_b1_volume_split(volume_sweep, current_z=ENTRY)
    assert is_depth_axis_correct(ax)
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert lines["Proven | discovery"].get_color() == colour("proven")
    assert lines["Possible below exit | discovery"].get_color() == colour("possible")
    assert lines["Attic | dry hole"].get_color() == colour("attic")


# ------------------------------------------------------------------- B2
def test_b2_depth_axis_and_chance_regret_colours(volume_sweep):
    fig, ax = figures.fig_b2_chance_vs_regret(volume_sweep, current_z=ENTRY)
    assert is_depth_axis_correct(ax)
    labels = {line.get_label(): line for line in ax.get_lines()}
    chance = next(v for k, v in labels.items() if "P_{well}" in k)
    assert chance.get_color() == colour("p_well")
    proven = next(v for k, v in labels.items() if k.startswith("P(proven"))
    attic = next(v for k, v in labels.items() if k.startswith("P(attic"))
    assert proven.get_color() == colour("proven")
    assert attic.get_color() == colour("attic")


def test_b2_states_that_the_regret_curve_conditions_on_a_charged_prospect(volume_sweep):
    """The chance failures are indistinguishable from a dry hole at the bore.

    Folding them in roughly halves this probability, so a curve computed over
    charged dry holes only may not be labelled simply "| dry" -- the design
    plan's 5.08 vs 9.09 MMboe distinction, one level along.
    """
    fig, ax = figures.fig_b2_chance_vs_regret(volume_sweep)
    attic = next(ln for ln in ax.get_lines() if ln.get_label().startswith("P(attic"))
    assert "charged" in attic.get_label()


def test_b2_p_well_agrees_with_the_reference_engine_sweep(reduced, area_depth):
    """B2 and A3 must not disagree about the one curve they share."""
    vs = run_volume_sweep(reduced, area_depth, POS, z_min=3450.0, z_max=3600.0, n=7, mefs=14.0)
    ref = run_sweep(reduced, POS, z_min=3450.0, z_max=3600.0, n=7)
    assert np.allclose(vs.p_well, ref.p_well)


def test_b2_requires_a_mefs_threshold(reduced, area_depth):
    no_mefs = run_volume_sweep(reduced, area_depth, POS, n=5)
    with pytest.raises(ValueError):
        figures.fig_b2_chance_vs_regret(no_mefs)


# ------------------------------------------------------------------- B4
ELEMENTS_EXAMPLE = {"charge": 0.92, "trap": 0.94, "reservoir": 0.95, "retention": 0.93}
R_EXAMPLE = 0.6017


POS_EXAMPLE = 0.7605


def _scatter_by_label(ax, label):
    """Look scatters up by legend label, never by index into ax.collections."""
    handles, labels = ax.get_legend_handles_labels()
    if label in labels:
        return handles[labels.index(label)]
    for coll in ax.collections:
        if coll.get_label() == label:
            return coll
    raise AssertionError(f"no scatter labelled {label!r}; have {labels}")


# The waterfall's arithmetic is tested in test_chance.py against p_well()
# itself. These only check that the drawing honours the roles it is given.
def test_b4_has_no_depth_axis_and_a_log_scale():
    fig, ax = figures.fig_b4_chance_waterfall(ELEMENTS_EXAMPLE, R_EXAMPLE, POS_EXAMPLE)
    assert not is_depth_axis_correct(ax)
    assert ax.get_yscale() == "log"


def test_b4_marks_the_total_as_p_well_not_as_the_tables_own_product():
    """The label a reader takes the number off must name P_well, and the value
    must be the app's P_well -- not prod(elements) x r, which is what the figure
    would total if it ignored the POS in use."""
    fig, ax = figures.fig_b4_chance_waterfall(ELEMENTS_EXAMPLE, R_EXAMPLE, POS_EXAMPLE)
    said = " ".join(t.get_text() for t in ax.texts)
    assert "P_{well}" in said
    assert f"{POS_EXAMPLE * R_EXAMPLE:.4f}" in said
    assert f"{float(np.prod(list(ELEMENTS_EXAMPLE.values()))) * R_EXAMPLE:.4f}" not in said


def test_b4_keeps_location_steps_blue_but_separable_by_hatch():
    """r is a chance and A3 draws it blue, so B4 must not give it a second
    colour; the location contribution is separated by hatching instead."""
    fig, ax = figures.fig_b4_chance_waterfall(ELEMENTS_EXAMPLE, R_EXAMPLE, POS_EXAMPLE, scheme="none")
    hatched = [p for p in ax.patches if p.get_hatch()]
    plain = [p for p in ax.patches if not p.get_hatch() and _rgb(p.get_facecolor()) == pytest.approx(_rgb(colour("p_well")))]
    assert len(hatched) == 1                       # the standalone location bar
    assert len(plain) == len(figures.ELEMENTS)      # one per chance element
    assert _rgb(hatched[0].get_facecolor()) == pytest.approx(_rgb(colour("p_well")))


def test_b4_reconciliation_step_is_neither_chance_nor_location_coloured():
    fig, ax = figures.fig_b4_chance_waterfall(ELEMENTS_EXAMPLE, R_EXAMPLE, POS_EXAMPLE)
    muted = _rgb(palette()["muted"])
    assert any(_rgb(p.get_facecolor()) == pytest.approx(muted) for p in ax.patches)


def test_b4_says_so_rather_than_drawing_a_stub_when_r_is_zero():
    # Reachable: the entry slider's maximum is the deepest sampled contact.
    fig, ax = figures.fig_b4_chance_waterfall(ELEMENTS_EXAMPLE, 0.0, POS_EXAMPLE)
    said = " ".join(t.get_text() for t in ax.texts)
    assert "r = 0" in said
    assert ax.patches == [] or all(p.get_height() == 0 for p in ax.patches)


# ------------------------------------------------------------------- B5
def test_b5_has_no_depth_axis_and_one_panel_per_shipped_scheme():
    fig, axes = figures.fig_b5_allocation_dumbbell(ELEMENTS_EXAMPLE, R_EXAMPLE)
    assert len(axes) == len(figures.SHIPPED_SCHEMES)
    for ax in axes:
        assert not is_depth_axis_correct(ax)


def test_b5_panels_share_one_comparable_x_axis():
    """Three panels whose whole purpose is comparison must not be on
    three different scales."""
    fig, axes = figures.fig_b5_allocation_dumbbell(ELEMENTS_EXAMPLE, R_EXAMPLE)
    xlims = {ax.get_xlim() for ax in axes}
    assert len(xlims) == 1


def test_b5_reservoir_shows_no_movement_under_any_scheme():
    """Reservoir is exempt from every shipped scheme -- allocate() proves it;
    here the dumbbell's own baseline/revised markers must coincide because of it."""
    fig, axes = figures.fig_b5_allocation_dumbbell(ELEMENTS_EXAMPLE, R_EXAMPLE)
    reservoir_idx = list(figures.ELEMENTS).index("reservoir")
    for ax in axes:
        bx = _scatter_by_label(ax, "Baseline").get_offsets()[reservoir_idx][0]
        rx = _scatter_by_label(ax, "At the well").get_offsets()[reservoir_idx][0]
        assert bx == pytest.approx(rx, abs=1e-9)


@pytest.mark.parametrize(
    "scheme_idx, element, should_move",
    [
        (0, "charge", False),      # none: nothing moves
        (0, "trap", False),
        (1, "charge", True),       # equal cube-root: charge/trap/retention move
        (1, "retention", True),
        (2, "trap", True),         # all to trap: only trap moves
        (2, "charge", False),
        (2, "retention", False),
    ],
)
def test_b5_only_the_elements_a_scheme_weights_move(scheme_idx, element, should_move):
    fig, axes = figures.fig_b5_allocation_dumbbell(ELEMENTS_EXAMPLE, R_EXAMPLE)
    ax = axes[scheme_idx]
    idx = list(figures.ELEMENTS).index(element)
    bx = _scatter_by_label(ax, "Baseline").get_offsets()[idx][0]
    rx = _scatter_by_label(ax, "At the well").get_offsets()[idx][0]
    assert (abs(bx - rx) > 1e-9) == should_move


def test_b5_revised_markers_use_chance_colour():
    fig, axes = figures.fig_b5_allocation_dumbbell(ELEMENTS_EXAMPLE, R_EXAMPLE)
    for ax in axes:
        revised = _scatter_by_label(ax, "At the well")
        assert _rgb(revised.get_facecolor()[0]) == pytest.approx(_rgb(colour("p_well")))


def test_b5_explains_where_r_went_in_the_none_panel():
    """Otherwise panel 1 is four dots and no movement, which reads as a
    panel that failed to draw rather than as Milkov's point."""
    fig, axes = figures.fig_b5_allocation_dumbbell(ELEMENTS_EXAMPLE, R_EXAMPLE)
    said = " ".join(t.get_text() for t in axes[0].texts)
    assert "separately" in said


def test_b5_draws_the_shared_p_well_when_given_a_pos():
    fig, axes = figures.fig_b5_allocation_dumbbell(
        ELEMENTS_EXAMPLE, R_EXAMPLE, pos_prospect=POS_EXAMPLE
    )
    expected = POS_EXAMPLE * R_EXAMPLE
    for ax in axes:
        xs = [ln.get_xdata()[0] for ln in ax.get_lines() if len(set(ln.get_xdata())) == 1]
        assert any(abs(x - expected) < 1e-9 for x in xs)


# ------------------------------------------------------------------- B6
@pytest.fixture(scope="module")
def vsweep_banded(reduced, area_depth):
    return run_volume_sweep(reduced, area_depth, POS, n=30, mefs=14.0, z_gap=50.0, n_boot=150)


def test_b6_puts_required_depth_on_y_inverted(vsweep_banded):
    """B6's y-axis carries a depth, so non-negotiable 2 applies to it too."""
    fig, ax = figures.fig_b6_inverse(vsweep_banded, target=20.0)
    assert is_depth_axis_correct(ax)
    assert "TVDSS" in ax.get_ylabel()


def test_b6_encodes_p_well_as_colour_not_a_second_axis(vsweep_banded):
    """Dual y-axes are forbidden, so the cost side of the trade is the colour."""
    fig, ax = figures.fig_b6_inverse(vsweep_banded, target=20.0)
    assert len(fig.axes) == 2                       # the second is the colourbar
    assert fig.axes[1].get_ylabel().startswith("$P_{well}$")
    assert ax.get_shared_x_axes().get_siblings(ax) == [ax]   # no twinned axis


def test_b6_marks_the_requested_target(vsweep_banded):
    fig, ax = figures.fig_b6_inverse(vsweep_banded, target=20.0)
    said = " ".join(t.get_text() for t in ax.texts)
    assert "P_{well}" in said


def test_b6_draws_a_band_only_when_the_sweep_carried_one(reduced, area_depth, vsweep_banded):
    banded, _ = figures.fig_b6_inverse(vsweep_banded)
    assert banded.axes[0].collections, "expected a filled bootstrap band"
    plain_sweep = run_volume_sweep(reduced, area_depth, POS, n=20, z_gap=50.0)
    fig, ax = figures.fig_b6_inverse(plain_sweep)
    labels = ax.get_legend_handles_labels()[1]
    assert not any("band" in lbl for lbl in labels)


def test_b6_says_so_rather_than_drawing_nothing_when_there_is_no_curve(reduced, area_depth):
    empty = run_volume_sweep(reduced, area_depth, POS, z_min=4000.0, z_max=4100.0, n=5, z_gap=50.0)
    fig, ax = figures.fig_b6_inverse(empty)
    said = " ".join(t.get_text() for t in ax.texts)
    assert "invert" in said


# --------------------------------------------------- thinning the deep end
def test_b1_leaves_undersupported_steps_undrawn(vsweep_banded):
    """A mean of 8 trials must not be drawn as boldly as a mean of 4 000."""
    fig, ax = figures.fig_b1_volume_split(vsweep_banded)
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    proven = lines["Proven | discovery"].get_xdata()
    assert np.isnan(np.asarray(proven, dtype=float)).any()


def test_b1_draws_everything_when_the_floor_is_lowered(vsweep_banded):
    """The floor is a presentation choice, so it has to be a real argument."""
    fig, ax = figures.fig_b1_volume_split(vsweep_banded, min_support=0)
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    proven = np.asarray(lines["Proven | discovery"].get_xdata(), dtype=float)
    # Only genuinely empty steps stay NaN at a zero floor.
    assert np.isnan(proven).sum() <= int((vsweep_banded.n_discovery == 0).sum())


def test_b2_thins_conditional_curves_but_never_p_well(vsweep_banded):
    """P_well is a chance over all trials, so it is supported everywhere."""
    fig, ax = figures.fig_b2_chance_vs_regret(vsweep_banded)
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    p_well_x = np.asarray(lines[r"$P_{well}$"].get_xdata(), dtype=float)
    assert not np.isnan(p_well_x).any()
    proven_x = np.asarray(lines["P(proven > MEFS | discovery)"].get_xdata(), dtype=float)
    assert np.isnan(proven_x).any()


def test_b2_names_the_curves_that_meet_rather_than_claiming_chance_equals_regret(vsweep_banded):
    """P_well is unconditional; the regret curve is conditional on dry *and*
    charged. Calling their intersection "chance = regret" would put those two
    on one scale, which is the conflation this project keeps having to undo."""
    fig, ax = figures.fig_b2_chance_vs_regret(vsweep_banded)
    said = " ".join(t.get_text() for t in ax.texts)
    assert "chance = regret" not in said
    assert "dry & charged" in said


# ------------------------------------------------- the export-path twins
# The concepts figure, the map view and the colour key were designed on the
# interactive path and only later drawn in matplotlib. These tests exist because
# the export path is the half nobody looks at, so a twin that quietly stopped
# matching would not be noticed until an exported document was already circulating.


def test_every_plotly_figure_has_an_export_twin():
    """The rule the twins exist to satisfy. Anything drawn in the app must be
    drawable in an export, or the export silently ships a different document."""
    from wellvolpos.viz import interactive as I

    plotly_names = {n[len("pfig_"):] for n in dir(I) if n.startswith("pfig_")}
    mpl_names = {n[len("fig_"):] for n in dir(figures) if n.startswith("fig_")}
    assert plotly_names <= mpl_names, f"no export twin for {sorted(plotly_names - mpl_names)}"


def test_the_c2_twin_starts_its_curves_at_their_own_chance(reduced, area_depth, groups, vc):
    """The teaching mechanism, on the export path. The curves are risked, so the
    prospect curve starts at POS_prospect and the well-associated curve at
    P_well -- and the gap between those two starts is the location penalty. If a
    curve started at 100 % the figure would be making the opposite argument."""
    fig, ax_exc = figures.fig_c2_exceedance(
        reduced, groups, vc, pos_prospect=POS, p_well=0.4576, mefs=14.0,
    )
    # Each curve's *maximum* is its own chance -- not 100 %. The earlier version of
    # this test asserted 100 % for the first two curves, which contradicted its own
    # docstring and was passing only because the figure zero-padded with the trial
    # file's masks; on prospect A under the "trials are risked" convention the two
    # coincide, which is exactly why it went unnoticed.
    # Each concept now draws *two* curves: a solid conditional one reaching 100 %
    # and a dashed unconditional one reaching its own chance. Only the conditional
    # line carries a legend label, so the dashed twins are identified by style.
    dashed = [l for l in ax_exc.get_lines()
              if l.get_linestyle() == "--" and len(l.get_ydata()) > 10]
    solid = [l for l in ax_exc.get_lines()
             if l.get_linestyle() == "-" and not l.get_label().startswith("_")
             and len(l.get_ydata()) > 10]
    for l in solid:
        assert float(np.nanmax(l.get_ydata())) == pytest.approx(100.0, abs=0.5)
    tops_dashed = sorted(float(np.nanmax(l.get_ydata())) for l in dashed)
    # POS, P_well twice (well associated and tested), and POS - P_well.
    expected = sorted([POS * 100.0, 45.76, 45.76, (POS - 0.4576) * 100.0])
    assert tops_dashed == pytest.approx(expected, abs=0.6)


def test_the_c1_twin_carries_a_labelled_depth_axis_and_the_well(
    reduced, area_depth, groups, vc
):
    """C1 is fully labelled (Lars, 2026-08-11), and is *not* an exemption from
    non-negotiable 2.

    It briefly was one -- a stripped thumbnail whose job beside C2 was to be
    recognised rather than read. That failed in use: it is the first figure on the
    teaching tab, and a structure with no depth axis cannot show that the up-dip
    volume sits *above* the well at some particular depth, which is the whole
    content of the panel.

    So this asserts the ordinary rule -- depth on y, inverted, labelled in m TVDSS
    -- plus the two well rules, which is the other half of what came back.
    """
    fig, ax_sec = figures.fig_c1_section(
        area_depth, reduced, z_entry=ENTRY, z_exit=EXIT,
    )
    lo, hi = ax_sec.get_ylim()
    assert lo > hi                                       # inverted
    assert "TVDSS" in ax_sec.get_ylabel()
    assert ax_sec.get_xlabel()                           # the area axis is named
    assert len(ax_sec.get_yticks()) > 0

    # The well: one rule at entry, one at exit, both in the well colour and both
    # named. An unnamed rule is the state this test exists to prevent returning to.
    ys = {round(float(line.get_ydata()[0]), 3) for line in ax_sec.get_lines()
          if len(set(line.get_ydata())) == 1}
    assert round(ENTRY, 3) in ys and round(EXIT, 3) in ys
    texts = {t.get_text() for t in ax_sec.texts}
    assert "well entry" in texts and "well exit" in texts


def test_the_c2_twin_colours_by_the_same_roles_as_its_plotly_original(
    reduced, area_depth, groups, vc
):
    """One colour per concept, in both backends. Drift here is exactly the drift
    theme.py exists to prevent."""
    fig, ax_exc = figures.fig_c2_exceedance(
        reduced, groups, vc, pos_prospect=POS, p_well=0.4576,
    )
    from matplotlib.colors import to_hex

    used = {to_hex(l.get_color()).lower() for l in ax_exc.get_lines()
            if not l.get_label().startswith("_")}
    for role in ("prospect", "well_associated", "tested", "up_dip"):
        assert colour(role, False).lower() in used


def test_the_a1_twin_degrades_rather_than_inventing_a_base_reservoir(
    reduced, area_depth, groups, vc
):
    """No recoverable thickness means no base reservoir -- and A1, which carries the
    reservoir band now, says so instead of drawing a surface it cannot support."""
    import copy

    ts = copy.deepcopy(reduced)
    ts.frame["hc_grv"] = ts.frame["hc_grv"] * 1000.0     # nothing can be inverted
    fig, ax = figures.fig_a1_area_depth(area_depth, ts=ts, current_entry=ENTRY,
                                        current_exit=EXIT)
    # No base curve and no shaded wedges -- the figure simply shows A(z).
    assert not any("Base" in str(l.get_label()) for l in ax.get_lines())


def test_the_map_twin_is_plan_view_with_equal_aspect(area_depth):
    """Both axes are map kilometres, so this is one of the figures exempt from
    the depth rule -- but the areas must stay comparable, which needs equal
    aspect. Without it a resize makes one contour look bigger than another that
    encloses more."""
    apex = area_depth.apex_estimate()
    fig, ax = figures.fig_map_view(area_depth, apex=apex, z_entry=ENTRY, z_exit=EXIT)
    assert "TVDSS" not in ax.get_ylabel()
    assert ax.get_aspect() == 1.0
    assert ax.get_xlim() == pytest.approx(ax.get_ylim())


def test_the_map_twin_puts_the_well_on_its_own_entry_contour(area_depth):
    """The one thing about the well's map position that means anything."""
    apex = area_depth.apex_estimate()
    fig, ax = figures.fig_map_view(area_depth, apex=apex, z_entry=ENTRY, z_exit=EXIT,
                             well_azimuth_deg=0.0)
    r_entry = area_depth.radius_at(ENTRY, apex)
    xs = [float(l.get_xdata()[0]) for l in ax.get_lines() if len(l.get_xdata()) == 1]
    assert any(x == pytest.approx(r_entry, rel=1e-6) for x in xs)


def test_the_map_twin_names_the_three_areas_the_well_divides_the_closure_into(area_depth):
    apex = area_depth.apex_estimate()
    fig, ax = figures.fig_map_view(area_depth, apex=apex, z_entry=ENTRY, z_exit=EXIT)
    labels = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert "attic" in labels and "proven" in labels and "Possible" in labels


def test_the_map_twin_honours_the_contour_interval(area_depth):
    """Contours on round multiples, so a coarser interval draws fewer of them."""
    apex = area_depth.apex_estimate()
    coarse = figures.fig_map_view(area_depth, apex=apex, z_entry=ENTRY, interval=100.0)[1]
    fine = figures.fig_map_view(area_depth, apex=apex, z_entry=ENTRY, interval=25.0)[1]
    assert len(fine.get_lines()) > len(coarse.get_lines())


def test_the_colour_key_twin_lists_the_same_concepts_as_the_interactive_one():
    """Read from one source, so the two keys cannot disagree about what a colour
    means -- which is the only thing a colour key promises."""
    from wellvolpos.viz.interactive import CONCEPT_KEY

    fig, ax = figures.fig_colour_key()
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == len(CONCEPT_KEY)
    for (_, label, _), text in zip(CONCEPT_KEY, texts):
        assert label in text
    from matplotlib.colors import to_hex

    swatches = [to_hex(patch.get_facecolor()).lower() for patch in ax.patches]
    assert swatches == [colour(role, False).lower() for role, _, _ in CONCEPT_KEY]
