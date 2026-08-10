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


def test_a5_has_no_depth_axis_and_uses_the_four_canonical_roles(reduced, groups, vc):
    fig, ax = figures.fig_a5_exceedance(reduced, groups, vc, mefs=14.0)
    assert not is_depth_axis_correct(ax)
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert lines["Prospect (all trials)"].get_color() == colour("prospect")
    assert lines["Discovery case"].get_color() == colour("discovery")
    assert lines["Proven at well"].get_color() == colour("proven")
    assert lines["Attic | dry hole"].get_color() == colour("attic")


def test_b3_depth_axis_and_optimum_is_plotted_inside_the_swept_range(sweep):
    fig, ax = figures.fig_b3_uncertainty_reduction(sweep, current_z=ENTRY)
    assert is_depth_axis_correct(ax)
    assert sweep.z.min() <= sweep.z_optimum <= sweep.z.max()


def test_a5_prospect_curve_settles_at_pos_trials_at_the_zero_pileup(reduced, groups, vc):
    # 2 395 chance-failure trials tie at resource = 0, so P(X >= 0) = 100% but
    # the curve steps down to P(X > 0) = POS_trials right after that pile --
    # the drop sits on the y-axis spine. The three conditional series have no
    # such pileup and start flat at 100%.
    fig, ax = figures.fig_a5_exceedance(reduced, groups, vc)
    lines = {line.get_label(): line for line in ax.get_lines()}
    xs, ys = lines["Prospect (all trials)"].get_data()
    assert ys[xs > 0].max() == pytest.approx(100.0 * POS, abs=1e-6)
    assert lines["Discovery case"].get_ydata()[0] == pytest.approx(100.0)
    assert lines["Attic | dry hole"].get_ydata()[0] == pytest.approx(100.0)


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
    assert bands["Discovery, contact seen"] == pytest.approx(_rgb(colour("discovery")))
    assert bands["Discovery, HC to exit"] == pytest.approx(_rgb(colour("possible")))


# ------------------------------------------------------------------- A6
def test_a6_has_no_depth_axis_and_uses_proven_and_attic_colours(groups, vc):
    fig, ax = figures.fig_a6_overlap(vc, groups, mefs=14.0)
    assert not is_depth_axis_correct(ax)
    bars = _hist_containers(ax)
    assert set(bars) == {"Attic | dry hole", "Proven | discovery"}
    assert _rgb(bars["Attic | dry hole"].patches[0].get_facecolor()) == pytest.approx(_rgb(colour("attic")))
    assert _rgb(bars["Proven | discovery"].patches[0].get_facecolor()) == pytest.approx(_rgb(colour("proven")))


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
