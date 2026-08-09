"""Phase 1 figures: depth-axis compliance and colour-role compliance.

Correctness of the underlying numbers is covered by test_sweep.py,
test_groups.py and test_classes.py; this file only checks that the figures
honour the two non-negotiables from CLAUDE.md -- depth always on y and
inverted, colour assigned by meaning -- since those are easy to silently
break while iterating on a plot.
"""

import matplotlib

matplotlib.use("Agg")

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
