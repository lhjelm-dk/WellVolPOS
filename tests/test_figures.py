"""Phase 1 figures: depth-axis compliance and colour-role compliance.

Correctness of the underlying numbers is covered by test_sweep.py,
test_groups.py and test_classes.py; this file only checks that the figures
honour the two non-negotiables from CLAUDE.md -- depth always on y and
inverted, colour assigned by meaning -- since those are easy to silently
break while iterating on a plot.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from wellvolpos.core.classes import split_trials
from wellvolpos.core.sweep import run_sweep
from wellvolpos.viz import figures
from wellvolpos.viz.theme import colour, is_depth_axis_correct

from .conftest import ENTRY, EXIT

POS = 0.7605


@pytest.fixture(scope="module")
def sweep(reduced):
    return run_sweep(reduced, POS, n=40)


@pytest.fixture(scope="module")
def vc(reduced, area_depth, groups):
    return split_trials(reduced, area_depth, groups, ENTRY, EXIT)


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
