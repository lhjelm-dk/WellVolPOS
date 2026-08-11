"""The two backends must say the same thing, not merely both exist.

``test_every_plotly_figure_has_an_export_twin`` compares ``dir()``: it proves a
twin is *present* and never that it *agrees*. Every figure change has to be made
twice by hand, and on 2026-08-11 an audit found four figures whose axes had
drifted apart -- including B6, where the on-screen y-axis said the answer was a
guarantee ("or deeper") and the exported one did not.

That matters more than it sounds. The export is what leaves the building: it goes
into a well proposal, and it loses the app's captions on the way. A figure that
disagrees with the screen is a figure nobody can check.

So this compares what is cheap to compare and expensive to lose -- the axis
titles. It is deliberately not a pixel comparison; the two backends are allowed
to look different. They are not allowed to *say* different things.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wellvolpos.core import AreaDepth, group_trials, p_well, split_trials
from wellvolpos.core.sweep import run_sweep, run_volume_sweep
from wellvolpos.viz import figures as F
from wellvolpos.viz import interactive as I

from .conftest import ENTRY, EXIT

POS = 0.7605


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def kit(reduced, area_depth):
    g = group_trials(reduced, ENTRY, EXIT)
    return {
        "ts": reduced,
        "ad": area_depth,
        "g": g,
        "vc": split_trials(reduced, area_depth, g, ENTRY, EXIT),
        "sw": run_sweep(reduced, POS, n=20, z_gap=EXIT - ENTRY),
        "vs": run_volume_sweep(reduced, area_depth, POS, n=12, mefs=14.0, z_gap=EXIT - ENTRY),
        "ch": p_well(reduced, ENTRY, POS),
    }


def _pairs(k):
    """(name, plotly figure, matplotlib (fig, ax)) for every figure with a twin."""
    ts, ad, g, vc, sw, vs, ch = (k["ts"], k["ad"], k["g"], k["vc"], k["sw"], k["vs"], k["ch"])
    return [
        ("A1", I.pfig_a1_area_depth(ad, ts=ts, current_entry=ENTRY, current_exit=EXIT),
         F.fig_a1_area_depth(ad, ts=ts, current_entry=ENTRY, current_exit=EXIT)),
        ("A2", I.pfig_a2_outcome_tree(sw, current_z=ENTRY),
         F.fig_a2_outcome_tree(sw, current_z=ENTRY)),
        ("A3", I.pfig_a3_chance_decomposition(sw, pos_prospect=POS),
         F.fig_a3_chance_decomposition(sw, pos_trials=POS)),
        ("A4", I.pfig_a4_resource_vs_depth(ts, current_entry=ENTRY, mefs=14.0),
         F.fig_a4_resource_vs_depth(ts, current_entry=ENTRY, mefs=14.0)),
        ("A5", I.pfig_a5_exceedance(ts, g, vc, mefs=14.0, pos_prospect=POS),
         F.fig_a5_exceedance(ts, g, vc, mefs=14.0, pos_prospect=POS)),
        ("A6", I.pfig_a6_overlap(vc, g, mefs=14.0), F.fig_a6_overlap(vc, g, mefs=14.0)),
        ("A8", I.pfig_a8_contact_distribution(ts, current_entry=ENTRY),
         F.fig_a8_contact_distribution(ts, current_entry=ENTRY)),
        ("A9", I.pfig_a9_prospect_density(ts, mefs=14.0), F.fig_a9_prospect_density(ts, mefs=14.0)),
        ("B0", I.pfig_b0_section(ad, z_entry=ENTRY, z_exit=EXIT),
         F.fig_b0_section(ad, z_entry=ENTRY, z_exit=EXIT)),
        ("B1", I.pfig_b1_volume_split(vs, current_z=ENTRY),
         F.fig_b1_volume_split(vs, current_z=ENTRY)),
        ("B2", I.pfig_b2_chance_vs_regret(vs, current_z=ENTRY),
         F.fig_b2_chance_vs_regret(vs, current_z=ENTRY)),
        ("B3", I.pfig_b3_uncertainty_reduction(sw, current_z=ENTRY),
         F.fig_b3_uncertainty_reduction(sw, current_z=ENTRY)),
        ("B6", I.pfig_b6_inverse(vs, target=14.0, ts=ts, mefs=14.0),
         F.fig_b6_inverse(vs, target=14.0, ts=ts, mefs=14.0)),
        ("B7", I.pfig_b7_frontier(vs, current_z=ENTRY), F.fig_b7_frontier(vs, current_z=ENTRY)),
        ("B8", I.pfig_b8_commercial_chance(vs, current_z=ENTRY),
         F.fig_b8_commercial_chance(vs, current_z=ENTRY)),
        ("B9", I.pfig_b9_chance_weighted(vs, current_z=ENTRY),
         F.fig_b9_chance_weighted(vs, current_z=ENTRY)),
        ("C1", I.pfig_c1_section(ad, ts, z_entry=ENTRY, z_exit=EXIT),
         F.fig_c1_section(ad, ts, z_entry=ENTRY, z_exit=EXIT)),
        ("C2", I.pfig_c2_exceedance(ts, g, vc, pos_prospect=POS, p_well=ch.p_well),
         F.fig_c2_exceedance(ts, g, vc, pos_prospect=POS, p_well=ch.p_well)),
    ]


def _normalise(s: str) -> str:
    """Strip the markup each backend spells differently, keep the words.

    plotly takes a little HTML, matplotlib takes mathtext. Neither is content, so
    both come off before comparing -- otherwise the guard would fail on ``<i>``
    and pass on a genuinely different sentence.
    """
    for tag in ("<sub>", "</sub>", "<sup>", "</sup>", "<i>", "</i>", "<b>", "</b>", "<br>"):
        s = s.replace(tag, " ")
    for ch in "${}\\":
        s = s.replace(ch, "")
    return " ".join(s.split()).lower()


def test_every_twin_agrees_on_its_axis_titles(kit):
    """The words on the axes, in both backends, for all eighteen figures."""
    mismatched = []
    for name, pfig, (_mfig, max_) in _pairs(kit):
        ax = np.atleast_1d(max_).ravel()[0]
        px = _normalise((pfig.layout.xaxis.title.text or "") if pfig.layout.xaxis else "")
        py = _normalise((pfig.layout.yaxis.title.text or "") if pfig.layout.yaxis else "")
        mx, my = _normalise(ax.get_xlabel()), _normalise(ax.get_ylabel())
        if px != mx:
            mismatched.append(f"{name} x: plotly {px!r} vs mpl {mx!r}")
        if py != my:
            mismatched.append(f"{name} y: plotly {py!r} vs mpl {my!r}")
        plt.close("all")
    assert not mismatched, "twins disagree:\n  " + "\n  ".join(mismatched)


def test_every_depth_carrying_twin_inverts_its_axis_in_both_backends(kit):
    """Non-negotiable 2 is backend-independent, so it is checked on the pair.

    A figure that puts depth on y and inverts it on screen, but not in the export,
    is worse than one that gets it wrong in both places -- the reader has no cue
    that the two differ.
    """
    for name, pfig, (_mfig, max_) in _pairs(kit):
        ax = np.atleast_1d(max_).ravel()[0]
        py = _normalise((pfig.layout.yaxis.title.text or "") if pfig.layout.yaxis else "")
        if "tvdss" not in py:
            continue
        lo, hi = pfig.layout.yaxis.range
        assert lo > hi, f"{name}: plotly depth axis is not inverted"
        m_lo, m_hi = ax.get_ylim()
        assert m_lo > m_hi, f"{name}: matplotlib depth axis is not inverted"
        plt.close("all")
