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
    # ``P_well`` and ``P<sub>well</sub>`` are one symbol written two ways -- plain
    # text for matplotlib, which cannot take HTML, and a subscript for plotly, which
    # cannot take mathtext. Neither spelling is content.
    s = s.replace("_", " ")
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


#: Series a backend is allowed to name where its twin does not, because the two
#: draw the same thing by different means. Kept explicit and short: an exemption
#: list that grows without argument is how the guard stops guarding.
NAME_EXEMPT = {
    # matplotlib rules are Line2D objects and carry legend labels; plotly draws
    # them as layout shapes with annotations, which have no trace name.
    "A1": {"well entry", "well exit"},
    "A4": {"well entry", "well exit"},
    "A8": {"well entry"},
    "A9": {"prospect (n=7,605)"},
    # B0's bands are annotated in both, but only plotly names the traces.
    "B0": {"attic if dry", "proven", "unproven below lkh"},
    "B3": {"reduction"},
}


def test_every_twin_draws_the_same_named_series(kit):
    """Not just the axes -- the *curves*.

    The audit that found the axis drift also found A1's export missing the entire
    area P90/P50/mean/P10 family and one of the four base-reservoir curves, and C1
    drawing one base where the screen drew four. Both were features Lars had asked
    for, absent from the artefact that goes in a well proposal, with nothing on the
    page to say so.

    Comparing *names* rather than pixels keeps this honest without freezing the
    styling: the backends may render a curve differently, they may not omit it.
    """
    missing = []
    for name, pfig, (_mfig, max_) in _pairs(kit):
        # Normalised, like the axis titles: the two backends spell markup
        # differently and neither spelling is content.
        p_names = {
            _normalise(t.name) for t in pfig.data
            if getattr(t, "name", None) and "lines" in (getattr(t, "mode", "") or "lines")
        }
        # **Every axes on the figure, not only the one returned.** A twinned axis
        # is a sibling rather than a child, so a series drawn on a second x-axis --
        # A1's and A8's contact histograms -- was invisible to this guard and had to
        # be exempted by name. An exemption for a series that *is* drawn is the guard
        # lying about its own coverage.
        m_names = set()
        for ax in np.atleast_1d(max_).ravel():
            for sibling in ax.figure.axes:
                m_names |= {_normalise(lbl)
                            for lbl in sibling.get_legend_handles_labels()[1]
                            if not lbl.startswith("_")}
        exempt = NAME_EXEMPT.get(name, set())
        only_plotly = p_names - m_names - exempt
        only_mpl = m_names - p_names - exempt
        if only_plotly:
            missing.append(f"{name}: on screen but not exported: {sorted(only_plotly)}")
        if only_mpl:
            missing.append(f"{name}: exported but not on screen: {sorted(only_mpl)}")
        plt.close("all")
    assert not missing, "twins draw different series:\n  " + "\n  ".join(missing)


def test_every_figure_with_a_twin_reaches_the_export_bundle(reduced, area_depth):
    """A twin that agrees perfectly is no use if the export never asks for it.

    A9 had a matplotlib twin, passed the agreement guard, and was absent from every
    exported document because ``build_figures`` simply did not list it. That is a
    different hole from divergence and needs its own check: the export path is the
    only consumer of ``viz.figures``, so anything with a twin and no bundle entry is
    dead weight on screen-only.

    Exemptions are for helpers that are not figures in their own right.
    """
    from wellvolpos.report import export as E
    from wellvolpos.report.case import Case
    from wellvolpos.viz import figures as Fmod

    case = Case(
        entry=ENTRY, exit=EXIT, mefs=14.0, risking_convention="success_case_only",
        chance_table={"charge": 0.9, "trap": 1.0, "reservoir": 0.6, "retention": 0.8},
        play_elements={e: 1.0 for e in ("charge", "trap", "reservoir", "retention")},
        reference="crest", scheme="equal_cube_root",
    )
    bundle = E.assemble(reduced, case, pos=POS, pos_source="the chance table")
    exported = " ".join(E.build_figures(bundle)).lower()

    #: Not figures: ``fig_concepts`` was split into C1/C2, and the colour key and
    #: map view are exported under their own names already.
    skip = {"fig_concepts"}
    missing = [
        name for name in dir(Fmod)
        if name.startswith("fig_") and name not in skip
        and name[len("fig_"):].split("_")[0] not in exported
    ]
    assert not missing, f"have a twin but never reach an export: {missing}"
