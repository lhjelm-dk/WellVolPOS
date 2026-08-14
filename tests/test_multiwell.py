"""Every candidate well reaches every figure on tab ③ that can carry it.

This drifted once already and silently: the rules were wired to the seven figures
that happened to be built into variables, and the five built inline inside their
``_chart(...)`` call -- 3.8, 3.9, 3.10, 3.11, 3.12 -- got nothing. Nothing failed,
nothing looked wrong, and half the tab simply did not know about wells B and C.

So the contract is asserted at the level it broke: **every figure on tab ③ either
draws the candidates or is on the list of figures that provably cannot**, and that
list is spelled out with the reason rather than left as whatever the code does today.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from wellvolpos.core import AreaDepth, run_volume_sweep
from wellvolpos.ui.wells import MAX_WELLS, WELL_LABELS, WellOption
from wellvolpos.viz import add_well_markers, add_well_points
from wellvolpos.viz import interactive as I

TAB3 = Path(__file__).resolve().parents[1] / "wellvolpos" / "ui" / "tab3_where.py"

#: Figures on tab ③ that take **rules**, because their y-axis is a depth.
# b0 left the tab on 2026-08-14 -- 4.3 draws the same section at the chosen well.
RULE_FIGURES = {"a2", "a3", "b3", "b11", "b1", "b2", "b8", "b9", "b6"}

#: And the two that cannot, with the reason. 3.8's axes are volume and chance, so a
#: well is a *point* on the frontier; 3.12 is re-banded by whichever well is selected,
#: because a band straddling the entry would mix dry trials with discoveries -- so a
#: second candidate is a different figure, not another curve.
POINT_FIGURES = {"b7"}
RE_BANDED = {"b12"}


@pytest.fixture(scope="module")
def wells():
    return (WellOption("A", 2205.0, 2255.0),
            WellOption("B", 2255.0, 2305.0),
            WellOption("C", 2305.0, 2355.0))


@pytest.fixture(scope="module")
def vsweep(full):
    ad = AreaDepth.from_trials(full.col("contact"), full.col("area"))
    return run_volume_sweep(full, ad, 0.43, n=20, z_gap=50.0, mefs=103.0)


# ------------------------------------------------------------------ the wiring
def test_every_charted_figure_on_tab_three_is_accounted_for():
    """A new figure must be given the candidates or listed as unable to take them."""
    src = TAB3.read_text(encoding="utf-8")
    charted = {m.group(1) for m in re.finditer(r'_chart\([^)]*key="(\w+)"', src, re.S)}
    known = RULE_FIGURES | POINT_FIGURES | RE_BANDED
    assert charted <= known, (
        f"tab ③ charts figures the multi-well contract has not been told about: "
        f"{sorted(charted - known)}"
    )


def test_every_depth_figure_is_given_the_candidates():
    """The exact defect: rules wired to the figures that had variables, and to no
    others. Checked against the source, because it is a wiring fact rather than a
    property of any one figure."""
    src = TAB3.read_text(encoding="utf-8")
    marked = set()
    for m in re.finditer(r"add_well_markers\(\s*(\w+)", src):
        marked.add(m.group(1))
    for m in re.finditer(r"for _f in \(([^)]*)\):\s*\n\s*add_well_markers", src):
        marked |= {t.strip() for t in m.group(1).split(",") if t.strip()}
    # Every rule figure has a variable named after it that is passed to the helper.
    missing = [k for k in RULE_FIGURES
               if not any(v.endswith(k) for v in marked)]
    assert not missing, f"depth figures with no candidate rules: {sorted(missing)}"
    assert re.search(r"add_well_points\(\s*_f_b7", src), "3.8 has no candidate points"


# --------------------------------------------------------------- the drawing
def test_markers_skip_the_selected_well(wells):
    """Each figure already marks ``current_z`` in its own style, so drawing it here
    too put two rules at one depth in two colours."""
    fig = I.pfig_b1_volume_split.__wrapped__ if hasattr(I.pfig_b1_volume_split, "__wrapped__") \
        else None
    import plotly.graph_objects as go

    blank = go.Figure()
    add_well_markers(blank, wells, selected="A")
    texts = [a.text for a in blank.layout.annotations]
    assert not any("A ·" in (t or "") for t in texts), texts
    assert any("B ·" in (t or "") for t in texts)
    assert any("C ·" in (t or "") for t in texts)
    assert len(blank.layout.shapes) == 2


def test_every_marker_carries_its_letter_and_its_depth(wells):
    import plotly.graph_objects as go

    blank = go.Figure()
    add_well_markers(blank, wells, selected=None)
    texts = [a.text or "" for a in blank.layout.annotations]
    assert len(texts) == len(wells)
    for w in wells:
        assert any(w.label in t and f"{w.entry:,.0f}" in t for t in texts), (w, texts)


def test_frontier_points_land_on_the_curve_the_figure_draws(vsweep, wells):
    """Interpolated onto the *thinned* series, so a marker cannot appear at a volume
    3.8 itself declined to plot."""
    fig = I.pfig_b7_frontier(vsweep, current_z=wells[0].entry)
    before = len(fig.data)
    add_well_points(fig, vsweep, wells, selected="A")
    added = fig.data[before:]
    assert len(added) == len(wells) - 1
    xs = np.concatenate([np.asarray(t.x, dtype=float) for t in fig.data[:before]
                         if t.x is not None and len(t.x) > 2])
    lo, hi = float(np.nanmin(xs)), float(np.nanmax(xs))
    for t in added:
        assert lo <= float(t.x[0]) <= hi, (t.name, t.x, lo, hi)
        assert 0.0 <= float(t.y[0]) <= 110.0


def test_3_12_names_the_well_it_was_banded_on(full):
    """Each candidate re-bands it, so which one is on screen has to be on the figure."""
    from wellvolpos.core import group_trials, split_trials, thickness_from_pay
    from wellvolpos.core.bands import banded_percentiles

    ad = AreaDepth.from_trials(full.col("contact"), full.col("area"))
    g = group_trials(full, 2205.0, 2255.0)
    th = thickness_from_pay(full, ad).thickness
    vc = split_trials(full, ad, g, 2205.0, 2255.0, thickness=th, apex=ad.apex_estimate())
    bp = banded_percentiles(full, g, vc, z_entry=2205.0, z_exit=2255.0)

    titled = I.pfig_b12_banded_percentiles(bp, well_label="B")
    assert "Well B" in titled.layout.title.text
    # And without a label it still says which depths, rather than claiming a well.
    plain = I.pfig_b12_banded_percentiles(bp)
    assert "Well" not in plain.layout.title.text.split("<br>")[0].replace("well ", "")


# --------------------------------------------------------------- the model
def test_the_labels_and_the_limit_agree():
    assert len(WELL_LABELS) == MAX_WELLS
    assert WELL_LABELS[0] == "A", "Well A is the one that always exists"
