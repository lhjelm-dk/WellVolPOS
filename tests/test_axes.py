"""Depth is always on the y-axis, increasing downward.

Not a style preference. A depth axis on y makes a plot spatially congruent with
the subsurface: higher on the page is shallower is up-dip, so a row of panels
sharing one axis can be read straight across at constant depth beside a well log
or a structural section, and the attic sits literally above the well marker.
This test exists so the rule cannot quietly rot.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wellvolpos.viz.theme import apply, colour, depth_axis, is_depth_axis_correct, palette


def test_depth_axis_inverts():
    apply()
    fig, ax = plt.subplots()
    depth_axis(ax, zlim=(3350, 3700))
    assert is_depth_axis_correct(ax)
    assert ax.get_ylim() == (3700.0, 3350.0)
    plt.close(fig)


def test_depth_axis_inverts_an_already_plotted_axis():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3400, 3600])
    depth_axis(ax)
    assert is_depth_axis_correct(ax)
    plt.close(fig)


def test_shared_row_hides_repeat_labels():
    fig, (a, b) = plt.subplots(1, 2)
    depth_axis(a, zlim=(3350, 3700))
    depth_axis(b, ylabel=None, zlim=(3350, 3700))
    assert a.get_ylim() == b.get_ylim()
    assert b.yaxis.get_label().get_text() == ""
    plt.close(fig)


def test_colours_are_addressed_by_meaning_not_position():
    assert colour("attic") == colour("regret")
    assert colour("discovery") == colour("p_well")
    assert colour("attic") != colour("proven")


# The rule is backend-independent: the interactive path has to obey it too, or
# the half the user actually looks at is the unchecked half. Per-figure
# compliance is in test_interactive.py; these cover the helper itself.
def test_plotly_depth_axis_inverts_with_an_explicit_range():
    import plotly.graph_objects as go

    from wellvolpos.viz.theme import depth_axis_plotly, is_depth_axis_correct_plotly

    fig = go.Figure()
    depth_axis_plotly(fig, zlim=(3350, 3700))
    assert is_depth_axis_correct_plotly(fig)
    assert tuple(fig.layout.yaxis.range) == (3700.0, 3350.0)


def test_plotly_depth_axis_inverts_without_an_explicit_range():
    import plotly.graph_objects as go

    from wellvolpos.viz.theme import depth_axis_plotly, is_depth_axis_correct_plotly

    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3400, 3600])
    depth_axis_plotly(fig)
    assert is_depth_axis_correct_plotly(fig)


def test_plotly_shared_row_hides_repeat_labels():
    import plotly.graph_objects as go

    from wellvolpos.viz.theme import depth_axis_plotly

    a, b = go.Figure(), go.Figure()
    depth_axis_plotly(a, zlim=(3350, 3700))
    depth_axis_plotly(b, zlim=(3350, 3700), show_ticklabels=False)
    assert tuple(a.layout.yaxis.range) == tuple(b.layout.yaxis.range)
    assert b.layout.yaxis.showticklabels is False


def test_dark_mode_is_a_selected_palette_not_an_inversion():
    light, dark = palette(False), palette(True)
    assert light["well_associated"] != dark["well_associated"]
    assert light["surface"] != dark["surface"]


# ------------------------------------------------- colour by volume concept
def test_roles_follow_the_volume_concepts_not_the_old_palette():
    """The mapping Lars's teaching figure uses, so the app and the material he
    explains the concepts with agree. Replaced an earlier
    blue=discovery/orange=attic/yellow=proven/aqua=prospect scheme."""
    from wellvolpos.viz.theme import palette as pal

    p = pal()
    assert colour("attic") == p["up_dip"]
    assert colour("up_dip") == colour("attic") == colour("regret")
    assert colour("discovery") == p["well_associated"] == colour("well_associated")
    assert colour("proven") == p["tested"] == colour("tested")
    assert colour("prospect") == p["prospect"] == colour("pos_prospect")
    assert colour("minimum") == p["minimum"] == colour("mefs")


def test_a_chance_takes_the_colour_of_the_volume_it_belongs_to():
    """P_well is the chance of the well-associated case, so it is olive; the
    prospect's POS is navy. The two POS values on an exceedance plot then read
    against the two distributions they risk."""
    assert colour("p_well") == colour("well_associated")
    assert colour("pos_prospect") == colour("prospect")
    assert colour("p_well") != colour("pos_prospect")


def test_possible_is_no_longer_an_alias_for_prospect():
    """It was, which made aqua mean both "prospect totals" and "possible below
    exit" -- a collision carried over from the mock-ups."""
    assert colour("below_lkh") != colour("prospect")
    assert colour("below_lkh") != colour("proven")


def _simulate_cvd(hex_colour: str, matrix):
    """sRGB hex -> CIELab after a linear-RGB colour-vision-deficiency transform."""
    rgb = np.array([int(hex_colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    sim = np.clip(lin @ np.asarray(matrix).T, 0.0, 1.0)
    back = np.where(sim <= 0.0031308, sim * 12.92, 1.055 * sim ** (1 / 2.4) - 0.055)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    lin2 = np.where(back <= 0.04045, back / 12.92, ((back + 0.055) / 1.055) ** 2.4)
    xyz = (lin2 @ m.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.maximum(xyz, 0) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


CVD_MATRICES = {
    "deuteranopia": [[0.625, 0.375, 0.0], [0.70, 0.30, 0.0], [0.0, 0.30, 0.70]],
    "protanopia": [[0.567, 0.433, 0.0], [0.558, 0.442, 0.0], [0.0, 0.242, 0.758]],
    "tritanopia": [[0.95, 0.05, 0.0], [0.0, 0.433, 0.567], [0.0, 0.475, 0.525]],
    "normal": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
}


@pytest.mark.parametrize("dark", [False, True])
def test_palette_survives_colour_vision_deficiency(dark):
    """Re-derives the separation the palette claims, instead of trusting a comment.

    Checked within each set of roles that can share a figure rather than across
    all pairs: seven categorical colours cannot all separate under every CVD
    type, and demanding it would force a worse palette on figures that never
    show them together. dE 15 is the bar; the tuned palette clears it by a
    hair, so a casual colour edit will fail this.
    """
    from itertools import combinations

    from wellvolpos.viz.theme import CO_OCCURRING, element_colour
    from wellvolpos.viz.theme import palette as pal

    p = pal(dark)

    def _hex(member: str) -> str:
        # An "element:" member is a *risk element*, not a volume role -- a separate
        # family with its own table, added when Lars supplied four element colours
        # on 2026-08-12. Resolved here so one test covers both families.
        if member.startswith("element:"):
            return element_colour(member.split(":", 1)[1], dark)
        return p[member]

    for group, members in CO_OCCURRING.items():
        for cvd, matrix in CVD_MATRICES.items():
            for a, b in combinations(members, 2):
                d = float(np.linalg.norm(
                    _simulate_cvd(_hex(a), matrix) - _simulate_cvd(_hex(b), matrix)))
                assert d >= 15.0, f"{group}: {a} vs {b} under {cvd} is dE {d:.1f}"


def test_the_pale_element_tints_are_never_used_as_the_only_channel():
    """The tints are for a chip with the name written on it, and nothing else.

    Measured: as pale fills, charge and reservoir separate by **dE 6.8** under
    simulated tritanopia -- less than half this project's bar. That is fine behind a
    written label and not fine on a line or a bar, so the saturated set is what the
    figures use and this records why the two tables exist.
    """
    from itertools import combinations

    from wellvolpos.viz.theme import ELEMENT_COLOURS, ELEMENT_TINTS

    assert set(ELEMENT_TINTS) == set(ELEMENT_COLOURS)
    worst_tint = min(
        float(np.linalg.norm(_simulate_cvd(ELEMENT_TINTS[a], m)
                             - _simulate_cvd(ELEMENT_TINTS[b], m)))
        for m in CVD_MATRICES.values() for a, b in combinations(ELEMENT_TINTS, 2)
    )
    worst_solid = min(
        float(np.linalg.norm(_simulate_cvd(ELEMENT_COLOURS[a], m)
                             - _simulate_cvd(ELEMENT_COLOURS[b], m)))
        for m in CVD_MATRICES.values() for a, b in combinations(ELEMENT_COLOURS, 2)
    )
    assert worst_tint < 15.0, "tints now pass -- update the comment, not the test"
    assert worst_solid >= 15.0, f"the saturated element colours fell to dE {worst_solid:.1f}"


def test_the_type_scale_makes_a_hierarchy():
    """Title, subtitle and body must be three distinct sizes.

    The title was 14 against a 12 body -- two points is not a hierarchy -- and several
    titles carry a ``<sub>`` subtitle that was rendering at the title's own size.
    """
    from wellvolpos.viz.theme import FONT_SIZES

    assert FONT_SIZES["title"] > FONT_SIZES["body"] + 3
    assert FONT_SIZES["subtitle"] < FONT_SIZES["body"]
    assert FONT_SIZES["axis_title"] >= FONT_SIZES["tick"]


def test_both_backends_ask_for_the_same_typeface():
    """An export set in a different face from the app is the same defect as an export
    that disagrees about the data — the export path is how this work reaches people."""
    import matplotlib as mpl

    from wellvolpos.viz.theme import FONT_STACK, apply

    apply(False)
    first_mpl = mpl.rcParams["font.sans-serif"][0]
    assert FONT_STACK.split(",")[0].strip() == first_mpl

    # DejaVu leads deliberately: Segoe UI has no PROPORTIONAL TO glyph, which A1's
    # caption uses, so putting it first drew a box in the exported PDF. Glyph coverage
    # beats typeface parity.
    assert first_mpl == "DejaVu Sans"


def test_the_probability_figures_carry_a_crosshair(reduced):
    """Design plan §6.4. On an exceedance curve, reading a probability off a volume is
    *the* interaction, and there were no spikelines."""
    import wellvolpos.viz.interactive as I
    from wellvolpos.core import AreaDepth, group_trials, split_trials

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    groups = group_trials(reduced, 3500.0, 3550.0)
    vc = split_trials(reduced, ad, groups, 3500.0, 3550.0)

    for fig in (I.pfig_a5_exceedance(reduced, groups, vc, mefs=14.0,
                                     pos_prospect=0.76, p_well=0.46),
                I.pfig_c2_exceedance(reduced, groups, vc, pos_prospect=0.76,
                                     p_well=0.46, mefs=14.0)):
        assert fig.layout.xaxis.showspikes and fig.layout.yaxis.showspikes
        assert fig.layout.xaxis.spikemode == "across"


def test_red_means_mefs_and_only_mefs():
    """The design review read ``minimum`` as covering the assessment minimum too, and a
    new hue was cut for it — then measured: the assessment minimum is never drawn.

    It is a numbers-only mapping, so there is nothing to collide with, and an eighth
    categorical hue would have cost separation in a palette already at the CVD limit.
    """
    from wellvolpos.viz.theme import LIGHT

    assert "assessment_min" not in LIGHT


def test_the_top_margin_grows_with_the_title_and_with_a_top_axis(reduced, area_depth):
    """Room is reserved for what the figure actually carries.

    ``autoexpand`` is off across the project — deliberately, so a legend cannot shrink
    one panel of a row — which means nothing grows a margin on its own. The first
    version of this reserved a fixed band and A1, whose title runs to three lines,
    printed its top axis straight through the subtitles. A constant is only ever right
    for the figure it was measured on.
    """
    import wellvolpos.viz.interactive as I

    plain = I.pfig_a1_area_depth(area_depth)                 # one title line, no top axis
    full = I.pfig_a1_area_depth(area_depth, ts=reduced)      # three lines and a top axis

    assert plain.layout.title.text.count("<br>") == 0
    assert full.layout.title.text.count("<br>") == 2
    assert full.layout.margin.t > plain.layout.margin.t

    # The extra room comes out of the margin *and* the height, so a panel sharing a
    # depth row keeps the same plot area as its neighbours.
    def plot_height(fig):
        return fig.layout.height - fig.layout.margin.t - fig.layout.margin.b

    assert plot_height(full) == pytest.approx(plot_height(plain), abs=1)


def test_no_figure_hides_its_top_axis_title_under_the_figure_title(reduced, area_depth):
    """Every top axis is titled, and every titled top axis has its band."""
    import wellvolpos.viz.interactive as I
    from wellvolpos.viz.theme import TOP_AXIS_BAND

    for fig in (I.pfig_a1_area_depth(area_depth, ts=reduced),
                I.pfig_a8_contact_distribution(reduced)):
        tops = [k for k in fig.layout if str(k).startswith("xaxis")
                and getattr(fig.layout[k], "side", None) == "top"]
        assert tops, "expected a top axis"
        for k in tops:
            assert fig.layout[k].title.text, f"{k}: an unlabelled second scale is a trap"
        lines = fig.layout.title.text.count("<br>") + 1
        assert fig.layout.margin.t >= 55 + TOP_AXIS_BAND + 17 * max(0, lines - 2)
