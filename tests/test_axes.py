"""Depth is always on the y-axis, increasing downward.

Not a style preference. A depth axis on y makes a plot spatially congruent with
the subsurface: higher on the page is shallower is up-dip, so a row of panels
sharing one axis can be read straight across at constant depth beside a well log
or a structural section, and the attic sits literally above the well marker.
This test exists so the rule cannot quietly rot.
"""

import matplotlib

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


def test_dark_mode_is_a_selected_palette_not_an_inversion():
    light, dark = palette(False), palette(True)
    assert light["discovery"] != dark["discovery"]
    assert light["surface"] != dark["surface"]
