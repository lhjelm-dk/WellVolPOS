"""One styling entry point, shared by every figure in the project.

Two rules here are load-bearing rather than decorative.

**Colour is assigned by the volume concept, never cycled.** The mapping follows
Lars's teaching figure, so the app and the material he explains the concepts
with agree:

===================  ==============  ==================================
concept              colour          what it is
===================  ==============  ==================================
prospect             dark navy       the whole un-cut prospect
well associated      olive / khaki   the discovery case -- what the well
                                     has access to if it finds anything
tested by well       mauve           proven between entry and exit
possible below exit  light khaki      well associated but not tested
up-dip / attic       light blue      what a dry hole leaves behind
minimum volume       red             a threshold: MEFS, assessment min
well                 purple          the well itself
===================  ==============  ==================================

The nesting is the point, and the colours carry it: minimum ⊂ up-dip ⊂ tested by
well ⊂ well associated ⊂ prospect. A chance takes the colour of the volume it
belongs to -- ``P_well`` is olive like the well-associated case it is the chance
of, and ``POS_prospect`` is navy -- so the two POS values on an exceedance plot
read against the two distributions they risk.

This mapping replaced an earlier one (blue = discovery, orange = attic, yellow =
proven, aqua = prospect). It was changed deliberately, on Lars's instruction, to
match the reference figure; ``tests/test_axes.py`` and the figure tests enforce
the current one. Colourblind separation is checked by
``tests/test_axes.py::test_palette_survives_colour_vision_deficiency`` rather
than asserted in prose.

**Any axis carrying a depth goes on y, increasing downward.** This is not taste.
A depth axis on y makes the plot spatially congruent with the subsurface: higher
on the page is shallower is up-dip. A row of panels sharing one depth axis can
then be read straight across at constant depth, beside a well log or a
structural section; the attic sits literally above the well marker and the
possible volume literally below it. Put depth on x and the picture is rotated 90
degrees from the thing it describes. :func:`depth_axis` is the only sanctioned
way to set up such an axis, and ``tests/test_axes.py`` enforces it.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri

# --------------------------------------------------------------- palette
# The hues follow Lars's teaching figure; the *lightness* of each was then tuned
# until every pair that can appear in one figure stayed separable under
# simulated deuteranopia, protanopia and tritanopia. That tuning is why these are
# not the figure's exact swatches: olive and mauve, and navy and purple, are hue
# distinctions that colour-vision deficiency collapses, so only a lightness or
# red-channel difference survives. Worst separation within any co-occurring set
# is dE 15.4 (light) / 15.5 (dark) -- see
# tests/test_axes.py::test_palette_survives_colour_vision_deficiency, which
# re-derives this rather than trusting the comment.
LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#8a8983",
    "grid": "#e6e5e1",
    "prospect": "#16294a",         # dark navy    -- prospect resource potential
    "tested": "#7d2f5f",           # deep mauve   -- proven between entry and exit
    # Red is **MEFS / MCFS and nothing else**. The design review of 2026-08-15 read
    # this role as covering the *assessment minimum* too, and a new colour was cut for
    # it -- then measured: the assessment minimum is never drawn on any figure. It is a
    # numbers-only mapping (minimum column height -> contact depth, area, percentile),
    # so there is nothing to collide with, and adding an eighth categorical hue to a
    # palette that is already at the CVD limit would have cost separation for nothing.
    # If it is ever plotted, give it a style rather than a hue.
    "minimum": "#e04b2f",          # red          -- MEFS / MCFS, the economic threshold
    "well_associated": "#b3a02f",  # olive        -- the discovery case
    "up_dip": "#4cb8e0",           # light blue   -- attic / up-dip / regret
    "below_lkh": "#cf9a4e",         # tan          -- well associated, not tested
    "well": "#7a4bb8",             # violet       -- the well itself
    # Green is the one hue the palette had not spent, and it reads as "commercial"
    # without being borrowed from anything else. Added 2026-08-15 with the commercial
    # volume class; the value is the output of a search over the green-teal region for
    # the largest CVD separation from every role it can share a figure with.
    "commercial": "#007c58",       # deep green   -- the accumulation given it clears MEFS
}

DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#a8a79f",
    "grid": "#383835",
    "prospect": "#4a6fb0",
    "tested": "#8a3a68",
    "minimum": "#e8593f",
    "well_associated": "#d6c04a",
    "up_dip": "#86dbf5",
    "below_lkh": "#e0ad63",
    "well": "#b184e6",
    "commercial": "#73be64",
}

# Sets of roles that can appear together in a single figure, and therefore have
# to be mutually distinguishable. Listed explicitly because the alternative --
# requiring all 21 pairs to separate -- is not achievable with seven categorical
# colours and would force a worse palette on figures that never show them side
# by side. `well` is exempt: it is always a labelled marker or a line with a text
# annotation, never a colour read off a legend.
CO_OCCURRING = {
    "A1 area-depth": ("prospect", "muted"),
    "A2 outcome tree": ("muted", "up_dip", "tested", "below_lkh"),
    "A3 chance decomposition": ("well_associated", "muted"),
    "A4 resource vs depth": ("prospect", "muted", "minimum"),
    "A5 exceedance": ("prospect", "well_associated", "tested", "up_dip", "minimum"),
    "C2 exceedance": ("prospect", "well_associated", "tested", "up_dip", "minimum",
                      "commercial"),
    "A6 overlap": ("tested", "up_dip", "minimum", "commercial"),
    "B0 section": ("up_dip", "tested", "below_lkh"),
    "B1 volume split": ("tested", "below_lkh", "up_dip", "muted"),
    "B2 chance vs regret": ("well_associated", "tested", "up_dip", "muted"),
    "B4 waterfall": ("well_associated", "muted"),
    "B12 bands": ("prospect", "tested", "minimum"),
    "map view": ("up_dip", "prospect", "tested"),
    # The risk elements, which share 5.1 and 5.2. Named with an "element:"
    # prefix so the CVD test can resolve them against ELEMENT_COLOURS rather
    # than against ROLES -- they are a different family.
    "B4/B5 risk elements": ("element:charge", "element:trap",
                            "element:reservoir", "element:retention"),
}

# Meaning -> palette key. Never index the palette by position.
#
# The left column is what the code asks for; the right is where it lands. Two
# aliases are deliberate: `discovery` is the well-associated case (the parity
# suite calls the same distribution "well associated volume"), and `proven` is
# what the well tests. `p_well` follows the volume it is the chance of.
ROLES = {
    "prospect": "prospect",
    "pos_prospect": "prospect",
    "discovery": "well_associated",
    "well_associated": "well_associated",
    "p_well": "well_associated",
    "proven": "tested",
    "tested": "tested",
    "below_lkh": "below_lkh",
    "attic": "up_dip",
    "up_dip": "up_dip",
    "regret": "up_dip",
    "minimum": "minimum",
    "mefs": "minimum",
    "well": "well",
}

#: One colour per geological risk element (Lars's card, 2026-08-12). A *different*
#: family from the volume palette and it has to stay that way: the volume roles say
#: which volume concept a curve is, these say which chance element a bar is, and no
#: figure shows both encodings at once.
#:
#: Saturated, not the pale fills on the card. Measured: as pale fills, charge and
#: reservoir separate by only dE 6.8 under simulated tritanopia, against this
#: project's dE 15 bar; saturated, the worst pair over all three deficiencies is
#: dE 16.8. The hues are unchanged -- red, blue, yellow, green -- only the lightness
#: is, which is exactly the tuning the volume palette had.
ELEMENT_COLOURS = {
    "charge": "#c62828",       # red
    "trap": "#1565c0",         # blue     -- displayed as "Closure"
    "reservoir": "#e6a700",    # yellow
    "retention": "#2e7d32",    # green
}

#: The pale versions, for a chip or a table cell *behind a written label*. Never for
#: a line or a bar, where the name is not on the mark and colour carries it alone.
ELEMENT_TINTS = {
    "charge": "#f6d6d6",
    "trap": "#d3e3f7",
    "reservoir": "#faedc4",
    "retention": "#d6e9d7",
}


def element_colour(key: str, dark: bool = False, *, tint: bool = False) -> str:
    """The colour for a risk element, by its stable key (``trap``, not ``Closure``)."""
    table = ELEMENT_TINTS if tint else ELEMENT_COLOURS
    if key not in table:
        raise KeyError(f"no colour for risk element {key!r}; expected {sorted(table)}")
    return table[key]


AREA_SCALES = {
    "area": ("Productive area (km²)", lambda a: a),
    "area²": ("Productive area² (km⁴)", lambda a: a ** 2),
    "√area": ("√ productive area (km)", lambda a: np.sqrt(np.maximum(a, 0.0))),
}
"""x-axis transforms for the area-depth panels.

GeoX plots its area-depth curve against area **squared**, so that convention is
offered rather than only ours. `sqrt(area)` is included too because it is the one
that straightens a conical closure, which makes departures from a simple cone easy
to see. The transform touches the axis only -- every number the tool computes is
in km2 regardless (non-negotiable 4).
"""


#: Kept for callers that still reference it; the legend is anchored to the figure
#: bottom now (see :func:`apply_plotly`) rather than to a fraction of the plot,
#: because a plot-relative offset drifted with the plot height and clipped.
LEGEND_Y = -0.20

#: The POS_prospect values the fan is drawn at. The workbook draws all one hundred
#: (`Well pos cal.` columns D..CZ, one per percent) which is unreadable on screen and
#: mostly redundant -- neighbouring curves differ by a hundredth. Ten deciles carry
#: the same message and can be told apart.
FAN_POS_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


SEQUENTIAL_CMAP = "Blues"   # single hue, light -> dark; never a rainbow

#: For colour that has to be read as a *value* rather than as "more or less":
#: A4's trial counts per cell, B6's ``P_well`` per marker.
#:
#: ``SEQUENTIAL_CMAP`` remains the default and the rule -- one hue, light to dark,
#: never a rainbow. But a single hue is genuinely hard to read as a quantity at
#: small mark sizes, which is what Lars reported of B6's 9 px points on
#: 2026-08-11, and it is why he asked for inferno on A4 before that. Inferno is
#: **perceptually uniform and monotonic in lightness**, so it is not the rainbow
#: the rule forbids: it survives greyscale printing and colour-vision deficiency
#: for the same reason a single hue does, while spending far more perceptual
#: distance over the range.
#:
#: Named here rather than written as a literal in each figure so the two cannot
#: drift apart -- a legend the reader learns on A4 has to still be true on B6.
VALUE_CMAP = "Inferno"


def palette(dark: bool = False) -> dict[str, str]:
    return dict(DARK if dark else LIGHT)


def colour(role: str, dark: bool = False) -> str:
    p = palette(dark)
    return p[ROLES.get(role, role)]


def rgba(role: str, alpha: float, dark: bool = False) -> str:
    """A role's colour as a plotly ``rgba(...)`` string at the given alpha.

    Exists so a translucent fill on the interactive path comes from the same
    role lookup as the opaque line beside it. Hardcoding an rgba literal is how
    a fill ends up frozen at its light-mode value while everything around it
    follows the palette.
    """
    hex_colour = colour(role, dark).lstrip("#")
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def apply(dark: bool = False) -> dict[str, str]:
    """Set matplotlib rcParams for the project. Returns the palette in use."""
    p = palette(dark)
    mpl.rcParams.update(
        {
            "figure.facecolor": p["surface"],
            "axes.facecolor": p["surface"],
            "savefig.facecolor": p["surface"],
            # Scaled from FONT_SIZES so the two backends stay in step. matplotlib
            # figures are drawn small and scaled up, hence the factor.
            "font.size": FONT_SIZES["body"] * 0.70,
            "axes.titlesize": FONT_SIZES["title"] * 0.70,
            "axes.labelsize": FONT_SIZES["axis_title"] * 0.70,
            "xtick.labelsize": FONT_SIZES["tick"] * 0.70,
            "ytick.labelsize": FONT_SIZES["tick"] * 0.70,
            "legend.fontsize": FONT_SIZES["legend"] * 0.70,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
            "axes.edgecolor": p["grid"],
            "axes.labelcolor": p["text_secondary"],
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": p["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": p["muted"],
            "ytick.color": p["muted"],
            "grid.color": p["grid"],
            "legend.frameon": False,
        }
    )
    return p


def depth_axis(ax, ylabel: str | None = "Depth (m TVDSS)", zlim: tuple[float, float] | None = None):
    """Configure ``ax`` to carry depth on y, increasing downward.

    ``zlim`` is given shallow-first, e.g. ``(3350, 3700)``; the axis is inverted
    for you. Pass ``ylabel=None`` for the second and later panels in a row that
    share a depth axis, so the tick labels appear only once.
    """
    if zlim is not None:
        lo, hi = float(zlim[0]), float(zlim[1])
        ax.set_ylim(max(lo, hi), min(lo, hi))
    elif ax.get_ylim()[0] < ax.get_ylim()[1]:
        ax.invert_yaxis()
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.tick_params(labelleft=False)
    ax.grid(True, lw=0.6, alpha=0.7)
    return ax


def is_depth_axis_correct(ax) -> bool:
    """True when y is inverted -- used by the axis test."""
    lo, hi = ax.get_ylim()
    return lo > hi


def new_figure(nrows=1, ncols=1, figsize=(12, 6), dark=False, **kw):
    apply(dark)
    return plt.subplots(nrows, ncols, figsize=figsize, **kw)


# ------------------------------------------------------------------- plotly
# The interactive figures live in viz/interactive.py and the export figures in
# viz/figures.py, but both are styled from the palette and the depth rule
# above, so the two paths cannot drift apart. Anything below is the plotly
# translation of a rule already stated for matplotlib -- never a second,
# independent set of choices.

# One height for every panel, in pixels. Figures are kept individual rather
# than merged into a subplot grid, so a row lines up only if each panel is the
# same height and carries the same depth range -- see depth_axis_plotly.
PANEL_HEIGHT = 560


#: Vertical space one row of horizontal legend entries takes, in px, measured off
#: the rendered app rather than guessed.
#: A quarter taller than a row panel, for the two figures Lars asked for more room
#: on (2026-08-12): 3.11, whose leader lines and five-deep contact family need the
#: vertical space to stay legible, and 3.12, which stacks up to ten curve families
#: on one pair of axes. Not a general default -- PANEL_HEIGHT still means "the height
#: a row of depth panels shares", and a row must stay a row.
TALL_PANEL_HEIGHT = int(PANEL_HEIGHT * 1.25)
BAND_PANEL_HEIGHT = TALL_PANEL_HEIGHT

LEGEND_ROW_PX = 21
#: Room for the x-axis title and tick labels, below which the legend starts.
AXIS_FOOT_PX = 62
#: Never reserve more than this. Beyond it the legend is the figure, and the answer
#: is fewer series rather than more margin.
MAX_LEGEND_PX = 260
#: A colourbar needs its own strip above the legend: the bar, its ticks and its title.
COLOURBAR_BAND_PX = 54


def legend_entries(fig) -> int:
    """How many entries the legend will actually show."""
    return sum(
        1 for tr in fig.data
        if getattr(tr, "name", None) and getattr(tr, "showlegend", None) is not False
    )


def _has_colourbar(fig) -> bool:
    for tr in fig.data:
        if getattr(tr, "colorbar", None) is not None and tr.colorbar.x is not None:
            return True
        marker = getattr(tr, "marker", None)
        if marker is not None and getattr(marker, "colorbar", None) is not None                 and marker.colorbar.x is not None:
            return True
    return False


def _place_colourbars(fig, n_entries: int, height: int | None,
                      bottom: int | None = None) -> None:
    """Put any colourbar in the reserved band, *above* the legend.

    Both were being anchored independently -- the legend to the figure bottom, the
    colourbar to a fraction of the plot -- and on A4 they landed on top of each
    other. There is one band of reserved space below the x-axis title, so one
    function has to divide it: the legend takes the bottom and the colourbar sits
    above it.

    **Anchored to the axis foot, not to the legend** (Lars, 2026-08-12: *"the colour
    bar on 3.11 needs to go down a bit"*). Measured up from the legend's *worst case*
    it was pushed hard against the x-axis title -- 14 px below it on 3.11, where the
    legend reserves for twelve entries and then renders two rows. The axis foot is
    the stable reference: it does not depend on how the browser wraps the legend. The
    floor keeps a realistic three-row legend clear, which is the collision that
    actually matters.
    """
    if height is None or not height:
        return
    legend_floor = LEGEND_ROW_PX * 3 + 18
    if bottom:
        y_px = max(int(bottom) - AXIS_FOOT_PX - 20, legend_floor)
    else:
        y_px = LEGEND_ROW_PX * max(int(n_entries), 1) + 18
    y = y_px / float(height)                      # container fraction, from the bottom
    spec = dict(orientation="h", x=0.5, xanchor="center", xref="container",
                yref="container", yanchor="bottom", y=y,
                len=0.42, thickness=10, tickfont=dict(size=9))
    for tr in fig.data:
        if getattr(tr, "colorbar", None) is not None and tr.colorbar.x is not None:
            tr.colorbar.update(**spec)
        marker = getattr(tr, "marker", None)
        if marker is not None and getattr(marker, "colorbar", None) is not None                 and marker.colorbar.x is not None:
            marker.colorbar.update(**spec)


def legend_margin(n_entries: int, *, colourbar: bool = False) -> int:
    """Bottom margin needed to show *all* of a horizontal legend, worst case.

    **Worst case is one entry per row**, and that is not pessimism -- it is what
    happens in a three-column row on this app, measured: B1 with six entries wrapped
    to six rows and was clipped by 75 px against the old fixed 125 px margin. Because
    Streamlit charts are width-responsive, the number of rows is decided in the
    browser at render time and cannot be known here; reserving for the worst case is
    the only thing that is always right.

    The cost is paid in figure *height*, not in plot area -- see
    :func:`apply_plotly`, which grows the figure by whatever the margin grew by. A
    taller figure with the same plot area is a strictly better trade than a legend
    with half its entries cut off.
    """
    needed = AXIS_FOOT_PX + LEGEND_ROW_PX * max(int(n_entries), 0)
    if colourbar:
        needed += COLOURBAR_BAND_PX
    cap = MAX_LEGEND_PX + (COLOURBAR_BAND_PX if colourbar else 0)
    return int(min(max(needed, AXIS_FOOT_PX + LEGEND_ROW_PX * 3), cap))


def level_row(*figs, height: int | None = None) -> None:
    """Give every figure in a row the same bottom margin and the same height.

    Panels in a row must share a plot area, not just a depth range -- otherwise the
    same depth lands on a different pixel row in each and the row cannot be read
    across, which is what non-negotiable 2 is for. Since the legend now sets the
    bottom margin and each panel has its own number of series, that sharing has to
    be imposed *after* the figures are built.

    The row takes the **largest** margin and the **largest** height among its
    members, so nothing is clipped anywhere and every panel still lines up. Mirrors
    :func:`wellvolpos.viz.interactive.row_zlim`, which does the same job for the
    depth range, and is called from the same place.
    """
    figs = [f for f in figs if f is not None]
    if not figs:
        return
    b = max(int(f.layout.margin.b or 0) for f in figs)
    h = height or max(int(f.layout.height or PANEL_HEIGHT) for f in figs)
    for f in figs:
        f.update_layout(margin=dict(b=b), height=h)
#: How much of the plot width a context histogram drawn on a second x-axis may take.
#: A sixth: the bars are context for the curve they sit behind, and given the whole
#: width they become the figure while the curve reads as an annotation on them.
CONTACT_SHARE = 1.0 / 6.0

#: Extra top margin for a figure carrying a titled x-axis on top of the plot area.
TOP_AXIS_BAND = 38


def _has_top_axis(fig) -> bool:
    """True when some x-axis is drawn on top *and* carries a title."""
    for key in fig.layout:
        if not str(key).startswith("xaxis"):
            continue
        ax = fig.layout[key]
        if getattr(ax, "side", None) == "top" and getattr(
                getattr(ax, "title", None), "text", None):
            return True
    return False




def apply_plotly(fig, dark: bool = False, height: int | None = PANEL_HEIGHT):
    """Style a plotly figure to match :func:`apply`'s matplotlib rcParams.

    The bottom margin is sized to the legend rather than fixed, and ``height`` grows
    by the same amount -- so a figure with twelve series is taller than one with two
    and both have the same plot area. See :func:`legend_margin`.
    """
    p = palette(dark)
    n_entries = legend_entries(fig)
    has_bar = _has_colourbar(fig)
    bottom = legend_margin(n_entries, colourbar=has_bar)
    # **A titled top axis gets its own band** (Lars, 2026-08-18). ``t=55`` fits a
    # title and its ``<sub>`` line and nothing else, so a figure with a second x-axis
    # on top printed "trials per bin" straight through the subtitle. ``autoexpand``
    # is off across the whole project -- deliberately, so a legend cannot shrink one
    # panel of a row -- which means nothing grows the margin on its own. Reserving
    # the band here rather than in the one figure that has such an axis today keeps
    # the next one from rediscovering the collision.
    top = 55 + (TOP_AXIS_BAND if _has_top_axis(fig) else 0)
    if height is not None:
        height = int(height) + max(0, bottom - legend_margin(3)) + (top - 55)
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=p["surface"],
        plot_bgcolor=p["surface"],
        font=dict(family=FONT_STACK, size=FONT_SIZES["body"], color=p["text"]),
        title=dict(font=dict(size=FONT_SIZES["title"], color=p["text"])),
        # autoexpand=False is load-bearing, not tidiness. With it on, plotly
        # grows the margins to make room for a legend or colour bar placed
        # outside the axes -- so one panel in a row acquiring a legend shrinks
        # its plot area and the row stops being level even though every axis
        # carries the identical range. Fixed margins plus one PANEL_HEIGHT mean
        # a given depth lands on the same pixel row in every panel.
        # b=125 reserves room *below the x-axis title* for the legend and any
        # colourbar (Lars, 2026-08-12). autoexpand stays off -- it is load-bearing,
        # not tidiness: with it on, plotly grows the margins to fit whatever sits
        # outside the axes, so one panel in a row acquiring a legend shrinks its
        # plot area and the row stops being level even though every axis carries
        # the identical range. Reserving the space on *every* figure instead keeps
        # a given depth on the same pixel row in every panel.
        margin=dict(l=70, r=25, t=top, b=bottom, autoexpand=False),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=10),
            orientation="h",
            # **Anchored to the figure, not to the plot area.** With the default
            # ``yref="paper"`` the legend's y is a fraction of the *plot* height, so
            # the gap below the axis grew with the plot and a legend that fitted on a
            # short figure overflowed a tall one -- three of them still clipped after
            # the margin was made legend-aware. ``yref="container"`` measures from the
            # figure edge instead, so the legend sits at the bottom and grows upward
            # into the reserved margin. It cannot run off the page.
            yref="container", yanchor="bottom", y=0.012,
            xref="container", xanchor="center", x=0.5,
        ),
        hovermode="closest",
    )
    if height is not None:
        fig.update_layout(height=height)
    _place_colourbars(fig, n_entries, height, bottom)
    axis = dict(
        gridcolor=p["grid"], zeroline=False, linecolor=p["grid"],
        tickfont=dict(color=p["muted"], size=11),
        title=dict(font=dict(color=p["text_secondary"], size=12)),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def depth_axis_plotly(
    fig,
    zlim: tuple[float, float] | None = None,
    title: str | None = "Depth (m TVDSS)",
    *,
    show_ticklabels: bool = True,
    row: int | None = None,
    col: int | None = None,
):
    """Configure a plotly y-axis to carry depth, increasing downward.

    The plotly counterpart of :func:`depth_axis`, and the only sanctioned way
    to set up such an axis on an interactive figure. ``zlim`` is given
    shallow-first and the axis is reversed for you; pass the *same* ``zlim`` to
    every figure in a row so the row can be read straight across at constant
    depth, which is what non-negotiable 2 is for. ``show_ticklabels=False``
    suppresses the repeated numbers on the second and later panels of a row,
    matching ``depth_axis``' ``ylabel=None`` behaviour.
    """
    kw: dict = {"title": title if title else None}
    if zlim is not None:
        lo, hi = float(zlim[0]), float(zlim[1])
        # Descending range is what inverts a plotly axis; autorange="reversed"
        # would fight an explicit range.
        kw["range"] = [max(lo, hi), min(lo, hi)]
        kw["autorange"] = False
    else:
        kw["autorange"] = "reversed"
    if not show_ticklabels:
        kw["showticklabels"] = False
        kw["title"] = None
    target = fig.update_yaxes
    if row is not None and col is not None:
        target(**kw, row=row, col=col)
    else:
        target(**kw)
    return fig


def is_depth_axis_correct_plotly(fig, axis: str = "yaxis") -> bool:
    """True when the named plotly y-axis increases downward -- the axis test."""
    ax = getattr(fig.layout, axis, None)
    if ax is None:
        return False
    if ax.autorange == "reversed":
        return True
    rng = ax.range
    return rng is not None and rng[0] > rng[1]

#: Short names for the reference contour, for figure titles. A figure that draws
#: ``P_well`` without saying which contour ``r_location`` was measured against is
#: showing a number whose meaning depends on a setting the reader cannot see --
#: against non-negotiable 5, and it matters most on the export path, where the
#: app's caption is not there to make up for it.
REFERENCE_SHORT = {"crest": "crest-referenced", "p90_area": "P90-area-referenced"}


def reference_label(reference) -> str:
    """``'crest-referenced'`` / ``'P90-area-referenced'`` from an enum or a string."""
    value = getattr(reference, "value", reference)
    return REFERENCE_SHORT.get(str(value), str(value))


# --------------------------------------------------------- log-probit axes
#: Default histogram fill on 4.5. Lower than the 0.45 it was, because that figure
#: is *about* the overlap and five series at 0.45 hide each other. Adjustable in the
#: app; this is only where it starts.
OVERLAP_OPACITY = 0.20

#: Bars on 4.7, doubled from 40 on 2026-08-16 (Lars). With five overlapping
#: series the figure is about where one series' mass sits relative to another,
#: and 40 bins over a range running to ~480 MMboe is a 12 MMboe bar -- wide
#: enough to hide the shoulder where two classes part company.
OVERLAP_BINS = 80

#: The type scale, shared by both backends so an exported figure is set like the one
#: on screen. The title was 14 against a 12 body -- two points is not a hierarchy, and
#: several titles now carry a ``<sub>`` subtitle that was rendering at the same size as
#: the title itself (Lars, 2026-08-15, design review).
#:
#: **One stack for both backends, DejaVu first -- and that is a compromise, not a
#: solution.** matplotlib asked for DejaVu Sans; plotly asked for
#: ``DejaVu Sans, Arial, sans-serif`` and in a browser almost always landed on Arial,
#: so a PNG in a report and the same figure on screen were set in different type.
#:
#: Putting Segoe UI first closed that gap and immediately broke the export: Segoe UI
#: has no PROPORTIONAL TO glyph, which A1's caption uses, and matplotlib drew a box.
#: **Glyph coverage beats typeface parity** -- a missing character in an exported PDF
#: is worse than a slightly different sans on screen. So DejaVu leads, matplotlib uses
#: it (it ships with matplotlib), and a browser without it falls through to Segoe UI.
#:
#: True parity needs a bundled webfont. Worth doing if the export path ever matters
#: more than it does today; it is not worth a dependency yet.
FONT_STACK = "DejaVu Sans, Segoe UI, Helvetica, Arial, sans-serif"
FONT_SIZES = {
    "title": 17,
    "subtitle": 11,
    "axis_title": 12,
    "tick": 11,
    "legend": 11,
    "body": 12,
    "annotation": 10,
}

#: Characters before a long figure title wraps. Without it plotly breaks a title
#: wherever it happens to run out of room, which at a narrow viewport is mid-word.
TITLE_WRAP = 78

def crosshair(fig, dark: bool = False):
    """Spikelines on both axes, for a figure whose axes are both readable quantities.

    Design plan §6.4 asked for this and it was never built. On an exceedance curve,
    reading a probability off a volume *is* the interaction, and ``hovermode="closest"``
    without spikelines leaves the reader estimating against a gridline.

    **Not on the depth panels.** There the shared depth range already lets a reader
    carry a value across a row, and a spike on every one of ten panels is noise. This
    is for 2.3, 4.2 and 3.12, where x is a volume and y is a probability.
    """
    p = palette(dark)
    for axis in (fig.update_xaxes, fig.update_yaxes):
        axis(showspikes=True, spikemode="across", spikesnap="cursor",
             spikecolor=p["muted"], spikethickness=1, spikedash="dot")
    fig.update_layout(hovermode="closest")
    return fig


#: The exceedance percentiles the probit grid is ruled at. Fixed, and deliberately
#: *not* the ladder a given figure ends up drawing (see
#: :func:`wellvolpos.core.bands.supported_percentiles`): two figures with
#: different trial counts must still share a y grid, or they cannot be compared.
PROBIT_TICKS = (99, 95, 90, 75, 50, 25, 10, 5, 1)

#: A little beyond the outermost tick, so P99 and P1 are not on the frame.
PROBIT_PAD = 0.4


def probit(p):
    """Probit coordinate of a probability in *percent*.

    ``ndtri`` is the inverse standard-normal CDF, the same spelling
    :mod:`wellvolpos.io.synthetic` uses. Straightness on a log-probit plot is
    lognormality, which is the whole reason for the transform: it turns a
    distributional claim into a question about a ruler.
    """
    a = np.clip(np.asarray(p, dtype=float) / 100.0, 1e-6, 1 - 1e-6)
    return ndtri(a)


def probit_axis_plotly(fig, *, title="Exceedance probability", ticks=PROBIT_TICKS):
    """Rule a plotly y-axis as an exceedance-probability probit scale.

    P99 sits at the top and P1 at the bottom, because a **P90 is a small volume**
    under this project's exceedance convention -- so volume increasing rightward
    makes every curve descend. Schneider et al. (2023) Figure 9 runs the other
    way, on cumulative-less-than percentiles; it is the same distribution read
    from the other end, and the axis title says which is on screen.
    """
    fig.update_yaxes(
        title=title,
        tickmode="array",
        tickvals=[float(probit(t)) for t in ticks],
        ticktext=[f"P{t}" for t in ticks],
        range=[float(probit(min(ticks))) - PROBIT_PAD,
               float(probit(max(ticks))) + PROBIT_PAD],
        autorange=False,
    )
    return fig


def probit_axis(ax, *, ylabel="Exceedance probability", ticks=PROBIT_TICKS):
    """The matplotlib twin of :func:`probit_axis_plotly`."""
    ax.set_ylabel(ylabel)
    ax.set_yticks([float(probit(t)) for t in ticks])
    ax.set_yticklabels([f"P{t}" for t in ticks])
    ax.set_ylim(float(probit(min(ticks))) - PROBIT_PAD,
                float(probit(max(ticks))) + PROBIT_PAD)
    return ax


#: How the exceedance probability axis may be scaled. Probit is the default because
#: it turns a lognormal into a straight line, which is a claim a reader can check
#: with a ruler; linear is offered because a probability *is* linear and the probit
#: distortion has to be earned rather than assumed.
PROBABILITY_SCALES = ("probit", "linear")

#: How a volume axis may be scaled, where the figure offers the choice.
VOLUME_SCALES = ("log", "linear")


def probability_coords(p, scale: str = "probit"):
    """Plot coordinates for an exceedance probability in percent, under ``scale``."""
    if scale not in PROBABILITY_SCALES:
        raise ValueError(
            f"unknown probability scale {scale!r}; expected one of {PROBABILITY_SCALES}"
        )
    return probit(p) if scale == "probit" else np.asarray(p, dtype=float)


def probability_axis_plotly(fig, scale: str = "probit", *,
                            title="Exceedance probability", ticks=PROBIT_TICKS):
    """Rule a plotly y-axis as an exceedance probability, probit or linear."""
    if scale == "probit":
        return probit_axis_plotly(fig, title=f"{title} · probit scale", ticks=ticks)
    fig.update_yaxes(
        title=f"{title} (%)", tickmode="array",
        tickvals=list(range(0, 101, 10)),
        ticktext=[f"{t}" for t in range(0, 101, 10)],
        range=[0, 102], autorange=False,
    )
    return fig


def probability_axis(ax, scale: str = "probit", *,
                     ylabel="Exceedance probability", ticks=PROBIT_TICKS):
    """The matplotlib twin of :func:`probability_axis_plotly`."""
    if scale == "probit":
        return probit_axis(ax, ylabel=f"{ylabel} · probit scale", ticks=ticks)
    ax.set_ylabel(f"{ylabel} (%)")
    ax.set_yticks(list(range(0, 101, 10)))
    ax.set_yticklabels([str(t) for t in range(0, 101, 10)])
    ax.set_ylim(0, 102)
    return ax


def probability_axis_range(scale: str = "probit", *, ticks=PROBIT_TICKS):
    """The (low, high) plot coordinates a probability axis spans, under ``scale``.

    Needed by anything drawn *across* the axis rather than on it -- a reference rule
    given as a trace, for instance, which has to know where the axis ends because a
    trace has no equivalent of a shape's "span the whole plot".
    """
    if scale == "probit":
        return (float(probit(min(ticks))) - PROBIT_PAD,
                float(probit(max(ticks))) + PROBIT_PAD)
    return (0.0, 102.0)


def concept_shades(role: str, n: int, dark: bool = False, *,
                   lo: float = 0.28, hi: float = 1.0):
    """``n`` hex colours ramping light to dark **through a palette role's own hue**.

    For a figure that has to order a quantity (depth, here) within *each* of two
    volume concepts. One sequential ramp per concept keeps the concept in the hue,
    where the palette puts it, and the ordering in the lightness -- which is what
    :data:`SEQUENTIAL_CMAP` does for a single family and cannot do for two.

    Both families were drawn from ``SEQUENTIAL_CMAP`` at first and Lars reported the
    obvious consequence: *"they are both blues now"*. Line style alone does not
    separate two families whose members interleave.
    """
    import matplotlib as mpl

    base = colour(role, dark)
    p = palette(dark)
    # Light end: the role's colour blended most of the way to the page; dark end:
    # blended towards black. Built from the role rather than hand-picked, so a
    # palette change carries through and the hue cannot drift from the concept.
    light = mpl.colors.to_rgb(p["surface"])
    mid = np.asarray(mpl.colors.to_rgb(base), dtype=float)
    dark_end = mid * 0.45
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{role}_ramp", [light, tuple(mid), tuple(dark_end)]
    )
    if n <= 1:
        return [mpl.colors.to_hex(cmap(0.6))]
    return [mpl.colors.to_hex(cmap(t)) for t in np.linspace(lo, hi * 0.85, int(n))]


def depth_shades(n: int, dark: bool = False, *, lo: float = 0.32, hi: float = 0.95):
    """``n`` hex colours from the sequential scale, shallow (light) to deep (dark).

    The sanctioned use of :data:`SEQUENTIAL_CMAP`: depth here is a *quantity* with
    an order, so one hue light-to-dark is exactly right and a categorical cycle
    would be exactly wrong. Sampling starts at ``lo`` rather than at 0 because
    the pale end of a single-hue scale is invisible on white.

    Drawn from matplotlib rather than from plotly's copy of the same scale so the
    two backends cannot pick different blues for the same band.
    """
    import matplotlib as mpl

    cmap = mpl.colormaps[SEQUENTIAL_CMAP]
    if n <= 1:
        return [mpl.colors.to_hex(cmap(hi))]
    return [mpl.colors.to_hex(cmap(t)) for t in np.linspace(lo, hi, int(n))]

def log_ticks(lo: float, hi: float, *, subs=(1.0, 2.0, 5.0)):
    """Tick values for a logarithmic *volume* axis: 1-2-5 per decade, as plain numbers.

    Both backends label a log axis badly by default and badly in different ways --
    matplotlib writes ``4 x 10^0``, which is a physicist's notation for an axis
    carrying MMboe, and plotly labels every minor decade step, which ran the numbers
    into each other across prospect B's 3.6-420 MMboe range. Shared from here so the
    two cannot drift, since a figure and its export twin disagreeing about the ticks
    is the same defect as disagreeing about the data.
    """
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    d0 = int(np.floor(np.log10(lo))) - 1
    d1 = int(np.ceil(np.log10(hi))) + 1
    vals = [s * 10.0 ** d for d in range(d0, d1 + 1) for s in subs]
    return [v for v in vals if lo / 2.0 <= v <= hi * 2.0]


def log_tick_text(vals):
    """Plain labels for :func:`log_ticks` -- no exponents, no trailing zeros."""
    return [f"{v:g}" for v in vals]
