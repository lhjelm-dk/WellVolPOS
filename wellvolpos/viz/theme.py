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
    "minimum": "#e04b2f",          # red          -- a threshold volume
    "well_associated": "#b3a02f",  # olive        -- the discovery case
    "up_dip": "#4cb8e0",           # light blue   -- attic / up-dip / regret
    "possible": "#ebe4bc",         # pale khaki   -- well associated, not tested
    "well": "#7a4bb8",             # violet       -- the well itself
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
    "possible": "#faf4dd",
    "well": "#b184e6",
}

# Sets of roles that can appear together in a single figure, and therefore have
# to be mutually distinguishable. Listed explicitly because the alternative --
# requiring all 21 pairs to separate -- is not achievable with seven categorical
# colours and would force a worse palette on figures that never show them side
# by side. `well` is exempt: it is always a labelled marker or a line with a text
# annotation, never a colour read off a legend.
CO_OCCURRING = {
    "A1 area-depth": ("prospect", "muted"),
    "A2 outcome tree": ("muted", "up_dip", "tested", "possible"),
    "A3 chance decomposition": ("well_associated", "muted"),
    "A4 resource vs depth": ("prospect", "muted", "minimum"),
    "A5 exceedance": ("prospect", "well_associated", "tested", "up_dip", "minimum"),
    "A6 overlap": ("tested", "up_dip", "minimum"),
    "B0 section": ("up_dip", "tested", "possible"),
    "B1 volume split": ("tested", "possible", "up_dip", "muted"),
    "B2 chance vs regret": ("well_associated", "tested", "up_dip", "muted"),
    "B4 waterfall": ("well_associated", "muted"),
    "map view": ("up_dip", "prospect", "tested"),
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
    "possible": "possible",
    "attic": "up_dip",
    "up_dip": "up_dip",
    "regret": "up_dip",
    "minimum": "minimum",
    "mefs": "minimum",
    "well": "well",
}

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
            "font.size": 8.5,
            "font.family": "DejaVu Sans",
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
PANEL_HEIGHT = 470


def apply_plotly(fig, dark: bool = False, height: int | None = PANEL_HEIGHT):
    """Style a plotly figure to match :func:`apply`'s matplotlib rcParams."""
    p = palette(dark)
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=p["surface"],
        plot_bgcolor=p["surface"],
        font=dict(family="DejaVu Sans, Arial, sans-serif", size=12, color=p["text"]),
        title=dict(font=dict(size=14, color=p["text"])),
        # autoexpand=False is load-bearing, not tidiness. With it on, plotly
        # grows the margins to make room for a legend or colour bar placed
        # outside the axes -- so one panel in a row acquiring a legend shrinks
        # its plot area and the row stops being level even though every axis
        # carries the identical range. Fixed margins plus one PANEL_HEIGHT mean
        # a given depth lands on the same pixel row in every panel.
        margin=dict(l=70, r=25, t=55, b=55, autoexpand=False),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11),
            yanchor="top", y=0.99, xanchor="right", x=0.99,
        ),
        hovermode="closest",
    )
    if height is not None:
        fig.update_layout(height=height)
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
