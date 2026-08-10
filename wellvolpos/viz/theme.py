"""One styling entry point, shared by every figure in the project.

Two rules here are load-bearing rather than decorative.

**Colour is assigned by meaning, never cycled.** Blue is always the discovery
case or a chance; orange is always the attic / up-dip / regret; yellow is always
proven; aqua is always the prospect total. A reader who learns the mapping once
reads every figure in the tool. The palette validates colourblind-safe across
all pairs in both light and dark mode.

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

# --------------------------------------------------------------- palette
LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#8a8983",
    "grid": "#e6e5e1",
    "discovery": "#2a78d6",   # blue    -- discovery case, chance
    "attic": "#eb6834",       # orange  -- attic / up-dip / regret
    "prospect": "#1baf7a",    # aqua    -- prospect totals
    "proven": "#eda100",      # yellow  -- proven at the well
    "well": "#4a3aa7",        # violet  -- the well itself
}

DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#8a8983",
    "grid": "#383835",
    "discovery": "#3987e5",
    "attic": "#d95926",
    "prospect": "#199e70",
    "proven": "#c98500",
    "well": "#9085e9",
}

# Meaning -> palette key. Never index the palette by position.
ROLES = {
    "prospect": "prospect",
    "discovery": "discovery",
    "proven": "proven",
    "possible": "prospect",
    "attic": "attic",
    "regret": "attic",
    "p_well": "discovery",
    "well": "well",
}

SEQUENTIAL_CMAP = "Blues"   # single hue, light -> dark; never a rainbow


def palette(dark: bool = False) -> dict[str, str]:
    return dict(DARK if dark else LIGHT)


def colour(role: str, dark: bool = False) -> str:
    p = palette(dark)
    return p[ROLES.get(role, role)]


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
