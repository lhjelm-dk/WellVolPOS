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
