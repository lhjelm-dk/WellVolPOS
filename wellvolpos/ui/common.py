"""Helpers every tab needs, in one place rather than four.

Nothing here decides anything -- it is presentation only. The arithmetic lives in
``wellvolpos.core`` and the figures in ``wellvolpos.viz``, which is what lets this
module be imported by every tab without creating a cycle.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..core.chance import ELEMENT_LABELS
from ..viz.theme import (
    PANEL_HEIGHT,
    element_colour,
)
from .context import Ctx
from .numbering import renumber_title

#: C2 is a *stacked composite*, not a panel in a row. At the shared panel height
#: both of its halves were squashed to the point where the braces collided with
#: the axis, so it gets its own.
# A quarter taller again on 2026-08-14: eight curves, their value labels and the
# nesting braces below the zero line, all on one pair of axes.
C2_HEIGHT = 775

__all__ = ["C2_HEIGHT", "badge", "chart", "element_chip", "split_caveat"]


def badge(level: str) -> str:
    return {"pass": "✅", "warn": "⚠️", "fail": "⛔"}[level]


def chart(fig, key: str, height: int | None = None):
    """Renumber a figure by its tab, then render it.

    The renumbering happens here because ``chart`` is the one place every figure
    passes through *and* already knows its key -- so no figure function has to be
    told which tab it is on, and ``ui/numbering.py`` stays the single source of the
    mapping. See that module for why the letter codes survive in the code while the
    numbers appear on screen.

    ``height`` is pinned rather than left at Streamlit's default of ``"content"``:
    on ``"content"`` each chart is sized from its own contents, so panels in a
    row end up different heights and a shared depth range still does not put a
    given depth on the same pixel row.

    Left unset it takes the **figure's own** ``layout.height``, which is where
    ``theme.apply_plotly`` and ``theme.level_row`` have already put the answer --
    including the extra space a tall legend needs. It used to default to
    ``PANEL_HEIGHT`` instead, which silently squashed every figure that had asked
    for more room: 3.11 and 3.12 came back at 560 px having requested 700, because
    the Streamlit container height wins over the layout's. Callers that pass a
    height still override, and the row callers that pass ``int(fig.layout.height)``
    now say the same thing twice harmlessly.

    ``theme=None`` keeps ``wellvolpos.viz.theme`` authoritative. Streamlit's own
    plotly theme otherwise restyles fonts, title and template on top of ours,
    which is exactly the drift between the two backends that
    "both driven from viz/theme.py" rule exists to prevent.
    """
    title = getattr(getattr(fig.layout, "title", None), "text", None)
    if title:
        fig.update_layout(title=dict(text=renumber_title(title, key)))
    if height is None:
        height = int(getattr(fig.layout, "height", None) or PANEL_HEIGHT)
    return st.plotly_chart(fig, width="stretch", height=height, theme=None, key=key)


def split_caveat(ctx: Ctx) -> None:
    """Say when the proven/possible split cannot be trusted, or was approximated.

    Raised wherever the split's own numbers are drawn, not only in the QC list --
    a reader who goes straight to tab ③ never sees that list.

    The correlation check *warns* rather than fails (decided 2026-08-10): the
    assumption it tests belongs to the extension alone, and blocking on it closed
    the reference engine, which apportions nothing, along with it.
    """
    if (ctx.split_level == "warn" and np.isfinite(ctx.split_r)
            and abs(ctx.split_r) >= 0.5):
        st.warning(
            f"**The proven/possible split is not defensible on this data.** "
            f"{ctx.split_message}"
        )
    if ctx.vc is None:
        return
    # The apportionment changed on 2026-08-11 and moved the headline numbers by
    # about 8 %, so it is stated wherever the split's own numbers are drawn rather
    # than only in the footer.
    if ctx.vc.apportionment == "area":
        st.info(
            "**Split apportioned by map area**, not on the wedge — this file carries no "
            "gross pay and no HC gross rock volume, so the reservoir thickness cannot be "
            "recovered and the wedge cannot be built. The area rule assumes uniform pay per "
            "unit area, which understates proven and overstates possible."
        )
    elif ctx.vc.n_thickness_assumed:
        st.caption(
            f"{ctx.vc.n_thickness_assumed:,} discovery trials could not resolve a reservoir "
            f"thickness from pay and were treated as **charged to base**, which is what the "
            f"thickness inversion flags them as."
        )


def element_chip(key: str) -> str:
    """A small coloured nameplate for one risk element, as HTML.

    Lars asked for the four element colours on the chance-table inputs as well as on
    5.1 and 5.2, so that a number typed here can be found again on the figures
    without counting rows. A Streamlit ``number_input`` label cannot be coloured, so
    the colour goes on a chip above it.

    **The name is written on the chip**, which is the point rather than decoration:
    the tint alone would put charge and reservoir at dE 6.8 under simulated
    tritanopia, well inside this project's dE 15 bar. With the name on the mark the
    colour is a second channel and never the only one.
    """
    fill = element_colour(key, tint=True)
    edge = element_colour(key)
    return (
        f'<div style="background:{fill};border-left:5px solid {edge};'
        f'border-radius:3px;padding:2px 8px;margin-bottom:2px;'
        f'font-size:0.82rem;font-weight:600;color:#1b1b1b;">'
        f'{ELEMENT_LABELS[key]}</div>'
    )


# ------------------------------------------------------------------ KPI ladder
#: The rungs, in reading order. ``Pmean`` sits between P50 and P10 because that is
#: where a right-skewed distribution puts it, and it is labelled so it cannot be read
#: as a percentile.
LADDER = ("p99", "p90", "p50", "mean", "p10", "p1")
LADDER_LABELS = {"p99": "P99", "p90": "P90", "p50": "P50",
                 "mean": "Pmean", "p10": "P10", "p1": "P1"}


def kpi_ladder(*, chance_label: str, chance: float, values: dict,
               chance_help: str = "", value_help: str = "",
               deltas: dict | None = None) -> None:
    """One chance and one percentile ladder, in one shape, wherever it appears.

    Lars, 2026-08-15, from the design review: tab ② and tab ④ reported comparable
    quantities in different layouts, so the eye could not carry one across to the
    other -- which is the comparison the whole tool exists to make. Now the prospect's
    row and the well's row are the same row twice and can be read one above the other.

    **Chance first, then the volumes**, because that is the order the argument runs in
    and because it puts the one unconditional number where it cannot be mistaken for
    one of the six conditional ones beside it.
    """
    cols = st.columns(len(LADDER) + 1)
    cols[0].metric(chance_label, f"{chance:.1%}", help=chance_help or None,
                   delta=(deltas or {}).get("chance"))
    for col, key in zip(cols[1:], LADDER):
        v = values.get(key, float("nan"))
        col.metric(LADDER_LABELS[key], "—" if v != v else f"{v:,.2f}",
                   help=value_help or None, delta=(deltas or {}).get(key))


def track_deltas(slot: str, fingerprint: str, well: tuple, values: dict,
                 fmt: str = "{:+.2f}") -> dict:
    """What each number did when the well last moved.

    The tool's whole subject is sensitivity to one depth, and moving that depth
    re-rendered everything with no indication of what moved. This holds the values
    from before the last change and returns the differences.

    **The comparison is dropped when the fingerprint changes.** A delta measured
    across a different trial file, a different chance table or a different threshold
    is not a sensitivity, it is two unrelated numbers subtracted -- and it would look
    exactly like a real one. ``fingerprint`` is whatever the caller decides makes a
    comparison meaningful.

    **It persists while the well is still.** Streamlit reruns on every widget touch,
    so comparing against the immediately previous run would show zero for everything
    the moment the user clicked anything else. The stored snapshot only advances when
    the well actually moves, so the delta on screen answers "what did that move do"
    until the next one.
    """
    prev = st.session_state.get(f"_delta_prev_{slot}")
    if prev is None or prev.get("fp") != fingerprint:
        st.session_state[f"_delta_prev_{slot}"] = {
            "fp": fingerprint, "well": well, "vals": dict(values)}
        st.session_state.pop(f"_delta_last_{slot}", None)
        return {}
    if prev.get("well") != well:
        out = {k: fmt.format(values[k] - prev["vals"][k])
               for k in values
               if k in prev["vals"] and values[k] == values[k]
               and prev["vals"][k] == prev["vals"][k]}
        st.session_state[f"_delta_prev_{slot}"] = {
            "fp": fingerprint, "well": well, "vals": dict(values)}
        st.session_state[f"_delta_last_{slot}"] = out
        return out
    return st.session_state.get(f"_delta_last_{slot}", {})


def figure_note(headline: str, detail: str = "", *,
                label: str = "How to read this") -> None:
    """One line under a figure, with the argument folded away behind it.

    Tab (3) carried 2,082 words of caption under twelve figures -- around seventeen
    minutes of reading to choose a depth, which meant it got skimmed, which meant the
    one caption that would have changed the answer got skimmed too. The writing was
    not the problem; the volume was.

    So the caption becomes two tiers. ``headline`` is one or two sentences carrying a
    live number and nothing else -- no restating the axis labels, no throat-clearing.
    ``detail`` is what was there before, one click away for the reader who wants it.

    Nothing is deleted by this. It moves.
    """
    st.caption(headline)
    if detail:
        with st.expander(label):
            st.markdown(detail)


def well_readout(entry: float, exit_: float, *, note: str = "") -> None:
    """The current well, at the top of a tab that is about it.

    Lars, 2026-08-15, design review: the sliders live on tab ① and a reader on ③ or ④
    could not see what they were set to without going back. This is **display only**.

    Not a second pair of sliders, and that is a constraint rather than a preference:
    two widgets cannot own one session-state key, so a duplicate would either raise or
    need its own key -- and a second key is a second source of truth for the one number
    every figure in the app is computed from. One place to set it, everywhere to see it.
    """
    st.caption(
        f"**Well:** {entry:,.0f} – {exit_:,.0f} m TVDSS  ·  "
        f"{exit_ - entry:,.0f} m of reservoir penetrated"
        + (f"  ·  {note}" if note else "")
        + "  ·  *set on tab ①*"
    )
