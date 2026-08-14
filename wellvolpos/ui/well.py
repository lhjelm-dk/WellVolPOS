"""The well: one reservoir entry, one reservoir exit.

The app held up to four candidate locations between 2026-08-13 and 2026-08-14, on the
argument that *"a prospect assessor doesn't ask what does a well at 2205 m give me,
they ask A, B or C"*. **Lars removed it**: the cost was not the model, which is small,
but that only two of the twelve quantities a swept figure draws actually move with the
exit, so most of the comparison was four copies of one curve and the two that did
differ needed a curve each on six figures before the tab was honest. Do not rebuild it
without asking -- see CLAUDE.md for the full argument.

**The widgets live in tab ① and are read at the top of ``app.py``.** That is the same
arrangement the chance table uses and it works for the same reason: Streamlit runs top
to bottom, ``entry``/``exit_`` are needed before any tab renders, and a widget owns its
key so the next rerun sees the change with no second copy of the state.
:func:`read_well` is the top-of-script read; :func:`well_editor` creates the widgets
later.

Defaults are seeded **beside the read**, never beside the widget -- seeding them in the
tab left the first run after a file change reading the old default under a control
already showing the new one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

#: Default entry-to-exit spacing, in metres. The same number the sweeps use as their
#: hypothetical gap, so the well is the same shape as the swept one.
DEFAULT_GAP_M = 50.0

ENTRY_KEY = "w_entry"
EXIT_KEY = "w_exit"


@dataclass(frozen=True)
class Well:
    """The proposed location: a reservoir entry and a reservoir exit."""

    entry: float
    exit: float

    @property
    def gap(self) -> float:
        return self.exit - self.entry

    def describe(self) -> str:
        return f"{self.entry:,.0f}–{self.exit:,.0f} m"


def default_entry(ts, zmin: float, zmax: float) -> float:
    """The opening entry depth: the median successful contact, rounded to 5 m.

    Derived from the data rather than hardcoded. It used to be 3500 m -- the reference
    well of one particular workbook -- which on the other demo prospect landed on its
    *deepest* contact and collapsed the exit range to a single point.
    """
    succ = ts.col("contact")[ts.col("resource") > 0]
    if not succ.size:
        return float(zmin)
    value = float(np.round(np.median(succ) / 5.0) * 5.0)
    return float(np.clip(value, zmin, zmax))


def read_well(ts, zmin: float, zmax: float) -> Well:
    """The well, seeding it on the first run.

    Called at the **top** of ``app.py``, before any tab renders, because the entry and
    exit drive the grouping, the split and the chance -- all computed once and passed
    down in the ``Ctx``.
    """
    seed = default_entry(ts, zmin, zmax)
    entry = float(st.session_state.setdefault(ENTRY_KEY, seed))
    # **Clamped and written back**, not just clamped. A new trial file moves the
    # contact range, and a stored depth outside a slider's bounds is a hard error in
    # Streamlit rather than a silently corrected value -- so the state has to be made
    # valid here, where the range is known, and not in the widget.
    entry = float(np.clip(entry, zmin, zmax))
    st.session_state[ENTRY_KEY] = entry

    exit_ = float(st.session_state.setdefault(
        EXIT_KEY, min(entry + DEFAULT_GAP_M, max(zmax, entry + 5.0))))
    exit_ = float(np.clip(exit_, entry, max(zmax, entry + 5.0)))
    st.session_state[EXIT_KEY] = exit_
    return Well(entry=entry, exit=exit_)


def well_editor(well: Well, zmin: float, zmax: float):
    """The two sliders, drawn wherever the tab wants them.

    Creates the widgets that own the keys :func:`read_well` reads. The *values* are not
    returned -- the next rerun picks them up at the top of the script, which is what
    keeps a single source of truth for the well geometry.

    What **is** returned is an empty slot for the deviation verdict, because that
    verdict needs the reservoir thickness and therefore ``A(z)``, which does not exist
    yet at the point in ``app.py`` where these sliders are drawn. ``st.empty()``
    reserves its position where it is *declared*, so the caption still lands directly
    under the sliders. Fill it with :func:`deviation_caption`.
    """
    left, right = st.columns(2)
    with left:
        # **No `value=` argument.** :func:`read_well` has already seeded and clamped
        # the key, and passing a default as well is the one thing Streamlit refuses
        # outright: "created with a default value but also had its value set via the
        # Session State API". The key is the single source.
        entry = st.slider("Reservoir entry (m TVDSS)", min_value=zmin, max_value=zmax,
                          step=5.0, key=ENTRY_KEY)
    with right:
        # ``max_value`` cannot equal ``min_value``, and an exit at or below the deepest
        # contact is meaningful -- it says the well passes through the whole reservoir
        # -- so the range is widened rather than clamped.
        st.slider("Reservoir exit (m TVDSS)", min_value=entry,
                  max_value=max(zmax, entry + 5.0), step=5.0, key=EXIT_KEY)

    return st.empty()


def deviation_caption(slot, well: Well, thickness) -> None:
    """**Is this spacing a vertical well?**

    The reservoir has a true vertical thickness, recovered per trial from the pay, and
    a vertical well entering at the top and leaving at the base sees exactly that. So
    the entry-to-exit spacing is not a free choice unless the well is deviated -- a
    wider spacing is a commitment to one drilled down-dip inside the layer, and the
    volumes are computed as though it had been drilled.

    Reported, never enforced: all three verdicts are legitimate wells.
    """
    if thickness is None:
        return
    from ..core.dependence import check_deviation

    chk = check_deviation(thickness, well.gap)
    icon = {"vertical": "⟂", "deviated": "⟋", "partial": "⊣", "unknown": "?"}[chk.verdict]
    slot.caption(f"{icon} **{chk.verdict}** — {well.gap:,.0f} m of section against a "
                 f"{chk.thickness_p50:,.0f} m reservoir. {chk.message()}")
