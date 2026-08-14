"""Well candidates: up to four locations, defined once and compared everywhere.

Lars, 2026-08-13, after the review named this the biggest practical gap: *"a prospect
assessor doesn't ask what does a well at 2205 m give me, they ask A, B or C"*. The
sliders used to sit in the sidebar and describe exactly one well, so comparing two
locations meant moving a slider and trying to remember the previous number.

The model is deliberately small:

* **Well A always exists.** There is no state in which the app has no well, so every
  tab downstream keeps working unchanged.
* **B, C and D are added on demand**, up to :data:`MAX_WELLS`. They are *options under
  investigation*, drawn on tab ③'s swept figures as extra rules so their positions can
  be read against every curve at once.
* **Exactly one is carried into tab ④.** That tab answers "what do I get at the depth
  I chose", which is a question about one well; letting four through would turn every
  figure on it into a comparison and lose the thing it is for.

**The widgets live in the tabs and are read at the top of ``app.py``.** That is the
same arrangement the chance table uses and it works for the same reason: Streamlit
runs top to bottom, ``entry``/``exit_`` are needed before any tab renders, and a
widget owns its key so the next rerun sees the change with no second copy of the
state. :func:`read_wells` is the top-of-script read; :func:`well_editor` creates the
widgets later.

Defaults are seeded **beside the read**, never beside the widget -- seeding them in
the tab left the first run after a file change reading the old default under a
control already showing the new one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

#: How many candidates the app will hold. Four covers every well proposal argument
#: worth having on one page; beyond that the comparison table stops being readable
#: and the rules on the swept figures stop being tellable apart.
MAX_WELLS = 4

#: Their names, in order. Fixed rather than free text: a label that can be edited is a
#: label that can be duplicated, and these are used as figure annotations and as the
#: key a case round-trips on.
WELL_LABELS = ("A", "B", "C", "D")

#: Default entry-to-exit spacing, in metres. The same number the sweeps use as their
#: hypothetical gap, so a freshly added well is the same shape as the swept one.
DEFAULT_GAP_M = 50.0


@dataclass(frozen=True)
class WellOption:
    """One candidate location: a reservoir entry and exit, and a name to argue about."""

    label: str
    entry: float
    exit: float

    @property
    def gap(self) -> float:
        return self.exit - self.entry

    @property
    def title(self) -> str:
        return f"Well {self.label}"

    def describe(self) -> str:
        return f"Well {self.label} · {self.entry:,.0f}–{self.exit:,.0f} m"


def _key(label: str, field: str) -> str:
    return f"w_well_{label}_{field}"


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


def read_wells(ts, zmin: float, zmax: float) -> tuple[WellOption, ...]:
    """Every candidate currently defined, seeding Well A on the first run.

    Called at the **top** of ``app.py``, before any tab renders, because the selected
    well's entry and exit drive the grouping, the split and the chance -- all of which
    are computed once and passed down in the ``Ctx``.
    """
    base = default_entry(ts, zmin, zmax)
    st.session_state.setdefault("w_n_wells", 1)
    n = int(np.clip(st.session_state.get("w_n_wells", 1), 1, MAX_WELLS))

    wells: list[WellOption] = []
    for i, label in enumerate(WELL_LABELS[:n]):
        # Each added well opens a little deeper than the last, so a fresh B is a
        # *different* location rather than a duplicate of A that has to be moved
        # before it says anything.
        seed_entry = float(np.clip(base + i * DEFAULT_GAP_M, zmin, zmax))
        entry = float(st.session_state.setdefault(_key(label, "entry"), seed_entry))
        # **Clamped and written back**, not just clamped. A new trial file moves the
        # contact range, and a stored depth outside a slider's bounds is a hard error
        # in Streamlit rather than a silently corrected value -- so the state has to
        # be made valid here, where the range is known, and not in the widget.
        entry = float(np.clip(entry, zmin, zmax))
        st.session_state[_key(label, "entry")] = entry

        exit_ = float(st.session_state.setdefault(
            _key(label, "exit"), min(entry + DEFAULT_GAP_M, max(zmax, entry + 5.0))))
        exit_ = float(np.clip(exit_, entry, max(zmax, entry + 5.0)))
        st.session_state[_key(label, "exit")] = exit_
        wells.append(WellOption(label=label, entry=entry, exit=exit_))
    return tuple(wells)


def read_selected(wells: tuple[WellOption, ...]) -> WellOption:
    """The one candidate tab ④ and tab ⑤ are about.

    Falls back to Well A when the selection names a well that has since been removed,
    rather than raising: removing a well is a normal action and it must not be able to
    break the tabs downstream of it.
    """
    label = st.session_state.get("w_selected_well", wells[0].label)
    for w in wells:
        if w.label == label:
            return w
    st.session_state["w_selected_well"] = wells[0].label
    return wells[0]


def well_editor(wells: tuple[WellOption, ...], zmin: float, zmax: float,
                thickness=None) -> None:
    """The controls, drawn wherever the tab wants them.

    Creates the widgets that own the keys :func:`read_wells` reads. Nothing is
    returned: the next rerun picks the values up at the top of the script, which is
    what keeps a single source of truth for the well geometry.
    """
    n = len(wells)
    cols = st.columns(max(n, 1))
    for col, w in zip(cols, wells):
        with col:
            st.markdown(f"**Well {w.label}**"
                        + ("  ·  *default*" if w.label == "A" else ""))
            # **No `value=` argument.** :func:`read_wells` has already seeded and
            # clamped the key, and passing a default as well is the one thing
            # Streamlit refuses outright: "created with a default value but also had
            # its value set via the Session State API". The key is the single source.
            entry = st.slider(
                f"Reservoir entry (m TVDSS) — {w.label}",
                min_value=zmin, max_value=zmax, step=5.0,
                key=_key(w.label, "entry"),
            )
            # ``max_value`` cannot equal ``min_value``, and an exit at or below the
            # deepest contact is meaningful -- it says the well passes through the
            # whole reservoir -- so the range is widened rather than clamped.
            st.slider(
                f"Reservoir exit (m TVDSS) — {w.label}",
                min_value=entry, max_value=max(zmax, entry + 5.0), step=5.0,
                key=_key(w.label, "exit"),
            )
            # **Is this spacing a vertical well?** The reservoir has a thickness and a
            # vertical well sees exactly that, so the entry-to-exit spacing is not a
            # free choice unless the well is deviated. Said here, where it is chosen.
            if thickness is not None:
                from ..core.dependence import check_deviation

                chk = check_deviation(thickness, w.exit - w.entry)
                icon = {"vertical": "⟂", "deviated": "⟋", "partial": "⊣",
                        "unknown": "?"}[chk.verdict]
                st.caption(f"{icon} **{chk.verdict}** — {w.exit - w.entry:,.0f} m of "
                           f"section, reservoir {chk.thickness_p50:,.0f} m")

    add, remove = st.columns([1, 1])
    with add:
        if n < MAX_WELLS:
            if st.button(f"➕ Add well option {WELL_LABELS[n]}", key="w_add_well",
                         width="stretch"):
                st.session_state["w_n_wells"] = n + 1
                st.rerun()
        else:
            st.button(f"➕ Add well option", key="w_add_well", disabled=True,
                      width="stretch",
                      help=f"{MAX_WELLS} is the limit — beyond that the comparison "
                           f"stops being readable.")
    with remove:
        if n > 1:
            if st.button(f"➖ Remove well {WELL_LABELS[n - 1]}", key="w_del_well",
                         width="stretch"):
                st.session_state["w_n_wells"] = n - 1
                # A selection pointing at the removed well would otherwise survive
                # into the next run and be silently corrected somewhere else.
                if st.session_state.get("w_selected_well") == WELL_LABELS[n - 1]:
                    st.session_state["w_selected_well"] = WELL_LABELS[0]
                st.rerun()
        else:
            st.button("➖ Remove well", key="w_del_well", disabled=True,
                      width="stretch", help="Well A is always defined.")
