"""The Streamlit layer: one module per tab, plus the state they share.

``app.py`` was 1 629 lines of layout, wiring and prose, and a change to tab ②
meant scrolling past tab ①. Split on 2026-08-11 (task 2 of the audit): app.py
keeps the things that are genuinely global -- page config, the sidebar, loading,
and the single resolution of which POS is in force -- and each tab's body moves
to its own module with one ``render`` entry point.

**The split is layout only.** Not one number moved; the guard is
``tests/test_app_shell.py``, which drives the real app through Streamlit's own
``AppTest`` and would notice.

Two ordering facts survive the move and are the reason this is not a free
rearrangement:

* **Tab ① is built in two halves, around the sidebar.** The entry/exit sliders
  take their range from the trial file's contact column, so the file has to be
  chosen first; the QC report is appended to the same tab afterwards. Streamlit
  renders in call order, so ``tab1_data`` has two entry points rather than one,
  and that is a real constraint rather than an accident of history.
* **Tab ② owns the chance widgets but app.py reads them.** ``POS_prospect`` is
  needed before tab ② renders, so app.py reads the widget keys out of
  ``st.session_state`` at the top and tab ② creates the widgets later. See
  ``Ctx`` and the note beside that read in app.py.
"""

from .context import Ctx

__all__ = ["Ctx"]
