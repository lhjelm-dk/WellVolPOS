"""The app shell, driven through Streamlit's own test harness.

Everything else in this suite tests the library. These four test ``app.py``
itself, because the defects they cover were only reachable through the UI and were
found by using it rather than by reading it:

* Choosing *Upload your own…* before picking a file **made the whole sidebar
  disappear**, with no way back that looked like a way back. The cause was
  ``st.stop()`` running before the sidebar was ever created — so the fix is not
  "put the sidebar back", it is "say why the controls are not there yet".
* Case load and trial upload were two file uploaders in two different places, which
  invited the reasonable question of whether they did the same thing.

``AppTest`` runs the real script with no browser, which is the only way to assert
on these without clicking. It cannot set a ``file_uploader``'s value, so the
in-memory upload path is covered in ``test_io_layer.py`` instead — at the level
where it actually lives.
"""

import pytest

pytest.importorskip("streamlit.testing.v1")

from pathlib import Path

from streamlit.testing.v1 import AppTest

# Absolute, because AppTest resolves a relative path against the *calling file* --
# so "app.py" here means tests/app.py.
APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 240


@pytest.fixture(scope="module")
def fresh():
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def test_the_app_runs_on_the_demo_data_with_no_exception(fresh):
    assert not fresh.exception
    # The sidebar carries the threshold and the conventions. The well geometry moved
    # to tab (3) on 2026-08-13 -- see the test below -- because a location decision
    # compares candidates, and four pairs of sliders do not belong in a sidebar
    # beside four global conventions.
    assert fresh.sidebar.slider.len == 0


def test_choosing_upload_says_why_the_controls_are_not_there_yet():
    """Descendant of the original sidebar bug, which was that the controls vanished
    with no way back that looked like a way back.

    There is **no sidebar at all** since 2026-08-14 -- everything moved into tab ①,
    now "Setup and input" -- so the same failure mode would be a tab that goes
    silent. It says why instead, in the tab where the file is chosen.
    """
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.selectbox[0].select("Upload your own…").run()

    assert not at.exception
    assert at.sidebar.slider.len == 0
    assert at.sidebar.number_input.len == 0, "the sidebar is gone; nothing may live there"
    said = " ".join(e.value for e in at.info)
    assert "Waiting for trial data" in said
    # And it says where the well geometry is, rather than leaving a reader hunting.
    assert "tab ③" in said


def test_going_back_to_a_demo_restores_the_controls():
    """The other half of the same bug: it has to be recoverable."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.selectbox[0].select("Upload your own…").run()
    assert not any("MEFS" in (w.label or "") for w in at.number_input)
    # Whichever demo is present, not a hardcoded label: which files ship is a
    # publication decision (prospect B is gitignored) and this test is about
    # recovering from "Upload your own…", not about the dataset.
    _demo = next(o for o in at.selectbox[0].options if not o.startswith("Upload"))
    at.selectbox[0].select(_demo).run()
    assert not at.exception
    # The threshold and the four conventions are in tab ①'s body now, not a sidebar.
    assert any("MEFS" in (w.label or "") for w in at.number_input)
    assert any("Reference contour" in (w.label or "") for w in at.radio)
    assert any("allocation" in (w.label or "") for w in at.selectbox)


def test_nothing_lives_in_a_sidebar(fresh):
    """The sidebar was removed on 2026-08-14: one place to set a session up, and six
    tabs that are then all output. A control that reappears there is a control a
    reader has to find twice."""
    sb = fresh.sidebar
    assert (sb.slider.len + sb.number_input.len + sb.selectbox.len
            + sb.radio.len + sb.checkbox.len) == 0


def test_the_well_geometry_lives_in_a_tab_and_not_in_a_sidebar(fresh):
    """One well, its two sliders in tab ①, read at the top of ``app.py``.

    The same arrangement the chance table uses, and for the same reason: the entry
    and exit are needed before any tab renders, so a widget owns the key and the read
    happens at the top of the script.
    """
    assert not fresh.exception
    labels = [s.label for s in fresh.slider]
    assert any(l.startswith("Reservoir entry") for l in labels), labels
    assert any(l.startswith("Reservoir exit") for l in labels), labels
    # Exactly one of each -- the four-candidate model was removed on 2026-08-14 and a
    # second pair of these sliders means it has come back by accident.
    assert sum(l.startswith("Reservoir entry") for l in labels) == 1, labels
    assert sum(l.startswith("Reservoir exit") for l in labels) == 1, labels


def test_case_load_and_save_sit_together_in_tab_one_not_the_sidebar(fresh):
    """Both are session *inputs*, and having them in two places made it look as
    though the case uploader was a second way to load trials. It is not: a case
    carries settings, a trial file carries data."""
    assert not fresh.exception
    sidebar_text = " ".join(e.value for e in fresh.sidebar.markdown) + " ".join(
        e.value for e in fresh.sidebar.caption
    )
    assert "Load a case" not in sidebar_text
    body = " ".join(e.value for e in fresh.subheader)
    assert "Case — the settings, not the data" in body
    labels = [b.label for b in fresh.download_button]
    assert "⬇ Save this case (.json)" in labels


def test_the_reader_and_its_confidence_are_stated(fresh):
    """Which adapter claimed the file, and by how much, is provenance a reader
    should be able to see rather than infer from the numbers looking plausible."""
    captions = " ".join(e.value for e in fresh.caption)
    assert "Reader:" in captions and "GeoX" in captions


def test_the_units_verdict_appears_in_the_qc_report(fresh):
    """The check exists to be read. A file in feet is refused, and a file that
    declares nothing says what was assumed instead."""
    body = " ".join(e.value for e in fresh.markdown)
    assert "declares no units" in body or "Units confirmed" in body


def _all_text(at) -> str:
    """Every string the app put on screen, in one blob -- widget labels included.

    The labels matter here: the risk elements are named *only* in a
    ``number_input`` label, so a check that reads markdown alone cannot see them.
    """
    parts = []
    for group in (at.markdown, at.caption, at.subheader, at.header, at.warning,
                  at.info, at.success, at.error):
        parts += [e.value for e in group if isinstance(e.value, str)]
    for name in ("number_input", "slider", "radio", "checkbox", "selectbox",
                 "text_input", "multiselect"):
        for e in getattr(at, name, []):
            label = getattr(e, "label", None)
            if isinstance(label, str):
                parts.append(label)
    return "\n".join(parts)


def test_no_unresolved_figure_placeholder_reaches_the_reader(fresh):
    """``ref()`` resolves ``{b2}`` to a number -- unless the string is never passed
    through it, and then the reader is shown the brace form.

    That happened: the guide's references block ended with *"{b2}'s regret curve is
    the place they would land"*, because it is a plain triple-quoted string and the
    wrap around it had been left off. A placeholder is silent when it works and
    obvious when it does not, which is exactly the kind of thing worth a test.
    """
    import re

    from wellvolpos.ui.numbering import FIGURE_NUMBERS

    text = _all_text(fresh)
    stray = {m.group(1) for m in re.finditer(r"\{([a-z][a-z0-9_]*)\}", text)
             if m.group(1) in FIGURE_NUMBERS}
    assert not stray, f"unresolved figure placeholders on screen: {sorted(stray)}"


def test_no_legacy_letter_code_is_shown_to_the_reader(fresh):
    """Figures are numbered by tab now. ``B6`` told a reader nothing about where it
    lived, and the letters survive only in the code, the export keys and this
    project's own notes -- never on screen."""
    import re

    # Two codes on screen are *not* figure references and must survive:
    #
    #   * Rose's prospect **segment A2**, in the guide's discussion of his poster --
    #     his label for one of his segments, nothing to do with our outcome tree.
    #   * the guide's paragraph explaining that the exported figure files and this
    #     project's own notes still carry the letters. A reader who unzips the
    #     figures and finds B6_inverse.svg is better off having been told.
    #
    # Listed rather than pattern-matched, so a *new* stray code fails instead of
    # being swallowed by a loose exception.
    allowed = ("segment A2", 'about "B6" by letter')
    text = _all_text(fresh)
    for phrase in allowed:
        text = text.replace(phrase, "")
    # Word-boundary matched, so "A1" is caught but "MMboe" and "P90" are not.
    codes = sorted({m.group(0)
                    for m in re.finditer(r"\b(?:A[0-9]|B(?:1[0-2]|[0-9])|C[12])\b", text)})
    assert not codes, f"legacy figure codes on screen: {codes}"


def test_the_risk_elements_are_named_closure_not_trap(fresh):
    """Lars, 2026-08-12: *Closure* is the better term for what the element assesses.

    The dict keys stay ``trap`` -- behaviour must never depend on label text -- so
    this checks the half that is visible.
    """
    import re

    text = _all_text(fresh)
    assert "Closure" in text, "the risk element is not labelled Closure anywhere"
    # "trapezoid" is an integration rule and stays; the geological sense must be gone.
    stray = re.findall(r"\btrap(?!ezoid)\w*", text, re.I)
    assert not stray, f"'trap' still shown to the reader: {sorted(set(stray))}"




def test_tab_three_is_grouped_and_leads_with_its_two_inputs(fresh):
    """The tab's figures under three questions, and the two inputs they are swept
    against at the top of it.

    The well sliders were on tab ① with the conventions until 2026-08-18. That is
    defensible for a setting and wrong for these: this is the tab that sweeps every
    depth and prices every threshold, so the pair being swept against belongs where
    the sweeping is.
    """
    assert not fresh.exception
    subs = [s.value for s in fresh.subheader]

    # Three headings, in the order a reader meets the question.
    for a, b in (("What changes as the well moves", "What the well would prove"),
                 ("What the well would prove", "Where the optimum sits")):
        assert a in subs and b in subs, (a, b)
        assert subs.index(a) < subs.index(b), subs

    # The well leads tab ③ — before every heading on it.
    assert subs.index("The well being tested") < subs.index("What changes as the well moves")
    # And it is no longer on tab ①, which now carries only setup.
    assert subs.index("The well being tested") > subs.index("Quality control"), subs

    # 3.12 is behind an expander rather than in the scroll.
    labels = " | ".join(e.label for e in fresh.expander)
    assert "Resource by contact-depth band" in labels, labels

    # And the current well is readable from the tabs that are about it, without
    # being a second widget that could disagree with tab ①'s.
    # Keyed on the penetrated thickness, not on the "**Well:**" prefix: tab ④ drops
    # the depths from its readout because the headline sentence above it already
    # gives them, and that sentence is the one written to be quoted (2026-08-18).
    readouts = [c.value for c in fresh.caption if "of reservoir penetrated" in c.value]
    assert len(readouts) >= 2, readouts
    assert all("top of tab ③" in r for r in readouts)
    # Exactly one of them repeats the depths -- tab ③'s, where nothing above it has.
    assert sum(1 for r in readouts if r.startswith("**Well:**")) == 1, readouts


def test_the_guide_opens_with_what_this_tool_assumes(fresh):
    """Phase 6: assume competence, explain only what is local.

    The tab opened by defining percentiles and exceedance to a reader who has known
    both since university, while the assumptions that would change how they read a
    number were spread through long passages further down.
    """
    assert not fresh.exception
    body = _all_text(fresh)
    i = body.find("What this tool assumes")
    assert i >= 0, "the assumptions section is missing"

    # It leads: nothing else in tab ⑥ comes before it.
    for later in ("Guidelines — six things to get right",
                  "The one idea everything rests on",
                  "Colour associations"):
        j = body.find(later)
        assert j > i, f"{later} appears before the assumptions"

    # Each of the six is present by its distinguishing phrase, so a rewrite that drops
    # one fails rather than quietly shortening the list.
    for phrase in ("wedge, not by map area", "extrapolated, never mapped",
                   "modelled, not logged", "exactly two volume classes",
                   "never applied to the distributions", "uniform inside the charged"):
        assert phrase in body, phrase
