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
    assert [s.label for s in fresh.sidebar.slider] == [
        "Reservoir entry depth (m TVDSS)", "Reservoir exit depth (m TVDSS)",
    ]


def test_choosing_upload_does_not_take_the_sidebar_away():
    """The reported bug. Before the fix the sidebar was empty — which reads as a
    crash — and the controls could not come back without knowing to change the
    selector in tab ①."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.selectbox[0].select("Upload your own…").run()

    assert not at.exception
    # The sliders genuinely cannot exist: their range comes from the contact
    # column. So the sidebar explains itself rather than vanishing.
    assert at.sidebar.slider.len == 0
    assert [h.value for h in at.sidebar.subheader] == ["Well"]
    said = " ".join(e.value for e in at.sidebar.info)
    assert "Waiting for trial data" in said
    assert "tab ①" in said


def test_going_back_to_a_demo_restores_the_controls():
    """The other half of the same bug: it has to be recoverable."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.selectbox[0].select("Upload your own…").run()
    assert at.sidebar.slider.len == 0
    at.selectbox[0].select("Prospect A — reduced (7 columns)").run()
    assert not at.exception
    assert at.sidebar.slider.len == 2


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
