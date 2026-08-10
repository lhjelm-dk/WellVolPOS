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
