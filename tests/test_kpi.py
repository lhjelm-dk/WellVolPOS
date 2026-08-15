"""The shared KPI ladder, and the deltas that follow the well.

Both exist so a reader can compare: the prospect row against the well row, and this
well against the one before it. Both are therefore only useful if they are comparing
the same kind of thing, which is what these tests hold.
"""

import pytest
from streamlit.testing.v1 import AppTest

from wellvolpos.core import AreaDepth, class_summary, group_trials, split_trials
from wellvolpos.ui.common import LADDER, LADDER_LABELS

from .conftest import DATA, ENTRY, EXIT

APP = str(DATA.parent / "app.py")
TIMEOUT = 300


def test_the_ladder_has_one_shape_and_the_mean_is_not_a_percentile():
    """Pmean sits between P50 and P10 because the distribution is skewed, not because
    it is a rung — so it is labelled Pmean and never P50-something."""
    assert LADDER == ("p99", "p90", "p50", "mean", "p10", "p1")
    assert LADDER_LABELS["mean"] == "Pmean"
    assert [LADDER_LABELS[k] for k in LADDER] == \
        ["P99", "P90", "P50", "Pmean", "P10", "P1"]


def test_class_summary_supplies_every_rung(reduced):
    """The ladder is rendered straight from ``class_summary``, so a missing rung would
    be a dash on screen rather than an error."""
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    g = group_trials(reduced, ENTRY, EXIT)
    vc = split_trials(reduced, ad, g, ENTRY, EXIT)
    cs = class_summary(vc, g)
    for key, stats in cs.items():
        for rung in LADDER:
            assert rung in stats, (key, rung)
        # Petroleum orientation across all six.
        assert stats["p99"] <= stats["p90"] <= stats["p50"] <= stats["p10"] <= stats["p1"] + 1e-9


def test_the_prospect_row_and_the_well_row_are_the_same_row_twice():
    """The comparison the tool exists to make was previously two layouts two tabs
    apart, so the eye could not carry one across to the other."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    for rung in ("P99", "P90", "P50", "Pmean", "P10", "P1"):
        assert labels.count(rung) >= 2, (rung, labels)
    # And each row leads with its own chance, not with a volume.
    assert "POS prospect" in labels and "P well" in labels


def test_a_delta_is_dropped_when_the_comparison_would_be_meaningless():
    """A difference measured across a different file, chance table or threshold is
    two unrelated numbers subtracted, and it would look exactly like a real one."""
    import streamlit as st
    from wellvolpos.ui.common import track_deltas

    # Streamlit's session state is only available inside a script run, so this
    # exercises the logic through a stand-in mapping of the same shape.
    class _State(dict):
        def pop(self, k, default=None):
            return dict.pop(self, k, default)

    st.session_state = _State()  # type: ignore[assignment]
    try:
        first = track_deltas("t", "fileA", (100.0, 150.0), {"a": 1.0})
        assert first == {}, "no comparison exists on the first run"

        moved = track_deltas("t", "fileA", (120.0, 170.0), {"a": 3.0})
        assert moved["a"] == "+2.00"

        # Still, with the well unchanged: the last move's answer persists rather than
        # collapsing to zero the moment any other widget is touched.
        assert track_deltas("t", "fileA", (120.0, 170.0), {"a": 3.0})["a"] == "+2.00"

        # A different file clears it outright.
        assert track_deltas("t", "fileB", (120.0, 170.0), {"a": 9.0}) == {}
    finally:
        del st.session_state
