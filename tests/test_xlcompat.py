"""Excel function fidelity, including the behaviours that are easy to miss."""

import numpy as np
import pytest

from wellvolpos.core.xlcompat import (
    percentile_exc,
    percentile_inc,
    percentrank_exc,
    truncate_to_significant_digits,
)


def test_truncation_truncates_rather_than_rounds():
    assert truncate_to_significant_digits(0.5424124254241237) == pytest.approx(0.542)
    assert truncate_to_significant_digits(0.5429999) == pytest.approx(0.542)  # not 0.543
    assert truncate_to_significant_digits(0.0) == 0.0


def test_percentrank_default_truncation_is_applied(reduced):
    contact = reduced.col("contact")
    truncated = percentrank_exc(contact, 3500.0)
    raw = percentrank_exc(contact, 3500.0, significance=None)
    assert truncated == pytest.approx(0.542, abs=1e-12)
    assert raw == pytest.approx(0.5424124254241237, abs=1e-12)
    assert truncated != raw


def test_percentile_inc_matches_numpy():
    a = np.arange(1.0, 101.0)
    for p in (0.1, 0.25, 0.5, 0.9):
        assert percentile_inc(a, p) == pytest.approx(np.percentile(a, p * 100))


def test_percentile_exc_is_not_numpy_percentile():
    a = np.arange(1.0, 11.0)
    # position = p*(n+1) = 0.5*11 = 5.5 -> midway between the 5th and 6th values
    assert percentile_exc(a, 0.5) == pytest.approx(5.5)
    # Excel returns #NUM! outside 1/(n+1) .. n/(n+1)
    assert np.isnan(percentile_exc(a, 0.05))
    assert np.isnan(percentile_exc(a, 0.95))


def test_ranges_ignore_text_and_blanks():
    """Excel skips text in a range; so must we, or the count is wrong."""
    a = [1.0, 2.0, np.nan, 3.0, np.inf]
    assert percentile_inc(a, 0.5) == pytest.approx(2.0)
