"""Sampling error: bootstrap intervals and the support behind a swept curve."""

import numpy as np
import pytest

from wellvolpos.core.stats import (
    MIN_SUPPORT,
    bootstrap_mean_ci,
    bootstrap_proportion_ci,
    describe_support,
    support_mask,
    thin,
)


def test_bootstrap_interval_brackets_the_sample_mean():
    rng = np.random.default_rng(1)
    v = rng.normal(15.0, 4.0, 2000)
    lo, hi = bootstrap_mean_ci(v, n_boot=400)
    assert lo < v.mean() < hi


def test_bootstrap_interval_narrows_as_n_grows():
    """The property the band exists to show: more trials, tighter statement."""
    rng = np.random.default_rng(2)
    wide = bootstrap_mean_ci(rng.normal(15.0, 4.0, 30), n_boot=400)
    narrow = bootstrap_mean_ci(rng.normal(15.0, 4.0, 3000), n_boot=400)
    assert (wide[1] - wide[0]) > 3.0 * (narrow[1] - narrow[0])


def test_bootstrap_is_reproducible_by_default():
    """A band that moved every time a slider was nudged would read as new
    information rather than as noise in the band itself."""
    v = np.linspace(1.0, 20.0, 500)
    assert bootstrap_mean_ci(v, n_boot=200) == bootstrap_mean_ci(v, n_boot=200)
    assert bootstrap_mean_ci(v, n_boot=200, seed=1) != bootstrap_mean_ci(v, n_boot=200, seed=2)


def test_bootstrap_declines_rather_than_inventing_an_interval():
    lo, hi = bootstrap_mean_ci(np.array([4.0]))
    assert np.isnan(lo) and np.isnan(hi)
    lo, hi = bootstrap_mean_ci(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)


def test_a_wider_interval_is_requested_by_a_smaller_alpha():
    rng = np.random.default_rng(3)
    v = rng.normal(10.0, 2.0, 800)
    lo90, hi90 = bootstrap_mean_ci(v, n_boot=400, alpha=0.10)
    lo50, hi50 = bootstrap_mean_ci(v, n_boot=400, alpha=0.50)
    assert (hi90 - lo90) > (hi50 - lo50)


def test_proportion_interval_stays_inside_zero_to_one():
    flags = np.zeros(500, dtype=bool)
    flags[:37] = True
    lo, hi = bootstrap_proportion_ci(flags, n_boot=400)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < flags.mean() < hi


# --------------------------------------------------------------- support
def test_support_mask_and_thin_blank_but_do_not_shorten():
    counts = np.array([500, 100, 29, 3, 0])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert support_mask(counts, 30).tolist() == [True, True, False, False, False]
    out = thin(values, counts, 30)
    assert out.size == values.size            # stays aligned with its depth grid
    assert out[:2].tolist() == [1.0, 2.0]
    assert np.isnan(out[2:]).all()


def test_thin_leaves_a_fully_supported_curve_untouched():
    values = np.array([1.0, 2.0, 3.0])
    out = thin(values, np.array([100, 200, 300]), 30)
    assert np.array_equal(out, values)


def test_describe_support_names_where_support_first_fails():
    counts = np.array([500, 400, 20, 5])
    z = np.array([3400.0, 3500.0, 3600.0, 3700.0])
    s = describe_support(counts, z, min_support=30, name="discovery")
    assert not s.all_supported
    assert s.n_below == 2
    assert s.n_min == 5
    assert s.shallowest_thin == pytest.approx(3600.0)
    assert s.deepest_thin == pytest.approx(3700.0)
    assert "3600" in s.message()
    assert "discovery" in s.message()


def test_describe_support_does_not_assume_thinning_happens_with_depth():
    """The dry-with-attic group thins at the *shallow* end, because near the
    crest almost nothing is dry. A hardcoded "downward" was wrong for it."""
    counts = np.array([0, 7, 400, 500])
    z = np.array([3350.0, 3360.0, 3500.0, 3600.0])
    s = describe_support(counts, z, min_support=30, name="dry-with-attic")
    assert s.shallowest_thin == pytest.approx(3350.0)
    assert s.deepest_thin == pytest.approx(3360.0)
    assert "downward" not in s.message()
    assert "3350" in s.message() and "3360" in s.message()


def test_describe_support_says_so_when_everything_is_supported():
    s = describe_support(np.array([500, 400, 300]), np.array([3400.0, 3500.0, 3600.0]))
    assert s.all_supported
    assert s.shallowest_thin is None and s.deepest_thin is None
    assert "at least" in s.message()


def test_the_support_floor_is_a_default_not_a_buried_constant():
    counts = np.array([25])
    assert support_mask(counts, MIN_SUPPORT).tolist() == [False]
    assert support_mask(counts, 10).tolist() == [True]
