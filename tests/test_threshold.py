"""Minimum column height, and how far it really is from a volume percentile."""

import numpy as np
import pytest

from wellvolpos.core.threshold import (
    apply_min_column_height,
    compare_definitions,
    spread_at_fixed_column,
    volume_percentile_threshold,
)


def test_resource_still_spreads_at_a_fixed_column_height(reduced, area_depth):
    """Area is pinned, but pay and yield are not — so the two cuts differ."""
    apex = area_depth.apex_estimate()
    for h in (150.0, 200.0, 250.0):
        s = spread_at_fixed_column(reduced, apex, h)
        if s["n"] >= 5:
            assert s["ratio"] > 2.0


def test_the_two_definitions_mostly_but_not_exactly_agree(reduced, area_depth):
    apex = area_depth.apex_estimate()
    cmp = compare_definitions(reduced, apex, 175.0)
    assert cmp["comparable"]
    assert cmp["disagreement_frac"] < 0.10       # close
    assert cmp["disagreement_frac"] > 0.0        # but not identical


def test_a_loose_minimum_binds_on_nothing_and_says_so(reduced, area_depth):
    """The simulator's own assessment minimum is already stricter."""
    apex = area_depth.apex_estimate()
    m = apply_min_column_height(reduced, area_depth, apex, 30.0)
    assert not m.binds
    assert m.n_excluded == 0
    assert "excludes nothing" in m.message


def test_a_strict_minimum_does_bind(reduced, area_depth):
    apex = area_depth.apex_estimate()
    m = apply_min_column_height(reduced, area_depth, apex, 260.0)
    assert m.binds
    assert m.n_excluded > 0
    assert m.min_area is not None and m.min_area > 0


def test_volume_percentile_alternative_is_available(reduced):
    v = volume_percentile_threshold(reduced, 0.995)
    assert np.isfinite(v) and v > 0
