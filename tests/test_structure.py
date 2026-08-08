"""A(z): the curve that turns a well depth into a position on the structure."""

import numpy as np
import pytest

from wellvolpos.core import AreaDepth


def test_fit_is_essentially_exact(area_depth):
    """GeoX evaluates area as a deterministic function of contact depth."""
    assert area_depth.r2 > 0.999999
    assert area_depth.resid_sd < 1e-4
    assert area_depth.quality()[0] == "pass"


def test_monotone_and_invertible(area_depth):
    a = area_depth.a
    assert np.all(np.diff(a) >= 0)
    for z in (3400.0, 3500.0, 3600.0):
        assert area_depth.depth_at(area_depth.area_at(z)) == pytest.approx(z, abs=1.0)


def test_known_lookups(area_depth):
    assert area_depth.area_at(3500.0) == pytest.approx(2.734, abs=5e-3)


def test_apex_estimate_is_flagged_as_an_estimate(area_depth):
    """Extrapolating the shallow tail is a convenience, not a result."""
    apex = area_depth.apex_estimate()
    assert 3150.0 < apex < 3300.0
    assert apex < area_depth.shallowest


def test_refuses_to_fit_from_too_few_points():
    with pytest.raises(ValueError, match="at least 10"):
        AreaDepth.from_trials([3400.0, 3500.0], [1.0, 2.0])
