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


# ----------------------------------------------------- map-view geometry
def test_area_at_tapered_goes_to_zero_at_the_apex_instead_of_clipping(area_depth):
    """area_at is np.interp, which clips: above the shallowest sampled contact it
    keeps returning that contact's area. For a map view that drew a closure with
    a wide flat top and two identical contours where the trials simply never
    reached the crest."""
    apex = area_depth.apex_estimate()
    assert area_depth.area_at(apex) == pytest.approx(area_depth.a[0])        # the clip
    assert area_depth.area_at_tapered(apex, apex) == pytest.approx(0.0)     # the fix
    mid = 0.5 * (apex + area_depth.shallowest)
    tapered = float(area_depth.area_at_tapered(mid, apex))
    assert 0.0 < tapered < float(area_depth.a[0])


def test_area_at_tapered_leaves_the_sampled_range_untouched(area_depth):
    apex = area_depth.apex_estimate()
    for z in (area_depth.shallowest, 3500.0, area_depth.deepest):
        assert area_depth.area_at_tapered(z, apex) == pytest.approx(area_depth.area_at(z))


def test_contour_radii_are_monotone_and_flag_the_extrapolated_ones(area_depth):
    apex = area_depth.apex_estimate()
    depths, radii, extrap = area_depth.contour_radii(apex, interval=50.0)
    assert depths.size == radii.size == extrap.size
    assert np.all(np.diff(depths) > 0)
    # Deeper contact encloses more area, so radius can only grow.
    assert np.all(np.diff(radii) >= -1e-12)
    # Contours above the shallowest sampled contact are the extrapolated ones.
    assert np.array_equal(extrap, depths < area_depth.shallowest)
    assert extrap.any() and not extrap.all()


def test_contour_radii_reach_the_deepest_sampled_contact(area_depth):
    """Lars asked for the deepest depth in the data set to be shown, so the
    outer ring must land on it even when the interval does not divide evenly."""
    apex = area_depth.apex_estimate()
    depths, _, _ = area_depth.contour_radii(apex, interval=37.0)
    assert depths[-1] == pytest.approx(area_depth.deepest)


def test_contour_radius_matches_the_enclosed_area(area_depth):
    apex = area_depth.apex_estimate()
    depths, radii, _ = area_depth.contour_radii(apex, interval=50.0)
    for z, r in zip(depths, radii):
        assert np.pi * r * r == pytest.approx(float(area_depth.area_at_tapered(z, apex)), rel=1e-9)


def test_contour_interval_must_be_positive(area_depth):
    with pytest.raises(ValueError):
        area_depth.contour_radii(area_depth.apex_estimate(), interval=0.0)
