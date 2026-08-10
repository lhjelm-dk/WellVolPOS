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


def test_contours_sit_on_round_multiples_of_the_interval(area_depth):
    """Not stepped off the apex. 3250/3300/3350 for a 50 m interval, not
    3268.3/3318.3 -- the latter are artefacts of where the apex was guessed and
    are not contours anyone would draw on a depth map."""
    apex = area_depth.apex_estimate()
    for interval in (25.0, 50.0, 100.0):
        c = area_depth.contour_radii(apex, interval=interval)
        rounds = c.depths[~c.at_data_limit]
        assert rounds.size > 0
        assert np.allclose(np.remainder(rounds, interval), 0.0), interval


def test_the_shallowest_contour_is_the_first_multiple_below_the_apex(area_depth):
    apex = area_depth.apex_estimate()          # 3218.3 on the reference file
    assert area_depth.contour_radii(apex, interval=50.0).depths[0] == pytest.approx(3250.0)
    assert area_depth.contour_radii(apex, interval=25.0).depths[0] == pytest.approx(3225.0)
    assert area_depth.contour_radii(apex, interval=100.0).depths[0] == pytest.approx(3300.0)


def test_an_apex_exactly_on_a_multiple_still_starts_one_interval_down(area_depth):
    """A contour at the apex encloses no area and would draw as a point."""
    c = area_depth.contour_radii(3200.0, interval=100.0)
    assert c.depths[0] == pytest.approx(3300.0)


def test_contour_depths_do_not_move_when_the_apex_is_nudged(area_depth):
    """The reason for absolute depths rather than apex-relative ones: the apex is
    usually an estimate, and contours referenced to it would shift under it,
    making two runs of the same prospect incomparable."""
    base = area_depth.contour_radii(area_depth.apex_estimate(), interval=50.0)
    for delta in (-13.0, +7.0, +21.0):
        moved = area_depth.contour_radii(area_depth.apex_estimate() + delta, interval=50.0)
        assert np.array_equal(base.depths, moved.depths)


def test_the_deepest_sampled_contact_is_flagged_not_inferred(area_depth):
    """It is the base of the data, not a round contour, so a figure should be able
    to label it as such rather than guessing from position."""
    c = area_depth.contour_radii(area_depth.apex_estimate(), interval=50.0)
    assert c.at_data_limit.sum() == 1
    assert c.at_data_limit[-1]
    assert c.depths[-1] == pytest.approx(area_depth.deepest)
    assert not np.isclose(c.depths[-1] % 50.0, 0.0)      # genuinely not a round one


def test_an_interval_larger_than_the_closure_still_gives_the_data_limit(area_depth):
    c = area_depth.contour_radii(area_depth.apex_estimate(), interval=5000.0)
    assert c.depths.size == 1
    assert c.at_data_limit[0]
    assert c.depths[0] == pytest.approx(area_depth.deepest)
