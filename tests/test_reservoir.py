"""Reservoir thickness recovered from pay, by inverting the wedge.

The identity under test: with a flat contact and a dipping layer, the
hydrocarbon-bearing gross rock volume is the integral of A(z) from (contact -
thickness) to the contact. That makes thickness recoverable per trial, and --
where the export carries its own thickness column -- independently checkable.
"""

import numpy as np
import pytest

from wellvolpos.core import AreaDepth
from wellvolpos.core.reservoir import thickness_from_pay


def test_the_inversion_reproduces_geox_own_thickness_column(full, area_depth):
    """The check that validates the whole wedge identity.

    Two independent routes to one number: GeoX sampled a reservoir thickness and
    used it to build the pay; we invert the pay to get the thickness back. They
    agree to a mean difference of 0.01 m at r = 0.9998 on the reference file. If
    this test fails, the wedge geometry assumed in core.reservoir is not the one
    the simulator used.
    """
    ad = AreaDepth.from_trials(full.col("contact"), full.col("area"))
    r = thickness_from_pay(full, ad)
    m = r.resolved
    assert m.sum() > 7_000
    recovered, column = r.thickness[m], full.col("thickness")[m]
    assert float(np.mean(recovered - column)) == pytest.approx(0.0, abs=0.1)
    assert float(np.median(recovered - column)) == pytest.approx(0.0, abs=0.1)
    assert float(np.corrcoef(recovered, column)[0, 1]) > 0.999


def test_it_works_on_the_seven_column_paste_which_has_no_thickness(reduced, full):
    """The practical payoff. The everyday export carries area, pay and contact
    but no reservoir thickness — and those three are enough."""
    assert not reduced.has("thickness")
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    r = thickness_from_pay(reduced, ad)
    assert r.n_resolved > 7_000
    # Same run exported twice, so it must recover the full file's own column.
    m = r.resolved
    assert float(np.mean(r.thickness[m] - full.col("thickness")[m])) == pytest.approx(0.0, abs=0.1)


def test_recovered_thickness_is_never_less_than_the_pay_it_came_from(reduced):
    """Pay is an average over the wedge, so it is bounded above by the reservoir
    thickness — the geometric fact the whole distinction rests on."""
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    r = thickness_from_pay(reduced, ad)
    m = r.resolved
    assert np.all(r.thickness[m] >= reduced.col("gross_pay")[m] - 1e-6)


def test_pay_over_thickness_ratio_is_the_area_growth_it_should_be(reduced):
    """The corollary of the identity: mean gross pay / thickness equals the mean
    of A over the top T metres divided by A at the contact. On this file that
    lands at about 0.83, which is a statement about how fast area grows in the
    last ~45 m rather than a free parameter."""
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    r = thickness_from_pay(reduced, ad)
    m = r.resolved
    ratio = reduced.col("gross_pay")[m] / r.thickness[m]
    assert np.all(ratio <= 1.0 + 1e-9)
    assert float(np.median(ratio)) == pytest.approx(0.83, abs=0.03)


def test_the_volume_identity_round_trips(reduced):
    """Rebuild the GRV from the recovered thickness and get back what went in."""
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    r = thickness_from_pay(reduced, ad)
    m = r.resolved
    contact = reduced.col("contact")[m]
    rebuilt = ad.volume_above(contact, r.apex) - ad.volume_above(contact - r.thickness[m], r.apex)
    original = reduced.col("hc_grv")[m]
    assert np.allclose(rebuilt, original, rtol=1e-3, atol=1e-3)


def test_full_to_base_trials_are_excluded_not_guessed(reduced):
    """Where the closure is charged top-to-base the thickness is only bounded
    below, so the trial cannot pin it down and must not pretend to."""
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    r = thickness_from_pay(reduced, ad)
    assert np.all(np.isnan(r.thickness[~r.resolved]))
    assert r.n_full_to_base + r.n_inconsistent + r.n_resolved <= int((reduced.col("resource") > 0).sum())


def test_an_impossible_pay_is_flagged_rather_than_inverted(reduced):
    """More HC gross rock volume than the closure holds above the contact is a
    QC problem with the export, not a numerical edge case."""
    import copy

    ts = copy.deepcopy(reduced)
    ad = AreaDepth.from_trials(ts.col("contact"), ts.col("area"))
    ts.frame["hc_grv"] = ts.frame["hc_grv"] * 1000.0        # physically impossible
    r = thickness_from_pay(ts, ad)
    assert r.n_inconsistent > 7_000
    assert r.n_resolved == 0
    assert "more HC gross rock volume than the closure holds" in r.message()


def test_it_says_so_when_it_has_nothing_to_invert(reduced):
    import copy

    ts = copy.deepcopy(reduced)
    ad = AreaDepth.from_trials(ts.col("contact"), ts.col("area"))
    ts.frame = ts.frame.drop(columns=["hc_grv", "gross_pay"])
    with pytest.raises(ValueError, match="invert the wedge"):
        thickness_from_pay(ts, ad)


def test_volume_above_and_depth_for_volume_are_inverses(area_depth):
    apex = area_depth.apex_estimate()
    for z in (3400.0, 3500.0, 3600.0, area_depth.deepest):
        v = area_depth.volume_above(z, apex)
        assert float(area_depth.depth_for_volume(v, apex)) == pytest.approx(z, abs=0.2)


def test_volume_above_is_zero_at_the_apex_and_grows_with_depth(area_depth):
    apex = area_depth.apex_estimate()
    assert float(area_depth.volume_above(apex, apex)) == pytest.approx(0.0, abs=1e-9)
    depths = np.array([3300.0, 3400.0, 3500.0, 3600.0, area_depth.deepest])
    v = area_depth.volume_above(depths, apex)
    assert np.all(np.diff(v) > 0)
