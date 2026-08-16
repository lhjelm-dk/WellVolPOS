"""Two criteria the expectation curve cannot express.

Both are standard decision analysis rather than anything this project invented, so the
tests check the *properties that make them what they are* — risk neutrality as a
limiting case, monotonicity of the constraint — rather than reproducing arithmetic that
would just be the implementation written twice.
"""

import numpy as np
import pytest

from wellvolpos.core import (
    AreaDepth,
    ce_curve,
    certainty_equivalent,
    constrained_best,
    run_volume_sweep,
)

POS, MEFS = 0.7605, 14.0


@pytest.fixture(scope="module")
def sweep(reduced):
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    return run_volume_sweep(reduced, ad, POS, z_gap=50.0, mefs=MEFS, n=40)


# ------------------------------------------------------------ certainty equivalent
def test_a_large_risk_tolerance_is_risk_neutrality():
    """``CE -> p x E[V]`` as ``rho -> inf``. If this fails the utility is wrong, and
    every conclusion drawn from it is a different question's answer."""
    v = np.array([10.0, 50.0, 120.0, 300.0])
    p = 0.4
    assert certainty_equivalent(v, p, rho=1e7) == pytest.approx(p * v.mean(), rel=1e-3)


def test_risk_aversion_always_discounts_and_never_flatters():
    """A risk-averse party never values a gamble above its expectation, and values it
    less the more averse they are. Both directions, because a sign slip in the
    exponential would still produce a smooth, plausible-looking curve."""
    v = np.array([5.0, 40.0, 400.0])
    p, expectation = 0.3, 0.3 * np.mean([5.0, 40.0, 400.0])
    prev = expectation
    for rho in (1e6, 500.0, 200.0, 80.0, 30.0):
        ce = certainty_equivalent(v, p, rho=rho)
        assert ce <= expectation + 1e-9
        assert ce <= prev + 1e-9, f"tightening rho to {rho} raised the CE"
        prev = ce
    assert prev < 0.5 * expectation, "a tight tolerance should bite hard"


def test_a_nonpositive_tolerance_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="risk tolerance"):
        certainty_equivalent(np.array([1.0]), 0.5, rho=0.0)


def test_the_ce_optimum_never_sits_deeper_than_the_expectation_peak(reduced, sweep):
    """The whole point: risk aversion penalises the low-chance / high-volume tail, and
    that tail is down-dip. So tightening the tolerance can only move the answer up-dip.
    """
    from wellvolpos.core.stats import thin

    exp = (thin(sweep.p_well, sweep.n_discovery, 30)
           * thin(sweep.discovery_mean, sweep.n_discovery, 30))
    z_exp = float(sweep.z[int(np.nanargmax(exp))])

    mean_succ = float(reduced.col("resource")[reduced.col("resource") > 0].mean())
    last = np.inf
    for frac in (4.0, 1.0, 0.4):
        c = ce_curve(reduced, sweep, rho=mean_succ * frac)
        assert c.best is not None
        assert c.best_depth <= z_exp + 1e-9, (frac, c.best_depth, z_exp)
        assert c.best_depth <= last + 1e-9
        last = c.best_depth


def test_the_ce_discount_widens_down_dip(reduced, sweep):
    """The reading when the peak cannot move: the gap between the expectation and its
    risk-adjusted twin grows with depth, because that is where the tail lives."""
    from wellvolpos.core.stats import thin

    exp = (thin(sweep.p_well, sweep.n_discovery, 30)
           * thin(sweep.discovery_mean, sweep.n_discovery, 30))
    mean_succ = float(reduced.col("resource")[reduced.col("resource") > 0].mean())
    ce = ce_curve(reduced, sweep, rho=mean_succ).ce

    ok = np.isfinite(exp) & np.isfinite(ce) & (exp > 0)
    discount = 1.0 - ce[ok] / exp[ok]
    assert discount[0] > 0, "no discount at all — the utility is not biting"
    assert discount[-1] > discount[0], "the discount must widen down-dip"


# ------------------------------------------------------------ chance-constrained
def test_a_tighter_hurdle_is_never_shallower_and_never_better_odds(sweep):
    """The trade the constraint exists to price: more confidence costs chance."""
    prev_depth, prev_pw = -np.inf, np.inf
    for q in (0.50, 0.70, 0.85, 0.95):
        r = constrained_best(sweep, confidence=q)
        if not r.feasible:
            continue
        assert r.depth >= prev_depth - 1e-9, q
        assert r.p_well_at <= prev_pw + 1e-9, q
        prev_depth, prev_pw = r.depth, r.p_well_at


def test_the_hurdle_is_a_guarantee_not_a_first_crossing(sweep):
    """Same rule B6 uses. The commerciality curve dips wherever the discovery group is
    small, so a first crossing returns depths that deeper locations contradict."""
    from wellvolpos.core.stats import thin

    r = constrained_best(sweep, confidence=0.80)
    assert r.feasible
    pm = thin(sweep.p_discovery_exceeds_mefs, sweep.n_discovery, 30)
    below = np.isfinite(pm) & (sweep.z >= r.depth)
    assert np.all(pm[below] >= 0.80 - 1e-9), "the hurdle is broken deeper down"


def test_an_unreachable_hurdle_says_so_rather_than_returning_the_deepest(sweep):
    r = constrained_best(sweep, confidence=1.0)
    if not r.feasible:
        assert "cannot be met" in r.message()
        assert r.depth is None


def test_the_constraint_needs_a_threshold_and_a_sane_confidence(reduced):
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    no_mefs = run_volume_sweep(reduced, ad, POS, z_gap=50.0, n=15)
    with pytest.raises(ValueError, match="mefs"):
        constrained_best(no_mefs)
    with pytest.raises(ValueError, match="confidence"):
        constrained_best(run_volume_sweep(reduced, ad, POS, z_gap=50.0, mefs=MEFS, n=15),
                         confidence=1.5)


# ---------------------------------------------------------------- the hurdle curve
def test_the_hurdle_curve_is_the_panel_swept(sweep):
    """Every point is one ``constrained_best`` call, so 3.13 and the row above it
    cannot disagree about any single confidence — the row is a point on the curve."""
    from wellvolpos.core import hurdle_curve

    h = hurdle_curve(sweep)
    for q in (0.60, 0.80, 0.90):
        i = int(np.argmin(np.abs(h.confidence - q)))
        if not h.feasible[i]:
            continue
        r = constrained_best(sweep, confidence=float(h.confidence[i]))
        assert r.feasible
        assert h.depth[i] == pytest.approx(r.depth, abs=1e-9)
        assert h.p_well[i] == pytest.approx(r.p_well_at, abs=1e-9)
        assert h.pc[i] == pytest.approx(r.pc_at, abs=1e-9)


def test_a_tighter_hurdle_costs_chance_and_never_buys_it_back(sweep):
    """The shape 3.13 exists to draw: both curves fall to the right.

    ``P_well`` falling is obvious. ``Pc`` falling is not — it is the product of a
    falling chance and a rising conditional, and a reader who expects "more confident"
    to mean "better" has read the constraint as an objective.
    """
    from wellvolpos.core import hurdle_curve

    h = hurdle_curve(sweep)
    ok = h.feasible
    assert ok.sum() > 5, "too few feasible hurdles to say anything"

    pw, pc = h.p_well[ok], h.pc[ok]
    assert np.all(np.diff(pw) <= 1e-9), "P_well must not rise as the hurdle tightens"
    # Pc is allowed to plateau but must end below where it started, and must never
    # exceed P_well anywhere -- it is that times a probability.
    assert pc[-1] < pc[0] - 1e-9, "Pc should fall as the hurdle tightens"
    assert np.all(pc <= pw + 1e-9)


def test_an_impossible_hurdle_leaves_a_gap_rather_than_a_guess(reduced):
    """Where the constraint cannot be met the curve is not drawn, rather than pinned
    to the deepest supported step -- which would read as an achievable answer."""
    from wellvolpos.core import hurdle_curve, run_volume_sweep

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    # A threshold far above anything the prospect holds: nothing can clear it.
    vs = run_volume_sweep(reduced, ad, POS, z_gap=50.0, mefs=1e6, n=20)
    h = hurdle_curve(vs)
    assert not h.feasible.any()
    assert np.all(np.isnan(h.depth))
