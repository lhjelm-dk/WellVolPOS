"""Contact-depth bands and the log-probit figure built on them (3.12).

The arithmetic here is easy to get subtly wrong in two ways, and both are pinned:
the exceedance convention (a P90 is a *small* volume) and the proven family's
conditioning (discovery trials only, never mixed with a band's dry ones).
"""

import numpy as np
import pytest

from wellvolpos.core import split_trials, thickness_from_pay
from wellvolpos.core.bands import (
    BAND_PERCENTILES,
    banded_percentiles,
    supported_percentiles,
)
from wellvolpos.viz.figures import fig_b12_banded_percentiles
from wellvolpos.viz.interactive import pfig_b12_banded_percentiles
from wellvolpos.viz.theme import PROBIT_TICKS, probability_axis_range, probit

from .conftest import ENTRY, EXIT


def _members(bp, band, contact, positive):
    """Which trials a band holds -- ``(top, base]``, with the shallowest inclusive.

    Open at the shallow end because that is ``group_trials``' own rule for a
    discovery, ``contact > z_entry``. Duplicated here on purpose: a test that
    asked the module for its own membership could not catch the module getting
    the rule wrong.
    """
    first = band is bp.bands[0]
    lower = contact >= band.z_top if first else contact > band.z_top
    return lower & (contact <= band.z_base) & positive


def _classes(ts, ad, groups):
    th = thickness_from_pay(ts, ad).thickness
    return split_trials(ts, ad, groups, ENTRY, EXIT,
                        thickness=th, apex=ad.apex_estimate())


@pytest.fixture(scope="module")
def vclasses(reduced, area_depth, groups):
    return _classes(reduced, area_depth, groups)


@pytest.fixture(scope="module")
def bp(reduced, groups, vclasses):
    return banded_percentiles(reduced, groups, vclasses, z_entry=ENTRY, z_exit=EXIT)


# ------------------------------------------------------------------ the ladder
def test_a_percentile_needs_trials_beyond_it():
    """P99 takes 200 trials and P50 takes four. Below that it is interpolation."""
    assert supported_percentiles(10_000) == BAND_PERCENTILES
    assert 99 not in supported_percentiles(100)
    assert 1 not in supported_percentiles(100)
    assert 50 in supported_percentiles(4)
    assert supported_percentiles(3) == ()


def test_every_band_reports_the_same_ladder(bp):
    """Bands with different percentiles cannot be compared, which is the point."""
    for band in bp.bands:
        assert band.total.size == len(bp.percentiles)
        if band.proven is not None:
            assert band.proven.size == len(bp.percentiles)


# ------------------------------------------------------ the exceedance convention
def test_percentiles_are_exceedance_so_p90_is_the_small_one(bp):
    """Milkov 2021: a P90 is exceeded 90 % of the time, so it is a small volume."""
    assert bp.percentiles[0] > bp.percentiles[-1]        # P99 comes first
    for band in bp.bands:
        assert np.all(np.diff(band.total) > 0)           # so the volume increases
        if band.proven is not None:
            assert np.all(np.diff(band.proven) > 0)


def test_the_mean_carries_its_own_probability(reduced, bp):
    """A mean is not a percentile, so it is drawn where it actually falls."""
    res = reduced.col("resource")
    contact = reduced.col("contact")
    for band in bp.bands:
        m = _members(bp, band, contact, res > 0)
        assert band.n == int(m.sum())
        assert np.isclose(band.total_mean, res[m].mean(), rtol=1e-9)
        share = np.count_nonzero(res[m] > band.total_mean) / m.sum() * 100.0
        assert np.isclose(band.total_mean_p, share, rtol=1e-9)


# ----------------------------------------------------------- the two families
def test_no_proven_curve_where_nothing_is_proven(bp):
    """A band wholly above the well entry holds no discoveries and so no curve."""
    for band in bp.bands:
        if band.z_base < bp.z_entry:
            assert band.n_discovery == 0
            assert band.proven is None


def test_the_proven_family_never_mixes_in_the_dry_trials(reduced, groups, vclasses, bp):
    """A zero drawn on a log axis becomes the smallest thing on the plot."""
    contact = reduced.col("contact")
    disc = np.asarray(groups.discovery, dtype=bool)
    for band in bp.bands:
        if band.proven is None:
            continue
        assert np.all(band.proven > 0.0)
        m = _members(bp, band, contact, disc)
        assert band.n_discovery == int(m.sum())
        assert np.isclose(band.proven_mean, vclasses.proven[m].mean(), rtol=1e-9)


def test_proven_never_exceeds_total_in_the_same_band(bp):
    """The well cannot prove more than the band holds, at any percentile."""
    for band in bp.bands:
        if band.proven is None:
            continue
        assert np.all(band.proven <= band.total * (1.0 + 1e-9))


def test_going_deeper_buys_volume_but_not_proof(bp):
    """The figure's message, as an inequality on the demo data.

    Down-dip bands hold more resource, and the share of it this well proves
    falls -- which is exactly why the two families diverge down-dip.
    """
    totals = [float(b.total_mean) for b in bp.bands]
    assert totals == sorted(totals)
    withp = [b for b in bp.bands if b.proven is not None]
    i = bp.percentiles.index(50)
    shares = [float(b.proven[i]) / float(b.total[i]) for b in withp]
    assert shares[0] > shares[-1]
    assert shares[-1] < 0.95        # the deepest band is materially short


# --------------------------------------------------------------- the settings
def test_an_unknown_band_mode_raises(reduced, groups, vclasses):
    """A silent fallback would answer a different question under the same label."""
    with pytest.raises(ValueError, match="unknown band mode"):
        banded_percentiles(reduced, groups, vclasses, z_entry=ENTRY, z_exit=EXIT,
                           mode="quantile")


def test_equal_width_bands_are_equally_wide(reduced, groups, vclasses):
    bp = banded_percentiles(reduced, groups, vclasses, z_entry=ENTRY, z_exit=EXIT,
                            mode="equal_width", interval_m=50.0)
    widths = [b.z_base - b.z_top for b in bp.bands]
    assert np.allclose(widths, 50.0)


def test_equal_count_bands_hold_equal_counts_within_each_side(bp):
    """Equal counts is a promise *inside* each side of the entry, not across it.

    The entry boundary is worth more than exact equality -- see
    ``test_the_well_entry_is_always_a_band_boundary`` -- so the quota is shared
    out between the two sides in proportion and cut equally within each.
    """
    for side in (
        [b for b in bp.bands if b.z_base <= bp.z_entry],
        [b for b in bp.bands if b.z_top >= bp.z_entry],
    ):
        assert len(side) >= 1
        counts = np.array([b.n for b in side], dtype=float)
        assert counts.std() / counts.mean() < 0.01


def test_the_well_entry_is_always_a_band_boundary(bp):
    """No band may mix dry trials with discoveries; the fix for a real defect.

    A straddling band took ``total`` over everything in it and ``proven`` over the
    deeper discovery subset only, so the dotted curve came out to the *right* of
    its own solid one -- a well appearing to prove more than the band holds.
    """
    assert any(np.isclose(b.z_top, bp.z_entry) for b in bp.bands)
    for band in bp.bands:
        assert band.n_discovery in (0, band.n)
        assert band.discovery_fraction in (0.0, 1.0)


def test_the_entry_is_a_boundary_in_equal_width_too(reduced, groups, vclasses):
    """Its grid is anchored on the entry, which is how it keeps equal widths."""
    bp = banded_percentiles(reduced, groups, vclasses, z_entry=ENTRY, z_exit=EXIT,
                            mode="equal_width", interval_m=37.0)
    assert any(np.isclose(b.z_top, ENTRY) for b in bp.bands)
    assert np.allclose([b.z_base - b.z_top for b in bp.bands], 37.0)
    for band in bp.bands:
        assert band.n_discovery in (0, band.n)


# ----------------------------------------------------------------- the figures
def test_pb12_is_log_probit_with_a_labelled_scale(bp):
    fig = pfig_b12_banded_percentiles(bp, mefs=15.0)
    assert fig.layout.xaxis.type == "log"
    assert "log" in fig.layout.xaxis.title.text.lower()
    assert list(fig.layout.yaxis.ticktext) == [f"P{t}" for t in PROBIT_TICKS]
    # No dual axis anywhere -- the actual rule, not merely "no yaxis2".
    for name in (k for k in fig.layout if str(k).startswith("yaxis")):
        assert getattr(fig.layout[name], "overlaying", None) is None


def test_pb12_draws_the_proven_family_solid_and_the_total_dotted(bp):
    """**Solid is what the well proves** (Lars, 2026-08-18).

    It was the other way round. The band's whole resource is context and the proven
    part is the subject, so the emphasis was inverted -- and on a figure carrying two
    families per band in two colour ramps, weight is what the eye sorts by first.
    """
    fig = pfig_b12_banded_percentiles(bp, show_mean=False)
    width = len(bp.percentiles)
    solid = [t for t in fig.data
             if t.x is not None and len(t.x) == width
             and t.line.dash in (None, "solid")]
    dotted = [t for t in fig.data
              if t.x is not None and len(t.x) == width and t.line.dash == "dot"]
    # One dotted total per band; one solid proven per band that has one.
    assert len(dotted) == len(bp.bands)
    assert len(solid) == sum(1 for b in bp.bands if b.proven is not None)
    # And the solid family is the heavier of the two, so weight and meaning agree.
    assert min(t.line.width for t in solid) > max(t.line.width for t in dotted)
    assert "solid = the part this well would prove" in fig.layout.title.text


def test_pb12_names_every_band_in_the_legend_exactly_once(bp):
    fig = pfig_b12_banded_percentiles(bp)
    named = [t.name for t in fig.data if t.showlegend is not False and t.name]
    for band in bp.bands:
        assert sum(1 for n in named if n.startswith(band.label)) == 1


def test_fig_b12_twin_agrees_on_the_axes(bp):
    fig, ax = fig_b12_banded_percentiles(bp, mefs=15.0)
    assert ax.get_xscale() == "log"
    assert [t.get_text() for t in ax.get_yticklabels()] == [f"P{t}" for t in PROBIT_TICKS]
    assert "Exceedance" in ax.get_ylabel()


def test_probit_puts_p99_above_p1():
    """Volume rightward with P99 on top is what makes every curve descend."""
    assert probit(99) > probit(50) > probit(1)
    assert np.isclose(probit(50), 0.0)

def test_the_mefs_rule_is_a_trace_so_a_log_axis_cannot_move_it(bp):
    """Measured, not reasoned about: ``add_vline`` on a log axis put this rule at
    2.9 MMboe while its stored coordinate said 103, and passing ``log10(mefs)``
    instead put it against the plot's left edge. Two wrong answers from two
    plausible conventions.

    A trace is in data coordinates by definition -- which is why every *curve* here
    was correct throughout while the reference line was not -- so the rule is drawn
    as a two-point trace with its label attached. This asserts that, in both volume
    scales, because the failure only showed in one of them.
    """
    mefs = 15.0
    lo, hi = bp.volume_range
    for scale in ("log", "linear"):
        fig = pfig_b12_banded_percentiles(bp, mefs=mefs, volume_scale=scale)
        rules = [t for t in fig.data
                 if t.x is not None and len(t.x) == 2
                 and np.allclose(np.asarray(t.x, dtype=float), mefs)]
        assert len(rules) == 1, f"{scale}: expected exactly one MEFS rule trace"
        rule = rules[0]
        assert lo / 2.0 < float(rule.x[0]) < hi * 2.0, scale
        assert any("MEFS" in (s or "") for s in rule.text), \
            f"{scale}: the rule carries no label"
        assert rule.showlegend is False, f"{scale}: the rule should not take a legend row"
        # No shape at all any more, so nothing is left to be reinterpreted.
        assert not [s for s in fig.layout.shapes if s.x0 is not None], scale


def test_the_mefs_rule_spans_the_whole_probability_axis(bp):
    """A rule that stops short reads as a curve rather than as a threshold."""
    for scale in ("probit", "linear"):
        fig = pfig_b12_banded_percentiles(bp, mefs=15.0, probability_scale=scale)
        lo, hi = probability_axis_range(scale)
        rule = next(t for t in fig.data
                    if t.x is not None and len(t.x) == 2
                    and np.allclose(np.asarray(t.x, dtype=float), 15.0))
        assert np.isclose(min(rule.y), lo) and np.isclose(max(rule.y), hi), scale
        assert list(fig.layout.yaxis.range) == [lo, hi], scale


def test_a_band_above_the_well_is_named_rather_than_left_blank(bp):
    """A missing curve and a coinciding curve look identical.

    Three of prospect C's six bands lie entirely up-dip of the default entry, so the
    well never enters them and there is no proven curve to draw. Before 2026-08-18 the
    figure simply showed nothing for them, which reads as a rendering failure rather
    than as the geological statement it is.
    """
    above = [b for b in bp.bands if b.above_the_well]
    if not above:
        pytest.skip("this fixture's bands all hold discoveries")
    for band in above:
        assert band.n_discovery == 0
        # No share to report, and nan rather than 0: nothing is proven there because
        # the well is not in it, not because it is there and proves none.
        assert np.isnan(band.proven_share(50, bp.percentiles))

    fig = pfig_b12_banded_percentiles(bp)
    named = [t.name for t in fig.data if t.name]
    for band in above:
        assert any(band.label in n and "above the well" in n for n in named), band.label
    for band in bp.bands:
        if not band.above_the_well:
            assert any(band.label in n and "above the well" not in n for n in named)


def test_the_proven_share_falls_with_depth_and_is_a_real_ratio(bp):
    """The figure's decision content as a number, not as a gap between two curves.

    On a log axis a gap *is* a ratio and does not look like one, which is exactly why
    the share is computed rather than left to the eye. It must fall down-dip: the total
    keeps growing and the proven part cannot, because the well stays where it is.
    """
    withp = [b for b in bp.bands if not b.above_the_well]
    assert len(withp) >= 2
    shares = [b.proven_share(50, bp.percentiles) for b in withp]
    assert all(0.0 < s <= 1.0 + 1e-9 for s in shares), shares
    assert shares == sorted(shares, reverse=True), shares
    # And it agrees with the curves it summarises.
    i = bp.percentiles.index(50)
    for band, share in zip(withp, shares):
        assert share == pytest.approx(float(band.proven[i]) / float(band.total[i]))
