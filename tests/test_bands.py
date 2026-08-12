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
from wellvolpos.viz.theme import PROBIT_TICKS, probit

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


def test_pb12_draws_one_solid_and_one_dotted_family_per_band(bp):
    fig = pfig_b12_banded_percentiles(bp, show_mean=False)
    width = len(bp.percentiles)
    solid = [t for t in fig.data
             if t.x is not None and len(t.x) == width
             and t.line.dash in (None, "solid")]
    dotted = [t for t in fig.data
              if t.x is not None and len(t.x) == width and t.line.dash == "dot"]
    assert len(solid) == len(bp.bands)
    assert len(dotted) == sum(1 for b in bp.bands if b.proven is not None)


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

def test_a_reference_line_on_a_log_axis_is_given_as_log10(bp):
    """Plotly shape coordinates are *axis* coordinates, and on a log axis that is
    the exponent -- so passing the volume itself put the MEFS rule at 10^103 and
    stretched the axis to 10^96, with every curve crushed into the left edge.

    Traces are unaffected, which is exactly why it read as a data problem rather
    than an axis one, and the matplotlib twin was right all along because
    ``axvline`` takes the value. Any future rule on a log axis has the same trap.
    """
    mefs = 15.0
    fig = pfig_b12_banded_percentiles(bp, mefs=mefs)
    shapes = [sh for sh in fig.layout.shapes if sh.x0 is not None]
    assert shapes, "expected the MEFS rule as a shape"
    for sh in shapes:
        assert np.isclose(10.0 ** float(sh.x0), mefs, rtol=1e-9)
    lo, hi = bp.volume_range
    for sh in shapes:
        assert np.log10(lo) - 1 < float(sh.x0) < np.log10(hi) + 1
