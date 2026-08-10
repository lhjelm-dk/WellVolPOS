"""The location sweep: both engines evaluated across a depth grid."""

import numpy as np
import pytest

from wellvolpos.core.chance import ReferenceContour, p_well
from wellvolpos.core.classes import class_summary, split_trials
from wellvolpos.core.groups import group_trials
from wellvolpos.core.stats import support_mask
from wellvolpos.core.sweep import (
    find_crossing,
    invert_volume_target,
    run_sweep,
    run_volume_sweep,
    volume_target_curve,
)

from .conftest import ENTRY, EXIT

POS = 0.7605


def test_sweep_at_a_single_point_matches_direct_evaluation(reduced):
    sweep = run_sweep(reduced, POS, z_min=ENTRY, z_max=ENTRY, n=1)
    direct = p_well(reduced, ENTRY, POS)
    assert sweep.z[0] == pytest.approx(ENTRY)
    assert sweep.p_well[0] == pytest.approx(direct.p_well, abs=1e-9)
    assert sweep.r_location[0] == pytest.approx(direct.r_location, abs=1e-9)


def test_r_location_is_non_increasing_with_depth(reduced):
    # Deeper entry can only exclude contacts, never add one back.
    sweep = run_sweep(reduced, POS, n=60)
    assert np.all(np.diff(sweep.r_location) <= 1e-9)


def test_p_well_is_pos_times_r_location_at_every_step(reduced):
    sweep = run_sweep(reduced, POS, n=60)
    assert np.allclose(sweep.p_well, POS * sweep.r_location)


def test_uncertainty_reduction_is_finite_and_capped_at_100(reduced):
    sweep = run_sweep(reduced, POS, n=60)
    assert np.all(np.isfinite(sweep.uncertainty_reduction))
    assert sweep.uncertainty_reduction.max() <= 100.0 + 1e-6


def test_optimum_is_the_argmax_within_the_swept_range(reduced):
    sweep = run_sweep(reduced, POS, n=60)
    assert sweep.z.min() <= sweep.z_optimum <= sweep.z.max()
    assert sweep.reduction_optimum == pytest.approx(sweep.uncertainty_reduction.max())
    assert sweep.reduction_optimum == pytest.approx(
        sweep.uncertainty_reduction[np.argmin(np.abs(sweep.z - sweep.z_optimum))]
    )


def test_reference_contour_changes_r_location_but_not_its_shape_family(reduced):
    crest = run_sweep(reduced, POS, n=40, reference=ReferenceContour.CREST)
    rose = run_sweep(reduced, POS, n=40, reference=ReferenceContour.P90_AREA)
    assert not np.allclose(crest.r_location, rose.r_location)
    # Rose is a flat 1.11x uplift up to its cap -- never below the crest curve.
    assert np.all(rose.r_location >= crest.r_location - 1e-9)


def test_explicit_bounds_are_not_padded(reduced):
    sweep = run_sweep(reduced, POS, z_min=3400.0, z_max=3600.0, n=21)
    assert sweep.z[0] == pytest.approx(3400.0)
    assert sweep.z[-1] == pytest.approx(3600.0)


def test_default_range_pads_the_shallow_end_only(reduced):
    res, contact = reduced.col("resource"), reduced.col("contact")
    succ_contact = contact[res > 0.0]
    sweep = run_sweep(reduced, POS, n=50)
    assert sweep.z[0] < succ_contact.min()
    assert sweep.z[-1] == pytest.approx(succ_contact.max())


# ------------------------------------------------------- A2's outcome tree
def test_outcome_shares_sum_to_one_at_every_step(reduced):
    sweep = run_sweep(reduced, POS, n=40)
    total = (
        sweep.share_chance_failure
        + sweep.share_dry_with_attic
        + sweep.share_contact_seen
        + sweep.share_hc_to_exit
    )
    assert np.allclose(total, 1.0)


def test_outcome_shares_do_not_perturb_r_location_or_p_well(reduced):
    # z_gap only changes how discovery is split into seen/past-exit; the
    # discovery mask itself -- and therefore r_location and p_well -- must
    # be identical regardless of what z_gap is asked for.
    a = run_sweep(reduced, POS, n=30, z_gap=20.0)
    b = run_sweep(reduced, POS, n=30, z_gap=200.0)
    assert np.allclose(a.r_location, b.r_location)
    assert np.allclose(a.p_well, b.p_well)


@pytest.mark.parametrize("pos", [POS, 0.60, 1.0])
def test_discovery_mass_of_the_outcome_tree_equals_p_well(reduced, pos):
    """A2 and A3 must be incapable of disagreeing.

    The two figures sit side by side in the same row, so if the outcome tree's
    discovery bands did not sum to exactly the chance curve's ``P_well`` a
    reader would be looking at two different answers to one question. The
    parametrisation is the point: with the tree built from the trial masks
    instead, this passes at POS_trials and fails at every other POS.
    """
    sweep = run_sweep(reduced, pos, n=30)
    assert np.allclose(sweep.share_contact_seen + sweep.share_hc_to_exit, sweep.p_well)
    assert np.isclose(sweep.share_chance_failure, 1.0 - pos)
    assert np.all(sweep.share_dry_with_attic >= -1e-12)


def test_outcome_tree_reproduces_the_locked_group_shares_at_the_reference_well(reduced):
    """Bridge the sweep to the shares test_groups.py already locks."""
    gap = EXIT - ENTRY
    sweep = run_sweep(reduced, POS, z_min=ENTRY, z_max=ENTRY, n=1, z_gap=gap)
    expected = group_trials(reduced, ENTRY, EXIT).shares()
    assert np.isclose(sweep.share_chance_failure, expected["chance_failure"], atol=1e-9)
    assert np.isclose(sweep.share_dry_with_attic[0], expected["dry_with_attic"], atol=1e-9)
    assert np.isclose(sweep.share_contact_seen[0], expected["contact_seen"], atol=1e-9)
    assert np.isclose(sweep.share_hc_to_exit[0], expected["hc_to_exit"], atol=1e-9)


def test_haskett_curve_is_independent_of_the_entered_pos(reduced):
    """The reduction curve is a statement about the trial set's own spread.

    Unlike the outcome tree it is *not* re-risked: how much the spread of
    recoverable resource collapses on each outcome is a property of the
    realisations, which the entered chance table does not change.
    """
    a = run_sweep(reduced, POS, n=30)
    b = run_sweep(reduced, 0.40, n=30)
    assert np.allclose(a.uncertainty_reduction, b.uncertainty_reduction)
    assert a.z_optimum == pytest.approx(b.z_optimum)


# ----------------------------------------------------------- volume sweep
def test_volume_sweep_matches_direct_split_at_the_reference_point(reduced, area_depth):
    vsweep = run_volume_sweep(
        reduced, area_depth, POS, z_gap=EXIT - ENTRY, z_min=ENTRY, z_max=ENTRY, n=1, mefs=14.0,
    )
    groups_ref = group_trials(reduced, ENTRY, EXIT)
    vc_ref = split_trials(reduced, area_depth, groups_ref, ENTRY, EXIT)
    cs = class_summary(vc_ref, groups_ref)

    assert vsweep.z[0] == pytest.approx(ENTRY)
    assert vsweep.z_exit[0] == pytest.approx(EXIT)
    assert vsweep.proven_mean[0] == pytest.approx(cs["proven"]["mean"], abs=1e-9)
    assert vsweep.possible_mean[0] == pytest.approx(cs["possible"]["mean"], abs=1e-9)
    assert vsweep.attic_mean[0] == pytest.approx(cs["attic_dry_hole"]["mean"], abs=1e-9)

    direct = p_well(reduced, ENTRY, POS)
    assert vsweep.p_well[0] == pytest.approx(direct.p_well, abs=1e-9)

    disc_proven = vc_ref.proven[groups_ref.discovery]
    dry_attic = vc_ref.attic[groups_ref.dry_with_attic]
    assert vsweep.p_proven_exceeds_mefs[0] == pytest.approx(float((disc_proven > 14.0).mean()), abs=1e-9)
    assert vsweep.p_attic_exceeds_mefs[0] == pytest.approx(float((dry_attic > 14.0).mean()), abs=1e-9)


def test_volume_sweep_exceedance_is_a_probability(reduced, area_depth):
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=15, mefs=14.0)
    for arr in (vsweep.p_proven_exceeds_mefs, vsweep.p_attic_exceeds_mefs):
        finite = arr[np.isfinite(arr)]
        assert finite.size > 0
        assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_volume_sweep_without_mefs_leaves_exceedance_unset(reduced, area_depth):
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=10)
    assert vsweep.mefs is None
    assert vsweep.p_proven_exceeds_mefs is None
    assert vsweep.p_attic_exceeds_mefs is None


def test_volume_sweep_z_exit_never_precedes_z_entry(reduced, area_depth):
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=40, z_gap=50.0)
    assert np.all(vsweep.z_exit >= vsweep.z - 1e-9)


def test_volume_sweep_holds_the_gap_past_the_deepest_contact(reduced, area_depth):
    """A sweep may legitimately run below the deepest sampled contact.

    Clipping the exit to the data range once made it shallower than its own
    entry down here, which split_trials rightly rejects -- so this asked for
    an impossible well and got an exception rather than a curve.
    """
    vsweep = run_volume_sweep(reduced, area_depth, POS, z_min=3500.0, z_max=3800.0, n=13, z_gap=50.0)
    assert np.allclose(vsweep.z_exit - vsweep.z, 50.0)
    assert vsweep.z[-1] == pytest.approx(3800.0)


def test_volume_sweep_reproduces_the_locked_proven_mean_on_a_real_grid(reduced, area_depth):
    """The interior of the loop, not just a degenerate one-point grid.

    15.76 MMboe at entry 3500 / exit 3550 is the design plan's headline KPI, so
    a grid that straddles the reference well must reproduce it at that depth.
    """
    vsweep = run_volume_sweep(
        reduced, area_depth, POS, z_min=3400.0, z_max=3600.0, n=41, z_gap=EXIT - ENTRY,
    )
    i = int(np.argmin(np.abs(vsweep.z - ENTRY)))
    assert vsweep.z[i] == pytest.approx(ENTRY)
    assert vsweep.proven_mean[i] == pytest.approx(15.756, abs=5e-3)
    assert vsweep.attic_mean[i] == pytest.approx(9.090, abs=5e-3)


# ------------------------------------------------- sample-size diagnostics
def test_volume_sweep_reports_the_sample_size_behind_every_step(reduced, area_depth):
    """The counts are what let a figure decline to draw what it cannot support."""
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=40, z_gap=50.0)
    assert vsweep.n_discovery.shape == vsweep.z.shape
    assert vsweep.n_dry.shape == vsweep.z.shape
    # Deeper entry can only shrink the discovery group.
    assert np.all(np.diff(vsweep.n_discovery) <= 0)
    # On the reference data it collapses from thousands to single figures.
    assert vsweep.n_discovery.max() > 5_000
    assert vsweep.n_discovery.min() < 30


def test_volume_sweep_carries_the_conventions_that_produced_it(reduced, area_depth):
    """A curve without its conventions cannot state which contour it used."""
    vsweep = run_volume_sweep(
        reduced, area_depth, POS, n=10, reference=ReferenceContour.P90_AREA
    )
    assert vsweep.pos_prospect == pytest.approx(POS)
    assert vsweep.reference is ReferenceContour.P90_AREA


def test_bootstrap_band_is_absent_unless_asked_for(reduced, area_depth):
    plain = run_volume_sweep(reduced, area_depth, POS, n=8)
    assert plain.proven_mean_lo is None and plain.alpha is None
    banded = run_volume_sweep(reduced, area_depth, POS, n=8, n_boot=100)
    assert banded.proven_mean_lo is not None and banded.alpha == pytest.approx(0.10)


def test_bootstrap_band_brackets_the_proven_mean_where_supported(reduced, area_depth):
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=20, n_boot=200)
    ok = support_mask(vsweep.n_discovery) & np.isfinite(vsweep.proven_mean_lo)
    assert ok.any()
    assert np.all(vsweep.proven_mean_lo[ok] <= vsweep.proven_mean[ok] + 1e-9)
    assert np.all(vsweep.proven_mean[ok] <= vsweep.proven_mean_hi[ok] + 1e-9)


def test_bootstrap_band_widens_where_the_discovery_group_thins(reduced, area_depth):
    """Down-dip the band should widen by itself -- that is the whole point of
    resampling within each step rather than applying one global width."""
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=30, n_boot=300)
    width = vsweep.proven_mean_hi - vsweep.proven_mean_lo
    ok = np.isfinite(width) & (vsweep.n_discovery > 0)
    n, w = vsweep.n_discovery[ok], width[ok]
    shallow = w[n > 2000].mean()          # thousands of trials
    deep = w[n < 200].mean()              # a few dozen
    assert deep > shallow


def test_proven_mean_rises_with_depth_over_the_well_supported_range(reduced, area_depth):
    """Deeper entry can only mean a bigger discovery, where there is data.

    Restricted to the range where the discovery group is not down to a handful
    of trials -- past that the curve is noise, which is exactly why the deep
    end still needs a visual caveat (recorded in CLAUDE.md).
    """
    vsweep = run_volume_sweep(reduced, area_depth, POS, z_min=3400.0, z_max=3600.0, n=30, z_gap=50.0)
    ok = np.isfinite(vsweep.proven_mean)
    assert np.all(np.diff(vsweep.proven_mean[ok]) > -1e-6)


# ------------------------------------------------------------ B6: the inverse
def test_inverting_the_locked_proven_mean_returns_the_reference_well(reduced, area_depth):
    """The round trip that ties B6 to the rest of the tool.

    The design plan's headline KPI is a proven mean of 15.76 MMboe at entry
    3500 / exit 3550. Asking the inverse for that volume must hand back that
    entry depth, and the P_well there must be the locked 0.4576 -- otherwise
    B6 is answering a different question from tab 4.
    """
    vsweep = run_volume_sweep(
        reduced, area_depth, POS, z_min=3400.0, z_max=3600.0, n=201, z_gap=EXIT - ENTRY,
    )
    res = invert_volume_target(vsweep, 15.756, ts=reduced)
    assert res.achievable
    # Tolerances near the real error (~5e-4 m), not a whole grid step: at
    # abs=1.5 the test still passed with interpolation removed entirely.
    assert res.z_required == pytest.approx(ENTRY, abs=0.05)
    assert res.p_well_at == pytest.approx(p_well(reduced, res.z_required, POS).p_well, abs=1e-12)
    assert res.p_well_at == pytest.approx(0.4576, abs=1e-3)


def test_demanding_more_volume_requires_a_deeper_well_and_costs_chance(reduced, area_depth):
    """The trade B6 exists to show, asserted rather than assumed."""
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=60, z_gap=50.0)
    prev_z, prev_p = -np.inf, np.inf
    for target in (12.0, 16.0, 20.0, 24.0):
        res = invert_volume_target(vsweep, target)
        assert res.achievable, target
        assert res.z_required >= prev_z - 1e-9
        assert res.p_well_at <= prev_p + 1e-9
        prev_z, prev_p = res.z_required, res.p_well_at


def test_an_unreachable_target_says_nowhere_rather_than_the_deepest_step(reduced, area_depth):
    """Asking for more than the closure holds is a fair question whose honest
    answer is 'nowhere' -- returning the deepest depth would answer a different
    question."""
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=20, z_gap=50.0)
    res = invert_volume_target(vsweep, 10_000.0)
    assert not res.achievable
    assert res.z_required is None and res.p_well_at is None
    assert "No location" in res.message()


def test_the_inverse_reports_the_support_behind_its_answer(reduced, area_depth):
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=40, z_gap=50.0)
    # The deepest *supported* target, not the raw curve's peak: the peak sits in
    # the region the inverse now refuses to answer in.
    targets, _, _ = volume_target_curve(vsweep, n=2)
    shallow = invert_volume_target(vsweep, 12.0)
    deep = invert_volume_target(vsweep, float(targets[-1]))
    assert deep.achievable
    assert shallow.n_discovery_at > deep.n_discovery_at


def test_the_inverse_band_brackets_the_requirement(reduced, area_depth):
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=40, z_gap=50.0, n_boot=200)
    res = invert_volume_target(vsweep, 20.0)
    assert res.achievable
    assert res.z_lo is not None and res.z_hi is not None
    assert res.z_lo <= res.z_required + 1e-6 <= res.z_hi + 1e-6
    assert "band" in res.message()


def test_no_band_is_offered_when_the_sweep_carried_no_bootstrap(reduced, area_depth):
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=20, z_gap=50.0)
    res = invert_volume_target(vsweep, 20.0)
    assert res.z_lo is None and res.z_hi is None
    assert "band" not in res.message()


def test_volume_target_curve_is_monotone_in_depth(reduced, area_depth):
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=40, z_gap=50.0)
    targets, z_req, p_at = volume_target_curve(vsweep, n=25)
    ok = np.isfinite(z_req)
    assert ok.sum() > 5
    assert np.all(np.diff(z_req[ok]) >= -1e-6)
    assert np.all(np.diff(p_at[ok]) <= 1e-6)


# ---------------------------------------------------------- B2's crossings
def test_find_crossing_locates_a_sign_change_by_interpolation():
    z = np.array([0.0, 1.0, 2.0, 3.0])
    a = np.array([3.0, 2.0, 1.0, 0.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    assert find_crossing(z, a, b) == pytest.approx(1.5)


def test_find_crossing_returns_none_when_curves_never_meet():
    z = np.array([0.0, 1.0, 2.0])
    assert find_crossing(z, np.array([5.0, 5.0, 5.0]), np.array([1.0, 1.0, 1.0])) is None


def test_find_crossing_tolerates_the_nan_gaps_thinning_leaves():
    z = np.array([0.0, 1.0, 2.0, 3.0])
    a = np.array([3.0, 2.0, 1.0, 0.0])
    b = np.array([0.0, 1.0, np.nan, 3.0])
    assert find_crossing(z, a, b) is not None


def test_chance_and_regret_cross_on_the_reference_data(reduced, area_depth):
    """The crossing is the decision depth B2 exists to make visible."""
    vsweep = run_volume_sweep(reduced, area_depth, POS, n=60, mefs=14.0, z_gap=50.0)
    z_cross = find_crossing(vsweep.z, vsweep.p_well, vsweep.p_attic_exceeds_mefs)
    assert z_cross is not None
    assert vsweep.z.min() < z_cross < vsweep.z.max()
