"""The location sweep: both engines evaluated across a depth grid."""

import numpy as np
import pytest

from wellvolpos.core.chance import ReferenceContour, p_well
from wellvolpos.core.sweep import run_sweep

from .conftest import ENTRY

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
