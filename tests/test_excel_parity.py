"""The source workbook is the specification. These numbers lock it.

Every value below was read from `WELL Location POS and Resources
V10052017_prospect A.xlsx` and reproduced independently in numpy. If one of
these ever fails, the port has drifted from the tool Lars and his colleagues
already trust -- which is a bug in the port, not in the workbook.

Written before any figure code, deliberately.
"""

import numpy as np
import pytest

from wellvolpos.core import cube_root_factor, group_summary, r_location
from wellvolpos.core.xlcompat import percentile_inc, percentrank_exc

from .conftest import ENTRY, EXIT

TOL = 1e-6


def _pct(v, petroleum_p):
    """Petroleum percentile: P90 is exceeded 90 % of the time."""
    return percentile_inc(v, 1.0 - petroleum_p / 100.0)


# --------------------------------------------------------- prospect totals
def test_prospect_statistics(reduced):
    res = reduced.col("resource")
    assert res.size == 10_000
    assert np.isclose(res.mean(), 10.313254923, atol=TOL)       # Results!M4
    assert np.isclose(_pct(res, 50), 10.5754, atol=TOL)          # Results!L4
    assert np.isclose(_pct(res, 10), 19.41543, atol=TOL)         # Results!N4


# ------------------------------------------------- well-associated volume
def test_well_associated_volume(reduced):
    res, contact = reduced.col("resource"), reduced.col("contact")
    v = res[contact >= ENTRY]
    assert v.size == 4576                                        # Results!Q5
    assert np.isclose(_pct(v, 99.5), 8.069616250000001, atol=TOL)  # Results!J5
    assert np.isclose(_pct(v, 90), 10.94055, atol=TOL)             # Results!K5
    assert np.isclose(_pct(v, 50), 15.3987, atol=TOL)              # Results!L5
    assert np.isclose(v.mean(), 16.520602897727272, atol=TOL)      # Results!M5
    assert np.isclose(_pct(v, 10), 23.523249999999997, atol=TOL)   # Results!N5


# ------------------------------------------------------ tested by the well
def test_volume_tested_by_well(reduced):
    res, contact = reduced.col("resource"), reduced.col("contact")
    v = res[(contact >= ENTRY) & (contact <= EXIT)]
    assert v.size == 3706                                        # Results!Q6
    assert np.isclose(_pct(v, 90), 10.60365, atol=TOL)             # Results!K6
    assert np.isclose(_pct(v, 50), 14.50345, atol=TOL)             # Results!L6
    assert np.isclose(v.mean(), 14.775307220723105, atol=TOL)      # Results!M6
    assert np.isclose(_pct(v, 10), 19.403, atol=TOL)               # Results!N6


# --------------------------------------------------------- up-dip volume
def test_updip_volume_matches_workbook(reduced):
    """Results!M7 and N7 -- including the chance-failure zeros, as the workbook does."""
    res, contact = reduced.col("resource"), reduced.col("contact")
    v = res[contact < ENTRY]
    assert np.isclose(v.mean(), 5.076377280604707, atol=TOL)       # Results!M7
    assert np.isclose(_pct(v, 10), 11.356839999999998, atol=TOL)   # Results!N7


def test_updip_volume_conditioned_on_charge(reduced, groups):
    """The same quantity with the chance failures removed.

    The workbook averages 2 395 zero-volume chance failures into the attic. Both
    numbers are legitimate and answer different questions -- but this is the one
    that matters when somebody argues for a sidetrack, and it is 79 % larger.
    """
    res = reduced.col("resource")
    v = res[groups.dry_with_attic]
    assert v.size == 3029
    assert np.isclose(v.mean(), 9.090218, atol=1e-5)


# --------------------------------------------------------------- well POS
def test_well_pos_reproduces_cell_E6(reduced):
    """Results!E6 = (1 - PERCENTRANK.EXC(contacts, 3500)) * E3, with E3 = 1.0.

    Excel truncates PERCENTRANK to three significant digits by default, so the
    cell shows 0.458 rather than 0.45759. The truncated value propagates.
    """
    contact = reduced.col("contact")
    rank = percentrank_exc(contact, ENTRY)          # 3-sig-digit truncation applied
    assert np.isclose(rank, 0.542, atol=1e-12)
    assert np.isclose(1.0 - rank, 0.458, atol=1e-12)


def test_well_pos_untruncated(reduced):
    """The statistically clean value, for comparison."""
    contact = reduced.col("contact")
    raw = percentrank_exc(contact, ENTRY, significance=None)
    assert np.isclose(1.0 - raw, 0.4575875745758763, atol=1e-12)


# ------------------------------------------------------------ Results!V15
def test_v15_correction_factor(reduced):
    """V15 = (P_well / POS_prospect)^(1/3) = r_location^(1/3).

    Reproduces the cell to all 16 digits, which is what identifies it as an
    equal split of the location log-risk across three elements.
    """
    contact = reduced.col("contact")
    rank = percentrank_exc(contact, ENTRY)
    r = 1.0 - rank                                   # = 0.458, POS_prospect = 1.0 in the workbook
    assert cube_root_factor(r) == pytest.approx(0.7708238778159993, abs=1e-15)


# ------------------------------------------------------- the decomposition
def test_chance_decomposition(reduced):
    """POS x r_location must reproduce the workbook's well POS."""
    res = reduced.col("resource")
    pos = float((res > 0).mean())
    assert np.isclose(pos, 0.7605, atol=1e-9)

    r, _ = r_location(reduced, ENTRY)
    assert np.isclose(r, 4576 / 7605, atol=1e-12)
    assert np.isclose(r, 0.6017, atol=1e-4)
    assert np.isclose(pos * r, 0.4576, atol=1e-4)


def test_group_summary_shape(reduced, groups):
    """The two locked workbook values, and the set of groups reported.

    ``prospect_success`` was added on 2026-08-12. The key *set* here is a shape
    assertion rather than one of the fifteen locked values -- both of those still
    hold below -- so extending it is an addition to the specification, not a
    contradiction of it.
    """
    s = group_summary(reduced, groups)
    assert set(s) == {"prospect", "prospect_success", "discovery",
                      "attic_dry_hole", "attic_incl_failures"}
    assert np.isclose(s["discovery"]["mean"], 16.520602897727272, atol=TOL)
    assert np.isclose(s["attic_incl_failures"]["mean"], 5.076377280604707, atol=TOL)


def test_the_prospect_mean_is_already_risked_and_the_success_one_is_not(reduced, groups):
    """The identity that makes double-risking an expected volume detectable.

    ``prospect`` spans every trial, chance failures included, so it *is* the
    success-case mean already multiplied by the trial file's own POS. Risking it
    again is the recurring bug in this codebase, and it reported 7.84 MMboe on this
    file where the answer is 10.31.

    On a trial file with no zero-volume rows the two means coincide, which is exactly
    why the error survived on one demo prospect and not the other -- so this is
    asserted on the file that has them.
    """
    from wellvolpos.core import expected_volume

    s = group_summary(reduced, groups)
    res = reduced.col("resource")
    pos_trials = float((res > 0).mean())
    assert pos_trials < 1.0, "this test needs the file with chance failures"

    # prospect == prospect_success x POS_trials, exactly.
    assert np.isclose(s["prospect"]["mean"],
                      s["prospect_success"]["mean"] * pos_trials, atol=TOL)
    # So the expected volume, risked once, comes back to the all-trial mean.
    assert np.isclose(expected_volume(s["prospect_success"]["mean"], pos_trials),
                      s["prospect"]["mean"], atol=TOL)
    # And risking the unconditional one again is measurably wrong.
    wrong = expected_volume(s["prospect"]["mean"], pos_trials)
    assert wrong < s["prospect"]["mean"] * 0.9
