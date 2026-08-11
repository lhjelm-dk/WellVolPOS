"""The risk summary: the entered chance table times the computed location factor.

Reproduces the summary block Lars keeps in the workbook. The tests below pin two
things: that it *is* the workbook's arithmetic, and that its bottom line is
``P_well`` under every allocation scheme -- cross-checked against
``core.chance.p_well`` rather than against the table's own product, which is the
defence CLAUDE.md asks for after the B4 defect.
"""

import numpy as np
import pytest

from wellvolpos.core.chance import (
    SHIPPED_SCHEMES,
    SUMMARY_COLUMNS,
    p_well,
    risk_summary,
)

from .conftest import ENTRY

# Lars's worked example, from the sheet this reproduces.
ELEMENTS = {"charge": 0.90, "trap": 1.00, "reservoir": 0.60, "retention": 0.80}
R = 0.5625


def test_it_reproduces_the_workbook_summary_to_the_printed_digits():
    """The target: charge 74.3 %, trap 82.6 %, reservoir 60 % (exempt), retention
    66.0 %, multiplying to a well POS of 24.3 % with a correction factor of 0.826."""
    s = risk_summary(ELEMENTS, R, scheme="equal_cube_root")
    at_well = {r["Chance element"]: r[SUMMARY_COLUMNS[2]] for r in s.as_records()}
    assert at_well["Charge"] == pytest.approx(0.7429, abs=5e-4)
    assert at_well["Trap"] == pytest.approx(0.8255, abs=5e-4)
    assert at_well["Reservoir"] == pytest.approx(0.60, abs=5e-4)
    assert at_well["Retention"] == pytest.approx(0.6604, abs=5e-4)
    assert s.prospect_pos == pytest.approx(0.432)
    assert s.well_pos == pytest.approx(0.243, abs=5e-4)
    assert s.correction_factor == pytest.approx(0.8255, abs=5e-4)


def test_the_correction_factor_is_the_cube_root_because_reservoir_is_exempt():
    """Three of the four elements share the penalty, which is why it is a cube root
    and not a fourth root. If reservoir ever started carrying it, this fails."""
    s = risk_summary(ELEMENTS, R, scheme="equal_cube_root")
    assert s.correction_factor == pytest.approx(R ** (1 / 3), abs=1e-9)
    carries = {r["Chance element"]: r["Carries the location penalty"] for r in s.as_records()}
    assert carries["Reservoir"] is False
    assert all(carries[k] for k in ("Charge", "Trap", "Retention"))


@pytest.mark.parametrize("scheme", SHIPPED_SCHEMES)
def test_the_summary_multiplies_to_p_well_under_every_scheme(reduced, scheme):
    """The load-bearing assertion, and the reason this is not computed as the
    product of the third column.

    Under a scheme that allocates the penalty to elements the two agree. Under
    "none" -- which reports ``r`` separately by design -- the third column equals
    the second, so multiplying it out gives POS_prospect. Reporting *that* as the
    well POS would be an unrisked number under a risked label for the fifth time in
    this codebase, so ``none`` gets an explicit location-factor row instead and the
    column still multiplies to the bottom line.
    """
    pos = float(np.prod(list(ELEMENTS.values())))
    expected = p_well(reduced, ENTRY, pos)
    s = risk_summary(ELEMENTS, expected.r_location, scheme=scheme)
    assert s.well_pos == pytest.approx(expected.p_well, abs=1e-9)
    column = float(np.prod([r[SUMMARY_COLUMNS[2]] for r in s.as_records()]))
    assert column == pytest.approx(expected.p_well, abs=1e-9), scheme


def test_the_none_scheme_gets_a_location_row_and_the_others_do_not():
    none = risk_summary(ELEMENTS, R, scheme="none")
    cube = risk_summary(ELEMENTS, R, scheme="equal_cube_root")
    assert [r["Chance element"] for r in none.as_records()][-1] == "Location factor r"
    assert none.correction_factor == 1.0            # nothing was attributed
    assert "Location factor r" not in [r["Chance element"] for r in cube.as_records()]


def test_every_scheme_gives_the_same_well_pos_only_the_split_differs():
    """Allocation is a convention, not a fact -- the point B5 draws."""
    well = [risk_summary(ELEMENTS, R, scheme=s).well_pos for s in SHIPPED_SCHEMES]
    assert well == pytest.approx([well[0]] * len(well), abs=1e-12)
    splits = [
        tuple(round(r[SUMMARY_COLUMNS[2]], 6) for r in risk_summary(ELEMENTS, R, scheme=s).rows)
        for s in SHIPPED_SCHEMES
    ]
    assert len(set(splits)) == len(splits)          # every scheme attributes differently


def test_the_play_column_is_one_and_says_why():
    """This tool assesses one prospect segment and models no play level above it
    (decision 10), so the first column is 1.0 unless a caller states otherwise."""
    s = risk_summary(ELEMENTS, R)
    assert s.play_chance == 1.0
    assert all(r[SUMMARY_COLUMNS[0]] == 1.0 for r in s.as_records())
    assert s.conditional_prospect_chance == pytest.approx(s.prospect_pos)


def test_a_play_chance_scales_the_prospect_and_the_well_together():
    """Carried because the summary is read beside sheets that do have a play level.
    It must move both results, not one."""
    s = risk_summary(ELEMENTS, R, play_chance=0.5)
    base = risk_summary(ELEMENTS, R)
    assert s.prospect_pos == pytest.approx(base.prospect_pos * 0.5)
    assert s.well_pos == pytest.approx(base.well_pos * 0.5)


def test_the_result_lines_are_in_the_order_they_are_read():
    s = risk_summary(ELEMENTS, R)
    assert [r["result"] for r in s.result_records()] == [
        "Play chance", "Cond. prospect chance", "Final prospect POS", "Well location POS",
    ]


def test_a_custom_weighting_reports_no_single_correction_factor():
    """Where each element gets a different factor there is no one number to print,
    and an average would look like one."""
    s = risk_summary(ELEMENTS, R, scheme={"charge": 0.7, "trap": 0.3, "retention": 0.0})
    assert np.isnan(s.correction_factor)
