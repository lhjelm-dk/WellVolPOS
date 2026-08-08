"""The location factor, the two reference conventions, and risk allocation."""

import numpy as np
import pytest

from wellvolpos.core import ReferenceContour, allocate, cube_root_factor, p_well, r_location
from wellvolpos.core.chance import SCHEMES

from .conftest import ENTRY


def test_r_location_is_conditional_on_success(reduced):
    """The failure trials belong to POS, not to the location factor."""
    r, _ = r_location(reduced, ENTRY)
    assert r == pytest.approx(4576 / 7605, abs=1e-12)


def test_p_well_decomposition(reduced):
    res = reduced.col("resource")
    pos = float((res > 0).mean())
    result = p_well(reduced, ENTRY, pos)
    assert result.p_well == pytest.approx(0.4576, abs=1e-4)
    assert result.pos_prospect * result.r_location == pytest.approx(result.p_well)


def test_rose_convention_is_a_flat_uplift(reduced):
    """Normalising at the P90 area gives a constant 1/0.90 = 1.11x versus the crest."""
    for z in (3450.0, 3500.0, 3550.0):
        crest, _ = r_location(reduced, z, reference=ReferenceContour.CREST)
        rose, ref_depth = r_location(reduced, z, reference=ReferenceContour.P90_AREA)
        assert ref_depth is not None
        if rose < 1.0:                       # away from the cap
            assert rose / crest == pytest.approx(1 / 0.90, abs=1e-9)


def test_rose_convention_caps_updip_of_the_reference(reduced):
    """Up-dip of the P90-area contour, P_well = Pg and the factor stops rising."""
    r, ref_depth = r_location(reduced, 3300.0, reference=ReferenceContour.P90_AREA)
    assert r == pytest.approx(1.0)
    assert 3400.0 < ref_depth < 3500.0


def test_v15_is_the_cube_root(reduced):
    assert cube_root_factor(0.458) == pytest.approx(0.7708238778159993, abs=1e-15)


def test_all_allocating_schemes_give_the_same_p_well():
    """Only the attribution differs. This is the property the figures assert."""
    elements = {"charge": 0.92, "trap": 0.94, "reservoir": 0.95, "retention": 0.93}
    r = 0.6017
    baseline = np.prod(list(elements.values())) * r
    for scheme in ("equal_cube_root", "all_to_trap"):
        revised, _ = allocate(elements, r, scheme)
        assert np.prod(list(revised.values())) == pytest.approx(baseline, rel=1e-12)


def test_the_none_scheme_leaves_the_table_untouched():
    """The default reports r as its own term rather than folding it into elements."""
    elements = {"charge": 0.92, "trap": 0.94, "reservoir": 0.95, "retention": 0.93}
    revised, _ = allocate(elements, 0.6017, "none")
    assert revised == pytest.approx(elements)


def test_reservoir_is_exempt_under_every_shipped_scheme():
    elements = {"charge": 0.92, "trap": 0.94, "reservoir": 0.95, "retention": 0.93}
    for scheme in SCHEMES:
        revised, _ = allocate(elements, 0.6017, scheme)
        assert revised["reservoir"] == pytest.approx(0.95)


def test_cube_root_scheme_matches_the_workbook():
    elements = {"charge": 1.0, "trap": 1.0, "reservoir": 1.0, "retention": 1.0}
    revised, _ = allocate(elements, 0.458, "equal_cube_root")
    for el in ("charge", "trap", "retention"):
        assert revised[el] == pytest.approx(0.7708238778159993, abs=1e-15)


def test_custom_weights_are_normalised():
    revised, warnings = allocate({"charge": 1.0, "trap": 1.0, "reservoir": 1.0, "retention": 1.0},
                                 0.5, {"charge": 1.0, "trap": 1.0})
    assert any("normalised" in w for w in warnings)
    assert np.prod(list(revised.values())) == pytest.approx(0.5, rel=1e-12)


def test_floor_warning_fires():
    _, warnings = allocate({"charge": 0.15, "trap": 0.9, "reservoir": 0.9, "retention": 0.9},
                           0.05, "equal_cube_root")
    assert any("floor" in w for w in warnings)
