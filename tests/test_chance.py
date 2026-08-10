"""The location factor, the two reference conventions, and risk allocation."""

import numpy as np
import pytest

from wellvolpos.core import ReferenceContour, allocate, cube_root_factor, p_well, r_location
from wellvolpos.core.chance import (
    ELEMENTS,
    SCHEMES,
    SHIPPED_SCHEMES,
    normalised_weights,
    waterfall_steps,
)

from .conftest import ENTRY

TABLE = {"charge": 0.92, "trap": 0.94, "reservoir": 0.95, "retention": 0.93}


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


# ------------------------------------------------------- normalised weights
def test_shipped_scheme_weights_are_zero_or_normalised_to_one():
    for scheme in SHIPPED_SCHEMES:
        w, _ = normalised_weights(scheme)
        total = sum(w.values())
        assert total == pytest.approx(0.0) or total == pytest.approx(1.0)


def test_partial_weights_are_normalised_not_taken_at_face_value():
    """The bug this guards: reading a raw weight sum of 0.6 and then adding a
    separate r^0.4 term double-counts the location factor, because allocate has
    already scaled those weights up to sum to 1."""
    w, notes = normalised_weights({"charge": 0.3, "trap": 0.3})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["charge"] == pytest.approx(0.5)
    assert any("normalised" in n for n in notes)


# ------------------------------------------------------------ B4 waterfall
@pytest.mark.parametrize("scheme", SHIPPED_SCHEMES)
def test_waterfall_total_is_p_well_from_the_real_function(reduced, scheme):
    """Cross-checked against p_well itself, not against the waterfall's own product.

    This is the assertion that catches a waterfall drawn from the chance table
    while the app's P_well comes from the trials: the two ran 0.6017 against
    0.4576 on the demo data, and a self-consistency check passed throughout.
    """
    pos = 0.7605
    expected = p_well(reduced, ENTRY, pos).p_well
    steps = waterfall_steps(TABLE, r_location(reduced, ENTRY)[0], pos, scheme)
    total = float(np.prod([f for _, f, _ in steps]))
    assert total == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("pos", [0.7605, 0.60, 1.0])
@pytest.mark.parametrize("scheme", SHIPPED_SCHEMES)
def test_waterfall_total_holds_for_any_pos_and_scheme(pos, scheme):
    r = 0.6017094
    steps = waterfall_steps(TABLE, r, pos, scheme)
    assert float(np.prod([f for _, f, _ in steps])) == pytest.approx(pos * r, abs=1e-12)


def test_waterfall_names_a_reconciliation_step_when_the_table_disagrees_with_pos():
    """A table that does not multiply to the POS in use is the normal case when
    the trials carry the risking. The gap has to be named, not absorbed."""
    steps = waterfall_steps(TABLE, 0.6017094, 0.7605, "none")
    roles = [role for _, _, role in steps]
    assert "reconcile" in roles
    labels = [lab for lab, _, role in steps if role == "reconcile"]
    assert labels == ["POS reconciliation"]


def test_waterfall_has_no_reconciliation_step_when_the_table_is_the_pos():
    pos = float(np.prod([TABLE[e] for e in ELEMENTS]))
    steps = waterfall_steps(TABLE, 0.6017094, pos, "none")
    assert "reconcile" not in [role for _, _, role in steps]


def test_waterfall_reports_r_separately_only_under_the_none_scheme():
    r = 0.6017094
    none_steps = waterfall_steps(TABLE, r, 0.7605, "none")
    standalone = [f for lab, f, role in none_steps if role == "location" and lab.startswith("Location")]
    assert standalone == [pytest.approx(r)]

    for scheme in ("equal_cube_root", "all_to_trap"):
        steps = waterfall_steps(TABLE, r, 0.7605, scheme)
        assert not [lab for lab, _, role in steps if role == "location" and lab.startswith("Location")]
        # ...but the location penalty is still shown, attached to the elements
        # that carry it, rather than vanishing into them silently.
        assert [lab for lab, _, role in steps if role == "location"]


def test_waterfall_attaches_the_penalty_to_the_elements_a_scheme_weights():
    steps = waterfall_steps(TABLE, 0.6017094, 0.7605, "all_to_trap")
    carriers = [lab.split(" ")[0] for lab, _, role in steps if role == "location"]
    assert carriers == ["Trap"]

    steps = waterfall_steps(TABLE, 0.6017094, 0.7605, "equal_cube_root")
    carriers = [lab.split(" ")[0] for lab, _, role in steps if role == "location"]
    assert carriers == ["Charge", "Trap", "Retention"]      # reservoir exempt
    assert "Reservoir" not in carriers
