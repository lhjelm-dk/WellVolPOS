"""Proven / possible / attic — the extension beyond the reference grouping."""

import numpy as np
import pytest

from wellvolpos.core import class_summary, split_trials
from wellvolpos.core.classes import check_area_pay_correlation

from .conftest import ENTRY, EXIT


def test_headline_kpi_is_proven_mean(reduced, area_depth, groups):
    """16.04 on the wedge, against 15.76 under the area rule it replaced.

    The number moved on 2026-08-11 and the move is the point rather than drift --
    see ``test_the_wedge_moves_volume_up_dip_against_the_old_area_rule`` for the
    argument. Both are pinned so neither can change unnoticed.
    """
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    s = class_summary(vc, groups)
    assert s["proven"]["mean"] == pytest.approx(16.04, abs=0.02)
    assert s["proven"]["n"] == 4576


def test_the_two_possible_readings_are_different_and_both_are_kept(reduced, area_depth,
                                                                  groups):
    """One is additive, the other is the size of the upside. Neither replaces the other.

    ``possible_of_discovery`` spans every discovery trial, zeros included, and is the
    member that makes the split a *decomposition*: proven + it = discovery, exactly.
    ``possible`` is conditional on there being anything below the exit at all, which is
    the event the class is named after -- and on this file 81 % of the discovery group
    contributes an exact zero, which is what dragged the reported P50 to 0.00 and the
    mean to a fifth of the real upside (Lars, 2026-08-14).
    """
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    s = class_summary(vc, groups)

    # The additive one, and the identity that has to survive *any* apportionment: the
    # rule decides where the boundary falls, never how much there is in total.
    assert s["possible_of_discovery"]["mean"] == pytest.approx(0.48, abs=0.02)
    assert s["discovery"]["mean"] == pytest.approx(
        s["proven"]["mean"] + s["possible_of_discovery"]["mean"], abs=1e-9
    )

    # The conditional one: a smaller population, a larger volume, and no zeros in it.
    assert s["possible"]["mean"] == pytest.approx(2.53, abs=0.02)
    assert s["possible"]["n"] < s["possible_of_discovery"]["n"]
    assert s["possible"]["mean"] > s["possible_of_discovery"]["mean"] * 4
    assert s["possible"]["p50"] > 0.0, "a conditional percentile must not be a zero"
    assert s["possible_of_discovery"]["p50"] == pytest.approx(0.0)

    # And it must NOT be the additive one, which is the mistake being guarded against.
    assert s["discovery"]["mean"] != pytest.approx(
        s["proven"]["mean"] + s["possible"]["mean"], abs=0.01
    )


def test_the_wedge_moves_volume_up_dip_against_the_old_area_rule(
    reduced, area_depth, groups
):
    """The split apportions on the wedge, and that is not a tuning.

    The rule until 2026-08-11 was ``A(lkh) / A(contact)`` -- a ratio of *map
    areas*, i.e. uniform pay and yield per unit area. It contradicted the geometry
    ``core/reservoir.py`` is built on and validated against GeoX to 0.01 m: the
    charged interval is a **wedge**, full reservoir thickness up-dip, pinching out
    to zero at the contact. Volume therefore sits further up-dip than a per-area
    rule allows.

    So the direction is a consequence of the geometry, not of this dataset, and
    that is what is asserted: the wedge must move volume from *possible* into
    *proven*, on every trial where the well left the reservoir still in
    hydrocarbons. Lars asked for the switch after the bias was measured at about
    six points of the accumulation on both demo prospects.
    """
    wedge = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    area = split_trials(reduced, area_depth, groups, ENTRY, EXIT, apportionment="area")
    d = groups.discovery

    assert wedge.apportionment == "wedge" and area.apportionment == "area"
    assert wedge.proven[d].mean() > area.proven[d].mean()
    assert wedge.possible[d].mean() < area.possible[d].mean()
    # ...and never the other way round on any single trial
    assert np.all(wedge.proven[d] >= area.proven[d] - 1e-9)

    # The total is untouched: an apportionment moves the boundary, it does not
    # create or destroy resource.
    assert np.allclose(wedge.proven[d] + wedge.possible[d],
                       area.proven[d] + area.possible[d])

    # Trials whose contact is *above* the exit have no possible volume under
    # either rule -- the well logged the contact, so nothing is left untested.
    seen = d & (np.asarray(reduced.col("contact"), dtype=float) <= EXIT)
    assert seen.any()
    assert np.allclose(wedge.possible[seen], 0.0)
    assert np.allclose(area.possible[seen], 0.0)


def test_an_unknown_apportionment_is_refused(reduced, area_depth, groups):
    """Same rule as the risking convention and the target statistic: a silent
    fallback answers a different question under the caller's label."""
    with pytest.raises(ValueError, match="unknown apportionment"):
        split_trials(reduced, area_depth, groups, ENTRY, EXIT, apportionment="areal")


def test_attic_conditioned_on_charge(reduced, area_depth, groups):
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    s = class_summary(vc, groups)
    assert s["attic_dry_hole"]["mean"] == pytest.approx(9.09, abs=0.02)


def test_split_conserves_resource(reduced, area_depth, groups):
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    res = reduced.col("resource")
    recomposed = vc.proven + vc.possible + vc.attic
    assert np.allclose(recomposed[groups.success], res[groups.success], atol=1e-9)


def test_nothing_is_proven_in_a_dry_hole(reduced, area_depth, groups):
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    assert np.all(vc.proven[~groups.discovery] == 0.0)
    assert np.all(vc.possible[~groups.discovery] == 0.0)


def test_contact_seen_means_nothing_left_below(reduced, area_depth, groups):
    """If the well logs the contact, the accumulation is fully determined."""
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    assert np.allclose(vc.possible[groups.contact_seen], 0.0, atol=1e-9)


def test_deeper_exit_proves_more(reduced, area_depth):
    from wellvolpos.core import group_trials

    out = []
    for exit_depth in (3520.0, 3560.0, 3600.0):
        g = group_trials(reduced, ENTRY, exit_depth)
        vc = split_trials(reduced, area_depth, g, ENTRY, exit_depth)
        out.append(vc.proven[g.discovery].mean())
    assert out[0] < out[1] < out[2]


def test_exit_above_entry_is_rejected(reduced, area_depth, groups):
    with pytest.raises(ValueError, match="shallower than entry"):
        split_trials(reduced, area_depth, groups, 3500.0, 3400.0)


def test_uniform_yield_assumption_is_checked(reduced):
    """On this dataset gross pay is independent of depth, so the split is sound."""
    level, msg, r = check_area_pay_correlation(reduced)
    assert level == "pass"
    assert abs(r) < 0.2
