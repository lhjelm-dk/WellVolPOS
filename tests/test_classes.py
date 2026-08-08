"""Proven / possible / attic — the extension beyond the reference grouping."""

import numpy as np
import pytest

from wellvolpos.core import class_summary, split_trials
from wellvolpos.core.classes import check_area_pay_correlation

from .conftest import ENTRY, EXIT


def test_headline_kpi_is_proven_mean(reduced, area_depth, groups):
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    s = class_summary(vc, groups)
    assert s["proven"]["mean"] == pytest.approx(15.76, abs=0.02)
    assert s["proven"]["n"] == 4576


def test_possible_below_exit_is_separate_and_small_here(reduced, area_depth, groups):
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    s = class_summary(vc, groups)
    assert s["possible"]["mean"] == pytest.approx(0.76, abs=0.02)
    # proven + possible must reconstruct the discovery-case total exactly
    assert s["discovery"]["mean"] == pytest.approx(
        s["proven"]["mean"] + s["possible"]["mean"], abs=1e-9
    )


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
