"""Rose's three named quantities.

These are the poster's definitions, not ours, and two of them deliberately
disagree with what the app computes by default. The tests below pin the
definitions *and* the disagreements, because a later tidy-up that quietly made
``p_mcfs_downdip`` agree with what B2 draws would be a change of meaning
disguised as a simplification.

Provenance: `no_regrets(...).at_entry_mean` is checked against the source
workbook's `Results!G8`, which is the one cell in the workbook that computes an
entry-associated volume probabilistically.
"""

import numpy as np
import pytest

from wellvolpos.core.chance import p_well as chance_at
from wellvolpos.core.classes import split_trials
from wellvolpos.core.rose import commercial_chance, no_regrets

from .conftest import ENTRY, EXIT

POS = 0.7605

# Results!G8, "Entry depth asso. vol.", read from the workbook.
WORKBOOK_G8 = 11.666908745874588


def test_at_entry_mean_reproduces_the_workbook_cell(reduced, area_depth):
    """The probabilistic No Regrets volume, against the only cell that has one."""
    nr = no_regrets(reduced, area_depth, ENTRY)
    assert nr.at_entry_mean == pytest.approx(WORKBOOK_G8, rel=1e-9)
    assert nr.at_entry_n > 30


def test_the_deterministic_form_is_area_times_mean_pay_times_mean_yield(reduced, area_depth):
    """The poster's definition, arithmetic and all, so a refactor cannot drift it
    into something that merely lands nearby."""
    nr = no_regrets(reduced, area_depth, ENTRY)
    assert nr.deterministic == pytest.approx(nr.area_at_entry * nr.mean_pay * nr.mean_yield)
    assert nr.area_at_entry == pytest.approx(float(area_depth.area_at(ENTRY)))


def test_the_two_no_regrets_answers_are_close_but_not_the_same(reduced, area_depth):
    """Collapsing pay and yield to their means is the whole difference between
    them. If it ever made no difference at all, one of the two is not being
    computed."""
    nr = no_regrets(reduced, area_depth, ENTRY)
    assert nr.deterministic != pytest.approx(nr.at_entry_mean, rel=1e-6)
    assert abs(nr.deterministic - nr.at_entry_mean) / nr.at_entry_mean < 0.10


def test_no_regrets_grows_as_the_well_moves_down_dip(reduced, area_depth):
    """More area above the entry means more volume a discovery would not regret."""
    v = [no_regrets(reduced, area_depth, z).deterministic for z in (3400.0, 3500.0, 3600.0)]
    assert v[0] < v[1] < v[2]


def test_a_wider_window_pulls_in_more_trials(reduced, area_depth):
    narrow = no_regrets(reduced, area_depth, ENTRY, window_m=1.0)
    wide = no_regrets(reduced, area_depth, ENTRY, window_m=10.0)
    assert wide.at_entry_n > narrow.at_entry_n
    assert narrow.deterministic == pytest.approx(wide.deterministic)  # window is not in it


def test_pc_well_is_p_well_times_pmcfs_and_never_exceeds_p_well(reduced, area_depth, groups):
    """Rose Equation 2. Pc(well) is a chance, so it is bounded by the chance of a
    discovery in the first place — the sanity check that catches a stray
    unconditional probability being substituted for a conditional one."""
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    pw = chance_at(reduced, ENTRY, POS).p_well
    cc = commercial_chance(reduced, groups, vc.proven, pw, mcfs=14.0)
    assert cc.pc_well == pytest.approx(pw * cc.p_mcfs_downdip)
    assert 0.0 < cc.pc_well <= pw
    assert cc.n_discovery == int(groups.discovery.sum())


def test_pmcfs_is_conditional_on_a_discovery_not_on_the_prospect(reduced, area_depth, groups):
    """The recurring bug class in this codebase, checked once more: an unrisked
    number under a risked label. Pmcfs is computed over discovery trials only, so
    changing the entered POS must not move it."""
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    a = commercial_chance(reduced, groups, vc.proven, chance_at(reduced, ENTRY, 0.40).p_well, 14.0)
    b = commercial_chance(reduced, groups, vc.proven, chance_at(reduced, ENTRY, 0.95).p_well, 14.0)
    assert a.p_mcfs_downdip == pytest.approx(b.p_mcfs_downdip)
    assert a.pc_well < b.pc_well          # only Pc(well) carries the POS


def test_the_proven_variant_is_the_smaller_of_the_two(reduced, area_depth, groups):
    """The proven volume is a subset of the well-associated volume, so its chance
    of clearing the same threshold cannot be larger. Both are reported precisely
    so that neither gets quoted as the other."""
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    pw = chance_at(reduced, ENTRY, POS).p_well
    cc = commercial_chance(reduced, groups, vc.proven, pw, mcfs=14.0)
    assert cc.p_mcfs_proven <= cc.p_mcfs_downdip
    assert "Pmcfs(well)" in cc.message() and "Pc(well)" in cc.message()


def test_a_threshold_nothing_clears_gives_zero_not_an_error(reduced, area_depth, groups):
    vc = split_trials(reduced, area_depth, groups, ENTRY, EXIT)
    pw = chance_at(reduced, ENTRY, POS).p_well
    cc = commercial_chance(reduced, groups, vc.proven, pw, mcfs=1e6)
    assert cc.p_mcfs_downdip == 0.0 and cc.pc_well == 0.0


def test_a_location_below_every_contact_has_no_discovery_to_condition_on(reduced, area_depth):
    """Deep enough and the conditional is empty. It must say NaN rather than
    return a number computed from nothing."""
    from wellvolpos.core import group_trials

    deep = float(reduced.col("contact").max()) + 50.0
    g = group_trials(reduced, deep, deep + 50.0)
    vc = split_trials(reduced, area_depth, g, deep, deep + 50.0)
    cc = commercial_chance(reduced, g, vc.proven, 0.0, mcfs=14.0)
    assert cc.n_discovery == 0
    assert np.isnan(cc.p_mcfs_downdip) and np.isnan(cc.pc_well)


def test_no_regrets_refuses_a_trial_set_with_no_successes(reduced, area_depth):
    import copy

    ts = copy.deepcopy(reduced)
    ts.frame["resource"] = 0.0
    with pytest.raises(ValueError, match="no successful trials"):
        no_regrets(ts, area_depth, ENTRY)
