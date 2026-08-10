"""The reference engine: whole-trial grouping, after Schneider et al. (2023)."""

import numpy as np
import pytest

from wellvolpos.core import group_trials

from .conftest import ENTRY, EXIT


def test_outcome_tree_shares(groups):
    s = groups.shares()
    assert s["chance_failure"] == pytest.approx(0.2395, abs=1e-9)
    assert s["dry_with_attic"] == pytest.approx(0.3029, abs=1e-9)
    assert s["contact_seen"] == pytest.approx(0.3706, abs=1e-9)
    assert s["hc_to_exit"] == pytest.approx(0.0870, abs=1e-9)
    assert s["p_well"] == pytest.approx(0.4576, abs=1e-9)


def test_the_tree_is_exhaustive_and_exclusive(groups):
    total = (
        groups.chance_failure.astype(int)
        + groups.dry_with_attic.astype(int)
        + groups.contact_seen.astype(int)
        + groups.hc_to_exit.astype(int)
    )
    assert np.all(total == 1)


def test_discovery_is_seen_plus_past_exit(groups):
    assert np.array_equal(groups.discovery, groups.contact_seen | groups.hc_to_exit)


def test_moving_the_well_downdip_lowers_p_well(reduced):
    shallow = group_trials(reduced, 3450.0, 3500.0).shares()["p_well"]
    deep = group_trials(reduced, 3550.0, 3600.0).shares()["p_well"]
    assert shallow > deep


def test_exit_omitted_is_handled(reduced):
    g = group_trials(reduced, ENTRY)
    assert np.array_equal(g.contact_seen, g.discovery)
    assert not g.hc_to_exit.any()


# --------------------------------------------------------------- risked_shares
def test_risked_shares_matches_shares_at_pos_trials(groups):
    """The two must agree exactly when the entered POS is POS_trials."""
    plain = groups.shares()
    risked = groups.risked_shares(0.7605, plain["p_well"])
    for k in ("chance_failure", "dry_with_attic", "contact_seen", "hc_to_exit"):
        assert risked[k] == pytest.approx(plain[k], abs=1e-9)


def test_risked_shares_diverges_from_shares_under_a_different_pos(groups):
    """The bug this exists to prevent: a chance table that isn't POS_trials.

    Tab ④'s inline outcome-tree text used groups.shares() directly until this
    was added, so it kept reporting POS_trials's 23.9% chance failure even
    after the sidebar's chance table set POS_prospect to something else --
    right next to a P_well metric that *had* moved.
    """
    plain = groups.shares()
    pos, p_well = 0.50, 0.50 * (plain["p_well"] / 0.7605)  # same r_location, a different POS
    risked = groups.risked_shares(pos, p_well)
    assert risked["chance_failure"] == pytest.approx(0.50, abs=1e-9)
    assert risked["chance_failure"] != pytest.approx(plain["chance_failure"], abs=1e-3)


def test_risked_shares_partitions_to_one_and_pins_discovery_to_p_well(groups):
    for pos, p_well in ((0.7605, 0.4576), (0.50, 0.3009), (1.0, 0.6017)):
        r = groups.risked_shares(pos, p_well)
        total = r["chance_failure"] + r["dry_with_attic"] + r["contact_seen"] + r["hc_to_exit"]
        assert total == pytest.approx(1.0, abs=1e-9)
        assert r["contact_seen"] + r["hc_to_exit"] == pytest.approx(p_well, abs=1e-9)


def test_risked_shares_handles_a_location_with_no_discoveries(reduced):
    """Deep enough and the discovery set is empty, so there is no seen/past-exit
    proportion to carry. p_well is 0 there too, so the bands stay valid."""
    deep = group_trials(reduced, 4000.0, 4050.0)
    assert not deep.discovery.any()
    r = deep.risked_shares(0.7605, 0.0)
    total = r["chance_failure"] + r["dry_with_attic"] + r["contact_seen"] + r["hc_to_exit"]
    assert total == pytest.approx(1.0, abs=1e-9)
    assert r["contact_seen"] == pytest.approx(0.0)
    assert r["hc_to_exit"] == pytest.approx(0.0)
    assert r["dry_with_attic"] == pytest.approx(0.7605)


def test_risked_shares_keeps_the_seen_vs_past_exit_proportions(groups):
    """The split between the two discovery outcomes should be a proportion
    carried over from the trial file, not a value that itself depends on POS."""
    plain = groups.shares()
    expected_seen_frac = plain["contact_seen"] / plain["p_well"]
    risked = groups.risked_shares(0.50, 0.30)
    assert risked["contact_seen"] / 0.30 == pytest.approx(expected_seen_frac, abs=1e-9)
