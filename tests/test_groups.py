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
