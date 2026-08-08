"""The failure detector, which decides how geological risk is counted.

Getting this wrong double-counts risk, so the evidence it relies on is asserted
piece by piece rather than only the verdict.
"""

import numpy as np

from wellvolpos.io import detect_failures


def test_detects_chance_failures_in_reduced(reduced):
    rep = detect_failures(reduced)
    assert rep.verdict == "chance_failure"
    assert rep.n_total == 10_000
    assert rep.n_zero == 2395
    assert np.isclose(rep.frac_zero, 0.2395, atol=1e-9)
    assert np.isclose(rep.pos_trials, 0.7605, atol=1e-9)


def test_the_sentinel_signature(reduced):
    """One contact value, a clean gap, hydrocarbons collapsed, the rest still sampled."""
    rep = detect_failures(reduced)
    assert rep.n_unique_zero_contacts == 1
    assert np.isclose(rep.sentinel_contact, 3166.59, atol=1e-9)
    assert np.isclose(rep.shallowest_success_contact, 3360.13, atol=1e-2)
    assert rep.gap_to_shallowest_success > 190.0          # 193.5 m of empty space
    for f in ("resource", "area", "gross_pay"):
        assert f in rep.collapsed_fields


def test_full_export_agrees(full):
    """A different realisation of the same model must give the same verdict."""
    rep = detect_failures(full)
    assert rep.verdict == "chance_failure"
    assert np.isclose(rep.pos_trials, 0.7605, atol=1e-9)
    assert np.isclose(rep.sentinel_contact, 3166.59, atol=1e-9)


def test_non_hydrocarbon_parameters_still_sampled(full):
    """The discriminator: a chance failure zeroes the HC quantities only.

    If the contact distribution itself had put mass above the crest, area and
    pay would be zero *and* the contact would vary. Here porosity, net/gross and
    thickness keep on being sampled inside the failure trials, which is what a
    simulator does when it applies a chance factor after the volumetrics.
    """
    rep = detect_failures(full)
    assert set(rep.still_sampled_fields) >= {"porosity", "net_gross", "thickness"}


def test_success_case_only_file_reports_none(reduced):
    """Strip the failures and the detector must not invent them."""
    import copy

    ts = copy.deepcopy(reduced)
    ts.frame = ts.frame[ts.frame["resource"] > 0].reset_index(drop=True)
    rep = detect_failures(ts)
    assert rep.verdict == "none"
    assert rep.pos_trials is None
    assert not rep.has_failures


def test_geometric_failures_are_distinguished(reduced):
    """Spread the failure contacts out and the verdict must change.

    This is the case where POS is *not* readable from the file, and the app must
    fall back to the user's chance table instead of silently using 1 - n0/n.
    """
    import copy

    ts = copy.deepcopy(reduced)
    zero = ts.frame["resource"] <= 0
    rng = np.random.default_rng(0)
    ts.frame.loc[zero, "contact"] = rng.uniform(3355.0, 3365.0, int(zero.sum()))
    rep = detect_failures(ts)
    assert rep.verdict == "geometric"
    assert rep.pos_trials is None
