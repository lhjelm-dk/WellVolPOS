"""The two synthetic trial files, and the branches only they can reach.

These fixtures exist because the demo data is one GeoX run of one prospect, and
two code paths never execute on it: the area / net-pay correlation guard, which
on the reference file is correctly silent, and the risking branch that has to
fall through to the chance table when there are no chance failures to read a POS
from.

Both generators emit GeoX-shaped CSV and are read back through the real adapter,
so these tests exercise the import layer too rather than around it. The cone
gives an exactly known ``A(z)``, which lets the structure recovery be checked
against the truth instead of against itself -- the one thing the real data cannot
provide.
"""

import numpy as np
import pytest

from wellvolpos.core import AreaDepth, group_trials, split_trials
from wellvolpos.core.classes import check_area_pay_correlation
from wellvolpos.core.reservoir import thickness_from_pay
from wellvolpos.io.adapters import read_trials
from wellvolpos.io.failure import detect_failures
from wellvolpos.io.qc import run_qc
from wellvolpos.io.synthetic import (
    FAILURE_SENTINEL,
    correlated_area_pay,
    make_trials,
    success_case_only,
    write,
)

APEX, SPILL, AREA_AT_SPILL = 3200.0, 3700.0, 6.0


def _read(frame, tmp_path, name):
    """Through the adapter, never straight into a TrialSet."""
    path = tmp_path / f"{name}.csv"
    write(frame, path)
    return read_trials(path)


@pytest.fixture(scope="module")
def synthetic_default():
    return make_trials(4_000, seed=7)


# --------------------------------------------------------------- the geometry
def test_the_generated_area_depth_curve_is_the_cone_it_claims(synthetic_default, tmp_path):
    """The only test in the suite that can check ``AreaDepth`` against a known
    truth rather than against its own fit. A(z) = k(z - apex)^2 by construction,
    so the recovered curve has to reproduce it."""
    ts = _read(synthetic_default, tmp_path, "cone")
    ad = AreaDepth.from_trials(ts.col("contact"), ts.col("area"))
    k = AREA_AT_SPILL / (SPILL - APEX) ** 2
    for z in (3300.0, 3450.0, 3600.0):
        assert float(ad.area_at(z)) == pytest.approx(k * (z - APEX) ** 2, rel=1e-3)
    assert ad.r2 > 0.999999


def test_the_recovered_apex_lands_near_the_true_one(synthetic_default, tmp_path):
    """Still an extrapolation -- the trials do not reach the crest -- so this is
    a tolerance, not an identity. It is the honest size of that error."""
    ts = _read(synthetic_default, tmp_path, "apex")
    ad = AreaDepth.from_trials(ts.col("contact"), ts.col("area"))
    assert float(ad.apex_estimate()) == pytest.approx(APEX, abs=40.0)


def test_the_wedge_inversion_recovers_a_thickness_on_synthetic_data(synthetic_default, tmp_path):
    """The inversion is validated against GeoX's own column on the real file;
    here it is validated against a closure whose geometry is known exactly."""
    ts = _read(synthetic_default, tmp_path, "wedge")
    ad = AreaDepth.from_trials(ts.col("contact"), ts.col("area"))
    tfp = thickness_from_pay(ts, ad, apex=APEX)
    assert tfp.n_resolved > 2_000
    assert tfp.n_inconsistent == 0
    resolved = tfp.thickness[tfp.resolved]
    assert np.all(resolved >= ts.col("gross_pay")[tfp.resolved] - 1e-6)


def test_the_inconsistent_count_is_sensitive_to_the_apex_estimate(synthetic_default, tmp_path):
    """A finding this fixture exists to make visible, and the only place it can
    be shown: given the true apex the inversion flags nothing, but given the
    *derived* apex — the extrapolation of A(z) to zero area, ~34 m too deep here —
    it flags a couple of per cent of trials as carrying more hydrocarbon rock than
    the closure holds.

    They are not bad data. A deeper apex makes the modelled closure smaller, so a
    perfectly consistent trial can exceed it. The consequence for reading the QC
    gate: ``n_inconsistent`` is an upper bound on export problems, not a count of
    them. It happens to be zero on the reference file, so nothing there is
    affected — but a shallower-tailed closure would show the same artefact.
    """
    ts = _read(synthetic_default, tmp_path, "apex_sensitivity")
    ad = AreaDepth.from_trials(ts.col("contact"), ts.col("area"))
    derived = float(ad.apex_estimate())
    assert derived > APEX + 10.0                      # the extrapolation runs deep

    truth = thickness_from_pay(ts, ad, apex=APEX)
    estimated = thickness_from_pay(ts, ad)
    n_success = int((ts.col("resource") > 0).sum())
    assert truth.n_inconsistent == 0
    assert 0 < estimated.n_inconsistent < 0.05 * n_success


# ------------------------------------------------------- the risking branch
def test_a_success_case_only_file_offers_no_pos_to_read(tmp_path):
    """The branch the real data cannot reach. With no chance failures POS_trials
    is unreadable, so the app must take POS from the chance table -- and say so
    in its provenance stamp."""
    ts = _read(success_case_only(3_000), tmp_path, "success")
    assert np.all(ts.col("resource") > 0.0)
    report = detect_failures(ts)
    assert not report.has_failures
    assert not run_qc(ts).blocked


def test_a_risked_file_carries_its_failures_the_way_geox_writes_them(tmp_path):
    """Every hydrocarbon quantity exactly zero and the contact stamped with the
    sentinel, while the non-hydrocarbon parameters keep being sampled. That
    signature is what ``io/failure.py`` detects."""
    ts = _read(make_trials(5_000, pos=0.6, seed=3), tmp_path, "risked")
    res = ts.col("resource")
    zero = res == 0.0
    assert zero.sum() == pytest.approx(2_000, abs=1)
    assert np.all(ts.col("contact")[zero] == FAILURE_SENTINEL)
    assert np.all(ts.col("area")[zero] == 0.0)
    report = detect_failures(ts)
    assert report.has_failures
    assert report.pos_trials == pytest.approx(0.6, abs=1e-9)


def test_the_generated_pos_is_the_pos_asked_for(tmp_path):
    for pos in (0.4, 0.76, 1.0):
        ts = _read(make_trials(2_000, pos=pos, seed=11), tmp_path, f"pos{pos}")
        assert float((ts.col("resource") > 0).mean()) == pytest.approx(pos, abs=1e-9)


def test_the_failures_are_not_grouped_at_one_end(tmp_path):
    """A real export does not sort its failures, and this codebase already knows
    ``TrialNumber`` is not a reliable key. Anything depending on row order should
    break on this file."""
    ts = _read(make_trials(2_000, pos=0.7, seed=5), tmp_path, "shuffled")
    zero = np.flatnonzero(ts.col("resource") == 0.0)
    # Failures spread through the file rather than sitting in one block.
    assert zero.min() < 200 and zero.max() > 1_800


# ------------------------------------------------- the correlation guard
def test_the_correlated_file_makes_the_split_guard_speak(tmp_path):
    """The whole point of this fixture. The per-trial split apportions resource by
    *area*, which needs pay and yield roughly uniform across the closure. When
    they are not, the guard has to say so out loud rather than let the split pass
    for an unqualified answer."""
    ts = _read(correlated_area_pay(4_000), tmp_path, "correlated")
    level, message, r = check_area_pay_correlation(ts)
    assert level == "fail"
    assert r > 0.5
    assert "not defensible" in message and "reference grouping" in message


def test_the_independent_file_leaves_the_guard_silent(tmp_path):
    """The control. Without it, a guard that fired on everything would look like
    a guard that works."""
    ts = _read(make_trials(4_000, seed=9), tmp_path, "independent")
    level, _, r = check_area_pay_correlation(ts)
    assert level == "pass"
    assert abs(r) < 0.1


def test_the_correlation_is_the_one_that_was_asked_for(tmp_path):
    """The copula changes the dependence, not the structure, so a correlated file
    and an independent one drawn on the same seed are comparable -- which is what
    makes the pair usable as a controlled experiment.

    Contact and area are preserved exactly. Pay is preserved *except* where the
    closure-volume cap bites, and it has to be applied after the pairing because
    what a trial can hold depends on which contact it drew. So the pay
    distributions are checked as close, not identical, and the assertion pins how
    close."""
    independent = _read(make_trials(4_000, seed=9), tmp_path, "m_ind")
    correlated = _read(make_trials(4_000, seed=9, area_pay_correlation=0.8), tmp_path, "m_cor")
    for name in ("area", "contact"):
        assert np.allclose(np.sort(independent.col(name)), np.sort(correlated.col(name)))

    a, b = np.sort(independent.col("gross_pay")), np.sort(correlated.col("gross_pay"))
    assert float(np.mean(np.abs(a - b))) < 0.5                      # metres, on a mean of ~40
    assert float(abs(a.mean() - b.mean()) / a.mean()) < 0.02
    # The same pay values, differently attached: that is the whole manipulation.
    assert not np.allclose(independent.col("gross_pay"), correlated.col("gross_pay"))


def test_the_split_still_runs_on_correlated_data(tmp_path):
    """It has to: nothing better is available from a trial file. The guard's job
    is to qualify the answer, not to withhold it."""
    ts = _read(correlated_area_pay(2_000), tmp_path, "split_corr")
    ad = AreaDepth.from_trials(ts.col("contact"), ts.col("area"))
    groups = group_trials(ts, 3450.0, 3500.0)
    vc = split_trials(ts, ad, groups, 3450.0, 3500.0)
    disc = groups.discovery
    assert disc.sum() > 100
    total = vc.proven[disc] + vc.possible[disc]
    assert np.allclose(total, vc.discovery_total[disc], rtol=1e-9)


# ------------------------------------------------------------------ contracts
def test_the_generator_refuses_impossible_geometry():
    with pytest.raises(ValueError, match="deeper than the apex"):
        make_trials(100, apex=3700.0, spill=3200.0)
    for pos in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="pos must be"):
            make_trials(100, pos=pos)


def test_the_generated_columns_are_the_geox_ones(synthetic_default):
    from wellvolpos.io.synthetic import COLUMNS

    assert list(synthetic_default.columns) == list(COLUMNS)


def test_the_generated_trials_are_internally_consistent(synthetic_default, tmp_path):
    """``HC GRV = area x gross pay`` is the identity the QC gate checks on every
    import, so a fixture that violated it would fail for the wrong reason."""
    ts = _read(synthetic_default, tmp_path, "consistent")
    m = ts.col("resource") > 0
    assert np.allclose(ts.col("hc_grv")[m], ts.col("area")[m] * ts.col("gross_pay")[m])
    assert np.all(ts.col("gross_pay")[m] <= ts.col("contact")[m] - APEX + 1e-9)


def test_the_generator_is_deterministic():
    a = make_trials(500, seed=42)
    b = make_trials(500, seed=42)
    assert a.equals(b)
    assert not a.equals(make_trials(500, seed=43))
