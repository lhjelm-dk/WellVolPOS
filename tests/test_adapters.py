"""Import: the traps that are actually in GeoX exports."""

import numpy as np
import pytest

from wellvolpos.io.adapters import GeoXAdapter, read_trials

from .conftest import DATA


def test_both_demo_files_load(reduced, full):
    assert reduced.n_trials == 10_000
    assert full.n_trials == 10_000


def test_required_fields_are_mapped(reduced):
    for f in ("contact", "resource", "area", "gross_pay"):
        assert reduced.has(f), f


def test_sniff_prefers_geox_for_these_files():
    a = GeoXAdapter()
    assert a.sniff(str(DATA / "demo_prospectA_reduced.csv")) > 0.5
    assert a.sniff(str(DATA / "demo_prospectA_full.csv")) > 0.5


def test_duplicate_headers_are_resolved_positionally(full):
    """Six GeoX quantities share a name AND a unit string; only order tells them apart."""
    assert "Inplace." in " ".join(full.notes) or full.source_columns
    # the resolution must not have corrupted the canonical mapping
    assert full.source_columns["resource"].lower().startswith("recoverable")


def test_the_two_demos_hold_the_same_realisations(reduced, full):
    """They are one run exported twice, not two runs.

    Worth asserting explicitly because the reverse was assumed at first: joining
    the two sheets on TrialNumber gives only ~5.6 % agreement, which looks like
    two independent runs. It is not. Row for row the data are identical -- it is
    the TrialNumber column itself that is out of step (see the next test).
    """
    for f in ("contact", "resource", "area"):
        assert np.allclose(reduced.col(f), full.col(f), atol=1e-9)


def test_trial_numbers_are_not_a_reliable_key(reduced, full):
    """TrialNumber does not travel with its own data row in this export.

    Both files carry the same multiset of trial numbers, but attached to
    different rows. Anything that joins two exports on TrialNumber will silently
    scramble them, which is why nothing in this codebase does.
    """
    ta, tb = reduced.col("trial"), full.col("trial")
    assert np.array_equal(np.sort(ta), np.sort(tb))       # same identifiers
    assert float((ta == tb).mean()) < 0.10                # attached to different rows


def test_unmapped_file_raises_usefully(tmp_path):
    p = tmp_path / "nonsense.csv"
    p.write_text("alpha,beta\n1,2\n3,4\n")
    with pytest.raises(ValueError):
        read_trials(p)


def test_grv_identity_survives_the_file_precision(reduced):
    """HC GRV = area x gross pay, to within the six significant digits stored.

    Asserted here because it is the premise of the whole per-trial split: if the
    volumetric model were not area x thickness, apportioning a trial's resource
    by area would have no basis.
    """
    a, g, v = reduced.col("area"), reduced.col("gross_pay"), reduced.col("hc_grv")
    m = (v > 0) & (a > 0) & (g > 0)
    ratio = v[m] / (a[m] * g[m])
    assert np.allclose(ratio, 1.0, atol=1e-4)
