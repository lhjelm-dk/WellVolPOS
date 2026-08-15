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


# ------------------------------------------------------------------ the stubs
def test_a_stub_never_outranks_a_working_adapter():
    """Design plan §8. The stubs exist so the protocol is exercised by six
    implementations rather than two — but a stub that scored anything above zero on an
    unverified format is how a real GeoX export ends up on a half-written path.
    """
    from pathlib import Path

    from wellvolpos.io.adapters import ADAPTERS, STUB_ADAPTERS, score_adapters

    assert len(STUB_ADAPTERS) == 4
    assert all(a in ADAPTERS for a in STUB_ADAPTERS)

    data = Path(__file__).resolve().parents[1] / "data" / "demo_prospectA_reduced.csv"
    ranked = score_adapters(data)
    assert all(score == 0.0 for score, a in ranked if a in STUB_ADAPTERS)
    # And the real one still wins outright.
    assert ranked[0][0] > 0.0 and ranked[0][1] not in STUB_ADAPTERS


def test_a_stub_refuses_rather_than_half_importing():
    """A file that half-loads and produces plausible numbers is worse than one that
    refuses, so every stub raises and says what is missing."""
    import pytest

    from wellvolpos.io.adapters import STUB_ADAPTERS

    for adapter in STUB_ADAPTERS:
        with pytest.raises(NotImplementedError) as e:
            adapter.read("anything.csv")
        msg = str(e.value)
        assert adapter.name in msg
        # It has to point somewhere useful, not just decline.
        assert "generic" in msg.lower() or "GeoX" in msg
        assert adapter.needs and adapter.needs in msg


def test_every_stub_satisfies_the_adapter_protocol():
    """The claim the stubs exist to keep honest: adding a simulator is a file, not a
    refactor. If the protocol changes, this fails here rather than in six months."""
    from wellvolpos.io.adapters import STUB_ADAPTERS
    from wellvolpos.io.adapters.base import TrialAdapter

    for adapter in STUB_ADAPTERS:
        assert isinstance(adapter, TrialAdapter), adapter.name
