"""The import layer: units, in-memory sources, and the generic reader.

Three things landed together on 2026-08-10 and they are tested together because
they are one path: a file arrives as bytes, some adapter claims it, and its units
are checked before anything numerical happens.

The unit tests are the ones that matter most. Until this existed, ``TrialSet.units``
was parsed and then never looked at, so a file in feet imported silently as metres
and every figure was wrong by a factor of 3.28 with all the shapes still looking
right. That is the hardest class of error to notice, which is why the check
refuses rather than converts.
"""

import copy
import io

import numpy as np
import pandas as pd
import pytest

from wellvolpos.io.adapters import (
    GenericCsvAdapter,
    GeoXAdapter,
    Source,
    propose,
    read_trials,
    score_adapters,
    signature,
)
from wellvolpos.io.qc import run_qc
from wellvolpos.io.units import check_declared, check_plausibility, normalise, verdict

DEMO = "data/demo_prospectA_reduced.csv"


# --------------------------------------------------------------------- source
def test_a_path_and_its_bytes_read_identically(reduced):
    """The point of `Source`: one code path for a demo file and an upload, rather
    than two that can drift."""
    with open(DEMO, "rb") as fh:
        raw = fh.read()
    from_bytes = read_trials(Source(name="upload.csv", data=raw))
    assert from_bytes.n_trials == reduced.n_trials
    assert np.array_equal(from_bytes.col("contact"), reduced.col("contact"))


def test_an_upload_is_never_written_to_disk(tmp_path, monkeypatch):
    """Design plan §10. It is the licensee's data, so the app has no business
    leaving copies of it in a working directory.

    Enforced by reading a file-like object with no path at all: if any adapter
    still needed one, this cannot work.
    """
    with open(DEMO, "rb") as fh:
        raw = fh.read()

    class FakeUpload(io.BytesIO):
        name = "my_prospect.csv"

    monkeypatch.chdir(tmp_path)
    ts = read_trials(FakeUpload(raw))
    assert ts.n_trials == 10_000
    assert list(tmp_path.iterdir()) == []          # nothing spilled anywhere


def test_a_source_survives_being_read_twice(reduced):
    """A reader peeks at the header and then parses, and Streamlit replays the
    whole script on every interaction. A cursor left at EOF would give an empty
    frame the second time."""
    with open(DEMO, "rb") as fh:
        src = Source(name="x.csv", data=fh.read())
    assert read_trials(src).n_trials == read_trials(src).n_trials == 10_000


def test_a_source_spots_a_spreadsheet_whatever_it_is_called(tmp_path):
    """An upload's name is whatever the browser supplied, and a spreadsheet saved
    as .txt is a real thing."""
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    path = tmp_path / "sheet.xlsx"
    frame.to_excel(path, index=False)
    renamed = Source(name="sheet.txt", data=path.read_bytes())
    assert renamed.is_excel                        # by the zip magic number, not the suffix


def test_a_source_refuses_what_it_cannot_read():
    with pytest.raises(TypeError, match="cannot read trials from"):
        Source.from_any(object())


# ---------------------------------------------------------------------- units
def test_the_demo_files_pass_and_say_what_was_assumed(reduced, full):
    """Neither demo carries a unit row -- a 7-column paste has nowhere to put one --
    so the honest verdict is not "units pass" but "these were assumed, and here is
    what was checked instead"."""
    for ts in (reduced, full):
        level, message = verdict(ts)
        assert level == "pass"
        assert "declares no units" in message and "assumed" in message
        assert "sanity-checked" in message


def test_a_declared_wrong_unit_is_refused_and_never_converted(reduced):
    """The check the plan asked for. A factor is a guess about what the number
    means; getting it wrong makes every figure wrong by a constant."""
    for field, unit, fragment in (
        ("contact", "ft", "feet"),
        ("area", "acres", "acres"),
        ("resource", "MMbbl", "not oil-equivalent"),
        ("area", "m2", "square metres"),
    ):
        ts = copy.deepcopy(reduced)
        ts.units[field] = unit
        level, message = verdict(ts)
        assert level == "fail", (field, unit)
        assert fragment in message
        assert "nothing is converted" in message.lower()


def test_a_matching_declared_unit_passes_in_any_reasonable_spelling(reduced):
    for spelling in ("km2", "km^2", "km²", "KM2", " sq km "):
        ts = copy.deepcopy(reduced)
        ts.units["area"] = spelling
        assert verdict(ts)[0] == "pass", spelling


def test_an_unrecognised_unit_warns_rather_than_refusing(reduced):
    """Refusing a file over a spelling nobody anticipated would be worse than
    saying so."""
    ts = copy.deepcopy(reduced)
    ts.units["resource"] = "kilo-widgets"
    level, message = verdict(ts)
    assert level == "warn"
    assert "not a spelling" in message


def test_depths_in_feet_are_caught_from_the_numbers_alone(reduced):
    """The check that matters in practice, because the everyday file declares no
    units at all. 3 500 m is 11 483 ft, so the two populations do not overlap."""
    ts = copy.deepcopy(reduced)
    ts.frame["contact"] = ts.frame["contact"] * 3.2808
    level, message = verdict(ts)
    assert level == "fail"
    assert "not a plausible depth in metres" in message
    assert run_qc(ts).blocked                       # and it stops the analysis tabs


def test_a_fraction_given_as_a_percentage_is_caught(full):
    ts = copy.deepcopy(full)
    ts.frame["net_gross"] = ts.frame["net_gross"] * 100.0
    level, message = verdict(ts)
    assert level == "fail"
    assert "percentage rather than a fraction" in message


def test_the_module_does_not_claim_checks_it_cannot_make(reduced):
    """Area in acres and pay in feet are *not* detectable from magnitude -- 3.2 km²
    is 790 acres, and a 790 km² prospect is possible; 45 m is 148 ft, still a
    plausible thickness. The module says so in its docstring and this test holds it
    to that rather than letting a future change quietly start guessing."""
    acres = copy.deepcopy(reduced)
    acres.frame["area"] = acres.frame["area"] * 247.1
    assert check_plausibility(acres) == []
    feet_pay = copy.deepcopy(reduced)
    feet_pay.frame["gross_pay"] = feet_pay.frame["gross_pay"] * 3.2808
    assert check_plausibility(feet_pay) == []


def test_normalise_is_case_and_space_insensitive():
    assert normalise("  M   TVDSS ") == "m tvdss"


def test_a_field_with_no_declared_unit_is_not_warned_about_twice(reduced):
    """Absence is reported once, by the verdict, not once per field."""
    assert check_declared(reduced) == []


# ------------------------------------------------------------ generic adapter
def _write(tmp_path, name, frame, **kw):
    path = tmp_path / name
    frame.to_csv(path, index=False, **kw)
    return path


@pytest.fixture
def european(tmp_path):
    """Semicolon-separated with a decimal comma, and an in-place column to avoid."""
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({
        "Run": np.arange(n),
        "Free Water Level (m)": 3200 + rng.random(n) * 400,
        "Recoverable Volume (MMboe)": rng.lognormal(2.5, 0.6, n),
        "Closure Area": rng.random(n) * 5,
        "Pay thickness": 20 + rng.random(n) * 40,
        "STOIIP inplace": rng.lognormal(3.0, 0.6, n),
    })
    return _write(tmp_path, "euro.csv", frame, sep=";", decimal=",")


def test_the_generic_reader_never_outranks_an_adapter_that_knows_the_format(reduced):
    """The whole mechanism that keeps a real GeoX export off the fallback path.
    GeoX scores 1.0 on the demos; the generic reader is capped at 0.30."""
    for f in (DEMO, "data/demo_prospectA_full.csv"):
        scored = score_adapters(f)
        assert isinstance(scored[0][1], GeoXAdapter)
        assert scored[0][0] > 0.9
        generic = next(s for s, a in scored if isinstance(a, GenericCsvAdapter))
        assert generic <= 0.3
        assert read_trials(f).source == GeoXAdapter().name


def test_the_generic_reader_handles_a_decimal_comma_and_a_semicolon(european):
    ts = read_trials(european)
    assert ts.source == GenericCsvAdapter().name
    assert ts.n_trials == 400
    # Parsed as numbers, not as text that silently became NaN.
    assert 3200.0 < float(np.median(ts.col("contact"))) < 3600.0
    assert any("Decimal comma" in n for n in ts.notes)


def test_the_generic_reader_does_not_mistake_an_in_place_volume_for_a_resource(european):
    """The trap worth guarding: STOIIP reads as a resource and is the wrong
    quantity, so a file carrying both must resolve to the recoverable one."""
    proposal = propose(european)
    assert proposal.mapping["resource"] == "Recoverable Volume (MMboe)"
    assert "inplace" not in proposal.mapping["resource"].lower()


def test_a_weak_header_match_is_flagged_rather_than_trusted(european):
    """"Free Water Level" is a contact, but the header does not say so plainly --
    which is exactly the case a person should confirm."""
    proposal = propose(european)
    assert "contact" in proposal.needs_confirmation
    assert proposal.confidence["contact"] < 0.6
    assert any("weak header match" in n for n in read_trials(european).notes)


def test_a_manual_mapping_overrides_the_proposal_and_clears_the_flag(european):
    ts = GenericCsvAdapter(mapping={"contact": "Free Water Level (m)"}).read(european)
    assert ts.source_columns["contact"] == "Free Water Level (m)"
    assert not any("weak header match" in n for n in ts.notes)


def test_a_manual_mapping_that_names_a_missing_column_says_so(european):
    with pytest.raises(ValueError, match="is not in"):
        propose(european, mapping={"contact": "No Such Column"})
    with pytest.raises(ValueError, match="not a canonical field"):
        propose(european, mapping={"nonsense": "Run"})


def test_a_title_row_does_not_defeat_the_delimiter_sniff(tmp_path):
    """Found by testing rather than reasoning: a leading title row has no
    delimiter in it, which made every candidate look inconsistent, and a
    tab-separated file was then read as comma-separated and refused."""
    path = tmp_path / "tabbed.txt"
    rows = ["My Prospect export", "\t".join(["Iteration", "HC Water Contact", "EUR", "Productive area"]),
            "\t".join(["-", "m", "MMboe", "km2"])]
    rows += ["\t".join([str(i), f"{3300 + i * 0.2:.2f}", f"{10 + i * 0.01:.3f}",
                        f"{1 + i * 0.001:.3f}"]) for i in range(300)]
    path.write_text("\n".join(rows), encoding="utf-8")

    ts = read_trials(path)
    assert ts.n_trials == 300
    assert ts.source_columns["contact"] == "HC Water Contact"
    assert any("Delimiter detected" in n for n in ts.notes)


def test_a_units_row_under_the_header_is_kept_and_then_checked(tmp_path):
    """The two features meeting: the generic reader finds the units row, and the
    unit check confirms them instead of ignoring them."""
    path = tmp_path / "withunits.csv"
    rows = ["Iteration,HC Water Contact,EUR,Productive area", "-,m,MMboe,km2"]
    rows += [f"{i},{3300 + i * 0.2:.2f},{10 + i * 0.01:.3f},{1 + i * 0.001:.3f}" for i in range(200)]
    path.write_text("\n".join(rows), encoding="utf-8")

    ts = read_trials(path)
    assert ts.units["contact"] == "m" and ts.units["resource"] == "MMboe"
    level, message = verdict(ts)
    assert level == "pass" and "confirmed against the file" in message


def test_a_file_it_cannot_map_is_refused_with_what_it_saw(tmp_path):
    path = _write(tmp_path, "junk.csv", pd.DataFrame({"alpha": [1, 2], "beta": [3, 4]}))
    with pytest.raises(ValueError, match="No adapter recognised"):
        read_trials(path)


def test_the_signature_is_stable_and_column_order_independent(tmp_path, european):
    """What makes "remember this mapping for the next export like it" possible."""
    assert signature(european) == signature(european)
    frame = pd.read_csv(european, sep=";", decimal=",")
    reordered = _write(tmp_path, "reordered.csv", frame[list(frame.columns)[::-1]],
                       sep=";", decimal=",")
    assert signature(reordered) == signature(european)


def test_the_signature_changes_when_the_columns_do(tmp_path, european):
    frame = pd.read_csv(european, sep=";", decimal=",").rename(columns={"Closure Area": "Area"})
    other = _write(tmp_path, "renamed.csv", frame, sep=";", decimal=",")
    assert signature(other) != signature(european)


def test_the_generic_reader_reads_a_spreadsheet_too(tmp_path, european):
    frame = pd.read_csv(european, sep=";", decimal=",")
    path = tmp_path / "book.xlsx"
    frame.to_excel(path, index=False)
    assert read_trials(path).n_trials == len(frame)


def test_rows_without_a_contact_or_a_resource_are_dropped_and_counted(tmp_path):
    """Both readers drop such rows -- they have to -- but neither may do it
    quietly. A trial count that silently differs from the one the exporter
    reports is the sort of discrepancy that surfaces halfway through arguing
    about a number.

    The GeoX reader was the one doing it silently, and this test is what found
    it: "HC Water Contact" is enough for GeoX to claim the file, so the generic
    reader never saw it.
    """
    frame = pd.DataFrame({
        "HC Water Contact": [3300.0, np.nan, 3400.0],
        "EUR": [10.0, 11.0, np.nan],
    })
    path = _write(tmp_path, "gappy.csv", frame)
    ts = read_trials(path)
    assert ts.source == GeoXAdapter().name
    assert ts.n_trials == 1
    assert any("Dropped 2" in n for n in ts.notes)

    # And the same file through the generic reader, which is the other half.
    forced = GenericCsvAdapter().read(path)
    assert forced.n_trials == 1
    assert any("Dropped 2" in n for n in forced.notes)
