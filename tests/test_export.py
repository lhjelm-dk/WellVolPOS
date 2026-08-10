"""Phase 5: the case file, and the four export formats.

Two things are worth testing here and one is not. Worth testing: that a case
round-trips and refuses what is not a case, and that every artefact carries the
POS provenance and agrees with what the app computes. Not worth testing: whether
matplotlib can write a PNG.

The agreement checks are the point. This codebase has three times shipped a
figure showing an unrisked number under a risked label, and an export path
multiplies the chances of a fourth -- four writers, each able to recompute
something slightly differently. So the assertions below tie the workbook back to
``p_well()`` and to the class summaries rather than to each other.
"""

import io
import json
import zipfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from wellvolpos.core.chance import p_well
from wellvolpos.io.qc import run_qc
from wellvolpos.report import export as E
from wellvolpos.report.case import FORMAT, Case, fingerprint

from .conftest import ENTRY, EXIT

POS = 0.7605


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def case(reduced):
    return Case(
        entry=ENTRY, exit=EXIT, mefs=14.0, risking_convention="trials_risked",
        dataset="Prospect A — reduced", n_trials=reduced.n_trials,
        fingerprint=fingerprint(reduced),
    )


@pytest.fixture(scope="module")
def vc(reduced, area_depth, groups):
    from wellvolpos.core.classes import split_trials

    return split_trials(reduced, area_depth, groups, ENTRY, EXIT)


@pytest.fixture(scope="module")
def bundle(reduced, case):
    return E.assemble(reduced, case, pos=POS, pos_source="the trials",
                      qc=run_qc(reduced), n_sweep=12, n_volume_sweep=8)


# ------------------------------------------------------------------- the case
def test_a_case_round_trips_through_json(case):
    again = Case.from_json(case.to_json())
    assert again == case


def test_two_saves_of_one_case_differ_only_in_the_timestamp(case):
    """JSON with sorted keys, so a case file is diffable in git -- which is the
    reason to choose it over a pickle."""
    a = json.loads(case.to_json())
    b = json.loads(case.to_json())
    a["settings"].pop("saved_utc")
    b["settings"].pop("saved_utc")
    assert a == b


def test_a_case_stores_settings_and_never_results(case):
    """The load-bearing decision. A case that carried its own KPIs would be a way
    to show last month's numbers under this month's version with nothing on
    screen saying so."""
    stored = json.loads(case.to_json())["settings"]
    forbidden = ("p_well", "r_location", "pos", "proven_mean", "results", "kpi")
    assert not [k for k in stored if any(f in k.lower() for f in forbidden)]


def test_a_case_refuses_things_that_are_not_cases():
    for bad, why in (
        ("{not json", "valid case file"),
        (json.dumps({"format": "something-else"}), "not a WellVolPOS case"),
        (json.dumps({"format": FORMAT, "format_version": 99, "settings": {}}), "newer than"),
        (json.dumps({"format": FORMAT, "format_version": 1}), "no settings block"),
    ):
        with pytest.raises(ValueError, match=why):
            Case.from_json(bad)


def test_a_case_refuses_settings_this_build_does_not_know():
    """A file with extra keys was written by something that knew more than this
    reader. Naming them beats dropping them in silence."""
    payload = json.dumps({
        "format": FORMAT, "format_version": 1,
        "settings": {"entry": 3500.0, "exit": 3550.0, "mefs": 14.0, "vertical_risk_model": "x"},
    })
    with pytest.raises(ValueError, match="does not know"):
        Case.from_json(payload)


def test_a_case_refuses_an_unknown_convention_rather_than_defaulting():
    """Silently falling back would answer a different question under the same
    label -- the one mistake this codebase keeps making."""
    for kwargs, why in (
        ({"risking_convention": "made_up"}, "unknown risking convention"),
        ({"reference": "made_up"}, "made_up"),
        ({"scheme": "made_up"}, "unknown allocation scheme"),
    ):
        with pytest.raises(ValueError, match=why):
            Case(entry=3500.0, exit=3550.0, mefs=14.0, **kwargs)


def test_a_case_refuses_an_exit_above_its_entry():
    with pytest.raises(ValueError, match="cannot leave the reservoir shallower"):
        Case(entry=3550.0, exit=3500.0, mefs=14.0)


def test_the_fingerprint_sees_one_run_exported_twice_as_one_run(reduced, full):
    """``data/`` holds one GeoX run exported at 7 columns and at 60, so a case
    saved from one must recognise the other. Hashing every column present would
    call them different trials, which answers the wrong question."""
    assert fingerprint(reduced) == fingerprint(full)


def test_the_fingerprint_changes_when_the_trials_do(reduced):
    import copy

    other = copy.deepcopy(reduced)
    other.frame.loc[0, "contact"] += 1.0
    assert fingerprint(other) != fingerprint(reduced)


def test_a_file_without_area_fingerprints_differently(reduced):
    """Correct rather than pedantic: without ``area`` there is no A(z) and no
    proven/possible split, so the two files are not interchangeable."""
    import copy

    thin = copy.deepcopy(reduced)
    thin.frame = thin.frame.drop(columns=["area"])
    assert fingerprint(thin) != fingerprint(reduced)


def test_reopening_a_case_on_other_trials_says_so(reduced, case):
    """It must still open -- re-running a case on a new export is a reasonable
    thing to want -- but it may not present the result as a reproduction.

    Checked against a genuinely different trial set rather than against the full
    export, which is the *same* run and must therefore not warn."""
    import copy

    other = copy.deepcopy(reduced)
    other.frame["resource"] = other.frame["resource"] * 1.05
    assert case.check_against(reduced) == []
    warnings = case.check_against(other)
    assert warnings and "not the ones this case was saved against" in warnings[0]


def test_a_case_flags_a_well_outside_the_new_files_contact_range(reduced):
    deep = float(reduced.col("contact").max()) + 500.0
    c = Case(entry=deep, exit=deep + 10.0, mefs=14.0)
    assert any("outside this file's contact range" in w for w in c.check_against(reduced))


# ----------------------------------------------------------------- the bundle
def test_the_bundle_stamp_carries_the_three_numbers_and_their_provenance(bundle):
    """A figure can be cropped out of a screenshot; the stamp is what cannot be
    cropped out of a file."""
    s = bundle.stamp
    for fragment in ("POS_prospect 0.7605", "r_location", "P_well", "reference contour",
                     "allocation", "the trials"):
        assert fragment in s


def test_the_workbook_kpis_agree_with_p_well_not_with_themselves(bundle, reduced):
    """Cross-checked against ``core.chance.p_well`` rather than against the
    bundle's own product, which is how the B4 bug survived its first test."""
    kpi = {r.quantity: r.value for r in E.tables(bundle)["KPIs"].itertuples()}
    expected = p_well(reduced, ENTRY, POS)
    assert kpi["POS_prospect"] == pytest.approx(POS)
    assert kpi["r_location"] == pytest.approx(expected.r_location)
    assert kpi["P_well"] == pytest.approx(expected.p_well)
    assert kpi["P_well"] == pytest.approx(kpi["POS_prospect"] * kpi["r_location"])


def test_the_workbook_keeps_pos_and_r_location_on_separate_rows(bundle):
    """The one idea the whole tool rests on. An exported sheet must not be the
    place the two are collapsed into one reported figure."""
    kpis = E.tables(bundle)["KPIs"]
    names = list(kpis["quantity"])
    assert "POS_prospect" in names and "r_location" in names and "P_well" in names


def test_the_chance_waterfall_multiplies_to_p_well(bundle, reduced):
    """Same assertion as B4's, on the exported table, for the same reason: the
    figure and the sheet must not be able to disagree."""
    steps = E.tables(bundle)["Chance waterfall"]
    expected = p_well(reduced, ENTRY, POS).p_well
    assert float(np.prod(steps["factor"])) == pytest.approx(expected)
    assert float(steps["running"].iloc[-1]) == pytest.approx(expected)


def test_the_proven_mean_in_the_workbook_is_the_headline_kpi(bundle, vc, groups):
    from wellvolpos.core.classes import class_summary

    kpi = {r.quantity: r.value for r in E.tables(bundle)["KPIs"].itertuples()}
    assert kpi["Proven mean"] == pytest.approx(class_summary(vc, groups)["proven"]["mean"])


def test_the_rose_sheet_labels_both_pmcfs_definitions(bundle):
    """The poster's and ours are different numbers; the sheet exists so neither
    gets quoted as the other."""
    rose = E.tables(bundle)["Rose quantities"]
    names = " ".join(rose["quantity"])
    assert "Pmcfs(well), whole volume" in names and "Pmcfs, proven only" in names
    values = dict(zip(rose["quantity"], rose["value"]))
    assert values["Pmcfs, proven only"] <= values["Pmcfs(well), whole volume"]
    assert values["Pc(well)"] <= values["Pmcfs(well), whole volume"]


def test_the_case_sheet_records_the_conventions_and_the_fingerprint(bundle):
    sheet = dict(zip(E.tables(bundle)["Case"]["setting"], E.tables(bundle)["Case"]["value"]))
    assert sheet["Reference contour"] == "crest"
    assert sheet["Risking convention"] == "trials_risked"
    assert sheet["Trial fingerprint"] == fingerprint(bundle.ts)


def test_the_sweep_sheets_are_columns_not_a_picture(bundle):
    """The workbook is the artefact a reviewer opens to check arithmetic, so the
    sweeps have to be numbers they can re-add."""
    depth = E.tables(bundle)["Depth sweep"]
    assert {"entry_depth_m", "r_location", "p_well"} <= set(depth.columns)
    assert np.all(np.diff(depth["entry_depth_m"]) > 0)
    shares = depth[[c for c in depth.columns if c.startswith("share_")]].sum(axis=1)
    assert np.allclose(shares, 1.0, atol=1e-9)         # the four outcomes partition every trial


def test_every_sheet_name_fits_excels_limit(bundle):
    assert all(len(name) <= 31 for name in E.tables(bundle))


# ---------------------------------------------------------------- the formats
def test_the_workbook_opens_and_holds_every_table(bundle):
    raw = E.workbook_bytes(bundle)
    sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
    assert set(sheets) == {name[:31] for name in E.tables(bundle)}
    assert not sheets["KPIs"].empty


def test_the_pdf_is_one_page_per_figure_plus_a_stamped_cover(bundle):
    from pypdf import PdfReader

    raw = E.pdf_bytes(bundle)
    reader = PdfReader(io.BytesIO(raw))
    n_figures = len(E.build_figures(bundle))
    plt.close("all")
    assert len(reader.pages) == n_figures + 1
    # Broken across lines on the page, so each part is checked rather than the
    # single 160-character string, which would run off an A4 sheet.
    cover = reader.pages[0].extract_text()
    for part in bundle.stamp.split(" · "):
        assert part.strip() in cover


def test_the_figure_archive_carries_the_stamp_and_the_case(bundle):
    """A figure dropped into a slide is separated from its provenance
    immediately, so the provenance travels inside the archive."""
    for fmt in ("png", "svg"):
        raw = E.figures_zip(bundle, fmt)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            assert "README.txt" in names and "case.json" in names
            readme = z.read("README.txt").decode("utf-8")
            for part in bundle.stamp.split(" · "):
                assert part.strip() in readme
            assert Case.from_json(z.read("case.json")) == bundle.case
            figures = [n for n in names if n.endswith(f".{fmt}")]
            assert len(figures) == len(E.build_figures(bundle))
            plt.close("all")


def test_an_unknown_figure_format_is_refused(bundle):
    figs = E.build_figures(bundle)
    fig = next(iter(figs.values()))
    with pytest.raises(ValueError, match="unknown figure format"):
        E.figure_bytes(fig, "tiff")


def test_the_export_closes_the_figures_it_draws(bundle):
    """A dozen figures per rerun, and pyplot keeps every one alive otherwise.
    ``figures_zip`` and ``pdf_bytes`` own the figures they build."""
    plt.close("all")
    E.figures_zip(bundle, "png")
    assert plt.get_fignums() == []
    E.pdf_bytes(bundle)
    assert plt.get_fignums() == []


def test_the_export_says_so_rather_than_faking_the_area_figures(reduced, case):
    """Without a productive-area column there is no split, no A(z) and no map.
    They are absent from the export and named in the warnings, not invented."""
    import copy

    ts = copy.deepcopy(reduced)
    ts.frame = ts.frame.drop(columns=["area"])
    b = E.assemble(ts, case, pos=POS, pos_source="the trials", n_sweep=8)
    assert b.ad is None and b.vc is None and b.vsweep is None
    assert any("No productive-area column" in w for w in b.warnings)
    assert "Extension split" not in E.tables(b)
    figs = E.build_figures(b)
    assert "concepts" not in figs and "map_view" not in figs
    assert "B4_chance_waterfall" in figs           # chance does not need area
    plt.close("all")


def test_the_warnings_from_the_session_reach_the_exported_file(reduced):
    """Including the "these are not the trials this case was saved against"
    warning, which is the whole point of fingerprinting."""
    import copy

    other = copy.deepcopy(reduced)
    other.frame["contact"] = other.frame["contact"] + 3.0
    c = Case(entry=ENTRY, exit=EXIT, mefs=14.0, dataset="other",
             fingerprint=fingerprint(other), n_trials=other.n_trials)
    b = E.assemble(reduced, c, pos=POS, pos_source="the trials", n_sweep=8, n_volume_sweep=6)
    assert any("not the ones this case was saved against" in w for w in b.warnings)
    assert "Warnings" in E.tables(b)
    with zipfile.ZipFile(io.BytesIO(E.figures_zip(b, "png"))) as z:
        assert "not the ones this case was saved against" in z.read("README.txt").decode("utf-8")
    plt.close("all")


def test_the_guide_tab_can_be_told_which_palette_to_draw_in():
    """``report/guide.py`` draws the colour key, and it is reached by a direct
    import rather than through app.py's namespace -- so the ``partial`` that binds
    the dark palette into every ``pfig_*`` cannot reach it. It has to be handed
    the theme explicitly, and it was not: the key rendered light on a dark page,
    which for a colour key is the one thing it must not do.

    A signature check rather than a render, because rendering needs a Streamlit
    script context. It catches the defect that actually occurred.
    """
    import inspect

    from wellvolpos.report.guide import render

    assert "dark" in inspect.signature(render).parameters
