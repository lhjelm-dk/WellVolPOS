"""Figure numbers by tab, and the two ways they have already gone wrong.

Both failures here happened in practice within a day of the renumbering, and both
are invisible to lose: a number the reader can still cross-reference, and a number
that is simply the wrong one.
"""

import re

import pytest

from wellvolpos.ui.numbering import (
    FIGURE_GUIDE,
    FIGURE_NUMBERS,
    LEGACY_CODE,
    guide_table,
    ref,
    renumber_title,
)

CODE = re.compile(r"\b(?:A|B|C)\d{1,2}\b")


# ------------------------------------------------------------- the mapping
def test_every_number_is_unique():
    numbers = list(FIGURE_NUMBERS.values())
    assert len(numbers) == len(set(numbers))


def test_the_guide_documents_exactly_the_numbered_figures():
    """A figure with a number and no guide entry is one the reader cannot look up."""
    assert set(FIGURE_GUIDE) == set(FIGURE_NUMBERS)


def test_numbers_are_grouped_by_tab_in_render_order():
    """3.4 must be on tab 3 and the fourth of them, or the scheme says nothing."""
    seen: dict[str, list[int]] = {}
    for number in FIGURE_NUMBERS.values():
        tab, _, idx = number.partition(".")
        seen.setdefault(tab, []).append(int(idx))
    for tab, order in seen.items():
        assert order == sorted(order), f"tab {tab} is numbered out of render order"
        assert order == list(range(1, len(order) + 1)), f"tab {tab} skips a number"


# ------------------------------------------------------- stripping the code
@pytest.mark.parametrize("separator", ["·", "-", "–", "—", ":"])
def test_a_leading_code_is_replaced_whatever_the_separator(separator):
    """Rewriting one title with a hyphen was enough to break this.

    ``renumber_title`` matched only the middot, so a hyphenated title *prepended*
    and the reader was shown ``"3.12 · B12 - Resource by contact-depth band"`` --
    the very code the renumbering exists to retire, with the new number bolted on
    in front of it.
    """
    out = renumber_title(f"B12 {separator} Resource by contact-depth band", "b12")
    # Read from the mapping, never spelled out: hardcoding "3.12" here is the same
    # mistake the whole module exists to stop, and it broke the moment 3.5 was
    # removed and tab ③ renumbered.
    assert out == f"{FIGURE_NUMBERS['b12']} · Resource by contact-depth band"
    assert not CODE.search(out)


def test_a_title_with_no_code_gets_the_number_prepended():
    """The live section and the map view had no index at all, so no way to refer
    to them -- which is how "the one next to A6" became the only way to say it."""
    assert renumber_title("Live section", "live").startswith(
        f"{FIGURE_NUMBERS['live']} · ")


def test_prose_in_a_title_is_not_mistaken_for_a_code():
    title = "Chance against volume · the location trade-off"
    assert renumber_title(title, "b7") == f"{FIGURE_NUMBERS['b7']} · {title}"


def test_an_unmapped_key_leaves_the_title_alone():
    """A figure the mapping has not been told about should look wrong obviously."""
    assert renumber_title("B99 · something new", "b99") == "B99 · something new"


# ------------------------------------------------------------ prose references
def test_ref_resolves_a_key_to_the_current_number():
    assert ref("the attic curve on {b2}") == f"the attic curve on {FIGURE_NUMBERS['b2']}"


def test_ref_leaves_an_unknown_placeholder_alone():
    """So wrapping a whole caption is safe even when it contains other braces."""
    assert ref("{not_a_figure} and {b2}") == f"{{not_a_figure}} and {FIGURE_NUMBERS['b2']}"


def test_the_guide_table_shows_no_legacy_codes_and_no_hardcoded_numbers():
    """"(was A5)" meant nothing to anyone who had not used the app before the
    renumbering, and a number spelled out by hand goes stale the moment a figure
    is inserted above it -- which is exactly what happened to 2.3."""
    table = guide_table()
    assert "was A" not in table and "was B" not in table
    for key, code in LEGACY_CODE.items():
        assert f"({code})" not in table
    # Every number in the table has to be one the mapping currently issues.
    for number in re.findall(r"\*\*(\d+\.\d+)\*\*", table):
        assert number in set(FIGURE_NUMBERS.values())
    assert "{" not in table, "an unresolved ref() placeholder reached the reader"


def test_every_figure_the_app_charts_has_a_number():
    """The mapping is checked against the *tabs*, not only against itself.

    Every test here inspected ``FIGURE_NUMBERS`` and never the code that draws, so
    three figures — the outcome tree, the two partitions and the wedge — were charted
    for a day carrying raw letter codes (``C6 · …``) while everything around them read
    ``4.1``, ``4.2``. Nothing failed, because the mapping was internally consistent and
    simply did not mention them.

    ``chart(key=…)`` leaves an unmapped title alone by design, which is right for a
    one-off but is exactly what made this silent.
    """
    import re
    from pathlib import Path

    from wellvolpos.ui.numbering import FIGURE_NUMBERS

    ui = Path(__file__).resolve().parents[1] / "wellvolpos" / "ui"
    charted: dict[str, str] = {}
    for f in sorted(ui.glob("tab*.py")) + [
            Path(__file__).resolve().parents[1] / "wellvolpos" / "report" / "guide.py"]:
        s = f.read_text(encoding="utf-8")
        # `_chart(..., key="x")` — widget keys start with `w_` and are excluded.
        for m in re.finditer(r'_chart\((?:[^()]|\([^()]*\))*?key="([a-z0-9]+)"', s):
            charted.setdefault(m.group(1), f.name)

    # Not figures, and listed rather than pattern-matched so a *new* stray fails.
    NOT_FIGURES = {
        "colourkey": "the palette legend on tab ⑥ — a key to the figures, not one",
    }

    assert charted, "found no charted figures — the pattern above has drifted"
    missing = {k: v for k, v in charted.items()
               if k not in FIGURE_NUMBERS and k not in NOT_FIGURES}
    assert not missing, (
        f"charted but unnumbered: {missing}. A figure with no entry in FIGURE_NUMBERS "
        f"keeps its raw letter code on screen while its neighbours are numbered."
    )


def test_the_guide_table_runs_in_figure_number_order():
    """A reader scans it for a number they have just seen on screen.

    ``FIGURE_GUIDE`` is keyed by figure letter and written in the order the figures
    were built, so before 2026-08-18 the table listed 3.7 before 3.5, 3.8 after 3.13
    and 6.1 in the middle of tab ④. Sorting on the *string* would be just as wrong —
    3.10 would follow 3.1 — so the key parses the two halves as integers.
    """
    from wellvolpos.ui.numbering import _number_key

    keys = sorted(FIGURE_GUIDE, key=_number_key)
    numbers = [FIGURE_NUMBERS[k] for k in keys]
    parsed = [tuple(int(x) for x in n.split(".")) for n in numbers]
    assert parsed == sorted(parsed), numbers

    # And the rendered table follows that order, not the dict's.
    body = guide_table()
    positions = [body.index(f"**{n}**") for n in numbers]
    assert positions == sorted(positions), numbers


# --------------------------------------------------- the report carries app numbers
# Lars, 2026-08-18: "the figures carry names like A4, B2, C4 ... I want the 3.12, 2.2,
# 4.2 kind of numbering." The app renumbered on screen and the export path did not, so
# the same figure was 2.1 in the browser and A1 in the PDF.


def test_every_exported_figure_maps_to_a_number():
    """No bundle entry may be unnumbered except the ones deliberately listed.

    The colour key is a *legend*, not a figure — it has no number on screen either, and
    inventing one for the report would give the document a figure the app does not have.
    """
    from wellvolpos.ui.numbering import (EXPORT_FIGURE_KEYS, UNNUMBERED_EXPORT_KEYS,
                                         export_number)

    for key in EXPORT_FIGURE_KEYS:
        assert export_number(key), key
    assert not (set(EXPORT_FIGURE_KEYS) & UNNUMBERED_EXPORT_KEYS)



def test_the_export_key_map_covers_the_bundle(reduced):
    """A new export figure without a number would ship as a letter code again."""
    from wellvolpos.report import export as E
    from wellvolpos.report.case import Case
    from wellvolpos.ui.numbering import EXPORT_FIGURE_KEYS, UNNUMBERED_EXPORT_KEYS

    case = Case(entry=3500.0, exit=3550.0, mefs=14.0,
                risking_convention="trials_risked", reference="crest",
                scheme="equal_cube_root",
                chance_table={"charge": 0.9, "trap": 0.9, "reservoir": 0.7,
                              "retention": 0.8},
                dataset="A", n_trials=reduced.n_trials)
    bundle = E.assemble(reduced, case, pos=0.7605, pos_source="the trials")
    known = set(EXPORT_FIGURE_KEYS) | set(UNNUMBERED_EXPORT_KEYS)
    built = set(E.build_figure_keys(bundle))
    assert built <= known, sorted(built - known)
    # The cheap key list must match what the builder actually produces, or the
    # cover page's contents list describes a different document.
    import matplotlib.pyplot as plt
    assert built == set(E.build_figures(bundle))
    plt.close("all")


def test_export_file_names_are_the_numbers_the_app_shows():
    from wellvolpos.ui.numbering import export_filename

    assert export_filename("A4_resource_vs_depth") == "2.2_resource_vs_depth"
    assert export_filename("B8_commercial_chance") == "3.10_commercial_chance"
    assert export_filename("C4_wedge") == "6.1_wedge"
    # No legacy code survives in a file name.
    assert not re.match(r"[A-Za-z]\d", export_filename("B12_banded_percentiles"))
    # A key with no legacy prefix keeps its whole descriptive name: partitioning
    # "map_view" blindly once gave "4.8_view".
    assert export_filename("map_view") == "4.8_map_view"
    # Unnumbered entries are left exactly as they are.
    assert export_filename("colour_key") == "colour_key"


def test_the_bundle_is_ordered_by_figure_number():
    from wellvolpos.ui.numbering import EXPORT_FIGURE_KEYS, export_number, export_sort_key

    keys = sorted(list(EXPORT_FIGURE_KEYS) + ["colour_key"], key=export_sort_key)
    assert keys[0] == "colour_key", "the key that explains the rest leads"
    parsed = [tuple(int(x) for x in export_number(k).split(".")) for k in keys[1:]]
    assert parsed == sorted(parsed), parsed
