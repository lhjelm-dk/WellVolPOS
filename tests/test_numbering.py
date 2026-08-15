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
