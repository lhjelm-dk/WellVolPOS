"""A ceiling on the always-visible caption text, per tab.

Tab ③ carried 2,082 words of caption under twelve figures — around seventeen minutes
of reading to choose a depth. Every paragraph was justified on its own, which is
exactly how it got there and exactly why a ceiling is the only thing that holds: the
next addition will be justified too.

The three-tier arrangement moved the argument behind ``st.expander`` rather than
deleting it (``ui.common.figure_note``), so what this measures is what a reader has to
read *before* deciding they want more.

**Raising a budget is a decision, not a fix.** If a change needs more words on screen,
move something into the expander or into tab ⑥ first, and only raise the number when
the words genuinely belong in front of the reader.
"""

import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "wellvolpos" / "ui"

#: Tab -> the most always-visible caption words it may carry.
#:
#: ``tab3`` is the tiered one and its budget is real: 344 words today against a ceiling
#: of 600, which is room for a few more figures at tier-1 length.
#:
#: ``tab4`` was tiered the same day, 1,388 -> 513.
#:
#: ``tab2`` is **provisional** — the last still to be tiered, and its number is today's
#: count with a little headroom so the test locks the current state in rather than
#: blessing it. Bring it to 600 when it is done.
BUDGET = {
    "tab2_prospect": 1_100,
    "tab3_where": 600,
    "tab4_well": 700,
    "tab5_report": 400,
}


def visible_caption_words(path: Path) -> int:
    """Words a reader sees without opening anything.

    Counts ``st.caption(...)`` in full and only the *headline* of
    ``figure_note(headline, detail=...)`` — the detail is behind an expander, which is
    the whole point of the arrangement.
    """
    s = path.read_text(encoding="utf-8")
    total = 0
    for m in re.finditer(r"(st\.caption|figure_note)\(", s):
        i = m.end()
        depth, j = 1, i
        while depth:
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
            j += 1
        body = s[i:j - 1]
        if m.group(1) == "figure_note":
            body = body.split("detail=")[0]
        total += sum(len(t.split()) for t in re.findall(r'"([^"]{4,})"', body))
    return total


@pytest.mark.parametrize("stem,ceiling", sorted(BUDGET.items()))
def test_a_tab_stays_within_its_caption_budget(stem, ceiling):
    words = visible_caption_words(UI / f"{stem}.py")
    assert words <= ceiling, (
        f"{stem} shows {words:,} caption words, over its {ceiling:,} budget. "
        f"Move the argument into a figure_note detail or into tab ⑥ rather than "
        f"raising the ceiling."
    )


@pytest.mark.parametrize("stem", ["tab3_where", "tab4_well"])
def test_the_tiered_tabs_actually_came_down(stem):
    """If either regresses, the tiering has been undone rather than extended."""
    assert visible_caption_words(UI / f"{stem}.py") < 700


def test_every_long_caption_on_a_tiered_tab_is_behind_an_expander():
    """A tier-1 line is one or two sentences. Anything longer belongs in the detail."""
    s = "\n".join((UI / f"{stem}.py").read_text(encoding="utf-8")
                  for stem in ("tab3_where", "tab4_well"))
    for m in re.finditer(r"figure_note\(", s):
        i = m.end()
        depth, j = 1, i
        while depth:
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
            j += 1
        head = s[i:j - 1].split("detail=")[0]
        words = sum(len(t.split()) for t in re.findall(r'"([^"]{4,})"', head))
        assert words <= 60, f"a tier-1 headline runs to {words} words:\n{head[:160]}"
