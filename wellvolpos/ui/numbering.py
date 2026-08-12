"""Figure numbers by tab: ``2.1``, ``3.4`` and so on, instead of ``A1`` and ``B0``.

Lars asked for this on 2026-08-12, and the reason is navigation: a reader looking
at figure **3.4** knows to find it on tab ③, fourth down. ``B0`` told them nothing
about where it lived, and the letters had stopped meaning anything anyway -- the
A/B/C series came from the design plan's own grouping, which the tab order no
longer follows.

**Only the displayed prefix changes.** The function names stay ``pfig_b0_section``
and ``fig_b0_section``, the export keys stay ``B0_section``, and CLAUDE.md still
argues about B6 and C2 by letter. Renaming those would churn every test, every
docstring and every commit message that discusses a figure, for no gain: the
letter is the figure's *identity*, the number is where it sits today. If a figure
moves tabs, one line here changes and nothing else does.

Applied by :func:`wellvolpos.ui.common.chart`, which already takes the key -- so no
figure function knows or cares about its number, and there is exactly one place
where the mapping lives.

Tab ① has no figures (it is data, QC and the import mapping), so the numbering
starts at 2.1.
"""

from __future__ import annotations

#: chart key -> displayed figure number. Order within a tab is render order, which
#: is the order a reader meets them scrolling down.
FIGURE_NUMBERS = {
    # ② Prospect
    "a1": "2.1",
    "a4": "2.2",
    "a5": "2.3",
    "a9": "2.4",
    "a8": "2.5",
    # ③ Where to drill
    "a2": "3.1",
    "a3": "3.2",
    "b3": "3.3",
    "b0": "3.4",
    "b1": "3.5",
    "b2": "3.6",
    "b7": "3.7",
    "b8": "3.8",
    "b9": "3.9",
    "b6": "3.10",
    # ④ At this well
    "c1": "4.1",
    "c2": "4.2",
    "live": "4.3",
    "a6": "4.4",
    "mapview": "4.5",
    # ⑤ Risk & report
    "b4": "5.1",
    "b5": "5.2",
}

#: The letter code each number replaced, so a caption or the guide can still say
#: "B6" where the argument is about B6 rather than about tab ③'s tenth figure.
LEGACY_CODE = {
    "a1": "A1", "a2": "A2", "a3": "A3", "a4": "A4", "a5": "A5", "a6": "A6",
    "a8": "A8", "a9": "A9", "b0": "B0", "b1": "B1", "b2": "B2", "b3": "B3",
    "b4": "B4", "b5": "B5", "b6": "B6", "b7": "B7", "b8": "B8", "b9": "B9",
    "c1": "C1", "c2": "C2",
}

__all__ = ["FIGURE_NUMBERS", "LEGACY_CODE", "renumber_title"]


def renumber_title(title: str, key: str) -> str:
    """Swap a figure title's leading letter code for its tab-relative number.

    ``"B0 · Schematic section"`` -> ``"3.4 · Schematic section"``. The descriptive
    half is kept verbatim, which is the point: the name is what the figure *is* and
    only the index is being renumbered.

    A title that never had a code gets the number **prepended** instead. That is not
    a nicety: the live section and the map view had no index at all, so a reader
    could not refer to them, which is how "the one next to A6" became the only way
    to say which figure was meant.

    A key with no number assigned comes back untouched -- a figure the mapping has
    not been told about should look wrong in an obvious way rather than silently
    lose its identity.
    """
    number = FIGURE_NUMBERS.get(key)
    if not number or not title:
        return title
    head, sep, rest = title.partition("·")
    if sep:
        # Replace a leading code only if it looks like one: a letter then digits and
        # nothing else. Anything longer is a title whose first clause is prose.
        stripped = head.strip()
        if 2 <= len(stripped) <= 3 and stripped[0].isalpha() and stripped[1:].isdigit():
            return f"{number} {sep}{rest}"
    return f"{number} · {title}"
