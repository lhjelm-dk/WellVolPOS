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

import re

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
    "b11": "3.4",
    # b0 (the schematic section) was removed from this tab on 2026-08-14: 4.3
    # draws the same section at the chosen well, from the same A(z), and two
    # copies of one figure on two tabs is one more place for them to disagree.
    # The figure function survives -- 4.3 *is* it -- and keeps its export key.
    "b1": "3.5",
    # 3.6 before 3.7 on Lars's order (2026-08-14): the regret curve reads directly
    # off the volume split above it, and the unproven volume is the follow-on
    # question -- "and how much is under the well I have just described".
    "b2": "3.6",
    "b13": "3.7",
    "b7": "3.8",
    "b8": "3.9",
    "b9": "3.10",
    "b6": "3.11",
    "b12": "3.12",
    # ④ At this well
    "c1": "4.1",
    "c2": "4.2",
    "c3": "4.3",
    "live": "4.4",
    "a6": "4.5",
    "mapview": "4.6",
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
    "b11": "B11", "b12": "B12", "b13": "B13",
    "c1": "C1", "c2": "C2", "c3": "C3", "c4": "C4", "c6": "C6",
}

__all__ = ["FIGURE_NUMBERS", "LEGACY_CODE", "ref", "renumber_title"]


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
    # The separator is matched loosely -- middot, hyphen, dash or colon -- rather
    # than only "·". Rewriting one figure's title with a hyphen was enough to make
    # this prepend instead of replace, and the result was "3.12 · B12 - Resource by
    # ...", which shows the reader the very code the renumbering exists to retire.
    # A leading code is a letter and one or two digits and nothing else; anything
    # longer is a title whose first clause is prose.
    m = re.match(r"\s*([A-Za-z]\d{1,2})\s*[·\-–—:]\s*(.+)", title, re.S)
    if m:
        return f"{number} · {m.group(2)}"
    return f"{number} · {title}"


#: key -> (what it is, the question it answers). The guide's "How to read each
#: figure" table is generated from this together with :data:`FIGURE_NUMBERS`, so the
#: numbers in the guide cannot drift from the numbers on the figures -- which they
#: had, within a day of the renumbering, because the table spelled them out by hand.
#:
#: Order here is the order the table is rendered in, which is tab order.
FIGURE_GUIDE = {
    "a1": ("area–depth curve and reservoir",
           "The structural spine. Everything that splits a trial at the well rests on "
           "this curve. Carries the area uncertainty as P90/P50/mean/P10 and the base "
           "reservoir four times over, because the thickness recovered from pay is a "
           "distribution and one base line implied a surface the trials do not support."),
    "a4": ("resource vs contact depth",
           "Where the volume actually sits with depth, success trials only. Colour is a "
           "trial count on a log scale, because the modal cell holds two orders of "
           "magnitude more trials than the tails — and the tails are where a location "
           "question lives."),
    "a5": ("prospect exceedance, both readings",
           "Solid conditional from 100 %, dashed unconditional from POS_prospect. The "
           "volumes are identical between the two; only the probability attached to them "
           "changes, and the risked one is what a portfolio adds up."),
    "a9": ("prospect resource density",
           "The same distribution {a5} draws as a curve, drawn as a shape. Where the mass "
           "sits, how long the tail is, and how far the mean sits from the P50."),
    "a8": ("contact distribution and P(deeper)",
           "The HCWC distribution read back out of the trials, with the inverse cumulative "
           "beside it. That cumulative *is* r_location at every depth, so {a8} and {a3} must "
           "agree everywhere."),
    "a2": ("outcome tree vs location",
           "What moving the well does to the four outcomes. Risked onto the entered POS, so "
           "it cannot contradict {a3}."),
    "a3": ("chance decomposition vs location",
           "`P_well` and `r_location` as separate curves — the decomposition made "
           "un-mistakable. Never multiplied into one number."),
    "b3": ("uncertainty reduction vs location",
           "Haskett's value-of-information optimum, found by argmax rather than by eye. See "
           "the section below on what it is measuring here, which is *not* an appraisal well."),
    "b11": ("P_well sensitivity to POS_prospect",
            "How much of your answer is the chance table rather than the geometry? Every "
            "curve is P_well against depth for a different POS_prospect. They are all the "
            "same shape scaled vertically, because only r_location moves with depth -- so "
            "revising the chance table and moving the well are independent levers."),
    "b13": ("volume below the reservoir exit",
            "What is left under the well, in the cases where the well left the reservoir "
            "still in hydrocarbons — bold mean, dotted P90/P50/P10. Its own figure because "
            "it is conditional on a *different* event from the three on {b1}, so its curves "
            "were never on the same footing as theirs."),
    "b1": ("volume split vs location",
           "Proven, possible and attic against location, with the proven P90/P50/P10 family "
           "around the mean."),
    "b2": ("chance vs regret",
           "The most decision-relevant plot: where chance stops outweighing what a dry hole "
           "leaves. Four curves, and only `P_well` is unconditional — so the crossing names "
           "the two curves that meet rather than claiming chance equals regret."),
    "b7": ("chance against volume",
           "The trade-off stated directly: moving down-dip buys volume with chance. Depth "
           "appears as labels along the curve. Switch the volume axis to log to read the "
           "*proportional* rate of exchange instead of the absolute one."),
    "b8": ("commercial chance vs location",
           "A rising conditional times a falling `P_well` gives an interior maximum, and that "
           "starred peak is where the well goes commercially."),
    "b9": ("chance-weighted resource vs location",
           "`P_well × the mean`, swept — where the expectation peaks. An expected value "
           "describes no outcome that can happen: the well either finds something near the "
           "success-case mean or it finds nothing. Right for ranking locations, wrong to "
           "quote as a volume."),
    "b6": ("the inverse",
           "Given a volume to prove, where must the well go and what does it cost? Answers a "
           "**guarantee** — the shallowest depth from which the statistic stays at or above "
           "the target all the way down. The grey family is the spread of contacts consistent "
           "with the same volume, on borrowed axes, so where the two families cross means "
           "nothing."),
    "b12": ("resource by contact-depth band",
            "One resource distribution per contact-depth interval, on log-probit axes where a "
            "lognormal is a straight line. Solid is the whole resource in the band, dotted is "
            "what this well would prove in it — so where the dotted curve peels away from the "
            "solid one is where going deeper stops buying proven volume."),
    "c1": ("the structure, with the volume classes",
           "Where each volume sits in the structure at this well. Read as a pair with 4.2."),
    "c2": ("the same volumes as exceedance curves",
           "Two curves per concept in one colour. The risked curves start at their own chance, "
           "so the vertical gap between the top two *is* the location penalty. Values are "
           "labelled on the conditional curves only — risking scales the probability, never "
           "the volume."),
    "c3": ("the chance of clearing MEFS, per volume",
           "The eight numbers 4.2 marks on its MEFS line, given an axis of their own. "
           "Solid is unrisked and hatched is risked; the ratio within a pair is the chance "
           "of the case itself, which differs per row — the up-dip bar is risked by "
           "POS − P_well, not by P_well."),
    "live": ("the section at this well",
             "The closure in section, from A(z), with the three volume classes shaded where "
             "the current entry and exit actually put them. This *is* the schematic section — "
             "it was drawn twice, on two tabs, until 2026-08-14; one copy is one fewer place "
             "for them to disagree."),
    "a6": ("where the four classes overlap",
           "Schneider's *“surprising overlap”*: a dry hole's attic against a discovery's "
           "proven volume, seen against the two larger distributions they are carved out of. "
           "Switch to peak-normalised to compare *shapes* when one class is far narrower."),
    "mapview": ("conceptual map view",
                "The entry contour in plan, and the three areas a well at this depth divides "
                "the closure into. Contours on round absolute depths, not stepped off the "
                "apex, so they do not move when the apex estimate is nudged."),
    "b4": ("chance waterfall",
           "The chance elements then the location factor, as a running product on a log axis. "
           "Totals to `pos_prospect × r` by construction."),
    "b5": ("allocation dumbbell",
           "Which risk elements carry the location penalty. Every scheme gives the same "
           "`P_well` — only the attribution differs."),
}


def ref(text: str) -> str:
    """Replace ``{key}`` placeholders with the figure's current number.

    ``ref("the attic curve on {b2}")`` -> ``"the attic curve on 3.7"``.

    Prose that spells a number out goes stale the moment a figure is inserted above
    it, and that happened the same day the numbering was introduced: adding the
    sensitivity fan as 3.4 pushed five figures down one, and six sentences elsewhere
    kept pointing at the old numbers. Naming the key instead means a renumbering
    fixes the prose along with the figures.
    """
    def sub(m):
        key = m.group(1)
        return FIGURE_NUMBERS.get(key, m.group(0))
    return re.sub(r"\{([a-z][a-z0-9_]*)\}", sub, text)


def guide_table() -> str:
    """The guide's figure table, in markdown, numbered from :data:`FIGURE_NUMBERS`."""
    rows = ["| Figure | The question it answers |", "|---|---|"]
    for key, (what, question) in FIGURE_GUIDE.items():
        number = FIGURE_NUMBERS.get(key, "—")
        # No legacy code in the reader-facing table (Lars, 2026-08-12): "(was A5)"
        # meant nothing to anyone who had not used the app before the renumbering,
        # and it made every row look like it was mid-migration. LEGACY_CODE stays --
        # the export file names and this project's own notes still use the letters --
        # it just is not shown to the reader.
        rows.append(f"| **{number}** {what} | {ref(question)} |")
    return "\n".join(rows)
