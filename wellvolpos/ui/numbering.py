"""Figure numbers by tab: ``2.1``, ``3.4`` and so on, instead of ``A1`` and ``B0``.

Lars asked for this on 2026-08-12, and the reason is navigation: a reader looking
at figure **3.4** knows to find it on tab ③, fourth down. ``B0`` told them nothing
about where it lived, and the letters had stopped meaning anything anyway -- the
A/B/C series came from the design plan's own grouping, which the tab order no
longer follows.

**Only the displayed prefix changes.** The function names stay ``pfig_b0_section``
and ``fig_b0_section``, the export keys stay ``B0_section``, and the notes still
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
    # **B14 renders here**, inside the "where the optimum sits" block and above the
    # frontier, so it takes 3.8 and the four below it shift by one (Lars, 2026-08-18:
    # the numbers must be chronological). Nothing moved on screen -- this is one line,
    # and every prose reference resolves through ref() and follows by itself.
    "b14": "3.8",
    "b7": "3.9",
    "b8": "3.10",
    "b9": "3.11",
    "b6": "3.12",
    "b12": "3.13",
    # ④ At this well
    "c6": "4.1",
    "c5": "4.2",
    "c1": "4.3",
    "c2": "4.4",
    "c3": "4.5",
    "live": "4.6",
    "a6": "4.7",
    "mapview": "4.8",
    # ⑤ Risk & report
    "b4": "5.1",
    "b5": "5.2",

    # ⑥ Theory & guide — the wedge is a schematic and the only figure on that tab.
    "c4": "6.1",
}

#: The letter code each number replaced, so a caption or the guide can still say
#: "B6" where the argument is about B6 rather than about tab ③'s tenth figure.
LEGACY_CODE = {
    "a1": "A1", "a2": "A2", "a3": "A3", "a4": "A4", "a5": "A5", "a6": "A6",
    "a8": "A8", "a9": "A9", "b0": "B0", "b1": "B1", "b2": "B2", "b3": "B3",
    "b4": "B4", "b5": "B5", "b6": "B6", "b7": "B7", "b8": "B8", "b9": "B9",
    "b11": "B11", "b12": "B12", "b13": "B13", "b14": "B14",
    "c1": "C1", "c2": "C2", "c3": "C3", "c4": "C4", "c5": "C5", "c6": "C6",
}

__all__ = ["FIGURE_NUMBERS", "LEGACY_CODE", "ref", "renumber_title"]


#: export-bundle key -> chart key, so an exported figure can find its number.
#:
#: **Explicit, not derived from the prefix** (Lars, 2026-08-18: *"the figures carry
#: names like A4, B2, C4 ... I want the 3.12, 2.2, 4.2 kind of numbering"*). Splitting
#: ``"A4_resource_vs_depth"`` on the underscore gets 27 of the 30 right and silently
#: wrong on the three that matter: ``B0_section`` is the *live section*, which the app
#: charts under the key ``live`` as 4.6, and ``map_view`` is ``mapview``. A rule that is
#: right most of the time is how a figure ends up numbered as a different figure.
EXPORT_FIGURE_KEYS = {
    "A1_area_depth": "a1",
    "A4_resource_vs_depth": "a4",
    "A5_exceedance": "a5",
    "A9_prospect_density": "a9",
    "A8_contact_distribution": "a8",
    "A2_outcome_tree": "a2",
    "A3_chance_decomposition": "a3",
    "B3_uncertainty_reduction": "b3",
    "B11_pos_sensitivity": "b11",
    "B1_volume_split": "b1",
    "B2_chance_vs_regret": "b2",
    "B13_below_exit": "b13",
    "B14_hurdle_cost": "b14",
    "B7_frontier": "b7",
    "B8_commercial_chance": "b8",
    "B9_chance_weighted": "b9",
    "B6_inverse": "b6",
    "B12_banded_percentiles": "b12",
    "C6_outcome_tree": "c6",
    "C5_partitions": "c5",
    "C1_section": "c1",
    "C2_exceedance": "c2",
    "C3_mefs_bars": "c3",
    # The same function the app draws as 4.6 "Live section", called again for export.
    "B0_section": "live",
    "A6_overlap": "a6",
    "map_view": "mapview",
    "B4_chance_waterfall": "b4",
    "B5_allocation_dumbbell": "b5",
    "C4_wedge": "c4",
}

#: Bundle entries that are deliberately unnumbered. The colour key is a *legend*, not
#: a figure -- it has no number on screen either, and giving it one in the report would
#: invent a figure the app does not have.
UNNUMBERED_EXPORT_KEYS = frozenset({"colour_key"})


def export_number(export_key: str) -> str | None:
    """The displayed figure number for a bundle entry, or ``None`` if it has none."""
    chart_key = EXPORT_FIGURE_KEYS.get(export_key)
    return FIGURE_NUMBERS.get(chart_key) if chart_key else None


def export_sort_key(export_key: str) -> tuple[int, int, str]:
    """Order a bundle by figure number, so the PDF reads in the app's order.

    Unnumbered entries lead: the colour key is the one page that explains every other,
    so a reader meeting the archive first meets it.
    """
    number = export_number(export_key)
    if not number:
        return (-1, -1, export_key)
    tab, _, within = number.partition(".")
    return (int(tab), int(within), export_key)


def export_filename(export_key: str) -> str:
    """The name a figure file carries in the archive.

    ``"A4_resource_vs_depth"`` -> ``"2.2_resource_vs_depth"``. The *descriptive* half
    is kept: the number says where the figure sits and the words say what it is, and a
    file called ``3.10.png`` tells a reader nothing once it is on their desktop.

    **The number is exact, not zero-padded.** ``3.10`` therefore sorts before ``3.2``
    in a naive alphabetical listing, and that is the deliberate trade: the whole point
    is that the file name matches what is on screen. The archive is written in report
    order and carries a contents list in ``README.txt``, which is the fix for ordering
    that does not require lying about the number.
    """
    number = export_number(export_key)
    if not number:
        return export_key
    # Strip the prefix only where it *is* a legacy code. ``map_view`` has none, and
    # partitioning it blindly gave "4.8_view" -- the descriptive half silently halved.
    code, _, rest = export_key.partition("_")
    if rest and re.fullmatch(r"[A-Za-z]\d{1,2}", code):
        return f"{number}_{rest}"
    return f"{number}_{export_key}"


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
    "b9": ("chance-weighted resource, and its risk-adjusted twin",
           "`P_well × the mean`, swept — where the expectation peaks. An expected value "
           "describes no outcome that can happen: the well either finds something near the "
           "success-case mean or it finds nothing. Right for ranking locations, wrong to "
           "quote as a volume. "
           "**The dashed green curve is the same distribution risk-adjusted** — "
           "exponential utility at the risk tolerance set above the panel. An "
           "expectation is risk-neutral; this is not, and the gap between them widens "
           "down-dip, because that is where the low-chance, high-volume tail lives."),
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
    "b14": ("what the commerciality hurdle costs",
            "Sweeps the *requirement* rather than the depth: x is the confidence you "
            "insist on, y is what it buys. The counterintuitive part is that Pc falls "
            "as the hurdle tightens — more confidence pushes the well deeper, and "
            "deeper costs chance faster than it buys commerciality. Labels on the "
            "upper curve are the entry depth each hurdle requires."),
    "c6": ("what happens if this well is drilled",
           "The four outcomes as one bar: chance failure, dry with attic, and a discovery "
           "split at MEFS into sub-commercial and commercial. Every share is risked onto "
           "the POS in use, so the leaves partition to 100 % and the discovery branch is "
           "P_well by construction."),
    "c5": ("two cuts of one closure",
           "Rose partitions at the well; this app partitions at the interval the well "
           "penetrates. Same closure twice, and the violet band is the slice they "
           "disagree about — Rose counts it below his cut, this app above. Both sum to "
           "the well-associated volume."),
    "c4": ("the wedge — why pay is less than reservoir thickness",
           "Schematic. A dipping layer of constant true vertical thickness under a flat "
           "contact gives a charged interval at full thickness up-dip, pinching to zero "
           "where the top surface meets the contact. The two rules on the right are "
           "area-averaged pay against the thickness; the gap between them is the point."),
    "c1": ("the structure, with the volume classes",
           "Where each volume sits in the structure at this well. Read as a pair with {c2}."),
    "c2": ("the same volumes as exceedance curves",
           "Two curves per concept in one colour. The risked curves start at their own chance, "
           "so the vertical gap between the top two *is* the location penalty. Values are "
           "labelled on the conditional curves only — risking scales the probability, never "
           "the volume."),
    "c3": ("the chance of clearing MEFS, per volume",
           "The eight numbers {c2} marks on its MEFS line, given an axis of their own. "
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
    "b5": ("how the location penalty is shared out",
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


def _number_key(key: str) -> tuple[int, int]:
    """Sort key from a figure number, so 3.10 follows 3.9 rather than 3.1.

    Anything unmapped sorts last rather than raising: a figure with no number is a
    defect ``test_every_figure_the_app_charts_has_a_number`` already catches, and it
    should fail there rather than by crashing the guide.
    """
    number = FIGURE_NUMBERS.get(key)
    if not number or "." not in number:
        return (99, 99)
    tab, _, within = number.partition(".")
    return (int(tab), int(within))


def guide_table() -> str:
    """The guide's figure table, in markdown, numbered from :data:`FIGURE_NUMBERS`.

    **Sorted by number, not by dict order** (Lars, 2026-08-18). ``FIGURE_GUIDE`` is
    keyed by figure *letter* and was written in the order the figures were built, so
    the table listed 3.7 before 3.5, 3.8 after 3.13, and 6.1 in the middle of tab ④.
    A reader scanning for a number they have just seen on screen has no reason to
    expect any of that.
    """
    rows = ["| Figure | The question it answers |", "|---|---|"]
    for key in sorted(FIGURE_GUIDE, key=_number_key):
        what, question = FIGURE_GUIDE[key]
        number = FIGURE_NUMBERS.get(key, "—")
        # No legacy code in the reader-facing table (Lars, 2026-08-12): "(was A5)"
        # meant nothing to anyone who had not used the app before the renumbering,
        # and it made every row look like it was mid-migration. LEGACY_CODE stays --
        # the export file names and this project's own notes still use the letters --
        # it just is not shown to the reader.
        rows.append(f"| **{number}** {what} | {ref(question)} |")
    return "\n".join(rows)
