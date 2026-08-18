"""Tab ② — the prospect, before any well.

**Well-free on purpose**, and enforced by argument rather than convention:
A1 is drawn with ``show_classes=False`` and no entry/exit, A4 without the
well rules, A8 without the entry marker. The same figures appear on tab ④
with the well. If a well marker ever reappears here, the tab has lost its
subject.

It also **owns the eight chance widgets and the risking convention**, which
are ``POS_prospect`` -- a property of the prospect, so they sit above the
distributions they risk. ``app.py`` reads their values out of session state
before this renders; see :class:`~wellvolpos.ui.context.Ctx`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..core import (
    check_column_heights,
    ELEMENT_LABELS,
    SUMMARY_COLUMNS,
    risk_summary,
    ELEMENTS,
    class_percentiles,
    group_summary,
)
from ..viz import (
    element_colour,
    pfig_a1_area_depth,
    pfig_a4_resource_vs_depth,
    pfig_a5_exceedance,
    pfig_a8_contact_distribution,
    pfig_a9_prospect_density,
    row_zlim,
    suggest_grid,
)
from .conventions import (
    CHANCE_DEFAULTS,
    CHANCE_HELP,
    CONVENTION_KEYS,
    CONVENTION_LABELS,
    PLAY_DEFAULTS,
    PLAY_HELP,
)
from .common import (chart as _chart, element_chip, figure_note,
                     kpi_ladder, split_caveat)
from .context import Ctx
from .numbering import ref as fig_ref


def render(ctx: Ctx) -> None:
    ts, ad, has_area = ctx.ts, ctx.ad, ctx.has_area
    groups, vc, chance = ctx.groups, ctx.vc, ctx.chance
    entry, exit_, mefs = ctx.entry, ctx.exit_, ctx.mefs
    ref, scheme, area_scale = ctx.ref, ctx.scheme, ctx.area_scale
    pos, pos_source, pos_from_table = ctx.pos, ctx.pos_source, ctx.pos_from_table
    pos_trials, risking_convention = ctx.pos_trials, ctx.risking_convention
    elements, play_elements, play_chance = ctx.elements, ctx.play_elements, ctx.play_chance
    qc, gap = ctx.qc, ctx.gap
    source, overrides = ctx.source, ctx.overrides

    def _split_caveat() -> None:
        split_caveat(ctx)

    st.subheader("Chance table — the geological risk elements")
    st.caption(
        "**Inputs, not results.** Per-element chance of success, in two levels, multiplied "
        "together to give POS_prospect. Nothing here depends on where the well goes: the "
        "location factor is computed from the trials, and the two are brought together in the "
        "risk summary in tab ⑤."
    )
    st.markdown("**The prospect, given the play works**")
    # The element colours from 5.1 and 5.2, on a chip above each input, so a number
    # typed here can be found again on those figures without counting rows. The name
    # is written on the chip because the tint alone does not survive tritanopia --
    # see `common.element_chip`.
    ec = st.columns(4)
    for i, el in enumerate(ELEMENTS):
        ec[i].markdown(element_chip(el), unsafe_allow_html=True)
        ec[i].number_input(
            ELEMENT_LABELS[el], 0.01, 1.0, CHANCE_DEFAULTS[el], 0.01,
            key=f"w_chance_{el}",
            help=CHANCE_HELP[el],
            label_visibility="collapsed",
        )
    st.markdown("**The play — the same four elements, one level up**")
    st.caption(
        "*Is there a working petroleum system here at all?* Assessed element by element rather "
        "than as one number (Lars, 2026-08-11), because that question and *does this closure "
        "have a seal* draw on different evidence, and a single play number cannot be argued "
        "about element by element the way a column can. The four above are read as **conditional "
        "on the play working**."
    )
    pc = st.columns(4)
    for i, el in enumerate(ELEMENTS):
        pc[i].markdown(element_chip(el), unsafe_allow_html=True)
        pc[i].number_input(
            f"{ELEMENT_LABELS[el]} (play)", 0.01, 1.0, PLAY_DEFAULTS[el], 0.01,
            key=f"w_play_{el}", help=PLAY_HELP[el],
            label_visibility="collapsed",
        )
    _cond = float(np.prod(list(elements.values())))

    # **The eight inputs, totalled where they are entered** (Lars, 2026-08-18). The
    # summary in tab ⑤ was the only place these multiplied out, three tabs from the
    # controls that set them -- so revising a chance meant leaving the tab to see what
    # it did. This is the same table with **the well column removed**: `r_location`
    # needs a well and a well belongs to tab ③, and a column of ones under a heading
    # about location teaches the wrong thing.
    #
    # Built from :func:`risk_summary` with ``r = 1`` rather than by multiplying here,
    # so the two tables cannot drift: one function owns the arithmetic and this one
    # drops a column from it.
    _prospect_only = risk_summary(elements, 1.0, scheme="none",
                                  play_elements=play_elements)
    _rows = pd.DataFrame(_prospect_only.as_records()).drop(
        columns=[SUMMARY_COLUMNS[2], "Carries the location penalty"])
    _keys = _prospect_only.element_keys()

    def _tint(_col):
        return [f"background-color: {element_colour(k, tint=True)}" if k else ""
                for k in _keys]

    sc1, sc2 = st.columns([3, 2])
    with sc1:
        st.dataframe(
            _rows.style.apply(_tint, subset=["Chance element"]),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="percent")
                           for c in SUMMARY_COLUMNS[:2]},
        )
    with sc2:
        st.dataframe(
            pd.DataFrame([r for r in _prospect_only.result_records()
                          if r["result"] != "Well location POS"]),
            hide_index=True, width="stretch",
            column_config={"value": st.column_config.NumberColumn(format="percent")},
        )
        st.caption(
            fig_ref("The well's own chance is not here: it needs `r_location`, which "
                    "needs a well. Tab ③ sweeps it and tab ⑤ multiplies it in."))

    st.caption(
        f"**POS_prospect = play × prospect-given-play = {play_chance:.4f} × {_cond:.4f} = "
        f"{play_chance * _cond:.4f}** — the product of all eight inputs above."
    )

    # The convention: what the zero-volume trials mean, and therefore where POS comes
    # from. The *evidence* for the decision stays in tab ①'s QC report, because
    # whether those trials are zero is a fact about the file; this is the judgement.
    _f = qc.failure
    if _f and _f.has_failures:
        st.markdown("**Risking convention — where POS_prospect comes from**")
        st.caption(
            f"The detector found: {_f.summary()} The evidence is in tab ① under *Zero-volume "
            f"trials*. Reading them as chance failures makes the trial file itself carry the "
            f"chance; reading the export as success-case only makes the table above carry it."
        )
        # Branch on a stable key, never on the label text: the label is user copy and
        # rewording it must not be able to change which POS the whole app uses. The
        # radio *owns* ``risking_convention``, so a loaded case that writes that key
        # sets the control rather than fighting it. The key is seeded further up,
        # where POS is read -- see the comment there.
        conv = st.radio(
            "How should the zero-volume trials be read?",
            CONVENTION_KEYS,
            format_func=lambda k: CONVENTION_LABELS[k],
            key="risking_convention",
        )
        if conv == "geometric":
            st.info(
                "**To be implemented in a later update.** The principle is in Lowry, Suttill & "
                "Taylor (2005), *Advances in risking exploration prospects*, APPEA Journal 45(1) "
                "179–188, [doi:10.1071/AJ04012](https://doi.org/10.1071/AJ04012): the geological "
                "chance factors and the *geometric* chance that a given location is within the "
                "accumulation are separate things, and conflating them double-counts. Until it is "
                "built this option behaves as *success-case only*, and the caveat below says why "
                "that is not the same answer."
            )
            st.warning(
                "Not yet implemented. Reading the zeros as geometric means they are *charged* "
                "trials with no hydrocarbon column above the crest, so they belong in the "
                "denominator of r_location — which conditions on `resource > 0` and therefore "
                "currently drops them. Until that is built, this option behaves like "
                "'success-case only' and r_location is not what this reading requires."
            )
    st.caption(f"**Effective POS_prospect: {pos:.4f}**, from {pos_source}.")

    st.divider()
    st.subheader("Prospect — defined from the trials")
    res_all = ts.col("resource")
    s = group_summary(ts, groups)["prospect_success"]
    # **One kind of mean on this tab** (Lars, 2026-08-12). This strip used to be taken
    # over *every* trial while the table under 2.3 was taken over the success cases,
    # so the same tab showed a "Mean" of 10.31 and a "Pmean" of 13.56 one screen
    # apart. Everything here is now the **success case**, which is what the solid
    # curve on 2.3 draws and what its table reports; the risked reading is stated
    # once, below, as the single number it is.
    _succ = res_all[res_all > 0.0]
    p99 = float(np.percentile(_succ, 1.0)) if _succ.size else float("nan")
    p1 = float(np.percentile(_succ, 99.0)) if _succ.size else float("nan")
    # One shape, shared with tab ④'s well-associated row, so the prospect and the
    # well can be read one above the other -- see ui.common.kpi_ladder.
    kpi_ladder(
        chance_label="POS prospect", chance=chance.pos_prospect,
        values={"p99": p99, "p90": s["p90"], "p50": s["p50"],
                "mean": s["mean"], "p10": s["p10"], "p1": p1},
        chance_help="The chance the PROSPECT holds hydrocarbons, before any well is "
                    "placed. Unconditional; the six beside it are not.",
        value_help="MMboe, success case — conditional on the prospect working.",
    )
    _n_zero = int(np.count_nonzero(res_all <= 0.0))
    _f_zero = _n_zero / res_all.size if res_all.size else 0.0
    _risked = s["mean"] * chance.pos_prospect
    _zero_note = (
        f"This file carries **{_f_zero:.1%}** zero-volume trials — chance failures, which "
        f"belong to POS and not to the shape of the distribution, so they are excluded here."
        if _n_zero else
        "This file carries **no** zero-volume trials, so the success case is the whole file "
        "and the geological chance is entirely in the chance table above."
    )
    st.caption(
        f"MMboe, **success case** — conditional on the prospect working, which is where "
        f"percentiles live: P99 is the low end, exceeded 99 % of the time. {_zero_note} "
        f"The **risked** (unconditional) mean is the one number that folds the chance in: "
        f"**{s['mean']:.2f} × {chance.pos_prospect:.1%} = {_risked:.2f} MMboe**, which is what a "
        f"portfolio adds up and what no single outcome ever equals."
    )

    succ_contact = ts.col("contact")[res_all > 0.0]

    # ---------------------------------------------- apex and minimum column height
    # Asked for on 2026-08-12: show the apex and the minimum hydrocarbon column the
    # trials imply, so the reader can inspect the extrapolation rather than inherit
    # it silently. Decision 6 makes the apex derived, never entered -- it is A(z)'s
    # shallow tail run out to zero area, and the trials do not contain the crest, so
    # every column height quoted anywhere in this app carries this error.
    if has_area:
        _apex = float(ad.apex_estimate())
        _shallowest = float(np.min(succ_contact)) if succ_contact.size else float("nan")
        _deepest = float(np.max(succ_contact)) if succ_contact.size else float("nan")
        _min_col = _shallowest - _apex
        _max_col = _deepest - _apex
        with st.expander(
            f"Apex and column height — minimum {_min_col:,.0f} m of hydrocarbon column "
            f"in this data set", expanded=False
        ):
            ac = st.columns(4)
            ac[0].metric("Apex (derived)", f"{_apex:,.0f} m",
                         help="Where A(z) extrapolates to zero productive area. Never an "
                              "input — one apex per session, from one source (decision 6).")
            ac[1].metric("Shallowest sampled contact", f"{_shallowest:,.0f} m",
                         help="The shallowest hydrocarbon–water contact any successful "
                              "trial produced.")
            ac[2].metric("Minimum column height", f"{_min_col:,.0f} m",
                         help="Shallowest contact − apex. The thinnest accumulation the "
                              "trials actually contain.")
            ac[3].metric("Maximum column height", f"{_max_col:,.0f} m",
                         help="Deepest contact − apex, at the deepest trial.")
            _gap = _shallowest - _apex
            st.warning(
                f"**The apex is an extrapolation, not a mapped depth.** The trials stop at "
                f"{_shallowest:,.0f} m, so the top {_gap:,.0f} m of this closure is inferred "
                f"from the shape of A(z) alone and no trial constrains it. Any minimum column "
                f"height below **{_min_col:,.0f} m** is therefore outside the modelled range: "
                f"it would not exclude a single trial, because there are none up there to "
                f"exclude. Treat a column-height cut as binding only above that figure."
            )
            st.caption(
                "This is the one number every column-height statement in the app inherits — "
                "the minimum-flowable mapping, the map view's contours and the section's "
                "crest all measure from this apex. It is shown here so the size of the "
                "extrapolation is visible before any of them is quoted."
            )

            st.divider()
            st.markdown("**Is any trial too thin to flow?**")
            # The check Lars asked for on 2026-08-13. Two questions, and only the first
            # is a defect: a success trial at or above the apex has a positive volume
            # and no column at once, which is a contradiction rather than a thin
            # accumulation. The second is a what-if, and its treatment is settled --
            # **sub-minimum trials lower POS**, because an accumulation too thin to flow
            # is a failed well and belongs in the denominator, not a smaller success.
            _mch = st.number_input(
                "Minimum column height to test (m), 0 = none", 0.0, 500.0, 0.0, 5.0,
                key="w_min_column",
                help="Reported, never applied: nothing in the app filters on it. It is "
                     "here so the cost of a minimum can be argued about with the count "
                     "in front of you.",
            )
            _cc = check_column_heights(ts, _apex, _mch if _mch > 0 else None)
            if _cc.contradicts:
                st.error(_cc.message())
            elif _cc.binds:
                st.warning(_cc.message())
            else:
                st.success(_cc.message())
            figure_note(
                "Every successful trial sits below the derived apex, as it must — a contact "
            "above the crest would be hydrocarbon outside the closure.",
                detail="**A trial with its contact at or above the apex would be a contradiction** "
                "— positive volume, no column — and would mean either the apex "
                "extrapolation has overshot or the export is inconsistent. Ideally there "
                "are none, and on both demo files there are none. If a minimum column "
                "height is set, sub-minimum trials are counted as **chance failures**, so "
                "the cut lowers POS rather than renormalising what is left: too thin to "
                "flow is a failed well, not a smaller discovery.",
            )

    if not has_area:
        st.warning(
            f"No productive-area column in this export — the map view, {fig_ref('{a1}')}, "
            f"{fig_ref('{a4}')} and {fig_ref('{a5}')} need it "
            "and are skipped."
        )
    else:
        st.divider()
        # One depth range for the row, so A1 and A4 can be read straight across
        # at constant depth (non-negotiable 2). The map view is in plan view and
        # A5 has no depth axis, so neither joins the alignment.

        zrow_prospect = row_zlim(
            (ad.shallowest, ad.deepest),
            (float(succ_contact.min()), float(succ_contact.max())),
            pad_frac=0.02,
        )
        # A1 and A4 **stacked, full width** (Lars, 2026-08-11). They were side by side
        # in two columns, which halved both: A1's four base-reservoir curves crowded
        # into one another and A4's grid cells stopped being distinguishable. They
        # still share one depth range, so the reading across still works -- it is now
        # top-to-bottom at constant depth rather than left-to-right, which costs
        # nothing because neither figure needs the other's x.
        #
        # No well on the prospect tab: A1 here is A(z) and the reservoir band, both
        # properties of the prospect. The entry/exit rules and the three shaded
        # volume classes are a *well* result and appear on tab ④, where the same
        # figure is drawn again with them.
        _chart(pfig_a1_area_depth(
                ad, ts=ts, zlim=zrow_prospect, area_scale=area_scale,
                show_classes=False,
            ), key="a1")

        st.divider()
        _auto_r, _auto_z = suggest_grid(res_all[res_all > 0.0], succ_contact)
        ac1, ac2, ac3 = st.columns([2, 1, 1])
        a4_render = ac1.radio(
            "Rendering", ["grid", "hexbin"], horizontal=True, key="w_a4_render",
            format_func=lambda k: {"grid": "Trial-count grid",
                                   "hexbin": "Log-density hexbin"}[k],
        )
        a4_nx = ac2.number_input("Resource bins", 10, 120, _auto_r, 2, key="w_grid_res",
                                 disabled=a4_render != "grid")
        a4_ny = ac3.number_input("Depth bins", 10, 120, _auto_z, 2, key="w_grid_z",
                                 disabled=a4_render != "grid")
        # Depth labels back on: stacked, A4 is no longer the second panel of a row
        # borrowing the first's tick labels -- it is a figure in its own right and
        # needs its own.
        _chart(pfig_a4_resource_vs_depth(
                ts, mefs=mefs,
                render=a4_render, n_resource=int(a4_nx), n_depth=int(a4_ny),
                zlim=zrow_prospect,
            ), key="a4")
        figure_note(
            fig_ref("**{a1}** — the closure in section: top reservoir from A(z), the base a thickness below it, and the area uncertainty around both."),
            detail=f"**{fig_ref('{a1}')}** now carries the reservoir too: top reservoir is A(z), and the base is that "
            f"curve shifted down by the thickness back-calculated from pay — drawn four times, "
            f"P90/P50/mean/P10, because that thickness is a distribution and one base line "
            f"implied a surface the trials do not support. The three shaded classes are the same "
            f"colours {fig_ref('{c2}')} uses below. **{fig_ref('{a4}')}**'s grid default is "
            f"{_auto_r} × {_auto_z} from Freedman–Diaconis per axis; counts are on a log scale "
            f"either way, because the modal cell holds two orders of magnitude more trials than "
            f"the tails and the tails are where a location question lives.\n\n"
            f"{fig_ref('{a1}')} and {fig_ref('{a4}')} are stacked rather than side by side, and "
            f"still share one depth range "
            f"({zrow_prospect[0]:.0f}–{zrow_prospect[1]:.0f} m TVDSS) — so they read straight "
            f"**down** at constant depth instead of across, which costs nothing because neither "
            f"needs the other's x-axis and buys both of them full width. Both draw the mean thick "
            f"and the P90/P50/P10 family "
            f"thin and grey — the mean is the number that gets quoted, and on a skewed distribution "
            f"it is not the P50. {fig_ref('{a4}')} uses success trials only: the chance-failure "
            f"zeros belong to "
            f"POS, not to the shape of the resource distribution.",
        )

        st.divider()
        # POS_prospect, not P_well: the risked curve here is the *prospect's*, which
        # is a property of the prospect and belongs on this tab. It stopped being
        # drawn when the tab went well-free, because the chance argument was
        # stripped along with the well ones -- and without a chance the figure
        # declines to invent an unconditional curve rather than guessing one.
        _chart(pfig_a5_exceedance(
                ts, groups, vc, mefs=mefs, pos_prospect=chance.pos_prospect,
            ), key="a5")
        figure_note(
            fig_ref("**{a5}** — the prospect's resource. Solid is the success case and starts at 100 %; dashed folds the chance in and starts at it."),
            detail=f"**{fig_ref('{a5}')} — the prospect's resource, both readings.** The **solid** curve is "
            f"*conditional*: the success case, given the prospect works. It starts at 100 % and "
            f"it is where the percentiles live — that is what anyone means by \"the P50\". The "
            f"**dashed** curve is *unconditional* (risked): the same volumes with POS_prospect "
            f"folded in, so it starts at **{chance.pos_prospect:.0%}** instead.\n\n"
            f"The volumes are identical between the two — only the probability attached to them "
            f"changes, and the risked one is what a portfolio adds up. No depth axis, so this "
            f"sits below the row rather than in it.",
        )

        # One wide row rather than six long ones (Lars, 2026-08-12). The long form
        # gave every cell a label, but it spent six rows and two probability columns
        # saying what the percentile names already say -- an unrisked P90 reads 90 %
        # by definition, and the risked one is that times a chance printed in the
        # same table. So the chance is stated once and the rest is the volume ladder,
        # which is the thing a reader actually copies out.
        st.markdown(fig_ref("**The numbers behind {a5}**"))
        _a5_stats = class_percentiles(res_all[res_all > 0], chance.pos_prospect)
        _row = {
            "case": "Prospect recoverable resource",
            "prospect POS": chance.pos_prospect,
            "P99": _a5_stats["p99"],
            "P90": _a5_stats["p90"],
            "P50": _a5_stats["p50"],
            "Pmean": _a5_stats["mean"],
            "P10": _a5_stats["p10"],
            "P1": _a5_stats["p1"],
        }
        _vol_cols = {
            c: st.column_config.NumberColumn(f"{c} (MMboe)", format="%.2f")
            for c in ("P99", "P90", "P50", "Pmean", "P10", "P1")
        }
        st.dataframe(
            pd.DataFrame([_row]), hide_index=True, width="stretch",
            column_config={
                "prospect POS": st.column_config.NumberColumn(format="percent"),
                **_vol_cols,
            },
        )
        figure_note(
            "Conditional volumes — the success case, given the prospect works. That is "
        "where percentiles live; the chance is applied once, separately.",
            detail="**Volumes are conditional** — the success case, given the prospect works. That is "
            "where percentiles live and it is the distribution the industry quotes: an unrisked "
            "P90 is exceeded 90 % of the time *by definition*. To get the unconditional "
            f"(risked) probability of any of them, multiply by the prospect POS beside it — "
            f"**{chance.pos_prospect:.1%}** — which is what the dashed curve above already does. "
            "The volumes themselves do not change between the two readings; only the probability "
            "attached to them does."
            "\n\n"
            "**Pmean is not a percentile.** It is the arithmetic mean, and on this distribution it "
            f"is exceeded {_a5_stats['mean_at']:.0f} % of the time rather than 50 % — above the "
            "P50, as it must be on a right-skewed distribution. It sits at the end of the ladder "
            "rather than between P50 and P10 so that it cannot be read as one of them.",
        )

        st.divider()
        # **The exceedance curve, offered here rather than assumed** (Lars,
        # 2026-08-18). It is the same trials as the bars, integrated from the right,
        # so putting a volume against its chance no longer means moving back up the
        # tab. Off by default: a density and a cumulative answer different questions,
        # and this figure's own subject is the shape.
        _a9_exc = st.checkbox(
            fig_ref("Add the exceedance curve from {a5}"), value=False,
            key="a9_exceedance",
            help="Conditional only, on a right-hand axis — the bars are success "
                 "trials, so a risked curve over them would run against a histogram "
                 "that has already dropped the failures.")
        _chart(pfig_a9_prospect_density(ts, mefs=mefs, show_exceedance=_a9_exc),
               key="a9")
        _a9_vals = res_all[res_all > 0]
        figure_note(
            fig_ref("**{a9}** — the same distribution as a shape rather than a curve: where the mass sits, not how likely each volume is."),
            detail=f"**{fig_ref('{a9}')}** — the same distribution {fig_ref('{a5}')} draws as a curve, "
            f"drawn as a shape. {fig_ref('{a5}')} answers "
            f"*how likely is at least this much*; {fig_ref('{a9}')} answers *where does the mass "
            f"actually sit*, "
            f"which is the question a long right tail makes hard to read off a cumulative curve. "
            f"The **mean is drawn thicker than the P50** because on a right-skewed resource "
            f"distribution they are different numbers and the mean is the one that gets quoted — "
            f"here {float(np.mean(_a9_vals)):.1f} against a P50 of "
            f"{float(np.percentile(_a9_vals, 50)):.1f} MMboe. It is the only volume figure on "
            f"this tab that needs no well at all.",
        )

        st.divider()
        _chart(pfig_a8_contact_distribution(ts), key="a8")
        figure_note(
            fig_ref("**{a8}** — where the contact lands, and the share of the prospect lying below any depth. This is what r_location reads off."),
            detail=f"**{fig_ref('{a8}')}** — the contact distribution recovered from the trials, and "
            f"`P(contact deeper "
            "than this depth)` over it. Read a depth off the y-axis and the line gives the "
            "fraction of success trials whose contact lies below it, which **is** `r_location` at "
            f"that entry, crest-referenced. So {fig_ref('{a8}')} is {fig_ref('{a3}')}'s raw "
            f"material shown as a distribution "
            "instead of as a chance curve, and the two agree at every depth by construction. "
            "This distribution is what the HCWC Builder produces and GeoX consumes; every "
            "location result in this tool ultimately rests on its shape.",
        )
